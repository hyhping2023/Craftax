"""SessionActor：环境会话的唯一可变状态所有者。

职责：
- 持有 (key, state, params, revision, timestep, terminated, episode_id)；
- 串行执行 reset / step（threading.Lock），JAX 推理在锁内完成；
- 构造 host 层 StateSummary / Snapshot / TransitionRecord；
- command_id 幂等 + expected_revision 乐观并发控制；
- 帧缓存（revision -> uint8 RGB），并按需编码 PNG。

本模块不 import gui / tasks / recording / dataset；tasks 与 recording
均通过 craftax.contracts 惰性接入，失败时回退到空实现。
"""
from __future__ import annotations

import dataclasses
import threading
import warnings
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from craftax.contracts import (
    DEFAULT_ENERGY_RATE,
    DEFAULT_THIRST_RATE,
    ActionSpec,
    NullRecorder,
    RecordingConfig,
    Snapshot,
    StateSummary,
    TaskEval,
    TransitionRecord,
    new_episode_id,
)
from craftax.service.frame_encoder import encode_png
from craftax.service.world_window import (
    CHUNK_SIZE,
    ChunkCoord,
    WorldOrigin,
    crop_window,
    global_position,
)
from craftax.service.chunk_store import ChunkStore


def env_params_to_dict(params: Any) -> Dict[str, Any]:
    """EnvParams -> JSON 可序列化 dict，用于写进 shard manifest（数据集自描述）。

    逐字段导出而不是白名单：以后往 EnvParams 里加参数（像本轮的 thirst_rate）
    会自动被记录，不需要再改这里——漏记等于数据集失去可复现性。
    """

    def scalar(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:  # noqa: BLE001 - 非标量数组：退回字符串表示
                pass
        if isinstance(value, (list, tuple)):
            return [scalar(v) for v in value]
        return str(value)

    out: Dict[str, Any] = {}
    for f in dataclasses.fields(params):
        out[f.name] = scalar(getattr(params, f.name))
    return out


# 动作枚举名（name）的权威来源。
_ACTION_NAMES: Dict[int, str] = {}


def _ensure_action_names() -> None:
    if not _ACTION_NAMES:
        from craftax.craftax.constants import Action

        for a in Action:
            _ACTION_NAMES[int(a.value)] = a.name


def action_spec_from_id(action_id: int) -> ActionSpec:
    """由 action id 构造 ActionSpec；id 无效时抛 ValueError。"""
    _ensure_action_names()
    name = _ACTION_NAMES.get(int(action_id))
    if name is None:
        raise ValueError(f"invalid action id: {action_id}")
    return ActionSpec(id=int(action_id), name=name)


def resolve_action_spec(action: Any) -> ActionSpec:
    """把 int 或 {id, name} 归一化为 ActionSpec（name 以 Action 枚举为准）。"""
    if isinstance(action, ActionSpec):
        return action
    if isinstance(action, int):
        return action_spec_from_id(action)
    # dict 或 pydantic ActionRef
    if isinstance(action, dict):
        return action_spec_from_id(int(action["id"]))
    action_id = int(action.id)
    spec = action_spec_from_id(action_id)
    provided_name = getattr(action, "name", None)
    if provided_name and provided_name != spec.name:
        raise ValueError(
            f"action name mismatch: id={action_id} expects {spec.name!r}, "
            f"got {provided_name!r}"
        )
    return spec


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class RevisionConflictError(Exception):
    """expected_revision 与当前 revision 不匹配（409）。"""

    def __init__(self, current_revision: int, expected_revision: int):
        self.current_revision = current_revision
        self.expected_revision = expected_revision
        super().__init__(
            f"revision conflict: current={current_revision}, expected={expected_revision}"
        )


class SessionTerminatedError(Exception):
    """episode 已终止，必须 reset 后才能继续（400）。"""

    def __init__(self, revision: int, terminated: bool, truncated: bool):
        self.revision = revision
        self.terminated = terminated
        self.truncated = truncated
        super().__init__(
            f"session is done (terminated={terminated}, truncated={truncated}); "
            f"call reset() first"
        )


class FrameNotFoundError(Exception):
    """请求的帧不存在或已被帧缓存淘汰（404）。"""

    def __init__(self, session_id: str, revision: int):
        self.session_id = session_id
        self.revision = revision
        super().__init__(f"frame {revision} not found for session {session_id}")


# ---------------------------------------------------------------------------
# Task 适配器（惰性，tasks 模块未就绪时回退为空实现）
# ---------------------------------------------------------------------------


class _NullTaskAdapter:
    task_id = ""
    version = ""
    spec = None

    def evaluate(self, state: Any, info: Dict[str, Any]) -> TaskEval:
        return TaskEval(progress=0.0, done=False, instruction="", event_tokens=[])


def _resolve_task_adapter(task_id: str, version: str) -> Any:
    try:
        from craftax.contracts import get_task_adapter

        return get_task_adapter(task_id, version)
    except Exception as e:  # noqa: BLE001 - tasks 模块可能尚未实现
        warnings.warn(f"task adapter unavailable for {task_id}@{version}: {e}")
        return _NullTaskAdapter()


# ---------------------------------------------------------------------------
# Recorder 钩子（惰性，recording 模块未就绪时回退为 NullRecorder）
# ---------------------------------------------------------------------------


def _make_recorder(recording: RecordingConfig):
    if not recording.enabled:
        return NullRecorder()
    try:
        from craftax.recording.recorder import AsyncRecorder

        return AsyncRecorder(
            recording_config=recording,
            spool_dir=recording.spool_dir,
            run_id=recording.dataset_run_id,
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"AsyncRecorder unavailable, using NullRecorder: {e}")
        return NullRecorder()


class _GuardedHook:
    """包装 recorder hook，使 recording 模块的运行时故障不影响服务主流程。"""

    def __init__(self, hook: Any):
        self._hook = hook
        self._warned = False

    def _guard(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                warnings.warn(f"recorder hook failed and was skipped: {e}")
                self._warned = True

    def on_episode_start(self, *args, **kwargs) -> None:
        self._guard(self._hook.on_episode_start, *args, **kwargs)

    def on_transition(self, record: TransitionRecord) -> None:
        self._guard(self._hook.on_transition, record)

    def on_episode_end(self, *args, **kwargs) -> None:
        self._guard(self._hook.on_episode_end, *args, **kwargs)

    def close(self) -> None:
        self._guard(self._hook.close)


# ---------------------------------------------------------------------------
# 像素渲染器缓存（Symbolic 环境需要单独 renderer）
# ---------------------------------------------------------------------------

_PIXEL_RENDERER_CACHE: Dict[int, Any] = {}


def _get_pixel_renderer(block_pixel_size: int):
    renderer = _PIXEL_RENDERER_CACHE.get(block_pixel_size)
    if renderer is None:
        from craftax.craftax.renderer import make_craftax_pixel_renderer

        renderer = jax.jit(make_craftax_pixel_renderer(block_pixel_size))
        _PIXEL_RENDERER_CACHE[block_pixel_size] = renderer
    return renderer


def _make_env(env_name: str) -> Any:
    """创建 NoAutoReset 环境实例。"""
    if env_name == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import (
            CraftaxPixelsEnvNoAutoReset,
        )

        return CraftaxPixelsEnvNoAutoReset()
    if env_name == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import (
            CraftaxSymbolicEnvNoAutoReset,
        )

        return CraftaxSymbolicEnvNoAutoReset()
    raise ValueError(f"unsupported env_name: {env_name}")


# ---------------------------------------------------------------------------
# host 值转换工具
# ---------------------------------------------------------------------------


def _to_py(value: Any) -> Any:
    """把 numpy / jax host 标量递归转换为 python 原生类型。"""
    if isinstance(value, dict):
        return {k: _to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_py(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return value


# ---------------------------------------------------------------------------
# SessionActor
# ---------------------------------------------------------------------------


class SessionActor:
    """单个环境会话的串行执行器，实现 craftax.contracts.SessionDriver 协议。"""

    def __init__(
        self,
        session_id: str,
        env_name: str,
        seed: Optional[int],
        task: Any,
        render: Any,
        recording: RecordingConfig,
        *,
        env: Optional[Any] = None,
        max_timesteps: Optional[int] = None,
        god_mode: bool = False,
        thirst_rate: Optional[float] = None,
        energy_rate: Optional[float] = None,
    ):
        self.session_id = session_id
        self.env_name = env_name
        self.controller: str = "human"
        self.task_id = task.task_id
        self.task_version = task.version

        self._render_cfg = render
        self._is_pixels = env_name == "Craftax-Pixels-v1"
        self._block_pixel_size = self._resolve_block_pixel_size(render)

        self._lock = threading.Lock()

        self._env = env if env is not None else _make_env(env_name)
        self._params = self._env.default_params
        if max_timesteps is not None:
            self._params = self._params.replace(max_timesteps=int(max_timesteps))
        if god_mode:
            self._params = self._params.replace(god_mode=True)
        # 口渴衰减：具身会话默认放缓（contracts.DEFAULT_THIRST_RATE=0.15），因为长程
        # 任务在原版速率下会被"找水"挤占；显式传 1.0 可恢复原版。EnvParams 本身
        # 的默认值不变，RL 基准不受影响。
        self.thirst_rate = float(
            DEFAULT_THIRST_RATE if thirst_rate is None else thirst_rate
        )
        self._params = self._params.replace(thirst_rate=self.thirst_rate)
        self.energy_rate = float(
            DEFAULT_ENERGY_RATE if energy_rate is None else energy_rate
        )
        self._params = self._params.replace(energy_rate=self.energy_rate)
        # 环境参数快照写进录制配置 → shard manifest（数据集自描述）。没有它，
        # 一批数据里混着不同 thirst_rate 的 episode 无法区分：transition
        # 结构完全相同，动力学却不同。
        self._recording_cfg = dataclasses.replace(
            recording, env_params=env_params_to_dict(self._params)
        )

        self._task_adapter = _resolve_task_adapter(task.task_id, task.version)
        self._hook = _GuardedHook(_make_recorder(recording))

        # 可变状态（仅在 _lock 内读写）；revision 从 -1 起步，
        # 首次 reset 后为 0（POST /sessions 返回 revision 0 的 Snapshot）。
        self._key: Any = None
        self._state: Any = None
        self._revision: int = -1
        self._timestep: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._episode_id: str = ""
        self._seed: Optional[int] = None
        # Host-side world coordinates.  The current backend starts at origin
        # (0, 0); future streamed chunks will advance this origin while the
        # JAX state keeps the same fixed-size active window.
        self._world_origin = WorldOrigin()
        self._floor_origins: Dict[int, WorldOrigin] = {0: self._world_origin}
        self._chunk_store: Optional[ChunkStore] = None

        self._frames: Dict[int, np.ndarray] = {}
        self._snapshots: Dict[int, Snapshot] = {}
        self._commands: Dict[str, Snapshot] = {}
        self._max_frames = 512
        self._max_snapshots = 1024
        self._max_commands = 512

        self._current_snapshot: Optional[Snapshot] = None

    # -- 只读访问 ---------------------------------------------------------

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def terminated(self) -> bool:
        with self._lock:
            return self._terminated

    def frame_dims(self, revision: int) -> Optional[Tuple[int, int]]:
        """返回帧 (height, width)；帧已淘汰返回 None。"""
        with self._lock:
            rgb = self._frames.get(revision)
            if rgb is None:
                return None
            return int(rgb.shape[0]), int(rgb.shape[1])

    # -- SessionDriver 协议 -------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        expected_revision: Optional[int] = None,
        command_id: Optional[str] = None,
    ) -> Snapshot:
        with self._lock:
            return self._reset_locked(seed, expected_revision, command_id)

    def step(
        self,
        action: ActionSpec,
        command_id: Optional[str] = None,
        wait_frame: bool = True,
        expected_revision: Optional[int] = None,
    ) -> Snapshot:
        with self._lock:
            return self._step_locked(action, command_id, wait_frame, expected_revision)

    def get_snapshot(self, revision: Optional[int] = None) -> Snapshot:
        with self._lock:
            if revision is None:
                if self._current_snapshot is None:
                    raise RuntimeError("session has no snapshot yet")
                return self._current_snapshot
            snap = self._snapshots.get(revision)
            if snap is None:
                raise KeyError(f"snapshot revision {revision} not found")
            return snap

    def get_frame_png(self, revision: int) -> bytes:
        with self._lock:
            rgb = self._frames.get(revision)
            if rgb is None:
                raise FrameNotFoundError(self.session_id, revision)
            return encode_png(rgb)

    def get_map(
        self, floor: Optional[int] = None, window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """返回指定楼层的地图窗口、玩家位置与实体状态。

        ``window_size=None`` 保持兼容行为，返回当前固定活动地图；传入窗口
        大小后只返回以玩家为中心的固定区域，并附带 ``map_origin`` 与绝对坐标。

        供规划器/demo 客户端读取全图；返回 host numpy/list,不做 JAX 编译。
        新增字段供技能链执行器使用：
        - mob_positions / mob_masks：本层怪物坐标与存活掩码（melee/ranged/passive）
        - ladder_down / ladder_up：本层下/上梯坐标
        - monsters_killed：本层已击杀数（下楼门槛用）
        - chest_positions：本层未开的 CHEST 方块坐标
        """
        with self._lock:
            if self._state is None:
                raise RuntimeError("session has no state yet")
            level = int(np.asarray(self._state.player_level).item())
            if floor is None:
                floor = level
            if not 0 <= floor < self._state.map.shape[0]:
                raise ValueError(f"floor {floor} 越界（共 {self._state.map.shape[0]} 层）")
            state = self._state
            host_map = np.asarray(jax.device_get(state.map[floor]))
            pos = jax.device_get(state.player_position)
            local_pos = [int(pos[0].item()), int(pos[1].item())]
            returned_local_pos = list(local_pos)
            origin = self._floor_origins.get(floor, WorldOrigin(floor=floor))
            map_origin = [origin.x, origin.y]
            local_map_origin = (0, 0)
            if window_size is not None:
                size = int(window_size)
                # Persist the authoritative grid for *any* requested floor
                # before stitching.  This keeps edits made by the JAX state
                # authoritative while allowing the planner to ask for a
                # larger target-floor window as well as the current floor.
                if self._chunk_store is not None:
                    self._chunk_store.merge_window(
                        floor, origin.x, origin.y, host_map,
                        np.asarray(jax.device_get(state.item_map[floor])),
                    )
                    if floor == level:
                        center_global = global_position(origin, local_pos)
                    else:
                        # Non-current floors have no player anchor.  Center a
                        # requested view on the remembered floor anchor's
                        # 48x48 initial window; absolute coordinates remain
                        # explicit in ``map_origin``.
                        center_global = [
                            origin.x + host_map.shape[0] // 2,
                            origin.y + host_map.shape[1] // 2,
                        ]
                    start_x = center_global[0] - size // 2
                    start_y = center_global[1] - size // 2
                    host_map, _ = self._chunk_store.render_window(
                        floor, start_x, start_y, size
                    )
                    local_map_origin = (start_x - origin.x, start_y - origin.y)
                else:
                    # Legacy/no-store path: crop the currently selected JAX
                    # grid around the player (or its center on another floor).
                    anchor = local_pos if floor == level else [
                        host_map.shape[0] // 2, host_map.shape[1] // 2
                    ]
                    host_map, local_map_origin = crop_window(host_map, anchor, size)
                if floor == level:
                    returned_local_pos = [
                        local_pos[0] - local_map_origin[0],
                        local_pos[1] - local_map_origin[1],
                    ]
                map_origin = [origin.x + local_map_origin[0], origin.y + local_map_origin[1]]
            direction = int(np.asarray(state.player_direction).item())

            # 怪物坐标与掩码（本层）
            mobs = {}
            for key in ("melee", "ranged", "passive"):
                mob = getattr(state, f"{key}_mobs", None)
                if mob is None:
                    mobs[key] = {"positions": [], "masks": []}
                    continue
                m_pos = jax.device_get(mob.position[floor])
                m_mask = jax.device_get(mob.mask[floor])
                positions = []
                global_positions = []
                visible_masks = []
                mobs[key] = {
                    "positions": positions,
                    "global_positions": global_positions,
                    "masks": visible_masks,
                }
                for mob_index, p in enumerate(m_pos):
                    point = [int(p[0].item()), int(p[1].item())]
                    absolute = global_position(origin, point)
                    local = [point[0] - local_map_origin[0], point[1] - local_map_origin[1]]
                    if 0 <= local[0] < host_map.shape[0] and 0 <= local[1] < host_map.shape[1]:
                        positions.append(local)
                        global_positions.append(absolute)
                        visible_masks.append(bool(m_mask[mob_index].item()))

            # 梯子与击杀数（本层）
            down = jax.device_get(state.down_ladders[floor])
            up = jax.device_get(state.up_ladders[floor])
            ladder_down = [int(down[0].item()), int(down[1].item())]
            ladder_up = [int(up[0].item()), int(up[1].item())]
            monsters_killed = int(
                np.asarray(state.monsters_killed[floor]).item()
            )

            # 未开宝箱位置（本层 CHEST 块，排除已开）
            map_arr = np.asarray(host_map)
            from craftax.craftax.constants import BlockType

            # ``host_map`` is already the map returned to the caller (the
            # active 48x48 grid or the requested cropped/stitched window).
            # Therefore chest coordinates are local to ``map_origin``.  The
            # previous implementation subtracted ``local_map_origin`` a
            # second time for a requested window and produced wrong/negative
            # positions, which made the planner lose a locked chest target.
            chest_rows, chest_cols = np.where(map_arr == BlockType.CHEST.value)
            chest_positions = [
                [int(x), int(y)] for x, y in zip(chest_rows, chest_cols)
            ]
            chest_global_positions = [
                [int(map_origin[0]) + p[0], int(map_origin[1]) + p[1]]
                for p in chest_positions
            ]
            if window_size is not None and floor == level:
                ladder_down = [
                    ladder_down[0] - local_map_origin[0],
                    ladder_down[1] - local_map_origin[1],
                ]
                ladder_up = [
                    ladder_up[0] - local_map_origin[0],
                    ladder_up[1] - local_map_origin[1],
            ]
            if self._chunk_store is not None:
                center_x = (int(map_origin[0]) + host_map.shape[0] // 2) // CHUNK_SIZE
                center_y = (int(map_origin[1]) + host_map.shape[1] // 2) // CHUNK_SIZE

                for cx in range(center_x - 1, center_x + 2):
                    for cy in range(center_y - 1, center_y + 2):
                        self._chunk_store.get(ChunkCoord(floor, cx, cy))

            # The player belongs to the current floor even when a caller asks
            # for another floor's planning window.  Report that absolute
            # position from the current world anchor; map tiles themselves are
            # anchored by ``map_origin`` below.
            global_pos = global_position(self._world_origin, local_pos)
            # Positions of entities remain local to the returned window for
            # backwards compatibility; absolute copies are provided alongside
            # them for streamed-world planners.
            return {
                "floor": floor,
                "current_level": level,
                "map": host_map.tolist(),
                "player_position": returned_local_pos,
                "player_global_position": global_pos,
                "map_origin": map_origin,
                "world_origin": [origin.x, origin.y],
                "world_origin_chunk": [origin.x // CHUNK_SIZE, origin.y // CHUNK_SIZE],
                "chunk_size": CHUNK_SIZE,
                "window_size": int(host_map.shape[0]),
                "world_mode": "streamed_chunk_v1",
                "world_seed": self._seed,
                "chunk_store": "craftax_generator_with_offscreen_tick_v1",
                "chunk_generator": "craftax_smoothworld_v1",
                "loaded_chunks": [
                    [c.floor, c.x, c.y]
                    for c in (self._chunk_store.loaded() if self._chunk_store else ())
                ],
                "player_direction": direction,
                "mob_positions": mobs,
                "ladder_down": ladder_down,
                "ladder_up": ladder_up,
                "ladder_down_global": global_position(origin, [int(down[0].item()), int(down[1].item())]),
                "ladder_up_global": global_position(origin, [int(up[0].item()), int(up[1].item())]),
                "ladder_network": (
                    self._chunk_store.ladders(floor)
                    if self._chunk_store is not None else {"up": [], "down": []}
                ),
                "monsters_killed": monsters_killed,
                "chest_positions": chest_positions,
                "chest_global_positions": chest_global_positions,
            }

    def close(self) -> None:
        with self._lock:
            self._hook.close()

    # -- 内部实现 -----------------------------------------------------------

    def _reset_locked(
        self,
        seed: Optional[int],
        expected_revision: Optional[int],
        command_id: Optional[str],
    ) -> Snapshot:
        # 1. command_id 幂等：重复命令返回首次结果
        if command_id is not None:
            cached = self._commands.get(command_id)
            if cached is not None:
                return cached

        # 2. expected_revision 乐观并发控制
        if expected_revision is not None and expected_revision != self._revision:
            raise RevisionConflictError(self._revision, expected_revision)

        # 3. 新 episode
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._seed = seed
        self._world_origin = WorldOrigin()
        self._floor_origins = {0: self._world_origin}
        self._chunk_store = ChunkStore(seed)
        self._episode_id = new_episode_id(self.session_id)

        # 4. PRNG：split 后使用子 key，避免复用
        self._key = jax.random.PRNGKey(seed)
        key, reset_key = jax.random.split(self._key)
        self._key = key

        obs, state = self._env.reset(reset_key, self._params)
        host_state = jax.device_get(state)
        host_obs = jax.device_get(obs)
        # Preserve the authoritative Craftax generator for the initial active
        # world. Future chunks are generated lazily by ChunkStore, while this
        # 48x48 window exactly matches the JAX reset state.
        for floor_index in range(int(host_state.map.shape[0])):
            self._chunk_store.hydrate_window(
                floor_index,
                0,
                0,
                np.asarray(host_state.map[floor_index]),
                np.asarray(host_state.item_map[floor_index]),
            )
            self._chunk_store.register_ladders(
                floor_index, "down", np.asarray(host_state.down_ladders[floor_index])
            )
            self._chunk_store.register_ladders(
                floor_index, "up", np.asarray(host_state.up_ladders[floor_index])
            )

        timestep = int(np.asarray(host_state.timestep).item())
        frame_rgb = self._render_frame(host_obs, state)

        # 5. 推进 revision
        self._revision += 1
        self._state = state
        self._timestep = timestep
        self._terminated = False
        self._truncated = False

        self._frames[self._revision] = frame_rgb
        self._trim_frames()

        _, summary = self._evaluate_and_summary(host_state, info={})
        snap = Snapshot(
            session_id=self.session_id,
            revision=self._revision,
            timestep=timestep,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            summary=summary,
            frame_png=encode_png(frame_rgb),
            frame_rgb=frame_rgb,
            frame_revision=self._revision,
            command_id=command_id or "",
            info={},
        )
        self._store_snapshot(snap)
        if command_id is not None:
            self._commands[command_id] = snap
            self._trim_commands()
        self._current_snapshot = snap

        # 6. recorder：frame 0（is_initial_frame）
        self._hook.on_episode_start(
            session_id=self.session_id,
            episode_id=self._episode_id,
            task_id=self.task_id,
            task_version=self.task_version,
            seed=self._seed,
            recording_config=self._recording_cfg,
            state=host_state,
            frame=frame_rgb,
        )
        return snap

    def _step_locked(
        self,
        action: ActionSpec,
        command_id: Optional[str],
        wait_frame: bool,
        expected_revision: Optional[int],
    ) -> Snapshot:
        # 1. command_id 幂等：重复命令返回首次结果
        if command_id is not None:
            cached = self._commands.get(command_id)
            if cached is not None:
                return cached

        # 2. expected_revision 乐观并发控制
        if expected_revision is not None and expected_revision != self._revision:
            raise RevisionConflictError(self._revision, expected_revision)

        # 3. 终局校验：终止后必须 reset
        if self._terminated or self._truncated:
            raise SessionTerminatedError(
                self._revision, self._terminated, self._truncated
            )
        if self._state is None:
            raise SessionTerminatedError(0, False, False)

        stream_event = self._maybe_shift_active_window(action.id)

        # 4. JAX step
        key, step_key = jax.random.split(self._key)
        self._key = key
        obs, state, reward, done, info = self._env.step(
            step_key, self._state, action.id, self._params
        )
        host_state = jax.device_get(state)
        host_obs = jax.device_get(obs)
        current_level = int(np.asarray(host_state.player_level).item())
        if current_level != self._world_origin.floor:
            self._world_origin = self._floor_origins.setdefault(
                current_level, WorldOrigin(floor=current_level)
            )
        if self._chunk_store is not None:
            height, width = self._state.map.shape[-2:]
            self._chunk_store.tick_offscreen(
                current_level, self._world_origin.x, self._world_origin.y,
                min(int(height), int(width)),
            )
        host_info = _to_py(jax.device_get(info))
        host_info.update(self._world_metadata(host_state))
        if stream_event is not None:
            host_info.update(stream_event)
        reward = float(np.asarray(reward).item())

        # 5. 终局判定：到达 max_timesteps 视为截断，其余视为终止
        pre_timestep = self._timestep
        post_timestep = int(np.asarray(host_state.timestep).item())
        done = bool(np.asarray(done).item())
        if done:
            if post_timestep >= int(self._params.max_timesteps):
                truncated, terminated = True, False
            else:
                truncated, terminated = False, True
        else:
            terminated, truncated = False, False

        frame_rgb = self._render_frame(host_obs, state)

        # 6. 推进状态
        self._revision += 1
        self._state = state
        self._timestep = post_timestep
        self._terminated = terminated
        self._truncated = truncated

        self._frames[self._revision] = frame_rgb
        self._trim_frames()

        eval_result, summary = self._evaluate_and_summary(host_state, info=host_info)
        if isinstance(host_info, dict):
            host_info["event_tokens"] = list(eval_result.event_tokens)

        snap = Snapshot(
            session_id=self.session_id,
            revision=self._revision,
            timestep=post_timestep,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            summary=summary,
            frame_png=encode_png(frame_rgb) if wait_frame else None,
            frame_rgb=frame_rgb,
            frame_revision=self._revision,
            command_id=command_id or "",
            info=host_info,
        )
        self._store_snapshot(snap)
        if command_id is not None:
            self._commands[command_id] = snap
            self._trim_commands()
        self._current_snapshot = snap

        # 7. recorder transition
        if self._recording_cfg.enabled:
            frame_sample = self._recording_cfg.frame_sample
            is_sampled = frame_sample.is_sampled(post_timestep)
            is_terminal_frame = terminated or truncated
            has_frame = is_sampled or is_terminal_frame
            record = TransitionRecord(
                session_id=self.session_id,
                episode_id=self._episode_id,
                timestep=pre_timestep,
                action=action,
                action_source=self.controller,
                command_id=command_id or "",
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                state=host_state,
                frame=frame_rgb if has_frame else None,
                is_sampled_frame=is_sampled,
                is_initial_frame=False,
                is_terminal_frame=is_terminal_frame,
                info=host_info,
                event_tokens=list(eval_result.event_tokens),
                instruction=eval_result.instruction,
                task_id=self.task_id,
                task_version=self.task_version,
            )
            self._hook.on_transition(record)

        # 8. episode 结束通知
        if terminated or truncated:
            self._hook.on_episode_end(
                self.session_id, self._episode_id, terminated=terminated
            )

        return snap

    def _world_metadata(self, host_state: Any) -> Dict[str, Any]:
        """Stable absolute-coordinate metadata attached to every transition."""
        origin = self._world_origin
        pos = np.asarray(host_state.player_position)
        absolute = global_position(origin, [int(pos[0]), int(pos[1])])
        return {
            "world_seed": self._seed,
            "world_origin": [origin.x, origin.y],
            "world_origin_chunk": [origin.x // CHUNK_SIZE, origin.y // CHUNK_SIZE],
            "player_global_position": absolute,
            "chunk_size": CHUNK_SIZE,
        }

    def _maybe_shift_active_window(self, action_id: int) -> Optional[Dict[str, Any]]:
        """Move the fixed JAX window when a cardinal action crosses its edge."""
        if self._state is None or self._chunk_store is None:
            return None
        from craftax.craftax.constants import Action, BlockType, SOLID_BLOCKS

        deltas = {
            int(Action.LEFT.value): (0, -1),
            int(Action.RIGHT.value): (0, 1),
            int(Action.UP.value): (-1, 0),
            int(Action.DOWN.value): (1, 0),
        }
        delta = deltas.get(int(action_id))
        if delta is None:
            return None
        level = int(np.asarray(self._state.player_level).item())
        pos = np.asarray(self._state.player_position, dtype=np.int32)
        height, width = self._state.map.shape[-2:]
        next_pos = (int(pos[0]) + delta[0], int(pos[1]) + delta[1])
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width:
            return None

        # Crossing the active-window edge is a real movement, not merely a
        # viewport refresh.  Check the deterministic destination chunk first;
        # otherwise a blocked edge action could shift the window and let the
        # player walk through a solid/water/lava tile that the native collision
        # routine would have rejected.
        global_row = int(self._world_origin.x) + next_pos[0]
        global_col = int(self._world_origin.y) + next_pos[1]
        destination_coord = ChunkCoord(
            level, global_col // CHUNK_SIZE, global_row // CHUNK_SIZE
        )
        destination_chunk = self._chunk_store.get(destination_coord)
        destination_tile = int(
            destination_chunk.blocks[global_row % CHUNK_SIZE, global_col % CHUNK_SIZE]
        )
        if destination_tile in set(SOLID_BLOCKS) or destination_tile in (
            BlockType.WATER.value, BlockType.LAVA.value
        ):
            return None

        old_origin = self._world_origin
        old_map = np.asarray(jax.device_get(self._state.map[level]))
        old_items = np.asarray(jax.device_get(self._state.item_map[level]))
        self._chunk_store.merge_window(
            level, old_origin.x, old_origin.y, old_map, old_items
        )
        entity_arrays = {
            name: {
                field: np.asarray(
                    jax.device_get(getattr(getattr(self._state, name), field)[level])
                )
                for field in ("position", "health", "mask", "attack_cooldown", "type_id")
            }
            for name in (
                "melee_mobs", "ranged_mobs", "passive_mobs",
                "mob_projectiles", "player_projectiles",
            )
        }
        self._chunk_store.save_entities(level, old_origin.x, old_origin.y, entity_arrays)
        new_x, new_y = old_origin.x, old_origin.y
        if next_pos[0] < 0:
            new_x -= CHUNK_SIZE
        elif next_pos[0] >= height:
            new_x += CHUNK_SIZE
        if next_pos[1] < 0:
            new_y -= CHUNK_SIZE
        elif next_pos[1] >= width:
            new_y += CHUNK_SIZE

        window, item_window = self._chunk_store.render_window(level, new_x, new_y, height)
        global_row = old_origin.x + int(pos[0])
        global_col = old_origin.y + int(pos[1])
        new_pos = jnp.asarray([global_row - new_x, global_col - new_y], dtype=jnp.int32)
        # The player may land at offset 31 (a one-chunk shift), not at the
        # window center.  The generated chunk owns this cell; never overwrite
        # it with PATH here, otherwise merely refreshing the active window
        # would mutate the absolute world's block type.  Craftax chunk
        # generation keeps chunk borders traversable, so the handoff cell is
        # already valid in the rendered window.
        new_pos_host = [int(new_pos[0]), int(new_pos[1])]
        new_map = self._state.map.at[level].set(jnp.asarray(window))
        new_items = self._state.item_map.at[level].set(jnp.asarray(item_window))
        translated = self._translate_active_entities(level, old_origin, new_x, new_y)
        self._state = self._state.replace(
            map=new_map,
            item_map=new_items,
            player_position=new_pos,
            **translated,
        )
        self._world_origin = WorldOrigin(floor=level, x=new_x, y=new_y)
        # ``get_map`` resolves the origin through ``_floor_origins``.  Keep
        # that floor anchor in sync with every streamed shift; otherwise the
        # next response would silently report stale absolute coordinates even
        # though the JAX window had moved.
        self._floor_origins[level] = self._world_origin
        return {
            "world_expanded": True,
            "world_origin": [new_x, new_y],
            "world_origin_chunk": [new_x // CHUNK_SIZE, new_y // CHUNK_SIZE],
            "world_transition": {
                "from_origin": [old_origin.x, old_origin.y],
                "to_origin": [new_x, new_y],
                "floor": level,
            },
        }

    def _translate_active_entities(
        self, level: int, old_origin: WorldOrigin, new_x: int, new_y: int
    ) -> Dict[str, Any]:
        """Translate host-window entity coordinates across a window shift.

        Entities outside the new active window are masked out; their persistent
        simulation will be reintroduced by the chunk/entity store in the next
        streaming phase instead of leaving stale local coordinates in JAX.
        """
        state = self._state
        delta_r = int(old_origin.x - new_x)
        delta_c = int(old_origin.y - new_y)
        height, width = state.map.shape[-2:]
        updates: Dict[str, Any] = {}

        def translate_mobs(name: str) -> None:
            mob = getattr(state, name)
            pos = mob.position
            current = np.asarray(jax.device_get(pos[level]))
            mask = np.asarray(jax.device_get(mob.mask[level])).copy()
            moved = current.copy()
            moved[:, 0] += delta_r
            moved[:, 1] += delta_c
            visible = (
                (moved[:, 0] >= 0) & (moved[:, 0] < height)
                & (moved[:, 1] >= 0) & (moved[:, 1] < width)
            )
            mask &= visible
            updates[name] = mob.replace(
                position=pos.at[level].set(jnp.asarray(moved)),
                mask=mob.mask.at[level].set(jnp.asarray(mask)),
            )

        for name in (
            "melee_mobs", "ranged_mobs", "passive_mobs",
            "mob_projectiles", "player_projectiles",
        ):
            translate_mobs(name)
            mob = updates[name]
            records = self._chunk_store.load_entities(
                level, new_x, new_y, height, name
            )
            # Replace the active-floor arrays with persisted records visible in
            # the new window. Newly generated windows simply have no records;
            # normal Craftax spawning will repopulate them on subsequent steps.
            pos_arr = np.zeros_like(np.asarray(jax.device_get(mob.position[level])))
            health_arr = np.zeros_like(np.asarray(jax.device_get(mob.health[level])))
            mask_arr = np.zeros_like(np.asarray(jax.device_get(mob.mask[level])))
            cooldown_arr = np.zeros_like(np.asarray(jax.device_get(mob.attack_cooldown[level])))
            type_arr = np.zeros_like(np.asarray(jax.device_get(mob.type_id[level])))
            for i, record in enumerate(records[: len(mask_arr)]):
                pos_arr[i] = record["position"]
                health_arr[i] = record["health"]
                mask_arr[i] = True
                cooldown_arr[i] = record["attack_cooldown"]
                type_arr[i] = record["type_id"]
            updates[name] = mob.replace(
                position=mob.position.at[level].set(jnp.asarray(pos_arr)),
                health=mob.health.at[level].set(jnp.asarray(health_arr)),
                mask=mob.mask.at[level].set(jnp.asarray(mask_arr)),
                attack_cooldown=mob.attack_cooldown.at[level].set(jnp.asarray(cooldown_arr)),
                type_id=mob.type_id.at[level].set(jnp.asarray(type_arr)),
            )

        # mob_map is an occupancy cache used by spawning/collision logic; it
        # must be rebuilt together with the translated entity arrays.
        translated_map = np.zeros_like(np.asarray(jax.device_get(state.mob_map)))
        for name in ("melee_mobs", "ranged_mobs", "passive_mobs"):
            mob = updates[name]
            positions = np.asarray(jax.device_get(mob.position[level]))
            masks = np.asarray(jax.device_get(mob.mask[level]))
            for p, alive in zip(positions, masks):
                if bool(alive) and 0 <= int(p[0]) < height and 0 <= int(p[1]) < width:
                    translated_map[level, int(p[0]), int(p[1])] = True
        updates["mob_map"] = jnp.asarray(translated_map)

        plants = np.asarray(jax.device_get(state.growing_plants_positions)).copy()
        plant_mask = np.asarray(jax.device_get(state.growing_plants_mask)).copy()
        plants[:, 0] += delta_r
        plants[:, 1] += delta_c
        plant_visible = (
            (plants[:, 0] >= 0) & (plants[:, 0] < height)
            & (plants[:, 1] >= 0) & (plants[:, 1] < width)
        )
        updates["growing_plants_positions"] = jnp.asarray(plants)
        updates["growing_plants_mask"] = jnp.asarray(plant_mask & plant_visible)
        return updates

    # -- 渲染 / 摘要 ---------------------------------------------------------

    def _render_frame(self, host_obs: Any, state: Any) -> np.ndarray:
        """生成 uint8 HWC RGB 帧，分辨率由 block_pixel_size 决定。

        对所有环境统一使用参数化像素 renderer（Pixels 环境的 obs 本身也是
        renderer 的输出，统一后分辨率完全可控）。
        """
        renderer = _get_pixel_renderer(self._block_pixel_size)
        pixels = renderer(state)
        arr = np.asarray(jax.device_get(pixels), dtype=np.float32)
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)

    @staticmethod
    def _resolve_block_pixel_size(render: Any) -> int:
        """解析方块像素尺寸：显式 block_pixel_size > mode 默认（human=16, agent=10）。"""
        explicit = getattr(render, "block_pixel_size", None)
        if explicit is not None:
            size = int(explicit)
            if size <= 0:
                raise ValueError(f"block_pixel_size 必须为正数，收到 {size}")
            return size
        mode = getattr(render, "mode", "human")
        return 16 if mode == "human" else 10

    def _evaluate_and_summary(
        self, host_state: Any, info: Dict[str, Any]
    ) -> Tuple[TaskEval, StateSummary]:
        """计算任务评估结果并构造 StateSummary。"""
        eval_result: TaskEval = TaskEval(
            progress=0.0, done=False, instruction="", event_tokens=[]
        )
        try:
            eval_result = self._task_adapter.evaluate(host_state, info)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"task adapter evaluate failed: {e}")

        achievements = np.asarray(host_state.achievements).ravel()
        from craftax.craftax.constants import Achievement

        achieved = [a.name for a in Achievement if bool(achievements[int(a.value)])]

        inv = host_state.inventory
        inventory = {
            "wood": int(np.asarray(inv.wood).item()),
            "stone": int(np.asarray(inv.stone).item()),
            "coal": int(np.asarray(inv.coal).item()),
            "iron": int(np.asarray(inv.iron).item()),
            "diamond": int(np.asarray(inv.diamond).item()),
            "sapling": int(np.asarray(inv.sapling).item()),
            "pickaxe": int(np.asarray(inv.pickaxe).item()),
            "sword": int(np.asarray(inv.sword).item()),
            "bow": int(np.asarray(inv.bow).item()),
            "arrows": int(np.asarray(inv.arrows).item()),
            "water": int(np.asarray(inv.water).item()),
            "torches": int(np.asarray(inv.torches).item()),
            "ruby": int(np.asarray(inv.ruby).item()),
            "sapphire": int(np.asarray(inv.sapphire).item()),
            "books": int(np.asarray(inv.books).item()),
            "armour": [int(v) for v in np.asarray(inv.armour).ravel()],
            "potions": [int(v) for v in np.asarray(inv.potions).ravel()],
        }

        summary = StateSummary(
            timestep=int(np.asarray(host_state.timestep).item()),
            health=float(np.asarray(host_state.player_health).item()),
            food=float(np.asarray(host_state.player_food).item()),
            drink=float(np.asarray(host_state.player_drink).item()),
            energy=float(np.asarray(host_state.player_energy).item()),
            mana=float(np.asarray(host_state.player_mana).item()),
            floor=int(np.asarray(host_state.player_level).item()),
            xp=int(np.asarray(host_state.player_xp).item()),
            dexterity=int(np.asarray(host_state.player_dexterity).item()),
            strength=int(np.asarray(host_state.player_strength).item()),
            intelligence=int(np.asarray(host_state.player_intelligence).item()),
            is_sleeping=bool(np.asarray(host_state.is_sleeping).item()),
            is_resting=bool(np.asarray(host_state.is_resting).item()),
            inventory=inventory,
            achievements=achieved,
            sword_enchantment=int(np.asarray(host_state.sword_enchantment).item()),
            bow_enchantment=int(np.asarray(host_state.bow_enchantment).item()),
            armour_enchantments=[int(v) for v in np.asarray(host_state.armour_enchantments).ravel()],
            learned_spells=[bool(v) for v in np.asarray(host_state.learned_spells).ravel()],
            task_progress=float(eval_result.progress),
            task_done=bool(eval_result.done),
            instruction=eval_result.instruction,
            task_id=self.task_id,
            task_version=self.task_version,
            player_position=[
                int(np.asarray(host_state.player_position[0]).item()),
                int(np.asarray(host_state.player_position[1]).item()),
            ],
            player_direction=int(np.asarray(host_state.player_direction).item()),
        )
        return eval_result, summary

    # -- 缓存 ---------------------------------------------------------------

    def _store_snapshot(self, snap: Snapshot) -> None:
        # 历史快照不保留帧字节（帧从缓存按需读取）
        hist = Snapshot(
            session_id=snap.session_id,
            revision=snap.revision,
            timestep=snap.timestep,
            action=snap.action,
            reward=snap.reward,
            terminated=snap.terminated,
            truncated=snap.truncated,
            summary=snap.summary,
            frame_png=None,
            frame_rgb=None,
            frame_revision=snap.frame_revision,
            command_id=snap.command_id,
            info=snap.info,
        )
        self._snapshots[snap.revision] = hist
        while len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(next(iter(self._snapshots)))

    def _trim_frames(self) -> None:
        while len(self._frames) > self._max_frames:
            self._frames.pop(next(iter(self._frames)))

    def _trim_commands(self) -> None:
        while len(self._commands) > self._max_commands:
            self._commands.pop(next(iter(self._commands)))
