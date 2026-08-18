"""Craftax Embodied 层共享契约。

本文件是 service / gui / tasks / recording / dataset 五个模块之间唯一的共享依赖。
规则：
- 只依赖标准库 + numpy；不依赖 flax / jax / pygame / fastapi 具体类型。
- 所有跨模块类型与常量必须在此定义，模块间不得相互 import。
- state 一律以“host 化 numpy 数组的扁平 dict（树路径 -> np.ndarray）”形式传递，
  由 recording.state_codec 负责 EnvState PyTree <-> dict 转换。
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# 动作
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    """离散动作的稳定表示。name 来自 craftax.craftax.constants.Action 枚举名。"""

    id: int
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ActionSpec":
        return ActionSpec(id=int(d["id"]), name=str(d["name"]))


def all_action_specs() -> List[ActionSpec]:
    """从 craftax Action 枚举生成全部动作；惰性 import，避免启动时依赖游戏包。"""
    from craftax.craftax.constants import Action

    return [ActionSpec(a.value, a.name) for a in Action]


# ---------------------------------------------------------------------------
# 帧采样与录制配置（MP4 精确对齐契约）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSampleConfig:
    """CFR MP4 采样配置。

    step_rate_hz(S) / video_fps(F) 必须满足 S % F == 0，
    帧间隔 R == S // F：视频第 i 帧恒对应 timestep = i * R。
    """

    step_rate_hz: int = 20  # S：环境每秒执行 step 数
    video_fps: int = 10  # F：MP4 固定帧率

    @property
    def frame_stride(self) -> int:
        """R = S / F：每 R 个环境 step 采样一帧。"""
        return self.step_rate_hz // self.video_fps

    def validate(self) -> None:
        if self.step_rate_hz <= 0 or self.video_fps <= 0:
            raise ValueError("step_rate_hz 与 video_fps 必须为正")
        if self.step_rate_hz % self.video_fps != 0:
            raise ValueError(
                f"step_rate_hz({self.step_rate_hz}) 必须能被 video_fps({self.video_fps}) 整除"
            )
        if self.video_fps > self.step_rate_hz:
            raise ValueError("video_fps 不能大于 step_rate_hz")

    def is_sampled(self, timestep: int) -> bool:
        """常规采样帧判定（不含 frame 0 与 terminal frame）。"""
        return timestep > 0 and timestep % self.frame_stride == 0

    def to_dict(self) -> Dict[str, Any]:
        return {"step_rate_hz": self.step_rate_hz, "video_fps": self.video_fps}


def default_data_dir() -> str:
    """数据集/录制输出的默认根目录。

    默认 <仓库根>/data；可用环境变量 CRAFTAX_DATA_DIR 覆盖。
    shard 目录布局为 <root>/<run_id>/<producer_id>/<attempt_id>/。
    """
    if "CRAFTAX_DATA_DIR" in os.environ:
        return os.environ["CRAFTAX_DATA_DIR"]
    return str(Path(__file__).resolve().parent.parent / "data")


@dataclass(frozen=True)
class RecordingConfig:
    """数据集录制配置。spool_dir 为本地暂存目录，shard 封存后不可变。"""

    enabled: bool = True
    dataset_run_id: str = "default"
    frame_sample: FrameSampleConfig = FrameSampleConfig()
    gold_frames: bool = False  # 是否额外保存每 step PNG 金标帧
    spool_dir: str = field(default_factory=default_data_dir)
    shard_max_transitions: int = 50_000  # 一个 shard 的 transition 上限（待基准后调整）

    def validate(self) -> None:
        self.frame_sample.validate()


# ---------------------------------------------------------------------------
# Transition / Episode 记录
# ---------------------------------------------------------------------------


@dataclass
class TransitionRecord:
    """一步动作的完整记录（host 层，录制器消费）。

    时间轴语义（与 embodied_environment_plan.md 第 7.2 节一致）：
        action[t] 作用于 state[t] -> reward[t], terminated[t], truncated[t], state[t+1]
        frame 描述 state[t]；常规采样帧满足 timestep % R == 0（frame 0 与 terminal frame 除外）。
    """

    session_id: str
    episode_id: str
    timestep: int  # 本步动作作用前的时间步 t（state[t]）
    action: ActionSpec
    action_source: str  # human | scripted | policy | random | replay | curriculum_generator
    command_id: str
    reward: float
    terminated: bool
    truncated: bool
    state: Any  # host 化（numpy 值）的 EnvState，由 recording.state_codec 扁平化后写入 Zarr
    frame: Optional[np.ndarray] = None  # state[t+1] 的画面，uint8 HWC RGB；None 表示本步未采样
    is_sampled_frame: bool = False
    is_initial_frame: bool = False  # 仅 reset 后的 frame 0
    is_terminal_frame: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    event_tokens: List[str] = field(default_factory=list)
    instruction: str = ""
    task_id: str = ""
    task_version: str = ""
    sim_time_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "timestep": self.timestep,
            "action": self.action.to_dict(),
            "action_source": self.action_source,
            "command_id": self.command_id,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "is_sampled_frame": self.is_sampled_frame,
            "is_initial_frame": self.is_initial_frame,
            "is_terminal_frame": self.is_terminal_frame,
            "info": self.info,
            "event_tokens": self.event_tokens,
            "instruction": self.instruction,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "sim_time_ns": self.sim_time_ns,
        }


def new_episode_id(session_id: str) -> str:
    return f"{session_id[:8]}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 任务（原生 Craftax 只读标注）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """版本化任务定义。只描述“解释与标注”，不得改变游戏规则。

    success_predicate / annotation_predicates 为可序列化的表达式（如
    {"type": "achievement", "name": "COLLECT_WOOD"}），由 tasks 模块解析。

    dependencies：前置任务 task_id 列表（严格依赖）。例如 collect_coal 需要
    镐子才能挖矿，故依赖 native.craft_wood_pickaxe；enter_sewers 需要先到
    gnomish_mines，故依赖 native.enter_gnomish_mines。用于任务规划/排序/展示，
    不改变游戏规则。
    """

    task_id: str
    version: str
    instruction: str
    objective: str
    success_predicate: Dict[str, Any]
    annotation_predicates: List[Dict[str, Any]] = field(default_factory=list)
    renderer_config: Dict[str, Any] = field(default_factory=dict)
    action_vocabulary_version: str = "craftax-native-v1"
    dependencies: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskEval:
    """TaskAdapter 对一步 transition 的只读评估结果。"""

    progress: float  # 0..1
    done: bool  # 成功（仅评估，不影响环境）
    instruction: str
    event_tokens: List[str] = field(default_factory=list)


class TaskAdapter(Protocol):
    """只读任务适配器协议。不得修改 state / reward / 终局 / 动作。"""

    task_id: str
    version: str
    spec: TaskSpec

    def evaluate(self, state: Any, info: Dict[str, Any]) -> TaskEval:
        """基于 host 化 state 与 info（含原生 achievements）计算进度与事件。

        实现须为只读：不得修改 state / reward / 终局 / 动作。
        """
        ...


def get_task_adapter(task_id: str, version: str) -> TaskAdapter:
    """惰性解析任务适配器；tasks 模块通过 registry 注册实现。"""
    from craftax.tasks.registry import get_task_adapter as _impl

    return _impl(task_id, version)


def list_task_ids() -> List[str]:
    from craftax.tasks.registry import list_task_ids as _impl

    return _impl()


# ---------------------------------------------------------------------------
# Recorder 钩子（service 调用，recording 模块实现）
# ---------------------------------------------------------------------------


class RecorderHook(Protocol):
    """异步录制钩子。service 的 SessionActor 在每个事件上调用。"""

    def on_episode_start(
        self,
        session_id: str,
        episode_id: str,
        task_id: str,
        task_version: str,
        seed: int,
        recording_config: RecordingConfig,
        state: Any,  # host 化 EnvState
        frame: Optional[np.ndarray],
    ) -> None: ...

    def on_transition(self, record: TransitionRecord) -> None: ...

    def on_episode_end(self, session_id: str, episode_id: str, terminated: bool) -> None: ...

    def close(self) -> None: ...


class NullRecorder:
    """录制禁用时的空实现。"""

    def on_episode_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_transition(self, record: TransitionRecord) -> None:
        pass

    def on_episode_end(self, session_id: str, episode_id: str, terminated: bool) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 会话驱动（gui 与 service 的桥）
# ---------------------------------------------------------------------------


@dataclass
class StateSummary:
    """状态面板展示所需的轻量摘要（host 标量）。"""

    timestep: int
    health: float
    food: float
    drink: float
    energy: float
    mana: float
    floor: int
    xp: int
    dexterity: int
    strength: int
    intelligence: int
    is_sleeping: bool
    is_resting: bool
    inventory: Dict[str, Any] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)
    sword_enchantment: int = 0  # 0=无, 1=火, 2=冰
    bow_enchantment: int = 0
    armour_enchantments: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    learned_spells: List[bool] = field(default_factory=lambda: [False, False])  # 火球/冰球
    task_progress: float = 0.0
    task_done: bool = False
    instruction: str = ""
    task_id: str = ""
    task_version: str = ""
    player_position: List[int] = field(default_factory=lambda: [0, 0])
    player_direction: int = 0


@dataclass
class Snapshot:
    """一次 reset/step 后返回给客户端的结果。"""

    session_id: str
    revision: int  # 单调递增，冲突检测用
    timestep: int
    action: Optional[ActionSpec] = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    summary: Optional[StateSummary] = None
    frame_png: Optional[bytes] = None  # API/Pygame remote 模式
    frame_rgb: Optional[np.ndarray] = None  # embedded 模式，uint8 HWC
    frame_revision: Optional[int] = None
    command_id: str = ""
    info: Dict[str, Any] = field(default_factory=dict)


class SessionDriver(Protocol):
    """环境会话驱动。gui / 脚本 / 测试均通过该接口交互。"""

    def reset(self, seed: Optional[int] = None) -> Snapshot: ...

    def step(
        self,
        action: ActionSpec,
        command_id: Optional[str] = None,
        wait_frame: bool = True,
    ) -> Snapshot: ...

    def get_snapshot(self, revision: Optional[int] = None) -> Snapshot: ...

    def get_frame_png(self, revision: int) -> bytes: ...

    @property
    def revision(self) -> int: ...


# ---------------------------------------------------------------------------
# Shard 目录布局（recording 与 dataset 的共同契约）
# ---------------------------------------------------------------------------

# 一个 sealed shard 的目录内容：
#   shard_manifest.json       不可变 manifest（含全部 hash）
#   tensors.zarr/             状态/动作/奖励等固定 shape 张量（时间维 = 总 transition 数）
#   episodes.parquet          episode 边界与任务/seed 元数据
#   transitions.parquet       每 transition 的 action metadata / 事件 / 指令
#   frame_index.parquet       视频帧 -> timestep/state_index 显式映射
#   video-<video_id>.mp4      CFR H.264 视频（每 episode 一个）
#   gold-frames/              可选 PNG 金标帧（每 step 无损，用于校验）

SHARD_MANIFEST_FILENAME = "shard_manifest.json"
ZARR_DIRNAME = "tensors.zarr"
EPISODES_PARQUET_FILENAME = "episodes.parquet"
TRANSITIONS_PARQUET_FILENAME = "transitions.parquet"
FRAME_INDEX_PARQUET_FILENAME = "frame_index.parquet"
VIDEO_FILENAME_PREFIX = "video-"
VIDEO_FILENAME_SUFFIX = ".mp4"
GOLD_FRAMES_DIRNAME = "gold-frames"

# frame_index.parquet 的列（recording 必须写出，dataset 必须按此读取）
FRAME_INDEX_COLUMNS = [
    "episode_id",
    "video_id",
    "frame_index",  # 视频帧序号，从 0 开始
    "timestep",  # 环境时间步（state[timestep]）
    "state_index",  # 在 tensors.zarr 时间维的全局偏移（可直接索引 state 数组；与 episodes.parquet 的 state_start/state_end 闭区间对应）
    "sim_time_ns",
    "is_initial_frame",
    "is_terminal_frame",
    "encoder_pts",  # 视频容器时间戳（仅信息性，不作对齐依据）
]

# transitions.parquet 的固定列（可为 null；额外自定义列允许）
TRANSITIONS_PARQUET_COLUMNS = [
    "session_id",
    "episode_id",
    "timestep",
    "action_id",
    "action_name",
    "action_source",
    "command_id",
    "reward",
    "terminated",
    "truncated",
    "is_sampled_frame",
    "is_initial_frame",
    "is_terminal_frame",
    "instruction",
    "task_id",
    "task_version",
    "event_tokens",  # JSON 字符串数组
    "sim_time_ns",
]

# episodes.parquet 的固定列
EPISODES_PARQUET_COLUMNS = [
    "session_id",
    "episode_id",
    "task_id",
    "task_version",
    "seed",
    "num_states",  # T+1
    "num_transitions",  # T
    "num_frames",
    "terminated",
    "truncated",
    "video_id",
    "state_start",  # 在 Zarr 时间维的起始偏移（含）
    "state_end",  # 在 Zarr 时间维的结束偏移（含）
    "start_wall_ns",
    "end_wall_ns",
]

# 状态扁平化后保留的顶层路径（recording.state_codec 使用）
# 值形如 "map"、"player_position"、"inventory/wood" 等；约定用 "/" 分隔路径段。

# tensors.zarr 内部布局（recording 写出 / dataset 读取的共同契约）
#   state/<path>          : [N_total_states, ...]（episode 内 state 数组；episodes.parquet 的
#                           state_start/state_end 为闭区间偏移）
#   actions               : [T_total] int32，action_id（作用于 state[t]）
#   rewards               : [T_total] float32
#   terminated            : [T_total] bool
#   truncated             : [T_total] bool
#   state_timesteps       : [N_total_states] int32，每个 state 的全局 timestep
#   state_episode_ids     : [N_total_states] object/int，每个 state 所属 episode 在 shard 内的
#                           episode 序号（0..n_episodes-1，与 episodes.parquet 行序一致）
#   event_tokens          : [T_total] object，事件 token 列表（JSON 字符串）
ZARR_STATE_GROUP = "state"
ZARR_ARRAY_ACTIONS = "actions"
ZARR_ARRAY_REWARDS = "rewards"
ZARR_ARRAY_TERMINATED = "terminated"
ZARR_ARRAY_TRUNCATED = "truncated"
ZARR_ARRAY_STATE_TIMESTEPS = "state_timesteps"
ZARR_ARRAY_STATE_EPISODE_IDS = "state_episode_ids"
ZARR_ARRAY_EVENT_TOKENS = "event_tokens"
