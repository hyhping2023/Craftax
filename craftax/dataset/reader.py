"""Canonical shard 只读访问器（VLA / World Model 训练样本导出的基础层）。

只读消费 recording 模块产出的 sealed shard。目录布局、Parquet 列名与 Zarr
布局常量定义在 ``craftax.contracts``（本模块不修改它），也不 import recording /
service / tasks 的实现模块；依赖仅限标准库 + zarr + pyarrow + imageio + numpy。

时间轴语义（embodied_environment_plan.md 7.2 节）::

    action[t] 作用于 state[t] -> reward[t], terminated[t], truncated[t], state[t+1]
    frame[t] 描述 state[t]

视频帧 -> timestep/state 的映射由 frame_index.parquet 显式记录，reader 不根据
视频容器的 PTS 推断对齐关系。

本模块的读取约定（与 recording 写序一致，作为假设记录在案）：

- transitions.parquet / frame_index.parquet 以 ``episode_id`` 列过滤出 episode
  行后按 ``timestep`` / ``frame_index`` 升序排序；episode 内 timestep 从 0 连续
  递增（reset 后从 0 开始），``(episode_id, timestep)`` 唯一。
- 状态数组按 episodes.parquet 的 ``state_start``/``state_end``（闭区间）在
  tensors.zarr 的 ``state/<path>`` 数组中定位；``state_timesteps`` 提供
  state -> timestep 的显式映射。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import imageio.v2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from craftax.contracts import (
    EPISODES_PARQUET_FILENAME,
    FRAME_INDEX_PARQUET_FILENAME,
    SHARD_MANIFEST_FILENAME,
    TRANSITIONS_PARQUET_FILENAME,
    VIDEO_FILENAME_PREFIX,
    VIDEO_FILENAME_SUFFIX,
    ZARR_ARRAY_STATE_TIMESTEPS,
    ZARR_DIRNAME,
    ZARR_STATE_GROUP,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def collect_state_paths(group: "zarr.Group") -> List[str]:
    """递归收集 state 组内全部数组路径（含嵌套分组，如 "inventory/wood"）。

    ``Group.array_keys()`` 只返回直接子数组，不进入嵌套分组；recording 按
    flattened 路径（"path/to/field"）写出的嵌套数组必须递归才能枚举到。
    """
    paths: List[str] = []

    def walk(node: "zarr.Group", prefix: str) -> None:
        for name, child in node.members():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(child, zarr.Group):
                walk(child, path)
            else:
                paths.append(path)

    walk(group, "")
    return paths


def parse_event_tokens(raw: Any) -> List[str]:
    """把 event_tokens 的原始值归一化为 str 列表。

    契约规定 event_tokens 为 JSON 字符串数组；zarr v3 中可能以 StringDType /
    v2 object（pickle 后反序列化为 list）保存，Parquet 侧为 JSON 字符串。
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return [raw.decode("utf-8", "replace")]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, list):
                return [str(x) for x in value]
        return [text]
    return [str(raw)]


@dataclass(frozen=True)
class Frame:
    """一帧解码后的视频帧与其显式 timestep 映射。"""

    frame_index: int
    timestep: int
    rgb: np.ndarray  # uint8 HWC RGB


class FrameIter:
    """按 frame_index 顺序解码视频帧的迭代器。

    顺序迭代为流式解码（逐帧 yield）；随机访问 ``frame(index)`` 会解码整个
    视频并缓存（标注复杂度 O(解码一次)），后续按索引 O(len(rows)) 查找。
    解码帧数必须严格等于 frame_index 行数，否则视为数据完整性错误。
    """

    def __init__(
        self, video_path: Optional[Path], frame_rows: List[Dict[str, Any]]
    ) -> None:
        self.video_path = video_path
        self.frame_rows = frame_rows
        self._decoded: Optional[List[np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.frame_rows)

    def __iter__(self) -> Iterator[Frame]:
        if self._decoded is not None:
            for i, row in enumerate(self.frame_rows):
                yield Frame(
                    frame_index=int(row["frame_index"]),
                    timestep=int(row["timestep"]),
                    rgb=self._decoded[i],
                )
            return
        reader = self._open_reader()
        n = 0
        try:
            for img in reader:
                if n >= len(self.frame_rows):
                    raise self._count_mismatch_error(n + 1)
                row = self.frame_rows[n]
                yield Frame(
                    frame_index=int(row["frame_index"]),
                    timestep=int(row["timestep"]),
                    rgb=np.asarray(img),
                )
                n += 1
            if n != len(self.frame_rows):
                raise self._count_mismatch_error(n)
        finally:
            reader.close()

    def frame(self, frame_index: int) -> Frame:
        decoded = self._decode_all()
        for i, row in enumerate(self.frame_rows):
            if int(row["frame_index"]) == int(frame_index):
                return Frame(
                    frame_index=int(row["frame_index"]),
                    timestep=int(row["timestep"]),
                    rgb=decoded[i],
                )
        raise KeyError(f"frame_index {frame_index} 不在 frame_index.parquet 中")

    def _open_reader(self):
        if self.video_path is None or not self.video_path.exists():
            raise FileNotFoundError(f"episode 视频不存在: {self.video_path}")
        return imageio.v2.get_reader(str(self.video_path), format="ffmpeg")

    def _decode_all(self) -> List[np.ndarray]:
        if self._decoded is None:
            reader = self._open_reader()
            try:
                frames = [np.asarray(img) for img in reader]
            finally:
                reader.close()
            if len(frames) != len(self.frame_rows):
                raise self._count_mismatch_error(len(frames))
            self._decoded = frames
        return self._decoded

    def _count_mismatch_error(self, decoded: int) -> RuntimeError:
        return RuntimeError(
            f"视频解码帧数({decoded})与 frame_index 行数({len(self.frame_rows)})不一致: "
            f"{self.video_path}"
        )


class EpisodeView:
    """单个 episode 的只读视图。

    状态数组按 episode 内 state_index 定位 Zarr（state_start 为闭区间起始）；
    索引方法（``state_at_timestep`` / ``action_at_timestep``）以 timestep 查询。
    """

    def __init__(self, shard: "ShardReader", row_index: int) -> None:
        self._shard = shard
        self._row_index = row_index
        self._row = shard._episode_rows[row_index]
        self._trans_rows: Optional[List[Dict[str, Any]]] = None
        self._trans_by_ts: Optional[Dict[int, Dict[str, Any]]] = None
        self._frame_rows: Optional[List[Dict[str, Any]]] = None
        self._frame_by_index: Optional[Dict[int, Dict[str, Any]]] = None
        self._state_ts: Optional[Dict[int, int]] = None
        self._frame_iter: Optional[FrameIter] = None

    # ------------------------------------------------------------------ 元数据

    @property
    def meta(self) -> Dict[str, Any]:
        """episodes.parquet 的原始行（Python 标量）。"""
        return dict(self._row)

    @property
    def session_id(self) -> str:
        return str(self._row["session_id"])

    @property
    def episode_id(self) -> str:
        return str(self._row["episode_id"])

    @property
    def task_id(self) -> str:
        return str(self._row["task_id"])

    @property
    def task_version(self) -> str:
        return str(self._row.get("task_version") or "")

    @property
    def seed(self) -> int:
        return int(self._row.get("seed") or 0)

    @property
    def num_states(self) -> int:
        return int(self._row["num_states"])

    @property
    def num_transitions(self) -> int:
        return int(self._row["num_transitions"])

    @property
    def num_frames(self) -> int:
        return int(self._row.get("num_frames") or 0)

    @property
    def terminated(self) -> bool:
        return bool(self._row.get("terminated") or False)

    @property
    def truncated(self) -> bool:
        return bool(self._row.get("truncated") or False)

    @property
    def video_id(self) -> Optional[str]:
        vid = self._row.get("video_id")
        return str(vid) if vid else None

    @property
    def video_path(self) -> Optional[Path]:
        vid = self.video_id
        if vid is None:
            return None
        return (
            self._shard.shard_dir
            / f"{VIDEO_FILENAME_PREFIX}{vid}{VIDEO_FILENAME_SUFFIX}"
        )

    @property
    def state_start(self) -> int:
        """在 Zarr 时间维的起始偏移（闭区间，含）。"""
        return int(self._row["state_start"])

    @property
    def state_end(self) -> int:
        """在 Zarr 时间维的结束偏移（闭区间，含）。"""
        return int(self._row["state_end"])

    @property
    def instruction(self) -> str:
        """episode 的 task instruction：取 transitions 中第一个非空指令。"""
        for row in self.transition_rows():
            text = (row.get("instruction") or "").strip()
            if text:
                return text
        return ""

    # ------------------------------------------------------------- transition

    def transitions(self) -> pa.Table:
        """本 episode 的 transitions.parquet 行（按 timestep 升序）。"""
        table = self._shard.transitions_table.filter(
            pc.field("episode_id") == self.episode_id
        )
        order = pc.sort_indices(table, sort_keys=[("timestep", "ascending")])
        return table.take(order)

    def transition_rows(self) -> List[Dict[str, Any]]:
        """本 episode 的 transition 行（dict 列表，按 timestep 升序）。"""
        if self._trans_rows is None:
            self._trans_rows = self.transitions().to_pylist()
            self._trans_by_ts = {int(r["timestep"]): r for r in self._trans_rows}
            expected = list(range(self.num_transitions))
            actual = [int(r["timestep"]) for r in self._trans_rows]
            if actual != expected:
                logger.warning(
                    "episode %s 的 transition timestep 不连续（%s != %s），"
                    "写入方可能未满足契约的 episode 内 timestep 从 0 递增假设",
                    self.episode_id,
                    actual,
                    expected,
                )
        return self._trans_rows

    def actions(self) -> Tuple[np.ndarray, List[str]]:
        """每 transition 的 (action_id 数组, action_name 列表)。"""
        rows = self.transition_rows()
        ids = np.asarray([int(r["action_id"]) for r in rows], dtype=np.int64)
        names = [str(r["action_name"]) for r in rows]
        return ids, names

    def transition_at_timestep(self, t: int) -> Optional[Dict[str, Any]]:
        """timestep t 的 transition 行；无 transition（如终局）返回 None。"""
        self.transition_rows()
        return self._trans_by_ts.get(int(t))

    def action_at_timestep(self, t: int) -> Optional[Tuple[int, str]]:
        """timestep t 的 (action_id, action_name)；终局步无 action 返回 None。"""
        row = self.transition_at_timestep(t)
        if row is None:
            return None
        return int(row["action_id"]), str(row["action_name"])

    # ---------------------------------------------------------------- state

    def states(
        self, start: int = 0, end: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """读取 episode 内 state_index 区间 [start, end) 的状态。

        返回 ``{state_path: ndarray}``；state_path 为扁平化状态树路径
        （如 ``"map"``、``"inventory/wood"``），数组首维为区间长度。
        """
        start = max(0, int(start))
        end = self.num_states if end is None else min(self.num_states, int(end))
        lo = self.state_start + start
        hi = self.state_start + end
        return {
            path: np.asarray(self._shard.state_group[path][lo:hi])
            for path in self._shard.state_paths
        }

    def state_index_by_timestep(self) -> Dict[int, int]:
        """timestep -> episode 内 state_index 的映射（来自 state_timesteps）。"""
        if self._state_ts is None:
            arr = self._shard.zarr_root[ZARR_ARRAY_STATE_TIMESTEPS]
            timesteps = np.asarray(
                arr[self.state_start : self.state_end + 1], dtype=np.int64
            )
            self._state_ts = {int(v): i for i, v in enumerate(timesteps)}
        return self._state_ts

    def state_at_timestep(self, t: int) -> Optional[Dict[str, np.ndarray]]:
        """timestep t 的单个状态（dict）；该 timestep 无 state 时返回 None。"""
        local = self.state_index_by_timestep().get(int(t))
        if local is None:
            return None
        return self.states(local, local + 1)

    def states_at_timesteps(
        self, timesteps: Sequence[int]
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """按 timestep 列表批量读状态；任一 timestep 缺失返回 None。

        与逐个 ``state_at_timestep`` 相比，只对 Zarr 做一次区间读取。
        """
        index = self.state_index_by_timestep()
        locals_: List[int] = []
        for t in timesteps:
            v = index.get(int(t))
            if v is None:
                return None
            locals_.append(v)
        lo, hi = min(locals_), max(locals_) + 1
        raw = self.states(lo, hi)
        rel = np.asarray([i - lo for i in locals_])
        return [{p: np.asarray(raw[p][j : j + 1]) for p in raw} for j in rel]

    # ----------------------------------------------------------------- frame

    def frame_rows(self) -> List[Dict[str, Any]]:
        """本 episode 的 frame_index 行（按 frame_index 升序）。"""
        if self._frame_rows is None:
            table = self._shard.frame_index_table.filter(
                pc.field("episode_id") == self.episode_id
            )
            order = pc.sort_indices(table, sort_keys=[("frame_index", "ascending")])
            self._frame_rows = table.take(order).to_pylist()
            self._frame_by_index = {int(r["frame_index"]): r for r in self._frame_rows}
        return self._frame_rows

    def frames(self) -> FrameIter:
        """按 frame_index 顺序解码视频帧。"""
        if self._frame_iter is None:
            self._frame_iter = FrameIter(self.video_path, self.frame_rows())
        return self._frame_iter

    def frame(self, frame_index: int) -> Frame:
        """随机访问指定 frame_index 的视频帧（解码一次后缓存）。"""
        return self.frames().frame(frame_index)

    def frame_at_timestep(self, t: int) -> Optional[np.ndarray]:
        """timestep t 对应的视频帧 RGB；该 timestep 无帧返回 None。"""
        for row in self.frame_rows():
            if int(row["timestep"]) == int(t):
                return self.frames().frame(int(row["frame_index"])).rgb
        return None


class ShardReader:
    """只读打开一个 sealed shard 目录。

    shard 目录内容见 ``craftax.contracts`` 的布局注释；本类只读，不修改任何文件。
    """

    def __init__(self, shard_dir: PathLike) -> None:
        self.shard_dir = Path(shard_dir)
        manifest_path = self.shard_dir / SHARD_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"缺少 {SHARD_MANIFEST_FILENAME}: {manifest_path}")
        self.manifest: Dict[str, Any] = json.loads(manifest_path.read_text("utf-8"))

        zarr_dir = self.shard_dir / ZARR_DIRNAME
        if not zarr_dir.exists():
            raise FileNotFoundError(f"缺少 Zarr 目录: {zarr_dir}")
        self.zarr_root = zarr.open_group(str(zarr_dir), mode="r")
        self.state_group = self.zarr_root[ZARR_STATE_GROUP]
        self.state_paths: List[str] = sorted(collect_state_paths(self.state_group))
        if not self.state_paths:
            logger.warning("shard %s 的 state 组为空", self.shard_dir)

        self.episodes_table = pq.read_table(self.shard_dir / EPISODES_PARQUET_FILENAME)
        self.transitions_table = pq.read_table(
            self.shard_dir / TRANSITIONS_PARQUET_FILENAME
        )
        self.frame_index_table = pq.read_table(
            self.shard_dir / FRAME_INDEX_PARQUET_FILENAME
        )

        self._episode_rows = self.episodes_table.to_pylist()
        self._episode_lookup: Dict[str, int] = {}
        for i, row in enumerate(self._episode_rows):
            self._episode_lookup[str(row["episode_id"])] = i

    @classmethod
    def open(cls, shard_dir: PathLike) -> "ShardReader":
        """打开 sealed shard 的便捷入口。"""
        return cls(shard_dir)

    @property
    def shard_id(self) -> str:
        return str(self.manifest.get("shard_id") or self.shard_dir.name)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_rows)

    def __len__(self) -> int:
        return self.num_episodes

    def episodes(self) -> Iterator[EpisodeView]:
        """遍历 shard 内全部 episode（按 episodes.parquet 行序）。"""
        for i, row in enumerate(self._episode_rows):
            yield EpisodeView(self, self._episode_lookup[str(row["episode_id"])])

    def get_episode(self, episode_id: str) -> EpisodeView:
        if episode_id not in self._episode_lookup:
            raise KeyError(f"shard 中不存在 episode: {episode_id}")
        return EpisodeView(self, self._episode_lookup[episode_id])

    def close(self) -> None:
        """兼容接口；zarr v3 无显式 close 语义。"""


class DatasetReader:
    """遍历 dataset_root（revisions/shards 或直接含多个 shard）下的 sealed shard。

    ``list_shards()`` 递归查找含 ``shard_manifest.json`` 的目录；``open_shard``
    按目录名（shard_id）打开。
    """

    def __init__(self, dataset_root: PathLike) -> None:
        self.dataset_root = Path(dataset_root)
        self._shard_dirs: Dict[str, Path] = {}
        if (self.dataset_root / SHARD_MANIFEST_FILENAME).exists():
            self._shard_dirs[self.dataset_root.name] = self.dataset_root
        for manifest in sorted(self.dataset_root.rglob(SHARD_MANIFEST_FILENAME)):
            self._shard_dirs[manifest.parent.name] = manifest.parent

    def list_shards(self) -> List[str]:
        return sorted(self._shard_dirs)

    def open_shard(self, shard_id: str) -> ShardReader:
        if shard_id not in self._shard_dirs:
            raise KeyError(f"未找到 shard: {shard_id}")
        return ShardReader(self._shard_dirs[shard_id])

    def __len__(self) -> int:
        return len(self._shard_dirs)
