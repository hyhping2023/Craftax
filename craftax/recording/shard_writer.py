"""单 writer 封存 shard：Zarr 张量 + Parquet 元数据 + CFR MP4 + manifest。

目录布局（contracts.SHARD_* 常量）：
    tensors.zarr/          state/<path> 与 actions/rewards/terminated/truncated/
                           state_timesteps/state_episode_ids/event_tokens
    episodes.parquet       episode 边界与元数据
    transitions.parquet    每 transition 的 action 元数据 / 事件 / 指令
    frame_index.parquet    视频帧 -> timestep 显式映射
    video-<video_id>.mp4   每 episode 一个 CFR H.264 视频
    gold-frames/           可选 PNG 金标帧（按采样帧保存）
    shard_manifest.json    finalize() 时写入，之后 shard 不可变

时间轴不变量：num_states == num_transitions + 1。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from craftax.contracts import (
    EPISODES_PARQUET_COLUMNS,
    EPISODES_PARQUET_FILENAME,
    FRAME_INDEX_COLUMNS,
    FRAME_INDEX_PARQUET_FILENAME,
    TRANSITIONS_PARQUET_COLUMNS,
    TRANSITIONS_PARQUET_FILENAME,
    ZARR_ARRAY_ACTIONS,
    ZARR_ARRAY_EVENT_TOKENS,
    ZARR_ARRAY_REWARDS,
    ZARR_ARRAY_STATE_EPISODE_IDS,
    ZARR_ARRAY_STATE_TIMESTEPS,
    ZARR_ARRAY_TERMINATED,
    ZARR_ARRAY_TRUNCATED,
    ZARR_DIRNAME,
    ZARR_STATE_GROUP,
    FrameSampleConfig,
    TransitionRecord,
)
from craftax.recording.manifest import build_shard_manifest, write_manifest
from craftax.recording.state_codec import flatten_state, schema_hash
from craftax.recording.video_writer import VideoWriter

CHUNK_T = 128  # Zarr 时间维 chunk 大小

# 类型标注仅在 typing 层面使用，避免引入运行时依赖
try:  # pragma: no cover
    from zarr.codecs import VLenUTF8Codec
    from zarr.storage import LocalStore
except ImportError:  # pragma: no cover
    VLenUTF8Codec = None  # type: ignore
    LocalStore = None  # type: ignore


@dataclasses.dataclass
class FrameEntry:
    """一帧 RGB 及其 frame_index 行。"""

    rgb: np.ndarray  # uint8 HWC
    row: Dict[str, Any]  # frame_index.parquet 行（含 FRAME_INDEX_COLUMNS）


@dataclasses.dataclass
class EpisodeData:
    """一个 episode 的全部录制内容。states 长度 = transitions 长度 + 1。"""

    episode_id: str
    session_id: str
    task_id: str
    task_version: str
    seed: int
    states: List[Any]  # host 化 EnvState，长度 T+1
    transitions: List[TransitionRecord]  # 长度 T
    frames: List[FrameEntry]  # 采样帧（含初始帧与 terminal frame）
    terminated: bool
    truncated: bool
    start_wall_ns: int = 0
    end_wall_ns: int = 0
    video_id: str = ""


def make_shard_dir(spool_dir: str, run_id: str, producer_id: str, attempt_id: str) -> str:
    """按契约 7.9 创建 spool/{run_id}/{producer_id}/{attempt_id} 目录。"""
    path = Path(spool_dir) / run_id / producer_id / attempt_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _task_hash(task_id: str, task_version: str, instruction: str, objective: str) -> str:
    payload = json.dumps(
        {
            "task_id": task_id,
            "task_version": task_version,
            "instruction": instruction,
            "objective": objective,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class ShardWriter:
    """向一个 shard 追加 episode；finalize() 后 shard 不可变。"""

    def __init__(
        self,
        shard_dir: str,
        *,
        run_id: str = "default",
        producer_id: str = "recorder",
        attempt_id: str = "",
        task_id: str = "",
        task_version: str = "",
        task_hash: str = "",
        frame_sample: FrameSampleConfig,
        gold_frames: bool = False,
        shard_max_transitions: Optional[int] = None,
        video_fps: Optional[int] = None,
    ):
        frame_sample.validate()
        self.shard_dir = str(shard_dir)
        self.run_id = run_id
        self.producer_id = producer_id
        self.attempt_id = attempt_id or uuid.uuid4().hex[:8]
        self.task_id = task_id
        self.task_version = task_version
        self.task_hash = task_hash
        self.frame_sample = frame_sample
        self.gold_frames = gold_frames
        self.shard_max_transitions = shard_max_transitions
        self.video_fps = int(video_fps or frame_sample.video_fps)
        self.shard_id = f"shard-{run_id[:12]}-{self.attempt_id}"

        self._sealed = False
        self._num_states = 0
        self._num_transitions = 0
        self._num_frames = 0
        self._num_videos = 0
        self._episode_index = 0
        self._state_schema_hash = ""
        self._state_arrays: Dict[str, Any] = {}

        os.makedirs(self.shard_dir, exist_ok=True)
        self._zarr = self._open_zarr()
        # 固定形状数组
        self._arr_actions = self._zarr.create_array(
            ZARR_ARRAY_ACTIONS, shape=(0,), dtype=np.int32, chunks=(CHUNK_T,), overwrite=True
        )
        self._arr_rewards = self._zarr.create_array(
            ZARR_ARRAY_REWARDS, shape=(0,), dtype=np.float32, chunks=(CHUNK_T,), overwrite=True
        )
        self._arr_terminated = self._zarr.create_array(
            ZARR_ARRAY_TERMINATED, shape=(0,), dtype=np.bool_, chunks=(CHUNK_T,), overwrite=True
        )
        self._arr_truncated = self._zarr.create_array(
            ZARR_ARRAY_TRUNCATED, shape=(0,), dtype=np.bool_, chunks=(CHUNK_T,), overwrite=True
        )
        self._arr_timesteps = self._zarr.create_array(
            ZARR_ARRAY_STATE_TIMESTEPS, shape=(0,), dtype=np.int32, chunks=(CHUNK_T,), overwrite=True
        )
        self._arr_episode_ids = self._zarr.create_array(
            ZARR_ARRAY_STATE_EPISODE_IDS, shape=(0,), dtype=np.int32, chunks=(CHUNK_T,), overwrite=True
        )
        if VLenUTF8Codec is not None:
            self._arr_event_tokens = self._zarr.create_array(
                ZARR_ARRAY_EVENT_TOKENS,
                shape=(0,),
                dtype=str,
                chunks=(CHUNK_T,),
                serializer=VLenUTF8Codec(),
                overwrite=True,
            )
        else:  # pragma: no cover - 极老 zarr 兼容路径
            self._arr_event_tokens = self._zarr.create_array(
                ZARR_ARRAY_EVENT_TOKENS, shape=(0,), dtype=object, chunks=(CHUNK_T,), overwrite=True
            )

        self._episode_rows: List[Dict[str, Any]] = []
        self._transition_rows: List[Dict[str, Any]] = []
        self._frame_rows: List[Dict[str, Any]] = []

    # -- Zarr -------------------------------------------------------------

    def _open_zarr(self):
        zarr_dir = os.path.join(self.shard_dir, ZARR_DIRNAME)
        store = LocalStore(zarr_dir)
        group = zarr_module_open_group(store)
        return group

    def _state_group(self, path: str) -> Any:
        """state/<path> 所在 group；自动创建中间分组。"""
        group = self._zarr.require_group(ZARR_STATE_GROUP)
        parts = path.split("/")
        for part in parts[:-1]:
            group = group.require_group(part)
        return group, parts[-1]

    def _ensure_state_array(self, path: str, first_value: np.ndarray) -> Any:
        arr = self._state_arrays.get(path)
        if arr is None:
            group, name = self._state_group(path)
            arr = group.create_array(
                name,
                shape=(0, *first_value.shape),
                dtype=first_value.dtype,
                chunks=(CHUNK_T, *first_value.shape),
                overwrite=True,
            )
            self._state_arrays[path] = arr
        return arr

    def _append_state(self, flattened: Dict[str, np.ndarray]) -> None:
        for path, value in sorted(flattened.items()):
            arr = self._ensure_state_array(path, value)
            row = np.asarray(value).reshape(1, *value.shape)
            if row.dtype != arr.dtype:
                row = row.astype(arr.dtype)
            arr.append(row)

    # -- 对外接口 ----------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_transitions(self) -> int:
        return self._num_transitions

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def is_full(self) -> bool:
        return (
            self.shard_max_transitions is not None
            and self._num_transitions >= self.shard_max_transitions
        )

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def add_episode(self, episode: EpisodeData) -> None:
        """写入一个 episode 的 Zarr 张量、视频与 Parquet 行。"""
        if self._sealed:
            raise RuntimeError("shard 已封存，不能再写入")
        transitions = episode.transitions
        states = episode.states
        if len(states) != len(transitions) + 1:
            raise ValueError(
                f"episode {episode.episode_id}: len(states)={len(states)} "
                f"必须等于 len(transitions)+1={len(transitions) + 1}"
            )

        # 1) state 张量
        for i, state in enumerate(states):
            flattened = flatten_state(state)
            if i == 0 and not self._state_schema_hash:
                self._state_schema_hash = schema_hash(flattened)
            self._append_state(flattened)
            self._num_states += 1
            timestep = int(np.asarray(getattr(state, "timestep", 0)))
            self._arr_timesteps.append(np.array([timestep], dtype=np.int32))
            self._arr_episode_ids.append(np.array([self._episode_index], dtype=np.int32))

        # 2) transition 张量（action/reward/done/event_tokens）
        actions = np.array([tr.action.id for tr in transitions], dtype=np.int32)
        rewards = np.array([float(tr.reward) for tr in transitions], dtype=np.float32)
        terminated = np.array([bool(tr.terminated) for tr in transitions], dtype=np.bool_)
        truncated = np.array([bool(tr.truncated) for tr in transitions], dtype=np.bool_)
        if actions.size:
            self._arr_actions.append(actions)
            self._arr_rewards.append(rewards)
            self._arr_terminated.append(terminated)
            self._arr_truncated.append(truncated)
            self._arr_event_tokens.append(
                np.array(
                    [json.dumps(tr.event_tokens, ensure_ascii=False) for tr in transitions],
                    dtype=str,
                )
            )
        self._num_transitions += len(transitions)

        # 3) 视频（每 episode 一个）
        video_id = episode.video_id or f"ep{self._episode_index:06d}"
        if episode.frames:
            first = episode.frames[0].rgb
            writer = VideoWriter(
                self.shard_dir, video_id, self.video_fps, first.shape[1], first.shape[0]
            )
            for fe in episode.frames:
                writer.add_frame(fe.rgb)
            writer.close()
            self._num_videos += 1
            expected = len(episode.frames)
            if writer.frame_count != expected:  # pragma: no cover - 防御性检查
                raise RuntimeError(
                    f"video {video_id} 写入帧数 {writer.frame_count} != 预期 {expected}"
                )

        # 4) gold-frames（可选，按采样帧保存 PNG）
        if self.gold_frames and episode.frames:
            gold_dir = os.path.join(self.shard_dir, "gold-frames", episode.episode_id)
            os.makedirs(gold_dir, exist_ok=True)
            for fe in episode.frames:
                from PIL import Image  # 惰性 import

                timestep = int(fe.row["timestep"])
                fi = int(fe.row["frame_index"])
                Image.fromarray(fe.rgb).save(
                    os.path.join(gold_dir, f"frame-{fi:04d}-t{timestep}.png")
                )

        # 5) Parquet 行
        state_start = self._num_states - len(states)
        state_end = self._num_states - 1
        self._episode_rows.append(
            {
                "session_id": episode.session_id,
                "episode_id": episode.episode_id,
                "task_id": episode.task_id or self.task_id,
                "task_version": episode.task_version or self.task_version,
                "seed": int(episode.seed),
                "num_states": len(states),
                "num_transitions": len(transitions),
                "num_frames": len(episode.frames),
                "terminated": bool(episode.terminated),
                "truncated": bool(episode.truncated),
                "video_id": video_id,
                "state_start": state_start,
                "state_end": state_end,
                "start_wall_ns": int(episode.start_wall_ns),
                "end_wall_ns": int(episode.end_wall_ns),
            }
        )
        for tr in transitions:
            self._transition_rows.append(self._transition_row(tr, episode.episode_id))
        for fe in episode.frames:
            # state_index 为 Zarr 时间维的全局偏移：episode 内偏移 + state_start
            row = dict(fe.row)
            row["state_index"] = int(row["state_index"]) + state_start
            self._frame_rows.append(row)

        self._num_frames += len(episode.frames)
        self._episode_index += 1

    @staticmethod
    def _transition_row(tr: TransitionRecord, episode_id: str) -> Dict[str, Any]:
        return {
            "session_id": tr.session_id,
            "episode_id": episode_id,
            "timestep": int(tr.timestep),
            "action_id": int(tr.action.id),
            "action_name": tr.action.name,
            "action_source": tr.action_source,
            "command_id": tr.command_id,
            "reward": float(tr.reward),
            "terminated": bool(tr.terminated),
            "truncated": bool(tr.truncated),
            "is_sampled_frame": bool(tr.is_sampled_frame),
            "is_initial_frame": bool(tr.is_initial_frame),
            "is_terminal_frame": bool(tr.is_terminal_frame),
            "instruction": tr.instruction,
            "task_id": tr.task_id,
            "task_version": tr.task_version,
            "event_tokens": json.dumps(tr.event_tokens, ensure_ascii=False),
            "sim_time_ns": int(tr.sim_time_ns),
        }

    # -- 封存 --------------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """写 Parquet 与 manifest，封存 shard（此后不可再写）。"""
        if self._sealed:
            from craftax.recording.manifest import load_manifest

            return load_manifest(self.shard_dir)
        self._write_parquets()
        task_hash = self.task_hash or _task_hash(
            self.task_id, self.task_version, "", ""
        )
        manifest = build_shard_manifest(
            self.shard_dir,
            shard_id=self.shard_id,
            run_id=self.run_id,
            producer_id=self.producer_id,
            attempt_id=self.attempt_id,
            task_id=self.task_id,
            task_version=self.task_version,
            task_hash=task_hash,
            state_schema_hash=self._state_schema_hash,
            frame_sample=self.frame_sample.to_dict(),
            counts={
                "num_episodes": self._episode_index,
                "num_transitions": self._num_transitions,
                "num_states": self._num_states,
                "num_frames": self._num_frames,
                "num_videos": self._num_videos,
            },
        )
        write_manifest(manifest, self.shard_dir)
        self._sealed = True
        return manifest

    def _write_parquets(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for filename, columns, rows in (
            (EPISODES_PARQUET_FILENAME, EPISODES_PARQUET_COLUMNS, self._episode_rows),
            (TRANSITIONS_PARQUET_FILENAME, TRANSITIONS_PARQUET_COLUMNS, self._transition_rows),
            (FRAME_INDEX_PARQUET_FILENAME, FRAME_INDEX_COLUMNS, self._frame_rows),
        ):
            normalized = [{col: row.get(col) for col in columns} for row in rows]
            table = pa.Table.from_pylist(normalized)
            pq.write_table(table, os.path.join(self.shard_dir, filename))


def zarr_module_open_group(store):
    """小封装：隔离 zarr 导入，便于测试替换。"""
    import zarr

    return zarr.open_group(store=store, mode="a", zarr_format=3)
