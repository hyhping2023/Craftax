"""帧采样判定与 frame_index 行构造（MP4 -> timestep 显式映射）。

时间轴规则（embodied_environment_plan.md 7.3）：
- frame_index=0 对应 reset 后的 state[0]（is_initial_frame）；
- 常规帧只在 timestep % R == 0 时保存（R = S // F），对应 state[timestep]；
- terminal state 即使不落在采样周期上，也必须单独保存为 terminal frame；
- encoder_pts 仅信息性（frame_index / F），不作对齐依据。
"""
from __future__ import annotations

from typing import Any, Dict

from craftax.contracts import FRAME_INDEX_COLUMNS, FrameSampleConfig


def decide_sample(
    config: FrameSampleConfig,
    timestep: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> bool:
    """该 timestep 是否需要保存一帧。terminal/truncated 强制采样。"""
    if timestep == 0:
        return True
    if terminated or truncated:
        return True
    return config.is_sampled(timestep)


def frame_flags(
    config: FrameSampleConfig,
    timestep: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> Dict[str, bool]:
    """返回该帧的初始/终局标志。终局帧优先标记（即使落在常规采样周期上）。"""
    is_terminal = bool(terminated or truncated)
    return {
        "is_initial_frame": timestep == 0,
        "is_terminal_frame": is_terminal,
    }


def frame_index_row(
    config: FrameSampleConfig,
    *,
    episode_id: str,
    video_id: str,
    frame_index: int,
    timestep: int,
    state_index: int,
    sim_time_ns: int = 0,
    terminated: bool = False,
    truncated: bool = False,
) -> Dict[str, Any]:
    """构造 frame_index.parquet 的一行（列名与 contracts.FRAME_INDEX_COLUMNS 一致）。"""
    flags = frame_flags(config, timestep, terminated=terminated, truncated=truncated)
    return {
        "episode_id": episode_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "timestep": int(timestep),
        "state_index": int(state_index),
        "sim_time_ns": int(sim_time_ns),
        "is_initial_frame": flags["is_initial_frame"],
        "is_terminal_frame": flags["is_terminal_frame"],
        "encoder_pts": float(frame_index / config.video_fps),
    }


def frame_index_columns() -> list:
    return list(FRAME_INDEX_COLUMNS)


class FrameIndexBuilder:
    """按 episode 递增生成 frame_index 行，管理视频帧序号与采样判定。

    用法：builder.reset() -> 对每个 state 调用 should_sample()/row()。
    """

    def __init__(self, config: FrameSampleConfig, *, episode_id: str, video_id: str):
        self.config = config
        self.episode_id = episode_id
        self.video_id = video_id
        self._next_frame_index = 0
        self._num_frames = 0

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def should_sample(
        self,
        timestep: int,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> bool:
        return decide_sample(self.config, timestep, terminated=terminated, truncated=truncated)

    def row(
        self,
        *,
        timestep: int,
        state_index: int,
        sim_time_ns: int = 0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> Dict[str, Any]:
        row = frame_index_row(
            self.config,
            episode_id=self.episode_id,
            video_id=self.video_id,
            frame_index=self._next_frame_index,
            timestep=timestep,
            state_index=state_index,
            sim_time_ns=sim_time_ns,
            terminated=terminated,
            truncated=truncated,
        )
        self._next_frame_index += 1
        self._num_frames += 1
        return row
