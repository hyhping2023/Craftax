"""World Model 训练样本导出。

从 canonical shard 生成连续 ``(state_t, frame_t, action_t, state_t+1, event_t)``
序列。只消费 ``craftax.dataset.reader`` 的只读接口。

对齐规则（embodied_environment_plan.md 7.2 节）：

- 一个窗口覆盖 timestep t0..t0+L：states L+1 个、actions L 个、next_states L 个、
  rewards L 个、events L 个。
- 帧只能出现在 frame_index 中存在的 sampled timestep；``require_frames=True``
  时窗口锚定在 sampled timestep（t0 必有帧）。若 sampled timestep 应有帧而
  数据缺失，跳过该窗口并记录日志。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence

from craftax.contracts import ActionSpec
from craftax.dataset.reader import EpisodeView, ShardReader, parse_event_tokens

logger = logging.getLogger(__name__)


def wm_samples(
    shard_reader: ShardReader,
    window_len: int,
    require_frames: bool = True,
    episode_ids: Optional[Sequence[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """产出 World Model 序列样本迭代器。

    参数:
        shard_reader: 只读 shard。
        window_len: 窗口内 transition 数 L（states = L+1）。
        require_frames: True 时窗口锚定在 sampled timestep（t0 有帧）；
            False 时锚定在每个 transition timestep。
        episode_ids: 限制产出的 episode；None 表示全部。

    每样本 dict 字段::

        states           list[dict[str, np.ndarray]]   timestep t0..t0+L 的状态
        frames           list[np.ndarray]              对齐的帧（仅 sampled timestep）
        frame_timesteps  list[int]                     frames 对应的 timestep
        actions          list[ActionSpec]              t0..t0+L-1 的 action
        next_states      list[dict[str, np.ndarray]]   states[1:]（action 的转移结果）
        events           list[list[str]]               t0..t0+L-1 的 event token
        rewards          list[float]
        terminated       bool                          末个 transition 的 terminated
        episode_id       str
        timesteps        list[int]                     t0..t0+L
    """
    if window_len <= 0:
        raise ValueError("window_len 必须为正")

    allowed = set(episode_ids) if episode_ids is not None else None
    for episode in shard_reader.episodes():
        if allowed is not None and episode.episode_id not in allowed:
            continue
        yield from _episode_wm_samples(episode, window_len, require_frames)


def _episode_wm_samples(
    episode: EpisodeView, window_len: int, require_frames: bool
) -> Iterator[Dict[str, Any]]:
    num_transitions = episode.num_transitions
    if num_transitions < window_len:
        return
    max_anchor = num_transitions - window_len  # 保证 t0+L <= T
    if require_frames:
        # 只锚定在 frame_index 中出现的 sampled timestep，且留足 L 个 transition。
        anchors = sorted(
            {
                int(r["timestep"])
                for r in episode.frame_rows()
                if int(r["timestep"]) <= max_anchor
            }
        )
    else:
        anchors = list(range(0, max_anchor + 1))

    frame_ts = {int(r["timestep"]) for r in episode.frame_rows()}
    skipped = 0
    for t0 in anchors:
        sample = _build_sample(episode, t0, window_len, require_frames, frame_ts)
        if sample is None:
            skipped += 1
            continue
        yield sample
    if skipped:
        logger.warning(
            "episode %s: 因帧缺失跳过 %d 个 WM 窗口", episode.episode_id, skipped
        )


def _build_sample(
    episode: EpisodeView,
    t0: int,
    window_len: int,
    require_frames: bool,
    frame_ts: set[int],
) -> Optional[Dict[str, Any]]:
    t_end = t0 + window_len
    states = episode.states_at_timesteps(range(t0, t_end + 1))
    if states is None:
        return None

    actions: List[ActionSpec] = []
    rewards: List[float] = []
    events: List[List[str]] = []
    for t in range(t0, t_end):
        transition = episode.transition_at_timestep(t)
        if transition is None:
            return None
        actions.append(
            ActionSpec(int(transition["action_id"]), str(transition["action_name"]))
        )
        rewards.append(float(transition["reward"]))
        events.append(parse_event_tokens(transition.get("event_tokens")))

    frames: List[Any] = []
    frame_timesteps: List[int] = []
    for t in range(t0, t_end + 1):
        if t not in frame_ts:
            continue  # 非 sampled timestep 无帧，帧对齐由 frame_timesteps 显式给出
        frame_rgb = episode.frame_at_timestep(t)
        if frame_rgb is None:
            if require_frames:
                return None  # sampled timestep 应有帧却缺失：跳过并标注
            continue
        frames.append(frame_rgb)
        frame_timesteps.append(t)
    if require_frames and not frame_timesteps:
        return None

    last_transition = episode.transition_at_timestep(t_end - 1)
    return {
        "states": states,
        "frames": frames,
        "frame_timesteps": frame_timesteps,
        "actions": actions,
        "next_states": states[1:],
        "events": events,
        "rewards": rewards,
        "terminated": bool(last_transition["terminated"]) if last_transition else False,
        "episode_id": episode.episode_id,
        "timesteps": list(range(t0, t_end + 1)),
    }
