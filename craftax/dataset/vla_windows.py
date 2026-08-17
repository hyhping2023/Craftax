"""VLA 训练样本导出。

从 canonical shard 生成 ``(instruction, RGB 帧序列, action 序列)`` 定长窗口。
只消费 ``craftax.dataset.reader`` 暴露的只读接口，不修改任何 shard 文件。

对齐规则：

- 窗口只跨越同一 episode；帧按 frame_index 顺序从视频解码。
- 每帧关联其 timestep 上的 action（``action[t]`` 作用于 ``state[t]``）。
  终局帧（timestep == num_transitions）没有 action，包含它的窗口被跳过。
- ``frame_stride`` 对窗口内帧降采样：每 ``frame_stride`` 帧取 1 帧。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

from craftax.contracts import ActionSpec
from craftax.dataset.reader import EpisodeView, ShardReader


def vla_samples(
    shard_reader: ShardReader,
    window_len: int,
    frame_stride: int = 1,
    episode_ids: Optional[Sequence[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """产出固定帧长的 VLA 样本迭代器。

    参数:
        shard_reader: 只读 shard。
        window_len: 每个样本的帧数（视频帧，按 frame_index 计）。
        frame_stride: 帧降采样步长（每 stride 帧取 1）。
        episode_ids: 限制产出的 episode；None 表示全部。

    每样本 dict 字段::

        instruction   str                    episode 的 task instruction
        task_id       str
        frames        list[np.ndarray]       RGB 帧序列（按 frame_index 顺序）
        actions       list[ActionSpec]       与 frames 对齐的 action
        action_names  list[str]
        rewards       list[float]
        terminated    bool                   窗口末帧 transition 的 terminated
        episode_id    str
        timesteps     list[int]              与 frames 对齐的 timestep
    """
    if window_len <= 0:
        raise ValueError("window_len 必须为正")
    if frame_stride <= 0:
        raise ValueError("frame_stride 必须为正")

    allowed = set(episode_ids) if episode_ids is not None else None
    for episode in shard_reader.episodes():
        if allowed is not None and episode.episode_id not in allowed:
            continue
        yield from _episode_vla_samples(episode, window_len, frame_stride)


def _episode_vla_samples(
    episode: EpisodeView, window_len: int, frame_stride: int
) -> Iterator[Dict[str, Any]]:
    rows = episode.frame_rows()
    if len(rows) < window_len:
        return
    instruction = episode.instruction
    # 最后一个可用窗起点：需要覆盖 window_len 帧且末帧有 action。
    max_start = len(rows) - window_len
    for start in range(0, max_start + 1):
        selected = rows[start : start + window_len][::frame_stride]
        sample = _build_sample(episode, selected, instruction)
        if sample is not None:
            yield sample


def _build_sample(
    episode: EpisodeView,
    rows: List[Dict[str, Any]],
    instruction: str,
) -> Optional[Dict[str, Any]]:
    frames: List[Any] = []
    actions: List[ActionSpec] = []
    action_names: List[str] = []
    rewards: List[float] = []
    timesteps: List[int] = []
    frame_iter = episode.frames()  # 解码一次后缓存，供窗口内随机访问
    for row in rows:
        ts = int(row["timestep"])
        action = episode.action_at_timestep(ts)
        transition = episode.transition_at_timestep(ts)
        if action is None or transition is None:
            return None  # 含终局帧（无 action）的窗口不构成完整 VLA 样本
        frames.append(frame_iter.frame(int(row["frame_index"])).rgb)
        actions.append(ActionSpec(action[0], action[1]))
        action_names.append(action[1])
        rewards.append(float(transition["reward"]))
        timesteps.append(ts)
    last_transition = episode.transition_at_timestep(timesteps[-1])
    return {
        "instruction": instruction,
        "task_id": episode.task_id,
        "frames": frames,
        "actions": actions,
        "action_names": action_names,
        "rewards": rewards,
        "terminated": bool(last_transition["terminated"]) if last_transition else False,
        "episode_id": episode.episode_id,
        "timesteps": timesteps,
    }
