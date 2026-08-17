"""sealed shard 结构校验器。

输入：已封存的 shard 目录。校验：
- validate_timeline：episode 内 num_states == num_transitions+1；transitions 行数 ==
  Zarr actions 长度；state_start/state_end 与 state 数组长度一致。
- validate_frame_index：常规帧满足 timestep % R == 0；(episode_id, video_id,
  frame_index) 唯一；MP4 解码帧数 == frame_index 行数；frame_index=0 为初始帧。
返回 (ok: bool, errors: list[str])。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from craftax.contracts import (
    EPISODES_PARQUET_FILENAME,
    FRAME_INDEX_PARQUET_FILENAME,
    TRANSITIONS_PARQUET_FILENAME,
    ZARR_ARRAY_ACTIONS,
    ZARR_ARRAY_STATE_EPISODE_IDS,
    ZARR_ARRAY_STATE_TIMESTEPS,
    ZARR_DIRNAME,
    FrameSampleConfig,
)

MANIFEST_FILENAME = "shard_manifest.json"


def _read_parquet(shard_dir: str, filename: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.join(shard_dir, filename))
    return table.to_pylist()


def _open_zarr(shard_dir: str) -> Any:
    import zarr

    store = zarr.storage.LocalStore(os.path.join(shard_dir, ZARR_DIRNAME))
    return zarr.open_group(store=store, mode="r")


def _manifest_frame_sample(shard_dir: str) -> Optional[FrameSampleConfig]:
    path = os.path.join(shard_dir, MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        manifest = json.load(f)
    fs = manifest.get("frame_sample") or {}
    if not fs:
        return None
    return FrameSampleConfig(
        step_rate_hz=int(fs.get("step_rate_hz", 20)),
        video_fps=int(fs.get("video_fps", 10)),
    )


def validate_timeline(shard_dir: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        episodes = _read_parquet(shard_dir, EPISODES_PARQUET_FILENAME)
        transitions = _read_parquet(shard_dir, TRANSITIONS_PARQUET_FILENAME)
        zarr = _open_zarr(shard_dir)
    except Exception as exc:  # noqa: BLE001 - 校验器吞异常转错误列表
        return False, [f"读取 shard 失败: {exc!r}"]

    # 1) episode 内 states == transitions + 1
    for ep in episodes:
        if int(ep["num_states"]) != int(ep["num_transitions"]) + 1:
            errors.append(
                f"episode {ep['episode_id']}: num_states={ep['num_states']} "
                f"!= num_transitions+1={ep['num_transitions'] + 1}"
            )
        if int(ep["state_end"]) - int(ep["state_start"]) + 1 != int(ep["num_states"]):
            errors.append(
                f"episode {ep['episode_id']}: state 区间 "
                f"[{ep['state_start']},{ep['state_end']}] 长度 != num_states={ep['num_states']}"
            )
    # 相邻 episode 的 state 区间必须连续且不重叠
    episodes_sorted = sorted(episodes, key=lambda e: int(e["state_start"]))
    for prev, cur in zip(episodes_sorted, episodes_sorted[1:]):
        if int(prev["state_end"]) + 1 != int(cur["state_start"]):
            errors.append(
                f"episode 区间不连续: {prev['episode_id']} end={prev['state_end']}, "
                f"{cur['episode_id']} start={cur['state_start']}"
            )

    # 2) transitions 行数 == Zarr actions 长度
    n_trans = len(transitions)
    n_actions = int(zarr[ZARR_ARRAY_ACTIONS].shape[0])
    if n_trans != n_actions:
        errors.append(f"transitions 行数 {n_trans} != Zarr actions 长度 {n_actions}")

    # 3) state 数组长度一致
    state_timesteps = zarr[ZARR_ARRAY_STATE_TIMESTEPS]
    state_episode_ids = zarr[ZARR_ARRAY_STATE_EPISODE_IDS]
    n_states = int(state_timesteps.shape[0])
    if n_states != int(state_episode_ids.shape[0]):
        errors.append(
            f"state_timesteps 长度 {n_states} != state_episode_ids 长度 "
            f"{state_episode_ids.shape[0]}"
        )
    # 取一个代表性 state 数组核对时间维
    first_state_name = None
    for name in zarr["state"].array_keys():
        first_state_name = name
        break
    if first_state_name is None:
        errors.append("tensors.zarr/state 下没有数组")
    else:
        first_state_array = zarr["state"][first_state_name]
        if int(first_state_array.shape[0]) != n_states:
            errors.append(
                f"state/{first_state_name} 长度 {first_state_array.shape[0]} "
                f"!= 期望状态数 {n_states}"
            )

    # 4) episode 计数与 episode_ids 范围一致
    if episodes:
        ids = np.asarray(state_episode_ids[:])
        if int(ids.min()) != 0 or int(ids.max()) != len(episodes) - 1:
            errors.append(
                f"state_episode_ids 范围 [{int(ids.min())},{int(ids.max())}] "
                f"与 episode 行数 {len(episodes)} 不一致"
            )

    return not errors, errors


def validate_frame_index(
    shard_dir: str, frame_sample: Optional[FrameSampleConfig] = None
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    config = frame_sample or _manifest_frame_sample(shard_dir)
    if config is None:
        return False, ["缺少 frame_sample 配置（manifest 或参数）"]

    try:
        frames = _read_parquet(shard_dir, FRAME_INDEX_PARQUET_FILENAME)
        episodes = _read_parquet(shard_dir, EPISODES_PARQUET_FILENAME)
    except Exception as exc:  # noqa: BLE001
        return False, [f"读取 frame_index 失败: {exc!r}"]

    # 无视频帧时跳过（允许只有状态/动作的 shard）
    if not frames:
        return True, []

    ep_by_id = {ep["episode_id"]: ep for ep in episodes}
    per_video: Dict[str, List[Dict[str, Any]]] = {}
    for row in frames:
        key = (row["episode_id"], row["video_id"])
        per_video.setdefault(key, []).append(row)

    for (episode_id, video_id), rows in per_video.items():
        # frame_index 唯一且从 0 连续
        idxs = sorted(int(r["frame_index"]) for r in rows)
        if idxs != list(range(len(rows))):
            errors.append(
                f"{video_id}: frame_index 非连续 {idxs}（应有 0..{len(rows) - 1}）"
            )
        if rows[0]["frame_index"] != 0 or not rows[0]["is_initial_frame"]:
            errors.append(f"{video_id}: 缺少 frame_index=0 的初始帧")
        # 常规帧满足 timestep % R == 0，终局帧除外
        for r in rows:
            if not r["is_initial_frame"] and not r["is_terminal_frame"]:
                ts = int(r["timestep"])
                if config.frame_stride and ts % config.frame_stride != 0:
                    errors.append(
                        f"{video_id}: 常规帧 timestep={ts} 不满足 %R==0 "
                        f"(R={config.frame_stride})"
                    )
        # timestep/state_index 必须落在 episode 的 state 区间内
        ep = ep_by_id.get(episode_id)
        if ep is None:
            errors.append(f"{video_id}: episode {episode_id} 不在 episodes.parquet 中")
            continue
        lo, hi = int(ep["state_start"]), int(ep["state_end"])
        for r in rows:
            if not (lo <= int(r["state_index"]) <= hi):
                errors.append(
                    f"{video_id}: state_index={r['state_index']} 超出 episode "
                    f"[{lo},{hi}]"
                )

    # MP4 解码帧数 == frame_index 行数
    import imageio.v3 as iio

    for (episode_id, video_id), rows in per_video.items():
        video_path = os.path.join(shard_dir, f"video-{video_id}.mp4")
        if not os.path.isfile(video_path):
            errors.append(f"缺少视频文件: video-{video_id}.mp4")
            continue
        try:
            decoded = iio.imread(video_path)
            n_decoded = decoded.shape[0] if decoded.ndim == 4 else 1
            if n_decoded != len(rows):
                errors.append(
                    f"video-{video_id}.mp4 解码帧数 {n_decoded} != frame_index 行数 {len(rows)}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"video-{video_id}.mp4 解码失败: {exc!r}")

    return not errors, errors


def validate_shard(shard_dir: str) -> Tuple[bool, List[str]]:
    """timeline + frame_index 汇总校验。"""
    ok_t, err_t = validate_timeline(shard_dir)
    ok_f, err_f = validate_frame_index(shard_dir)
    return ok_t and ok_f, err_t + err_f
