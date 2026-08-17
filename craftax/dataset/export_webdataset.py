"""WebDataset 格式导出：按 episode 打包 tar。

每个 tar 包含同一条样本（basename = episode_id）的三个文件::

    <episode_id>.mp4            复制自 shard 的视频（流式拷贝，不重新编码）
    <episode_id>.json           episode 元数据 + frame_index 摘要
    <episode_id>.actions.json   action 序列（含 reward/terminated/event tokens）

文件名（含 tar 内部 arcname）经 ``sanitize_filename`` 清洗；本模块只读 shard，
产出目录由调用方指定。
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from craftax.dataset.reader import EpisodeView, ShardReader, parse_event_tokens

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """清洗为安全的文件名片段（保留字母/数字/._-）。"""
    return _FILENAME_SAFE.sub("_", name).strip("._")


def export_webdataset(
    shard_reader: ShardReader,
    out_dir: Any,
    tar_prefix: str,
    episode_ids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """把 shard 按 episode 导出为 WebDataset tar。

    返回生成的 tar 文件路径列表；tar 文件名为
    ``<tar_prefix>-<episode_id>.tar``（二者均已 sanitize）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = sanitize_filename(tar_prefix) or "shard"

    allowed = set(episode_ids) if episode_ids is not None else None
    written: List[Path] = []
    for episode in shard_reader.episodes():
        if allowed is not None and episode.episode_id not in allowed:
            continue
        tar_path = out / f"{prefix}-{sanitize_filename(episode.episode_id)}.tar"
        _write_episode_tar(episode, tar_path)
        written.append(tar_path)
    return written


def _write_episode_tar(episode: EpisodeView, tar_path: Path) -> None:
    base = sanitize_filename(episode.episode_id)
    with tarfile.open(tar_path, mode="w") as tar:
        video_path = episode.video_path
        if video_path is not None and video_path.exists():
            tar.add(str(video_path), arcname=f"{base}.mp4")

        frame_rows = episode.frame_rows()
        summary = {
            "num_frames": len(frame_rows),
            "first_timestep": int(frame_rows[0]["timestep"]) if frame_rows else None,
            "last_timestep": int(frame_rows[-1]["timestep"]) if frame_rows else None,
            "frames": frame_rows,
        }
        _add_json(
            tar, f"{base}.json", {"episode": episode.meta, "frame_index": summary}
        )
        _add_json(tar, f"{base}.actions.json", _action_sequence(episode))


def _action_sequence(episode: EpisodeView) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in episode.transition_rows():
        out.append(
            {
                "timestep": row["timestep"],
                "action_id": row["action_id"],
                "action_name": row["action_name"],
                "action_source": row.get("action_source"),
                "command_id": row.get("command_id"),
                "reward": row.get("reward"),
                "terminated": row.get("terminated"),
                "truncated": row.get("truncated"),
                "event_tokens": parse_event_tokens(row.get("event_tokens")),
                "sim_time_ns": row.get("sim_time_ns"),
                "instruction": row.get("instruction"),
                "task_id": row.get("task_id"),
            }
        )
    return out


def _add_json(tar: tarfile.TarFile, arcname: str, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, default=_json_default).encode("utf-8")
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法 JSON 序列化: {type(value)!r}")
