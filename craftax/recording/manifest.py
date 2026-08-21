"""shard manifest 工具：文件 hash、manifest 构建/写入/校验。

manifest 结构（写入 SHARD_MANIFEST_FILENAME，封存后不可变）：
{
  "format_version": "1.0",
  "shard_id": str,
  "run_id"/"producer_id"/"attempt_id": str,
  "created_at": ISO8601,
  "task": {"task_id", "task_version", "task_hash"},
  "state_schema_hash": str,
  "frame_sample": {"step_rate_hz", "video_fps"},
  "env_params": {"thirst_rate", "day_length", "god_mode", ...},   # EnvParams 快照
  "counts": {"num_episodes", "num_transitions", "num_states", "num_frames", "num_videos"},
  "arrays": {"state/map": {"dtype", "shape", "chunks"}, ...},
  "files": {"相对路径": {"size", "sha256"}, ...}
}
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from craftax.contracts import (
    EPISODES_PARQUET_FILENAME,
    FRAME_INDEX_PARQUET_FILENAME,
    SHARD_MANIFEST_FILENAME,
    TRANSITIONS_PARQUET_FILENAME,
    ZARR_DIRNAME,
)

MANIFEST_FORMAT_VERSION = "1.0"

CHUNK_BYTES = 1 << 20  # 大文件分块读取，避免整体载入内存


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: str) -> Dict[str, Any]:
    return {"size": os.path.getsize(path), "sha256": compute_sha256(path)}


def collect_dir_hashes(root: str, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """递归收集目录内每个文件的 size/sha256（相对路径为 key）。"""
    result: Dict[str, Dict[str, Any]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.join(prefix, os.path.relpath(full, root))
            result[rel] = file_meta(full)
    return result


def _inspect_zarr_arrays(shard_dir: str) -> Dict[str, Dict[str, Any]]:
    """读取 tensors.zarr 内各数组的 dtype/shape/chunks（manifest.arrays 用）。"""
    zarr_dir = os.path.join(shard_dir, ZARR_DIRNAME)
    if not os.path.isdir(zarr_dir):
        return {}
    import zarr

    store = zarr.storage.LocalStore(zarr_dir)
    group = zarr.open_group(store=store, mode="r")
    arrays: Dict[str, Dict[str, Any]] = {}

    def walk(node: zarr.Group, prefix: str) -> None:
        for name, child in node.members():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(child, zarr.Group):
                walk(child, path)
            else:
                chunks = getattr(child, "chunks", None)
                arrays[path] = {
                    "dtype": str(child.dtype),
                    "shape": list(map(int, child.shape)),
                    "chunks": list(map(int, chunks)) if chunks is not None else None,
                }

    walk(group, "")
    return arrays


def build_shard_manifest(
    shard_dir: str,
    *,
    shard_id: str,
    run_id: str = "",
    producer_id: str = "",
    attempt_id: str = "",
    task_id: str = "",
    task_version: str = "",
    task_hash: str = "",
    state_schema_hash: str = "",
    frame_sample: Optional[Dict[str, Any]] = None,
    env_params: Optional[Dict[str, Any]] = None,
    counts: Optional[Dict[str, int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建（不写入）shard manifest。所有文件须已写完。"""
    files: Dict[str, Dict[str, Any]] = {}
    # tensors.zarr 为目录，逐文件 hash
    zarr_dir = os.path.join(shard_dir, ZARR_DIRNAME)
    if os.path.isdir(zarr_dir):
        files.update(collect_dir_hashes(zarr_dir, prefix=ZARR_DIRNAME))
    for name in (
        EPISODES_PARQUET_FILENAME,
        TRANSITIONS_PARQUET_FILENAME,
        FRAME_INDEX_PARQUET_FILENAME,
    ):
        path = os.path.join(shard_dir, name)
        if os.path.isfile(path):
            files[name] = file_meta(path)
    # 视频与 gold-frames
    for name in sorted(os.listdir(shard_dir)):
        if name.startswith("video-") and name.endswith(".mp4"):
            files[name] = file_meta(os.path.join(shard_dir, name))
    gold_dir = os.path.join(shard_dir, "gold-frames")
    if os.path.isdir(gold_dir):
        files.update(collect_dir_hashes(gold_dir, prefix="gold-frames"))

    manifest: Dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "shard_id": shard_id,
        "run_id": run_id,
        "producer_id": producer_id,
        "attempt_id": attempt_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "task_id": task_id,
            "task_version": task_version,
            "task_hash": task_hash,
        },
        "state_schema_hash": state_schema_hash,
        "frame_sample": frame_sample or {},
        # 环境参数快照：动力学不同的数据不能混用（如 thirst_rate 0.15 vs 1.0）
        "env_params": env_params or {},
        "counts": counts or {},
        "arrays": _inspect_zarr_arrays(shard_dir),
        "files": files,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: Dict[str, Any], shard_dir: str) -> str:
    path = os.path.join(shard_dir, SHARD_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def load_manifest(shard_dir: str) -> Dict[str, Any]:
    with open(os.path.join(shard_dir, SHARD_MANIFEST_FILENAME)) as f:
        return json.load(f)


def verify_manifest(shard_dir: str) -> Tuple[bool, List[str]]:
    """校验 manifest 内所有文件的 size/sha256 与实际一致。"""
    errors: List[str] = []
    manifest_path = os.path.join(shard_dir, SHARD_MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return False, [f"缺少 {SHARD_MANIFEST_FILENAME}"]
    manifest = load_manifest(shard_dir)
    for rel, meta in sorted(manifest.get("files", {}).items()):
        full = os.path.join(shard_dir, rel)
        if not os.path.isfile(full):
            errors.append(f"manifest 文件缺失: {rel}")
            continue
        actual_size = os.path.getsize(full)
        if actual_size != meta["size"]:
            errors.append(f"{rel}: size {actual_size} != manifest {meta['size']}")
            continue
        actual_sha = compute_sha256(full)
        if actual_sha != meta["sha256"]:
            errors.append(f"{rel}: sha256 不匹配")
    # 反向：目录中多出的核心文件未在 manifest 中
    for name in (EPISODES_PARQUET_FILENAME, TRANSITIONS_PARQUET_FILENAME):
        if os.path.isfile(os.path.join(shard_dir, name)) and name not in manifest.get("files", {}):
            errors.append(f"文件未在 manifest 中: {name}")
    return not errors, errors
