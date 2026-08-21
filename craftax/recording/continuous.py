"""Join streamed recording shards into one continuous replay video."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def discover_segments(root: str, episode_id: str | None = None) -> List[Dict[str, Any]]:
    """Return video segments in simulation order across all sealed shards."""
    import pyarrow.parquet as pq

    rows: List[Dict[str, Any]] = []
    for episodes_path in sorted(Path(root).rglob("episodes.parquet")):
        shard = episodes_path.parent
        for row in pq.read_table(episodes_path).to_pylist():
            current = str(row["episode_id"])
            source = current.split("-seg", 1)[0]
            if episode_id is not None and source != episode_id and current != episode_id:
                continue
            video = shard / f"video-{row['video_id']}.mp4"
            if video.exists():
                rows.append({
                    "episode_id": current,
                    "source_episode_id": source,
                    "segment_index": int(current.rsplit("-seg", 1)[1]) if "-seg" in current else 0,
                    "state_start": int(row.get("state_start", 0)),
                    "video": str(video.resolve()),
                    "num_frames": int(row.get("num_frames", 0)),
                    "start_wall_ns": int(row.get("start_wall_ns", 0)),
                })
    rows.sort(key=lambda r: (r["source_episode_id"], r["segment_index"], r["state_start"], r["video"]))
    return rows


def concatenate_recording(root: str, output: str | None = None, episode_id: str | None = None) -> Dict[str, Any]:
    """Concatenate segments with ffmpeg and write a replay manifest."""
    root_path = Path(root)
    segments = discover_segments(str(root_path), episode_id)
    if not segments:
        raise FileNotFoundError(f"no sealed video segments found under {root}")
    output_path = Path(output) if output else root_path / "continuous.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to concatenate recording segments")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        listing_path = listing.name
        for segment in segments:
            escaped = segment["video"].replace("'", "'\\''")
            listing.write(f"file '{escaped}'\n")
    try:
        command = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listing_path,
                   "-c", "copy", str(output_path)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            # Different recorder shards can carry slightly different H.264
            # headers. Re-encode once so replay remains continuous.
            subprocess.run(
                [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listing_path,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
    finally:
        Path(listing_path).unlink(missing_ok=True)
    manifest = {
        "format_version": "continuous-replay-v1",
        "output": str(output_path),
        "episode_id": episode_id,
        "segments": segments,
        "num_segments": len(segments),
        "num_frames": sum(s["num_frames"] for s in segments),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
