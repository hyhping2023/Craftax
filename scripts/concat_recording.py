#!/usr/bin/env python3
"""Join all MP4 segments from a Craftax recording into a continuous replay."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craftax.recording.continuous import concatenate_recording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", help="recording root or data directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args()
    manifest = concatenate_recording(args.task_dir, args.output, args.episode_id)
    print(f"wrote {manifest['output']} ({manifest['num_segments']} segments)")


if __name__ == "__main__":
    main()
