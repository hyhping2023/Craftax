"""Demo 4：数据集读取与训练样本导出。

读取 sealed shard（demo_record 的产物），展示 episode 摘要、帧对齐，
导出 VLA / World Model 样本；可选导出 WebDataset tar。

用法：
    python scripts/demos/demo_dataset.py --latest
    python scripts/demos/demo_dataset.py --shard <shard-dir>
    python scripts/demos/demo_dataset.py --latest --export data/webdataset/
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from craftax.contracts import default_data_dir  # noqa: E402


def find_latest_shard(root: Path) -> Path:
    manifests = sorted(root.rglob("shard_manifest.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        raise SystemExit(f"没有找到 sealed shard，请先运行 demo_record.py（数据目录 {root}）")
    return manifests[0].parent


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集读取 demo")
    parser.add_argument("--shard", help="shard 目录路径")
    parser.add_argument("--latest", action="store_true",
                        help=f"自动选择默认数据目录（{default_data_dir()}）下最新的 shard")
    parser.add_argument("--window", type=int, default=4, help="窗口长度")
    parser.add_argument("--export", help="导出 WebDataset tar 的目标目录")
    args = parser.parse_args()

    from craftax.dataset.export_webdataset import export_webdataset
    from craftax.dataset.reader import ShardReader
    from craftax.dataset.vla_windows import vla_samples
    from craftax.dataset.world_model_windows import wm_samples

    if args.shard:
        shard_dir = Path(args.shard)
    elif args.latest:
        shard_dir = find_latest_shard(Path(default_data_dir()))
    else:
        parser.error("请提供 --shard <path> 或 --latest")

    reader = ShardReader(shard_dir)
    print(f"[reader] shard: {shard_dir}")

    episodes = list(reader.episodes())
    print(f"[episodes] {len(episodes)} 个 episode")
    for ep in episodes:
        rows = ep.frame_rows()
        frames = list(ep.frames())
        print(
            f"    ep={ep.episode_id} task={ep.task_id} seed={ep.seed} "
            f"states={ep.num_states} trans={ep.num_transitions} "
            f"frames={ep.num_frames} video_decoded={len(frames)} "
            f"last_frame_ts={rows[-1]['timestep']}"
        )

    vs = list(vla_samples(reader, window_len=args.window))
    ws = list(wm_samples(reader, window_len=args.window))
    print(f"[samples] vla={len(vs)} wm={len(ws)} (window={args.window})")
    if vs:
        s = vs[0]
        print(f"    vla[0]: frames={len(s['frames'])} actions={len(s['actions'])} "
              f"instruction={s['instruction'][:36]}")
    if ws:
        s = ws[0]
        print(f"    wm[0]: states={len(s['states'])} actions={len(s['actions'])} "
              f"events={s['events'][:1]}")

    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        tars = export_webdataset(reader, out, tar_prefix="demo")
        print(f"[webdataset] 导出 {len(tars)} 个 tar 到 {out}/")
        for p in tars:
            print(f"    {p}")


if __name__ == "__main__":
    main()
