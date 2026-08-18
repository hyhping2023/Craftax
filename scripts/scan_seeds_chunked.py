"""分块种子扫描：JAX CPU 后端在单进程内编译多次会触发 LLVM 内存映射失败
（约 50+ 次后 Cannot allocate memory），因此每个 chunk 用独立子进程扫描，
最后合并为一个大 JSON。

用法：
    JAX_PLATFORMS=cpu conda run -n craftax python scripts/scan_seeds_chunked.py \
        --start 2000 --count 600 --chunk 40 --out data/seed_scan_2000.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from craftax.contracts import default_data_dir  # noqa: E402


def merge_scans(inputs: list[str], out: str) -> dict:
    """合并多个 seed_scan JSON（含 chunk 产物），按 seed 去重排序，写 out。"""
    results: dict[int, dict] = {}
    for path in inputs:
        if not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for r in data.get("results", []):
            results[int(r["seed"])] = r
    merged = [results[s] for s in sorted(results)]
    golden = sorted(s for s, r in results.items() if r["all_ladders_reachable"])
    payload = {"scanned": len(merged), "golden_seeds": golden, "results": merged}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    armor = [
        s for s, r in results.items()
        if r["all_ladders_reachable"]
        and r["floors"][0]["ore"]["iron"] >= 3
        and r["floors"][0]["ore"]["coal"] >= 3
    ]
    print(f"=== merged {len(merged)} seeds → {out_path}")
    print(f"=== golden ({len(golden)}): {golden}")
    print(f"=== golden + L0 armor-feasible ({len(armor)}): {armor}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="分块种子扫描（每个 chunk 独立进程）")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=40, help="每个子进程扫描的种子数")
    parser.add_argument("--out", default=None, help="合并输出 JSON 路径")
    parser.add_argument("--parallel", type=int, default=1, help="并行 chunk 数（默认 1，JAX 内存受限时勿开大）")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(default_data_dir()) / "seed_scan_chunked.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / (out_path.stem + "_chunks")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for s in range(args.start, args.start + args.count, args.chunk):
        chunks.append((s, min(args.chunk, args.start + args.count - s)))

    print(f"[chunked] {len(chunks)} chunks x ~{args.chunk} seeds → {out_path}")
    for idx, (s, n) in enumerate(chunks):
        chunk_path = tmp_dir / f"chunk_{s:06d}.json"
        if chunk_path.exists():
            print(f"  [{idx+1}/{len(chunks)}] skip existing {chunk_path.name}")
        else:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "scan_seeds.py"),
                "--seeds", ",".join(str(x) for x in range(s, s + n)),
                "--out", str(chunk_path),
            ]
            print(f"  [{idx+1}/{len(chunks)}] scanning {s}..{s+n-1} ...")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # 子进程可能因 JAX 内存问题崩溃：重试一次
                print(f"    retry after exit {proc.returncode} ...")
                proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"    FAILED chunk {s}: {proc.stderr[-300:]}")
                continue
        # 校验 chunk 可解析（子进程崩溃可能留下残缺文件）
        try:
            json.loads(chunk_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            print(f"    corrupt chunk {s}: remove and re-scan")
            chunk_path.unlink(missing_ok=True)

    # 合并本批 chunk + 已有的其他 seed_scan 文件（2000 范围 + 3000 范围）
    inputs = sorted(str(p) for p in tmp_dir.glob("chunk_*.json"))
    for extra in (Path(default_data_dir()) / "seed_scan.json",
                  Path(default_data_dir()) / "seed_scan_3030.json"):
        if extra.exists() and str(extra) not in inputs:
            inputs.append(str(extra))
    merge_scans(inputs, str(out_path))


if __name__ == "__main__":
    main()
