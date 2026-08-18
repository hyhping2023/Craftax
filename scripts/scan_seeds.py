"""种子扫描：检查各 seed 的地牢下行链可达性，输出"好种子"列表。

用途：深层任务（collect_diamond/sapphire/ruby、enter_graveyard、Boss 战等）需要
依次下楼 L1..L8。由于部分 seed 的梯子/矿石被 WATER 分隔（放石头不可跨水——放下的
石头是 SOLID 块，玩家不可走），同一 seed 下有些层梯子不可达。本脚本对每个候选 seed
逐层检查：
- L0：从玩家出生点 BFS，ladder_down 是否可达；
- L1..L7：从该层 ladder_up（玩家下行进入点）BFS，ladder_down 是否可达；
- 顺带统计各层矿石（coal/iron/diamond/sapphire/ruby）在可达区的数量。

输出一个按 seed 排序的 JSON 文件（默认 data_dir/seed_scan.json），
并在 stdout 打印完全可通行的 seed（golden seeds）。

用法：
    JAX_PLATFORMS=cpu conda run -n craftax python scripts/scan_seeds.py --start 2000 --count 60
    JAX_PLATFORMS=cpu conda run -n craftax python scripts/scan_seeds.py --seeds 2026,2027,2028
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import jax  # noqa: E402

from craftax.contracts import default_data_dir  # noqa: E402
from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.planner.path_planner import blocked  # noqa: E402

# 需要统计可达性的矿石方块
ORE_BLOCKS = {
    "coal": 8,
    "iron": 9,
    "diamond": 10,
    "sapphire": 21,
    "ruby": 22,
}
NUM_LEVELS = 9


def reachable_from(map2d, start):
    """从 start 出发的 BFS 可达格集合（WATER/LAVA/SOLID 不可走）。"""
    from collections import deque

    h, w = map2d.shape
    visited = {start}
    queue = deque([start])
    deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    while queue:
        cell = queue.popleft()
        for d in deltas:
            nxt = (cell[0] + d[0], cell[1] + d[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if nxt in visited:
                continue
            if not blocked(int(map2d[nxt[0], nxt[1]])):
                visited.add(nxt)
                queue.append(nxt)
    return visited


def scan_seed(seed: int) -> dict:
    env = CraftaxSymbolicEnvNoAutoReset()
    state = env.reset(jax.random.PRNGKey(seed), EnvParams())[1]
    hs = jax.device_get(state)

    floors = []
    all_ladders_reachable = True
    ore_totals = {k: 0 for k in ORE_BLOCKS}
    for level in range(NUM_LEVELS):
        map2d = np.asarray(hs.map[level])
        # 进入点：L0 用玩家出生点，L1+ 用该层 ladder_up
        if level == 0:
            enter = (int(hs.player_position[0]), int(hs.player_position[1]))
        else:
            up = hs.up_ladders[level]
            enter = (int(up[0]), int(up[1]))
        reach = reachable_from(map2d, enter)
        down = hs.down_ladders[level]
        down_pos = (int(down[0]), int(down[1]))
        ladder_down_reachable = bool(down_pos in reach) if level < NUM_LEVELS - 1 else None
        if level < NUM_LEVELS - 1 and not ladder_down_reachable:
            all_ladders_reachable = False
        # 矿石计数：矿石方块是 solid，不在可达(可走)集内；统计与可达格 4-邻接的矿石
        ore_counts = {}
        for name, block in ORE_BLOCKS.items():
            block_cells = set(zip(*np.where(map2d == block)))
            adj = set()
            for r in reach:
                for d in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    a = (r[0] + d[0], r[1] + d[1])
                    if a in block_cells:
                        adj.add(a)
            ore_counts[name] = len(adj)
            ore_totals[name] += len(adj)
        floors.append(
            {
                "floor": level,
                "enter": [enter[0], enter[1]],
                "ladder_down": [down_pos[0], down_pos[1]],
                "ladder_down_reachable": ladder_down_reachable,
                "reachable_cells": len(reach),
                "ore": ore_counts,
            }
        )
    return {
        "seed": seed,
        "all_ladders_reachable": all_ladders_reachable,
        "ore_totals": ore_totals,
        "floors": floors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描种子的地牢下行链可达性")
    parser.add_argument("--start", type=int, default=2000, help="起始 seed")
    parser.add_argument("--count", type=int, default=60, help="扫描连续 seed 个数")
    parser.add_argument("--seeds", help="显式指定 seed 列表（逗号分隔），优先于 start/count")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（默认 <data_dir>/seed_scan.json）")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.start, args.start + args.count))

    results = []
    golden = []
    for seed in seeds:
        r = scan_seed(seed)
        results.append(r)
        if r["all_ladders_reachable"]:
            golden.append(seed)
            tag = "GOLD"
        else:
            tag = "   "
        print(f"[{tag}] seed={seed:<6} ladders_all_reachable={r['all_ladders_reachable']} "
              f"ore={r['ore_totals']}")

    out_path = Path(args.out) if args.out else Path(default_data_dir()) / "seed_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scanned": len(results),
        "golden_seeds": golden,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n=== golden seeds ({len(golden)}/{len(seeds)}): {golden}")
    print(f"写入 {out_path}")


if __name__ == "__main__":
    main()
