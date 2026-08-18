"""全量验证脚本：遍历全部任务，用 SkillChainExecutor 驱动真实 env，统计完成情况。

用法（CPU，慢）：
    python craftax/planner/tests/validate_all_tasks.py
可选 --tasks 指定子集、--seeds 指定种子列表。
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "")

from craftax.tasks.registry import list_task_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", help="逗号分隔任务子集；默认全部")
    parser.add_argument("--seeds", default="2027",
                        help="逗号分隔种子列表，依次尝试直到任务完成")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="单任务步数上限；0=按任务依赖链自动估算（estimate_steps）")
    args = parser.parse_args()

    from test_executor import run_task  # noqa: E402
    from craftax.planner.executor import SkillChainExecutor  # noqa: E402

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = list_task_ids()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    done_list = []
    fail_list = []
    for tid in sorted(task_ids):
        result = None
        max_steps = args.max_steps
        if max_steps <= 0:
            try:
                max_steps = SkillChainExecutor(tid).estimate_steps()
            except Exception:  # noqa: BLE001
                max_steps = 5000
        for seed in seeds:
            result = run_task(tid, seed=seed, max_steps=max_steps)
            if result["done"]:
                break
        tag = "OK " if result["done"] else "ERR"
        print(f"[{tag}] {tid:<44} steps={result.get('steps','-'):<5} "
              f"floor={result.get('floor','-')} err={result.get('error','')}")
        (done_list if result["done"] else fail_list).append(tid)

    print(f"\n=== 完成 {len(done_list)}/{len(task_ids)} ===")
    print("未完成:", fail_list)


if __name__ == "__main__":
    main()
