"""Demo 5：任务依赖图与分级任务展示。

从 registry 实时构建静态任务图（92 个任务：77 基础 + 15 个由并发子 agent
提议的分级复合任务），展示：
- 图统计与 DAG 校验；
- 类别划分（native/collect/crafting/combat/exploration）；
- 拓扑层级（从根任务出发的最长依赖深度）；
- 语义层级（atomic / composite / root_goal）；
- 根任务与叶子任务；
- 若干典型根目标的前置依赖链（closure）；
- 新增的分级任务及其依赖。

用法：
    python scripts/demos/demo_task_graph.py            # 实时构建展示
    python scripts/demos/demo_task_graph.py --json     # 打印静态产物 task_graph.json
    python scripts/demos/demo_task_graph.py --tree native.dungeon_campaign
                                                        # 展开某任务的前置树
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from craftax.tasks.graph import TaskGraph, TaskSemanticLevel, build_task_graph  # noqa: E402

ARTIFACT = PROJECT_ROOT / "craftax" / "tasks" / "task_graph.json"

# 展示用示例任务（各阶段代表性根目标）
EXAMPLE_ROOTS = [
    "native.survival_starter_kit",   # native: 收集木材 + 制作基础工具
    "native.collect_all_primary_resources",  # collect: 全部基础资源
    "native.iron_gear",              # crafting: 全套铁质装备
    "native.conquer_dungeon_bosses", # combat: 击败地牢 Boss
    "native.reach_boss_floor",       # exploration: 到达 Boss 层
]


def _level_name(graph: TaskGraph, tid: str) -> str:
    return graph.node(tid).semantic_level


def print_stats(graph: TaskGraph) -> None:
    report = graph.validate()
    print("=" * 72)
    print("任务依赖图（静态 TaskGraph）")
    print("=" * 72)
    print(f"节点数: {report.node_count}   边数: {report.edge_count}   DAG 校验: {'OK' if report.ok else 'FAILED'}")
    if not report.ok:
        print("  dangling:", report.dangling)
        print("  self_deps:", report.self_deps)
        print("  cycles:", report.cycles)
    print(f"根任务(无前置): {len(graph.roots())}   叶子任务(不被依赖): {len(graph.leaves())}")
    print(f"最大拓扑层级: {max(n.topological_level for n in graph.nodes.values())}")


def print_categories(graph: TaskGraph) -> None:
    print()
    print("─" * 72)
    print("按类别划分")
    print("─" * 72)
    for category, ids in graph.by_category().items():
        levels = {tid: graph.node(tid).topological_level for tid in ids}
        max_lv = max(levels.values())
        print(f"[{category}] {len(ids)} 个任务  最深层级={max_lv}")
        for tid in ids:
            n = graph.node(tid)
            print(f"    L{n.topological_level:>2} {tid:<52} {n.semantic_level}")


def print_semantic_distribution(graph: TaskGraph) -> None:
    print()
    print("─" * 72)
    print("语义层级分布")
    print("─" * 72)
    buckets: dict = {}
    for n in graph.nodes.values():
        buckets.setdefault(n.semantic_level, []).append(n.task_id)
    for lv in (TaskSemanticLevel.ATOMIC, TaskSemanticLevel.COMPOSITE, TaskSemanticLevel.ROOT_GOAL):
        ids = sorted(buckets.get(lv.value, []))
        print(f"{lv.value:<10} {len(ids):>3} 个: {', '.join(t.split('.')[-1] for t in ids[:12])}"
              + (" ..." if len(ids) > 12 else ""))


def print_roots_and_leaves(graph: TaskGraph) -> None:
    print()
    print("─" * 72)
    print("根任务（可作独立起点 / 层级 0）")
    print("─" * 72)
    for tid in graph.roots():
        print(f"    {tid:<52} {graph.node(tid).semantic_level}")
    print()
    print("叶子任务（可作为最终根目标）")
    print("─" * 72)
    for tid in graph.leaves():
        print(f"    {tid:<52} {graph.node(tid).semantic_level}")


def print_dependency_chains(graph: TaskGraph) -> None:
    print()
    print("─" * 72)
    print("典型根目标的前置依赖链（closure，即要完成它需先完成哪些任务）")
    print("─" * 72)
    for root in EXAMPLE_ROOTS:
        if root not in graph.task_ids():
            continue
        closure = graph.closure(root)
        # 按拓扑层级排序，呈现依赖顺序
        ordered = sorted(closure, key=lambda t: graph.node(t).topological_level)
        print(f"\n  {root}  (层级 {graph.node(root).topological_level}, {graph.node(root).semantic_level})")
        print(f"      前置任务数: {len(ordered)}")
        for tid in ordered:
            print(f"      L{graph.node(tid).topological_level:>2}  {tid}")


def print_hierarchy_tasks(graph: TaskGraph) -> None:
    print()
    print("─" * 72)
    print("子 agent 提议的分级复合任务（hierarchy_tasks.py，已注册）")
    print("─" * 72)
    # 15 个新增任务：不在 77 基础任务集合中，直接按 task_id 清单展示
    hierarchy_ids = [
        "native.survival_starter_kit", "native.basic_combat_training",
        "native.dungeon_campaign", "native.collect_all_primary_resources",
        "native.survival_sustenance", "native.collect_all_ores",
        "native.starter_toolkit", "native.iron_gear", "native.crafting_mastery",
        "native.clear_surface_threats", "native.conquer_mid_tier_foes",
        "native.conquer_dungeon_bosses", "native.reach_mid_dungeon",
        "native.conquer_lower_realms", "native.build_home_base",
    ]
    for tid in hierarchy_ids:
        if tid not in graph.task_ids():
            continue
        n = graph.node(tid)
        deps = graph.dependencies(tid)
        print(f"  {tid:<50} L{n.topological_level:>2} {n.semantic_level}")
        print(f"      依赖: {', '.join(d.split('.')[-1] for d in deps) if deps else '(无)'}")


def print_tree(graph: TaskGraph, root: str, max_depth: int = 6) -> None:
    """以树形打印某任务的全部前置任务（按拓扑层级缩进）。"""
    if root not in graph.task_ids():
        print(f"任务 {root} 不在图中", file=sys.stderr)
        return
    print()
    print("─" * 72)
    print(f"前置依赖树: {root}")
    print("─" * 72)

    def walk(tid: str, depth: int, seen: set) -> None:
        if depth > max_depth:
            print("    " * depth + "…")
            return
        n = graph.node(tid)
        marker = "▶" if tid == root else " "
        print(f"{'    ' * depth}{marker} {tid}  (L{n.topological_level}, {n.semantic_level})")
        for dep in graph.dependencies(tid):
            if dep in seen:
                print(f"{'    ' * (depth + 1)}↺ {dep}")
                continue
            seen.add(dep)
            walk(dep, depth + 1, seen)

    walk(root, 0, set())


def main() -> None:
    parser = argparse.ArgumentParser(description="任务依赖图与分级 demo")
    parser.add_argument("--json", action="store_true",
                        help="打印静态产物 craftax/tasks/task_graph.json")
    parser.add_argument("--tree", metavar="TASK_ID",
                        help="以树形打印指定任务的前置依赖链")
    parser.add_argument("--artifact", action="store_true",
                        help="从静态产物加载而非实时构建")
    args = parser.parse_args()

    if args.json:
        if not ARTIFACT.exists():
            print(f"未找到产物 {ARTIFACT}，请先运行 scripts/build_task_graph.py merge", file=sys.stderr)
            sys.exit(1)
        with open(ARTIFACT, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
        return

    graph = TaskGraph.load(str(ARTIFACT)) if args.artifact else build_task_graph()

    if args.tree:
        print_tree(graph, args.tree)
        return

    print_stats(graph)
    print_categories(graph)
    print_semantic_distribution(graph)
    print_hierarchy_tasks(graph)
    print_dependency_chains(graph)
    print_roots_and_leaves(graph)


if __name__ == "__main__":
    main()
