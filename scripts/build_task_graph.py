#!/usr/bin/env python
"""任务图构建 CLI：确定性建图 / 校验 / 合并子 agent 提案并落地。

用法：
  # 1) 生成子 agent 输入（按类别划分，确定性）
  python scripts/build_task_graph.py --emit-inputs task_graph_agents/

  # 2) 合并 task_graph_agents/*_proposals.json：
  #    校验 -> 生成 craftax/tasks/builtin/hierarchy_tasks.py
  #          -> 生成 craftax/tasks/data/task_graph.json（含全部节点/边/层级/审计）
  python scripts/build_task_graph.py --merge task_graph_agents/

  # 3) 校验/查看已生成产物
  python scripts/build_task_graph.py --validate
  python scripts/build_task_graph.py --load craftax/tasks/data/task_graph.json --print

约束：不修改现有 77 个 builtin 任务；图构建与校验确定性；
子 agent 提案经 craftax.tasks.agent_spec.validate_proposals 校验后才可落地。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pprint
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from craftax.tasks.agent_spec import (  # noqa: E402
    emit_agent_inputs,
    validate_proposals,
)
from craftax.tasks.graph import TaskGraph, TaskSemanticLevel, category_of  # noqa: E402

DEFAULT_AGENTS_DIR = os.path.join(REPO_ROOT, "task_graph_agents")
HIERARCHY_MODULE = os.path.join(
    REPO_ROOT, "craftax", "tasks", "builtin", "hierarchy_tasks.py"
)
GRAPH_JSON = os.path.join(REPO_ROOT, "craftax", "tasks", "task_graph.json")

TASK_VERSION = "1.0.0"

# 语义层级归属：提案中语义层级条目可能引用尚未入图的提案任务；
# 合并时按 proposed_tasks 中的语义层级优先，其次按 success_predicate 推断。
_INFERRED = {
    "and": TaskSemanticLevel.COMPOSITE.value,
    "or": TaskSemanticLevel.COMPOSITE.value,
    "achievement": TaskSemanticLevel.ATOMIC.value,
    "level_ge": TaskSemanticLevel.ROOT_GOAL.value,
    "always": TaskSemanticLevel.ROOT_GOAL.value,
    "field": TaskSemanticLevel.ROOT_GOAL.value,
}


def _load_proposals(agents_dir: str):
    proposals = {}
    for path in sorted(glob.glob(os.path.join(agents_dir, "*_proposals.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        proposals[d["category"]] = d
    return proposals


def _collect_achievements(expr):
    out = []

    def walk(e):
        if isinstance(e, dict):
            if str(e.get("type", "")) == "achievement":
                name = str(e.get("name", ""))
                if name and name not in out:
                    out.append(name)
            for v in e.values():
                walk(v)
        elif isinstance(e, (list, tuple)):
            for item in e:
                walk(item)

    walk(expr)
    return out


def _render_python_literal(obj) -> str:
    return pprint.pformat(obj, width=88, sort_dicts=False)


def _render_hierarchy_module(accepted_proposals) -> str:
    """把通过校验的 proposed_tasks 渲染为 hierarchy_tasks.py 源码。"""
    lines = [
        '"""子 agent 提议的分级/复合任务（经校验后合并生成，勿手改）。',
        "",
        "来源：task_graph_agents/*_proposals.json，经",
        "craftax.tasks.agent_spec.validate_proposals 校验通过后由",
        "scripts/build_task_graph.py --merge 生成。如需修改，调整 proposals 后重新生成。",
        '"""',
        "from __future__ import annotations",
        "",
        "from typing import Any, Dict, List",
        "",
        "from craftax.contracts import TaskSpec",
        "from craftax.tasks.base import BaseTaskAdapter",
        "",
        f"TASK_VERSION = {TASK_VERSION!r}",
        "",
        "# (task_id, instruction, objective, success_predicate, dependencies)",
        "_HIERARCHY_DEFS: List[Dict[str, Any]] = [",
    ]
    for p in sorted(accepted_proposals, key=lambda x: x["task_id"]):
        instruction = f"{p['instruction_en']} / {p['instruction_zh']}"
        lines.append("    {")
        lines.append(f"        \"task_id\": {p['task_id']!r},")
        lines.append(f"        \"instruction\": {instruction!r},")
        lines.append(f"        \"objective\": {p['objective']!r},")
        lines.append(
            "        \"success_predicate\": "
            + _render_python_literal(p["success_predicate"]).replace("\n", "\n        ")
            + ","
        )
        lines.append(
            "        \"dependencies\": "
            + _render_python_literal(list(p["dependencies"])).replace(
                "\n", "\n        "
            )
            + ","
        )
        lines.append("    },")
    lines.append("]")
    lines.append("")
    lines.append("")
    lines.append("def register_hierarchy_tasks() -> None:")
    lines.append("    from craftax.tasks.registry import register")
    lines.append("")
    lines.append("    for d in _HIERARCHY_DEFS:")
    lines.append("        achievements = _collect_achievements(d['success_predicate'])")
    lines.append("        spec = TaskSpec(")
    lines.append("            task_id=d['task_id'],")
    lines.append("            version=TASK_VERSION,")
    lines.append("            instruction=d['instruction'],")
    lines.append("            objective=d['objective'],")
    lines.append("            success_predicate=d['success_predicate'],")
    lines.append(
        "            annotation_predicates="
        "[{'type': 'achievement', 'name': a} for a in achievements],"
    )
    lines.append("            renderer_config={},")
    lines.append("            dependencies=d['dependencies'],")
    lines.append("        )")
    lines.append(
        "        register(spec.task_id, spec.version, lambda spec=spec: BaseTaskAdapter(spec))"
    )
    lines.append("")
    lines.append("")
    lines.append("def _collect_achievements(expr: Any) -> List[str]:")
    lines.append('    """递归收集谓词中引用的全部成就名（保持顺序、去重）。"""')
    lines.append("    out: List[str] = []")
    lines.append("")
    lines.append("    def walk(e: Any) -> None:")
    lines.append("        if isinstance(e, dict):")
    lines.append(
        "            if str(e.get('type', '')) == 'achievement':"
    )
    lines.append("                name = str(e.get('name', ''))")
    lines.append("                if name and name not in out:")
    lines.append("                    out.append(name)")
    lines.append("            for value in e.values():")
    lines.append("                walk(value)")
    lines.append("        elif isinstance(e, (list, tuple)):")
    lines.append("            for item in e:")
    lines.append("                walk(item)")
    lines.append("")
    lines.append("    walk(expr)")
    lines.append("    return out")
    lines.append("")
    lines.append("")
    lines.append("register_hierarchy_tasks()")
    lines.append("")
    return "\n".join(lines)


def cmd_emit_inputs(args) -> int:
    graph = TaskGraph.build_from_registry()
    report = graph.validate()
    if not report.ok:
        print(f"基础图校验失败: {report}", file=sys.stderr)
        return 1
    paths = emit_agent_inputs(graph, args.agents_dir)
    print(f"节点数: {report.node_count}  边数: {report.edge_count}  ok: {report.ok}")
    for p in paths:
        print(f"  input -> {p}")
    return 0


def _existing_hierarchy_task_ids() -> list:
    """已注册的 hierarchy 任务 task_id（供重复 merge 时从校验图中剔除）。"""
    mod_path = os.path.join(REPO_ROOT, "craftax", "tasks", "builtin", "hierarchy_tasks.py")
    if not os.path.exists(mod_path):
        return []
    import importlib.util

    spec = importlib.util.spec_from_file_location("_hierarchy_probe", mod_path)
    mod = importlib.util.module_from_spec(spec)
    # 只读取数据常量，不执行 register（避免重复注册）。
    source = open(mod_path, encoding="utf-8").read()
    # 直接从源码中提取 _HIERARCHY_DEFS 里的 task_id（简易、无副作用）。
    import re

    return re.findall(r'"task_id": \'(native\.[a-z0-9_]+)\'', source)


def cmd_merge(args) -> int:
    proposals = _load_proposals(args.agents_dir)
    if not proposals:
        print(f"在 {args.agents_dir} 下未找到 *_proposals.json", file=sys.stderr)
        return 1

    graph = TaskGraph.build_from_registry()
    # 重复 merge 时，已注册的 hierarchy 任务先从校验图中剔除，保证幂等。
    existing_hierarchy = set(_existing_hierarchy_task_ids())
    if existing_hierarchy:
        graph = graph.subgraph(
            tid for tid in graph.task_ids() if tid not in existing_hierarchy
        )
    accepted, rejected = validate_proposals(graph, proposals)
    if not accepted:
        print("没有通过校验的提案，未生成任何文件。", file=sys.stderr)
        for r in rejected:
            print(f"  rejected: {r.get('category')} {r.get('task_id')} {r.get('reasons', r.get('reason', ''))}")
        return 1

    # 1) 应用语义层级覆盖（仅限已在图中的任务）。
    applied = 0
    for entry in accepted:
        if entry.get("kind") == "semantic_level":
            applied += graph.apply_semantic_levels(
                {entry["task_id"]: entry["semantic_level"]}
            )

    # 2) 找出 proposed_tasks 对应的 accepted 条目。
    proposed_entries = [e for e in accepted if e.get("kind") != "semantic_level"]
    # agent 显式给出的提案任务语义层级（可能因“任务不在图中”被 reject，优先采用）。
    proposed_semantic: dict = {}
    for proposal in proposals.values():
        for tid, lv in proposal.get("semantic_levels", {}).items():
            if any(e["task_id"] == tid for e in proposed_entries):
                proposed_semantic[tid] = lv
    if args.dry_run:
        print(f"通过校验: {len(proposed_entries)} 个提议任务; 语义层级覆盖 {applied} 项。")
        for e in proposed_entries:
            print(f"  - {e['task_id']} (category={e['category']})")
        for r in rejected:
            print(f"  rejected: {r.get('category')} {r.get('task_id')} {r.get('reasons', r.get('reason', ''))}")
        return 0

    # 3) 生成并写入 hierarchy_tasks.py。
    os.makedirs(os.path.dirname(HIERARCHY_MODULE), exist_ok=True)
    with open(HIERARCHY_MODULE, "w", encoding="utf-8") as f:
        f.write(_render_hierarchy_module(proposed_entries))
    print(f"生成 {HIERARCHY_MODULE}")

    # 4) 重新从 registry 建图（此时 hierarchy 任务已注册），再叠加语义覆盖，
    #    并写入 task_graph.json。
    import craftax.tasks.builtin.hierarchy_tasks  # noqa: F401  确保注册

    graph2 = TaskGraph.build_from_registry()
    report = graph2.validate()
    if not report.ok:
        print(f"合并后图校验失败: {report}", file=sys.stderr)
        return 1
    graph2.set_provenance(
        "agent",
        {
            "proposal_dir": args.agents_dir,
            "accepted_proposals": len(proposed_entries),
            "rejected": [
                {"category": r.get("category"), "task_id": r.get("task_id"),
                 "reason": r.get("reasons") or r.get("reason")}
                for r in rejected
            ],
            "semantic_level_overrides": applied,
        },
    )
    # 重新应用全部语义层级（含 new 任务推断值）。
    semantic_map: dict = {}
    for entry in accepted:
        if entry.get("kind") == "semantic_level":
            semantic_map[entry["task_id"]] = entry["semantic_level"]
    for entry in proposed_entries:
        tid = entry["task_id"]
        if tid in proposed_semantic:
            semantic_map[tid] = proposed_semantic[tid]
        else:
            semantic_map[tid] = _INFERRED.get(
                str(entry["success_predicate"].get("type", "and")),
                TaskSemanticLevel.COMPOSITE.value,
            )
    graph2.apply_semantic_levels(semantic_map)

    os.makedirs(os.path.dirname(GRAPH_JSON), exist_ok=True)
    graph2.save(GRAPH_JSON)
    print(f"生成 {GRAPH_JSON}")
    print(f"最终节点数: {report.node_count}  边数: {report.edge_count}  ok: {report.ok}")
    print(f"拒绝的提案条目: {len(rejected)}（详见 JSON provenance）")
    return 0


def cmd_validate(args) -> int:
    path = args.path
    if not os.path.exists(path):
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1
    graph = TaskGraph.load(path)
    report = graph.validate()
    print(f"节点数: {report.node_count}  边数: {report.edge_count}  ok: {report.ok}")
    if not report.ok:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 1
    return 0


def cmd_load_print(args) -> int:
    if not os.path.exists(args.path):
        print(f"文件不存在: {args.path}", file=sys.stderr)
        return 1
    graph = TaskGraph.load(args.path)
    report = graph.validate()
    print(f"节点数: {report.node_count}  边数: {report.edge_count}  ok: {report.ok}")
    print(f"roots: {graph.roots()}")
    print(f"leaves: {graph.leaves()}")
    for cat, ids in graph.by_category().items():
        print(f"  [{cat}] {len(ids)}: {ids}")
    if args.verbose:
        print()
        print(graph.to_json())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_emit = sub.add_parser("emit-inputs", help="按类别写出子 agent 输入 JSON")
    p_emit.add_argument("--agents-dir", default=DEFAULT_AGENTS_DIR)

    p_merge = sub.add_parser("merge", help="校验并合并子 agent 提案")
    p_merge.add_argument("agents_dir", nargs="?", default=DEFAULT_AGENTS_DIR)
    p_merge.add_argument("--dry-run", action="store_true", help="只报告不写文件")

    p_val = sub.add_parser("validate", help="校验已生成的 task_graph.json")
    p_val.add_argument("--path", default=GRAPH_JSON)

    p_load = sub.add_parser("load", help="加载并打印 task_graph.json")
    p_load.add_argument("path", nargs="?", default=GRAPH_JSON)
    p_load.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.cmd == "emit-inputs":
        return cmd_emit_inputs(args)
    if args.cmd == "merge":
        return cmd_merge(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "load":
        return cmd_load_print(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
