"""并发子 agent 制定任务图与分级任务定义的输出契约与校验。

设计（对应 embodied_environment_plan.md §6.1.2 静态图 + 分级任务定义）：
- 将 registry 中的 77 个 builtin 任务按类别（native/collect/crafting/combat/exploration）分组；
- 每个类别由独立子 agent 读取输入（category_input）并产出提案
  {category}_proposals.json（semantic_levels / proposed_tasks / anomalies）；
- validate_proposals 对提案做确定性校验（task_id 唯一、achievement 名合法、
  谓词类型合法、依赖无悬空、加入后仍为 DAG），通过者才可合并落地。

本模块不修改任何 TaskSpec、不改变游戏规则；图构建与校验全部确定性。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from craftax.contracts import TaskSpec
from craftax.tasks import registry
from craftax.tasks.graph import (
    TaskGraph,
    TaskSemanticLevel,
    category_of,
    known_categories,
)

# 提案谓词允许的类型（与 craftax/tasks/base.py 的谓词注册表对齐，
# 但不允许 always/never/field_* 作为新任务成功条件，保持可解释）。
_ALLOWED_PREDICATE_TYPES = {"achievement", "and", "or", "not", "level_ge"}

_TASK_ID_RE = re.compile(r"^native\.[a-z][a-z0-9_]*$")

PROPOSAL_SCHEMA = {
    "category": "str，类别名（native/collect/crafting/combat/exploration）",
    "semantic_levels": {
        "task_id": '"atomic" | "composite" | "root_goal"，该类别内每个任务的语义层级'
    },
    "proposed_tasks": [
        {
            "task_id": "str，native.<snake_case>，与现有 77 个任务不冲突",
            "instruction_en": "str，英文指令",
            "instruction_zh": "str，中文指令",
            "objective": "str，目标描述",
            "success_predicate": "dict，只含 achievement/and/or/not/level_ge 谓词，"
            "achievement 名必须存在于 craftax.craftax.constants.Achievement 枚举",
            "dependencies": ["str，现有或本次提案中的 task_id（严格前置）"],
            "rationale": "str，为什么定义这个分级任务",
        }
    ],
    "anomalies": [
        {"task_id": "str", "issue": "str，发现的问题", "suggestion": "str，建议"}
    ],
    "rationale": "str，类别级总结",
}


def _spec_of(task_id: str) -> TaskSpec:
    versions = registry.list_versions(task_id)
    if not versions:
        raise KeyError(f"任务 {task_id!r} 未注册")
    return registry.get_task_adapter(task_id, versions[-1]).spec


def category_input(graph: TaskGraph, category: str) -> Dict[str, Any]:
    """生成子 agent 的类别输入：该类别全部任务的可序列化描述。"""
    tasks: List[Dict[str, Any]] = []
    for tid in graph.category_subgraph(category).task_ids():
        spec = _spec_of(tid)
        node = graph.node(tid)
        tasks.append(
            {
                "task_id": tid,
                "version": spec.version,
                "category": node.category,
                "instruction": spec.instruction,
                "objective": spec.objective,
                "success_predicate": spec.success_predicate,
                "annotation_predicates": list(spec.annotation_predicates),
                "achievements": list(node.achievements),
                "dependencies": list(node.dependencies),
                "success_type": node.success_type,
                "topological_level": node.topological_level,
                "semantic_level": node.semantic_level,
            }
        )
    return {"category": category, "task_count": len(tasks), "tasks": tasks}


def emit_agent_inputs(graph: TaskGraph, out_dir: str) -> List[str]:
    """为每个类别写出 {category}_input.json，返回写入的文件路径列表。"""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for category in known_categories():
        path = os.path.join(out_dir, f"{category}_input.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(category_input(graph, category), f, ensure_ascii=False, indent=2)
        paths.append(path)
    return paths


def _valid_achievement_names() -> set:
    from craftax.craftax.constants import Achievement

    return set(Achievement.__members__.keys())


def _validate_predicate(expr: Any, valid_achievements: set, path: str) -> List[str]:
    """递归校验谓词表达式。返回错误列表（空表示合法）。"""
    errors: List[str] = []
    if expr is None:
        return errors
    if not isinstance(expr, dict):
        errors.append(f"{path}: 谓词必须是 dict，got {type(expr).__name__}")
        return errors
    type_name = str(expr.get("type", ""))
    if type_name not in _ALLOWED_PREDICATE_TYPES:
        errors.append(f"{path}: 谓词类型 {type_name!r} 不允许（可用 {sorted(_ALLOWED_PREDICATE_TYPES)}）")
        return errors
    if type_name == "achievement":
        name = str(expr.get("name", ""))
        if name not in valid_achievements:
            errors.append(f"{path}: achievement 名 {name!r} 不在 Achievement 枚举中")
    elif type_name in ("and", "or"):
        for i, sub in enumerate(expr.get("predicates", [])):
            errors.extend(_validate_predicate(sub, valid_achievements, f"{path}.predicates[{i}]"))
    elif type_name == "not":
        errors.extend(_validate_predicate(expr.get("predicate"), valid_achievements, f"{path}.predicate"))
    # level_ge：value 为 int，无需额外校验
    return errors


def validate_proposals(
    graph: TaskGraph,
    proposals_by_category: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """校验各子 agent 的提案。返回 (accepted, rejected)。

    accepted 元素保留 category 与原始提案；rejected 元素含 task_id + reasons。
    """
    valid_achievements = _valid_achievement_names()
    valid_levels = {lv.value for lv in TaskSemanticLevel}

    # 1) 聚合语义层级覆盖（类别内先验）。
    semantic_levels: Dict[str, str] = {}
    level_rejects: List[Dict[str, Any]] = []
    for category, proposal in proposals_by_category.items():
        for tid, lv in proposal.get("semantic_levels", {}).items():
            if tid not in graph._nodes:
                level_rejects.append(
                    {"category": category, "task_id": tid, "reason": "任务不在图中", "kind": "semantic_level"}
                )
                continue
            if lv not in valid_levels:
                level_rejects.append(
                    {"category": category, "task_id": tid, "reason": f"层级 {lv!r} 非法", "kind": "semantic_level"}
                )
                continue
            semantic_levels[tid] = lv

    # 2) 校验 proposed_tasks。
    all_existing = set(graph._nodes.keys())
    proposed_ids: Dict[str, Dict[str, Any]] = {}
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for category, proposal in proposals_by_category.items():
        seen_in_category: set = set()
        for p in proposal.get("proposed_tasks", []):
            reasons: List[str] = []
            tid = str(p.get("task_id", ""))
            if not tid:
                reasons.append("缺少 task_id")
            else:
                if not _TASK_ID_RE.match(tid):
                    reasons.append("task_id 必须形如 native.<snake_case>")
                if tid in all_existing:
                    reasons.append("task_id 与现有任务冲突")
                if tid in proposed_ids:
                    reasons.append("task_id 在多个类别提案中重复")
                if tid in seen_in_category:
                    reasons.append("task_id 在类别内重复")

            if not isinstance(p.get("success_predicate"), dict):
                reasons.append("success_predicate 必须是 dict")
            else:
                reasons.extend(
                    _validate_predicate(
                        p["success_predicate"],
                        valid_achievements,
                        f"{tid}.success_predicate",
                    )
                )

            deps = p.get("dependencies", [])
            if not isinstance(deps, list):
                reasons.append("dependencies 必须是 list")
            else:
                for d in deps:
                    if d not in all_existing and d not in proposed_ids and d != tid:
                        reasons.append(f"依赖 {d!r} 既不存在于现有任务，也不在本轮提案中")

            for field_name in ("instruction_en", "instruction_zh", "objective", "rationale"):
                if not str(p.get(field_name, "")).strip():
                    reasons.append(f"缺少字段 {field_name}")

            if reasons:
                rejected.append(
                    {"category": category, "task_id": tid, "reasons": reasons, "proposal": p}
                )
                continue

            record = dict(p)
            record["category"] = category
            proposed_ids[tid] = record
            seen_in_category.add(tid)
            accepted.append(record)

    # 3) 把提案并入图副本，验证整体仍为无环 DAG、无悬空。
    if accepted:
        probe = graph.to_dict()
        for record in accepted:
            tid = record["task_id"]
            probe["nodes"][tid] = {
                "task_id": tid,
                "version": "1.0.0",
                "category": record["category"],
                "success_type": str(record["success_predicate"].get("type", "and")),
                "achievements": [],
                "dependencies": list(record.get("dependencies", [])),
                "topological_level": 0,
                "semantic_level": "composite",
            }
        probe_graph = TaskGraph.from_dict(probe)
        probe_report = probe_graph.validate()
        if not probe_report.ok:
            rejected.extend(
                [
                    {
                        "category": record["category"],
                        "task_id": record["task_id"],
                        "reason": f"并入后图校验失败: {probe_report.dangling or probe_report.cycles or probe_report.self_deps}",
                        "kind": "graph_merge",
                    }
                    for record in accepted
                ]
            )
            accepted = []

    # 语义层级覆盖也作为 accepted 的一部分返回（供合并时 apply）。
    for tid, lv in semantic_levels.items():
        accepted.append(
            {"category": graph.node(tid).category, "task_id": tid, "semantic_level": lv, "kind": "semantic_level"}
        )

    return accepted, rejected + level_rejects


def write_proposal(
    category: str,
    proposal: Dict[str, Any],
    out_dir: str,
) -> str:
    """写入 {category}_proposals.json。返回文件路径。"""
    import os

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{category}_proposals.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)
    return path
