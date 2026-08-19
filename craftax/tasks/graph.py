"""静态任务依赖图（Task Dependency Graph）的确定性构建与校验。

对应 embodied_environment_plan.md §6.1.2 的“静态任务依赖图”：
- 节点是版本化 TaskSpec.task_id；
- 边来自 TaskSpec.dependencies（A 依赖 B，边为 A -> B，B 是 A 的严格前置）；
- 图必须为 DAG；本模块负责建图、环/悬空/自依赖校验、拓扑层级计算、
  前置闭包（供 runtime planner 消费）与 JSON 序列化。

本模块只读取 registry，不修改任何 TaskSpec、不改变游戏规则。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from craftax.tasks import registry


class TaskSemanticLevel(str, Enum):
    """任务语义层级。

    - ATOMIC：单个原生成就即可完成（如 collect_wood）。
    - COMPOSITE：组合多个成就/子目标（and/or 谓词），如 craft_full_kit。
    - ROOT_GOAL：作为顶层目标的任务（如 survive / explore_dungeon）。
    """

    ATOMIC = "atomic"
    COMPOSITE = "composite"
    ROOT_GOAL = "root_goal"


# ---------------------------------------------------------------------------
# 类别划分（与 builtin 模块划分一致，供并发子 agent 分组）
# ---------------------------------------------------------------------------

# 命名不直观/跨类别模块的任务走显式映射；其余按前缀回退。
_CATEGORY_OVERRIDES: Dict[str, str] = {
    "native.survive": "native",
    "native.collect_wood": "native",  # 定义于 native.py，但语义属 collect
    "native.craft_tools": "native",
    "native.defeat_enemy": "native",
    "native.explore_dungeon": "native",
    "native.collect_ore": "collect",
    "native.collect_all_gems": "collect",
    "native.eat_food": "collect",
    "native.craft_full_kit": "crafting",
    "native.master_crafter": "crafting",
    "native.deep_explorer": "exploration",
    "native.defeat_elemental": "combat",
    "native.defeat_three_enemies": "combat",
    "native.defeat_undead": "combat",
    "native.damage_necromancer": "combat",
    # 子 agent 提议的分级复合任务（hierarchy_tasks.py，经校验合并）
    "native.basic_combat_training": "native",
    "native.dungeon_campaign": "native",
    "native.survival_starter_kit": "native",
    "native.collect_all_ores": "collect",
    "native.collect_all_primary_resources": "collect",
    "native.survival_sustenance": "collect",
    "native.crafting_mastery": "crafting",
    "native.iron_gear": "crafting",
    "native.starter_toolkit": "crafting",
    "native.clear_surface_threats": "combat",
    "native.conquer_dungeon_bosses": "combat",
    "native.conquer_mid_tier_foes": "combat",
    "native.build_home_base": "exploration",
    "native.build_shelter": "exploration",
    "native.conquer_lower_realms": "exploration",
    "native.reach_mid_dungeon": "exploration",
}

# (task_id 前缀（去掉 native. 后）, category)
_CATEGORY_PREFIXES: List[Tuple[str, str]] = [
    ("collect_", "collect"),
    ("eat_", "collect"),
    ("drink_", "collect"),
    ("craft_", "crafting"),
    ("find_", "crafting"),
    ("fire_", "crafting"),
    ("enchant_", "crafting"),
    ("learn_", "crafting"),
    ("cast_", "crafting"),
    ("defeat_", "combat"),
    ("damage_", "combat"),
    ("enter_", "exploration"),
    ("place_", "exploration"),
    ("reach_", "exploration"),
    ("open_chest", "exploration"),
    ("wake_up", "exploration"),
]

DEFAULT_CATEGORY = "misc"

# 已识别的依赖/语义异常（只作 advisory，不自动修改现有任务）。
# 来源：builtin 模块代码审阅。不作为构建/校验失败项。
#
# 已修复并从本表移除（2026-08）：
# - craft_tools 缺 craft_wood_pickaxe；place_furnace/place_stone 缺 collect_stone；
# - defeat_skeleton 误依赖 enter_dungeon（骷髅在 L0）；
# - defeat_orc_soldier/orc_mage 误依赖 enter_sewers（兽人在 L1）；
# - learn_fireball/iceball 与 enchant_* 的因果方向反了（书/附魔台在 L3/L4，
#   且元素能力是下 L6/L7 的前置而非其结果）→ 已改为 enter_sewers/enter_vault，
#   并给 enter_fire_realm/enter_ice_realm 补上元素能力前置边。
# 余下两项涉及任务语义（成功谓词 / 公开 task_id），需产品确认后再动，
# 因为改动会影响已录制数据集的标签口径。
_KNOWN_ANOMALIES: List[Dict[str, str]] = [
    {
        "task_id": "native.defeat_enemy",
        "issue": "成功谓词覆盖 15 种敌人，combat 模块单独任务覆盖 17 种（DEFEAT_KNIGHT/DEFEAT_ARCHER 未纳入）。",
        "suggestion": "确认是否应把 KNIGHT/ARCHER 纳入 defeat_enemy（改成功谓词=改任务语义，需版本化）。",
    },
    {
        "task_id": "native.explore_dungeon",
        "issue": "与 native.reach_boss_floor 语义重复（均为 level_ge 8）。",
        "suggestion": "确认是否为有意保留的两个入口（两者都是公开 task_id，删除会破坏既有数据集）。",
    },
]


def category_of(task_id: str) -> str:
    """由 task_id 推断任务类别（builtin 模块名）。确定性映射，未命中回退 DEFAULT_CATEGORY。"""
    if task_id in _CATEGORY_OVERRIDES:
        return _CATEGORY_OVERRIDES[task_id]
    short = task_id.split(".", 1)[-1]
    for prefix, category in _CATEGORY_PREFIXES:
        if short.startswith(prefix):
            return category
    return DEFAULT_CATEGORY


def known_categories() -> List[str]:
    """返回全部已知类别（按 builtin 模块顺序）。"""
    return ["native", "collect", "crafting", "combat", "exploration"]


# ---------------------------------------------------------------------------
# 谓词解析（success_predicate -> success_type / achievements）
# ---------------------------------------------------------------------------


def _predicate_type(expr: Optional[Dict[str, Any]]) -> str:
    if not expr:
        return "always"
    type_name = str(expr.get("type", "always"))
    if type_name.startswith("field_"):
        return "field"
    return type_name


def _collect_achievements(expr: Optional[Dict[str, Any]]) -> List[str]:
    """递归收集谓词中引用的全部成就名（保持出现顺序、去重）。"""
    out: List[str] = []

    def walk(e: Any) -> None:
        if isinstance(e, dict):
            if str(e.get("type", "")) == "achievement":
                name = str(e.get("name", ""))
                if name and name not in out:
                    out.append(name)
            for value in e.values():
                walk(value)
        elif isinstance(e, (list, tuple)):
            for item in e:
                walk(item)

    walk(expr)
    return out


# ---------------------------------------------------------------------------
# 节点与图
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskNode:
    task_id: str
    version: str
    category: str
    success_type: str
    achievements: Tuple[str, ...]
    dependencies: Tuple[str, ...]
    topological_level: int = 0
    semantic_level: str = TaskSemanticLevel.ATOMIC.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "version": self.version,
            "category": self.category,
            "success_type": self.success_type,
            "achievements": list(self.achievements),
            "dependencies": list(self.dependencies),
            "topological_level": self.topological_level,
            "semantic_level": self.semantic_level,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskNode":
        return TaskNode(
            task_id=str(d["task_id"]),
            version=str(d["version"]),
            category=str(d["category"]),
            success_type=str(d["success_type"]),
            achievements=tuple(str(a) for a in d.get("achievements", [])),
            dependencies=tuple(str(x) for x in d.get("dependencies", [])),
            topological_level=int(d.get("topological_level", 0)),
            semantic_level=str(d.get("semantic_level", TaskSemanticLevel.ATOMIC.value)),
        )


@dataclass(frozen=True)
class GraphReport:
    """构建/校验报告。ok=False 时列出具体问题，供 CLI 与测试消费。"""

    node_count: int
    edge_count: int
    ok: bool
    dangling: Tuple[str, ...] = ()
    self_deps: Tuple[str, ...] = ()
    cycles: Tuple[Tuple[str, ...], ...] = ()
    duplicate_versions: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _base_semantic_level(success_type: str) -> TaskSemanticLevel:
    """基础语义层级：achievement -> ATOMIC；and/or -> COMPOSITE；其余(always/level_ge/field) -> ROOT_GOAL。

    该规则是确定性基线；子 agent 提案可通过 apply_semantic_levels 覆盖（存 JSON）。
    """
    if success_type == "achievement":
        return TaskSemanticLevel.ATOMIC
    if success_type in ("and", "or"):
        return TaskSemanticLevel.COMPOSITE
    return TaskSemanticLevel.ROOT_GOAL


class TaskGraph:
    """静态任务依赖图。

    语义：
    - 边 A -> B 表示 A 的 dependencies 含 B（B 是 A 的严格前置，必须先完成）。
    - topological_level = 从任一 root（无前置）出发的最长路径深度；root 为 0。
    - closure(task_id) 返回其全部（传递）前置任务，供 runtime planner 展开子图。
    """

    schema_version = "1"

    def __init__(self) -> None:
        self._nodes: Dict[str, TaskNode] = {}
        self._semantic_overrides: Dict[str, str] = {}
        self._provenance: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": "deterministic:registry",
            "note": "语义层级为确定性基线；agent 提案通过 apply_semantic_levels 覆盖。",
        }

    # -- 构建 --------------------------------------------------------------

    @classmethod
    def build_from_registry(cls) -> "TaskGraph":
        """从 tasks.registry 读取全部注册任务，确定性建图。

        使用每个 task_id 的最新版本（list_versions 排序后取末位）。
        """
        graph = cls()
        deps_map: Dict[str, List[str]] = {}
        for task_id in registry.list_task_ids():
            versions = registry.list_versions(task_id)
            if not versions:
                continue
            version = versions[-1]
            adapter = registry.get_task_adapter(task_id, version)
            spec = adapter.spec
            category = category_of(task_id)
            success_type = _predicate_type(spec.success_predicate)
            achievements = tuple(_collect_achievements(spec.success_predicate))
            deps = tuple(spec.dependencies)
            graph._nodes[task_id] = TaskNode(
                task_id=task_id,
                version=version,
                category=category,
                success_type=success_type,
                achievements=achievements,
                dependencies=deps,
                semantic_level=_base_semantic_level(success_type).value,
            )
            deps_map[task_id] = list(deps)
        graph._assign_topological_levels(deps_map)
        return graph

    def _assign_topological_levels(self, deps_map: Dict[str, List[str]]) -> None:
        """Kahn 拓扑排序计算最长路径深度（level）。依赖环会导致本方法抛错。"""
        from collections import deque

        # 先校验无环（环在 validate 中也会报，这里防御性提前抛错）
        in_degree: Dict[str, int] = {tid: 0 for tid in deps_map}
        children: Dict[str, List[str]] = {tid: [] for tid in deps_map}
        for tid, deps in deps_map.items():
            in_degree[tid] = len(deps)
            for d in deps:
                children.setdefault(d, []).append(tid)

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        level: Dict[str, int] = {tid: 0 for tid in deps_map}
        seen = 0
        while queue:
            tid = queue.popleft()
            seen += 1
            for child in children.get(tid, []):
                level[child] = max(level[child], level[tid] + 1)
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if seen != len(deps_map):
            raise ValueError(
                f"任务图存在环，无法计算拓扑层级（已处理 {seen}/{len(deps_map)} 节点）。"
            )
        for tid, lv in level.items():
            if tid in self._nodes:
                node = self._nodes[tid]
                self._nodes[tid] = TaskNode(
                    task_id=node.task_id,
                    version=node.version,
                    category=node.category,
                    success_type=node.success_type,
                    achievements=node.achievements,
                    dependencies=node.dependencies,
                    topological_level=lv,
                    semantic_level=node.semantic_level,
                )

    # -- 查询 --------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, TaskNode]:
        return dict(self._nodes)

    def task_ids(self) -> List[str]:
        return sorted(self._nodes)

    def node(self, task_id: str) -> TaskNode:
        if task_id not in self._nodes:
            raise KeyError(f"任务 {task_id!r} 不在图中")
        return self._nodes[task_id]

    def dependencies(self, task_id: str) -> List[str]:
        return list(self._nodes[task_id].dependencies)

    def dependents(self, task_id: str) -> List[str]:
        """直接依赖 task_id 的任务（反向边消费者）。"""
        return sorted(
            tid for tid, n in self._nodes.items() if task_id in n.dependencies
        )

    def roots(self) -> List[str]:
        """无任何前置（依赖为空）的任务。"""
        return sorted(tid for tid, n in self._nodes.items() if not n.dependencies)

    def leaves(self) -> List[str]:
        """不被任何其他任务依赖的任务。"""
        depended = {d for n in self._nodes.values() for d in n.dependencies}
        return sorted(tid for tid in self._nodes if tid not in depended)

    def closure(self, task_id: str, include_self: bool = False) -> Set[str]:
        """task_id 的全部（传递）前置任务集合。供 runtime planner 展开子图。"""
        if task_id not in self._nodes:
            raise KeyError(f"任务 {task_id!r} 不在图中")
        result: Set[str] = set()
        stack = list(self._nodes[task_id].dependencies)
        while stack:
            tid = stack.pop()
            if tid in result:
                continue
            if tid not in self._nodes:
                continue  # 悬空引用由 validate() 报告
            result.add(tid)
            stack.extend(self._nodes[tid].dependencies)
        if include_self:
            result.add(task_id)
        return result

    def subgraph(self, task_ids: Iterable[str]) -> "TaskGraph":
        """仅保留指定 task_id 的节点；依赖边限制在该集合内。"""
        keep = set(task_ids)
        g = TaskGraph()
        for tid in sorted(keep):
            if tid not in self._nodes:
                continue
            n = self._nodes[tid]
            deps = tuple(d for d in n.dependencies if d in keep)
            g._nodes[tid] = TaskNode(
                task_id=n.task_id,
                version=n.version,
                category=n.category,
                success_type=n.success_type,
                achievements=n.achievements,
                dependencies=deps,
                topological_level=n.topological_level,
                semantic_level=n.semantic_level,
            )
        g._assign_topological_levels(
            {tid: list(n.dependencies) for tid, n in g._nodes.items()}
        )
        g._provenance = dict(self._provenance)
        return g

    def category_subgraph(self, category: str) -> "TaskGraph":
        return self.subgraph(
            tid for tid, n in self._nodes.items() if n.category == category
        )

    def by_category(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for tid, n in self._nodes.items():
            out.setdefault(n.category, []).append(tid)
        return {k: sorted(v) for k, v in out.items()}

    # -- 语义层级 ----------------------------------------------------------

    def apply_semantic_levels(self, levels: Dict[str, str]) -> int:
        """应用子 agent 提案的语义层级覆盖。返回实际生效的任务数。

        值必须为 TaskSemanticLevel 的合法值；不存在的 task_id 忽略。
        """
        valid = {lv.value for lv in TaskSemanticLevel}
        applied = 0
        for tid, lv in levels.items():
            if tid not in self._nodes:
                continue
            if lv not in valid:
                raise ValueError(
                    f"任务 {tid!r} 的语义层级 {lv!r} 非法，可用 {sorted(valid)}"
                )
            self._semantic_overrides[tid] = lv
            n = self._nodes[tid]
            self._nodes[tid] = TaskNode(
                task_id=n.task_id,
                version=n.version,
                category=n.category,
                success_type=n.success_type,
                achievements=n.achievements,
                dependencies=n.dependencies,
                topological_level=n.topological_level,
                semantic_level=lv,
            )
            applied += 1
        return applied

    # -- 校验 --------------------------------------------------------------

    def validate(self) -> GraphReport:
        """校验 DAG 不变量：悬空依赖、自依赖、环、版本一致性。

        注意：悬空依赖/环会使图不可用；版本一致性为 advisory（builtin 应统一）。
        """
        dangling: List[str] = []
        self_deps: List[str] = []
        for tid, n in self._nodes.items():
            for d in n.dependencies:
                if d not in self._nodes:
                    dangling.append(f"{tid} -> {d}")
                elif d == tid:
                    self_deps.append(tid)
        cycles = self._find_cycles()
        versions: Dict[str, Set[str]] = {}
        for n in self._nodes.values():
            versions.setdefault(n.task_id, set()).add(n.version)
        dup = sorted(tid for tid, vs in versions.items() if len(vs) > 1)
        return GraphReport(
            node_count=len(self._nodes),
            edge_count=sum(len(n.dependencies) for n in self._nodes.values()),
            ok=not dangling and not self_deps and not cycles,
            dangling=tuple(dangling),
            self_deps=tuple(self_deps),
            cycles=tuple(cycles),
            duplicate_versions=tuple(dup),
        )

    def _find_cycles(self) -> List[Tuple[str, ...]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in self._nodes}
        cycles: List[Tuple[str, ...]] = []

        def dfs(start: str) -> None:
            stack = [(start, iter(self._nodes[start].dependencies))]
            color[start] = GRAY
            path: List[str] = [start]
            while stack:
                tid, it = stack[-1]
                try:
                    nxt = next(it)
                except StopIteration:
                    color[tid] = BLACK
                    stack.pop()
                    if path and path[-1] == tid:
                        path.pop()
                    continue
                if nxt not in color:
                    continue  # 悬空依赖：由 validate() 的 dangling 报告处理
                if color[nxt] == GRAY:
                    idx = path.index(nxt)
                    cycle = tuple(path[idx:] + [nxt])
                    if cycle not in cycles:
                        cycles.append(cycle)
                elif color[nxt] == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append((nxt, iter(self._nodes[nxt].dependencies)))
        for tid in self._nodes:
            if color[tid] == WHITE:
                dfs(tid)
        return cycles

    # -- 序列化 ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        report = self.validate()
        return {
            "schema_version": self.schema_version,
            "node_count": len(self._nodes),
            "edge_count": report.edge_count,
            "ok": report.ok,
            "categories": self.by_category(),
            "nodes": {tid: n.to_dict() for tid, n in sorted(self._nodes.items())},
            "semantic_overrides": dict(sorted(self._semantic_overrides.items())),
            "advisory": list(_KNOWN_ANOMALIES),
            "validation": report.to_dict(),
            "provenance": self._provenance,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskGraph":
        g = cls()
        for tid, node_dict in d["nodes"].items():
            g._nodes[tid] = TaskNode.from_dict(node_dict)
        g._semantic_overrides = dict(d.get("semantic_overrides", {}))
        if "provenance" in d:
            g._provenance = dict(d["provenance"])
        return g

    @classmethod
    def load(cls, path: str) -> "TaskGraph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def set_provenance(self, key: str, value: Any) -> None:
        self._provenance[key] = value


def build_task_graph() -> TaskGraph:
    """便捷入口：从 registry 确定性构建静态任务图。"""
    return TaskGraph.build_from_registry()
