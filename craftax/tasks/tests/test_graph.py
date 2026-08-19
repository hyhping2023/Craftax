"""craftax/tasks/graph 静态任务依赖图测试。

不依赖环境 reset/step（图构建只读 registry），因此本文件无需 JAX 编译。
"""
from __future__ import annotations

import pytest

from craftax.tasks.agent_spec import validate_proposals
from craftax.tasks.graph import (
    TaskGraph,
    TaskSemanticLevel,
    build_task_graph,
    category_of,
)

# 77 个基础任务 + 16 个 agent 提议的分级任务（hierarchy_tasks.py）
# 16 = 原 15 + native.build_shelter（2026-08-19：把"搭掩体防守"显式建成任务节点）
BASE_TASK_COUNT = 77
HIERARCHY_TASK_COUNT = 16
TOTAL_TASK_COUNT = BASE_TASK_COUNT + HIERARCHY_TASK_COUNT


@pytest.fixture(scope="module")
def graph() -> TaskGraph:
    return build_task_graph()


# ---------------------------------------------------------------------------
# 构建与校验
# ---------------------------------------------------------------------------


def test_build_from_registry_counts(graph):
    assert len(graph.task_ids()) == TOTAL_TASK_COUNT


def test_graph_is_valid_dag(graph):
    report = graph.validate()
    assert report.ok, report.to_dict()
    assert report.node_count == TOTAL_TASK_COUNT
    assert report.dangling == ()
    assert report.self_deps == ()
    assert report.cycles == ()
    assert report.duplicate_versions == ()


def test_no_dangling_hierarchy_deps(graph):
    for tid in graph.task_ids():
        for dep in graph.dependencies(tid):
            assert dep in graph.task_ids(), f"{tid} 依赖 {dep} 悬空"


def test_validate_detects_dangling():
    g = TaskGraph.from_dict(
        {
            "nodes": {
                "native.a": {
                    "task_id": "native.a",
                    "version": "1.0.0",
                    "category": "misc",
                    "success_type": "achievement",
                    "achievements": ["COLLECT_WOOD"],
                    "dependencies": ["native.missing"],
                    "topological_level": 0,
                    "semantic_level": "atomic",
                }
            }
        }
    )
    report = g.validate()
    assert not report.ok
    assert "native.a -> native.missing" in report.dangling


def test_validate_detects_cycle():
    g = TaskGraph.from_dict(
        {
            "nodes": {
                "native.a": {
                    "task_id": "native.a",
                    "version": "1.0.0",
                    "category": "misc",
                    "success_type": "achievement",
                    "achievements": [],
                    "dependencies": ["native.b"],
                    "topological_level": 0,
                    "semantic_level": "atomic",
                },
                "native.b": {
                    "task_id": "native.b",
                    "version": "1.0.0",
                    "category": "misc",
                    "success_type": "achievement",
                    "achievements": [],
                    "dependencies": ["native.a"],
                    "topological_level": 0,
                    "semantic_level": "atomic",
                },
            }
        }
    )
    report = g.validate()
    assert not report.ok
    assert report.cycles


# ---------------------------------------------------------------------------
# 拓扑层级与闭包
# ---------------------------------------------------------------------------


def test_roots_have_zero_level(graph):
    for tid in graph.roots():
        assert graph.node(tid).topological_level == 0


def test_floor_chain_levels(graph):
    """enter_* 楼层链应严格递增层级。"""
    levels = {
        "native.enter_dungeon": 0,
        "native.enter_gnomish_mines": 1,
        "native.enter_sewers": 2,
        "native.enter_vault": 3,
        "native.enter_troll_mines": 4,
        "native.enter_fire_realm": 5,
        "native.enter_ice_realm": 6,
        "native.enter_graveyard": 7,
    }
    for tid, expected in levels.items():
        assert graph.node(tid).topological_level == expected, tid


def test_reach_boss_floor_is_deep(graph):
    assert graph.node("native.reach_boss_floor").topological_level >= 8
    # 最深层级由组合任务（collect_all_gems 等）占据
    max_level = max(n.topological_level for n in graph.nodes.values())
    assert max_level >= graph.node("native.reach_boss_floor").topological_level
    # 层级单调性：依赖层级必须严格小于自身层级
    for tid, n in graph.nodes.items():
        for dep in n.dependencies:
            assert graph.node(dep).topological_level < n.topological_level, (
                f"{dep}({graph.node(dep).topological_level}) -> {tid}({n.topological_level})"
            )


def test_closure_collect_all_gems(graph):
    closure = graph.closure("native.collect_all_gems")
    assert "native.collect_sapphire" in closure
    assert "native.collect_ruby" in closure
    assert "native.craft_diamond_pickaxe" in closure
    assert "native.collect_wood" in closure
    assert "native.collect_all_gems" not in closure  # 默认不含自身


def test_closure_include_self(graph):
    closure = graph.closure("native.collect_all_gems", include_self=True)
    assert "native.collect_all_gems" in closure


def test_closure_hierarchy_root(graph):
    closure = graph.closure("native.conquer_dungeon_bosses")
    assert "native.defeat_troll" in closure
    assert "native.defeat_knight" in closure
    assert "native.defeat_necromancer" in closure
    assert "native.enter_graveyard" in closure


# ---------------------------------------------------------------------------
# 类别划分
# ---------------------------------------------------------------------------


def test_category_of_base_tasks():
    assert category_of("native.collect_wood") == "native"
    assert category_of("native.collect_stone") == "collect"
    assert category_of("native.craft_iron_pickaxe") == "crafting"
    assert category_of("native.defeat_zombie") == "combat"
    assert category_of("native.enter_dungeon") == "exploration"


def test_category_of_hierarchy_tasks(graph):
    assert category_of("native.dungeon_campaign") == "native"
    assert category_of("native.collect_all_ores") == "collect"
    assert category_of("native.iron_gear") == "crafting"
    assert category_of("native.conquer_dungeon_bosses") == "combat"
    assert category_of("native.build_home_base") == "exploration"


def test_category_subgraph_contains_all(graph):
    for category in ("native", "collect", "crafting", "combat", "exploration"):
        sub = graph.category_subgraph(category)
        assert len(sub.task_ids()) > 0, category
        for tid in sub.task_ids():
            assert graph.node(tid).category == category


# ---------------------------------------------------------------------------
# 语义层级
# ---------------------------------------------------------------------------


def test_semantic_levels_base_rule(graph):
    assert graph.node("native.survive").semantic_level == TaskSemanticLevel.ROOT_GOAL.value
    assert graph.node("native.collect_wood").semantic_level == TaskSemanticLevel.ATOMIC.value
    assert (
        graph.node("native.craft_full_kit").semantic_level
        == TaskSemanticLevel.COMPOSITE.value
    )
    assert (
        graph.node("native.explore_dungeon").semantic_level
        == TaskSemanticLevel.ROOT_GOAL.value
    )


def test_semantic_levels_hierarchy_tasks_artifact():
    """agent 显式语义层级覆盖应存在于生成的 task_graph.json 中。"""
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "task_graph.json")
    if not os.path.exists(path):
        pytest.skip("task_graph.json 未生成")
    g = TaskGraph.load(path)
    assert (
        g.node("native.conquer_dungeon_bosses").semantic_level
        == TaskSemanticLevel.ROOT_GOAL.value
    )
    assert (
        g.node("native.clear_surface_threats").semantic_level
        == TaskSemanticLevel.COMPOSITE.value
    )


def test_apply_semantic_levels_override(graph):
    g = graph.subgraph(["native.collect_wood"])
    g.apply_semantic_levels({"native.collect_wood": "root_goal"})
    assert g.node("native.collect_wood").semantic_level == "root_goal"


def test_apply_semantic_levels_invalid_value(graph):
    g = graph.subgraph(["native.collect_wood"])
    with pytest.raises(ValueError):
        g.apply_semantic_levels({"native.collect_wood": "bogus"})


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def test_json_round_trip(graph, tmp_path):
    path = tmp_path / "graph.json"
    graph.save(str(path))
    loaded = TaskGraph.load(str(path))
    report = loaded.validate()
    assert report.ok
    assert loaded.task_ids() == graph.task_ids()
    for tid in graph.task_ids():
        assert loaded.node(tid).to_dict() == graph.node(tid).to_dict()


def test_saved_artifact_matches_registry():
    """仓库内 task_graph.json 应与 registry 实时构建一致。"""
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "task_graph.json")
    if not os.path.exists(path):
        pytest.skip("task_graph.json 未生成")
    artifact = TaskGraph.load(path)
    live = build_task_graph()
    report = artifact.validate()
    assert report.ok
    assert artifact.task_ids() == live.task_ids()


# ---------------------------------------------------------------------------
# 子 agent 提案校验（graph/agent_spec 契约）
# ---------------------------------------------------------------------------


def _valid_proposal(predicate, deps):
    return {
        "task_id": "native.probe_new_goal",
        "instruction_en": "Probe.",
        "instruction_zh": "探测。",
        "objective": "probe",
        "success_predicate": predicate,
        "dependencies": deps,
        "rationale": "test",
    }


def test_validate_proposals_accepts_valid(graph):
    proposals = {
        "collect": {
            "category": "collect",
            "semantic_levels": {},
            "proposed_tasks": [
                _valid_proposal(
                    {
                        "type": "and",
                        "predicates": [
                            {"type": "achievement", "name": "COLLECT_WOOD"},
                            {"type": "achievement", "name": "COLLECT_STONE"},
                        ],
                    },
                    ["native.collect_wood", "native.craft_wood_pickaxe"],
                )
            ],
        }
    }
    accepted, rejected = validate_proposals(graph, proposals)
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0]["task_id"] == "native.probe_new_goal"


def test_validate_proposals_rejects_bad_achievement(graph):
    proposals = {
        "collect": {
            "category": "collect",
            "semantic_levels": {},
            "proposed_tasks": [
                _valid_proposal(
                    {"type": "achievement", "name": "NOT_A_REAL_ACHIEVEMENT"}, []
                )
            ],
        }
    }
    accepted, rejected = validate_proposals(graph, proposals)
    assert accepted == []
    assert any("Achievement" in str(r.get("reasons", r.get("reason", ""))) for r in rejected)


def test_validate_proposals_rejects_conflict_with_existing(graph):
    proposals = {
        "collect": {
            "category": "collect",
            "semantic_levels": {},
            "proposed_tasks": [
                _valid_proposal({"type": "achievement", "name": "COLLECT_WOOD"}, [])
            ],
        }
    }
    proposals["collect"]["proposed_tasks"][0]["task_id"] = "native.collect_wood"
    accepted, rejected = validate_proposals(graph, proposals)
    assert accepted == []
    assert any("冲突" in str(r.get("reasons", "")) for r in rejected)


def test_validate_proposals_rejects_dangling_dependency(graph):
    proposals = {
        "collect": {
            "category": "collect",
            "semantic_levels": {},
            "proposed_tasks": [
                _valid_proposal(
                    {"type": "achievement", "name": "COLLECT_WOOD"},
                    ["native.no_such_task"],
                )
            ],
        }
    }
    accepted, rejected = validate_proposals(graph, proposals)
    assert accepted == []
    assert any("依赖" in str(r.get("reasons", "")) for r in rejected)
