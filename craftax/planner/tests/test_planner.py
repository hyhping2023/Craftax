"""planner.py 单测：楼层就绪门与条件规划。"""
from __future__ import annotations

import pytest

from craftax.planner.planner import (
    FLOOR_GEAR_REQ,
    build_plan,
    check_floor_readiness,
    gear_from_summary,
    has_elemental,
)
from craftax.planner.world import WorldFacts

SCAN = "data/seed_scan.json"


def _summary(**overrides):
    inv = overrides.pop("inventory", {})
    default_inv = {
        "wood": 0, "stone": 0, "coal": 0, "iron": 0, "diamond": 0,
        "pickaxe": 0, "sword": 0, "armour": [0, 0, 0, 0],
    }
    default_inv.update(inv)
    base = {
        "floor": 0,
        "strength": 1,
        "dexterity": 1,
        "intelligence": 1,
        "sword_enchantment": 0,
        "bow_enchantment": 0,
        "learned_spells": [False, False],
        "inventory": default_inv,
    }
    base.update(overrides)
    return base


def test_gear_from_summary():
    g = gear_from_summary(_summary(inventory={"sword": 3, "armour": [1, 1, 0, 0]},
                                   strength=3, dexterity=2), has_elemental=True)
    assert g.sword == 3
    assert g.armour == 2
    assert g.strength == 3
    assert g.dexterity == 2
    assert g.has_elemental is True


def test_has_elemental():
    # L0/L1 不需要元素能力
    assert has_elemental(_summary(), 1) is True
    # L6 需要冰（sword_enchantment==2 或 iceball）
    assert has_elemental(_summary(), 6) is False
    assert has_elemental(_summary(sword_enchantment=2), 6) is True
    assert has_elemental(_summary(learned_spells=[False, True]), 6) is True
    assert has_elemental(_summary(sword_enchantment=1), 6) is False  # 火附魔对 L6 无效
    # L7 需要火
    assert has_elemental(_summary(sword_enchantment=1), 7) is True
    assert has_elemental(_summary(learned_spells=[True, False]), 7) is True


def test_check_floor_readiness_ok():
    s = _summary(
        inventory={"sword": 2, "armour": [0, 0, 0, 0]},
        strength=2,
    )
    ok, missing = check_floor_readiness(1, s)
    assert ok, missing


def test_check_floor_readiness_missing_sword_armour():
    s = _summary(inventory={"sword": 1, "armour": [0, 0, 0, 0]}, strength=2)
    ok, missing = check_floor_readiness(2, s)
    assert not ok
    kinds = [m[0] for m in missing]
    assert "sword" in kinds
    assert "armour" in kinds


def test_check_floor_readiness_elemental_gate():
    s = _summary(
        inventory={"sword": 4, "armour": [1, 1, 1, 1]}, strength=5,
    )
    ok, missing = check_floor_readiness(6, s)
    assert not ok
    assert ("elemental", 6) in missing
    # 学冰球后通过元素门
    s2 = dict(s)
    s2["learned_spells"] = [False, True]
    ok2, missing2 = check_floor_readiness(6, s2)
    assert ok2, missing2


def test_check_floor_readiness_hopeless_descent_blocked():
    # 木剑(0) + 无甲 + 力量1 下 L2：sword/armour 门槛拦截；combat 判定 MARGINAL（可打但危险）
    s = _summary(inventory={"sword": 0, "armour": [0, 0, 0, 0]}, strength=1)
    ok, missing = check_floor_readiness(2, s)
    assert not ok
    kinds = [m[0] for m in missing]
    assert "sword" in kinds
    assert "armour" in kinds
    # MARGINAL 作为软警告放行，不出现在缺失里（INFEASIBLE 才硬拦截）
    assert "survival" not in kinds


def test_check_floor_readiness_elemental_also_survival_infeasible():
    # L6 无元素能力：元素门 + combat_model INFEASIBLE 双重拦截
    s = _summary(inventory={"sword": 4, "armour": [1, 1, 1, 1]}, strength=5)
    ok, missing = check_floor_readiness(6, s)
    assert not ok
    assert ("elemental", 6) in missing
    assert ("survival", "INFEASIBLE") in missing


def test_build_plan_has_gear_gates():
    plan = build_plan("native.collect_diamond")
    desc = plan.describe()
    assert plan.root_task == "native.collect_diamond"
    # 依赖在前：collect_wood 先于 craft_wood_pickaxe
    order = [s.task_id for s in plan.steps]
    assert order.index("native.collect_wood") < order.index("native.craft_wood_pickaxe")
    assert order.index("native.craft_wood_pickaxe") < order.index("native.craft_stone_pickaxe")


def test_build_plan_combat_floor():
    plan = build_plan("native.defeat_gnome_warrior")
    combat = [s for s in plan.steps if s.task_id == "native.defeat_gnome_warrior"]
    assert combat and combat[0].target_floor == 2
    # 楼层门控标注在 enter_dungeon / enter_gnomish_mines 上
    assert any("floor 2" in d and "gates:" in d for d in plan.describe())


def test_floor_gear_req_completeness():
    # L1-L8 都有门槛定义
    assert set(FLOOR_GEAR_REQ) == set(range(1, 9))


# ---------------------------------------------------------------------------
# WorldFacts 真正参与门控（旧实现里 world_facts 是个从不被读的死参数）
# ---------------------------------------------------------------------------


def _facts(per_floor):
    """构造一个最小 WorldFacts：per_floor[floor] = (ladder_down, ore dict)。"""
    from craftax.planner.world import FloorFacts

    floors = {}
    for floor, (ladder, ore) in per_floor.items():
        floors[floor] = FloorFacts(floor=floor, ore=dict(ore),
                                   ladder_down_reachable=ladder)
    return WorldFacts(seed=-1, floors=floors)


def test_readiness_reports_broken_ladder_chain():
    """seed 的梯子链在中途断掉 → 给出硬中止项 ("ladder", floor)。"""
    facts = _facts({0: (True, {}), 1: (False, {}), 2: (True, {})})
    s = _summary(inventory={"sword": 4, "armour": [1, 1, 1, 1]}, strength=5)
    ok, missing = check_floor_readiness(3, s, facts)
    assert not ok
    assert ("ladder", 3) in missing


def test_readiness_drops_unsatisfiable_armour_requirement():
    """铁/煤在目标层及以上都不够 → 不再要求护甲（否则执行器为一件永远做不出
    的甲反复采矿）。资源够时仍然要求。"""
    s = _summary(inventory={"sword": 3, "armour": [0, 0, 0, 0], "bow": 0}, strength=5)
    poor = _facts({0: (True, {"iron": 1, "coal": 1}), 1: (True, {}), 2: (True, {})})
    ok, missing = check_floor_readiness(2, s, poor)
    assert not any(k == "armour" for k, _ in missing)
    assert ok  # 其余门槛已满足 → 不因不可能的护甲要求卡住下行
    rich = _facts({0: (True, {"iron": 6, "coal": 6}), 1: (True, {}), 2: (True, {})})
    _, missing_rich = check_floor_readiness(2, s, rich)
    assert ("armour", 1) in missing_rich


def test_build_plan_never_raises_for_reach_floor_tasks():
    """回归：REACH_FLOOR.get(tid, ENTER_FLOOR[tid]) 的默认值会被立即求值，
    导致 reach_floor_* / explore_dungeon 的计划构造直接 KeyError。"""
    for task_id in (
        "native.reach_floor_3",
        "native.reach_floor_5",
        "native.reach_boss_floor",
        "native.explore_dungeon",
        "native.dungeon_campaign",
    ):
        plan = build_plan(task_id)
        assert plan.steps, task_id


def test_build_plan_matches_executor_chain():
    """离线计划顺序 == 执行器实际任务链顺序（同一个 build_task_chain）。"""
    from craftax.planner.executor import build_task_chain

    for task_id in ("native.defeat_necromancer", "native.collect_diamond"):
        plan = build_plan(task_id)
        assert [s.task_id for s in plan.steps] == build_task_chain(task_id)
