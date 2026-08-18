"""楼层就绪门（Floor Readiness Gate）与条件规划。

在依赖图"任务顺序"之上增加一层环境感知的门控：进入/清怪某个楼层前，
按该层怪的数值（combat_model）与当前装备/属性（summary）判定"就绪度"，
把缺失项（sword/armour/strength/elemental/clear/survival）返回给执行器，
由执行器用原语技能补齐（造剑/造甲/升级/学元素/清上一层）。

- check_floor_readiness：纯函数判定，返回 (ok, missing[])；
- build_plan：离线分析用的有序 PlanSteps（链 + 门控标注）；
- 执行器侧 resolve_gate 见 executor.py（需地图与合成原语）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from craftax.planner.combat_model import Gear, survival_verdict
from craftax.planner.world import WorldFacts

# ---------------------------------------------------------------------------
# 每层最低就绪门槛（初始建议值；按慢速集成测试标定）
#   sword:   最低剑等级（1木 2石 3铁 4钻）
#   armour:  最低铁甲件数（每件 10% 物免）
#   strength:最低力量（默认执行器力量优先，此值为"建议线"）
#   elemental: L6/L7 需要对应元素能力
# ---------------------------------------------------------------------------
FLOOR_GEAR_REQ: Dict[int, Dict[str, Any]] = {
    1: {"sword": 2, "strength": 2},                # orc 3dmg/7HP
    2: {"sword": 3, "armour": 1, "strength": 2},   # gnome 4dmg/9HP
    3: {"sword": 3, "armour": 1, "strength": 3},   # lizard 5dmg/11HP
    4: {"sword": 3, "armour": 2, "strength": 3},   # knight 6dmg/12HP 50%物免
    5: {"sword": 4, "armour": 2, "strength": 4},   # troll 8dmg/20HP 20%物免
    6: {"sword": 3, "armour": 2, "strength": 4, "elemental": True},  # pigman
    7: {"sword": 4, "armour": 2, "strength": 4, "elemental": True},  # ice troll
    8: {"sword": 4, "armour": 4, "strength": 5},   # boss
}


def gear_from_summary(summary: Dict[str, Any], has_elemental: bool) -> Gear:
    """从 state_summary 构建战斗模型的 Gear。"""
    inventory = summary.get("inventory") or {}
    return Gear(
        sword=int(inventory.get("sword", 0)),
        armour=sum(int(x) for x in inventory.get("armour", [0])),
        strength=int(summary.get("strength", 1)),
        dexterity=int(summary.get("dexterity", 1)),
        intelligence=int(summary.get("intelligence", 1)),
        has_elemental=has_elemental,
    )


def has_elemental(summary: Dict[str, Any], floor: int) -> bool:
    """L6(火界) 需冰系、L7(冰界) 需火系；判断元素能力是否具备。

    与 executor._has_elemental_capability 语义一致。
    """
    if floor not in (6, 7):
        return True
    need_ice = floor == 6
    sword_ench = int(summary.get("sword_enchantment", 0))
    bow_ench = int(summary.get("bow_enchantment", 0))
    spells = summary.get("learned_spells") or [False, False]
    iceball = bool(spells[1]) if len(spells) > 1 else False
    fireball = bool(spells[0]) if len(spells) > 0 else False
    if need_ice:
        return sword_ench == 2 or bow_ench == 2 or iceball
    return sword_ench == 1 or bow_ench == 1 or fireball


def check_floor_readiness(
    floor: int,
    summary: Dict[str, Any],
    world_facts: Optional[WorldFacts] = None,
) -> Tuple[bool, List[Tuple[str, Any]]]:
    """目标楼层 floor 的就绪门判定。

    返回 (ok, missing)。missing 元素为 (kind, value)：
    - ("sword", min_level)：剑等级不足；
    - ("armour", pieces)：铁甲件数不足；
    - ("strength", min_str)：力量不足（软门槛，力量优先升级可自动满足）；
    - ("elemental", floor)：L6/L7 缺元素能力；
    - ("survival", verdict)：combat_model 判定 INFEASIBLE。
    """
    missing: List[Tuple[str, Any]] = []
    req = FLOOR_GEAR_REQ.get(floor, {})
    inventory = summary.get("inventory") or {}
    sword = int(inventory.get("sword", 0))
    armour = sum(int(x) for x in inventory.get("armour", [0]))
    strength = int(summary.get("strength", 1))

    if req.get("sword", 0) > sword:
        missing.append(("sword", req["sword"]))
    if req.get("armour", 0) > armour:
        missing.append(("armour", req["armour"]))
    if req.get("strength", 1) > strength:
        missing.append(("strength", req["strength"]))
    if req.get("elemental") and not has_elemental(summary, floor):
        missing.append(("elemental", floor))

    elem = has_elemental(summary, floor)
    gear = gear_from_summary(summary, elem)
    verdict = survival_verdict(floor, gear)
    if verdict == "INFEASIBLE":
        missing.append(("survival", verdict))

    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# 离线条件规划（供分析/测试/文档演示）
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    task_id: str
    kind: str                          # collect/craft/combat/descend/...
    gates: List[str] = field(default_factory=list)   # 前置门控描述
    target_floor: Optional[int] = None


@dataclass
class Plan:
    root_task: str
    steps: List[PlanStep] = field(default_factory=list)

    def describe(self) -> List[str]:
        out = []
        for s in self.steps:
            line = f"{s.task_id} [{s.kind}]"
            if s.target_floor is not None:
                line += f" (floor {s.target_floor})"
            if s.gates:
                line += " gates: " + ", ".join(s.gates)
            out.append(line)
        return out


def build_plan(
    task_id: str,
    world_facts: Optional[WorldFacts] = None,
) -> Plan:
    """从依赖图 closure 构造有序 PlanSteps，并按楼层就绪门槛标注门控。"""
    from craftax.tasks.graph import TaskGraph

    graph = TaskGraph.build_from_registry()
    closure = graph.closure(task_id, include_self=True)
    chain = sorted(
        closure, key=lambda t: (graph.node(t).topological_level, t)
    )

    from craftax.planner.executor import (
        COLLECT_TARGET_BLOCKS,
        CRAFT_ACTIONS,
        DEFEAT_TASKS,
        ENTER_FLOOR,
        LEARN_FLOOR,
        REACH_FLOOR,
    )

    steps: List[PlanStep] = []
    for tid in chain:
        if tid in CRAFT_ACTIONS:
            kind = "craft"
        elif tid in COLLECT_TARGET_BLOCKS or tid.startswith("native.collect_"):
            kind = "collect"
        elif tid in DEFEAT_TASKS:
            kind = "combat"
        elif tid in ENTER_FLOOR or tid in REACH_FLOOR:
            kind = "descend"
        elif tid in LEARN_FLOOR:
            kind = "learn"
        else:
            kind = "action"
        target = None
        gates: List[str] = []
        if tid in ENTER_FLOOR or tid in REACH_FLOOR:
            target = REACH_FLOOR.get(tid, ENTER_FLOOR[tid])
            req = FLOOR_GEAR_REQ.get(target, {})
            if req:
                gates = [f"{k}={v}" for k, v in req.items()]
        elif tid in DEFEAT_TASKS:
            from craftax.planner.executor import DEFEAT_MOB_LOCATIONS

            loc = DEFEAT_MOB_LOCATIONS.get(tid)
            if loc:
                target = max(loc[2]) if loc[2] else None
        steps.append(PlanStep(tid, kind, gates, target))
    return Plan(task_id, steps)
