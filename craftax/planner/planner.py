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

from craftax.planner.combat_model import Gear, arrows_for_clear, survival_verdict
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
        bow=int(inventory.get("bow", 0)),
        bow_enchant=int(summary.get("bow_enchantment", 0)),
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


def _armour_reachable(
    world_facts: Optional[WorldFacts], floor: int, pieces: int
) -> bool:
    """按 seed 事实判断"还能再做出 pieces 件铁甲吗"（每件 3 铁 + 3 煤）。

    无扫描数据（world_facts 为 None 或该 seed 未扫描）时返回 True——
    未知不等于不可行，交由执行器就地尝试（原有行为）。
    """
    if world_facts is None or pieces <= 0:
        return True
    floors = list(range(0, max(floor, 0) + 1))
    known = any(world_facts.floor(f) is not None for f in floors)
    if not known:
        return True
    return world_facts.armor_pieces_feasible(floors, pieces=pieces)


def check_floor_readiness(
    floor: int,
    summary: Dict[str, Any],
    world_facts: Optional[WorldFacts] = None,
    arrows_restockable: bool = False,
) -> Tuple[bool, List[Tuple[str, Any]]]:
    """目标楼层 floor 的就绪门判定。

    返回 (ok, missing)。missing 元素为 (kind, value)：
    - ("sword", min_level)：剑等级不足；
    - ("armour", pieces)：铁甲件数不足；
    - ("strength", min_str)：力量不足（软门槛，力量优先升级可自动满足）；
    - ("elemental", floor)：L6/L7 缺元素能力；
    - ("survival", verdict)：combat_model 判定 INFEASIBLE；
    - ("ladder", floor)：本 seed 的梯子链在 floor 之前就断了（硬中止）。

    world_facts（本 seed 的跨层事实）用于把"本层看不见"的信息纳入判定：
    - 铁/煤在 floor 及其以上层都不够 → 铁甲门槛不可满足，不再要求（软化门槛，
      避免执行器为一件永远做不出的甲反复采矿）；
    - 梯子链不可达 → 直接给出硬中止项，不必下去撞墙。

    arrows_restockable：目标层能否就地合成箭（1 木 + 1 石）。弓能否顶替剑/甲
    取决于**弹药够不够清满 8 怪**：地牢层没有树，箭不可再生，带下去的箭打完
    就只剩剑——所以此时剑/甲门槛必须恢复（见 §6.2.6a 出路 a）。
    """
    missing: List[Tuple[str, Any]] = []
    req = FLOOR_GEAR_REQ.get(floor, {})
    inventory = summary.get("inventory") or {}
    sword = int(inventory.get("sword", 0))
    armour = sum(int(x) for x in inventory.get("armour", [0]))
    strength = int(summary.get("strength", 1))
    # 弓（L1 首箱必出）覆盖 L1-L3 的清怪需求：箭 1-3 发/怪、0-1 受击，
    # 不再强制剑/甲作为下楼前置（L4 起骑士/巨魔物免高，仍需剑/甲或附魔）。
    # 但"覆盖"以弹药充足为前提——不可补弹层上箭数不够清满 8 怪时，弓只能覆盖
    # 前半段，残余仍要用剑打，此时不能豁免剑/甲门槛。
    has_bow = int(inventory.get("bow", 0)) >= 1
    elem_now = has_elemental(summary, floor)
    gear = gear_from_summary(summary, elem_now)
    arrows_needed = arrows_for_clear(floor, gear, restockable=arrows_restockable)
    ammo_ok = arrows_restockable or (
        arrows_needed > 0 and int(inventory.get("arrows", 0)) >= arrows_needed
    )
    bow_covers_gear = has_bow and floor <= 3 and ammo_ok

    if req.get("sword", 0) > sword and not bow_covers_gear:
        missing.append(("sword", req["sword"]))
    need_armour = req.get("armour", 0)
    if need_armour > armour and not bow_covers_gear:
        if _armour_reachable(world_facts, floor, need_armour - armour):
            missing.append(("armour", need_armour))
    if req.get("strength", 1) > strength:
        missing.append(("strength", req["strength"]))
    if req.get("elemental") and not elem_now:
        missing.append(("elemental", floor))
    if world_facts is not None and not world_facts.reaches(floor):
        current = int(summary.get("floor", 0))
        if floor > current:  # 已经站在该层就不必再质疑可达性
            missing.append(("ladder", floor))

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
    """从依赖图 closure 构造有序 PlanSteps，并按楼层就绪门槛标注门控。

    顺序与执行器一致（同一个 build_task_chain：成本感知的拓扑排序），
    使离线看到的计划就是运行时会走的计划。
    """
    from craftax.planner.executor import (
        COLLECT_TARGET_BLOCKS,
        CRAFT_ACTIONS,
        DEFEAT_TASKS,
        ENTER_FLOOR,
        LEARN_FLOOR,
        REACH_FLOOR,
        build_task_chain,
        task_target_floor,
    )

    chain = build_task_chain(task_id, world_facts=world_facts)

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
        target = task_target_floor(tid, world_facts=world_facts)
        gates: List[str] = []
        if tid in ENTER_FLOOR or tid in REACH_FLOOR:
            req = FLOOR_GEAR_REQ.get(target or 0, {})
            if req:
                gates = [f"{k}={v}" for k, v in req.items()]
        steps.append(PlanStep(tid, kind, gates, target))
    return Plan(task_id, steps)
