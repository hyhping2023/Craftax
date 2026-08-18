"""战斗/生存模型：把游戏数值映射为"清层可行性"判定（纯函数，可独立测试）。

数据来源（与 craftax/craftax/constants.py 保持一致，见 test_combat_model.py 交叉校验）：
- FLOOR_MOB_MAPPING：每层 (passive, melee, ranged) 的怪 type_id
- MOB_TYPE_DAMAGE_MAPPING：各 type 的近战/投射伤害向量
- MOB_TYPE_HEALTH_MAPPING：每层 (passive, melee, ranged) HP
- MOB_TYPE_DEFENSE_MAPPING：每层 (passive, melee, ranged) 防御向量

游戏机制要点（供标定参考）：
- 下楼需本层 monsters_killed >= 8；
- 怪相邻且 attack_cooldown<=0 才攻击，命中后冷却重置 5（每回合递减 1）；
- 新刷怪槽冷却恒 <=0 → 每次接战约命中 1 次，之后 5 回合内可安全击杀；
- SLEEP 受击 x3.5、回血 x2；REST 受击 x1、回血 x1（26 步/HP）；醒着 26 步/HP；
- 铁甲每件 10% 物免（满 4 件 40%）；力量每点 +25% 物伤 +1 血；敏捷每点 +2 能量上限；
- L6(火) / L7(冰) 怪 90% 物免 + 对应对应元素免疫 → 必须元素能力（剑/弓附魔或法术）。

本模块只做数值估算；批量清怪 + 回上一层锚点恢复的"确定性生存机制"在 executor 中实现，
模型判定的 CLEARABLE/MARGINAL/INFEASIBLE 用于楼层就绪门（避免必死下行）与战术选择。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# 标定系数（对真实运行 adjust；脚本 scripts/measure_combat.py 可回填）
# ---------------------------------------------------------------------------

# 每层怪表：floor -> (melee_dmg_total, melee_hp, melee_phys_def,
#                     ranged_dmg_total, ranged_hp, ranged_phys_def,
#                     requires_elemental)
# melee_dmg_total = 物理+火+冰之和（护甲减免统一作用于三系）；防御仅物理。
# requires_elemental：L6 火免需冰、L7 冰免需火（90% 物免 + 对应元素免疫）。
MOB_STATS: Dict[int, Tuple[float, float, float, float, float, float, bool]] = {
    0: (2.0, 5.0, 0.0, 2.0, 3.0, 0.0, False),   # zombie / skeleton
    1: (3.0, 7.0, 0.0, 3.0, 5.0, 0.0, False),   # orc / orc mage (fireball)
    2: (4.0, 9.0, 0.0, 4.0, 6.0, 0.0, False),   # gnome warrior / archer
    3: (5.0, 11.0, 0.0, 3.0, 8.0, 0.0, False),  # lizard / kobold (iceball)
    4: (6.0, 12.0, 0.5, 5.0, 12.0, 0.5, False), # knight / archer (50% 物免)
    5: (8.0, 20.0, 0.2, 10.0, 4.0, 0.0, False), # troll (近战 20% 物免) / slime
    6: (8.0, 20.0, 0.9, 8.0, 14.0, 0.9, True),  # pigman（火界）
    7: (9.0, 24.0, 0.9, 9.0, 16.0, 0.9, True),  # ice troll（冰界）
    8: (6.0, 40.0, 0.0, 6.0, 40.0, 0.0, False), # boss 层（波次怪估算；Boss 血量另计）
}

SWORD_DAMAGE: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0)
STRENGTH_DAMAGE_PER_POINT = 0.25   # 力量每点 +25% 物伤（力量 5 = 双倍）
ELEMENTAL_FRACTION = 0.5           # 附魔元素伤 = 物伤 x 0.5
INT_DAMAGE_PER_POINT = 0.05        # 附魔元素伤随智力 +5%/点
SPELL_BASE_DAMAGE = 3.0            # 火球/冰球基础 3
INT_SPELL_PER_POINT = 0.5          # 法术随智力 +50%/点
MAX_ATTRIBUTE = 5

MAX_HEALTH_BASE = 8.0
HEALTH_PER_STRENGTH = 1.0
MAX_ENERGY_BASE = 7.0
ENERGY_PER_DEXTERITY = 2.0
ENERGY_DECAY_PER_DEX = 0.125       # 每点敏捷 -12.5% 疲劳/口渴增速

HIT_COOLDOWN_TURNS = 5.0           # 怪命中后冷却 5 回合（贴脸 stand）
KITE_EFFECTIVE_PERIOD = 7.0        # 风筝时怪有效攻击周期（冷却 5 + 撤退/回身 2）
HITS_PER_KILL_BASE = 1.0           # 每次接战必中 1 次（新怪冷却<=0）
APPROACH_HITS = 0.2                # 走位/追击期间额外受击系数（标定项）
KITE_APPROACH_HITS = 0.1           # 风筝走位更少受击（标定项）

ARMOR_REDUCTION_PER_PIECE = 0.1    # 铁甲每件 10% 物免
CLEAR_TARGET = 8                   # 每层需杀怪数
PASSIVE_REGEN_STEPS_PER_HP = 26.0  # 醒着回 1 血所需步数
ENERGY_STEPS_PER_POINT = 31.0      # 醒着每消耗 1 能量所需步数
BATCH_ABORT_HEALTH = 4.0           # 清怪中止血量（低于则回上层恢复）
BATCH_STEPS_OVERHEAD = 40.0        # 每批往返/等待刷怪的固定步数开销（标定项）
STEPS_PER_KILL_APPROACH = 4.0      # 每怪寻路/追击步数（标定项）


@dataclass(frozen=True)
class Gear:
    """玩家装备/属性快照（用于战斗模型估算）。"""
    sword: int = 0
    armour: int = 0
    strength: int = 1
    dexterity: int = 1
    intelligence: int = 1
    has_elemental: bool = False     # 对 L6/7：剑/弓附魔或对应法术已具备


# ---------------------------------------------------------------------------
# 玩家/怪数值
# ---------------------------------------------------------------------------


def max_health(strength: int) -> float:
    return MAX_HEALTH_BASE + HEALTH_PER_STRENGTH * (strength - 1)


def max_energy(dexterity: int) -> float:
    return MAX_ENERGY_BASE + ENERGY_PER_DEXTERITY * (dexterity - 1)


def energy_decay_factor(dexterity: int) -> float:
    return 1.0 - ENERGY_DECAY_PER_DEX * (dexterity - 1)


def player_melee_dps(sword: int, strength: int) -> float:
    """玩家近战物伤 DPS（未扣除怪防御）。"""
    return SWORD_DAMAGE[sword] * (1.0 + STRENGTH_DAMAGE_PER_POINT * (strength - 1))


def effective_dps_vs_mob(sword: int, strength: int, phys_def: float) -> float:
    """物理近战对某怪的等效 DPS（扣除怪物免）。"""
    return player_melee_dps(sword, strength) * (1.0 - phys_def)


def elemental_dps(
    sword: int, strength: int, intelligence: int, elemental_on_floor: bool
) -> float:
    """剑附魔（元素）对 L6/L7 的等效 DPS。

    elemental_on_floor=True 表示本层怪对当前附魔元素无免疫（L6 用冰、L7 用火）。
    元素伤 = 0.5 x 物伤 x (1 + 0.05 x (int-1))，且不受 90% 物免影响。
    """
    phys = player_melee_dps(sword, strength)
    if not elemental_on_floor:
        return 0.0
    return ELEMENTAL_FRACTION * phys * (1.0 + INT_DAMAGE_PER_POINT * (intelligence - 1))


def turns_to_kill(hp: float, dps: float) -> float:
    """击杀某怪所需 DO 回合数（近似向上取整；dps<=0 视为打不动）。"""
    if dps <= 0.0:
        return 999.0
    return max(1.0, math.ceil(hp / dps))


def hits_per_kill(turns: float, tactic: str = "stand") -> float:
    """击杀过程中玩家被该怪命中的期望次数。

    stand：每次接战必中 1 次（首击），此后每 5 回合（冷却）多中 1 次；
    kite：命中后拉开，怪需额外 2 回合回身 → 有效攻击周期 7 回合，后续命中降低。
    """
    if tactic == "kite":
        period = KITE_EFFECTIVE_PERIOD
        approach = KITE_APPROACH_HITS
    else:
        period = HIT_COOLDOWN_TURNS
        approach = APPROACH_HITS
    hits = HITS_PER_KILL_BASE + max(0.0, turns - 1.0) / period + approach
    return hits


def damage_per_kill(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """玩家在本层平均击杀 1 怪（近战/远程等权混合）受到的伤害。"""
    melee_dmg, melee_hp, mdef, ranged_dmg, ranged_hp, rdef, requires_elem = MOB_STATS[floor]

    def _mob(hp: float, dmg: float, phys_def: float, uses_elemental: bool) -> float:
        if requires_elem and uses_elemental:
            dps = elemental_dps(gear.sword, gear.strength, gear.intelligence, True)
        else:
            dps = effective_dps_vs_mob(gear.sword, gear.strength, phys_def)
        turns = turns_to_kill(hp, dps)
        hits = hits_per_kill(turns, tactic)
        return hits * dmg

    armor_reduction = ARMOR_REDUCTION_PER_PIECE * gear.armour
    melee_in = _mob(melee_hp, melee_dmg, mdef, uses_elemental=(floor in (6, 7)))
    ranged_in = _mob(ranged_hp, ranged_dmg, rdef, uses_elemental=(floor in (6, 7)))
    # 8 杀中约 5 近战 + 3 远程（3 近战 + 2 远程上限的刷新比例）
    avg = (5.0 * melee_in + 3.0 * ranged_in) / 8.0
    return avg * (1.0 - armor_reduction)


def damage_per_clear(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """本层清满 8 怪的期望累计伤害（含每次接战首击）。"""
    return CLEAR_TARGET * damage_per_kill(floor, gear, tactic)


def mobs_per_batch(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """每次恢复前可击杀的怪数（用到中止血量为止）。"""
    mh = max_health(gear.strength)
    usable = max(1.0, mh - BATCH_ABORT_HEALTH)
    dmg = damage_per_kill(floor, gear, tactic)
    if dmg <= 0.0:
        return CLEAR_TARGET + 1.0
    return max(1.0, usable / dmg)


def batches_for_clear(floor: int, gear: Gear, tactic: str = "stand") -> int:
    return max(1, int(math.ceil(CLEAR_TARGET / mobs_per_batch(floor, gear, tactic))))


def estimated_steps(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """清完本层 8 怪的期望步数（每批往返开销 + 击杀/追击步数）。"""
    melee_dmg, melee_hp, mdef, ranged_dmg, ranged_hp, rdef, requires_elem = MOB_STATS[floor]

    def _mob_turns(hp: float, phys_def: float, uses_elemental: bool) -> float:
        if requires_elem and uses_elemental:
            dps = elemental_dps(gear.sword, gear.strength, gear.intelligence, True)
        else:
            dps = effective_dps_vs_mob(gear.sword, gear.strength, phys_def)
        return turns_to_kill(hp, dps) + STEPS_PER_KILL_APPROACH

    avg_turns = (
        5.0 * _mob_turns(melee_hp, mdef, uses_elemental=(floor in (6, 7)))
        + 3.0 * _mob_turns(ranged_hp, rdef, uses_elemental=(floor in (6, 7)))
    ) / 8.0
    batches = batches_for_clear(floor, gear, tactic)
    return CLEAR_TARGET * avg_turns + batches * BATCH_STEPS_OVERHEAD


def energy_consumed(steps: float, dexterity: int) -> float:
    """清醒 steps 步消耗的能量（每 ~31 步 1 点，随敏捷减缓）。"""
    return steps / (ENERGY_STEPS_PER_POINT / energy_decay_factor(dexterity))


def awake_budget_steps(dexterity: int) -> float:
    """当前敏捷下"从满能量到耗尽"可支撑的清醒步数。

    能量上限 7+2*(dex-1)，每点能量消耗约 31/decay 步。
    dex1≈217、dex2≈318、dex3≈454、dex4≈621、dex5≈930 步。
    """
    return max_energy(dexterity) * ENERGY_STEPS_PER_POINT / energy_decay_factor(dexterity)


def energy_is_bottleneck(
    floor: int, gear: Gear, margin: float = 0.8
) -> bool:
    """判定能量是否是清本层的瓶颈（**单批工作段**所需能量 > 敏捷预算 × margin）。

    关键：执行器的"批量清怪 + 锚点恢复"在每批之间回锚点睡觉、能量重置，
    因此敏捷只在该批工作段（一轮清怪 + 往返）超过能量预算时才有收益。
    L2 单批 ~46 步 vs dex1 预算 217 步 → 不是瓶颈 → 力量优先。
    """
    total = estimated_steps(floor, gear)
    batches = max(1, batches_for_clear(floor, gear))
    per_batch = total / batches
    budget = awake_budget_steps(gear.dexterity)
    return per_batch > budget * margin


def passive_regen(steps: float) -> float:
    """清醒 steps 步的被动回血。"""
    return steps / PASSIVE_REGEN_STEPS_PER_HP


def survival_verdict(floor: int, gear: Gear, tactic: str = "stand") -> str:
    """清层可行性判定：CLEARABLE / MARGINAL / INFEASIBLE。

    - INFEASIBLE：元素层无元素能力；或单怪受击伤害 >= 满血（无法交换）；
      或当前武器对该层怪几乎打不动（dps<=0，turns_to_kill 天文数字）。
    - MARGINAL：清层所需能量超过当前能量预算的 2 倍（需多次回上层恢复）或
      单怪伤害已接近中止血量（每次只能清 1-2 只，耗时过长）。
    - CLEARABLE：其余。
    """
    if floor in (6, 7) and not gear.has_elemental:
        return "INFEASIBLE"
    melee_dmg, melee_hp, mdef, ranged_dmg, ranged_hp, rdef, requires_elem = MOB_STATS[floor]

    def _dps(phys_def: float, uses_elemental: bool) -> float:
        if requires_elem and uses_elemental:
            return elemental_dps(gear.sword, gear.strength, gear.intelligence, True)
        return effective_dps_vs_mob(gear.sword, gear.strength, phys_def)

    max_turns = max(
        turns_to_kill(melee_hp, _dps(mdef, uses_elemental=(floor in (6, 7)))),
        turns_to_kill(ranged_hp, _dps(rdef, uses_elemental=(floor in (6, 7)))),
    )
    if max_turns >= 999.0:
        return "INFEASIBLE"  # 当前武器对该层怪几乎打不动

    mh = max_health(gear.strength)
    armor_reduction = ARMOR_REDUCTION_PER_PIECE * gear.armour
    # 近战单怪单次命中伤害（未计额外命中次数）——判断"能否交换"
    worst_hit = max(melee_dmg, ranged_dmg) * (1.0 - armor_reduction)
    if worst_hit >= mh:
        return "INFEASIBLE"
    dmg_per_kill = damage_per_kill(floor, gear, tactic)
    usable = max(1.0, mh - BATCH_ABORT_HEALTH)
    if dmg_per_kill >= usable and dmg_per_kill >= 0.75 * mh:
        # 每次只能清 1 只且受击过半血：勉强能打但非常危险
        return "MARGINAL"
    steps = estimated_steps(floor, gear, tactic)
    energy = energy_consumed(steps, gear.dexterity)
    if energy > 2.0 * max_energy(gear.dexterity):
        return "MARGINAL"
    return "CLEARABLE"


def recommend_tactic(floor: int, gear: Gear) -> str:
    """选择受击更小的战术：stand（贴脸速杀）vs kite（命中后拉开）。

    风筝只显著降低"慢速击杀"的后续命中；速杀（1-2 回合）时差距很小，
    仅在风筝受击 < stand x 0.9 时切换。
    """
    stand = damage_per_kill(floor, gear, "stand")
    kite = damage_per_kill(floor, gear, "kite")
    if kite < stand * 0.9:
        return "kite"
    return "stand"
