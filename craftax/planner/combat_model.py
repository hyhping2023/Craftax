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
from typing import Dict, Optional, Tuple

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

# 弓（L1 首箱必出，定位为"0-1 受击清怪"的核心武器）
BOW_ARROW_DAMAGE = 5.0             # ARROW2 基础物伤（MOB_TYPE_DAMAGE_MAPPING[4, PROJECTILE]）
BOW_DAMAGE_PER_DEX = 0.2           # 箭伤随敏捷 +20%/点
BOW_ELEMENTAL_FRACTION = 0.5       # 附魔弓元素半伤（不受物免；对正确元素防御为 0）

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

# ---------------------------------------------------------------------------
# 内在状态衰减/恢复速率（逐条对应 game_logic.update_player_intrinsics，
# 用于"睡前投影"等有限步前瞻；数值由 test_combat_model 与源码交叉校验）
#   饥饿累加 >25 掉 1 food；口渴累加 >20 掉 1 drink；疲劳 >30 掉 1 energy；
#   睡眠时饥饿/口渴累加减半、疲劳 -1/步（energy 每 11 步 +1）；
#   三项必需品（food>0, drink>0, energy>0 或睡眠）齐备时 recover +1（睡眠 +2），
#   >25 回 1 血；任一为 0 则 recover -1（睡眠 -0.5），< -15 掉 1 血。
# ---------------------------------------------------------------------------
HUNGER_STEPS_PER_POINT = 26.0        # 醒着每掉 1 food 的步数
THIRST_STEPS_PER_POINT = 21.0        # 醒着每掉 1 drink 的步数
SLEEP_INTRINSIC_DECAY = 0.5          # 睡眠时饥饿/口渴累加系数
SLEEP_ENERGY_STEPS_PER_POINT = 11.0  # 睡眠每回 1 能量的步数
SLEEP_REGEN_STEPS_PER_HP = 13.0      # 睡眠每回 1 血的步数（必需品齐备）
STARVE_STEPS_PER_HP_AWAKE = 16.0     # 缺必需品时醒着每掉 1 血的步数
STARVE_STEPS_PER_HP_ASLEEP = 31.0    # 缺必需品时睡眠每掉 1 血的步数
SLEEP_DAMAGE_MULTIPLIER = 3.5        # 睡眠中受击倍率
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
    bow: int = 0                    # 弓等级（0=无弓；1=已有弓）
    bow_enchant: int = 0            # 弓附魔（0=无；1=火；2=冰；L6 需冰、L7 需火）


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


# ---------------------------------------------------------------------------
# 弓战斗模型（L1 首箱必出弓；箭伤 5 + 敏捷缩放，点射 0-1 受击清怪）
# ---------------------------------------------------------------------------


def bow_arrow_damage(
    dexterity: int,
    phys_def: float,
    bow_enchant: int = 0,
    elemental_required: bool = False,
    has_elemental: bool = False,
) -> float:
    """单支箭对某怪的等效伤害。

    - 物伤 = 5 × (1+0.2×(dex-1)) × (1-物免)；
    - 附魔弓（fire/ice）额外元素半伤，不受物免；
      L6/L7（elemental_required=True）需对应元素才生效（has_elemental 语义：
       L6 需冰、L7 需火），其余层任意附魔均有效。
    """
    base = BOW_ARROW_DAMAGE * (1.0 + BOW_DAMAGE_PER_DEX * (dexterity - 1))
    phys = base * (1.0 - phys_def)
    if bow_enchant > 0 and (not elemental_required or has_elemental):
        phys += base * BOW_ELEMENTAL_FRACTION
    return phys


def turns_to_kill_bow(
    hp: float,
    gear: Gear,
    phys_def: float,
    elemental_required: bool = False,
) -> float:
    """用弓击杀某怪所需箭数（打不动返回 999）。"""
    dmg = bow_arrow_damage(
        gear.dexterity, phys_def, gear.bow_enchant, elemental_required, gear.has_elemental
    )
    if dmg <= 0.0:
        return 999.0
    # 小 epsilon 避免浮点边界（如 0.4999... 导致 20/0.5 算出 41）
    return max(1.0, math.ceil(hp / dmg - 1e-9))


def damage_per_kill_bow(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """弓清层时平均击杀 1 怪受到的伤害（近战/远程等权混合）。"""
    melee_dmg, melee_hp, mdef, ranged_dmg, ranged_hp, rdef, requires_elem = MOB_STATS[floor]
    elem_floor = floor in (6, 7)

    def _mob(hp: float, dmg: float, phys_def: float) -> float:
        turns = turns_to_kill_bow(hp, gear, phys_def, elemental_required=elem_floor)
        hits = hits_per_kill(turns, tactic)
        return hits * dmg

    armor_reduction = ARMOR_REDUCTION_PER_PIECE * gear.armour
    melee_in = _mob(melee_hp, melee_dmg, mdef)
    ranged_in = _mob(ranged_hp, ranged_dmg, rdef)
    avg = (5.0 * melee_in + 3.0 * ranged_in) / 8.0
    return avg * (1.0 - armor_reduction)


def damage_per_clear_bow(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """弓清满 8 怪的期望累计伤害。"""
    return CLEAR_TARGET * damage_per_kill_bow(floor, gear, tactic)


def mobs_per_batch_bow(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """弓清层时每次恢复前可击杀的怪数。"""
    mh = max_health(gear.strength)
    usable = max(1.0, mh - BATCH_ABORT_HEALTH)
    dmg = damage_per_kill_bow(floor, gear, tactic)
    if dmg <= 0.0:
        return CLEAR_TARGET + 1.0
    return max(1.0, usable / dmg)


def batches_for_clear_bow(floor: int, gear: Gear, tactic: str = "stand") -> int:
    return max(1, int(math.ceil(CLEAR_TARGET / mobs_per_batch_bow(floor, gear, tactic))))


def estimated_steps_bow(floor: int, gear: Gear, tactic: str = "stand") -> float:
    """弓清完本层 8 怪的期望步数（含每批往返开销）。"""
    melee_dmg, melee_hp, mdef, ranged_dmg, ranged_hp, rdef, requires_elem = MOB_STATS[floor]
    elem_floor = floor in (6, 7)

    def _mob_turns(hp: float, phys_def: float) -> float:
        return turns_to_kill_bow(hp, gear, phys_def, elemental_required=elem_floor) \
            + STEPS_PER_KILL_APPROACH

    avg_turns = (
        5.0 * _mob_turns(melee_hp, mdef)
        + 3.0 * _mob_turns(ranged_hp, rdef)
    ) / 8.0
    batches = batches_for_clear_bow(floor, gear, tactic)
    return CLEAR_TARGET * avg_turns + batches * BATCH_STEPS_OVERHEAD


# 箭预算余量：射空/怪走位/被动怪误伤的浪费系数（标定项）
ARROW_WASTE_MARGIN = 1.25
# 不可补弹层的预留系数：弹尽即失去武器，且该层无法再合成 → 按尾部而不是均值备。
# （1.25 是"平均浪费"；在 L1 这类无木层，17 支刚够理论均值，一次走位失手就断供。）
ARROW_NO_RESTOCK_MARGIN = 1.75


def arrows_per_kill(floor: int, gear: Gear) -> float:
    """本层平均击杀 1 怪所需箭数（按 5 近战 + 3 远程的刷新比例）；打不动返回 0。"""
    _, melee_hp, mdef, _, ranged_hp, rdef, _ = MOB_STATS[floor]
    elem = floor in (6, 7)
    turns_melee = turns_to_kill_bow(melee_hp, gear, mdef, elemental_required=elem)
    turns_ranged = turns_to_kill_bow(ranged_hp, gear, rdef, elemental_required=elem)
    if max(turns_melee, turns_ranged) >= 999.0:
        return 0.0
    return (5.0 * turns_melee + 3.0 * turns_ranged) / 8.0


def arrows_for_clear(floor: int, gear: Gear, restockable: bool = True) -> int:
    """用弓清满本层 8 怪需要的箭数（含浪费余量）；弓打不动该层返回 0。

    MAKE_ARROW 一次消耗 1 木 + 1 石产出 2 箭（game_logic），所以这个数字
    直接决定"下楼前要在有木石的层备多少料"。L1 兽人（7/5 HP）在敏捷 1 下
    约 17 支——而首箱只给 0 支、执行器过去只备 8 支，正是深层清怪半途弹尽的原因。

    restockable=False（目标层无木、箭不可再生）时用更大的预留系数：17 支只是
    均值，弹尽后剩下的怪只能用剑打，而"剑够不够"由 recommend_clear_prep 决定。
    """
    per_kill = arrows_per_kill(floor, gear)
    if per_kill <= 0.0:
        return 0
    margin = ARROW_WASTE_MARGIN if restockable else ARROW_NO_RESTOCK_MARGIN
    return int(math.ceil(CLEAR_TARGET * per_kill * margin))


# ---------------------------------------------------------------------------
# 弹药经济学：备箭 vs 造石/铁剑的择优（§6.2.6a 出路 a）
#
# 关键事实：MAKE_ARROW 一次 1 木 + 1 石 → 2 箭，而**石剑同样只要 1 木 + 1 石**
# （铁剑再加 1 铁 + 1 煤 + 熔炉）。地牢层（L1-L5）没有树 → 下楼后箭是不可再生
# 资源：弹尽时剩下的怪只能用剑打。因此"有弓就跳过深制备"在不可补弹的层上不成立。
#
# 两个选项换算到同一货币——**每花掉 1 木 + 1 石 能省下的清层受击伤害**：
#   备箭：2 支箭把 2/arrows_per_kill 只怪从近战转为远程 → 省
#         (damage_per_kill - damage_per_kill_bow) × 该怪数；
#   升剑：把**弹尽后的残余击杀**的近战单价降一档 → 省
#         残余比例 × (damage_per_clear(旧剑) - damage_per_clear(新剑))。
# 残余为 0（箭够/可就地补）时升剑收益恒为 0 → 自动退回"有弓就跳过深制备"，
# 即旧行为成为新规则在"弹药可补"这一特例下的推论，而不再是无条件假设。
# ---------------------------------------------------------------------------

ARROWS_PER_CRAFT = 2               # MAKE_ARROW: 1 木 + 1 石 → 2 箭
# 升剑成本（以"1 木 + 1 石"为 1 单位）：石剑就是 1 单位；铁剑多 1 铁 + 1 煤，
# 且铁需石镐 + 熔炉 + 采矿往返 → 记 3 单位额外暴露；钻石剑要下 L2+ 挖 2 钻。
SWORD_UPGRADE_UNITS: Dict[int, float] = {2: 1.0, 3: 4.0, 4: 7.0}
# 升剑要赢过备箭这么多才值得：备箭是可分割、随时可停的投入（每 2 支立刻生效），
# 而升剑是一次性承诺（半程放弃的采铁毫无价值）→ 收益接近时选风险小的那条。
SWORD_PREFER_MARGIN = 1.25
# 升剑的绝对门槛：每单位材料至少省下这么多伤害才值得为它多跑一趟采集。
# 没有这条门槛时，"相对更优"会一路推到钻石剑（每单位省 0.3 伤），把制备变成
# 无限升级（实测：整局在地表升级链上打转，从未下楼）。
MIN_SWORD_GAIN_ABS = 1.0


def bow_kill_coverage(floor: int, gear: Gear, arrows: int) -> float:
    """给定箭数，弓能覆盖的击杀比例（0..1，按平均浪费系数折算）。"""
    per_kill = arrows_per_kill(floor, gear)
    if per_kill <= 0.0 or arrows <= 0:
        return 0.0
    kills = arrows / (per_kill * ARROW_WASTE_MARGIN)
    return min(1.0, kills / CLEAR_TARGET)


def mixed_damage_per_clear(
    floor: int, gear: Gear, arrows: int, tactic: str = "stand"
) -> float:
    """清满 8 怪的期望受击：前 coverage 比例用弓，弹尽后的残余用剑。

    这是"备箭还是升剑"的目标函数——两种投入都只是在压低这个数。
    """
    coverage = bow_kill_coverage(floor, gear, arrows)
    bow = damage_per_clear_bow(floor, gear, tactic) if coverage > 0.0 else 0.0
    melee = damage_per_clear(floor, gear, tactic)
    return coverage * bow + (1.0 - coverage) * melee


@dataclass(frozen=True)
class ClearPrep:
    """一层"清 8 怪"的制备建议（纯数值择优，执行器据此排动作顺序）。"""

    floor: int
    arrows_have: int
    arrows_min: int         # 值得专程采料备到的箭数（均值余量 1.25x）
    arrows_target: int      # 不可补层的预留目标（1.75x）；只用手头余料补到这里
    coverage: float         # 当前箭数能覆盖的击杀比例
    damage_now: float       # 按"当前箭 + 当前剑"清满 8 怪的期望受击
    damage_if_stocked: float  # 备满箭后的期望受击（弓上限）
    bow_advantage: float    # 清满 8 怪时"用剑" - "用弓"的伤害差（<=0 → 备箭无意义）
    sword_target: int       # 建议升级到的剑等级（0=此刻升剑不如备箭/无需升）
    gain_per_unit_arrows: float   # 每 1 木+1 石 换成箭能省的伤害
    gain_per_unit_sword: float    # 同样成本花在升剑上能省的伤害
    prefer: str             # "arrows" | "sword" | "ready"
    reason: str


def recommend_clear_prep(
    floor: int,
    gear: Gear,
    arrows: int,
    restockable: bool = False,
    iron_available: bool = False,
    diamond_available: bool = False,
    tactic: str = "stand",
    arrow_cap: Optional[int] = None,
    max_sword: int = 4,
) -> ClearPrep:
    """备箭 vs 升剑的择优。

    restockable：目标层能否就地合成箭（同时有木与石）。可补时残余为 0，
      备箭恒占优（= 旧的"有弓就跳过深制备"）。
    iron_available / diamond_available：本层（制备地点）能否拿到铁+煤 / 钻石，
      决定铁剑/钻石剑是否是可选项——不可得的升级不参与比较。
    arrow_cap：执行器的携带上限（备箭目标据此截断）。
    max_sword：调用方**当场做得出来**的最高剑等级。给出做不出来的目标会让
      执行器在制备链上无限打转（实测：钻石剑目标使整局停在地表升级）。

    箭数给两档：arrows_min（均值 1.25x，值得专程采料）与 arrows_target
    （不可补层的 1.75x 预留，只用手头余料补）。两档是必要的——地表持续刷怪、
    弓持续消耗，若把预留当硬目标，产量≈消耗会让补给永远不满足（实测：整个
    episode 变成地表箭工厂，一次没下楼）。
    """
    needed = arrows_for_clear(floor, gear, restockable=restockable)
    arrows_min = arrows_for_clear(floor, gear, restockable=True)
    arrows_target = needed
    if arrow_cap is not None:
        cap = int(arrow_cap)
        arrows_target = min(arrows_target, cap) if arrows_target > 0 else 0
        arrows_min = min(arrows_min, cap) if arrows_min > 0 else 0
    coverage = 1.0 if (restockable and needed > 0) else bow_kill_coverage(
        floor, gear, arrows
    )
    melee_clear = damage_per_clear(floor, gear, tactic)
    bow_clear = damage_per_clear_bow(floor, gear, tactic) if needed > 0 else melee_clear
    damage_now = coverage * bow_clear + (1.0 - coverage) * melee_clear
    residual = max(0.0, 1.0 - coverage)
    # 弓相对当前剑的优势：<=0 表示剑已经打得和弓一样省血（L1 + 铁剑就是这样，
    # 兽人 2 击杀 = 2 箭），此时再为"清层"备箭买不到任何减伤，只留生存储备。
    bow_advantage = melee_clear - bow_clear

    # 备箭的边际收益：2 支箭能把多少比例的击杀从近战转为远程
    per_kill = arrows_per_kill(floor, gear)
    if per_kill > 0.0 and residual > 0.0:
        d_cov = min(
            residual,
            (ARROWS_PER_CRAFT / (per_kill * ARROW_WASTE_MARGIN)) / CLEAR_TARGET,
        )
        gain_arrows = d_cov * max(0.0, bow_advantage)
    else:
        gain_arrows = 0.0

    # 升剑的边际收益：残余击杀的近战单价降一档（除以材料/采矿成本单位）
    best_sword = 0
    best_gain = 0.0
    for level in range(gear.sword + 1, min(max_sword, len(SWORD_DAMAGE) - 1) + 1):
        cost = SWORD_UPGRADE_UNITS.get(level)
        if cost is None:
            continue
        if level == 3 and not iron_available:
            continue
        if level == 4 and not diamond_available:
            continue
        upgraded = damage_per_clear(
            floor,
            Gear(
                sword=level,
                armour=gear.armour,
                strength=gear.strength,
                dexterity=gear.dexterity,
                intelligence=gear.intelligence,
                has_elemental=gear.has_elemental,
                bow=gear.bow,
                bow_enchant=gear.bow_enchant,
            ),
            tactic,
        )
        gain = residual * max(0.0, melee_clear - upgraded) / cost
        if gain > best_gain:
            best_gain, best_sword = gain, level

    want_arrows = bow_advantage > 0.0 and not restockable and arrows < arrows_target
    if (best_gain >= MIN_SWORD_GAIN_ABS
            and best_gain > gain_arrows * SWORD_PREFER_MARGIN):
        prefer = "sword"
        reason = (
            f"L{floor}: 箭 {arrows}/{needed} 只覆盖 {coverage:.0%}，残余 "
            f"{residual:.0%} 要用剑打；升剑 {gear.sword}→{best_sword} 每单位省 "
            f"{best_gain:.1f} 伤 > 备箭 {gain_arrows:.1f}"
        )
    elif want_arrows:
        prefer = "arrows"
        best_sword = 0
        if residual <= 1e-9:
            reason = (
                f"L{floor}: 箭 {arrows} 已够均值，不可补层的预留 {arrows_target} 未满"
                " → 用手头余料继续备箭（弹尽即失去武器）"
            )
        else:
            reason = (
                f"L{floor}: 箭 {arrows}/{arrows_min}(预留 {arrows_target})，"
                f"备箭每单位省 {gain_arrows:.1f} 伤 >= 升剑 {best_gain:.1f} → 先备箭"
            )
    else:
        prefer = "ready"
        best_sword = 0
        if restockable:
            reason = f"L{floor}: 本层可就地补箭 → 无需深制备"
        elif bow_advantage <= 0.0:
            reason = (
                f"L{floor}: 当前剑已与弓等效（清层受击 {melee_clear:.0f}）→ "
                "不再为清层备箭/升剑，只留生存储备"
            )
        else:
            reason = f"L{floor}: 箭 {arrows}/{arrows_target} 已备齐 → 无需深制备"
    return ClearPrep(
        floor=floor,
        arrows_have=int(arrows),
        arrows_min=int(arrows_min),
        arrows_target=int(arrows_target),
        coverage=coverage,
        damage_now=damage_now,
        damage_if_stocked=bow_clear if needed > 0 else melee_clear,
        bow_advantage=bow_advantage,
        sword_target=best_sword,
        gain_per_unit_arrows=gain_arrows,
        gain_per_unit_sword=best_gain,
        prefer=prefer,
        reason=reason,
    )


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


@dataclass(frozen=True)
class SleepProjection:
    """一次 SLEEP 的有限步前瞻结果（纯算术，无需推进模拟器）。

    SLEEP 是不可撤销承诺：动作被锁为 NOOP 直到能量回满或被怪打醒，
    因此必须在按下之前把整段时长的口渴/饥饿/掉血走完一遍再决定。
    """

    steps: float          # 预计睡眠时长（回满能量所需步数）
    health_end: float     # 睡醒时血量（不含被怪打醒的伤害）
    drink_end: float      # 睡醒时水
    food_end: float       # 睡醒时食物
    dies: bool            # 期间是否会掉到 0 血


def project_sleep(
    energy: float,
    health: float,
    drink: float,
    food: float,
    strength: int = 1,
    dexterity: int = 1,
    thirst_rate: float = 1.0,
) -> SleepProjection:
    """投影"现在睡下去"的结果（对应 update_player_intrinsics 的逐步累加）。

    要点：睡眠只解决能量，不解决口渴/饥饿；任一必需品归零时睡眠反而**掉血**
    （31 步/HP），且睡眠期间无法自救。执行器用它否掉"渴着睡"这类必死决策。

    thirst_rate 对应 `EnvParams.thirst_rate`（1.0 = 原版）。两边必须用同一个值，
    否则会话把水调慢后投影仍按原版算，执行器会拒绝本来安全的睡眠。
    """
    decay = energy_decay_factor(dexterity)
    mh = max_health(strength)
    steps = max(0.0, (max_energy(dexterity) - energy)) * SLEEP_ENERGY_STEPS_PER_POINT
    # 睡眠期间口渴/饥饿按半速累加
    thirst_steps = THIRST_STEPS_PER_POINT / (SLEEP_INTRINSIC_DECAY * decay * thirst_rate)
    hunger_steps = HUNGER_STEPS_PER_POINT / (SLEEP_INTRINSIC_DECAY * decay)
    drink_end = max(0.0, drink - steps / thirst_steps)
    food_end = max(0.0, food - steps / hunger_steps)

    # 分段：必需品齐备的前缀按 13 步/HP 回血；此后按 31 步/HP 掉血
    if drink > 0 and food > 0:
        steps_to_empty = min(
            drink * thirst_steps if drink > 0 else 0.0,
            food * hunger_steps if food > 0 else 0.0,
        )
    else:
        steps_to_empty = 0.0
    ok_steps = min(steps, steps_to_empty)
    bad_steps = max(0.0, steps - ok_steps)
    health_end = min(mh, health + ok_steps / SLEEP_REGEN_STEPS_PER_HP)
    health_end -= bad_steps / STARVE_STEPS_PER_HP_ASLEEP
    return SleepProjection(
        steps=steps,
        health_end=health_end,
        drink_end=drink_end,
        food_end=food_end,
        dies=health_end <= 0.0,
    )


def projected_awake_health(
    steps: float, health: float, drink: float, food: float, energy: float,
    strength: int = 1, dexterity: int = 1, thirst_rate: float = 1.0,
) -> float:
    """投影"清醒原地待机 steps 步"后的血量。

    用于判定"原地等被动回血"是否真的在回血——缺水/缺食物时被动回血是负的
    （16 步/HP 掉血 vs 26 步/HP 回血），原地等待即死亡螺旋。
    thirst_rate 语义同 project_sleep（对应 EnvParams.thirst_rate）。
    """
    decay = energy_decay_factor(dexterity)
    mh = max_health(strength)
    thirst_steps = THIRST_STEPS_PER_POINT / (decay * thirst_rate)
    hunger_steps = HUNGER_STEPS_PER_POINT / decay
    if drink > 0 and food > 0 and energy > 0:
        steps_to_empty = min(drink * thirst_steps, food * hunger_steps,
                             energy * ENERGY_STEPS_PER_POINT / decay)
    else:
        steps_to_empty = 0.0
    ok_steps = min(steps, steps_to_empty)
    bad_steps = max(0.0, steps - ok_steps)
    out = min(mh, health + ok_steps / PASSIVE_REGEN_STEPS_PER_HP)
    return out - bad_steps / STARVE_STEPS_PER_HP_AWAKE


def survival_verdict(floor: int, gear: Gear, tactic: str = "stand") -> str:
    """清层可行性判定：CLEARABLE / MARGINAL / INFEASIBLE。

    - INFEASIBLE：元素层无元素能力；或单怪受击伤害 >= 满血（无法交换）；
      或当前武器对该层怪几乎打不动（dps<=0，turns_to_kill 天文数字）。
    - MARGINAL：清层所需能量超过当前能量预算的 2 倍（需多次回上层恢复）或
      单怪伤害已接近中止血量（每次只能清 1-2 只，耗时过长）。
    - CLEARABLE：其余。

    有弓（gear.bow>=1）时改用弓模型：L1-L3 箭 1-3 发即可击杀，
    L4 骑士/ L5 巨魔靠物免衰减箭伤（需附魔/近战配合），L6/L7 仍需元素能力。
    """
    if gear.bow >= 1:
        return _survival_verdict_bow(floor, gear, tactic)
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


def _survival_verdict_bow(floor: int, gear: Gear, tactic: str = "stand") -> str:
    """弓模型下的清层可行性判定（结构同近战判定）。"""
    if floor in (6, 7) and not gear.has_elemental:
        return "INFEASIBLE"
    melee_dmg, melee_hp, mdef, ranged_dmg, ranged_hp, rdef, requires_elem = MOB_STATS[floor]
    elem_floor = floor in (6, 7)

    max_turns = max(
        turns_to_kill_bow(melee_hp, gear, mdef, elemental_required=elem_floor),
        turns_to_kill_bow(ranged_hp, gear, rdef, elemental_required=elem_floor),
    )
    if max_turns >= 999.0:
        return "INFEASIBLE"

    mh = max_health(gear.strength)
    armor_reduction = ARMOR_REDUCTION_PER_PIECE * gear.armour
    worst_hit = max(melee_dmg, ranged_dmg) * (1.0 - armor_reduction)
    if worst_hit >= mh:
        return "INFEASIBLE"
    dmg_per_kill = damage_per_kill_bow(floor, gear, tactic)
    usable = max(1.0, mh - BATCH_ABORT_HEALTH)
    if dmg_per_kill >= usable and dmg_per_kill >= 0.75 * mh:
        return "MARGINAL"
    steps = estimated_steps_bow(floor, gear, tactic)
    energy = energy_consumed(steps, gear.dexterity)
    if energy > 2.0 * max_energy(gear.dexterity):
        return "MARGINAL"
    return "CLEARABLE"


def recommend_tactic(floor: int, gear: Gear) -> str:
    """选择受击更小的战术：bow（远程点射）/ kite（风筝）/ stand（贴脸速杀）。

    有弓且弓清层受击不显著劣于近战时优先 bow——弓把 L1-L3 清怪从 2-3 受击
    降到 0-1 受击，是打破 L0 制备生存墙的核心武器。
    """
    if gear.bow >= 1:
        bow_dmg = damage_per_clear_bow(floor, gear, "stand")
        melee_dmg = damage_per_clear(floor, gear, "stand")
        if bow_dmg <= melee_dmg * 1.1:
            return "bow"
    stand = damage_per_kill(floor, gear, "stand")
    kite = damage_per_kill(floor, gear, "kite")
    if kite < stand * 0.9:
        return "kite"
    return "stand"
