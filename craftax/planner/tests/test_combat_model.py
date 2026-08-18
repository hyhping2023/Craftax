"""combat_model 单测：数值与判定逻辑。"""
from __future__ import annotations

import math

import pytest

from craftax.planner.combat_model import (
    Gear,
    awake_budget_steps,
    batches_for_clear,
    damage_per_clear,
    damage_per_kill,
    effective_dps_vs_mob,
    elemental_dps,
    energy_consumed,
    energy_is_bottleneck,
    estimated_steps,
    hits_per_kill,
    max_energy,
    max_health,
    mobs_per_batch,
    player_melee_dps,
    recommend_tactic,
    survival_verdict,
    turns_to_kill,
)


def test_sword_damage_table():
    assert player_melee_dps(0, 1) == pytest.approx(1.0)
    assert player_melee_dps(1, 1) == pytest.approx(2.0)
    assert player_melee_dps(2, 1) == pytest.approx(3.0)
    assert player_melee_dps(3, 1) == pytest.approx(5.0)
    assert player_melee_dps(4, 1) == pytest.approx(8.0)


def test_strength_scaling():
    # 力量每点 +25%：str=2 铁剑 = 5*1.25=6.25；str=5 铁剑 = 5*2=10
    assert player_melee_dps(3, 2) == pytest.approx(6.25)
    assert player_melee_dps(3, 5) == pytest.approx(10.0)


def test_effective_dps_vs_defense():
    # L4 knight 50% 物免
    assert effective_dps_vs_mob(3, 1, 0.5) == pytest.approx(2.5)
    # L5 troll 20% 物免：钻石剑(8) x 力量5(双倍) = 16，x0.8 = 12.8
    assert effective_dps_vs_mob(4, 5, 0.2) == pytest.approx(12.8)


def test_turns_to_kill():
    assert turns_to_kill(5, 5) == 1.0
    assert turns_to_kill(9, 6.25) == 2.0  # ceil(9/6.25)=ceil(1.44)=2
    assert turns_to_kill(999, 0.0) == 999.0  # 打不动


def test_hits_per_kill():
    # 1 回合击杀（首击）：约 1.2 次命中
    h = hits_per_kill(1.0, "stand")
    assert h == pytest.approx(1.0 + 0.2)
    # 长时间击杀命中更多
    assert hits_per_kill(10.0, "stand") > hits_per_kill(2.0, "stand")
    # kite 降低受击
    assert hits_per_kill(3.0, "kite") < hits_per_kill(3.0, "stand")


def test_damage_per_clear_no_armor_l1():
    # L1 orc 3 伤；铁剑 str1：击杀回合 2，命中 ~1.4，单怪 ~4.2，8 怪 ~33
    gear = Gear(sword=3, armour=0, strength=1)
    assert damage_per_clear(1, gear, "stand") == pytest.approx(
        damage_per_kill(1, gear, "stand") * 8
    )


def test_armour_reduces_damage():
    gear0 = Gear(sword=3, armour=0, strength=1)
    gear2 = Gear(sword=3, armour=2, strength=1)
    # 2 件甲 = 20% 减免
    assert damage_per_clear(1, gear2, "stand") == pytest.approx(
        damage_per_clear(1, gear0, "stand") * 0.8
    )


def test_batches_and_steps():
    gear = Gear(sword=3, armour=0, strength=3)
    mpb = mobs_per_batch(1, gear)
    assert mpb >= 1.0
    assert batches_for_clear(1, gear) >= 1
    assert estimated_steps(1, gear) > 0


def test_survival_verdict_elemental_gate():
    # L6/L7 无元素能力 → INFEASIBLE（元素门）
    assert survival_verdict(6, Gear(sword=3, armour=4, strength=5), "stand") == "INFEASIBLE"
    assert survival_verdict(7, Gear(sword=3, armour=4, strength=5), "stand") == "INFEASIBLE"
    # 有元素能力 + 足够防御 → 不再 INFEASIBLE
    gear_e = Gear(sword=3, armour=4, strength=5, has_elemental=True)
    assert survival_verdict(6, gear_e, "stand") != "INFEASIBLE"


def test_survival_verdict_weak_gear_l2():
    # L2 gnome 4 伤 vs 满血 str1=8：单怪受击 ~5.6 < 8，可清但伤（应为 CLEARABLE/MARGINAL）
    g = Gear(sword=2, armour=0, strength=1)
    assert survival_verdict(2, g, "stand") in ("CLEARABLE", "MARGINAL")
    # 无剑（dps≈0 打不动）→ 单怪受击按 999 命中估算，不应 CLEARABLE
    g0 = Gear(sword=0, armour=0, strength=1)
    assert survival_verdict(2, g0, "stand") != "CLEARABLE"


def test_recommend_tactic():
    # 剑强+层弱 → stand；剑弱/层强 → kite 不劣于 stand
    g_strong = Gear(sword=4, armour=4, strength=5, has_elemental=True)
    assert recommend_tactic(1, g_strong) == "stand"
    g_weak = Gear(sword=2, armour=0, strength=1)
    assert recommend_tactic(2, g_weak) in ("stand", "kite")


def test_max_health_energy():
    assert max_health(1) == pytest.approx(8.0)
    assert max_health(5) == pytest.approx(12.0)
    assert max_energy(1) == pytest.approx(7.0)
    assert max_energy(5) == pytest.approx(15.0)


def test_energy_consumed():
    # 敏捷越高，单位步数消耗越低
    assert energy_consumed(310.0, 1) > energy_consumed(310.0, 5)


def test_awake_budget_steps():
    # 满能量到耗尽可支撑的清醒步数：随敏捷大幅增长
    b1 = awake_budget_steps(1)
    b5 = awake_budget_steps(5)
    assert b1 == pytest.approx(217.0, rel=0.01)
    assert b5 > b1 * 3
    # 能量上限线性 +2/点
    assert max_energy(3) == max_energy(2) + 2


def test_energy_is_bottleneck_per_batch():
    """批量+锚点恢复下能量按"单批工作段"预算；L1-L5 单批 ~46-64 步 << dex1
    预算 217 步 → 不是瓶颈（力量优先）。这是量化结论：锚点恢复让敏捷几乎不
    值得花 XP；敏捷只在力量满后（深层长程）或不可恢复的单批超长段才有价值。
    """
    g = Gear(sword=3, armour=1, strength=3, dexterity=1)
    total = estimated_steps(2, g)
    per_batch = total / max(1, batches_for_clear(2, g))
    assert per_batch < awake_budget_steps(1) * 0.8
    assert energy_is_bottleneck(2, g) is False
    # 弱装备单批也不超过预算（锚点恢复兜底）
    g_weak = Gear(sword=0, armour=0, strength=1, dexterity=1)
    for f in (1, 2, 5):
        total = estimated_steps(f, g_weak)
        per_batch = total / max(1, batches_for_clear(f, g_weak))
        assert per_batch < awake_budget_steps(1) * 0.9, f"L{f} per_batch={per_batch:.0f}"
        assert energy_is_bottleneck(f, g_weak) is False


# ---------------------------------------------------------------------------
# 与 JAX 常量交叉校验（MOB_STATS 表要与 game 常量一致）
# ---------------------------------------------------------------------------


def test_mob_stats_match_jax_constants():
    jax = pytest.importorskip("jax")
    import numpy as np

    from craftax.craftax.constants import (
        FLOOR_MOB_MAPPING,
        MOB_TYPE_DAMAGE_MAPPING,
        MOB_TYPE_DEFENSE_MAPPING,
        MOB_TYPE_HEALTH_MAPPING,
    )
    from craftax.planner.combat_model import MOB_STATS

    for floor in range(8):
        melee_type = int(np.asarray(FLOOR_MOB_MAPPING[floor, 1]))
        ranged_type = int(np.asarray(FLOOR_MOB_MAPPING[floor, 2]))
        melee_dmg = float(
            np.asarray(MOB_TYPE_DAMAGE_MAPPING[melee_type, 1]).sum()
        )
        ranged_dmg = float(
            np.asarray(MOB_TYPE_DAMAGE_MAPPING[ranged_type, 3]).sum()
        )  # 远程怪伤害在 projectile 槽（ranged 槽恒 NO_DAMAGE）
        melee_hp = float(np.asarray(MOB_TYPE_HEALTH_MAPPING[floor, 1]))
        ranged_hp = float(np.asarray(MOB_TYPE_HEALTH_MAPPING[floor, 2]))
        melee_def = float(np.asarray(MOB_TYPE_DEFENSE_MAPPING[floor, 1])[0])
        ranged_def = float(np.asarray(MOB_TYPE_DEFENSE_MAPPING[floor, 2])[0])  # 仅物理防御

        stats = MOB_STATS[floor]
        assert stats[0] == pytest.approx(melee_dmg), f"L{floor} melee dmg"
        assert stats[1] == pytest.approx(melee_hp), f"L{floor} melee hp"
        assert stats[2] == pytest.approx(melee_def), f"L{floor} melee def"
        assert stats[3] == pytest.approx(ranged_dmg), f"L{floor} ranged dmg"
        assert stats[4] == pytest.approx(ranged_hp), f"L{floor} ranged hp"
        assert stats[5] == pytest.approx(ranged_def), f"L{floor} ranged def"


def test_floor_names_match_executor_mapping():
    """MOB_STATS 的楼层编号须与 executor.DEFEAT_MOB_LOCATIONS 一致。"""
    from craftax.planner.executor import DEFEAT_MOB_LOCATIONS

    assert DEFEAT_MOB_LOCATIONS["native.defeat_gnome_warrior"][2] == [2]
    assert DEFEAT_MOB_LOCATIONS["native.defeat_orc_soldier"][2] == [1]
    assert DEFEAT_MOB_LOCATIONS["native.defeat_knight"][2] == [4]
    assert DEFEAT_MOB_LOCATIONS["native.defeat_troll"][2] == [5]
