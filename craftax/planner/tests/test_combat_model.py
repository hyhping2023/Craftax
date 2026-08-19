"""combat_model 单测：数值与判定逻辑。"""
from __future__ import annotations

import math

import pytest

from craftax.planner.combat_model import (
    Gear,
    awake_budget_steps,
    batches_for_clear,
    bow_arrow_damage,
    damage_per_clear,
    damage_per_clear_bow,
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
    turns_to_kill_bow,
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


# ---------------------------------------------------------------------------
# 弓模型（L1 首箱必出；箭伤 5 + 敏捷缩放，0-1 受击清怪）
# ---------------------------------------------------------------------------


def test_bow_arrow_damage():
    # 箭伤 5 物理（dex1 无物免），随敏捷 +20%/点
    assert bow_arrow_damage(1, 0.0) == pytest.approx(5.0)
    assert bow_arrow_damage(2, 0.0) == pytest.approx(6.0)
    assert bow_arrow_damage(5, 0.0) == pytest.approx(9.0)
    # 物免衰减（L4 knight 50%）
    assert bow_arrow_damage(1, 0.5) == pytest.approx(2.5)
    # 附魔弓元素半伤不受物免：L4 任意附魔有效（5*0.5 + 2.5 = 5）
    assert bow_arrow_damage(1, 0.5, bow_enchant=1) == pytest.approx(5.0)
    # L6 元素层：正确元素（冰）生效（5*0.1 + 2.5 = 3）；错误/无元素被 90% 物免压制
    assert bow_arrow_damage(1, 0.9, bow_enchant=2, elemental_required=True,
                            has_elemental=True) == pytest.approx(3.0)
    assert bow_arrow_damage(1, 0.9, bow_enchant=1, elemental_required=True,
                            has_elemental=False) == pytest.approx(0.5)


def test_bow_turns_to_kill():
    g = Gear(sword=1, strength=1, bow=1)  # str1/dex1，无附魔
    # L0 僵尸 5HP → 1 箭；L1 兽人 7HP → 2 箭；L2 侏儒 9HP → 2 箭；L3 蜥蜴 11HP → 3 箭
    assert turns_to_kill_bow(5, g, 0.0) == 1.0
    assert turns_to_kill_bow(7, g, 0.0) == 2.0
    assert turns_to_kill_bow(9, g, 0.0) == 2.0
    assert turns_to_kill_bow(11, g, 0.0) == 3.0
    # L4 骑士 50% 物免：无附魔 5 箭；附魔弓（任意元素）3 箭
    assert turns_to_kill_bow(12, g, 0.5) == 5.0
    g_ench = Gear(sword=1, strength=1, bow=1, bow_enchant=1)
    assert turns_to_kill_bow(12, g_ench, 0.5) == 3.0
    # L6 猪人 90% 物免：无元素箭伤仅 0.5/箭 → 40 箭（慢但非 0）；冰弓 7 箭
    assert turns_to_kill_bow(20, g, 0.9, elemental_required=True) == 40.0
    g_ice = Gear(sword=1, strength=1, bow=1, bow_enchant=2, has_elemental=True)
    assert turns_to_kill_bow(20, g_ice, 0.9, elemental_required=True) == 7.0


def test_bow_survival_verdict():
    # 弓让 L1/L2 直接 CLEARABLE（无需剑/甲）；L3 蜥蜴伤高 → CLEARABLE/MARGINAL
    g_bow = Gear(sword=0, armour=0, strength=1, bow=1)
    assert survival_verdict(1, g_bow) == "CLEARABLE"
    assert survival_verdict(2, g_bow) == "CLEARABLE"
    assert survival_verdict(3, g_bow) in ("CLEARABLE", "MARGINAL")
    # L4 骑士物免：无附魔弓 MARGINAL；附魔弓 3 箭/怪、无甲仍 MARGINAL（骑士 6 伤/击）
    # —— 不 INFEASIBLE 即可，护甲/锚点恢复兜底（对应 FLOOR_GEAR_REQ L4 需甲）
    assert survival_verdict(4, g_bow) == "MARGINAL"
    g_ench = Gear(sword=0, armour=0, strength=1, bow=1, bow_enchant=1)
    assert survival_verdict(4, g_ench) != "INFEASIBLE"
    assert survival_verdict(4, g_ench) in ("CLEARABLE", "MARGINAL")
    # L6/L7 无元素 → 弓也 INFEASIBLE
    assert survival_verdict(6, g_bow) == "INFEASIBLE"


def test_bow_recommend_tactic():
    # 有弓 + 弓不劣于近战 → bow 战术
    g_bow = Gear(sword=1, armour=0, strength=1, bow=1)
    assert recommend_tactic(1, g_bow) == "bow"
    # 弓 + 高近战装备：L4 无附魔弓不如近战 → 不用 bow
    g_strong = Gear(sword=3, armour=2, strength=3, bow=1)
    assert recommend_tactic(4, g_strong) != "bow"
    # 无弓 → 不进 bow 分支
    assert recommend_tactic(1, Gear(sword=1)) in ("stand", "kite")


def test_bow_damage_per_clear_positive():
    g_bow = Gear(sword=0, armour=0, strength=1, bow=1)
    for f in (0, 1, 2, 3):
        assert damage_per_clear_bow(f, g_bow) > 0
        assert damage_per_clear_bow(f, g_bow) <= damage_per_clear(f, Gear(sword=1, strength=1))


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


# ---------------------------------------------------------------------------
# 内在状态前瞻（睡眠/等待投影）与弹药预算
# ---------------------------------------------------------------------------


def test_project_sleep_healthy_recovers():
    """必需品充足时睡眠回血回能量：投影血量应上升且不掉水/食到 0。"""
    from craftax.planner.combat_model import project_sleep

    proj = project_sleep(energy=2.0, health=6.0, drink=9.0, food=9.0,
                         strength=1, dexterity=1)
    assert proj.steps == pytest.approx(5.0 * 11.0)   # (7-2) 点 × 11 步/点
    assert proj.health_end > 6.0
    assert not proj.dies
    assert proj.drink_end > 0.0 and proj.food_end > 0.0


def test_project_sleep_thirsty_is_net_damage():
    """渴着睡是净掉血：drink=0 时 recover 为负（31 步/HP），必须能被识别。"""
    from craftax.planner.combat_model import project_sleep

    proj = project_sleep(energy=2.0, health=6.0, drink=0.0, food=9.0,
                         strength=1, dexterity=1)
    assert proj.health_end < 6.0
    proj_low = project_sleep(energy=0.0, health=2.0, drink=0.0, food=0.0,
                             strength=1, dexterity=1)
    assert proj_low.dies


def test_projected_awake_health_matches_game_rates():
    """醒着待机：必需品齐备 26 步/HP 回血，缺任一项 16 步/HP 掉血。"""
    from craftax.planner.combat_model import projected_awake_health

    up = projected_awake_health(steps=52.0, health=5.0, drink=9.0, food=9.0,
                                energy=9.0)
    assert up == pytest.approx(7.0)
    down = projected_awake_health(steps=32.0, health=5.0, drink=0.0, food=9.0,
                                  energy=9.0)
    assert down == pytest.approx(3.0)


def test_arrows_for_clear_l1_and_elemental():
    """弹药预算：L1 兽人约 17 支；L6/L7 无元素能力时弓需求爆炸（不该靠弓）。"""
    from craftax.planner.combat_model import Gear, arrows_for_clear

    gear = Gear(bow=1, dexterity=1, strength=1)
    assert arrows_for_clear(1, gear) == 17
    assert arrows_for_clear(2, gear) == 20
    # 90% 物免层：普通箭需求数百支 → 执行器据此判定"不能靠弓清元素层"
    assert arrows_for_clear(6, gear) > 100


# ---------------------------------------------------------------------------
# 弹药经济学：备箭 vs 造石/铁剑（§6.2.6a 出路 a）
# ---------------------------------------------------------------------------


def test_arrows_for_clear_reserves_more_when_not_restockable():
    """不可补弹层（地牢无木）按 1.75x 预留而不是 1.25x 均值：
    L1 的 17 支只是理论均值，打空后剩下的怪只能用剑。"""
    from craftax.planner.combat_model import Gear, arrows_for_clear

    gear = Gear(bow=1, dexterity=1, strength=1)
    assert arrows_for_clear(1, gear, restockable=False) == 23
    assert arrows_for_clear(2, gear, restockable=False) == 28
    for floor in (1, 2, 3):
        assert (arrows_for_clear(floor, gear, restockable=False)
                > arrows_for_clear(floor, gear, restockable=True))


def test_bow_coverage_and_mixed_damage():
    """混合模型：箭只够打前一段，残余按剑算 → 伤害在弓与剑之间线性插值。"""
    from craftax.planner.combat_model import (
        Gear,
        bow_kill_coverage,
        damage_per_clear_bow,
        mixed_damage_per_clear,
    )

    gear = Gear(sword=1, bow=1, strength=1)   # 木剑 + 弓
    assert bow_kill_coverage(1, gear, 0) == 0.0
    assert bow_kill_coverage(1, gear, 17) == pytest.approx(1.0)
    assert 0.4 < bow_kill_coverage(1, gear, 8) < 0.6
    # 0 箭 = 纯近战；备满 = 纯弓；中间严格居中
    assert mixed_damage_per_clear(1, gear, 0) == pytest.approx(
        damage_per_clear(1, gear)
    )
    assert mixed_damage_per_clear(1, gear, 17) == pytest.approx(
        damage_per_clear_bow(1, gear)
    )
    assert (damage_per_clear_bow(1, gear)
            < mixed_damage_per_clear(1, gear, 8)
            < damage_per_clear(1, gear))


def test_clear_prep_prefers_stone_sword_when_ammo_short():
    """L1 的核心结论：石剑与 2 支箭同价（1 木 + 1 石），而弹药缺口大时
    石剑压低的是**全部残余击杀**的单价 → 先造石剑再备箭。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    gear = Gear(sword=1, bow=1, strength=1)   # 首箱弓 + 木剑
    prep = recommend_clear_prep(1, gear, arrows=8, restockable=False,
                                iron_available=False)
    assert prep.prefer == "sword"
    assert prep.sword_target == 2
    assert prep.gain_per_unit_sword > prep.gain_per_unit_arrows
    # 石剑到手后，同样的材料买箭更划算（无铁时不再有升剑候选）
    after = recommend_clear_prep(1, Gear(sword=2, bow=1, strength=1), arrows=8,
                                 restockable=False, iron_available=False)
    assert after.prefer == "arrows"
    assert after.sword_target == 0
    assert after.damage_now < prep.damage_now


def test_clear_prep_skips_deep_prep_when_ammo_covers_clear():
    """旧行为（有弓就跳过深制备）应作为"弹药够/可补"的推论保留下来。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    gear = Gear(sword=1, bow=1, strength=1)
    stocked = recommend_clear_prep(1, gear, arrows=23, restockable=False,
                                   iron_available=True)
    assert stocked.prefer == "ready"
    assert stocked.sword_target == 0
    # 本层能就地补箭（有木有石，如 L0）→ 残余为 0，同样不做深制备
    restock = recommend_clear_prep(1, gear, arrows=2, restockable=True,
                                   iron_available=True)
    assert restock.prefer == "ready"
    assert restock.sword_target == 0


def test_clear_prep_iron_sword_only_when_iron_reachable():
    """铁剑要 1 铁 + 1 煤 + 熔炉（记 4 单位成本）：拿不到铁时不作为候选。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    gear = Gear(sword=2, bow=1, strength=1)   # 已有石剑
    no_iron = recommend_clear_prep(1, gear, arrows=0, restockable=False,
                                   iron_available=False)
    assert no_iron.sword_target == 0          # 无候选 → 只能备箭
    with_iron = recommend_clear_prep(1, gear, arrows=0, restockable=False,
                                     iron_available=True)
    assert with_iron.sword_target == 3
    assert with_iron.prefer == "sword"


def test_clear_prep_arrow_target_respects_cap():
    """携带上限：备箭目标不超过 arrow_cap（L3 的 37 支要 19 木 19 石，不现实）。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    gear = Gear(sword=1, bow=1, strength=1)
    prep = recommend_clear_prep(3, gear, arrows=0, restockable=False,
                                arrow_cap=30)
    assert prep.arrows_target == 30


def test_clear_prep_bow_useless_floor_falls_back_to_melee():
    """弓打不动的层（L6 无元素，箭伤 0.5）→ 残余 100%，只能靠剑/元素。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    gear = Gear(sword=1, bow=1, strength=1)
    prep = recommend_clear_prep(6, gear, arrows=8, restockable=False,
                                iron_available=True)
    assert prep.prefer == "sword"
    assert prep.coverage < 0.1


def test_clear_prep_stops_when_sword_matches_bow():
    """铁剑在 L1 已经和弓等效（兽人 2 击杀 = 2 箭）→ 再为清层备箭/升剑都买不到
    减伤，制备必须收敛（否则执行器会一直"再多备一点"，整局停在地表）。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    iron = Gear(sword=3, bow=1, strength=1)
    prep = recommend_clear_prep(1, iron, arrows=0, restockable=False,
                                iron_available=True, max_sword=3)
    assert prep.bow_advantage == pytest.approx(0.0)
    assert prep.prefer == "ready"
    assert prep.sword_target == 0


def test_clear_prep_ignores_swords_it_cannot_build():
    """max_sword 限制候选：只给"当场做得出来"的等级。放开到钻石剑时，
    每单位收益 0.4 也会赢过 0 收益的备箭 → 制备链无限打转（实测死因）。"""
    from craftax.planner.combat_model import Gear, recommend_clear_prep

    gear = Gear(sword=3, bow=1, strength=1)
    open_ended = recommend_clear_prep(1, gear, arrows=0, restockable=False,
                                      iron_available=True, diamond_available=True,
                                      max_sword=4)
    assert open_ended.sword_target == 0   # 收益 0.4 < MIN_SWORD_GAIN_ABS
    capped = recommend_clear_prep(1, gear, arrows=0, restockable=False,
                                  iron_available=True, diamond_available=True,
                                  max_sword=3)
    assert capped.sword_target == 0
