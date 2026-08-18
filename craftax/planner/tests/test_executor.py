"""SkillChainExecutor 集成测试：直接用真实 EnvState 驱动。

每步从 host_state 构造 map_payload + summary（等价于 GET /map + step 响应），
再交给 executor，验证依赖图推导的技能链能真正完成任务。
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.constants import Achievement  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.planner.executor import (  # noqa: E402
    LEVEL_UP_DEXTERITY,
    LEVEL_UP_STRENGTH,
    MAKE_WOOD_PICKAXE,
    NOOP,
    PLACE_TABLE,
    SkillChainExecutor,
)


def _host(state):
    return jax.device_get(state)


def _map_payload(hs):
    level = int(hs.player_level)
    mobs = {}
    for key in ("melee", "ranged", "passive"):
        mob = getattr(hs, f"{key}_mobs")
        m_pos = np.asarray(mob.position[level])
        m_mask = np.asarray(mob.mask[level])
        mobs[key] = {
            "positions": [p.tolist() for p in m_pos],
            "masks": [bool(x) for x in m_mask],
        }
    map_arr = np.asarray(hs.map[level])
    from craftax.craftax.constants import BlockType

    chest_rows, chest_cols = np.where(map_arr == BlockType.CHEST.value)
    return {
        "floor": level,
        "map": map_arr.tolist(),
        "player_position": [int(x) for x in hs.player_position],
        "player_direction": int(hs.player_direction),
        "mob_positions": mobs,
        "ladder_down": [int(x) for x in hs.down_ladders[level]],
        "ladder_up": [int(x) for x in hs.up_ladders[level]],
        "monsters_killed": int(hs.monsters_killed[level]),
        "chest_positions": [[int(x), int(y)] for x, y in zip(chest_rows, chest_cols)],
    }


def _summary(hs):
    inv = hs.inventory
    return {
        "floor": int(hs.player_level),
        "player_position": [int(x) for x in hs.player_position],
        "player_direction": int(hs.player_direction),
        "energy": float(hs.player_energy),
        "food": float(hs.player_food),
        "drink": float(hs.player_drink),
        "health": float(hs.player_health),
        "mana": float(hs.player_mana),
        "xp": int(hs.player_xp),
        "strength": int(hs.player_strength),
        "dexterity": int(hs.player_dexterity),
        "intelligence": int(hs.player_intelligence),
        "is_sleeping": bool(hs.is_sleeping),
        "is_resting": bool(hs.is_resting),
        "sword_enchantment": int(hs.sword_enchantment),
        "bow_enchantment": int(hs.bow_enchantment),
        "armour_enchantments": [int(x) for x in np.asarray(hs.armour_enchantments).ravel()],
        "learned_spells": [bool(x) for x in np.asarray(hs.learned_spells).ravel()],
        "inventory": {
            "wood": int(inv.wood),
            "stone": int(inv.stone),
            "coal": int(inv.coal),
            "iron": int(inv.iron),
            "diamond": int(inv.diamond),
            "sapphire": int(inv.sapphire),
            "ruby": int(inv.ruby),
            "sapling": int(inv.sapling),
            "pickaxe": int(inv.pickaxe),
            "sword": int(inv.sword),
            "bow": int(inv.bow),
            "arrows": int(inv.arrows),
            "torches": int(inv.torches),
            "armour": [int(x) for x in inv.armour],
            "books": int(inv.books),
            "potions": [int(x) for x in inv.potions],
        },
        "achievements": [
            a.name for a in Achievement if bool(np.asarray(hs.achievements)[int(a.value)])
        ],
    }


def run_task(task_id: str, seed: int = 2026, max_steps: int = 2000) -> dict:
    env = CraftaxSymbolicEnvNoAutoReset()
    state = env.reset(jax.random.PRNGKey(seed), EnvParams())[1]
    executor = SkillChainExecutor(task_id)
    if max_steps <= 0:
        max_steps = executor.estimate_steps()
    key_rng = jax.random.PRNGKey(seed + 1)
    result = {"task_id": task_id, "steps": 0, "done": False, "floor": 0}
    for i in range(max_steps):
        hs = _host(state)
        payload = _map_payload(hs)
        summ = _summary(hs)
        if executor.is_done(summ):
            result["done"] = True
            result["steps"] = i
            result["floor"] = int(hs.player_level)
            result["inventory"] = summ["inventory"]
            result["achievements"] = summ["achievements"]
            return result
        action = executor.next_action(payload, summ)
        if action is None:
            result["error"] = "no_action"
            result["steps"] = i
            result["floor"] = int(hs.player_level)
            result["goal"] = executor._chain[executor._chain_idx] if executor._chain_idx < len(executor._chain) else None
            return result
        key_rng, k2 = jax.random.split(key_rng)
        obs, state, reward, done, info = env.step(k2, state, action, EnvParams())
        if bool(np.asarray(done)) and not executor.is_done(_summary(_host(state))):
            # 玩家死亡（health<=0）或超时：episode 结束且任务未完成
            result["error"] = "died_or_truncated"
            result["steps"] = i + 1
            result["floor"] = int(_host(state).player_level)
            return result
    hs = _host(state)
    result["steps"] = max_steps
    result["floor"] = int(hs.player_level)
    result["inventory"] = _summary(hs)["inventory"]
    result["achievements"] = _summary(hs)["achievements"]
    return result


# 深层任务使用的"好种子"：scan_seeds.py 扫描出的 L0-L7 梯子全部可达的种子。
# 普通种子（2026/2027/2028）存在 L0 或深层梯子被 WATER 分隔而不可达的问题，
# 深层任务必须优先用这些 golden seeds。3017/3050 为早期发现；
# 2011/2111 为扩大扫描（2000-2999）发现的 golden∩L0 装甲可行（可做铁甲）种子。
GOOD_SEEDS = (2011, 2111, 3017, 3050)


def _deep_seeds(task_id: str) -> tuple:
    """深层任务的候选种子：先取 scan_seeds 就绪度排序的候选，再补 golden + 常规。"""
    from craftax.planner.world import best_seeds, load_scan_results

    try:
        cands = best_seeds(task_id, n=5)
    except Exception:  # noqa: BLE001  seed_scan.json 缺失/损坏时回退
        cands = []
    seen = set(cands)
    for s in (*cands, *GOOD_SEEDS, 2027, 2028, 2026):
        if s not in seen:
            seen.add(s)
            cands.append(s)
    return tuple(cands)


@pytest.mark.parametrize(
    "task_id",
    [
        "native.collect_wood",
        "native.collect_coal",
        "native.collect_stone",
        "native.collect_iron",
        "native.collect_diamond",
    ],
)
def test_collect_tasks_complete(task_id):
    """收集类任务：尝试多个 seed（某些 seed 梯子不可达），任一成功即通过。"""
    result = None
    for seed in (2026, 2027, 2028, *GOOD_SEEDS):
        result = run_task(task_id, seed=seed, max_steps=6000)
        if result["done"]:
            break
    assert result is not None and result["done"], (
        f"{task_id} 在所有 seed 下均未完成: {result}"
    )


def test_craft_tasks_complete():
    """地表合成任务（不含需下 L2 的钻石镐——见 test_deep_task_slow）。"""
    for task_id in (
        "native.craft_wood_pickaxe",
        "native.craft_stone_pickaxe",
        "native.craft_iron_pickaxe",
    ):
        result = None
        for seed in (2026, 2027, 2028, *GOOD_SEEDS):
            result = run_task(task_id, seed=seed, max_steps=0)  # 0=按依赖链估算
            if result["done"]:
                break
        assert result is not None and result["done"], f"{task_id} 未完成: {result}"


def test_executor_chain_has_dependencies_first():
    ex = SkillChainExecutor("native.collect_coal")
    chain = ex.chain()
    # 依赖在前：collect_wood 必须先于 craft_wood_pickaxe
    assert chain.index("native.collect_wood") < chain.index("native.craft_wood_pickaxe")
    assert chain.index("native.craft_wood_pickaxe") < chain.index("native.collect_coal")


def test_should_use_bow():
    """弓主战判定：有弓时 L0-L5 用弓；L6/L7 需元素能力。"""
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ_bow = {"inventory": {"bow": 1}}
    summ_nobow = {"inventory": {"bow": 0}}
    assert ex._should_use_bow(1, summ_bow) is True
    assert ex._should_use_bow(0, summ_bow) is True
    assert ex._should_use_bow(5, summ_bow) is True
    assert ex._should_use_bow(1, summ_nobow) is False
    # L6 需冰系能力
    assert ex._should_use_bow(6, summ_bow) is False
    assert ex._should_use_bow(
        6, {"inventory": {"bow": 1}, "learned_spells": [False, True],
            "sword_enchantment": 0, "bow_enchantment": 0}
    ) is True


def test_line_clear():
    """直线射检查：中间无 solid 阻挡才射；target 本身是怪格不算阻挡。"""
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    # 3x3 全草地：从 (0,0) 到 (0,2) 无障碍
    assert ex._line_clear([[2, 2, 2], [2, 2, 2], [2, 2, 2]], (0, 0), (0, 2)) is True
    # 中间是 STONE(4)：被挡
    assert ex._line_clear([[2, 4, 2], [2, 2, 2], [2, 2, 2]], (0, 0), (0, 2)) is False
    # WATER(3) 也挡（箭可过水，但保守不射）
    assert ex._line_clear([[2, 3, 2], [2, 2, 2], [2, 2, 2]], (0, 0), (0, 2)) is False


def test_collect_resource_checks_pickaxe():
    """采集目标需要更高镐时先合成（修复：无镐挖石会卡死）。"""
    ex = SkillChainExecutor("native.collect_diamond")
    # 石头需木镐：无镐 → 返回合成木镐的动作
    map2d = np.zeros((4, 4), dtype=np.int32) + 2  # 全草地
    map2d[1][1] = 4  # STONE
    payload = {
        "map": map2d,
        "player_position": [0, 0],
        "player_direction": 2,
        "mob_positions": {"melee": {"positions": [], "masks": []},
                          "ranged": {"positions": [], "masks": []},
                          "passive": {"positions": [], "masks": []}},
        "monsters_killed": 10,
    }
    summ = {"floor": 0, "player_position": [0, 0], "player_direction": 2,
            "inventory": {"wood": 2, "stone": 0, "pickaxe": 0, "sword": 1,
                          "bow": 0, "arrows": 0, "armour": [0, 0, 0, 0]}}
    a = ex._collect_resource("native.collect_stone", payload, summ)
    # 先做木镐：无台子先放台（PLACE_TABLE），有台则直接 MAKE_WOOD_PICKAXE
    assert a in (PLACE_TABLE, MAKE_WOOD_PICKAXE)


def test_bow_rush_starts_with_sword_then_descends():
    """弓先制：无弓时先保证木剑，再下 L1（L0 已清无杀怪门槛）。"""
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    map2d = np.zeros((5, 5), dtype=np.int32) + 2  # 全草地
    map2d[1][1] = 5  # TREE
    # 有 ladder_down 在 (4,4)
    payload = {
        "map": map2d,
        "player_position": [0, 0],
        "player_direction": 2,
        "mob_positions": {"melee": {"positions": [], "masks": []},
                          "ranged": {"positions": [], "masks": []},
                          "passive": {"positions": [], "masks": []}},
        "ladder_down": [4, 4],
        "monsters_killed": 10,
    }
    summ = {"floor": 0, "player_position": [0, 0], "player_direction": 2,
            "inventory": {"wood": 0, "stone": 0, "pickaxe": 0, "sword": 0,
                          "bow": 0, "arrows": 0, "armour": [0, 0, 0, 0]}}
    # 无木 → 先采木
    a = ex._bow_rush(payload, summ)
    assert a is not None
    assert a != NOOP
    # 有弓 → 不触发弓先制
    summ["inventory"]["bow"] = 1
    assert ex._bow_rush(payload, summ) is None


def test_enter_dungeon_fast():
    """进入 L1（只需到达）：表层制备木剑后快速下行，不强制清怪。"""
    result = None
    for seed in (2026, 2027, 2028, *GOOD_SEEDS):
        result = run_task("native.enter_dungeon", seed=seed, max_steps=4000)
        if result["done"]:
            break
    assert result is not None and result["done"], f"enter_dungeon 未完成: {result}"


def test_level_up_choice_strength_first():
    """升级策略：力量优先；仅当能量瓶颈（深层长单批）或力量满后补敏捷。"""
    ex = SkillChainExecutor("native.enter_gnomish_mines")  # max_floor=2
    inv = {"sword": 3, "armour": [1, 0, 0, 0]}
    # 地表（floor 0）：力量优先
    assert ex._level_up_choice({"xp": 1, "strength": 1, "dexterity": 1,
                                "intelligence": 1, "floor": 0,
                                "inventory": inv}) == LEVEL_UP_STRENGTH
    # 深层且已有战力（非能量瓶颈）：力量优先
    assert ex._level_up_choice({"xp": 1, "strength": 2, "dexterity": 1,
                                "intelligence": 1, "floor": 2,
                                "inventory": inv}) == LEVEL_UP_STRENGTH
    # 力量满后：深层任务补敏捷
    ex_deep = SkillChainExecutor("native.defeat_necromancer")  # max_floor=8
    assert ex_deep._level_up_choice({"xp": 1, "strength": 5, "dexterity": 1,
                                     "intelligence": 1, "floor": 5,
                                     "inventory": inv}) == LEVEL_UP_DEXTERITY


# ---------------------------------------------------------------------------
# 深层任务集成测试（真实 env 步进，耗时较长 → 标记 slow，默认被 pyproject 跳过）
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "task_id",
    [
        "native.defeat_skeleton",        # 骷髅只在 L0 刷新：验证战斗楼层选择（回地表）
        "native.enter_gnomish_mines",    # 下到 L2
        "native.reach_floor_3",          # 下到 L3
        "native.collect_diamond",        # 铁镐 + 下 L2 挖钻石
        "native.collect_sapphire",       # 钻石镐 + 深矿
        "native.collect_ruby",
    ],
)
def test_deep_task_slow(task_id):
    """深层任务：优先就绪度候选/golden seeds（梯子全部可达），任一成功即通过。"""
    result = None
    for seed in _deep_seeds(task_id):
        result = run_task(task_id, seed=seed, max_steps=0)
        if result["done"]:
            break
    assert result is not None and result["done"], f"{task_id} 未完成: {result}"


@pytest.mark.slow
def test_magic_tasks_slow():
    """法术/射弓任务：learn_fireball / cast_fireball / fire_bow。"""
    for task_id in (
        "native.learn_fireball",
        "native.cast_fireball",
        "native.fire_bow",
    ):
        result = None
        for seed in _deep_seeds(task_id):
            result = run_task(task_id, seed=seed, max_steps=0)
            if result["done"]:
                break
        assert result is not None and result["done"], f"{task_id} 未完成: {result}"


@pytest.mark.slow
def test_enchant_tasks_slow():
    """附魔任务：enchant_sword / enchant_armour（需宝石 + 附魔台 + 满蓝）。"""
    for task_id in ("native.enchant_sword", "native.enchant_armour"):
        result = None
        for seed in _deep_seeds(task_id):
            result = run_task(task_id, seed=seed, max_steps=0)
            if result["done"]:
                break
        assert result is not None and result["done"], f"{task_id} 未完成: {result}"


@pytest.mark.slow
def test_boss_task_slow():
    """Boss 战：defeat_necromancer（需清 7 层 + 元素附魔 + 打 Boss，极慢）。"""
    result = None
    for seed in _deep_seeds("native.defeat_necromancer"):
        result = run_task("native.defeat_necromancer", seed=seed, max_steps=0)
        if result["done"]:
            break
    assert result is not None and result["done"], f"defeat_necromancer 未完成: {result}"
