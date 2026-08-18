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
from craftax.planner.executor import SkillChainExecutor  # noqa: E402


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
        "is_sleeping": bool(hs.is_sleeping),
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
# 深层任务必须优先用这些 golden seeds。
GOOD_SEEDS = (3017, 3050)


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
    """深层任务：优先 golden seeds（梯子全部可达），任一成功即通过。"""
    result = None
    for seed in (*GOOD_SEEDS, 2027, 2028, 2026):
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
        for seed in (*GOOD_SEEDS, 2027, 2028, 2026):
            result = run_task(task_id, seed=seed, max_steps=0)
            if result["done"]:
                break
        assert result is not None and result["done"], f"{task_id} 未完成: {result}"


@pytest.mark.slow
def test_enchant_tasks_slow():
    """附魔任务：enchant_sword / enchant_armour（需宝石 + 附魔台 + 满蓝）。"""
    for task_id in ("native.enchant_sword", "native.enchant_armour"):
        result = None
        for seed in (*GOOD_SEEDS, 2027, 2028, 2026):
            result = run_task(task_id, seed=seed, max_steps=0)
            if result["done"]:
                break
        assert result is not None and result["done"], f"{task_id} 未完成: {result}"


@pytest.mark.slow
def test_boss_task_slow():
    """Boss 战：defeat_necromancer（需清 7 层 + 元素附魔 + 打 Boss，极慢）。"""
    result = None
    for seed in (*GOOD_SEEDS, 2027, 2028, 2026):
        result = run_task("native.defeat_necromancer", seed=seed, max_steps=0)
        if result["done"]:
            break
    assert result is not None and result["done"], f"defeat_necromancer 未完成: {result}"
