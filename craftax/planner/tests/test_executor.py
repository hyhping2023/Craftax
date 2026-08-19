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
    DO,
    LEVEL_UP_DEXTERITY,
    LEVEL_UP_STRENGTH,
    MAKE_WOOD_PICKAXE,
    NOOP,
    PLACE_TABLE,
    RIGHT,
    SLEEP,
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


def _floor_map_payload(hs, floor: int) -> dict:
    """任意楼层的 map payload（等价 GET /map?floor=N）。

    供执行器的跨层观察使用：下楼之前就能知道目标层有没有要采的矿。
    """
    import numpy as _np

    from craftax.craftax.constants import BlockType

    map_arr = _np.asarray(hs.map[floor])
    chest_rows, chest_cols = _np.where(map_arr == BlockType.CHEST.value)
    return {
        "floor": floor,
        "map": map_arr.tolist(),
        "ladder_down": [int(x) for x in hs.down_ladders[floor]],
        "ladder_up": [int(x) for x in hs.up_ladders[floor]],
        "monsters_killed": int(hs.monsters_killed[floor]),
        "chest_positions": [[int(x), int(y)] for x, y in zip(chest_rows, chest_cols)],
    }


def run_task(task_id: str, seed: int = 2026, max_steps: int = 2000) -> dict:
    env = CraftaxSymbolicEnvNoAutoReset()
    state = env.reset(jax.random.PRNGKey(seed), EnvParams())[1]
    # seed 透传：执行器据此加载 WorldFacts（跨层矿石/梯子事实）；
    # floor_map_provider：任意层全图，让"该层到底有没有这种矿"成为已知量。
    holder: dict = {}

    def provider(floor: int):
        hs = holder.get("hs")
        if hs is None or not 0 <= floor < hs.map.shape[0]:
            return None
        return _floor_map_payload(hs, floor)

    executor = SkillChainExecutor(task_id, seed=seed, floor_map_provider=provider)
    if max_steps <= 0:
        max_steps = executor.estimate_steps()
    key_rng = jax.random.PRNGKey(seed + 1)
    result = {"task_id": task_id, "steps": 0, "done": False, "floor": 0}
    for i in range(max_steps):
        hs = _host(state)
        holder["hs"] = hs  # 执行器按 TTL 自行失效跨层地图缓存
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


# ---------------------------------------------------------------------------
# 回归测试：本轮修复的缺陷（崩溃 / 完成判定 / 排序 / 生存优先级 / 看门狗）
# ---------------------------------------------------------------------------


def _fake_payload(map2d=None, monsters_killed: int = 10, mobs=None,
                  ladder_down=(10, 10), ladder_up=(30, 30), chests=()):
    if map2d is None:
        map2d = np.full((48, 48), 2, dtype=np.int32)  # 全草地
    empty = {"positions": [], "masks": []}
    return {
        "floor": 0,
        "map": map2d,
        "mob_positions": mobs or {k: dict(empty) for k in ("melee", "ranged", "passive")},
        "ladder_down": list(ladder_down),
        "ladder_up": list(ladder_up),
        "monsters_killed": monsters_killed,
        "chest_positions": [list(c) for c in chests],
    }


def _fake_summary(**over):
    summary = {
        "floor": 0,
        "player_position": [24, 24],
        "player_direction": 4,
        "health": 9.0, "energy": 9.0, "food": 9.0, "drink": 9.0, "mana": 9.0,
        "xp": 0, "strength": 5, "dexterity": 1, "intelligence": 1,
        "is_sleeping": False, "is_resting": False,
        "sword_enchantment": 0, "bow_enchantment": 0,
        "learned_spells": [False, False],
        "achievements": [],
        "inventory": {"wood": 5, "stone": 5, "coal": 1, "iron": 1, "diamond": 0,
                      "sapphire": 0, "ruby": 0, "sapling": 0, "pickaxe": 2,
                      "sword": 2, "bow": 1, "arrows": 8, "torches": 0,
                      "armour": [0, 0, 0, 0], "books": 0, "potions": [0] * 6},
    }
    inv = over.pop("inventory", None)
    summary.update(over)
    if inv:
        summary["inventory"].update(inv)
    return summary


@pytest.mark.parametrize(
    "task_id",
    ["native.explore_dungeon", "native.dungeon_campaign", "native.reach_floor_3",
     "native.reach_floor_5", "native.reach_boss_floor"],
)
def test_reach_floor_tasks_never_raise(task_id):
    """回归：REACH_FLOOR.get(tid, ENTER_FLOOR[tid]) 的默认值被立即求值，
    使 explore_dungeon / dungeon_campaign 在 floor>=1 时抛 KeyError。"""
    for floor in range(0, 9):
        for killed in (10, 3):
            ex = SkillChainExecutor(task_id)
            ex.next_action(_fake_payload(monsters_killed=killed),
                           _fake_summary(floor=floor))


def test_composite_goals_complete_via_registry_predicate():
    """回归：复合/根目标过去恒为"未完成"（_task_is_complete 兜底 return False），
    即使成就全开也报失败。完成判定现在走 registry 的 success_predicate。"""
    from craftax.planner.executor import COMPOSITE_TASKS

    full = _fake_summary(
        floor=8,
        achievements=[a.name for a in Achievement],
        inventory={"sword": 4, "pickaxe": 4, "armour": [2, 2, 2, 2]},
    )
    for task_id in sorted(COMPOSITE_TASKS):
        ex = SkillChainExecutor(task_id)
        assert ex.is_done(full), f"{task_id} 应判定完成"
    # 反向：什么都没做时不该判定完成
    empty = _fake_summary(floor=0, achievements=[])
    assert not SkillChainExecutor("native.master_crafter").is_done(empty)


def test_max_floor_uses_nearest_spawn_floor():
    """回归：战斗任务的 _max_floor 取 max(刷新层) → 骷髅([0,8]) 被当成 L8 任务，
    触发整套深层制备（实测让地表任务绕道 L1 并暴死）。"""
    assert SkillChainExecutor("native.defeat_skeleton")._max_floor == 0
    assert SkillChainExecutor("native.defeat_zombie")._max_floor == 0
    assert SkillChainExecutor("native.collect_diamond")._max_floor == 2
    assert SkillChainExecutor("native.defeat_necromancer")._max_floor == 8


def test_chain_is_valid_topological_order():
    """成本感知排序仍须是合法拓扑序（依赖先于消费者）。"""
    from craftax.tasks.graph import TaskGraph

    graph = TaskGraph.build_from_registry()
    for task_id in ("native.defeat_necromancer", "native.collect_all_ores",
                    "native.crafting_mastery", "native.defeat_troll"):
        chain = SkillChainExecutor(task_id).chain()
        seen = set()
        for tid in chain:
            for dep in graph.dependencies(tid):
                if dep in chain:
                    assert dep in seen, f"{task_id}: {tid} 早于其依赖 {dep}"
            seen.add(tid)


def test_chain_groups_work_by_floor():
    """成本感知排序：装备/合成在"路过该层时"做掉，而不是按 task_id 字母序
    插在下楼动作之间。旧顺序会先下 L2 再回头放工作台，钻石剑排到 L8 之后。"""
    chain = SkillChainExecutor("native.defeat_necromancer").chain()
    idx = {tid: i for i, tid in enumerate(chain)}
    # 地表制备全部早于第一次下楼
    for tid in ("native.collect_wood", "native.place_table",
                "native.craft_wood_pickaxe", "native.collect_stone"):
        assert idx[tid] < idx["native.enter_dungeon"], tid
    # 钻石装备在拿到钻石的矿层就做，不拖到 Boss 层
    assert idx["native.craft_diamond_sword"] < idx["native.enter_graveyard"]
    assert idx["native.collect_diamond"] < idx["native.craft_diamond_sword"]
    # 学法术在书层（L3/L4），且早于需要元素能力的火/冰界
    assert idx["native.learn_fireball"] < idx["native.enter_ice_realm"]
    assert idx["native.learn_iceball"] < idx["native.enter_fire_realm"]


def test_supply_takes_priority_over_regen_and_sleep():
    """回归（实测死因）：drink=0 时执行器先"原地回血"再"睡觉"，
    而缺水会让被动回血变成掉血、睡眠更掉血——必须先去喝水。"""
    map2d = np.full((48, 48), 2, dtype=np.int32)
    map2d[24, 27] = 3  # 右侧 3 格处有水
    ex = SkillChainExecutor("native.collect_diamond")
    summ = _fake_summary(health=5.0, energy=2.0, drink=0.0, food=6.0)
    action = ex._survival_action(_fake_payload(map2d=map2d), summ)
    assert action not in (SLEEP, None), "应去补水而不是睡觉/干等"
    # 水在正右方 → 朝右移动/转向或直接 DO 取水
    assert action in (RIGHT, DO)


def test_sleep_projection_refuses_lethal_sleep():
    """睡眠前瞻：渴着睡（会掉血且无法自救）必须被否掉；补给充足时才睡。"""
    ex = SkillChainExecutor("native.collect_diamond")
    assert not ex._sleep_is_safe(_fake_summary(energy=2.0, health=6.0, drink=0.0))
    assert not ex._sleep_is_safe(_fake_summary(energy=0.0, health=3.0, drink=1.0,
                                              food=1.0))
    assert ex._sleep_is_safe(_fake_summary(energy=2.0, health=8.0, drink=9.0,
                                          food=9.0))


def test_stall_watchdog_breaks_and_aborts():
    """看门狗：状态完全不变时先换动作打破僵持，长期无进展则中止该 seed。"""
    ex = SkillChainExecutor("native.collect_diamond")
    payload = _fake_payload()
    summ = _fake_summary()
    actions = [ex.next_action(payload, summ) for _ in range(200)]
    assert ex._stall_steps > 0, "应识别出无进展"
    assert len(set(a for a in actions if a is not None)) > 1, "应尝试不同动作"
    for _ in range(600):
        if ex.next_action(payload, summ) is None:
            break
    assert ex.abort_reason() is not None and "停滞" in ex.abort_reason()


def test_cross_floor_resource_choice_skips_known_empty_floor():
    """跨层观察：目标层"已知没有该矿"就不去（旧实现只能按静态偏好表试错）。"""
    from craftax.planner.executor import _preferred_collect_floor
    from craftax.planner.world import FloorFacts, WorldFacts

    # collect_iron 偏好 [2, 5, 0]；令 L2 无铁、L5 有铁
    facts = WorldFacts(seed=-1, floors={
        0: FloorFacts(floor=0, ore={"iron": 0}, ladder_down_reachable=True),
        2: FloorFacts(floor=2, ore={"iron": 0}, ladder_down_reachable=True),
        5: FloorFacts(floor=5, ore={"iron": 4}, ladder_down_reachable=True),
    })
    assert _preferred_collect_floor("native.collect_iron", 0, facts) == 5
    # 没有事实时保持旧行为（偏好表首位）
    assert _preferred_collect_floor("native.collect_iron", 0, None) == 2


def test_floor_map_provider_counts_blocks():
    """floor_map_provider 注入的任意层地图应被用于资源计数。"""
    other = np.full((48, 48), 2, dtype=np.int32)
    other[5, 5] = 9  # IRON
    ex = SkillChainExecutor(
        "native.collect_iron",
        floor_map_provider=lambda f: ({"floor": f, "map": other} if f == 5 else
                                      {"floor": f, "map": np.full((48, 48), 2,
                                                                  dtype=np.int32)}),
    )
    assert ex._floor_resource_count(5, [9]) == 1
    assert ex._floor_resource_count(2, [9]) == 0


def test_floor_can_restock_arrows_uses_cross_floor_map():
    """弹药可补性：MAKE_ARROW 要 1 木 + 1 石，地牢层没有树 → 不可补。
    有跨层地图就实测该层树/石，没有则按地形约定（只有 L0/L6 可能有木）。"""
    surface = np.full((48, 48), 2, dtype=np.int32)
    surface[3, 3] = 5   # TREE
    surface[3, 4] = 4   # STONE
    dungeon = np.full((48, 48), 4, dtype=np.int32)  # 全石头，无树
    ex = SkillChainExecutor(
        "native.enter_gnomish_mines",
        floor_map_provider=lambda f: {"floor": f,
                                      "map": surface if f == 0 else dungeon},
    )
    assert ex._floor_can_restock_arrows(0) is True
    assert ex._floor_can_restock_arrows(1) is False
    # 无 provider（未知）→ 按地形约定
    ex2 = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex2._floor_can_restock_arrows(0) is True
    assert ex2._floor_can_restock_arrows(1) is False


def test_clear_prep_prefers_stone_sword_then_arrows():
    """备箭 vs 升剑的择优接到执行器：L1 不可补弹 + 箭不足 → 先造石剑；
    石剑到手后转为备箭（旧逻辑"有弓就跳过深制备"在这里会一直备箭）。"""
    ex = SkillChainExecutor("native.enter_gnomish_mines")  # max_floor=2 → 需清 L1
    map2d = np.full((16, 16), 2, dtype=np.int32)
    map2d[0, 0] = 5   # TREE（本层可采木）
    map2d[0, 1] = 4   # STONE
    payload = _fake_payload(map2d=map2d, ladder_down=(15, 15))
    summ = _fake_summary(floor=0, strength=1,
                         inventory={"sword": 1, "pickaxe": 1, "bow": 1,
                                    "arrows": 8, "wood": 3, "stone": 3})
    prep = ex._clear_prep(1, payload, summ)
    assert prep.prefer == "sword" and prep.sword_target == 2, prep.reason
    assert prep.arrows_target == 23   # 不可补层按 1.75x 预留
    # 箭备齐后不再要求深制备（旧行为作为特例保留）
    summ_stocked = _fake_summary(floor=0, strength=1,
                                 inventory={"sword": 1, "pickaxe": 1, "bow": 1,
                                            "arrows": 23, "wood": 3, "stone": 3})
    assert ex._clear_prep(1, payload, summ_stocked).sword_target == 0


def test_descend_prep_crafts_sword_before_stocking_arrows():
    """下楼前的动作顺序应遵循择优结果：箭缺口大时先造石剑，而不是先合成箭。"""
    from craftax.planner.executor import (
        CRAFT_TABLE_BLOCK,
        MAKE_ARROW,
        MAKE_STONE_SWORD,
    )

    map2d = np.full((16, 16), 2, dtype=np.int32)
    map2d[12, 12] = CRAFT_TABLE_BLOCK   # 工作台就在玩家旁边
    payload = _fake_payload(map2d=map2d, ladder_down=(15, 15))
    summ = _fake_summary(floor=0, strength=1, player_position=[12, 13],
                         inventory={"sword": 1, "pickaxe": 1, "bow": 1,
                                    "arrows": 8, "wood": 4, "stone": 4})
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex._descend_to(payload, summ, 2) == MAKE_STONE_SWORD
    # 石剑到手 → 同样的材料转去备箭
    summ["inventory"]["sword"] = 2
    ex2 = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex2._descend_to(payload, summ, 2) == MAKE_ARROW


def test_arrow_feedstock_carried_for_deeper_floors():
    """过路层还在更深处时（L1 之后还要清 L2），木料只有地表有 →
    下楼前先把合成箭用的木带够（+2 供深层放工作台）。"""
    ex = SkillChainExecutor("native.reach_floor_3")   # 需清 L1、L2
    summ = _fake_summary(floor=0, strength=1,
                         inventory={"bow": 1, "arrows": 30, "wood": 0})
    assert ex._arrow_feedstock_target(1, summ) == ex.ARROW_FEEDSTOCK_CAP
    # 下一层就是最深层（不再有过路层）→ 不必带木料
    ex_shallow = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex_shallow._arrow_feedstock_target(2, summ) == 0


def test_survival_restock_stops_at_base_reserve():
    """生存层补箭只补到基础储备（8）：清层用的完整弹药预算由下楼路径负责。
    否则地表持续刷怪 + 弓持续消耗使"补到 23 支"永远不满足——实测整局 1800 步
    里 635 步在合成箭、一次没下楼。"""
    from craftax.planner.executor import (
        BOW_ARROW_RESERVE,
        CRAFT_TABLE_BLOCK,
        MAKE_ARROW,
    )

    map2d = np.full((48, 48), 2, dtype=np.int32)
    map2d[24, 25] = CRAFT_TABLE_BLOCK
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    ex._restock_target = 23        # 下楼路径算出的清层预算
    payload = _fake_payload(map2d=map2d)
    # 已达基础储备 → 不再为补箭停留
    full = _fake_summary(inventory={"arrows": BOW_ARROW_RESERVE, "wood": 5, "stone": 5})
    assert ex._survival_action(payload, full) != MAKE_ARROW
    # 低于基础储备 → 仍然补（0-1 箭等于待宰）
    low = _fake_summary(inventory={"arrows": 2, "wood": 5, "stone": 5})
    assert ex._survival_action(payload, low) == MAKE_ARROW


def test_defeat_mob_locations_match_game_constants():
    """DEFEAT_MOB_LOCATIONS 必须与 FLOOR_MOB_MAPPING 一致（单一真相源），
    且任务依赖图声明的入口层不得比怪的刷新层更深（旧图把兽人标在 L3）。"""
    from craftax.craftax.constants import FLOOR_MOB_MAPPING
    from craftax.planner.executor import DEFEAT_MOB_LOCATIONS, ENTER_FLOOR
    from craftax.tasks.graph import TaskGraph

    mapping = np.asarray(FLOOR_MOB_MAPPING)
    class_index = {"melee": 1, "ranged": 2}
    graph = TaskGraph.build_from_registry()
    for task_id, (mob_class, mob_type, floors) in DEFEAT_MOB_LOCATIONS.items():
        if mob_class == "boss":
            continue
        for floor in floors:
            if floor == 8:
                continue  # Boss 层刷 type 0 波次怪，不在 FLOOR_MOB_MAPPING 语义内
            assert int(mapping[floor][class_index[mob_class]]) == mob_type, (
                f"{task_id}: L{floor} 的 {mob_class} 不是 type {mob_type}"
            )
        deps = graph.closure(task_id)
        entered = [ENTER_FLOOR[d] for d in deps if d in ENTER_FLOOR]
        if entered:
            assert max(entered) <= max(floors), (
                f"{task_id}: 依赖图要求下到 L{max(entered)}，但怪在 L{floors}"
            )
