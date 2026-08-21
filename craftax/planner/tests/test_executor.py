"""SkillChainExecutor 集成测试：直接用真实 EnvState 驱动。

每步从 host_state 构造 map_payload + summary（等价于 GET /map + step 响应），
再交给 executor，验证依赖图推导的技能链能真正完成任务。
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from craftax.contracts import DEFAULT_THIRST_RATE  # noqa: E402
from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.constants import Achievement  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.planner.executor import (  # noqa: E402
    DO,
    DOWN,
    LEVEL_UP_DEXTERITY,
    LEVEL_UP_STRENGTH,
    LEFT,
    MAKE_WOOD_PICKAXE,
    NOOP,
    PLACE_TABLE,
    RIGHT,
    SLEEP,
    SkillChainExecutor,
    UP,
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
        # timestep 是夜间策略的输入（light_level 由它推出）；服务端 summary 一直有
        # 这个字段（session_actor），测试侧缺失会让夜间逻辑在 rollout 里静默失效。
        "timestep": int(hs.timestep),
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
            "water": int(inv.water),
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


def run_task(task_id: str, seed: int = 2026, max_steps: int = 2000,
             thirst_rate: float = DEFAULT_THIRST_RATE) -> dict:
    env = CraftaxSymbolicEnvNoAutoReset()
    # 与具身会话一致的环境参数：口渴衰减放缓（contracts.DEFAULT_THIRST_RATE）。
    # env 与 executor 必须用同一个 thirst_rate，否则执行器的睡眠/等待投影会脱节。
    params = EnvParams(thirst_rate=thirst_rate)
    state = env.reset(jax.random.PRNGKey(seed), params)[1]
    # seed 透传：执行器据此加载 WorldFacts（跨层矿石/梯子事实）；
    # floor_map_provider：任意层全图，让"该层到底有没有这种矿"成为已知量。
    holder: dict = {}

    def provider(floor: int):
        hs = holder.get("hs")
        if hs is None or not 0 <= floor < hs.map.shape[0]:
            return None
        return _floor_map_payload(hs, floor)

    executor = SkillChainExecutor(task_id, seed=seed, thirst_rate=thirst_rate,
                                  floor_map_provider=provider)
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
        obs, state, reward, done, info = env.step(k2, state, action, params)
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
        "timestep": 0,          # 默认白天（light_level ≈ 0.8）
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


def test_collect_goal_lock_survives_window_refresh():
    """边缘换窗后仍追踪同一绝对资源，不能改选新窗口里的同类方块。"""
    from craftax.craftax.constants import BlockType

    ex = SkillChainExecutor("native.collect_wood")
    ex._mob_cells = set()
    summary = {
        "floor": 0,
        "player_position": [4, 4],
        "player_direction": RIGHT,
        "inventory": {},
        "achievements": [],
    }
    first_grid = np.full((9, 9), BlockType.PATH.value, dtype=np.int32)
    first_grid[4, 8] = BlockType.TREE.value
    first = {
        "map": first_grid,
        "map_origin": [100, 200],
        "player_global_position": [104, 204],
    }
    assert ex._seek_and_do(first, summary, [BlockType.TREE.value]) == RIGHT
    assert ex._goal_locks["collect:0:(5,)"] == (0, 104, 208)

    # 窗口向右平移四格：玩家和树的局部坐标变了，但绝对坐标不变。
    summary["player_position"] = [4, 0]
    refreshed_grid = np.full((9, 9), BlockType.PATH.value, dtype=np.int32)
    refreshed_grid[4, 4] = BlockType.TREE.value
    refreshed = {
        "map": refreshed_grid,
        "map_origin": [100, 204],
        "player_global_position": [104, 204],
    }
    assert ex._seek_and_do(refreshed, summary, [BlockType.TREE.value]) == RIGHT
    assert ex._goal_locks["collect:0:(5,)"] == (0, 104, 208)


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


def test_survival_restock_targets_real_ammo_budget_and_yields_to_push_mode():
    """补箭目标是战斗模型算出的真实弹药预算，不再截到 8 支。

    旧设计把生存层的补给目标截到 BOW_ARROW_RESERVE，为的是压掉"整局 1800 步里
    635 步在合成箭、一次没下楼"。但那限制的是目标而不是成本：8 支箭只够约 3 个
    击杀，而下楼门要求清 8 只怪，于是执行器带着永远不够的弹药反复下楼送死。
    现在"箭工厂"由推进优先模式兜——链推进预算一耗尽就关掉补箭。
    （另一个方案是给补箭单独加步数窗口，8 局面板 A/B 实测为净负，见
    _restock_active 的注释。）"""
    from craftax.planner.executor import (
        CHAIN_PUSH_STEPS,
        CRAFT_TABLE_BLOCK,
        MAKE_ARROW,
    )

    map2d = np.full((48, 48), 2, dtype=np.int32)
    map2d[24, 25] = CRAFT_TABLE_BLOCK
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    ex._restock_target = 23        # 下楼路径算出的清层预算
    payload = _fake_payload(map2d=map2d)
    summ = _fake_summary(inventory={"arrows": 8, "wood": 5, "stone": 5})
    # 8 支远低于真实预算 23 → 现在会补（旧设计在这里就停手了）
    assert ex._survival_action(payload, summ) == MAKE_ARROW
    # 已达目标 → 不补
    ex2 = SkillChainExecutor("native.enter_gnomish_mines")
    ex2._restock_target = 23
    stocked = _fake_summary(inventory={"arrows": 23, "wood": 5, "stone": 5})
    assert ex2._survival_action(payload, stocked) != MAKE_ARROW
    # 链推进预算耗尽 → 停止补箭，让位给推进
    ex3 = SkillChainExecutor("native.enter_gnomish_mines")
    ex3._restock_target = 23
    for _ in range(CHAIN_PUSH_STEPS + 1):
        ex3._update_chain_progress(summ)
    assert ex3._push_now()
    assert ex3._survival_action(payload, summ) != MAKE_ARROW


def test_push_mode_engages_after_chain_stalls_and_stops_optional_holds():
    """链推进预算：位置/背包一直在变但楼层与子目标不动 → 进推进优先模式。

    普通停滞检测的签名含位置与背包，"在浅层来回走着采集"每步都算有进展
    （实测 1615 步里 1539 步在 L0，而 _stall_steps 从未累积）。这里的预算只看
    (楼层, chain_idx, 成就数)。"""
    from craftax.planner.executor import CHAIN_PUSH_STEPS, PUSH_MAX_STEPS

    ex = SkillChainExecutor("native.reach_floor_3")
    # 每步都换位置、每步都多一根木头 —— 旧签名会认为一直有进展
    # +1：首次观测只是记下签名（计数 0），此后每步无推进才累加
    for i in range(CHAIN_PUSH_STEPS + 1):
        ex._update_chain_progress(
            _fake_summary(player_position=[24, 24 + (i % 5)], inventory={"wood": i})
        )
    assert ex._push_now()
    # 推进优先模式下不再进入回血 / 夜间驻守 / 补箭
    assert ex._regen_mode_active(2.0) is False
    assert ex._night_hold_active(_fake_summary(health=2.0, timestep=180)) is False
    assert ex._restock_active() is False
    # 楼层推进 → 预算清零、结束推进窗口
    ex._update_chain_progress(_fake_summary(floor=1))
    assert not ex._push_now()

    # 推进优先是**脉冲**而非常开：窗口最多 PUSH_MAX_STEPS 步，之后进冷却，
    # 让恢复类行为重新可用（否则深挖类任务会在几千步里全程无法回血）。
    ex2 = SkillChainExecutor("native.reach_floor_3")
    for _ in range(CHAIN_PUSH_STEPS + 1):
        ex2._update_chain_progress(_fake_summary())
    assert ex2._push_now()
    for _ in range(PUSH_MAX_STEPS):
        ex2._update_chain_progress(_fake_summary())
    assert not ex2._push_now(), "推进窗口必须有步数上限"
    assert ex2._push_cooldown > 0
    assert ex2._regen_mode_active(2.0) is True, "冷却期内恢复行为必须恢复可用"


def test_regen_budget_counts_sleep_steps_and_cools_down_after_normal_exit():
    """回血预算必须计入睡眠步数，且正常退出也要冷却。

    两个原实现缺陷：(1) `_survival_action` 在睡眠时早退，走不到
    `_regen_steps += 1`，于是 68 步的睡眠只消耗 0 预算；(2) 冷却只在预算耗尽
    时设置，正常回满退出不设 → "下楼→挨打→回血→下楼"的循环次数无上限。"""
    payload = _fake_payload()
    ex = SkillChainExecutor("native.reach_floor_3")
    assert ex._regen_mode_active(2.0) is True          # 进入回血模式
    before = ex._regen_steps
    ex._survival_action(payload, _fake_summary(health=2.0, is_sleeping=True))
    assert ex._regen_steps == before + 1, "睡眠步数必须计入回血预算"
    # 回满到 REGEN_EXIT_HEALTH 后正常退出 → 设冷却，限制再次进入的频率
    assert ex._regen_mode_active(SkillChainExecutor.REGEN_EXIT_HEALTH) is False
    assert ex._regen_cooldown == SkillChainExecutor.REGEN_EXIT_COOLDOWN_STEPS
    assert ex._regen_mode_active(2.0) is False, "冷却期内不得立刻再次进入回血"


def test_progress_latch_keeps_descend_path_through_small_health_dips():
    """推进门的滞回：一旦开始推进，血量小幅回落不再把控制权交回生存维护。

    旧实现是裸阈值 health>=6：地表刷怪使血量在 6 附近震荡时整条下楼路径被
    无限期关闭（实测"下楼→挨打→回地表恢复"占一局 53% 步数）。"""
    ex = SkillChainExecutor("native.reach_floor_3")
    assert ex._safe_to_progress(_fake_summary(health=6.0)) is True
    assert ex._safe_to_progress(_fake_summary(health=5.0)) is True   # 小幅回落仍推进
    assert ex._safe_to_progress(_fake_summary(health=3.5)) is False  # 真正见底才交还
    # 交还后需重新达到进入门槛才再推进
    assert ex._safe_to_progress(_fake_summary(health=5.0)) is False
    assert ex._safe_to_progress(_fake_summary(health=6.0)) is True
    # 某项 vital 见底同样交还控制权（缺水/缺食由生存维护跨层处理）
    assert ex._safe_to_progress(_fake_summary(health=9.0, drink=0.0)) is False


def test_arrow_feedstock_carries_stone_as_well_as_wood():
    """MAKE_ARROW = 1 木 + 1 石 → 只带木等于深层一支箭也做不出。

    实测三次下楼时背包都是 8 木 + 0 石，弓在 L1 打空后彻底失去补给能力。"""
    ex = SkillChainExecutor("native.reach_floor_3")   # 需穿过 L1、L2
    summ = _fake_summary(floor=0, inventory={"bow": 1, "arrows": 30, "wood": 0})
    crafts = ex._arrow_feedstock_crafts(1, summ)
    wood = ex._arrow_feedstock_target(1, summ)
    stone = ex._arrow_feedstock_stone_target(1, summ)
    cap = SkillChainExecutor.ARROW_FEEDSTOCK_CAP
    assert crafts > 0 and wood > 0 and stone > 0
    assert stone == min(cap, crafts)
    assert wood == min(cap, crafts + 2)     # 木多留 2 个给深层放工作台
    assert stone <= wood
    # 下一层就是最深层（不再有过路层）→ 两种料都不必带
    ex_shallow = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex_shallow._arrow_feedstock_target(2, summ) == 0
    assert ex_shallow._arrow_feedstock_stone_target(2, summ) == 0


def test_descend_requires_arrow_reserve_before_crossing_another_floor():
    """弹药门：还要穿过下一层继续下潜时，不许带着 2 支箭下楼。

    实测两局都是带 2 支箭下 L1、箭尽后被近战打死（其中一局回地表补 2 支又下去，
    仍死在 L1）。prep 的 arrows_min 是均值档（很小），2 支就能满足 → 旧路径直接
    放行；这道门把下限提到 BOW_ARROW_RESERVE。"""
    from craftax.planner.executor import (
        BOW_ARROW_RESERVE,
        CRAFT_TABLE_BLOCK,
        MAKE_ARROW,
    )

    map2d = np.full((16, 16), 2, dtype=np.int32)
    map2d[12, 12] = CRAFT_TABLE_BLOCK       # 工作台就在玩家旁边
    payload = _fake_payload(map2d=map2d, ladder_down=(15, 15), monsters_killed=10)
    # L1 已清怪、装备阶梯已满足（剑 3/镐 2）→ 只差弹药
    inv = {"sword": 3, "pickaxe": 2, "bow": 1, "wood": 6, "stone": 6}
    summ = _fake_summary(floor=1, player_position=[12, 13],
                         inventory=dict(inv, arrows=2))
    ex = SkillChainExecutor("native.reach_floor_3")   # max_floor=3 → 还要穿过 L2
    assert ex._descend_to(payload, summ, 2) == MAKE_ARROW

    # 备到基础储备后不再为箭停留
    stocked = _fake_summary(floor=1, player_position=[12, 13],
                            inventory=dict(inv, arrows=BOW_ARROW_RESERVE))
    ex2 = SkillChainExecutor("native.reach_floor_3")
    assert ex2._descend_to(payload, stocked, 2) != MAKE_ARROW

    # 下一层就是最终目标层（只需到达，不清怪）→ 这道门不阻塞下楼
    ex3 = SkillChainExecutor("native.enter_gnomish_mines")   # max_floor=2
    assert ex3._descend_to(payload, summ, 2) != MAKE_ARROW

    # 逃逸：地牢层没有树，木料耗尽 → 造不出箭就必须放行，
    # 否则玩家会被一直推回工作台空转（旧版 635 步都在"合成箭"）。
    dry = _fake_summary(floor=1, player_position=[12, 13],
                        inventory=dict(inv, arrows=2, wood=0))
    ex4 = SkillChainExecutor("native.reach_floor_3")
    assert ex4._descend_to(payload, dry, 2) != MAKE_ARROW


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


# ---------------------------------------------------------------------------
# 掩体 / 庇护所 / 工具阶梯（2026-08-19）
# ---------------------------------------------------------------------------


def test_bow_shoots_along_line_beyond_point_blank():
    """回归：直线射击分支曾恒不触发。

    旧实现按 `DELTA_TO_ACTION.get((dx, dy))` 取朝向，而该表只有 4 个**单位**
    向量键 → dist>1 时恒为 None 并 continue，于是弓只在贴脸时才射。实测后果是
    带着 13-18 支箭被 4-5 格外的远程怪点死（每次掉血都发生在无近身怪时）。
    """
    from craftax.planner.executor import DOWN, SHOOT_ARROW

    map2d = np.full((48, 48), 2, dtype=np.int32)
    mobs = {
        "melee": {"positions": [], "masks": []},
        "ranged": {"positions": [[29, 24]], "masks": [True]},   # 同列，5 格外
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=map2d, mobs=mobs)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    # 已面向目标（direction=DOWN=4）→ 直接射
    summ = _fake_summary(player_direction=DOWN, inventory={"bow": 1, "arrows": 5})
    assert ex._bow_combat(payload, summ) == SHOOT_ARROW
    # 朝向不对 → 先转向（旧实现在这里返回 None，弓完全不动）
    summ_turned = _fake_summary(player_direction=1, inventory={"bow": 1, "arrows": 5})
    assert ex._bow_combat(payload, summ_turned) == DOWN


def test_bow_does_not_shoot_through_wall():
    """直线上有 solid 方块 → 不浪费箭（投射物会撞墙消失）。"""
    map2d = np.full((48, 48), 2, dtype=np.int32)
    map2d[26, 24] = 4  # STONE 挡在玩家与怪之间
    mobs = {
        "melee": {"positions": [], "masks": []},
        "ranged": {"positions": [[29, 24]], "masks": [True]},
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=map2d, mobs=mobs)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_direction=4, inventory={"bow": 1, "arrows": 5})
    from craftax.planner.executor import SHOOT_ARROW

    assert ex._bow_combat(payload, summ, chase="none") != SHOOT_ARROW


def _pocket_map(size: int = 48):
    """(24, 21) 处三面石墙的天然坑位，其余为草地。"""
    map2d = np.full((size, size), 2, dtype=np.int32)
    map2d[23, 21] = map2d[25, 21] = map2d[24, 20] = 4
    return map2d


def test_take_cover_walks_to_natural_pocket():
    from craftax.planner.executor import LEFT

    payload = _fake_payload(map2d=_pocket_map())
    summ = _fake_summary(player_position=[24, 24])
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex._take_cover(payload, summ) == LEFT   # 朝坑位走
    # 已在坑位里 → 不再移动
    assert ex._take_cover(payload, _fake_summary(player_position=[24, 21])) is None


def test_take_cover_digs_into_stone_when_no_pocket():
    """旷野无坑位但有石堆 + 木镐 → 挖一格造坑位（DO 面向石头）。"""
    from craftax.planner.executor import DO, DOWN

    map2d = np.full((48, 48), 2, dtype=np.int32)
    map2d[25:29, 22:26] = 4          # 石堆，玩家贴在它上边缘
    payload = _fake_payload(map2d=map2d)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_position=[24, 24], player_direction=DOWN,
                         inventory={"pickaxe": 1})
    assert ex._take_cover(payload, summ) == DO
    # 无镐则挖不动 → 不返回挖掘动作
    no_pick = _fake_summary(player_position=[24, 24], player_direction=DOWN,
                            inventory={"pickaxe": 0, "stone": 0})
    assert ex._take_cover(payload, no_pick) is None


def test_survival_takes_cover_under_ranged_pressure():
    """掉血却无近身怪 = 被投射物命中 → 先进掩体（墙会把箭吃掉），
    而不是继续在旷野推进（实测 L1 死因）。"""
    from craftax.planner.executor import LEFT

    payload = _fake_payload(map2d=_pocket_map())
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_position=[24, 24], health=9.0)
    # 无压制时不触发掩体逻辑
    assert ex._ranged_pressure() is False
    ex._ranged_hit_step = ex._steps
    assert ex._ranged_pressure() is True
    assert ex._survival_action(payload, summ) == LEFT


def test_ranged_pressure_ignores_starvation_damage():
    """饥/渴掉血恒 1 点，不能被误判成远程压制（否则断补给时会去躲墙角等死）。"""
    payload = _fake_payload()
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    ex._prev_health = 6.0
    ex._next_action_inner(payload, _fake_summary(health=5.0, drink=0.0))
    assert ex._ranged_pressure() is False
    ex2 = SkillChainExecutor("native.enter_gnomish_mines")
    ex2._prev_health = 9.0
    ex2._next_action_inner(payload, _fake_summary(health=6.0))   # -3 = 兽人法师的箭
    assert ex2._ranged_pressure() is True


def test_sleeps_in_shelter_when_no_safe_distance():
    """三面墙的坑位里可以睡：只剩一个开口，睡中最多被一只怪打醒。
    旧规则要求"距怪 >=14 或血 >=8"，低血 + 甩不掉怪时永远睡不着（能量归零死锁）。"""
    map2d = np.full((12, 12), 4, dtype=np.int32)     # 全是石头
    for cell in ((5, 5), (5, 6), (5, 7)):            # 一条 3 格死胡同
        map2d[cell] = 2
    mobs = {
        "melee": {"positions": [[5, 7]], "masks": [True]},
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=map2d, mobs=mobs)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_position=[5, 5], health=5.0, energy=2.0,
                         inventory={"bow": 0, "arrows": 0})
    assert ex._in_shelter(payload, summ) is True
    assert ex._survival_action(payload, summ) == SLEEP


def test_night_hold_is_bounded_and_conditional():
    """夜间驻守只在血/能量不满时进入，且有步数预算与冷却（否则天黑就躲会吃掉
    一局约 30% 的步数）。"""
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    night = dict(timestep=210)          # light_level ≈ 0（见 calculate_light_level）
    assert ex._is_night(_fake_summary(**night)) is True
    assert ex._is_night(_fake_summary(timestep=0)) is False
    # 满血满能量 → 不驻守
    assert ex._night_hold_active(_fake_summary(**night)) is False
    # 血不满 → 驻守，但用满预算后进入冷却
    low = _fake_summary(health=6.0, **night)
    assert ex._night_hold_active(low) is True
    for _ in range(ex.NIGHT_HOLD_MAX_STEPS + 1):
        ex._night_hold_active(low)
    assert ex._night_hold_active(low) is False       # 冷却中
    # 深层不做夜间驻守（地牢没有昼夜刷新差异）
    ex2 = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex2._night_hold_active(_fake_summary(floor=2, health=6.0, **night)) is False


def test_clearing_waits_inside_cover():
    """清怪时"等刷怪"要在坑位里等：怪 75% 概率朝玩家走、只在 10-14 格环上刷新，
    在旷野等 = 四面可能被贴脸且远程怪可自由射击。"""
    from craftax.planner.executor import LEFT

    payload = _fake_payload(map2d=_pocket_map(), monsters_killed=0)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_position=[24, 24], floor=1)
    assert ex._cover_or_wait(payload, summ, clearing=True) == LEFT
    # 已清层不需要驻守，回到原来的等待语义
    assert ex._cover_or_wait(payload, summ, clearing=False) in (SLEEP, DO)


def test_descend_prep_upgrades_pickaxe_even_with_bow():
    """回归：石镐曾挂在 `sword_target >= 3 or not has_bow` 下，而弹药充足时
    择优给 sword_target=0 → 木镐锁死 → 采不到铁 → 铁装链整条断掉
    （实测 8/8 局死亡时 pickaxe 恒为 1）。"""
    from craftax.planner.executor import CRAFT_TABLE_BLOCK, MAKE_STONE_PICKAXE

    map2d = np.full((16, 16), 2, dtype=np.int32)
    map2d[12, 12] = CRAFT_TABLE_BLOCK
    payload = _fake_payload(map2d=map2d, ladder_down=(15, 15))
    summ = _fake_summary(floor=0, strength=1, player_position=[12, 13],
                         inventory={"sword": 2, "pickaxe": 1, "bow": 1,
                                    "arrows": 23, "wood": 4, "stone": 4})
    ex = SkillChainExecutor("native.enter_gnomish_mines")   # _max_floor=2 → 需石镐
    assert ex._clear_prep(1, payload, summ).sword_target == 0   # 弹药已备齐
    assert ex._descend_to(payload, summ, 2) == MAKE_STONE_PICKAXE


def test_readiness_gate_resolves_pickaxe():
    """就绪门给出 ("pickaxe", n) 时执行器逐级升镐（做上一级镐才能采下一级的料）。"""
    from craftax.planner.executor import CRAFT_TABLE_BLOCK, MAKE_STONE_PICKAXE

    map2d = np.full((16, 16), 2, dtype=np.int32)
    map2d[12, 12] = CRAFT_TABLE_BLOCK
    payload = _fake_payload(map2d=map2d)
    summ = _fake_summary(player_position=[12, 13],
                         inventory={"pickaxe": 1, "wood": 4, "stone": 4})
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    assert ex._resolve_gate([("pickaxe", 3)], payload, summ, 2) == MAKE_STONE_PICKAXE


def test_bow_engagement_range_depends_on_whether_kills_count():
    """已清层不做 14 格无差别点射：那里的击杀不计入任何门槛，而地表持续刷新，
    无差别点射会把整局变成"地表箭工厂"（实测：L0 击杀 17→29、箭耗到 0、
    一次没下楼、成就 18→17）。未清层则相反——每一杀都算下楼门槛，远距射杀
    既推进目标又免接战。"""
    from craftax.planner.executor import DOWN, SHOOT_ARROW

    map2d = np.full((48, 48), 2, dtype=np.int32)
    mobs = {
        "melee": {"positions": [[34, 24]], "masks": [True]},   # 同列 10 格外
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    summ = _fake_summary(player_direction=DOWN, inventory={"bow": 1, "arrows": 8})
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    # 已清层（L0 初始 monsters_killed=10）→ 10 格外的怪不值得花箭
    cleared = _fake_payload(map2d=map2d, mobs=mobs, monsters_killed=10)
    assert ex._survival_action(cleared, summ) != SHOOT_ARROW
    # 未清层 → 远距射杀（这才是弓的价值）
    uncleared = _fake_payload(map2d=map2d, mobs=mobs, monsters_killed=0)
    assert ex._survival_action(uncleared, summ) == SHOOT_ARROW


# ---------------------------------------------------------------------------
# 口渴衰减倍率（EnvParams.thirst_rate）
# ---------------------------------------------------------------------------


def test_thirst_rate_slows_water_decay_in_env():
    """thirst_rate 线性放缓掉水速度（原版 1.0 ≈ 21 步/点）。

    这是长程任务能否完成的前提：原版速率下满水 9 点只够约 190 步，2000 步的
    任务要被"去找水"打断十余次，实测死亡轨迹里"低血 + 缺水 + 夜间"最常见。
    """
    from craftax.craftax.constants import Action

    def drink_after(steps: int, thirst_rate: float) -> float:
        env = CraftaxSymbolicEnvNoAutoReset()
        params = EnvParams(thirst_rate=thirst_rate)
        state = env.reset(jax.random.PRNGKey(2026), params)[1]
        key = jax.random.PRNGKey(7)
        for _ in range(steps):
            key, sub = jax.random.split(key)
            _o, state, _r, _d, _i = env.step(sub, state, Action.NOOP.value, params)
        return float(_host(state).player_drink)

    vanilla = drink_after(120, 1.0)
    slowed = drink_after(120, 0.25)
    assert vanilla < 9.0                    # 原版 120 步已经掉了好几点
    assert slowed > vanilla                 # 放缓后掉得更少
    # 倍率是线性的：0.25 倍速下掉的点数应约为原版的 1/4（允许 ±1 的取整误差）
    assert abs((9.0 - slowed) - (9.0 - vanilla) / 4) <= 1.0


def test_energy_rate_slows_natural_decay_in_env():
    """具身 energy_rate=0.25 时，清醒精力自然消耗约为原版四分之一。"""
    from craftax.craftax.constants import Action

    def energy_after(steps: int, energy_rate: float) -> float:
        env = CraftaxSymbolicEnvNoAutoReset()
        params = EnvParams(energy_rate=energy_rate)
        state = env.reset(jax.random.PRNGKey(2026), params)[1]
        key = jax.random.PRNGKey(7)
        for _ in range(steps):
            key, sub = jax.random.split(key)
            _o, state, _r, _d, _i = env.step(sub, state, Action.NOOP.value, params)
        return float(_host(state).player_energy)

    vanilla = energy_after(120, 1.0)
    slowed = energy_after(120, 0.25)
    assert slowed > vanilla
    assert abs((9.0 - slowed) - (9.0 - vanilla) / 4) <= 1.0


def test_sleep_can_heal_in_a_safe_shelter_even_at_full_energy():
    """SLEEP 不再只为回能量，也可在安全掩体中快速回血。"""
    map2d = np.full((12, 12), 4, dtype=np.int32)
    for cell in ((5, 5), (5, 6), (5, 7)):
        map2d[cell] = 2
    payload = _fake_payload(map2d=map2d)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_position=[5, 5], health=5.0, energy=9.0)
    assert ex._wait_action(summ, payload) == SLEEP


def test_executor_projections_follow_session_thirst_rate():
    """执行器的睡眠投影必须用会话的 thirst_rate：否则把水调慢后仍按原版投影，
    会拒绝本来安全的睡眠（"渴着睡"的保护变成"永远不睡"）。"""
    ex_vanilla = SkillChainExecutor("native.enter_gnomish_mines", thirst_rate=1.0)
    ex_slow = SkillChainExecutor("native.enter_gnomish_mines", thirst_rate=0.25)
    # 能量 1 → 整段睡眠 (7-1)×11 = 66 步（dex1 的能量上限是 7）。原版睡眠掉水
    # 42 步/点 → 66 步正好把仅剩的 1 点水耗干（醒来即进入掉血状态）→ 不睡；
    # 0.25 倍速下 168 步/点 → 只掉 0.4 点，睡醒还有水 → 可以安全睡。
    summ = _fake_summary(energy=1.0, health=7.0, drink=1.0, food=9.0)
    assert ex_vanilla._sleep_is_safe(summ) is False
    assert ex_slow._sleep_is_safe(summ) is True


def test_never_sleeps_in_the_open_with_mobs_around():
    """回归：满血也不能在旷野睡。实测 seed 2011 夜里"血足→睡等刷怪"，僵尸走近
    一击 10→3（2 伤 ×3.5 睡眠倍率），253 步暴毙。要么距怪 >=14（怪会消失），
    要么身处三面墙坑位，才允许 SLEEP。"""
    map2d = np.full((48, 48), 2, dtype=np.int32)
    mobs = {
        "melee": {"positions": [[29, 24]], "masks": [True]},   # 5 格外，远小于 14
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=map2d, mobs=mobs)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(health=9.0)
    assert ex._wait_action(summ, payload) != SLEEP
    # 怪很远（>=14）→ 可以睡
    far = {**mobs, "melee": {"positions": [[44, 24]], "masks": [True]}}
    assert ex._wait_action(summ, _fake_payload(map2d=map2d, mobs=far)) == SLEEP
    # 坑位里即使怪在 5 格外也可以睡（只有一个开口）
    pocket = _pocket_map()
    assert ex._wait_action(
        _fake_summary(health=9.0, player_position=[24, 21]),
        _fake_payload(map2d=pocket, mobs=mobs),
    ) == SLEEP


def test_idle_does_not_mine_away_own_shelter():
    """在坑位里"原地待命"不能用 DO——DO 作用于朝向格，而走进坑位时正面朝墙，
    DO 会把墙挖掉，掩体当场作废。"""
    from craftax.planner.executor import UP

    pocket = _pocket_map()
    payload = _fake_payload(map2d=pocket)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    # 站在坑位里、面朝石墙 (23,21) → NOOP
    in_pocket = _fake_summary(player_position=[24, 21], player_direction=UP, health=5.0)
    assert ex._idle_action(payload, in_pocket) == NOOP
    # 旷野里朝向草地 → 仍然是 DO（等刷怪/被动回血的原语义）
    open_field = _fake_summary(player_position=[24, 30], player_direction=UP, health=5.0)
    assert ex._idle_action(payload, open_field) == DO


def test_stall_guard_breaks_repeated_idle_do_and_attacks_adjacent_mob():
    """等待动作也必须经过停滞看门狗，近身怪优先转向/攻击。"""
    map2d = np.full((48, 48), 2, dtype=np.int32)
    map2d[24, 25] = 3  # 面朝水时旧逻辑会无限返回 DO
    payload = _fake_payload(map2d=map2d)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(player_position=[24, 24], player_direction=RIGHT, health=9.0)
    ex._stall_steps = 4
    escaped = ex._guard_stall(DO, payload, summ)
    assert escaped in (LEFT, RIGHT, UP, DOWN)

    # A planner movement must not be rewritten as DO merely because water is
    # adjacent.  That was the shore starvation loop seen in the demo.
    ex._stall_steps = 2
    assert ex._guard_stall(UP, payload, summ) == UP

    mobs = {
        "melee": {"positions": [[25, 24]], "masks": [True]},
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    ex._stall_steps = 2
    attack = ex._guard_stall(
        DO, _fake_payload(map2d=map2d, mobs=mobs), summ
    )
    assert attack == DOWN  # 先转向南侧相邻的近战怪


def test_ambush_holds_the_pocket_instead_of_chasing():
    """清怪层 + 坑位 + 有弓 → 守住开口不追击：怪 75% 概率自己走过来，
    走出去追等于把"只有一个开口"的优势还回去（每次接战固定挨一次首击）。"""
    pocket = _pocket_map()
    # 怪在斜向：既不贴脸也不在直线上 —— 唯一的区别就是"追不追"
    mobs = {
        "melee": {"positions": [[21, 28]], "masks": [True]},
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=pocket, mobs=mobs, monsters_killed=0)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    # 坑位里（距怪 10 格）：守住开口
    summ = _fake_summary(player_position=[24, 21], floor=1,
                         inventory={"bow": 1, "arrows": 5})
    assert ex._combat_any(payload, summ) in (NOOP, DO, SLEEP)
    # 旷野里且怪在追击距离内（5 格）：照常迎上去（返回移动动作）
    summ_open = _fake_summary(player_position=[24, 30], floor=1,
                              inventory={"bow": 1, "arrows": 5})
    assert ex._combat_any(payload, summ_open) not in (NOOP, DO, SLEEP)


def test_low_energy_does_not_gamble_on_open_field_sleep():
    """能量将尽也不能在旷野睡：SLEEP 一按就锁到能量回满（60+ 步），14 格内的怪
    一定会走到，3.5x 一击就是 7 伤（实测 seed 2011 满血 10→3 随即被补刀）。
    宁可顶着低能量推进——能量归零只是每 16 步掉 1 血的慢性消耗，随时可自救。"""
    map2d = np.full((48, 48), 2, dtype=np.int32)
    mobs = {
        "melee": {"positions": [[30, 24]], "masks": [True]},   # 6 格外：够得着
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=map2d, mobs=mobs)
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    summ = _fake_summary(energy=2.0, health=10.0, inventory={"bow": 0, "arrows": 0})
    assert ex._survival_action(payload, summ) != SLEEP
    # 同样能量、但怪在 14 格外（会消失）→ 照常睡
    far = {**mobs, "melee": {"positions": [[44, 24]], "masks": [True]}}
    assert ex._survival_action(_fake_payload(map2d=map2d, mobs=far), summ) == SLEEP


def test_planner_strikes_at_sword_reach_instead_of_closing_in():
    """持剑时正前方两格的怪要**直接打**，不能走近再打。

    走到相邻格必然先吃怪的一次首击（怪刷新即冷却<=0），而两格外它打不到我们——
    这正是 game_logic.do_action 里 sword_reach 的全部价值。旧实现的
    DELTA_TO_ACTION 只有 4 个单位偏移，两格外的怪一律走近，等于把射程收益还回去。
    """
    from craftax.planner.executor import SWORD_REACH

    map2d = np.full((48, 48), 2, dtype=np.int32)   # 全草地：视线通透
    pos = [24, 24]
    two_ahead = [24, 24 + SWORD_REACH]             # 正前方（RIGHT）两格
    mobs = {
        "melee": {"positions": [two_ahead], "masks": [True]},
        "ranged": {"positions": [], "masks": []},
        "passive": {"positions": [], "masks": []},
    }
    payload = _fake_payload(map2d=map2d, mobs=mobs)
    armed = _fake_summary(player_position=pos, player_direction=RIGHT,
                          inventory={"sword": 1, "bow": 0, "arrows": 0})
    # 已朝向目标 → 直接 DO
    assert SkillChainExecutor._adjacent_hostile_action(payload, armed) == DO
    ex = SkillChainExecutor("native.enter_gnomish_mines")
    ex._mob_cells = set()
    assert ex._combat_any(payload, armed) == DO

    # 没朝向 → 先转身，而不是走过去
    facing_away = _fake_summary(player_position=pos, player_direction=LEFT,
                                inventory={"sword": 1, "bow": 0, "arrows": 0})
    assert SkillChainExecutor._adjacent_hostile_action(payload, facing_away) == RIGHT

    # 无剑 → 两格打不到，退回原来的"走近"行为
    unarmed = _fake_summary(player_position=pos, player_direction=RIGHT,
                            inventory={"sword": 0, "bow": 0, "arrows": 0})
    assert SkillChainExecutor._adjacent_hostile_action(payload, unarmed) is None

    # 中间格是实心方块 → 打不到（与游戏侧的视线门一致），不要白转身
    walled = np.array(map2d)
    walled[24, 25] = 4                              # STONE 挡在中间
    blocked_payload = _fake_payload(map2d=walled, mobs=mobs)
    assert SkillChainExecutor._adjacent_hostile_action(blocked_payload, armed) is None

    # 斜向两格不在射程内（游戏只判正前方第二格）
    diag = {**mobs, "melee": {"positions": [[25, 25]], "masks": [True]}}
    diag_payload = _fake_payload(map2d=map2d, mobs=diag)
    assert SkillChainExecutor._adjacent_hostile_action(diag_payload, armed) is None

    # 贴脸的怪优先于两格外的（这一回合它就会打到我们）
    both = {**mobs, "melee": {"positions": [two_ahead, [24, 23]], "masks": [True, True]}}
    both_payload = _fake_payload(map2d=map2d, mobs=both)
    assert SkillChainExecutor._adjacent_hostile_action(both_payload, armed) == LEFT


def test_sword_reach_needs_line_of_sight_in_env():
    """游戏侧：两格攻击必须要求中间格通透。

    attack_mob 只按位置精确相等匹配怪、不看地形，没有视线门时剑会穿墙命中——
    地牢里等于隔墙单方面打怪（怪的近战判定要求相邻），是个白给的无敌位。
    """
    from craftax.craftax.constants import Action, BlockType

    def hit_far_mob(*, wall_between: bool, sword: int) -> float:
        env = CraftaxSymbolicEnvNoAutoReset()
        params = EnvParams()
        state = env.reset(jax.random.PRNGKey(2026), params)[1]
        level = int(_host(state).player_level)
        pos = np.asarray(_host(state).player_position)
        target = (int(pos[0]), int(pos[1]) + 2)
        mid = (int(pos[0]), int(pos[1]) + 1)
        # 把一只近战怪放到正前方两格，中间格按需设成石头或路
        mobs = state.melee_mobs
        state = state.replace(
            player_direction=jax.numpy.asarray(Action.RIGHT.value, dtype=jax.numpy.int32),
            inventory=state.inventory.replace(
                sword=jax.numpy.asarray(sword, dtype=jax.numpy.int32)
            ),
            melee_mobs=mobs.replace(
                position=mobs.position.at[level, 0].set(
                    jax.numpy.asarray(target, dtype=jax.numpy.int32)
                ),
                health=mobs.health.at[level, 0].set(jax.numpy.asarray(20.0)),
                mask=mobs.mask.at[level, 0].set(True),
                type_id=mobs.type_id.at[level, 0].set(jax.numpy.asarray(0, dtype=jax.numpy.int32)),
            ),
            map=state.map.at[level, mid[0], mid[1]].set(
                BlockType.STONE.value if wall_between else BlockType.PATH.value
            ),
        )
        before = float(_host(state).melee_mobs.health[level, 0])
        _o, state, _r, _d, _i = env.step(
            jax.random.PRNGKey(7), state, Action.DO.value, params
        )
        return before - float(_host(state).melee_mobs.health[level, 0])

    assert hit_far_mob(wall_between=False, sword=1) > 0, "通透时两格应能命中"
    assert hit_far_mob(wall_between=True, sword=1) == 0, "隔墙不得命中"
    assert hit_far_mob(wall_between=False, sword=0) == 0, "无剑不得有两格射程"
