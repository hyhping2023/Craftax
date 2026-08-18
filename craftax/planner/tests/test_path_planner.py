"""craftax/planner 确定性路径规划测试。

- BFS 寻路：无障碍直线、绕障、堵死返回 None；
- 收集类任务映射与缺镐 fallback；
- 真实世界：在地表能找树/石头，collect_wood 可规划到可执行动作；
- 集成冒烟：planner 驱动真实 env 在有限步内达成 collect_wood。

CPU JAX 首次 reset 编译较慢，EnvState fixture 只生成一次。
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.planner.path_planner import (  # noqa: E402
    DELTA_TO_ACTION,
    PICKAXE_REQUIRED,
    blocked,
    find_nearest_target,
    is_collect_task,
    next_action,
    plan_path,
)

GRASS, WATER, STONE, TREE, COAL, IRON, WALL = 2, 3, 4, 5, 8, 9, 17


@pytest.fixture(scope="module")
def real_world():
    env = CraftaxSymbolicEnvNoAutoReset()
    obs, state = env.reset(jax.random.PRNGKey(2026), EnvParams())
    return env, jax.device_get(state)


# ---------------------------------------------------------------------------
# BFS 寻路（纯 numpy 布局）
# ---------------------------------------------------------------------------


def _grid():
    g = np.full((10, 10), GRASS, dtype=np.int32)
    return g


def test_find_target_adjacent():
    g = _grid()
    g[4, 5] = TREE  # 起点旁
    result = find_nearest_target(g, (4, 4), [TREE])
    assert result is not None
    target, first_delta = result
    assert target == (4, 5)
    assert first_delta in DELTA_TO_ACTION  # 可映射到动作


def test_find_target_straight_line():
    g = _grid()
    g[4, 8] = TREE
    result = find_nearest_target(g, (4, 4), [TREE])
    assert result is not None
    target, first_delta = result
    assert target == (4, 8)
    assert first_delta == (0, 1)  # 第一步向右


def test_find_target_around_obstacle():
    g = _grid()
    g[4, 5] = WALL
    g[5, 6] = TREE  # 石头挡路，需绕行
    result = find_nearest_target(g, (4, 4), [TREE])
    assert result is not None
    target, first_delta = result
    assert target == (5, 6)
    # 第一步不能是向右（撞墙），BFS 会走其他方向
    assert first_delta != (0, 1)


def test_find_target_unreachable_returns_none():
    g = _grid()
    # 用 WALL 围住玩家
    g[3:6, 3] = WALL
    g[3:6, 5] = WALL
    g[3, 3:6] = WALL
    g[5, 3:6] = WALL
    g[5, 5] = TREE  # 在墙内
    g[1, 1] = TREE  # 在墙外但被墙隔开（起点被围住）
    result = find_nearest_target(g, (4, 4), [TREE])
    assert result is None


def test_blocked_rules():
    assert blocked(TREE)
    assert blocked(STONE)
    assert blocked(WALL)
    assert blocked(WATER)  # 陆地生物不能进水面
    assert not blocked(GRASS)


# ---------------------------------------------------------------------------
# 收集类任务映射与缺镐 fallback
# ---------------------------------------------------------------------------


def test_collect_task_mapping():
    assert is_collect_task("native.collect_wood")
    assert is_collect_task("native.collect_diamond")
    assert not is_collect_task("native.survive")
    assert not is_collect_task("native.craft_wood_pickaxe")
    assert PICKAXE_REQUIRED["native.collect_wood"] == 0
    assert PICKAXE_REQUIRED["native.collect_iron"] == 2


def test_next_action_missing_pickaxe_returns_none():
    g = _grid()
    g[4, 5] = IRON
    # 无镐（pickaxe=0 < 2）→ 返回 None，交由调用方 fallback
    assert next_action(g, (4, 4), 2, 0, "native.collect_iron") is None


def test_next_action_with_pickaxe_returns_do():
    g = _grid()
    g[4, 5] = IRON
    # 站在铁旁、面向右、有石镐 → 直接 DO
    assert next_action(g, (4, 4), 2, 2, "native.collect_iron") == 5


def test_next_action_turns_then_do():
    g = _grid()
    g[4, 5] = STONE
    # 站在石头旁但面向下 → 先转向右（移动被 solid 挡、仅更新朝向）
    assert next_action(g, (4, 4), 4, 1, "native.collect_stone") == 2


# ---------------------------------------------------------------------------
# 真实世界
# ---------------------------------------------------------------------------


def test_real_world_finds_targets(real_world):
    _, hs = real_world
    level = int(hs.player_level)
    map2d = np.asarray(hs.map[level])
    pos = (int(hs.player_position[0]), int(hs.player_position[1]))
    assert map2d.shape == (48, 48)
    assert find_nearest_target(map2d, pos, [TREE]) is not None
    assert find_nearest_target(map2d, pos, [STONE]) is not None


def test_real_world_plan_path_collect_wood(real_world):
    """planner 在真实地图上能生成一条可行动作链（移动+转向+DO）。"""
    _, hs = real_world
    level = int(hs.player_level)
    map2d = np.asarray(hs.map[level])
    pos = (int(hs.player_position[0]), int(hs.player_position[1]))
    actions = plan_path(map2d, pos, [TREE])
    assert actions is not None
    assert actions[-1] == 5  # 末尾是 DO
    assert all(a in (1, 2, 3, 4, 5) for a in actions)


# ---------------------------------------------------------------------------
# 集成冒烟：planner 驱动真实 env 达成 collect_wood
# ---------------------------------------------------------------------------


def test_planner_drives_env_to_collect_wood(real_world):
    env, hs = real_world
    state = env.reset(jax.random.PRNGKey(2026), EnvParams())[1]
    key_rng = jax.random.PRNGKey(1)
    wood_before = int(np.asarray(hs.inventory.wood).item())
    for _ in range(300):
        s = jax.device_get(state)
        level = int(s.player_level)
        map2d = np.asarray(s.map[level])
        pos = (int(s.player_position[0]), int(s.player_position[1]))
        action = next_action(
            map2d, pos, int(s.player_direction),
            int(s.inventory.pickaxe), "native.collect_wood",
        )
        if action is None:
            break
        key_rng, k2 = jax.random.split(key_rng)
        obs, state, reward, done, info = env.step(k2, state, action, EnvParams())
    s = jax.device_get(state)
    wood_after = int(np.asarray(s.inventory.wood).item())
    assert wood_after > wood_before


# ---------------------------------------------------------------------------
# 技能链：挖煤（collect_coal）
# ---------------------------------------------------------------------------


def test_skill_chain_task_mapping():
    from craftax.planner.path_planner import (
        SKILL_CHAIN_TASKS,
        is_skill_chain_task,
    )

    assert is_skill_chain_task("native.collect_coal")
    assert not is_skill_chain_task("native.collect_wood")
    assert isinstance(SKILL_CHAIN_TASKS["native.collect_coal"], type)


def test_skill_chain_phase_switch_no_pickaxe_no_wood(real_world):
    """无镐且 wood<3 → 返回采木动作（在真实地图上应可规划）。"""
    from craftax.planner.path_planner import SkillChainPlanner

    _, hs = real_world
    level = int(hs.player_level)
    map2d = np.asarray(hs.map[level])
    pos = (int(hs.player_position[0]), int(hs.player_position[1]))
    planner = SkillChainPlanner("native.collect_coal")
    action = planner.next_action(
        map2d, pos, int(hs.player_direction),
        {"wood": 0, "pickaxe": 0, "coal": 0},
        [],
    )
    assert action in (1, 2, 3, 4, 5)


def test_skill_chain_phase_place_table(real_world):
    """wood=3 且无工作台 → 应返回朝向可放置格的移动或 PLACE_TABLE。"""
    from craftax.planner.path_planner import (
        PLACE_TABLE,
        SkillChainPlanner,
    )

    _, hs = real_world
    level = int(hs.player_level)
    map2d = np.asarray(hs.map[level])
    pos = (int(hs.player_position[0]), int(hs.player_position[1]))
    planner = SkillChainPlanner("native.collect_coal")
    action = planner.next_action(
        map2d, pos, int(hs.player_direction),
        {"wood": 3, "pickaxe": 0, "coal": 0},
        [],
    )
    assert action in (1, 2, 3, 4, PLACE_TABLE)


def test_skill_chain_phase_craft_pickaxe():
    """有工作台成就 + wood>=1 → 走向工作台或 MAKE_WOOD_PICKAXE（构造地图）。"""
    from craftax.planner.path_planner import (
        MAKE_WOOD_PICKAXE,
        SkillChainPlanner,
    )

    g = _grid()
    g[4, 5] = 11  # CRAFTING_TABLE 在起点右边
    planner = SkillChainPlanner("native.collect_coal")
    # 玩家在 (4,4) 面向右，工作台在 (4,5) 相邻 → 直接 MAKE_WOOD_PICKAXE
    action = planner.next_action(
        g, (4, 4), 2,
        {"wood": 1, "pickaxe": 0, "coal": 0},
        ["PLACE_TABLE"],
    )
    assert action == MAKE_WOOD_PICKAXE
    # 玩家在远处 → 返回移动动作走向工作台
    action2 = planner.next_action(
        g, (6, 6), 4,
        {"wood": 1, "pickaxe": 0, "coal": 0},
        ["PLACE_TABLE"],
    )
    assert action2 in (1, 2, 3, 4)


def test_skill_chain_phase_mine_coal(real_world):
    """有木镐 → 挖煤阶段（目标 COAL，需要镐）。"""
    from craftax.planner.path_planner import SkillChainPlanner

    _, hs = real_world
    level = int(hs.player_level)
    map2d = np.asarray(hs.map[level])
    pos = (int(hs.player_position[0]), int(hs.player_position[1]))
    planner = SkillChainPlanner("native.collect_coal")
    action = planner.next_action(
        map2d, pos, int(hs.player_direction),
        {"wood": 0, "pickaxe": 1, "coal": 0},
        ["PLACE_TABLE", "MAKE_WOOD_PICKAXE"],
    )
    assert action in (1, 2, 3, 4, 5)


def test_skill_chain_is_done():
    from craftax.planner.path_planner import SkillChainPlanner

    planner = SkillChainPlanner("native.collect_coal")
    assert planner.is_done({"coal": 1}, [])
    assert planner.is_done({"coal": 0}, ["COLLECT_COAL"])
    assert not planner.is_done({"coal": 0}, [])


def test_skill_chain_drives_env_to_collect_coal():
    """集成冒烟：skill chain 驱动真实 env 完成挖煤（coal>0）。"""
    from craftax.craftax.constants import Achievement
    from craftax.planner.path_planner import SkillChainPlanner

    env = CraftaxSymbolicEnvNoAutoReset()
    state = env.reset(jax.random.PRNGKey(2026), EnvParams())[1]
    planner = SkillChainPlanner("native.collect_coal")
    key_rng = jax.random.PRNGKey(7)
    for _ in range(400):
        s = jax.device_get(state)
        level = int(s.player_level)
        map2d = np.asarray(s.map[level])
        pos = (int(s.player_position[0]), int(s.player_position[1]))
        inv = {
            "wood": int(s.inventory.wood),
            "coal": int(s.inventory.coal),
            "pickaxe": int(s.inventory.pickaxe),
        }
        achievements = [
            a.name for a in Achievement if bool(np.asarray(s.achievements)[int(a.value)])
        ]
        if planner.is_done(inv, achievements):
            break
        action = planner.next_action(
            map2d, pos, int(s.player_direction), inv, achievements
        )
        if action is None:
            break
        key_rng, k2 = jax.random.split(key_rng)
        obs, state, reward, done, info = env.step(k2, state, action, EnvParams())
    s = jax.device_get(state)
    assert int(np.asarray(s.inventory.coal).item()) > 0
