"""craftax/tasks/builtin/exploration_tasks 测试。

CPU JAX 首次 reset 会触发约 30-60s 编译，因此 EnvState 用 module 级 fixture 只生成一次。
"""
from __future__ import annotations

import dataclasses
import re

import jax
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from craftax.contracts import TaskEval  # noqa: E402
from craftax.craftax.constants import Achievement  # noqa: E402
from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.tasks.registry import get_task_adapter, list_task_ids  # noqa: E402

TASK_VERSION = "1.0.0"

EXPLORATION_TASKS = [
    ("native.enter_dungeon", TASK_VERSION),
    ("native.enter_gnomish_mines", TASK_VERSION),
    ("native.enter_sewers", TASK_VERSION),
    ("native.enter_vault", TASK_VERSION),
    ("native.enter_troll_mines", TASK_VERSION),
    ("native.enter_fire_realm", TASK_VERSION),
    ("native.enter_ice_realm", TASK_VERSION),
    ("native.enter_graveyard", TASK_VERSION),
    ("native.open_chest", TASK_VERSION),
    ("native.place_table", TASK_VERSION),
    ("native.place_furnace", TASK_VERSION),
    ("native.place_stone", TASK_VERSION),
    ("native.place_plant", TASK_VERSION),
    ("native.place_torch", TASK_VERSION),
    ("native.wake_up", TASK_VERSION),
    ("native.reach_floor_3", TASK_VERSION),
    ("native.reach_floor_5", TASK_VERSION),
    ("native.reach_boss_floor", TASK_VERSION),
    ("native.deep_explorer", TASK_VERSION),
]

SINGLE_ACHIEVEMENT_TASKS = [
    ("native.enter_dungeon", "ENTER_DUNGEON"),
    ("native.enter_gnomish_mines", "ENTER_GNOMISH_MINES"),
    ("native.enter_sewers", "ENTER_SEWERS"),
    ("native.enter_vault", "ENTER_VAULT"),
    ("native.enter_troll_mines", "ENTER_TROLL_MINES"),
    ("native.enter_fire_realm", "ENTER_FIRE_REALM"),
    ("native.enter_ice_realm", "ENTER_ICE_REALM"),
    ("native.enter_graveyard", "ENTER_GRAVEYARD"),
    ("native.open_chest", "OPEN_CHEST"),
    ("native.place_table", "PLACE_TABLE"),
    ("native.place_furnace", "PLACE_FURNACE"),
    ("native.place_stone", "PLACE_STONE"),
    ("native.place_plant", "PLACE_PLANT"),
    ("native.place_torch", "PLACE_TORCH"),
    ("native.wake_up", "WAKE_UP"),
]

_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_HAS_LATIN = re.compile(r"[A-Za-z]")


@pytest.fixture(scope="module")
def env_state():
    """真实 EnvState（host 化 numpy 值）。只 reset 一次，不 step。"""
    env = CraftaxSymbolicEnvNoAutoReset()
    obs, state = env.reset(jax.random.PRNGKey(0), EnvParams())
    return jax.device_get(state)


@pytest.fixture(scope="module")
def info():
    return {"discount": 1.0}


def _with_achievements(env_state, names):
    arr = np.zeros(env_state.achievements.shape, dtype=bool)
    for name in names:
        arr[Achievement[name].value] = True
    return dataclasses.replace(env_state, achievements=arr)


def test_all_exploration_tasks_registered():
    ids = list_task_ids()
    for task_id, _ in EXPLORATION_TASKS:
        assert task_id in ids


@pytest.mark.parametrize("task_id,version", EXPLORATION_TASKS)
def test_adapter_smoke(env_state, info, task_id, version):
    """适配器可用、evaluate 不抛错、instruction 中英双语、progress 在 [0,1]。"""
    adapter = get_task_adapter(task_id, version)
    result: TaskEval = adapter.evaluate(env_state, info)
    assert isinstance(result, TaskEval)
    assert 0.0 <= result.progress <= 1.0
    assert isinstance(result.done, bool)
    assert " / " in result.instruction
    assert _HAS_LATIN.search(result.instruction)
    assert _HAS_CJK.search(result.instruction)
    assert isinstance(result.event_tokens, list)


@pytest.mark.parametrize("task_id,achievement", SINGLE_ACHIEVEMENT_TASKS)
def test_single_achievement_truth(env_state, info, task_id, achievement):
    """手工改 achievements 数组：已达成 -> done/progress=1，未达成 -> 0/0。"""
    adapter = get_task_adapter(task_id, TASK_VERSION)
    s_true = _with_achievements(env_state, [achievement])
    result = adapter.evaluate(s_true, info)
    assert result.done is True
    assert result.progress == 1.0
    assert achievement in result.event_tokens
    s_false = _with_achievements(env_state, [])
    result = adapter.evaluate(s_false, info)
    assert result.done is False
    assert result.progress == 0.0
    assert achievement not in result.event_tokens


@pytest.mark.parametrize(
    "task_id,achieved_at,level_below",
    [
        ("native.reach_floor_3", 3, 2),
        ("native.reach_floor_5", 5, 4),
        ("native.reach_boss_floor", 8, 7),
    ],
)
def test_reach_floor_boundary(env_state, info, task_id, achieved_at, level_below):
    """level_ge 边界：目标层前未达成，达到目标层即达成；progress = level/8。"""
    adapter = get_task_adapter(task_id, TASK_VERSION)
    s_below = dataclasses.replace(env_state, player_level=np.asarray(level_below))
    result = adapter.evaluate(s_below, info)
    assert result.done is False
    assert result.progress == pytest.approx(level_below / 8)
    s_at = dataclasses.replace(env_state, player_level=np.asarray(achieved_at))
    result = adapter.evaluate(s_at, info)
    assert result.done is True
    assert result.progress == pytest.approx(achieved_at / 8)


def test_reach_floor_progress_capped(env_state, info):
    adapter = get_task_adapter("native.reach_boss_floor", TASK_VERSION)
    s = dataclasses.replace(env_state, player_level=np.asarray(12))
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0


def test_deep_explorer_and_predicate(env_state, info):
    """and 嵌套：三个深层区域成就全部达成才成功，progress 按比例推进。"""
    adapter = get_task_adapter("native.deep_explorer", TASK_VERSION)
    s_one = _with_achievements(env_state, ["ENTER_SEWERS"])
    result = adapter.evaluate(s_one, info)
    assert result.done is False
    assert result.progress == pytest.approx(1 / 3)
    s_two = _with_achievements(env_state, ["ENTER_SEWERS", "ENTER_VAULT"])
    result = adapter.evaluate(s_two, info)
    assert result.done is False
    assert result.progress == pytest.approx(2 / 3)
    s_all = _with_achievements(
        env_state, ["ENTER_SEWERS", "ENTER_VAULT", "ENTER_GRAVEYARD"]
    )
    result = adapter.evaluate(s_all, info)
    assert result.done is True
    assert result.progress == 1.0
    assert result.event_tokens.count("ENTER_SEWERS") >= 1


def test_evaluate_is_readonly(env_state, info):
    before = np.array(env_state.achievements)
    adapter = get_task_adapter("native.enter_sewers", TASK_VERSION)
    adapter.evaluate(env_state, info)
    assert np.array_equal(env_state.achievements, before)
