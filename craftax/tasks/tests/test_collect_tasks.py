"""collect_tasks 模块测试。

CPU JAX 首次 reset 会触发约 30-60s 编译，因此 EnvState 用 session 级 fixture
只生成一次。谓词正确性通过手工修改真实 EnvState 的 achievements 数组构造
"已达成" / "未达成" 两种场景。
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
from craftax.tasks.registry import get_task_adapter  # noqa: E402

# (task_id, version, achievement 名)
SINGLE_COLLECT_TASKS = [
    ("native.collect_stone", "1.0.0", "COLLECT_STONE"),
    ("native.collect_coal", "1.0.0", "COLLECT_COAL"),
    ("native.collect_iron", "1.0.0", "COLLECT_IRON"),
    ("native.collect_diamond", "1.0.0", "COLLECT_DIAMOND"),
    ("native.collect_sapphire", "1.0.0", "COLLECT_SAPPHIRE"),
    ("native.collect_ruby", "1.0.0", "COLLECT_RUBY"),
    ("native.collect_drink", "1.0.0", "COLLECT_DRINK"),
    ("native.collect_sapling", "1.0.0", "COLLECT_SAPLING"),
    ("native.eat_cow", "1.0.0", "EAT_COW"),
    ("native.eat_plant", "1.0.0", "EAT_PLANT"),
    ("native.eat_bat", "1.0.0", "EAT_BAT"),
    ("native.eat_snail", "1.0.0", "EAT_SNAIL"),
    ("native.drink_potion", "1.0.0", "DRINK_POTION"),
]

# (task_id, version)
COMBINATION_TASKS = [
    ("native.collect_ore", "1.0.0"),
    ("native.collect_all_gems", "1.0.0"),
    ("native.eat_food", "1.0.0"),
]

ALL_TASKS = [(tid, ver) for tid, ver, _ in SINGLE_COLLECT_TASKS] + COMBINATION_TASKS

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_RE = re.compile(r"[A-Za-z]")


@pytest.fixture(scope="session")
def env_state():
    """真实 EnvState（host 化 numpy 值）。只 reset 一次，不 step。"""
    env = CraftaxSymbolicEnvNoAutoReset()
    obs, state = env.reset(jax.random.PRNGKey(0), EnvParams())
    return jax.device_get(state)


@pytest.fixture(scope="session")
def info():
    return {"discount": 1.0}


def _with_achievements(env_state, names):
    """伪造 achievements 数组：指定成就置真，其余为假。"""
    arr = np.zeros(env_state.achievements.shape, dtype=bool)
    for name in names:
        arr[Achievement[name].value] = True
    return dataclasses.replace(env_state, achievements=arr)


@pytest.mark.parametrize("task_id,version", ALL_TASKS)
def test_evaluate_basic(env_state, info, task_id, version):
    """每个任务：adapter 可用、evaluate 不抛错、instruction 中英双语、progress 在 [0,1]。"""
    adapter = get_task_adapter(task_id, version)
    result: TaskEval = adapter.evaluate(env_state, info)
    assert isinstance(result, TaskEval)
    assert 0.0 <= result.progress <= 1.0
    assert isinstance(result.done, bool)
    assert result.instruction  # 非空
    assert _CJK_RE.search(result.instruction), "instruction 应包含中文"
    assert _ASCII_RE.search(result.instruction), "instruction 应包含英文"
    assert isinstance(result.event_tokens, list)


@pytest.mark.parametrize("task_id,version,achievement", SINGLE_COLLECT_TASKS)
def test_single_collect_achievement_truth(env_state, info, task_id, version, achievement):
    """单成就任务：达成 -> done/1.0 + 事件 token；未达成 -> not done/0.0。"""
    adapter = get_task_adapter(task_id, version)
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


def test_collect_ore_partial(env_state, info):
    """or 谓词组合任务：任一矿石即成功；progress 随达成种类数增长。"""
    adapter = get_task_adapter("native.collect_ore", "1.0.0")

    s = _with_achievements(env_state, ["COLLECT_COAL"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(1 / 5)
    assert "COLLECT_COAL" in result.event_tokens

    s = _with_achievements(env_state, ["COLLECT_COAL", "COLLECT_RUBY"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(2 / 5)

    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_collect_all_gems_partial(env_state, info):
    """and 谓词组合任务：仅达成一种不算成功；两种都达成才 done。"""
    adapter = get_task_adapter("native.collect_all_gems", "1.0.0")

    s = _with_achievements(env_state, ["COLLECT_SAPPHIRE"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(0.5)
    assert "COLLECT_SAPPHIRE" in result.event_tokens

    s = _with_achievements(env_state, ["COLLECT_SAPPHIRE", "COLLECT_RUBY"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0
    assert "COLLECT_SAPPHIRE" in result.event_tokens
    assert "COLLECT_RUBY" in result.event_tokens

    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_eat_food_partial(env_state, info):
    """or 谓词组合任务：任一食物即成功；progress 随已吃种类数增长。"""
    adapter = get_task_adapter("native.eat_food", "1.0.0")

    s = _with_achievements(env_state, ["EAT_COW"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(1 / 4)
    assert "EAT_COW" in result.event_tokens

    s = _with_achievements(env_state, ["EAT_COW", "EAT_SNAIL"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(2 / 4)

    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0
