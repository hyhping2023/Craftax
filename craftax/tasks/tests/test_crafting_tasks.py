"""craftax/tasks/builtin/crafting_tasks 测试。

CPU JAX 首次 reset 会触发约 30-60s 编译，因此 EnvState 用模块级 fixture 只生成一次。
"""
from __future__ import annotations

import dataclasses

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

ALL_CRAFTING = [
    ("native.craft_wood_pickaxe", "1.0.0"),
    ("native.craft_stone_pickaxe", "1.0.0"),
    ("native.craft_iron_pickaxe", "1.0.0"),
    ("native.craft_diamond_pickaxe", "1.0.0"),
    ("native.craft_wood_sword", "1.0.0"),
    ("native.craft_stone_sword", "1.0.0"),
    ("native.craft_iron_sword", "1.0.0"),
    ("native.craft_diamond_sword", "1.0.0"),
    ("native.craft_iron_armour", "1.0.0"),
    ("native.craft_diamond_armour", "1.0.0"),
    ("native.craft_arrow", "1.0.0"),
    ("native.craft_torch", "1.0.0"),
    ("native.enchant_sword", "1.0.0"),
    ("native.enchant_armour", "1.0.0"),
    ("native.learn_fireball", "1.0.0"),
    ("native.learn_iceball", "1.0.0"),
    ("native.find_bow", "1.0.0"),
    ("native.fire_bow", "1.0.0"),
    ("native.cast_fireball", "1.0.0"),
    ("native.cast_iceball", "1.0.0"),
    ("native.craft_full_kit", "1.0.0"),
    ("native.master_crafter", "1.0.0"),
]

ACHIEVEMENT_BY_TASK = {
    "native.craft_wood_pickaxe": "MAKE_WOOD_PICKAXE",
    "native.craft_stone_pickaxe": "MAKE_STONE_PICKAXE",
    "native.craft_iron_pickaxe": "MAKE_IRON_PICKAXE",
    "native.craft_diamond_pickaxe": "MAKE_DIAMOND_PICKAXE",
    "native.craft_wood_sword": "MAKE_WOOD_SWORD",
    "native.craft_stone_sword": "MAKE_STONE_SWORD",
    "native.craft_iron_sword": "MAKE_IRON_SWORD",
    "native.craft_diamond_sword": "MAKE_DIAMOND_SWORD",
    "native.craft_iron_armour": "MAKE_IRON_ARMOUR",
    "native.craft_diamond_armour": "MAKE_DIAMOND_ARMOUR",
    "native.craft_arrow": "MAKE_ARROW",
    "native.craft_torch": "MAKE_TORCH",
    "native.enchant_sword": "ENCHANT_SWORD",
    "native.enchant_armour": "ENCHANT_ARMOUR",
    "native.learn_fireball": "LEARN_FIREBALL",
    "native.learn_iceball": "LEARN_ICEBALL",
    "native.find_bow": "FIND_BOW",
    "native.fire_bow": "FIRE_BOW",
    "native.cast_fireball": "CAST_FIREBALL",
    "native.cast_iceball": "CAST_ICEBALL",
}

PICKAXES = [
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "MAKE_DIAMOND_PICKAXE",
]
SWORDS = [
    "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD",
    "MAKE_IRON_SWORD",
    "MAKE_DIAMOND_SWORD",
]
ARMOURS = ["MAKE_IRON_ARMOUR", "MAKE_DIAMOND_ARMOUR"]


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


def test_registry_contains_all_crafting_tasks():
    ids = list_task_ids()
    for task_id, _ in ALL_CRAFTING:
        assert task_id in ids


@pytest.mark.parametrize("task_id,version", ALL_CRAFTING)
def test_evaluate_on_real_state(env_state, info, task_id, version):
    adapter = get_task_adapter(task_id, version)
    result: TaskEval = adapter.evaluate(env_state, info)
    assert isinstance(result, TaskEval)
    assert 0.0 <= result.progress <= 1.0
    assert isinstance(result.done, bool)
    assert result.instruction  # 非空（中英双语）
    assert isinstance(result.event_tokens, list)


@pytest.mark.parametrize("task_id,version", ALL_CRAFTING)
def test_instruction_is_bilingual(env_state, info, task_id, version):
    adapter = get_task_adapter(task_id, version)
    instruction = adapter.instruction()
    assert "/" in instruction  # 中英分隔符
    assert any("\u4e00" <= ch <= "\u9fff" for ch in instruction)  # 含中文
    assert any(ch.isascii() and ch.isalpha() for ch in instruction)  # 含英文


@pytest.mark.parametrize(
    "task_id,achievement", sorted(ACHIEVEMENT_BY_TASK.items())
)
def test_single_achievement_truth(env_state, info, task_id, achievement):
    adapter = get_task_adapter(task_id, "1.0.0")
    # 成就达成 -> done，progress 1，事件 token 出现
    s_true = _with_achievements(env_state, [achievement])
    result = adapter.evaluate(s_true, info)
    assert result.done is True
    assert result.progress == 1.0
    assert achievement in result.event_tokens
    # 未达成 -> not done，progress 0，无 token
    s_false = _with_achievements(env_state, [])
    result = adapter.evaluate(s_false, info)
    assert result.done is False
    assert result.progress == 0.0
    assert achievement not in result.event_tokens


def test_craft_full_kit_and_or_logic(env_state, info):
    adapter = get_task_adapter("native.craft_full_kit", "1.0.0")
    total = len(PICKAXES) + len(SWORDS) + len(ARMOURS)
    # 只有一把镐：未完成，进度 1/total
    s = _with_achievements(env_state, ["MAKE_IRON_PICKAXE"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(1 / total)
    # 镐 + 剑：仍未完成（缺盔甲）
    s = _with_achievements(env_state, ["MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(2 / total)
    # 镐 + 剑 + 铁甲：完成
    s = _with_achievements(
        env_state, ["MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD", "MAKE_IRON_ARMOUR"]
    )
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(3 / total)
    # 钻石盔甲同样满足盔甲分支
    s = _with_achievements(
        env_state, ["MAKE_STONE_PICKAXE", "MAKE_DIAMOND_SWORD", "MAKE_DIAMOND_ARMOUR"]
    )
    assert adapter.evaluate(s, info).done is True
    # 空成就
    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_master_crafter_progress(env_state, info):
    adapter = get_task_adapter("native.master_crafter", "1.0.0")
    # 两把镐：进度 0.5，未完成
    s = _with_achievements(env_state, ["MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(0.5)
    # 三把镐：进度 0.75，仍未完成
    s = _with_achievements(
        env_state,
        ["MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE"],
    )
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(0.75)
    # 全部四把镐：完成
    s = _with_achievements(env_state, PICKAXES)
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0
    # 剑不计入镐进度
    s = _with_achievements(env_state, ["MAKE_WOOD_SWORD"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_evaluate_is_readonly(env_state, info):
    before = np.array(env_state.achievements)
    for task_id, _ in ALL_CRAFTING:
        get_task_adapter(task_id, "1.0.0").evaluate(env_state, info)
    assert np.array_equal(env_state.achievements, before)
