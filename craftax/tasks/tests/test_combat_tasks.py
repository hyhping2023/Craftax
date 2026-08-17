"""combat_tasks 模块测试。

CPU JAX 首次 reset 会触发约 30-60s 编译，因此 EnvState 用模块级 fixture 只生成一次。
谓词正确性通过手工修改真实 EnvState 的 achievements 数组构造"已达成"/"未达成"场景。
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

# 单敌人击杀任务：(task_id, version, achievement 名)
SINGLE_DEFEAT_TASKS = [
    ("native.defeat_zombie", "1.0.0", "DEFEAT_ZOMBIE"),
    ("native.defeat_skeleton", "1.0.0", "DEFEAT_SKELETON"),
    ("native.defeat_gnome_warrior", "1.0.0", "DEFEAT_GNOME_WARRIOR"),
    ("native.defeat_gnome_archer", "1.0.0", "DEFEAT_GNOME_ARCHER"),
    ("native.defeat_orc_soldier", "1.0.0", "DEFEAT_ORC_SOLIDER"),
    ("native.defeat_orc_mage", "1.0.0", "DEFEAT_ORC_MAGE"),
    ("native.defeat_troll", "1.0.0", "DEFEAT_TROLL"),
    ("native.defeat_kobold", "1.0.0", "DEFEAT_KOBOLD"),
    ("native.defeat_necromancer", "1.0.0", "DEFEAT_NECROMANCER"),
    ("native.defeat_knight", "1.0.0", "DEFEAT_KNIGHT"),
    ("native.defeat_archer", "1.0.0", "DEFEAT_ARCHER"),
    ("native.damage_necromancer", "1.0.0", "DAMAGE_NECROMANCER"),
]

# 组合击杀任务：(task_id, version)
COMBINATION_TASKS = [
    ("native.defeat_elemental", "1.0.0"),
    ("native.defeat_three_enemies", "1.0.0"),
    ("native.defeat_undead", "1.0.0"),
]

ALL_TASKS = [(tid, ver) for tid, ver, _ in SINGLE_DEFEAT_TASKS] + COMBINATION_TASKS

# 组合任务背后的成就集合
ELEMENTAL_ACHIEVEMENTS = [
    "DEFEAT_FIRE_ELEMENTAL",
    "DEFEAT_FROST_TROLL",
    "DEFEAT_ICE_ELEMENTAL",
]
UNDEAD_ACHIEVEMENTS = ["DEFEAT_ZOMBIE", "DEFEAT_SKELETON", "DEFEAT_NECROMANCER"]

# VLA 指令规范：英文部分必须动词开头（Defeat / Damage），禁止 and/or/then 复合句
_EN_VERB_RE = re.compile(r"^\s*(?:Defeat|Damage)\b")
_EN_CONJUNCTION_RE = re.compile(r"\b(and|or|then)\b", re.IGNORECASE)


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
    """伪造 achievements 数组：指定成就置真，其余为假。"""
    arr = np.zeros(env_state.achievements.shape, dtype=bool)
    for name in names:
        arr[Achievement[name].value] = True
    return dataclasses.replace(env_state, achievements=arr)


def test_registry_contains_all_combat_tasks():
    ids = list_task_ids()
    for task_id, _ in ALL_TASKS:
        assert task_id in ids


@pytest.mark.parametrize("task_id,version", ALL_TASKS)
def test_evaluate_on_real_state(env_state, info, task_id, version):
    """每个任务：adapter 可用、evaluate 不抛错、progress 在 [0,1]。"""
    adapter = get_task_adapter(task_id, version)
    result: TaskEval = adapter.evaluate(env_state, info)
    assert isinstance(result, TaskEval)
    assert 0.0 <= result.progress <= 1.0
    assert isinstance(result.done, bool)
    assert result.instruction  # 非空（中英双语）
    assert isinstance(result.event_tokens, list)


@pytest.mark.parametrize("task_id,version", ALL_TASKS)
def test_instruction_is_vla_style(env_state, info, task_id, version):
    """VLA 指令规范：动词开头、单宾语短句、中英双语、无 and/or/then 复合句。"""
    adapter = get_task_adapter(task_id, version)
    instruction = adapter.instruction()
    assert "/" in instruction  # 中英分隔符
    assert any("\u4e00" <= ch <= "\u9fff" for ch in instruction)  # 含中文
    assert any(ch.isascii() and ch.isalpha() for ch in instruction)  # 含英文
    en_part = instruction.split("/", 1)[0]
    assert _EN_VERB_RE.search(en_part), f"英文指令应以 Defeat 开头: {instruction!r}"
    assert not _EN_CONJUNCTION_RE.search(en_part), (
        f"指令禁止 and/or/then 复合句: {instruction!r}"
    )


@pytest.mark.parametrize(
    "task_id,version,achievement", SINGLE_DEFEAT_TASKS
)
def test_single_defeat_achievement_truth(env_state, info, task_id, version, achievement):
    """单敌人任务：达成 -> done/1.0 + 事件 token；未达成 -> not done/0.0。"""
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


def test_defeat_elemental_logic(env_state, info):
    """or 谓词组合：任一元素击杀成就即成功；progress 随达成种类数增长。"""
    adapter = get_task_adapter("native.defeat_elemental", "1.0.0")

    s = _with_achievements(env_state, ["DEFEAT_FIRE_ELEMENTAL"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(1 / 3)
    assert "DEFEAT_FIRE_ELEMENTAL" in result.event_tokens

    s = _with_achievements(
        env_state, ["DEFEAT_FIRE_ELEMENTAL", "DEFEAT_ICE_ELEMENTAL"]
    )
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == pytest.approx(2 / 3)

    s = _with_achievements(env_state, ELEMENTAL_ACHIEVEMENTS)
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0

    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_defeat_three_enemies_logic(env_state, info):
    """任意三种敌人：不足 3 种不算成功；达 3 种才 done，progress = count/3。"""
    adapter = get_task_adapter("native.defeat_three_enemies", "1.0.0")

    s = _with_achievements(env_state, ["DEFEAT_ZOMBIE", "DEFEAT_SKELETON"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(2 / 3)

    s = _with_achievements(
        env_state, ["DEFEAT_ZOMBIE", "DEFEAT_SKELETON", "DEFEAT_TROLL"]
    )
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0
    assert "DEFEAT_ZOMBIE" in result.event_tokens
    assert "DEFEAT_TROLL" in result.event_tokens

    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_defeat_undead_logic(env_state, info):
    """亡灵任二：1 种不算成功；2 种才 done，progress = count/2。"""
    adapter = get_task_adapter("native.defeat_undead", "1.0.0")

    s = _with_achievements(env_state, ["DEFEAT_ZOMBIE"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(0.5)

    s = _with_achievements(env_state, ["DEFEAT_ZOMBIE", "DEFEAT_SKELETON"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0
    assert "DEFEAT_ZOMBIE" in result.event_tokens
    assert "DEFEAT_SKELETON" in result.event_tokens

    # 亡灵法师单独不算成功（需任二）
    s = _with_achievements(env_state, ["DEFEAT_NECROMANCER"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(0.5)

    s = _with_achievements(env_state, UNDEAD_ACHIEVEMENTS)
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0

    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0


def test_damage_necromancer_is_damage_not_kill(env_state, info):
    """DAMAGE_NECROMANCER 对应"伤害 Boss"，与击杀成就 DEFEAT_NECROMANCER 不同。"""
    adapter = get_task_adapter("native.damage_necromancer", "1.0.0")

    s = _with_achievements(env_state, ["DAMAGE_NECROMANCER"])
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0
    assert "DAMAGE_NECROMANCER" in result.event_tokens

    # 仅击杀亡灵法师不算伤害成就
    s = _with_achievements(env_state, ["DEFEAT_NECROMANCER"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == 0.0
    assert "DAMAGE_NECROMANCER" not in result.event_tokens


def test_defeat_three_enemies_includes_knight_archer(env_state, info):
    """骑士/弓箭手击杀成就计入 defeat_three_enemies 的组合池。"""
    adapter = get_task_adapter("native.defeat_three_enemies", "1.0.0")

    s = _with_achievements(
        env_state, ["DEFEAT_KNIGHT", "DEFEAT_ARCHER", "DEFEAT_ZOMBIE"]
    )
    result = adapter.evaluate(s, info)
    assert result.done is True
    assert result.progress == 1.0

    # 骑士 + 弓箭手不足 3 种
    s = _with_achievements(env_state, ["DEFEAT_KNIGHT", "DEFEAT_ARCHER"])
    result = adapter.evaluate(s, info)
    assert result.done is False
    assert result.progress == pytest.approx(2 / 3)


def test_evaluate_is_readonly(env_state, info):
    before = np.array(env_state.achievements)
    for task_id, _ in ALL_TASKS:
        get_task_adapter(task_id, "1.0.0").evaluate(env_state, info)
    assert np.array_equal(env_state.achievements, before)
