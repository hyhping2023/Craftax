"""craftax/tasks 测试。

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
from craftax.tasks.base import eval_predicate  # noqa: E402
from craftax.tasks.registry import (  # noqa: E402
    get_task_adapter,
    list_task_ids,
    list_versions,
)

ALL_BUILTIN = [
    ("native.survive", "1.0.0"),
    ("native.collect_wood", "1.0.0"),
    ("native.craft_tools", "1.0.0"),
    ("native.defeat_enemy", "1.0.0"),
    ("native.explore_dungeon", "1.0.0"),
]


@pytest.fixture(scope="module")
def env_state():
    """真实 EnvState（host 化 numpy 值）。只 reset 一次，不 step。"""
    env = CraftaxSymbolicEnvNoAutoReset()
    obs, state = env.reset(jax.random.PRNGKey(0), EnvParams())
    return jax.device_get(state)


@pytest.fixture(scope="module")
def info():
    return {"discount": 1.0}


def test_registry_lists_all_builtin():
    ids = list_task_ids()
    for task_id, _ in ALL_BUILTIN:
        assert task_id in ids
    assert list_versions("native.survive") == ["1.0.0"]


def test_registry_version_mismatch_raises():
    with pytest.raises(ValueError, match="版本"):
        get_task_adapter("native.survive", "9.9.9")


def test_registry_unknown_task_raises():
    with pytest.raises(KeyError):
        get_task_adapter("no.such.task", "1.0.0")


@pytest.mark.parametrize("task_id,version", ALL_BUILTIN)
def test_builtin_evaluate_on_real_state(env_state, info, task_id, version):
    adapter = get_task_adapter(task_id, version)
    result: TaskEval = adapter.evaluate(env_state, info)
    assert isinstance(result, TaskEval)
    assert 0.0 <= result.progress <= 1.0
    assert isinstance(result.done, bool)
    assert result.instruction  # 非空（中英双语）
    assert isinstance(result.event_tokens, list)


def test_survive_progress_from_timestep(env_state, info):
    adapter = get_task_adapter("native.survive", "1.0.0")
    s = dataclasses.replace(env_state, timestep=np.asarray(50_000))
    assert adapter.evaluate(s, info).progress == pytest.approx(0.5)
    s = dataclasses.replace(env_state, timestep=np.asarray(200_000))
    assert adapter.evaluate(s, info).progress == 1.0  # 超上限被 clamp


def test_explore_dungeon_progress_from_level(env_state, info):
    adapter = get_task_adapter("native.explore_dungeon", "1.0.0")
    s = dataclasses.replace(env_state, player_level=np.asarray(4))
    result = adapter.evaluate(s, info)
    assert result.progress == pytest.approx(4 / 8)
    assert not result.done
    s = dataclasses.replace(env_state, player_level=np.asarray(8))
    result = adapter.evaluate(s, info)
    assert result.progress == 1.0
    assert result.done


def _with_achievements(env_state, names):
    arr = np.zeros(env_state.achievements.shape, dtype=bool)
    for name in names:
        arr[Achievement[name].value] = True
    return dataclasses.replace(env_state, achievements=arr)


def test_collect_wood_achievement_truth(env_state, info):
    adapter = get_task_adapter("native.collect_wood", "1.0.0")
    # 伪造 achievements：COLLECT_WOOD 为真 -> done，事件 token 出现一次
    s_true = _with_achievements(env_state, ["COLLECT_WOOD"])
    result = adapter.evaluate(s_true, info)
    assert result.done is True
    assert result.progress == 1.0
    assert "COLLECT_WOOD" in result.event_tokens
    # 未达成 -> not done，progress 0
    s_false = _with_achievements(env_state, [])
    result = adapter.evaluate(s_false, info)
    assert result.done is False
    assert result.progress == 0.0
    assert "COLLECT_WOOD" not in result.event_tokens


def test_craft_tools_partial_progress(env_state, info):
    adapter = get_task_adapter("native.craft_tools", "1.0.0")
    s = _with_achievements(env_state, ["MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE"])
    result = adapter.evaluate(s, info)
    assert result.progress == pytest.approx(2 / 4)
    assert result.done is True  # 任一镐即成功（or 谓词）
    s = _with_achievements(env_state, ["MAKE_IRON_PICKAXE"])
    assert adapter.evaluate(s, info).done is True
    # 一个都没做：progress 0，not done
    s = _with_achievements(env_state, [])
    result = adapter.evaluate(s, info)
    assert result.progress == 0.0
    assert result.done is False


def test_defeat_enemy_any(env_state, info):
    adapter = get_task_adapter("native.defeat_enemy", "1.0.0")
    s = _with_achievements(env_state, ["DEFEAT_SKELETON"])
    assert adapter.evaluate(s, info).done is True
    s = _with_achievements(env_state, ["DEFEAT_NECROMANCER"])
    assert adapter.evaluate(s, info).done is True


def test_predicate_field_ge(env_state):
    assert eval_predicate({"type": "field_ge", "path": "player_health", "value": 5}, env_state, {})
    assert not eval_predicate(
        {"type": "field_ge", "path": "player_health", "value": 999}, env_state, {}
    )
    assert eval_predicate({"type": "field_ge", "path": "inventory.wood", "value": 0}, env_state, {})
    assert not eval_predicate({"type": "field_gt", "path": "inventory.wood", "value": 1}, env_state, {})


def test_predicate_combinators(env_state):
    always = {"type": "always"}
    never = {"type": "never"}
    assert eval_predicate({"type": "and", "predicates": [always, always]}, env_state, {})
    assert not eval_predicate({"type": "and", "predicates": [always, never]}, env_state, {})
    assert eval_predicate({"type": "or", "predicates": [never, always]}, env_state, {})
    assert not eval_predicate({"type": "not", "predicate": always}, env_state, {})


def test_predicate_info_achievements_list(env_state):
    # info 提供 achievements_list 时同样生效
    info = {"achievements_list": ["COLLECT_WOOD"]}
    assert eval_predicate({"type": "achievement", "name": "COLLECT_WOOD"}, env_state, info)
    assert not eval_predicate({"type": "achievement", "name": "DEFEAT_ZOMBIE"}, env_state, info)


def test_evaluate_is_readonly(env_state, info):
    """TaskAdapter.evaluate 不得修改 state。"""
    before = np.array(env_state.achievements)
    adapter = get_task_adapter("native.collect_wood", "1.0.0")
    adapter.evaluate(env_state, info)
    assert np.array_equal(env_state.achievements, before)
    # 谓词求值也不修改
    eval_predicate({"type": "field_ge", "path": "player_health", "value": 0}, env_state, info)
    assert np.array_equal(env_state.achievements, before)
