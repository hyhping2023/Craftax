"""首批 builtin 任务（仅标注，不改变游戏规则）。

任务适配器全部继承 BaseTaskAdapter；success/annotation 由 TaskSpec 的
可序列化谓词定义，progress 由子类提供。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from craftax.contracts import TaskSpec
from craftax.tasks.base import BaseTaskAdapter

# 版本：所有 builtin 任务首版统一 1.0.0
TASK_VERSION = "1.0.0"

# 环境默认最大步数（EnvParams.max_timesteps），survive 任务进度用。
DEFAULT_MAX_TIMESTEPS = 100_000
# 默认楼层数（StaticEnvParams.num_levels），explore_dungeon 用。
DEFAULT_NUM_LEVELS = 9


def _survive_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.survive",
        version=TASK_VERSION,
        instruction="Survive as long as possible. / 尽可能长久地存活下来。",
        objective="在不死亡的前提下存活到环境允许的最大步数。",
        success_predicate={"type": "always"},
        annotation_predicates=[
            {"type": "achievement", "name": "WAKE_UP"},
            {"type": "achievement", "name": "DEFEAT_ZOMBIE"},
        ],
        renderer_config={},
        dependencies=[],
    )


class SurviveTask(BaseTaskAdapter):
    """进度 = timestep / max_timesteps；成功条件由环境终局决定，故 success 恒真。"""

    def __init__(self) -> None:
        super().__init__(_survive_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        timestep = int(np.asarray(state.timestep))
        return self.clamp01(timestep / DEFAULT_MAX_TIMESTEPS)


def _collect_wood_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.collect_wood",
        version=TASK_VERSION,
        instruction="Collect wood. / 收集木材。",
        objective="获取并拾取至少一块木材（COLLECT_WOOD 成就）。",
        success_predicate={"type": "achievement", "name": "COLLECT_WOOD"},
        annotation_predicates=[{"type": "achievement", "name": "COLLECT_WOOD"}],
        renderer_config={},
        dependencies=[],  # 徒手砍树即可，无需工具
    )


class CollectWoodTask(BaseTaskAdapter):
    def __init__(self) -> None:
        super().__init__(_collect_wood_spec())


def _craft_tools_spec() -> TaskSpec:
    pickaxes = [
        {"type": "achievement", "name": "MAKE_WOOD_PICKAXE"},
        {"type": "achievement", "name": "MAKE_STONE_PICKAXE"},
        {"type": "achievement", "name": "MAKE_IRON_PICKAXE"},
    ]
    return TaskSpec(
        task_id="native.craft_tools",
        version=TASK_VERSION,
        instruction="Craft a pickaxe. / 制作一把镐。",
        objective="制作任意一种镐（木/石/铁镐任一成就达成）。",
        success_predicate={
            "type": "or",
            "predicates": [
                {"type": "achievement", "name": "MAKE_WOOD_PICKAXE"},
                {"type": "achievement", "name": "MAKE_STONE_PICKAXE"},
                {"type": "achievement", "name": "MAKE_IRON_PICKAXE"},
                {"type": "achievement", "name": "MAKE_DIAMOND_PICKAXE"},
            ],
        },
        annotation_predicates=pickaxes,
        renderer_config={},
        dependencies=[
            "native.collect_wood",
            "native.place_table",  # 合成需在工作台旁
            "native.craft_wood_pickaxe",  # 成功谓词要求任一镐：木镐是最低成本达成路径
        ],
    )


class CraftToolsTask(BaseTaskAdapter):
    _PICKAXE_ACHIEVEMENTS = [
        "MAKE_WOOD_PICKAXE",
        "MAKE_STONE_PICKAXE",
        "MAKE_IRON_PICKAXE",
        "MAKE_DIAMOND_PICKAXE",
    ]

    def __init__(self) -> None:
        super().__init__(_craft_tools_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(self._PICKAXE_ACHIEVEMENTS, state, info)


def _defeat_enemy_spec() -> TaskSpec:
    enemies = [
        "DEFEAT_ZOMBIE",
        "DEFEAT_SKELETON",
        "DEFEAT_GNOME_WARRIOR",
        "DEFEAT_GNOME_ARCHER",
        "DEFEAT_ORC_SOLIDER",
        "DEFEAT_ORC_MAGE",
        "DEFEAT_LIZARD",
        "DEFEAT_KOBOLD",
        "DEFEAT_TROLL",
        "DEFEAT_DEEP_THING",
        "DEFEAT_PIGMAN",
        "DEFEAT_FIRE_ELEMENTAL",
        "DEFEAT_FROST_TROLL",
        "DEFEAT_ICE_ELEMENTAL",
        "DEFEAT_NECROMANCER",
    ]
    return TaskSpec(
        task_id="native.defeat_enemy",
        version=TASK_VERSION,
        instruction="Defeat an enemy. / 击败一个敌人。",
        objective="击败任意敌人（僵尸/骷髅等任一击杀成就达成）。",
        success_predicate={
            "type": "or",
            "predicates": [{"type": "achievement", "name": name} for name in enemies],
        },
        annotation_predicates=[{"type": "achievement", "name": name} for name in enemies],
        renderer_config={},
        dependencies=["native.craft_wood_sword"],  # 战斗需要近战武器
    )


class DefeatEnemyTask(BaseTaskAdapter):
    _ENEMY_ACHIEVEMENTS = [
        "DEFEAT_ZOMBIE",
        "DEFEAT_SKELETON",
        "DEFEAT_GNOME_WARRIOR",
        "DEFEAT_GNOME_ARCHER",
        "DEFEAT_ORC_SOLIDER",
        "DEFEAT_ORC_MAGE",
        "DEFEAT_LIZARD",
        "DEFEAT_KOBOLD",
        "DEFEAT_TROLL",
        "DEFEAT_DEEP_THING",
        "DEFEAT_PIGMAN",
        "DEFEAT_FIRE_ELEMENTAL",
        "DEFEAT_FROST_TROLL",
        "DEFEAT_ICE_ELEMENTAL",
        "DEFEAT_NECROMANCER",
    ]

    def __init__(self) -> None:
        super().__init__(_defeat_enemy_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(self._ENEMY_ACHIEVEMENTS, state, info)


def _explore_dungeon_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.explore_dungeon",
        version=TASK_VERSION,
        instruction="Explore the dungeon. / 探索地下城。",
        objective="沿楼层一路向下推进，最终到达 Boss 层（player_level=8）。",
        success_predicate={"type": "level_ge", "value": DEFAULT_NUM_LEVELS - 1},
        annotation_predicates=[
            {"type": "achievement", "name": "ENTER_DUNGEON"},
            {"type": "achievement", "name": "ENTER_GNOMISH_MINES"},
            {"type": "achievement", "name": "ENTER_SEWERS"},
            {"type": "achievement", "name": "ENTER_VAULT"},
            {"type": "achievement", "name": "ENTER_TROLL_MINES"},
            {"type": "achievement", "name": "ENTER_FIRE_REALM"},
            {"type": "achievement", "name": "ENTER_ICE_REALM"},
            {"type": "achievement", "name": "ENTER_GRAVEYARD"},
        ],
        renderer_config={},
        dependencies=["native.enter_dungeon"],  # 需先进入地下城沿楼层下探
    )


class ExploreDungeonTask(BaseTaskAdapter):
    """进度 = player_level / (num_levels - 1)；level 8（Boss 层）即成功。"""

    def __init__(self) -> None:
        super().__init__(_explore_dungeon_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        level = int(np.asarray(state.player_level))
        return self.clamp01(level / (DEFAULT_NUM_LEVELS - 1))


_BUILTIN_FACTORIES = [
    SurviveTask,
    CollectWoodTask,
    CraftToolsTask,
    DefeatEnemyTask,
    ExploreDungeonTask,
]


def register_builtin_tasks() -> None:
    from craftax.tasks.registry import register

    for factory in _BUILTIN_FACTORIES:
        adapter = factory()
        spec: TaskSpec = adapter.spec
        register(spec.task_id, spec.version, factory)


register_builtin_tasks()
