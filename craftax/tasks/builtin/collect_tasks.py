"""收集/食用/饮用类 builtin 任务（仅标注，不改变游戏规则）。

全部任务用原生 Achievement 成就谓词判定；组合任务（collect_ore / collect_all_gems /
eat_food）分别用 or / and 谓词组合多个成就。任务适配器继承 BaseTaskAdapter，progress
由 achievement_progress（已达成比例）提供连续进度。
"""
from __future__ import annotations

from typing import Any, Dict

from craftax.contracts import TaskSpec
from craftax.tasks.base import BaseTaskAdapter

# 版本：所有 builtin 任务首版统一 1.0.0
TASK_VERSION = "1.0.0"

# 单资源收集任务：成就名 -> (instruction, objective, dependencies)
# dependencies 为严格前置任务：采石/煤需木镐，铁需石镐，钻石需铁镐，宝石需钻石镐。
_COLLECT_ITEM_SPECS = {
    "COLLECT_STONE": (
        "Collect stone. / 收集石头。",
        "获取并拾取至少一块石头（COLLECT_STONE 成就）。",
        ["native.craft_wood_pickaxe"],  # 采石需木镐（pickaxe>=1）
    ),
    "COLLECT_COAL": (
        "Collect coal. / 收集煤炭。",
        "获取并拾取至少一块煤炭（COLLECT_COAL 成就）。",
        ["native.craft_wood_pickaxe"],  # 挖煤需木镐（pickaxe>=1）
    ),
    "COLLECT_IRON": (
        "Collect iron. / 收集铁矿石。",
        "获取并拾取至少一块铁矿石（COLLECT_IRON 成就）。",
        ["native.craft_stone_pickaxe"],  # 挖铁需石镐（pickaxe>=2）
    ),
    "COLLECT_DIAMOND": (
        "Collect diamond. / 收集钻石。",
        "获取并拾取至少一颗钻石（COLLECT_DIAMOND 成就）。",
        ["native.craft_iron_pickaxe"],  # 挖钻石需铁镐（pickaxe>=3）
    ),
    "COLLECT_SAPPHIRE": (
        "Collect sapphire. / 收集蓝宝石。",
        "获取并拾取至少一颗蓝宝石（COLLECT_SAPPHIRE 成就）。",
        ["native.craft_diamond_pickaxe", "native.enter_vault"],  # 深层矿脉需钻石镐
    ),
    "COLLECT_RUBY": (
        "Collect ruby. / 收集红宝石。",
        "获取并拾取至少一颗红宝石（COLLECT_RUBY 成就）。",
        ["native.craft_diamond_pickaxe", "native.enter_troll_mines"],  # 深层矿脉需钻石镐
    ),
    "COLLECT_DRINK": (
        "Collect drink. / 收集饮用水。",
        "获取饮用水以满足口渴（COLLECT_DRINK 成就）。",
        [],  # 水源旁 DO 即可
    ),
    "COLLECT_SAPLING": (
        "Collect a sapling. / 收集树苗。",
        "获取并拾取至少一棵树苗（COLLECT_SAPLING 成就）。",
        [],
    ),
    "EAT_COW": (
        "Eat beef. / 吃牛肉。",
        "食用一块牛肉（EAT_COW 成就达成）。",
        [],
    ),
    "EAT_PLANT": (
        "Eat a plant. / 吃一株植物。",
        "食用一株植物（EAT_PLANT 成就达成）。",
        [],
    ),
    "EAT_BAT": (
        "Eat a bat. / 吃一只蝙蝠。",
        "食用一只蝙蝠（EAT_BAT 成就达成）。",
        [],
    ),
    "EAT_SNAIL": (
        "Eat a snail. / 吃一只蜗牛。",
        "食用一只蜗牛（EAT_SNAIL 成就达成）。",
        [],
    ),
    "DRINK_POTION": (
        "Drink a potion. / 喝一瓶药水。",
        "饮用一瓶药水（DRINK_POTION 成就达成）。",
        [],
    ),
}

# 单资源任务的成就清单（按注册顺序）
SINGLE_COLLECT_ACHIEVEMENTS = [
    "COLLECT_STONE",
    "COLLECT_COAL",
    "COLLECT_IRON",
    "COLLECT_DIAMOND",
    "COLLECT_SAPPHIRE",
    "COLLECT_RUBY",
    "COLLECT_DRINK",
    "COLLECT_SAPLING",
    "EAT_COW",
    "EAT_PLANT",
    "EAT_BAT",
    "EAT_SNAIL",
    "DRINK_POTION",
]

# 组合任务：收集类矿石（collect_ore 用 or，任一达成即成功）
ORE_ACHIEVEMENTS = [
    "COLLECT_COAL",
    "COLLECT_IRON",
    "COLLECT_DIAMOND",
    "COLLECT_SAPPHIRE",
    "COLLECT_RUBY",
]

# 组合任务：两类宝石（collect_all_gems 用 and，全部达成才成功）
GEM_ACHIEVEMENTS = [
    "COLLECT_SAPPHIRE",
    "COLLECT_RUBY",
]

# 组合任务：食物类（eat_food 用 or，任一达成即成功）
FOOD_ACHIEVEMENTS = [
    "EAT_COW",
    "EAT_PLANT",
    "EAT_BAT",
    "EAT_SNAIL",
]


def _collect_item_spec(achievement: str) -> TaskSpec:
    """单成就收集任务 spec：task_id 由成就名派生（COLLECT_STONE -> native.collect_stone）。"""
    instruction, objective, deps = _COLLECT_ITEM_SPECS[achievement]
    return TaskSpec(
        task_id="native." + achievement.lower(),
        version=TASK_VERSION,
        instruction=instruction,
        objective=objective,
        success_predicate={"type": "achievement", "name": achievement},
        annotation_predicates=[{"type": "achievement", "name": achievement}],
        renderer_config={},
        dependencies=list(deps),
    )


class CollectItemTask(BaseTaskAdapter):
    """单资源收集任务：达成对应成就即成功，进度 0/1。"""

    def __init__(self, achievement: str) -> None:
        self.achievement = achievement
        super().__init__(_collect_item_spec(achievement))


def _collect_ore_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.collect_ore",
        version=TASK_VERSION,
        instruction="Collect any ore. / 收集任意一种矿石。",
        objective="收集煤/铁/钻石/蓝宝石/红宝石中任意一种矿石。",
        success_predicate={
            "type": "or",
            "predicates": [
                {"type": "achievement", "name": name} for name in ORE_ACHIEVEMENTS
            ],
        },
        annotation_predicates=[
            {"type": "achievement", "name": name} for name in ORE_ACHIEVEMENTS
        ],
        renderer_config={},
        dependencies=["native.craft_wood_pickaxe"],  # 任一矿石都需镐子
    )


class CollectOreTask(BaseTaskAdapter):
    """任意一种矿石即成功（or 谓词）；progress = 已收集矿石种类比例。"""

    def __init__(self) -> None:
        super().__init__(_collect_ore_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(ORE_ACHIEVEMENTS, state, info)


def _collect_all_gems_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.collect_all_gems",
        version=TASK_VERSION,
        instruction="Collect both gems. / 收集两种宝石。",
        objective="同时收集蓝宝石与红宝石。",
        success_predicate={
            "type": "and",
            "predicates": [
                {"type": "achievement", "name": name} for name in GEM_ACHIEVEMENTS
            ],
        },
        annotation_predicates=[
            {"type": "achievement", "name": name} for name in GEM_ACHIEVEMENTS
        ],
        renderer_config={},
        dependencies=[
            "native.collect_sapphire",
            "native.collect_ruby",
        ],
    )


class CollectAllGemsTask(BaseTaskAdapter):
    """蓝宝石与红宝石全部达成才成功（and 谓词）；progress = 已达成宝石比例。"""

    def __init__(self) -> None:
        super().__init__(_collect_all_gems_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(GEM_ACHIEVEMENTS, state, info)


def _eat_food_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.eat_food",
        version=TASK_VERSION,
        instruction="Eat some food. / 吃一些食物。",
        objective="食用牛/植物/蝙蝠/蜗牛中任意一种食物。",
        success_predicate={
            "type": "or",
            "predicates": [
                {"type": "achievement", "name": name} for name in FOOD_ACHIEVEMENTS
            ],
        },
        annotation_predicates=[
            {"type": "achievement", "name": name} for name in FOOD_ACHIEVEMENTS
        ],
        renderer_config={},
        dependencies=[],  # 任何食物均可，无严格前置
    )


class EatFoodTask(BaseTaskAdapter):
    """任意一种食物即成功（or 谓词）；progress = 已食用食物种类比例。"""

    def __init__(self) -> None:
        super().__init__(_eat_food_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(FOOD_ACHIEVEMENTS, state, info)


_BUILTIN_FACTORIES = [
    *[lambda name=name: CollectItemTask(name) for name in SINGLE_COLLECT_ACHIEVEMENTS],
    CollectOreTask,
    CollectAllGemsTask,
    EatFoodTask,
]


def register_builtin_tasks() -> None:
    from craftax.tasks.registry import register

    for factory in _BUILTIN_FACTORIES:
        adapter = factory()
        spec: TaskSpec = adapter.spec
        register(spec.task_id, spec.version, factory)


register_builtin_tasks()
