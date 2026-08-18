"""制作 / 附魔 / 魔法类任务（仅标注，不改变游戏规则）。

全部任务以原生 Craftax Achievement 成就为成功谓词：
- 单成就任务：默认 0/1 进度；
- 组合任务（craft_full_kit / master_crafter）：achievement_progress 连续进度。

任务适配器全部继承 BaseTaskAdapter；spec 为可序列化谓词。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from craftax.contracts import TaskSpec
from craftax.tasks.base import BaseTaskAdapter

# 版本：所有 builtin 任务首版统一 1.0.0
TASK_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# 单成就任务
# ---------------------------------------------------------------------------

# (task_id, achievement_name, instruction, objective, dependencies)
# 合成链严格依赖：木镐<-木+工作台；石镐<-木镐+石；铁镐<-石镐+铁+煤+熔炉+工作台；
# 钻石镐<-铁镐+钻石。剑/盔甲同理。
_SINGLE_TASK_DEFS: List[Tuple[str, str, str, str, List[str]]] = [
    # 镐
    (
        "native.craft_wood_pickaxe",
        "MAKE_WOOD_PICKAXE",
        "Craft a wooden pickaxe. / 制作一把木镐。",
        "制作一把木镐（MAKE_WOOD_PICKAXE 成就）。",
        ["native.collect_wood", "native.place_table"],
    ),
    (
        "native.craft_stone_pickaxe",
        "MAKE_STONE_PICKAXE",
        "Craft a stone pickaxe. / 制作一把石镐。",
        "制作一把石镐（MAKE_STONE_PICKAXE 成就）。",
        ["native.craft_wood_pickaxe", "native.collect_stone", "native.place_table"],
    ),
    (
        "native.craft_iron_pickaxe",
        "MAKE_IRON_PICKAXE",
        "Craft an iron pickaxe. / 制作一把铁镐。",
        "制作一把铁镐（MAKE_IRON_PICKAXE 成就）。",
        [
            "native.craft_stone_pickaxe",
            "native.collect_iron",
            "native.collect_coal",
            "native.place_furnace",
            "native.place_table",
        ],
    ),
    (
        "native.craft_diamond_pickaxe",
        "MAKE_DIAMOND_PICKAXE",
        "Craft a diamond pickaxe. / 制作一把钻石镐。",
        "制作一把钻石镐（MAKE_DIAMOND_PICKAXE 成就）。",
        ["native.craft_iron_pickaxe", "native.collect_diamond", "native.place_table"],
    ),
    # 剑
    (
        "native.craft_wood_sword",
        "MAKE_WOOD_SWORD",
        "Craft a wooden sword. / 制作一把木剑。",
        "制作一把木剑（MAKE_WOOD_SWORD 成就）。",
        ["native.collect_wood", "native.place_table"],
    ),
    (
        "native.craft_stone_sword",
        "MAKE_STONE_SWORD",
        "Craft a stone sword. / 制作一把石剑。",
        "制作一把石剑（MAKE_STONE_SWORD 成就）。",
        ["native.craft_wood_pickaxe", "native.collect_stone", "native.place_table"],
    ),
    (
        "native.craft_iron_sword",
        "MAKE_IRON_SWORD",
        "Craft an iron sword. / 制作一把铁剑。",
        "制作一把铁剑（MAKE_IRON_SWORD 成就）。",
        [
            "native.craft_stone_pickaxe",
            "native.collect_iron",
            "native.collect_coal",
            "native.place_furnace",
            "native.place_table",
        ],
    ),
    (
        "native.craft_diamond_sword",
        "MAKE_DIAMOND_SWORD",
        "Craft a diamond sword. / 制作一把钻石剑。",
        "制作一把钻石剑（MAKE_DIAMOND_SWORD 成就）。",
        ["native.craft_iron_pickaxe", "native.collect_diamond", "native.place_table"],
    ),
    # 盔甲
    (
        "native.craft_iron_armour",
        "MAKE_IRON_ARMOUR",
        "Craft iron armour. / 制作一套铁盔甲。",
        "制作一套铁盔甲（MAKE_IRON_ARMOUR 成就）。",
        [
            "native.craft_stone_pickaxe",
            "native.collect_iron",
            "native.collect_coal",
            "native.place_furnace",
            "native.place_table",
        ],
    ),
    (
        "native.craft_diamond_armour",
        "MAKE_DIAMOND_ARMOUR",
        "Craft diamond armour. / 制作一套钻石盔甲。",
        "制作一套钻石盔甲（MAKE_DIAMOND_ARMOUR 成就）。",
        [
            "native.craft_iron_pickaxe",
            "native.collect_diamond",
            "native.place_furnace",
            "native.place_table",
        ],
    ),
    # 道具
    (
        "native.craft_arrow",
        "MAKE_ARROW",
        "Craft an arrow. / 制作一支箭。",
        "制作一支箭（MAKE_ARROW 成就）。",
        ["native.craft_wood_pickaxe", "native.collect_stone", "native.collect_wood"],
    ),
    (
        "native.craft_torch",
        "MAKE_TORCH",
        "Craft a torch. / 制作一根火把。",
        "制作一根火把（MAKE_TORCH 成就）。",
        ["native.craft_wood_pickaxe", "native.collect_coal", "native.collect_wood"],
    ),
    # 弓
    (
        "native.find_bow",
        "FIND_BOW",
        "Find a bow. / 找到一张弓。",
        "找到一张弓（FIND_BOW 成就）。",
        [],  # 宝箱/掉落获得，无严格前置
    ),
    (
        "native.fire_bow",
        "FIRE_BOW",
        "Shoot an arrow. / 射出一支箭。",
        "用弓射出一支箭（FIRE_BOW 成就）。",
        ["native.find_bow", "native.craft_arrow"],
    ),
    # 附魔
    (
        "native.enchant_sword",
        "ENCHANT_SWORD",
        "Enchant a sword. / 附魔一把剑。",
        "在附魔台为剑附魔（ENCHANT_SWORD 成就）。",
        ["native.craft_stone_sword", "native.enter_fire_realm", "native.enter_ice_realm"],
    ),
    (
        "native.enchant_armour",
        "ENCHANT_ARMOUR",
        "Enchant armour. / 附魔盔甲。",
        "在附魔台为盔甲附魔（ENCHANT_ARMOUR 成就）。",
        ["native.craft_iron_armour", "native.enter_fire_realm", "native.enter_ice_realm"],
    ),
    # 魔法
    (
        "native.learn_fireball",
        "LEARN_FIREBALL",
        "Learn fireball. / 学会火球术。",
        "阅读魔法书学会火球术（LEARN_FIREBALL 成就）。",
        ["native.open_chest", "native.enter_fire_realm"],
    ),
    (
        "native.learn_iceball",
        "LEARN_ICEBALL",
        "Learn iceball. / 学会冰球术。",
        "阅读魔法书学会冰球术（LEARN_ICEBALL 成就）。",
        ["native.open_chest", "native.enter_ice_realm"],
    ),
    (
        "native.cast_fireball",
        "CAST_FIREBALL",
        "Cast a fireball. / 施放一个火球。",
        "施放一个火球（CAST_FIREBALL 成就）。",
        ["native.learn_fireball"],
    ),
    (
        "native.cast_iceball",
        "CAST_ICEBALL",
        "Cast an iceball. / 施放一个冰球。",
        "施放一个冰球（CAST_ICEBALL 成就）。",
        ["native.learn_iceball"],
    ),
]


def _single_task_spec(
    task_id: str,
    achievement: str,
    instruction: str,
    objective: str,
    dependencies: List[str],
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        version=TASK_VERSION,
        instruction=instruction,
        objective=objective,
        success_predicate={"type": "achievement", "name": achievement},
        annotation_predicates=[{"type": "achievement", "name": achievement}],
        renderer_config={},
        dependencies=list(dependencies),
    )


SINGLE_TASK_SPECS: List[TaskSpec] = [
    _single_task_spec(task_id, achievement, instruction, objective, dependencies)
    for task_id, achievement, instruction, objective, dependencies in _SINGLE_TASK_DEFS
]


class SingleAchievementTask(BaseTaskAdapter):
    """单成就驱动的制作/附魔/魔法任务；默认 0/1 进度。"""

    def __init__(self, spec: TaskSpec) -> None:
        super().__init__(spec)


def _single_task_factory(spec: TaskSpec) -> Callable[[], SingleAchievementTask]:
    def factory() -> SingleAchievementTask:
        return SingleAchievementTask(spec)

    return factory


# ---------------------------------------------------------------------------
# 组合任务
# ---------------------------------------------------------------------------

PICKAXE_ACHIEVEMENTS: List[str] = [
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "MAKE_DIAMOND_PICKAXE",
]
SWORD_ACHIEVEMENTS: List[str] = [
    "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD",
    "MAKE_IRON_SWORD",
    "MAKE_DIAMOND_SWORD",
]
ARMOUR_ACHIEVEMENTS: List[str] = [
    "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR",
]


def _achievement_expr(name: str) -> Dict[str, Any]:
    return {"type": "achievement", "name": name}


def _craft_full_kit_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.craft_full_kit",
        version=TASK_VERSION,
        instruction="Craft a full kit. / 打造一套完整装备。",
        objective="同时拥有任意一把镐、任意一把剑和一套盔甲（铁/钻石盔甲）。",
        success_predicate={
            "type": "and",
            "predicates": [
                {
                    "type": "or",
                    "predicates": [_achievement_expr(n) for n in PICKAXE_ACHIEVEMENTS],
                },
                {
                    "type": "or",
                    "predicates": [_achievement_expr(n) for n in SWORD_ACHIEVEMENTS],
                },
                {
                    "type": "or",
                    "predicates": [_achievement_expr(n) for n in ARMOUR_ACHIEVEMENTS],
                },
            ],
        },
        annotation_predicates=[
            _achievement_expr(n)
            for n in PICKAXE_ACHIEVEMENTS + SWORD_ACHIEVEMENTS + ARMOUR_ACHIEVEMENTS
        ],
        renderer_config={},
        dependencies=[
            "native.craft_wood_pickaxe",
            "native.craft_wood_sword",
            "native.craft_iron_armour",
        ],
    )


class CraftFullKitTask(BaseTaskAdapter):
    """进度 = 装备成就达成比例（10 个成就）。"""

    _KIT_ACHIEVEMENTS = PICKAXE_ACHIEVEMENTS + SWORD_ACHIEVEMENTS + ARMOUR_ACHIEVEMENTS

    def __init__(self) -> None:
        super().__init__(_craft_full_kit_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(self._KIT_ACHIEVEMENTS, state, info)


def _master_crafter_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.master_crafter",
        version=TASK_VERSION,
        instruction="Master crafting. / 精通制作。",
        objective="依次制作木、石、铁、钻石四种镐（全部镐类成就达成）。",
        success_predicate={
            "type": "and",
            "predicates": [_achievement_expr(n) for n in PICKAXE_ACHIEVEMENTS],
        },
        annotation_predicates=[_achievement_expr(n) for n in PICKAXE_ACHIEVEMENTS],
        renderer_config={},
        dependencies=[
            "native.craft_wood_pickaxe",
            "native.craft_stone_pickaxe",
            "native.craft_iron_pickaxe",
            "native.craft_diamond_pickaxe",
        ],
    )


class MasterCrafterTask(BaseTaskAdapter):
    """进度 = 四种镐成就达成比例。"""

    _PICKAXE_ACHIEVEMENTS = list(PICKAXE_ACHIEVEMENTS)

    def __init__(self) -> None:
        super().__init__(_master_crafter_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(self._PICKAXE_ACHIEVEMENTS, state, info)


# ---------------------------------------------------------------------------
# 模块级注册（import 即注册）
# ---------------------------------------------------------------------------

_COMPOSITE_FACTORIES = [CraftFullKitTask, MasterCrafterTask]


def register_crafting_tasks() -> None:
    from craftax.tasks.registry import register

    for spec in SINGLE_TASK_SPECS:
        register(spec.task_id, spec.version, _single_task_factory(spec))
    for factory in _COMPOSITE_FACTORIES:
        adapter = factory()
        spec: TaskSpec = adapter.spec
        register(spec.task_id, spec.version, factory)


register_crafting_tasks()
