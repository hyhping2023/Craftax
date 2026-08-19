"""探索/交互类任务（仅标注，不改变游戏规则）。

- 区域进入类：以原生 ENTER_* 成就谓词判定，进度 0/1；
- 放置/交互类：以 PLACE_* / OPEN_CHEST / WAKE_UP 成就谓词判定，进度 0/1；
- 层级推进类：reach_floor_* / reach_boss_floor 用 level_ge 谓词判定，
  进度 = player_level / 8（第 8 层即 Boss 层，与 native.explore_dungeon 一致）；
- 组合类：deep_explorer 用 and 嵌套多个 ENTER_* 成就谓词。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from craftax.contracts import TaskSpec
from craftax.tasks.base import BaseTaskAdapter

# 版本：所有 builtin 任务首版统一 1.0.0
TASK_VERSION = "1.0.0"

# 默认楼层数（StaticEnvParams.num_levels），层级推进任务 progress 的分母。
DEFAULT_NUM_LEVELS = 9


def _achievement_spec(
    task_id: str,
    achievement: str,
    instruction: str,
    objective: str,
    dependencies: List[str] = (),
) -> TaskSpec:
    return TaskSpec(
        task_id=f"native.{task_id}",
        version=TASK_VERSION,
        instruction=instruction,
        objective=objective,
        success_predicate={"type": "achievement", "name": achievement},
        annotation_predicates=[{"type": "achievement", "name": achievement}],
        renderer_config={},
        dependencies=list(dependencies),
    )


def _level_spec(
    task_id: str,
    target_level: int,
    instruction: str,
    objective: str,
    dependencies: List[str] = (),
) -> TaskSpec:
    return TaskSpec(
        task_id=f"native.{task_id}",
        version=TASK_VERSION,
        instruction=instruction,
        objective=objective,
        success_predicate={"type": "level_ge", "value": target_level},
        annotation_predicates=[{"type": "level_ge", "value": target_level}],
        renderer_config={},
        dependencies=list(dependencies),
    )


# ---------------------------------------------------------------------------
# 区域进入任务（ENTER_* 成就）。楼层严格递进：下一层依赖上一层入口成就。
# ---------------------------------------------------------------------------

_ENTER_TASKS: List[List[Any]] = [
    [
        "enter_dungeon",
        "ENTER_DUNGEON",
        "Enter the dungeon. / 进入地下城。",
        "到达地下城入口并踏入其中（ENTER_DUNGEON 成就）。",
        "",  # 无前置：地表第一道梯子默认开放
    ],
    [
        "enter_gnomish_mines",
        "ENTER_GNOMISH_MINES",
        "Enter the gnomish mines. / 进入侏儒矿洞。",
        "到达侏儒矿洞并踏入其中（ENTER_GNOMISH_MINES 成就）。",
        "native.enter_dungeon",
    ],
    [
        "enter_sewers",
        "ENTER_SEWERS",
        "Enter the sewers. / 进入下水道。",
        "到达下水道并踏入其中（ENTER_SEWERS 成就）。",
        "native.enter_gnomish_mines",
    ],
    [
        "enter_vault",
        "ENTER_VAULT",
        "Enter the vault. / 进入宝库。",
        "到达宝库并踏入其中（ENTER_VAULT 成就）。",
        "native.enter_sewers",
    ],
    [
        "enter_troll_mines",
        "ENTER_TROLL_MINES",
        "Enter the troll mines. / 进入巨魔矿洞。",
        "到达巨魔矿洞并踏入其中（ENTER_TROLL_MINES 成就）。",
        "native.enter_vault",
    ],
    [
        "enter_fire_realm",
        "ENTER_FIRE_REALM",
        "Enter the fire realm. / 进入火焰领域。",
        "到达火焰领域并踏入其中（ENTER_FIRE_REALM 成就）。",
        # 火界怪 90% 物免 + 火免 → 必须先具备冰系伤害才可能生存（冰球术；
        # 执行器也接受冰附魔剑/弓作为等价替代，见 _acquire_elemental_capability）。
        ["native.enter_troll_mines", "native.learn_iceball"],
    ],
    [
        "enter_ice_realm",
        "ENTER_ICE_REALM",
        "Enter the ice realm. / 进入寒冰领域。",
        "到达寒冰领域并踏入其中（ENTER_ICE_REALM 成就）。",
        # 冰界怪 90% 物免 + 冰免 → 需火系伤害（火球术或火附魔）。
        ["native.enter_fire_realm", "native.learn_fireball"],
    ],
    [
        "enter_graveyard",
        "ENTER_GRAVEYARD",
        "Enter the graveyard. / 进入墓地。",
        "到达墓地并踏入其中（ENTER_GRAVEYARD 成就）。",
        "native.enter_ice_realm",
    ],
]


# ---------------------------------------------------------------------------
# 放置类任务（PLACE_* 成就）
# ---------------------------------------------------------------------------

_PLACE_TASKS: List[List[Any]] = [
    [
        "place_table",
        "PLACE_TABLE",
        "Place a table. / 放置一张桌子。",
        "放置一张桌子（PLACE_TABLE 成就）。",
        "native.collect_wood",  # 需要木材 >= 2
    ],
    [
        "place_furnace",
        "PLACE_FURNACE",
        "Place a furnace. / 放置一个熔炉。",
        "放置一个熔炉（PLACE_FURNACE 成就）。",
        ["native.craft_wood_pickaxe", "native.collect_stone"],  # 需石头（采石需木镐）
    ],
    [
        "place_stone",
        "PLACE_STONE",
        "Place a stone. / 放置一块石头。",
        "放置一块石头（PLACE_STONE 成就）。",
        ["native.craft_wood_pickaxe", "native.collect_stone"],
    ],
    [
        "place_plant",
        "PLACE_PLANT",
        "Place a plant. / 种下一株植物。",
        "种下一株植物（PLACE_PLANT 成就）。",
        "native.collect_sapling",
    ],
    [
        "place_torch",
        "PLACE_TORCH",
        "Place a torch. / 放置一支火把。",
        "放置一支火把（PLACE_TORCH 成就）。",
        "native.craft_torch",
    ],
]


# ---------------------------------------------------------------------------
# 其他交互任务（OPEN_CHEST / WAKE_UP 成就）
# ---------------------------------------------------------------------------

_OTHER_TASKS: List[List[Any]] = [
    [
        "open_chest",
        "OPEN_CHEST",
        "Open a chest. / 打开一个宝箱。",
        "打开一个宝箱（OPEN_CHEST 成就）。",
        "",  # 地表/洞穴中寻找宝箱即可
    ],
    [
        "wake_up",
        "WAKE_UP",
        "Wake up. / 从睡梦中醒来。",
        "从睡梦中醒来（WAKE_UP 成就）。",
        "",  # 睡觉后醒来即可
    ],
]


# ---------------------------------------------------------------------------
# 层级推进任务（level_ge 谓词）
# ---------------------------------------------------------------------------


class _ReachFloorTask(BaseTaskAdapter):
    """progress = player_level / 8；成功 = 到达目标层。"""

    _TARGET_LEVEL: int = 0

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        level = int(np.asarray(state.player_level))
        return self.clamp01(level / (DEFAULT_NUM_LEVELS - 1))


class ReachFloor3Task(_ReachFloorTask):
    _TARGET_LEVEL = 3

    def __init__(self) -> None:
        super().__init__(
            _level_spec(
                "reach_floor_3",
                3,
                "Reach floor 3. / 到达第 3 层。",
                "沿楼梯向下推进到第 3 层（player_level >= 3）。",
                dependencies=["native.enter_sewers"],
            )
        )


class ReachFloor5Task(_ReachFloorTask):
    _TARGET_LEVEL = 5

    def __init__(self) -> None:
        super().__init__(
            _level_spec(
                "reach_floor_5",
                5,
                "Reach floor 5. / 到达第 5 层。",
                "沿楼梯向下推进到第 5 层（player_level >= 5）。",
                dependencies=["native.enter_troll_mines"],
            )
        )


class ReachBossFloorTask(_ReachFloorTask):
    _TARGET_LEVEL = 8

    def __init__(self) -> None:
        super().__init__(
            _level_spec(
                "reach_boss_floor",
                8,
                "Reach the boss floor. / 到达 Boss 层。",
                "推进到第 8 层 Boss 层（player_level >= 8）。",
                dependencies=["native.enter_graveyard"],
            )
        )


# ---------------------------------------------------------------------------
# 组合任务（and 嵌套多个成就谓词）
# ---------------------------------------------------------------------------


def _deep_explorer_spec() -> TaskSpec:
    regions = [
        {"type": "achievement", "name": "ENTER_SEWERS"},
        {"type": "achievement", "name": "ENTER_VAULT"},
        {"type": "achievement", "name": "ENTER_GRAVEYARD"},
    ]
    return TaskSpec(
        task_id="native.deep_explorer",
        version=TASK_VERSION,
        instruction="Explore the depths. / 探索最深处。",
        objective="进入下水道、宝库与墓地（三者 ENTER_* 成就全部达成）。",
        success_predicate={"type": "and", "predicates": regions},
        annotation_predicates=regions,
        renderer_config={},
        dependencies=[
            "native.enter_sewers",
            "native.enter_vault",
            "native.enter_graveyard",
        ],
    )


class DeepExplorerTask(BaseTaskAdapter):
    """进度 = 三个深层区域进入成就的达成比例。"""

    _REGIONS = ["ENTER_SEWERS", "ENTER_VAULT", "ENTER_GRAVEYARD"]

    def __init__(self) -> None:
        super().__init__(_deep_explorer_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(self._REGIONS, state, info)


# ---------------------------------------------------------------------------
# 模块级注册（import 即注册）
# ---------------------------------------------------------------------------

_SINGLE_ACHIEVEMENT_TASKS = _ENTER_TASKS + _PLACE_TASKS + _OTHER_TASKS


def register_exploration_tasks() -> None:
    from craftax.tasks.registry import register

    for row in _SINGLE_ACHIEVEMENT_TASKS:
        # 行格式：[task_id, achievement, instruction, objective, dep]
        # dep 可为单个 task_id 字符串，或多个前置的列表（"" / 省略表示无前置）。
        task_id, achievement, instruction, objective = row[:4]
        dep = row[4] if len(row) > 4 else ""
        if isinstance(dep, (list, tuple)):
            deps = [d for d in dep if d]
        else:
            deps = [dep] if dep else []
        spec = _achievement_spec(task_id, achievement, instruction, objective, deps)
        register(spec.task_id, spec.version, lambda spec=spec: BaseTaskAdapter(spec))

    for factory in (
        ReachFloor3Task,
        ReachFloor5Task,
        ReachBossFloorTask,
        DeepExplorerTask,
    ):
        adapter = factory()
        register(adapter.task_id, adapter.version, factory)


register_exploration_tasks()
