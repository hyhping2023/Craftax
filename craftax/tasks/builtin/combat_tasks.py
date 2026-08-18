"""战斗类任务（仅标注，不改变游戏规则）。

覆盖：
- 逐个击败任务（僵尸 / 骷髅 / 侏儒战士 / 侏儒弓箭手 / 兽人士兵 / 兽人法师 /
  巨魔 / 狗头人 / 亡灵法师 Boss）；
- 组合任务（元素生物任一、任意三种敌人、亡灵任二）。

全部以原生 Achievement 成就谓词判定；组合任务进度 = 已达成目标数 / 目标阈值。
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

from craftax.contracts import TaskSpec
from craftax.tasks.base import BaseTaskAdapter

# 版本：与 builtin 其他任务保持一致
TASK_VERSION = "1.0.0"

# 全部敌人击杀成就（含原生枚举中的骑士/弓箭手，共 17 项；
# 注意枚举拼写 DEFEAT_ORC_SOLIDER 无 "i"）。
ALL_ENEMY_ACHIEVEMENTS = [
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
    "DEFEAT_KNIGHT",
    "DEFEAT_ARCHER",
]

# 元素击杀成就（火 / 冰 / 霜巨魔）
ELEMENTAL_ACHIEVEMENTS = [
    "DEFEAT_FIRE_ELEMENTAL",
    "DEFEAT_FROST_TROLL",
    "DEFEAT_ICE_ELEMENTAL",
]

# 亡灵击杀成就
UNDEAD_ACHIEVEMENTS = ["DEFEAT_ZOMBIE", "DEFEAT_SKELETON", "DEFEAT_NECROMANCER"]


def _ach(name: str) -> Dict[str, Any]:
    return {"type": "achievement", "name": name}


def _count_achieved(
    names: List[str], state: Any, info: Dict[str, Any]
) -> int:
    """names 中已达成成就的个数。"""
    progress = BaseTaskAdapter.achievement_progress(names, state, info)
    return int(round(progress * len(names)))


# ---------------------------------------------------------------------------
# 逐个击败任务
# ---------------------------------------------------------------------------

# (task_id, achievement, 英文指令, 中文指令, 目标中文名, dependencies)
# 依赖：楼层（敌人所在区域，严格） + 武器（推荐但按严格前置记录最低要求）
_SINGLE_DEFEAT_TASKS: List[Tuple[str, str, str, str, str, List[str]]] = [
    (
        "native.defeat_zombie",
        "DEFEAT_ZOMBIE",
        "Defeat a zombie.",
        "击败一只僵尸。",
        "僵尸",
        ["native.craft_wood_sword"],  # 地表夜晚刷新
    ),
    (
        "native.defeat_skeleton",
        "DEFEAT_SKELETON",
        "Defeat a skeleton.",
        "击败一只骷髅。",
        "骷髅",
        ["native.enter_dungeon", "native.craft_wood_sword"],
    ),
    (
        "native.defeat_gnome_warrior",
        "DEFEAT_GNOME_WARRIOR",
        "Defeat a gnome warrior.",
        "击败一名侏儒战士。",
        "侏儒战士",
        ["native.enter_gnomish_mines", "native.craft_stone_sword"],
    ),
    (
        "native.defeat_gnome_archer",
        "DEFEAT_GNOME_ARCHER",
        "Defeat a gnome archer.",
        "击败一名侏儒弓箭手。",
        "侏儒弓箭手",
        ["native.enter_gnomish_mines", "native.craft_stone_sword"],
    ),
    (
        "native.defeat_orc_soldier",
        "DEFEAT_ORC_SOLIDER",
        "Defeat an orc soldier.",
        "击败一名兽人士兵。",
        "兽人士兵",
        ["native.enter_sewers", "native.craft_stone_sword"],
    ),
    (
        "native.defeat_orc_mage",
        "DEFEAT_ORC_MAGE",
        "Defeat an orc mage.",
        "击败一名兽人法师。",
        "兽人法师",
        ["native.enter_sewers", "native.craft_stone_sword"],
    ),
    (
        "native.defeat_troll",
        "DEFEAT_TROLL",
        "Defeat a troll.",
        "击败一只巨魔。",
        "巨魔",
        ["native.enter_troll_mines", "native.craft_iron_sword"],
    ),
    (
        "native.defeat_kobold",
        "DEFEAT_KOBOLD",
        "Defeat a kobold.",
        "击败一只狗头人。",
        "狗头人",
        ["native.enter_sewers", "native.craft_stone_sword"],
    ),
    (
        "native.defeat_necromancer",
        "DEFEAT_NECROMANCER",
        "Defeat the necromancer boss.",
        "击败亡灵法师 Boss。",
        "亡灵法师",
        ["native.enter_graveyard", "native.craft_diamond_sword"],
    ),
    (
        "native.defeat_knight",
        "DEFEAT_KNIGHT",
        "Defeat a knight.",
        "击败一名骑士。",
        "骑士",
        ["native.enter_vault", "native.craft_iron_sword"],
    ),
    (
        "native.defeat_archer",
        "DEFEAT_ARCHER",
        "Defeat an archer.",
        "击败一名弓箭手。",
        "弓箭手",
        ["native.enter_vault", "native.craft_iron_sword"],
    ),
]


def _defeat_single_spec(
    task_id: str,
    achievement: str,
    instruction_en: str,
    instruction_zh: str,
    enemy_zh: str,
    dependencies: List[str],
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        version=TASK_VERSION,
        instruction=f"{instruction_en} / {instruction_zh}",
        objective=f"击败并击杀一只{enemy_zh}（{achievement} 成就达成）。",
        success_predicate=_ach(achievement),
        annotation_predicates=[_ach(achievement)],
        renderer_config={},
        dependencies=list(dependencies),
    )


class SingleAchievementTask(BaseTaskAdapter):
    """单成就任务（击杀或伤害）的 0/1 进度适配器。"""

    def __init__(self, spec: TaskSpec) -> None:
        super().__init__(spec)


# ---------------------------------------------------------------------------
# 伤害 Boss（DAMAGE_NECROMANCER：对亡灵法师造成伤害，而非击杀）
# ---------------------------------------------------------------------------


def _damage_necromancer_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.damage_necromancer",
        version=TASK_VERSION,
        instruction="Damage the necromancer. / 伤害死灵法师。",
        objective="对亡灵法师 Boss 造成伤害（DAMAGE_NECROMANCER 成就达成，无需击杀）。",
        success_predicate=_ach("DAMAGE_NECROMANCER"),
        annotation_predicates=[_ach("DAMAGE_NECROMANCER")],
        renderer_config={},
        dependencies=["native.enter_graveyard", "native.craft_diamond_sword"],
    )


# ---------------------------------------------------------------------------
# 元素击杀（火元素 or 冰元素，任一）
# ---------------------------------------------------------------------------


def _defeat_elemental_spec() -> TaskSpec:
    return TaskSpec(
        task_id="native.defeat_elemental",
        version=TASK_VERSION,
        instruction="Defeat any elemental. / 击败任意元素生物。",
        objective="击败火元素、冰元素或霜巨魔任一（对应击杀成就达成）。",
        success_predicate={
            "type": "or",
            "predicates": [_ach(name) for name in ELEMENTAL_ACHIEVEMENTS],
        },
        annotation_predicates=[_ach(name) for name in ELEMENTAL_ACHIEVEMENTS],
        renderer_config={},
        dependencies=[
            "native.enter_fire_realm",
            "native.enter_ice_realm",
            "native.craft_diamond_sword",
        ],
    )


class DefeatElementalTask(BaseTaskAdapter):
    """进度 = 已击败元素数 / 元素总数。"""

    def __init__(self) -> None:
        super().__init__(_defeat_elemental_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        return self.achievement_progress(ELEMENTAL_ACHIEVEMENTS, state, info)


# ---------------------------------------------------------------------------
# 任意三种敌人
# ---------------------------------------------------------------------------


def _defeat_three_enemies_spec() -> TaskSpec:
    # success：任取 3 种击杀成就，即 "or" 全部三元组合
    triples = [
        {"type": "and", "predicates": [_ach(a), _ach(b), _ach(c)]}
        for a, b, c in itertools.combinations(ALL_ENEMY_ACHIEVEMENTS, 3)
    ]
    return TaskSpec(
        task_id="native.defeat_three_enemies",
        version=TASK_VERSION,
        instruction="Defeat enemies (×3). / 击败敌人（×3）。",
        objective="累计击败任意三种敌人（击杀成就数达到 3 个）。",
        success_predicate={"type": "or", "predicates": triples},
        annotation_predicates=[_ach(name) for name in ALL_ENEMY_ACHIEVEMENTS],
        renderer_config={},
        dependencies=[
            "native.defeat_zombie",
            "native.defeat_skeleton",
            "native.defeat_gnome_warrior",
        ],
    )


class DefeatThreeEnemiesTask(BaseTaskAdapter):
    """进度 = 已击败敌人数 / 3（上限 1.0），达到 3 种即成功。"""

    def __init__(self) -> None:
        super().__init__(_defeat_three_enemies_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        count = _count_achieved(ALL_ENEMY_ACHIEVEMENTS, state, info)
        return self.clamp01(count / 3)


# ---------------------------------------------------------------------------
# 亡灵任二
# ---------------------------------------------------------------------------


def _defeat_undead_spec() -> TaskSpec:
    zombie, skeleton, necromancer = (_ach(n) for n in UNDEAD_ACHIEVEMENTS)
    return TaskSpec(
        task_id="native.defeat_undead",
        version=TASK_VERSION,
        instruction="Defeat undead (×2). / 击败亡灵（×2）。",
        objective="击败僵尸、骷髅、亡灵法师中的任意两种（对应击杀成就达成两个）。",
        success_predicate={
            "type": "or",
            "predicates": [
                {"type": "and", "predicates": [zombie, skeleton]},
                {"type": "and", "predicates": [zombie, necromancer]},
                {"type": "and", "predicates": [skeleton, necromancer]},
            ],
        },
        annotation_predicates=[_ach(name) for name in UNDEAD_ACHIEVEMENTS],
        renderer_config={},
        dependencies=["native.defeat_zombie", "native.defeat_skeleton"],
    )


class DefeatUndeadTask(BaseTaskAdapter):
    """进度 = 已击败亡灵数 / 2（上限 1.0），两种即成功。"""

    def __init__(self) -> None:
        super().__init__(_defeat_undead_spec())

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        count = _count_achieved(UNDEAD_ACHIEVEMENTS, state, info)
        return self.clamp01(count / 2)


# ---------------------------------------------------------------------------
# 模块级注册（import 即注册）
# ---------------------------------------------------------------------------


def register_combat_tasks() -> None:
    from craftax.tasks.registry import register

    for task_id, achievement, en, zh, enemy_zh, deps in _SINGLE_DEFEAT_TASKS:
        spec = _defeat_single_spec(task_id, achievement, en, zh, enemy_zh, deps)
        register(spec.task_id, spec.version, lambda spec=spec: SingleAchievementTask(spec))

    damage = _damage_necromancer_spec()
    register(damage.task_id, damage.version, lambda: SingleAchievementTask(damage))

    elemental = _defeat_elemental_spec()
    register(elemental.task_id, elemental.version, DefeatElementalTask)

    three = _defeat_three_enemies_spec()
    register(three.task_id, three.version, DefeatThreeEnemiesTask)

    undead = _defeat_undead_spec()
    register(undead.task_id, undead.version, DefeatUndeadTask)


register_combat_tasks()
