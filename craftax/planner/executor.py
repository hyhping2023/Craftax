"""通用任务执行器：基于任务依赖图自动推导技能链，保证任务成功。

设计（对应 embodied_environment_plan.md §6.1.4 rolling planner）：
- 给定目标任务 task_id，用 TaskGraph.closure(include_self=True) 得到全部前置任务闭包，
  按拓扑层级排序形成任务链；
- 每步从链中取第一个"未完成"的任务作为当前子目标，按其类型派发到对应原语技能
  （采集 / 合成 / 放置 / 开箱 / 下楼 / 战斗 / 进食 / 生存维护）；
- 生存维护（饥饿/口渴/疲劳/HP）优先级最高，任何子目标之前先保证存活；
- 每步输入：GET /map 响应（map/mob/ladder/chest/monsters_killed）+
  step 响应的 state_summary（inventory/achievements/floor/player_position）。

本模块只读全图做规划与决策，执行仍走原生动作；不改游戏逻辑。
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from craftax.planner.path_planner import (
    ACTION_DELTA,
    CRAFTING_TABLE_TILE,
    DELTA_TO_ACTION,
    PLACEABLE_TILES,
    blocked,
    find_nearest_target,
)

# ---------------------------------------------------------------------------
# 动作 id（与 craftax.craftax.constants.Action 一致）
# ---------------------------------------------------------------------------
NOOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
DO = 5
SLEEP = 6
PLACE_STONE = 7
PLACE_TABLE = 8
PLACE_FURNACE = 9
PLACE_PLANT = 10
MAKE_WOOD_PICKAXE = 11
MAKE_STONE_PICKAXE = 12
MAKE_IRON_PICKAXE = 13
MAKE_WOOD_SWORD = 14
MAKE_STONE_SWORD = 15
MAKE_IRON_SWORD = 16
DESCEND = 18
ASCEND = 19
MAKE_DIAMOND_PICKAXE = 20
MAKE_DIAMOND_SWORD = 21
MAKE_IRON_ARMOUR = 22
MAKE_DIAMOND_ARMOUR = 23
SHOOT_ARROW = 24
MAKE_ARROW = 25
CAST_FIREBALL = 26
CAST_ICEBALL = 27
PLACE_TORCH = 28
DRINK_POTION_RED = 29
DRINK_POTION_BLUE = 31
DRINK_POTION_PINK = 32
DRINK_POTION_CYAN = 33
DRINK_POTION_YELLOW = 34
READ_BOOK = 35
ENCHANT_SWORD = 36
ENCHANT_ARMOUR = 37
MAKE_TORCH = 38

# ---------------------------------------------------------------------------
# 方块类型（BlockType 枚举值）
# ---------------------------------------------------------------------------
GRASS = 2
WATER = 3
STONE = 4
TREE = 5
COAL = 8
IRON = 9
DIAMOND = 10
FURNACE = 12
RIPE_PLANT = 16
CHEST = 23
FOUNTAIN = 24
FIRE_TREE = 28
SAPPHIRE = 21
RUBY = 22
ENCHANT_TABLE_FIRE = 30
ENCHANT_TABLE_ICE = 31

# 采集类任务 -> 目标方块
COLLECT_TARGET_BLOCKS: Dict[str, List[int]] = {
    "native.collect_wood": [TREE, FIRE_TREE],
    "native.collect_stone": [STONE],
    "native.collect_coal": [COAL],
    "native.collect_iron": [IRON],
    "native.collect_diamond": [DIAMOND],
    "native.collect_sapphire": [SAPPHIRE],
    "native.collect_ruby": [RUBY],
    "native.collect_drink": [WATER, FOUNTAIN],
}

# 采集任务的优先楼层（矿石主要分布层，按丰富度排序；地表缺则下楼）
# 宝石优先 L5（蓝/红宝石密度均 1%，且 L5 有水源+被动怪，生存资源最全）；
# L7 蓝宝密度 2% 但无 passive（无食物），L6 红宝 2.5% 但无水源。
COLLECT_TARGET_FLOORS: Dict[str, List[int]] = {
    "native.collect_wood": [0],
    "native.collect_stone": [0, 2],
    "native.collect_coal": [0, 2],
    "native.collect_iron": [2, 5, 0],
    "native.collect_diamond": [2, 5, 7],
    "native.collect_sapphire": [5, 7, 2],
    "native.collect_ruby": [5, 6, 2],
    "native.collect_drink": [0],
}

# 无水源楼层（火界 L6 为熔岩海，Boss 层 L8 无水源）
NO_DRINK_FLOORS = {6, 8}
# 无食物楼层（L7 冰界 passive 刷新率为 0，无植物）
NO_FOOD_FLOORS = {7}

# 亡灵法师方块（Boss 战 DO 目标）
NECROMANCER_BLOCK = 32

# 合成类任务 -> (制作动作 id, 需要熔炉?)
CRAFT_ACTIONS: Dict[str, Tuple[int, bool]] = {
    "native.craft_wood_pickaxe": (MAKE_WOOD_PICKAXE, False),
    "native.craft_stone_pickaxe": (MAKE_STONE_PICKAXE, False),
    "native.craft_iron_pickaxe": (MAKE_IRON_PICKAXE, True),
    "native.craft_diamond_pickaxe": (MAKE_DIAMOND_PICKAXE, False),
    "native.craft_wood_sword": (MAKE_WOOD_SWORD, False),
    "native.craft_stone_sword": (MAKE_STONE_SWORD, False),
    "native.craft_iron_sword": (MAKE_IRON_SWORD, True),
    "native.craft_diamond_sword": (MAKE_DIAMOND_SWORD, False),
    "native.craft_iron_armour": (MAKE_IRON_ARMOUR, True),
    "native.craft_diamond_armour": (MAKE_DIAMOND_ARMOUR, False),
    "native.craft_arrow": (MAKE_ARROW, False),
    "native.craft_torch": (MAKE_TORCH, False),
}

# 放置类任务 -> 放置动作 id
PLACE_ACTIONS: Dict[str, int] = {
    "native.place_stone": PLACE_STONE,
    "native.place_table": PLACE_TABLE,
    "native.place_furnace": PLACE_FURNACE,
    "native.place_plant": PLACE_PLANT,
    "native.place_torch": PLACE_TORCH,
}

# 战斗类任务：击杀成就名（DefeatEnemy 等）
DEFEAT_TASKS: Set[str] = {
    "native.defeat_zombie", "native.defeat_skeleton", "native.defeat_gnome_warrior",
    "native.defeat_gnome_archer", "native.defeat_orc_soldier", "native.defeat_orc_mage",
    "native.defeat_kobold", "native.defeat_troll", "native.defeat_necromancer",
    "native.defeat_knight", "native.defeat_archer", "native.damage_necromancer",
}
DEFEAT_ACHIEVEMENT: Dict[str, str] = {
    "native.defeat_zombie": "DEFEAT_ZOMBIE",
    "native.defeat_skeleton": "DEFEAT_SKELETON",
    "native.defeat_gnome_warrior": "DEFEAT_GNOME_WARRIOR",
    "native.defeat_gnome_archer": "DEFEAT_GNOME_ARCHER",
    "native.defeat_orc_soldier": "DEFEAT_ORC_SOLIDER",
    "native.defeat_orc_mage": "DEFEAT_ORC_MAGE",
    "native.defeat_kobold": "DEFEAT_KOBOLD",
    "native.defeat_troll": "DEFEAT_TROLL",
    "native.defeat_necromancer": "DEFEAT_NECROMANCER",
    "native.defeat_knight": "DEFEAT_KNIGHT",
    "native.defeat_archer": "DEFEAT_ARCHER",
    "native.damage_necromancer": "DEFEAT_NECROMANCER",
}

# 战斗任务 -> 目标怪物所在楼层（依据 FLOOR_MOB_MAPPING：每层仅一种 melee + 一种 ranged）。
# 用于在正确楼层战斗：骷髅只在 L0(地表) 刷新，而任务依赖图误标 enter_dungeon，
# 此处执行器层绕过，直接导航到目标怪实际刷新的楼层。
# (mob_class, mob_type_id, floors)。type_id 仅用于诊断；战斗按楼层+类别即可。
DEFEAT_MOB_LOCATIONS: Dict[str, Tuple[str, int, List[int]]] = {
    "native.defeat_zombie": ("melee", 0, [0]),
    "native.defeat_skeleton": ("ranged", 0, [0, 8]),  # L8 Boss 波次也刷 type0
    "native.defeat_gnome_warrior": ("melee", 1, [2]),
    "native.defeat_gnome_archer": ("ranged", 1, [2]),
    "native.defeat_orc_soldier": ("melee", 2, [1]),
    "native.defeat_orc_mage": ("ranged", 2, [1]),
    "native.defeat_kobold": ("ranged", 3, [3]),
    "native.defeat_troll": ("melee", 5, [5]),
    "native.defeat_knight": ("melee", 4, [4]),
    "native.defeat_archer": ("ranged", 4, [4]),
    "native.defeat_necromancer": ("boss", -1, [8]),
    "native.damage_necromancer": ("boss", -1, [8]),
}

# 进入地牢类任务 -> 目标楼层（player_level）
ENTER_FLOOR: Dict[str, int] = {
    "native.enter_dungeon": 1,
    "native.enter_gnomish_mines": 2,
    "native.enter_sewers": 3,
    "native.enter_vault": 4,
    "native.enter_troll_mines": 5,
    "native.enter_fire_realm": 6,
    "native.enter_ice_realm": 7,
    "native.enter_graveyard": 8,
}

# 到达指定楼层任务 -> 目标 player_level
REACH_FLOOR: Dict[str, int] = {
    "native.reach_floor_3": 3,
    "native.reach_floor_5": 5,
    "native.reach_boss_floor": 8,
    "native.explore_dungeon": 8,
}

# 学法术任务 -> 目标楼层（拿书）
LEARN_FLOOR: Dict[str, int] = {
    "native.learn_fireball": 3,
    "native.learn_iceball": 4,
}

# 施法任务 -> 施法动作
CAST_ACTIONS: Dict[str, int] = {
    "native.cast_fireball": CAST_FIREBALL,
    "native.cast_iceball": CAST_ICEBALL,
}

# 附魔任务 -> 附魔动作 + 需要附魔台类型
ENCHANT_ACTIONS: Dict[str, Tuple[int, List[int]]] = {
    "native.enchant_sword": (ENCHANT_SWORD, [ENCHANT_TABLE_FIRE, ENCHANT_TABLE_ICE]),
    "native.enchant_armour": (ENCHANT_ARMOUR, [ENCHANT_TABLE_FIRE, ENCHANT_TABLE_ICE]),
}

# 需要站立于其旁(8邻域)才能触发动作的方块
CRAFT_TABLE_BLOCK = CRAFTING_TABLE_TILE
FURNACE_BLOCK = FURNACE

# 资源采集任务 -> (库存字段, 目标储备量)
# 储备量取"当前链段最临近消耗"即可：合成时若不足，_craft_action 会按
# CRAFT_RESOURCE_COSTS 就地补采。设太高会迫使采集任务为凑数而提前深下
# （浅层装备不足时下深层清怪极危险）。深层矿石按需下行。
RESOURCE_TARGETS: Dict[str, Tuple[str, int]] = {
    "native.collect_wood": ("wood", 5),
    "native.collect_stone": ("stone", 8),
    "native.collect_coal": ("coal", 1),
    "native.collect_iron": ("iron", 2),
    "native.collect_diamond": ("diamond", 3),
    "native.collect_sapphire": ("sapphire", 1),
    "native.collect_ruby": ("ruby", 1),
}

# 合成动作 -> 消耗的资源（_craft_action 合成前检查，不足则就地补采）
CRAFT_RESOURCE_COSTS: Dict[str, Dict[str, int]] = {
    "native.craft_wood_pickaxe": {"wood": 1},
    "native.craft_stone_pickaxe": {"wood": 1, "stone": 1},
    "native.craft_iron_pickaxe": {"wood": 1, "stone": 1, "iron": 1, "coal": 1},
    "native.craft_diamond_pickaxe": {"wood": 1, "diamond": 3},
    "native.craft_wood_sword": {"wood": 1},
    "native.craft_stone_sword": {"wood": 1, "stone": 1},
    "native.craft_iron_sword": {"wood": 1, "stone": 1, "iron": 1, "coal": 1},
    "native.craft_diamond_sword": {"wood": 1, "diamond": 2},
    "native.craft_iron_armour": {"iron": 3, "coal": 3},
    "native.craft_diamond_armour": {"diamond": 3},
    "native.craft_arrow": {"wood": 1, "stone": 1},
    "native.craft_torch": {"wood": 1, "coal": 1},
}

# 采集任务 -> 对应库存字段（_craft_action 补资源时使用）
CRAFT_RESOURCE_COLLECT_TASK: Dict[str, str] = {
    "wood": "native.collect_wood",
    "stone": "native.collect_stone",
    "coal": "native.collect_coal",
    "iron": "native.collect_iron",
    "diamond": "native.collect_diamond",
    "sapphire": "native.collect_sapphire",
    "ruby": "native.collect_ruby",
}

# 作为“目标任务”时只需达成成就（收集 ≥1 即可完成任务本身）；
# 仅当作为前置任务（为后续 craft 供材）时才需要 RESOURCE_TARGETS 的储备量。
TARGET_ACHIEVEMENT_FIELDS: Dict[str, str] = {
    "native.collect_wood": "wood",
    "native.collect_stone": "stone",
    "native.collect_coal": "coal",
    "native.collect_iron": "iron",
    "native.collect_diamond": "diamond",
    "native.collect_sapphire": "sapphire",
    "native.collect_ruby": "ruby",
}

# 任务 -> 完成所依赖的成就名（用于从 summary.achievements 判定子目标完成）
ACHIEVEMENT_COMPLETION: Dict[str, List[str]] = {
    "native.collect_wood": ["COLLECT_WOOD"],
    "native.collect_stone": ["COLLECT_STONE"],
    "native.collect_coal": ["COLLECT_COAL"],
    "native.collect_iron": ["COLLECT_IRON"],
    "native.collect_diamond": ["COLLECT_DIAMOND"],
    "native.collect_sapphire": ["COLLECT_SAPPHIRE"],
    "native.collect_ruby": ["COLLECT_RUBY"],
    "native.collect_drink": ["COLLECT_DRINK"],
    "native.collect_sapling": ["COLLECT_SAPLING"],
    "native.eat_cow": ["EAT_COW"],
    "native.eat_plant": ["EAT_PLANT"],
    "native.eat_bat": ["EAT_BAT"],
    "native.eat_snail": ["EAT_SNAIL"],
    "native.drink_potion": ["DRINK_POTION"],
    "native.place_table": ["PLACE_TABLE"],
    "native.place_furnace": ["PLACE_FURNACE"],
    "native.place_stone": ["PLACE_STONE"],
    "native.place_plant": ["PLACE_PLANT"],
    "native.place_torch": ["PLACE_TORCH"],
    "native.open_chest": ["OPEN_CHEST"],
    "native.find_bow": ["FIND_BOW"],
    "native.fire_bow": ["FIRE_BOW"],
    "native.wake_up": ["WAKE_UP"],
    "native.enchant_sword": ["ENCHANT_SWORD"],
    "native.enchant_armour": ["ENCHANT_ARMOUR"],
    "native.learn_fireball": ["LEARN_FIREBALL"],
    "native.learn_iceball": ["LEARN_ICEBALL"],
    "native.cast_fireball": ["CAST_FIREBALL"],
    "native.cast_iceball": ["CAST_ICEBALL"],
}

for _tid, _ach in DEFEAT_ACHIEVEMENT.items():
    ACHIEVEMENT_COMPLETION.setdefault(_tid, [_ach])

# 复合任务（and/or 组合）依赖其子任务完成，不需要独立原语；
# 其完成由 success_predicate 引用成就判定，见 _task_is_complete。
COMPOSITE_TASKS: Set[str] = {
    "native.collect_ore", "native.collect_all_gems", "native.eat_food",
    "native.craft_full_kit", "native.master_crafter", "native.deep_explorer",
    "native.defeat_elemental", "native.defeat_three_enemies", "native.defeat_undead",
    "native.craft_tools", "native.defeat_enemy", "native.explore_dungeon",
    "native.survive", "native.survival_starter_kit", "native.basic_combat_training",
    "native.dungeon_campaign", "native.collect_all_primary_resources",
    "native.survival_sustenance", "native.collect_all_ores", "native.starter_toolkit",
    "native.iron_gear", "native.crafting_mastery", "native.clear_surface_threats",
    "native.conquer_mid_tier_foes", "native.conquer_dungeon_bosses",
    "native.reach_mid_dungeon", "native.conquer_lower_realms", "native.build_home_base",
}

# 可食用方块（DO 面向进食）
FOOD_BLOCKS = [RIPE_PLANT]
# 可饮水方块
DRINK_BLOCKS = [WATER, FOUNTAIN]


def _norm_pos(p: Any) -> Tuple[int, int]:
    return (int(p[0]), int(p[1]))


class SkillChainExecutor:
    """依赖图驱动的通用任务执行器。"""

    def __init__(self, task_id: str, *, max_steps: int = 2000) -> None:
        self.task_id = task_id
        self.max_steps = max_steps
        self._chain: List[str] = []
        self._chain_idx = 0
        self._build_chain()

    # -- 任务链构建 --------------------------------------------------------

    def _build_chain(self) -> None:
        from craftax.tasks.graph import TaskGraph

        graph = TaskGraph.build_from_registry()
        closure = graph.closure(self.task_id, include_self=True)
        # 拓扑排序：按 topological_level 升序（依赖在前）
        self._chain = sorted(
            closure,
            key=lambda t: (graph.node(t).topological_level, t),
        )

    def chain(self) -> List[str]:
        return list(self._chain)

    # -- 对外接口 ----------------------------------------------------------

    def next_action(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Optional[int]:
        """返回下一步动作 id；全部完成返回 None（调用方停止录制）。"""
        if "map" in map_payload and isinstance(map_payload["map"], list):
            map_payload = dict(map_payload)
            map_payload["map"] = np.asarray(map_payload["map"], dtype=np.int32)
        # 本层活怪占用的格子会挡住玩家移动，寻路时视为不可通行
        self._mob_cells = self._mob_blocked_cells(map_payload)
        if self._is_done(map_payload, summary):
            return None
        # 属性升级：优先力量（每点 +25% 物伤，str=5 时翻倍，清怪效率大增）。
        # XP 来自下楼（每层 +1），深层任务自然攒够。
        if int(summary.get("xp", 0)) >= 1 and int(summary.get("strength", 1)) < 5:
            return LEVEL_UP_STRENGTH
        # 尽早确保有近战武器：地表被怪攻击时无剑极危险；
        # 若 wood 充足且附近有工作台，优先做木剑自保
        inventory = summary.get("inventory") or {}
        if (int(inventory.get("sword", 0)) < 1
                and int(inventory.get("wood", 0)) >= 1
                and self._near_any(map_payload["map"], _norm_pos(summary.get("player_position", [0, 0])), [CRAFT_TABLE_BLOCK])):
            craft = self._craft_action("native.craft_wood_sword", map_payload, summary)
            if craft is not None:
                return craft
        # 当前子目标（首个未完成）决定是否优先下楼：本层已清 8 怪且需下楼时，
        # 优先下楼而非纠缠于防御战斗（避免在无水源的深层耗尽）
        floor = int(summary.get("floor", 0))
        need_descend = False
        while self._chain_idx < len(self._chain):
            tid = self._chain[self._chain_idx]
            if not self._task_is_complete(tid, summary):
                if tid in ENTER_FLOOR or tid in REACH_FLOOR:
                    need_descend = True
                break
            self._chain_idx += 1
        if (need_descend and floor > 0
                and int(map_payload.get("monsters_killed", 0)) >= 8):
            descend = self._descend_to(map_payload, summary,
                                       REACH_FLOOR.get(self._chain[self._chain_idx],
                                                       ENTER_FLOOR[self._chain[self._chain_idx]]))
            if descend is not None:
                return descend
        # 生存维护优先
        survival = self._survival_action(map_payload, summary)
        if survival is not None:
            return survival
        # 推进到第一个未完成的子目标
        while self._chain_idx < len(self._chain):
            tid = self._chain[self._chain_idx]
            if not self._task_is_complete(tid, summary):
                break
            self._chain_idx += 1
        if self._chain_idx >= len(self._chain):
            return None
        tid = self._chain[self._chain_idx]
        action = self._primitive(tid, map_payload, summary)
        return action

    def is_done(self, summary: Dict[str, Any]) -> bool:
        return self._task_is_complete(self.task_id, summary)

    def step_count_hint(self) -> int:
        return len(self._chain)

    def estimate_steps(self) -> int:
        """按依赖链深度估算完成任务所需步数上限（demo/测试用）。

        基础 800 + 每深一层清怪约 700 步 + 每个链上任务约 150 步，封顶 30000。
        深度取自链上任务的目标楼层：enter/reach/learn 楼层，以及采集/战斗任务
        需要的矿石/怪物分布层（这些任务隐含下行）。
        """
        max_floor = 0
        for tid in self._chain:
            f = ENTER_FLOOR.get(tid, REACH_FLOOR.get(tid, LEARN_FLOOR.get(tid, 0)))
            if tid in COLLECT_TARGET_FLOORS:
                f = max(f, max(COLLECT_TARGET_FLOORS[tid]))
            loc = DEFEAT_MOB_LOCATIONS.get(tid)
            if loc is not None:
                f = max(f, max(loc[2]))
            max_floor = max(max_floor, f)
        return min(30000, 800 + 700 * max_floor + 150 * len(self._chain))

    # -- 完成判定 ----------------------------------------------------------

    def _task_is_complete(self, tid: str, summary: Dict[str, Any]) -> bool:
        # 资源采集任务：目标任务只需 ≥1（成就达成），前置任务需储备量
        if tid in RESOURCE_TARGETS:
            field, target = RESOURCE_TARGETS[tid]
            if tid == self.task_id:
                target = 1
            inventory = summary.get("inventory") or {}
            return int(inventory.get(field, 0)) >= target
        # 合成任务：以背包工具等级/数量判定（MAKE_* 动作执行后即时更新）
        if tid in CRAFT_ACTIONS:
            inventory = summary.get("inventory") or {}
            return self._craft_done(tid, inventory)
        # 组合进食任务：任意食物成就达成即完成
        if tid in ("native.eat_food", "native.eat_plant", "native.eat_cow",
                   "native.eat_bat", "native.eat_snail"):
            achieved = set(summary.get("achievements", []))
            food_achs = {"EAT_COW", "EAT_PLANT", "EAT_BAT", "EAT_SNAIL"}
            if tid == "native.eat_food":
                return any(a in achieved for a in food_achs)
            return bool(ACHIEVEMENT_COMPLETION.get(tid) and
                        ACHIEVEMENT_COMPLETION[tid][0] in achieved)
        if tid in ACHIEVEMENT_COMPLETION:
            achieved = set(summary.get("achievements", []))
            return all(a in achieved for a in ACHIEVEMENT_COMPLETION[tid])
        if tid in REACH_FLOOR:
            return int(summary.get("floor", 0)) >= REACH_FLOOR[tid]
        if tid in ENTER_FLOOR:
            return int(summary.get("floor", 0)) >= ENTER_FLOOR[tid]
        if tid == "native.survive":
            return True  # 存活即完成
        # 其余（含复合任务）：无显式成就判定 → 交给依赖链与动作驱动
        return False

    def _craft_done(self, tid: str, inventory: Dict[str, Any]) -> bool:
        """合成任务完成判定：对应工具/物品等级或数量达标。"""
        if tid in ("native.craft_wood_pickaxe",):
            return int(inventory.get("pickaxe", 0)) >= 1
        if tid in ("native.craft_stone_pickaxe",):
            return int(inventory.get("pickaxe", 0)) >= 2
        if tid in ("native.craft_iron_pickaxe",):
            return int(inventory.get("pickaxe", 0)) >= 3
        if tid in ("native.craft_diamond_pickaxe",):
            return int(inventory.get("pickaxe", 0)) >= 4
        if tid in ("native.craft_wood_sword",):
            return int(inventory.get("sword", 0)) >= 1
        if tid in ("native.craft_stone_sword",):
            return int(inventory.get("sword", 0)) >= 2
        if tid in ("native.craft_iron_sword",):
            return int(inventory.get("sword", 0)) >= 3
        if tid in ("native.craft_diamond_sword",):
            return int(inventory.get("sword", 0)) >= 4
        if tid in ("native.craft_iron_armour",):
            return sum(int(x) for x in inventory.get("armour", [0])) >= 1
        if tid in ("native.craft_diamond_armour",):
            return sum(int(x) for x in inventory.get("armour", [0])) >= 2
        if tid in ("native.craft_arrow",):
            return int(inventory.get("arrows", 0)) >= 1
        if tid in ("native.craft_torch",):
            return int(inventory.get("torches", 0)) >= 1
        return False

    def _is_done(self, map_payload: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        return self._task_is_complete(self.task_id, summary)

    # -- 生存维护 ----------------------------------------------------------

    def _survival_action(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        energy = float(summary.get("energy", 9.0))
        food = float(summary.get("food", 9.0))
        drink = float(summary.get("drink", 9.0))
        health = float(summary.get("health", 9.0))
        is_sleeping = bool(summary.get("is_sleeping", False))
        inventory = summary.get("inventory") or {}
        floor = int(summary.get("floor", 0))
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))

        # 优先级：能量 → 口渴 → 饥饿 → 近身威胁(撤退) → 防御战斗。
        # 注意：SLEEP 会让动作变 NOOP 直到能量回满或被怪打醒（3.5x 伤害），
        # 因此低血时绝不睡觉（睡觉 = 待宰）；只靠被动回血 + 撤退保命。
        # 疲劳：能量低必须睡（不睡能量归零掉血）。但 SLEEP 会被怪 3.5x 打醒，
        # 低血时睡觉=待宰 → 健康 <8 时不睡（L0 僵尸 3.5x=7 伤，8 血才扛得住），
        # 先撤退靠被动回血等健康回够再睡。
        if energy < 2 and not is_sleeping:
            if health < 8:
                retreat = self._retreat_from_mobs(map_payload, summary)
                if retreat is not None:
                    return retreat
                return None  # 无可退处：硬扛（被动回血）
            return SLEEP
        # 口渴：已在水旁 → 喝满（避免反复远途喝水把 drink 拖在 1~3 卡死任务）；
        # 临界 → 找水；当前层无水则上浅一层（L6 火界等）
        if drink < 8 and self._near_any(map2d, pos, DRINK_BLOCKS):
            a = self._seek_and_do(map_payload, summary, DRINK_BLOCKS)
            if a is not None:
                return a
        if drink < 3:
            a = self._seek_and_do(map_payload, summary, DRINK_BLOCKS)
            if a is not None:
                return a
            if floor > 0:
                a = self._ascend_to(map_payload, summary, floor - 1)
                if a is not None:
                    return a
        # 饥饿：附近有熟植物 → 吃到 8；被动怪 → 主动进食；临界 → 找食物/等刷新
        if food < 8 and self._near_any(map2d, pos, FOOD_BLOCKS):
            a = self._seek_and_do(map_payload, summary, FOOD_BLOCKS)
            if a is not None:
                return a
        if food < 5:
            a = self._seek_and_eat_mob(map_payload, summary)
            if a is not None:
                return a
        if food < 3:
            a = self._seek_and_do(map_payload, summary, FOOD_BLOCKS)
            if a is not None:
                return a
            a = self._seek_and_eat_mob(map_payload, summary)
            if a is not None:
                return a
            if floor > 0:
                a = self._ascend_to(map_payload, summary, floor - 1)
                if a is not None:
                    return a
            # 无任何食物来源：站着等被动怪刷新（不睡：睡觉=待宰）
            return DO
        # 被 ≥2 只怪近身 → 先撤出包围（被围殴 dps 扛不住），拉开再逐个单挑
        if not is_sleeping and self._mob_count_within(map_payload, summary, 1) >= 2:
            retreat = self._retreat_from_mobs(map_payload, summary)
            if retreat is not None:
                return retreat
        # 生命低（<6）且近身有怪（≤5 格）→ 撤退脱离（避免低血硬拼，靠被动回血）。
        # 远处的远程怪不算"近身威胁"：应主动击杀（逃跑反而持续吃射击）。
        if health < 6 and not is_sleeping and self._mob_within(map_payload, summary, 5):
            retreat = self._retreat_from_mobs(map_payload, summary)
            if retreat is not None:
                return retreat
        # 防御性战斗：仅当怪真正紧邻（≤1 格，会攻击）时清怪；
        # 2 格外的怪不影响任务推进（走路/下楼/喝水优先），避免被反复打断
        if self._mob_adjacent(map_payload, summary, max_dist=1):
            combat = self._combat_any(map_payload, summary)
            if combat is not None and combat != SLEEP:
                return combat
        return None

    def _mob_within(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], max_dist: int
    ) -> bool:
        """玩家 max_dist 内是否有存活怪（含被动）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        for key in ("melee", "ranged", "passive"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                if abs(int(p[0]) - pos[0]) + abs(int(p[1]) - pos[1]) <= max_dist:
                    return True
        return False

    def _mob_count_within(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], max_dist: int
    ) -> int:
        """玩家 max_dist 内存活怪的数量（含被动）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        count = 0
        for key in ("melee", "ranged", "passive"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                if abs(int(p[0]) - pos[0]) + abs(int(p[1]) - pos[1]) <= max_dist:
                    count += 1
        return count

    def _retreat_from_mobs(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """向远离近身怪簇的方向移动一步（低血时脱离接触以便睡觉回血）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        nearby: List[Tuple[int, int]] = []
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                c = (int(p[0]), int(p[1]))
                if abs(c[0] - pos[0]) + abs(c[1] - pos[1]) <= 10:
                    nearby.append(c)
        if not nearby:
            return None
        map2d = map_payload["map"]
        h, w = len(map2d), len(map2d[0])
        blocked_extra = getattr(self, "_mob_cells", None) or set()
        best_delta = None
        best_score = -1
        for action, delta in ACTION_DELTA.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if blocked(int(map2d[nxt[0]][nxt[1]])):
                continue
            if nxt in blocked_extra:
                continue  # 怪占用的格不可走（避免原地撞怪）
            # 最大化"距最近怪的距离"——从怪簇中向出口跑
            score = min(abs(c[0] - nxt[0]) + abs(c[1] - nxt[1]) for c in nearby)
            if score > best_score:
                best_score = score
                best_delta = delta
        if best_delta is not None:
            return DELTA_TO_ACTION[best_delta]
        return None

    def _mob_adjacent(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], max_dist: int = 2
    ) -> bool:
        """玩家 max_dist 内是否有怪（紧邻威胁）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        for key in ("melee", "ranged", "passive"):
            entry = mobs.get(key, {})
            positions = entry.get("positions", [])
            masks = entry.get("masks", [])
            for i, p in enumerate(positions):
                if i < len(masks) and not masks[i]:
                    continue
                dist = abs(int(p[0]) - pos[0]) + abs(int(p[1]) - pos[1])
                if dist <= max_dist:
                    return True
        return False

    def _seek_and_eat_mob(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """找最近可达被动怪（牛/蝙蝠/蜗牛）并 DO 进食。"""
        mobs = map_payload.get("mob_positions", {})
        passive = mobs.get("passive", {})
        positions = passive.get("positions", [])
        masks = passive.get("masks", [])
        candidates = []
        for i, pos in enumerate(positions):
            if i < len(masks) and not masks[i]:
                continue
            candidates.append(_norm_pos(pos))
        if not candidates:
            return None
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        map2d = map_payload["map"]
        reachable = self._reachable_set(map2d, pos)
        candidates = [c for c in candidates if c in reachable]
        if not candidates:
            return None
        # 按距离依次尝试：相邻则 DO，否则找可走路；被挡则换下一个
        for c in sorted(
            candidates, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1])
        ):
            delta = (c[0] - pos[0], c[1] - pos[1])
            if delta in DELTA_TO_ACTION:
                if direction == DELTA_TO_ACTION[delta]:
                    return DO
                return DELTA_TO_ACTION[delta]
            walk = self._walk_to_mob(map2d, pos, c)
            if walk is not None:
                return walk
        return None

    # -- 原语技能 ----------------------------------------------------------

    def _primitive(
        self, tid: str, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        # 采集 / 喝水
        if tid in COLLECT_TARGET_BLOCKS:
            return self._collect_resource(tid, map_payload, summary)
        # 树苗：DO 在草地上
        if tid == "native.collect_sapling":
            return self._seek_and_do(map_payload, summary, [GRASS])
        # 进食（熟植物 / 被动怪）
        if tid in ("native.eat_plant", "native.eat_food",
                   "native.eat_cow", "native.eat_bat", "native.eat_snail"):
            a = self._seek_and_do(map_payload, summary, FOOD_BLOCKS)
            if a is not None:
                return a
            a = self._seek_and_eat_mob(map_payload, summary)
            if a is not None:
                return a
            # 无熟植物也无被动怪：等待刷新（先保持存活）
            return SLEEP
        # 合成
        if tid in CRAFT_ACTIONS:
            return self._craft_action(tid, map_payload, summary)
        # 放置
        if tid in PLACE_ACTIONS:
            return self._place_action(tid, map_payload, summary)
        # 开箱：宝箱在 L1/L3/L4；当前层无宝箱先下楼
        if tid in ("native.open_chest", "native.find_bow"):
            floor = int(summary.get("floor", 0))
            if not map_payload.get("chest_positions") and floor < 1:
                return self._descend_to(map_payload, summary, 1)
            return self._open_chest(map_payload, summary)
        # 下楼 / 到达层
        if tid in ENTER_FLOOR or tid in REACH_FLOOR:
            return self._descend_to(map_payload, summary, REACH_FLOOR.get(tid, ENTER_FLOOR[tid]))
        # 战斗
        if tid in DEFEAT_TASKS:
            return self._combat(map_payload, summary, tid)
        # 学法术：先到书层开箱拿书，再 READ_BOOK（可上可下）
        if tid in LEARN_FLOOR:
            return self._learn_spell(map_payload, summary, tid)
        # 施法：需蓝（mana>=2）
        if tid in CAST_ACTIONS:
            return self._cast_spell(map_payload, summary, tid)
        # 附魔：需到附魔台旁 + 满蓝 + 宝石
        if tid in ENCHANT_ACTIONS:
            return self._enchant_action(map_payload, summary, tid)
        # 射弓：需弓 + 箭
        if tid == "native.fire_bow":
            return self._fire_bow(map_payload, summary)
        # 喝药水
        if tid == "native.drink_potion":
            return self._drink_potion(map_payload, summary)
        # 其他：尝试 DO（万不得已）
        return None

    # -- 通用：走到某类方块旁并 DO -----------------------------------------

    def _collect_resource(
        self,
        tid: str,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Optional[int]:
        """采集任务：本层目标数量充足则采，不足则到更合适的矿石层（可下可上）。"""
        floor = int(summary.get("floor", 0))
        target_types = COLLECT_TARGET_BLOCKS[tid]
        map2d = map_payload["map"]
        # 统计本层目标方块数量
        count = 0
        for row in map2d:
            for tile in row:
                if int(tile) in target_types:
                    count += 1
        preferred = COLLECT_TARGET_FLOORS.get(tid, [floor])
        target_need = RESOURCE_TARGETS.get(tid, (None, 1))[1]
        if tid == self.task_id:
            target_need = 1
        # 本层目标充足且可采 → 就地采
        if count >= target_need:
            a = self._seek_and_do(map_payload, summary, target_types)
            if a is not None:
                self._collect_visited = set()
                return a
            # 本层目标方块都是不可达的（WATER 分隔）：按不足处理，去下一优先层
        # 本层不足或目标不可达 → 去下一个优先层（深/浅皆可；已去过的层不再去，防上下打转）
        if getattr(self, "_collect_visited_tid", None) != tid:
            self._collect_visited = set()
            self._collect_visited_tid = tid
        self._collect_visited.add(floor)
        target = None
        for f in preferred:
            if f != floor and f not in self._collect_visited:
                target = f
                break
        if target is None:
            # 所有优先层都去过：留本层继续采（尽力而为），不再换层防打转
            return self._seek_and_do(map_payload, summary, target_types)
        return self._move_to_floor(map_payload, summary, target)

    def _seek_and_do(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        target_types: Sequence[int],
    ) -> Optional[int]:
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        h, w = len(map2d), len(map2d[0])

        def tile(p: Tuple[int, int]) -> int:
            return int(map2d[p[0]][p[1]])

        # 已站在目标旁且面向它 → DO
        for delta, action in DELTA_TO_ACTION.items():
            adj = (pos[0] + delta[0], pos[1] + delta[1])
            if 0 <= adj[0] < h and 0 <= adj[1] < w and tile(adj) in target_types:
                if direction == action:
                    return DO
                return action  # 转向
        result = find_nearest_target(
            map2d, pos, list(target_types), extra_blocked=getattr(self, "_mob_cells", None)
        )
        if result is None:
            # mob 挡路（临时）：忽略怪可达则等待怪移动/消失，避免误判无目标
            blind = find_nearest_target(map2d, pos, list(target_types))
            if blind is not None:
                return SLEEP
            return None
        _, first_delta = result
        if first_delta in DELTA_TO_ACTION:
            return DELTA_TO_ACTION[first_delta]
        return None

    # -- 合成：走到工作台/熔炉旁 → 按制作动作 --------------------------------

    def _craft_action(
        self, tid: str, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        make_action, need_furnace = CRAFT_ACTIONS[tid]
        inventory = summary.get("inventory") or {}

        # 合成前检查资源：不足则就地补采（对应 collect_* 任务）
        for resource, cost in (CRAFT_RESOURCE_COSTS.get(tid) or {}).items():
            if int(inventory.get(resource, 0)) >= cost:
                continue
            collect_tid = CRAFT_RESOURCE_COLLECT_TASK.get(resource)
            if collect_tid is None:
                continue
            a = self._collect_resource(collect_tid, map_payload, summary)
            if a is not None:
                return a
            break  # 目标采不到：先做别的

        # 已在工作台 8 邻域内 → 按制作键（铁系需熔炉也在邻域）
        if self._near_any(map2d, pos, [CRAFT_TABLE_BLOCK]):
            if need_furnace and not self._near_any(map2d, pos, [FURNACE_BLOCK]):
                # 熔炉通常就在工作台旁（place_furnace 优先放台旁），但可能在玩家
                # 8 邻域之外：先走到熔炉旁（其旁一般也含工作台）
                result = find_nearest_target(
                    map2d, pos, [FURNACE_BLOCK], extra_blocked=getattr(self, "_mob_cells", None)
                )
                if result is not None:
                    _, first_delta = result
                    if first_delta in DELTA_TO_ACTION:
                        return DELTA_TO_ACTION[first_delta]
                # 本层没有熔炉：就地放一个（_place_action 优先放工作台旁）
                return self._place_action("native.place_furnace", map_payload, summary)
            return make_action
        # 附近无工作台 → 就地放一个（深层也能合成，不依赖地表基础设施）
        return self._place_action("native.place_table", map_payload, summary)

    def _near_any(self, map2d, pos, types) -> bool:
        h, w = len(map2d), len(map2d[0])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                p = (pos[0] + dx, pos[1] + dy)
                if 0 <= p[0] < h and 0 <= p[1] < w and int(map2d[p[0]][p[1]]) in types:
                    return True
        return False

    # -- 放置：找可放置格 → 面向 → 按 PLACE_* ------------------------------

    def _place_action(
        self, tid: str, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        from craftax.planner.path_planner import _find_placeable_target

        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        place_action = PLACE_ACTIONS[tid]
        inventory = summary.get("inventory") or {}

        # 放置所需物品检查：不足则先采集
        if place_action == PLACE_FURNACE and int(inventory.get("stone", 0)) < 1:
            return self._collect_resource("native.collect_stone", map_payload, summary)
        if place_action == PLACE_TABLE and int(inventory.get("wood", 0)) < 2:
            return self._collect_resource("native.collect_wood", map_payload, summary)
        if place_action == PLACE_PLANT and int(inventory.get("sapling", 0)) < 1:
            # 树苗：DO 在草地上（10% 概率）
            return self._seek_and_do(map_payload, summary, [GRASS])
        if place_action == PLACE_TORCH and int(inventory.get("torches", 0)) < 1:
            craft = self._craft_action("native.craft_torch", map_payload, summary)
            if craft is not None:
                return craft

        # 优先在已有工作台旁放置（便于后续 craft_iron 等同时接近 table+furnace）
        result = self._find_place_near_table(map2d, pos)
        if result is None:
            result = _find_placeable_target(
                map2d, pos, extra_blocked=getattr(self, "_mob_cells", None)
            )
        if result is None:
            return None
        target, first_delta = result
        if first_delta is None:
            facing = DELTA_TO_ACTION.get(
                (target[0] - pos[0], target[1] - pos[1])
            )
            if facing is None:
                return None
            if direction == facing:
                return place_action
            return facing
        return DELTA_TO_ACTION[first_delta]

    def _find_place_near_table(self, map2d, pos):
        """找已有工作台旁的 8 邻域内可放置格（供 PLACE_FURNACE 等）。

        返回 (可放置格, 从 pos 到其相邻可走格的第一步 delta 或 None)。
        没有工作台或无可放置格时返回 None。
        """
        h, w = len(map2d), len(map2d[0])

        def tile(p):
            return int(map2d[p[0]][p[1]])

        # 找到所有工作台位置
        tables = []
        for x in range(h):
            for y in range(w):
                if tile((x, y)) == CRAFT_TABLE_BLOCK:
                    tables.append((x, y))
        if not tables:
            return None
        # 工作台 8 邻域内的可放置格
        candidates = []
        for t in tables:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    p = (t[0] + dx, t[1] + dy)
                    if 0 <= p[0] < h and 0 <= p[1] < w and tile(p) in PLACEABLE_TILES:
                        candidates.append(p)
        if not candidates:
            return None
        # 最近候选
        best = min(candidates, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
        # 已在其旁
        delta = (best[0] - pos[0], best[1] - pos[1])
        if delta in DELTA_TO_ACTION:
            return best, None
        # BFS 走向其相邻可走格
        from collections import deque

        blocked_extra = getattr(self, "_mob_cells", None) or set()
        visited = {pos}
        queue = deque([(pos, None)])
        while queue:
            cell, first_delta = queue.popleft()
            for d in ACTION_DELTA.values():
                nxt = (cell[0] + d[0], cell[1] + d[1])
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in visited:
                    continue
                if nxt == best:
                    return best, (first_delta or d)
                if not blocked(tile(nxt)) and nxt not in blocked_extra:
                    visited.add(nxt)
                    queue.append((nxt, first_delta or d))
        return None

    # -- 开箱：找 CHEST → 面向 → DO ---------------------------------------

    def _open_chest(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        chests = map_payload.get("chest_positions", [])
        if not chests:
            return None
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        h, w = len(map2d), len(map2d[0])
        # 最近宝箱
        best = None
        best_dist = 10 ** 9
        for c in chests:
            cpos = _norm_pos(c)
            dist = abs(cpos[0] - pos[0]) + abs(cpos[1] - pos[1])
            if dist < best_dist:
                best_dist = dist
                best = cpos
        if best is None:
            return None
        # 已相邻
        delta = (best[0] - pos[0], best[1] - pos[1])
        if delta in DELTA_TO_ACTION:
            if direction == DELTA_TO_ACTION[delta]:
                return DO
            return DELTA_TO_ACTION[delta]
        # 走向它
        result = find_nearest_target(
            map2d, pos, [CHEST], extra_blocked=getattr(self, "_mob_cells", None)
        )
        if result is None:
            return None
        _, first_delta = result
        if first_delta in DELTA_TO_ACTION:
            return DELTA_TO_ACTION[first_delta]
        return None

    # -- 下楼 / 上楼 / 到达指定楼层 ------------------------------------------

    def _move_to_floor(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], target_floor: int
    ) -> Optional[int]:
        """向下或向上移动到指定楼层；已到达目标层返回 None。"""
        floor = int(summary.get("floor", 0))
        if floor == target_floor:
            return None
        if floor < target_floor:
            return self._descend_to(map_payload, summary, target_floor)
        return self._ascend_to(map_payload, summary, target_floor)

    def _descend_to(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], target_floor: int
    ) -> Optional[int]:
        floor = int(summary.get("floor", 0))
        if floor >= target_floor:
            return None  # 已在目标层
        inventory = summary.get("inventory") or {}
        monsters_killed = int(map_payload.get("monsters_killed", 0))

        # 下楼前储备生存资源（任何下行前；深层水源/食物可能稀缺）。
        # 下 L6(火界)/L8(Boss) 前强制喝满 9（这两层无水源），
        # 下 L7(冰界) 前强制吃满 9（该层无被动怪、无植物）。
        drink = float(summary.get("drink", 9.0))
        food = float(summary.get("food", 9.0))
        next_floor = floor + 1
        drink_thresh = 9 if next_floor in NO_DRINK_FLOORS else 7
        food_thresh = 9 if next_floor in NO_FOOD_FLOORS else 7
        if drink < drink_thresh:
            a = self._seek_and_do(map_payload, summary, DRINK_BLOCKS)
            if a is not None:
                return a
        if food < food_thresh:
            a = self._seek_and_do(map_payload, summary, FOOD_BLOCKS)
            if a is not None:
                return a
            a = self._seek_and_eat_mob(map_payload, summary)
            if a is not None:
                return a
            if food < 5:
                return SLEEP  # 等被动怪刷新补充食物

        # 下地牢前确保有够用的近战武器：深层怪血厚攻高，木剑(2伤)清不动。
        # 有石镐(可采铁)时优先铁剑(5伤，兽人 9HP 2 击可杀)；否则至少石剑(3伤)。
        pickaxe_level = int(inventory.get("pickaxe", 0))
        target_sword = 1  # wood
        if pickaxe_level >= 2 and target_floor >= 1:
            target_sword = 3  # iron
        elif pickaxe_level >= 1 and target_floor >= 1:
            target_sword = 2  # stone
        sword_level = int(inventory.get("sword", 0))
        if sword_level < target_sword:
            craft_tid = {
                1: "native.craft_wood_sword",
                2: "native.craft_stone_sword",
                3: "native.craft_iron_sword",
                4: "native.craft_diamond_sword",
            }.get(target_sword, "native.craft_stone_sword")
            craft = self._craft_action(craft_tid, map_payload, summary)
            if craft is not None:
                return craft
            # 目标剑做不了（缺资源）：至少保证有木剑
            if sword_level < 1:
                craft = self._craft_action("native.craft_wood_sword", map_payload, summary)
                if craft is not None:
                    return craft

        # 下地牢前尽量穿铁甲（4 件 40% 物抗，深层清怪生存关键）。
        # 浅层铁/煤充足时在出发地做；不足则放弃（深层种子铁/煤更丰富，到时再做）。
        armour_levels = [int(x) for x in inventory.get("armour", [0])]
        if (sum(armour_levels) < 1 and int(inventory.get("iron", 0)) >= 3
                and int(inventory.get("coal", 0)) >= 3):
            craft = self._craft_action("native.craft_iron_armour", map_payload, summary)
            if craft is not None:
                return craft

        # 下火界(L6)/冰界(L7)前需元素战斗能力（剑/弓附魔或对应法术），
        # 否则无附魔硬打（90% 物抗）几乎必死。无法获取 → 放弃该 seed。
        if next_floor in (6, 7) and not self._has_elemental_capability(summary, next_floor):
            a = self._acquire_elemental_capability(map_payload, summary, next_floor)
            if a is not None:
                return a
            return None

        # 若本层未清 8 怪，需清怪才能下楼
        if floor > 0 and monsters_killed < 8:
            return self._combat_any(map_payload, summary)
        # 站到 LADDER_DOWN 上并按 DESCEND
        ladder = map_payload.get("ladder_down")
        if ladder is None:
            return None
        ladder_pos = _norm_pos(ladder)
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        if pos == ladder_pos:
            return DESCEND
        map2d = map_payload["map"]
        h, w = len(map2d), len(map2d[0])
        if not (0 <= ladder_pos[0] < h and 0 <= ladder_pos[1] < w):
            return None
        a = self._walk_to(map2d, pos, ladder_pos)
        if a is not None:
            return a
        # 梯子被怪挡住：打掉近身怪；无近身怪则原地站着等怪移动/消失
        # （不用 SLEEP——SLEEP 会锁死动作直到能量回满或被怪 3.5x 打醒）。
        if self._mob_within(map_payload, summary, 2):
            combat = self._combat_any(map_payload, summary)
            if combat is not None:
                return combat
        if self._walk_to(map2d, pos, ladder_pos, mob_aware=False) is not None:
            return NOOP  # 等挡路怪移动
        return None

    def _ascend_to(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], target_floor: int
    ) -> Optional[int]:
        """上行到指定楼层（走回 ladder_up 并按 ASCEND）。L0 无 ladder_up，只能上到 L0。"""
        floor = int(summary.get("floor", 0))
        if floor <= target_floor:
            return None  # 已在目标层或更浅
        ladder = map_payload.get("ladder_up")
        if ladder is None:
            return None
        ladder_pos = _norm_pos(ladder)
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        if pos == ladder_pos:
            return ASCEND
        map2d = map_payload["map"]
        h, w = len(map2d), len(map2d[0])
        if not (0 <= ladder_pos[0] < h and 0 <= ladder_pos[1] < w):
            return None
        a = self._walk_to(map2d, pos, ladder_pos)
        if a is not None:
            return a
        # 梯子被怪挡住：打掉近身怪；无近身怪则原地等怪移动
        if self._mob_within(map_payload, summary, 2):
            combat = self._combat_any(map_payload, summary)
            if combat is not None:
                return combat
        if self._walk_to(map2d, pos, ladder_pos, mob_aware=False) is not None:
            return NOOP
        return None

    @staticmethod
    def _mob_blocked_cells(map_payload: Dict[str, Any]) -> Set[Tuple[int, int]]:
        """本层存活怪占用的格子（怪会挡住玩家移动，寻路时不可通行）。"""
        cells: Set[Tuple[int, int]] = set()
        mobs = map_payload.get("mob_positions", {})
        for key in ("melee", "ranged", "passive"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                cells.add(_norm_pos(p))
        return cells

    def _walk_to(
        self,
        map2d,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        mob_aware: bool = True,
    ) -> Optional[int]:
        """BFS 从 start 走到 goal（goal 自身可能为可走格），返回第一步动作。

        中间格被怪占用时绕路（怪会挡路）；goal 本身即使被怪占用也允许前往
        （怪每步会移动/消失，下一拍会重新规划）。mob_aware=False 时忽略怪。
        """
        blocked_extra = getattr(self, "_mob_cells", None) or set() if mob_aware else set()
        h, w = len(map2d), len(map2d[0])
        if start == goal:
            return None
        from collections import deque

        def tile(p):
            return int(map2d[p[0]][p[1]])

        visited = {start}
        queue = deque([(start, None)])
        while queue:
            cell, first_delta = queue.popleft()
            for delta in ACTION_DELTA.values():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in visited:
                    continue
                if nxt == goal:
                    if not mob_aware or nxt not in blocked_extra:
                        return DELTA_TO_ACTION[first_delta or delta]
                    continue  # goal 被怪占：暂时不可达，绕路/等待怪移动
                if blocked(tile(nxt)):
                    continue
                if nxt in blocked_extra:
                    continue
                visited.add(nxt)
                queue.append((nxt, first_delta or delta))
        return None

    # -- 战斗：找目标怪 → 走近 → 面向 → DO ---------------------------------

    def _combat(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], tid: str
    ) -> Optional[int]:
        """按目标怪所在楼层导航后再战斗（执行器层绕过依赖图的楼层不符）。"""
        floor = int(summary.get("floor", 0))
        loc = DEFEAT_MOB_LOCATIONS.get(tid)
        if loc is None:
            return self._combat_any(map_payload, summary)
        mob_class, _mob_type, floors = loc
        if mob_class == "boss":
            # Boss 战：先下到 L8（链上 enter_graveyard 已保证，防御性补一步）
            if floor < 8:
                return self._descend_to(map_payload, summary, 8)
            return self._combat_boss(map_payload, summary)
        # 目标怪不在当前层 → 导航到最近的目标层（可上可下）
        if floor not in floors:
            target = min(floors, key=lambda f: abs(f - floor))
            return self._move_to_floor(map_payload, summary, target)
        return self._combat_any(map_payload, summary)

    def _combat_boss(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """L8 Boss 战：有活怪先清波次；无活怪则 DO 亡灵法师方块（vulnerable 窗口）。"""
        floor = int(summary.get("floor", 0))
        if floor != 8:
            return None
        mobs = map_payload.get("mob_positions", {})
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i in range(len(masks)):
                if masks[i]:
                    return self._combat_any(map_payload, summary)
        # 无活怪：Boss vulnerable → DO 亡灵法师方块
        return self._do_necromancer(map_payload, summary)

    def _do_necromancer(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """走向亡灵法师方块旁，面向它并 DO（积累 boss_progress）。"""
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        h, w = len(map2d), len(map2d[0])
        # 已站在方块旁：面向它则 DO，否则转向
        for delta, action in DELTA_TO_ACTION.items():
            adj = (pos[0] + delta[0], pos[1] + delta[1])
            if 0 <= adj[0] < h and 0 <= adj[1] < w and int(map2d[adj[0]][adj[1]]) == NECROMANCER_BLOCK:
                if direction == action:
                    return DO
                return action
        result = find_nearest_target(
            map2d, pos, [NECROMANCER_BLOCK], extra_blocked=getattr(self, "_mob_cells", None)
        )
        if result is None:
            return None
        _, first_delta = result
        if first_delta in DELTA_TO_ACTION:
            return DELTA_TO_ACTION[first_delta]
        return None

    def _combat_any(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """找最近的、玩家可达的怪，走近并 DO。

        清怪（monsters_killed<8）时优先主动怪（melee/ranged），被动怪只用于进食；
        无主动怪且不饿时 SLEEP 等刷怪（sleep 更省食物/水消耗且可回蓝回血）。
        """
        mobs = map_payload.get("mob_positions", {})
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        map2d = map_payload["map"]
        reachable = self._reachable_set(map2d, pos)

        hostiles: List[Tuple[int, int]] = []
        passives: List[Tuple[int, int]] = []
        ranged_set: Set[Tuple[int, int]] = set()
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                c = _norm_pos(p)
                if c in reachable:
                    hostiles.append(c)
                    if key == "ranged":
                        ranged_set.add(c)
        entry = mobs.get("passive", {})
        masks = entry.get("masks", [])
        for i, p in enumerate(entry.get("positions", [])):
            if i < len(masks) and not masks[i]:
                continue
            c = _norm_pos(p)
            if c in reachable:
                passives.append(c)

        food = float(summary.get("food", 9.0))
        monsters_killed = int(map_payload.get("monsters_killed", 0))
        clearing = monsters_killed < 8
        if hostiles:
            candidates = hostiles
        elif passives and (food < 4 or not clearing):
            candidates = passives
        else:
            return SLEEP  # 无可攻击目标：等待刷怪（顺带回血回蓝）

        # 远程怪必须追（远距离射击威胁，等它接近只会持续挨打）；
        # 近战怪 5 格内能走到则追，更远 → SLEEP 等其接近（追太远既慢又易被包抄）。
        chase_limit = 5
        for c in sorted(
            candidates,
            key=lambda c: (
                0 if c in ranged_set else 1,
                abs(c[0] - pos[0]) + abs(c[1] - pos[1]),
            ),
        ):
            dist = abs(c[0] - pos[0]) + abs(c[1] - pos[1])
            delta = (c[0] - pos[0], c[1] - pos[1])
            if delta in DELTA_TO_ACTION:
                if direction == DELTA_TO_ACTION[delta]:
                    return DO
                return DELTA_TO_ACTION[delta]
            if c in ranged_set or dist <= chase_limit:
                walk = self._walk_to_mob(map2d, pos, c)
                if walk is not None:
                    return walk
            if dist > chase_limit and c not in ranged_set:
                break  # 超出近战追击距离：等近战怪接近
        return SLEEP

    def _reachable_set(
        self, map2d, start: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """BFS 返回从 start 可达（非 blocked）的格子集合。

        注意：不排除怪占用的格子——本集合用于判断"哪些怪可被攻击/吃"，怪格本身
        是目标而非可走格；走路时另由 _walk_to 等按 _mob_cells 绕路。
        """
        from collections import deque

        h, w = len(map2d), len(map2d[0])

        def tile(p):
            return int(map2d[p[0]][p[1]])

        visited = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for d in ACTION_DELTA.values():
                nxt = (cell[0] + d[0], cell[1] + d[1])
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in visited:
                    continue
                if not blocked(tile(nxt)):
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _walk_to_mob(
        self, map2d, start: Tuple[int, int], mob: Tuple[int, int]
    ) -> Optional[int]:
        """走到 mob 相邻格（mob 格不可走；中间格被其他怪占用时绕路）。"""
        blocked_extra = getattr(self, "_mob_cells", None) or set()
        h, w = len(map2d), len(map2d[0])
        from collections import deque

        def tile(p):
            return int(map2d[p[0]][p[1]])

        # 先看是否已在 mob 相邻格
        for delta in ACTION_DELTA.values():
            nxt = (start[0] + delta[0], start[1] + delta[1])
            if nxt == mob:
                return DELTA_TO_ACTION[delta]
        visited = {start}
        queue = deque([(start, None)])
        while queue:
            cell, first_delta = queue.popleft()
            for delta in ACTION_DELTA.values():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in visited:
                    continue
                if nxt == mob:
                    return DELTA_TO_ACTION[first_delta or delta]
                if not blocked(tile(nxt)) and nxt not in blocked_extra:
                    visited.add(nxt)
                    queue.append((nxt, first_delta or delta))
        return None

    # -- 学法术 ------------------------------------------------------------

    def _learn_spell(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], tid: str
    ) -> Optional[int]:
        """到书层（L3/L4）开箱拿书并 READ_BOOK。

        READ_BOOK 随机学一种未学法术；若第一次学了另一术，去另一书层拿第二本
        再读（两本必学全两术）。链中 enter_fire/ice_realm 会先下到 L6/L7，
        此处用 _move_to_floor 可上行回书层。
        """
        floor = int(summary.get("floor", 0))
        target = LEARN_FLOOR[tid]
        other = 4 if target == 3 else 3
        books = int((summary.get("inventory") or {}).get("books", 0))
        if books > 0:
            return READ_BOOK
        if floor != target:
            return self._move_to_floor(map_payload, summary, target)
        # 在书层但没书：开箱拿书（L3/L4 首箱必给书）
        chest_action = self._open_chest(map_payload, summary)
        if chest_action is not None:
            return chest_action
        # 本层无未开宝箱：去另一书层
        return self._move_to_floor(map_payload, summary, other)

    # -- 施法 / 附魔 / 射弓 ------------------------------------------------

    def _cast_spell(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], tid: str
    ) -> Optional[int]:
        """施法：需 mana>=2，否则睡觉回蓝。"""
        if float(summary.get("mana", 0)) < 2:
            return SLEEP
        return CAST_ACTIONS[tid]

    def _fire_bow(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """射弓：需弓（L1 首箱）与箭（就地合成）。"""
        inventory = summary.get("inventory") or {}
        if int(inventory.get("bow", 0)) < 1:
            floor = int(summary.get("floor", 0))
            if floor < 1:
                return self._descend_to(map_payload, summary, 1)
            return self._open_chest(map_payload, summary)
        if int(inventory.get("arrows", 0)) < 1:
            return self._craft_action("native.craft_arrow", map_payload, summary)
        return SHOOT_ARROW

    def _enchant_action(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], tid: str
    ) -> Optional[int]:
        """附魔：选台（ruby→L4 火台、sapphire→L3 冰台）+ 满蓝 + 宝石后按 ENCHANT_*。

        任务依赖图未声明宝石前置，这里自行获取：优先开宝箱（L1/L3/L4 箱子
        约 7.5% 概率出宝石），开完所有宝箱仍无再回退（多数 seed 24 箱够出 1 颗）。
        """
        inventory = summary.get("inventory") or {}
        ruby = int(inventory.get("ruby", 0))
        sapphire = int(inventory.get("sapphire", 0))
        if ruby >= 1:
            return self._enchant_at(map_payload, summary, 4, ENCHANT_TABLE_FIRE, tid)
        if sapphire >= 1:
            return self._enchant_at(map_payload, summary, 3, ENCHANT_TABLE_ICE, tid)
        # 无宝石：先开当前层宝箱，再去最近宝箱层（L1/L3/L4）
        floor = int(summary.get("floor", 0))
        chest_action = self._open_chest(map_payload, summary)
        if chest_action is not None:
            return chest_action
        if floor not in (1, 3, 4):
            target = min((1, 3, 4), key=lambda f: abs(f - floor))
            return self._move_to_floor(map_payload, summary, target)
        return None

    def _enchant_at(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        target_floor: int,
        table_block: int,
        tid: str,
    ) -> Optional[int]:
        """到指定附魔台：先到位 → 满蓝 → 站在台旁面向它 → 按 ENCHANT_SWORD/ARMOUR。"""
        make_action = ENCHANT_ACTIONS[tid][0]
        floor = int(summary.get("floor", 0))
        if floor != target_floor:
            return self._move_to_floor(map_payload, summary, target_floor)
        # 附魔需 9 mana（满值），睡觉回蓝最快
        if float(summary.get("mana", 0)) < 9:
            return SLEEP
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        h, w = len(map2d), len(map2d[0])
        for delta, action in DELTA_TO_ACTION.items():
            adj = (pos[0] + delta[0], pos[1] + delta[1])
            if 0 <= adj[0] < h and 0 <= adj[1] < w and int(map2d[adj[0]][adj[1]]) == table_block:
                if direction == action:
                    return make_action
                return action
        # 走向附魔台
        result = find_nearest_target(
            map2d, pos, [table_block], extra_blocked=getattr(self, "_mob_cells", None)
        )
        if result is None:
            return None
        _, first_delta = result
        if first_delta in DELTA_TO_ACTION:
            return DELTA_TO_ACTION[first_delta]
        return None

    # -- 元素能力（火界/冰界）---------------------------------------------

    def _has_elemental_capability(
        self, summary: Dict[str, Any], target_floor: int
    ) -> bool:
        """目标层所需元素战斗能力是否已具备。

        L6(火界)怪 90% 物抗 + 100% 火免 → 需要冰系（剑/弓冰附魔 或 学会冰球）；
        L7(冰界)怪 90% 物抗 + 100% 冰免 → 需要火系（剑/弓火附魔 或 学会火球）。
        """
        need_ice = target_floor == 6
        sword_ench = int(summary.get("sword_enchantment", 0))
        bow_ench = int(summary.get("bow_enchantment", 0))
        spells = summary.get("learned_spells") or [False, False]
        fireball = bool(spells[0]) if len(spells) > 0 else False
        iceball = bool(spells[1]) if len(spells) > 1 else False
        if need_ice:
            return sword_ench == 2 or bow_ench == 2 or iceball
        return sword_ench == 1 or bow_ench == 1 or fireball

    def _acquire_elemental_capability(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], target_floor: int
    ) -> Optional[int]:
        """为下 L6/L7 获取元素战斗能力。

        优先学法术（只需 L3/L4 的书，两本必学全两术）；法术不可得再走剑附魔
        （需对应宝石 + 剑 + 满蓝）。都无法获取返回 None（调用方放弃该 seed）。
        """
        need_ice = target_floor == 6
        spells = summary.get("learned_spells") or [False, False]
        iceball = bool(spells[1]) if len(spells) > 1 else False
        fireball = bool(spells[0]) if len(spells) > 0 else False
        inventory = summary.get("inventory") or {}

        # 1) 学法术
        if need_ice and not iceball:
            a = self._learn_spell(map_payload, summary, "native.learn_iceball")
            if a is not None:
                return a
        if not need_ice and not fireball:
            a = self._learn_spell(map_payload, summary, "native.learn_fireball")
            if a is not None:
                return a

        # 2) 剑附魔（需剑 + 对应宝石）
        if int(inventory.get("sword", 0)) < 1:
            return None
        if need_ice:
            if int(inventory.get("sapphire", 0)) < 1:
                a = self._get_gem(map_payload, summary, "native.collect_sapphire")
                if a is not None:
                    return a
                return None
            return self._enchant_at(
                map_payload, summary, 3, ENCHANT_TABLE_ICE, "native.enchant_sword"
            )
        else:
            if int(inventory.get("ruby", 0)) < 1:
                a = self._get_gem(map_payload, summary, "native.collect_ruby")
                if a is not None:
                    return a
                return None
            return self._enchant_at(
                map_payload, summary, 4, ENCHANT_TABLE_FIRE, "native.enchant_sword"
            )

    def _get_gem(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], collect_tid: str
    ) -> Optional[int]:
        """获取一颗宝石：有钻石镐则挖，否则开宝箱碰运气（L1/L3/L4 箱子约 7.5% 出宝石）。"""
        inventory = summary.get("inventory") or {}
        if int(inventory.get("pickaxe", 0)) >= 4:
            a = self._collect_resource(collect_tid, map_payload, summary)
            if a is not None:
                return a
        a = self._open_chest(map_payload, summary)
        if a is not None:
            return a
        floor = int(summary.get("floor", 0))
        if floor not in (1, 3, 4):
            target = min((1, 3, 4), key=lambda f: abs(f - floor))
            return self._move_to_floor(map_payload, summary, target)
        return None

    # -- 喝药水 ------------------------------------------------------------

    def _drink_potion(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        potions = (summary.get("inventory") or {}).get("potions", [])
        for i, count in enumerate(potions):
            if int(count) > 0:
                return DRINK_POTION_RED + i
        return None


# 任务注册：供 demo 脚本按任务选择 executor
EXECUTOR_FACTORIES: Dict[str, Any] = {}


def make_executor(task_id: str) -> SkillChainExecutor:
    return SkillChainExecutor(task_id)
