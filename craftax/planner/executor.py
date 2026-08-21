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
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from craftax.planner.path_planner import (
    ACTION_DELTA,
    CRAFTING_TABLE_TILE,
    DELTA_TO_ACTION,
    PICKAXE_REQUIRED,
    PLACEABLE_TILES,
    blocked,
    find_nearest_target,
)
from craftax.planner.combat_model import (
    ClearPrep,
    Gear,
    energy_is_bottleneck,
    estimated_steps_bow,
    project_sleep,
    projected_awake_health,
    recommend_tactic,
)
from craftax.contracts import (
    DEFAULT_THIRST_RATE,
    MAX_WATER_CANTEENS,
    WATER_DRINK_AMOUNT,
)
from craftax.planner.planner import check_floor_readiness, has_elemental
from craftax.planner.shelter import (
    MIN_SHELTER_WALLS,
    find_cover_tile,
    find_dig_pocket,
    seal_target,
    wall_count,
)
from craftax.planner.world import WorldFacts

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
REST = 17
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
ENCHANT_BOW = 42
FILL_WATER = 43
DRINK_WATER = 44
MAKE_TORCH = 38
LEVEL_UP_DEXTERITY = 39
LEVEL_UP_STRENGTH = 40
LEVEL_UP_INTELLIGENCE = 41

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

# 附魔任务 -> 附魔台所在楼层（冰台 L3 SEWER_CONFIG / 火台 L4 VAULTS_CONFIG；
# 取较浅的冰台作为规划目标层，执行器按手上宝石选台）
ENCHANT_FLOOR: Dict[str, int] = {
    "native.enchant_sword": 3,
    "native.enchant_armour": 3,
}

# 矿石方块 -> seed 扫描数据（WorldFacts）中的键，用于跨层"该层有没有这种矿"查询
ORE_BLOCK_TO_KEY: Dict[int, str] = {
    COAL: "coal",
    IRON: "iron",
    DIAMOND: "diamond",
    SAPPHIRE: "sapphire",
    RUBY: "ruby",
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
    "native.collect_iron": ("iron", 1),
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

# 消耗品合成：完成判定看背包数量而非一次性的 MAKE_* 成就（箭射掉就得再造）
CONSUMABLE_CRAFTS: Set[str] = {"native.craft_arrow", "native.craft_torch"}

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
    "native.build_shelter",
}

# 可食用方块（DO 面向进食）
FOOD_BLOCKS = [RIPE_PLANT]
# 可饮水方块
DRINK_BLOCKS = [WATER, FOUNTAIN]
# 便携水的容量和每瓶恢复量与游戏逻辑共享，避免规划器和环境漂移。
WATER_RESERVE_TARGET = MAX_WATER_CANTEENS

# 弓（L1 首箱必出）相关
BOW_ARROW_RESERVE = 8               # 箭数低于此值就补（wood+stone 在台上合成）

# 近战射程：持剑时 DO 也能打到正前方第二格（game_logic.do_action 的 sword_reach，
# 要求中间格通透）。规划器必须知道这个数——否则两格外的怪会被"走近再打"，
# 而走到相邻格就必然先吃一次怪的首击，正好抵消了加射程的收益。
SWORD_REACH = 2


# 停滞检测阈值（无进展 = 楼层/成就/背包/击杀数都不变）
STALL_NO_PROGRESS_STEPS = 150   # 连续无进展步数 → 判定停滞，开始打破
STALL_REPEAT_ACTIONS = 40       # 同一 (楼层,位置,朝向,动作) 重复次数 → 判定停滞
STALL_ABORT_STEPS = 600         # 停滞持续到此步数 → 放弃该 seed（abort_reason）

# 链推进预算（与上面的停滞检测互补，抓的是完全不同的故障）
#
# 停滞检测的进展签名含**位置与背包**：走一步、捡一根木头就算"有进展"。这对
# "原地抽搐"有效，但对"在浅层无限生存+采集"完全无效——实测一局 1615 步里
# 1539 步在 L0、楼层三次触到 1 又退回，而 _stall_steps 从未累积到 150。
# 因此这里再加一层**只看任务链是否真的前进**的预算：签名刻意只含
# (楼层, chain_idx, 成就数)，位置和背包一律不算进展。
CHAIN_PUSH_STEPS = 250          # 无链推进达此步数 → 进入"推进优先"模式
# 标定：正常制备期最长的一段无成就/无链推进间隔实测为 237 步（429→666，做铁甲），
# 故 PUSH 取 250，刚好放过正常制备。
#
# 刻意**没有**"无链推进就放弃 seed"这一档。试过 CHAIN_ABORT_STEPS=600，被实测
# 否掉：collect_diamond 逐 seed 实测的最长无推进间隔为
#   2026:210  2027:225  2028:129  2011:189  2111:1032  3017:3541(唯一通关)
# ——深挖类任务本来就会有几千步"楼层/子目标/成就都不动"的正常区间，任何对浅层
# 生存循环有意义的阈值（250-600）都会把唯一能通关的 seed 判死。硬死锁（位置与
# 背包也不再变化）已由 STALL_ABORT_STEPS 覆盖，这里只做优先级调整，不做放弃。
#
# 同理，推进优先模式是**脉冲**而不是常开状态：常开会在 3017 那样的 3541 步
# 区间里全程关掉回血/驻守，把"推不动"换成"死得快"。
PUSH_MAX_STEPS = 150            # 单次推进优先窗口的步数预算
PUSH_COOLDOWN_STEPS = 250       # 窗口结束后的冷却：让恢复类行为重新可用

# 睡眠投影门槛：睡醒血量低于此值就不睡（渴/饿着睡会掉血且无法自救）
SLEEP_MIN_PROJECTED_HEALTH = 3.0


def _norm_pos(p: Any) -> Tuple[int, int]:
    return (int(p[0]), int(p[1]))


class _SummaryPredicateState:
    """把 state_summary（dict）包装成 tasks.base.eval_predicate 可读的 state。

    builtin 任务的成功谓词只用到 achievement / level_ge / and / or / always
    （achievement 走 info["achievements_list"]，level_ge 读 player_level），
    因此执行器不必持有 EnvState 也能用**权威谓词**判定完成。
    若将来出现 field_* 谓词，在此按 path 补映射即可。
    """

    __slots__ = ("_summary",)

    def __init__(self, summary: Dict[str, Any]) -> None:
        self._summary = summary

    @property
    def player_level(self) -> int:
        return int(self._summary.get("floor", 0))

    @property
    def inventory(self) -> Dict[str, Any]:
        return self._summary.get("inventory") or {}

    @property
    def player_health(self) -> float:
        return float(self._summary.get("health", 0.0))

    def __getattr__(self, name: str) -> Any:  # 兜底：summary 里的同名字段
        summary = object.__getattribute__(self, "_summary")
        if name in summary:
            return summary[name]
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# 任务 -> 目标楼层 / 成本感知的任务链排序
# ---------------------------------------------------------------------------


def _preferred_collect_floor(
    tid: str,
    current_floor: int,
    world_facts: Optional[WorldFacts],
) -> int:
    """采集任务的目标层：优先"已知有该矿且最近"的层，未知则按静态偏好首位。

    这是把 seed 事实（跨层矿石计数）真正用进规划的地方：没有事实时行为与
    过去一致（偏好列表首位），有事实时会跳过"已知没有"的层。
    """
    preferred = COLLECT_TARGET_FLOORS.get(tid) or [current_floor]
    if world_facts is None:
        return preferred[0]
    blocks = COLLECT_TARGET_BLOCKS.get(tid) or []
    keys = [ORE_BLOCK_TO_KEY[b] for b in blocks if b in ORE_BLOCK_TO_KEY]
    if not keys:
        return preferred[0]
    scored: List[Tuple[int, int, int, int]] = []
    for f in preferred:
        facts = world_facts.floor(f)
        if facts is None:
            scored.append((1, abs(f - current_floor), f, f))  # 未知：保留但排在已知之后
            continue
        count = sum(facts.ore_count(k) for k in keys)
        if count <= 0:
            continue  # 已知该层没有 → 不去
        scored.append((0, abs(f - current_floor), f, f))
    if not scored:
        return preferred[0]
    scored.sort()
    return scored[0][3]


def task_target_floor(
    tid: str,
    current_floor: int = 0,
    world_facts: Optional[WorldFacts] = None,
) -> Optional[int]:
    """任务需要在哪一层执行；与楼层无关（合成/放置/复合）返回 None。

    与执行器的实际导航保持一致：
    - 战斗任务取"距当前层最近的刷新层"（骷髅 [0, 8] 从地表出发即 L0，
      而不是 max=8——旧实现按 max 会把地表任务当成 Boss 层任务来制备）；
    - 采集任务取偏好层（可被 world_facts 修正）。
    """
    if tid in REACH_FLOOR:
        return REACH_FLOOR[tid]
    if tid in ENTER_FLOOR:
        return ENTER_FLOOR[tid]
    if tid in LEARN_FLOOR:
        return LEARN_FLOOR[tid]
    if tid in ENCHANT_FLOOR:
        return ENCHANT_FLOOR[tid]
    loc = DEFEAT_MOB_LOCATIONS.get(tid)
    if loc is not None and loc[2]:
        return min(loc[2], key=lambda f: (abs(f - current_floor), f))
    if tid in COLLECT_TARGET_FLOORS:
        return _preferred_collect_floor(tid, current_floor, world_facts)
    if tid in ("native.open_chest", "native.find_bow", "native.fire_bow"):
        return 1  # 宝箱/弓：L1 首箱确定性掉落
    return None


def build_task_chain(
    task_id: str,
    world_facts: Optional[WorldFacts] = None,
) -> List[str]:
    """把依赖闭包排成一条**成本感知**的任务链（确定性）。

    旧实现按 (topological_level, task_id) 排序——字母序决定同层任务先后，
    于是 "native.enter_gnomish_mines" < "native.place_table" 让计划变成
    "先下 L2、再回头放工作台做木镐"，深链上表现为反复上下楼。

    新排序是一次贪心拓扑调度：在依赖已满足的候选中，优先
      1) 与楼层无关的任务（合成/放置/复合）——就地就能做，顺手做完；
      2) 距"当前虚拟楼层"最近的楼层任务（同层的先做完，即"路过就做掉"），
         同距离时取较浅层；
      3) 拓扑层级、task_id（保证确定性）。
    每选中一个楼层任务就把虚拟楼层移到该层，于是采集→合成→再下行自然成组。
    """
    from craftax.tasks.graph import TaskGraph

    graph = TaskGraph.build_from_registry()
    closure = graph.closure(task_id, include_self=True)
    deps = {
        tid: [d for d in graph.dependencies(tid) if d in closure] for tid in closure
    }
    remaining: Set[str] = set(closure)
    done: Set[str] = set()
    chain: List[str] = []
    virtual_floor = 0
    while remaining:
        ready = sorted(t for t in remaining if all(d in done for d in deps[t]))
        if not ready:  # 理论不可达（图已校验无环）；保底按拓扑层级推进
            ready = [
                min(remaining, key=lambda t: (graph.node(t).topological_level, t))
            ]

        def _cost(t: str) -> Tuple[int, int, int, int, str]:
            floor = task_target_floor(t, virtual_floor, world_facts)
            level = graph.node(t).topological_level
            if floor is None:
                return (0, 0, 0, level, t)
            return (1, abs(floor - virtual_floor), floor, level, t)

        pick = min(ready, key=_cost)
        floor = task_target_floor(pick, virtual_floor, world_facts)
        if floor is not None:
            virtual_floor = floor
        chain.append(pick)
        done.add(pick)
        remaining.discard(pick)
    return chain


class SkillChainExecutor:
    """依赖图驱动的通用任务执行器。"""

    def __init__(self, task_id: str, *, max_steps: Optional[int] = None,
                 seed: Optional[int] = None,
                 thirst_rate: float = DEFAULT_THIRST_RATE,
                 energy_rate: float = 1.0,
                 floor_map_provider: Optional[
                     Callable[[int], Optional[Dict[str, Any]]]
                 ] = None) -> None:
        """floor_map_provider：按楼层号返回该层 /map 响应（或 None=取不到）。

        这是"扩大观察边界"的注入点：服务端 GET /map?floor=N 能给出任意层的
        完整 48×48 网格，执行器用它在**下楼之前**知道目标层有没有要采的矿，
        而不是靠静态偏好表猜。未注入时退回 seed 扫描事实，再退回静态表。
        """
        self.task_id = task_id
        self.seed = seed
        # 会话的口渴衰减倍率（EnvParams.thirst_rate）。投影类前瞻必须用同一个值，
        # 否则会话把水调慢后执行器仍按原版投影，会拒绝本来安全的睡眠/等待。
        self.thirst_rate = float(thirst_rate)
        self.energy_rate = float(energy_rate)
        self._floor_map_provider = floor_map_provider
        self._floor_map_cache: Dict[int, Optional[Dict[str, Any]]] = {}
        self._floor_map_cache_step: Dict[int, int] = {}
        self._steps = 0
        self._world_facts: Optional[WorldFacts] = None
        self._abort_reason: Optional[str] = None
        self._tactic: str = "stand"
        self._tactic_floor: Optional[int] = None
        # 弓补给目标：>0 时在已清层补箭到该数量（0=无需补给）
        self._restock_target = 0
        # 最近一次"备箭 vs 升剑"择优的依据（可读，供日志/调试）
        self._prep_note: Optional[str] = None
        # 风筝：怪攻击冷却计时（被命中后冷却重置 5）与撤退步数
        self._mob_attack_timer = 0
        self._prev_health: Optional[float] = None
        self._kite_retreats = 0
        # 防御性回血的滞回状态（模式 / 已用步数 / 冷却）
        self._regen_mode = False
        self._regen_steps = 0
        self._regen_cooldown = 0
        # 远程压制：最近一次"掉血但无近身怪"（= 被投射物命中）的步号
        self._ranged_hit_step = -10 ** 9
        # 夜间驻守的滞回状态（模式 / 已用步数 / 冷却），语义同 _regen_*
        self._night_hold = False
        self._night_hold_steps = 0
        self._night_hold_cooldown = 0
        # 上下楼振荡保护：为补给上行的冷却 + 最近的换层动作步号
        self._supply_ascend_block = 0
        self._ladder_action_steps: List[int] = []
        self._ladder_recent_actions: List[int] = []
        # 链推进预算：距上次"任务链真正前进"的步数。签名只含
        # (楼层, chain_idx, 成就数)，不含位置与背包（见 CHAIN_PUSH_STEPS）
        self._chain_sig: Optional[Tuple[int, int, int]] = None
        self._steps_since_chain_progress = 0
        # 推进优先模式：链推进预算耗尽或全局步数预算告急时置位（脉冲 + 冷却）
        self._push_mode = False
        self._push_steps = 0
        self._push_cooldown = 0
        # 下楼/推进的滞回latch：一旦开始推进就不因血量小幅波动而交还给生存维护
        self._progress_latched = False
        # 停滞检测（无进展步数 / 重复动作计数 / 打破停滞的方向轮转）
        self._progress_sig: Optional[Tuple[Any, ...]] = None
        self._stall_steps = 0
        self._repeat_counter: "Counter[Tuple[int, Tuple[int, int], int, int]]" = Counter()
        self._stall_nudge = 0
        self._stall_reset_done = False
        # 资源目标的绝对坐标黑名单：跨 chunk 后仍然有效，避免对同一棵树/矿石
        # 反复 DO。只有目标方块在交互后仍未改变才会进入黑名单。
        self._blocked_collect_targets: Set[Tuple[int, int, int, Tuple[int, ...]]] = set()
        self._pending_collect_target: Optional[Tuple[int, int, int, Tuple[int, ...]]] = None
        self._pending_collect_armed = False
        # 当前子任务的绝对目标锁。chunk 换窗只改变局部路径，不改变目标点；
        # 目标完成、确认失效或任务链前进后才释放。
        self._goal_locks: Dict[str, Tuple[int, int, int]] = {}
        # 目标锁之外还要缓存一段到目标旁边的绝对路径。仅锁目标而每步
        # 重新 BFS 会在多个等长路径之间来回换首步，表现为左右抽搐。
        self._goal_routes: Dict[str, List[Tuple[int, int]]] = {}
        self._chain: List[str] = []
        self._chain_idx = 0
        self._adapters: Dict[str, Any] = {}
        self._estimate_cache: Optional[int] = None
        self._build_chain()
        # 步数预算：未显式指定时按任务链自行估算。旧默认值是硬编码的 2000，而
        # **没有任何调用方传过 max_steps**，于是深层任务的"预算"从一开始就低于
        # 估算需求；把它接进决策（_budget_is_tight）前必须先让它有意义。
        self.max_steps = int(max_steps) if max_steps else self.estimate_steps()

    # -- 任务链构建 --------------------------------------------------------

    def _build_chain(self) -> None:
        self._chain = build_task_chain(self.task_id, world_facts=self.world_facts())
        # 任务链需要到达的最深楼层（用于升级策略与表层制备）。
        wf = self.world_facts()
        self._max_floor = max(
            (task_target_floor(tid, 0, wf) or 0) for tid in self._chain
        )

    def world_facts(self) -> Optional[WorldFacts]:
        """当前 seed 的世界事实（离线就绪度/跨层矿石参考）。"""
        if self._world_facts is None and self.seed is not None:
            self._world_facts = WorldFacts.for_seed(self.seed)
        return self._world_facts

    def abort_reason(self) -> Optional[str]:
        """seed 不可行时的中止原因（否则 None）。"""
        return self._abort_reason

    def chain(self) -> List[str]:
        return list(self._chain)

    # -- 对外接口 ----------------------------------------------------------

    def next_action(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Optional[int]:
        """返回下一步动作 id；全部完成返回 None（调用方停止录制）。

        外层只做停滞看门狗：策略本体在 _next_action_inner。反应式规则级联
        会出现"原地重复同一动作但状态不变"的僵持（等挡路怪、缺料反复按合成键、
        等刷怪），这里统一检测并打破，持续太久则中止该 seed。
        """
        if "map" in map_payload and isinstance(map_payload["map"], list):
            map_payload = dict(map_payload)
            map_payload["map"] = np.asarray(map_payload["map"], dtype=np.int32)
        self._steps += 1
        self._update_collect_target_state(map_payload, summary)
        self._update_stall_state(map_payload, summary)
        self._update_chain_progress(summary)
        action = self._next_action_inner(map_payload, summary)
        if action is None:
            return None
        return self._guard_stall(action, map_payload, summary)

    # -- 链推进预算 --------------------------------------------------------

    def _chain_progress_signature(self, summary: Dict[str, Any]) -> Tuple[int, int, int]:
        """"任务链是否真的前进"的签名。

        刻意**只含**楼层、链下标与成就数。位置与背包被故意排除：把它们算进
        进展就等于把"在浅层来回走着采集"判成推进，这正是普通停滞检测抓不到
        本类故障的原因（见 CHAIN_PUSH_STEPS）。
        """
        return (
            int(summary.get("floor", 0)),
            int(self._chain_idx),
            len(summary.get("achievements", [])),
        )

    def _update_chain_progress(self, summary: Dict[str, Any]) -> None:
        """更新链推进预算与推进优先模式（脉冲 + 冷却，语义同 _regen_mode_active）。

        睡眠/休息**照常计入**——这与停滞检测相反。睡眠对停滞检测是"合法的长时
        间不动"，但对链推进预算恰恰是要限制的对象：实测三段 68/58/58 步的睡眠
        全部落在"下楼→挨打→回地表睡→再下楼"的循环里。
        """
        sig = self._chain_progress_signature(summary)
        if sig != self._chain_sig:
            self._chain_sig = sig
            self._steps_since_chain_progress = 0
            # 真的推进了 → 立刻结束推进优先窗口（它的目的已经达到）
            if self._push_mode:
                self._push_mode = False
                self._push_steps = 0
                self._push_cooldown = PUSH_COOLDOWN_STEPS
            return
        self._steps_since_chain_progress += 1
        if self._push_mode:
            self._push_steps += 1
            if self._push_steps >= PUSH_MAX_STEPS:
                self._push_mode = False
                self._push_steps = 0
                self._push_cooldown = PUSH_COOLDOWN_STEPS
            return
        if self._push_cooldown > 0:
            self._push_cooldown -= 1
            return
        # 触发条件二选一：链推进预算耗尽，或全局步数预算告急（剩余步数已不足以
        # 走完链上剩下的部分）。后者把 self.max_steps 接进了决策——它此前只是
        # 存下来从未被读取。
        if (self._steps_since_chain_progress >= CHAIN_PUSH_STEPS
                or self._budget_is_tight()):
            self._push_mode = True
            self._push_steps = 0

    # 全局步数预算的"告急"起点：走完这个比例之前一律不认为告急。
    # 必须有这道下限——调用方给的 max_steps 可能本来就小于 estimate_steps()
    # （默认 2000 对深层任务就是如此），否则第 0 步就判定告急，推进优先模式
    # 会从头开到尾，那些恢复行为的设计意图就全被抹掉了。
    BUDGET_PRESSURE_START = 0.5

    def _budget_is_tight(self) -> bool:
        """剩余步数是否已不足以走完链上剩余部分（按 estimate_steps 的比例折算）。"""
        if not self.max_steps or not self._chain:
            return False
        if self._steps < self.max_steps * self.BUDGET_PRESSURE_START:
            return False
        remaining_steps = int(self.max_steps) - self._steps
        if remaining_steps <= 0:
            return True
        done_ratio = min(1.0, self._chain_idx / float(len(self._chain)))
        remaining_need = self.estimate_steps() * (1.0 - done_ratio)
        return remaining_steps < remaining_need

    def _push_now(self) -> bool:
        """是否处于"推进优先"模式（降低推进门槛、停止一切可选的驻留行为）。"""
        return self._push_mode

    def _update_collect_target_state(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> None:
        pending = self._pending_collect_target
        if pending is None or not self._pending_collect_armed:
            return
        floor, row, col, target_types = pending
        origin = map_payload.get("map_origin", [0, 0])
        local = (row - int(origin[0]), col - int(origin[1]))
        grid = map_payload.get("map")
        still_target = (
            grid is not None and 0 <= local[0] < len(grid)
            and 0 <= local[1] < len(grid[0])
            and int(grid[local[0]][local[1]]) in target_types
        )
        if still_target:
            key = pending
            failures = getattr(self, "_collect_target_failures", Counter())
            failures[key] += 1
            self._collect_target_failures = failures
            if failures[key] >= 2:
                self._blocked_collect_targets.add(key)
        else:
            self._collect_target_failures.pop(pending, None) if hasattr(self, "_collect_target_failures") else None
        self._pending_collect_target = None
        self._pending_collect_armed = False

    # -- 停滞看门狗 --------------------------------------------------------

    @staticmethod
    def _progress_signature(
        map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """"是否有进展"的签名：楼层 / 成就数 / 背包 / 本层击杀数。

        故意不含血量与能量——观测到的死亡僵持里血量一直在掉（看着"有变化"，
        实际毫无进展），把它算进签名就永远检测不到停滞。
        """
        inventory = summary.get("inventory") or {}
        inv_sig = tuple(
            sorted(
                (k, int(v) if not isinstance(v, (list, tuple)) else sum(int(x) for x in v))
                for k, v in inventory.items()
            )
        )
        position = summary.get("player_global_position") or summary.get("player_position") or [0, 0]
        position_sig = tuple(int(x) for x in position[:2])
        return (
            int(summary.get("floor", 0)),
            len(summary.get("achievements", [])),
            inv_sig,
            int(map_payload.get("monsters_killed", 0)),
            position_sig,
        )

    def _update_stall_state(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> None:
        # 睡眠/休息是"合法的长时间不动"（在回能量/回血）→ 不计入停滞
        if bool(summary.get("is_sleeping", False)) or bool(summary.get("is_resting", False)):
            return
        sig = self._progress_signature(map_payload, summary)
        if sig != self._progress_sig:
            self._progress_sig = sig
            self._stall_steps = 0
            self._repeat_counter.clear()
            self._stall_reset_done = False
        else:
            self._stall_steps += 1

    def _guard_stall(
        self, action: int, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        if bool(summary.get("is_sleeping", False)) or bool(summary.get("is_resting", False)):
            return action
        # Waiting actions used to bypass the watchdog because they are not
        # cardinal moves.  In the demo this became hundreds of DO commands at
        # one coordinate: the player faced water/grass, ``_idle_action`` kept
        # returning DO, and a nearby mob eventually killed it.  Give adjacent
        # hostiles a direct turn/attack response, then force the normal
        # deterministic stall escape after a few unchanged steps.
        if action in (DO, NOOP) and self._stall_steps >= 2:
            adjacent = self._adjacent_hostile_action(map_payload, summary)
            if adjacent is not None:
                return adjacent
        if action in (DO, NOOP) and self._stall_steps >= 4:
            escaped = self._break_stall(action, map_payload, summary)
            if escaped != action:
                return escaped
        # 最后一层防护：若移动动作连续没有改变位置，而相邻格是可交互
        # 资源（最常见是树挡在锁定目标的路线上），优先原生 DO/转向，
        # 避免继续输出撞墙方向造成“抽搐”。这不释放绝对目标锁；
        # 交互完成后仍沿原目标继续规划。
        if action in ACTION_DELTA and self._stall_steps >= 2:
            pos = _norm_pos(summary.get("player_position", [0, 0]))
            direction = int(summary.get("player_direction", 0))
            grid = map_payload.get("map")
            # This fallback is for a movement route that is genuinely blocked by
            # a resource target.  Water is deliberately excluded: being next to
            # water is common while the planner is trying to find food, a ladder,
            # or a combat position, and turning that movement into DO traps the
            # player at the shore until hunger kills it.
            interactive = set()
            for tid, values in COLLECT_TARGET_BLOCKS.items():
                if tid != "native.collect_drink":
                    interactive.update(values)
            for delta, facing_action in DELTA_TO_ACTION.items():
                adj = (pos[0] + delta[0], pos[1] + delta[1])
                if (grid is not None and 0 <= adj[0] < len(grid)
                        and 0 <= adj[1] < len(grid[0])
                        and int(grid[adj[0]][adj[1]]) in interactive):
                    if direction == facing_action:
                        return DO
                    return facing_action
        # 上下楼振荡：20 步窗口内换层动作 >= 6 次 → 冻结"为补给上行"一段时间。
        # 楼层每步都在变时进展签名也每步在变，普通停滞检测抓不到这种循环。
        if action in (ASCEND, DESCEND):
            self._ladder_action_steps.append(self._steps)
            self._ladder_action_steps = [
                s for s in self._ladder_action_steps if self._steps - s <= 20
            ]
            self._ladder_recent_actions.append(action)
            self._ladder_recent_actions = self._ladder_recent_actions[-6:]
            # 供给不足时 ASCEND/DESCEND 交替会把执行器困在楼梯口；
            # 这不是有效进展，即使楼层字段在变化也应打断一次并冷却上行。
            if (
                len(self._ladder_recent_actions) >= 4
                and all(
                    self._ladder_recent_actions[-i]
                    != self._ladder_recent_actions[-i - 1]
                    for i in range(1, 4)
                )
            ):
                self._supply_ascend_block = max(self._supply_ascend_block, 120)
                self._ladder_recent_actions.clear()
                return self._break_stall(action, map_payload, summary)
            if len(self._ladder_action_steps) >= 6:
                self._supply_ascend_block = 200
                self._ladder_action_steps.clear()
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        key = (int(summary.get("floor", 0)), pos,
               int(summary.get("player_direction", 0)), int(action))
        self._repeat_counter[key] += 1
        stalled = (
            self._stall_steps >= STALL_NO_PROGRESS_STEPS
            or self._repeat_counter[key] >= STALL_REPEAT_ACTIONS
        )
        if not stalled:
            return action
        if self._stall_steps >= STALL_ABORT_STEPS:
            self._abort_reason = (
                f"停滞 {self._stall_steps} 步无进展（楼层/成就/背包/击杀均未变化）"
            )
            return None
        # 干预一次后清空重复计数：否则策略每步仍提出同一动作、计数只增不减，
        # 看门狗会一直接管（变成"永久乱走"），策略再也没有机会重试。
        self._repeat_counter.clear()
        return self._break_stall(action, map_payload, summary)

    @staticmethod
    def _melee_strike_action(
        map_payload: Dict[str, Any], summary: Dict[str, Any], cell: Tuple[int, int]
    ) -> Optional[int]:
        """把"打这只怪"翻译成动作：已朝向 → DO，否则先转身。None = 打不到。

        持剑时射程为 SWORD_REACH（2 格，见 game_logic.do_action）：正前方第二格
        的怪也能打到，前提是中间那一格通透——游戏侧的视线门用的是同一个条件。
        隔着墙/树打不到，别把转身浪费在打不到的目标上。
        """
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        delta = (cell[0] - pos[0], cell[1] - pos[1])
        attack = DELTA_TO_ACTION.get(delta)
        if attack is not None:
            return DO if direction == attack else attack
        inventory = summary.get("inventory") or {}
        if int(inventory.get("sword", 0)) < 1:
            return None
        # 两格：必须是同一方向的正前方（不含斜向），且中间格不是实心方块
        for move, step in ACTION_DELTA.items():
            if delta != (step[0] * SWORD_REACH, step[1] * SWORD_REACH):
                continue
            mid = (pos[0] + step[0], pos[1] + step[1])
            map2d = map_payload.get("map")
            if map2d is None:
                return None
            if not (0 <= mid[0] < len(map2d) and 0 <= mid[1] < len(map2d[0])):
                return None
            if blocked(int(map2d[mid[0]][mid[1]])):
                return None
            return DO if direction == move else move
        return None

    @classmethod
    def _adjacent_hostile_action(
        cls, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """Return turn/attack action for a melee/ranged mob within sword reach."""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        candidates: List[Tuple[int, Tuple[int, int]]] = []
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, point in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                cell = (int(point[0]), int(point[1]))
                dist = abs(cell[0] - pos[0]) + abs(cell[1] - pos[1])
                if dist <= SWORD_REACH:
                    candidates.append((dist, cell))
        # 近的优先：贴脸的怪这一回合就会打到我们，两格外的还不会
        for _dist, cell in sorted(candidates):
            action = cls._melee_strike_action(map_payload, summary, cell)
            if action is not None:
                return action
        return None

    def _break_stall(
        self, action: int, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> int:
        """打破僵持：清掉可能过期的缓存，再换一个确定性的备选动作。

        顺序：1) 一次性重置换层/战术缓存（可能因为"去过的层"判断而卡死）；
        2) 有箭且有主动怪 → 点射（挡路怪打掉就通了）；
        3) 按固定轮转方向走一步（绕开占位的怪/被 BFS 判死的角落）。
        """
        if not self._stall_reset_done:
            self._stall_reset_done = True
            self._collect_visited = set()
            self._collect_visited_tid = None
            self._tactic_floor = None
            bow = self._bow_combat(map_payload, summary)
            if bow is not None and bow != SLEEP:
                return bow
        inventory = summary.get("inventory") or {}
        if (self._has_bow(summary) and int(inventory.get("arrows", 0)) >= 1
                and self._nearest_hostile_dist(map_payload, summary) <= 14):
            bow = self._bow_combat(map_payload, summary)
            if bow is not None and bow != SLEEP:
                return bow
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        h, w = len(map2d), len(map2d[0])
        order = [RIGHT, DOWN, LEFT, UP]
        blocked_extra = getattr(self, "_mob_cells", None) or set()
        for i in range(4):
            move = order[(self._stall_nudge + i) % 4]
            delta = ACTION_DELTA[move]
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if blocked(int(map2d[nxt[0]][nxt[1]])) or nxt in blocked_extra:
                continue
            self._stall_nudge = (self._stall_nudge + i + 1) % 4
            return move
        self._stall_nudge = (self._stall_nudge + 1) % 4
        return action

    def _next_action_inner(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Optional[int]:
        # 本层活怪占用的格子会挡住玩家移动，寻路时视为不可通行
        self._mob_cells = self._mob_blocked_cells(map_payload)
        # 怪攻击冷却计时：健康比上一步下降说明被近战命中（命中后冷却重置 5）。
        # 计时用于风筝——冷却将到时（<=1）拉开，避免被命中。
        health = float(summary.get("health", 9.0))
        dropped = self._prev_health is not None and health < self._prev_health - 0.01
        if dropped and self._mob_adjacent(map_payload, summary, 1):
            self._mob_attack_timer = 5
        else:
            if self._mob_attack_timer > 0:
                self._mob_attack_timer -= 1
            # 掉血却没有近身怪 → 被远程怪的投射物命中（远程怪在 4-5 格开火）。
            # 饥/渴/力竭的掉血恒为 1 点/次，投射物 >=2（骷髅 2、兽人法师 3），
            # 故以 1.5 为界区分二者，避免把断补给误判成远程压制。
            if dropped and (self._prev_health or 0.0) - health >= 1.5:
                self._ranged_hit_step = self._steps
        self._prev_health = health
        # 风筝撤退收尾：连续拉开 2 步让怪在冷却归零时追不上
        if self._kite_retreats > 0:
            self._kite_retreats -= 1
            retreat = self._retreat_from_mobs(map_payload, summary)
            if retreat is not None:
                return retreat
        if self._is_done(map_payload, summary):
            return None
        # 近身主动敌人拥有最高优先级。补水/进食、制作和下楼都可能是
        # 合法的长期目标，但在僵尸贴脸时继续执行这些动作会让玩家原地
        # 挨打（尤其是站在水源旁反复 DRINK/FILL_WATER）。先交出一个战斗
        # 或转身动作，下一回合再由策略决定继续战斗还是撤退。
        if self._mob_adjacent(map_payload, summary, 1):
            emergency = self._adjacent_hostile_action(map_payload, summary)
            if emergency is not None:
                return emergency
            emergency = self._combat_any(map_payload, summary)
            if emergency is not None and emergency != SLEEP:
                return emergency
        # 属性升级：力量优先（每点 +25% 物伤、+1 血）；深层任务视能量瓶颈补敏捷
        level = self._level_up_choice(summary)
        if level is not None:
            return level
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
            # 注意：不能写 REACH_FLOOR.get(tid, ENTER_FLOOR[tid])——默认值会被
            # 立即求值，tid 只在 REACH_FLOOR 里时抛 KeyError（explore_dungeon /
            # dungeon_campaign 一进 L1 就崩）。
            target = self._floor_of_descend_task(self._chain[self._chain_idx])
            if target is not None:
                descend = self._descend_to(map_payload, summary, target)
                if descend is not None:
                    return descend

        # 当任务链仍需向更深楼层推进时，优先执行这条完整的制备/下楼路径。
        # 该路径内部已经包含补给、装备升级、备箭和 L1+ 清怪门；如果把它放在
        # _survival_action 后面，地表弓战会不断抢占动作，回到 L0 后永远不再
        # 走向梯子。紧急近身威胁已在上方处理，真正的危险补给也由路径自身处理。
        safe_to_progress = self._safe_to_progress(summary)
        # 深层任务先让后面的 bow-rush 完成首把弓；没有弓就直接推进会把玩家
        # 送进 L1 近战挨打，反而失去进入下一阶段的机会。
        # The first descent is how the player reaches the floor-1 chest that
        # supplies the bow.  Only gate *further* descent on bow preparation;
        # gating surface -> floor 1 makes reach_floor_3 wait forever with a
        # perfectly usable sword/pickaxe but no bow yet.
        needs_bow_prep = (
            self._max_floor >= 2
            and floor >= 1
            and not self._has_bow(summary)
        )
        if need_descend and safe_to_progress and not needs_bow_prep:
            target = self._floor_of_descend_task(self._chain[self._chain_idx])
            if target is not None and floor < target:
                descend = self._descend_to(map_payload, summary, target)
                if descend is not None:
                    return descend
        # 生存维护优先
        survival = self._survival_action(map_payload, summary)
        if survival is not None:
            return survival
        # 生存线稳定后，立即兑现已经满足的装备升级条件；不要等到链条
        # 再次切换到“制备”子任务才合成，避免带着可用材料空手遇怪。
        equipment = self._priority_equipment_action(map_payload, summary)
        if equipment is not None:
            return equipment
        # 空闲且在地表时顺手补满便携水，供无水楼层使用；只补到容量上限，
        # 不会在深层反复上楼抢占当前任务。
        inventory = summary.get("inventory") or {}
        if (int(summary.get("floor", 0)) == 0
                and int(inventory.get("water", 0)) < MAX_WATER_CANTEENS):
            reserve = self._fill_water(map_payload, summary)
            if reserve is not None:
                return reserve
        # 弓先制：深层任务（max_floor>=2）在深制备前先拿 L1 弓。
        # 打断当前子目标——拿到弓后制备/清怪受击降到 0-1，L0 制备墙解除。
        # （弓的补给/恢复由 _survival_action 的箭补给分支处理，这里只管获取。）
        if self._max_floor >= 2 and not self._has_bow(summary):
            bow_action = self._bow_rush(map_payload, summary)
            if bow_action is not None:
                return bow_action
        # 有弓即设补给目标（在已清层补满箭再推进，防止 0 箭待宰）
        if self._has_bow(summary) and self._restock_target == 0:
            self._restock_target = BOW_ARROW_RESERVE
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

    def _priority_equipment_action(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """把可立即完成的武器/工具升级提升到生存维护之后。"""
        inventory = summary.get("inventory") or {}
        max_floor = self._max_floor
        sword = int(inventory.get("sword", 0))
        pickaxe = int(inventory.get("pickaxe", 0))
        wood = int(inventory.get("wood", 0))
        stone = int(inventory.get("stone", 0))
        coal = int(inventory.get("coal", 0))
        iron = int(inventory.get("iron", 0))
        if max_floor >= 1 and sword < 1 and wood >= 1:
            return self._craft_action("native.craft_wood_sword", map_payload, summary)
        if max_floor >= 2 and sword < 2 and wood >= 1 and stone >= 1:
            return self._craft_action("native.craft_stone_sword", map_payload, summary)
        if max_floor >= 2 and pickaxe < 1 and wood >= 1:
            return self._craft_action("native.craft_wood_pickaxe", map_payload, summary)
        if max_floor >= 2 and pickaxe < 2 and wood >= 1 and stone >= 2:
            return self._craft_action("native.craft_stone_pickaxe", map_payload, summary)
        if max_floor >= 3 and sword < 3 and wood >= 1 and stone >= 1 and iron >= 1 and coal >= 1:
            return self._craft_action("native.craft_iron_sword", map_payload, summary)
        return None

    @staticmethod
    def _floor_of_descend_task(tid: str) -> Optional[int]:
        """下楼/到达类任务的目标层（REACH_FLOOR 优先，其次 ENTER_FLOOR）。"""
        if tid in REACH_FLOOR:
            return REACH_FLOOR[tid]
        return ENTER_FLOOR.get(tid)

    def is_done(self, summary: Dict[str, Any]) -> bool:
        return self._task_is_complete(self.task_id, summary)

    def step_count_hint(self) -> int:
        return len(self._chain)

    # 步数预算的标定项（travel 为单层往返寻路的经验值）
    SURFACE_PREP_STEPS = 500       # L0 制备（采木/石 + 合成 + 拿弓往返）
    FLOOR_TRAVEL_STEPS = 140       # 每层走到梯子/矿点的寻路开销
    PER_TASK_STEPS = 60            # 每个链上任务的交互/微调开销
    RECOVERY_STEPS_PER_FLOOR = 120  # 批量清怪之间回锚点恢复的往返

    def estimate_steps(self) -> int:
        """步数预算：用战斗模型逐层估算清怪成本，而不是拍一个线性公式。

        每层成本 = combat_model.estimated_steps_bow(层, 预期装备)（弓先制是
        默认路线）+ 寻路 + 锚点恢复往返；再加表层制备与每任务交互开销。
        预期装备取"到该层时应有的"保守值（剑 2 / 弓 1 / 力量 3），与
        FLOOR_GEAR_REQ 的软门槛一致。封顶 30000。
        """
        if self._estimate_cache is not None:
            return self._estimate_cache   # 链不变则估算不变，而 _budget_is_tight 每步都问
        total = float(self.SURFACE_PREP_STEPS + self.PER_TASK_STEPS * len(self._chain))
        for floor in range(1, self._max_floor + 1):
            gear = Gear(sword=2, armour=0, strength=3, dexterity=1, intelligence=1,
                        has_elemental=True, bow=1, bow_enchant=0)
            total += estimated_steps_bow(floor, gear)
            total += self.FLOOR_TRAVEL_STEPS + self.RECOVERY_STEPS_PER_FLOOR
        self._estimate_cache = int(min(30000, max(600, total)))
        return self._estimate_cache

    # -- 完成判定 ----------------------------------------------------------

    def _adapter(self, tid: str):
        """缓存 registry 里该任务的适配器（承载权威 success_predicate）。"""
        adapter = self._adapters.get(tid)
        if adapter is None:
            from craftax.tasks import registry

            versions = registry.list_versions(tid)
            if not versions:
                return None
            adapter = registry.get_task_adapter(tid, versions[-1])
            self._adapters[tid] = adapter
        return adapter

    def _spec_done(self, tid: str, summary: Dict[str, Any]) -> bool:
        """按 registry 的 success_predicate 判定完成——与数据集标注同源。

        谓词只需要成就名单与 player_level（builtin 任务用 achievement /
        level_ge / and / or / always），因此可直接在 state_summary 上求值，
        无需 EnvState。未注册的 task_id 返回 False。
        """
        adapter = self._adapter(tid)
        if adapter is None:
            return False
        info = {"achievements_list": list(summary.get("achievements", []))}
        return bool(adapter.success(_SummaryPredicateState(summary), info))

    def _task_is_complete(self, tid: str, summary: Dict[str, Any]) -> bool:
        """任务是否已完成。

        权威口径是 registry 的成功谓词；在此之上，**中间**任务还要满足
        "执行层可用性"，因为链上的后续步骤消费的是实物而不是成就：
        - 采集类：需要 RESOURCE_TARGETS 的储备量（成就早就达成也得补货）；
        - 消耗品合成（箭/火把）：看背包数量而不是一次性的 MAKE_* 成就；
        - 工具合成：成就或更高等级的同类工具都算达成（宝箱开出铁剑也算）。
        目标任务本身只看谓词——那才是任务的定义。
        """
        spec_done = self._spec_done(tid, summary)
        if tid == self.task_id:
            return spec_done
        inventory = summary.get("inventory") or {}
        if tid in RESOURCE_TARGETS:
            field, target = RESOURCE_TARGETS[tid]
            return int(inventory.get(field, 0)) >= target
        # collect_drink 作为 Boss/生存复合任务的前置时，不只要一次性成就，
        # 还要把便携水装到目标储备；否则离开地表后仍会被迫频繁回水源。
        # 作为根任务时保留原有语义：喝到一次水即可完成该原子目标。
        if tid == "native.collect_drink":
            return (
                spec_done
                and int(inventory.get("water", 0)) >= WATER_RESERVE_TARGET
            )
        if tid in CONSUMABLE_CRAFTS:
            return self._craft_done(tid, inventory)
        if tid in CRAFT_ACTIONS:
            return spec_done or self._craft_done(tid, inventory)
        return spec_done

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
        health = float(summary.get("health", 9.0))
        is_sleeping = bool(summary.get("is_sleeping", False))
        is_resting = bool(summary.get("is_resting", False))
        floor = int(summary.get("floor", 0))
        # 水/食物由 _supply_action 统一处理（见下方 0) 分支）

        if is_sleeping or is_resting:
            # 睡眠/休息中：保持（游戏会把动作变 NOOP，见 game_logic.craftax_step）。
            # 但**必须先把这些步数记到恢复类行为的预算上**：这里是早退点，
            # 下方 1a) 的 `self._regen_steps += 1` 根本走不到，于是一段 68 步的
            # 睡眠只消耗 0 预算——REGEN_MAX_STEPS=150 实际只在数"走去掩体的
            # 那几步"，名义上的预算完全没有约束到真正的恢复时间。
            if self._regen_mode:
                self._regen_steps += 1
            if self._night_hold:
                self._night_hold_steps += 1
            return NOOP

        mobs_close = self._mob_within(map_payload, summary, 5)
        mobs_adj = self._mob_adjacent(map_payload, summary, 1)

        # Keep this guard inside the survival routine as well as at the outer
        # dispatch boundary: callers/tests may invoke _survival_action
        # directly, and supply must never win over a zombie already beside us.
        if mobs_adj:
            emergency = self._adjacent_hostile_action(map_payload, summary)
            if emergency is not None:
                return emergency
            emergency = self._combat_any(map_payload, summary)
            if emergency is not None and emergency != SLEEP:
                return emergency

        # -1) 掩体优先于反击：被投射物命中（掉血却无近身怪）且不在掩体里 →
        #     先进掩体。墙会把投射物吃掉（_move_mob_projectile 的 in_wall），
        #     而追射手要走 4-5 格、路上继续挨打。实测 L1 死因正是这一段：
        #     hp 10→7→4→1 每一跳都发生在无近身怪时。
        #     这里的门槛刻意是 2 面墙（不是 3）：**挡住射线就够了**——一个 2 面墙
        #     的墙角已经封掉两个方向的箭道，而 3 面墙的坑位在地表 8 步内往往不
        #     存在（实测按 3 面墙找掩体时远程受击率没有下降：4.15→4.16 次/千步）。
        #     3 面墙的要求留给睡眠/驻守，那里要限制的是近战接战数。
        if self._ranged_pressure() and not mobs_adj:
            cover = self._take_cover(map_payload, summary, min_walls=2,
                                     max_steps=self.RANGED_COVER_STEPS)
            if cover is not None:
                return cover

        # 弓主动防御：弓主战层有弓+箭且主动怪进入交战半径 → 提前点射。
        # 交战半径**按击杀是否有价值区分**：
        # - 未清层（monsters_killed<8）：14 格 = 出生环（10-13）+ 余量。这里每一
        #   杀都算进下楼门槛，远距射杀既推进目标又免接战（0 受击）；
        # - 已清层（如 L0，初始 monsters_killed=10）：**杀怪买不到任何东西**，
        #   而地表持续刷新 → 按 14 格无差别点射会把整局变成"地表箭工厂"：
        #   实测新箭道打通后 L0 击杀 17→29、箭耗到 0、一次没下楼。因此已清层
        #   只打真正够得着我们的怪（远程怪射程 4-5，留 1 格余量）。
        # 这里不追远怪（chase）：地表怪持续刷新，追击会把整局变成原地打怪。
        # 血足时可追近怪（"all"），血不足只点射不移动——走位会引发更多接战。
        # 推进优先模式下，已清层的主动点射一律停掉：那里的击杀买不到任何东西，
        # 而它消耗的箭会立刻把执行器拉回"补箭"循环。真正的自卫由下面
        # SURVIVAL_ENGAGE_DIST 那一段兜住。
        clearing_now = int(map_payload.get("monsters_killed", 0)) < 8
        engage_range = 14 if clearing_now else self.CLEARED_FLOOR_ENGAGE_DIST
        if (self._should_use_bow(floor, summary)
                and (clearing_now or not self._push_now())
                and int((summary.get("inventory") or {}).get("arrows", 0)) >= 1
                and self._nearest_hostile_dist(map_payload, summary) <= engage_range):
            proactive = self._bow_combat(
                map_payload, summary,
                chase="all" if health >= self.REGEN_EXIT_HEALTH else "none",
                max_range=engage_range,
            )
            if proactive is not None and proactive != SLEEP:
                return proactive

        # 不把"第一次挨打"白送掉：可达敌人已进到 SURVIVAL_ENGAGE_DIST 内且有箭
        # → 现在就点射/走到点射位。旧规则要等怪贴脸（≤1）或血掉到 6 以下才反击，
        # 于是 2-4 格的怪总能免费走近并打出第一击（实测 L1 死因：hp 6 时对
        # 2 格外的兽人无动作，被走近连打两下致死）。
        # 距离刻意取 3（不是 10）：地表远程怪在 10-13 格持续刷新，追远怪会变成
        # 无限接战（实测 94 步暴死）。
        if (int((summary.get("inventory") or {}).get("arrows", 0)) >= 1
                and self._has_bow(summary)
                and self._reachable_hostile_within(map_payload, summary,
                                                   self.SURVIVAL_ENGAGE_DIST)):
            engage = self._bow_combat(map_payload, summary, chase="all",
                                      max_range=engage_range)
            if engage is not None and engage != SLEEP:
                return engage

        has_bow = self._has_bow(summary)

        # 0) 补给优先（水/食物）：这一段必须排在"防御回血"和"睡眠"之前。
        #    机制上 drink 或 food 归零后 recover 变负：醒着 16 步掉 1 血、
        #    睡着 31 步掉 1 血，而被动回血是 26 步/HP——渴着回血/睡觉是净掉血，
        #    等于用不可撤销的等待去解一个走几步就能解决的问题（实测死因）。
        supply = self._supply_action(map_payload, summary)
        if supply is not None:
            return supply

        # 1) 能量/健康维护：能量将尽（<3）→ 睡（回能量 + 回血）。
        #    睡中受击 3.5x（L0 僵尸 7）——先清近身怪、再找"距主动怪 >=14 格"的
        #    安全点睡（怪在 >14 会消失，睡中不会被打醒）。floor>0 时先回 L0 锚点
        #    （已清、怪弱）。
        #    睡前先做一次投影（project_sleep）：把整段睡眠时长的口渴/饥饿/掉血
        #    走完，若睡醒会低于 SLEEP_MIN_PROJECTED_HEALTH 就不睡——SLEEP 一旦
        #    按下就锁到能量回满，中途无法自救。
        if energy < 3 and self._sleep_is_safe(summary):
            # 睡前去清近身怪（弓点射 1-2 箭 / 近战），避免睡中被 3.5x 打醒致死
            if self._mob_adjacent(map_payload, summary, 1):
                a = (self._bow_combat(map_payload, summary) if has_bow
                     else self._combat_any(map_payload, summary))
                if a is not None and a != SLEEP:
                    return a
            if floor > 0:
                a = self._ascend_to(map_payload, summary, 0)
                if a is not None:
                    return a
            # 找安全点（<=20 步内距离主动怪 >=14 格）：怪在 >14 会消失，
            # 在安全点睡任何血量都不会被打醒（低血/低能量也能安全恢复）。
            safe = self._safe_sleep_spot_walk(map_payload, summary, max_steps=20)
            if safe is not None:
                return safe
            # 找不到"距怪 >=14"的安全点 → 退而求其次进掩体：三面墙的坑位只剩
            # 一个开口，睡中最多被一只怪打醒，且三个方向挡箭。
            cover = self._take_cover(map_payload, summary)
            if cover is not None:
                return cover
            # 当前已足够远（>=14）或身处掩体 → 就地睡
            if (self._nearest_hostile_dist(map_payload, summary) >= 14
                    or self._in_shelter(map_payload, summary)):
                return SLEEP
            # 不够远且无掩体 → **不睡**。旧规则是"血足就睡，扛一次 3.5x 打醒"，
            # 但 SLEEP 一按就锁到能量回满（60+ 步），怪只要在 14 格内就一定走到；
            # 实测 seed 2011 满血 10 被僵尸一击打到 3（2 伤 ×3.5）随即补刀，
            # 259 步暴毙。宁可顶着低能量继续推进：能量归零只是每 16 步掉 1 血
            # 的慢性消耗，且随时可自救，而睡眠中的 3.5x 是不可撤销的。
        # 1a) 血不足（<8）且在已清层（锚点）且箭尚足（>=2）→ 防御性原地回血
        #     （不推进）。先走到距主动怪尽量远的安全点/角落——角落怪只能沿直线
        #     靠近，弓可提前 14 格点射（0 受击）；无安全点则原地 DO 等被动回血
        #     （26 步/HP），回满到 8 后再睡/继续——避免低血+低能量死亡螺旋。
        #     前提：投影确认"等待确实在回血"（缺水/缺食/能量耗尽时是净掉血，
        #     此时不能原地等，必须继续推进去找补给，见 0)）。
        monsters_killed = int(map_payload.get("monsters_killed", 0))
        arrows_now = int((summary.get("inventory") or {}).get("arrows", 0))
        if (self._regen_mode_active(health) and monsters_killed >= 8 and energy > 1
                and arrows_now >= 2 and not mobs_adj
                and self._waiting_regenerates(summary)):
            self._regen_steps += 1
            safe = self._safe_sleep_spot_walk(map_payload, summary, max_steps=8)
            if safe is not None:
                return safe
            cover = self._take_cover(map_payload, summary)
            if cover is not None:
                return cover
            if self._in_shelter(map_payload, summary) and self._sleep_is_safe(summary):
                return SLEEP
            return DO
        # 1a') 夜间条件驻守：L0 夜间近战刷新 0.02 → 0.12/步，天黑后在旷野推进
        #      等于持续吃刷新。血/能量不满就进掩体睡到天亮；带滞回 + 预算 +
        #      冷却（同 _regen_mode_active 的教训：裸阈值会让子目标永远不推进）。
        if self._night_hold_active(summary) and not mobs_adj:
            cover = self._take_cover(map_payload, summary)
            if cover is not None:
                return cover
            if self._in_shelter(map_payload, summary) and self._sleep_is_safe(summary):
                return SLEEP
        # 1b) 箭补给：已清层 + 储备未满 + 无近身怪 → 补箭。
        #     弓是生存武器（0-1 箭=待宰），血低也补（0 箭必死，补箭才有机会）。
        #
        #     这里的目标改回**真实弹药预算**（_restock_target，L1 约 23 支），
        #     不再截到 BOW_ARROW_RESERVE。旧的截断是为了压掉"1800 步里 635 步
        #     在合成箭、一次没下楼"，但它限制的是**目标**而不是**成本**：8 支
        #     箭只够约 3 个击杀，而下楼门要求清 8 只怪，于是执行器带着永远不够
        #     的弹药反复下楼送死（实测两局都死在 L1）。真正该限制的是"为补箭
        #     花掉多少步"——由 _restock_active 的步数预算 + 冷却负责。
        if (has_bow and arrows_now < self._restock_target
                and monsters_killed >= 8 and energy >= 2 and not mobs_adj
                and self._restock_active()):
            craft = self._craft_action("native.craft_arrow", map_payload, summary)
            if craft is not None:
                return craft
        # 1c) 极低血（<3）→ 暂停推进：清近身怪 / 撤退，其余原地 DO 等被动回血。
        #     （致命的缺水/缺食已在 0) 处理，那里优先级更高。）
        if health < 3:
            if mobs_close and self._has_bow(summary):
                a = self._bow_combat(map_payload, summary)
                if a is not None and a != SLEEP:
                    return a
            if mobs_close:
                retreat = self._retreat_from_mobs(map_payload, summary)
                if retreat is not None:
                    return retreat
            if self._in_shelter(map_payload, summary) and self._sleep_is_safe(summary):
                return SLEEP
            return DO
        # 2) / 3) 口渴与饥饿已提前到 0) _supply_action（补给优先于回血/睡眠）
        # 4) 清怪中血/能量不足，或箭将耗尽 → 回 L0 锚点恢复/补给
        #    （L0 已清、怪弱，可安全睡觉 + 补箭）。直接走向梯子 ASCEND——
        #    不先撤退（撤退方向可能与梯子相反，来回打转致死）。
        monsters_killed = int(map_payload.get("monsters_killed", 0))
        arrows_now = int((summary.get("inventory") or {}).get("arrows", 0))
        if (floor > 0 and monsters_killed < 8
                and (health < 6 or energy < 3
                     or (self._has_bow(summary) and arrows_now < 4))):
            # 箭不足导致的回撤 → 回 L0 后补满箭再下（不下调下楼前算出的弹药预算）
            if self._has_bow(summary) and arrows_now < 4:
                self._restock_target = max(self._restock_target, BOW_ARROW_RESERVE)
            a = self._ascend_to(map_payload, summary, 0)
            if a is not None:
                return a
        # 5) 生命低（<6）且近身有怪 → 主动清掉近身怪（撤退甩不掉怪，只是拖长受击；
        #    清掉后能继续推进表层制备——铁剑是逃出死亡螺旋的唯一路径）。
        if health < 6 and mobs_close:
            combat = self._combat_any(map_payload, summary)
            if combat is not None and combat != SLEEP:
                return combat
        # 6) 被 ≥2 只怪近身 → 先撤出包围（被围殴 dps 扛不住），拉开再逐个单挑
        if self._mob_count_within(map_payload, summary, 1) >= 2:
            retreat = self._retreat_from_mobs(map_payload, summary)
            if retreat is not None:
                return retreat
        # 7) 防御性战斗：仅当怪真正紧邻（≤1 格，会攻击）时清怪；
        #    2 格外的怪不影响任务推进（走路/下楼/喝水优先），避免被反复打断
        if self._mob_adjacent(map_payload, summary, max_dist=1):
            combat = self._combat_any(map_payload, summary)
            if combat is not None and combat != SLEEP:
                return combat
        return None

    # -- 补给与"不可撤销决策"的有限步前瞻 ----------------------------------

    # 低于该值就值得专程走一趟补给（0 会开始掉血，留出行程余量）
    SUPPLY_SEEK_THRESHOLD = 5.0
    # 贴着水源/食物时顺手补到该值（无位移成本）
    SUPPLY_TOPUP_TARGET = 9.0

    def _supply_action(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """水/食物维持。返回 None 表示"当前没有值得做的补给动作"。

        分两档：
        - 顺手档：已经站在水/熟植物旁 → 直接喝/吃到满（几乎零成本）；
        - 专程档：低于 SUPPLY_SEEK_THRESHOLD → 走过去补；本层没有来源且更浅层
          有（NO_DRINK_FLOORS / NO_FOOD_FLOORS 或本层确实找不到）→ 上行一层。
        食物完全断供（<3 且本层无来源、已在 L0）时原地 DO 等被动怪刷新——
        这是唯一合理的等待，且由停滞看门狗兜底。
        """
        drink = float(summary.get("drink", 9.0))
        food = float(summary.get("food", 9.0))
        floor = int(summary.get("floor", 0))
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        carried_water = int((summary.get("inventory") or {}).get("water", 0))

        # 先顺手补水，保留便携水给无水楼层；只有附近没有水源且低于 seek
        # 阈值时才消耗瓶子，避免在水源旁提前浪费储备。
        if drink < self.SUPPLY_TOPUP_TARGET and self._near_any(map2d, pos, DRINK_BLOCKS):
            a = self._seek_and_do(map_payload, summary, DRINK_BLOCKS)
            if a is not None:
                return a
        if food < self.SUPPLY_TOPUP_TARGET and self._near_any(map2d, pos, FOOD_BLOCKS):
            a = self._seek_and_do(map_payload, summary, FOOD_BLOCKS)
            if a is not None:
                return a

        # 随身水把补水从一次往返变成一个动作，尤其适合 L6/L8 等无水层。
        if drink < self.SUPPLY_SEEK_THRESHOLD and carried_water > 0:
            return DRINK_WATER

        if drink < self.SUPPLY_SEEK_THRESHOLD:
            a = self._seek_and_do(map_payload, summary, DRINK_BLOCKS)
            if a is not None:
                return a
            if floor > 0:
                a = self._supply_ascend(map_payload, summary, DRINK_BLOCKS,
                                        self._nearest_drink_floor(floor))
                if a is not None:
                    return a
        if food < self.SUPPLY_SEEK_THRESHOLD:
            a = self._seek_and_do(map_payload, summary, FOOD_BLOCKS)
            if a is not None:
                return a
            a = self._seek_and_eat_mob(map_payload, summary)
            if a is not None:
                return a
            if floor > 0:
                # 食物还能来自被动怪（不在地图方块里），因此不按方块数量否决上行
                a = self._supply_ascend(map_payload, summary, None,
                                        self._nearest_food_floor(floor))
                if a is not None:
                    return a
            if food < 3:
                # 已在 L0 且本层暂无食物来源：等被动怪刷新（不睡：睡觉=待宰）
                return DO
        return None

    def _supply_ascend(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        blocks: Optional[Sequence[int]],
        target_floor: int,
    ) -> Optional[int]:
        """为补给上行到 target_floor——但先确认"上去真的有"。

        没有这个确认就会出现梯子来回：本层没水 → 上一层；上一层的水也不可达
        → 计划继续下楼 → 又没水 → 再上去……（实测 seed 3050 每步一次上下楼，
        且楼层每步都在变，连停滞检测都测不出来）。
        blocks 为 None 时不按方块否决（食物可来自被动怪）。
        """
        if self._supply_ascend_block > 0:
            self._supply_ascend_block -= 1
            return None
        if blocks is not None:
            count = self._floor_resource_count(target_floor, blocks)
            if count is not None and count <= 0:
                return None  # 跨层观察：上去也没有 → 不折返
        return self._ascend_to(map_payload, summary, target_floor)

    def _nearest_drink_floor(self, floor: int) -> int:
        """向上找最近的有水层（L6 熔岩海 / L8 Boss 层无水）。"""
        for f in range(floor - 1, -1, -1):
            if f not in NO_DRINK_FLOORS:
                return f
        return 0

    def _nearest_food_floor(self, floor: int) -> int:
        """向上找最近的有食物层（L7 冰界无被动怪/植物）。"""
        for f in range(floor - 1, -1, -1):
            if f not in NO_FOOD_FLOORS:
                return f
        return 0

    # 推进门（safe_to_progress）的滞回区间。旧实现是裸阈值 health>=6 —— 与
    # _regen_mode_active 的文档教训自相矛盾：地表持续刷怪使血量在 6 附近震荡时，
    # 整条下楼路径会被无限期关掉，控制权交给 _survival_action，而后者的恢复
    # 与下楼路径自己的恢复门（REGEN_EXIT_HEALTH）互不协调，形成
    # "下楼→挨打→回地表恢复→下楼"的无限循环（实测占一局 53% 步数）。
    # 加 latch 后：进入需要 6 血，一旦开始推进就一直由下楼路径掌控（它内部
    # 自带补给/回血/清怪门），只有真正见底（<4 血或某项 <1）才交还给生存维护。
    PROGRESS_ENTER_HEALTH = 6.0
    PROGRESS_EXIT_HEALTH = 4.0
    PROGRESS_ENTER_VITAL = 3.0     # 能量/食物/水的进入门槛
    PROGRESS_EXIT_VITAL = 1.0      # 低于此值才交还控制权（0 会开始掉血）
    PUSH_ENTER_HEALTH = 4.0        # 推进优先模式下的进入门槛
    PUSH_ENTER_VITAL = 2.0
    PUSH_DESCEND_HEALTH = 6.0      # 推进优先模式下"下楼前回血"的门槛（常规是 8）

    def _safe_to_progress(self, summary: Dict[str, Any]) -> bool:
        """现在是否适合把控制权交给"制备/下楼"路径（带 latch 的滞回）。"""
        health = float(summary.get("health", 9.0))
        vitals = (
            float(summary.get("energy", 9.0)),
            float(summary.get("food", 9.0)),
            float(summary.get("drink", 9.0)),
        )
        if self._progress_latched:
            if health < self.PROGRESS_EXIT_HEALTH or min(vitals) < self.PROGRESS_EXIT_VITAL:
                self._progress_latched = False
            return self._progress_latched
        enter_health = (self.PUSH_ENTER_HEALTH if self._push_now()
                        else self.PROGRESS_ENTER_HEALTH)
        enter_vital = (self.PUSH_ENTER_VITAL if self._push_now()
                       else self.PROGRESS_ENTER_VITAL)
        if health >= enter_health and min(vitals) >= enter_vital:
            self._progress_latched = True
        return self._progress_latched

    # 防御性回血的滞回区间与预算（避免"永远在回血、永远不推进"）
    REGEN_ENTER_HEALTH = 5.0   # 低于此值进入回血模式
    REGEN_EXIT_HEALTH = 8.0    # 回到此值退出回血模式
    REGEN_MAX_STEPS = 150      # 单次回血模式的步数预算，超出就带着中等血量推进
    REGEN_COOLDOWN_STEPS = 200  # 预算耗尽后的冷却：这段时间不再进入回血模式
    # 正常退出（回满到 8）后的冷却。缺了这一条，"下楼→挨打→回血到 8→下楼"
    # 的循环次数就没有上限：预算只约束单次时长，冷却才约束频率。
    REGEN_EXIT_COOLDOWN_STEPS = 60

    def _regen_mode_active(self, health: float) -> bool:
        """带滞回的"要不要停下来回血"。

        旧实现是 health < 8 的裸阈值：地表 monsters_killed 初始就是 10（≥8），
        于是只要掉到 7 就原地回血，回到 8 又被怪打回 7——子目标可以几百步不动
        （实测 enter_gnomish_mines 在 L0 卡了 700+ 步）。滞回 + 预算 + 冷却把
        "恢复"限制成有限行为：低于 5 才停，回到 8 才走，且最多停 150 步。

        推进优先模式下完全不进入：那时的问题恰恰是"恢复得太称职"。
        """
        if self._push_now():
            self._regen_mode = False
            self._regen_steps = 0
            return False
        if self._regen_cooldown > 0:
            self._regen_cooldown -= 1
            self._regen_mode = False
            return False
        if self._regen_mode:
            if health >= self.REGEN_EXIT_HEALTH:
                self._regen_mode = False
                self._regen_steps = 0
                self._regen_cooldown = self.REGEN_EXIT_COOLDOWN_STEPS
            elif self._regen_steps >= self.REGEN_MAX_STEPS:
                self._regen_mode = False
                self._regen_steps = 0
                self._regen_cooldown = self.REGEN_COOLDOWN_STEPS
        elif health < self.REGEN_ENTER_HEALTH:
            self._regen_mode = True
            self._regen_steps = 0
        return self._regen_mode

    def _restock_active(self) -> bool:
        """当前是否还允许为补箭花步数。

        "地表箭工厂"（产量≈消耗、永远追不上目标，实测 1800 步里 635 步在合成箭）
        由**推进优先模式**兜住：链推进预算一耗尽就关掉补箭。

        试过、被实测否掉的改动（勿重复）：给补箭单独加 120 步窗口 + 200 步冷却，
        想用"限制成本"取代"限制目标"。8 局面板 A/B 显示这是净负——
        `defeat_gnome_warrior/3017` 从 done(1210 步, 15 箭) 退化成 died(767 步, 4 箭)，
        通关 4→3。原因是走到工作台、采料、被战斗打断都吃这个窗口，120 步往往在
        箭到手之前就用完，等于把"无限步补到 8 支"换成了"补不到就下楼"。
        去掉窗口后同一面板回到 4 通关。
        """
        return not self._push_now()

    def _sleep_is_safe(self, summary: Dict[str, Any]) -> bool:
        """睡下去会不会更糟——SLEEP 的有限步前瞻（见 combat_model.project_sleep）。

        SLEEP 锁死动作直到能量回满（约 11 步/点，实测可达 60+ 步），期间无法
        喝水/吃饭/反击。因此按真实衰减速率把整段时长走一遍：睡醒血量过低、
        或睡醒时水/食物见底（醒来即进入掉血状态）→ 不睡，改去补给/推进。
        """
        proj = project_sleep(
            energy=float(summary.get("energy", 9.0)),
            health=float(summary.get("health", 9.0)),
            drink=float(summary.get("drink", 9.0)),
            food=float(summary.get("food", 9.0)),
            strength=int(summary.get("strength", 1)),
            dexterity=int(summary.get("dexterity", 1)),
            thirst_rate=self.thirst_rate,
        )
        if proj.dies or proj.health_end < SLEEP_MIN_PROJECTED_HEALTH:
            return False
        if proj.drink_end <= 0.0 or proj.food_end <= 0.0:
            return False
        return True

    def _waiting_regenerates(self, summary: Dict[str, Any], steps: float = 30.0) -> bool:
        """原地等待是否真的在回血（缺水/缺食/能量耗尽时是净掉血）。"""
        health = float(summary.get("health", 9.0))
        projected = projected_awake_health(
            steps=steps,
            health=health,
            drink=float(summary.get("drink", 9.0)),
            food=float(summary.get("food", 9.0)),
            energy=float(summary.get("energy", 9.0)),
            strength=int(summary.get("strength", 1)),
            dexterity=int(summary.get("dexterity", 1)),
            thirst_rate=self.thirst_rate,
            energy_rate=self.energy_rate,
        )
        return projected > health

    def _mob_within(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], max_dist: int
    ) -> bool:
        """玩家 max_dist 内是否有存活怪（含被动）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        # 被动动物不是战斗威胁，不能阻止补水/睡眠或触发战斗优先级。
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                if abs(int(p[0]) - pos[0]) + abs(int(p[1]) - pos[1]) <= max_dist:
                    return True
        return False

    # 生存维护里"现在就该反击"的距离（可达敌人）；见 _survival_action 注释
    SURVIVAL_ENGAGE_DIST = 3
    # 已清层（monsters_killed>=8，如 L0）的交战半径：那里的击杀不计入任何门槛，
    # 只有"够得着我们的怪"值得花箭。6 = 远程怪射程 5 + 1 格余量。
    CLEARED_FLOOR_ENGAGE_DIST = 6

    def _reachable_hostile_within(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], max_dist: int
    ) -> bool:
        """max_dist 内是否有**可达**的主动怪。

        要求可达：隔着墙的怪打不到我们、我们也过不去（地牢常见），
        对它交战只会原地转圈。
        """
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions") or {}
        reachable = None
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                c = _norm_pos(p)
                if abs(c[0] - pos[0]) + abs(c[1] - pos[1]) > max_dist:
                    continue
                if reachable is None:
                    reachable = self._reachable_set(map_payload["map"], pos)
                if c in reachable:
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

    # -- 掩体 / 庇护所 ------------------------------------------------------
    #
    # 机制依据（game_logic.py 已核实，详见 planner/shelter.py 模块文档）：
    # 近战怪只在曼哈顿距离 ==1 时攻击、投射物撞 solid 即消失、怪在 10-14 格环上
    # 刷新且 75% 概率朝玩家走。因此"三面墙的坑位"同时做到三件事：
    # 同时接战数 4→1、三个方向挡箭、把"等刷怪"变成安全行为。
    # 放置只能封朝向格（move_player 能走就走），所以坑位靠**找**天然凹陷或
    # **挖**石堆得到，放石只用于把 2 面墙补成 3 面。

    COVER_SEARCH_STEPS = 8        # 找掩体的限步半径（移动本身就是暴露）
    RANGED_COVER_STEPS = 12       # 远程压制下的半径：多走几步换"箭道被挡"值得
    RANGED_PRESSURE_WINDOW = 8    # 最近这么多步内被投射物命中 → 判定远程压制
    DAY_LENGTH = 300              # EnvParams.day_length 默认值
    NIGHT_LIGHT_THRESHOLD = 0.3   # light_level 低于此值视为夜
    NIGHT_HOLD_MAX_STEPS = 120    # 单次夜间驻守的步数预算
    NIGHT_HOLD_COOLDOWN = 200     # 预算耗尽后的冷却

    @staticmethod
    def _hostile_cells(map_payload: Dict[str, Any]) -> List[Tuple[int, int]]:
        """本层存活的主动怪坐标。"""
        cells: List[Tuple[int, int]] = []
        mobs = map_payload.get("mob_positions", {})
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                cells.append(_norm_pos(p))
        return cells

    def _cover_walls(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> int:
        """玩家当前所站格的墙数（0-4）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        return wall_count(map_payload["map"], pos)

    def _in_shelter(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> bool:
        return self._cover_walls(map_payload, summary) >= MIN_SHELTER_WALLS

    def _ranged_pressure(self) -> bool:
        """最近是否被远程怪的投射物命中（见 _next_action_inner 的掉血分流）。"""
        return self._steps - self._ranged_hit_step <= self.RANGED_PRESSURE_WINDOW

    def _take_cover(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        min_walls: int = MIN_SHELTER_WALLS,
        max_steps: Optional[int] = None,
    ) -> Optional[int]:
        """进掩体：找坑位 → 挖坑位 → 放石封口。已在掩体或都不可行时返回 None。"""
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        if wall_count(map2d, pos) >= min_walls:
            return None  # 已在掩体里
        steps = self.COVER_SEARCH_STEPS if max_steps is None else max_steps
        mob_cells = getattr(self, "_mob_cells", None)
        hostiles = self._hostile_cells(map_payload)
        inventory = summary.get("inventory") or {}

        # ① 天然坑位（凹陷/走廊尽头）：不消耗任何资源，优先
        cell = find_cover_tile(
            map2d, pos, hostiles, max_steps=steps,
            min_walls=min_walls, extra_blocked=mob_cells,
        )
        if cell is not None:
            walk = self._walk_to(map2d, pos, cell)
            if walk is not None:
                return walk

        # ② 向石堆里挖一格（需木镐）：旷野里唯一能**造出**三面墙坑位的手段，
        #    且顺带 +1 石头。挖出的坑位下一拍会被 ① 找到并走进去。
        if int(inventory.get("pickaxe", 0)) >= 1:
            pocket = find_dig_pocket(
                map2d, pos, max_steps=steps, extra_blocked=mob_cells
            )
            if pocket is not None:
                stand, stone = pocket
                if pos == stand:
                    delta = (stone[0] - pos[0], stone[1] - pos[1])
                    action = DELTA_TO_ACTION.get(delta)
                    if action is not None:
                        direction = int(summary.get("player_direction", 0))
                        return DO if direction == action else action
                else:
                    walk = self._walk_to(map2d, pos, stand)
                    if walk is not None:
                        return walk

        # ③ 放石封口：只能封**朝向格**，所以仅用于把 2 面墙的角落补成 3 面
        if int(inventory.get("stone", 0)) >= 1 and wall_count(map2d, pos) >= min_walls - 1:
            delta = ACTION_DELTA.get(int(summary.get("player_direction", 0)))
            if delta is not None and seal_target(map2d, pos, delta, mob_cells):
                return PLACE_STONE
        return None

    def _is_night(self, summary: Dict[str, Any]) -> bool:
        """当前是否夜间（light_level < 阈值）。

        与 game_logic_utils.calculate_light_level 同式；summary 无 timestep
        （旧客户端）时按白天处理——夜间策略只做减法，缺数据即退回原行为。
        """
        timestep = summary.get("timestep")
        if timestep is None:
            return False
        progress = (int(timestep) / float(self.DAY_LENGTH)) % 1 + 0.3
        light = 1.0 - abs(math.cos(math.pi * progress)) ** 3
        return light < self.NIGHT_LIGHT_THRESHOLD

    def _night_hold_active(self, summary: Dict[str, Any]) -> bool:
        """要不要夜间驻守（带滞回 + 预算 + 冷却，语义同 _regen_mode_active）。

        L0 夜间近战刷新率 0.02 → 0.12/步（constants.FLOOR_MOB_SPAWN_CHANCE 的
        夜间项 ×(1-light)^2）：天黑后在旷野推进等于持续吃刷新。但"天黑就躲"
        会吃掉一局约 30% 的步数（实测 light<0.3 的步占比 546/2000），
        所以只在血/能量不满时驻守（条件驻守）。
        """
        if self._push_now():
            self._night_hold = False
            self._night_hold_steps = 0
            return False
        if self._night_hold_cooldown > 0:
            self._night_hold_cooldown -= 1
            self._night_hold = False
            return False
        night = self._is_night(summary)
        if self._night_hold:
            if not night or self._night_hold_steps >= self.NIGHT_HOLD_MAX_STEPS:
                self._night_hold = False
                self._night_hold_steps = 0
                self._night_hold_cooldown = self.NIGHT_HOLD_COOLDOWN
            else:
                self._night_hold_steps += 1
        elif (night and int(summary.get("floor", 0)) == 0
                and (float(summary.get("health", 9.0)) < 8
                     or float(summary.get("energy", 9.0)) < 5)):
            self._night_hold = True
            self._night_hold_steps = 0
        return self._night_hold

    def _mob_adjacent(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], max_dist: int = 2
    ) -> bool:
        """玩家 max_dist 内是否有主动敌人（紧邻威胁）。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        # 被动动物不是战斗威胁，不能阻止补水/睡眠或触发战斗优先级。
        for key in ("melee", "ranged"):
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
        if tid == "native.collect_drink":
            return self._fill_water(map_payload, summary)
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
            return self._wait_action(summary, map_payload)
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
            target = self._floor_of_descend_task(tid)
            if target is None:
                return None
            return self._descend_to(map_payload, summary, target)
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

    def _fill_water(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """在水源旁装便携水；满容量后退回普通取水逻辑。"""
        inventory = summary.get("inventory") or {}
        if int(inventory.get("water", 0)) >= MAX_WATER_CANTEENS:
            return self._seek_and_do(map_payload, summary, DRINK_BLOCKS)
        return self._seek_and_do(
            map_payload, summary, DRINK_BLOCKS, action_if_adjacent=FILL_WATER
        )

    # -- 通用：走到某类方块旁并 DO -----------------------------------------

    def _collect_resource(
        self,
        tid: str,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Optional[int]:
        """采集任务：本层目标数量充足则采，不足则到更合适的矿石层（可下可上）。

        先保证镐等级：目标方块需要更高镐（如石头需木镐、铁需石镐）时先合成，
        否则在方块旁 DO 永远挖不动（无镐挖石会卡死）。
        """
        floor = int(summary.get("floor", 0))
        target_types = COLLECT_TARGET_BLOCKS[tid]
        map2d = map_payload["map"]
        inventory = summary.get("inventory") or {}
        need_pickaxe = PICKAXE_REQUIRED.get(tid, 0)
        if int(inventory.get("pickaxe", 0)) < need_pickaxe:
            craft_tid = {
                1: "native.craft_wood_pickaxe",
                2: "native.craft_stone_pickaxe",
                3: "native.craft_iron_pickaxe",
                4: "native.craft_diamond_pickaxe",
            }[need_pickaxe]
            a = self._craft_action(craft_tid, map_payload, summary)
            if a is not None:
                return a
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
        # 本层有目标方块 → 就地采（哪怕不足储备量：合成 on-demand 只需 1 块，
        # 采完再按 task 语义决定是否换层）。只有本层完全没有才换层，避免
        # "为采铁先下楼"陷入无装备下深层的死亡螺旋。
        if count >= 1:
            a = self._seek_and_do(map_payload, summary, target_types)
            if a is not None:
                self._collect_visited = set()
                return a
            # 本层目标方块都是不可达的（WATER 分隔）：按不足处理，去下一优先层
        # 本层不足或目标不可达 → 换层。选层时先看**别的层到底有没有**：
        # 优先跨层地图（floor_map_provider，服务端可给任意层全图），其次 seed
        # 扫描事实；"已知没有"的层直接跳过，避免白跑一趟深层（旧实现只能按
        # 静态偏好表顺序试，错了要靠 _collect_visited 兜住上下打转）。
        if getattr(self, "_collect_visited_tid", None) != tid:
            self._collect_visited = set()
            self._collect_visited_tid = tid
        self._collect_visited.add(floor)
        candidates: List[Tuple[int, int, int, int]] = []
        for f in preferred:
            if f == floor or f in self._collect_visited:
                continue
            known = self._floor_resource_count(f, target_types)
            if known is not None and known <= 0:
                continue  # 已知该层没有 → 不去
            rank = 0 if known else 1  # 已知有 > 未知
            candidates.append((rank, abs(f - floor), f, f))
        if not candidates:
            # 无更好的层（都去过 / 都已知没有）：留本层继续采（尽力而为）
            return self._seek_and_do(map_payload, summary, target_types)
        candidates.sort()
        return self._move_to_floor(map_payload, summary, candidates[0][2])

    # -- 跨层观察（扩大边界）------------------------------------------------

    # 跨层地图缓存寿命（步）：地图会因挖掘/放置改变，过期即重新取；
    # 取图在真实部署里是一次 HTTP 调用，不宜每步都拉。
    FLOOR_MAP_TTL_STEPS = 60

    def _floor_map(self, floor: int) -> Optional[Dict[str, Any]]:
        """取指定层的 /map 响应（带 TTL 缓存）。未注入 provider 时返回 None。"""
        if self._floor_map_provider is None:
            return None
        cached_at = self._floor_map_cache_step.get(floor)
        if cached_at is not None and self._steps - cached_at > self.FLOOR_MAP_TTL_STEPS:
            self._floor_map_cache.pop(floor, None)
            self._floor_map_cache_step.pop(floor, None)
        if floor in self._floor_map_cache:
            return self._floor_map_cache[floor]
        self._floor_map_cache_step[floor] = self._steps
        try:
            payload = self._floor_map_provider(floor)
        except Exception:  # noqa: BLE001 - provider 是外部 HTTP 调用，失败即退回未知
            payload = None
        if payload is not None and isinstance(payload.get("map"), list):
            payload = dict(payload)
            payload["map"] = np.asarray(payload["map"], dtype=np.int32)
        self._floor_map_cache[floor] = payload
        return payload

    def _floor_resource_count(
        self, floor: int, target_types: Sequence[int]
    ) -> Optional[int]:
        """指定层上目标方块的数量；无法得知返回 None（未知≠没有）。

        依次尝试：跨层地图（精确）→ seed 扫描的矿石计数（近似，仅矿石）。
        """
        payload = self._floor_map(floor)
        if payload is not None and payload.get("map") is not None:
            return self._count_blocks(payload["map"], list(target_types))
        facts = self.world_facts()
        if facts is not None:
            floor_facts = facts.floor(floor)
            if floor_facts is not None:
                keys = [ORE_BLOCK_TO_KEY[b] for b in target_types if b in ORE_BLOCK_TO_KEY]
                if keys:
                    return sum(floor_facts.ore_count(k) for k in keys)
        return None

    def _seek_and_do(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        target_types: Sequence[int],
        action_if_adjacent: int = DO,
    ) -> Optional[int]:
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        h, w = len(map2d), len(map2d[0])
        target_key_types = tuple(sorted(int(t) for t in target_types))
        floor = int(summary.get("floor", 0))
        origin = map_payload.get("map_origin", [0, 0])
        goal_key = f"collect:{floor}:{target_key_types}"
        # 绝对目标锁只对 streamed-world payload 生效；旧的固定窗口/离线
        # 测试 payload 没有 map_origin，继续使用原有局部寻路语义。
        lock_enabled = "map_origin" in map_payload
        locked_goal = self._goal_locks.get(goal_key) if lock_enabled else None
        if locked_goal is not None:
            goal_local = (
                int(locked_goal[1]) - int(origin[0]),
                int(locked_goal[2]) - int(origin[1]),
            )
            if not (0 <= goal_local[0] < h and 0 <= goal_local[1] < w):
                chase = self._step_toward_global_goal(
                    map_payload, summary, locked_goal[1:]
                )
                if chase is not None:
                    return chase
                return NOOP
            elif int(map2d[goal_local[0]][goal_local[1]]) not in target_types:
                # 目标已被采集/世界状态已确认改变，允许选择下一个目标。
                self._goal_locks.pop(goal_key, None)
                self._goal_routes.pop(goal_key, None)
                locked_goal = None
        excluded = [
            (r - int(origin[0]), c - int(origin[1]))
            for f, r, c, types in self._blocked_collect_targets
            if f == floor and types == target_key_types
        ]
        if lock_enabled and locked_goal is not None and 0 <= goal_local[0] < h and 0 <= goal_local[1] < w:
            # 目标锁生效期间，当前窗口内其它同类资源全部排除；跨窗后
            # 仍沿用 locked_goal 的绝对坐标，不会因新 chunk 换目标。
            for r in range(h):
                for c in range(w):
                    if (r, c) != goal_local and int(map2d[r][c]) in target_types:
                        excluded.append((r, c))

        def arm_target(local_target: Tuple[int, int], armed: bool) -> None:
            global_target = (
                floor, int(origin[0]) + local_target[0],
                int(origin[1]) + local_target[1],
            )
            if lock_enabled:
                self._goal_locks[goal_key] = global_target
                self._pending_collect_target = (
                    floor, global_target[1], global_target[2], target_key_types,
                )
                self._pending_collect_armed = armed

        def tile(p: Tuple[int, int]) -> int:
            return int(map2d[p[0]][p[1]])

        def cached_route_step() -> Optional[int]:
            """Follow the cached absolute route until it leaves the window."""
            if not (lock_enabled and locked_goal is not None):
                return None
            global_pos = map_payload.get("player_global_position")
            if global_pos is None:
                global_pos = [int(origin[0]) + pos[0], int(origin[1]) + pos[1]]
            current = (int(global_pos[0]), int(global_pos[1]))
            route = self._goal_routes.get(goal_key)
            if route:
                while route and route[0] == current:
                    route.pop(0)
            if not route:
                # Build a route to a walkable cell adjacent to the locked target.
                # The target itself is normally solid (tree/ore/water).
                from collections import deque
                queue = deque([(pos, [])])
                visited = {pos}
                route_global: Optional[List[Tuple[int, int]]] = None
                while queue:
                    cell, path = queue.popleft()
                    for delta in ACTION_DELTA.values():
                        nxt = (cell[0] + delta[0], cell[1] + delta[1])
                        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                            continue
                        if nxt in visited:
                            continue
                        if nxt == goal_local:
                            route_global = [
                                (int(origin[0]) + p[0], int(origin[1]) + p[1])
                                for p in (path + [cell])
                                if p != pos
                            ]
                            break
                        if blocked(tile(nxt)) or nxt in getattr(self, "_mob_cells", set()):
                            continue
                        visited.add(nxt)
                        queue.append((nxt, path + [cell]))
                    if route_global is not None:
                        break
                if route_global is None:
                    return None
                self._goal_routes[goal_key] = route_global
                route = route_global
            if not route:
                return None
            next_global = route[0]
            local = (next_global[0] - int(origin[0]), next_global[1] - int(origin[1]))
            if not (0 <= local[0] < h and 0 <= local[1] < w):
                return self._step_toward_global_goal(map_payload, summary, next_global)
            # 放置/挖掘可能刚刚改变了缓存路径上的方块；不能继续朝
            # 已变成实心方块的旧 waypoint 撞击，否则又会形成固定方向僵持。
            if blocked(tile(local)):
                self._goal_routes.pop(goal_key, None)
                return None
            delta = (local[0] - pos[0], local[1] - pos[1])
            return DELTA_TO_ACTION.get(delta)

        # 已站在目标旁且面向它 → DO
        for delta, action in DELTA_TO_ACTION.items():
            adj = (pos[0] + delta[0], pos[1] + delta[1])
            if (0 <= adj[0] < h and 0 <= adj[1] < w
                    and tile(adj) in target_types
                    and (locked_goal is None or adj == goal_local)):
                if direction == action:
                    arm_target(adj, True)
                    return action_if_adjacent
                return action  # 转向
        result = find_nearest_target(
            map2d, pos, list(target_types),
            extra_blocked=getattr(self, "_mob_cells", None),
            excluded_targets=excluded,
        )
        if result is None:
            if lock_enabled and locked_goal is not None:
                # 锁定目标的路线可能被一棵树/矿石挡住。允许对相邻障碍
                # 做一次原生交互，但不改变最终绝对目标锁；否则盲目 fallback
                # 会每步返回“撞向树”的 LEFT/RIGHT，形成抽搐。
                for delta, action in DELTA_TO_ACTION.items():
                    adj = (pos[0] + delta[0], pos[1] + delta[1])
                    if (0 <= adj[0] < h and 0 <= adj[1] < w
                            and int(map2d[adj[0]][adj[1]]) in target_types):
                        if direction == action:
                            return action_if_adjacent
                        return action
                failures = getattr(self, "_goal_path_failures", Counter())
                failures[goal_key] += 1
                self._goal_path_failures = failures
                # 保持绝对目标优先，但动态怪物/局部编辑可能让当前路径
                # 暂时不可达；连续多次确认后才允许重新选同类目标。
                if failures[goal_key] >= 8:
                    self._goal_locks.pop(goal_key, None)
                    self._goal_routes.pop(goal_key, None)
                    self._goal_path_failures.pop(goal_key, None)
                    locked_goal = None
                    excluded = [
                        (r - int(origin[0]), c - int(origin[1]))
                        for f, r, c, types in self._blocked_collect_targets
                        if f == floor and types == target_key_types
                    ]
                    result = find_nearest_target(
                        map2d, pos, list(target_types),
                        extra_blocked=getattr(self, "_mob_cells", None),
                        excluded_targets=excluded,
                    )
                    if result is not None:
                        target_local, first_delta = result
                        self._goal_locks[goal_key] = (
                            floor, int(origin[0]) + target_local[0],
                            int(origin[1]) + target_local[1],
                        )
                        return DELTA_TO_ACTION.get(first_delta)
                # 目标锁仍然有效时，不能退回到会选择其它资源的 blind
                # fallback；等待下一次 chunk/怪物状态更新即可。
                return self._wait_action(summary, map_payload)
            # mob 挡路（临时）：忽略怪可达则等待怪移动/消失，避免误判无目标
            blind = find_nearest_target(map2d, pos, list(target_types))
            if blind is not None:
                return self._wait_action(summary, map_payload)
            return None
        target_local, first_delta = result
        if lock_enabled and locked_goal is None:
            self._goal_locks[goal_key] = (
                floor, int(origin[0]) + target_local[0],
                int(origin[1]) + target_local[1],
            )
            locked_goal = self._goal_locks[goal_key]
            goal_local = (
                int(locked_goal[1]) - int(origin[0]),
                int(locked_goal[2]) - int(origin[1]),
            )
            self._goal_routes.pop(goal_key, None)
        if lock_enabled and locked_goal is not None:
            routed = cached_route_step()
            if routed is not None:
                return routed
        if first_delta in DELTA_TO_ACTION:
            return DELTA_TO_ACTION[first_delta]
        return None

    def _step_toward_global_goal(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], goal: Sequence[int]
    ) -> Optional[int]:
        """Move one local step toward a locked absolute goal.

        This is only used while the target is outside the current 3x3/5x5
        observation window. The absolute goal remains unchanged; local map
        collisions may choose an alternate cardinal step, but never a new
        resource/ladder target.
        """
        origin = map_payload.get("map_origin", [0, 0])
        global_pos = map_payload.get("player_global_position")
        if global_pos is None:
            local = _norm_pos(summary.get("player_position", [0, 0]))
            global_pos = [int(origin[0]) + local[0], int(origin[1]) + local[1]]
        dr = int(goal[0]) - int(global_pos[0])
        dc = int(goal[1]) - int(global_pos[1])
        if dr == 0 and dc == 0:
            return None
        preferred = []
        if abs(dr) >= abs(dc) and dr:
            preferred.append(DOWN if dr > 0 else UP)
        if dc:
            preferred.append(RIGHT if dc > 0 else LEFT)
        if dr and preferred == []:
            preferred.append(DOWN if dr > 0 else UP)
        map2d = map_payload.get("map")
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        blocked_extra = getattr(self, "_mob_cells", None) or set()
        for action in preferred:
            delta = ACTION_DELTA[action]
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if (0 <= nxt[0] < len(map2d) and 0 <= nxt[1] < len(map2d[0])
                    and not blocked(int(map2d[nxt[0]][nxt[1]]))
                    and nxt not in blocked_extra):
                return action
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
        # 附近无工作台 → 先走向已有工作台（避免每步就地放台、散落一堆）；
        # 本层确实没有 → 就地放一个（深层也能合成，不依赖地表基础设施）
        result = find_nearest_target(
            map2d, pos, [CRAFT_TABLE_BLOCK], extra_blocked=getattr(self, "_mob_cells", None)
        )
        if result is not None:
            _, first_delta = result
            if first_delta in DELTA_TO_ACTION:
                return DELTA_TO_ACTION[first_delta]
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

    @staticmethod
    def _count_blocks(map2d, types) -> int:
        """统计本层目标方块数量。"""
        count = 0
        for row in map2d:
            for tile in row:
                if int(tile) in types:
                    count += 1
        return count

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
        floor = int(summary.get("floor", 0))
        origin = map_payload.get("map_origin", [0, 0])
        goal_key = f"chest:{floor}"
        locked = self._goal_locks.get(goal_key)
        if locked is not None:
            best = [int(locked[1]) - int(origin[0]), int(locked[2]) - int(origin[1])]
            if not (0 <= best[0] < h and 0 <= best[1] < w):
                chase = self._step_toward_global_goal(map_payload, summary, locked[1:])
                if chase is not None:
                    return chase
                return NOOP
            elif int(map2d[best[0]][best[1]]) != CHEST:
                self._goal_locks.pop(goal_key, None)
                locked = None
        # 最近宝箱
        if locked is None:
            best = None
            best_dist = 10 ** 9
            for c in chests:
                cpos = _norm_pos(c)
                dist = abs(cpos[0] - pos[0]) + abs(cpos[1] - pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best = cpos
            if best is not None:
                self._goal_locks[goal_key] = (
                    floor, int(origin[0]) + best[0], int(origin[1]) + best[1]
                )
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
        # 递归保护：_descend_to → 制备(_craft_action) → 补资源(_collect_resource)
        # → _move_to_floor → _descend_to 会无限递归（本层缺某矿时）。重入时放弃
        # 本轮制备（缺料装备按软门槛跳过，双轨制兜底）。
        if getattr(self, "_descend_depth", 0) > 0:
            return None
        self._descend_depth = getattr(self, "_descend_depth", 0) + 1
        try:
            return self._descend_to_inner(map_payload, summary, target_floor)
        finally:
            self._descend_depth -= 1

    def _descend_to_inner(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], target_floor: int
    ) -> Optional[int]:
        floor = int(summary.get("floor", 0))
        inventory = summary.get("inventory") or {}
        monsters_killed = int(map_payload.get("monsters_killed", 0))

        # 下楼前储备生存资源。注意：food/drink 由 _survival_action 全局维护，
        # 这里只保证"下无水源/食物层"前满值；普通层阈值放低（5），避免下楼前
        # 远途找水/食而过度暴露（表层制备阶段尤其致命）。
        drink = float(summary.get("drink", 9.0))
        food = float(summary.get("food", 9.0))
        next_floor = floor + 1
        drink_thresh = 9 if next_floor in NO_DRINK_FLOORS else 5
        food_thresh = 9 if next_floor in NO_FOOD_FLOORS else 5
        carried_water = int(inventory.get("water", 0))
        # 便携水是有效储备，不要求在下楼瞬间把 player_drink 补满；只在
        # 当前水量 + 可携带恢复量仍不足时回水源，减少无意义的往返。
        effective_drink = drink + carried_water * WATER_DRINK_AMOUNT
        if effective_drink < drink_thresh:
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
                return self._wait_action(summary, map_payload)  # 等被动怪刷新补充食物

        # 血线门（下楼前）：链上还要"穿过"下一层 → 到那层就得清 8 怪，
        # 而深层没有水/被动怪可依赖，恢复要折返。因此在**当前层（已清、相对安全）**
        # 恢复到接近满血再下：能量未满就睡（13 步/HP + 回能量），否则原地被动回血。
        # 实测：带着 4-6 血下 L1 会在抵达后 20 步内被兽人打死。
        # 推进优先模式下把这道门降到 PUSH_DESCEND_HEALTH：否则"进入推进模式"只是
        # 把控制权交给下楼路径，而下楼路径自己又在这里睡到 8 血，循环照旧。
        health_now = float(summary.get("health", 9.0))
        regen_gate = (self.PUSH_DESCEND_HEALTH if self._push_now()
                      else self.REGEN_EXIT_HEALTH)
        if (self._max_floor > next_floor
                and health_now < regen_gate
                and not self._mob_adjacent(map_payload, summary, 1)):
            if (self._sleep_is_safe(summary)
                    and self._nearest_hostile_dist(map_payload, summary) >= 14
                    and float(summary.get("energy", 9.0)) < 9.0):
                cover = self._take_cover(
                    map_payload, summary, min_walls=MIN_SHELTER_WALLS
                )
                if cover is not None:
                    return cover
                return SLEEP
            safe = self._safe_sleep_spot_walk(map_payload, summary, max_steps=8)
            if safe is not None:
                return safe
            return DO

        # 弹药经济学（下楼前）：链上还要"穿过"下一层时必须在那层清 8 怪，而
        # **地牢层没有树 → 箭在下楼后不可再生**（MAKE_ARROW 要 1 木 + 1 石）。
        # 因此"有弓就跳过深制备"只在弹药可补时成立；不可补时按 arrows_for_clear
        # （备箭）与 damage_per_clear（升剑）折算到同一货币——每 1 木+1 石 能省下
        # 的清层受击——择优（combat_model.recommend_clear_prep）。
        # 石剑与 2 支箭同价（1 木 + 1 石），所以弹药缺口大时"先造石剑"通常更划算。
        prep = None
        feedstock = 0
        stone_feedstock = 0
        if self._max_floor > next_floor:
            prep = self._clear_prep(next_floor, map_payload, summary)
            self._prep_note = prep.reason
            if prep.arrows_min > 0:
                self._restock_target = max(self._restock_target, prep.arrows_min)
            # 更深的过路层：木头只有地表有，箭在 L1+ 只能靠背包木料现做 →
            # 预留这些木不参与"给下一层多备几支箭"（否则采木/合成箭互相消耗打转）。
            if floor == 0:
                feedstock = self._arrow_feedstock_target(next_floor, summary)
                stone_feedstock = self._arrow_feedstock_stone_target(
                    next_floor, summary
                )
        # 备箭：只在有木石的本层补（软门槛，不为了备箭跨层跑——那会引出上下楼循环）。
        # prefer=="sword" 时先升剑再回来备箭（同样的材料买到更多减伤）；
        # prefer=="ready" 时不再为清层备箭（剑已与弓等效或已备齐）。
        # 两档目标：arrows_min（均值）值得专程采料；超出的预留只用手头余料补，
        # 否则地表刷怪的持续消耗会让"备满预留"永远达不成（补给循环吃掉整局）。
        if prep is not None and prep.prefer == "arrows" and self._has_bow(summary):
            arrows_now = int(inventory.get("arrows", 0))
            spare_materials = (int(inventory.get("wood", 0)) >= feedstock + 1
                               and int(inventory.get("stone", 0)) >= stone_feedstock + 1)
            if (arrows_now < prep.arrows_min
                    and self._arrow_materials_available(map_payload, summary)) or (
                    arrows_now < prep.arrows_target and spare_materials):
                craft = self._craft_action("native.craft_arrow", map_payload, summary)
                if craft is not None:
                    return craft
        # 过路层的原料：木与石都要带够再下（MAKE_ARROW = 1 木 + 1 石）。
        # 只带木是原来的缺陷：三次下楼都是 8 木 + 0 石 → 深层一支箭也做不出。
        if feedstock > 0 and int(inventory.get("wood", 0)) < feedstock:
            a = self._collect_resource("native.collect_wood", map_payload, summary)
            if a is not None:
                return a
        if stone_feedstock > 0 and int(inventory.get("stone", 0)) < stone_feedstock:
            a = self._collect_resource("native.collect_stone", map_payload, summary)
            if a is not None:
                return a

        # 下地牢前准备工具与武器链（表层制备，避免无装备硬下深层）。
        # 按链上"最深层需求"（_max_floor）制备——enter_dungeon 的 target 只是
        # L1（只需到达，不强制清怪）→ 木剑即可快速下；L2+（需清 L1 8 怪）→
        # 完整制备。顺序刻意让"剑"尽早升级（击杀越快受击越少）：
        # 木剑 → 木镐 → 石剑(2击杀 L0 僵尸，受击减半) → 石镐 → 铁剑(1击，0受击)。
        # 有弓时不再无条件跳过：由上面的择优给出 sword_target（弹药可补/已备齐
        # → 0，即退回"跳过深制备"；弹药有缺口 → 石剑/铁剑先做）。
        deep_need = self._max_floor >= 2
        has_bow = self._has_bow(summary)
        pickaxe_level = int(inventory.get("pickaxe", 0))
        sword_level = int(inventory.get("sword", 0))
        if self._max_floor >= 1 and sword_level < 1:
            craft = self._craft_action("native.craft_wood_sword", map_payload, summary)
            if craft is not None:
                return craft
        # 木镐排在最前：石头是石剑/石镐/箭/熔炉的共同原料，没有木镐一样都做不出。
        if deep_need and pickaxe_level < 1:
            craft = self._craft_action("native.craft_wood_pickaxe", map_payload, summary)
            if craft is not None:
                return craft

        # 剑阶梯（先于镐阶梯）：剑直接降低下一层的清怪受击，是**当下**的生存收益；
        # 镐是**后续**采矿链的前置。两者同价（1 木 + 1 石），故先剑后镐。
        sword_target = 0 if has_bow else (3 if deep_need else 0)
        if has_bow and prep is not None:
            sword_target = prep.sword_target
        if deep_need and sword_target > sword_level:
            if sword_level < 2:
                craft = self._craft_action("native.craft_stone_sword", map_payload, summary)
                if craft is not None:
                    return craft
            # 铁剑：本层能凑齐铁+煤则做（兽人 2 击杀，单怪受击与弓等效）；否则放弃。
            if (sword_level < 3 and sword_target >= 3
                    and self._iron_craft_feasible(map_payload, summary)):
                craft = self._craft_action("native.craft_iron_sword", map_payload, summary)
                if craft is not None:
                    return craft

        # 镐阶梯：与"备箭 vs 升剑"的择优**解耦**。镐不是武器，它决定能采到什么：
        # 木镐→石头（石剑/箭/熔炉），石镐→铁（铁剑/铁甲/铁镐），铁镐→钻石。
        # 旧实现把石镐挂在 `sword_target >= 3 or not has_bow` 下，而择优通常给
        # sword_target=2 → 木镐锁死 → 永远采不到铁 → 铁剑/铁甲/钻石整条链断掉
        # （实测 8/8 局死亡时 pickaxe 恒为 1）。这里按链上最深层需求逐级升。
        pickaxe_target = 1 if self._max_floor >= 1 else 0
        if deep_need:
            pickaxe_target = 2          # 石镐 = 采铁的前置
        if self._max_floor >= 4:
            pickaxe_target = 3          # 铁镐 = 采钻石的前置
        if pickaxe_target > pickaxe_level:
            next_pickaxe = {
                1: "native.craft_wood_pickaxe",
                2: "native.craft_stone_pickaxe",
                3: "native.craft_iron_pickaxe",
            }[min(pickaxe_level + 1, 3)]
            # 逐级升（做上一级镐才能采下一级的料）；铁镐需要铁+煤+熔炉，
            # 本层凑不齐就不为它停留（软门槛，双轨制兜底）。
            if (next_pickaxe != "native.craft_iron_pickaxe"
                    or self._iron_craft_feasible(map_payload, summary)):
                craft = self._craft_action(next_pickaxe, map_payload, summary)
                if craft is not None:
                    return craft

        # 择优选了升剑但本层做不出来（缺石/缺铁）→ 退回备箭，别空手下楼
        if (prep is not None and prep.prefer == "sword" and has_bow
                and int(inventory.get("arrows", 0)) < prep.arrows_min
                and self._arrow_materials_available(map_payload, summary)):
            craft = self._craft_action("native.craft_arrow", map_payload, summary)
            if craft is not None:
                return craft

        # 下地牢前尽量穿铁甲（4 件 40% 物抗，深层清怪生存关键）。
        # 仅深层任务（max_floor>=2）尝试：仅本层采铁/煤（_craft_armour_until
        # 不跨层）；浅层不足则放弃，双轨制兜底。L1 任务（只需到达）不做——
        # 采 3 铁 3 煤的暴露远大于收益。有弓时不做（弓点射已足够，甲留待深层）。
        armour_levels = [int(x) for x in inventory.get("armour", [0])]
        if sum(armour_levels) < 1 and self._max_floor >= 2 and not has_bow:
            craft = self._craft_armour_until(map_payload, summary, 1)
            if craft is not None:
                return craft

        # 下火界(L6)/冰界(L7)前需元素战斗能力（剑/弓附魔或对应法术），
        # 否则无附魔硬打（90% 物抗）几乎必死。无法获取 → 放弃该 seed。
        if next_floor in (6, 7) and not self._has_elemental_capability(summary, next_floor):
            a = self._acquire_elemental_capability(map_payload, summary, next_floor)
            if a is not None:
                return a
            return None

        # 弹药门：无箭的弓不算战斗装备。旧的"软准备"路径允许打完最后一支箭就下楼，
        # 玩家会被困在敌对层里既无远程也无近战。实测两局都是带 2 支箭下 L1、
        # 箭尽后被近战打死（其中一局返回地表补了 2 支又下去，仍死在 L1），
        # 所以门槛不能是"至少 1 支"，按基础储备 BOW_ARROW_RESERVE 卡。
        # 只在"还要穿过本层继续下潜"时生效（max_floor > next_floor）：
        # 首次下楼与到达最终目标层不受此门阻塞。清层的完整预算由准备路径给（更大）。
        # 逃逸条件（_arrow_materials_available）不可省：地牢层没有树，木料耗尽后
        # 这道门若仍然生效，_craft_action 会一直把玩家推回工作台却永远造不出箭
        # （旧版实测 635 步都在"合成箭"、一次没下楼）。补不了就放行，别原地饿死。
        arrows_now = int(inventory.get("arrows", 0))
        if (self._has_bow(summary) and self._max_floor > next_floor
                and arrows_now < BOW_ARROW_RESERVE
                and self._arrow_materials_available(map_payload, summary)):
            craft = self._craft_action("native.craft_arrow", map_payload, summary)
            if craft is not None:
                return craft
            stone = int(inventory.get("stone", 0))
            if stone < 1:
                collect = self._collect_resource(
                    "native.collect_stone", map_payload, summary
                )
                if collect is not None:
                    return collect

        # 本层未清 8 怪 → 需清怪才能下楼（原生规则）。
        # 注：L0 不主动强清——采集/自守期间僵尸会被顺带击杀，强清反而多挨打；
        #     在梯子口按住 DESCEND，游戏会在 monsters_killed[0]>=8 后放行（原行为）。
        if floor > 0 and monsters_killed < 8:
            return self._combat_any(map_payload, summary)

        # 楼层就绪门：仅当链还需"穿过"本层下更深（max_floor > target_floor）时
        # 强校验装备——到达最终目标层（任务在到达时即完成）只保证能到，不强求清怪装备。
        # 缺失项（剑/甲/力量/元素/生存）先尽力补齐；硬门槛（元素/生存 INFEASIBLE）
        # 补不上 → 中止该 seed（批量+锚点恢复只兜底 MARGINAL，不兜底 INFEASIBLE）。
        if self._max_floor > target_floor:
            ok, missing = check_floor_readiness(
                target_floor, summary, self.world_facts(),
                arrows_restockable=self._floor_can_restock_arrows(target_floor),
            )
            if not ok:
                action = self._resolve_gate(missing, map_payload, summary, target_floor)
                if action is not None:
                    return action
                if any(k in ("elemental", "survival", "ladder") for k, _ in missing):
                    self._abort_reason = (
                        f"floor {target_floor} 就绪门无法通过: {missing}"
                    )
                    return None
                # 软门槛（剑/甲缺资源）补不上 → 继续（双轨制：无甲走风筝+锚点恢复）

        # 站到 LADDER_DOWN 上并按 DESCEND
        return self._ladder_descend(map_payload, summary)

    def _ladder_descend(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """走向 LADDER_DOWN 并按 DESCEND（含挡路怪处理）。

        用于常规下行与弓先制的"快速下 L1"（L0 已清、无杀怪门槛）。
        """
        floor = int(summary.get("floor", 0))
        ladder = map_payload.get("ladder_down")
        ladder_global = map_payload.get("ladder_down_global")
        goal_key = f"ladder_down:{floor}"
        locked = self._goal_locks.get(goal_key)
        if locked is None and ladder_global is not None:
            locked = (floor, int(ladder_global[0]), int(ladder_global[1]))
            self._goal_locks[goal_key] = locked
        if locked is not None:
            origin = map_payload.get("map_origin", [0, 0])
            ladder = [int(locked[1]) - int(origin[0]), int(locked[2]) - int(origin[1])]
        if ladder is None:
            return None
        ladder_pos = _norm_pos(ladder)
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        if pos == ladder_pos:
            return DESCEND
        map2d = map_payload["map"]
        h, w = len(map2d), len(map2d[0])
        if not (0 <= ladder_pos[0] < h and 0 <= ladder_pos[1] < w):
            return self._step_toward_global_goal(map_payload, summary, locked[1:]) if locked else None
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
        self._goal_locks.pop(goal_key, None)
        return None

    def _bow_rush(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> Optional[int]:
        """弓先制：深层任务在深制备前先拿 L1 弓（打断当前子目标）。

        流程：保证木剑（自保）→ 下 L1（L0 已清、无杀怪门槛）→ 开首箱拿弓。
        L1 首箱必出弓（game_logic.add_items_from_chest 确定性规则），
        是打破"L0 制备生存墙"的关键前置：拿到弓后制备/清怪都只需 0-1 受击。
        已拿弓或拿不到弓返回 None（调用方继续正常流程）。
        """
        if self._has_bow(summary):
            # 首次确认有弓 → 设置补给目标（在锚点补满箭后再推进下楼）
            if self._restock_target == 0:
                self._restock_target = BOW_ARROW_RESERVE
            return None
        inventory = summary.get("inventory") or {}
        floor = int(summary.get("floor", 0))
        # 1) 自保木剑（1 木；缺木就地采）
        if int(inventory.get("sword", 0)) < 1:
            if int(inventory.get("wood", 0)) < 2:
                a = self._collect_resource("native.collect_wood", map_payload, summary)
                if a is not None:
                    return a
            craft = self._craft_action("native.craft_wood_sword", map_payload, summary)
            if craft is not None:
                return craft
        # 2) 预储备箭料（2 木 + 2 石）：拿弓回程后可立即合成箭自保，
        #    避免在 L0 无箭状态下长途采料被接战拖死。
        if floor == 0 and self._max_floor >= 2:
            if int(inventory.get("wood", 0)) < 2:
                a = self._collect_resource("native.collect_wood", map_payload, summary)
                if a is not None:
                    return a
            if int(inventory.get("stone", 0)) < 2:
                a = self._collect_resource("native.collect_stone", map_payload, summary)
                if a is not None:
                    return a
        # 3) 下 L1 拿弓
        if floor < 1:
            a = self._ladder_descend(map_payload, summary)
            if a is not None:
                return a
            return None  # 梯子不可达（非 golden seed）：放弃弓先制，走常规
        if floor == 1:
            a = self._open_chest(map_payload, summary)
            if a is not None:
                return a
            return None  # 无宝箱可开（异常）：放弃
        return None

    def _ascend_to(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], target_floor: int
    ) -> Optional[int]:
        """上行到指定楼层（走回 ladder_up 并按 ASCEND）。L0 无 ladder_up，只能上到 L0。"""
        floor = int(summary.get("floor", 0))
        if floor <= target_floor:
            return None  # 已在目标层或更浅
        ladder = map_payload.get("ladder_up")
        ladder_global = map_payload.get("ladder_up_global")
        goal_key = f"ladder_up:{floor}:{target_floor}"
        locked = self._goal_locks.get(goal_key)
        if locked is None and ladder_global is not None:
            locked = (floor, int(ladder_global[0]), int(ladder_global[1]))
            self._goal_locks[goal_key] = locked
        if locked is not None:
            origin = map_payload.get("map_origin", [0, 0])
            ladder = [int(locked[1]) - int(origin[0]), int(locked[2]) - int(origin[1])]
        if ladder is None:
            return None
        ladder_pos = _norm_pos(ladder)
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        if pos == ladder_pos:
            return ASCEND
        map2d = map_payload["map"]
        h, w = len(map2d), len(map2d[0])
        if not (0 <= ladder_pos[0] < h and 0 <= ladder_pos[1] < w):
            return self._step_toward_global_goal(map_payload, summary, locked[1:]) if locked else None
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
        self._goal_locks.pop(goal_key, None)
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

    # -- 弓战斗：远程点射（L1 首箱弓，0-1 受击清怪的核心武器）---------------

    ARROW_BUDGET_CAP = 30  # 备箭上限（再多也带不动，且备箭本身要采料）
    # 试过、被实测否掉的改动（勿重复）：把"下楼前值得专程采料"的门槛从
    # arrows_min（清满 8 只的预算）降到"一个批次"（10 支），理由是批量清怪 +
    # 锚点恢复本来就是分批的。16 局成对 A/B：死亡 8→7、成就 18.6→18.8、
    # L1 击杀 12→12，但**唯一一次通关（seed 2027 到 L2）没了**（done→timeout）。
    # 净收益为负，已回退。
    ARROW_FEEDSTOCK_CAP = 8  # 带下楼的木料上限（每 1 木 = 2 箭；+2 供深层放工作台）

    def _gear_for(self, floor: int, summary: Dict[str, Any]) -> Gear:
        """按 summary 构造战斗模型的装备快照（floor 决定元素能力口径）。"""
        inventory = summary.get("inventory") or {}
        return Gear(
            sword=int(inventory.get("sword", 0)),
            armour=sum(int(x) for x in inventory.get("armour", [0])),
            strength=int(summary.get("strength", 1)),
            dexterity=int(summary.get("dexterity", 1)),
            intelligence=int(summary.get("intelligence", 1)),
            has_elemental=has_elemental(summary, floor),
            bow=int(inventory.get("bow", 0)),
            bow_enchant=int(summary.get("bow_enchantment", 0)),
        )

    def _arrow_budget(
        self, floor: int, summary: Dict[str, Any], restockable: bool = False
    ) -> int:
        """清目标层所需箭数（combat_model.arrows_for_clear），不低于基础储备。

        restockable=False（默认，地牢层无木 → 箭不可再生）时按更大的预留系数算。
        """
        from craftax.planner.combat_model import arrows_for_clear

        need = arrows_for_clear(floor, self._gear_for(floor, summary), restockable)
        if need <= 0:
            return 0
        return int(min(self.ARROW_BUDGET_CAP, max(BOW_ARROW_RESERVE, need)))

    def _floor_can_restock_arrows(self, floor: int) -> bool:
        """目标层能否就地补箭（同时有树与石）。

        MAKE_ARROW = 1 木 + 1 石，而地牢层（L1-L5）没有树 —— 这是"备箭 vs 升剑"
        的决定性事实。有跨层地图（GET /map?floor=N）就实测，没有则按游戏地形
        约定：只有地表（L0）与火界（L6，FIRE_TREE）可能有木。
        """
        trees = self._floor_resource_count(floor, [TREE, FIRE_TREE])
        stones = self._floor_resource_count(floor, [STONE])
        if trees is None or stones is None:
            return floor in (0, 6)
        return trees > 0 and stones > 0

    def _clear_prep(
        self, floor: int, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> ClearPrep:
        """目标层"清 8 怪"的制备择优：备箭还是先造石/铁剑。

        把两个选项折算到同一货币（每 1 木 + 1 石 能省下的清层受击）后比较，
        依据是 combat_model.arrows_for_clear 与 damage_per_clear；
        铁剑只在**本层**（制备地点）拿得到铁与煤时才作为候选。
        max_sword=3：这里只考虑下楼前做得出来的剑（钻石剑属于链上任务，
        当成目标会让制备链无限打转）。
        """
        from craftax.planner.combat_model import recommend_clear_prep

        inventory = summary.get("inventory") or {}
        map2d = map_payload["map"]
        iron_here = (int(inventory.get("iron", 0)) >= 1
                     or self._count_blocks(map2d, [IRON]) > 0)
        coal_here = (int(inventory.get("coal", 0)) >= 1
                     or self._count_blocks(map2d, [COAL]) > 0)
        return recommend_clear_prep(
            floor,
            self._gear_for(floor, summary),
            arrows=int(inventory.get("arrows", 0)),
            restockable=self._floor_can_restock_arrows(floor),
            iron_available=iron_here and coal_here,
            arrow_cap=self.ARROW_BUDGET_CAP,
            max_sword=3,
        )

    def _arrow_feedstock_crafts(self, next_floor: int, summary: Dict[str, Any]) -> int:
        """深层现做箭所需的合成次数（每次 1 木 + 1 石 → ARROWS_PER_CRAFT 支）。

        只统计**还要穿过**的层（f < _max_floor：终点层只需到达，不必清怪），
        且下一层本身的箭已在本层备齐 → 从 next_floor+1 起算；按其中最贵的一层
        估算。0 = 不必带料。
        """
        deeper = [f for f in range(next_floor + 1, self._max_floor)
                  if not self._floor_can_restock_arrows(f)]
        if not deeper:
            return 0
        need = max(self._arrow_budget(f, summary) for f in deeper)
        if need <= 0:
            return 0
        from craftax.planner.combat_model import ARROWS_PER_CRAFT

        return int(math.ceil(need / ARROWS_PER_CRAFT))

    def _arrow_feedstock_target(self, next_floor: int, summary: Dict[str, Any]) -> int:
        """带下楼的木料数（合成次数 + 2 木给深层放工作台）。"""
        crafts = self._arrow_feedstock_crafts(next_floor, summary)
        if crafts <= 0:
            return 0
        return int(min(self.ARROW_FEEDSTOCK_CAP, crafts + 2))

    def _arrow_feedstock_stone_target(
        self, next_floor: int, summary: Dict[str, Any]
    ) -> int:
        """带下楼的**石料**数。

        MAKE_ARROW = 1 木 + 1 石，所以只带木等于一支箭也做不出来。旧实现只有
        木料这一路（_arrow_feedstock_target），实测三次下楼时背包石头都是 0、
        木头都是 8——弓在 L1 打空后彻底失去补给能力，而"无箭的弓不算装备"的
        弹药门也因此永远判定"补不了"而放行。石头不留放工作台的余量（工作台
        只要木），故比木料目标少 2。
        """
        crafts = self._arrow_feedstock_crafts(next_floor, summary)
        if crafts <= 0:
            return 0
        return int(min(self.ARROW_FEEDSTOCK_CAP, crafts))

    def _arrow_materials_available(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> bool:
        """本层能否就地补箭（背包或本层地图里同时有木与石）。"""
        inventory = summary.get("inventory") or {}
        map2d = map_payload["map"]
        wood_ok = int(inventory.get("wood", 0)) >= 1 or self._count_blocks(
            map2d, [TREE, FIRE_TREE]
        ) > 0
        stone_ok = int(inventory.get("stone", 0)) >= 1 or self._count_blocks(
            map2d, [STONE]
        ) > 0
        return wood_ok and stone_ok

    def _has_bow(self, summary: Dict[str, Any]) -> bool:
        inventory = summary.get("inventory") or {}
        return int(inventory.get("bow", 0)) >= 1

    def _should_use_bow(self, floor: int, summary: Dict[str, Any]) -> bool:
        """当前层是否用弓主战。

        L0-L3 弓优于/等价近战（箭 1-3 发/怪，受击 0-1；弓主动防御防骷髅耗血）；
        L4+ 骑士/巨魔物免高，需附魔弓或近战配合；L6/L7 只有附魔对应元素时弓才有用。
        """
        if not self._has_bow(summary):
            return False
        if floor in (6, 7):
            return self._has_elemental_capability(summary, floor)
        return True

    @staticmethod
    def _line_clear(map2d, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """检查从 pos 沿直线到 target 是否无 solid 阻挡（target 格本身除外）。

        target 必须与 pos 同行或同列；用于判断"射箭能否命中"，避免射进墙里浪费箭。
        """
        h, w = len(map2d), len(map2d[0])
        dx = 1 if target[0] > pos[0] else (-1 if target[0] < pos[0] else 0)
        dy = 1 if target[1] > pos[1] else (-1 if target[1] < pos[1] else 0)
        x, y = pos
        while (x, y) != target:
            x += dx
            y += dy
            if not (0 <= x < h and 0 <= y < w):
                return False
            if (x, y) == target:
                return True
            if blocked(int(map2d[x][y])):
                return False
        return True

    def _nearest_hostile_dist(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> int:
        """距最近主动怪（melee/ranged）的曼哈顿距离；无怪返回 999。"""
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        mobs = map_payload.get("mob_positions", {})
        best = 999
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                d = abs(int(p[0]) - pos[0]) + abs(int(p[1]) - pos[1])
                best = min(best, d)
        return best

    def _bow_combat(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        chase: str = "all",
        max_range: int = 14,
    ) -> Optional[int]:
        """弓战斗原语：贴脸点射（必中）→ 同行列直线射（<=max_range，提前射杀免接战）
        → 按 chase 模式走近目标点射。

        返回动作；无弓/无目标/无法射时返回 None（调用方回退近战/等待）。
        chase 模式：
        - "all"：追近战与远程（清怪子目标用）；
        - "ranged"：只追**远程**怪（生存维护用）——远程怪会隔空射击，干等只是
          白挨打；近战怪会自己走过来，不必迎上去（迎上去=多挨首击）。
        - "none"：只点射不移动。
        max_range：直线射击的最远距离。清怪层用 14（出生环 10-13，远距射杀=0 受击），
        已清层用 CLEARED_FLOOR_ENGAGE_DIST——那里杀怪买不到任何东西，无差别点射会
        把整局变成"地表箭工厂"（实测 L0 击杀 29、箭耗尽、一次没下楼）。
        """
        if isinstance(chase, bool):  # 兼容旧调用
            chase = "all" if chase else "none"
        if not self._has_bow(summary):
            return None
        inventory = summary.get("inventory") or {}
        if int(inventory.get("arrows", 0)) < 1:
            return self._craft_action("native.craft_arrow", map_payload, summary)
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        direction = int(summary.get("player_direction", 0))
        map2d = map_payload["map"]
        mobs = map_payload.get("mob_positions", {})

        hostiles: List[Tuple[int, int]] = []
        ranged_set: Set[Tuple[int, int]] = set()
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and not masks[i]:
                    continue
                c = _norm_pos(p)
                hostiles.append(c)
                if key == "ranged":
                    ranged_set.add(c)
        if not hostiles:
            return None
        # 远程优先（会先打你），其次距离近
        hostiles.sort(key=lambda c: (
            0 if c in ranged_set else 1,
            abs(c[0] - pos[0]) + abs(c[1] - pos[1]),
        ))
        for c in hostiles:
            dx, dy = c[0] - pos[0], c[1] - pos[1]
            dist = abs(dx) + abs(dy)
            # 贴脸：转向 + 点射（投射物判定当前格+下一格，必中）
            if dist == 1:
                action = DELTA_TO_ACTION.get((dx, dy))
                if action is None:
                    continue
                if direction == action:
                    return SHOOT_ARROW
                return action  # 转向
            # 同行/列且直线无阻挡且距离 <= 14：直线射（怪 75% 概率迎面走来）。
            # 怪在 10-13 出生，提前在远距射杀可免接战（0 受击）——这是弓清怪
            # 的核心价值，也是 L0 制备墙的突破口。
            # 注意：DELTA_TO_ACTION 只有 4 个**单位**向量键，(dx, dy) 在 dist>1 时
            # 恒不命中——旧实现写 DELTA_TO_ACTION.get((dx, dy)) 使整段直线射击成为
            # 死代码：弓只在贴脸时才射，玩家带着十几支箭被 4-5 格外的远程怪点死
            # （实测 L1 死因：每次掉血都发生在无近身怪时）。这里取**符号向量**。
            if (dx == 0 or dy == 0) and dist <= max_range and self._line_clear(map2d, pos, c):
                step = (0 if dx == 0 else (1 if dx > 0 else -1),
                        0 if dy == 0 else (1 if dy > 0 else -1))
                action = DELTA_TO_ACTION.get(step)
                if action is None:
                    continue
                if direction == action:
                    return SHOOT_ARROW
                return action  # 转向（同行列，转向后仍对齐，下一拍即可射）
        # 无贴脸/直线目标：按 chase 模式走近（近战 5 格内、远程 8 格内）。
        if chase != "none" and hostiles:
            targets = [c for c in hostiles if chase == "all" or c in ranged_set]
            if targets:
                c = min(
                    targets,
                    key=lambda x: abs(x[0] - pos[0]) + abs(x[1] - pos[1]),
                )
                dist = abs(c[0] - pos[0]) + abs(c[1] - pos[1])
                ch = 8 if c in ranged_set else 5
                if dist <= ch:
                    walk = self._walk_to_mob(map2d, pos, c)
                    if walk is not None:
                        return walk
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
        # 按层缓存战术：bow（远程点射）vs kite（命中后拉开）vs stand（贴脸速杀）
        floor = int(summary.get("floor", 0))
        if self._tactic_floor != floor:
            elem = self._has_elemental_capability(summary, floor)
            gear = Gear(
                sword=int((summary.get("inventory") or {}).get("sword", 0)),
                armour=sum(int(x) for x in (summary.get("inventory") or {}).get("armour", [0])),
                strength=int(summary.get("strength", 1)),
                dexterity=int(summary.get("dexterity", 1)),
                intelligence=int(summary.get("intelligence", 1)),
                has_elemental=elem,
                bow=int((summary.get("inventory") or {}).get("bow", 0)),
                bow_enchant=int(summary.get("bow_enchantment", 0)),
            )
            if self._should_use_bow(floor, summary):
                self._tactic = "bow"
            else:
                self._tactic = recommend_tactic(floor, gear)
            self._tactic_floor = floor

        # 坑位伏击（清怪层 + 已在三面墙的坑位 + 有弓）：**守住不追**。
        # 怪 75% 概率朝玩家走、且只在 10-14 格环上刷新（spawn_mobs），所以"等"
        # 一定等得到；而走出坑位追击等于把"只有一个开口"的优势还回去——每次接战
        # 固定挨一次首击（怪刷新即冷却<=0），这正是 L1 清怪墙的成本来源。
        # 坑位里只需盯住那一个开口：怪进开口=贴脸必中点射，或沿走廊直线提前射杀。
        ambush = (int(map_payload.get("monsters_killed", 0)) < 8
                  and self._has_bow(summary)
                  and int((summary.get("inventory") or {}).get("arrows", 0)) >= 1
                  and self._in_shelter(map_payload, summary))

        # 弓主战：先尝试远程点射；无箭/无目标时回退近战/等待
        if self._tactic == "bow":
            bow_action = self._bow_combat(
                map_payload, summary, chase="none" if ambush else "all"
            )
            if bow_action is not None:
                return bow_action
            if ambush:
                return self._wait_action(summary, map_payload)  # 守住开口，别走出去

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
            # 无目标：等刷怪。清怪中先退进掩体再等（见 _cover_or_wait）
            return self._cover_or_wait(map_payload, summary, clearing)

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
            # 射程内（贴脸 1 格，或持剑时正前方 2 格且中间通透）→ 打，不要走近。
            # 走到相邻格必然先吃怪的首击（怪刷新即冷却<=0），而两格外它打不到我们，
            # 这正是 2 格射程的全部价值所在。
            strike = self._melee_strike_action(map_payload, summary, c)
            if strike is not None:
                if strike == DO:
                    # 风筝（recommend_tactic=kite 时）：跟踪怪冷却窗口。被命中后
                    # 冷却重置 5，计时递减；当 timer==1（怪冷却将归零）且怪紧邻
                    # → 拉开 2 步，让怪在冷却归零时追不上（攻击判定在回合初的
                    # 相邻状态）。timer==0 表示"无近期命中"（新鲜怪），照常攻击
                    # 并承担必中的首击——首击无法避免，风筝只规避后续命中。
                    # 两格外的怪本回合打不到我们，不必风筝。
                    if (self._tactic == "kite" and c not in ranged_set
                            and dist <= 1 and self._mob_attack_timer == 1):
                        self._kite_retreats = 2
                        retreat = self._retreat_from_mobs(map_payload, summary)
                        if retreat is not None:
                            return retreat
                return strike
            if c in ranged_set or dist <= chase_limit:
                walk = self._walk_to_mob(map2d, pos, c)
                if walk is not None:
                    return walk
            if dist > chase_limit and c not in ranged_set:
                break  # 超出近战追击距离：等近战怪接近
        return self._cover_or_wait(map_payload, summary, clearing)

    def _cover_or_wait(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], clearing: bool
    ) -> int:
        """清怪时的"等"：先退进掩体再等。

        怪 75% 概率朝玩家走、且只在 10-14 格环上刷新（spawn_mobs），所以"等怪
        上门"本来就是有效战术；差别在于**在哪里等**。旷野里 4 个方向都可能被
        贴脸、远程怪可自由射击；三面墙的坑位把同时接战数降到 1 并挡掉三面的箭。
        """
        if clearing:
            cover = self._take_cover(map_payload, summary)
            if cover is not None:
                return cover
        return self._wait_action(summary, map_payload)

    def _wait_action(
        self, summary: Dict[str, Any],
        map_payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """等待刷怪/无事可做：安全时睡（省资源+回蓝回血），否则原地待命。

        两道门都必要：
        - 血 <8 时 SLEEP 会被怪 3.5x 打醒致死（L0 僵尸 3.5x=7，8 血才扛得住）；
        - **满血也不能在旷野睡**：实测 seed 2011 在夜里"血足→睡等刷怪"，僵尸走到
          身边一击 10→3（2 伤 ×3.5），醒来后又被补刀，253 步暴毙。因此要么距怪
          >=14（怪会消失），要么身处三面墙的坑位（只有一个开口）才睡。
        "原地待命"默认是 DO，但传入 map_payload 时会避免拆掉自己的掩体
        （见 _idle_action）。
        """
        if map_payload is None:  # 老调用点（无地图信息）：保持原语义
            return DO if float(summary.get("health", 9.0)) < 8 else SLEEP
        # 睡眠前先把安全点实体化：优先天然凹槽，其次挖坑/补石，避免“找到了
        # 睡眠点但仍站在四面开阔地”导致睡眠伤害放大。掩体建好后本次返回的
        # 动作会在下一拍重新规划，最终才发 SLEEP。
        needs_sleep_shelter = (
            float(summary.get("health", 9.0)) < 8
            or float(summary.get("energy", 9.0)) < 3
        )
        if needs_sleep_shelter and not self._in_shelter(map_payload, summary):
            cover = self._take_cover(map_payload, summary, min_walls=MIN_SHELTER_WALLS)
            if cover is not None:
                return cover
        if float(summary.get("health", 9.0)) < 8:
            if self._in_shelter(map_payload, summary) and self._sleep_is_safe(summary):
                return SLEEP
            return self._idle_action(map_payload, summary)
        sleep_safe = (self._nearest_hostile_dist(map_payload, summary) >= 14
                      or self._in_shelter(map_payload, summary))
        if not sleep_safe:
            return self._idle_action(map_payload, summary)
        return SLEEP

    def _idle_action(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> int:
        """原地待命的动作：默认 DO，但**不能拆掉自己的掩体**。

        DO 作用于朝向格。在坑位里朝向格通常正是墙（走进坑位时面朝里），DO 会把
        它挖掉——掩体当场作废，玩家从"三面墙"变回"站在开阔地"，而且这发生在
        每一次"低血原地回血"的等待里。此时改用 NOOP。
        朝向水/泉时保留 DO（那是喝水，正是想要的行为）。
        """
        map2d = map_payload["map"]
        pos = _norm_pos(summary.get("player_position", [0, 0]))
        delta = ACTION_DELTA.get(int(summary.get("player_direction", 0)))
        if delta is None:
            return DO
        facing = (pos[0] + delta[0], pos[1] + delta[1])
        h, w = len(map2d), len(map2d[0])
        if not (0 <= facing[0] < h and 0 <= facing[1] < w):
            return DO
        tile = int(map2d[facing[0]][facing[1]])
        if blocked(tile) and tile not in DRINK_BLOCKS and self._cover_walls(
            map_payload, summary
        ) >= MIN_SHELTER_WALLS - 1:
            return NOOP
        return DO

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
        """施法：需 mana>=2，否则睡觉回蓝（血低/旷野不睡）。"""
        if float(summary.get("mana", 0)) < 2:
            return self._wait_action(summary, map_payload)
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
        make_action: Optional[int] = None,
    ) -> Optional[int]:
        """到指定附魔台：先到位 → 满蓝 → 站在台旁面向它 → 按 ENCHANT_*。

        make_action 覆盖默认动作（如 ENCHANT_BOW）；默认按 ENCHANT_ACTIONS[tid]。
        """
        if make_action is None:
            make_action = ENCHANT_ACTIONS[tid][0]
        floor = int(summary.get("floor", 0))
        if floor != target_floor:
            return self._move_to_floor(map_payload, summary, target_floor)
        # 附魔需 9 mana（满值），睡觉回蓝最快（血低/旷野不睡）
        if float(summary.get("mana", 0)) < 9:
            return self._wait_action(summary, map_payload)
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

        # 2) 弓附魔（首选：弓元素半伤 + 远程点射，需弓 + 对应宝石）
        if self._has_bow(summary):
            if need_ice:
                if int(inventory.get("sapphire", 0)) < 1:
                    a = self._get_gem(map_payload, summary, "native.collect_sapphire")
                    if a is not None:
                        return a
                    return None
                return self._enchant_at(
                    map_payload, summary, 3, ENCHANT_TABLE_ICE,
                    "native.enchant_sword", make_action=ENCHANT_BOW,
                )
            else:
                if int(inventory.get("ruby", 0)) < 1:
                    a = self._get_gem(map_payload, summary, "native.collect_ruby")
                    if a is not None:
                        return a
                    return None
                return self._enchant_at(
                    map_payload, summary, 4, ENCHANT_TABLE_FIRE,
                    "native.enchant_sword", make_action=ENCHANT_BOW,
                )

        # 3) 剑附魔（需剑 + 对应宝石）
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

    # -- 就绪门 / 升级 / 恢复战术 -------------------------------------------

    def _level_up_choice(self, summary: Dict[str, Any]) -> Optional[int]:
        """属性升级策略（量化评估）。

        默认力量优先（每点 +25% 物伤、+1 血）。**敏捷量化评估**：用战斗模型估
        算清当前层所需能量，若超出敏捷能量预算（`energy_is_bottleneck`，即
        estimated_steps > 7+2dex 上限 × 衰减修正的 80%），则敏捷（+2 能量上限、
        -12.5% 疲劳衰减）的收益 > 力量 → 点敏捷。深层任务（max_floor>=2）才评估。
        """
        xp = int(summary.get("xp", 0))
        if xp < 1:
            return None
        strength = int(summary.get("strength", 1))
        dexterity = int(summary.get("dexterity", 1))
        floor = int(summary.get("floor", 0))
        if floor > 0 and self._max_floor >= 2:
            inventory = summary.get("inventory") or {}
            gear = Gear(
                sword=int(inventory.get("sword", 0)),
                armour=sum(int(x) for x in inventory.get("armour", [0])),
                strength=strength,
                dexterity=dexterity,
                intelligence=int(summary.get("intelligence", 1)),
                has_elemental=has_elemental(summary, floor),
                bow=int(inventory.get("bow", 0)),
                bow_enchant=int(summary.get("bow_enchantment", 0)),
            )
            if energy_is_bottleneck(floor, gear) and dexterity < 5:
                return LEVEL_UP_DEXTERITY
        if strength < 5:
            return LEVEL_UP_STRENGTH
        if self._max_floor >= 5 and dexterity < 5:
            # 深层长程任务：力量满后补敏捷保能量
            return LEVEL_UP_DEXTERITY
        return None

    def _safe_sleep_spot_walk(
        self,
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        max_steps: int = 6,
    ) -> Optional[int]:
        """睡前去更安全的点：找可达格中"距最近主动怪最远"的格。

        规则：
        - 当前已 >=14 格（怪在 >14 格会消失）→ 返回 None（就地睡）；
        - 存在明显更远的可达点（min-dist >= 14）且路径 <= max_steps → 走一步；
        - 否则返回 None（就地睡 / 由锚点恢复兜底）。
        限步数避免远途跋涉暴露（表层制备阶段尤其致命）。
        """
        mobs = map_payload.get("mob_positions", {})
        hostile: List[Tuple[int, int]] = []
        for key in ("melee", "ranged"):
            entry = mobs.get(key, {})
            masks = entry.get("masks", [])
            for i, p in enumerate(entry.get("positions", [])):
                if i < len(masks) and masks[i]:
                    hostile.append(_norm_pos(p))
        if not hostile:
            return None  # 无主动怪 → 就地睡
        from collections import deque

        map2d = map_payload["map"]
        h, w = len(map2d), len(map2d[0])
        pos = _norm_pos(summary.get("player_position", [0, 0]))

        # 多源 BFS：每格到最近主动怪的距离
        dist_mob: Dict[Tuple[int, int], int] = {}
        q = deque()
        for c in hostile:
            if 0 <= c[0] < h and 0 <= c[1] < w:
                dist_mob[c] = 0
                q.append(c)
        while q:
            cell = q.popleft()
            d = dist_mob[cell]
            for delta in ACTION_DELTA.values():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in dist_mob:
                    continue
                dist_mob[nxt] = d + 1
                q.append(nxt)

        current = dist_mob.get(pos, 99)
        if current >= 14:
            return None  # 已足够远（怪会消失）

        # 玩家限步 BFS 的可达格中，选 min-dist 最大的（要求 >=14 才算安全点）
        visited = {pos}
        frontier = {pos}
        best: Optional[Tuple[int, int]] = None
        best_score = current
        for _ in range(max_steps):
            nxt_frontier = set()
            for cell in frontier:
                for d in ACTION_DELTA.values():
                    nxt = (cell[0] + d[0], cell[1] + d[1])
                    if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                        continue
                    if nxt in visited or blocked(int(map2d[nxt[0]][nxt[1]])):
                        continue
                    visited.add(nxt)
                    nxt_frontier.add(nxt)
                    score = dist_mob.get(nxt, 0)
                    if score >= 14 and score > best_score:
                        best_score = score
                        best = nxt
            frontier = nxt_frontier
        if best is None or best == pos:
            return None
        return self._walk_to(map2d, pos, best)

    def _iron_craft_feasible(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any]
    ) -> bool:
        """本层能否就地做铁装：铁与煤各 >=1（背包里有，或本层地图上有）。

        旧实现只看铁（`has_iron_here`），但 MAKE_IRON_* 同时要 1 煤 + 熔炉——
        只有铁没有煤时会一路走到合成键前才发现缺料。
        """
        inventory = summary.get("inventory") or {}
        map2d = map_payload["map"]
        iron_ok = (int(inventory.get("iron", 0)) >= 1
                   or self._count_blocks(map2d, [IRON]) > 0)
        coal_ok = (int(inventory.get("coal", 0)) >= 1
                   or self._count_blocks(map2d, [COAL]) > 0)
        return iron_ok and coal_ok

    def _craft_armour_until(
        self, map_payload: Dict[str, Any], summary: Dict[str, Any], pieces: int
    ) -> Optional[int]:
        """补足铁甲到 pieces 件：缺料则仅在本层采（不跨层），齐料则就地做一件。

        本层采不到铁/煤 → 返回 None（软门槛，调用方决定是否放弃）。
        """
        inventory = summary.get("inventory") or {}
        armour = sum(int(x) for x in inventory.get("armour", [0]))
        if armour >= pieces:
            return None
        if int(inventory.get("iron", 0)) < 3:
            return self._seek_and_do(map_payload, summary, [IRON])
        if int(inventory.get("coal", 0)) < 3:
            return self._seek_and_do(map_payload, summary, [COAL])
        return self._craft_action("native.craft_iron_armour", map_payload, summary)

    def _resolve_gate(
        self,
        missing: Sequence[Tuple[str, Any]],
        map_payload: Dict[str, Any],
        summary: Dict[str, Any],
        target_floor: int,
    ) -> Optional[int]:
        """按缺失门槛返回一个补齐动作；无法补齐返回 None（调用方决定中止/继续）。"""
        for kind, val in missing:
            if kind == "sword":
                craft_tid = {
                    2: "native.craft_stone_sword",
                    3: "native.craft_iron_sword",
                    4: "native.craft_diamond_sword",
                }.get(int(val))
                if craft_tid is not None:
                    a = self._craft_action(craft_tid, map_payload, summary)
                    if a is not None:
                        return a
            elif kind == "pickaxe":
                # 逐级升镐：只做"下一级"，因为做上一级镐才能采到下一级的料。
                have = int((summary.get("inventory") or {}).get("pickaxe", 0))
                craft_tid = {
                    1: "native.craft_wood_pickaxe",
                    2: "native.craft_stone_pickaxe",
                    3: "native.craft_iron_pickaxe",
                    4: "native.craft_diamond_pickaxe",
                }.get(min(have + 1, int(val)))
                if craft_tid is not None:
                    a = self._craft_action(craft_tid, map_payload, summary)
                    if a is not None:
                        return a
            elif kind == "armour":
                a = self._craft_armour_until(map_payload, summary, int(val))
                if a is not None:
                    return a
            elif kind == "elemental":
                a = self._acquire_elemental_capability(map_payload, summary, target_floor)
                if a is not None:
                    return a
                return None  # 元素能力无法获得 → 硬中止
            elif kind == "survival":
                return None  # combat INFEASIBLE → 硬中止（由调用方处理）
            elif kind == "ladder":
                return None  # seed 的梯子链断了（world_facts）→ 硬中止
            # strength：力量优先升级已自动处理 → 忽略
        return None


# 任务注册：供 demo 脚本按任务选择 executor
EXECUTOR_FACTORIES: Dict[str, Any] = {}


def make_executor(task_id: str) -> SkillChainExecutor:
    return SkillChainExecutor(task_id)
