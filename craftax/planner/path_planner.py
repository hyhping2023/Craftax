"""确定性路径规划器（收集类任务）。

对应 embodied_environment_plan.md §6.1.4 的 rolling planner 首期实现：
- 输入完整地图（服务端 /v1/sessions/{sid}/map 暴露的当前楼层 48×48 网格）、
  玩家位置/朝向、背包镐等级、任务 id；
- 用 BFS 寻路到最近的目标方块（如 TREE），生成确定性动作序列：
  走到目标旁 → 转向面向它 → DO 采集；
- 无足够镐或找不到目标时返回 None，由调用方回退到随机策略。

本模块只读地图做规划，不改游戏逻辑、不注入特权修改；
执行仍走原生动作（LEFT/RIGHT/UP/DOWN/DO）。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 动作 id（与 craftax.craftax.constants.Action 一致）
NOOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
DO = 5
PLACE_TABLE = 8
MAKE_WOOD_PICKAXE = 11

# 可放置方块（place_block 允许放置于其上的地面块，CAN_PLACE_ITEM_BLOCKS）
PLACEABLE_TILES = {2, 7, 13, 25, 26}  # GRASS, PATH, SAND, FIRE_GRASS, ICE_GRASS

# 工作台方块值（BlockType.CRAFTING_TABLE）
CRAFTING_TABLE_TILE = 11

# 收集类任务 -> 目标方块类型（BlockType 枚举值）
COLLECT_TARGETS: Dict[str, List[int]] = {
    "native.collect_wood": [5, 28],       # TREE, FIRE_TREE
    "native.collect_stone": [4],          # STONE
    "native.collect_coal": [8],           # COAL
    "native.collect_iron": [9],           # IRON
    "native.collect_diamond": [10],       # DIAMOND
    "native.collect_sapphire": [21],      # SAPPHIRE
    "native.collect_ruby": [22],          # RUBY
}

# 采集所需最低镐等级（0=徒手，1=木镐，2=石镐，3=铁镐，4=钻石镐）
PICKAXE_REQUIRED: Dict[str, int] = {
    "native.collect_wood": 0,
    "native.collect_stone": 1,
    "native.collect_coal": 1,
    "native.collect_iron": 2,
    "native.collect_diamond": 3,
    "native.collect_sapphire": 4,
    "native.collect_ruby": 4,
}

# 动作 id -> 位移 delta（与 game_logic.DIRECTIONS 一致）
ACTION_DELTA: Dict[int, Tuple[int, int]] = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}
DELTA_TO_ACTION: Dict[Tuple[int, int], int] = {
    delta: action for action, delta in ACTION_DELTA.items()
}

_SOLID_BLOCKS: Optional[Sequence[int]] = None
WATER = 3
LAVA = 14


def _solid_blocks() -> Sequence[int]:
    """惰性加载 SOLID_BLOCKS（避免 import 时加载 JAX）。"""
    global _SOLID_BLOCKS
    if _SOLID_BLOCKS is None:
        from craftax.craftax.constants import SOLID_BLOCKS

        _SOLID_BLOCKS = list(SOLID_BLOCKS)
    return _SOLID_BLOCKS


def is_collect_task(task_id: str) -> bool:
    return task_id in COLLECT_TARGETS


def blocked(tile: int) -> bool:
    """玩家能否走上该方块。solid 或水/岩浆均不可走。"""
    return tile in _solid_blocks() or tile in (WATER, LAVA)


def find_nearest_target(
    map2d: Any,
    start: Tuple[int, int],
    target_types: Sequence[int],
    extra_blocked: Optional[Sequence[Tuple[int, int]]] = None,
    excluded_targets: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """BFS 找最近目标方块。

    返回 (目标方块坐标, 从 start 出发的第一步 delta)；
    目标方块本身是 solid，玩家只需走到其相邻格并面向它。
    找不到返回 None。

    map2d: 当前楼层 48×48 int 网格（`map[y][x]` 或 numpy array，下标 [0]=x/[1]=y
    与 player_position 一致）。start: (x, y)。
    extra_blocked: 额外不可通行的格子（如被怪占用的格；怪会挡路）。
    """
    target_set = set(target_types)
    blocked_extra = set(extra_blocked) if extra_blocked else set()
    excluded = set(excluded_targets) if excluded_targets else set()
    h, w = map2d.shape[0], map2d.shape[1]

    def tile_at(pos: Tuple[int, int]) -> int:
        return int(map2d[pos[0], pos[1]])

    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    # 玩家起点的周围直接有目标方块（站在目标旁）
    for delta in ACTION_DELTA.values():
        adj = (start[0] + delta[0], start[1] + delta[1])
        if (0 <= adj[0] < h and 0 <= adj[1] < w
                and tile_at(adj) in target_set and adj not in excluded):
            return adj, delta

    visited = {start}
    queue: deque = deque([(start, None)])  # (cell, 从 start 到此的第一步 delta)
    while queue:
        cell, first_delta = queue.popleft()
        for delta in ACTION_DELTA.values():
            nxt = (cell[0] + delta[0], cell[1] + delta[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if nxt in visited:
                continue
            if tile_at(nxt) in target_set and nxt not in excluded:
                return nxt, (first_delta or delta)
            if blocked(tile_at(nxt)):
                continue
            if nxt in blocked_extra:
                continue  # 怪占用的格不可通行（绕路）
            visited.add(nxt)
            queue.append((nxt, first_delta or delta))
    return None


def next_action(
    map2d: Any,
    player_pos: Tuple[int, int],
    player_dir: int,
    inventory_pickaxe: int,
    task_id: str,
) -> Optional[int]:
    """返回下一步动作 id；无法规划（缺镐/无目标）返回 None。

    map2d: 当前楼层网格；player_pos: (x, y)；player_dir: 当前朝向动作 id
    （1=LEFT,2=RIGHT,3=UP,4=DOWN）；inventory_pickaxe: 背包镐等级（0..4）。
    """
    if task_id not in COLLECT_TARGETS:
        return None
    if PICKAXE_REQUIRED[task_id] > inventory_pickaxe:
        return None
    target_types = COLLECT_TARGETS[task_id]

    h, w = map2d.shape[0], map2d.shape[1]

    def tile_at(pos: Tuple[int, int]) -> int:
        return int(map2d[pos[0], pos[1]])

    # 已经站在某个目标方块旁：先确认是否正面向它
    for delta, action in DELTA_TO_ACTION.items():
        adj = (player_pos[0] + delta[0], player_pos[1] + delta[1])
        if 0 <= adj[0] < h and 0 <= adj[1] < w and tile_at(adj) in target_types:
            if player_dir == action:
                return DO
            # 朝向不对：按方向键转向（目标为 solid，移动被挡、仅更新朝向）
            return action

    result = find_nearest_target(map2d, player_pos, target_types)
    if result is None:
        return None
    target, first_delta = result
    if first_delta in DELTA_TO_ACTION:
        return DELTA_TO_ACTION[first_delta]
    # 已站在目标旁且朝向正确但 DO 未生效（如树没被判定）：重试 DO
    if (target[0] - player_pos[0], target[1] - player_pos[1]) in DELTA_TO_ACTION:
        return DO
    return None


def plan_path(
    map2d: Any,
    start: Tuple[int, int],
    target_types: Sequence[int],
    max_steps: int = 200,
) -> Optional[List[int]]:
    """返回从 start 到目标方块相邻格的确定性移动动作序列（含末尾转向+DO）。

    供调试/测试直接得到完整动作串；demo 录制使用 next_action 逐拍驱动。
    """
    if not find_nearest_target(map2d, start, target_types):
        return None
    actions: List[int] = []
    pos = start
    dir_action = DOWN
    seen = set()
    for _ in range(max_steps):
        a = _next_action_for_targets(map2d, pos, dir_action, target_types)
        if a is None:
            return None
        actions.append(a)
        key = (pos, a)
        if key in seen:
            return None  # 防环
        seen.add(key)
        if a == DO:
            return actions
        if a in ACTION_DELTA:
            delta = ACTION_DELTA[a]
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
            # 仅当目标格可走才移动；否则视为转向
            if not blocked(int(map2d[new_pos[0], new_pos[1]])):
                pos = new_pos
        dir_action = a
    return None


def _next_action_for_targets(
    map2d: Any,
    player_pos: Tuple[int, int],
    player_dir: int,
    target_types: Sequence[int],
) -> Optional[int]:
    """next_action 的非 task 版本（plan_path 复用）。"""
    h, w = map2d.shape[0], map2d.shape[1]

    def tile_at(pos: Tuple[int, int]) -> int:
        return int(map2d[pos[0], pos[1]])

    for delta, action in DELTA_TO_ACTION.items():
        adj = (player_pos[0] + delta[0], player_pos[1] + delta[1])
        if 0 <= adj[0] < h and 0 <= adj[1] < w and tile_at(adj) in target_types:
            if player_dir == action:
                return DO
            return action

    result = find_nearest_target(map2d, player_pos, target_types)
    if result is None:
        return None
    _, first_delta = result
    if first_delta in DELTA_TO_ACTION:
        return DELTA_TO_ACTION[first_delta]
    return None


# ---------------------------------------------------------------------------
# 技能链：挖煤（collect_coal）
# ---------------------------------------------------------------------------
# 阶段：采木 x3 → 放工作台(wood-2) → 做木镐(wood-1, 需靠近工作台) → 挖煤
# 每个阶段一个动作选择函数；由 SkillChainPlanner 按当前 inventory/成就切换。
# 注意：放置工作台时目标格(GRASS 等)是可走方块，不能靠"朝 solid 转向"，
# 必须走到可放置格旁边、面向它再 PLACE_TABLE。


def _find_placeable_target(
    map2d: Any,
    start: Tuple[int, int],
    extra_blocked: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]:
    """找最近的可放置格(GRASS/SAND/PATH 等)。

    返回 (可放置格坐标, 从 start 到其相邻可走格的第一步 delta 或 None(已在其旁))。
    可放置格本身可走，玩家站到它旁边，面向它，按 PLACE_TABLE。
    """
    blocked_extra = set(extra_blocked) if extra_blocked else set()
    h, w = map2d.shape[0], map2d.shape[1]

    def tile_at(pos: Tuple[int, int]) -> int:
        return int(map2d[pos[0], pos[1]])

    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None

    # 已站在可放置格旁：直接返回该格
    for delta in ACTION_DELTA.values():
        adj = (start[0] + delta[0], start[1] + delta[1])
        if 0 <= adj[0] < h and 0 <= adj[1] < w and tile_at(adj) in PLACEABLE_TILES:
            return adj, None

    visited = {start}
    queue: deque = deque([(start, None)])
    while queue:
        cell, first_delta = queue.popleft()
        for delta in ACTION_DELTA.values():
            nxt = (cell[0] + delta[0], cell[1] + delta[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if nxt in visited:
                continue
            if tile_at(nxt) in PLACEABLE_TILES:
                # nxt 是可放置格；玩家需走到 cell（nxt 的相邻可走格）
                return nxt, (first_delta or delta)
            if blocked(tile_at(nxt)):
                continue
            if nxt in blocked_extra:
                continue
            visited.add(nxt)
            queue.append((nxt, first_delta or delta))
    return None


def _find_adjacent_block(
    map2d: Any,
    start: Tuple[int, int],
    target_types: Sequence[int],
    extra_blocked: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]:
    """找最近的目标方块(target_types)，玩家站到其相邻可走格。

    返回 (目标方块坐标, 从 start 到相邻格的第一步 delta 或 None(已在相邻格))。
    与 find_nearest_target 类似，但额外返回"已在旁"语义（None delta）。
    """
    blocked_extra = set(extra_blocked) if extra_blocked else set()
    h, w = map2d.shape[0], map2d.shape[1]
    target_set = set(target_types)

    def tile_at(pos: Tuple[int, int]) -> int:
        return int(map2d[pos[0], pos[1]])

    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    for delta in ACTION_DELTA.values():
        adj = (start[0] + delta[0], start[1] + delta[1])
        if 0 <= adj[0] < h and 0 <= adj[1] < w and tile_at(adj) in target_set:
            return adj, None

    visited = {start}
    queue: deque = deque([(start, None)])
    while queue:
        cell, first_delta = queue.popleft()
        for delta in ACTION_DELTA.values():
            nxt = (cell[0] + delta[0], cell[1] + delta[1])
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if nxt in visited:
                continue
            if tile_at(nxt) in target_set:
                return nxt, (first_delta or delta)
            if blocked(tile_at(nxt)):
                continue
            if nxt in blocked_extra:
                continue
            visited.add(nxt)
            queue.append((nxt, first_delta or delta))
    return None


class SkillChainPlanner:
    """有状态技能链规划器：挖煤（collect_coal）。

    状态由当前阶段推断（不单独维护），每步根据 inventory/成就返回动作：
      - 无木镐 & wood<3       → 采木（next_action collect_wood）
      - 无木镐 & 未放工作台   → 走到可放置格旁并 PLACE_TABLE
      - 无木镐 & 有工作台     → 走到工作台旁并 MAKE_WOOD_PICKAXE
      - 有木镐                → 挖煤（next_action collect_coal）
    """

    def __init__(self, task_id: str = "native.collect_coal") -> None:
        self.task_id = task_id

    def next_action(
        self,
        map2d: Any,
        player_pos: Tuple[int, int],
        player_dir: int,
        inventory: Dict[str, Any],
        achievements: Sequence[str],
    ) -> Optional[int]:
        if self.task_id == "native.collect_coal":
            return self._next_coal(map2d, player_pos, player_dir, inventory, achievements)
        return None

    def _next_coal(
        self,
        map2d: Any,
        player_pos: Tuple[int, int],
        player_dir: int,
        inventory: Dict[str, Any],
        achievements: Sequence[str],
    ) -> Optional[int]:
        pickaxe = int(inventory.get("pickaxe", 0))
        wood = int(inventory.get("wood", 0))

        # 有木镐 → 挖煤
        if pickaxe >= 1:
            return next_action(map2d, player_pos, player_dir, pickaxe, "native.collect_coal")

        table_placed = any(
            a == "PLACE_TABLE" or a == "PlaceTable" for a in achievements
        )
        if not table_placed:
            # 采木到 wood>=3（2 放台 + 1 做镐）
            if wood < 3:
                return next_action(map2d, player_pos, player_dir, 0, "native.collect_wood")
            # 放工作台
            result = _find_placeable_target(map2d, player_pos)
            if result is None:
                return None
            target, first_delta = result
            if first_delta is None:
                # 已站在可放置格旁：面向它并放置
                facing = _action_facing(player_pos, target)
                if facing is None:
                    return None
                if player_dir == facing:
                    return PLACE_TABLE
                return facing
            return DELTA_TO_ACTION[first_delta]

        # 已放工作台：若 wood==0 需再采 1 木做镐
        if wood < 1:
            return next_action(map2d, player_pos, player_dir, 0, "native.collect_wood")
        # 做木镐：走到工作台旁
        result = _find_adjacent_block(map2d, player_pos, [CRAFTING_TABLE_TILE])
        if result is None:
            return None
        _, first_delta = result
        if first_delta is None:
            return MAKE_WOOD_PICKAXE
        return DELTA_TO_ACTION[first_delta]

    def is_done(self, inventory: Dict[str, Any], achievements: Sequence[str]) -> bool:
        """任务是否已完成：collect_coal 达成（背包 coal>0 或成就含 COLLECT_COAL）。"""
        if int(inventory.get("coal", 0)) > 0:
            return True
        return any(
            a == "COLLECT_COAL" or a == "CollectCoal" for a in achievements
        )


def _action_facing(player_pos: Tuple[int, int], target: Tuple[int, int]) -> Optional[int]:
    """返回朝向 target 的动作 id（target 在玩家 4 邻域内时）。"""
    delta = (target[0] - player_pos[0], target[1] - player_pos[1])
    return DELTA_TO_ACTION.get(delta)


SKILL_CHAIN_TASKS = {"native.collect_coal": SkillChainPlanner}


def is_skill_chain_task(task_id: str) -> bool:
    return task_id in SKILL_CHAIN_TASKS
