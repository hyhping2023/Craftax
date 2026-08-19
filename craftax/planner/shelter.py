"""庇护所/掩体几何：找坑位、挖洞、封口（纯函数，可单测）。

机制依据（均已从 `craftax/craftax/game_logic.py` 核实）：

- **近战怪只在曼哈顿距离 ==1 时攻击**（`_move_melee_mob` 的 `is_attacking_player`），
  且移动受 collision 阻挡 → 三面是墙的坑位把"同时可接战的怪"从 4 只降到 1 只。
- **怪的投射物撞到 solid 方块即消失**（`_move_mob_projectile` 的 `in_wall`）→ 墙挡箭。
  远程怪在距离 4-5 时开火（`_move_ranged_mob`），所以"暴露射线"只需看 5 格内有没有墙。
- **怪只在距玩家 >9 且 < despawn(14) 的环上刷新**（`spawn_mobs`），且 75% 概率朝玩家走
  → 站在坑位里等怪来，比在旷野迎上去更省血。
- **放置方块落在朝向格**（`place_block`），而按方向键时"能走就走、走不动才只转向"
  （`move_player`）→ **只能封住行进方向正前方的格**。推论：4 面自封在 Craftax 里
  做不到（最后一面开口的方向无法在原地面向），可行的最好结果是"开口 1 个"的坑位：
  天然凹陷、或用镐向石堆里挖一格。本模块因此以"找/挖坑位"为主、"放石封口"为辅。

坐标约定与 `path_planner` 一致：`cell = (x, y)`，`map2d[x][y]` 为方块类型。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from craftax.planner.path_planner import ACTION_DELTA, PLACEABLE_TILES, blocked

# 远程怪的开火距离上限（game_logic `_move_ranged_mob`：距离 4-5 时开火）。
# 判断某方向"暴露"时只看这么远——更远的怪不会射，近了它反而后退。
RANGED_REACH = 5

# 可作为掩体坑位的最低墙数（3 面墙 = 只剩一个开口）
MIN_SHELTER_WALLS = 3

# 石头方块（BlockType.STONE）：挖洞造坑位的唯一目标，需木镐（PICKAXE_REQUIRED）
STONE = 4


def _dims(map2d: Any) -> Tuple[int, int]:
    return len(map2d), len(map2d[0])


def in_bounds(map2d: Any, cell: Tuple[int, int]) -> bool:
    h, w = _dims(map2d)
    return 0 <= cell[0] < h and 0 <= cell[1] < w


def tile_at(map2d: Any, cell: Tuple[int, int]) -> int:
    return int(map2d[cell[0]][cell[1]])


def is_wall(map2d: Any, cell: Tuple[int, int]) -> bool:
    """该格能否挡住玩家/怪/箭。越界视为墙（地图边界同样不可穿过）。

    注意：`blocked()` 把水/岩浆也算不可走。水**不挡箭**（`_move_mob_projectile`
    对 WATER 有例外），但挡近战怪的移动，因此这里仍算墙；`exposed_dirs` 另行
    按"挡箭"的口径处理。
    """
    if not in_bounds(map2d, cell):
        return True
    return blocked(tile_at(map2d, cell))


def _blocks_arrow(map2d: Any, cell: Tuple[int, int]) -> bool:
    """该格能否挡住箭：solid 即可，水例外（箭可越水飞行）。"""
    if not in_bounds(map2d, cell):
        return True
    tile = tile_at(map2d, cell)
    if tile == 3:  # WATER：箭可飞过
        return False
    return blocked(tile)


def wall_count(map2d: Any, cell: Tuple[int, int]) -> int:
    """4 邻域中墙的数量（0-4）。3 = 只剩一个开口的坑位。"""
    return sum(
        1
        for d in ACTION_DELTA.values()
        if is_wall(map2d, (cell[0] + d[0], cell[1] + d[1]))
    )


def exposed_dirs(
    map2d: Any, cell: Tuple[int, int], reach: int = RANGED_REACH
) -> List[Tuple[int, int]]:
    """返回"箭能射进来"的方向：该方向 reach 格内没有挡箭的方块。"""
    out: List[Tuple[int, int]] = []
    for d in ACTION_DELTA.values():
        for step in range(1, reach + 1):
            probe = (cell[0] + d[0] * step, cell[1] + d[1] * step)
            if _blocks_arrow(map2d, probe):
                break
        else:
            out.append(d)
    return out


def cover_rating(map2d: Any, cell: Tuple[int, int]) -> Tuple[int, int]:
    """掩体评分 `(墙数, -暴露射线数)`，可直接用于排序（越大越好）。"""
    return (wall_count(map2d, cell), -len(exposed_dirs(map2d, cell)))


def _walkable(
    map2d: Any, cell: Tuple[int, int], extra_blocked: Optional[Set[Tuple[int, int]]]
) -> bool:
    if not in_bounds(map2d, cell) or blocked(tile_at(map2d, cell)):
        return False
    return not (extra_blocked and cell in extra_blocked)


def reachable_cells(
    map2d: Any,
    start: Tuple[int, int],
    max_steps: int,
    extra_blocked: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], int]:
    """限步 BFS：可达格 -> 步数（含 start=0）。

    限步是为了不为找掩体远途跋涉——移动本身就是暴露（与 `_safe_sleep_spot_walk`
    的既有约定一致）。
    """
    extra = set(extra_blocked) if extra_blocked else set()
    dist: Dict[Tuple[int, int], int] = {start: 0}
    queue: deque = deque([start])
    while queue:
        cell = queue.popleft()
        if dist[cell] >= max_steps:
            continue
        for d in ACTION_DELTA.values():
            nxt = (cell[0] + d[0], cell[1] + d[1])
            if nxt in dist or not _walkable(map2d, nxt, extra):
                continue
            dist[nxt] = dist[cell] + 1
            queue.append(nxt)
    return dist


def _min_mob_dist(cell: Tuple[int, int], hostiles: Sequence[Tuple[int, int]]) -> int:
    if not hostiles:
        return 999
    return min(abs(cell[0] - c[0]) + abs(cell[1] - c[1]) for c in hostiles)


def find_cover_tile(
    map2d: Any,
    pos: Tuple[int, int],
    hostiles: Sequence[Tuple[int, int]] = (),
    max_steps: int = 8,
    min_walls: int = MIN_SHELTER_WALLS,
    extra_blocked: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[int, int]]:
    """限步内找最好的掩体格；没有达标格返回 None。

    排序：墙数 > 暴露射线少 > 离怪远 > 步数少。要求严格优于当前所站格，
    否则返回 None（避免"为了持平的掩体反复走动"）。
    """
    dist = reachable_cells(map2d, pos, max_steps, extra_blocked)
    here = cover_rating(map2d, pos)
    best: Optional[Tuple[int, int]] = None
    best_key: Optional[Tuple[int, int, int, int]] = None
    for cell, steps in dist.items():
        walls, neg_exposed = cover_rating(map2d, cell)
        if walls < min_walls:
            continue
        key = (walls, neg_exposed, _min_mob_dist(cell, hostiles), -steps)
        if cell == pos:
            continue
        if key[:2] <= here:  # 不比当前位置更好 → 不值得走
            continue
        if best_key is None or key > best_key:
            best_key = key
            best = cell
    return best


def find_dig_pocket(
    map2d: Any,
    pos: Tuple[int, int],
    max_steps: int = 8,
    extra_blocked: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """找"挖一块石头就能得到 3 面墙坑位"的位置。

    条件：石头格 `s` 的 4 邻域里恰有 1 个可走格（玩家的站位），其余 3 面是墙。
    挖掉 `s` 后它变成 PATH，墙数 3 → 走进去即成掩体。返回 `(站位, 石头格)`。

    这是旷野里唯一能主动**造出**坑位的手段：放石只能封行进方向正前方，
    因此"封到只剩一个开口"通常不可行，而向石堆里挖一格必定得到 3 面墙。
    """
    dist = reachable_cells(map2d, pos, max_steps, extra_blocked)
    best: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    best_key: Optional[Tuple[int, int]] = None
    for stand, steps in dist.items():
        for d in ACTION_DELTA.values():
            s = (stand[0] + d[0], stand[1] + d[1])
            if not in_bounds(map2d, s) or tile_at(map2d, s) != STONE:
                continue
            open_neighbours = [
                n
                for n in (
                    (s[0] + e[0], s[1] + e[1]) for e in ACTION_DELTA.values()
                )
                if not is_wall(map2d, n)
            ]
            if len(open_neighbours) != 1 or open_neighbours[0] != stand:
                continue
            key = (-steps, wall_count(map2d, stand))
            if best_key is None or key > best_key:
                best_key = key
                best = (stand, s)
    return best


def seal_target(
    map2d: Any,
    pos: Tuple[int, int],
    direction_delta: Tuple[int, int],
    mob_cells: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[int, int]]:
    """当前朝向放石能否封出掩体：可行则返回被封的格，否则 None。

    要求朝向格可放置（`PLACEABLE_TILES`）、当前不是墙、且没有怪站在上面
    （放到怪身上不产生阻挡）。放石只能作用于朝向格——这正是"4 面自封不可能"
    的根源，此函数只用于把 2 面墙的角落补成 3 面。
    """
    target = (pos[0] + direction_delta[0], pos[1] + direction_delta[1])
    if not in_bounds(map2d, target) or is_wall(map2d, target):
        return None
    if tile_at(map2d, target) not in PLACEABLE_TILES:
        return None
    if mob_cells and target in set(mob_cells):
        return None
    return target
