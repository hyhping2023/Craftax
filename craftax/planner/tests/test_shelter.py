"""shelter.py 单测：掩体几何（找坑位 / 挖坑位 / 封口）。

地图约定与 path_planner 一致：`map2d[x][y]`；2=GRASS（可走）、4=STONE（墙）、
3=WATER（挡人挡怪但不挡箭）。
"""
from __future__ import annotations

import numpy as np
import pytest

from craftax.planner.shelter import (
    MIN_SHELTER_WALLS,
    cover_rating,
    exposed_dirs,
    find_cover_tile,
    find_dig_pocket,
    seal_target,
    wall_count,
)

GRASS = 2
WATER = 3
STONE = 4


def _open_map(size: int = 12) -> np.ndarray:
    return np.full((size, size), GRASS, dtype=np.int32)


def test_wall_count_open_field_and_pocket():
    m = _open_map()
    assert wall_count(m, (5, 5)) == 0
    # 三面石头 → 只剩一个开口
    m[4, 5] = m[6, 5] = m[5, 4] = STONE
    assert wall_count(m, (5, 5)) == MIN_SHELTER_WALLS


def test_wall_count_counts_out_of_bounds_as_wall():
    """地图边界与石头等价：怪同样过不来。"""
    m = _open_map()
    assert wall_count(m, (0, 0)) == 2  # 上、左越界


def test_exposed_dirs_blocked_by_stone_but_not_by_water():
    """墙挡箭（_move_mob_projectile 的 in_wall），水不挡箭（对 WATER 有例外）。"""
    m = _open_map()
    assert len(exposed_dirs(m, (5, 5))) == 4
    m[3, 5] = STONE                       # 上方 2 格处一堵墙
    assert len(exposed_dirs(m, (5, 5))) == 3
    m[7, 5] = WATER                       # 下方是水：箭能飞过 → 仍然暴露
    assert len(exposed_dirs(m, (5, 5))) == 3


def test_exposed_dirs_only_looks_within_ranged_reach():
    """远程怪在 4-5 格开火，更远的墙不改变暴露判定。"""
    m = _open_map(size=20)
    m[5 + 9, 5] = STONE  # 9 格外的墙：够不着，不算掩护
    assert len(exposed_dirs(m, (5, 5))) == 4


def test_cover_rating_prefers_more_walls():
    m = _open_map()
    m[4, 5] = m[6, 5] = m[5, 4] = STONE
    assert cover_rating(m, (5, 5)) > cover_rating(m, (8, 8))


def test_find_cover_tile_finds_natural_pocket():
    m = _open_map()
    # (5,5) 是三面墙的凹陷，玩家在 (5,8) 的旷野上
    m[4, 5] = m[6, 5] = m[5, 4] = STONE
    cell = find_cover_tile(m, (5, 8), hostiles=[(9, 8)], max_steps=8)
    assert cell == (5, 5)


def test_find_cover_tile_none_in_open_field():
    """旷野里没有达标坑位 → None（调用方回退到挖洞/放石/原行为）。"""
    assert find_cover_tile(_open_map(), (5, 5), hostiles=[(9, 5)], max_steps=8) is None


def test_find_cover_tile_none_when_already_in_best_cover():
    """已经站在坑位里就不再走动（避免为持平的掩体来回移动）。"""
    m = _open_map()
    m[4, 5] = m[6, 5] = m[5, 4] = STONE
    assert find_cover_tile(m, (5, 5), hostiles=[(9, 5)], max_steps=8) is None


def test_find_cover_tile_respects_step_budget():
    m = _open_map(size=30)
    m[20, 5] = m[22, 5] = m[21, 4] = STONE   # 坑位在 (21,5)，距玩家 16 步
    assert find_cover_tile(m, (5, 5), hostiles=[], max_steps=8) is None
    assert find_cover_tile(m, (5, 5), hostiles=[], max_steps=20) == (21, 5)


def test_find_dig_pocket_into_stone_mass():
    """向石堆里挖一格 → 三面墙坑位。这是旷野里唯一能造出坑位的手段
    （放石只能封朝向格，四面自封在 Craftax 里做不到）。"""
    m = _open_map()
    m[6:10, 3:7] = STONE          # 一块石堆
    m[5, 5] = GRASS               # 玩家站位就在石堆边
    pocket = find_dig_pocket(m, (5, 5), max_steps=4)
    assert pocket is not None
    stand, stone = pocket
    assert stand == (5, 5)
    assert stone == (6, 5)        # 挖进去后 (6,5) 的另外三面仍是石头
    assert int(m[stone]) == STONE


def test_find_dig_pocket_rejects_thin_wall():
    """只有一层厚的石墙：挖开后是通道不是坑位（两侧都通）→ 不作为庇护所目标。"""
    m = _open_map()
    m[6, :] = STONE               # 一条贯穿的单层石墙
    assert find_dig_pocket(m, (5, 5), max_steps=4) is None


def test_find_dig_pocket_none_without_stone():
    assert find_dig_pocket(_open_map(), (5, 5), max_steps=6) is None


def test_seal_target_only_for_placeable_open_tile():
    m = _open_map()
    assert seal_target(m, (5, 5), (1, 0)) == (6, 5)      # 草地可放置
    m[6, 5] = STONE
    assert seal_target(m, (5, 5), (1, 0)) is None        # 已经是墙
    m[6, 5] = GRASS
    # 怪站在朝向格上：放石不产生阻挡（怪仍可从那里攻击）→ 不放
    assert seal_target(m, (5, 5), (1, 0), mob_cells=[(6, 5)]) is None


def test_seal_target_out_of_bounds():
    assert seal_target(_open_map(), (0, 0), (-1, 0)) is None


@pytest.mark.parametrize("cell", [(0, 0), (11, 11), (0, 11)])
def test_geometry_functions_survive_map_corners(cell):
    """角落/边界不应抛异常（越界一律按墙处理）。"""
    m = _open_map()
    assert 0 <= wall_count(m, cell) <= 4
    assert len(exposed_dirs(m, cell)) <= 4
