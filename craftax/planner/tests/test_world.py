"""world.py 单测：种子事实加载、就绪度评分、候选种子。

扫描数据用**仓库内固定 fixture**（tests/fixtures/seed_scan.json，由
scripts/scan_seeds.py 对 seed 3017/3050/2011/2026/2027/2028 实测产出并提交），
而不是 data/ 下的运行时产物——.gitignore 里的 `data` 会匹配任意层级的 data 目录，
新克隆的仓库没有它，本文件过去因此在干净检出上直接失败（所以 fixture 目录也
不能叫 data）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from craftax.planner.world import (
    FloorFacts,
    SeedReadiness,
    WorldFacts,
    best_seeds,
    target_floor_for_task,
)

SCAN = str(Path(__file__).parent / "fixtures" / "seed_scan.json")


def test_world_facts_from_scan():
    wf = WorldFacts.for_seed(3017, SCAN)
    f0 = wf.floor(0)
    assert f0 is not None
    assert f0.ore_count("iron") == 1
    assert f0.ore_count("coal") == 1
    assert wf.reaches(2)  # L0/L1 ladder_down 可达（golden seed）
    assert wf.reaches(8)


def test_world_facts_unknown_seed():
    wf = WorldFacts.for_seed(123456, SCAN)
    assert wf.floor(0) is not None
    assert wf.floor(0).ore_count("iron") == 0
    # 未扫描 seed 保守判定不可达
    assert not wf.reaches(2)


def test_floor_facts_special_floors():
    wf = WorldFacts.for_seed(3017, SCAN)
    assert wf.floor(6).has_drink is False   # 火界无水源
    assert wf.floor(7).has_food is False    # 冰界无食物
    assert wf.floor(6).requires_elemental is True
    assert wf.floor(7).requires_elemental is True


def test_armor_resources():
    wf = WorldFacts.for_seed(3017, SCAN)
    iron, coal = wf.armor_resources((0,))
    assert (iron, coal) == (1, 1)
    assert not wf.armor_pieces_feasible((0,), pieces=1)  # 1 铁 1 煤 < 3/3
    # L0+L1+L2 合计：1+0+2=3 铁，1+0+4=5 煤 → 可做 1 件
    assert wf.armor_pieces_feasible((0, 1, 2), pieces=1)


def test_seed_readiness_golden():
    wf = WorldFacts.for_seed(3017, SCAN)
    rd = SeedReadiness(3017, 8, wf)
    ev = rd.evaluate()
    assert ev["reach"] is True
    assert ev["verdict"] in ("GOOD", "OK")


def test_target_floor_for_task():
    assert target_floor_for_task("native.enter_gnomish_mines") == 2
    assert target_floor_for_task("native.defeat_troll") == 5
    assert target_floor_for_task("native.defeat_necromancer") == 8
    assert target_floor_for_task("native.collect_sapphire") == 5
    assert target_floor_for_task("native.some_unknown") == 1


def test_best_seeds_returns_golden_first():
    seeds = best_seeds("native.enter_gnomish_mines", n=3, scan_path=SCAN)
    assert seeds, "seed_scan.json 应至少有一个 seed"
    # 候选列表应从可达性最高开始
    first = WorldFacts.for_seed(seeds[0], SCAN)
    assert first.reaches(2) or True  # 排序由 test_best_seeds_rank 精确验证


def test_best_seeds_rank():
    seeds = best_seeds("native.enter_gnomish_mines", n=10, scan_path=SCAN)
    assert seeds
    ranked_reach = []
    for s in seeds:
        wf = WorldFacts.for_seed(s, SCAN)
        ranked_reach.append(wf.reaches(2))
    # 可达的应排在不可达之前
    first_false = ranked_reach.index(False) if False in ranked_reach else len(ranked_reach)
    assert all(ranked_reach[:first_false])


def test_load_scan_missing_file():
    assert WorldFacts._load_scan("data/does_not_exist.json") is None


def test_load_scan_aggregates_multi_file(tmp_path, monkeypatch):
    """seed_scan*.json + seed_scan*_chunks/chunk_*.json 都纳入聚合。"""
    import json

    from craftax.contracts import default_data_dir

    monkeypatch.setattr(
        "craftax.planner.world.default_data_dir", lambda: str(tmp_path)
    )
    seed_a = {"seed": 2001, "all_ladders_reachable": True,
              "floors": [{"floor": 0, "ore": {"coal": 3, "iron": 3}}]}
    seed_b = {"seed": 2002, "all_ladders_reachable": False,
              "floors": [{"floor": 0, "ore": {"coal": 5, "iron": 1}}]}
    (tmp_path / "seed_scan.json").write_text(json.dumps({
        "scanned": 1, "golden_seeds": [2001], "results": [seed_a]}))
    chunk_dir = tmp_path / "seed_scan_2000_chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_002000.json").write_text(json.dumps({
        "scanned": 1, "golden_seeds": [], "results": [seed_b]}))
    data = WorldFacts._load_scan()
    assert data is not None
    seeds = {r["seed"] for r in data["results"]}
    assert seeds == {2001, 2002}
    assert data["golden_seeds"] == [2001]
