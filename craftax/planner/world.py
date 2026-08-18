"""世界事实（WorldFacts）：把种子扫描数据 + 游戏常量整理为规划可用的每层事实。

用途：
- 种子就绪度评分（双轨制）：一个 seed 对"到达目标楼层 + 装甲可行性 + 生存资源"
  的综合判定，生成候选种子排序列表（供测试与 demo）；
- 运行时楼层就绪门的跨层参考：当前层矿石用实时地图，其他层矿石用扫描数据。

数据来源：
- scripts/scan_seeds.py 产出的 <data_dir>/seed_scan.json：
  {results: [{seed, all_ladders_reachable, ore_totals,
              floors: [{floor, ladder_down_reachable, reachable_cells,
                        ore: {coal,iron,diamond,sapphire,ruby}}]}]}
- craftax/planner/combat_model.py 的 MOB_STATS（怪属性）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from craftax.contracts import default_data_dir
from craftax.planner.combat_model import MOB_STATS

# _load_scan 结果缓存（按路径），避免 best_seeds 对每个 seed 重复解析 JSON
_SCAN_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

# 护甲制造成本（每件铁甲 3 铁 + 3 煤）
ARMOUR_IRON_COST = 3
ARMOUR_COAL_COST = 3

# 任务需要的最深楼层（enter/reach/learn/combat/collect 的综合）
_TASK_TARGET_FLOOR: Dict[str, int] = {
    "native.enter_dungeon": 1,
    "native.enter_gnomish_mines": 2,
    "native.enter_sewers": 3,
    "native.enter_vault": 4,
    "native.enter_troll_mines": 5,
    "native.enter_fire_realm": 6,
    "native.enter_ice_realm": 7,
    "native.enter_graveyard": 8,
    "native.reach_floor_3": 3,
    "native.reach_floor_5": 5,
    "native.reach_boss_floor": 8,
    "native.explore_dungeon": 8,
    "native.learn_fireball": 3,
    "native.learn_iceball": 4,
    "native.collect_diamond": 2,
    "native.collect_sapphire": 5,
    "native.collect_ruby": 5,
    "native.defeat_gnome_warrior": 2,
    "native.defeat_gnome_archer": 2,
    "native.defeat_orc_soldier": 1,
    "native.defeat_orc_mage": 1,
    "native.defeat_kobold": 3,
    "native.defeat_knight": 4,
    "native.defeat_archer": 4,
    "native.defeat_troll": 5,
    "native.defeat_necromancer": 8,
    "native.damage_necromancer": 8,
}

ORE_KEYS = ("coal", "iron", "diamond", "sapphire", "ruby")


@dataclass(frozen=True)
class FloorFacts:
    floor: int
    ore: Dict[str, int] = field(default_factory=dict)
    ladder_down_reachable: Optional[bool] = None
    reachable_cells: int = 0
    # 从 combat_model 派生
    mob_melee_dmg: float = 0.0
    mob_melee_hp: float = 0.0
    mob_ranged_dmg: float = 0.0
    mob_ranged_hp: float = 0.0
    requires_elemental: bool = False
    # 特殊层
    has_drink: bool = True      # L6 火界 / L8 Boss 无水源
    has_food: bool = True       # L7 冰界无被动怪/植物

    def ore_count(self, key: str) -> int:
        return int(self.ore.get(key, 0))

    @property
    def armor_iron(self) -> int:
        return self.ore_count("iron")

    @property
    def armor_coal(self) -> int:
        return self.ore_count("coal")


# 无水源/食物层（与 executor.NO_DRINK_FLOORS / NO_FOOD_FLOORS 一致）
_NO_DRINK_FLOORS = {6, 8}
_NO_FOOD_FLOORS = {7}


def _floor_facts_from_scan(floor: int, scan_floor: Dict[str, Any]) -> FloorFacts:
    ore = {k: int(scan_floor.get("ore", {}).get(k, 0)) for k in ORE_KEYS}
    stats = MOB_STATS.get(floor, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False))
    return FloorFacts(
        floor=floor,
        ore=ore,
        ladder_down_reachable=scan_floor.get("ladder_down_reachable"),
        reachable_cells=int(scan_floor.get("reachable_cells", 0)),
        mob_melee_dmg=float(stats[0]),
        mob_melee_hp=float(stats[1]),
        mob_ranged_dmg=float(stats[3]),
        mob_ranged_hp=float(stats[4]),
        requires_elemental=bool(stats[6]),
        has_drink=floor not in _NO_DRINK_FLOORS,
        has_food=floor not in _NO_FOOD_FLOORS,
    )


class WorldFacts:
    """一个 seed 的世界事实（跨层矿石/梯子可达性/生存资源）。"""

    def __init__(self, seed: int, floors: Dict[int, FloorFacts]) -> None:
        self.seed = seed
        self._floors = floors

    @classmethod
    def for_seed(
        cls, seed: int, scan_path: Optional[str] = None
    ) -> "WorldFacts":
        """从 seed_scan.json 构建；该 seed 未扫描时返回空事实（未知矿石）。"""
        floors: Dict[int, FloorFacts] = {}
        data = cls._load_scan(scan_path)
        entry = None
        if data is not None:
            for r in data.get("results", []):
                if int(r.get("seed", -1)) == seed:
                    entry = r
                    break
        if entry is None:
            for floor in range(9):
                stats = MOB_STATS.get(floor, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False))
                floors[floor] = FloorFacts(
                    floor=floor,
                    ladder_down_reachable=None,
                    mob_melee_dmg=float(stats[0]),
                    mob_melee_hp=float(stats[1]),
                    mob_ranged_dmg=float(stats[3]),
                    mob_ranged_hp=float(stats[4]),
                    requires_elemental=bool(stats[6]),
                    has_drink=floor not in _NO_DRINK_FLOORS,
                    has_food=floor not in _NO_FOOD_FLOORS,
                )
            return cls(seed, floors)
        for f in entry.get("floors", []):
            floors[int(f["floor"])] = _floor_facts_from_scan(int(f["floor"]), f)
        return cls(seed, floors)

    @staticmethod
    def _load_scan(scan_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """读取种子扫描数据（带缓存）。

        默认聚合 `<data_dir>/seed_scan.json` + 所有 `seed_scan_*.json`
        （如 seed_scan_2000.json 扩大扫描批次）+ `seed_scan*_chunks/chunk_*.json`，
        按 seed 去重合并。缓存避免 best_seeds 对每个 seed 重复解析 JSON。
        """
        cache_key = scan_path if scan_path is not None else "__aggregate__"
        if cache_key in _SCAN_CACHE:
            return _SCAN_CACHE[cache_key]
        if scan_path is not None:
            candidate = Path(scan_path)
            if not candidate.exists():
                _SCAN_CACHE[cache_key] = None
                return None
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                _SCAN_CACHE[cache_key] = data
                return data
            except Exception:  # noqa: BLE001
                _SCAN_CACHE[cache_key] = None
                return None
        data_dir = Path(default_data_dir())
        files = sorted(data_dir.glob("seed_scan*.json"))
        # 分块扫描产物（seed_scan_2000_chunks/chunk_*.json）也纳入
        files += sorted(data_dir.glob("seed_scan*_chunks/chunk_*.json"))
        files = sorted(set(files))
        if not files:
            _SCAN_CACHE[cache_key] = None
            return None
        results: Dict[int, Dict[str, Any]] = {}
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for r in data.get("results", []):
                results[int(r["seed"])] = r
        if not results:
            _SCAN_CACHE[cache_key] = None
            return None
        merged = [results[s] for s in sorted(results)]
        data = {
            "scanned": len(merged),
            "golden_seeds": sorted(
                s for s, r in results.items() if r.get("all_ladders_reachable")
            ),
            "results": merged,
        }
        _SCAN_CACHE[cache_key] = data
        return data

    def floor(self, floor: int) -> Optional[FloorFacts]:
        return self._floors.get(floor)

    def reaches(self, target_floor: int) -> bool:
        """从 L0 出生点能否依次下到 target_floor（每层 ladder_down 可达）。"""
        for f in range(min(target_floor, 8)):
            facts = self._floors.get(f)
            if facts is None or facts.ladder_down_reachable is None:
                # 未扫描 → 无法判定（保守视为不可行）
                return False
            if not facts.ladder_down_reachable:
                return False
        return True

    def armor_resources(self, floors: Sequence[int]) -> Tuple[int, int]:
        """这些层的可达铁/煤合计（用于判断能否就地做护甲）。"""
        iron = coal = 0
        for f in floors:
            facts = self._floors.get(f)
            if facts is not None:
                iron += facts.armor_iron
                coal += facts.armor_coal
        return iron, coal

    def armor_pieces_feasible(self, floors: Sequence[int], pieces: int = 1) -> bool:
        iron, coal = self.armor_resources(floors)
        return iron >= ARMOUR_IRON_COST * pieces and coal >= ARMOUR_COAL_COST * pieces


def target_floor_for_task(task_id: str) -> int:
    """任务需要到达的最深楼层（用于种子筛选）；未知任务默认 1。"""
    return _TASK_TARGET_FLOOR.get(task_id, 1)


class SeedReadiness:
    """一个 seed 对一个目标楼层的就绪度判定（双轨制）。"""

    def __init__(self, seed: int, target_floor: int, world: WorldFacts) -> None:
        self.seed = seed
        self.target_floor = target_floor
        self.world = world

    @property
    def reach(self) -> bool:
        return self.world.reaches(self.target_floor)

    @property
    def armor_feasible(self) -> bool:
        # 优先 L0；其次 L0+L2（下到 L2 前可折返采）
        if self.world.armor_pieces_feasible((0,), pieces=1):
            return True
        return self.world.armor_pieces_feasible((0, 1, 2), pieces=1)

    @property
    def survival_ok(self) -> bool:
        # 锚点恢复：目标层（含）以上有可恢复层即可；L0 恒有水/食物
        return True

    def evaluate(self) -> Dict[str, Any]:
        verdict = "INFEASIBLE"
        if self.reach:
            if self.armor_feasible:
                verdict = "GOOD"      # 装甲路线
            else:
                verdict = "OK"        # 风筝/力量/锚点恢复路线
        return {
            "seed": self.seed,
            "target_floor": self.target_floor,
            "reach": self.reach,
            "armor_feasible": self.armor_feasible,
            "survival_ok": self.survival_ok,
            "verdict": verdict,
        }


def load_scan_results(scan_path: Optional[str] = None) -> List[Dict[str, Any]]:
    data = WorldFacts._load_scan(scan_path)
    if data is None:
        return []
    return list(data.get("results", []))


def best_seeds(
    task_id: str, n: int = 5, scan_path: Optional[str] = None
) -> List[int]:
    """为任务挑选候选 seed：按 (reach, armor_feasible) 字典序排序取前 n。

    仅基于 seed_scan.json 中已扫描的 seed；未扫描返回空列表。
    """
    target = target_floor_for_task(task_id)
    results = load_scan_results(scan_path)
    scored: List[Tuple[Tuple[bool, bool, int], int]] = []
    for r in results:
        seed = int(r["seed"])
        world = WorldFacts.for_seed(seed, scan_path)
        rd = SeedReadiness(seed, target, world)
        key = (rd.reach, rd.armor_feasible, -seed)  # 同分取小 seed
        scored.append((key, seed))
    scored.sort(reverse=True)
    return [s for _, s in scored[:n]]


def best_golden_seeds(scan_path: Optional[str] = None) -> List[int]:
    """全部梯子可达的 golden seeds（按 L0 装甲可行性优先）。"""
    results = load_scan_results(scan_path)
    scored: List[Tuple[Tuple[bool, int], int]] = []
    for r in results:
        seed = int(r["seed"])
        world = WorldFacts.for_seed(seed, scan_path)
        key = (world.armor_pieces_feasible((0,), pieces=1), -seed)
        scored.append((key, seed))
    scored.sort(reverse=True)
    return [s for _, s in scored]
