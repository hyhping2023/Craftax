from __future__ import annotations

import random

from scripts.demo_task_videos import MOVE_IDS, DO, planner_action, select_demo_action


def _critical_summary() -> dict:
    return {
        "energy": 9.0,
        "food": 9.0,
        "drink": 0.0,
        "is_sleeping": False,
    }


def test_planner_action_wins_over_supply_fallback():
    """危险补给状态不能把规划器的补水/移动/战斗动作改写成 DO。"""
    assert select_demo_action(
        _critical_summary(), 2, use_planner=True, rng=random.Random(0)
    ) == 2


def test_planner_critical_fallback_moves_instead_of_repeating_do():
    """规划暂时无结果时，危险状态也要离开水边等待下一轮规划。"""
    action = select_demo_action(
        _critical_summary(), None, use_planner=True, rng=random.Random(0)
    )
    assert action in MOVE_IDS
    assert action != DO


def test_no_planner_keeps_legacy_supply_fallback():
    assert select_demo_action(
        _critical_summary(), None, use_planner=False, rng=random.Random(0)
    ) == DO


def test_planner_window_rebase_is_not_accumulated():
    """Cached window coordinates are rebased from the fetch anchor each call."""
    class Capture:
        def __init__(self):
            self.positions = []

        def next_action(self, payload, summary):
            self.positions.append(tuple(payload["player_position"]))
            return 1

    chain = Capture()
    payload = {
        "map": [[0]],
        "player_position": [40, 40],
        "player_global_position": [100, 100],
        "player_direction": 0,
        "_summary_anchor_position": [10, 10],
    }
    summary = {"player_position": [11, 10], "player_direction": 1}
    assert planner_action(payload, summary, "native.test", chain) == 1
    summary["player_position"] = [12, 10]
    assert planner_action(payload, summary, "native.test", chain) == 1
    assert chain.positions == [(41, 40), (42, 40)]
