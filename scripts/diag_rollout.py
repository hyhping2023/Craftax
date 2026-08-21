#!/usr/bin/env python
"""Rollout 诊断：跑固定 (任务, seed) 组合并记录死亡前的逐步轨迹，用于定位死因。

与快速单测不同，这里关心的是"整局怎么死的"：每步记录血/能量/水/食物、光照、
近身怪数、本层击杀数、关键装备等级，并保留结局前 25 步的轨迹。掉血却 `adj=0`
的步数即"被远程投射物命中"的次数——这是 L1 的主要死因，普通成功率统计看不到。

用法：
  PYTHONPATH=. python scripts/diag_rollout.py            # 默认 8 组，2000 步
  PYTHONPATH=. python scripts/diag_rollout.py --steps 1200 --out /tmp/a.json
  PYTHONPATH=. python scripts/diag_rollout.py --job native.reach_floor_3:3017

输出：一行一局的摘要 + 完整 JSON（含轨迹），可作 A/B 对比的基线。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402

from craftax.contracts import DEFAULT_ENERGY_RATE, DEFAULT_THIRST_RATE  # noqa: E402
from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.planner.executor import SkillChainExecutor  # noqa: E402
from craftax.planner.tests.test_executor import (  # noqa: E402
    _floor_map_payload,
    _host,
    _map_payload,
    _summary,
)

# 默认诊断组合：L1 清怪墙（enter_gnomish_mines）、表层制备（collect_iron）、
# 深链（reach_floor_3）、战斗任务（defeat_gnome_warrior）× golden seeds。
DEFAULT_JOBS = [
    ("native.enter_gnomish_mines", 3017),
    ("native.enter_gnomish_mines", 3050),
    ("native.reach_floor_3", 3017),
    ("native.reach_floor_3", 3050),
    ("native.defeat_gnome_warrior", 3017),
    ("native.collect_iron", 3017),
    ("native.enter_gnomish_mines", 2011),
    ("native.enter_gnomish_mines", 2111),
]

TRACE_TAIL = 25


def light_level(timestep: int, day_length: int = 300) -> float:
    """与 game_logic_utils.calculate_light_level 同式。"""
    progress = (timestep / day_length) % 1 + 0.3
    return float(1 - abs(np.cos(np.pi * progress)) ** 3)


def _adjacent_hostiles(payload, pos) -> int:
    total = 0
    for key in ("melee", "ranged"):
        entry = payload["mob_positions"][key]
        positions = np.array(entry["positions"])
        masks = np.array(entry["masks"], dtype=bool)
        if masks.any() and positions.size:
            dist = np.abs(positions[masks] - np.asarray(pos)).sum(axis=1)
            total += int((dist <= 1).sum())
    return total


def run(task_id: str, seed: int, max_steps: int = 2000,
        thirst_rate: float = DEFAULT_THIRST_RATE,
        energy_rate: float = DEFAULT_ENERGY_RATE) -> dict:
    env = CraftaxSymbolicEnvNoAutoReset()
    # env 与 executor 必须共用同一个 thirst_rate（见 contracts.DEFAULT_THIRST_RATE）
    params = EnvParams(thirst_rate=thirst_rate, energy_rate=energy_rate)
    state = env.reset(jax.random.PRNGKey(seed), params)[1]
    holder: dict = {}

    def provider(floor: int):
        hs = holder.get("hs")
        if hs is None or not 0 <= floor < hs.map.shape[0]:
            return None
        return _floor_map_payload(hs, floor)

    ex = SkillChainExecutor(task_id, seed=seed, thirst_rate=thirst_rate,
                            energy_rate=energy_rate,
                            floor_map_provider=provider)
    key = jax.random.PRNGKey(seed + 1)
    trace: list = []
    stats = {"adj_steps": 0, "night_steps": 0, "ranged_hits": 0, "melee_hits": 0}
    prev_health = None

    def finish(outcome: str, steps: int, hs) -> dict:
        summ = _summary(hs)
        return dict(
            task=task_id, seed=seed, outcome=outcome, steps=steps,
            floor=int(hs.player_level), trace=trace[-TRACE_TAIL:],
            inventory=summ["inventory"], n_achievements=len(summ["achievements"]),
            monsters_killed=[int(x) for x in np.asarray(hs.monsters_killed)],
            abort=ex.abort_reason(), **stats,
        )

    for i in range(max_steps):
        hs = _host(state)
        holder["hs"] = hs
        payload = _map_payload(hs)
        summ = _summary(hs)
        if ex.is_done(summ):
            return finish("done", i, hs)
        action = ex.next_action(payload, summ)
        if action is None:
            out = finish("no_action", i, hs)
            idx = ex._chain_idx
            out["goal"] = ex._chain[idx] if idx < len(ex._chain) else None
            return out

        health = float(hs.player_health)
        nadj = _adjacent_hostiles(payload, summ["player_position"])
        if prev_health is not None and health < prev_health - 0.01:
            # 掉血分流：有近身怪 = 近战命中；无近身怪且落差 >=1.5 = 远程投射物
            if nadj:
                stats["melee_hits"] += 1
            elif prev_health - health >= 1.5:
                stats["ranged_hits"] += 1
        prev_health = health
        light = light_level(int(hs.timestep))
        stats["adj_steps"] += nadj > 0
        stats["night_steps"] += light < 0.3
        goal_idx = ex._chain_idx
        trace.append(dict(
            i=i, floor=int(hs.player_level), hp=round(health, 1),
            energy=round(float(hs.player_energy), 1),
            drink=round(float(hs.player_drink), 1),
            food=round(float(hs.player_food), 1),
            action=int(action), adj=nadj, light=round(light, 2),
            kills=int(payload["monsters_killed"]),
            arrows=int(summ["inventory"]["arrows"]),
            sword=int(summ["inventory"]["sword"]),
            pickaxe=int(summ["inventory"]["pickaxe"]),
            stone=int(summ["inventory"]["stone"]),
            goal=ex._chain[goal_idx] if goal_idx < len(ex._chain) else None,
        ))
        key, subkey = jax.random.split(key)
        _obs, state, _r, done, _info = env.step(subkey, state, action, params)
        if bool(np.asarray(done)):
            return finish("died", i + 1, _host(state))
    return finish("timeout", max_steps, _host(state))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000, help="每局步数预算")
    parser.add_argument("--thirst-rate", type=float, default=DEFAULT_THIRST_RATE,
                        help="口渴衰减倍率（1.0=原版，越小水掉得越慢）")
    parser.add_argument("--energy-rate", type=float, default=DEFAULT_ENERGY_RATE,
                        help="精力衰减倍率（1.0=原版，0.25=原版四分之一）")
    parser.add_argument("--out", default="/tmp/diag_rollout.json", help="JSON 输出路径")
    parser.add_argument(
        "--job", action="append", default=None,
        help="task_id:seed（可重复）；不给则跑 DEFAULT_JOBS",
    )
    args = parser.parse_args()
    jobs = DEFAULT_JOBS
    if args.job:
        jobs = []
        for item in args.job:
            task, _, seed = item.rpartition(":")
            jobs.append((task, int(seed)))

    results = []
    for task, seed in jobs:
        r = run(task, seed, max_steps=args.steps, thirst_rate=args.thirst_rate,
                energy_rate=args.energy_rate)
        results.append(r)
        inv = r["inventory"]
        print(
            f"{r['task']:32s} seed={r['seed']} {r['outcome']:8s} steps={r['steps']:5d} "
            f"floor={r['floor']} kills={r['monsters_killed'][:3]} "
            f"sword={inv['sword']} pickaxe={inv['pickaxe']} arrows={inv['arrows']} "
            f"ach={r['n_achievements']} ranged_hits={r['ranged_hits']} "
            f"melee_hits={r['melee_hits']} night={r['night_steps']}",
            flush=True,
        )
    with open(args.out, "w") as handle:
        json.dump(results, handle)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
