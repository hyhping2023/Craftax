"""Demo 3：数据录制（API -> 异步 recorder -> sealed shard -> 校验）。

用法：
    python scripts/demos/demo_record.py --steps 40 --seed 2026
    python scripts/demos/demo_record.py --task native.collect_wood --run-id demo-001
输出：<CRAFTAX_DATA_DIR 或 <仓库根>/data>/<run-id>/ 下的 sealed shard。
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from craftax.craftax.constants import Action  # noqa: E402
from craftax.contracts import default_data_dir  # noqa: E402


def post(base: str, path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def delete(base: str, path: str) -> int:
    req = urllib.request.Request(base + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    parser = argparse.ArgumentParser(description="数据录制 demo")
    parser.add_argument("--base", default="http://127.0.0.1:8321")
    parser.add_argument("--env", default="Craftax-Pixels-v1")
    parser.add_argument("--steps", type=int, default=60, help="max_timesteps（episode 长度）")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--task", default="native.collect_wood",
                        help="collect_wood：启发式策略十几步内真实砍树完成（推荐演示）；"
                             "explore_dungeon：进度按楼层，随机策略下通常不完成")
    parser.add_argument("--task-version", default="1.0.0")
    parser.add_argument("--run-id", default="demo-record")
    parser.add_argument("--gold-frames", action="store_true")
    parser.add_argument("--god-mode", action="store_true",
                        help="默认关闭：god_mode 会让玩家开局满背包（99 木材+全套装备），"
                             "收集类成就第 0 步即达成，任务演示无意义")
    parser.add_argument("--block-pixel-size", type=int, default=64,
                        help="方块像素尺寸：64 ≈ 720p(704x832) demo 高清；"
                             "真实批量录制建议 18 ≈ 240p(198x234)")
    args = parser.parse_args()

    action_names = {a.value: a.name for a in Action}
    rng = random.Random(args.seed)

    def sample_action() -> int:
        """启发式随机策略：40% 移动 + 40% DO(互动/砍树) + 20% 随机。

        移动保证画面在动；DO 在树旁会真实砍树，收集类任务（collect_wood）
        通常十几步内即可完成，演示能看到"完成任务"的过程。
        """
        r = rng.random()
        if r < 0.4:
            return rng.choice((1, 2, 3, 4))  # LEFT/RIGHT/UP/DOWN
        if r < 0.8:
            return 5  # DO
        return rng.randint(1, len(Action) - 1)

    # 1. 创建带录制的会话
    resp = post(
        args.base,
        "/v1/sessions",
        {
            "env_name": args.env,
            "seed": args.seed,
            "task": {"task_id": args.task, "version": args.task_version},
            "render": {"format": "png", "mode": "human",
                       "block_pixel_size": args.block_pixel_size},
            "recording": {
                "enabled": True,
                "dataset_run_id": args.run_id,
                "frame_sample": {"step_rate_hz": 20, "video_fps": 10},
                "gold_frames": args.gold_frames,
                "spool_dir": default_data_dir(),
            },
            "max_timesteps": args.steps,
            "god_mode": args.god_mode,
        },
    )
    sid = resp["session_id"]
    s = resp.get("state_summary") or {}
    print(f"[1] session={sid} task={s.get('task_id')}@{s.get('task_version')} "
          f"instruction={s.get('instruction', '')[:40]}")

    # 2. 随机策略 step 直到 episode 结束
    t = 0
    while True:
        action_id = sample_action()
        r = post(
            args.base,
            f"/v1/sessions/{sid}/step",
            {
                "action": {"id": action_id, "name": action_names[action_id]},
                "command_id": f"rec-{t}",
            },
        )
        t += 1
        if r.get("terminated") or r.get("truncated"):
            print(f"[2] episode end at step {t}: "
                  f"terminated={r['terminated']} truncated={r['truncated']} "
                  f"events={r.get('info', {}).get('event_tokens')}")
            break

    # 3. 删除会话 -> 触发 recorder 封存
    code = delete(args.base, f"/v1/sessions/{sid}")
    print(f"[3] delete session -> {code}")

    # 4. 等待 sealed shard
    run_dir = Path(default_data_dir()) / args.run_id
    manifests: list[Path] = []
    deadline = time.time() + 90
    while time.time() < deadline and run_dir.exists():
        manifests = list(run_dir.rglob("shard_manifest.json"))
        if manifests:
            break
        time.sleep(1)
    if not manifests:
        print("[4] FAIL: shard not found")
        return
    shard_dir = manifests[0].parent
    print(f"[4] sealed shard: {shard_dir}")

    # 5. 校验 + 摘要
    from craftax.recording.validators import validate_shard

    ok, errors = validate_shard(shard_dir)
    print(f"[5] validate_shard -> ok={ok} errors={errors[:3] if errors else []}")

    from craftax.dataset.reader import ShardReader

    reader = ShardReader(shard_dir)
    for ep in reader.episodes():
        print(f"    ep={ep.episode_id} task={ep.task_id} states={ep.num_states} "
              f"trans={ep.num_transitions} frames={ep.num_frames} video={ep.video_id}")


if __name__ == "__main__":
    main()
