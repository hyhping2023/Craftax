"""模拟录制任务：API 创建 session -> 脚本策略 step -> DELETE 触发封存 -> 校验 + 读取。

用法：python scripts/record_demo.py
输出：data/spool/<dataset_run_id>/ 下的 sealed shard（已 gitignore）。
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = "http://127.0.0.1:8321"
SPOOL = PROJECT_ROOT / "data" / "spool"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def delete(path: str) -> int:
    req = urllib.request.Request(BASE + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    SPOOL.mkdir(exist_ok=True)

    # 1. 创建带录制的会话（collect_wood 任务，40 步上限，god_mode 保证流程稳定）
    resp = post(
        "/v1/sessions",
        {
            "env_name": "Craftax-Pixels-v1",
            "seed": 2026,
            "task": {"task_id": "native.collect_wood", "version": "1.0.0"},
            "recording": {
                "enabled": True,
                "dataset_run_id": "sim-record-1",
                "frame_sample": {"step_rate_hz": 20, "video_fps": 10},
                "gold_frames": True,
                "spool_dir": str(SPOOL),
            },
            "max_timesteps": 40,
            "god_mode": True,
        },
    )
    sid = resp["session_id"]
    summary = resp.get("state_summary") or {}
    print(f"[1] session={sid} rev={resp['revision']}")
    print(f"    task={summary.get('task_id')}@{summary.get('task_version')}")
    print(f"    instruction={summary.get('instruction', '')[:50]}")
    print(f"    frame={resp.get('frame')}")

    # 2. 脚本策略：随机动作直到 episode 结束
    from craftax.craftax.constants import Action

    action_names = {a.value: a.name for a in Action}
    rng = random.Random(7)
    actions = list(range(1, 43))  # 排除 NOOP
    t = 0
    while True:
        action_id = rng.choice(actions)
        r = post(
            f"/v1/sessions/{sid}/step",
            {
                "action": {"id": action_id, "name": action_names[action_id]},
                "command_id": f"sim-{t}",
            },
        )
        t += 1
        if r.get("terminated") or r.get("truncated"):
            print(
                f"[2] episode end at step {t}: "
                f"terminated={r['terminated']} truncated={r['truncated']} "
                f"reward={r['reward']:.2f} events={r.get('info', {}).get('event_tokens')}"
            )
            break
        if t % 10 == 0:
            s = r.get("state_summary") or {}
            print(
                f"    step {t}: ts={r['timestep']} reward={r['reward']:.2f} "
                f"progress={s.get('task_progress', 0):.2f} events={r.get('info', {}).get('event_tokens')}"
            )

    # 3. 删除会话 -> 触发 recorder close + shard 封存
    code = delete(f"/v1/sessions/{sid}")
    print(f"[3] delete session -> {code}")

    # 4. 等待 shard 落盘（异步写入）
    shard_manifests: list[Path] = []
    deadline = time.time() + 90
    while time.time() < deadline:
        run_dir = SPOOL / "sim-record-1"
        if run_dir.exists():
            shard_manifests = list(run_dir.rglob("shard_manifest.json"))
        if shard_manifests:
            break
        time.sleep(1)
    print(f"[4] sealed shards: {len(shard_manifests)}")
    if not shard_manifests:
        print("    FAIL: shard not found")
        return

    # 5. 校验 + 读取
    from craftax.dataset.reader import ShardReader
    from craftax.dataset.vla_windows import vla_samples
    from craftax.dataset.world_model_windows import wm_samples
    from craftax.recording.validators import validate_shard

    shard_dir = shard_manifests[0].parent
    ok, errors = validate_shard(shard_dir)
    print(f"[5] validate_shard -> ok={ok} errors={errors[:3] if errors else []}")

    reader = ShardReader(shard_dir)
    episodes = list(reader.episodes())
    print(f"    episodes: {len(episodes)}")
    for ep in episodes:
        rows = ep.frame_rows()
        print(
            f"    ep={ep.episode_id} task={ep.task_id} states={ep.num_states} "
            f"trans={ep.num_transitions} frames={ep.num_frames} video={ep.video_id}"
        )
        print(
            f"      frame_index: rows={len(rows)} first_ts={rows[0]['timestep']} "
            f"last_ts={rows[-1]['timestep']} terminal={rows[-1]['is_terminal_frame']}"
        )
        frames = list(ep.frames())
        print(f"      decoded video frames: {len(frames)}")

    vs = list(vla_samples(reader, window_len=4))
    ws = list(wm_samples(reader, window_len=4))
    print(f"    vla_samples: {len(vs)} | wm_samples: {len(ws)}")
    if vs:
        s = vs[0]
        print(
            f"      vla[0]: frames={len(s['frames'])} actions={len(s['actions'])} "
            f"instruction={s['instruction'][:40]} episode={s['episode_id']}"
        )
    if ws:
        s = ws[0]
        print(
            f"      wm[0]: states={len(s['states'])} actions={len(s['actions'])} "
            f"events={s['events'][:2]} episode={s['episode_id']}"
        )

    print("[6] shard 文件清单:")
    for f in sorted(shard_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(shard_dir)
            print(f"    {rel}  {f.stat().st_size} B")


if __name__ == "__main__":
    main()
