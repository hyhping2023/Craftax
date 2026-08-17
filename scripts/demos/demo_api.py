"""Demo 1：模型式 API 交互。

展示具身模型如何通过 REST API 与环境交互：创建会话 -> 提交动作 ->
获取状态摘要与场景帧。默认随机策略（可复现），可换成任意策略。

用法：
    python scripts/demos/demo_api.py --steps 10 --seed 42
    python scripts/demos/demo_api.py --save-frames --steps 5   # 保存 PNG 到 data/demo_frames/
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from craftax.craftax.constants import Action  # noqa: E402


def post(base: str, path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_bytes(base: str, path: str) -> bytes:
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return r.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="模型式 API 交互 demo")
    parser.add_argument("--base", default="http://127.0.0.1:8321")
    parser.add_argument("--env", default="Craftax-Symbolic-v1",
                        help="Craftax-Symbolic-v1 | Craftax-Pixels-v1")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="native.survive")
    parser.add_argument("--block-pixel-size", type=int, default=64,
                        help="方块像素尺寸：64 ≈ 720p(704x832)；真实录制建议 18 ≈ 240p")
    parser.add_argument("--save-frames", action="store_true",
                        help="把每步 PNG 帧保存到 data/demo_frames/")
    args = parser.parse_args()

    action_names = {a.value: a.name for a in Action}
    rng = random.Random(args.seed)

    def sample_action() -> int:
        """移动优先的随机策略：60% 移动，40% 随机动作，保证演示画面在动。"""
        if rng.random() < 0.6:
            return rng.choice((1, 2, 3, 4))  # LEFT/RIGHT/UP/DOWN
        return rng.randint(1, len(Action) - 1)

    # 1. 创建会话（模型视角：一次 reset）
    resp = post(
        args.base,
        "/v1/sessions",
        {
            "env_name": args.env,
            "seed": args.seed,
            "task": {"task_id": args.task, "version": "1.0.0"},
            "render": {"format": "png", "mode": "human",
                       "block_pixel_size": args.block_pixel_size},
        },
    )
    sid = resp["session_id"]
    s = resp.get("state_summary") or {}
    print(f"[reset] session={sid} revision={resp['revision']} "
          f"task={s.get('task_id')}@{s.get('task_version')}")
    print(f"        instruction: {s.get('instruction', '')}")
    print(f"        health={s.get('health')} food={s.get('food')} "
          f"drink={s.get('drink')} floor={s.get('floor')}")

    frame_dir = PROJECT_ROOT / "data" / "demo_frames"
    if args.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    # 2. 循环交互（模型视角：obs -> action -> obs'）
    for t in range(args.steps):
        action_id = sample_action()
        r = post(
            args.base,
            f"/v1/sessions/{sid}/step",
            {
                "action": {"id": action_id, "name": action_names[action_id]},
                "command_id": f"demo-{t}",
            },
        )
        s = r.get("state_summary") or {}
        events = r.get("info", {}).get("event_tokens", [])
        print(
            f"[step {t:02d}] rev={r['revision']} action={action_names[action_id]:<24} "
            f"reward={r['reward']:+.2f} progress={s.get('task_progress', 0):.2f} "
            f"events={events}"
        )
        frame = r.get("frame")
        if args.save_frames and frame:
            png = get_bytes(args.base, frame["url"])
            (frame_dir / f"{sid[:8]}-rev{r['revision']}.png").write_bytes(png)

        if r.get("terminated") or r.get("truncated"):
            print(f"[end] episode done: terminated={r['terminated']} "
                  f"truncated={r['truncated']}")
            break

    if args.save_frames:
        print(f"[frames] saved to {frame_dir}/")


if __name__ == "__main__":
    main()
