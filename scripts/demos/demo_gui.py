"""Demo 2：Pygame GUI（remote HTTP 或 embedded 同进程）。

remote：连接已启动的 service（uvicorn）。
embedded：本进程直接创建 SessionActor，无需服务。

用法：
    python scripts/demos/demo_gui.py --base http://127.0.0.1:8321
    python scripts/demos/demo_gui.py --embedded
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from craftax.gui.pygame_client import PygameGUI  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pygame GUI demo")
    parser.add_argument("--base", default="http://127.0.0.1:8321",
                        help="remote 模式的 service 地址")
    parser.add_argument("--embedded", action="store_true",
                        help="同进程 embedded 模式（不连接服务）")
    parser.add_argument("--env", default="Craftax-Pixels-v1")
    parser.add_argument("--block-pixel-size", type=int, default=64,
                        help="服务端渲染方块像素尺寸：64 ≈ 720p(704x832)")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if args.embedded:
        from craftax.contracts import RecordingConfig
        from craftax.craftax.envs.craftax_symbolic_env import (
            CraftaxSymbolicEnvNoAutoReset,
        )
        from craftax.service.session_actor import SessionActor

        env = CraftaxSymbolicEnvNoAutoReset()
        actor = SessionActor(
            session_id="demo-embedded",
            env_name="Craftax-Symbolic-v1",
            seed=42,
            task=SimpleNamespace(task_id="native.survive", version="1.0.0"),
            render=SimpleNamespace(format="png", mode="human"),
            recording=RecordingConfig(enabled=False),
            env=env,
        )
        actor.reset(seed=42)
        gui = PygameGUI(driver=actor, title="Craftax (embedded)", fps=args.fps)
        print("[embedded] 直接使用本地 SessionActor，无需服务")
        gui.run()
    else:
        gui = PygameGUI.connect_http(
            args.base,
            env_name=args.env,
            fps=args.fps,
            block_pixel_size=args.block_pixel_size,
        )
        print(f"[remote] 连接 {args.base} (env={args.env}, "
              f"block_pixel_size={args.block_pixel_size})")
        gui.run()


if __name__ == "__main__":
    main()
