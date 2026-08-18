"""为每个已注册任务生成一个 demo 视频（CFR MP4）。

流程：
1. 确保 uvicorn service 在 127.0.0.1:8321 可访问（不可访问则自动拉起子进程）；
2. 可选 warmup：编译 env step + pixel renderer，并生成目标分辨率的纹理（一次）；
3. 遍历任务（--tasks 指定子集，默认全部已注册任务）：
   - POST /v1/sessions 创建带录制的会话（max_timesteps=--steps）
   - 生存优先的启发式策略 step 直到 episode 结束
   - DELETE 会话触发 AsyncRecorder 封存 shard
   - 等待 shard_manifest.json 出现（异步写入保险）
   - 把 shard 里的 video-*.mp4 复制为短名 demo.mp4
4. 写 <data_dir>/demo_videos.json 汇总每个任务的视频路径与任务结果。

目录布局（task_id = "native.collect_wood"）：
    <data_dir>/native/collect_wood/demo.mp4        1 级=task 命名空间，2 级=短 id，文件为 demo.mp4
    <data_dir>/native/collect_wood/recorder/<id>/  完整 sealed shard

用法：
    conda run -n craftax python scripts/demo_task_videos.py --parallel 8
    conda run -n craftax python scripts/demo_task_videos.py --tasks native.collect_wood,native.explore_dungeon
    conda run -n craftax python scripts/demo_task_videos.py --steps 120 --block-pixel-size 24
    conda run -n craftax python scripts/demo_task_videos.py --no-planner   # 收集类任务也走随机策略
    # 深层任务（下 L2+）优先用 golden seeds（梯子全部可达）：
    conda run -n craftax python scripts/demo_task_videos.py --tasks native.collect_diamond,native.reach_floor_3 --seeds 3017,3050,2027 --steps 0

默认保存位置：<CRAFTAX_DATA_DIR 或 <仓库根>/data>/。
--steps 0 = 按任务依赖链自动估算（深层任务会自动放大到数千~数万步）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 注意：不要在此处 import craftax.craftax.*（会触发 JAX 加载）；JAX 的线程模型
# 与 subprocess.Popen 的 fork 不兼容。服务器拉起后再按需惰性 import。
from craftax.contracts import default_data_dir  # noqa: E402

BASE = "http://127.0.0.1:8321"
DEFAULT_TASK_VERSION = "1.0.0"
# 系统 HTTP(S)_PROXY 会把 localhost 流量错误发送到外部 squid；录制服务必须
# 直连本机，显式禁用 urllib 代理。
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 默认 CPU 平台：GPU 常被其他任务占满显存，导致 JAX 编译/执行长时间卡死。
# 该环境变量在 jax 首次 import 时生效，且会继承给 uvicorn 子进程。
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# 非 NOOP 的动作 id（1..42）
MOVE_IDS = (1, 2, 3, 4)  # LEFT/RIGHT/UP/DOWN
DO = 5
SLEEP = 6

# 多样化动作：覆盖制作/放置/魔法/交互，让 demo 画面更多展示任务相关行为
# （资源不足时这些动作会被游戏忽略，不会出错）
VARIETY_IDS = (
    7,   # PLACE_STONE
    8,   # PLACE_TABLE
    9,   # PLACE_FURNACE
    10,  # PLACE_PLANT
    11,  # MAKE_WOOD_PICKAXE
    12,  # MAKE_STONE_PICKAXE
    13,  # MAKE_IRON_PICKAXE
    14,  # MAKE_WOOD_SWORD
    18,  # DESCEND
    19,  # ASCEND
    24,  # SHOOT_ARROW
    26,  # CAST_FIREBALL
    27,  # CAST_ICEBALL
    28,  # PLACE_TORCH
    35,  # READ_BOOK
    36,  # ENCHANT_SWORD
    38,  # MAKE_TORCH
)


def http_json(path: str, *, method: str = "GET", body: dict | None = None,
              timeout: int = 300) -> dict | int:
    """发 HTTP 请求，返回 JSON dict；非 2xx 返回 int 状态码。"""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with LOCAL_HTTP.open(req, timeout=timeout) as r:
            payload = r.read()
            return json.loads(payload) if payload else r.status
    except urllib.error.HTTPError as e:
        return e.code


def server_ready(base: str) -> bool:
    """FastAPI 默认暴露 /openapi.json，可作健康检查。"""
    try:
        req = urllib.request.Request(base + "/openapi.json", method="GET")
        with LOCAL_HTTP.open(req, timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_server(port: int = 8321) -> subprocess.Popen:
    """在当前进程的 Python 环境拉起 uvicorn，等待就绪后返回子进程。

    服务器输出重定向到临时文件而非管道：uvicorn 对每个请求写访问日志，
    若用 subprocess.PIPE 而父进程不读，64KB 管道缓冲填满后服务器写日志被
    阻塞，HTTP 请求就再也无法响应（表现为“编译很久/卡死”）。
    """
    log_file = tempfile.NamedTemporaryFile(
        prefix="craftax-server-", suffix=".log", delete=False
    )
    log_path = log_file.name
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "craftax.service.app:create_app", "--factory",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()  # 父进程不再持有句柄；子进程继续写文件
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(log_path, "rb") as f:
                out = f.read().decode(errors="replace")
            raise RuntimeError(f"uvicorn 启动失败，退出码 {proc.returncode}\n{out}")
        if server_ready(BASE):
            print(f"[server] uvicorn 已就绪 (pid={proc.pid}, log={log_path})")
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("uvicorn 启动超时")


def choose_action(summary: dict | None, rng: random.Random) -> int:
    """生存优先 + 探索：避免 demo 视频中途饿死/渴死/疲劳死。

    - 能量低 → SLEEP（6）
    - 食物/水低 → DO（5，靠近食物/水源时吃喝）
    - 其余：40% 移动 + 35% DO + 25% 多样化任务动作
    """
    if summary is None:
        return rng.choice((*MOVE_IDS, DO))
    energy = summary.get("energy", 9.0)
    food = summary.get("food", 9.0)
    drink = summary.get("drink", 9.0)
    is_sleeping = summary.get("is_sleeping", False)
    if energy < 2 and not is_sleeping:
        return SLEEP
    if food < 1 or drink < 1:
        return DO
    r = rng.random()
    if r < 0.4:
        return rng.choice(MOVE_IDS)
    if r < 0.8:
        return DO
    return rng.choice(VARIETY_IDS)


def get_floor_map(sid: str, floor: int | None = None) -> dict | None:
    """GET /v1/sessions/{sid}/map：返回 {floor, map(48×48 int 嵌套 list),
    player_position, player_direction}；失败返回 None。"""
    q = f"?floor={floor}" if floor is not None else ""
    resp = http_json(f"/v1/sessions/{sid}/map{q}")
    if not isinstance(resp, dict) or "map" not in resp:
        return None
    return resp


def planner_action(
    map_payload: dict | None,
    summary: dict | None,
    task_id: str,
    skill_chain: Any = None,
) -> int | None:
    """用确定性任务执行器返回下一步动作；无规划能力返回 None（调用方 fallback）。

    map_payload 缺失或玩家位置缺失时无法规划，返回 None。
    skill_chain 为 SkillChainExecutor 实例（全任务通用），
    内部根据任务依赖图 + 当前状态派发原语技能。
    """
    if map_payload is None or summary is None:
        return None
    if not summary.get("player_position"):
        return None

    if skill_chain is not None:
        return skill_chain.next_action(map_payload, summary)

    import numpy as np

    from craftax.planner.path_planner import next_action

    pos = summary["player_position"]
    direction = summary.get("player_direction", 0)
    inventory = summary.get("inventory") or {}
    map2d = np.asarray(map_payload["map"], dtype=np.int32)
    pickaxe = inventory.get("pickaxe", 0)
    return next_action(
        map2d,
        (int(pos[0]), int(pos[1])),
        int(direction),
        int(pickaxe),
        task_id,
    )


def wait_for_shard(spool_dir: str, run_id: str, before: set[Path],
                   timeout: float = 90.0) -> Path | None:
    """等待 run_id 目录下出现新的 sealed shard（按 manifest 增量判定）。"""
    run_dir = Path(spool_dir) / run_id
    deadline = time.time() + timeout
    while time.time() < deadline:
        manifests = set(run_dir.rglob("shard_manifest.json"))
        new = manifests - before
        if new:
            return sorted(new)[0].parent
        time.sleep(0.5)
    return None


def _survival_critical(summary: dict | None) -> bool:
    """是否处于生存临界状态（能量/食物/水危险）——此类状态不走 planner。"""
    if summary is None:
        return False
    energy = summary.get("energy", 9.0)
    food = summary.get("food", 9.0)
    drink = summary.get("drink", 9.0)
    is_sleeping = summary.get("is_sleeping", False)
    return (energy < 2 and not is_sleeping) or food < 1 or drink < 1


def record_one_task(
    task_id: str,
    *,
    version: str,
    spool_dir: str,
    steps: int,
    block_pixel_size: int,
    seed: int,
    step_rate_hz: int,
    video_fps: int,
    use_planner: bool = True,
) -> dict:
    """录制一个任务的一个 demo episode，返回结果摘要。

    目录布局（task_id = "native.collect_wood"）：
        <data_dir>/native/collect_wood/demo.mp4       短文件名 demo 视频
        <data_dir>/native/collect_wood/recorder/<id>/ 完整 sealed shard
    """
    entry: dict = {"task_id": task_id, "version": version}
    from craftax.craftax.constants import Action  # 惰性 import，避免主进程过早加载 JAX

    # 1 级文件夹 = task 命名空间（如 native），2 级文件夹 = 短 id（如 collect_wood）
    namespace, _, short_id = task_id.partition(".")
    if not short_id:
        namespace, short_id = "misc", task_id
    task_spool = str(Path(spool_dir) / namespace)
    run_id = short_id
    task_dir = Path(task_spool) / run_id

    before = set(task_dir.rglob("shard_manifest.json"))
    rng = random.Random((seed + zlib.crc32(task_id.encode())) & 0xFFFFFFFF)
    action_names = {a.value: a.name for a in Action}

    # 0. 任务执行器（供 planner 驱动 + 步数自动估算）
    skill_chain = None
    if use_planner:
        from craftax.planner.executor import SkillChainExecutor  # 惰性加载，避免拉起服务前加载 JAX

        skill_chain = SkillChainExecutor(task_id)
    if steps <= 0 and skill_chain is not None:
        steps = skill_chain.estimate_steps()

    # 1. 创建带录制的会话
    resp = http_json(
        "/v1/sessions",
        method="POST",
        body={
            "env_name": "Craftax-Pixels-v1",
            "seed": seed,
            "task": {"task_id": task_id, "version": version},
            "render": {"format": "png", "mode": "human",
                       "block_pixel_size": block_pixel_size},
            "recording": {
                "enabled": True,
                "dataset_run_id": run_id,
                "frame_sample": {"step_rate_hz": step_rate_hz, "video_fps": video_fps},
                "gold_frames": False,
                "spool_dir": task_spool,
            },
            "max_timesteps": steps,
            "god_mode": False,
        },
    )
    if not isinstance(resp, dict) or "session_id" not in resp:
        entry["error"] = f"create_session failed: {resp}"
        return entry
    sid = resp["session_id"]
    s = resp.get("state_summary") or {}
    entry["instruction"] = s.get("instruction", "")
    entry["task_id"] = s.get("task_id", task_id)
    entry["namespace"] = namespace
    entry["short_id"] = short_id
    entry["max_steps"] = steps

    # 2. 策略 step 直到 episode 结束
    t = 0
    summary = s
    map_payload = None  # planner 用：懒加载全图
    map_fetch_failures = 0
    stalled = 0  # 连续无进展步数（位置/成就均未变化）
    last_pos = None
    last_achievements = set(summary.get("achievements", []))
    try:
        while t < steps:
            # 任务已完成 → 提前结束 episode（不硬跑满 steps）
            if skill_chain is not None and skill_chain.is_done(summary):
                break
            # 生存优先保护优先于 planner（避免 demo 中途饿死/渴死/疲劳死）
            survival_action = choose_action(summary, rng) if _survival_critical(summary) else None

            if use_planner and survival_action is None:
                # 每 3 步或没有地图时拉一次全图（服务端权威状态，含怪物移动）
                if map_payload is None or t % 3 == 0:
                    map_payload = get_floor_map(sid, summary.get("floor"))
                    if map_payload is None:
                        map_fetch_failures += 1
                planned = planner_action(map_payload, summary, task_id, skill_chain)
            else:
                planned = None

            action_id = planned if planned is not None else (
                survival_action if survival_action is not None else choose_action(summary, rng)
            )
            r = http_json(
                f"/v1/sessions/{sid}/step",
                method="POST",
                body={
                    "action": {"id": action_id, "name": action_names[action_id]},
                    "command_id": f"demo-{task_id}-{t}",
                    "wait_frame": False,
                },
            )
            if not isinstance(r, dict):
                entry["error"] = f"step failed at t={t}: {r}"
                break
            t += 1
            summary = r.get("state_summary") or {}
            if r.get("terminated") or r.get("truncated"):
                entry["terminated"] = bool(r.get("terminated"))
                entry["truncated"] = bool(r.get("truncated"))
                break
            # 防死循环：位置与成就均无变化超过 10 步 → 强制重拉地图并走随机一步
            pos = summary.get("player_position")
            achievements = set(summary.get("achievements", []))
            progress_made = (pos is not None and pos != last_pos) or achievements != last_achievements
            if progress_made:
                stalled = 0
                last_pos = pos
                last_achievements = achievements
            else:
                stalled += 1
                if stalled >= 10:
                    map_payload = get_floor_map(sid, summary.get("floor"))
                    stalled = 0
        entry["num_steps"] = t
        entry["task_done"] = bool((summary or {}).get("task_done"))
        entry["task_progress"] = float((summary or {}).get("task_progress", 0.0))
        entry["achievements"] = (summary or {}).get("achievements", [])
        if map_fetch_failures:
            entry["map_fetch_failures"] = map_fetch_failures
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"step loop failed: {exc!r}"

    # 3. DELETE 会话 → 触发 recorder 封存
    code = http_json(f"/v1/sessions/{sid}", method="DELETE")
    if isinstance(code, int) and code not in (204, 404):
        entry["error"] = f"delete session failed: {code}"

    # 4. 等待 sealed shard 并复制短视频
    shard_dir = wait_for_shard(task_spool, run_id, before)
    if shard_dir is None:
        entry["error"] = "shard not sealed within timeout"
        return entry
    entry["shard_dir"] = str(shard_dir)

    videos = sorted(shard_dir.glob("video-*.mp4"))
    if not videos:
        entry["error"] = "no video in shard"
        return entry
    src = videos[0]

    task_dir.mkdir(parents=True, exist_ok=True)
    dst = task_dir / "demo.mp4"
    try:
        import shutil

        shutil.copyfile(src, dst)
        entry["video"] = str(dst)
        entry["video_bytes"] = dst.stat().st_size
        import imageio.v3 as iio

        decoded = iio.imread(src)
        entry["video_frames"] = decoded.shape[0] if decoded.ndim == 4 else 1
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"video copy/decode failed: {exc!r}"
    return entry


def record_one_task_multi_seed(
    task_id: str,
    *,
    seeds: list[int],
    version: str,
    spool_dir: str,
    steps: int,
    block_pixel_size: int,
    step_rate_hz: int,
    video_fps: int,
    use_planner: bool = True,
) -> dict:
    """为任务尝试多个世界种子，返回第一个成功（有视频）的 entry。

    全部种子都失败时返回最后一次尝试的 entry，并合并 seed_attempts 供审计。
    """
    attempts: list[dict] = []
    last_entry: dict = {"task_id": task_id, "version": version}
    for seed in seeds:
        entry = record_one_task(
            task_id,
            version=version,
            spool_dir=spool_dir,
            steps=steps,
            block_pixel_size=block_pixel_size,
            seed=seed,
            step_rate_hz=step_rate_hz,
            video_fps=video_fps,
            use_planner=use_planner,
        )
        entry["seed"] = seed
        ok = "video" in entry and "error" not in entry
        done = bool(entry.get("task_done", False))
        attempts.append(
            {
                "seed": seed,
                "ok": ok,
                "task_done": done,
                "steps": entry.get("num_steps", 0),
                "error": entry.get("error"),
            }
        )
        last_entry = entry
        # 任务已完成即停；否则换下一个种子重试（多种子提升成功率）
        if done:
            break
    last_entry["seed_attempts"] = attempts
    if not last_entry.get("task_done") and "video" not in last_entry:
        last_entry["error"] = (
            f"all {len(seeds)} seeds failed: "
            + "; ".join(f"seed={a['seed']} {a['error']}" for a in attempts)
        )
    return last_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="为每个任务生成 demo 视频")
    global BASE
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--tasks", help="逗号分隔的任务 id 子集；默认全部已注册任务")
    parser.add_argument("--task-version", default=DEFAULT_TASK_VERSION)
    parser.add_argument("--steps", type=int, default=1500, help="每个 episode 的最大步数")
    parser.add_argument("--seed", type=int, default=2026,
                        help="世界生成种子（--seeds 未提供时使用）")
    parser.add_argument("--seeds", help="逗号分隔的世界种子列表；每个任务依次尝试多个种子直到成功")
    parser.add_argument("--block-pixel-size", type=int, default=24,
                        help="方块像素尺寸：24≈264x312（demo 适中），18≈240p，64≈720p")
    parser.add_argument("--step-rate-hz", type=int, default=20)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--data-dir", default=default_data_dir(),
                        help=f"录制根目录（默认 {default_data_dir()}）")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu",
                        help="JAX 运行平台：默认 cpu（GPU 显存常被其他任务占满，会导致卡死）")
    parser.add_argument("--parallel", type=int, default=8, help="并发任务数")
    parser.add_argument("--no-planner", action="store_true",
                        help="关闭确定性路径规划（收集类任务也走随机策略）")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有 demo.mp4 的任务（断点续录）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过 warmup 会话")
    parser.add_argument("--keep-server", action="store_true",
                        help="结束后不关闭由本脚本拉起的 uvicorn")
    args = parser.parse_args()
    BASE = args.base
    if args.platform == "gpu":
        os.environ["JAX_PLATFORMS"] = "gpu"

    seeds: list[int] = []
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = [args.seed]
    print(f"[seeds] 每个任务尝试种子: {seeds}")

    spool_dir = str(Path(args.data_dir))
    Path(spool_dir).mkdir(parents=True, exist_ok=True)
    use_planner = not args.no_planner

    # 0. 服务就绪（自动拉起）
    started = False
    if not server_ready(args.base):
        proc = start_server()
        started = True
    else:
        print(f"[server] 已存在 service: {args.base}")

    try:
        # 1. warmup：编译 env step + renderer + 生成目标尺寸纹理
        if not args.no_warmup:
            resp = http_json(
                "/v1/sessions",
                method="POST",
                body={
                    "env_name": "Craftax-Pixels-v1",
                    "seed": 0,
                    "task": {"task_id": "native.survive", "version": args.task_version},
                    "render": {"format": "png", "mode": "human",
                               "block_pixel_size": args.block_pixel_size},
                    "recording": {"enabled": False},
                    "max_timesteps": 2,
                },
            )
            if isinstance(resp, dict) and "session_id" in resp:
                sid = resp["session_id"]
                for i in range(2):
                    http_json(f"/v1/sessions/{sid}/step", method="POST",
                              body={"action": {"id": 5, "name": "DO"},
                                    "command_id": f"warm-{i}", "wait_frame": False})
                http_json(f"/v1/sessions/{sid}", method="DELETE")
                print("[warmup] env step + renderer 已编译")

        # 2. 任务清单
        if args.tasks:
            task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        else:
            from craftax.tasks.registry import list_task_ids

            task_ids = list_task_ids()
        if args.skip_existing:
            def _demo_path(t: str) -> Path:
                ns, _, sid = t.partition(".")
                if not sid:
                    ns, sid = "misc", t
                return Path(spool_dir) / ns / sid / "demo.mp4"

            skipped = [t for t in task_ids if _demo_path(t).exists()]
            task_ids = [t for t in task_ids if not _demo_path(t).exists()]
            print(f"[tasks] skip-existing 跳过 {len(skipped)} 个已录制任务")
        print(f"[tasks] 共 {len(task_ids)} 个任务待录制")
        if use_planner:
            from craftax.planner import is_collect_task  # noqa: E402 惰性加载

            planned = [t for t in task_ids if is_collect_task(t)]
            print(f"[planner] 收集类任务确定性路径: {len(planned)} 个 {planned}")

        # 3. 逐任务录制（每个任务依次尝试多个种子直到成功）
        results: list[dict] = []
        if args.parallel <= 1:
            for task_id in task_ids:
                entry = record_one_task_multi_seed(
                    task_id,
                    seeds=seeds,
                    version=args.task_version,
                    spool_dir=spool_dir,
                    steps=args.steps,
                    block_pixel_size=args.block_pixel_size,
                    step_rate_hz=args.step_rate_hz,
                    video_fps=args.video_fps,
                    use_planner=use_planner,
                )
                ok = "video" in entry and "error" not in entry
                print(f"  [{'OK ' if ok else 'ERR'}] {task_id:<38} "
                      f"steps={entry.get('num_steps', '-'):<4} "
                      f"done={entry.get('task_done', '-')} "
                      f"frames={entry.get('video_frames', '-')} "
                      f"{entry.get('error', '')}")
                results.append(entry)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futures = {
                    ex.submit(
                        record_one_task_multi_seed,
                        task_id,
                        seeds=seeds,
                        version=args.task_version,
                        spool_dir=spool_dir,
                        steps=args.steps,
                        block_pixel_size=args.block_pixel_size,
                        step_rate_hz=args.step_rate_hz,
                        video_fps=args.video_fps,
                        use_planner=use_planner,
                    ): task_id
                    for task_id in task_ids
                }
                for fut in as_completed(futures):
                    task_id = futures[fut]
                    try:
                        entry = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        entry = {"task_id": task_id, "error": f"{exc!r}"}
                    ok = "video" in entry and "error" not in entry
                    print(f"  [{'OK ' if ok else 'ERR'}] {task_id:<38} "
                          f"steps={entry.get('num_steps', '-'):<4} "
                          f"done={entry.get('task_done', '-')} "
                          f"frames={entry.get('video_frames', '-')} "
                          f"{entry.get('error', '')}")
                    results.append(entry)

        # 4. 汇总（skip-existing 续录时与旧汇总合并，保证覆盖全部任务）
        summary_path = Path(spool_dir) / "demo_videos.json"
        old_videos: dict[str, dict] = {}
        if args.skip_existing and summary_path.exists():
            try:
                old = json.loads(summary_path.read_text(encoding="utf-8"))
                old_videos = {e["task_id"]: e for e in old.get("videos", [])}
            except Exception:  # noqa: BLE001
                old_videos = {}
        for e in results:
            old_videos[e["task_id"]] = e
        merged = sorted(old_videos.values(), key=lambda e: e["task_id"])
        summary = {
            "data_dir": spool_dir,
            "layout": "<data_dir>/<namespace>/<short_id>/demo.mp4",
            "steps": args.steps,
            "block_pixel_size": args.block_pixel_size,
            "step_rate_hz": args.step_rate_hz,
            "video_fps": args.video_fps,
            "seeds": seeds,
            "count": len(merged),
            "ok": sum(1 for e in merged if "video" in e and "error" not in e),
            "failed": [e["task_id"] for e in merged if "error" in e],
            "videos": merged,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[summary] ok={summary['ok']}/{summary['count']} "
              f"failed={summary['failed']}")
        print(f"[summary] 写入 {summary_path}")
    finally:
        if started and not args.keep_server:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[server] uvicorn 已关闭")


if __name__ == "__main__":
    main()
