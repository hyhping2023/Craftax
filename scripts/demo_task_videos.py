"""为每个已注册任务生成一个 demo 视频（CFR MP4）。

流程：
1. 确保 uvicorn service 在 127.0.0.1:8321 可访问（不可访问则自动拉起子进程）；
2. 可选 warmup：编译 env step + pixel renderer，并生成目标分辨率的纹理（一次）；
3. 遍历任务（--tasks 指定子集，默认全部已注册任务）：
   - POST /v1/sessions 创建带录制的会话（max_timesteps=--steps；--until-done 时由任务终止）
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
    conda run -n craftax python scripts/demo_task_videos.py --tasks native.reach_floor_3 --until-done
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
from craftax.contracts import DEFAULT_ENERGY_RATE, default_data_dir  # noqa: E402

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


def get_floor_map(
    sid: str, floor: int | None = None, window_size: int | None = 80
) -> dict | None:
    """获取规划窗口；窗口大小由调用方按路径需求决定。

    ``map_origin``/``player_global_position`` 记录绝对坐标；默认返回以玩家为中心
    的 80×80（5×5 chunk）窗口；请求其他楼层时同样由 chunk store 拼接。
    """
    query = []
    if floor is not None:
        query.append(f"floor={floor}")
    if window_size is not None:
        query.append(f"window_size={int(window_size)}")
    q = "?" + "&".join(query) if query else ""
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
        # GET /map?window_size=N returns positions relative to that cropped
        # window, while state_summary keeps the fixed 48x48 JAX coordinates.
        # Planning with the latter makes the agent target the wrong cells and
        # commonly degenerates into repeated DO actions at episode start.
        planning_summary = dict(summary)
        # The terrain window is intentionally cached for a few steps, but the
        # player moves every step.  Rebase the cached local/global player
        # coordinates from the latest fixed-window summary so a locked route
        # consumes one waypoint per actual movement instead of repeatedly
        # planning from the stale fetch position.
        current_raw = summary.get("player_position")
        anchor_raw = map_payload.get("_summary_anchor_position")
        if current_raw is not None and anchor_raw is not None:
            delta = [
                int(current_raw[0]) - int(anchor_raw[0]),
                int(current_raw[1]) - int(anchor_raw[1]),
            ]
            planning_local = [
                int(map_payload["player_position"][0]) + delta[0],
                int(map_payload["player_position"][1]) + delta[1],
            ]
            planning_global = [
                int(map_payload["player_global_position"][0]) + delta[0],
                int(map_payload["player_global_position"][1]) + delta[1],
            ]
        else:
            planning_local = list(map_payload["player_position"])
            planning_global = map_payload.get("player_global_position")
        planning_summary["player_position"] = planning_local
        # The terrain window is cached, but facing direction is part of the
        # live summary and may change on every action.
        planning_summary["player_direction"] = summary.get(
            "player_direction", map_payload.get("player_direction", 0)
        )
        planning_payload = dict(map_payload)
        planning_payload["player_position"] = planning_local
        if planning_global is not None:
            planning_summary["player_global_position"] = list(planning_global)
            planning_payload["player_global_position"] = list(planning_global)
        return skill_chain.next_action(planning_payload, planning_summary)

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
    """是否处于生存临界状态（能量/食物/水危险）。"""
    if summary is None:
        return False
    energy = summary.get("energy", 9.0)
    food = summary.get("food", 9.0)
    drink = summary.get("drink", 9.0)
    is_sleeping = summary.get("is_sleeping", False)
    return (energy < 2 and not is_sleeping) or food < 1 or drink < 1


def select_demo_action(
    summary: dict | None,
    planned: int | None,
    *,
    use_planner: bool,
    rng: random.Random,
) -> int:
    """选择 demo 要发送的动作。

    规划器的补给/战斗决策必须优先于 demo 层的生存兜底。尤其不能因为
    ``food``/``drink`` 进入危险阈值，就把规划器返回的动作改写成 DO；这会
    让玩家在水源旁反复交互，永远没有机会走开或战斗。只有规划器本身暂时
    没有动作时，危险状态才使用移动作为短暂 fallback，下一步重新规划。
    无规划器模式保留原来的随机生存策略。
    """
    if planned is not None:
        return planned
    if use_planner and _survival_critical(summary):
        return rng.choice(MOVE_IDS)
    return choose_action(summary, rng)


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
    #    seed 透传 → 执行器可用 WorldFacts（跨层矿石/梯子事实）；
    #    floor_map_provider → GET /map?floor=N，让执行器在下楼前看到目标层全图。
    #    provider 用后绑定的 sid_holder：执行器要在建会话之前构造（步数估算）。
    skill_chain = None
    sid_holder: dict[str, str] = {}
    if use_planner:
        from craftax.planner.executor import SkillChainExecutor  # 惰性加载，避免拉起服务前加载 JAX

        def _floor_map_provider(floor: int) -> dict | None:
            sid_now = sid_holder.get("sid")
            if not sid_now:
                return None
            return get_floor_map(sid_now, floor)

        skill_chain = SkillChainExecutor(
            task_id, seed=seed, energy_rate=DEFAULT_ENERGY_RATE,
            floor_map_provider=_floor_map_provider
        )
    until_done = steps < 0
    if not until_done and steps <= 0 and skill_chain is not None:
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
            # None 使用服务端的长程默认上限；客户端在任务完成/死亡时主动
            # 封存。--until-done 不再因为一个人为 demo 步数提前截断。
            "max_timesteps": None if until_done else steps,
            "god_mode": False,
        },
    )
    if not isinstance(resp, dict) or "session_id" not in resp:
        entry["error"] = f"create_session failed: {resp}"
        return entry
    sid = resp["session_id"]
    sid_holder["sid"] = sid  # 绑定跨层地图 provider
    s = resp.get("state_summary") or {}
    entry["instruction"] = s.get("instruction", "")
    entry["task_id"] = s.get("task_id", task_id)
    entry["namespace"] = namespace
    entry["short_id"] = short_id
    entry["max_steps"] = None if until_done else steps
    entry["until_done"] = until_done

    # 2. 策略 step 直到 episode 结束
    t = 0
    summary = s
    map_payload = None  # planner 用：懒加载全图
    # The planner's world model is currently the authoritative 48x48 active
    # Craftax grid.  Keep the demo on that exact coordinate frame; cropped
    # windows are useful for clients, but can hide a ladder/resource target
    # outside the crop and make long locked routes brittle.
    planner_window_size = None
    map_fetch_failures = 0
    stalled = 0  # 连续无进展步数（位置/成就均未变化）
    last_pos = None
    last_achievements = set(summary.get("achievements", []))
    try:
        while until_done or t < steps:
            # 任务已完成 → 提前结束 episode（不硬跑满 steps）
            if skill_chain is not None and skill_chain.is_done(summary):
                break
            # 规划器自身包含补水、进食、撤退和战斗优先级；启用规划器时即使
            # 已经进入危险补给线，也不能用盲目的 DO 覆盖它。旧逻辑在水边
            # 直接反复发送 DO，直到食物耗尽死亡。无规划器模式才保留随机
            # demo 的生存兜底。
            if use_planner:
                # 每步拉一次权威地图；规划器需要最新的怪物坐标，尤其是
                # 地下层的远程怪会在三步缓存窗口内改变战斗风险。
                if map_payload is None or t % 1 == 0:
                    map_payload = get_floor_map(
                        sid, summary.get("floor"), window_size=planner_window_size
                    )
                    if map_payload is not None and summary.get("player_position") is not None:
                        map_payload["_summary_anchor_position"] = list(summary["player_position"])
                    if map_payload is None:
                        map_fetch_failures += 1
                planned = planner_action(map_payload, summary, task_id, skill_chain)
                # 80×80 通常足够，但目标可能在窗口外或被局部障碍隔开。
                # 当前窗口没有可执行动作时扩大请求范围，并保留扩大后的
                # 窗口供后续子目标继续使用；服务端会按 chunk 持久化拼接。
                if (
                    planned is None
                    and map_payload is not None
                    and planner_window_size is not None
                    and planner_window_size < 512
                ):
                    planner_window_size = min(512, planner_window_size * 2)
                    expanded = get_floor_map(
                        sid, summary.get("floor"), window_size=planner_window_size
                    )
                    if expanded is not None:
                        if summary.get("player_position") is not None:
                            expanded["_summary_anchor_position"] = list(
                                summary["player_position"]
                            )
                        map_payload = expanded
                        planned = planner_action(
                            map_payload, summary, task_id, skill_chain
                        )
            else:
                planned = None

            # 规划器动作（包括它决定的 DO）优先；只在规划暂时无结果时，
            # 危险补给状态才用移动打破原地僵持并等待下一轮重规划。
            action_id = select_demo_action(
                summary, planned, use_planner=use_planner, rng=rng
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

    # 4. 等待 sealed shard 并拼接完整视频
    shard_dir = wait_for_shard(task_spool, run_id, before)
    if shard_dir is None:
        entry["error"] = "shard not sealed within timeout"
        return entry
    entry["shard_dir"] = str(shard_dir)

    # 一个长 episode 会被 recorder 按 segment_timesteps 切成多个 video-*.mp4。
    # 必须按 recorder 的 episodes.parquet 顺序拼接全部 segment，不能只取最后一段，
    # 否则 demo 看起来像是在末尾突然开始/结束。
    videos = sorted(shard_dir.glob("video-*.mp4"))
    if not videos:
        entry["error"] = "no video in shard"
        return entry

    task_dir.mkdir(parents=True, exist_ok=True)
    dst = task_dir / "demo.mp4"
    try:
        from craftax.recording.continuous import concatenate_recording

        continuous = concatenate_recording(str(shard_dir), output=str(dst))
        entry["video"] = str(dst)
        entry["video_bytes"] = dst.stat().st_size
        entry["video_segments"] = int(continuous["num_segments"])
        import imageio.v3 as iio

        decoded = iio.imread(dst)
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
    parser.add_argument(
        "--until-done", action="store_true",
        help="不使用 demo 步数上限，持续录制到任务完成、死亡或服务端安全上限",
    )
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
    if args.until_done:
        # -1 是内部 sentinel；保留 --steps 0 的既有“按任务估算”语义。
        args.steps = -1
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
            "until_done": args.until_done,
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
