"""端到端集成验证：service → recording → dataset + GUI headless HTTP。

链路：
  FastAPI service（SessionActor）→ AsyncRecorder → ShardWriter → sealed shard
  → validators.validate_shard → dataset.reader.ShardReader → vla/wm_samples

约定：
- 使用 Craftax-Symbolic-v1（避免双像素渲染开销）+ god_mode=True +
  max_timesteps=10，保证 episode 确定性地以 truncated 结束，T=10。
- recorder 的 flush 通过 DELETE /v1/sessions/{sid} 触发（actor.close()）。
- CPU JAX 首次编译耗时数秒~数十秒；manager / client / http_server
  均为 module 级共享 fixture，进程内只编译一次。

全部用例可独立运行（fixture 按需创建）。
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import threading
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest
from fastapi.testclient import TestClient

from craftax.contracts import (
    EPISODES_PARQUET_FILENAME,
    FRAME_INDEX_PARQUET_FILENAME,
    SHARD_MANIFEST_FILENAME,
    TRANSITIONS_PARQUET_FILENAME,
    ZARR_DIRNAME,
    StateSummary,
)
from craftax.dataset.reader import ShardReader
from craftax.dataset.vla_windows import vla_samples
from craftax.dataset.world_model_windows import wm_samples
from craftax.gui.pygame_client import PygameGUI, decode_png_rgb
from craftax.recording.validators import validate_shard
from craftax.service.app import create_app
from craftax.service.session_manager import SessionManager

ENV_NAME = "Craftax-Symbolic-v1"
TASK = {"task_id": "native.survive", "version": "1.0.0", "params": {}}
RENDER = {"format": "png", "mode": "human"}
MAX_T = 10  # transition 数；state 数 = MAX_T + 1
STRIDE = 2  # step_rate_hz=20 / video_fps=10

# 期望的视频帧：timestep 0（初始）、2/4/6/8（常规采样）、10（terminal）
EXPECTED_FRAME_TIMESTEPS = [0, 2, 4, 6, 8, 10]


@pytest.fixture(scope="module")
def manager() -> SessionManager:
    return SessionManager()


@pytest.fixture(scope="module")
def client(manager: SessionManager):
    app = create_app(manager=manager)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def http_server(manager: SessionManager):
    """真实 uvicorn 服务（随机空闲端口，后台线程）。"""
    import uvicorn

    # 预热：先编译 reset/step/renderer，避免首个 HTTP 请求内触发 JAX 编译。
    manager.warmup()

    port = _free_port()
    app = create_app(manager=manager)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn 启动超时"
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.should_exit = True
    thread.join(timeout=30)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _find_shards(spool_dir: Path) -> list[Path]:
    return sorted(p.parent for p in spool_dir.rglob(SHARD_MANIFEST_FILENAME))


def _create_recording_session(client: TestClient, spool_dir: Path, seed: int = 1234) -> dict:
    body = {
        "env_name": ENV_NAME,
        "seed": seed,
        "task": TASK,
        "render": RENDER,
        "recording": {
            "enabled": True,
            "dataset_run_id": "integration-test",
            "frame_sample": {"step_rate_hz": 20, "video_fps": 10},
            "gold_frames": True,
            "spool_dir": str(spool_dir),
        },
        "max_timesteps": MAX_T,
        "god_mode": True,
    }
    resp = client.post("/v1/sessions", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _step_until_done(
    client: TestClient, sid: str, *, action: dict, max_steps: int
) -> list[dict]:
    snaps: list[dict] = []
    for _ in range(max_steps):
        resp = client.post(f"/v1/sessions/{sid}/step", json={"action": action})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        snaps.append(data)
        if data["terminated"] or data["truncated"]:
            return snaps
    raise AssertionError(f"episode 未在 {max_steps} 步内结束")


# ---------------------------------------------------------------------------
# B. service → recording → dataset 端到端
# ---------------------------------------------------------------------------


def test_service_recording_dataset_e2e(client: TestClient, tmp_path_factory):
    """整条录制链路：HTTP step → recorder → sealed shard → dataset 读取。"""
    spool = tmp_path_factory.mktemp("e2e-spool")

    # 1) 创建会话（revision 0 / frame 0），录制开启
    created = _create_recording_session(client, spool, seed=1234)
    sid = created["session_id"]
    try:
        assert created["revision"] == 0
        assert created["timestep"] == 0
        assert created["state_summary"] is not None
        assert created["state_summary"]["timestep"] == 0
        assert created["action"] is None

        # 2) 连续 step 直到 truncated（T=10）
        snaps = _step_until_done(
            client, sid, action={"id": 0, "name": "NOOP"}, max_steps=MAX_T + 1
        )
        assert len(snaps) == MAX_T
        last = snaps[-1]
        assert last["terminated"] is False
        assert last["truncated"] is True
        assert last["timestep"] == MAX_T

        # step 响应的契约：
        #   state_summary 键名 == contracts.StateSummary 字段
        #   action 为嵌套 {"requested", "applied"}
        expected_keys = {f.name for f in dataclasses.fields(StateSummary)}
        for snap in snaps:
            assert snap["state_summary"] is not None
            assert set(snap["state_summary"].keys()) == expected_keys
            assert snap["action"] == {
                "requested": {"id": 0, "name": "NOOP"},
                "applied": {"id": 0, "name": "NOOP"},
            }
            assert snap["revision"] == snap["state_summary"]["timestep"]
        assert snaps[0]["timestep"] == 1
        assert snaps[-2]["truncated"] is False  # 只有最后一步结束

        # 保留 reset 帧 PNG（revision 0），供与 shard 中的帧对比
        resp0 = client.get(f"/v1/sessions/{sid}/frames/0")
        assert resp0.status_code == 200
        reset_png = resp0.content

    finally:
        # 3) DELETE 触发 recorder.close() → shard finalize
        resp = client.delete(f"/v1/sessions/{sid}")
        assert resp.status_code == 204

    # 4) spool 产出完整 shard
    shards = _find_shards(spool)
    assert len(shards) == 1, f"期望 1 个 shard，得到 {shards}"
    shard_dir = shards[0]
    assert (shard_dir / ZARR_DIRNAME).is_dir()
    for name in (
        EPISODES_PARQUET_FILENAME,
        TRANSITIONS_PARQUET_FILENAME,
        FRAME_INDEX_PARQUET_FILENAME,
    ):
        assert (shard_dir / name).is_file()
    videos = sorted(shard_dir.glob("video-*.mp4"))
    assert len(videos) == 1

    # 5) validate_shard 通过
    ok, errors = validate_shard(str(shard_dir))
    assert ok, f"validate_shard 失败: {errors}"

    # 6) ShardReader 读取
    reader = ShardReader(shard_dir)
    episodes = list(reader.episodes())
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.num_states == MAX_T + 1
    assert ep.num_transitions == MAX_T
    assert ep.num_frames == len(EXPECTED_FRAME_TIMESTEPS)
    assert ep.terminated is False and ep.truncated is True

    # ---- 关键不变量 ----
    # len(state) == len(action) + 1
    n_states = int(reader.zarr_root["state_timesteps"].shape[0])
    n_actions = int(reader.zarr_root["actions"].shape[0])
    assert n_states == n_actions + 1
    assert ep.num_states == ep.num_transitions + 1

    # frame_index：常规帧 timestep % R == 0，terminal 帧除外
    rows = ep.frame_rows()
    assert [r["timestep"] for r in rows] == EXPECTED_FRAME_TIMESTEPS
    regular = [r for r in rows if not r["is_initial_frame"] and not r["is_terminal_frame"]]
    assert regular, "缺少常规采样帧"
    assert all(r["timestep"] % STRIDE == 0 for r in regular)
    initial = [r for r in rows if r["is_initial_frame"]]
    assert len(initial) == 1 and initial[0]["timestep"] == 0
    terminal = [r for r in rows if r["is_terminal_frame"]]
    assert len(terminal) == 1 and terminal[0]["timestep"] == MAX_T

    # state_index ∈ [state_start, state_end]
    for r in rows:
        assert ep.state_start <= r["state_index"] <= ep.state_end

    # 帧解码数与 frame_index 行数一致
    frames = list(ep.frames())
    assert len(frames) == len(rows)
    for f, r in zip(frames, rows):
        assert f.frame_index == r["frame_index"]
        assert f.timestep == r["timestep"]
        assert f.rgb.ndim == 3 and f.rgb.dtype == np.uint8

    # frame_at_timestep(0) 与 reset 帧对应
    gold_dir = shard_dir / "gold-frames" / ep.episode_id
    gold_pngs = sorted(gold_dir.glob("*.png"))
    assert len(gold_pngs) == len(EXPECTED_FRAME_TIMESTEPS)
    gold0_path = gold_dir / "frame-0000-t0.png"
    assert gold0_path.is_file()
    gold0_rgb = decode_png_rgb(gold0_path.read_bytes())
    mp40_rgb = ep.frame_at_timestep(0)
    assert mp40_rgb is not None and gold0_rgb.shape == mp40_rgb.shape
    # API PNG（无损）与 gold frame（无损）应像素一致
    assert np.array_equal(decode_png_rgb(reset_png), gold0_rgb)
    # MP4（H.264 有损）与 gold 帧允许小容差
    diff = np.abs(gold0_rgb.astype(int) - mp40_rgb.astype(int))
    assert diff.mean() < 8.0, f"MP4 frame0 与 gold frame0 偏差过大: mean={diff.mean():.2f}"

    # 状态读取：state_at_timestep，含嵌套路径（inventory/wood）
    s0 = ep.state_at_timestep(0)
    assert s0 is not None
    assert "player_position" in s0
    assert "inventory/wood" in s0  # 修复：reader 递归枚举嵌套 state 数组
    assert "achievements" in s0
    sT = ep.state_at_timestep(MAX_T)
    assert sT is not None
    assert ep.state_at_timestep(MAX_T + 1) is None

    # transitions 连续性
    assert [r["timestep"] for r in ep.transition_rows()] == list(range(MAX_T))
    assert ep.action_at_timestep(0) == (0, "NOOP")
    assert ep.action_at_timestep(MAX_T) is None  # 终局步无 action

    # ---- vla_samples ----
    vla = list(vla_samples(reader, window_len=4))
    assert len(vla) == 2, f"期望 2 个 VLA 样本（含终局帧的窗口被跳过），得到 {len(vla)}"
    for s in vla:
        assert s["episode_id"] == ep.episode_id
        assert s["task_id"] == "native.survive"
        assert s["instruction"].strip()
        assert len(s["frames"]) == 4
        assert len(s["actions"]) == 4
        assert len(s["timesteps"]) == 4
        assert all(t % STRIDE == 0 for t in s["timesteps"])
        for t, a in zip(s["timesteps"], s["actions"]):
            assert ep.action_at_timestep(t) == (a.id, a.name)
        assert all(f.shape[-1] == 3 for f in s["frames"])

    # ---- wm_samples ----
    wm = list(wm_samples(reader, window_len=4, require_frames=True))
    assert len(wm) == 4, f"期望 4 个 WM 样本，得到 {len(wm)}"
    for s in wm:
        assert len(s["states"]) == 5
        assert len(s["actions"]) == 4
        assert len(s["next_states"]) == 4
        assert len(s["events"]) == 4
        assert len(s["rewards"]) == 4
        assert s["timesteps"] == list(range(s["timesteps"][0], s["timesteps"][0] + 5))
        assert len(s["frames"]) == len(s["frame_timesteps"])
        for ft in s["frame_timesteps"]:
            assert ft % STRIDE == 0
        assert "inventory/wood" in s["states"][0]

    # manifest 可读且记录数一致
    manifest = json.loads((shard_dir / SHARD_MANIFEST_FILENAME).read_text())
    assert manifest["counts"]["num_episodes"] == 1
    assert manifest["counts"]["num_transitions"] == MAX_T
    assert manifest["counts"]["num_states"] == MAX_T + 1
    assert manifest["counts"]["num_frames"] == len(EXPECTED_FRAME_TIMESTEPS)
    assert manifest["counts"]["num_videos"] == 1
    assert "state/inventory/wood" in manifest["arrays"]


# ---------------------------------------------------------------------------
# C. GUI headless 连接真实 HTTP service
# ---------------------------------------------------------------------------


def test_gui_headless_connects_real_http(http_server: str):
    gui = PygameGUI.connect_http(
        http_server,
        env_name=ENV_NAME,
        fps=100,
        recording={"enabled": False},
    )
    try:
        # create_session 已完成（revision 0 / frame 0）
        assert gui.driver.revision == 0
        assert gui._snapshot is not None and gui._snapshot.revision == 0
        assert gui._frame_rgb is not None and gui._frame_rgb.shape[-1] == 3
        assert gui.driver.session_id is not None

        # K_d → RIGHT：post 事件并单步执行
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        steps = gui.run(max_steps=1)
        assert steps == 1
        assert gui.driver.revision == 1
        assert gui._snapshot is not None and gui._snapshot.timestep == 1
        # 渲染路径（PNG 解码 → 场景画布）无异常
        assert gui._frame_rgb is not None and gui._frame_rgb.shape == (130, 110, 3)

        # GET /state 轮询端点
        snap = gui.driver.get_snapshot()
        assert snap.revision == 1
        assert snap.summary is not None and snap.summary.timestep == 1
    finally:
        sid = gui.driver.session_id
        if sid:
            resp = client_delete(http_server, sid)
            assert resp == 204, f"DELETE 会话失败: {resp}"


def client_delete(base_url: str, sid: str) -> int:
    import urllib.request

    req = urllib.request.Request(f"{base_url}/v1/sessions/{sid}", method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status
