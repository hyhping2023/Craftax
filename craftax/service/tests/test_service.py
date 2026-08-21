"""service 模块端到端测试。

使用 Craftax-Symbolic-v1（避免像素环境双渲染），recording 关闭，
max_timesteps 调小以在有限步内结束 episode。CPU JAX 首次编译需要
数十秒，因此 TestClient / 环境实例为 session 级共享，仅编译一次。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from craftax.service.app import create_app

ENV_NAME = "Craftax-Symbolic-v1"
TASK = {"task_id": "native.survive", "version": "1.0.0", "params": {}}
RENDER = {"format": "png", "mode": "human"}
RECORDING = {"enabled": False}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="session")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def session_ids() -> List[str]:
    ids: List[str] = []
    yield ids


@pytest.fixture(autouse=True)
def _cleanup_sessions(client: TestClient, session_ids: List[str]) -> None:
    yield
    for sid in session_ids:
        try:
            client.delete(f"/v1/sessions/{sid}")
        except Exception:  # noqa: BLE001 - 清理失败不影响其余断言
            pass


def create_session(
    client: TestClient,
    session_ids: List[str],
    *,
    seed: Optional[int] = None,
    max_timesteps: int = 6,
    task: Optional[dict] = None,
    **overrides,
) -> dict:
    body = {
        "env_name": ENV_NAME,
        "seed": seed,
        "max_timesteps": max_timesteps,
        "task": task or TASK,
        "render": RENDER,
        "recording": RECORDING,
        **overrides,
    }
    resp = client.post("/v1/sessions", json=body)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    session_ids.append(data["session_id"])
    return data


def step(client: TestClient, sid: str, action, **overrides):
    body: Dict = {"action": action, **overrides}
    return client.post(f"/v1/sessions/{sid}/step", json=body)


# ---------------------------------------------------------------------------
# 创建会话
# ---------------------------------------------------------------------------


def test_create_session_returns_revision_zero(client, session_ids):
    data = create_session(client, session_ids, seed=42, max_timesteps=20)
    assert data["session_id"].startswith("sess_")
    assert data["revision"] == 0
    assert data["timestep"] == 0
    assert data["terminated"] is False
    assert data["truncated"] is False
    assert data["action"] is None
    assert data["state_summary"] is not None
    assert data["state_summary"]["timestep"] == 0
    assert data["state_summary"]["health"] > 0
    assert isinstance(data["state_summary"]["inventory"], dict)
    assert data["frame"] is not None
    assert data["frame"]["revision"] == 0
    assert data["frame"]["url"] == f"/v1/sessions/{data['session_id']}/frames/0"
    assert data["frame"]["content_type"] == "image/png"
    assert data["frame"]["width"] > 0 and data["frame"]["height"] > 0


def test_map_window_exposes_absolute_coordinates(client, session_ids):
    data = create_session(client, session_ids, seed=42, max_timesteps=20)
    sid = data["session_id"]
    response = client.get(f"/v1/sessions/{sid}/map", params={"window_size": 16})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["map"]) == 16
    assert len(payload["map"][0]) == 16
    assert payload["window_size"] == 16
    assert payload["chunk_size"] == 16
    assert payload["world_mode"] == "streamed_chunk_v1"
    assert payload["player_global_position"] == [
        payload["map_origin"][0] + payload["player_position"][0],
        payload["map_origin"][1] + payload["player_position"][1],
    ]
    wide = client.get(f"/v1/sessions/{sid}/map", params={"window_size": 80}).json()
    assert len(wide["map"]) == 80
    assert wide["player_global_position"] == [
        wide["map_origin"][0] + wide["player_position"][0],
        wide["map_origin"][1] + wide["player_position"][1],
    ]
    # 规划器可在当前范围不够时继续扩大请求，而不是被旧的 96 格上限截断。
    expanded = client.get(f"/v1/sessions/{sid}/map", params={"window_size": 128}).json()
    assert len(expanded["map"]) == 128
    assert len(expanded["map"][0]) == 128
    assert expanded["player_global_position"] == [
        expanded["map_origin"][0] + expanded["player_position"][0],
        expanded["map_origin"][1] + expanded["player_position"][1],
    ]
    assert len(payload["map_origin"]) == 2


def test_edge_refresh_preserves_absolute_coordinates_and_blocks(client, session_ids):
    """跨活动窗口后，重叠格仍指向同一绝对方块且原点随窗口同步。"""
    import jax.numpy as jnp

    data = create_session(client, session_ids, seed=42, max_timesteps=20)
    sid = data["session_id"]
    actor = client.app.state.manager.get(sid)
    # 将玩家置于右边界；新 chunk 的边界格由生成契约保证为可通行 PATH。
    with actor._lock:
        actor._state = actor._state.replace(
            player_position=jnp.asarray([24, 47], dtype=jnp.int32)
        )
    before = client.get(f"/v1/sessions/{sid}/map").json()
    # 绝对坐标 (24, 40) 在换窗前为局部 (24, 40)，换窗后为 (24, 24)。
    stable_block = before["map"][24][40]

    moved = step(client, sid, 2)  # RIGHT
    assert moved.status_code == 200, moved.text
    assert moved.json()["info"]["world_expanded"] is True

    after = client.get(f"/v1/sessions/{sid}/map").json()
    assert after["world_origin"] == [0, 16]
    assert after["player_global_position"] == [24, 48]
    assert after["player_global_position"] == [
        after["map_origin"][0] + after["player_position"][0],
        after["map_origin"][1] + after["player_position"][1],
    ]
    assert after["map"][24][24] == stable_block


# ---------------------------------------------------------------------------
# step 与 revision
# ---------------------------------------------------------------------------


def test_step_increments_revision_and_timestep(client, session_ids):
    data = create_session(client, session_ids, seed=1)
    sid = data["session_id"]

    r1 = step(client, sid, {"id": 0, "name": "NOOP"})
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["revision"] == 1
    assert d1["timestep"] == 1
    assert d1["action"] == {
        "requested": {"id": 0, "name": "NOOP"},
        "applied": {"id": 0, "name": "NOOP"},
    }

    r2 = step(client, sid, 1)  # int action id 也可以
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["revision"] == 2
    assert d2["timestep"] == 2
    assert d2["action"] == {
        "requested": {"id": 1, "name": "LEFT"},
        "applied": {"id": 1, "name": "LEFT"},
    }

    # snapshot 按 revision 查询
    rs = client.get(f"/v1/sessions/{sid}/snapshot", params={"revision": 1})
    assert rs.status_code == 200
    assert rs.json()["revision"] == 1

    # 当前 snapshot
    rcur = client.get(f"/v1/sessions/{sid}/snapshot")
    assert rcur.status_code == 200
    assert rcur.json()["revision"] == 2


def test_state_endpoint_matches_snapshot_contract(client, session_ids):
    """GUI 契约：GET /v1/sessions/{sid}/state?revision=N&detail=summary。"""
    data = create_session(client, session_ids, seed=1)
    sid = data["session_id"]

    step(client, sid, 0, command_id="cmd-1")
    step(client, sid, 1, command_id="cmd-2")

    # 指定 revision
    rs = client.get(
        f"/v1/sessions/{sid}/state", params={"revision": 1, "detail": "summary"}
    )
    assert rs.status_code == 200, rs.text
    body = rs.json()
    assert body["revision"] == 1
    assert body["state_summary"] is not None
    assert body["state_summary"]["timestep"] == 1
    # state 与 snapshot 端点同构
    ref = client.get(f"/v1/sessions/{sid}/snapshot", params={"revision": 1}).json()
    assert body == ref

    # 当前 revision（不传 revision）
    rcur = client.get(f"/v1/sessions/{sid}/state")
    assert rcur.status_code == 200
    assert rcur.json()["revision"] == 2

    # 不存在的 revision -> 404（GUI 据此退化到本地缓存）
    r404 = client.get(f"/v1/sessions/{sid}/state", params={"revision": 999})
    assert r404.status_code == 404
    assert r404.json()["error"] == "snapshot_not_found"


def test_step_accepts_return_field(client, session_ids):
    """GUI 契约：/step 请求体携带 return 字段必须被接受。"""
    data = create_session(client, session_ids, seed=1)
    sid = data["session_id"]

    body: Dict = {
        "action": {"id": 0, "name": "NOOP"},
        "return": {"frame": "reference", "observation": "summary"},
    }
    r = client.post(f"/v1/sessions/{sid}/step", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 1


def test_invalid_action_returns_400(client, session_ids):
    data = create_session(client, session_ids, seed=1)
    sid = data["session_id"]
    r = step(client, sid, 999)
    assert r.status_code == 400
    r2 = step(client, sid, {"id": 0, "name": "WRONG_NAME"})
    assert r2.status_code == 400


# ---------------------------------------------------------------------------
# command_id 幂等
# ---------------------------------------------------------------------------


def test_command_id_idempotent(client, session_ids):
    data = create_session(client, session_ids, seed=1)
    sid = data["session_id"]

    r1 = step(client, sid, 0, command_id="cmd-1")
    assert r1.status_code == 200, r1.text
    d1 = r1.json()

    # 相同 command_id（即使 action 不同）返回首次结果，不推进状态
    r2 = step(client, sid, 5, command_id="cmd-1")
    assert r2.status_code == 200, r2.text
    assert r2.json() == d1

    # 新 command_id 正常推进
    r3 = step(client, sid, 0, command_id="cmd-2")
    assert r3.status_code == 200, r3.text
    assert r3.json()["revision"] == d1["revision"] + 1


# ---------------------------------------------------------------------------
# expected_revision 乐观并发控制
# ---------------------------------------------------------------------------


def test_expected_revision_conflict_409(client, session_ids):
    data = create_session(client, session_ids, seed=1)
    sid = data["session_id"]

    r1 = step(client, sid, 0)
    assert r1.status_code == 200
    assert r1.json()["revision"] == 1

    # 旧 expected_revision -> 409，并返回最新 revision
    r2 = step(client, sid, 0, expected_revision=0)
    assert r2.status_code == 409
    body = r2.json()
    assert body["error"] == "revision_conflict"
    assert body["detail"]["current_revision"] == 1

    # 正确 expected_revision 可继续
    r3 = step(client, sid, 0, expected_revision=1)
    assert r3.status_code == 200
    assert r3.json()["revision"] == 2


# ---------------------------------------------------------------------------
# 终局行为
# ---------------------------------------------------------------------------


def test_step_after_terminated_returns_400(client, session_ids):
    data = create_session(client, session_ids, seed=1, max_timesteps=2)
    sid = data["session_id"]

    step(client, sid, 0)
    step(client, sid, 0)
    # 到达 max_timesteps -> truncated
    snap = client.get(f"/v1/sessions/{sid}/snapshot").json()
    assert snap["truncated"] is True

    r = step(client, sid, 0)
    assert r.status_code == 400
    assert r.json()["error"] == "session_terminated"


def test_reset_starts_new_episode(client, session_ids):
    data = create_session(client, session_ids, seed=1, max_timesteps=2)
    sid = data["session_id"]

    step(client, sid, 0)
    step(client, sid, 0)
    assert client.get(f"/v1/sessions/{sid}/snapshot").json()["truncated"] is True

    r = client.post(f"/v1/sessions/{sid}/reset", json={"seed": 5})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["truncated"] is False
    assert d["terminated"] is False
    assert d["timestep"] == 0
    assert d["revision"] > 2

    r2 = step(client, sid, 0)
    assert r2.status_code == 200, r2.text
    assert r2.json()["revision"] == d["revision"] + 1
    assert r2.json()["timestep"] == 1


# ---------------------------------------------------------------------------
# 帧
# ---------------------------------------------------------------------------


def test_get_frame_returns_png(client, session_ids):
    data = create_session(client, session_ids, seed=2)
    sid = data["session_id"]

    r0 = client.get(f"/v1/sessions/{sid}/frames/0")
    assert r0.status_code == 200
    assert r0.headers["content-type"] == "image/png"
    assert r0.content[:8] == PNG_MAGIC

    step(client, sid, 0)
    r1 = client.get(f"/v1/sessions/{sid}/frames/1")
    assert r1.status_code == 200
    assert r1.content[:8] == PNG_MAGIC

    rn = client.get(f"/v1/sessions/{sid}/frames/999")
    assert rn.status_code == 404


# ---------------------------------------------------------------------------
# 会话不存在
# ---------------------------------------------------------------------------


def test_unknown_session_returns_410(client):
    r = client.get("/v1/sessions/sess_nope/snapshot")
    assert r.status_code == 410
    assert r.json()["error"] == "session_not_found"


# ---------------------------------------------------------------------------
# 确定性 replay
# ---------------------------------------------------------------------------


def _summary_signature(data: dict) -> dict:
    s = data["state_summary"]
    return {
        "timestep": s["timestep"],
        "health": s["health"],
        "food": s["food"],
        "drink": s["drink"],
        "energy": s["energy"],
        "mana": s["mana"],
        "floor": s["floor"],
        "xp": s["xp"],
        "dexterity": s["dexterity"],
        "strength": s["strength"],
        "intelligence": s["intelligence"],
        "inventory": s["inventory"],
        "achievements": s["achievements"],
    }


def test_same_seed_and_actions_replay_identical(client, session_ids):
    actions = [0, 0, 1, 2, 3, 4, 0, 1]

    a = create_session(client, session_ids, seed=777, max_timesteps=20)
    b = create_session(client, session_ids, seed=777, max_timesteps=20)
    sid_a, sid_b = a["session_id"], b["session_id"]

    sigs_a: List[dict] = []
    sigs_b: List[dict] = []
    for act in actions:
        ra = step(client, sid_a, act)
        rb = step(client, sid_b, act)
        assert ra.status_code == 200, ra.text
        assert rb.status_code == 200, rb.text
        sigs_a.append(_summary_signature(ra.json()))
        sigs_b.append(_summary_signature(rb.json()))

    assert sigs_a == sigs_b
