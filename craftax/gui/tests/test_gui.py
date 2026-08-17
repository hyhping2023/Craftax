"""gui 模块测试。强制 SDL dummy 视频驱动，不涉及真实网络。"""

from __future__ import annotations

import json
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

from io import BytesIO

import numpy as np
import pygame
import pytest

from craftax.contracts import ActionSpec, Snapshot, StateSummary
from craftax.gui.controls import (
    KEY_MAPPING,
    ControllerMode,
    action_names,
    key_to_action,
    parse_controller,
)
from craftax.gui.pygame_client import (
    HttpSessionDriver,
    PygameGUI,
    decode_png_rgb,
)
from craftax.gui.view_models import (
    DebugPanel,
    InventoryPanel,
    StatusPanel,
    TaskPanel,
    panels_from_snapshot,
)


def make_png_bytes(size: int = 8, value: int = 42) -> bytes:
    from PIL import Image

    img = Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeDriver:
    """合成 SessionDriver：reset 返回固定 32x32 帧，step 递增 revision 并生成新帧。"""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._revision = 0
        self._timestep = 0
        self.reset_count = 0
        self.step_count = 0

    def _make_frame(self) -> np.ndarray:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = (self._revision * 7) % 256
        frame[0, 0, :] = 255
        return frame

    def _make_summary(self) -> StateSummary:
        return StateSummary(
            timestep=self._timestep,
            health=100.0,
            food=80.0,
            drink=70.0,
            energy=90.0,
            mana=10.0,
            floor=1,
            xp=5,
            dexterity=1,
            strength=2,
            intelligence=3,
            is_sleeping=False,
            is_resting=False,
            inventory={
                "wood": 4,
                "stone": 2,
                "armour": np.array([1, 0, 0, 0, 0], dtype=np.int32),
            },
            achievements=["COLLECT_WOOD"],
            task_progress=0.5,
            task_done=False,
            instruction="Collect 10 wood",
        )

    def reset(self, seed: int | None = None) -> Snapshot:
        if seed is not None:
            self.seed = seed
        self._revision = 0
        self._timestep = 0
        self.reset_count += 1
        return Snapshot(
            session_id="fake-session",
            revision=self._revision,
            timestep=self._timestep,
            summary=self._make_summary(),
            frame_rgb=self._make_frame(),
            frame_revision=self._revision,
            info={"seed": self.seed, "recording": "off"},
        )

    def step(
        self,
        action: ActionSpec,
        command_id: str | None = None,
        wait_frame: bool = True,
    ) -> Snapshot:
        self.step_count += 1
        self._revision += 1
        self._timestep += 1
        return Snapshot(
            session_id="fake-session",
            revision=self._revision,
            timestep=self._timestep,
            action=action,
            reward=1.0,
            summary=self._make_summary(),
            frame_rgb=self._make_frame(),
            frame_revision=self._revision,
            info={"seed": self.seed, "event_tokens": ["COLLECT_WOOD"]},
        )

    def get_snapshot(self, revision: int | None = None) -> Snapshot:
        return Snapshot(
            session_id="fake-session",
            revision=self._revision,
            timestep=self._timestep,
        )

    def get_frame_png(self, revision: int) -> bytes:
        return make_png_bytes()

    @property
    def revision(self) -> int:
        return self._revision


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


def test_key_mapping_roundtrip():
    for key, action in KEY_MAPPING.items():
        spec = key_to_action(key)
        assert spec is not None
        assert spec.id == action.value
        assert spec.name == action.name
    assert key_to_action(pygame.K_F1) is None
    assert key_to_action(-1) is None


def test_key_mapping_special_keys():
    assert key_to_action(pygame.K_q).name == "NOOP"
    assert key_to_action(pygame.K_TAB).name == "SLEEP"
    assert key_to_action(pygame.K_SPACE).name == "DO"


def test_action_names():
    from craftax.craftax.constants import Action

    names = action_names()
    assert len(names) == len(Action)
    assert "NOOP" in names
    assert "LEFT" in names
    assert len(set(names)) == len(names)


def test_controller_modes():
    assert ControllerMode.HUMAN.value == "human"
    assert ControllerMode.MODEL.value == "model"
    assert ControllerMode.REPLAY.value == "replay"
    assert parse_controller("model") is ControllerMode.MODEL
    assert parse_controller(" replay ") is ControllerMode.REPLAY
    with pytest.raises(ValueError):
        parse_controller("bogus")


# ---------------------------------------------------------------------------
# view models
# ---------------------------------------------------------------------------


def _sample_summary() -> StateSummary:
    return StateSummary(
        timestep=10,
        health=95.5,
        food=80.0,
        drink=70.0,
        energy=90.0,
        mana=5.0,
        floor=2,
        xp=100,
        dexterity=3,
        strength=4,
        intelligence=5,
        is_sleeping=True,
        is_resting=False,
        inventory={
            "wood": 3,
            "stone": 1,
            "potions": np.array([1, 0, 0, 0, 0], dtype=np.int32),
            "armour": [0, 1, 0, 0, 0],
        },
        achievements=["COLLECT_WOOD", "MAKE_PICKAXE"],
        task_progress=0.75,
        task_done=False,
        instruction="Chop trees",
    )


def test_status_panel_render():
    lines = StatusPanel.from_summary(_sample_summary()).render()
    assert lines and lines[0] == "[STATUS]"
    text = " ".join(lines)
    for token in (
        "health",
        "food",
        "drink",
        "energy",
        "mana",
        "floor",
        "xp",
        "dex",
        "sleep",
    ):
        assert token in text


def test_inventory_panel_render():
    lines = InventoryPanel.from_summary(_sample_summary()).render()
    text = " ".join(lines)
    assert lines[0] == "[INVENTORY]"
    assert "wood: 3" in text
    assert "stone: 1" in text
    assert "potions: [1, 0, 0, 0, 0]" in text


def test_inventory_panel_empty():
    lines = InventoryPanel(inventory={}).render()
    assert "(empty)" in " ".join(lines)


def test_task_panel_render():
    lines = TaskPanel.from_summary(_sample_summary(), ["WOOD", "STONE"]).render()
    text = " ".join(lines)
    assert "Chop trees" in text
    assert "75.0%" in text
    assert "WOOD" in text


def test_debug_panel_render():
    snap = Snapshot(
        session_id="sess-abc",
        revision=7,
        timestep=10,
        reward=2.0,
        summary=_sample_summary(),
        info={"seed": 1, "recording": "on"},
    )
    lines = DebugPanel.from_snapshot(snap).render()
    text = " ".join(lines)
    assert lines[0] == "[DEBUG]"
    assert "sess-abc" in text
    assert "seed:    1" in text
    assert "revision: 7" in text
    assert "record:  on" in text


def test_panels_from_snapshot():
    snap = Snapshot(
        session_id="s1",
        revision=0,
        timestep=0,
        summary=_sample_summary(),
        info={"event_tokens": ["X"], "seed": 9},
    )
    blocks = panels_from_snapshot(snap)
    assert len(blocks) == 4
    for block in blocks:
        assert isinstance(block, list) and block
        assert all(isinstance(line, str) for line in block)


def test_panels_from_snapshot_without_summary():
    snap = Snapshot(session_id="s1", revision=0, timestep=0, summary=None)
    blocks = panels_from_snapshot(snap)
    assert len(blocks) == 4
    assert all(block for block in blocks)


# ---------------------------------------------------------------------------
# pygame client
# ---------------------------------------------------------------------------


def test_decode_png_rgb():
    arr = decode_png_rgb(make_png_bytes(size=8, value=42))
    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0].tolist() == [42, 42, 42]


def test_gui_smoke():
    driver = FakeDriver()
    gui = PygameGUI(driver, fps=100, pixel_render_size=2)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    steps = gui.run(max_steps=100)
    assert steps == 2
    assert driver.step_count == 2
    assert driver.revision == 2
    assert gui._frame_rgb is not None
    assert gui._frame_rgb.shape == (32, 32, 3)


def test_gui_max_steps_bounds_loop():
    driver = FakeDriver()
    gui = PygameGUI(driver, fps=100)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
    steps = gui.run(max_steps=1)
    assert steps == 1
    assert driver.revision == 1


def test_gui_reset_key():
    driver = FakeDriver()
    gui = PygameGUI(driver, fps=100)
    driver.step(ActionSpec(1, "LEFT"))  # driver revision -> 1
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    gui.run(max_steps=10)
    # run() 初始会 reset 一次，随后 R 键再次 reset
    assert driver.revision == 0
    assert driver.reset_count >= 2


def test_gui_model_mode_ignores_action_keys():
    driver = FakeDriver()
    gui = PygameGUI(driver, fps=100)
    gui.set_controller("model")
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    steps = gui.run(max_steps=10)
    assert steps == 0
    assert driver.step_count == 0


def test_gui_controller_cycle():
    driver = FakeDriver()
    gui = PygameGUI(driver, fps=100)
    assert gui.controller is ControllerMode.HUMAN
    gui.cycle_controller()
    assert gui.controller is ControllerMode.MODEL
    gui.cycle_controller()
    assert gui.controller is ControllerMode.REPLAY
    gui.cycle_controller()
    assert gui.controller is ControllerMode.HUMAN


# ---------------------------------------------------------------------------
# remote HTTP driver（不触碰真实网络，monkeypatch _request）
# ---------------------------------------------------------------------------


def test_http_driver_contract(monkeypatch):
    driver = HttpSessionDriver("http://localhost:8000")
    captured = {}

    def fake_request(method: str, path: str, body):
        if method == "POST" and path == "/v1/sessions":
            captured["create"] = body
            return json.dumps(
                {
                    "session_id": "sess-1",
                    "revision": 0,
                    "state_summary": {
                        "timestep": 0,
                        "health": 100.0,
                        "inventory": {"wood": 3},
                    },
                    "frame": {"revision": 0, "url": "/v1/sessions/sess-1/frames/0"},
                }
            ).encode()
        if method == "GET" and path == "/v1/sessions/sess-1/frames/0":
            return make_png_bytes()
        if method == "POST" and path == "/v1/sessions/sess-1/step":
            captured["step"] = body
            return json.dumps(
                {
                    "session_id": "sess-1",
                    "revision": 1,
                    "reward": 0.5,
                    "terminated": False,
                    "truncated": False,
                    "action": {
                        "requested": {"id": 1, "name": "LEFT"},
                        "applied": {"id": 1, "name": "LEFT"},
                    },
                    "state_summary": {"timestep": 1, "health": 99.0},
                    "frame": {"revision": 1, "url": "/v1/sessions/sess-1/frames/1"},
                }
            ).encode()
        if method == "GET" and path == "/v1/sessions/sess-1/frames/1":
            return make_png_bytes()
        raise AssertionError(f"unexpected request: {method} {path} {body!r}")

    monkeypatch.setattr(driver, "_request", fake_request)

    snap = driver.create_session()
    assert snap.session_id == "sess-1"
    assert snap.revision == 0
    assert snap.summary.health == 100.0
    assert snap.summary.inventory["wood"] == 3
    assert snap.frame_png is not None
    assert snap.frame_revision == 0
    assert captured["create"]["env_name"] == "Craftax-Pixels-v1"

    snap2 = driver.step(ActionSpec(1, "LEFT"))
    assert snap2.revision == 1
    assert snap2.reward == 0.5
    assert snap2.summary.timestep == 1
    assert snap2.action is not None and snap2.action.id == 1
    assert snap2.frame_png is not None
    assert snap2.frame_revision == 1

    step_body = captured["step"]
    assert step_body["action"] == {"id": 1, "name": "LEFT"}
    assert step_body["expected_revision"] == 0
    assert step_body["command_id"]


def test_http_driver_reset(monkeypatch):
    driver = HttpSessionDriver("http://localhost:8000", session_id="sess-9")
    driver._revision = 5
    captured = {}

    def fake_request(method: str, path: str, body):
        if method == "POST" and path == "/v1/sessions/sess-9/reset":
            captured["reset"] = body
            return json.dumps(
                {
                    "session_id": "sess-9",
                    "revision": 0,
                    "state_summary": {"timestep": 0},
                }
            ).encode()
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(driver, "_request", fake_request)
    snap = driver.reset(seed=7)
    assert snap.revision == 0
    assert snap.summary.timestep == 0
    assert captured["reset"]["seed"] == 7
    assert captured["reset"]["expected_revision"] == 5
    assert driver.revision == 0


def test_http_driver_url_to_path():
    assert HttpSessionDriver._url_to_path("/a/b") == "/a/b"
    assert HttpSessionDriver._url_to_path("a/b") == "/a/b"
    assert (
        HttpSessionDriver._url_to_path("http://h:8000/v1/sessions/s/frames/1")
        == "/v1/sessions/s/frames/1"
    )
