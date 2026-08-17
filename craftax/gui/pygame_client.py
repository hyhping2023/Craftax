"""Pygame GUI 客户端（embedded / HTTP polling 两种模式）。

embedded：调用方直接传入实现 ``craftax.contracts.SessionDriver`` 的实例，
          场景画布使用 ``Snapshot.frame_rgb``（uint8 HWC，无编码开销）。
remote：  ``PygameGUI.connect_http(base_url)`` 用标准库 urllib 走 REST：
          ``POST /v1/sessions``、``/step``、``/reset``、``GET /v1/sessions/{id}/frames/{rev}``。
          帧以 PNG bytes（``Snapshot.frame_png``）返回，由 PIL 解码。

Pygame 仅是 frame 消费者，不持有环境状态。
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from craftax.contracts import ActionSpec, SessionDriver, Snapshot, StateSummary
from craftax.gui.controls import ControllerMode, key_to_action, parse_controller
from craftax.gui.view_models import panels_from_snapshot

CONTENT_TYPE_JSON = "application/json"
HTTP_TIMEOUT_SECONDS = 30


def decode_png_rgb(png_bytes: bytes) -> np.ndarray:
    """把 PNG bytes 解码为 uint8 HWC RGB 数组（惰性 import PIL）。"""
    from PIL import Image

    with Image.open(BytesIO(png_bytes)) as img:
        return np.asarray(img.convert("RGB"))


class HttpDriverError(RuntimeError):
    """HTTP 驱动错误；status 为 HTTP 状态码（网络层错误时为 None）。"""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class HttpSessionDriver:
    """REST polling 版 SessionDriver，仅依赖标准库 urllib。

    协议遵循 embodied_environment_plan.md 第 5 节草案。响应解析做防御性处理，
    以兼容服务端字段命名差异（见模块内 _parse_* 辅助函数）。
    """

    def __init__(
        self,
        base_url: str,
        env_name: str = "Craftax-Pixels-v1",
        seed: Optional[int] = None,
        recording: Optional[Dict] = None,
        session_id: Optional[str] = None,
        block_pixel_size: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.env_name = env_name
        self.seed = seed
        self.recording = recording or {}
        self.block_pixel_size = block_pixel_size
        self._session_id = session_id
        self._revision = -1
        self._last_snapshot: Optional[Snapshot] = None

    # -- SessionDriver 协议 -------------------------------------------------

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def create_session(self) -> Snapshot:
        """POST /v1/sessions 创建会话，返回 revision 0 的 Snapshot。"""
        body: Dict = {
            "env_name": self.env_name,
            "render": {
                "format": "png",
                "mode": "human",
                **(
                    {"block_pixel_size": self.block_pixel_size}
                    if self.block_pixel_size is not None
                    else {}
                ),
            },
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.recording:
            body["recording"] = self.recording
        resp = self._post_json("/v1/sessions", body)
        self._session_id = resp.get("session_id") or self._session_id
        if not self._session_id:
            raise HttpDriverError("POST /v1/sessions response missing session_id")
        snap = self._snapshot_from_json(resp)
        self._revision = snap.revision
        self._attach_frame(snap, resp)
        self._last_snapshot = snap
        return snap

    def reset(self, seed: Optional[int] = None) -> Snapshot:
        self._ensure_session()
        body: Dict = {
            "expected_revision": self._revision if self._revision >= 0 else None,
            "command_id": uuid.uuid4().hex,
        }
        if seed is not None:
            body["seed"] = seed
        resp = self._post_json(f"/v1/sessions/{self._session_id}/reset", body)
        snap = self._snapshot_from_json(resp)
        self._revision = snap.revision
        self._attach_frame(snap, resp)
        self._last_snapshot = snap
        return snap

    def step(
        self,
        action: ActionSpec,
        command_id: Optional[str] = None,
        wait_frame: bool = True,
    ) -> Snapshot:
        self._ensure_session()
        body: Dict = {
            "action": action.to_dict(),
            "expected_revision": self._revision if self._revision >= 0 else None,
            "command_id": command_id or uuid.uuid4().hex,
            "return": {"frame": "reference", "observation": "summary"},
        }
        resp = self._post_json(f"/v1/sessions/{self._session_id}/step", body)
        snap = self._snapshot_from_json(resp)
        self._revision = snap.revision
        if wait_frame:
            self._attach_frame(snap, resp)
        self._last_snapshot = snap
        return snap

    def get_snapshot(self, revision: Optional[int] = None) -> Snapshot:
        self._ensure_session()
        rev = self._revision if revision is None else revision
        path = f"/v1/sessions/{self._session_id}/state?revision={rev}&detail=summary"
        try:
            resp = self._get_json(path)
            snap = self._snapshot_from_json(resp)
            self._last_snapshot = snap
            return snap
        except HttpDriverError:
            if self._last_snapshot is not None:
                return self._last_snapshot
            raise

    def get_frame_png(self, revision: int) -> bytes:
        self._ensure_session()
        return self._get_bytes(f"/v1/sessions/{self._session_id}/frames/{revision}")

    # -- HTTP 原语 ----------------------------------------------------------

    def _ensure_session(self) -> None:
        if self._session_id is None:
            self.create_session()

    def _request(self, method: str, path: str, body: Optional[Dict]) -> bytes:
        url = self.base_url + path
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = CONTENT_TYPE_JSON
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")
            raise HttpDriverError(
                f"HTTP {err.code} from {method} {path}: {detail}", status=err.code
            )
        except urllib.error.URLError as err:
            raise HttpDriverError(f"URL error {method} {path}: {err}", status=None)

    def _post_json(self, path: str, body: Dict) -> Dict:
        return json.loads(self._request("POST", path, body).decode("utf-8"))

    def _get_json(self, path: str) -> Dict:
        return json.loads(self._request("GET", path, None).decode("utf-8"))

    def _get_bytes(self, path: str) -> bytes:
        return self._request("GET", path, None)

    # -- 响应解析 -----------------------------------------------------------

    def _snapshot_from_json(self, resp: Dict) -> Snapshot:
        summary_raw = resp.get("state_summary") or resp.get("summary") or {}
        return Snapshot(
            session_id=str(resp.get("session_id") or self._session_id or ""),
            revision=int(resp.get("revision", self._revision)),
            timestep=self._parse_timestep(resp, summary_raw),
            action=self._parse_action(resp.get("action")),
            reward=float(resp.get("reward", 0.0)),
            terminated=bool(resp.get("terminated", False)),
            truncated=bool(resp.get("truncated", False)),
            summary=self._summary_from_json(summary_raw)
            if isinstance(summary_raw, dict)
            else None,
            info=dict(resp.get("info") or {}),
            command_id=str(resp.get("command_id", "")),
        )

    @staticmethod
    def _parse_timestep(resp: Dict, summary_raw: Dict) -> int:
        if isinstance(summary_raw, dict) and summary_raw.get("timestep") is not None:
            return int(summary_raw["timestep"])
        return int(resp.get("timestep", 0))

    @staticmethod
    def _parse_action(action_raw) -> Optional[ActionSpec]:
        if not action_raw:
            return None
        if isinstance(action_raw, dict):
            for key in ("applied", "requested"):
                candidate = action_raw.get(key)
                if isinstance(candidate, dict):
                    action_raw = candidate
                    break
        try:
            return ActionSpec.from_dict(action_raw)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _summary_from_json(d: Dict) -> StateSummary:
        return StateSummary(
            timestep=int(d.get("timestep", 0)),
            health=float(d.get("health", 0.0)),
            food=float(d.get("food", 0.0)),
            drink=float(d.get("drink", 0.0)),
            energy=float(d.get("energy", 0.0)),
            mana=float(d.get("mana", 0.0)),
            floor=int(d.get("floor", 0)),
            xp=int(d.get("xp", 0)),
            dexterity=int(d.get("dexterity", 0)),
            strength=int(d.get("strength", 0)),
            intelligence=int(d.get("intelligence", 0)),
            is_sleeping=bool(d.get("is_sleeping", False)),
            is_resting=bool(d.get("is_resting", False)),
            inventory=dict(d.get("inventory") or {}),
            achievements=list(d.get("achievements") or []),
            task_progress=float(d.get("task_progress", 0.0)),
            task_done=bool(d.get("task_done", False)),
            instruction=str(d.get("instruction", "")),
            task_id=str(d.get("task_id", "")),
            task_version=str(d.get("task_version", "")),
        )

    def _attach_frame(self, snap: Snapshot, resp: Dict) -> None:
        frame_info = resp.get("frame") or {}
        revision = frame_info.get("revision") or snap.revision
        png: Optional[bytes] = None
        url = frame_info.get("url")
        if url:
            try:
                png = self._get_bytes(self._url_to_path(url))
            except HttpDriverError:
                png = None
        elif frame_info.get("frame_png"):
            png = base64.b64decode(frame_info["frame_png"])
        if png is not None:
            snap.frame_png = png
            snap.frame_revision = revision

    @staticmethod
    def _url_to_path(url: str) -> str:
        if url.startswith(("http://", "https://")):
            url = urllib.parse.urlsplit(url).path
        return url if url.startswith("/") else "/" + url


class PygameGUI:
    """本地 Pygame GUI。

    构造时传入 SessionDriver 即 embedded 模式；remote 模式请用
    ``PygameGUI.connect_http(base_url)``。
    """

    def __init__(
        self,
        driver: SessionDriver,
        title: str = "Craftax GUI",
        fps: int = 20,
        pixel_render_size: Optional[int] = None,
    ) -> None:
        """pixel_render_size=None 时按目标窗口高度自适应放大场景。"""
        self.driver = driver
        self.title = title
        self.fps = int(fps)
        self.pixel_render_size = pixel_render_size
        self.controller = ControllerMode.HUMAN
        self.recording_status = "off"

        self._snapshot: Optional[Snapshot] = None
        self._frame_rgb: Optional[np.ndarray] = None
        self._events: List = []
        self._running = False
        self._screen = None
        self._font = None
        self._clock = pygame.time.Clock()

        self._target_height = 720  # 自适应缩放的目标窗口高度
        self._scene_scale = 3  # 在 _ensure_screen 中最终确定
        self._panel_width = 340
        self._row_height = 16
        self._panel_margin = 10
        self._panel_gap = 10
        self._header_height = 18

        pygame.init()

    # -- 工厂 ---------------------------------------------------------------

    @classmethod
    def connect_http(
        cls,
        base_url: str,
        title: str = "Craftax GUI (remote)",
        fps: int = 20,
        pixel_render_size: Optional[int] = None,
        env_name: str = "Craftax-Pixels-v1",
        seed: Optional[int] = None,
        recording: Optional[Dict] = None,
        block_pixel_size: Optional[int] = None,
    ) -> "PygameGUI":
        """连接远程 service，创建会话并返回就绪的 GUI。"""
        driver = HttpSessionDriver(
            base_url,
            env_name=env_name,
            seed=seed,
            recording=recording,
            block_pixel_size=block_pixel_size,
        )
        gui = cls(driver, title=title, fps=fps, pixel_render_size=pixel_render_size)
        gui._on_snapshot(driver.create_session())
        return gui

    # -- 控制 ---------------------------------------------------------------

    def set_controller(self, mode: ControllerMode | str) -> None:
        if isinstance(mode, str):
            mode = parse_controller(mode)
        self.controller = mode

    def cycle_controller(self) -> None:
        modes = list(ControllerMode)
        self.controller = modes[(modes.index(self.controller) + 1) % len(modes)]

    def poll_events(self) -> None:
        self._events = list(pygame.event.get())

    def is_quit_requested(self) -> bool:
        return any(e.type == pygame.QUIT for e in self._events)

    def _handle_event(self, event) -> bool:
        """处理单个事件；返回 True 表示执行了一次环境 step。"""
        if event.type == pygame.QUIT:
            self._running = False
            return False
        if event.type != pygame.KEYDOWN:
            return False
        key = event.key
        if key == pygame.K_r:  # 重置
            self._on_snapshot(self.driver.reset())
            return False
        if key == pygame.K_c:  # 切换 controller
            self.cycle_controller()
            return False
        if self.controller != ControllerMode.HUMAN:
            return False
        action = key_to_action(key)
        if action is None:
            return False
        self._on_snapshot(self.driver.step(action))
        return True

    def run(self, max_steps: Optional[int] = None) -> int:
        """主循环。max_steps 限制环境 step 数（便于测试，不会无限循环）。"""
        if self._snapshot is None:
            self._on_snapshot(self.driver.reset())
        self._running = True
        step_count = 0
        while self._running:
            if max_steps is not None and step_count >= max_steps:
                break
            self.poll_events()
            for event in self._events:
                if self._handle_event(event):
                    step_count += 1
                if not self._running:
                    break
            if not self._running:
                break
            self._render()
            pygame.display.flip()
            self._clock.tick(self.fps)
        self._running = False
        return step_count

    # -- 渲染 ---------------------------------------------------------------

    def _on_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        self._frame_rgb = self._snapshot_to_rgb(snapshot)
        self._ensure_screen()

    @staticmethod
    def _snapshot_to_rgb(snapshot: Snapshot) -> Optional[np.ndarray]:
        if snapshot.frame_rgb is not None:
            frame = np.asarray(snapshot.frame_rgb)
            return frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
        if snapshot.frame_png is not None:
            return decode_png_rgb(snapshot.frame_png)
        return None

    def _ensure_screen(self) -> None:
        if self._screen is not None:
            return
        if self._frame_rgb is not None:
            h, w = self._frame_rgb.shape[:2]
        else:
            h, w = 64, 64
        # 场景自适应放大：尽量填满目标窗口高度，同时保留像素风最近邻
        if self.pixel_render_size is None:
            self._scene_scale = max(1, min(8, self._target_height // max(h, 1)))
        else:
            self._scene_scale = int(self.pixel_render_size)
        scene_h = h * self._scene_scale
        # 面板总高（含换行）；场景与面板取较高者，保证 task 内容完整可见
        panels_h = self._panels_total_height()
        screen_w = w * self._scene_scale + self._panel_width + self._panel_margin * 2
        screen_h = max(scene_h, panels_h) + self._panel_margin * 2
        # 不超过屏幕可用高度（macOS 窗口标题栏约 28px）
        try:
            max_h = pygame.display.Info().current_h - 60
            screen_h = min(screen_h, max_h)
        except pygame.error:
            pass
        pygame.display.set_caption(self.title)
        self._screen = pygame.display.set_mode((screen_w, screen_h))
        self._font = _make_font(13)

    def _panels_total_height(self) -> int:
        if self._snapshot is None:
            return 0
        blocks = panels_from_snapshot(self._snapshot, self.recording_status)
        return sum(self._box_height(b) for b in blocks) + self._panel_gap * (
            len(blocks) - 1
        )

    @property
    def _scene_width(self) -> int:
        if self._frame_rgb is None:
            return 0
        return int(self._frame_rgb.shape[1]) * self._scene_scale

    def _render(self) -> None:
        if self._screen is None or self._font is None:
            return
        self._screen.fill((12, 12, 16))
        self._draw_scene()
        self._draw_panels()

    def _draw_scene(self) -> None:
        if self._frame_rgb is None:
            return
        scaled = _scale_frame(self._frame_rgb, self._scene_scale)
        surface = pygame.surfarray.make_surface(np.transpose(scaled, (1, 0, 2)))
        rect = pygame.Rect(
            self._panel_margin, self._panel_margin, scaled.shape[1], scaled.shape[0]
        )
        self._screen.blit(surface, rect.topleft)
        pygame.draw.rect(self._screen, (90, 100, 120), rect, 1)

    def _draw_panels(self) -> None:
        if self._snapshot is None:
            return
        x = self._scene_width + self._panel_margin * 2
        y = self._panel_margin
        blocks = panels_from_snapshot(self._snapshot, self.recording_status)
        for (title, color), block in zip(PANEL_THEMES, blocks):
            self._draw_panel_box(x, y, title, color, block)
            y += self._box_height(block) + self._panel_gap

    def _box_height(self, block: List[str]) -> int:
        body = block[1:] if block and block[0].startswith("[") else block
        n_lines = sum(len(self._wrap_line(line)) for line in body)
        return 18 + n_lines * 16 + 8

    def _wrap_line(self, line: str) -> List[str]:
        """按像素宽度换行（中文字符按双宽自然换行），返回多行。"""
        if not line:
            return [""]
        max_px = self._panel_width - 14
        lines: List[str] = []
        current = ""
        for char in line:
            if self._text_width_px(current + char) <= max_px:
                current += char
            else:
                lines.append(current)
                current = char
        lines.append(current)
        return lines

    def _text_width_px(self, text: str) -> int:
        """估算文本像素宽；font 可用时用精确测量，否则按 CJK 双宽估算。"""
        if self._font is not None:
            return self._font.size(text)[0]
        return sum(16 if ord(c) > 0x2E7F else 8 for c in text)

    def _draw_panel_box(
        self,
        x: int,
        y: int,
        title: str,
        color: Tuple[int, int, int],
        block: List[str],
    ) -> None:
        """带彩色边框与标题栏的面板块，内容行自动换行。"""
        body_lines = block[1:] if block and block[0].startswith("[") else block
        wrapped = [w for line in body_lines for w in self._wrap_line(line)]
        height = 18 + len(wrapped) * 16 + 8
        box = pygame.Rect(x, y, self._panel_width, height)
        pygame.draw.rect(self._screen, color, box, 1)
        # 标题栏
        header = pygame.Rect(x + 1, y + 1, self._panel_width - 2, 16)
        pygame.draw.rect(self._screen, color, header)
        title_surface = self._font.render(title, True, (10, 10, 12))
        self._screen.blit(title_surface, (x + 6, y + 2))
        # 内容
        for i, line in enumerate(wrapped):
            surface = self._font.render(line, True, (218, 220, 224))
            self._screen.blit(surface, (x + 6, y + 18 + i * 16))


# 面板标题与主题色（按 panels_from_snapshot 返回顺序）
PANEL_THEMES = (
    ("STATUS", (87, 187, 138)),  # 绿
    ("INVENTORY", (94, 129, 244)),  # 蓝
    ("TASK", (240, 190, 78)),  # 黄
    ("DEBUG", (150, 150, 150)),  # 灰
)

# 支持 CJK 的字体候选（macOS 黑体/冬青/Unicode，Linux 思源/文泉驿）
CJK_FONT_CANDIDATES = (
    "stheitimedium",
    "hiraginosansgb",
    "arialunicode",
    "notosanscjksc",
    "wenquanyi",
    "pingfangsc",
)


def _find_cjk_font() -> Optional[str]:
    """查找可渲染中文的系统字体路径；找不到返回 None。"""
    try:
        available = set(pygame.font.get_fonts())
        for name in CJK_FONT_CANDIDATES:
            if name in available:
                return pygame.font.match_font(name)
    except pygame.error:
        return None
    return None


def _make_font(size: int) -> pygame.font.Font:
    """创建支持中文的字体；无 CJK 字体时回退 monospace。"""
    path = _find_cjk_font()
    if path is not None:
        return pygame.font.Font(path, size)
    return pygame.font.SysFont("monospace", size)


def _scale_frame(frame: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)
