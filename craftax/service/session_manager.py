"""SessionManager：单机内存会话注册表与生命周期。

职责：
- 创建 / 查询 / 删除 SessionActor；
- 每 env_name 缓存一个 NoAutoReset 环境实例，复用 JIT 编译；
- controller lease 简化实现：actor.controller 字符串（默认 "human"）。
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

from craftax.contracts import FrameSampleConfig, RecordingConfig
from craftax.service.api_models import SessionCreateRequest
from craftax.service.session_actor import SessionActor, _make_env


class SessionNotFoundError(Exception):
    """会话不存在或已被删除（410）。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"session not found: {session_id}")


def _to_recording_config(model: Any) -> RecordingConfig:
    fs = FrameSampleConfig(
        step_rate_hz=model.frame_sample.step_rate_hz,
        video_fps=model.frame_sample.video_fps,
    )
    rec = RecordingConfig(
        enabled=model.enabled,
        dataset_run_id=model.dataset_run_id,
        frame_sample=fs,
        gold_frames=model.gold_frames,
        spool_dir=model.spool_dir,
    )
    rec.validate()
    return rec


class SessionManager:
    """进程内会话管理器。"""

    def __init__(self, env_factory: Optional[Any] = None):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionActor] = {}
        self._envs: Dict[str, Any] = {}
        self._env_factory = env_factory or _make_env

    def _get_env(self, env_name: str) -> Any:
        """按 env_name 缓存环境实例，避免每个会话重复 JIT 编译。"""
        env = self._envs.get(env_name)
        if env is None:
            env = self._env_factory(env_name)
            self._envs[env_name] = env
        return env

    # -- 生命周期 ---------------------------------------------------------

    def create_session(self, req: SessionCreateRequest) -> SessionActor:
        session_id = "sess_" + uuid.uuid4().hex[:12]
        recording = _to_recording_config(req.recording)
        env = self._get_env(req.env_name)
        actor = SessionActor(
            session_id=session_id,
            env_name=req.env_name,
            seed=req.seed,
            task=req.task,
            render=req.render,
            recording=recording,
            env=env,
            max_timesteps=req.max_timesteps,
            god_mode=req.god_mode,
            thirst_rate=req.thirst_rate,
        )
        with self._lock:
            self._sessions[session_id] = actor
        try:
            # 创建即产生初始 snapshot（revision 0 / frame 0）
            actor.reset(seed=req.seed)
        except Exception:
            with self._lock:
                self._sessions.pop(session_id, None)
            actor.close()
            raise
        return actor

    def get(self, session_id: str) -> SessionActor:
        with self._lock:
            actor = self._sessions.get(session_id)
        if actor is None:
            raise SessionNotFoundError(session_id)
        return actor

    def delete(self, session_id: str) -> None:
        with self._lock:
            actor = self._sessions.pop(session_id, None)
        if actor is None:
            raise SessionNotFoundError(session_id)
        actor.close()

    def list_session_ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def close_all(self) -> None:
        with self._lock:
            actors = list(self._sessions.values())
            self._sessions.clear()
        for actor in actors:
            actor.close()

    def warmup(self, env_name: str = "Craftax-Symbolic-v1") -> None:
        """可选预热：触发一次 reset+step 编译，之后首个请求不再等待编译。"""
        req = SessionCreateRequest(
            env_name=env_name,
            seed=0,
            recording={"enabled": False},
            max_timesteps=2,
        )
        actor = self.create_session(req)
        try:
            from craftax.contracts import ActionSpec

            actor.step(ActionSpec(id=0, name="NOOP"))
        finally:
            self.delete(actor.session_id)
