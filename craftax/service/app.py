"""FastAPI 应用工厂。

端点：
- POST   /v1/sessions                      创建会话（201，返回初始 Snapshot）
- POST   /v1/sessions/{sid}/reset          显式 reset
- POST   /v1/sessions/{sid}/step           执行动作
- GET    /v1/sessions/{sid}/snapshot       结构化 observation / 摘要
- GET    /v1/sessions/{sid}/map            指定楼层的完整方块网格与玩家位置
- GET    /v1/sessions/{sid}/frames/{rev}   以 image/png 返回指定 revision 的帧
- DELETE /v1/sessions/{sid}                删除会话

错误语义：
- 409  RevisionConflictError（expected_revision 过期）
- 400  SessionTerminatedError（终止后 step）/ 参数校验失败
- 410  SessionNotFoundError（会话不存在）
- 404  FrameNotFoundError（帧不存在或已被淘汰）

JAX 推理在同步端点（线程池）中执行，避免阻塞事件循环。
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from craftax.contracts import ActionSpec, Snapshot
from craftax.service.api_models import (
    ActionRef,
    ActionStateRef,
    FrameRef,
    ResetRequest,
    SessionCreateRequest,
    SnapshotResponse,
    StateSummaryModel,
    StepRequest,
)
from craftax.service.session_actor import (
    FrameNotFoundError,
    RevisionConflictError,
    SessionTerminatedError,
    resolve_action_spec,
)
from craftax.service.session_manager import SessionManager, SessionNotFoundError


def _summary_to_model(summary: Any) -> Optional[StateSummaryModel]:
    if summary is None:
        return None
    return StateSummaryModel(
        timestep=summary.timestep,
        health=summary.health,
        food=summary.food,
        drink=summary.drink,
        energy=summary.energy,
        mana=summary.mana,
        floor=summary.floor,
        xp=summary.xp,
        dexterity=summary.dexterity,
        strength=summary.strength,
        intelligence=summary.intelligence,
        is_sleeping=summary.is_sleeping,
        is_resting=summary.is_resting,
        inventory=summary.inventory,
        achievements=summary.achievements,
        task_progress=summary.task_progress,
        task_done=summary.task_done,
        instruction=summary.instruction,
        task_id=summary.task_id,
        task_version=summary.task_version,
        player_position=list(summary.player_position),
        player_direction=int(summary.player_direction),
    )


def _to_snapshot_response(actor: Any, snap: Snapshot) -> SnapshotResponse:
    frame: Optional[FrameRef] = None
    if snap.frame_revision is not None:
        dims = actor.frame_dims(snap.frame_revision)
        if dims is not None:
            height, width = dims
            frame = FrameRef(
                revision=snap.frame_revision,
                url=(
                    f"/v1/sessions/{actor.session_id}"
                    f"/frames/{snap.frame_revision}"
                ),
                content_type="image/png",
                width=width,
                height=height,
            )
    return SnapshotResponse(
        session_id=snap.session_id,
        revision=snap.revision,
        timestep=snap.timestep,
        action=(
            ActionStateRef(
                requested=ActionRef(id=snap.action.id, name=snap.action.name),
                applied=ActionRef(id=snap.action.id, name=snap.action.name),
            )
            if snap.action is not None
            else None
        ),
        reward=snap.reward,
        terminated=snap.terminated,
        truncated=snap.truncated,
        state_summary=_summary_to_model(snap.summary),
        frame=frame,
        command_id=snap.command_id,
        info=snap.info,
    )


def create_app(manager: Optional[SessionManager] = None) -> FastAPI:
    """创建 FastAPI 应用。manager 可选注入（测试可传共享 manager）。"""
    mgr = manager if manager is not None else SessionManager()
    app = FastAPI(title="Craftax Embodied Service", version="0.1.0")
    app.state.manager = mgr

    # -- 会话 ---------------------------------------------------------------

    @app.post("/v1/sessions", status_code=201, response_model=SnapshotResponse)
    def create_session(req: SessionCreateRequest) -> SnapshotResponse:
        actor = mgr.create_session(req)
        return _to_snapshot_response(actor, actor.get_snapshot())

    @app.delete("/v1/sessions/{sid}", status_code=204)
    def delete_session(sid: str) -> Response:
        mgr.delete(sid)
        return Response(status_code=204)

    # -- 动作 ---------------------------------------------------------------

    @app.post(
        "/v1/sessions/{sid}/step", response_model=SnapshotResponse
    )
    def step_session(sid: str, req: StepRequest) -> SnapshotResponse:
        actor = mgr.get(sid)
        action: ActionSpec = resolve_action_spec(req.action)
        snap = actor.step(
            action,
            command_id=req.command_id,
            wait_frame=req.wait_frame,
            expected_revision=req.expected_revision,
        )
        return _to_snapshot_response(actor, snap)

    @app.post(
        "/v1/sessions/{sid}/reset", response_model=SnapshotResponse
    )
    def reset_session(sid: str, req: ResetRequest) -> SnapshotResponse:
        actor = mgr.get(sid)
        snap = actor.reset(
            seed=req.seed,
            expected_revision=req.expected_revision,
            command_id=req.command_id,
        )
        return _to_snapshot_response(actor, snap)

    # -- 读取 ---------------------------------------------------------------

    @app.get(
        "/v1/sessions/{sid}/snapshot", response_model=SnapshotResponse
    )
    def get_snapshot(sid: str, revision: Optional[int] = None) -> SnapshotResponse:
        actor = mgr.get(sid)
        snap = actor.get_snapshot(revision)
        return _to_snapshot_response(actor, snap)

    @app.get(
        "/v1/sessions/{sid}/state", response_model=SnapshotResponse
    )
    def get_state(
        sid: str, revision: Optional[int] = None, detail: str = "summary"
    ) -> SnapshotResponse:
        """GUI 契约的快照查询端点（/snapshot 的别名）。

        detail 目前仅支持 "summary"（默认）；响应始终携带完整 state_summary。
        """
        actor = mgr.get(sid)
        snap = actor.get_snapshot(revision)
        return _to_snapshot_response(actor, snap)

    @app.get("/v1/sessions/{sid}/frames/{revision}")
    def get_frame(sid: str, revision: int) -> Response:
        actor = mgr.get(sid)
        png = actor.get_frame_png(revision)
        return Response(content=png, media_type="image/png")

    @app.get("/v1/sessions/{sid}/map")
    def get_map(sid: str, floor: Optional[int] = None) -> JSONResponse:
        """返回指定楼层（默认当前楼层）的完整方块网格与玩家位置。

        供路径规划器读取全图：map 为 48×48 的 BlockType int 网格，
        player_position 为 [x, y]，player_direction 为朝向动作 id。
        """
        actor = mgr.get(sid)
        try:
            payload = actor.get_map(floor)
        except ValueError as e:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_floor", "detail": {"message": str(e)}},
            )
        except RuntimeError as e:  # noqa: BLE001
            return JSONResponse(
                status_code=409,
                content={"error": "no_state", "detail": {"message": str(e)}},
            )
        return JSONResponse(content=payload)

    # -- 错误处理 -----------------------------------------------------------

    @app.exception_handler(RevisionConflictError)
    def _handle_revision_conflict(
        request: Request, exc: RevisionConflictError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": "revision_conflict",
                "detail": {
                    "current_revision": exc.current_revision,
                    "expected_revision": exc.expected_revision,
                },
            },
        )

    @app.exception_handler(SessionTerminatedError)
    def _handle_terminated(
        request: Request, exc: SessionTerminatedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": "session_terminated",
                "detail": {
                    "revision": exc.revision,
                    "terminated": exc.terminated,
                    "truncated": exc.truncated,
                },
            },
        )

    @app.exception_handler(SessionNotFoundError)
    def _handle_session_not_found(
        request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=410,
            content={
                "error": "session_not_found",
                "detail": {"session_id": exc.session_id},
            },
        )

    @app.exception_handler(FrameNotFoundError)
    def _handle_frame_not_found(
        request: Request, exc: FrameNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "frame_not_found",
                "detail": {"session_id": exc.session_id, "revision": exc.revision},
            },
        )

    @app.exception_handler(KeyError)
    def _handle_snapshot_not_found(
        request: Request, exc: KeyError
    ) -> JSONResponse:
        # actor.get_snapshot(revision) 对不存在的 revision 抛 KeyError。
        return JSONResponse(
            status_code=404,
            content={
                "error": "snapshot_not_found",
                "detail": {"message": str(exc)},
            },
        )

    @app.exception_handler(ValueError)
    def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "detail": {"message": str(exc)}},
        )

    return app
