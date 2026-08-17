"""service 层 REST API 的 Pydantic 请求 / 响应模型。

只依赖 pydantic，不依赖 contracts / jax。
跨模块契约仍以 craftax.contracts 为准；本文件仅承载 HTTP 协议层 schema。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ActionRef(BaseModel):
    """离散动作引用。name 应来自 craftax.craftax.constants.Action 枚举名。"""

    id: int
    name: str


class ActionStateRef(BaseModel):
    """动作的嵌套形式：requested 为客户端请求的动作，applied 为实际执行的动作。

    当前实现下两者总相同（command_id 幂等命中时返回缓存快照）；保留区分字段
    以便未来支持“请求动作被环境归一化/拒绝”等场景。
    """

    requested: ActionRef
    applied: ActionRef


class TaskSpecModel(BaseModel):
    """任务定义（只读标注语义，见 contracts.TaskSpec）。"""

    task_id: str = "native.survive"
    version: str = "1.0.0"
    params: Dict[str, Any] = Field(default_factory=dict)


class FrameSampleModel(BaseModel):
    """MP4 定频采样配置。step_rate_hz 必须能被 video_fps 整除。"""

    step_rate_hz: int = 20
    video_fps: int = 10


class RecordingModel(BaseModel):
    """数据集录制配置。"""

    enabled: bool = True
    dataset_run_id: str = "default"
    frame_sample: FrameSampleModel = Field(default_factory=FrameSampleModel)
    gold_frames: bool = False
    spool_dir: str = "spool"


class RenderModel(BaseModel):
    """帧返回方式。format: "png" | "rgb"；mode: "human" | "agent"。"""

    format: str = "png"
    mode: str = "human"


class SessionCreateRequest(BaseModel):
    """创建会话。

    env_name 仅支持 Craftax-Pixels-v1 / Craftax-Symbolic-v1（均使用 NoAutoReset）。
    """

    env_name: str = "Craftax-Pixels-v1"
    seed: Optional[int] = None
    task: TaskSpecModel = Field(default_factory=TaskSpecModel)
    render: RenderModel = Field(default_factory=RenderModel)
    recording: RecordingModel = Field(default_factory=RecordingModel)

    # 环境参数覆盖（测试 / 短 episode 用）；None 时使用默认 EnvParams。
    max_timesteps: Optional[int] = None
    god_mode: bool = False


class StepRequest(BaseModel):
    """执行一个动作。action 可以是整数 action id 或 {id, name}。

    return 指定响应中 frame 与 observation 的返回方式（GUI 约定
    {"frame": "reference", "observation": "summary"}）；服务端当前固定
    返回帧引用 + 完整 state_summary，该字段仅作契约声明。
    """

    action: Union[int, ActionRef]
    expected_revision: Optional[int] = None
    command_id: Optional[str] = None
    wait_frame: bool = True
    return_: Optional[Dict[str, str]] = Field(default=None, alias="return")


class ResetRequest(BaseModel):
    """显式重置为新 episode。"""

    seed: Optional[int] = None
    expected_revision: Optional[int] = None
    command_id: Optional[str] = None


class FrameRef(BaseModel):
    """帧引用：客户端通过 url 拉取 PNG，不把 base64 塞入 JSON。"""

    revision: int
    url: str
    content_type: str = "image/png"
    width: int
    height: int


class StateSummaryModel(BaseModel):
    """状态面板摘要（host 标量）。"""

    timestep: int
    health: float
    food: float
    drink: float
    energy: float
    mana: float
    floor: int
    xp: int
    dexterity: int
    strength: int
    intelligence: int
    is_sleeping: bool
    is_resting: bool
    inventory: Dict[str, Any] = Field(default_factory=dict)
    achievements: List[str] = Field(default_factory=list)
    task_progress: float = 0.0
    task_done: bool = False
    instruction: str = ""
    task_id: str = ""
    task_version: str = ""


class SnapshotResponse(BaseModel):
    """一次 reset / step / snapshot 查询的结果。

    顶层字段遵循 GUI REST 契约：state_summary（GUI 兼容别名 summary）、
    action（嵌套 requested/applied 形式）。
    """

    session_id: str
    revision: int
    timestep: int
    action: Optional[ActionStateRef] = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    state_summary: Optional[StateSummaryModel] = None
    frame: Optional[FrameRef] = None
    command_id: str = ""
    info: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """统一错误体。detail 因错误类型而异。"""

    error: str
    detail: Dict[str, Any] = Field(default_factory=dict)
