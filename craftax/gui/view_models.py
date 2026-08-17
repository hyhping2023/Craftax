"""展示面板的纯数据模型。

所有 ``render()`` 返回 ``list[str]``（每行一个字符串），供 Pygame 文本渲染消费。
不依赖 pygame，只依赖 ``craftax.contracts``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from craftax.contracts import Snapshot, StateSummary

# 物品栏展示顺序（其余键按字母序追加）
INVENTORY_ITEM_ORDER = (
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapphire",
    "ruby",
    "sapling",
    "pickaxe",
    "sword",
    "bow",
    "torches",
    "arrows",
    "books",
    "potions",
    "armour",
)

RECENT_ACHIEVEMENTS_LIMIT = 3


def _format_value(value: Any) -> str:
    """把 inventory 里的标量 / numpy 标量 / 数组格式化为可读文本。"""
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return str(arr.item())
        if arr.size <= 12:
            return "[" + ", ".join(str(v) for v in arr.reshape(-1).tolist()) + "]"
        return f"array(shape={arr.shape})"
    if isinstance(value, (list, tuple)) and len(value) <= 12:
        return "[" + ", ".join(str(v) for v in value) + "]"
    if hasattr(value, "item"):  # numpy 标量
        return str(value.item())
    return str(value)


@dataclass(frozen=True)
class InventoryPanel:
    """物品栏摘要：按 INVENTORY_ITEM_ORDER 逐行展示。"""

    inventory: Dict[str, Any] = field(default_factory=dict)
    item_order: tuple = INVENTORY_ITEM_ORDER

    @classmethod
    def from_summary(cls, summary: StateSummary) -> "InventoryPanel":
        return cls(inventory=dict(summary.inventory))

    def render(self) -> List[str]:
        lines = ["[INVENTORY]"]
        keys = set(self.inventory)
        for item in self.item_order:
            if item in keys:
                lines.append(f"  {item}: {_format_value(self.inventory[item])}")
                keys.discard(item)
        for item in sorted(keys):
            lines.append(f"  {item}: {_format_value(self.inventory[item])}")
        if not self.inventory:
            lines.append("  (empty)")
        return lines


@dataclass(frozen=True)
class StatusPanel:
    """角色状态：生命/饱食/饮水/能量/法力/楼层/经验/属性/睡觉/休息。"""

    health: float = 0.0
    food: float = 0.0
    drink: float = 0.0
    energy: float = 0.0
    mana: float = 0.0
    floor: int = 0
    xp: int = 0
    dexterity: int = 0
    strength: int = 0
    intelligence: int = 0
    is_sleeping: bool = False
    is_resting: bool = False

    @classmethod
    def from_summary(cls, summary: StateSummary) -> "StatusPanel":
        return cls(
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
        )

    def render(self) -> List[str]:
        lines = ["[STATUS]"]
        lines.append(f"  health: {self.health:6.1f}")
        lines.append(f"  food:   {self.food:6.1f}")
        lines.append(f"  drink:  {self.drink:6.1f}")
        lines.append(f"  energy: {self.energy:6.1f}")
        lines.append(f"  mana:   {self.mana:6.1f}")
        lines.append(f"  floor:  {self.floor}")
        lines.append(f"  xp:     {self.xp}")
        lines.append(f"  dex:    {self.dexterity}")
        lines.append(f"  str:    {self.strength}")
        lines.append(f"  int:    {self.intelligence}")
        lines.append(f"  sleep:  {self.is_sleeping}")
        lines.append(f"  rest:   {self.is_resting}")
        return lines


@dataclass(frozen=True)
class TaskPanel:
    """任务面板：指令 / 进度 / 成功 / 事件 token。"""

    instruction: str = ""
    task_id: str = ""
    progress: float = 0.0
    done: bool = False
    event_tokens: List[str] = field(default_factory=list)

    @classmethod
    def from_summary(
        cls, summary: StateSummary, event_tokens: List[str] = ()
    ) -> "TaskPanel":
        return cls(
            task_id=summary.task_id,
            instruction=summary.instruction,
            progress=summary.task_progress,
            done=summary.task_done,
            event_tokens=list(event_tokens),
        )

    def render(self) -> List[str]:
        lines = ["[TASK]"]
        lines.append(f"  id:     {self.task_id or '-'}")
        lines.append(f"  instr:  {self.instruction or '-'}")
        lines.append(f"  progress: {self.progress * 100:5.1f}%")
        lines.append(f"  done:   {self.done}")
        tokens = self.event_tokens[-5:] or ["-"]
        lines.append(f"  events: {', '.join(tokens)}")
        return lines


@dataclass(frozen=True)
class DebugPanel:
    """调试面板：session / seed / revision / timestep / reward / 成就 / 录制状态。"""

    session_id: str = "-"
    seed: str = "-"
    revision: int = 0
    timestep: int = 0
    reward: float = 0.0
    achievements: List[str] = field(default_factory=list)
    recording: str = "off"

    @classmethod
    def from_snapshot(
        cls, snapshot: Snapshot, recording_status: str = "off"
    ) -> "DebugPanel":
        info = snapshot.info or {}
        seed = info.get("seed", info.get("env_seed", "-"))
        achievements = (
            list(snapshot.summary.achievements) if snapshot.summary is not None else []
        )
        recording = _resolve_recording(recording_status, info)
        return cls(
            session_id=snapshot.session_id or "-",
            seed="-" if seed is None else str(seed),
            revision=snapshot.revision,
            timestep=snapshot.timestep,
            reward=snapshot.reward,
            achievements=achievements,
            recording=recording,
        )

    def render(self) -> List[str]:
        lines = ["[DEBUG]"]
        lines.append(f"  session: {self.session_id}")
        lines.append(f"  seed:    {self.seed}")
        lines.append(f"  revision: {self.revision}")
        lines.append(f"  timestep: {self.timestep}")
        lines.append(f"  reward:  {self.reward:+.3f}")
        recent = self.achievements[-RECENT_ACHIEVEMENTS_LIMIT:]
        lines.append(f"  achiev:  {', '.join(recent) or '-'}")
        lines.append(f"  record:  {self.recording}")
        return lines


def _resolve_recording(recording_status: str, info: Dict[str, Any]) -> str:
    """优先用 info 中的录制字段，其次用调用方传入的状态。"""
    if info:
        for key in ("recording_status", "recording"):
            value = info.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                return "on" if value else "off"
            return str(value)
    return recording_status or "off"


def panels_from_snapshot(
    snapshot: Snapshot, recording_status: str = "off"
) -> List[List[str]]:
    """由 Snapshot 一次性生成全部面板的渲染行。"""
    if snapshot.summary is None:
        summary = StateSummary(
            timestep=snapshot.timestep,
            health=0.0,
            food=0.0,
            drink=0.0,
            energy=0.0,
            mana=0.0,
            floor=0,
            xp=0,
            dexterity=0,
            strength=0,
            intelligence=0,
            is_sleeping=False,
            is_resting=False,
        )
    else:
        summary = snapshot.summary
    event_tokens = list((snapshot.info or {}).get("event_tokens", []))
    return [
        StatusPanel.from_summary(summary).render(),
        InventoryPanel.from_summary(summary).render(),
        TaskPanel.from_summary(summary, event_tokens).render(),
        DebugPanel.from_snapshot(snapshot, recording_status).render(),
    ]
