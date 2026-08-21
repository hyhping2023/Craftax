"""AsyncRecorder：实现 contracts.RecorderHook 的异步录制器。

设计：
- 有界队列（queue.Queue(maxsize)）+ 后台消费线程；on_episode_start /
  on_transition / on_episode_end 只负责入队，主线程不被录制阻塞。
- 背压策略：队列满时 put() 阻塞调用方（不允许静默丢数据），并累计
  backpressure_events / backpressure_wait_seconds 供观测。
- 按 episode 组装：states[0..T]（on_episode_start 提供 state[0]，
  on_transition 的 record.state 依次追加）；帧只保留初始帧/采样帧/终局帧。
- 事件 token：TaskAdapter.evaluate(record.state, info) 的结果填入
  record.event_tokens / instruction；一个 episode 内相同 token 只出现一次
  （首次达成时记录，之后忽略），避免重复标注。
- close()：flush 所有未完成 episode（标记 truncated）+ 关闭线程。

任务适配器解析：
- 构造时传入 task_adapter 则全 episode 使用；
- 否则按 on_episode_start 的 (task_id, task_version) 从 registry 惰性解析。
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np

from craftax.contracts import (
    RecordingConfig,
    TransitionRecord,
)
from craftax.recording.shard_writer import (
    EpisodeData,
    FrameEntry,
    ShardWriter,
    make_shard_dir,
)

# 队列事件类型
_START = "start"
_TRANSITION = "transition"
_END = "end"
_STOP = "stop"


class AsyncRecorder:
    """有界队列 + 后台线程的录制器。线程安全地调用 RecorderHook 方法。"""

    def __init__(
        self,
        *,
        spool_dir: Optional[str] = None,
        run_id: str = "default",
        producer_id: str = "recorder",
        attempt_id: Optional[str] = None,
        recording_config: Optional[RecordingConfig] = None,
        task_adapter: Any = None,
        queue_maxsize: int = 1000,
        shard_max_transitions: Optional[int] = None,
        segment_max_transitions: int = 500,
    ):
        self._config = recording_config or RecordingConfig()
        self._config.validate()
        self._task_adapter = task_adapter
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max(1, queue_maxsize))
        self._backpressure_events = 0
        self._backpressure_wait_seconds = 0.0
        self._spool_dir = spool_dir or self._config.spool_dir
        self._run_id = run_id
        self._producer_id = producer_id
        self._attempt_id = attempt_id or uuid.uuid4().hex[:8]
        self._shard_max_transitions = shard_max_transitions or self._config.shard_max_transitions
        # 一局可持续数万步；若等到终局才把整个 episode 交给 ShardWriter，
        # states/frames 会无界堆在内存。分段只切录制产物，不切游戏 session。
        self._segment_max_transitions = max(1, int(segment_max_transitions))

        # 后台线程状态
        self._lock = threading.Lock()
        self._writer: Optional[ShardWriter] = None
        self._current: Optional[Dict[str, Any]] = None  # episode 组装器
        self._episode_counter = 0
        self._closed = False

        self._thread = threading.Thread(target=self._consume, name="craftax-recorder", daemon=True)
        self._thread.start()

    # -- 钩子（入队，不阻塞） ------------------------------------------------

    def on_episode_start(
        self,
        session_id: str,
        episode_id: str,
        task_id: str,
        task_version: str,
        seed: int,
        recording_config: RecordingConfig,
        state: Any,
        frame: Optional[np.ndarray],
    ) -> None:
        self._put((_START, (session_id, episode_id, task_id, task_version, seed, recording_config, state, frame)))

    def on_transition(self, record: TransitionRecord) -> None:
        self._put((_TRANSITION, record))

    def on_episode_end(self, session_id: str, episode_id: str, terminated: bool) -> None:
        self._put((_END, (session_id, episode_id, terminated)))

    def close(self) -> None:
        """flush 所有未完成 episode 并关闭消费线程。"""
        self._put((_STOP, None))
        self._thread.join(timeout=120)
        self._closed = True

    # -- 队列与背压 ---------------------------------------------------------

    def _put(self, item) -> None:
        if self._closed:
            raise RuntimeError("recorder 已关闭")
        started = time.monotonic()
        try:
            self._queue.put(item, timeout=300)
        except queue.Full:
            # 阻塞等待（不丢数据）；统计背压等待时长
            while True:
                try:
                    self._queue.put(item, timeout=300)
                    break
                except queue.Full:  # pragma: no cover - 长超时防御
                    continue
        wait = time.monotonic() - started
        if wait > 0.05:
            with self._lock:
                self._backpressure_events += 1
                self._backpressure_wait_seconds += wait

    @property
    def backpressure_events(self) -> int:
        with self._lock:
            return self._backpressure_events

    @property
    def backpressure_wait_seconds(self) -> float:
        with self._lock:
            return self._backpressure_wait_seconds

    # -- 消费线程 -----------------------------------------------------------

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            kind, payload = item
            if kind == _STOP:
                break
            try:
                if kind == _START:
                    self._on_start(*payload)
                elif kind == _TRANSITION:
                    self._on_transition(payload)
                elif kind == _END:
                    self._on_end(*payload)
            except Exception:  # noqa: BLE001 - 消费线程不允许崩溃
                # 单个事件失败不应中断录制线程；错误打印保留现场
                import traceback

                traceback.print_exc()
        # 线程退出前 flush 未完成 episode（标记 truncated）
        if self._current is not None:
            self._flush_current(truncated=True)
        if self._writer is not None and not self._writer.is_sealed:
            self._writer.finalize()

    def _resolve_adapter(self, task_id: str, task_version: str):
        if self._task_adapter is not None:
            return self._task_adapter
        from craftax.tasks.registry import get_task_adapter

        return get_task_adapter(task_id, task_version)

    def _ensure_writer(
        self, task_id: str, task_version: str, frame_sample, gold_frames: bool,
        env_params: Optional[Dict[str, Any]] = None,
    ) -> ShardWriter:
        if self._writer is None:
            shard_dir = make_shard_dir(
                self._spool_dir, self._run_id, self._producer_id, self._attempt_id
            )
            self._writer = ShardWriter(
                shard_dir,
                run_id=self._run_id,
                producer_id=self._producer_id,
                attempt_id=self._attempt_id,
                task_id=task_id,
                task_version=task_version,
                frame_sample=frame_sample,
                env_params=env_params,
                gold_frames=gold_frames,
                shard_max_transitions=self._shard_max_transitions,
                video_fps=frame_sample.video_fps,
            )
        return self._writer

    def _on_start(
        self,
        session_id: str,
        episode_id: str,
        task_id: str,
        task_version: str,
        seed: int,
        recording_config: RecordingConfig,
        state: Any,
        frame: Optional[np.ndarray],
    ) -> None:
        if self._current is not None:
            # 上一个 episode 未正常结束：强制 flush 并标记 truncated
            self._flush_current(truncated=True)
        config = recording_config or self._config
        adapter = self._resolve_adapter(task_id, task_version)
        self._current = self._new_segment(
            session_id=session_id,
            source_episode_id=episode_id,
            task_id=task_id or adapter.task_id,
            task_version=task_version or adapter.version,
            seed=int(seed),
            config=config,
            state=state,
            frame=frame,
            segment_index=0,
            seen_tokens=set(),
        )

    def _new_segment(
        self,
        *,
        session_id: str,
        source_episode_id: str,
        task_id: str,
        task_version: str,
        seed: int,
        config: RecordingConfig,
        state: Any,
        frame: Optional[np.ndarray],
        segment_index: int,
        seen_tokens: set,
    ) -> Dict[str, Any]:
        """创建一个录制段。段间共享边界 state，游戏 session 从不中断。"""
        # 首段保留原 episode_id，兼容既有数据和下游；续段才追加后缀。
        episode_id = (
            source_episode_id
            if segment_index == 0
            else f"{source_episode_id}-seg{segment_index:04d}"
        )
        acc: Dict[str, Any] = {
            "session_id": session_id,
            "source_episode_id": source_episode_id,
            "episode_id": episode_id,
            "segment_index": segment_index,
            "task_id": task_id,
            "task_version": task_version,
            "seed": seed,
            "states": [state],
            "transitions": [],
            "frames": [],
            "seen_tokens": seen_tokens,
            "terminated": False,
            "truncated": False,
            "config": config,
            "start_wall_ns": time.time_ns(),
        }
        # state[0] 的初始帧
        if frame is not None:
            from craftax.recording.frame_sampler import frame_index_row

            timestep = int(np.asarray(getattr(state, "timestep", 0)))
            row = frame_index_row(
                config.frame_sample,
                episode_id=episode_id,
                video_id=self._video_id_for(episode_id),
                frame_index=0,
                timestep=timestep,
                state_index=0,
                terminated=False,
                truncated=False,
            )
            # 续段沿用上一段边界 state 的 timestep，通常不为 0；对每个
            # 独立 MP4 来说它仍然是该视频的首帧，必须显式标记为初始帧。
            if segment_index > 0:
                row["is_initial_frame"] = True
            acc["frames"].append(FrameEntry(rgb=np.asarray(frame), row=row))
        return acc

    def _video_id_for(self, episode_id: str) -> str:
        return f"ep{self._episode_counter:06d}"

    def _on_transition(self, record: TransitionRecord) -> None:
        acc = self._current
        if acc is None:
            raise RuntimeError("收到 transition 前没有 episode 开始")
        # 任务标注（只读）
        adapter = self._resolve_adapter(record.task_id or acc["task_id"], record.task_version or acc["task_version"])
        try:
            teval = adapter.evaluate(record.state, record.info)
        except Exception:  # noqa: BLE001 - 评估失败不中断录制
            teval = None
        if teval is not None:
            if not record.instruction:
                record.instruction = teval.instruction
            new_tokens = [t for t in teval.event_tokens if t not in acc["seen_tokens"]]
            acc["seen_tokens"].update(new_tokens)
            record.event_tokens = list(new_tokens)
        record.task_id = record.task_id or adapter.task_id
        record.task_version = record.task_version or adapter.version

        acc["transitions"].append(record)
        state_after = record.state
        acc["states"].append(state_after)

        # 采样帧（初始/采样/终局任一生效且有 RGB 才记录）
        if record.frame is not None and (
            record.is_sampled_frame or record.is_initial_frame or record.is_terminal_frame
        ):
            from craftax.recording.frame_sampler import frame_index_row

            state_index = len(acc["states"]) - 1
            timestep = int(np.asarray(getattr(state_after, "timestep", 0)))
            row = frame_index_row(
                acc["config"].frame_sample,
                episode_id=acc["episode_id"],
                video_id=self._video_id_for(acc["episode_id"]),
                frame_index=len(acc["frames"]),
                timestep=timestep,
                state_index=state_index,
                sim_time_ns=record.sim_time_ns,
                terminated=record.terminated,
                truncated=record.truncated,
            )
            acc["frames"].append(FrameEntry(rgb=np.asarray(record.frame), row=row))

        if record.terminated:
            acc["terminated"] = True
        if record.truncated:
            acc["truncated"] = True

        # 流式封存：达到段上限立即写盘，并用本 transition 的后继 state
        # 作为下一段 state[0]。这不会改变环境，也不会人为终止长程任务。
        if (
            len(acc["transitions"]) >= self._segment_max_transitions
            and not record.terminated
            and not record.truncated
        ):
            self._roll_segment(record.state, record.frame)

    def _on_end(self, session_id: str, episode_id: str, terminated: bool) -> None:
        acc = self._current
        if acc is None or acc["source_episode_id"] != episode_id:
            return
        if terminated:
            acc["terminated"] = True
        self._flush_current(truncated=False)

    def _roll_segment(self, state: Any, frame: Optional[np.ndarray]) -> None:
        """封存当前录制段，并无缝开始下一段。"""
        acc = self._current
        if acc is None:
            return
        meta = {
            "session_id": acc["session_id"],
            "source_episode_id": acc["source_episode_id"],
            "task_id": acc["task_id"],
            "task_version": acc["task_version"],
            "seed": acc["seed"],
            "config": acc["config"],
            "segment_index": int(acc["segment_index"]) + 1,
            "seen_tokens": acc["seen_tokens"],
        }
        self._flush_current(truncated=True)
        self._current = self._new_segment(
            state=state,
            frame=frame,
            **meta,
        )

    def _flush_current(self, truncated: bool) -> None:
        acc = self._current
        if acc is None:
            return
        episode = EpisodeData(
            episode_id=acc["episode_id"],
            session_id=acc["session_id"],
            task_id=acc["task_id"],
            task_version=acc["task_version"],
            seed=acc["seed"],
            states=acc["states"],
            transitions=acc["transitions"],
            frames=acc["frames"],
            terminated=bool(acc["terminated"]),
            truncated=bool(acc["truncated"] or truncated),
            start_wall_ns=acc["start_wall_ns"],
            end_wall_ns=time.time_ns(),
            video_id=self._video_id_for(acc["episode_id"]),
        )
        writer = self._ensure_writer(
            acc["task_id"], acc["task_version"], acc["config"].frame_sample,
            acc["config"].gold_frames, acc["config"].env_params,
        )
        writer.add_episode(episode)
        self._episode_counter += 1
        self._current = None
        # shard 满 → 封存并开启新 shard（新 attempt_id，避免覆盖旧 shard 目录）
        if writer.is_full:
            writer.finalize()
            self._writer = None
            self._attempt_id = uuid.uuid4().hex[:8]
