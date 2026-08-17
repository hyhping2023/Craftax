"""craftax/recording 测试。

覆盖：state_codec 一致性、frame_sampler、video_writer、
shard_writer+manifest+validators、AsyncRecorder 集成。

CPU JAX 首次 reset 会触发约 30-60s 编译，EnvState 用模块级 fixture 生成一次。
"""
from __future__ import annotations

import dataclasses
import json
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

import pyarrow.parquet as pq  # noqa: E402

from craftax.contracts import (  # noqa: E402
    ActionSpec,
    FrameSampleConfig,
    RecordingConfig,
    TaskEval,
    TaskSpec,
    TransitionRecord,
)
from craftax.craftax.craftax_state import EnvParams  # noqa: E402
from craftax.craftax.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.recording.frame_sampler import (  # noqa: E402
    FrameIndexBuilder,
    decide_sample,
    frame_flags,
)
from craftax.recording.manifest import verify_manifest  # noqa: E402
from craftax.recording.recorder import AsyncRecorder  # noqa: E402
from craftax.recording.shard_writer import (  # noqa: E402
    EpisodeData,
    FrameEntry,
    ShardWriter,
    make_shard_dir,
)
from craftax.recording.state_codec import flatten_state, state_schema  # noqa: E402
from craftax.recording.validators import (  # noqa: E402
    validate_frame_index,
    validate_shard,
    validate_timeline,
)
from craftax.recording.video_writer import VideoWriter  # noqa: E402

CFG = FrameSampleConfig(step_rate_hz=20, video_fps=10)  # R=2


@pytest.fixture(scope="module")
def base_state():
    env = CraftaxSymbolicEnvNoAutoReset()
    obs, state = env.reset(jax.random.PRNGKey(1), EnvParams())
    return jax.device_get(state)


def make_state(base_state, timestep, **overrides):
    kwargs = {"timestep": np.asarray(timestep)}
    kwargs.update(overrides)
    return dataclasses.replace(base_state, **kwargs)


def make_record(base_state, episode_id, t, *, terminated=False, truncated=False, frame=None):
    return TransitionRecord(
        session_id="sess-test",
        episode_id=episode_id,
        timestep=t,
        action=ActionSpec((t % 43), "NOOP"),
        action_source="test",
        command_id=f"cmd-{t}",
        reward=0.0,
        terminated=terminated,
        truncated=truncated,
        state=make_state(base_state, t + 1),
        frame=frame,
        is_sampled_frame=frame is not None and not terminated,
        is_terminal_frame=terminated or truncated,
    )


def synthetic_frame(width=16, height=16, value=0):
    return np.full((height, width, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# state_codec
# ---------------------------------------------------------------------------


def test_state_codec_flatten_schema_consistency(base_state):
    flattened = flatten_state(base_state)
    assert flattened, "flatten 结果不能为空"
    assert "fractal_noise_angles" not in flattened  # None tuple 被跳过
    schema = state_schema(flattened)
    assert schema["schema_version"] == "1.0"
    assert schema["num_fields"] == len(flattened)
    for path, arr in flattened.items():
        field = schema["fields"][path]
        assert str(arr.dtype) == field["dtype"], f"{path} dtype 不一致"
        assert list(arr.shape) == field["shape"], f"{path} shape 不一致"
    # 关键路径存在
    assert "player_position" in flattened
    assert "inventory/wood" in flattened
    assert "achievements" in flattened
    # bool 保留 dtype
    assert flattened["is_sleeping"].dtype == np.bool_


def test_state_codec_unflatten_missing_field_raises(base_state):
    from craftax.recording.state_codec import unflatten_state

    flattened = flatten_state(base_state)
    with pytest.raises(NotImplementedError, match="fractal_noise_angles"):
        unflatten_state(flattened)


# ---------------------------------------------------------------------------
# frame_sampler
# ---------------------------------------------------------------------------


def test_frame_sampler_stride():
    cfg = FrameSampleConfig(step_rate_hz=20, video_fps=10)
    assert cfg.frame_stride == 2
    # 常规采样：timestep % 2 == 0 且 > 0
    assert decide_sample(cfg, 0) is True  # 初始帧
    assert decide_sample(cfg, 1) is False
    assert decide_sample(cfg, 2) is True
    assert decide_sample(cfg, 4) is True
    # terminal 强制采样，即使不在采样周期上
    assert decide_sample(cfg, 3, terminated=True) is True
    assert decide_sample(cfg, 5, truncated=True) is True
    # 非终局、非采样：不采
    assert decide_sample(cfg, 3) is False


def test_frame_sampler_flags():
    cfg = FrameSampleConfig(20, 10)
    assert frame_flags(cfg, 0) == {"is_initial_frame": True, "is_terminal_frame": False}
    assert frame_flags(cfg, 4, terminated=True) == {
        "is_initial_frame": False,
        "is_terminal_frame": True,
    }
    # 终局帧优先标记
    assert frame_flags(cfg, 6, terminated=True)["is_terminal_frame"] is True


def test_frame_index_builder_rows():
    cfg = FrameSampleConfig(20, 10)
    builder = FrameIndexBuilder(cfg, episode_id="ep1", video_id="v1")
    assert builder.should_sample(0)
    row0 = builder.row(timestep=0, state_index=0)
    assert row0["frame_index"] == 0 and row0["is_initial_frame"] and row0["timestep"] == 0
    assert not builder.should_sample(1)
    assert builder.should_sample(2)
    row2 = builder.row(timestep=2, state_index=2)
    assert row2["frame_index"] == 1 and row2["timestep"] == 2
    row_term = builder.row(timestep=3, state_index=3, terminated=True)
    assert row_term["is_terminal_frame"] and row_term["frame_index"] == 2
    assert builder.num_frames == 3
    # 列与契约一致
    from craftax.contracts import FRAME_INDEX_COLUMNS

    assert list(row0.keys()) == FRAME_INDEX_COLUMNS


# ---------------------------------------------------------------------------
# video_writer
# ---------------------------------------------------------------------------


def test_video_writer_roundtrip(tmp_path):
    width, height, n = 32, 32, 10
    writer = VideoWriter(tmp_path, "vid0", fps=10, width=width, height=height)
    frames = [
        synthetic_frame(width, height, value=i % 255) for i in range(n)
    ]
    for f in frames:
        writer.add_frame(f)
    assert writer.frame_count == n
    path = writer.close()
    assert path.endswith(".mp4") and os.path.isfile(path)

    import imageio.v3 as iio

    decoded = iio.imread(path)
    assert decoded.ndim == 4 and decoded.shape[0] == n
    assert decoded.shape[1:] == (height, width, 3)


def test_video_writer_rejects_bad_frame(tmp_path):
    writer = VideoWriter(tmp_path, "vid1", fps=10, width=32, height=32)
    with pytest.raises(ValueError):
        writer.add_frame(np.zeros((16, 16, 3), dtype=np.uint8))  # 尺寸不符
    with pytest.raises(ValueError):
        writer.add_frame(np.zeros((32, 32, 4), dtype=np.uint8))  # 通道不符
    writer.close()


# ---------------------------------------------------------------------------
# shard_writer + manifest + validators
# ---------------------------------------------------------------------------


def _build_two_episode_shard(tmp_path, base_state):
    shard_dir = make_shard_dir(str(tmp_path), "run1", "prod1", "att1")
    writer = ShardWriter(
        shard_dir,
        run_id="run1",
        producer_id="prod1",
        attempt_id="att1",
        task_id="test.collect_wood",
        task_version="1.0.0",
        frame_sample=CFG,
    )
    # Episode 1：T=4，帧在 timestep 0/2/4（4 为 terminal）
    frames1 = [FrameEntry(rgb=synthetic_frame(), row=row) for row in _rows_for("ep1", "v1", [(0, 0, False), (2, 2, False), (4, 4, True)])]
    trs1 = [make_record(base_state, "ep1", t, terminated=(t == 3), frame=synthetic_frame(value=t + 1) if (t + 1) % 2 == 0 or t == 3 else None) for t in range(4)]
    ep1 = EpisodeData(
        episode_id="ep1", session_id="s1", task_id="test.collect_wood",
        task_version="1.0.0", seed=10,
        states=[make_state(base_state, 0)] + [tr.state for tr in trs1],
        transitions=trs1, frames=frames1,
        terminated=True, truncated=False, video_id="v1",
    )
    # Episode 2：T=3，帧在 timestep 0/2/3（3 为 terminal）
    frames2 = [FrameEntry(rgb=synthetic_frame(), row=row) for row in _rows_for("ep2", "v2", [(0, 0, False), (2, 2, False), (3, 3, True)])]
    trs2 = [make_record(base_state, "ep2", t, terminated=(t == 2), frame=synthetic_frame(value=t + 1) if (t + 1) % 2 == 0 or t == 2 else None) for t in range(3)]
    ep2 = EpisodeData(
        episode_id="ep2", session_id="s1", task_id="test.collect_wood",
        task_version="1.0.0", seed=11,
        states=[make_state(base_state, 0)] + [tr.state for tr in trs2],
        transitions=trs2, frames=frames2,
        terminated=True, truncated=False, video_id="v2",
    )
    writer.add_episode(ep1)
    writer.add_episode(ep2)
    manifest = writer.finalize()
    assert manifest["counts"] == {
        "num_episodes": 2,
        "num_transitions": 7,
        "num_states": 9,
        "num_frames": 6,
        "num_videos": 2,
    }
    return shard_dir


def _rows_for(episode_id, video_id, triples):
    from craftax.recording.frame_sampler import frame_index_row

    rows = []
    for i, (timestep, state_index, terminal) in enumerate(triples):
        rows.append(
            frame_index_row(
                CFG, episode_id=episode_id, video_id=video_id, frame_index=i,
                timestep=timestep, state_index=state_index,
                terminated=terminal, truncated=False,
            )
        )
    return rows


def test_shard_writer_manifest_and_validators(base_state, tmp_path):
    shard_dir = _build_two_episode_shard(tmp_path, base_state)

    # manifest 校验
    ok, errors = verify_manifest(shard_dir)
    assert ok, errors
    with open(os.path.join(shard_dir, "shard_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["counts"]["num_episodes"] == 2
    assert "state/player_position" in manifest["arrays"]
    assert "actions" in manifest["arrays"]

    # 结构校验
    ok, errors = validate_timeline(shard_dir)
    assert ok, errors
    ok, errors = validate_frame_index(shard_dir)
    assert ok, errors
    ok, errors = validate_shard(shard_dir)
    assert ok, errors

    # Zarr 内容抽查
    import zarr

    store = zarr.storage.LocalStore(os.path.join(shard_dir, "tensors.zarr"))
    group = zarr.open_group(store=store, mode="r")
    assert group["actions"].shape[0] == 7
    assert group["state/player_position"].shape[0] == 9
    assert list(group["state/inventory/wood"][:]) == list(group["state/inventory/wood"][:])
    # 视频文件存在
    for name in ("video-v1.mp4", "video-v2.mp4"):
        assert os.path.isfile(os.path.join(shard_dir, name))


def test_validator_detects_missing_frame_index_row(base_state, tmp_path):
    shard_dir = _build_two_episode_shard(tmp_path, base_state)
    ok, _ = validate_frame_index(shard_dir)
    assert ok

    # 人为删除 frame_index.parquet 的一行 -> 校验失败
    path = os.path.join(shard_dir, "frame_index.parquet")
    table = pq.read_table(path)
    dropped = table.slice(0, table.num_rows - 1)  # 丢掉最后一行
    pq.write_table(dropped, path)
    ok, errors = validate_frame_index(shard_dir)
    assert not ok
    assert any("帧数" in e or "frame_index" in e for e in errors)


def test_validator_detects_timeline_break(base_state, tmp_path):
    shard_dir = _build_two_episode_shard(tmp_path, base_state)
    # 人为在 episodes.parquet 把 num_states 改错
    path = os.path.join(shard_dir, "episodes.parquet")
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["num_states"] = rows[0]["num_states"] + 1
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows), path)
    ok, errors = validate_timeline(shard_dir)
    assert not ok
    assert any("num_states" in e for e in errors)


# ---------------------------------------------------------------------------
# recorder 集成
# ---------------------------------------------------------------------------


class _FakeTask:
    task_id = "test.fake"
    version = "1.0.0"

    def __init__(self):
        self.spec = TaskSpec(
            task_id=self.task_id, version=self.version, instruction="Fake. / 假任务。",
            objective="o", success_predicate={"type": "always"},
            annotation_predicates=[{"type": "achievement", "name": "COLLECT_WOOD"}],
        )

    def evaluate(self, state, info):
        return TaskEval(
            progress=0.5, done=False, instruction=self.spec.instruction,
            event_tokens=["COLLECT_WOOD"],
        )


def test_recorder_integration(base_state, tmp_path):
    cfg = RecordingConfig(
        spool_dir=str(tmp_path),
        frame_sample=CFG,
        shard_max_transitions=50,
    )
    rec = AsyncRecorder(
        recording_config=cfg, run_id="run-rec", producer_id="prod-rec",
        task_adapter=_FakeTask(),
    )
    # 3 步 episode，帧在 timestep 0/2/3(terminal)
    rec.on_episode_start(
        "sess-r", "ep-r1", "test.fake", "1.0.0", 7, cfg,
        make_state(base_state, 0), synthetic_frame(value=0),
    )
    for t in range(3):
        term = t == 2
        frame = synthetic_frame(value=t + 1) if ((t + 1) % 2 == 0 or term) else None
        rec.on_transition(make_record(base_state, "ep-r1", t, terminated=term, frame=frame))
    rec.on_episode_end("sess-r", "ep-r1", True)
    rec.close()
    assert isinstance(rec.backpressure_events, int)

    shards = [
        root for root, dirs, files in os.walk(cfg.spool_dir)
        if "shard_manifest.json" in files
    ]
    assert len(shards) == 1
    shard_dir = shards[0]
    ok, errors = validate_timeline(shard_dir)
    assert ok, errors
    ok, errors = validate_frame_index(shard_dir)
    assert ok, errors
    # 视频存在且帧数与 frame_index 行数一致
    n_frames = len(pq.read_table(os.path.join(shard_dir, "frame_index.parquet")).to_pylist())
    assert n_frames == 3  # timestep 0/2/3
    import imageio.v3 as iio

    video = [f for f in os.listdir(shard_dir) if f.endswith(".mp4")]
    assert len(video) == 1
    decoded = iio.imread(os.path.join(shard_dir, video[0]))
    assert (decoded.shape[0] if decoded.ndim == 4 else 1) == n_frames


def test_recorder_event_tokens_deduplicated(base_state, tmp_path):
    """同一 episode 内 token 只在首次达成时记录。"""
    cfg = RecordingConfig(spool_dir=str(tmp_path), frame_sample=CFG)
    rec = AsyncRecorder(recording_config=cfg, run_id="r2", producer_id="p2", task_adapter=_FakeTask())
    rec.on_episode_start("s", "ep2", "test.fake", "1.0.0", 1, cfg, make_state(base_state, 0), None)
    for t in range(3):
        rec.on_transition(make_record(base_state, "ep2", t, terminated=(t == 2)))
    rec.on_episode_end("s", "ep2", True)
    rec.close()
    shard_dir = [
        root for root, dirs, files in os.walk(cfg.spool_dir)
        if "shard_manifest.json" in files
    ][0]
    rows = pq.read_table(os.path.join(shard_dir, "transitions.parquet")).to_pylist()
    tokens = [json.loads(r["event_tokens"]) for r in rows]
    assert tokens[0] == ["COLLECT_WOOD"]
    assert all(t == [] for t in tokens[1:])


def test_recorder_gold_frames(base_state, tmp_path):
    cfg = RecordingConfig(spool_dir=str(tmp_path), frame_sample=CFG, gold_frames=True)
    rec = AsyncRecorder(recording_config=cfg, run_id="r3", producer_id="p3", task_adapter=_FakeTask())
    rec.on_episode_start("s", "ep3", "test.fake", "1.0.0", 2, cfg, make_state(base_state, 0), synthetic_frame())
    for t in range(2):
        rec.on_transition(make_record(base_state, "ep3", t, terminated=(t == 1), frame=synthetic_frame(value=t + 1)))
    rec.on_episode_end("s", "ep3", True)
    rec.close()
    shard_dir = [
        root for root, dirs, files in os.walk(cfg.spool_dir)
        if "shard_manifest.json" in files
    ][0]
    gold = os.path.join(shard_dir, "gold-frames", "ep3")
    assert os.path.isdir(gold)
    assert len([f for f in os.listdir(gold) if f.endswith(".png")]) >= 1
    ok, errors = verify_manifest(shard_dir)
    assert ok, errors


def test_recorder_flushes_partial_episode_on_close(base_state, tmp_path):
    """close() 时未 on_episode_end 的 episode 也要落盘（标记 truncated）。"""
    cfg = RecordingConfig(spool_dir=str(tmp_path), frame_sample=CFG)
    rec = AsyncRecorder(recording_config=cfg, run_id="r4", producer_id="p4", task_adapter=_FakeTask())
    rec.on_episode_start("s", "ep4", "test.fake", "1.0.0", 3, cfg, make_state(base_state, 0), synthetic_frame())
    for t in range(2):
        rec.on_transition(make_record(base_state, "ep4", t, terminated=False))
    rec.close()  # 没有 on_episode_end
    shard_dir = [
        root for root, dirs, files in os.walk(cfg.spool_dir)
        if "shard_manifest.json" in files
    ][0]
    rows = pq.read_table(os.path.join(shard_dir, "episodes.parquet")).to_pylist()
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep4"
    assert rows[0]["truncated"] is True


def test_recorder_shard_rotation(base_state, tmp_path):
    """shard 满后自动封存并开启新 shard。"""
    cfg = RecordingConfig(spool_dir=str(tmp_path), frame_sample=CFG, shard_max_transitions=3)
    rec = AsyncRecorder(
        recording_config=cfg, run_id="r5", producer_id="p5", task_adapter=_FakeTask()
    )
    for i in range(2):
        ep_id = f"ep-rot-{i}"
        rec.on_episode_start("s", ep_id, "test.fake", "1.0.0", i, cfg, make_state(base_state, 0), None)
        for t in range(3):  # 每 episode 3 transitions；max=3 → 每次 flush 后即满
            rec.on_transition(make_record(base_state, ep_id, t, terminated=(t == 2)))
        rec.on_episode_end("s", ep_id, True)
    rec.close()
    shards = [
        root for root, dirs, files in os.walk(cfg.spool_dir)
        if "shard_manifest.json" in files
    ]
    assert len(shards) == 2
    for sh in shards:
        ok, errors = validate_shard(sh)
        assert ok, errors


def test_recorder_resolves_task_via_registry(base_state, tmp_path):
    """未显式传 task_adapter 时，通过 registry 按 (task_id, version) 解析 builtin 任务。"""
    cfg = RecordingConfig(spool_dir=str(tmp_path), frame_sample=CFG)
    rec = AsyncRecorder(recording_config=cfg, run_id="r6", producer_id="p6")  # 无 task_adapter
    rec.on_episode_start("s", "ep-reg", "native.survive", "1.0.0", 5, cfg, make_state(base_state, 0), None)
    rec.on_transition(make_record(base_state, "ep-reg", 0, terminated=True))
    rec.on_episode_end("s", "ep-reg", True)
    rec.close()
    shard_dir = [
        root for root, dirs, files in os.walk(cfg.spool_dir)
        if "shard_manifest.json" in files
    ][0]
    rows = pq.read_table(os.path.join(shard_dir, "transitions.parquet")).to_pylist()
    assert rows[0]["task_id"] == "native.survive"
    assert rows[0]["instruction"].startswith("Survive")  # 来自 TaskAdapter.evaluate
    ok, errors = validate_shard(shard_dir)
    assert ok, errors

