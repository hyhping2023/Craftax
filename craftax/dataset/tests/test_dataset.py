"""dataset 模块测试：基于合成 sealed shard fixture（不依赖 recording 模块）。"""

import json
import tarfile
from pathlib import Path

import imageio.v2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from craftax.contracts import (
    EPISODES_PARQUET_COLUMNS,
    EPISODES_PARQUET_FILENAME,
    FRAME_INDEX_COLUMNS,
    FRAME_INDEX_PARQUET_FILENAME,
    SHARD_MANIFEST_FILENAME,
    TRANSITIONS_PARQUET_COLUMNS,
    TRANSITIONS_PARQUET_FILENAME,
    VIDEO_FILENAME_PREFIX,
    VIDEO_FILENAME_SUFFIX,
    ZARR_DIRNAME,
    ZARR_STATE_GROUP,
)
from craftax.dataset.export_webdataset import export_webdataset
from craftax.dataset.reader import DatasetReader, ShardReader
from craftax.dataset.vla_windows import vla_samples
from craftax.dataset.world_model_windows import wm_samples

ACTION_NAMES = [
    "NOOP",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "JUMP",
    "USE",
    "DROP",
    "TOGGLE",
    "INTERACT",
]

EPISODE_SPECS = [
    {
        "episode_id": "ep-0000",
        "video_id": "v0",
        "task_id": "collect_wood",
        "task_version": "1.0.0",
        "seed": 101,
        "session_id": "sess-1",
        "instruction": "Collect wood.",
        "frame_base": 20,
    },
    {
        "episode_id": "ep-0001",
        "video_id": "v1",
        "task_id": "mine_stone",
        "task_version": "1.0.0",
        "seed": 202,
        "session_id": "sess-1",
        "instruction": "Mine stone.",
        "frame_base": 120,
    },
]

NUM_TRANSITIONS = 5  # 每 episode 5 个 transition（T=5，state 0..5）
FRAME_SIZE = 32
FRAME_TIMESTEPS = [0, 2, 4, 5]  # R=2 常规采样帧 + terminal frame


def _frame_rgb(base: int, frame_index: int) -> np.ndarray:
    img = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    img[..., 0] = base + frame_index * 10
    img[..., 1] = base
    img[..., 2] = 255 - frame_index * 10
    return img


def _events_for(spec: dict, t: int) -> list:
    if spec["task_id"] == "collect_wood" and t == 2:
        return ["COLLECT_WOOD"]
    if spec["task_id"] == "mine_stone" and t == 3:
        return ["MINE_STONE"]
    return []


def _write_video(path: Path, base: int) -> None:
    writer = imageio.v2.get_writer(
        str(path), fps=10, codec="libx264", quality=8, pixelformat="yuv420p"
    )
    for fi in range(len(FRAME_TIMESTEPS)):
        writer.append_data(_frame_rgb(base, fi))
    writer.close()


def _write_parquet(path: Path, columns: list, rows: list) -> None:
    table = pa.table({col: [r[col] for r in rows] for col in columns})
    pq.write_table(table, str(path))


def build_synthetic_shard(shard_dir: Path) -> None:
    """构造一个最小 sealed shard（契约见 craftax.contracts，不依赖 recording 模块）。

    2 个 episode × 5 transition；状态数组 shape [12, ...]；R=2 采样帧
    （timestep 0,2,4 + terminal frame 5）。
    """
    shard_dir.mkdir(parents=True, exist_ok=True)

    state_x_rows: list = []
    state_y_rows: list = []
    state_timesteps: list = []
    state_episode_ids: list = []
    transitions: list = []
    episode_rows: list = []
    frame_rows: list = []
    global_state = 0

    for ordinal, spec in enumerate(EPISODE_SPECS):
        state_start = global_state
        for t in range(NUM_TRANSITIONS + 1):
            state_x_rows.append(
                np.array([global_state * 10 + k for k in range(4)], dtype=np.int32)
            )
            state_y_rows.append(
                np.array([global_state * 100, global_state * 100 + 7], dtype=np.int32)
            )
            state_timesteps.append(t)
            state_episode_ids.append(ordinal)
            global_state += 1
        state_end = global_state - 1

        for t in range(NUM_TRANSITIONS):
            terminated = t == NUM_TRANSITIONS - 1
            # transition t 携带 state[t+1] 的帧；state[t+1] 在采样周期上或有终局帧
            sampled = (t + 1) % 2 == 0 or (t + 1) == NUM_TRANSITIONS
            transitions.append(
                {
                    "session_id": spec["session_id"],
                    "episode_id": spec["episode_id"],
                    "timestep": t,
                    "action_id": ordinal * NUM_TRANSITIONS + t,
                    "action_name": ACTION_NAMES[ordinal * NUM_TRANSITIONS + t],
                    "action_source": "random",
                    "command_id": f"cmd-{ordinal}-{t}",
                    "reward": float(t) * 0.1 + ordinal,
                    "terminated": terminated,
                    "truncated": False,
                    "is_sampled_frame": sampled,
                    "is_initial_frame": False,
                    "is_terminal_frame": terminated,
                    "instruction": spec["instruction"],
                    "task_id": spec["task_id"],
                    "task_version": spec["task_version"],
                    "event_tokens": json.dumps(_events_for(spec, t)),
                    "sim_time_ns": t * 50_000_000,
                }
            )

        episode_rows.append(
            {
                "session_id": spec["session_id"],
                "episode_id": spec["episode_id"],
                "task_id": spec["task_id"],
                "task_version": spec["task_version"],
                "seed": spec["seed"],
                "num_states": NUM_TRANSITIONS + 1,
                "num_transitions": NUM_TRANSITIONS,
                "num_frames": len(FRAME_TIMESTEPS),
                "terminated": True,
                "truncated": False,
                "video_id": spec["video_id"],
                "state_start": state_start,
                "state_end": state_end,
                "start_wall_ns": 0,
                "end_wall_ns": 1_000_000_000,
            }
        )

        for fi, ts in enumerate(FRAME_TIMESTEPS):
            frame_rows.append(
                {
                    "episode_id": spec["episode_id"],
                    "video_id": spec["video_id"],
                    "frame_index": fi,
                    "timestep": ts,
                    "state_index": state_start + ts,  # Zarr 全局偏移（与 recording 输出一致）
                    "sim_time_ns": ts * 50_000_000,
                    "is_initial_frame": fi == 0,
                    "is_terminal_frame": ts == NUM_TRANSITIONS,
                    "encoder_pts": fi * (1.0 / 10.0),
                }
            )

        _write_video(
            shard_dir
            / f"{VIDEO_FILENAME_PREFIX}{spec['video_id']}{VIDEO_FILENAME_SUFFIX}",
            spec["frame_base"],
        )

    # ---- tensors.zarr（v3 格式；读取路径兼容 v2/v3）----
    zarr_dir = shard_dir / ZARR_DIRNAME
    group = zarr.open_group(str(zarr_dir), mode="w")
    state_group = group.create_group(ZARR_STATE_GROUP)
    state_group.create_array(
        "x", data=np.stack(state_x_rows), chunks=(NUM_TRANSITIONS + 1, 4)
    )
    state_group.create_array(
        "y", data=np.stack(state_y_rows), chunks=(NUM_TRANSITIONS + 1, 2)
    )
    group.create_array(
        "actions",
        data=np.asarray([r["action_id"] for r in transitions], dtype=np.int32),
    )
    group.create_array(
        "rewards",
        data=np.asarray([r["reward"] for r in transitions], dtype=np.float32),
    )
    group.create_array(
        "terminated",
        data=np.asarray([r["terminated"] for r in transitions], dtype=bool),
    )
    group.create_array(
        "truncated",
        data=np.asarray([r["truncated"] for r in transitions], dtype=bool),
    )
    group.create_array(
        "state_timesteps",
        data=np.asarray(state_timesteps, dtype=np.int32),
    )
    group.create_array(
        "state_episode_ids",
        data=np.asarray(state_episode_ids, dtype=np.int32),
    )
    event_tokens = group.create_array(
        "event_tokens", shape=(len(transitions),), dtype="str"
    )
    event_tokens[:] = [r["event_tokens"] for r in transitions]

    # ---- parquet ----
    _write_parquet(
        shard_dir / EPISODES_PARQUET_FILENAME, EPISODES_PARQUET_COLUMNS, episode_rows
    )
    _write_parquet(
        shard_dir / TRANSITIONS_PARQUET_FILENAME,
        TRANSITIONS_PARQUET_COLUMNS,
        transitions,
    )
    _write_parquet(
        shard_dir / FRAME_INDEX_PARQUET_FILENAME, FRAME_INDEX_COLUMNS, frame_rows
    )

    # ---- manifest（内容 hash 在 fixture 中省略；读取路径不依赖 hash 校验）----
    manifest = {
        "shard_id": "shard-test-0001",
        "schema_version": 1,
        "num_episodes": len(EPISODE_SPECS),
        "num_states": global_state,
        "num_transitions": len(transitions),
        "num_frames": len(frame_rows),
        "video_fps": 10,
        "step_rate_hz": 20,
        "note": "test fixture; content hashes omitted (reader does not depend on them)",
    }
    (shard_dir / SHARD_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthetic_shard")
    build_synthetic_shard(Path(d))
    return Path(d)


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def test_manifest_and_episode_iteration(shard_dir):
    reader = ShardReader(shard_dir)
    assert reader.manifest["shard_id"] == "shard-test-0001"
    assert len(reader) == 2
    eps = list(reader.episodes())
    assert [e.episode_id for e in eps] == ["ep-0000", "ep-0001"]

    ep0 = reader.get_episode("ep-0000")
    assert ep0.task_id == "collect_wood"
    assert ep0.task_version == "1.0.0"
    assert ep0.seed == 101
    assert ep0.num_states == 6
    assert ep0.num_transitions == 5
    assert ep0.num_frames == 4
    assert ep0.state_start == 0 and ep0.state_end == 5

    ep1 = reader.get_episode("ep-0001")
    assert ep1.state_start == 6 and ep1.state_end == 11

    with pytest.raises(KeyError):
        reader.get_episode("no-such-episode")


def test_state_window_reading(shard_dir):
    reader = ShardReader(shard_dir)
    ep0 = reader.get_episode("ep-0000")
    win = ep0.states(0, 3)
    assert set(win) == {"x", "y"}
    assert win["x"].shape == (3, 4)
    assert win["y"].shape == (3, 2)
    np.testing.assert_array_equal(win["x"][0], [0, 1, 2, 3])
    np.testing.assert_array_equal(win["x"][2], [20, 21, 22, 23])
    np.testing.assert_array_equal(win["y"][2], [200, 207])

    # end 默认到 episode 末；区间半开 [start, end)
    assert ep0.states(2)["x"].shape == (4, 4)
    assert ep0.states(2, 3)["x"].shape == (1, 4)
    # 越界裁剪
    assert ep0.states(4, 99)["x"].shape == (2, 4)
    # 空区间
    assert ep0.states(3, 3)["x"].shape == (0, 4)

    ep1 = reader.get_episode("ep-0001")
    np.testing.assert_array_equal(ep1.states(0, 2)["x"][0], [60, 61, 62, 63])


def test_state_at_timestep(shard_dir):
    reader = ShardReader(shard_dir)
    ep0 = reader.get_episode("ep-0000")
    s = ep0.state_at_timestep(2)
    np.testing.assert_array_equal(s["x"][0], [20, 21, 22, 23])
    assert ep0.state_at_timestep(6) is None

    batch = ep0.states_at_timesteps([1, 3])
    assert [b["x"][0][0] for b in batch] == [10, 30]
    assert ep0.states_at_timesteps([2, 99]) is None


def test_actions_and_transitions(shard_dir):
    reader = ShardReader(shard_dir)
    ep0 = reader.get_episode("ep-0000")
    ids, names = ep0.actions()
    assert ids.tolist() == [0, 1, 2, 3, 4]
    assert names == ["NOOP", "LEFT", "RIGHT", "UP", "DOWN"]

    assert ep0.action_at_timestep(0) == (0, "NOOP")
    assert ep0.action_at_timestep(4) == (4, "DOWN")
    assert ep0.action_at_timestep(5) is None  # 终局步无 action

    tr = ep0.transition_at_timestep(2)
    assert tr["reward"] == pytest.approx(0.2)
    assert tr["command_id"] == "cmd-0-2"
    assert tr["event_tokens"] == '["COLLECT_WOOD"]'

    table = ep0.transitions()
    assert table.num_rows == 5
    assert table["timestep"].to_pylist() == [0, 1, 2, 3, 4]

    ep1 = reader.get_episode("ep-0001")
    assert ep1.actions()[0][0] == 5
    assert ep1.action_at_timestep(0) == (5, "JUMP")
    assert ep1.instruction == "Mine stone."
    assert ep0.instruction == "Collect wood."


def test_frames_sequential_and_random_access(shard_dir):
    reader = ShardReader(shard_dir)
    ep0 = reader.get_episode("ep-0000")
    frames = list(ep0.frames())
    assert len(frames) == 4
    assert [f.frame_index for f in frames] == [0, 1, 2, 3]
    assert [f.timestep for f in frames] == [0, 2, 4, 5]
    for f in frames:
        assert f.rgb.shape == (FRAME_SIZE, FRAME_SIZE, 3)
        assert f.rgb.dtype == np.uint8

    # 随机访问与顺序解码结果一致
    f2 = ep0.frame(2)
    np.testing.assert_array_equal(f2.rgb, frames[2].rgb)

    # 帧内容与合成视频一致（H.264/yuv420p 有损编码，允许小容差）
    ep1 = reader.get_episode("ep-0001")
    for fi, f in enumerate(ep1.frames()):
        diff = np.abs(f.rgb.astype(int) - _frame_rgb(120, fi).astype(int))
        assert diff.max() <= 5

    assert ep0.frame_at_timestep(4) is not None
    assert ep0.frame_at_timestep(1) is None
    with pytest.raises(KeyError):
        ep0.frame(99)


# ---------------------------------------------------------------------------
# vla_windows
# ---------------------------------------------------------------------------


def test_vla_samples(shard_dir):
    reader = ShardReader(shard_dir)
    samples = list(vla_samples(reader, window_len=3))
    # 每 episode 4 帧：起点 0（帧 0,1,2）有效；起点 1 含终局帧（ts=5）被跳过
    assert len(samples) == 2

    s = samples[0]
    assert s["episode_id"] == "ep-0000"
    assert s["task_id"] == "collect_wood"
    assert s["instruction"] == "Collect wood."
    assert len(s["frames"]) == 3
    assert all(f.shape == (FRAME_SIZE, FRAME_SIZE, 3) for f in s["frames"])
    assert s["timesteps"] == [0, 2, 4]
    assert [a.id for a in s["actions"]] == [0, 2, 4]
    assert [a.name for a in s["actions"]] == ["NOOP", "RIGHT", "DOWN"]
    assert s["action_names"] == ["NOOP", "RIGHT", "DOWN"]
    assert s["rewards"] == pytest.approx([0.0, 0.2, 0.4])
    assert s["terminated"] is True  # 窗口末帧 ts=4 的 transition terminated=True

    s1 = samples[1]
    assert s1["episode_id"] == "ep-0001"
    assert s1["timesteps"] == [0, 2, 4]
    assert [a.id for a in s1["actions"]] == [5, 7, 9]
    assert s1["rewards"] == pytest.approx([1.0, 1.2, 1.4])


def test_vla_samples_frame_stride(shard_dir):
    reader = ShardReader(shard_dir)
    samples = list(vla_samples(reader, window_len=4, frame_stride=2))
    assert len(samples) == 2
    s = samples[0]
    assert s["timesteps"] == [0, 4]  # 窗口帧 0..3，stride 2 取帧 0、2
    assert [a.id for a in s["actions"]] == [0, 4]
    assert len(s["frames"]) == 2

    # 窗口大于帧数 -> 无样本
    assert list(vla_samples(reader, window_len=8)) == []
    with pytest.raises(ValueError):
        list(vla_samples(reader, window_len=0))
    with pytest.raises(ValueError):
        list(vla_samples(reader, window_len=2, frame_stride=0))


# ---------------------------------------------------------------------------
# world_model_windows
# ---------------------------------------------------------------------------


def test_wm_samples_anchored_on_sampled_frames(shard_dir):
    reader = ShardReader(shard_dir)
    samples = list(wm_samples(reader, window_len=2, require_frames=True))
    # 每 episode 帧 timesteps {0,2,4,5}，max_anchor = T-L = 3 -> 锚点 {0,2}
    assert len(samples) == 4

    s = samples[0]
    assert s["episode_id"] == "ep-0000"
    assert s["timesteps"] == [0, 1, 2]
    assert len(s["states"]) == 3
    assert len(s["actions"]) == 2
    assert len(s["next_states"]) == 2
    np.testing.assert_array_equal(s["states"][0]["x"][0], [0, 1, 2, 3])
    np.testing.assert_array_equal(s["next_states"][0]["x"][0], [10, 11, 12, 13])
    assert [a.id for a in s["actions"]] == [0, 1]
    assert s["rewards"] == pytest.approx([0.0, 0.1])
    assert s["events"] == [[], []]
    assert s["frame_timesteps"] == [0, 2]  # 帧只出现在 sampled timestep
    assert len(s["frames"]) == 2
    assert s["terminated"] is False

    # 全部锚点都落在 sampled timestep（frame_index 中存在）
    anchors = {(sm["episode_id"], sm["timesteps"][0]) for sm in samples}
    assert anchors == {
        ("ep-0000", 0),
        ("ep-0000", 2),
        ("ep-0001", 0),
        ("ep-0001", 2),
    }

    # 事件 token 解析
    by_anchor = {(sm["episode_id"], sm["timesteps"][0]): sm for sm in samples}
    assert by_anchor[("ep-0000", 2)]["events"] == [["COLLECT_WOOD"], []]
    assert by_anchor[("ep-0001", 2)]["events"] == [[], ["MINE_STONE"]]


def test_wm_samples_no_frames_required(shard_dir):
    reader = ShardReader(shard_dir)
    samples = list(wm_samples(reader, window_len=2, require_frames=False))
    # 每 episode 锚定每个 transition timestep 0..T-L（T=5, L=2 -> 0..3）
    assert len(samples) == 8
    assert [sm["timesteps"][0] for sm in samples[:4]] == [0, 1, 2, 3]
    # 帧对齐始终指向 sampled timestep
    for sm in samples:
        for t in sm["frame_timesteps"]:
            assert t in {0, 2, 4, 5}


# ---------------------------------------------------------------------------
# export_webdataset
# ---------------------------------------------------------------------------


def test_export_webdataset(shard_dir, tmp_path):
    reader = ShardReader(shard_dir)
    out = tmp_path / "wds"
    paths = export_webdataset(reader, out, "demo")
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.name.startswith("demo-") and p.name.endswith(".tar")
        with tarfile.open(p, mode="r") as tar:
            names = tar.getnames()
            assert len(names) == 3
            mp4_name = next(n for n in names if n.endswith(".mp4"))
            json_name = next(
                n
                for n in names
                if n.endswith(".json") and not n.endswith("actions.json")
            )
            actions_name = next(n for n in names if n.endswith(".actions.json"))

            mp4_bytes = tar.extractfile(mp4_name).read()
            assert len(mp4_bytes) > 0

            meta = json.loads(tar.extractfile(json_name).read())
            assert meta["episode"]["episode_id"] in {"ep-0000", "ep-0001"}
            assert meta["episode"]["num_transitions"] == 5
            assert meta["frame_index"]["num_frames"] == 4
            assert meta["frame_index"]["frames"][0]["timestep"] == 0

            actions = json.loads(tar.extractfile(actions_name).read())
            assert len(actions) == 5
            assert actions[0]["timestep"] == 0
            assert actions[4]["terminated"] is True
            assert all(isinstance(a["event_tokens"], list) for a in actions)

    # episode 过滤
    paths_one = export_webdataset(reader, out, "demo", episode_ids=["ep-0000"])
    assert len(paths_one) == 1


def test_sanitize_filename():
    from craftax.dataset.export_webdataset import sanitize_filename

    assert sanitize_filename("ep-0000") == "ep-0000"
    assert sanitize_filename("a b/c*d") == "a_b_c_d"
    assert sanitize_filename("") == ""


# ---------------------------------------------------------------------------
# dataset reader
# ---------------------------------------------------------------------------


def test_dataset_reader(shard_dir):
    ds = DatasetReader(shard_dir)  # dataset_root 本身即 shard 目录
    shard_ids = ds.list_shards()
    assert len(shard_ids) == 1
    reader = ds.open_shard(shard_ids[0])
    assert len(reader) == 2

    # parent 目录递归发现 shard
    ds_parent = DatasetReader(shard_dir.parent)
    assert shard_dir.name in ds_parent.list_shards()

    with pytest.raises(KeyError):
        ds.open_shard("no-such-shard")
