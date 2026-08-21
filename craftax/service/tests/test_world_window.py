from __future__ import annotations

import numpy as np

from craftax.service.world_window import (
    CHUNK_SIZE,
    ChunkCoord,
    WorldOrigin,
    chunk_coord,
    crop_window,
    global_position,
)
from craftax.service.chunk_store import ChunkStore, generate_chunk


def test_global_and_chunk_coordinates_support_negative_space():
    origin = WorldOrigin(floor=2, x=32, y=-16)
    assert global_position(origin, [4, 7]) == [36, -9]
    assert chunk_coord(2, -1, -17) == ChunkCoord(2, -1, -2)
    assert CHUNK_SIZE == 16


def test_crop_window_has_stable_shape_and_origin_at_edge():
    grid = np.arange(25, dtype=np.int32).reshape(5, 5)
    cropped, origin = crop_window(grid, [0, 0], 4, fill_value=-1)
    assert origin == (-2, -2)
    assert cropped.tolist() == [
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, 0, 1],
        [-1, -1, 5, 6],
    ]


def test_chunk_generation_is_deterministic_and_edits_persist():
    coord = ChunkCoord(1, -2, 3)
    a = generate_chunk(3017, coord)
    b = generate_chunk(3017, coord)
    assert np.array_equal(a.blocks, b.blocks)
    store = ChunkStore(3017)
    before = store.get(coord).revision
    store.set_block(coord, 5, 6, 123)
    assert store.get(coord).blocks[5, 6] == 123
    assert store.get(coord).revision == before + 1
    assert store.get(coord) is store.get(coord)


def test_chunk_window_round_trip_preserves_items():
    store = ChunkStore(9)
    blocks = np.full((16, 16), 7, dtype=np.int32)
    items = np.zeros((16, 16), dtype=np.int32)
    # A non-symmetric sentinel catches accidental row/column transposition
    # when the active window is persisted and stitched back from chunks.
    blocks[3, 11] = 5
    items[3, 4] = 12
    store.merge_window(0, 0, 0, blocks, items)
    restored_blocks, restored_items = store.render_window(0, 0, 0, 16)
    assert restored_blocks[3, 11] == 5
    assert restored_blocks[3, 4] == 7
    assert restored_items[3, 4] == 12


def test_chunk_windows_keep_absolute_block_identity_when_recentered():
    store = ChunkStore(3017)
    blocks = np.full((16, 16), 7, dtype=np.int32)
    blocks[10, 11] = 5
    store.merge_window(0, 0, 0, blocks)
    first, _ = store.render_window(0, 0, 0, 16)
    recentered, _ = store.render_window(0, 4, 5, 16)
    assert first[10, 11] == 5
    # Global (10, 11) is local (6, 6) after the window moves to (4, 5).
    assert recentered[6, 6] == 5
