"""Host-side coordinates for a streamed, chunked world.

The JAX environment still steps a fixed-size active window.  This module keeps
the coordinate contract independent from that implementation detail so the
service can later swap windows without changing planner/API coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


CHUNK_SIZE = 16
DEFAULT_WINDOW_SIZE = 48
# Planning is allowed to ask for a wider view than the active JAX window.
# Keep a generous safety ceiling for accidental HTTP requests while allowing
# several surrounding chunks (the old 96-cell ceiling was too small for a
# planner that needs to expand its search radius).
MAX_WINDOW_SIZE = 512


@dataclass(frozen=True)
class ChunkCoord:
    floor: int
    x: int
    y: int


@dataclass(frozen=True)
class WorldOrigin:
    """Global coordinate of local map cell ``[0, 0]``."""

    floor: int = 0
    x: int = 0
    y: int = 0

    @property
    def chunk(self) -> ChunkCoord:
        return ChunkCoord(self.floor, self.x // CHUNK_SIZE, self.y // CHUNK_SIZE)


def global_position(origin: WorldOrigin, local_position: Sequence[int]) -> list[int]:
    return [origin.x + int(local_position[0]), origin.y + int(local_position[1])]


def chunk_coord(floor: int, global_x: int, global_y: int) -> ChunkCoord:
    return ChunkCoord(
        int(floor),
        int(np.floor_divide(global_x, CHUNK_SIZE)),
        int(np.floor_divide(global_y, CHUNK_SIZE)),
    )


def crop_window(
    grid: np.ndarray,
    center: Sequence[int],
    window_size: int,
    *,
    fill_value: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return a fixed-size window and its global/local-grid origin.

    ``center`` is expressed in the source grid's coordinates.  Out-of-bounds
    cells are padded with ``fill_value`` so planners always receive a stable
    shape, including near a world boundary.
    """
    size = int(window_size)
    if size <= 0 or size > MAX_WINDOW_SIZE:
        raise ValueError(f"window_size must be in [1, {MAX_WINDOW_SIZE}], got {size}")
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2-D, got shape {arr.shape}")
    start = (int(center[0]) - size // 2, int(center[1]) - size // 2)
    result = np.full((size, size), fill_value, dtype=arr.dtype)
    src_r0 = max(start[0], 0)
    src_c0 = max(start[1], 0)
    src_r1 = min(start[0] + size, arr.shape[0])
    src_c1 = min(start[1] + size, arr.shape[1])
    if src_r1 > src_r0 and src_c1 > src_c0:
        dst_r0 = src_r0 - start[0]
        dst_c0 = src_c0 - start[1]
        result[dst_r0 : dst_r0 + (src_r1 - src_r0),
               dst_c0 : dst_c0 + (src_c1 - src_c0)] = arr[src_r0:src_r1, src_c0:src_c1]
    return result, start
