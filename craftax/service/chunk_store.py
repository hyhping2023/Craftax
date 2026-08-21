"""Deterministic sparse chunk storage for the streamed-world backend.

This module is deliberately host-side: JAX receives only the active window.
Chunk generation is deterministic from ``(world_seed, floor, x, y)`` and edits
are retained when a chunk is unloaded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np

from craftax.craftax.constants import BlockType
from craftax.service.world_window import CHUNK_SIZE, ChunkCoord


@dataclass
class ChunkData:
    blocks: np.ndarray
    items: np.ndarray
    revision: int = 0
    generator: str = "craftax_smoothworld_v1"
    ladder_down: Tuple[int, int] | None = None
    ladder_up: Tuple[int, int] | None = None


def chunk_seed(world_seed: int, coord: ChunkCoord) -> int:
    raw = f"{int(world_seed)}:{coord.floor}:{coord.x}:{coord.y}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def generate_chunk(world_seed: int, coord: ChunkCoord) -> ChunkData:
    """Generate a deterministic chunk with the official Craftax generator.

    The active JAX environment still owns the initial 48x48 window.  Newly
    streamed regions are generated lazily here using the same smooth-world or
    dungeon routines as ``generate_world``.  A small NumPy fallback keeps the
    host service usable when JAX is unavailable (for example in a metadata
    reader process).
    """
    try:
        return _generate_craftax_chunk(world_seed, coord)
    except Exception:
        # Generation must never make a boundary crossing fatal.  The fallback
        # is deterministic and is intentionally kept compatible with old
        # recordings.
        return _generate_fallback_chunk(world_seed, coord)


def _generate_craftax_chunk(world_seed: int, coord: ChunkCoord) -> ChunkData:
    import jax
    import jax.numpy as jnp

    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from craftax.craftax.world_gen.world_gen import generate_dungeon, generate_smoothworld
    from craftax.craftax.world_gen.world_gen_configs import (
        ALL_DUNGEON_CONFIGS,
        ALL_SMOOTHGEN_CONFIGS,
    )

    static = StaticEnvParams(map_size=(CHUNK_SIZE, CHUNK_SIZE), num_levels=9)
    params = EnvParams()
    player = jnp.array([CHUNK_SIZE // 2, CHUNK_SIZE // 2], dtype=jnp.int32)
    rng = jax.random.PRNGKey(chunk_seed(world_seed, coord) & 0xFFFFFFFF)
    smooth_levels = {0: 0, 2: 1, 5: 2, 6: 3, 7: 4, 8: 5}
    dungeon_levels = {1: 0, 3: 1, 4: 2}
    if coord.floor in smooth_levels:
        config = jax.tree_util.tree_map(
            lambda value: value[smooth_levels[coord.floor]], ALL_SMOOTHGEN_CONFIGS
        )
        generated = generate_smoothworld(
            rng, static, player, config, params
        )
    elif coord.floor in dungeon_levels:
        config = jax.tree_util.tree_map(
            lambda value: value[dungeon_levels[coord.floor]], ALL_DUNGEON_CONFIGS
        )
        generated = generate_dungeon(
            rng, static, config
        )
    else:
        raise ValueError(f"invalid Craftax floor: {coord.floor}")
    blocks = np.array(generated[0], dtype=np.int32, copy=True)
    items = np.array(generated[1], dtype=np.int32, copy=True)
    # Keep chunk handoffs traversable.  The game generator remains responsible
    # for all terrain/resource semantics inside the chunk.
    blocks[0, :] = BlockType.PATH.value
    blocks[-1, :] = BlockType.PATH.value
    blocks[:, 0] = BlockType.PATH.value
    blocks[:, -1] = BlockType.PATH.value
    return ChunkData(
        blocks=blocks,
        items=items,
        generator="craftax_smoothworld_v1",
        ladder_down=tuple(np.asarray(generated[3], dtype=np.int32).tolist()),
        ladder_up=tuple(np.asarray(generated[4], dtype=np.int32).tolist()),
    )


def _generate_fallback_chunk(world_seed: int, coord: ChunkCoord) -> ChunkData:
    """Small deterministic fallback used only if the official generator fails."""
    rng = np.random.default_rng(chunk_seed(world_seed, coord))
    blocks = np.full((CHUNK_SIZE, CHUNK_SIZE), BlockType.GRASS.value, dtype=np.int32)
    items = np.zeros_like(blocks)
    if coord.floor > 0:
        blocks[:] = BlockType.STONE.value
        walls = rng.random(blocks.shape) < 0.18
        blocks[walls] = BlockType.WALL.value
    else:
        trees = rng.random(blocks.shape) < 0.08
        blocks[trees] = BlockType.TREE.value
        water = rng.random(blocks.shape) < 0.025
        blocks[water] = BlockType.WATER.value
    # Keep all four borders traversable for streaming handoff.
    blocks[0, :] = BlockType.PATH.value
    blocks[-1, :] = BlockType.PATH.value
    blocks[:, 0] = BlockType.PATH.value
    blocks[:, -1] = BlockType.PATH.value
    ore = rng.random(blocks.shape) < (0.035 if coord.floor > 0 else 0.02)
    blocks[ore & (blocks == BlockType.STONE.value)] = BlockType.IRON.value
    return ChunkData(blocks=blocks, items=items, generator="numpy_fallback_v1")


class ChunkStore:
    def __init__(self, world_seed: int):
        self.world_seed = int(world_seed)
        self._chunks: Dict[ChunkCoord, ChunkData] = {}
        # Sparse host-side entity snapshots.  Keys are floor and entity class;
        # values use absolute row/column coordinates and JAX mob attributes.
        self._entities: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        self._entity_ticks: Dict[int, int] = {}
        self._ladders: Dict[Tuple[int, str], List[List[int]]] = {}

    def get(self, coord: ChunkCoord) -> ChunkData:
        chunk = self._chunks.get(coord)
        if chunk is None:
            chunk = generate_chunk(self.world_seed, coord)
            self._chunks[coord] = chunk
            if chunk.ladder_down is not None:
                self.register_ladders(
                    coord.floor, "down",
                    [[coord.y * CHUNK_SIZE + chunk.ladder_down[0],
                      coord.x * CHUNK_SIZE + chunk.ladder_down[1]]],
                )
            if chunk.ladder_up is not None:
                self.register_ladders(
                    coord.floor, "up",
                    [[coord.y * CHUNK_SIZE + chunk.ladder_up[0],
                      coord.x * CHUNK_SIZE + chunk.ladder_up[1]]],
                )
        return chunk

    def hydrate_window(
        self, floor: int, origin_x: int, origin_y: int,
        blocks: np.ndarray, items: np.ndarray | None = None,
    ) -> None:
        """Seed the sparse store with an authoritative JAX-generated window."""
        # Discard lazily generated ladder metadata for this region before the
        # authoritative reset state is installed.
        for direction in ("up", "down"):
            points = self._ladders.get((int(floor), direction), [])
            self._ladders[(int(floor), direction)] = [
                p for p in points
                if not (origin_x <= p[0] < origin_x + np.asarray(blocks).shape[0]
                        and origin_y <= p[1] < origin_y + np.asarray(blocks).shape[1])
            ]
        self.merge_window(floor, origin_x, origin_y, blocks, items)

    def register_ladders(self, floor: int, direction: str, positions: Any) -> None:
        """Register absolute ladder coordinates for cross-chunk planning."""
        if direction not in {"up", "down"}:
            raise ValueError("ladder direction must be 'up' or 'down'")
        arr = np.asarray(positions)
        if arr.size == 0:
            return
        arr = arr.reshape((-1, 2))
        values = self._ladders.setdefault((int(floor), direction), [])
        for row, col in arr.tolist():
            point = [int(row), int(col)]
            if point not in values:
                values.append(point)

    def ladders(self, floor: int, direction: str | None = None) -> Dict[str, List[List[int]]]:
        directions = (direction,) if direction else ("up", "down")
        return {
            name: [list(p) for p in self._ladders.get((int(floor), name), [])]
            for name in directions
        }

    def nearest_ladder(self, floor: int, position: Any, direction: str) -> List[int] | None:
        points = self._ladders.get((int(floor), direction), [])
        if not points:
            return None
        row, col = map(int, np.asarray(position).tolist())
        return min(points, key=lambda p: abs(p[0] - row) + abs(p[1] - col))

    def route_between(self, start: Any, target: Any) -> List[List[int]]:
        """Return chunk-boundary waypoints for a global-coordinate objective."""
        current = np.asarray(start, dtype=np.int64).tolist()
        goal = np.asarray(target, dtype=np.int64).tolist()
        waypoints: List[List[int]] = [list(map(int, current))]
        while current != goal:
            if current[0] != goal[0]:
                current[0] += 1 if goal[0] > current[0] else -1
            elif current[1] != goal[1]:
                current[1] += 1 if goal[1] > current[1] else -1
            if current[0] % CHUNK_SIZE == 0 or current[1] % CHUNK_SIZE == 0 or current == goal:
                waypoints.append(list(map(int, current)))
        return waypoints

    def set_block(self, coord: ChunkCoord, row: int, col: int, block: int) -> None:
        if not (0 <= row < CHUNK_SIZE and 0 <= col < CHUNK_SIZE):
            raise ValueError("chunk-local coordinate out of range")
        chunk = self.get(coord)
        chunk.blocks[row, col] = int(block)
        chunk.revision += 1

    def loaded(self) -> Tuple[ChunkCoord, ...]:
        return tuple(sorted(self._chunks, key=lambda c: (c.floor, c.x, c.y)))

    def merge_window(
        self, floor: int, origin_x: int, origin_y: int,
        blocks: np.ndarray, items: np.ndarray | None = None,
    ) -> None:
        """Persist a local active-window block grid back into sparse chunks."""
        arr = np.asarray(blocks)
        if arr.ndim != 2:
            raise ValueError("blocks must be 2-D")
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                global_row = int(origin_x + r)
                global_col = int(origin_y + c)
                coord = ChunkCoord(
                    int(floor),
                    int(global_col // CHUNK_SIZE),
                    int(global_row // CHUNK_SIZE),
                )
                chunk = self.get(coord)
                lr = int(global_row % CHUNK_SIZE)
                lc = int(global_col % CHUNK_SIZE)
                value = int(arr[r, c])
                if int(chunk.blocks[lr, lc]) != value:
                    chunk.blocks[lr, lc] = value
                    chunk.revision += 1
                if items is not None:
                    item_value = int(np.asarray(items)[r, c])
                    if int(chunk.items[lr, lc]) != item_value:
                        chunk.items[lr, lc] = item_value
                        chunk.revision += 1

    def render_window(
        self, floor: int, origin_x: int, origin_y: int, size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stitch chunks into a fixed local window (blocks and items)."""
        result = np.full((size, size), BlockType.PATH.value, dtype=np.int32)
        item_result = np.zeros((size, size), dtype=np.int32)
        for r in range(size):
            for c in range(size):
                global_row = origin_x + r
                global_col = origin_y + c
                coord = ChunkCoord(
                    int(floor), global_col // CHUNK_SIZE, global_row // CHUNK_SIZE
                )
                chunk = self.get(coord)
                result[r, c] = chunk.blocks[global_row % CHUNK_SIZE, global_col % CHUNK_SIZE]
                item_result[r, c] = chunk.items[global_row % CHUNK_SIZE, global_col % CHUNK_SIZE]
        # Do not overwrite the center of a planner window. The center may
        # contain a real tree/ore/wall from the authoritative active window;
        # boundary handoff cells are made walkable explicitly by
        # SessionActor._maybe_shift_active_window().
        return result, item_result

    def save_entities(
        self, floor: int, origin_x: int, origin_y: int, entity_arrays: Dict[str, Any]
    ) -> None:
        """Replace the persisted snapshot for the active floor."""
        for kind, mob in entity_arrays.items():
            positions = np.asarray(mob["position"])
            health = np.asarray(mob["health"])
            mask = np.asarray(mob["mask"])
            cooldown = np.asarray(mob["attack_cooldown"])
            type_id = np.asarray(mob["type_id"])
            previous = self._entities.get((int(floor), kind), [])
            # Replace only the region being unloaded; preserve snapshots from
            # other chunks so returning to an older area restores its entities.
            records: List[Dict[str, Any]] = [
                record for record in previous
                if not (
                    origin_x <= record["position"][0] < origin_x + CHUNK_SIZE * 3
                    and origin_y <= record["position"][1] < origin_y + CHUNK_SIZE * 3
                )
            ]
            for i, alive in enumerate(mask):
                if not bool(alive):
                    continue
                records.append({
                    "position": [int(origin_x + positions[i, 0]), int(origin_y + positions[i, 1])],
                    "health": float(health[i]),
                    "attack_cooldown": int(cooldown[i]),
                    "type_id": int(type_id[i]),
                })
            self._entities[(int(floor), kind)] = records

    def load_entities(
        self, floor: int, origin_x: int, origin_y: int, size: int, kind: str
    ) -> List[Dict[str, Any]]:
        """Return persisted entities visible in the new active window."""
        records = self._entities.get((int(floor), kind), [])
        visible = []
        for record in records:
            r, c = record["position"]
            if origin_x <= r < origin_x + size and origin_y <= c < origin_y + size:
                copy = dict(record)
                copy["position"] = [r - origin_x, c - origin_y]
                visible.append(copy)
        return visible

    def tick_offscreen(self, floor: int, origin_x: int, origin_y: int, size: int) -> int:
        """Advance persisted entities outside the active window by one tick.

        This is deliberately lightweight host simulation: deterministic random
        walks, cooldown aging, and bounded despawning keep distant regions
        alive without allocating JAX arrays for every loaded chunk.
        """
        floor = int(floor)
        tick = self._entity_ticks.get(floor, 0) + 1
        self._entity_ticks[floor] = tick
        active = lambda p: origin_x <= p[0] < origin_x + size and origin_y <= p[1] < origin_y + size
        moved = 0
        for key, records in list(self._entities.items()):
            if key[0] != floor:
                continue
            kept = []
            for record in records:
                p = record["position"]
                if not active(p):
                    digest = hashlib.blake2b(
                        f"{self.world_seed}:{floor}:{key[1]}:{p[0]}:{p[1]}:{tick}".encode(),
                        digest_size=2,
                    ).digest()
                    direction = digest[0] % 5
                    if direction == 1:
                        p[0] += 1
                    elif direction == 2:
                        p[0] -= 1
                    elif direction == 3:
                        p[1] += 1
                    elif direction == 4:
                        p[1] -= 1
                    record["attack_cooldown"] = max(0, int(record["attack_cooldown"]) - 1)
                    moved += 1
                if abs(p[0] - origin_x) <= size * 8 and abs(p[1] - origin_y) <= size * 8:
                    kept.append(record)
            self._entities[key] = kept
        return moved
