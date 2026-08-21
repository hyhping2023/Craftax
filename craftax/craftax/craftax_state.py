from dataclasses import dataclass
from typing import Tuple, Any

import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    pickaxe: int
    sword: int
    bow: int
    arrows: int
    # 便携饮水次数（在水源旁 FILL_WATER，离开水源后 DRINK_WATER）。
    water: int
    armour: jnp.ndarray
    torches: int
    ruby: int
    sapphire: int
    potions: jnp.ndarray
    books: int


@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: jnp.ndarray
    mask: jnp.ndarray
    attack_cooldown: jnp.ndarray
    type_id: jnp.ndarray


@struct.dataclass
class EnvState:
    map: jnp.ndarray
    item_map: jnp.ndarray
    mob_map: jnp.ndarray
    light_map: jnp.ndarray
    down_ladders: jnp.ndarray
    up_ladders: jnp.ndarray
    chests_opened: jnp.ndarray
    monsters_killed: jnp.ndarray

    player_position: jnp.ndarray
    player_level: int
    player_direction: int

    # Intrinsics
    player_health: float
    player_food: int
    player_drink: int
    player_energy: int
    player_mana: int
    is_sleeping: bool
    is_resting: bool

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float
    player_recover_mana: float

    # Attributes
    player_xp: int
    player_dexterity: int
    player_strength: int
    player_intelligence: int

    inventory: Inventory

    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs

    mob_projectiles: Mobs
    mob_projectile_directions: jnp.ndarray
    player_projectiles: Mobs
    player_projectile_directions: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    potion_mapping: jnp.ndarray
    learned_spells: jnp.ndarray

    sword_enchantment: int
    bow_enchantment: int
    armour_enchantments: jnp.ndarray

    boss_progress: int
    boss_timesteps_to_spawn_this_round: int

    light_level: float

    achievements: jnp.ndarray

    state_rng: Any

    timestep: int

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    day_length: int = 300

    always_diamond: bool = False

    mob_despawn_distance: int = 14
    max_attribute: int = 5

    god_mode: bool = False

    # 口渴衰减倍率（1.0 = 原版：清醒每 ~21 步掉 1 drink）。<1 让水的压力线性变缓，
    # 用于长程具身任务——原版速率下满水 9 点只够约 190 步，2000 步的任务要被
    # 打断十余次去找水，"找水"会挤掉任务本身。默认保持 1.0，基准/RL 训练不受影响；
    # 具身层在 SessionActor 侧按 contracts.DEFAULT_THIRST_RATE 覆盖。
    thirst_rate: float = 1.0

    # 精力自然衰减倍率（1.0 = 原版）；具身长程会话由 contracts 覆盖为 0.25。
    energy_rate: float = 1.0

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (48, 48)
    num_levels: int = 9

    # Mobs
    max_melee_mobs: int = 3
    max_passive_mobs: int = 3
    max_growing_plants: int = 10
    max_ranged_mobs: int = 2
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3
