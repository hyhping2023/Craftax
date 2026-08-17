"""键盘 / 模型 / 回放控制映射。

键位表与 ``craftax/craftax/play_craftax.py`` 保持一致（不 import 该模块），
动作名以 ``craftax.craftax.constants.Action`` 为权威来源。
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

import pygame

from craftax.contracts import ActionSpec
from craftax.craftax.constants import Action

# 与 play_craftax.py 的 KEY_MAPPING 一致的键位表。
# 注意：GUI 层额外占用 R（reset）与 C（切换 controller），
# 二者不再映射为 PLACE_STONE / DRINK_POTION_BLUE。
KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_DIAMOND_PICKAXE,
    pygame.K_5: Action.MAKE_WOOD_SWORD,
    pygame.K_6: Action.MAKE_STONE_SWORD,
    pygame.K_7: Action.MAKE_IRON_SWORD,
    pygame.K_8: Action.MAKE_DIAMOND_SWORD,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_e: Action.REST,
    pygame.K_COMMA: Action.ASCEND,
    pygame.K_PERIOD: Action.DESCEND,
    pygame.K_y: Action.MAKE_IRON_ARMOUR,
    pygame.K_u: Action.MAKE_DIAMOND_ARMOUR,
    pygame.K_i: Action.SHOOT_ARROW,
    pygame.K_o: Action.MAKE_ARROW,
    pygame.K_g: Action.CAST_FIREBALL,
    pygame.K_h: Action.CAST_ICEBALL,
    pygame.K_j: Action.PLACE_TORCH,
    pygame.K_z: Action.DRINK_POTION_RED,
    pygame.K_x: Action.DRINK_POTION_GREEN,
    pygame.K_v: Action.DRINK_POTION_PINK,
    pygame.K_b: Action.DRINK_POTION_CYAN,
    pygame.K_n: Action.DRINK_POTION_YELLOW,
    pygame.K_m: Action.READ_BOOK,
    pygame.K_k: Action.ENCHANT_SWORD,
    pygame.K_l: Action.ENCHANT_ARMOUR,
    pygame.K_LEFTBRACKET: Action.MAKE_TORCH,
    pygame.K_RIGHTBRACKET: Action.LEVEL_UP_DEXTERITY,
    pygame.K_MINUS: Action.LEVEL_UP_STRENGTH,
    pygame.K_EQUALS: Action.LEVEL_UP_INTELLIGENCE,
    pygame.K_SEMICOLON: Action.ENCHANT_BOW,
}


class ControllerMode(Enum):
    """输入源模式。MODEL 模式下游戏动作键不生效，GUI 仅展示。"""

    HUMAN = "human"
    MODEL = "model"
    REPLAY = "replay"


def parse_controller(text: str) -> ControllerMode:
    """把 ``human`` / ``model`` / ``replay`` 文本解析为 ControllerMode。"""
    try:
        return ControllerMode(text.strip().lower())
    except ValueError:
        choices = ", ".join(m.value for m in ControllerMode)
        raise ValueError(f"unknown controller mode {text!r} (choices: {choices})")


def action_names() -> List[str]:
    """全部动作名（Action 枚举名顺序），用于控制面板展示。"""
    return [a.name for a in Action]


def key_to_action(key: int) -> Optional[ActionSpec]:
    """按键 -> 稳定动作。未映射键返回 None。"""
    action = KEY_MAPPING.get(key)
    if action is None:
        return None
    return ActionSpec(action.value, action.name)
