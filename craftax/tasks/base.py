"""原生 Craftax 任务的只读标注层。

职责：
- 解析并求值 TaskSpec 中的 success_predicate / annotation_predicates；
- 提供 BaseTaskAdapter，实现 contracts.TaskAdapter 协议；
- 所有求值都是只读的：不修改 state / reward / 终局 / 动作。

谓词表达式（可序列化 dict）：
- {"type": "always"}                         恒真
- {"type": "never"}                          恒假
- {"type": "achievement", "name": "COLLECT_WOOD"}
    achievements 数组（state.achievements 或 info["achievements"]）对应下标为真
    （若 info 提供 "achievements_list" 则按名称列表判断）
- {"type": "field_ge", "path": "player_health", "value": 5}
    字段比较，path 用 "." 或 "/" 分隔（如 "inventory.wood"），支持
    field_ge / field_gt / field_le / field_lt / field_eq / field_ne
- {"type": "level_ge", "value": 8}            player_level 楼层推进（等价 field_ge）
- {"type": "and", "predicates": [...]}       全部为真
- {"type": "or", "predicates": [...]}        任一为真
- {"type": "not", "predicate": {...}}        取反

扩展点：通过 register_predicate(type_name, fn) 注册自定义谓词类型，
fn 签名为 fn(state, info, expr) -> bool。
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List

import numpy as np

from craftax.contracts import TaskEval, TaskSpec

# ---------------------------------------------------------------------------
# 谓词注册表与求值
# ---------------------------------------------------------------------------

_PredicateFn = Callable[[Any, Dict[str, Any], Dict[str, Any]], bool]
_PREDICATE_REGISTRY: Dict[str, _PredicateFn] = {}


def register_predicate(type_name: str, fn: _PredicateFn) -> None:
    """注册自定义谓词类型。fn(state, info, expr) -> bool。"""
    if type_name in _PREDICATE_REGISTRY:
        raise ValueError(f"谓词类型 {type_name!r} 已注册")
    _PREDICATE_REGISTRY[type_name] = fn


def _get_field(state: Any, path: str) -> Any:
    """沿 path（"." 或 "/" 分隔）读取字段；数组支持下标访问。"""
    value = state
    for segment in path.replace("/", ".").split("."):
        if isinstance(value, (list, tuple)):
            value = value[int(segment)]
        else:
            value = getattr(value, segment)
    return value


def _to_bool(value: Any) -> bool:
    arr = np.asarray(value)
    return bool(arr)


def _pred_always(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    return True


def _pred_never(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    return False


def _achievement_index(name: str) -> int:
    """成就名 -> achievements 数组下标（与 craftax.constants.Achievement 对应）。"""
    from craftax.craftax.constants import Achievement

    return Achievement[name].value


def _achievement_value(state: Any, info: Dict[str, Any], name: str) -> bool:
    idx = _achievement_index(name)
    # 优先级：info 携带的成就（list/数组）> EnvState.achievements
    if "achievements_list" in info:
        return name in info["achievements_list"]
    if "achievements" in info:
        arr = np.asarray(info["achievements"])
        if arr.ndim > 0 and idx < arr.shape[0]:
            return _to_bool(arr[idx])
        return False
    if dataclasses.is_dataclass(state) and hasattr(state, "achievements"):
        arr = np.asarray(state.achievements)
        if arr.ndim > 0 and idx < arr.shape[0]:
            return _to_bool(arr[idx])
        return False
    raise KeyError(
        f"无法定位成就 {name!r}：state 与 info 中都没有 achievements 数组/list"
    )


def _pred_achievement(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    name = str(expr["name"])
    return _achievement_value(state, info, name)


_FIELD_CMP_OPS = {
    "field_ge": lambda a, b: a >= b,
    "field_gt": lambda a, b: a > b,
    "field_le": lambda a, b: a <= b,
    "field_lt": lambda a, b: a < b,
    "field_eq": lambda a, b: a == b,
    "field_ne": lambda a, b: a != b,
}


def _pred_field(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    path = str(expr["path"])
    value = np.asarray(_get_field(state, path))
    target = np.asarray(expr["value"])
    op = _FIELD_CMP_OPS[expr["type"]]
    return _to_bool(op(value, target))


def _pred_level_ge(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    level = np.asarray(_get_field(state, "player_level"))
    return _to_bool(level >= int(expr["value"]))


def _pred_and(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    return all(eval_predicate(sub, state, info) for sub in expr["predicates"])


def _pred_or(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    return any(eval_predicate(sub, state, info) for sub in expr["predicates"])


def _pred_not(state: Any, info: Dict[str, Any], expr: Dict[str, Any]) -> bool:
    return not eval_predicate(expr["predicate"], state, info)


def _register_default_predicates() -> None:
    register_predicate("always", _pred_always)
    register_predicate("never", _pred_never)
    register_predicate("achievement", _pred_achievement)
    register_predicate("level_ge", _pred_level_ge)
    register_predicate("and", _pred_and)
    register_predicate("or", _pred_or)
    register_predicate("not", _pred_not)
    for type_name in _FIELD_CMP_OPS:
        register_predicate(type_name, _pred_field)


_register_default_predicates()


def eval_predicate(expr: Dict[str, Any], state: Any, info: Dict[str, Any]) -> bool:
    """求值一个谓词表达式。expr 为 None/空 dict 视为恒真。"""
    if not expr:
        return True
    type_name = str(expr["type"])
    fn = _PREDICATE_REGISTRY.get(type_name)
    if fn is None:
        raise ValueError(
            f"未知谓词类型 {type_name!r}。可用：{sorted(_PREDICATE_REGISTRY)}"
        )
    return bool(fn(state, info, expr))


def predicate_token(expr: Dict[str, Any]) -> str:
    """谓词表达式对应的稳定事件 token（annotation 用）。"""
    type_name = str(expr["type"])
    if type_name == "achievement":
        return str(expr["name"])
    if type_name in ("field_ge", "field_gt", "field_le", "field_lt", "field_eq", "field_ne"):
        return f"{expr['path']}_{type_name}_{expr['value']}"
    if type_name == "level_ge":
        return f"PLAYER_LEVEL_GE_{expr['value']}"
    return type_name.upper()


# ---------------------------------------------------------------------------
# 基础任务适配器
# ---------------------------------------------------------------------------


class BaseTaskAdapter:
    """TaskAdapter 的默认实现。

    子类只需覆盖 progress()（可选）与 success/annotation 所需的 spec 谓词。
    """

    task_id: str
    version: str
    spec: TaskSpec

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.task_id = spec.task_id
        self.version = spec.version

    def success(self, state: Any, info: Dict[str, Any]) -> bool:
        return eval_predicate(self.spec.success_predicate, state, info)

    def annotation_tokens(self, state: Any, info: Dict[str, Any]) -> List[str]:
        """annotation_predicates 中当前为真的谓词对应的事件 token。"""
        tokens: List[str] = []
        for expr in self.spec.annotation_predicates:
            if eval_predicate(expr, state, info):
                tokens.append(predicate_token(expr))
        return tokens

    def progress(self, state: Any, info: Dict[str, Any]) -> float:
        """默认 0/1（成功前 0，成功后 1）；子类可覆盖为连续进度。"""
        return 1.0 if self.success(state, info) else 0.0

    def instruction(self) -> str:
        return self.spec.instruction

    def evaluate(self, state: Any, info: Dict[str, Any]) -> TaskEval:
        """只读评估：不修改 state / reward / 终局 / 动作。"""
        return TaskEval(
            progress=float(np.clip(self.progress(state, info), 0.0, 1.0)),
            done=bool(self.success(state, info)),
            instruction=self.spec.instruction,
            event_tokens=self.annotation_tokens(state, info),
        )

    # -- 工具方法（builtin 任务复用） --------------------------------------

    @staticmethod
    def achievement_progress(names: List[str], state: Any, info: Dict[str, Any]) -> float:
        """一组成就中已达成比例，作为连续进度。"""
        if not names:
            return 0.0
        achieved = 0
        for name in names:
            try:
                if _achievement_value(state, info, name):
                    achieved += 1
            except KeyError:
                # 成就不存在于当前环境版本：跳过，不阻塞任务评估
                continue
        return achieved / len(names)

    @staticmethod
    def any_achievement(names: List[str], state: Any, info: Dict[str, Any]) -> bool:
        """任一成就达成即真。"""
        return BaseTaskAdapter.achievement_progress(names, state, info) > 0.0

    @staticmethod
    def clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))
