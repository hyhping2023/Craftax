"""EnvState（flax struct dataclass PyTree）<-> 扁平 dict[np.ndarray] 编解码。

扁平化规则：
- 叶节点为 jax/numpy 数组、Python/numpy 标量或同质 tuple；
- 标量转为 0-d 数组；tuple 尝试转数组（含 None 元素的 tuple 整字段跳过）；
- dataclass（含嵌套的 Inventory/Mobs）递归展开，字段路径用 "/" 分隔；
- None / NoneType 字段（如 EnvState.fractal_noise_angles 的 (None,...)）跳过。

unflatten_state 为反向重构（用于 replay 验证）。对扁平化时跳过的
None 字段（无法从数值数组中还原）报 NotImplementedError 并说明原因。
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

import numpy as np

# 扁平化时明确跳过的字段（因其携带 None，数值数组无法表示）。
# 后续如需序列化可改用 vlen/变长存储。
SKIPPED_NONE_FIELDS = ("fractal_noise_angles",)


def _flatten(value: Any, path: str, out: Dict[str, np.ndarray]) -> None:
    if value is None:
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            child_path = f"{path}/{field.name}" if path else field.name
            _flatten(child, child_path, out)
        return
    if isinstance(value, (tuple, list)):
        if any(v is None for v in value):
            return
        arr = np.asarray(value)
        if arr.dtype == object:
            # 无法用数值数组表示（如 (None, None, None, None) 之外的异构 tuple）
            return
        out[path] = arr
        return
    arr = np.asarray(value)
    out[path] = arr


def flatten_state(state: Any) -> Dict[str, np.ndarray]:
    """把 host 化 EnvState 递归展开为 {"path/to/field": np.ndarray}。"""
    out: Dict[str, np.ndarray] = {}
    _flatten(state, "", out)
    return out


def state_schema(flattened: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """由扁平化结果生成稳定 schema（对应 state-v1.json 结构）。"""
    fields = {
        path: {"dtype": str(arr.dtype), "shape": list(map(int, arr.shape))}
        for path, arr in sorted(flattened.items())
    }
    return {
        "schema_version": "1.0",
        "num_fields": len(fields),
        "fields": fields,
    }


def unflatten_state(
    flattened: Dict[str, np.ndarray],
    schema: Optional[Dict[str, Any]] = None,
    state_type: Any = None,
) -> Any:
    """反向重构 EnvState（或任意 dataclass 结构）。

    - 只使用 flattened 中出现的字段；缺失字段抛出 NotImplementedError。
    - state_type 默认 EnvState（惰性 import）。
    """
    if state_type is None:
        from craftax.craftax.craftax_state import EnvState

        state_type = EnvState

    missing = []
    for f in dataclasses.fields(state_type):
        if dataclasses.is_dataclass(f.type):
            if not any(k.startswith(f"{f.name}/") for k in flattened):
                missing.append(f.name)
        elif f.name not in flattened:
            missing.append(f.name)
    if missing:
        raise NotImplementedError(
            f"无法重构 {state_type.__name__}：字段 {missing} 在扁平化结果中缺失"
            f"（通常因字段值为 None 而被跳过，如 {SKIPPED_NONE_FIELDS}）"
        )

    kwargs: Dict[str, Any] = {}
    for field in dataclasses.fields(state_type):
        if dataclasses.is_dataclass(field.type):
            sub = {
                k[len(field.name) + 1 :]: v
                for k, v in flattened.items()
                if k.startswith(f"{field.name}/")
            }
            kwargs[field.name] = unflatten_state(sub, state_type=field.type)
        else:
            kwargs[field.name] = flattened[field.name]
    return state_type(**kwargs)


def schema_hash(flattened: Dict[str, np.ndarray]) -> str:
    """对 flatten+schema 的确定性 hash（shard manifest 用）。"""
    import hashlib
    import json

    schema = state_schema(flattened)
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
