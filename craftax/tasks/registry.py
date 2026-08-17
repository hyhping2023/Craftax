"""任务适配器注册表。

注册表以 (task_id, version) 为键保存工厂函数；get 时 version 必须精确匹配，
不匹配则报错并给出可用版本。默认注册 builtin 任务。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from craftax.contracts import TaskAdapter, TaskSpec

_AdapterFactory = Callable[[], TaskAdapter]
_REGISTRY: Dict[Tuple[str, str], _AdapterFactory] = {}
_builtin_loaded = False


def _ensure_builtin() -> None:
    global _builtin_loaded
    if not _builtin_loaded:
        # 触发 craftax.tasks.builtin 的注册逻辑（在模块底部调用 register）
        import craftax.tasks.builtin  # noqa: F401

        _builtin_loaded = True


def register(task_id: str, version: str, factory: _AdapterFactory) -> None:
    """注册任务适配器工厂。重复注册相同 (task_id, version) 报错。"""
    if not callable(factory):
        raise TypeError("factory 必须是返回 TaskAdapter 的可调用对象")
    key = (task_id, version)
    if key in _REGISTRY:
        raise ValueError(f"任务 {task_id}@{version} 已注册")
    _REGISTRY[key] = factory


def register_spec(spec: TaskSpec, factory: _AdapterFactory) -> None:
    """以 TaskSpec 的 task_id/version 注册。"""
    register(spec.task_id, spec.version, factory)


def get_task_adapter(task_id: str, version: str) -> TaskAdapter:
    """获取任务适配器；version 不匹配时抛 ValueError。"""
    _ensure_builtin()
    key = (task_id, version)
    factory = _REGISTRY.get(key)
    if factory is None:
        available = sorted(v for (tid, v) in _REGISTRY if tid == task_id)
        if available:
            raise ValueError(
                f"任务 {task_id!r} 版本 {version!r} 不匹配，可用版本：{available}"
            )
        raise KeyError(f"任务 {task_id!r} 未注册")
    return factory()


def list_task_ids() -> List[str]:
    _ensure_builtin()
    return sorted({task_id for (task_id, _) in _REGISTRY})


def list_versions(task_id: str) -> List[str]:
    _ensure_builtin()
    return sorted(v for (tid, v) in _REGISTRY if tid == task_id)
