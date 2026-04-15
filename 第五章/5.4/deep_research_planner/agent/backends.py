# agent/backends.py
"""
文件系统后端配置。

为 Deep Agent 提供统一的文件系统视图，支持短期存储、长期记忆和本地文件系统。
"""

from pathlib import Path
from typing import Callable, Tuple

from .compat import apply_langgraph_runtime_compat

apply_langgraph_runtime_compat()

from deepagents.backends import (
    CompositeBackend,
    StateBackend,
    StoreBackend,
    FilesystemBackend,
)
from langgraph.store.memory import InMemoryStore


BackendFactory = Callable[[object], object]


def create_composite_backend(project_root: str) -> Tuple[BackendFactory, InMemoryStore]:
    """创建复合后端和存储实例"""
    root_path = Path(project_root).resolve()
    files_root = root_path / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    store = InMemoryStore()

    def backend_factory(runtime: object) -> CompositeBackend:
        """创建复合后端实例，路由不同路径到不同后端"""
        return CompositeBackend(
            default=StateBackend(runtime),  # 根路径：短期文件系统
            routes={
                "/memories/": StoreBackend(runtime),  # 长期记忆存储
                "/workspace/": FilesystemBackend(  # 本地文件系统映射
                    root_dir=str(files_root),
                    virtual_mode=True,
                ),
            },
        )

    return backend_factory, store
