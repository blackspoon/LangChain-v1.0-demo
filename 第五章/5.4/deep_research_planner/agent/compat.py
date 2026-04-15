"""
兼容性补丁。

用于处理 langgraph 1.0.2 与 deepagents 0.2.7 的类型导入差异，
避免在导入 deepagents 时触发 ImportError。
"""

from importlib import import_module
from typing import Any, TypedDict


class _ExecutionInfo(TypedDict, total=False):
    """兼容旧版 langgraph.runtime.ExecutionInfo。"""


class _ServerInfo(TypedDict, total=False):
    """兼容旧版 langgraph.runtime.ServerInfo。"""


def apply_langgraph_runtime_compat() -> None:
    """
    在旧版 langgraph 中补齐 deepagents 需要的类型符号。

    这些符号在 deepagents 导入阶段仅用于类型引用，不参与运行时逻辑。
    """
    runtime: Any = import_module("langgraph.runtime")
    if not hasattr(runtime, "ExecutionInfo"):
        runtime.ExecutionInfo = _ExecutionInfo
    if not hasattr(runtime, "ServerInfo"):
        runtime.ServerInfo = _ServerInfo

    runtime_cls = getattr(runtime, "Runtime", None)
    if runtime_cls is not None:
        # 新版 prebuilt.tool_node 会读取 runtime.execution_info/server_info；
        # 旧版 Runtime 无此属性时提供空值以保持兼容。
        if not hasattr(runtime_cls, "execution_info"):
            runtime_cls.execution_info = property(lambda self: None)
        if not hasattr(runtime_cls, "server_info"):
            runtime_cls.server_info = property(lambda self: None)
