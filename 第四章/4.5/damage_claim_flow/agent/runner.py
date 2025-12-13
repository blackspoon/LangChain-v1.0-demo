# agent/runner.py
from typing import Dict, Any, Iterator
from langgraph.types import Command
from agent.graphs.main_graph import build_main_graph
from agent.persistence import get_checkpointer

# 构建并编译图（应用级别单例）
_builder = build_main_graph()
_app = _builder.compile(checkpointer=get_checkpointer())

def get_app():
    return _app

def invoke_once(initial_state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    """执行一次图。如果中间遇到 interrupt，会在返回结果中包含 __interrupt__ 字段。"""
    config = {"configurable": {"thread_id": thread_id}}
    result: Dict[str, Any] = _app.invoke(initial_state, config=config)
    return result

def resume_with_decision(decision_payload: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    """当 approval_node 触发 interrupt 后，用人工决策恢复执行。"""
    config = {"configurable": {"thread_id": thread_id}}
    cmd = Command(resume=decision_payload)
    result: Dict[str, Any] = _app.invoke(cmd, config=config)
    return result

def stream_updates(initial_state: Dict[str, Any], thread_id: str) -> Iterator[Dict[str, Any]]:
    """使用 stream_mode='updates' 进行 streaming，可用于 CLI 或前端进度展示。"""
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in _app.stream(initial_state, config=config, stream_mode="updates"):
        yield chunk
