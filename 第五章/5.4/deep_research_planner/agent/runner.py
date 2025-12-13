# agent/runner.py
"""
Agent 构建与调用入口。

负责创建配置完备的 Deep Agent 实例，并提供调用和恢复执行的封装函数。
"""

from typing import Any, Dict, List

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_deepseek import ChatDeepSeek

from .backends import create_composite_backend
from .prompts import DEEP_RESEARCH_SYSTEM_PROMPT
from .subagents import build_subagents, internet_search


def build_deep_research_agent(project_root: str) -> Any:
    """构建并返回配置完备的 Deep Agent 实例"""
    # 配置主模型
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key="xxxxxx",
        temperature=0.3,
    )

    # 配置文件系统后端
    backend_factory, store = create_composite_backend(project_root)

    # 配置子智能体
    subagents = build_subagents()

    # 配置 Human-in-the-loop：对 write_file 操作启用人工审批
    interrupt_on: Dict[str, Any] = {
        "write_file": {
            "allowed_decisions": ["approve", "reject"],
        }
    }

    # 配置检查点保存器（Human-in-the-loop 必需）
    checkpointer = MemorySaver()

    # 创建并返回 Deep Agent
    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        backend=backend_factory,
        store=store,
        subagents=subagents,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )

    return agent


def invoke_with_hitl(
    agent: Any,
    user_input: str,
    thread_id: str,
) -> Dict[str, Any]:
    """首次调用 Agent，可能返回正常结果或中断结果"""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    return result


def resume_with_decisions(
    agent: Any,
    decisions: List[Dict[str, Any]],
    thread_id: str,
) -> Dict[str, Any]:
    """根据人工决策恢复 Agent 执行"""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )
    return result
