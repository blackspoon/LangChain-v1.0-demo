# agent/writer_graph.py
"""
基于 LangGraph 的写作子图。

将调研要点整理为结构化的 Markdown 报告，编译为 Runnable 供 CompiledSubAgent 使用。
"""

import operator

from typing import TypedDict, Annotated, List
from langchain.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek

from .prompts import WRITER_SUBAGENT_PROMPT


class WriterState(TypedDict):
    """写作子图的状态定义"""
    messages: Annotated[List[AnyMessage], operator.add]


# 写作专用模型实例
_writer_model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)


def writer_node(state: WriterState) -> WriterState:
    """写作节点：生成结构化报告"""
    messages = state["messages"]
    response = _writer_model.invoke(
        [SystemMessage(content=WRITER_SUBAGENT_PROMPT)] + messages
    )
    return {"messages": messages + [response]}


# 构建并编译 LangGraph 图
_graph_builder = StateGraph(WriterState)
_graph_builder.add_node("writer", writer_node)
_graph_builder.add_edge(START, "writer")
_graph_builder.add_edge("writer", END)

# 编译后的子智能体图
writer_agent = _graph_builder.compile()
