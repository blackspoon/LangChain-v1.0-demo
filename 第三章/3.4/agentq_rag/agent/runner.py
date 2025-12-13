from __future__ import annotations

"""
Agent runner 模块：
- 对 LangChain create_agent 进行最小封装，结合 MCP 工具与中间件
- 提供同步调用入口，方便在 Streamlit 等传统 Web 环境中复用
- 暴露 build_runner() 供应用层按需构建预配置的 AgentRunner
"""

import asyncio
from datetime import datetime
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

from agent.prompts import SYSTEM_PROMPT
from agent.mcp_tools import build_tools
from agent.memory import build_memory, ConversationWindow
from agent.middlewares import install_default_middlewares
from agent.company_kb import build_company_kb_tool


class AgentRunner:
    """
    一个最小可用的 Agent 调用器：
    - 使用 create_agent(model, tools, middlewares, system_prompt)
    - 通过 ConversationWindow 维护短期对话记忆（不落库）
    - 输入输出均走 Messages 机制（HumanMessage / AIMessage）
    """

    def __init__(self, *, agent, memory: ConversationWindow):
        self.agent = agent
        self.memory = memory

    def invoke(self, user_input: str) -> str:
        """单轮调用：拼接历史 + 本轮输入 -> 调用 Agent -> 回写记忆 -> 返回文本"""
        # 1) 拼接历史与当前输入（Messages 机制）
        history: List = self.memory.get()
        print(f"-----> Runner: History: {history}")
        state = {"messages": history + [HumanMessage(content=user_input)]}
        print(f"-----> Runner: State: {state}")

        # 2) 调用 Agent（内部自动处理ReAct、工具、中间件）
        # 使用asyncio.run()，它会自动处理事件循环的创建和关闭
        # 通过 config 设置递归限制，允许 Agent 调用更多工具
        config = {"recursion_limit": 50}
        new_state = asyncio.run(self.agent.ainvoke(state, config=config))
        print("--------------------------------")
        print(f"-----> Runner: New state: {new_state}")
        print("--------------------------------")

        # 3) 回写记忆为最新窗口
        # 只保存 HumanMessage 和最终的 AIMessage（没有 tool_calls 的），过滤掉 ToolMessage
        messages = new_state.get("messages", [])
        # 过滤消息：只保留 HumanMessage 和没有 tool_calls 的 AIMessage
        filtered_messages = [
            msg for msg in messages 
            if isinstance(msg, HumanMessage) or 
            (isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls))
        ]
        self.memory.clear()
        self.memory.add(filtered_messages)

        # 4) 返回本轮 AI 文本
        last = messages[-1] if messages else None
        return last.content if isinstance(last, AIMessage) else (str(last) if last is not None else "")

def build_runner() -> AgentRunner:
    """
    组装一个最简洁的 Runner：
    - DeepSeek 模型
    - MCP 工具（math + weather）
    - 中间件（预置 + 装饰器 + 类中间件，详见 middlewares.install_default_middlewares）
    - 短期记忆（InMemorySaver 封装的滑窗）
    """
    # app_streamlit.py 在页面初始化时会调用此方法，确保界面加载后即可复用同一套配置。
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key="xxxxxx",
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )
    # 原有 MCP 工具（math + weather）
    mcp_tools = build_tools()
    # 新增：公司规章制度 RAG 工具
    kb_tool = build_company_kb_tool()

    tools = mcp_tools + [kb_tool]

    # 中间件
    middlewares = install_default_middlewares()

    # 替换SYSTEM_PROMPT中的{today}占位符
    today_str = datetime.now().strftime("%Y年%m月%d日")
    system_prompt = SYSTEM_PROMPT.format(today=today_str)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middlewares,
    )

    memory = build_memory(window_size=5)
    return AgentRunner(agent=agent, memory=memory)