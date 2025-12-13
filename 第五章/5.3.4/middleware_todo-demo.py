"""
TodoListMiddleware 示例：
让智能体在处理多步骤任务时，先写 TODO 再执行。
"""

import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage

# 1. 初始化模型
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

# 2. 创建带 TodoListMiddleware 的 Agent
agent = create_agent(
    model=model,
    middleware=[
        TodoListMiddleware(
            system_prompt=(
                "面对多步骤任务时，先调用 write_todos 写出待办列表，"
                "然后根据 TODO 逐步执行，并在任务完成时更新 TODO。"
            )
        )
    ],
)

# 3. 发起一个明显是“多步骤”的请求
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "帮我规划一个为期三天的 LangGraph 学习计划："
                    "包括每天的学习目标和要完成的任务，并先给出你的 TODO 列表，再执行第一天的任务。"
                )
            )
        ]
    },
    config=config,
)

print("\n=== 最终回答 ===")
print(result["messages"][-1].content)
print("================\n")
