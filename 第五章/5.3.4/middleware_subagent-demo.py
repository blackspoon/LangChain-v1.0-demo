"""
SubAgentMiddleware 示例：
主智能体通过 task 工具，把“查询天气”的任务交给专门的 weather 子智能体。
"""

import uuid
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import SubAgentMiddleware

# 1. 定义子智能体要用到的工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    # 真实项目中可以调用天气 API，这里用固定文案示例
    return f"{city} 的天气是晴，气温 22℃。"

# 2. 初始化模型
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

# 3. 创建带 SubAgentMiddleware 的主智能体
agent = create_agent(
    model=model,
    middleware=[
        SubAgentMiddleware(
            # 新版 SubAgentMiddleware 需要显式提供 backend
            backend=FilesystemBackend(root_dir=".", virtual_mode=False),
            subagents=[
                {
                    "name": "weather",
                    "description": "专门负责查询城市天气的子智能体。",
                    "system_prompt": (
                        "你是一个天气查询助手，只负责回答与天气相关的问题。"
                        "遇到请求时，请使用 get_weather 工具获取天气信息。"
                    ),
                    "tools": [get_weather],
                    # 也可以在这里指定不同的模型，这里沿用同一个 model
                    "model": model,
                    # 子智能体本身也可以继续挂载 middleware，这里用空列表
                    "middleware": [],
                }
            ],
        )
    ],
)

# 4. 发起请求：让主智能体帮我查询北京天气
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="帮我看看北京今天的天气怎么样，需要不要带伞？"
            )
        ]
    },
    config=config,
)

print("\n=== 最终回答 ===")
print(result["messages"][-1].content)
print("================\n")
