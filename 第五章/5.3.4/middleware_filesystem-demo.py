"""
FilesystemMiddleware 示例：
让智能体通过虚拟文件系统工具写入并读取一个说明文件。
"""

import uuid
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage
from deepagents.middleware.filesystem import FilesystemMiddleware

# 1. 初始化模型
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

# 2. 创建带 FilesystemMiddleware 的 Agent
agent = create_agent(
    model=model,
    middleware=[
        FilesystemMiddleware(
            # 使用默认 backend（StateBackend），文件保存在图状态中，属于“短期文件系统”
            backend=None,
            system_prompt=(
                "当你需要为用户整理一份较长的说明或总结时，"
                "可以使用 write_file 将内容写入文件，再用 read_file 读取并总结给用户。"
            ),
        )
    ],
)

# 3. 发起请求：让模型创建一个 notes.txt 文件，然后再读取内容告诉我
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "请在虚拟文件系统中创建一个 notes.txt，"
                    "写入你对 LangGraph 核心概念的要点说明，"
                    "然后读取文件内容并向我总结其中的三点重点。"
                )
            )
        ]
    },
    config=config,
)

print("\n=== 最终回答 ===")
print(result["messages"][-1].content)
print("================\n")
