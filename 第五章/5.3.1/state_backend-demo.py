from deepagents import create_deep_agent
from langchain_deepseek import ChatDeepSeek

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

# 使用 StateBackend（其实默认就是这个，不写 backend 也一样）
agent = create_deep_agent(
    model=llm,
    system_prompt="""
你是一个善于做规划的 Deep Agent。
请把你对当前任务的执行计划写入 /scratch/plan.md，然后基于该文件给我一个简要说明。
"""
)

config = {"configurable": {"thread_id": "state-demo-001"}}

# 第一次调用：让 Agent 生成计划文件
result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我为一个 Deep Agents 入门教程列一个章节计划。"
            }
        ]
    },
    config=config,
)

print("【第一次回答】")
print(result1["messages"][-1].content)

# 同一线程下，第二次调用：要求继续使用之前的 plan.md
result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请基于你之前写入的 /scratch/plan.md，帮我再精简一版目录。"
            }
        ]
    },
    config=config,
)

print("\n【第二次回答（复用 /scratch/plan.md）】")
print(result2["messages"][-1].content)
