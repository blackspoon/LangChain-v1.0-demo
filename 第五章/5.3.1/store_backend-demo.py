from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore
from langchain_deepseek import ChatDeepSeek

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)
store = InMemoryStore()

agent = create_deep_agent(
    model=llm,
    backend=lambda rt: StoreBackend(rt),
    store=store,
    system_prompt="""
你有一个可持久化的文件系统。
请将用户的偏好信息写入 /profile/preferences.txt，
后续对话可以读取这个文件以保持一致的风格。
"""
)

# 第一次，在 thread A 里写入偏好
config_a = {"configurable": {"thread_id": "user-001-session-A"}}

result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我喜欢回答风格简洁、有条理，并尽量使用列表。",
            }
        ]
    },
    config=config_a,
)

print("【第一次写入偏好】")
print(result1["messages"][-1].content)

# 第二次，在 thread B 里读取同一个文件，验证跨线程持久化
config_b = {"configurable": {"thread_id": "user-001-session-B"}}

result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "现在请根据你之前记录的我的偏好，回答：什么是 Deep Agents？",
            }
        ]
    },
    config=config_b,
)

print("\n【跨线程读取偏好后的回答】")
print(result2["messages"][-1].content)
