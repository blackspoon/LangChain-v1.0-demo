
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage

# 主对话模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=20, # 限制每次生成不超过 20 tokens
)

# 摘要模型(这里也使用deepseek)
summ_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=20, # 限制每次生成不超过 20 tokens
)

# 会话摘要中间件：极低阈值，快速触发；摘要后仅保留 1 条原文
middleware = SummarizationMiddleware(
    model=summ_llm,
    max_tokens_before_summary=30,   # 累计上下文估算到阈值30时触发，把早期消息压缩成摘要。
    messages_to_keep=5,             # 摘要后仍保留的最近N条原文
    summary_prompt="用20个字以内概括要点。", # 自定义“如何摘要”的提示词（不设则用默认）
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="只用一句极短中文回答，且不超过30个字。",  # 只用“极简短句”回答
    middleware=[middleware],
)


conversation = [
    "RAG是什么？",
    "有何用途？",
    "主要缺点？",
    "一句话总结",
]

state = {"messages": []}

for i, question in enumerate(conversation, 1):
    # 用户问题
    user_msg = HumanMessage(content=question)
    # 智能体响应
    state = agent.invoke({"messages": state["messages"] + [user_msg]})
    answer = state["messages"][-1].content

    print(f"\n🧩 第 {i} 轮")
    print(f"Q: {question}")
    print(f"A: {answer}")

print("\n✅ 对话结束。")
