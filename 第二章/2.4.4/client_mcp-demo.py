import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# 1. 初始化 LLM（模型）
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# 2. 连接 MCP 服务（math + weather）
client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["./math_mcp_server.py"],   # 确保路径正确
    },
    "weather": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",  # Weather MCP Server
    },
})

def arun(coro):
    """同步封装：把异步协程在顶层跑完，主逻辑仍然是“同步写法”"""
    return asyncio.run(coro)

# 3. 获取 MCP 工具（注意：此方法在你本地是 async）
tools = arun(client.get_tools())
print("✅ 已加载的工具：", [t.name for t in tools])

# 4. 创建 Agent（基于 MCP 工具 + DeepSeek 模型）
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一个助理。涉及数学计算，使用 math 工具（add / multiply）；"
        "涉及天气，使用 weather 工具（geocode_city / get_current_weather / get_current_weather_by_city）。"
    ),
)

# 5. 调用：数学示例
math_question = "请帮我计算 (3 + 5) × 12 的结果"
print("\n🔢 数学任务：", math_question)
math_result = arun(agent.ainvoke({"messages": [HumanMessage(content=math_question)]}))
print("✅ 智能体输出：", math_result["messages"][-1].content)

# 6. 调用：天气示例
weather_question = "请告诉我北京现在的天气情况"
print("\n🌤 天气任务：", weather_question)
weather_result = arun(agent.ainvoke({"messages": [HumanMessage(content=weather_question)]}))
print("✅ 智能体输出：", weather_result["messages"][-1].content)
