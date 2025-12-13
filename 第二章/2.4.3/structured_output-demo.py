import json
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

# 1. 定义所需的输出结构 (Pydantic模型)
class UserProfile(BaseModel):
    """用于存储用户信息的结构。"""
    username: str = Field(description="用户的唯一用户名。")
    age: int = Field(description="用户的年龄，必须是整数。")
    is_active: bool = Field(description="用户的账户当前是否处于活跃状态。")

# 2. 辅助工具（保持一致性）
@tool
def dummy_tool(query: str) -> str:
    """一个占位符工具，确保智能体有可用的工具列表。"""
    return "Tool is available."

tools = [dummy_tool]
input_text = "请为我创建一个档案：用户名是 'Jane_D', 她今年 32 岁，账户当前是活跃状态。"

# --- 智能体: ToolStrategy (通用且兼容性强) ---
# 适用场景：当模型不支持原生JSON API，或需要强制通过 Tool Calling 机制输出结构时。
print(">>> 智能体: ToolStrategy")

llm_tool = ChatDeepSeek(
    model="deepseek-chat", 
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3
)

agent_tool = create_agent(
    model=llm_tool,
    tools=tools,
    # 核心：使用 ToolStrategy 封装模型
    response_format=ToolStrategy(UserProfile),
    system_prompt="你是一位通用信息提取助理，请使用 Tool Calling 机制提取信息。",
)

# 执行和结果
result_tool = agent_tool.invoke(
    {"messages": [HumanMessage(content=input_text)]}
)

structured_data = result_tool.get("structured_response")

print(f"类型: {type(structured_data).__name__}")
print(json.dumps(structured_data.model_dump(), indent=2, ensure_ascii=False))



