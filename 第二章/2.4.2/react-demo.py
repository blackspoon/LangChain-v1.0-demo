
from langchain_openai import ChatOpenAI
from langchain.tools import tool 
from langchain.agents import create_agent
from langchain.messages import HumanMessage 
from typing import List

# 1. 初始化 LLM（模型）
llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# 2. 定义工具（Tools）
# 工具一：数学计算器
@tool
def calculate(expression: str) -> str:
    """这是一个数学计算器。当需要计算数学表达式时调用此工具。
    输入必须是一个有效的Python表达式字符串，例如 '10 * 5 + 3'。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

# 工具二：查询汇率工具
@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """用于查询两种货币之间的实时汇率。返回 'from_currency' 兑换 'to_currency' 的比率。"""
    # 模拟实时汇率数据
    if from_currency == "USD" and to_currency == "CNY":
        return 7.21
    if from_currency == "EUR" and to_currency == "USD":
        return 1.08
    
    # 如果找不到汇率，则返回一个默认值
    return 1.0

# 遵照您的要求，工具列表包含两个工具
tools: List[tool] = [calculate, get_exchange_rate]

# 3. 创建 ReAct 智能体
agent_react = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一位专业的金融计算助理，必须使用工具进行汇率查询和数学运算。",
)

# 任务：一个需要汇率查询（获取数据）和乘法计算（处理数据）的复杂任务
complex_question = "如果当前美元兑人民币的汇率是实时汇率，那么 1500 美元可以兑换多少人民币？"

print("\n--- 任务三：多步 ReAct 机制演示（汇率查询 + 乘法计算） ---")
print(f"复杂问题: {complex_question}")

result_complex = agent_react.invoke(
    {"messages": [HumanMessage(content=complex_question)]}
)

print(f"智能体最终输出:\n{result_complex['messages'][-1].content}")
