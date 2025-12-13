
# 虽然可以使用 init_chat_model，但为确保 DeepSeekChat 的准确性，我们保留其导入
from langchain_openai import ChatOpenAI
from langchain.tools import tool 
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# 1. 初始化 LLM（模型）
# 创建 DeepSeek 模型实例
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
    输入必须是一个有效的Python表达式字符串。"""
    try:
        # 简化演示，生产环境应使用更安全的计算库
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

# 工具二：查询城市信息
@tool
def get_info(city_name: str) -> str:
    """用于查询指定城市的基本信息，例如城市别名、主要产业等。"""
    city_data = {
        "深圳": "别称“鹏城”，是中国改革开放建立的第一个经济特区，以高新技术产业闻名。",
        "上海": "别称“沪”或“申”，是中国最大的城市和金融中心，拥有繁荣的港口贸易。",
        "北京": "中国的首都，拥有深厚的历史文化底蕴，是政治和文化中心。"
    }
    return city_data.get(city_name, f"未找到关于城市 '{city_name}' 的特定信息。")


# 3. 创建智能体
agent = create_agent(
    model=llm,
    tools=[calculate, get_info],
    system_prompt="你是一位专业的智能助理，拥有计算和信息查询的能力。请根据用户需求，自主选择最合适的工具进行处理。",
)

# 4. 执行任务一：选择工具一 (计算)
print("--- 任务一：智能体自主选择 'calculate' 工具 ---")
question_one = "请计算 (256 减去 88) 再乘以 5 的结果是多少？"

result_one = agent.invoke(
    {"messages": [HumanMessage(content=question_one)]}
)

print(f"用户问题: {question_one}")
print(f"智能体最终输出:\n{result_one['messages'][-1].content}")


# 5. 执行任务二：选择工具二 (查询城市信息)
print("\n--- 任务二：智能体自主选择 'get_info' 工具 ---")
question_two = "深圳的别称是什么？它以什么产业闻名？"

result_two = agent.invoke(
    {"messages": [HumanMessage(content=question_two)]}
)

print(f"用户问题: {question_two}")
print(f"智能体最终输出:\n{result_two['messages'][-1].content}")