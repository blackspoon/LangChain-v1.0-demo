"""
SubAgent（Dictionary-based）示例：
主 Agent 负责总体协调，research-subagent 负责深度研究。
"""

from typing import Literal

from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

# ========= 1. 初始化模型 =========
research_model = init_chat_model(
    model="deepseek-chat",
    temperature=0.3,
    api_key="xxxxxx",
    base_url="https://api.deepseek.com/v1",
    model_provider="openai"
)

model = init_chat_model(
    model="deepseek-reasoner",
    temperature=0.3,
    api_key="xxxxxx",
    base_url="https://api.deepseek.com/v1",
    model_provider="openai"
)

# ========= 2. 定义工具 =========
tavily_client = TavilyClient(api_key="xxxxxx")

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """互联网搜索工具：供子智能体调用"""
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# ========= 3. 定义字典型 SubAgent =========
research_subagent = {
    "name": "research-agent",
    "description": (
        "用于处理需要多轮搜索和综合分析的深入研究任务，"
        "比如“系统性梳理某个技术方向”、“对比多个方案的优劣”等。"
    ),
    "system_prompt": """
你是一名严谨的研究员，负责对某个主题进行深入调研并输出结构化结论。

你的工作流程：
1. 将问题拆分成若干可搜索的子问题；
2. 调用 internet_search 工具多次检索信息；
3. 对信息进行筛选、归纳和对比分析；
4. 最终输出：简短结论 + 关键发现要点 + 可执行建议。

输出要求：
- 使用中文回答；
- 先给出 3～5 条要点总结，再给出一个 2～3 段的综合说明；
- 不要直接输出原始搜索结果内容。
""",
    "tools": [internet_search],
    # 可选：给子智能体换一个更适合“研究”的模型
    "model": research_model,
}

subagents = [research_subagent]

# ========= 4. 构建 Deep Agent =========
agent = create_deep_agent(
    model=model,
    system_prompt="""
你是一个任务协调者，负责理解用户需求、拆分任务，
并在需要时将“深入研究类任务”交给 research-agent 子智能体处理。

当你觉得需要多次搜索、信息较多时，
请使用 task() 工具调用 research-agent，并等待其返回总结结果。
最后，你负责把结果用用户容易理解的方式表达出来。
""",
    tools=[],  # 也可以挂一些通用工具
    subagents=subagents,
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我系统性地调研一下 LangGraph 在企业级场景中的典型用法，并给出落地建议。",
                }
            ]
        }
    )
    print(result["messages"][-1].content)
