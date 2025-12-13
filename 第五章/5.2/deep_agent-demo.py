"""
Deep Agent 示例代码

本示例演示如何使用 Deep Agents 框架创建一个具备自主规划和执行能力的智能 Agent。
该 Agent 能够：
- 使用 DeepSeek 模型作为底层推理引擎
- 集成自定义工具（如互联网搜索）
- 自主规划任务步骤并执行
- 将中间结果保存到虚拟文件系统以便后续参考

主要步骤：
1. 初始化 LLM 模型（DeepSeek）
2. 定义自定义工具（互联网搜索）
3. 编写系统提示词，指导 Agent 的行为
4. 创建 Deep Agent 实例
5. 调用 Agent 执行任务
"""

from typing import Literal

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from tavily import TavilyClient

# ==================== 步骤 1: 初始化 DeepSeek 模型 ====================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# ==================== 步骤 2: 定义自定义工具 ====================
# 初始化 Tavily 客户端，用于互联网搜索
# Tavily 是一个专门用于 AI 应用的搜索 API
tavily_client = TavilyClient(api_key="xxxxxx")

def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = False,
):
    """
    互联网搜索工具
    
    该工具封装了 Tavily 搜索功能，允许 Agent 在互联网上搜索信息。
    
    Args:
        query (str): 搜索查询字符串
        max_results (int): 返回的最大结果数量，默认为 3
        topic (Literal["general", "news"]): 搜索主题类型
            - "general": 通用搜索
            - "news": 新闻搜索
        include_raw_content (bool): 是否包含原始内容，默认为 False
    
    Returns:
        Tavily 搜索结果的字典，包含搜索结果列表和相关元数据
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# ==================== 步骤 3: 编写系统提示词 ====================
# 系统提示词定义了 Agent 的角色、能力和工作流程
# Deep Agent 会根据这些指令自主规划任务执行步骤
research_instructions = """
你是一个负责研究任务的 Deep Agent。

在处理复杂问题时，你需要：
1）先用待办工具规划要执行的步骤；
2）按步骤调用工具（例如 internet_search）收集信息；
3）必要时将中间结果写入虚拟文件系统以便反复查看；
4）最后输出结构化、清晰、可执行的中文结论。
"""

# ==================== 步骤 4: 创建 Deep Agent ====================
# 将 LLM 模型、自定义工具和系统提示词组合成一个完整的 Deep Agent
# Deep Agent 具备自主规划能力，可以：
# - 将复杂任务分解为多个步骤
# - 按顺序执行这些步骤
# - 使用工具收集信息
# - 管理中间状态和结果
agent = create_deep_agent(
    model=llm,
    tools=[internet_search],  # 额外挂载的自定义工具列表
    system_prompt=research_instructions,
)

# ==================== 步骤 5: 调用 Deep Agent 执行任务 ====================
if __name__ == "__main__":
    print("="*60)
    print("示例 1: 通用搜索")
    print("="*60 + "\n")
    
    # 示例 1: 使用 Deep Agent 进行通用信息搜索
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",  # 用户角色
                    "content": "请帮我简单调研一下 LangGraph 是什么，并用 3 条要点做中文总结。",
                }
            ]
        }
    )

    # Deep Agent 的最终回复一般在 messages 列表的最后一条
    # 前面的消息可能包含工具调用、中间结果等
    print(result["messages"][-1].content)
    
    print("\n" + "="*60)
    print("示例 2: 新闻搜索")
    print("="*60 + "\n")
    
    # 示例 2: 使用 Deep Agent 搜索最新新闻
    # 注意：Agent 会根据查询内容自动调用 internet_search 工具，并选择合适的搜索类型
    news_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我搜索一下最近关于人工智能的最新新闻，并总结 3 条要点。",
                }
            ]
        }
    )
    
    # 输出新闻搜索结果
    print(news_result["messages"][-1].content)
    
    print("\n" + "="*60)
    print("示例 3: 虚拟文件系统演示")
    print("="*60 + "\n")
    
    # 示例 3: 展示虚拟文件系统的使用
    # Deep Agent 内置了文件系统工具（write_file, read_file 等）
    # Agent 会自动使用这些工具保存和读取中间结果
    file_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请搜索 LangChain 的信息，将结果保存到文件 langchain.txt，然后读取该文件给我一个总结。",
                }
            ]
        }
    )
    
    print(file_result["messages"][-1].content)
    
    # 检查是否使用了文件系统工具
    used_tools = []
    for msg in file_result.get("messages", []):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get('name', '')
                if 'file' in tool_name.lower() or 'write' in tool_name.lower() or 'read' in tool_name.lower():
                    if tool_name not in used_tools:
                        used_tools.append(tool_name)
    
    if used_tools:
        print(f"\n使用的文件系统工具: {', '.join(used_tools)}")
        print("注意: 虚拟文件系统是内存中的，不会在磁盘上创建真实文件（langchain.txt）")
