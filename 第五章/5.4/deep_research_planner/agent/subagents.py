# agent/subagents.py
"""
子智能体定义。

包含调研子智能体和写作子智能体，供主 Agent 通过 task() 工具调用。
"""

from typing import Any, Dict, List, Literal

from tavily import TavilyClient
from deepagents import CompiledSubAgent
from langchain_deepseek import ChatDeepSeek

from .writer_graph import writer_agent


# Tavily 互联网检索客户端
_tavily_client = TavilyClient(
    api_key="xxxxxx"
)


def internet_search(
    query: str,
    max_results: int = 2,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    """Tavily 互联网检索工具"""
    return _tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


def build_subagents() -> List[Any]:
    """构建子智能体列表"""
    # 调研子智能体模型
    research_model = ChatDeepSeek(
        model="deepseek-chat",
        api_key="xxxxxx",
        temperature=0.3,
    )

    # 调研子智能体：负责互联网检索和要点整理
    research_subagent: Dict[str, Any] = {
        "name": "research-subagent",
        "description": "负责互联网检索与技术资料调研的子智能体。使用 internet_search 工具进行检索，整理调研要点。",
        "system_prompt": (
            "你是一名资深技术研究员，擅长阅读官方网站、博客、技术文档和社区讨论，"
            "围绕指定主题进行系统化调研。\n\n"
            "## 工作流程\n"
            "1. 根据主 Agent 提供的调研主题，使用 internet_search 工具进行多轮检索\n"
            "2. 检索关键词应覆盖：技术概念、应用场景、最佳实践、案例研究、社区讨论等\n"
            "3. 对检索结果进行整理和分析\n\n"
            "## 输出要求\n"
            "请将调研结果整理为结构化的要点，包含以下内容：\n"
            "- **关键信息来源**: 标明重要信息来源（站点名、文档标题、作者等）\n"
            "- **核心观点**: 提炼关键观点和技术要点\n"
            "- **对比分析**: 如有多个方案或观点，进行对比分析\n"
            "- **重要数据**: 如有统计数据、性能指标等，请记录\n"
            "- **典型应用场景**: 总结典型的应用场景和实践模式\n"
            "- **潜在问题**: 记录发现的风险、局限或争议点\n\n"
            "## 输出格式\n"
            "使用 Markdown 格式输出，结构清晰，便于主 Agent 整理后写入 research_notes.md。"
            "每个要点都应标明信息来源，确保可追溯。"
        ),
        "model": research_model,
        "tools": [internet_search],
    }

    # 写作子智能体：使用编译好的 LangGraph writer_agent
    writer_subagent = CompiledSubAgent(
        name="writer-subagent",
        description="负责将调研要点整理为标准 Markdown 报告的写作子智能体。",
        runnable=writer_agent,
    )

    return [research_subagent, writer_subagent]
