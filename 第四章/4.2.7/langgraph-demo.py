from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict

import langchain
from langgraph.func import entrypoint, task
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Compatibility guard: some langchain builds don't expose `debug`.
if not hasattr(langchain, "debug"):
    langchain.debug = False

def _call_chat_model(
    *,
    model_name: str,
    base_url: str,
    api_key: Optional[str],
    system_message: str, 
    human_message: str
) -> str:
    """统一的 LLM 调用封装"""
    model = init_chat_model(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        model_provider="openai",
    )
    response = model.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
    )
    return response.content.strip()

def deepseek_chat(prompt: str, model_name: str = "deepseek-chat") -> str:
    """调用 DeepSeek 分析问题"""
    return _call_chat_model(
        model_name=model_name,
        base_url="https://api.deepseek.com",
        api_key="xxxxxx",
        system_message="你是任务分析助手，请用简明中文解释用户的问题并规划回答结构。",
        human_message=prompt
    )

def qwen_chat(prompt: str, model_name: str = "qwen-plus") -> str:
    """调用通义千问生成回答"""
    return _call_chat_model(
        model_name=model_name,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="xxxxxx",
        system_message="你是专业的知识问答助手，请根据提供的分析和上下文生成详细、准确的中文回答。",
        human_message=prompt
    )

# ========== 1. 工具函数 ==========
def _ensure_question_text(raw: Any) -> str:
    """提取问题文本"""
    if isinstance(raw, str):
        return raw

    if isinstance(raw, dict):
        for key in ("question", "topic", "prompt"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        joined = " ".join(str(v) for v in raw.values())
        if joined.strip():
            return joined

    return str(raw)

# ========== 2. 状态定义 ==========
@dataclass
class AssistantContext:
    """运行时上下文"""
    analysis_model: str = "deepseek-chat"
    answer_model: str = "qwen-plus"
    detail_style: str = "auto"  # auto | summary | deep

class QAState(TypedDict):
    """问答状态"""
    question: str  # 用户输入
    analysis: Annotated[List[str], operator.add]  # 分析结果
    plan: Annotated[List[str], operator.add]  # 回答计划
    draft_sections: Annotated[List[str], operator.add]  # 章节草稿
    answer: Annotated[List[str], operator.add]  # 最终回答
    need_summarize: bool  # 是否需要总结
    expected_sections: int  # 章节数
    completed_sections: Annotated[int, operator.add]  # 已完成章节数
    ready_for_summary: bool  # 是否可总结
    current_section: NotRequired[str]  # 当前章节标题

# ========== 3. 规划节点 ==========
@task
def split_question(question: str) -> List[str]:
    """拆分问题为多个关注点"""
    question_text = _ensure_question_text(question)
    normalized = question_text
    for sep in ["，", "。", "？", "、", ";", "；", ",", "?", "以及", "并且", "和", "与"]:
        normalized = normalized.replace(sep, "|")

    segments = [seg.strip() for seg in normalized.split("|") if seg.strip()]
    if not segments and question_text.strip():
        segments = [question_text.strip()]

    return segments[:5] if segments else ["核心概念", "关键要点", "典型应用"]

@task
def enrich_outline(segments: List[str]) -> List[str]:
    """生成章节标题"""
    outline = []
    for idx, seg in enumerate(segments, 1):
        outline.append(f"第{idx}节：{seg}")
    return outline

@entrypoint(checkpointer=InMemorySaver())
def planning_workflow(question: str):
    """规划工作流"""
    question_text = _ensure_question_text(question)
    segments = split_question(question_text).result()
    outline = enrich_outline(segments).result()
    return {"steps": outline}

# ========== 4. 节点定义 ==========
def analyze_question(state: QAState, runtime: Runtime[AssistantContext]) -> Dict[str, Any]:
    """分析问题并制定计划"""
    question = state["question"]
    detail_pref = runtime.context.detail_style
    
    if detail_pref == "summary":
        need_summarize = True
    elif detail_pref == "deep":
        need_summarize = False
    else:
        need_summarize = not any(k in question for k in ["详细", "深度", "长篇", "展开"])

    content = deepseek_chat(
        f"请分析以下问题的主题、信息类型，并给出回答思路：{question}",
        model_name=runtime.context.analysis_model,
    )

    return {
        "analysis": [content],
        "need_summarize": need_summarize,
        "completed_sections": 0,
        "ready_for_summary": need_summarize,
    }

def plan_with_functional_api(state: QAState) -> Dict[str, Any]:
    """调用规划工作流"""
    workflow_result = planning_workflow.invoke({"question": state["question"]})
    outline = workflow_result.get("steps", [])
    if not outline:
        outline = ["第1节：核心概念速览", "第2节：关键能力", "第3节：企业落地建议"]
    return {
        "plan": outline,
        "analysis": [f"规划输出 {len(outline)} 个章节"],
        "expected_sections": len(outline),
    }

def dispatch_sections(state: QAState) -> Dict[str, Any]:
    """章节分发节点"""
    return {}

def fan_out_sections(state: QAState):
    """并行发送章节任务"""
    sections = state.get("plan", [])
    sends = [Send("expand_section", {"question": state["question"], "current_section": section}) 
             for section in sections]
    sends.append(Send("collect_sections", {}))
    return sends

def expand_section(state: QAState, runtime: Runtime[AssistantContext]) -> Dict[str, Any]:
    """生成单个章节内容"""
    section = state.get("current_section", "章节")
    prompt = f"""
        用户问题：
        {state["question"]}

        分析依据：
        {''.join(state.get('analysis', []))}

        当前需要展开的章节：
        {section}

        已有章节草稿：
        {' | '.join(state.get('draft_sections', [])) or '（暂无）'}

        请输出本章节的核心内容，强调与原问题的关联。
    """
    detailed = qwen_chat(prompt, model_name=runtime.context.answer_model)
    return {
        "draft_sections": [f"{section}\n{detailed}"],
        "completed_sections": 1,
    }

def collect_sections(state: QAState) -> Dict[str, Any]:
    """收集章节结果"""
    expected = state.get("expected_sections", 0)
    completed = state.get("completed_sections", 0)
    return {"ready_for_summary": True} if expected == 0 or completed >= expected else {}

def summarize_answer(state: QAState, runtime: Runtime[AssistantContext]) -> Dict[str, Any]:
    """生成最终回答"""
    if not state["need_summarize"] and not state.get("ready_for_summary", False):
        return {}

    prompt = f"""
        用户问题：
        {state["question"]}

        问题分析：
        {'\n'.join(state.get('analysis', []))}

        章节规划：
        {'; '.join(state.get('plan', [])) or '（无显式章节）'}

        章节草稿：
        {'\n\n'.join(state.get('draft_sections', [])) or '（已启用简要模式）'}

        请基于以上上下文，输出结构化、清晰且自然的中文最终回答。
    """
    answer = qwen_chat(prompt, model_name=runtime.context.answer_model)
    return {"answer": [answer]}

# ========== 5. 条件边 ==========
def route_after_plan(state: QAState) -> Literal["dispatch", "summarize"]:
    """路由决策"""
    return "summarize" if state.get("need_summarize") else "dispatch"

# ========== 6. 构建图 ==========
graph = StateGraph(QAState, context_schema=AssistantContext)

# 节点注册
graph.add_node("analyze", analyze_question)
graph.add_node("planner", plan_with_functional_api)
graph.add_node("dispatch", dispatch_sections)
graph.add_node("expand_section", expand_section)
graph.add_node("collect_sections", collect_sections)
graph.add_node("summarize", summarize_answer)

# 边定义
graph.add_edge(START, "analyze")
graph.add_edge("analyze", "planner")
graph.add_conditional_edges("planner", route_after_plan, {"dispatch": "dispatch", "summarize": "summarize"})
graph.add_conditional_edges("dispatch", fan_out_sections)
graph.add_edge("expand_section", "collect_sections")
graph.add_edge("collect_sections", "summarize")
graph.add_edge("summarize", END)

# 编译图
app = graph.compile()

# ========== 7. 执行 ==========
def create_initial_state(question: str) -> QAState:
    """创建初始状态"""
    return {
        "question": question,
        "analysis": [],
        "plan": [],
        "draft_sections": [],
        "answer": [],
        "need_summarize": False,
        "expected_sections": 0,
        "completed_sections": 0,
        "ready_for_summary": False,
    }

if __name__ == "__main__":
    user_input = input("请输入一个问题：").strip()
    initial_state = create_initial_state(user_input)
    runtime_context = {
        "analysis_model": "deepseek-chat",
        "answer_model": "qwen-plus",
        "detail_style": "auto",
    }

    print("\n=== LangGraph 智能问答助理启动 ===\n")
    print(f"运行时上下文：{runtime_context}")

    result = app.invoke(initial_state, context=runtime_context)

    print("\n--- 最终输出 ---")
    for msg in result["answer"]:
        print(msg)
