# agent/graphs/main_graph.py
"""
主图：破损理赔流程的核心工作流
流程：收集请求 -> 并行执行三个分支（规则检查/风险评估/案例检索）-> 汇总生成方案 -> 审批 -> 最终通知
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI

from agent.state import ClaimState
from agent.graphs.policy_graph import build_policy_graph
from agent.graphs.risk_graph import build_risk_graph
from agent.graphs.cases_graph import build_cases_graph
from agent.prompts import (
    SOLUTION_SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE_FOR_SOLUTION,
    FINAL_USER_NOTICE_SYSTEM_PROMPT,
)
from config import settings, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
)

# 预编译子图，提升执行效率
policy_app = build_policy_graph()
risk_app = build_risk_graph()
cases_app = build_cases_graph()

def collect_request(state: ClaimState) -> Dict[str, Any]:
    """受理理赔请求，可在这里做一些校验或补齐。"""
    logs = [f"[collect] 接收到理赔请求 request_id={state.get('request_id')}"]
    return {"logs": logs}

def run_policy_subgraph(state: ClaimState) -> Dict[str, Any]:
    sub_state = {
        "order_id": state.get("order_id", ""),
        "order_amount": state.get("order_amount", 0.0),
        "order_category": state.get("order_category", ""),
        "damage_description": state.get("damage_description", ""),
    }
    result = policy_app.invoke(sub_state)
    return {
        "policy_result": result.get("policy_result"),
        "policy_reason": result.get("policy_reason"),
        "logs": result.get("logs", []),
    }

def run_risk_subgraph(state: ClaimState) -> Dict[str, Any]:
    sub_state = {
        "user_id": state.get("user_id", ""),
        "total_claims_count": state.get("total_claims_count", 0),
        "recent_claims_count": state.get("recent_claims_count", 0),
        "total_orders_count": state.get("total_orders_count", 1),
    }
    result = risk_app.invoke(sub_state)
    return {
        "risk_level": result.get("risk_level"),
        "risk_reason": result.get("risk_reason"),
        "logs": result.get("logs", []),
    }

def run_cases_subgraph(state: ClaimState) -> Dict[str, Any]:
    sub_state = {
        "order_amount": state.get("order_amount", 0.0),
        "order_category": state.get("order_category", ""),
        "damage_description": state.get("damage_description", ""),
    }
    result = cases_app.invoke(sub_state)
    return {
        "similar_cases_summary": result.get("similar_cases_summary"),
        "logs": result.get("logs", []),
    }

def aggregate_and_propose(state: ClaimState) -> Dict[str, Any]:
    """汇总三个分支结果，生成推荐方案。
    
    如果规则检查不通过，直接拒绝；否则调用LLM生成推荐方案。
    根据风险等级和金额判断是否需要人工审批。
    """
    policy_result = state.get("policy_result")
    policy_reason = state.get("policy_reason", "")
    
    # 规则不通过时直接拒绝，不调用LLM
    if policy_result == "ineligible":
        logs = [
            "[aggregate] 规则检查不通过，直接拒绝理赔。",
            f"[aggregate] 拒绝原因: {policy_reason}",
        ]
        return {
            "proposed_solution": f"拒绝理赔。原因：{policy_reason}",
            "solution_reason": policy_reason,
            "need_approval": False,
            "logs": logs,
        }
    
    # 调用LLM生成推荐方案
    user_msg = USER_MESSAGE_TEMPLATE_FOR_SOLUTION.format(
        order_id=state.get("order_id", ""),
        order_amount=state.get("order_amount", 0.0),
        order_category=state.get("order_category", ""),
        damage_description=state.get("damage_description", ""),
        policy_result=state.get("policy_result", "unknown"),
        policy_reason=state.get("policy_reason", ""),
        risk_level=state.get("risk_level", "unknown"),
        risk_reason=state.get("risk_reason", ""),
        similar_cases_summary=state.get("similar_cases_summary", "（暂无）"),
    )

    prompt = (
        SOLUTION_SYSTEM_PROMPT.strip()
        + "\n\n下面是本次理赔的综合信息，请基于这些信息给出推荐方案：\n"
        + user_msg
    )

    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # 判断是否需要人工审批：高风险或高金额
    need_approval = False
    risk_level = state.get("risk_level") or "low"
    amount = float(state.get("order_amount", 0.0))

    if policy_result == "eligible" and (risk_level == settings.high_risk_level or amount >= settings.approval_amount_threshold):
        need_approval = True

    logs = [
        "[aggregate] 已汇总三条分支结果并生成推荐方案。",
        f"[aggregate] risk_level={risk_level}, amount={amount}, need_approval={need_approval}",
    ]

    return {
        "proposed_solution": text,
        "solution_reason": "综合规则、风险和历史案例自动生成的推荐方案。",
        "need_approval": need_approval,
        "logs": logs,
    }

def approval_node(state: ClaimState) -> Dict[str, Any]:
    """人工审批节点：如需审批则触发中断等待外部决策，否则自动跳过。
    
    审批决策格式：{"decision": "approve"|"modify"|"reject", "comment": "...", "override_solution": Optional[str]}
    """
    if not state.get("need_approval"):
        return {"logs": ["[approval] 本单无需人工审批，自动跳过。"]}

    payload = {
        "proposed_solution": state.get("proposed_solution"),
        "order_id": state.get("order_id"),
        "order_amount": state.get("order_amount"),
        "risk_level": state.get("risk_level"),
        "policy_result": state.get("policy_result"),
    }

    decision = interrupt(payload)  # type: ignore[assignment]
    decision_dict = decision or {}
    logs = [f"[approval] 收到人工审批结果: {decision_dict}"]

    final_solution_text = state.get("proposed_solution", "")
    override_solution = decision_dict.get("override_solution")
    decision_value = decision_dict.get("decision")
    
    # 处理审批结果：拒绝时设置拒绝方案，修改时使用覆盖方案
    if decision_value == "reject":
        final_solution_text = f"拒绝理赔。审批意见：{decision_dict.get('comment', '人工审批拒绝')}"
    elif override_solution:
        final_solution_text = override_solution

    return {
        "approval_decision": decision_value,
        "approval_comment": decision_dict.get("comment"),
        "proposed_solution": final_solution_text,  # 可能被人工修改
        "logs": logs,
    }

def finalize_or_skip(state: ClaimState) -> Dict[str, Any]:
    """生成最终用户通知：规则不通过或审批拒绝时生成拒绝通知，否则调用LLM生成友好通知。"""
    policy_result = state.get("policy_result")
    approval_decision = state.get("approval_decision")
    proposed_solution = state.get("proposed_solution", "")
    policy_reason = state.get("policy_reason", "")
    approval_comment = state.get("approval_comment", "")
    
    if policy_result == "ineligible":
        user_notice = f"很抱歉，您的理赔申请未能通过。{policy_reason}如有疑问，请联系客服。"
        logs = ["[finalize] 规则检查不通过，生成拒绝通知。"]
        return {
            "final_solution": proposed_solution,
            "user_notice": user_notice,
            "logs": logs,
        }
    
    if approval_decision == "reject":
        user_notice = f"很抱歉，您的理赔申请未能通过。{approval_comment or '人工审批拒绝'}如有疑问，请联系客服。"
        logs = ["[finalize] 人工审批拒绝，生成拒绝通知。"]
        return {
            "final_solution": proposed_solution,
            "user_notice": user_notice,
            "logs": logs,
        }
    
    # 正常流程：调用LLM生成友好通知
    prompt = (
        FINAL_USER_NOTICE_SYSTEM_PROMPT.strip()
        + "\n\n下面是内部视角的理赔决策说明，请将其改写成面向用户的友好通知：\n"
        + proposed_solution
    )

    resp = llm.invoke(prompt)
    user_notice = resp.content if hasattr(resp, "content") else str(resp)

    logs = ["[finalize] 最终理赔方案已确认并生成用户说明。"]

    return {
        "final_solution": proposed_solution,
        "user_notice": user_notice,
        "logs": logs,
    }

def build_main_graph() -> StateGraph:
    builder = StateGraph(ClaimState)

    # 节点注册
    builder.add_node("collect_request", collect_request)
    builder.add_node("policy_branch", run_policy_subgraph)
    builder.add_node("risk_branch", run_risk_subgraph)
    builder.add_node("cases_branch", run_cases_subgraph)
    builder.add_node("aggregate", aggregate_and_propose)
    builder.add_node("approval", approval_node)
    builder.add_node("finalize", finalize_or_skip)

    # START -> collect
    builder.add_edge(START, "collect_request")

    # collect -> 三个分支（并行）
    builder.add_edge("collect_request", "policy_branch")
    builder.add_edge("collect_request", "risk_branch")
    builder.add_edge("collect_request", "cases_branch")

    # 三分支 -> aggregate（汇总）
    builder.add_edge("policy_branch", "aggregate")
    builder.add_edge("risk_branch", "aggregate")
    builder.add_edge("cases_branch", "aggregate")

    # aggregate -> approval -> finalize -> END
    builder.add_edge("aggregate", "approval")
    builder.add_edge("approval", "finalize")
    builder.add_edge("finalize", END)

    return builder
