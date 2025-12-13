# agent/graphs/risk_graph.py
"""风险评估子图：根据用户历史理赔行为评估风险等级"""
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END

from agent.logic.risk_rules import evaluate_risk, RiskResult

class RiskState(TypedDict, total=False):
    user_id: str
    total_claims_count: int
    recent_claims_count: int
    total_orders_count: int
    risk_level: str
    risk_reason: str
    logs: list[str]

def risk_node(state: RiskState) -> Dict[str, Any]:
    total_claims_count = int(state.get("total_claims_count", 0))
    recent_claims_count = int(state.get("recent_claims_count", 0))
    total_orders_count = int(state.get("total_orders_count", 1))

    risk: RiskResult = evaluate_risk(
        total_claims_count=total_claims_count,
        recent_claims_count=recent_claims_count,
        total_orders_count=total_orders_count,
    )

    return {
        "risk_level": risk.level,
        "risk_reason": risk.reason,
        "logs": [f"[risk] 风险评估完成，level={risk.level}"],
    }

def build_risk_graph():
    builder = StateGraph(RiskState)
    builder.add_node("risk_check", risk_node)
    builder.add_edge(START, "risk_check")
    builder.add_edge("risk_check", END)
    return builder.compile()
