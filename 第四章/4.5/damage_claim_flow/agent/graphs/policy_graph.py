# agent/graphs/policy_graph.py
"""规则检查子图：检查订单是否符合理赔规则（品类限制等）"""
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END

from agent.logic.policy_rules import evaluate_policy

class PolicyState(TypedDict, total=False):
    order_id: str
    order_amount: float
    order_category: str
    damage_description: str
    policy_result: str
    policy_reason: str
    logs: list[str]

def policy_node(state: PolicyState) -> Dict[str, Any]:
    """规则检查节点：使用本地规则引擎生成结果。"""
    rule_result = evaluate_policy(
        order_category=state.get("order_category", ""),
    )

    return {
        "policy_result": rule_result.result,
        "policy_reason": rule_result.reason,
        "logs": [f"[policy] 规则检查完成，result={rule_result.result}"],
    }

def build_policy_graph():
    builder = StateGraph(PolicyState)
    builder.add_node("policy_check", policy_node)
    builder.add_edge(START, "policy_check")
    builder.add_edge("policy_check", END)
    return builder.compile()
