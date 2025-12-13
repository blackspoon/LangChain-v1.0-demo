# agent/graphs/cases_graph.py
"""相似案例检索子图：从历史案例中检索相似案例供决策参考"""
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END

from agent.logic.cases_store import load_cases, pick_relevant_cases

class CasesState(TypedDict, total=False):
    order_amount: float
    order_category: str
    damage_description: str
    similar_cases_summary: str
    logs: list[str]

def cases_node(state: CasesState) -> Dict[str, Any]:
    cases = load_cases("data/cases_examples.json")
    picked = pick_relevant_cases(
        all_cases=cases,
        order_category=state.get("order_category", ""),
        order_amount=float(state.get("order_amount", 0.0)),
    )

    if not picked:
        summary = "暂无可参考的历史破损理赔案例。"
    else:
        lines = []
        for c in picked:
            lines.append(
                f"- 案例 {c.id}: 品类={c.order_category}, 金额≈{c.order_amount}, "
                f"处理={c.decision}。说明：{c.summary}"
            )
        summary = "\n".join(lines)

    return {
        "similar_cases_summary": summary,
        "logs": [f"[cases] 相似案例检索完成，找到 {len(picked)} 条候选。"],
    }

def build_cases_graph():
    builder = StateGraph(CasesState)
    builder.add_node("cases", cases_node)
    builder.add_edge(START, "cases")
    builder.add_edge("cases", END)
    return builder.compile()
