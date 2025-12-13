# agent/state.py
from typing import Optional, List, Literal, TypedDict, Annotated

def append_logs(existing: List[str], new: List[str]) -> List[str]:
    return (existing or []) + (new or [])


class ClaimState(TypedDict, total=False):
    """破损理赔流程的统一 State 定义"""

    # 基本请求信息
    request_id: str
    user_id: str
    order_id: str
    order_amount: float
    order_category: str   # 商品品类，如 "electronics" / "fragile" / "virtual" 等
    damage_description: str

    # 风险评估用的 mock 统计数据
    total_claims_count: int
    recent_claims_count: int
    total_orders_count: int

    # 分支 A：规则检查结果
    policy_result: Optional[Literal["eligible", "ineligible"]]
    policy_reason: Optional[str]

    # 分支 B：风险评估结果
    risk_level: Optional[Literal["low", "medium", "high"]]
    risk_reason: Optional[str]

    # 分支 C：相似案例总结
    similar_cases_summary: Optional[str]

    # 推荐方案 & 审批
    proposed_solution: Optional[str]
    solution_reason: Optional[str]
    need_approval: Optional[bool]

    approval_decision: Optional[Literal["approve", "modify", "reject"]]
    approval_comment: Optional[str]

    # 最终方案 & 用户说明
    final_solution: Optional[str]
    user_notice: Optional[str]

    # 日志（用于 streaming updates）
    logs: Annotated[List[str], append_logs]