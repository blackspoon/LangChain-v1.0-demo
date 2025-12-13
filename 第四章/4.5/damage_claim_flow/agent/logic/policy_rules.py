# agent/logic/policy_rules.py
from dataclasses import dataclass

@dataclass
class PolicyCheckResult:
    result: str  # "eligible" / "ineligible"
    reason: str

def evaluate_policy(
    *,
    order_category: str,
) -> PolicyCheckResult:
    """评估订单是否符合理赔规则。
    
    检查项：
    - 品类限制：虚拟商品不支持破损理赔
    
    返回：PolicyCheckResult(result="eligible"|"ineligible", reason="...")
    """
    if order_category.lower() == "virtual":
        return PolicyCheckResult(
            result="ineligible",
            reason="该商品为虚拟商品，不支持到货破损理赔。",
        )

    return PolicyCheckResult(
        result="eligible",
        reason="商品品类支持破损理赔。",
    )
