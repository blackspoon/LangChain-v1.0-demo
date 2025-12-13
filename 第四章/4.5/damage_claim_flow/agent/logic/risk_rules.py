# agent/logic/risk_rules.py
from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["low", "medium", "high"]

@dataclass
class RiskResult:
    level: RiskLevel
    reason: str


def evaluate_risk(
    *,
    total_claims_count: int,
    recent_claims_count: int,
    total_orders_count: int,
) -> RiskResult:
    """评估用户理赔风险等级。
    
    规则：
    - high: 近期理赔≥3次 或 总理赔率>50%
    - medium: 近期理赔≥1次 或 总理赔率>20%
    - low: 其他情况
    """
    total_orders_count = max(total_orders_count, 1)
    claim_rate = total_claims_count / total_orders_count

    if recent_claims_count >= 3 or claim_rate > 0.5:
        return RiskResult(
            level="high",
            reason=f"近期待赔付次数较多（近期 {recent_claims_count} 次，总理赔率 {claim_rate:.0%}），疑似高风险用户。",
        )

    if recent_claims_count >= 1 or claim_rate > 0.2:
        return RiskResult(
            level="medium",
            reason=f"存在一定数量的历史理赔（近期 {recent_claims_count} 次，理赔率 {claim_rate:.0%}），需要适度关注。",
        )

    return RiskResult(
        level="low",
        reason="历史理赔记录较少，行为基本正常。",
    )
