# agent/logic/cases_store.py
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os

@dataclass
class CaseExample:
    id: str
    order_category: str
    order_amount: float
    damage_description: str
    decision: str
    summary: str

def load_cases(filepath: str) -> List[CaseExample]:
    """从本地 JSON 文件中加载案例，没有文件时返回内置 demo。"""
    if not os.path.exists(filepath):
        return [
            CaseExample(
                id="demo-1",
                order_category="fragile",
                order_amount=199.0,
                damage_description="玻璃杯运输过程中破裂",
                decision="全额退款",
                summary="易碎品破损，直接全额退款。",
            ),
            CaseExample(
                id="demo-2",
                order_category="electronics",
                order_amount=999.0,
                damage_description="外包装压损，机器可正常使用",
                decision="退款 20% + 优惠券",
                summary="功能正常但外观受损，部分退款加券。",
            ),
        ]

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases: List[CaseExample] = []
    for item in raw:
        cases.append(
            CaseExample(
                id=str(item.get("id")),
                order_category=item.get("order_category", ""),
                order_amount=float(item.get("order_amount", 0.0)),
                damage_description=item.get("damage_description", ""),
                decision=item.get("decision", ""),
                summary=item.get("summary", ""),
            )
        )
    return cases

def pick_relevant_cases(
    *,
    all_cases: List[CaseExample],
    order_category: str,
    order_amount: float,
    max_cases: int = 3,
) -> List[CaseExample]:
    """从所有案例中选取最相似的案例。
    
    相似度计算：品类相同+10分，金额差异越小分数越高。
    返回按相似度排序的前max_cases个案例。
    """
    scored: List[Dict[str, Any]] = []
    for c in all_cases:
        score = 0.0
        if c.order_category == order_category:
            score += 10.0
        score -= abs(c.order_amount - order_amount) / max(order_amount, 1.0)
        scored.append({"case": c, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [x["case"] for x in scored[:max_cases]]
