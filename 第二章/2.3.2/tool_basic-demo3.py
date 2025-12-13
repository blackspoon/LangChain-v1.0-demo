from langchain.tools import tool

@tool(
    "exchange_rate_helper",
    description="用于进行人民币与美元之间的汇率换算。当用户询问金额转换时使用此工具。"
)
def convert_currency(amount: float, to_currency: str) -> str:
    """执行基础的货币汇率换算（示例比例：1 USD = 7.2 CNY）。"""
    rate = 7.2
    if to_currency.lower() == "usd":
        result = amount / rate
        return f"{amount} 人民币 ≈ {result:.2f} 美元"
    elif to_currency.lower() == "cny":
        result = amount * rate
        return f"{amount} 美元 ≈ {result:.2f} 人民币"
    else:
        return "暂不支持该货币类型。"

# 查看工具信息
print(convert_currency.name)         # 输出：exchange_rate_helper
print(convert_currency.description)  # 输出：用于进行人民币与美元之间的汇率换算...
print(convert_currency.invoke({"amount": 100, "to_currency": "USD"}))  # 输出：100 人民币 ≈ 13.89 美元
