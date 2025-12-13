from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """返回两个整数相乘的结果"""
    return a * b

# 工具属性
print(multiply.name)         # 输出: "multiply"
print(multiply.description)  # 输出: "返回两个整数相乘的结果"
print(multiply.args)         # 输出: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}

# 手动调用
print(multiply.invoke({"a": 3, "b": 4}))  # 输出: 12
