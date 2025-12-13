from langchain.tools import tool

@tool
def process_user_data(user_id: str, is_active: bool) -> str:
    """
    根据用户ID和活动状态，更新用户记录。
    
    参数:
      user_id (str): 用户的唯一标识符。
      is_active (bool): 表示用户是否处于活动状态 (True/False)。
    """
    status = "激活" if is_active else "禁用"
    return f"用户ID: {user_id} 已成功更新为 {status} 状态。"

# --- 1. 使用 invoke 方法调用 Tool（传入字典） ---
# 注意：invoke 方法需要传入一个 Python 字典，而不是 JSON 字符串
tool_input_dict = {"user_id": "U001", "is_active": True}
print("--- 传入字典参数 ---")
result = process_user_data.invoke(tool_input_dict) 
print(f"调用结果: {result}") 
print("-" * 30)

# --- 2. 传入一个简化的字符串 ---
# 对于只需要一个简单字符串参数的工具，可以直接传入字符串
@tool
def simple_greeting(name: str) -> str:
    """对给定的人名说一句问候语。"""
    return f"你好，{name}！"

print("--- 传入简化的字符串 ---")
result_simple = simple_greeting.invoke("张三")
print(f"调用结果: {result_simple}")
