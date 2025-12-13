from langchain.messages import AIMessage, ToolMessage

# --- 1. 模拟模型决定调用工具 ---
# 假设模型被要求计算 "2 + 3"，并决定调用一个名为 'calculator' 的工具。
# 模型的回复 (AIMessage) 会包含 'tool_calls' 属性（此处简化为内容）
model_tool_call = AIMessage(
    content="", 
    # 实际在框架中，tool_calls 会包含 id 和 name
    tool_calls=[{"id": "call_123", "name": "calculator", "args": {"a": 2, "b": 3}}]
)

# --- 2. 模拟外部工具执行并返回结果 ---
calculation_result = "5" # 外部代码执行的结果

# --- 3. 创建 ToolMessage，将结果反馈给模型 ---
# ToolMessage 必须包含 tool_call_id，以便模型知道这个结果对应哪次调用
tool_feedback = ToolMessage(
    content=calculation_result,
    tool_call_id=model_tool_call.tool_calls[0]["id"]
)

# --- 4. 最终消息序列 (等待模型生成最终答案) ---
# 此时，消息列表会包含：
# [HumanMessage, AIMessage(Tool Call), ToolMessage(Result)]
# 框架将这个序列发送给模型，模型将基于 '5' 这个结果给出最终答案。
final_messages = [
    # ... 之前的 HumanMessage
    model_tool_call,
    tool_feedback
]

print("--- ToolMessage 反馈格式示例 ---")
print(f"类型: {tool_feedback.type}")
print(f"内容: {tool_feedback.content}")
print(f"关联ID: {tool_feedback.tool_call_id}")