from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

# 1. 定义 Agent 的 Context 结构
@dataclass
class UserContext:
    user_id: str = "GUEST_001"
    
# 2. 定义一个需要访问 Runtime Context 的工具
# ToolRuntime 是一个泛型，我们指定它携带 UserContext
@tool
def check_user_profile(runtime: ToolRuntime[UserContext]) -> str:
    """
    检查并返回当前用户的配置信息。
    必须在具有用户上下文的环境中调用。
    """
    # Tool 直接从 runtime 中访问 context 属性
    current_user_id = runtime.context.user_id
    
    if current_user_id == "GUEST_001":
        return "当前用户未登录，仅能执行公共查询。"
    else:
        # 实际操作中，工具会使用这个 ID 去查询数据库
        return f"用户 {current_user_id} 已登录，具有高级权限。"

# 3. 定义访问状态的工具
@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """获取当前对话中的消息数量"""
    messages = runtime.state.get("messages", [])
    return f"当前对话有 {len(messages)} 条消息"

# --- 4. 模拟 Runtime 环境并调用（演示概念） ---
print("--- 演示 ToolRuntime 的使用 ---\n")

# 在实际应用中，ToolRuntime 由 LangGraph 框架自动注入
# 这里我们手动构造所需的参数来模拟

# 模拟场景 1: 访问 Context
print("场景 1: 通过 runtime.context 访问用户上下文")
mock_context = UserContext(user_id="VIP_456")
# mock_context = UserContext(user_id="GUEST_001")
mock_state = {"messages": []}  # 模拟状态

# 创建模拟的 ToolRuntime（提供所有必需参数）
mock_runtime = ToolRuntime(
    context=mock_context,
    state=mock_state,
    config={},  # 配置对象
    stream_writer=lambda x: None,  # 流写入器（简单模拟）
    tool_call_id="mock_tool_call_001",  # 工具调用 ID
    store=None  # 持久化存储（可选）
)

result = check_user_profile.invoke({"runtime": mock_runtime})
print(f"✓ 工具返回: {result}\n")

# 模拟场景 2: 访问 State
print("场景 2: 通过 runtime.state 访问对话状态")
mock_state_with_messages = {
    "messages": ["msg1", "msg2", "msg3"]  # 模拟消息列表
}

# 这个工具不需要特定的 context，所以传入 None
mock_runtime2 = ToolRuntime(
    context=None,  # 不需要 context
    state=mock_state_with_messages,
    config={},
    stream_writer=lambda x: None,
    tool_call_id="mock_tool_call_002",
    store=None
)

result2 = get_message_count.invoke({"runtime": mock_runtime2})
print(f"✓ 工具返回: {result2}\n")

print("=" * 50)
print("说明:")
print("- runtime.context: 访问不可变的上下文信息（如 user_id）")
print("- runtime.state: 访问可变的状态信息（如 messages）")
print("- runtime.store: 访问持久化存储")
print("- runtime.stream_writer: 流式输出自定义更新")
print("\n在实际的 Agent 应用中，ToolRuntime 由 LangGraph")
print("框架自动注入，工具函数无需手动创建。")