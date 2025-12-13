"""
一个最小可运行示例：展示 Multiple Tool Calls 的 HITL 行为。

场景：
用户输入 “请同时备份文件和清理临时目录”，
智能体会规划调用两个工具：
- backup_data
- cleanup_temp

这两个工具都配置成需要人工审批，因此会一次性触发中断。
"""

import uuid
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage

# ==============================
# 1. 定义两个简单工具
# ==============================

@tool
def backup_data(target: str) -> str:
    """备份数据（敏感操作）。"""
    return f"Backup completed for {target}"

@tool
def cleanup_temp(folder: str) -> str:
    """清理临时目录（敏感操作）。"""
    return f"Temporary folder {folder} cleaned"

# ==============================
# 2. 创建 Agent
#    - 两个工具都需要 HITL
#    - 必须使用 MemorySaver 才能恢复中断
# ==============================

checkpointer = MemorySaver()

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

agent = create_deep_agent(
    model=model,
    tools=[backup_data, cleanup_temp],
    interrupt_on={
        "backup_data": True,
        "cleanup_temp": True,
    },
    system_prompt=(
        "你是一名系统助手，根据用户请求决定是否调用备份或清理工具。"
        "对于敏感操作会暂停并等待人工确认。"
    ),
    checkpointer=checkpointer,
)

# ==============================
# 3. 发起一次会触发多个工具调用的请求
# ==============================

# 每个对话线程必须带 thread_id
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="帮我备份 data.db，并清理一下 /tmp 目录。")
        ]
    },
    config=config,
)

# ==============================
# 4. 处理中断（可能需要循环处理多个中断）
# ==============================

while result.get("__interrupt__"):
    print("\n=== HITL 中断触发 ===")
    interrupt_value = result["__interrupt__"][0].value
    action_requests = interrupt_value["action_requests"]

    decisions = []
    for ar in action_requests:
        print(f"\n工具: {ar['name']}, 参数: {ar['args']}")
        decision = input("决策 (approve/reject): ").strip().lower()
        decisions.append({"type": decision})

    # 使用同一个 thread_id 继续执行
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )

# ==============================
# 5. 打印最终回答
# ==============================

print("\n=== 最终回应 ===")
if result.get("messages"):
    print(result["messages"][-1].content)
else:
    print("无消息返回")
print("================\n")
