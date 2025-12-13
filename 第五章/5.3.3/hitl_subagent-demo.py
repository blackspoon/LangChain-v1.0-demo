"""
展示 SubAgent 的 HITL 行为。

场景：
- 主智能体中 delete_file 需要审批，read_file 不需要审批
- 子智能体 file-manager 中 delete_file 和 read_file 都需要审批（覆写主配置）

重点：子智能体的 interrupt_on 配置可以独立于主智能体，实现对不同层级的精细控制。
"""
import uuid
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage


# ==============================
# 1. 定义工具
# ==============================

@tool
def delete_file(file_path: str) -> str:
    """删除文件（危险操作）。"""
    return f"文件 {file_path} 已删除"

@tool
def read_file(file_path: str) -> str:
    """读取文件内容。"""
    return f"文件 {file_path} 的内容：这是文件内容示例"


# ==============================
# 2. 创建 Agent
# ==============================

# HITL 必须使用 checkpointer 来持久化状态，以便在中断和恢复之间保持状态
checkpointer = MemorySaver()

# 初始化模型
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

agent = create_deep_agent(
    model=model,
    tools=[delete_file, read_file],
    # 主智能体的 interrupt_on 配置
    interrupt_on={
        "delete_file": True,   # 主智能体：删除需要审批
        "read_file": False,    # 主智能体：读取不需要审批
    },
    # 定义子智能体
    subagents=[{
        "name": "file-manager",
        "description": "专门负责文件管理的子智能体。",
        "system_prompt": "你是一个文件管理助手，可以谨慎地删除和读取文件。",
        "tools": [delete_file, read_file],
        # ✅ 子智能体的 interrupt_on 可以覆写主配置
        # 这使得子智能体可以有自己的审批规则
        "interrupt_on": {
            # delete_file 依然需要审批（与主智能体一致）
            "delete_file": True,
            # read_file 改成了需要审批（与主智能体不同：主智能体中不需要审批）
            "read_file": True,
        },
    }],
    checkpointer=checkpointer,  # 必须提供 checkpointer
)


# ==============================
# 3. 调用 Agent
# ==============================

# 每个对话线程必须带 thread_id，用于状态持久化
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="请让 file-manager 子智能体帮我读取 test.txt 文件，然后删除它。")
        ]
    },
    config=config,
)


# ==============================
# 4. 处理中断（循环处理，因为可能有多轮中断）
# ==============================

while result.get("__interrupt__"):
    print("\n=== HITL 中断触发 ===")
    # 提取中断信息
    interrupt_value = result["__interrupt__"][0].value
    # 获取待审批的工具调用列表
    action_requests = interrupt_value["action_requests"]

    # 收集用户决策（decisions 的顺序必须与 action_requests 完全一致）
    decisions = []
    for ar in action_requests:
        print(f"\n工具: {ar['name']}, 参数: {ar['args']}")
        decision = input("决策 (approve/reject): ").strip().lower()
        decisions.append({"type": decision})

    # 使用 Command(resume=...) 恢复执行，必须使用相同的 config（同一个 thread_id）
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )


# ==============================
# 5. 输出最终结果
# ==============================

print("\n=== 最终回应 ===")
if result.get("messages"):
    print(result["messages"][-1].content)
else:
    print("无消息返回")
print("================\n")
