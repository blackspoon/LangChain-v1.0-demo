from typing import TypedDict
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

# ========== 1. 状态定义 ==========
class AgentState(TypedDict):
    """智能体状态类型定义"""
    messages: list[dict]  # 消息列表，包含用户消息和 AI 响应

# ========== 2. 工具定义 ==========
@tool
def send_email(to: str, subject: str, body: str):
    """
    发送邮件工具：在发送前中断执行，等待人工审批
    
    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文
        
    Returns:
        str: 发送结果消息
    """
    # 在发送前暂停执行；信息会出现在 result["__interrupt__"] 中
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "是否批准发送此邮件？"
    })

    # 如果用户批准了发送
    if response.get("action") == "approve":
        # 获取最终参数（用户可能修改了参数）
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # 实际发送邮件
        print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"邮件已发送至 {final_to}"

    # 用户取消了发送
    return "邮件已被用户取消"

# ========== 3. 初始化模型 ==========
model = init_chat_model(
    model="deepseek-chat",
    temperature=0.7,
    api_key="xxxxxx",
    base_url="https://api.deepseek.com/v1",
    model_provider="openai"
).bind_tools([send_email])

# ========== 4. 节点定义 ==========
def agent_node(state: AgentState):
    """智能体节点：调用模型处理消息"""
    result = model.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}


def tools_node(state: AgentState):
    """工具执行节点：执行工具调用"""
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, 'tool_calls', [])
    
    tool_messages = []
    for tool_call in tool_calls:
        # 执行工具
        result = send_email.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    
    return {"messages": tool_messages}

# ========== 5. 构建图 ==========
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", "tools")
builder.add_edge("tools", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ========== 6. 执行演示 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "tools1"}}
    
    # 第一次执行：工作流在工具节点中断
    initial = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "向 xxxxxx@163.com 发送一封关于会议的邮件，主题为：会议通知，正文为：请参加会议，时间：2025-11-18 10:00，地点：会议室101"}
            ]
        },
        config=config
    )
    
    print("中断信息:")
    print(initial["__interrupt__"])
    
    # 恢复执行：传入审批结果和可选的编辑后的参数
    resumed = graph.invoke(
        Command(resume={"action": "approve", "subject": "会议通知", "body": "请参加会议，时间：2025-11-18 10:00，地点：会议室102"}),
        config=config
    )
    print("\n最终结果：", resumed["messages"][-1])