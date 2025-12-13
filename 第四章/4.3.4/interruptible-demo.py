import random
import operator
from typing import Dict, Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# ========== 1. 状态定义 ==========
class InterruptibleState(TypedDict):
    """可中断工作流状态"""
    messages: Annotated[list, operator.add]
    current_step: str

# ========== 2. 节点定义 ==========
def risky_step(state: InterruptibleState) -> Dict[str, Any]:
    """模拟可能失败的操作"""
    if random.choice([True, False]):
        raise Exception("模拟随机错误！工作流中断！")
    return {"messages": ["成功完成危险步骤!"], "current_step": "completed"}

# ========== 3. 构建图 ==========
def create_interruptible_graph():
    """创建可中断的工作流图"""
    builder = StateGraph(InterruptibleState)
    builder.add_node("risky_operation", risky_step)
    builder.add_edge(START, "risky_operation")
    builder.add_edge("risky_operation", END)
    
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)

# ========== 4. 执行演示 ==========
def print_state(title: str, state: Any):
    """打印状态信息"""
    print(f"\n{title}")
    print("-" * 50)
    if state and hasattr(state, "values"):
        print(f"当前步骤: {state.values.get('current_step', 'N/A')}")
        print(f"消息: {state.values.get('messages', [])}")
    else:
        print(state if state else "状态为空")

if __name__ == "__main__":
    graph = create_interruptible_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "interruptible_thread"}}
    
    try:
        print("=== 第一次执行 ===")
        result = graph.invoke(
            {"messages": ["开始执行"], "current_step": "start"}, 
            config=config
        )
        print("\n执行成功:")
        print(f"  当前步骤: {result.get('current_step', 'N/A')}")
        print(f"  消息: {result.get('messages', [])}")
    except Exception as e:
        print(f"\n执行中断: {e}")
        current_state = graph.get_state(config)
        print_state("中断时的状态", current_state)
