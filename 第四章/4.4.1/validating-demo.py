from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

# ========== 1. 状态定义 ==========
class FormState(TypedDict):
    """表单状态类型定义"""
    age: int | None  # 年龄，可以是整数或 None

# ========== 2. 节点定义 ==========
def get_age_node(state: FormState):
    """
    收集年龄节点：通过中断机制循环提示用户输入，直到获得有效数据
    
    Args:
        state: 当前状态，包含年龄字段
        
    Returns:
        dict: 更新后的状态，包含有效的年龄值
    """
    prompt = "请输入您的年龄："

    # 循环提示，直到获得有效输入
    while True:
        # 中断执行，提示用户输入；负载信息会出现在 result["__interrupt__"] 中
        answer = interrupt(prompt)

        # 验证输入：必须是正整数
        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        # 输入无效，更新提示信息，继续循环
        prompt = f"'{answer}' 不是有效的年龄。请输入一个正整数。"

# ========== 3. 构建图 ==========
builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ========== 4. 执行演示 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "validating1"}}
    
    # 第一次执行：工作流在收集年龄节点中断
    first = graph.invoke({"age": None}, config=config)
    print("第一次中断信息:")
    print(first["__interrupt__"])
    
    # 提供无效数据；节点会重新提示
    retry = graph.invoke(Command(resume="三十"), config=config)
    print("\n无效输入后的中断信息（包含错误提示）:")
    print(retry["__interrupt__"])
    
    # 提供有效数据；循环退出，状态更新
    final = graph.invoke(Command(resume=30), config=config)
    print("\n最终年龄:")
    print(final["age"])