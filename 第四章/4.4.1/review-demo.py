from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

# ========== 1. 状态定义 ==========
class ReviewState(TypedDict):
    """审核状态类型定义"""
    generated_text: str  # 生成的文本内容

# ========== 2. 节点定义 ==========
def review_node(state: ReviewState):
    """
    审核节点：中断执行，等待审核者编辑生成的内容
    
    Args:
        state: 当前状态，包含生成的文本
        
    Returns:
        dict: 更新后的状态，包含编辑后的文本
    """
    # 中断执行，请求审核者编辑生成的内容
    updated = interrupt({
        "instruction": "请审核并编辑此内容",
        "content": state["generated_text"],
    })
    return {"generated_text": updated}

# ========== 3. 构建图 ==========
builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ========== 4. 执行演示 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review1"}}
    
    # 第一次执行：工作流在审核节点中断
    initial = graph.invoke({"generated_text": "初始草稿"}, config=config)
    print("中断信息:")
    print(initial["__interrupt__"])
    
    # 恢复执行：传入审核者编辑后的文本
    final_state = graph.invoke(
        Command(resume="审核后改进的草稿"),
        config=config,
    )
    print("\n最终状态中的文本内容:")
    print(final_state["generated_text"])
