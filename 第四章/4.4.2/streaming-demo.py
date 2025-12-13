from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import asyncio

# ========== 1. 状态定义 ==========
class State(TypedDict):
    """状态类型定义"""
    topic: str  # 主题
    story: str  # 故事内容

# ========== 2. 节点定义 ==========
def refine_topic(state: State):
    """
    精炼主题节点：在原有主题基础上添加内容
    
    Args:
        state: 当前状态，包含主题
        
    Returns:
        dict: 更新后的状态，包含精炼后的主题
    """
    return {"topic": state["topic"] + "的故事"}


def generate_story(state: State):
    """
    生成故事节点：根据主题生成故事
    
    Args:
        state: 当前状态，包含主题
        
    Returns:
        dict: 更新后的状态，包含生成的故事
    """
    return {"story": f"这是一个关于{state['topic']}"}

# ========== 3. 构建图 ==========
graph = (
    StateGraph(State)
    .add_node("refine_topic", refine_topic)      # 添加精炼主题节点
    .add_node("generate_story", generate_story)  # 添加生成故事节点
    .add_edge(START, "refine_topic")             # 从开始节点到精炼主题节点
    .add_edge("refine_topic", "generate_story")  # 从精炼主题节点到生成故事节点
    .add_edge("generate_story", END)             # 从生成故事节点到结束节点
    .compile()                                   # 编译图
)

# ========== 4. 执行演示 ==========
if __name__ == "__main__":
    print("=== 同步指定流模式输出 ===")
    for chunk in graph.stream(
        {"topic": "童话"},  # 初始状态：主题为"童话"
        stream_mode="updates",
    ):
        print(chunk)
    
    print("\n=== 异步多模式流式输出 ===")
    async def stream_demo():
        async for chunk in graph.astream(
            {"topic": "猫"},
            stream_mode=["updates", "values"],
        ):
            print(chunk)
    asyncio.run(stream_demo())
    
    print("\n=== 调试模式流式输出 ===")
    for chunk in graph.stream(
        {"topic": "日常"},
        stream_mode="debug",
    ):
        print(chunk)
