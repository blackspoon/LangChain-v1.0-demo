from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# ========== 1. 定义子图 ==========
class SubgraphState(TypedDict):
    """子图状态"""
    parent_value: str  # 与父图共享
    child_value: str  # 子图私有

def subgraph_node_1(state: SubgraphState):
    """子图节点1"""
    return {"child_value": "你好！"}

def subgraph_node_2(state: SubgraphState):
    """子图节点2：更新共享状态"""
    return {"parent_value": state["child_value"] + state["parent_value"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# ========== 2. 定义父图 ==========
class ParentState(TypedDict):
    """父图状态"""
    parent_value: str  # 与子图共享

def node_1(state: ParentState):
    """父图节点1"""
    return {"parent_value": state["parent_value"]}

# ========== 3. 构建图 ==========
builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)  # 将子图直接作为节点
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

# ========== 4. 执行演示 ==========
if __name__ == "__main__":
    for chunk in graph.stream({"parent_value": "今天天气不错"}):
        print(chunk)