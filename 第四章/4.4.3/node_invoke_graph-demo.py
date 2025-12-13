from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# ========== 1. 定义子图 ==========
class SubgraphState(TypedDict):
    """子图状态"""
    child_value: str

def subgraph_node_1(state: SubgraphState):
    """子图节点"""
    return {"child_value": "你好! " + state["child_value"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# ========== 2. 定义父图 ==========
class State(TypedDict):
    """父图状态"""
    parent_value: str

def call_subgraph(state: State):
    """在节点内部调用子图"""
    # 将父图状态转换为子图状态并调用子图
    subgraph_output = subgraph.invoke({"child_value": state["parent_value"]})
    # 将子图输出转换回父图状态
    return {"parent_value": subgraph_output["child_value"]}

# ========== 3. 构建图 ==========
builder = StateGraph(State)
builder.add_node("node_1", call_subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()

# ========== 4. 执行演示 ==========
if __name__ == "__main__":
    for chunk in graph.stream({"parent_value": "今天天气不错"}, subgraphs=True):
        print(chunk)

