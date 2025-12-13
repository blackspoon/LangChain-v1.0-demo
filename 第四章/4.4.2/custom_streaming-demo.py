from typing import TypedDict
from langgraph.config import get_stream_writer
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

# ========== 1. 状态定义 ==========
class State(TypedDict):
    """状态类型定义"""
    query: str    # 查询内容
    answer: str   # 查询结果

# ========== 2. 工具定义 ==========
@tool
def query_database_tool(query: str) -> str:
    """
    查询数据库工具：执行数据库查询
    
    Args:
        query: 查询内容
        
    Returns:
        str: 查询结果
    """
    # 访问流式写入器以发送自定义数据
    writer = get_stream_writer()
    
    # 发送第一个进度更新
    writer({"data": "已检索 0/100 条记录", "type": "progress"})
    
    # 执行查询（模拟）
    # 这里可以执行实际的数据库查询操作
    
    # 发送第二个进度更新
    writer({"data": "已检索 100/100 条记录", "type": "progress"})
    
    # 返回查询结果
    return f"查询 '{query}' 的结果：找到 100 条相关记录"

# ========== 3. 节点定义 ==========
def process_query(state: State):
    """
    处理查询节点：验证和预处理查询请求
    
    Args:
        state: 当前状态，包含查询内容
        
    Returns:
        dict: 更新后的状态，包含处理后的查询
    """
    # 获取流式写入器以发送自定义数据
    writer = get_stream_writer()
    # 发送自定义键值对（例如：进度更新）
    writer({"status": "正在处理查询请求...", "type": "progress"})
    
    # 验证和预处理查询
    processed_query = state["query"].strip()
    if not processed_query:
        processed_query = "默认查询"
    
    writer({"status": f"查询已处理：{processed_query}", "type": "info"})
    return {"query": processed_query}


def query_database(state: State):
    """
    查询数据库节点：调用数据库查询工具
    
    Args:
        state: 当前状态，包含处理后的查询
        
    Returns:
        dict: 更新后的状态，包含查询结果
    """
    # 调用数据库查询工具
    result = query_database_tool.invoke(state["query"])
    return {"answer": result}

# ========== 4. 构建图 ==========
graph = (
    StateGraph(State)
    .add_node("process_query", process_query)        # 添加处理查询节点
    .add_node("query_database", query_database)      # 添加查询数据库节点
    .add_edge(START, "process_query")                # 从开始到处理查询节点
    .add_edge("process_query", "query_database")     # 从处理查询到查询数据库
    .add_edge("query_database", END)                 # 从查询数据库到结束
    .compile()                                       # 编译图
)

# ========== 5. 执行演示 ==========
if __name__ == "__main__":
    # 输入数据
    inputs = {"query": "用户信息"}
    
    # 设置 stream_mode="custom" 以在流中接收自定义数据
    print("=== 自定义流式输出 ===")
    for chunk in graph.stream(inputs, stream_mode="custom"):
        print(chunk)