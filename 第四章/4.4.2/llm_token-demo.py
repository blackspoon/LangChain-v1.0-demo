from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

# ========== 1. 状态定义 ==========
@dataclass
class MyState:
    """状态类"""
    topic: str 
    story: str = "" 

# ========== 2. 初始化模型 ==========
model = init_chat_model(
    model="deepseek-chat",
    temperature=0.7,
    api_key="xxxxxx",
    base_url="https://api.deepseek.com/v1",
    model_provider="openai"
)

# ========== 3. 节点定义 ==========
def call_model(state: MyState):
    """调用 LLM 生成故事"""
    model_response = model.invoke(
        [
            {"role": "user", "content": f"生成一个关于 {state.topic} 的故事"}
        ]
    )
    return {"story": model_response.content}

# ========== 4. 构建图 ==========
graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# ========== 5. 执行演示 ==========
if __name__ == "__main__":
    # stream_mode="messages" 返回 (message_chunk, metadata) 元组
    # message_chunk 是 LLM 流式输出的 token
    for message_chunk, metadata in graph.stream(
        {"topic": "悬疑"},
        stream_mode="messages",
    ):
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)