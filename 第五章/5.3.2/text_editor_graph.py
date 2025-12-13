# text_editor_graph.py
# 文本编辑器图：使用LangGraph构建的文本润色智能体

import operator
from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage, SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# 初始化聊天模型
model = init_chat_model(
    model="deepseek-chat",
    temperature=0.3,
    api_key="xxxxxx",
    base_url="https://api.deepseek.com/v1",
    model_provider="openai"
)

# 定义状态结构：消息列表
class TextEditorState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# 文本润色节点：调用模型进行文本润色
def rewrite_node(state: TextEditorState) -> TextEditorState:
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是一名中文文本润色助手..."
                )
            )
        ] + state["messages"]
    )
    return {"messages": [response]}

# 构建图：START -> rewrite -> END
graph_builder = StateGraph(TextEditorState)
graph_builder.add_node("rewrite", rewrite_node)
graph_builder.add_edge(START, "rewrite")
graph_builder.add_edge("rewrite", END)

# 编译图得到可执行的智能体
text_editor_agent = graph_builder.compile()

if __name__ == "__main__":
    result = text_editor_agent.invoke(
        {"messages": [HumanMessage(content="请将以下文本进行润色：今天天气真不错，适合出去玩。")]},
    )
    print(result["messages"][-1].content)