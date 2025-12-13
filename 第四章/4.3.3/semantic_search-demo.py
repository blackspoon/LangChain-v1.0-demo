from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, MessagesState, StateGraph
import uuid

# ========== 1. 初始化模型 ==========
model = init_chat_model(
    model="qwen-plus",
    temperature=0.7,
    api_key="xxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_provider="openai"
)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key="xxxxxx"
)

# ========== 2. 配置语义搜索存储 ==========
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536
    }
)

user_id = "user_001"
namespace_for_memory = (user_id, "memories")

store.put(namespace_for_memory, str(uuid.uuid4()), {"text": "我喜欢苹果"})
store.put(namespace_for_memory, str(uuid.uuid4()), {"text": "我是张三"})

# ========== 3. 定义节点 ==========
def chat(state, *, store: BaseStore):
    """使用语义搜索检索相关记忆"""
    items = store.search(
        namespace_for_memory, 
        query=state["messages"][-1].content,
        limit=1
    )
    
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## 用户记忆\n{memories}" if memories else ""
    
    response = model.invoke(
        [
            SystemMessage(content=f"你是一个帮助用户解决问题的助手。\n{memories}"),
            *state["messages"],
        ]
    )
    return {"messages": [response]}

# ========== 4. 构建图 ==========
builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

# ========== 5. 执行 ==========
if __name__ == "__main__":
    for message, metadata in graph.stream(
        input={"messages": [HumanMessage(content="我饿了")]},
        stream_mode="messages",
    ):
        print(message.content, end="")
