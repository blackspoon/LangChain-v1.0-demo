from langgraph.store.memory import InMemoryStore
import uuid
from typing import Tuple, Dict, Any

# ========== 1. 初始化存储 ==========
def create_store() -> InMemoryStore:
    """创建内存存储实例"""
    return InMemoryStore()

# ========== 2. 存储操作 ==========
def store_memory(
    store: InMemoryStore, 
    namespace: Tuple[str, str], 
    memory_id: str, 
    memory_data: Dict[str, Any]
) -> None:
    """存储记忆到指定命名空间"""
    store.put(namespace, memory_id, memory_data)

def retrieve_memories(store: InMemoryStore, namespace: Tuple[str, str]) -> list:
    """从指定命名空间检索所有记忆"""
    return store.search(namespace)

# ========== 3. 演示 ==========
if __name__ == "__main__":
    # 初始化
    in_memory_store = create_store()
    user_id = "1"
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memory_data = {"food_preference": "我喜欢苹果"}
    
    # 存储记忆
    store_memory(in_memory_store, namespace, memory_id, memory_data)
    print(f"已存储记忆 ID: {memory_id}")
    
    # 检索记忆
    memories = retrieve_memories(in_memory_store, namespace)
    print(f"\n命名空间 '{namespace}' 中的记忆数量: {len(memories)}")
    
    if memories:
        print("\n最新记忆内容:")
        print("-" * 50)
        latest_memory = memories[-1]
        if hasattr(latest_memory, "dict"):
            print(latest_memory.dict())
        else:
            print(latest_memory)
