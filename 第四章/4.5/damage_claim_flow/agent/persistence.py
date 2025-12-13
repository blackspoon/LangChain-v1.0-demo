# agent/persistence.py
from langgraph.checkpoint.memory import MemorySaver

# In-memory checkpointer，用于演示持久化和耐久执行。
_memory = MemorySaver()


def get_checkpointer() -> MemorySaver:
    return _memory
