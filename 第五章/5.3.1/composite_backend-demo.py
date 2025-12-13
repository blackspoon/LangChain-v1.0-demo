import os
from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    StateBackend,
    StoreBackend,
    FilesystemBackend,
)
from langgraph.store.memory import InMemoryStore
from langchain_deepseek import ChatDeepSeek

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

store = InMemoryStore()
docs_root = os.path.abspath("./project_docs")

def composite_backend_factory(rt):
    return CompositeBackend(
        default=StateBackend(rt),  # 默认：线程内临时
        routes={
            "/memories/": StoreBackend(rt),  # 长期记忆
            "/docs/": FilesystemBackend(root_dir=docs_root, virtual_mode=True),  # 本地文档
        },
    )

agent = create_deep_agent(
    model=llm,
    backend=composite_backend_factory,
    store=store,  # 提供给 StoreBackend 使用
    system_prompt="""
你有一个分层文件系统：
- /workspace/ 下是短期工作区（临时）
- /memories/ 下是长期记忆（持久化）
- /docs/ 下是本地项目文档（只在当前机器存在）

请合理使用这些路径来完成任务。
"""
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请从 /docs/ 目录中查找项目文档，"
                        "在 /workspace/ 下写一个草稿，"
                        "并把最终确认的结论保存到 /memories/summary.txt。"
                    ),
                }
            ]
        }
    )

    print(result["messages"][-1].content)
