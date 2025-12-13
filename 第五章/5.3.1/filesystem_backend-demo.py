import os
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="xxxxxx",
    temperature=0.3,
)

project_root = os.path.abspath(".")  # 当前目录作为 root_dir

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(
        root_dir=project_root,
        virtual_mode=True,   # 启用路径沙箱和规范化
    ),
    system_prompt="""
你可以访问一个本地项目目录。
请使用文件工具（例如 write_file）在项目根目录创建并写入一个 README.md 文件。
"""
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请为这个项目创建一个 README.md 文件，包含项目介绍、功能特性和使用方法等内容。"}]}
    )
    print(result["messages"][-1].content)
