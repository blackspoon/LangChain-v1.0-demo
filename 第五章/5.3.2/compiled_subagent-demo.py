# compiled_subagent-demo.py
# 编译子代理示例：使用deepagents创建包含子智能体的深度代理

from deepagents import create_deep_agent, CompiledSubAgent
from langchain.chat_models import init_chat_model
from text_editor_graph import text_editor_agent
from langchain.messages import HumanMessage

# 将文本编辑器图封装为子智能体
text_helper_subagent = CompiledSubAgent(
    name="text-helper",
    description="用于润色文本的子智能体",
    runnable=text_editor_agent,
)

# 初始化协调者模型（推理模型）
coordinator_model = init_chat_model(
    model="deepseek-reasoner",
    temperature=0.3,
    api_key="xxxxxx",
    base_url="https://api.deepseek.com/v1",
    model_provider="openai"
)

# 创建深度代理：协调者负责判断是否需要调用子智能体
agent = create_deep_agent(
    model=coordinator_model,
    system_prompt="""
你是一个智能文本助手的协调者，负责判断用户的文本加工需求。

你的决定逻辑如下：
1. 如果用户仅提出一般性问题，可以直接回答。
2. 如果用户提供了一段文本，并且提出"润色 / 优化 / 重写 / 精简 / 提升表达清晰度"等需求：
   - 请使用 task() 工具，将任务委托给 text-helper 子智能体处理；
   - text-helper 会返回润色后的文本；
   - 你需要将其原样返回给用户，不要添加额外解释。

请根据用户输入自行判断是否触发子智能体。
""",
    subagents=[text_helper_subagent],
)

if __name__ == "__main__":
    original_text = """
今天天气真不错，我和小明出去玩了，但是很遗憾，我们没有玩成，因为小明生病了。
"""
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=f"请帮我润色下面这段文字： {original_text}")
            ]
        }
    )
    print(result["messages"][-1].content)
