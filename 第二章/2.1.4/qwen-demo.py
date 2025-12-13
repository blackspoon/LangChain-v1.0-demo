# pip3 install -U dashscope

from langchain_community.llms.tongyi import Tongyi

# 1) 初始化 Qwen 文本补全模型（LLM）
model = Tongyi(
    model="qwen-plus",
    api_key="xxxxxx",
    temperature=0.7
)

# 2) 调用
result = model.invoke("请解释 LangChain 的核心理念。")
print(result)
