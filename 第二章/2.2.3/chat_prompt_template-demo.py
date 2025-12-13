from langchain_core.prompts import ChatPromptTemplate

# 定义 ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的AI助理。"),
    ("user", "请用简短的语言解释：{topic}")
])

# 渲染模板
pv = chat_prompt.invoke({"topic": "LangChain 的核心理念"})

print("=== ChatPromptTemplate 结果 ===")
for msg in pv.to_messages():
    print(msg.content)

# 你也可以得到完整文本：
print("\n=== 格式化后的文本 ===")
print(chat_prompt.format_prompt(topic="LangChain 的核心理念").to_string())

