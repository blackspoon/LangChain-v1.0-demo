from langchain.chat_models import init_chat_model

# 使用统一接口创建 ChatGPT 模型
model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.7, # 控制输出的随机性，0-1之间，值越大越随机
    api_key="xxxxxxxx",
    model_provider="openai"  # 指定模型提供方
)

# 调用模型
response = model.invoke("请用一句话总结LangChain的核心设计思想。")
print(response.content)

