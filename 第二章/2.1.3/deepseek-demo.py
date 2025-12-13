from langchain.chat_models import init_chat_model

# 使用统一接口创建 ChatGPT 模型
model = init_chat_model(
    model="deepseek-chat", # 可换成deepseek-reasoner推理模型
    temperature=0.7, # 控制输出的随机性，0-1之间，值越大越随机
    api_key="xxxxxx",  # 直接在代码中传入Deepseek API Key
    base_url="https://api.deepseek.com/v1", # DeepSeek API的base_url（使用OpenAI兼容接口）
    model_provider="openai"  # DeepSeek使用OpenAI兼容接口
)

# 调用模型
for chunk in model.stream("天为什么是蓝色的"):
    print(chunk.content, end="")
