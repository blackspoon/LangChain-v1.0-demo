from langchain_openai import ChatOpenAI

# 创建 ChatGPT 模型实例
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7, # 控制输出的随机性，0-1之间，值越大越随机
    api_key="xxxxxxx" 
)

# 调用模型
response = model.invoke("请解释LangChain模型接口的统一性。")
print(response.content)
