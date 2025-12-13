from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# 创建 DeepSeek 模型实例
model = ChatOpenAI(
    model="deepseek-chat", 
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# --- 第一轮对话 ---
print("--- 第一轮对话 ---")

# 手动构建第一轮 Messages 列表 (包含历史)
messages_round_1 = [
    # 设定角色和风格
    SystemMessage(content="你是一个幽默风趣的诗人，所有的回复必须包含一个笑话。"),
    # 用户输入
    HumanMessage(content="写一首关于'AI学习'的短诗。")
]

# 调用模型并获取响应，返回的是AIMessage
response_1 = model.invoke(messages_round_1)

print(f"AI回复: {response_1.content}")
print("-" * 20)

# --- 第二轮对话 ---
print("--- 第二轮对话 ---")

# 构建第二轮 Messages 列表
messages_round_2 = [
    # 历史：必须保留 SystemMessage
    messages_round_1[0], 
    
    # 历史：第一轮的 HumanMessage
    messages_round_1[1],
    
    # 历史：显式创建一个 AIMessage 对象来存储模型的第一轮回复
    # 注意：model.invoke() 返回的就是 AIMessage，这里也可以直接使用 response_1
    AIMessage(content=response_1.content),
    
    # 新的用户输入
    HumanMessage(content="非常好！现在将它翻译成英文。") 
]

# 调用模型并获取响应
response_2 = model.invoke(messages_round_2)

print(f"AI回复: {response_2.content}")