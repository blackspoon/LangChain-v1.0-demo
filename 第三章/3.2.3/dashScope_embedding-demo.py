from langchain_community.embeddings import DashScopeEmbeddings

# 1. 准备一段原始文本（直接写死，不依赖上一小节的 chunks）
document_text = """
XX销售有限公司员工守则：
公司要求全体员工遵守职业行为规范，包括准时上下班、客户接待礼仪、
办公环境维护、信息保密义务、安全生产责任制度等。
违反规定将根据情节轻重给予警告、记过、停职直至解除劳动合同的处理。
"""

# 2. 配置 DashScope（通义千问）的 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",         # 通义千问最新向量模型
    dashscope_api_key="xxxxxx"
)

# 3. 对文档文本进行向量化
doc_embedding = embeddings.embed_documents([document_text])[0]

print(f"文档向量维度：{len(doc_embedding)}")
print(f"前 10 个向量值：{doc_embedding[:10]}")

# 4. 示例：将一个查询转为向量
query = "公司的迟到早退处罚规则是什么？"
query_embedding = embeddings.embed_query(query)

print(f"查询向量维度：{len(query_embedding)}")
print(f"前 10 个向量值：{query_embedding[:10]}")
