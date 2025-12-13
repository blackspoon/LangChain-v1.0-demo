from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

# 1. 准备几条“员工守则”文档，封装为 Document
docs: List[Document] = [
    Document(
        page_content=(
            "公司要求全体员工遵守考勤制度，按时上下班。"
            "对于迟到、早退的员工，将视次数和情节轻重给予口头提醒、书面警告或绩效扣分。"
        ),
        metadata={"section": "考勤与纪律"},
    ),
    Document(
        page_content=(
            "员工在对外邮件和客户沟通中必须使用公司统一的邮件签名模板，"
            "严禁通过个人邮箱发送包含客户隐私或商业机密的信息。"
        ),
        metadata={"section": "对外沟通与信息安全"},
    ),
    Document(
        page_content=(
            "员工必须遵守信息安全制度，不得随意使用个人U盘等外部存储设备，"
            "不得将内部资料拷贝至非授权设备。"
        ),
        metadata={"section": "设备与资料管理"},
    ),
]

# 2. 配置通义千问 DashScope 的 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key="xxxxxx",
)

# 3. 初始化 InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# 4. 将文档写入向量存储（由 vector_store 内部调用 embeddings.embed_documents）
ids = vector_store.add_documents(documents=docs)
print(f"已写入 {len(ids)} 条文档到 InMemoryVectorStore")

# 5. 员工提问：检索与“迟到早退处罚”最相关的条款
query = "公司对迟到早退有什么处罚规定？"

results: List[Document] = vector_store.similarity_search(
    query,
    k=2,  # 返回两个最相似的文档
)

print(f"\n查询：{query}")
print("检索到的相关条款：")
for i, doc in enumerate(results, start=1):
    print(f"\n--- 结果 {i} ---")
    print(doc.page_content)
    print(f"metadata: {doc.metadata}")
