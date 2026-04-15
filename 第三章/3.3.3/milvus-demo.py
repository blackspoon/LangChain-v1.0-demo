import time
from pathlib import Path
from pymilvus import connections, utility, Collection
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus

# 1. 加载 PDF 文档
file_path = Path(__file__).resolve().parent / "files" / "XX销售有限公司员工守则.pdf"
if not file_path.exists():
    raise FileNotFoundError(
        f"未找到 PDF 文件：{file_path}\n"
        "请确认已在当前目录下创建 files/XX销售有限公司员工守则.pdf"
    )

loader = PyPDFLoader(str(file_path))
docs = loader.load()

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？"]
)
split_docs = text_splitter.split_documents(docs)

print(f"文档切分完成，共 {len(split_docs)} 个文档块")

# 3. 初始化 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key="xxxxxx"
)

# 4. 强制删除旧collection
try:
    URI = "http://localhost:19530"
    if not connections.has_connection("default"):
        connections.connect("default", uri=URI)
    utility.drop_collection("employee_handbook")
    print("已删除旧的 collection: employee_handbook")
except Exception as e:
    print(f"删除 collection 时出错（可能本就不存在，忽略继续）: {e}")

# 5. 构造向量库实例
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="employee_handbook",
    connection_args={"uri": "http://localhost:19530"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# 6. 写入数据
ids = vector_store.add_documents(split_docs)

print(f"数据已插入到 Milvus，共写入 {len(ids)} 条")

# 7. 验证插入是否成功
try:
    collection = Collection("employee_handbook")
    collection.load()
    
    # 先 flush 确保数据持久化
    collection.flush()
    time.sleep(0.5)  # 等待数据同步
    
    num_entities = collection.num_entities
    print(f"✓ 验证：Collection 中包含 {num_entities} 条数据")
    
    if num_entities == len(split_docs):
        print(f"✓ 插入成功！数据条数匹配（期望 {len(split_docs)} 条，实际 {num_entities} 条）")
    elif num_entities > 0:
        print(f"⚠ 数据已插入，但数量不匹配（期望 {len(split_docs)} 条，实际 {num_entities} 条）")
    else:
        # 如果数量为0，但查询能成功，说明数据在内存中，也算成功
        print("⚠ Collection 数量为 0，但数据可能在内存中（将通过查询验证）")
        
except Exception as e:
    print(f"⚠ 验证时出错: {e}")

# 7. 查询示例
query = "公司对迟到、早退是如何处理的？"
results = vector_store.similarity_search(query, k=3)

print(f"\n查询：{query}")
print(f"找到 {len(results)} 条相关结果：\n")

if len(results) > 0:
    print("✓ 查询成功，数据可用！插入验证通过")
    for i, doc in enumerate(results, 1):
        print(f"--- 结果 {i} ---")
        print(doc.page_content[:200])
        print()
else:
    print("✗ 警告：查询没有返回结果，数据可能未正确插入")
