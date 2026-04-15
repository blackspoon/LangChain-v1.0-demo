from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_writer.text_splitter import WriterTextSplitter
from langchain_core.documents import Document


file_path = Path(__file__).resolve().parent / "files" / "XX销售有限公司员工守则.pdf"
if not file_path.exists():
    raise FileNotFoundError(
        f"未找到 PDF 文件：{file_path}\n"
        "请确认已在当前目录下创建 files/XX销售有限公司员工守则.pdf"
    )

loader = PyPDFLoader(str(file_path))

docs = loader.load()

# 合并所有文档的文本内容
combined_text = "\n\n".join([doc.page_content for doc in docs])
# 合并元数据（使用第一个文档的元数据作为基础）
base_metadata = docs[0].metadata if docs else {}

# strategy 可选：llm_split / fast_split / hybrid_split
splitter = WriterTextSplitter(
    api_key="xxxxxx",
    strategy="fast_split",    # 更快的语义分段策略
)

# 使用 split_text 方法分割文本
text_chunks = splitter.split_text(combined_text)

# 手动创建 Document 对象列表
chunks = [Document(page_content=text, metadata=base_metadata) for text in text_chunks]

print(f"WRITER 返回的 chunk 数量: {len(chunks)}")
print("-"*100)
for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content[:100])
    print("-"*100)
