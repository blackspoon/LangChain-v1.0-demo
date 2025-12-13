from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "files/XX销售有限公司员工守则.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 每个chunk的长度（按字符）
    chunk_overlap=50, # 重叠部分，用来弥补边界处可能被截断的信息
    add_start_index=True, # 让 splitter 记录每个 chunk 在原文中的起始位置
    # 如果你希望显式控制“层级”，也可以自定义分隔符
    # separators=["\n\n", "\n", "。", "，", " "]
)

splits: List[Document] = text_splitter.split_documents(docs)

print(f"原始文档数量: {len(docs)}")
print(f"切分之后的文档块数量: {len(splits)}")

print("-"*100)
for split in splits:
    print(split.page_content[:100])
    print("-"*100)