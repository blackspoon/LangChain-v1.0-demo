from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = "files/XX销售有限公司员工守则.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", # 使用tiktoken编码器
    chunk_size=500, # 每个chunk的长度
    chunk_overlap=50 # 重叠部分，用来弥补边界处可能被截断的信息
)
texts = text_splitter.split_documents(docs)

print(f"共分割到 {len(texts)} 个 Chunk")
print("-"*100)
for text in texts:
    print(text.page_content[:100])
    print("-"*100)