from langchain_community.document_loaders import PyPDFLoader

file_path = "files/XX销售有限公司员工守则.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(f"共加载到 {len(docs)} 个 Document")

print("-"*100)

for doc in docs:
    print(doc.metadata)
    print(doc.page_content[:100])
    print("-"*100)

