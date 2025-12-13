from langchain_community.document_loaders import NotionDBLoader

loader = NotionDBLoader(
    integration_token="xxxxxx",
    database_id="xxxxxx",
    request_timeout_sec=30,
)

docs = loader.load()

print(f"共从 Notion 数据库加载到 {len(docs)} 个 Document")

print("-"*100)

for doc in docs:
    print(doc.metadata)
    print(doc.page_content[:100])
    print("-"*100)