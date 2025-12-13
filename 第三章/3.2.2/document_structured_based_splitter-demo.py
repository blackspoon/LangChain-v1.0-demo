from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# 产品使用手册

## 一、快速开始
这里介绍如何快速完成安装与登录。

### 1.1 安装步骤
详细安装步骤说明……

### 1.2 首次登录
首次登录需要注意的事项……

## 二、高级功能
这里是高级功能的概览。

### 2.1 自动化规则
如何配置自动化规则……

### 2.2 报表分析
如何查看和定制报表……
"""

# 指定希望追踪的 Markdown 级别
headers_to_split_on = [
    ("#", "level_1"),
    ("##", "level_2"),
    ("###", "level_3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    # 结果中的每个 Document 会附带对应标题层级的 metadata
    strip_headers=True,
)

md_docs = md_splitter.split_text(markdown_text)

for i, d in enumerate(md_docs[:4], start=1):
    print(f"--- Chunk {i} ---")
    print(d.metadata)        # 包含 level_1 / level_2 / level_3 信息
    print(d.page_content[:80])
