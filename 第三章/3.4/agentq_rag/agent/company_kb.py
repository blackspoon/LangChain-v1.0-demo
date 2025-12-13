"""
公司规章制度知识库（RAG）模块

提供三个核心能力：
1. init_company_kb(delete_existing=False)：从 PDF 初始化/重建知识库（写入 Milvus）
2. _search_company_policy(query)：从 Milvus 中做相似度检索（只读）
3. build_company_kb_tool()：把检索函数封装为 LangChain Tool，供 Agent 调用

设计要求：
- 平时问答只做查询，不写入 Milvus
- 只有在页面点击【初始化知识库】/【更新知识库】时，才执行插入
- 查询时若发现 collection 未创建，则返回"请先初始化知识库。"的提示
"""

import os
from functools import lru_cache
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility


# ============================================================================
# 配置常量
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(PROJECT_ROOT, "files", "XX销售有限公司员工守则.pdf")

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "employee_handbook"

DASHSCOPE_API_KEY = "xxxxxx"
EMBEDDING_MODEL = "text-embedding-v4"


# ============================================================================
# Milvus 连接管理
# ============================================================================

def _connect_milvus() -> None:
    """建立到 Milvus 的连接（幂等）。"""
    try:
        # 检查连接是否已存在
        existing_connections = connections.list_connections()
        if "default" in existing_connections:
            # 连接已存在，验证连接是否有效
            try:
                connections.get_connection_addr("default")
                print(f"[RAG] Milvus连接已存在: {MILVUS_URI}")
                return
            except Exception:
                # 连接无效，断开后重新连接
                try:
                    connections.disconnect("default")
                except Exception:
                    pass
    except Exception:
        # 如果检查连接时出错，继续尝试连接
        pass
    
    # 建立新连接
    try:
        connections.connect(alias="default", uri=MILVUS_URI)
        print(f"[RAG] Milvus连接已建立: {MILVUS_URI}")
    except Exception as e:
        raise ConnectionError(f"无法连接到 Milvus ({MILVUS_URI}): {e}") from e


# ============================================================================
# 向量存储管理
# ============================================================================

@lru_cache(maxsize=1)
def _get_embeddings() -> DashScopeEmbeddings:
    """获取 DashScope Embeddings 实例（缓存）。"""
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )


@lru_cache(maxsize=1)
def get_company_vectorstore() -> Milvus:
    """获取公司知识库向量存储实例（缓存）。"""
    embeddings = _get_embeddings()
    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )


# ============================================================================
# 知识库初始化
# ============================================================================

def init_company_kb(delete_existing: bool = False) -> None:
    """
    从 PDF 初始化公司规章制度知识库。
    
    Args:
        delete_existing: 如果为 True，删除现有集合后重新创建
    """
    # 1. 建立连接
    try:
        _connect_milvus()
    except Exception as e:
        raise ConnectionError(f"无法连接到 Milvus: {e}") from e

    # 2. 处理现有集合
    if delete_existing and utility.has_collection(COLLECTION_NAME):
        try:
            utility.drop_collection(COLLECTION_NAME)
            print(f"[RAG] 已删除现有集合: {COLLECTION_NAME}")
        except Exception as e:
            raise RuntimeError(f"删除集合失败: {e}") from e

    if (not delete_existing) and utility.has_collection(COLLECTION_NAME):
        get_company_vectorstore.cache_clear()
        print(f"[RAG] 集合已存在，跳过初始化: {COLLECTION_NAME}")
        return

    # 3. 检查 PDF 文件
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"未找到 PDF 文件：{PDF_PATH}")

    # 4. 加载和切分文档
    try:
        print(f"[RAG] 开始加载PDF: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        print(f"[RAG] PDF加载完成，共 {len(docs)} 页")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？"],
        )
        split_docs = text_splitter.split_documents(docs)
        print(f"[RAG] 文本切分完成，共 {len(split_docs)} 个文档块")
    except Exception as e:
        raise RuntimeError(f"加载或切分PDF失败: {e}") from e

    # 5. 初始化 Embeddings
    embeddings = _get_embeddings()

    # 6. 写入 Milvus
    try:
        _connect_milvus()  # 确保连接已建立
        print(f"[RAG] 开始写入 Milvus 集合: {COLLECTION_NAME}")
        
        Milvus.from_documents(
            documents=split_docs,
            embedding=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            index_params={"index_type": "FLAT", "metric_type": "L2"},
        )
        print(f"[RAG] 知识库初始化完成: {COLLECTION_NAME}")
        
        # 清除缓存，确保下次获取时使用最新的数据
        get_company_vectorstore.cache_clear()
    except Exception as e:
        # 如果失败，尝试重新连接后再试一次
        print(f"[RAG] 第一次写入失败，尝试重新连接: {e}")
        try:
            _connect_milvus()
            Milvus.from_documents(
                documents=split_docs,
                embedding=embeddings,
                connection_args={"uri": MILVUS_URI},
                collection_name=COLLECTION_NAME,
                index_params={"index_type": "FLAT", "metric_type": "L2"},
            )
            print(f"[RAG] 知识库初始化完成（重试后）: {COLLECTION_NAME}")
            get_company_vectorstore.cache_clear()
        except Exception as retry_e:
            raise RuntimeError(f"写入 Milvus 失败: {retry_e}") from retry_e


# ============================================================================
# 知识库查询
# ============================================================================

def _search_company_policy(query: str) -> str:
    """
    查询公司规章制度知识库。
    
    Args:
        query: 查询文本
        
    Returns:
        检索到的相关条款文本，如果未找到则返回提示信息
    """
    # 1. 检查连接和集合
    try:
        _connect_milvus()
        if not utility.has_collection(COLLECTION_NAME):
            return "请先初始化知识库。"
    except Exception as e:
        print(f"[RAG] 连接Milvus失败: {e}")
        return "请先初始化知识库。"

    # 2. 执行相似度搜索
    try:
        vector_store = get_company_vectorstore()
        docs: List[Document] = vector_store.similarity_search(query, k=3)

        if not docs:
            return (
                "【知识库无结果】未检索到与你问题相关的条款。\n"
                "你可以尝试换一种说法，或者咨询人力资源部门。"
            )

        # 3. 格式化返回结果
        chunks: List[str] = []
        for i, doc in enumerate(docs, start=1):
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            section = metadata.get("section") or metadata.get("page") or ""
            header = f"[条款 {i}]"
            if section:
                header = f"[条款 {i}] {section}"
            chunks.append(f"{header}\n{doc.page_content}")
            print(f"--------------------------------\n[RAG] 查询知识库完成: {chunks}")
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"[RAG] 查询知识库失败: {e}")
        return f"查询知识库时发生错误：{str(e)}。请检查知识库是否已正确初始化。"


# ============================================================================
# LangChain Tool 封装
# ============================================================================

@tool
def company_policy_knowledge_base(query: str) -> str:
    """
    查询公司内部规章制度、员工守则、考勤纪律、奖惩办法、请假制度等内容。
    
    当问题涉及"公司规定""员工守则""加班规定""考勤制度"等关键词时使用此工具。
    若知识库未初始化，将提示"请先初始化知识库。"
    
    Args:
        query: 查询问题，例如"员工迟到会有什么处理规定？"
        
    Returns:
        相关的规章制度条款内容
    """
    return _search_company_policy(query)


def build_company_kb_tool():
    """构建公司知识库工具（兼容旧接口）。"""
    return company_policy_knowledge_base
