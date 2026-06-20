import math
import json
import re
from pathlib import Path

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from core.settings import settings


CHROMA_DB_PATH = "./chroma_db_small_chunks"
RAG_SOURCE_FILE = "AcmeTech_Employee_Handbook.pdf"
DEFAULT_COMPATIBLE_EMBEDDING_MODEL = "text-embedding-v3-small"
DEFAULT_NO_HIT_CONTEXT = "未在当前示例知识库中检索到相关内容。"

QUERY_HINTS = {
    "远程办公": "remote work policy hybrid work manager approval core hours",
    "在家办公": "remote work policy hybrid work manager approval core hours",
    "异地办公": "remote work policy hybrid work manager approval core hours",
    "员工福利": "employee benefits health insurance 401k equity options professional development wellness programs",
    "福利": "employee benefits health insurance 401k equity options professional development wellness programs",
    "休假": "leave time-off policy PTO sick leave parental leave holidays HR portal",
    "请假": "leave time-off policy PTO sick leave parental leave holidays HR portal",
    "年假": "leave time-off policy PTO sick leave parental leave holidays HR portal",
    "病假": "leave time-off policy PTO sick leave parental leave holidays HR portal",
}


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class KnowledgeBaseSearchResult(BaseModel):
    hit_count: int = Field(description="检索命中的文档分块数量。")
    source: str = Field(description="当前知识库主要来源名称。")
    context: str = Field(description="供助手回答使用的检索上下文。")


def create_embedding_model() -> OpenAIEmbeddings:
    """按当前项目主线优先创建 compatible embeddings，其次才回退到 OpenAI。"""
    compatible_embedding_model = getattr(settings, "COMPATIBLE_EMBEDDING_MODEL", None)

    if settings.COMPATIBLE_API_KEY and settings.COMPATIBLE_BASE_URL:
        return OpenAIEmbeddings(
            model=compatible_embedding_model or DEFAULT_COMPATIBLE_EMBEDDING_MODEL,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    if settings.OPENAI_API_KEY:
        return OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            check_embedding_ctx_length=False,
        )

    raise RuntimeError(
        "无法初始化 embedding 模型。请优先配置 COMPATIBLE_API_KEY、"
        "COMPATIBLE_BASE_URL 和可选的 COMPATIBLE_EMBEDDING_MODEL；"
        "如果你确实要走 OpenAI，再配置 OPENAI_API_KEY。"
    )


def load_chroma_db():
    # 为当前示例知识库创建 embedding 模型
    embeddings = create_embedding_model()

    # 加载本地 Chroma 向量库
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


def filter_scored_documents(scored_documents, distance_threshold: float):
    """按距离阈值筛掉弱相关结果。距离越小，代表越相似。"""
    filtered_documents = []
    for document, distance in scored_documents:
        if distance <= distance_threshold:
            filtered_documents.append(document)
    return filtered_documents


def trim_documents_for_context(documents, max_context_chunks: int):
    """在已命中的结果里只保留最相关的前几个片段，减少回答上下文噪声。"""
    return documents[:max_context_chunks]


def expand_query_for_bilingual_handbook(query: str) -> str:
    """为中文提问补一段稳定的英文检索提示，缓解中文问句检索英文手册的召回问题。"""
    hint_parts = [english_hint for keyword, english_hint in QUERY_HINTS.items() if keyword in query]
    if not hint_parts:
        return query

    unique_hints = list(dict.fromkeys(hint_parts))
    return f"{query}\n\nRelevant English handbook topics: {' ; '.join(unique_hints)}"


def database_search_func(query: str) -> str:
    """检索示例员工手册知识库，并返回结构化结果。"""
    chroma_db = load_chroma_db()
    retrieval_query = expand_query_for_bilingual_handbook(query)
    scored_documents = chroma_db.similarity_search_with_score(
        retrieval_query,
        k=settings.RAG_SEARCH_K,
    )
    documents = filter_scored_documents(
        scored_documents,
        distance_threshold=settings.RAG_DISTANCE_THRESHOLD,
    )
    documents = trim_documents_for_context(
        documents,
        max_context_chunks=settings.RAG_MAX_CONTEXT_CHUNKS,
    )

    if not documents:
        result = KnowledgeBaseSearchResult(
            hit_count=0,
            source=RAG_SOURCE_FILE,
            context=DEFAULT_NO_HIT_CONTEXT,
        )
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

    # 格式化上下文
    context_str = format_contexts(documents)

    result = KnowledgeBaseSearchResult(
        hit_count=len(documents),
        source=RAG_SOURCE_FILE,
        context=context_str,
    )
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


database_search: BaseTool = tool(database_search_func)
database_search.name = "Knowledge_Base_Search"


def get_knowledge_base_status() -> str:
    """返回当前示例知识库的状态说明。"""
    data_file = Path("./data") / RAG_SOURCE_FILE
    chroma_dir = Path(CHROMA_DB_PATH)

    if not data_file.exists():
        return "未找到示例知识库源文件。"
    if not chroma_dir.exists():
        return "已检测到示例知识库源文件，但本地 Chroma 数据库尚未创建。"
    if not any(chroma_dir.iterdir()):
        return "本地 Chroma 数据库目录存在，但当前为空。"
    return f"已接入示例知识库：{RAG_SOURCE_FILE}"
