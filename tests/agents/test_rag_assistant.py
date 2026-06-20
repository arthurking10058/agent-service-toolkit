import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda

from agents.rag_assistant import (
    NO_HIT_RESPONSE_TEMPLATE,
    RAG_ERROR_RESPONSE_TEMPLATE,
    append_rag_observability_note,
    acall_model,
    build_rag_response_metadata,
    extract_knowledge_base_result,
    polish_rag_answer_content,
    retrieve_knowledge_base,
    wrap_model,
)
from agents.tools import (
    DEFAULT_NO_HIT_CONTEXT,
    expand_query_for_bilingual_handbook,
    filter_scored_documents,
    trim_documents_for_context,
)


def test_extract_knowledge_base_result():
    state = {
        "messages": [
            HumanMessage(content="员工手册里有休假吗？"),
            ToolMessage(
                content='{"hit_count": 3, "source": "AcmeTech_Employee_Handbook.pdf", "context": "..." }',
                tool_call_id="tool-1",
            ),
        ]
    }

    result = extract_knowledge_base_result(state)

    assert result is not None
    assert result["hit_count"] == 3
    assert result["source"] == "AcmeTech_Employee_Handbook.pdf"


def test_extract_knowledge_base_result_returns_none_for_non_json_tool_message():
    state = {
        "messages": [
            HumanMessage(content="员工手册里有休假吗？"),
            ToolMessage(content="plain text", tool_call_id="tool-1"),
        ]
    }

    result = extract_knowledge_base_result(state)

    assert result is None


def test_append_rag_observability_note():
    content = "根据员工手册，年假需要提前申请。"
    output = append_rag_observability_note(content, 3, "AcmeTech_Employee_Handbook.pdf")

    assert "年假需要提前申请。" in output
    assert "根据员工手册" not in output


def test_append_rag_observability_note_does_not_duplicate_same_note():
    note = "以上内容基于示例知识库《AcmeTech_Employee_Handbook.pdf》中的相关内容。"
    output = append_rag_observability_note(note, 3, "AcmeTech_Employee_Handbook.pdf")

    assert output == note


def test_append_rag_observability_note_normalizes_existing_note_to_real_source():
    note = "以上内容基于示例知识库《Another_Handbook.pdf》中的相关内容。"
    output = append_rag_observability_note(note, 3, "AcmeTech_Employee_Handbook.pdf")

    assert output == note


def test_append_rag_observability_note_removes_generic_closing():
    content = "公司支持混合办公，每周最多可远程 3 天。\n\n如果还有其他问题，欢迎继续提问。"

    output = append_rag_observability_note(content, 3, "AcmeTech_Employee_Handbook.pdf")

    assert "如果还有其他问题" not in output
    assert "公司支持混合办公" in output


def test_append_rag_observability_note_removes_internal_reference_tail():
    content = (
        "根据员工手册，AcmeTech 提供医疗、牙科和视力保障。\n\n"
        "以上内容基于示例知识库 AcmeTech_Employee_Handbook.pdf，参考了 1 个相关片段。"
    )

    output = append_rag_observability_note(content, 1, "AcmeTech_Employee_Handbook.pdf")

    assert output == "AcmeTech 提供医疗、牙科和视力保障。"


def test_polish_rag_answer_content_removes_formulaic_opening():
    content = "根据员工手册，公司支持混合办公，每周最多可远程 3 天。"

    output = polish_rag_answer_content(content)

    assert output == "公司支持混合办公，每周最多可远程 3 天。"


def test_polish_rag_answer_content_removes_redundant_source_text_and_extra_spacing():
    content = (
        "根据检索结果，员工福利主要包括医疗保险和培训支持。\n\n\n"
        "以上信息基于AcmeTech员工手册的知识库。"
    )

    output = polish_rag_answer_content(content)

    assert output == "员工福利主要包括医疗保险和培训支持。"


def test_build_rag_response_metadata():
    metadata = build_rag_response_metadata(
        {
            "hit_count": 2,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": "Remote Work Policy ...",
            "status": "hit",
        }
    )

    assert metadata["knowledge_base"]["hit"] is True
    assert metadata["knowledge_base"]["hit_count"] == 2
    assert metadata["knowledge_base"]["source"] == "AcmeTech_Employee_Handbook.pdf"
    assert metadata["knowledge_base"]["status"] == "hit"


def test_no_hit_template_is_stable():
    assert "没有找到" in NO_HIT_RESPONSE_TEMPLATE
    assert "福利" in NO_HIT_RESPONSE_TEMPLATE


def test_filter_scored_documents_applies_distance_threshold():
    docs = [
        (type("Doc", (), {"page_content": "remote"})(), 0.12),
        (type("Doc", (), {"page_content": "benefits"})(), 0.49),
        (type("Doc", (), {"page_content": "irrelevant"})(), 0.83),
    ]

    filtered = filter_scored_documents(docs, distance_threshold=0.55)

    assert len(filtered) == 2
    assert filtered[0].page_content == "remote"
    assert filtered[1].page_content == "benefits"


def test_expand_query_for_bilingual_handbook_adds_english_hint_for_chinese_keywords():
    query = "员工手册里有没有提到远程办公政策？"

    expanded = expand_query_for_bilingual_handbook(query)

    assert "remote work policy hybrid work manager approval core hours" in expanded
    assert query in expanded


def test_trim_documents_for_context_keeps_top_ranked_chunks_only():
    docs = [
        type("Doc", (), {"page_content": "chunk-1"})(),
        type("Doc", (), {"page_content": "chunk-2"})(),
        type("Doc", (), {"page_content": "chunk-3"})(),
        type("Doc", (), {"page_content": "chunk-4"})(),
    ]

    trimmed = trim_documents_for_context(docs, max_context_chunks=3)

    assert [doc.page_content for doc in trimmed] == ["chunk-1", "chunk-2", "chunk-3"]


def test_wrap_model_injects_retrieved_context():
    captured = {}

    def dummy_model(messages):
        captured["messages"] = messages
        return AIMessage(content="test")

    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
        "knowledge_base_result": {
            "hit_count": 2,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": "Remote Work Policy ...",
        },
    }

    runnable = wrap_model(RunnableLambda(dummy_model))
    runnable.invoke(state)

    system_message = captured["messages"][0]
    assert "当前知识库来源" in system_message.content
    assert "当前命中片段数：2" in system_message.content
    assert "Remote Work Policy ..." in system_message.content


@pytest.mark.asyncio
async def test_retrieve_knowledge_base_returns_structured_result():
    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
    }
    config = RunnableConfig(configurable={})

    with patch(
        "agents.rag_assistant.database_search_func",
        return_value=json.dumps(
            {
                "hit_count": 2,
                "source": "AcmeTech_Employee_Handbook.pdf",
                "context": "Remote Work Policy ...",
                "status": "hit",
            },
            ensure_ascii=False,
        ),
    ):
        output = await retrieve_knowledge_base(state, config)

    assert output["knowledge_base_result"]["hit_count"] == 2
    assert output["knowledge_base_result"]["source"] == "AcmeTech_Employee_Handbook.pdf"
    assert output["knowledge_base_result"]["status"] == "hit"


@pytest.mark.asyncio
async def test_retrieve_knowledge_base_marks_parse_failure_as_error():
    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
    }
    config = RunnableConfig(configurable={})

    with patch("agents.rag_assistant.database_search_func", return_value="not-json"):
        output = await retrieve_knowledge_base(state, config)

    assert output["knowledge_base_result"]["status"] == "error"
    assert output["knowledge_base_result"]["context"] == "知识库检索结果解析失败。"


@pytest.mark.asyncio
async def test_retrieve_knowledge_base_marks_search_exception_as_error():
    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
    }
    config = RunnableConfig(configurable={})

    with patch("agents.rag_assistant.database_search_func", side_effect=RuntimeError("network down")):
        output = await retrieve_knowledge_base(state, config)

    assert output["knowledge_base_result"]["status"] == "error"
    assert "知识库检索失败" in output["knowledge_base_result"]["context"]


@pytest.mark.asyncio
async def test_acall_model_returns_stable_no_hit_response_without_model_call():
    state = {
        "messages": [HumanMessage(content="2026 年公司股票期权发放比例是多少？")],
        "knowledge_base_result": {
            "hit_count": 0,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": DEFAULT_NO_HIT_CONTEXT,
            "status": "miss",
        },
    }
    config = RunnableConfig(configurable={})

    output = await acall_model(state, config)

    assert output["messages"][0].content == NO_HIT_RESPONSE_TEMPLATE
    assert output["messages"][0].response_metadata["knowledge_base"]["hit"] is False
    assert output["messages"][0].response_metadata["knowledge_base"]["hit_count"] == 0
    assert output["messages"][0].response_metadata["knowledge_base"]["status"] == "miss"


@pytest.mark.asyncio
async def test_acall_model_returns_error_response_for_knowledge_base_error():
    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
        "knowledge_base_result": {
            "hit_count": 0,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": "知识库检索结果解析失败。",
            "status": "error",
        },
    }
    config = RunnableConfig(configurable={})

    output = await acall_model(state, config)

    assert output["messages"][0].content == RAG_ERROR_RESPONSE_TEMPLATE
    assert output["messages"][0].response_metadata["knowledge_base"]["hit"] is False
    assert output["messages"][0].response_metadata["knowledge_base"]["status"] == "error"


@pytest.mark.asyncio
async def test_acall_model_appends_rag_metadata_on_hit():
    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
        "knowledge_base_result": {
            "hit_count": 2,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": "Remote Work Policy ...",
            "status": "hit",
        },
    }
    config = RunnableConfig(configurable={})

    with (
        patch("agents.rag_assistant.get_model") as mock_get_model,
        patch("agents.rag_assistant.wrap_model") as mock_wrap_model,
    ):
        mock_model_runnable = mock_wrap_model.return_value
        mock_model_runnable.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="手册说明公司支持混合办公，每周最多可远程 3 天。",
                response_metadata={"provider": "test"},
            )
        )

        output = await acall_model(state, config)

    mock_get_model.assert_called_once()
    response = output["messages"][0]
    assert response.response_metadata["provider"] == "test"
    assert response.response_metadata["knowledge_base"]["hit"] is True
    assert response.response_metadata["knowledge_base"]["hit_count"] == 2
    assert response.response_metadata["knowledge_base"]["status"] == "hit"
