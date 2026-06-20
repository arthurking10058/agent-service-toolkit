import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda

from agents.rag_assistant import (
    NO_HIT_RESPONSE_TEMPLATE,
    append_rag_observability_note,
    acall_model,
    build_rag_response_metadata,
    extract_knowledge_base_result,
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

    assert "参考了 3 个相关片段" in output
    assert "AcmeTech_Employee_Handbook.pdf" in output
    assert "以上内容基于示例知识库" in output
    assert content in output


def test_append_rag_observability_note_does_not_duplicate_same_note():
    note = "以上内容基于示例知识库 AcmeTech_Employee_Handbook.pdf，参考了 3 个相关片段。"
    output = append_rag_observability_note(note, 3, "AcmeTech_Employee_Handbook.pdf")

    assert output == note


def test_append_rag_observability_note_removes_generic_closing():
    content = "公司支持混合办公，每周最多可远程 3 天。\n\n如果还有其他问题，欢迎继续提问。"

    output = append_rag_observability_note(content, 3, "AcmeTech_Employee_Handbook.pdf")

    assert "如果还有其他问题" not in output
    assert "公司支持混合办公" in output


def test_build_rag_response_metadata():
    metadata = build_rag_response_metadata(
        {
            "hit_count": 2,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": "Remote Work Policy ...",
        }
    )

    assert metadata["knowledge_base"]["hit"] is True
    assert metadata["knowledge_base"]["hit_count"] == 2
    assert metadata["knowledge_base"]["source"] == "AcmeTech_Employee_Handbook.pdf"


def test_no_hit_template_is_stable():
    assert "没有检索到" in NO_HIT_RESPONSE_TEMPLATE
    assert "员工手册" in NO_HIT_RESPONSE_TEMPLATE


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
            },
            ensure_ascii=False,
        ),
    ):
        output = await retrieve_knowledge_base(state, config)

    assert output["knowledge_base_result"]["hit_count"] == 2
    assert output["knowledge_base_result"]["source"] == "AcmeTech_Employee_Handbook.pdf"


@pytest.mark.asyncio
async def test_acall_model_returns_stable_no_hit_response_without_model_call():
    state = {
        "messages": [HumanMessage(content="2026 年公司股票期权发放比例是多少？")],
        "knowledge_base_result": {
            "hit_count": 0,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": DEFAULT_NO_HIT_CONTEXT,
        },
    }
    config = RunnableConfig(configurable={})

    output = await acall_model(state, config)

    assert output["messages"][0].content == NO_HIT_RESPONSE_TEMPLATE
    assert output["messages"][0].response_metadata["knowledge_base"]["hit"] is False
    assert output["messages"][0].response_metadata["knowledge_base"]["hit_count"] == 0


@pytest.mark.asyncio
async def test_acall_model_appends_rag_metadata_on_hit():
    state = {
        "messages": [HumanMessage(content="员工手册里有没有提到远程办公政策？")],
        "knowledge_base_result": {
            "hit_count": 2,
            "source": "AcmeTech_Employee_Handbook.pdf",
            "context": "Remote Work Policy ...",
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
