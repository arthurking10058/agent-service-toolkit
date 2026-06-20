import json
from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph

from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from agents.tools import database_search, database_search_func
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: SafeguardOutput
    knowledge_base_result: dict[str, str | int]


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are AcmeBot, a knowledge-base assistant focused on answering questions from AcmeTech's Employee Handbook.
    Today's date is {current_date}.

    IMPORTANT:
    - The user cannot see the raw tool response.
    - The tool returns structured knowledge-base search results containing hit_count, source and context.
    - Only answer with information supported by the retrieved context.
    - When retrieved context is available, treat it as the content of the handbook you are allowed to use.
    - Do not claim that you cannot access the handbook, that you need the user to upload the file, or that you need the company name.
    - Do not switch to generic HR advice when the retrieved context already contains relevant information.

    Response rules:
    - If hit_count is 0, clearly say that the current example knowledge base did not return relevant content.
    - If content is found, answer the user's question directly in a concise, helpful and friendly tone.
    - Prefer summarizing the retrieved policy first, instead of explaining what you theoretically would do.
    - Mention that the answer is based on the Employee Handbook example knowledge base when appropriate.
    - Do not invent policies, benefits or citations that are not present in the retrieved context.
    """

NO_HIT_RESPONSE_TEMPLATE = (
    "当前示例员工手册知识库没有检索到与这个问题直接相关的内容。"
    "你可以换一种问法，或直接询问员工手册、福利、远程办公、休假政策等主题。"
)
RAG_RESPONSE_METADATA_KEY = "knowledge_base"


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    def build_messages(state: AgentState):
        knowledge_base_result = extract_knowledge_base_result(state) or {}
        hit_count = knowledge_base_result.get("hit_count", 0)
        source = knowledge_base_result.get("source", "示例知识库")
        context = knowledge_base_result.get("context", "")

        contextual_instructions = instructions
        if hit_count and context:
            contextual_instructions += (
                f"\n\n当前知识库来源：{source}"
                f"\n当前命中片段数：{hit_count}"
                f"\n以下是本次检索到的上下文，你必须把它当作当前可访问的员工手册内容，并优先直接回答用户问题：\n{context}"
            )

        return [SystemMessage(content=contextual_instructions)] + state["messages"]

    preprocessor = RunnableLambda(
        build_messages,
        name="StateModifier",
    )
    return preprocessor | model  # type: ignore[return-value]


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


def extract_knowledge_base_result(state: AgentState) -> dict | None:
    """从最近一次工具结果中提取结构化知识库检索信息。"""
    if "knowledge_base_result" in state and state["knowledge_base_result"]:
        return state["knowledge_base_result"]
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            try:
                data = json.loads(message.content)
            except Exception:
                continue
            if isinstance(data, dict) and {"hit_count", "source", "context"} <= data.keys():
                return data
    return None


def append_rag_observability_note(content: str, hit_count: int, source: str) -> str:
    """给回答补一条更自然的轻量来源说明。"""
    note = f"以上回答基于示例知识库 {source}，共参考了 {hit_count} 个相关片段。"
    cleaned_content = content.strip()
    cleaned_content = cleaned_content.replace(
        "如果您对手册中的其他内容还有疑问，欢迎随时向我提问！", ""
    ).replace(
        "如果您还有其他问题，欢迎随时提问！", ""
    ).replace(
        "以上回答基于 AcmeTech 员工手册示例知识库（参考了 1 个相关片段）。", ""
    ).replace(
        "以上回答基于 AcmeTech 员工手册示例知识库。", ""
    ).replace(
        "以上信息均基于AcmeTech员工手册（Employee Handbook）中的相关规定。", ""
    ).strip()
    if not cleaned_content:
        return note
    if note in cleaned_content:
        return cleaned_content
    return f"{cleaned_content}\n\n{note}"


def build_rag_response_metadata(knowledge_base_result: dict[str, str | int] | None) -> dict:
    """把知识库命中信息写入响应元数据，便于服务层和 UI 观察。"""
    if not knowledge_base_result:
        return {}

    hit_count = int(knowledge_base_result.get("hit_count", 0) or 0)
    source = str(knowledge_base_result.get("source", "示例知识库") or "示例知识库")
    return {
        RAG_RESPONSE_METADATA_KEY: {
            "hit": hit_count > 0,
            "hit_count": hit_count,
            "source": source,
        }
    }


async def retrieve_knowledge_base(state: AgentState, config: RunnableConfig) -> AgentState:
    """强制先执行知识库检索，再进入回答阶段。"""
    latest_human_message = next(
        (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        None,
    )
    if latest_human_message is None:
        return {"knowledge_base_result": {"hit_count": 0, "source": "", "context": ""}}

    raw_result = database_search_func(latest_human_message.content)
    try:
        result = json.loads(raw_result)
    except Exception:
        result = {
            "hit_count": 0,
            "source": "示例知识库",
            "context": "知识库检索结果解析失败。",
        }
    return {"knowledge_base_result": result}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    knowledge_base_result = extract_knowledge_base_result(state)
    if knowledge_base_result and knowledge_base_result.get("hit_count", 0) == 0:
        return {
            "messages": [
                AIMessage(
                    content=NO_HIT_RESPONSE_TEMPLATE,
                    response_metadata=build_rag_response_metadata(knowledge_base_result),
                )
            ]
        }

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if knowledge_base_result:
        hit_count = knowledge_base_result.get("hit_count", 0)
        source = knowledge_base_result.get("source", "示例知识库")
        response = AIMessage(
            content=append_rag_observability_note(response.content, hit_count, source),
            id=response.id,
            tool_calls=response.tool_calls,
            response_metadata={
                **(response.response_metadata or {}),
                **build_rag_response_metadata(knowledge_base_result),
            },
        )

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("guard_input", safeguard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("retrieve_knowledge_base", retrieve_knowledge_base)
agent.add_node("model", acall_model)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "retrieve_knowledge_base"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# 检索完成后固定进入回答阶段；回答后直接结束
agent.add_edge("retrieve_knowledge_base", "model")
agent.add_edge("model", END)

rag_assistant = agent.compile()
