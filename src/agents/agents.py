from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.github_mcp_agent.github_mcp_agent import github_mcp_agent
from agents.interrupt_agent import interrupt_agent
from agents.knowledge_base_agent import kb_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.langgraph_supervisor_hierarchy_agent import langgraph_supervisor_hierarchy_agent
from agents.lazy_agent import LazyLoadingAgent
from agents.rag_assistant import rag_assistant
from agents.research_assistant import research_assistant
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"

# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel  # What get_agent() returns (always loaded)
AgentGraphLike = CompiledStateGraph | Pregel | LazyLoadingAgent  # What can be stored in registry


@dataclass
class Agent:
    description: str
    graph_like: AgentGraphLike


agents: dict[str, Agent] = {
    "chatbot": Agent(description="基础对话助手，适合快速验证链路。", graph_like=chatbot),
    "research-assistant": Agent(
        description="带网页检索和计算能力的研究助手。",
        graph_like=research_assistant,
    ),
    "rag-assistant": Agent(
        description="基于示例知识库的文档问答助手。",
        graph_like=rag_assistant,
    ),
    "command-agent": Agent(description="命令式任务演示助手。", graph_like=command_agent),
    "bg-task-agent": Agent(description="支持后台任务状态展示的演示助手。", graph_like=bg_task_agent),
    "langgraph-supervisor-agent": Agent(
        description="多助手调度演示助手。", graph_like=langgraph_supervisor_agent
    ),
    "langgraph-supervisor-hierarchy-agent": Agent(
        description="带层级子助手结构的多助手调度演示。",
        graph_like=langgraph_supervisor_hierarchy_agent,
    ),
    "interrupt-agent": Agent(
        description="支持中断与恢复交互的演示助手。", graph_like=interrupt_agent
    ),
    "knowledge-base-agent": Agent(
        description="接入 Amazon Bedrock Knowledge Base 的知识库助手。",
        graph_like=kb_agent,
    ),
    "github-mcp-agent": Agent(
        description="带 GitHub MCP 工具的仓库协作助手。",
        graph_like=github_mcp_agent,
    ),
}


async def load_agent(agent_id: str) -> None:
    """Load lazy agents if needed."""
    graph_like = agents[agent_id].graph_like
    if isinstance(graph_like, LazyLoadingAgent):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraph:
    """Get an agent graph, loading lazy agents if needed."""
    agent_graph = agents[agent_id].graph_like

    # If it's a lazy loading agent, ensure it's loaded and return its graph
    if isinstance(agent_graph, LazyLoadingAgent):
        if not agent_graph._loaded:
            raise RuntimeError(f"Agent {agent_id} not loaded. Call load() first.")
        return agent_graph.get_graph()

    # Otherwise return the graph directly
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
