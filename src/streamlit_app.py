import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# 这是一个基于 Streamlit 的聊天界面，用来和后端智能体服务交互。
# 核心流程分为三部分：
#
# - main()：初始化页面、侧边栏和交互入口
# - draw_messages()：渲染历史消息和流式消息
# - handle_feedback()：收集用户反馈
#
# 页面层主要通过 AgentClient 调用 FastAPI 服务端接口。


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def format_model_label(model: str) -> str:
    """把模型标识转换成更适合界面展示的文本。"""
    if model == "openai-compatible":
        return "openai-compatible（兼容模式）"
    if model == "fake":
        return "fake（演示模式）"
    return model


def render_service_overview(service_info, agent_description_map: dict[str, str]) -> None:
    """在首页主区域展示当前服务摘要、默认配置和可用助手。"""
    with st.container(border=True):
        st.subheader("当前服务概览")
        if service_info.service_summary:
            st.write(service_info.service_summary)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("默认助手")
            st.write(f"`{service_info.default_agent}`")
            desc = agent_description_map.get(service_info.default_agent)
            if desc:
                st.write(desc)
        with col2:
            st.caption("默认模型")
            st.write(f"`{format_model_label(str(service_info.default_model))}`")

        if service_info.available_providers:
            st.caption("当前可用 Provider")
            st.write(" | ".join(service_info.available_providers))

        if service_info.configuration_warnings:
            st.warning("\n".join(service_info.configuration_warnings))

        with st.expander("查看可用助手与模型", expanded=False):
            st.write("可用助手：")
            for agent in service_info.agents:
                st.write(f"- `{agent.key}`：{agent.description}")
            st.write("可用模型：")
            for item in service_info.models:
                st.write(f"- `{format_model_label(str(item))}`")


def dialog_or_inline(title: str):
    """优先使用 st.dialog；旧版 Streamlit 环境则退化为普通函数。"""
    if hasattr(st, "dialog"):
        return st.dialog(title)

    def decorator(func):
        return func

    return decorator


def get_or_create_user_id() -> str:
    """从会话或 URL 里获取用户标识；没有就自动创建。"""
    # 先从当前会话里读取
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # 再尝试从 URL 参数恢复
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # 都没有时就新建一个
    user_id = str(uuid.uuid4())

    # 写入当前会话
    st.session_state[USER_ID_COOKIE] = user_id

    # 同时写回 URL，方便分享和恢复
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # 隐藏右上角默认状态控件，保留更干净的界面
    st.markdown(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # 获取或创建当前用户标识
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("正在连接服务端..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"连接服务端失败：{agent_url}\n\n{e}")
            st.markdown("服务可能还在启动中，请稍等几秒后重试。")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client
    service_info = agent_client.info
    agent_description_map = {agent.key: agent.description for agent in service_info.agents}

    # 每个会话只初始化一次语音管理器
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("没有找到这个 Thread ID 对应的消息历史。")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # 侧边栏配置区
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "这是一个基于 LangGraph、FastAPI 和 Streamlit 的可运行智能体项目骨架。"
        ""

        if st.button(":material/chat: 新对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # 新对话时清空上一次缓存的语音内容
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.popover(":material/settings: 设置", use_container_width=True):
            model_idx = service_info.models.index(service_info.default_model)
            model = st.selectbox(
                "选择模型",
                options=service_info.models,
                index=model_idx,
                format_func=format_model_label,
            )
            agent_list = [a.key for a in service_info.agents]
            agent_idx = agent_list.index(service_info.default_agent)
            agent_client.agent = st.selectbox(
                "选择助手",
                options=agent_list,
                index=agent_idx,
                format_func=lambda key: key,
            )
            st.caption(agent_description_map.get(agent_client.agent, ""))
            use_streaming = st.toggle("启用流式输出", value=True)
            # 关闭语音时顺手清掉缓存的语音结果
            enable_audio = st.toggle(
                "启用语音生成",
                value=True,
                disabled=not voice or not voice.tts,
                help="如需启用，请在 .env 中配置 VOICE_TTS_PROVIDER"
                if not voice or not voice.tts
                else None,
                on_change=lambda: st.session_state.pop("last_audio", None)
                if not st.session_state.get("enable_audio", True)
                else None,
                key="enable_audio",
            )

            # 展示当前会话使用的用户标识
            st.text_input("用户标识（只读）", value=user_id, disabled=True)

        @dialog_or_inline("架构")
        def architecture_dialog() -> None:
            st.image("media/agent_architecture.png")
            st.caption("当前显示的是仓库内置的架构示意图。")

        if st.button(":material/schema: 架构图", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/info: 服务信息", use_container_width=True):
            if service_info.service_summary:
                st.write(service_info.service_summary)
            st.write(f"默认助手：`{service_info.default_agent}`")
            st.write(f"默认模型：`{format_model_label(str(service_info.default_model))}`")
            if service_info.available_providers:
                st.write("当前可用 Provider：")
                for provider in service_info.available_providers:
                    st.write(f"- {provider}")
            if service_info.configuration_warnings:
                st.warning("\n".join(service_info.configuration_warnings))
            st.write("可用助手：")
            for agent in service_info.agents:
                st.write(f"- `{agent.key}`：{agent.description}")

        with st.popover(":material/policy: 隐私说明", use_container_width=True):
            st.write(
                "本应用中的提示词、回答和反馈可能会以匿名方式记录，用于调试、评估和后续改进。"
            )

        @dialog_or_inline("分享 / 恢复对话")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # 如果不是 localhost，默认优先展示 https 链接
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # 同时带上 thread_id 和 user_id，方便恢复同一会话
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**会话分享链接：**\n```text\n{chat_url}\n```")
            st.info("复制这个链接后，可以分享或再次打开当前会话。")

        if st.button(":material/upload: 分享 / 恢复对话", use_container_width=True):
            share_chat_dialog()

        st.caption("当前界面正在使用仓库内的本地配置运行。")

    # 渲染已有消息
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        render_service_overview(service_info, agent_description_map)
        match agent_client.agent:
            case "chatbot":
                WELCOME = "你好！我是一个基础对话助手，你可以直接向我提问。"
            case "interrupt-agent":
                WELCOME = "你好！我是一个可中断交互演示助手。告诉我你的生日，我会继续完成这次演示。"
            case "research-assistant":
                WELCOME = "你好！我是一个带有网页检索和计算能力的研究助手，你可以直接向我提问。"
            case "rag-assistant":
                WELCOME = """你好！我是一个接入示例知识库的文档问答助手，可以基于内置资料回答问题。
                你可以把这里当作知识库检索与问答的演示入口，直接向我提问。"""
            case _:
                WELCOME = "你好！我是当前项目里的默认助手，你可以直接向我提问。"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() 需要异步迭代器形式的消息输入
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # 重新渲染最近一条回答的语音，保证 rerun 后仍能保留
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # 处理用户新输入：优先走语音输入，否则使用普通输入框
    # 若要启用语音功能，需要在应用侧 .env 中配置
    # VOICE_STT_PROVIDER、VOICE_TTS_PROVIDER 和 OPENAI_API_KEY
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input("请输入你的消息")

    if user_input:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # 为流式回答补一段语音输出
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # 只有 AI 文本回答才生成语音
                    if last_msg.type == "ai" and last_msg.content:
                        # 文本已经流式渲染，这里只补音频
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # 非流式模式下直接渲染回答和可选语音
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            st.rerun()  # 清理旧容器，避免界面残留
        except AgentClientError as e:
            st.error(f"生成回答时出错：{e}")
            st.stop()

    # 只要已经有消息，就显示反馈组件
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    渲染一组消息，既支持回放历史消息，也支持展示流式消息。

    这里还额外处理了：
    - 流式 token 的增量渲染
    - 工具调用状态展示
    - 最后一条消息容器的追踪，方便后续挂反馈组件

    Args:
        messages_aiter: 待渲染消息的异步迭代器
        is_new: 当前是否为新生成消息
    """

    # 记录最近一次消息容器
    last_message_type = None
    st.session_state.last_message = None

    # 用于流式 token 的临时占位
    streaming_content = ""
    streaming_placeholder = None

    # 逐条渲染消息
    while msg := await anext(messages_agen, None):
        # 字符串消息表示流式输出中的中间 token
        if isinstance(msg, str):
            # 第一个 token 到来时先创建占位容器
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"收到未预期的消息类型：{type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # 用户消息：直接渲染
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # AI 消息：需要处理流式输出和工具调用
            case "ai":
                # 新消息写回会话状态
                if is_new:
                    st.session_state.messages.append(msg)

                # 如果前一条不是 AI 消息，就新建一个 AI 容器
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # 先渲染正文，再准备后续工具调用区域
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # 为每个工具调用创建单独状态块，并用 ID 建立映射
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # 转交助手和普通工具调用用不同标签
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 子助手：{tool_call["name"]}"""
                            else:
                                label = f"""🛠️ 工具调用：{tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # 每个工具调用后面都应跟一个 ToolMessage
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # 走到这里的是普通工具调用
                            status = call_results[tool_call["id"]]
                            status.write("输入：")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"收到未预期的 ChatMessage 类型：{tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # 新消息写回状态，并把结果更新到对应状态块
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("输出：")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # bg-task-agent 使用的自定义任务数据
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("收到 Agent 返回的未预期 CustomData 消息")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # 其他未知消息类型直接报错
            case _:
                st.error(f"收到未预期的 ChatMessage 类型：{msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """渲染反馈组件，并把反馈提交给后端。"""

    # 避免重复提交相同反馈
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # 反馈值或运行 ID 变化时再提交
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # 把星级索引换算成 0 到 1 的分数
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"记录反馈时出错：{e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("反馈已记录", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    把子助手执行过程收纳到状态容器里展示。

    它会从第一次转交消息开始，持续读取后续消息，
    直到子助手完成并把控制权交回上层。

    Args:
        messages_agen: 消息异步生成器
        status: 当前子助手对应的状态容器
        is_new: 当前是否为新消息
    """
    nested_popovers = {}

    # 第一条通常是转交成功的工具消息
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # 持续读取，直到明确收到交回控制权的消息
    while True:
        # 读取下一条子助手消息
        sub_msg = await anext(messages_agen)

        # 理论上这里只有结构化消息；保留给后续流式扩展
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # 工具结果如果对应已登记的弹出层，就直接补进去
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**输出：**")
            popover.write(sub_msg.content)
            continue

        # transfer_back_to 表示子助手把控制权交还上层
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # 处理所有交回控制权的工具调用
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # 读取对应的工具返回结果
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # 控制权已经交回，当前子助手流程结束
            if status:
                status.update(state="complete")
            break

        # 在同一个嵌套状态块里展示文本和工具调用
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # 如果又转交给了下一级子助手，就递归展示
                    if "transfer_to" in tc["name"]:
                        # 为下一级子助手创建嵌套状态块
                        nested_status = status.status(
                            f"""💼 子助手：{tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # 递归处理下一层子助手
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # 普通工具调用则用弹出层展示详情
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**工具：** {tc['name']}")
                        popover.write("**输入：**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
