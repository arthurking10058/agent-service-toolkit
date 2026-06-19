# 🧰 Agent Service Toolkit

这是一个基于 LangGraph、FastAPI 和 Streamlit 的 Agent 项目版本。

当前版本的重点不是继续复述上游仓库，而是把一个已经跑通、已经验证过、已经开始中文化的完整 Agent 工具链，整理成一个可继续开发、可准备上传、边界清晰的项目版本。

## 项目简介

这个项目目前包含一条完整的本地交互链路：

- `src/service/service.py` 提供 FastAPI 服务入口
- `src/client/client.py` 负责和服务通信
- `src/streamlit_app.py` 提供聊天式交互界面
- `src/agents/` 提供多个 Agent 实现
- `src/core/settings.py` 管理模型、环境变量和运行模式

当前这一版已经完成了基础运行验证，并在此基础上开始做中文化、配置收敛和项目边界整理。

## 当前版本定位

这不是一个全新从零搭建的模板仓库，而是基于现有可运行项目继续收敛后的版本。当前方向是：

- 保留当前仓库根目录作为真正可运行的项目主体
- 保留并继续完善 Streamlit 页面中文化
- 使用已经在本地跑通的 OpenAI-compatible 模型接法
- 排除本地记录、日志、缓存、数据库等非项目产物
- 逐步把 README、配置说明和界面表达改造成自己的版本

## 当前已验证状态

下面这些已经确认通过：

- FastAPI 服务可启动
- Streamlit 页面可打开
- `UI -> Client -> Service -> Agent` 链路已跑通
- fake model 演示模式可用
- 阿里云百炼 `OpenAI-compatible` 接法可用
- 页面主要可见文案已完成第一轮中文化

当前本地运行基线：

- FastAPI 端口：`8081`
- Streamlit 端口：`8501`
- 推荐默认模型：`openai-compatible`

## 目录说明

- `src/`：项目源码
- `tests/`：测试代码
- `docs/`：补充说明文档
- `media/`：项目图片资源
- `docker/`：Docker 相关配置
- `privatecredentials/`：本地凭据占位目录
- `.local/`：本地 SQLite 与运行数据目录，不上传
- `.env`：本地私有配置，不上传

## 快速开始

推荐按当前本地版本的方式启动。

### 1. 准备配置

```sh
cp .env.example .env
```

推荐按下面的方向填写：

```env
DEFAULT_MODEL=openai-compatible
COMPATIBLE_MODEL=qwen-plus
COMPATIBLE_API_KEY=your_compatible_api_key
COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PORT=8081
AGENT_URL=http://127.0.0.1:8081
```

如果当前 shell 或系统环境里残留 `OPENAI_API_KEY`，而你又只想保留 compatible provider，请先清掉它，否则 `AVAILABLE_MODELS` 里仍可能混入 `gpt-*`。

### 2. 安装依赖

```sh
uv sync --frozen
source .venv/bin/activate
```

### 3. 启动 FastAPI 服务

```sh
python src/run_service.py
```

### 4. 启动 Streamlit 界面

```sh
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

### 5. 访问页面

- Streamlit：`http://localhost:8501`
- FastAPI 文档：`http://127.0.0.1:8081/redoc`

## 推荐配置说明

当前这版更推荐使用 OpenAI-compatible 模式，而不是直接把 `OPENAI_API_KEY` 当默认起点。

推荐重点关注这些变量：

- `DEFAULT_MODEL=openai-compatible`
- `COMPATIBLE_MODEL`
- `COMPATIBLE_API_KEY`
- `COMPATIBLE_BASE_URL`
- `PORT=8081`
- `AGENT_URL=http://127.0.0.1:8081`

如果后续出现 provider 混乱，优先检查：

- 当前 shell 是否仍导出了 `OPENAI_API_KEY`
- 服务进程是否已经真正重启
- `src/core/settings.py` 实际读到的模型配置

## 当前主要能力

1. 多 Agent 切换与调用
2. 流式输出与工具调用展示
3. Streamlit 聊天界面
4. FastAPI 服务封装
5. 可切换的模型 provider 配置
6. 语音输入 / 输出扩展接口
7. RAG assistant 示例

## 后续适合继续改造的方向

- 继续完善页面中文化和品牌化
- 继续收敛默认 provider 策略
- 根据自己的使用场景定制 `src/agents/`
- 继续改造界面文案、项目命名和 README 结构

## 相关文档

- [Setting up Ollama](docs/Ollama.md)
- [Setting up VertexAI](docs/VertexAI.md)
- [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
- [Working with File-based Credentials](docs/File_Based_Credentials.md)

## 私有凭据文件

如果某些 Agent 或模型 provider 需要证书、凭据文件或其他本地私有材料，可以放到 `privatecredentials/`。该目录默认忽略真实内容，只保留 `.gitkeep` 占位文件。

## Docker

仓库保留了 `compose.yaml` 和 `docker/` 配置，后续如需回到 Docker 路线可以继续使用。

不过当前这版的主要验证工作是在本地 Python 运行路径下完成的，因此继续改造时更推荐优先沿用本地运行方式。

## 测试

可在本地虚拟环境中运行测试：

```sh
uv sync --frozen
source .venv/bin/activate
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
