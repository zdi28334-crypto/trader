# AGENTS.md

写给接手这个仓库的 AI 编码助手（Codex 等）。完整的来龙去脉、回测数字和待办在 [HANDOFF.md](HANDOFF.md)，动手前先读它。

## 项目一句话

A 股量化研究 Agent：用户输入一句研究需求，Agent 调真实数据和回测工具，生成带图的 Markdown 研究报告。终点是 5 个角色的多 Agent 团队（LangGraph）。仅用于学习研究，不接实盘、不给投资建议。

## 和用户打交道

- 用户 Di少（GitHub `zdi28334-crypto`）是 agent 开发初学者。**用中文回复**，讲大白话，少用术语。
- 用户更希望你直接在 Mac 上替他跑命令，而不是让他自己敲。真要他动手时，给可以整段复制的命令。
- **策略规则归用户所有。** 用户明确说过不想让项目全是 AI 的选择。任何交易规则、阈值、股票池的改动，先用大白话讲清楚、得到用户同意再改；你只负责把他的话翻译成代码和回测。用户没说的数字做成参数，标明"用户未定"。
- API key 只放 `.env`（已被 git 忽略），不要让用户把 key 贴进聊天，也不要提交。

## 环境与命令

- Python 3.12，包管理用 [uv](https://docs.astral.sh/uv/)。用户 Mac 上的代码在 `~/Desktop/01_课程学习/trader`。
- 安装依赖：`uv sync`
- 跑测试（离线、用假数据）：`uv run pytest`
- 取数：`uv run python -m trader fetch 600519 --start 2020-01-01`
- Agent 研究（需要 `.env` 里的 DeepSeek 或 Anthropic key）：`uv run python -m trader research "研究一下 600519，回测 20/60 均线"`
- 不用大模型直接回测用户策略：`uv run python -m trader strategy 600900 601088 --start 2023-01-01`
- 一律用 `python -m trader`，不要用 `uv run trader`：macOS 会把 `.venv` 标成隐藏，Python 跳过 `.pth`，报 `No module named 'trader'`（修复：`chflags -R nohidden .venv`）。
- AkShare 数据来自东方财富网页，**只能在用户本机联网取数**；云端/沙箱环境通常连不上（403）。需要真实数据的验证放到用户 Mac 上跑。

## 代码约定

- 平铺布局：包在 `trader/`（不是 `src/`；用户 Mac 上残留的 `src/` 是旧布局，可忽略）。
- 注释、docstring、CLI 输出、报告用中文；标识符用英文。
- 数据层：`trader/data/market.py`（个股日线 + Parquet 缓存）、`trader/data/index.py`（上证指数 + 两市成交额 = sh000001 + sz399106）。新的数据源也要先缓存再用。
- **数字只能来自工具返回值。** Agent 的系统提示词禁止编数字，报告末尾自动附上所有工具原始输出。新增 Agent 或工具时保持这条。
- **回测防未来函数**：第 t 天收盘后出信号，第 t+1 天开盘价成交（A 股 T+1）。回测都要含交易成本（默认 15bp/次）。挑股票池只能用当时已知的数据（参考 `trader/strategies/pool.py` 的每月滚动选池）。
- 策略的所有数字集中在参数 dataclass 里（例：`ShortTermParams`），改数字不动逻辑。每个策略一个文件，放 `trader/strategies/`。
- 测试不联网：用 `tests/conftest.py` 里的假行情，取数函数通过注入 fetcher 来替换。新增功能都要配离线测试。
- LLM 调用统一走 `trader/agent/llm.py`（LiteLLM）。litellm 传 `tools` 时需要 `mcp` 包，已在依赖里。

## Git

- 默认分支 `main`。每项工作开新分支、开 PR，用中文写 PR 标题和说明。
- 用户网页上登录的是另一个 GitHub 账号，常让 AI 帮他合并 PR；合并前要用户明确说"合并"。
- `reports/`、`data_cache/`、`.env` 都被 git 忽略，回测产物不要提交。
