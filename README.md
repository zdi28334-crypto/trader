# trader

A股量化研究 Agent 团队：输入一句研究需求，由多个 Agent 协作完成取数、分析、回测、风控审查，最后生成投研报告。

设计文档：[量化研究 Agent 团队 · 设计文档](https://claude.ai/artifact/VmiurkYicF6DwfakC5fN64)

> 本项目仅用于学习和研究，生成的内容不构成任何投资建议。

## 当前进度

- [x] M0 数据层：AkShare 取 A 股日线，本地 Parquet 缓存
- [x] M1 MVP：单 Agent + 取数 / 指标 / 回测三个工具，自动生成研究报告
- [ ] M2 多 Agent 团队（LangGraph）
- [ ] M3 Streamlit 界面、PDF 报告
- [ ] M4 进阶

## 快速开始

需要先安装 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/zdi28334-crypto/trader.git
cd trader
uv sync                                   # 安装依赖（自动建虚拟环境）
uv run python -m trader fetch 600519 --start 2020-01-01   # 下载贵州茅台 2020 年至今的前复权日线
uv run pytest                             # 跑测试（不联网）
```

> macOS 提示：也可以用 `uv run trader fetch 600519`，但如果 `.venv` 被系统标成隐藏，Python 会忽略 uv 写的 `.pth` 文件，报 `No module named 'trader'`。上面的 `python -m trader` 写法不受影响；也可以运行 `chflags -R nohidden .venv` 修复。

第一次运行会从网上下载，数据存在 `data_cache/daily/`；再次运行同样的区间会直接读本地，不再联网。

## 让 Agent 做研究（M1）

1. 复制配置文件并填入 API key（DeepSeek 或 Claude 任选其一）：

   ```bash
   cp .env.example .env
   # 然后用编辑器打开 .env，填 DEEPSEEK_API_KEY=... 或 ANTHROPIC_API_KEY=...
   ```

2. 提一个研究问题：

   ```bash
   uv run python -m trader research "研究一下贵州茅台（600519）2020 年以来的走势，回测 20/60 日均线策略，再换一组参数对比"
   ```

Agent 会自己决定依次调用三个工具：`fetch_price_data` 取数、`get_indicators` 看技术面、`run_sma_backtest` 回测。终端会打印每一步，最后在 `reports/` 下生成 Markdown 报告和净值图。报告末尾附有所有工具的原始输出，可以核对报告里每个数字的出处。

想换模型：`--model anthropic/claude-sonnet-5-5`，或在 `.env` 里设 `TRADER_MODEL`。

在代码里使用：

```python
from trader.data import get_daily

df = get_daily("600519", "2023-01-01", "2023-12-31")  # 默认前复权
print(df[["open", "close", "volume"]].tail())
```

## 目录结构

```
trader/
  __main__.py          # python -m trader 入口
  cli.py               # 命令行：fetch / research
  config.py            # 配置（缓存目录可用 TRADER_CACHE_DIR 环境变量修改）
  data/market.py       # 取数 + 缓存
  tools/indicators.py  # 技术指标：MA、RSI、MACD
  tools/backtest.py    # 均线交叉回测 + 净值图
  agent/llm.py         # LiteLLM 封装，DeepSeek / Claude 二选一
  agent/single.py      # 单 Agent 工具调用循环 + 报告生成
tests/             # 单元测试，用假数据，不依赖网络
```
