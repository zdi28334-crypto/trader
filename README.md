# trader

A股量化研究 Agent 团队：输入一句研究需求，由多个 Agent 协作完成取数、分析、回测、风控审查，最后生成投研报告。

设计文档：[量化研究 Agent 团队 · 设计文档](https://claude.ai/artifact/VmiurkYicF6DwfakC5fN64)

> 本项目仅用于学习和研究，生成的内容不构成任何投资建议。

## 当前进度

- [x] M0 数据层：AkShare 取 A 股日线，本地 Parquet 缓存
- [ ] M1 MVP：单 Agent + 取数 / 指标 / 回测三个工具
- [ ] M2 多 Agent 团队（LangGraph）
- [ ] M3 Streamlit 界面、PDF 报告
- [ ] M4 进阶

## 快速开始

需要先安装 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/zdi28334-crypto/trader.git
cd trader
uv sync                                   # 安装依赖（自动建虚拟环境）
uv run trader-fetch 600519 --start 2020-01-01   # 下载贵州茅台 2020 年至今的前复权日线
uv run pytest                             # 跑测试（不联网）
```

第一次运行会从网上下载，数据存在 `data_cache/daily/`；再次运行同样的区间会直接读本地，不再联网。

在代码里使用：

```python
from trader.data import get_daily

df = get_daily("600519", "2023-01-01", "2023-12-31")  # 默认前复权
print(df[["open", "close", "volume"]].tail())
```

## 目录结构

```
src/trader/
  config.py        # 配置（缓存目录可用 TRADER_CACHE_DIR 环境变量修改）
  cli.py           # 命令行入口 trader-fetch
  data/market.py   # 取数 + 缓存
tests/             # 单元测试，用假数据，不依赖网络
```
