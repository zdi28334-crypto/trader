"""M1：单个研究 Agent。

工作方式（经典的"工具调用循环"）：
1. 把用户问题和三个工具的说明发给大模型；
2. 大模型决定调哪个工具、传什么参数；我们在本地执行，把结果发回去；
3. 重复，直到大模型给出最终的报告正文。

关键约束：报告里的数字必须来自工具返回值。报告末尾会自动附上所有工具的原始输出，方便核对。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from trader.agent.llm import Reply, chat, pick_model
from trader.data import get_daily
from trader.tools import backtest, indicators

SYSTEM_PROMPT = """你是一名严谨的A股量化研究员。你可以调用工具获取真实数据、计算指标、运行回测。

工作要求：
- 先用 fetch_price_data 取数据，再用 get_indicators 看技术面，再用 run_sma_backtest 回测；需要时可以用不同参数多回测几组。
- 报告中出现的每一个数字都必须来自工具的返回结果，禁止编造或估算数字。
- 回测结果要和"买入持有"对比，并指出局限性（样本区间、过拟合风险、未考虑的成本等）。
- 不给出买卖建议，只做研究分析。

工具都用完后，直接输出 Markdown 格式的研究报告正文（不要用代码块包起来），结构：
# 标题
## 结论摘要
## 行情与技术面
## 策略回测
## 风险与局限
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_price_data",
            "description": "下载并缓存一只A股的前复权日线数据，返回数据概况。其他工具使用前必须先调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "6 位股票代码，如 600519"},
                    "start": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                    "end": {"type": "string", "description": "结束日期 YYYY-MM-DD，可省略表示今天"},
                },
                "required": ["symbol", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicators",
            "description": "计算最新交易日的技术指标：MA5/20/60、RSI14、MACD、近 20/60/250 日涨跌幅、250 日高低点。",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sma_backtest",
            "description": "回测均线交叉策略（快线上穿慢线买入、下穿卖出，次日执行，含交易成本），返回收益、夏普、最大回撤、胜率，并与买入持有对比。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "fast": {"type": "integer", "description": "快均线天数，默认 20"},
                    "slow": {"type": "integer", "description": "慢均线天数，默认 60"},
                },
                "required": ["symbol"],
            },
        },
    },
]

ChatFn = Callable[[list[dict], list[dict], str], Reply]


@dataclass
class ResearchResult:
    report_path: Path
    report: str
    tool_log: list[dict] = field(default_factory=list)


class ResearchAgent:
    def __init__(
        self,
        model: str | None = None,
        chat_fn: ChatFn = chat,
        data_fn=get_daily,
        report_dir: Path = Path("reports"),
        max_steps: int = 10,
        verbose: bool = True,
    ):
        self.model = model or (pick_model() if chat_fn is chat else "fake")
        self.chat_fn = chat_fn
        self.data_fn = data_fn
        self.report_dir = Path(report_dir)
        self.max_steps = max_steps
        self.verbose = verbose
        self.frames: dict[str, pd.DataFrame] = {}
        self.charts: list[Path] = []
        self.tool_log: list[dict] = []

    # ---------- 三个工具 ----------
    def fetch_price_data(self, symbol: str, start: str, end: str | None = None) -> dict:
        df = self.data_fn(symbol, start, end)
        if df.empty:
            return {"error": f"{symbol} 在该区间没有数据"}
        self.frames[symbol] = df
        return {
            "symbol": symbol,
            "rows": len(df),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "first_close": round(float(df["close"].iloc[0]), 2),
            "last_close": round(float(df["close"].iloc[-1]), 2),
        }

    def get_indicators(self, symbol: str) -> dict:
        df = self._frame(symbol)
        return indicators.summarize(df)

    def run_sma_backtest(self, symbol: str, fast: int = 20, slow: int = 60) -> dict:
        df = self._frame(symbol)
        result, daily = backtest.sma_cross(df, fast=fast, slow=slow)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        chart = self.report_dir / f"{symbol}_sma{fast}_{slow}_{date.today():%Y%m%d}.png"
        backtest.plot_equity(daily, f"{symbol} SMA({fast},{slow}) vs Buy & Hold", chart)
        if chart not in self.charts:
            self.charts.append(chart)
        return result.to_dict()

    def _frame(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.frames:
            raise ValueError(f"还没有 {symbol} 的数据，请先调用 fetch_price_data")
        return self.frames[symbol]

    def _run_tool(self, name: str, args: dict) -> dict:
        fn = {
            "fetch_price_data": self.fetch_price_data,
            "get_indicators": self.get_indicators,
            "run_sma_backtest": self.run_sma_backtest,
        }.get(name)
        if fn is None:
            return {"error": f"没有这个工具：{name}"}
        try:
            return fn(**args)
        except Exception as e:  # 把错误交还给模型，让它自己调整
            return {"error": f"{type(e).__name__}: {e}"}

    # ---------- 主循环 ----------
    def run(self, question: str) -> ResearchResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT + f"\n今天是 {date.today()}。"},
            {"role": "user", "content": question},
        ]
        for step in range(self.max_steps):
            reply = self.chat_fn(messages, TOOLS, self.model)
            if not reply.tool_calls:
                return self._finish(question, reply.content or "")

            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
                        }
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                out = self._run_tool(call.name, call.arguments)
                self.tool_log.append({"tool": call.name, "args": call.arguments, "result": out})
                if self.verbose:
                    print(f"[{step + 1}] {call.name}({call.arguments}) -> {str(out)[:120]}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(out, ensure_ascii=False)}
                )
        raise RuntimeError(f"Agent 在 {self.max_steps} 步内没有完成，请简化问题后重试")

    def _finish(self, question: str, body: str) -> ResearchResult:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        symbols = "_".join(self.frames) or "report"
        path = self.report_dir / f"{symbols}_{datetime.now():%Y%m%d_%H%M%S}.md"

        parts = [body.strip(), ""]
        for chart in self.charts:
            parts.append(f"![净值曲线]({chart.name})\n")
        parts.append("## 附录：工具原始输出\n")
        parts.append("报告中的数字均应能在下表中找到出处。\n")
        for i, entry in enumerate(self.tool_log, 1):
            parts.append(f"{i}. `{entry['tool']}({json.dumps(entry['args'], ensure_ascii=False)})`\n")
            parts.append("```json\n" + json.dumps(entry["result"], ensure_ascii=False, indent=2) + "\n```\n")
        parts.append(f"---\n研究问题：{question}  \n模型：{self.model}  \n")
        parts.append("> 本报告由程序自动生成，仅用于学习研究，不构成任何投资建议。\n")

        report = "\n".join(parts)
        path.write_text(report, encoding="utf-8")
        return ResearchResult(report_path=path, report=report, tool_log=self.tool_log)
