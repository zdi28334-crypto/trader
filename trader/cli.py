"""命令行入口。

    uv run python -m trader fetch 600519 --start 2020-01-01
    uv run python -m trader research "研究一下贵州茅台，回测 20/60 日均线策略"
    uv run python -m trader strategy 600519 000858 300750 --emotion 300750 --start 2023-01-01
"""

from __future__ import annotations

import argparse
import sys

from trader.data import get_daily


def _fetch(args: argparse.Namespace) -> None:
    df = get_daily(args.symbol, args.start, args.end, adjust=args.adjust)
    if df.empty:
        print("没有取到数据，请检查代码和日期。")
        return
    print(f"{args.symbol}: {len(df)} 条日线，{df.index[0].date()} 至 {df.index[-1].date()}")
    print(df.tail(5).to_string())


def _research(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    from trader.agent.single import ResearchAgent

    load_dotenv()
    try:
        agent = ResearchAgent(model=args.model)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"使用模型：{agent.model}\n")
    result = agent.run(args.question)
    print(f"\n报告已生成：{result.report_path}")


def _strategy(args: argparse.Namespace) -> None:
    """不用大模型，直接跑我的短线策略回测，打印结果并存图。"""
    import json
    from datetime import date
    from pathlib import Path

    from trader.data.index import get_market
    from trader.strategies import short_term

    emotion = set(args.emotion or [])
    stocks = {s: get_daily(s, args.start, args.end) for s in args.symbols}
    stocks = {s: df for s, df in stocks.items() if not df.empty}
    if not stocks:
        sys.exit("这些股票都没有取到数据。")
    market = get_market(args.start, args.end)
    kinds = {s: short_term.EMOTION for s in emotion}
    result, daily = short_term.backtest(stocks, kinds, market)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    chart = out_dir / f"my_short_term_{date.today():%Y%m%d}.png"
    short_term.plot(daily, "My short-term strategy vs SSE Composite", chart)

    print("当前大盘环境：")
    print(json.dumps(short_term.latest_snapshot(market), ensure_ascii=False, indent=2))
    print("\n回测结果：")
    summary = result.to_dict(max_trades=0)
    summary.pop("trade_list")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n共 {result.trades} 笔已平仓交易，最近 10 笔：")
    for tr in result.trade_list[-10:]:
        print(
            f"  {tr['symbol']}({tr['kind']}) {tr['buy_date']} 买 {tr['buy_price']} → "
            f"{tr['sell_date']} 卖 {tr['sell_price']}  {tr['return_pct']:+.2f}%  {tr['reason']}"
        )
    print(f"\n净值图：{chart}")


def _add_fetch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("symbol", help="6 位股票代码，如 600519")
    p.add_argument("--start", default="2020-01-01", help="开始日期，默认 2020-01-01")
    p.add_argument("--end", default=None, help="结束日期，默认今天")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""], help="复权方式")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trader", description="A股量化研究 Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="下载并缓存日线数据")
    _add_fetch_args(f)
    f.set_defaults(func=_fetch)

    r = sub.add_parser("research", help="让 Agent 做一次研究并生成报告")
    r.add_argument("question", help='研究问题，例如 "研究一下 600519，回测均线策略"')
    r.add_argument("--model", default=None, help="模型名，如 deepseek/deepseek-chat；默认按 .env 里的 key 自动选")
    r.set_defaults(func=_research)

    s = sub.add_parser("strategy", help="不用大模型，直接回测我的短线策略")
    s.add_argument("symbols", nargs="+", help="股票代码，可以多只")
    s.add_argument("--emotion", nargs="*", help="其中属于情绪类小票的代码")
    s.add_argument("--start", default="2023-01-01", help="开始日期，默认 2023-01-01")
    s.add_argument("--end", default=None, help="结束日期，默认今天")
    s.set_defaults(func=_strategy)

    args = p.parse_args(argv)
    args.func(args)


def fetch_main(argv: list[str] | None = None) -> None:
    """兼容旧的 trader-fetch 命令。"""
    main(["fetch", *(sys.argv[1:] if argv is None else argv)])
