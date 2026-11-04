"""命令行入口：uv run trader-fetch 600519 --start 2020-01-01"""

import argparse

from trader.data import get_daily


def fetch_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="下载并缓存A股日线数据")
    p.add_argument("symbol", help="6 位股票代码，如 600519")
    p.add_argument("--start", default="2020-01-01", help="开始日期，默认 2020-01-01")
    p.add_argument("--end", default=None, help="结束日期，默认今天")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""], help="复权方式")
    args = p.parse_args(argv)

    df = get_daily(args.symbol, args.start, args.end, adjust=args.adjust)
    if df.empty:
        print("没有取到数据，请检查代码和日期。")
        return
    print(f"{args.symbol}: {len(df)} 条日线，{df.index[0].date()} 至 {df.index[-1].date()}")
    print(df.tail(5).to_string())
