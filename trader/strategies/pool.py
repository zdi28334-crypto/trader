"""情绪票股票池：每个月初只用"当时已经知道"的数据重新挑一次，避免事后挑票。

挑法（对应说明书"题材偏情绪类小票"）：
- 回看过去 60 个交易日（不含当天），数每只票涨停了几次：
  主板涨幅 >= 9.8% 算涨停，创业板(30)/科创板(68) >= 19.8%；
- 流通市值 < 150 亿，用回看期最后一天的 成交额 / 换手率 估算（不用今天的市值，避免偷看）；
- 至少涨停 2 次，按次数取前 20 只。
"""

from __future__ import annotations

import pandas as pd


def limit_up_threshold(symbol: str) -> float:
    return 19.8 if symbol.startswith(("30", "68")) else 9.8


def float_cap(df: pd.DataFrame) -> pd.Series:
    """流通市值估算（元）= 成交额 / (换手率 / 100)。"""
    return df["amount"] / (df["turnover"] / 100)


def monthly_emotion_pool(
    stocks: dict[str, pd.DataFrame],
    start: str,
    end: str | None = None,
    lookback: int = 60,
    top: int = 20,
    max_float_cap: float = 150e8,
    min_limit_ups: int = 2,
) -> dict[pd.Timestamp, list[str]]:
    """返回 {每月第一个交易日: 当月情绪票列表}。"""
    all_dates = pd.DatetimeIndex(sorted(set().union(*(d.index for d in stocks.values()))))
    sel = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end or all_dates[-1]))]
    month_starts = pd.Series(sel, index=sel).groupby(sel.to_period("M")).first().tolist()

    pools = {}
    for t in month_starts:
        window = all_dates[all_dates < t][-lookback:]
        if len(window) < lookback:
            continue
        lo, hi = window[0], window[-1]
        scores = []
        for s, df in stocks.items():
            w = df.loc[lo:hi]
            if len(w) < lookback // 2 or pd.isna(w["turnover"].iloc[-1]) or w["turnover"].iloc[-1] <= 0:
                continue
            if float_cap(w).iloc[-1] >= max_float_cap:
                continue
            n = int((w["pct_chg"] >= limit_up_threshold(s)).sum())
            if n >= min_limit_ups:
                scores.append((n, s))
        scores.sort(key=lambda x: (-x[0], x[1]))
        pools[t] = [s for _, s in scores[:top]]
    return pools


def allowed_table(pools: dict[pd.Timestamp, list[str]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """把每月股票池展开成每天的"能不能买"表，给 short_term.backtest 的 allowed 参数用。"""
    symbols = sorted({s for v in pools.values() for s in v})
    table = pd.DataFrame(False, index=dates, columns=symbols)
    starts = sorted(pools)
    for i, t in enumerate(starts):
        until = starts[i + 1] if i + 1 < len(starts) else dates[-1] + pd.Timedelta(days=1)
        rows = (dates >= t) & (dates < until)
        table.loc[rows, pools[t]] = True
    return table
