"""大盘数据：上证指数行情 + 沪深两市成交额，用来判断"大盘环境"。

两市成交额 = 上证指数(sh000001) 成交额 + 深证综指(sz399106) 成交额。
用深证综指而不是深证成指，是因为成指只含 500 只成分股，综指覆盖深市全部股票。

缓存方式和个股一样：每个指数一个 Parquet 文件，已经下载过的区间直接读本地。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from trader import config
from trader.data.market import _covered_range, _write_cache

SH_INDEX = "sh000001"  # 上证指数
SZ_COMPOSITE = "sz399106"  # 深证综指

# (指数代码, 开始 YYYYMMDD, 结束 YYYYMMDD) -> 原始表，列含 date/open/close/high/low/volume/amount
IndexFetcher = Callable[[str, str, str], pd.DataFrame]


def _akshare_index(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    return ak.stock_zh_index_daily_em(symbol=symbol, start_date=start, end_date=end)


def _normalize(raw: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["open", "close", "high", "low", "volume", "amount"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[cols].astype(float)


def get_index_daily(
    symbol: str,
    start: str | date,
    end: str | date | None = None,
    cache_dir: Path | None = None,
    fetcher: IndexFetcher | None = None,
) -> pd.DataFrame:
    """取一个指数的日线（带本地缓存）。"""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end or date.today()).normalize()
    path = (cache_dir or config.CACHE_DIR) / "index" / f"{symbol}.parquet"
    lo, hi = _covered_range(path)
    if lo is not None and lo <= start_ts and hi >= end_ts:
        return pd.read_parquet(path).loc[start_ts:end_ts].copy()

    fetch = fetcher or _akshare_index
    new_lo = start_ts if lo is None else min(lo, start_ts)
    new_hi = end_ts if hi is None else max(hi, end_ts)
    df = _normalize(fetch(symbol, new_lo.strftime("%Y%m%d"), new_hi.strftime("%Y%m%d")))
    _write_cache(path, df, new_lo, new_hi)
    return df.loc[start_ts:end_ts].copy()


def get_market(
    start: str | date,
    end: str | date | None = None,
    cache_dir: Path | None = None,
    fetcher: IndexFetcher | None = None,
) -> pd.DataFrame:
    """返回大盘表：index_close（上证指数收盘）、amount（两市成交额，单位：元）。"""
    sh = get_index_daily(SH_INDEX, start, end, cache_dir, fetcher)
    sz = get_index_daily(SZ_COMPOSITE, start, end, cache_dir, fetcher)
    out = pd.DataFrame({"index_close": sh["close"], "amount": sh["amount"] + sz["amount"]})
    return out.dropna()
