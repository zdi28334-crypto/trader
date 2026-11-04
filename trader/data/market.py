"""A股日线行情：从 AkShare 取数，并缓存到本地 Parquet 文件。

为什么要缓存：
- 免费接口会限流，反复请求容易被封；
- 回测必须可复现，同一份数据跑出来的结果才能比较。

缓存规则：每只股票 + 复权方式存一个文件。请求的日期区间如果已经被缓存覆盖，
就直接读本地；否则只去网上补缺的那一段，再合并写回。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from trader import config

# AkShare 返回的中文列名 -> 项目内统一使用的英文列名
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
}
COLUMNS = list(COLUMN_MAP.values())[1:]  # 除 date 以外的列，date 作为索引

# 取数函数的签名：(代码, 开始日 YYYYMMDD, 结束日 YYYYMMDD, 复权) -> 原始 DataFrame
Fetcher = Callable[[str, str, str, str], pd.DataFrame]


def _akshare_fetch(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    import akshare as ak  # 延迟导入，测试时不需要联网

    return ak.stock_zh_a_hist(
        symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust
    )


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """把 AkShare 的原始表整理成：日期索引 + 统一英文列名 + 升序。"""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
    df = raw.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[[c for c in COLUMNS if c in df.columns]].astype(float)


def _cache_path(cache_dir: Path, symbol: str, adjust: str) -> Path:
    return cache_dir / "daily" / f"{symbol}_{adjust or 'none'}.parquet"


def _to_ts(d: str | date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def get_daily(
    symbol: str,
    start: str | date,
    end: str | date | None = None,
    adjust: str = "qfq",
    cache_dir: Path | None = None,
    fetcher: Fetcher | None = None,
) -> pd.DataFrame:
    """获取一只A股的日线数据（默认前复权）。

    参数:
        symbol: 6 位股票代码，例如 "600519"（贵州茅台）。
        start / end: 日期，"2020-01-01" 或 date 对象；end 默认今天。
        adjust: "qfq" 前复权，"hfq" 后复权，"" 不复权。
        cache_dir: 缓存目录，默认 config.CACHE_DIR。
        fetcher: 取数函数，测试时可以替换成假的。
    """
    if not (len(symbol) == 6 and symbol.isdigit()):
        raise ValueError(f"股票代码应为 6 位数字，收到：{symbol!r}")
    start_ts = _to_ts(start)
    end_ts = _to_ts(end or date.today())
    if start_ts > end_ts:
        raise ValueError(f"开始日期 {start_ts.date()} 晚于结束日期 {end_ts.date()}")

    fetch = fetcher or _akshare_fetch
    path = _cache_path(cache_dir or config.CACHE_DIR, symbol, adjust)

    cached = pd.read_parquet(path) if path.exists() else None
    # 缓存里记录了"已经向接口问过"的区间，避免节假日等没有数据的日子反复请求
    have_lo, have_hi = _covered_range(path)

    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if have_lo is None:
        gaps.append((start_ts, end_ts))
    else:
        if start_ts < have_lo:
            gaps.append((start_ts, have_lo - pd.Timedelta(days=1)))
        if end_ts > have_hi:
            gaps.append((have_hi + pd.Timedelta(days=1), end_ts))

    if gaps:
        parts = [] if cached is None else [cached]
        for lo, hi in gaps:
            parts.append(normalize(fetch(symbol, lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d"), adjust)))
        non_empty = [p for p in parts if not p.empty]
        merged = pd.concat(non_empty) if non_empty else normalize(None)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        new_lo = min(start_ts, have_lo) if have_lo is not None else start_ts
        new_hi = max(end_ts, have_hi) if have_hi is not None else end_ts
        _write_cache(path, merged, new_lo, new_hi)
        cached = merged

    return cached.loc[start_ts:end_ts].copy()


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".range")


def _covered_range(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    meta = _meta_path(path)
    if not (path.exists() and meta.exists()):
        return None, None
    lo, hi = meta.read_text().split()
    return pd.Timestamp(lo), pd.Timestamp(hi)


def _write_cache(path: Path, df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    _meta_path(path).write_text(f"{lo.date()} {hi.date()}")
