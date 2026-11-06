"""技术指标：均线、RSI、MACD。

用纯 pandas 实现，方便读懂每个指标是怎么算出来的。
每个指标在第 t 天只用到第 t 天及以前的收盘价，没有未来数据。
"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""
    return close.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标（Wilder 平滑），取值 0~100。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # 只涨不跌时 avg_loss 为 0，RSI 记为 100
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD：DIF = 快 EMA - 慢 EMA，DEA = DIF 的 EMA，柱 = 2 × (DIF - DEA)（国内常用口径）。"""
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": 2 * (dif - dea)})


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在日线表上加常用指标列，返回新表。"""
    out = df.copy()
    c = out["close"]
    for w in (5, 20, 60):
        out[f"ma{w}"] = sma(c, w)
    out["rsi14"] = rsi(c, 14)
    out = out.join(macd(c))
    return out


def summarize(df: pd.DataFrame) -> dict:
    """把最新一天的指标整理成给 LLM 看的小字典（数字保留两位）。"""
    ind = add_indicators(df)
    last = ind.iloc[-1]
    close = float(last["close"])

    def r(x):
        return None if pd.isna(x) else round(float(x), 2)

    def pct_change(days: int):
        if len(ind) <= days:
            return None
        return round((close / float(ind["close"].iloc[-1 - days]) - 1) * 100, 2)

    return {
        "date": str(ind.index[-1].date()),
        "close": r(close),
        "ma5": r(last["ma5"]),
        "ma20": r(last["ma20"]),
        "ma60": r(last["ma60"]),
        "rsi14": r(last["rsi14"]),
        "macd_dif": r(last["dif"]),
        "macd_dea": r(last["dea"]),
        "macd_hist": r(last["hist"]),
        "change_20d_pct": pct_change(20),
        "change_60d_pct": pct_change(60),
        "change_250d_pct": pct_change(250),
        "high_250d": r(ind["high"].tail(250).max()),
        "low_250d": r(ind["low"].tail(250).min()),
    }
