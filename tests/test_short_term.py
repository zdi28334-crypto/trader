"""我的短线策略：每条规则一个测试，用造出来的行情验证，不联网。"""

import numpy as np
import pandas as pd
import pytest

from trader.data.index import get_market
from trader.strategies import short_term as st


def make_df(close, start="2023-01-02"):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close), name="date")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6, "amount": 1e8},
        index=idx,
    )


def make_market(amounts, closes=None, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=len(amounts), name="date")
    closes = closes if closes is not None else np.full(len(amounts), 3000.0)
    return pd.DataFrame({"index_close": closes, "amount": np.asarray(amounts, dtype=float)}, index=idx)


# ---------- 大盘环境 ----------
def test_regime_low_volume_and_shrink():
    amounts = [1.2e12] * 25 + [0.7e12]  # 最后一天比 20 日均值少三成以上
    closes = [3000] * 22 + [2990, 2980, 2970, 2960]  # 最后 4 天连跌
    r = st.market_regime(make_market(amounts, closes))
    last = r.iloc[-1]
    assert last["low_volume"] and last["extreme_shrink"]
    assert last["down_streak"] == 4
    assert last["rebound_setup"]
    assert last["max_total"] == 0.5
    assert not r.iloc[-2]["extreme_shrink"]


def test_regime_normal_volume_allows_full_position():
    r = st.market_regime(make_market([2e12] * 30))
    assert not r.iloc[-1]["low_volume"]
    assert r.iloc[-1]["max_total"] == 1.0


# ---------- 买点 ----------
def rising_then_cross():
    """先稳步上涨（20 日线之上），中间小回调让 5 日线下穿 10 日线，再拉起上穿。"""
    up = list(np.linspace(10, 14, 40))
    dip = [13.8, 13.6, 13.5, 13.45, 13.5]
    again = list(np.linspace(13.7, 15, 10))
    return up + dip + again


def test_entry_signal_requires_cross_bull_and_hold():
    sig = st.entry_signals(make_df(rising_then_cross()))
    buys = sig.index[sig["buy"]]
    assert len(buys) >= 1
    row = sig.loc[buys[0]]
    assert row["ma5"] > row["ma10"] > row["ma20"]
    assert row["cross_5_10"] and row["pullback_held"]


def test_no_entry_in_downtrend():
    sig = st.entry_signals(make_df(np.linspace(20, 10, 80)))
    assert not sig["buy"].any()


def test_entry_has_no_lookahead():
    df = make_df(rising_then_cross())
    a = st.entry_signals(df)
    changed = df.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 0.5
    b = st.entry_signals(changed)
    pd.testing.assert_series_equal(a["buy"].iloc[:-1], b["buy"].iloc[:-1])


def test_w_bottom_detects_simple_w():
    w = [20, 18, 16, 14, 12, 10, 11, 12, 13, 12, 11, 10.1, 11, 12, 13.5, 14]
    close = pd.Series([20.0] * 10 + w, dtype=float)
    assert st.w_bottom(close, lookback=len(w)).iloc[-1]
    assert not st.w_bottom(pd.Series(np.linspace(20, 10, 30)), lookback=16).iloc[-1]


# ---------- 卖出 ----------
def row(close, ma10, ma20):
    return pd.Series({"close": close, "ma10": ma10, "ma20": ma20})


P = st.ShortTermParams()


def test_stop_loss_10pct():
    h = st._Holding(1, 10.0, pd.Timestamp("2023-01-02"), st.OTHER)
    assert "止损" in st._exit_reason(h, row(8.99, 8, 8), P)


def test_emotion_sells_below_ma10():
    h = st._Holding(1, 10.0, pd.Timestamp("2023-01-02"), st.EMOTION)
    assert st._exit_reason(h, row(10.5, 10.6, 9), P) == "情绪票跌破 10 日线"
    assert st._exit_reason(h, row(10.7, 10.6, 9), P) is None


def test_other_gets_one_day_to_reclaim_ma20():
    h = st._Holding(1, 10.0, pd.Timestamp("2023-01-02"), st.OTHER)
    assert st._exit_reason(h, row(10.5, 10.6, 10.4), P) is None  # 破 10 日线不卖
    assert st._exit_reason(h, row(10.3, 10.6, 10.4), P) is None  # 第一天破 20 日线，先看
    assert st._exit_reason(h, row(10.5, 10.6, 10.4), P) is None  # 次日收回，继续拿
    assert st._exit_reason(h, row(10.3, 10.6, 10.4), P) is None
    assert "次日未收回" in st._exit_reason(h, row(10.2, 10.6, 10.4), P)


# ---------- 组合回测 ----------
def test_backtest_respects_position_caps():
    df = make_df(rising_then_cross() + list(np.linspace(15, 16, 20)))
    stocks = {f"60000{i}": df for i in range(4)}
    low = make_market([1.0e12] * len(df))
    res, daily = st.backtest(stocks, market=low)
    # 量能低迷最多半仓（上限在买入时检查，之后股价上涨仓位会自然变大，不强制减仓）
    first = daily.index[daily["holdings"] > 0][0]
    assert daily.at[first, "exposure"] <= 0.5 + 0.01
    assert daily["exposure"].max() < 0.6
    normal = make_market([2.0e12] * len(df))
    res2, daily2 = st.backtest(stocks, market=normal)
    assert daily2["exposure"].max() > 0.5  # 不低迷时可以超过半仓
    # 单只不超过三成：一只票时仓位不超过 30%
    _, one = st.backtest({"600000": df}, market=normal)
    first = one.index[one["holdings"] > 0][0]
    assert one.at[first, "exposure"] <= 0.3 + 0.01


def test_backtest_max_holdings():
    df = make_df(rising_then_cross() + list(np.linspace(15, 16, 20)))
    stocks = {f"6000{i:02d}": df for i in range(12)}
    p = st.ShortTermParams(max_single=0.05)
    _, daily = st.backtest(stocks, market=make_market([2e12] * len(df)), p=p)
    assert daily["holdings"].max() == 8


def test_backtest_buys_next_open_and_stops_out():
    close = rising_then_cross() + [14.0, 12.0, 11.0, 11.0]  # 买入后暴跌
    df = make_df(close)
    res, daily = st.backtest({"600000": df}, kinds={"600000": st.EMOTION})
    assert res.trades == 1
    tr = res.trade_list[0]
    sig = st.entry_signals(df)
    first_buy = sig.index[sig["buy"]][0]
    assert pd.Timestamp(tr["buy_date"]) == df.index[df.index.get_loc(first_buy) + 1]
    assert tr["return_pct"] < 0


def test_get_market_sums_two_exchanges(tmp_path):
    def fake(symbol, start, end):
        dates = pd.bdate_range("2023-01-02", periods=3).strftime("%Y-%m-%d")
        amt = 6e11 if symbol == "sh000001" else 8e11
        return pd.DataFrame(
            {"date": dates, "open": 1.0, "close": [3000, 3010, 3020], "high": 1.0, "low": 1.0, "volume": 1.0, "amount": amt}
        )

    m = get_market("2023-01-02", "2023-01-04", cache_dir=tmp_path, fetcher=fake)
    assert m["amount"].tolist() == pytest.approx([1.4e12] * 3)
    assert m["index_close"].iloc[-1] == 3020
    # 第二次走缓存，不再调用取数函数
    m2 = get_market("2023-01-02", "2023-01-04", cache_dir=tmp_path, fetcher=None)
    pd.testing.assert_frame_equal(m, m2)


# ---------- 每月股票池 ----------
from trader.strategies import pool as pl  # noqa: E402


def _stock_with_limit_ups(days_up, n=150, turnover=5.0, amount=1e8):
    df = make_df(np.linspace(10, 12, n))
    df["pct_chg"] = 1.0
    df["turnover"] = turnover
    df["amount"] = amount
    for d in days_up:
        df.iloc[d, df.columns.get_loc("pct_chg")] = 10.0
    return df


def test_monthly_pool_uses_only_past_data():
    hot_late = _stock_with_limit_ups([120, 125, 130])  # 涨停发生在后面
    hot_early = _stock_with_limit_ups([20, 25, 30])
    big = _stock_with_limit_ups([20, 25, 30], amount=1e10)  # 流通市值 2000 亿，排除
    stocks = {"600001": hot_late, "600002": hot_early, "600003": big}
    pools = pl.monthly_emotion_pool(stocks, start="2023-04-01")
    first = min(pools)
    assert "600002" in pools[first]
    assert "600001" not in pools[first]  # 当时还没发生的涨停不能用
    assert all("600003" not in v for v in pools.values())


def test_chinext_threshold():
    assert pl.limit_up_threshold("300750") == 19.8
    assert pl.limit_up_threshold("600519") == 9.8


def test_backtest_respects_allowed():
    df = make_df(rising_then_cross() + list(np.linspace(15, 16, 20)))
    blocked = pd.DataFrame(False, index=df.index, columns=["600000"])
    res, _ = st.backtest({"600000": df}, allowed=blocked)
    assert res.trades == 0 and not res.open_positions
    res2, _ = st.backtest({"600000": df}, allowed=~blocked)
    assert res2.trades + len(res2.open_positions) >= 1
