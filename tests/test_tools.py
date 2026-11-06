import numpy as np
import pandas as pd
import pytest

from trader.tools import backtest, indicators


def test_sma_and_rsi_basic():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert indicators.sma(s, 2).tolist()[1:] == [1.5, 2.5, 3.5, 4.5]
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert indicators.rsi(up).iloc[-1] == 100.0
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert indicators.rsi(down).iloc[-1] == pytest.approx(0.0)


def test_indicators_have_no_lookahead(price_df):
    """把最后一天的价格改掉，前面各天的指标不应变化。"""
    a = indicators.add_indicators(price_df)
    changed = price_df.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 2
    b = indicators.add_indicators(changed)
    pd.testing.assert_frame_equal(a.iloc[:-1], b.iloc[:-1])


def test_summarize_keys(price_df):
    s = indicators.summarize(price_df)
    for k in ("close", "ma20", "rsi14", "macd_hist", "change_250d_pct"):
        assert s[k] is not None


def test_backtest_trades_next_day(price_df):
    res, daily = backtest.sma_cross(price_df, fast=10, slow=30)
    sig = (indicators.sma(price_df["close"], 10) > indicators.sma(price_df["close"], 30)).astype(float)
    # 持仓 = 前一天的信号
    assert (daily["position"].iloc[1:].values == sig.shift(1).iloc[1:].values).all()
    assert res.trades >= 1
    assert res.max_drawdown_pct <= 0


def test_backtest_no_lookahead(price_df):
    """改最后一天价格，不应改变之前任何一天的持仓。"""
    _, a = backtest.sma_cross(price_df, 10, 30)
    changed = price_df.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 0.5
    _, b = backtest.sma_cross(changed, 10, 30)
    pd.testing.assert_series_equal(a["position"], b["position"])


def test_costs_reduce_return(price_df):
    free, _ = backtest.sma_cross(price_df, 10, 30, cost_bps=0)
    paid, _ = backtest.sma_cross(price_df, 10, 30, cost_bps=50)
    assert paid.total_return_pct < free.total_return_pct


def test_always_in_market_matches_buy_hold():
    """价格一直涨时策略几乎全程持仓，扣掉一次买入成本后应接近买入持有。"""
    idx = pd.bdate_range("2023-01-02", periods=120, name="date")
    close = pd.Series(np.linspace(10, 20, 120), index=idx)
    df = pd.DataFrame({"close": close})
    res, daily = backtest.sma_cross(df, 2, 5, cost_bps=0)
    held = daily["position"] == 1
    expected = (close.pct_change()[held] + 1).prod() - 1
    assert res.total_return_pct == pytest.approx(expected * 100, abs=0.01)
    assert res.trades == 1


def test_backtest_rejects_bad_params(price_df):
    with pytest.raises(ValueError):
        backtest.sma_cross(price_df, 60, 20)
    with pytest.raises(ValueError):
        backtest.sma_cross(price_df.head(10), 5, 20)


def test_plot_equity(price_df, tmp_path):
    _, daily = backtest.sma_cross(price_df, 10, 30)
    out = tmp_path / "eq.png"
    backtest.plot_equity(daily, "test", out)
    assert out.stat().st_size > 1000
