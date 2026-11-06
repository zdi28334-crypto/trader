"""用可手算的行情检验成交时点与费用，完全离线。"""
import numpy as np
import pandas as pd
import pytest

from trader.tools.backtest import sma_cross


def prices(opening=None):
    # 第三天收盘出买入信号，第四天买入；第四天收盘出卖出信号，第五天卖出。
    return pd.DataFrame(
        {"open": opening or [10, 10, 11, 20, 4, 4],
         "close": [10, 10, 12, 6, 4, 4]},
        index=pd.bdate_range("2026-01-05", periods=6),
    )


def test_entry_exit_gaps_and_cash_days():
    result, daily = sma_cross(prices(), 1, 2, cost_bps=0)
    # 20 元买入，6 元收盘，次日 4 元卖出。不能赚到买入前的跳空。
    assert daily["strategy"].tolist() == pytest.approx([1, 1, 1, .3, .2, .2])
    assert daily["position"].tolist() == [0, 0, 0, 1, 0, 0]
    assert result.total_return_pct == -80
    assert result.trades == 1


def test_round_trip_includes_both_fees():
    df = prices([10, 10, 11, 10, 10.02, 4])
    # 毛利 0.2%，但买卖各 0.15% 后亏损，胜率必须为零。
    result, daily = sma_cross(df, 1, 2)
    expected = 10.02 * (1 - .0015) / (10 * (1 + .0015))
    assert daily["strategy"].iloc[-1] == pytest.approx(expected)
    assert result.win_rate_pct == 0
    assert result.cost_bps == 15


def test_hold_overnight_and_open_position_mark_to_close():
    df = pd.DataFrame({"open": [10, 10, 11, 20, 5, 9],
                       "close": [10, 10, 12, 13, 14, 15]},
                      index=pd.bdate_range("2026-01-05", periods=6))
    result, daily = sma_cross(df, 1, 2, cost_bps=0)
    assert daily["strategy"].tolist() == pytest.approx([1, 1, 1, 13/20, 14/20, 15/20])
    assert result.trades == 1  # 期末未平仓沿用原有计数口径
    assert result.total_return_pct == -25


def test_future_prices_do_not_change_past_equity():
    df = prices()
    _, before = sma_cross(df, 1, 2)
    df.loc[df.index[-1], ["open", "close"]] = [300, 400]
    _, after = sma_cross(df, 1, 2)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_last_close_signal_does_not_fill_without_next_open():
    df = prices()
    df["close"] = [10, 10, 10, 10, 10, 20]
    result, daily = sma_cross(df, 1, 2)
    assert result.trades == 0
    assert result.win_rate_pct is None
    assert (daily["strategy"] == 1).all()


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_open_is_not_silently_replaced(bad):
    df = prices().astype(float)
    df.loc[df.index[3], "open"] = bad
    with pytest.raises(ValueError, match="有限正数"):
        sma_cross(df, 1, 2)


def test_missing_open_is_rejected():
    with pytest.raises(ValueError, match="open"):
        sma_cross(prices().drop(columns="open"), 1, 2)


def test_reentry_uses_remaining_cash():
    df = pd.DataFrame(
        {"open": [10, 10, 11, 20, 4, 8, 10, 6, 2],
         "close": [10, 10, 12, 6, 4, 8, 9, 3, 2]},
        index=pd.bdate_range("2026-01-05", periods=9),
    )
    result, daily = sma_cross(df, 1, 2, cost_bps=0)
    # 第一笔 20 买 4 卖，剩 0.2；第二笔 10 买 2 卖，剩 0.04。
    assert daily["strategy"].iloc[-1] == pytest.approx(.04)
    assert result.trades == 2
    assert result.win_rate_pct == 0
