"""均线交叉策略回测（只做多）。

规则：快均线上穿慢均线时满仓买入，下穿时清仓。
防未来函数：第 t 天收盘后出信号，第 t+1 个交易日开盘成交。
成本：每次换手按 cost_bps 收取（默认万分之十五，近似佣金 + 印花税 + 滑点）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trader.tools.indicators import sma

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    fast: int
    slow: int
    start: str
    end: str
    days: int
    total_return_pct: float
    annual_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trades: int
    win_rate_pct: float | None
    exposure_pct: float
    buy_hold_return_pct: float
    buy_hold_max_drawdown_pct: float
    execution: str = "收盘产生信号，下一交易日开盘成交；期末持仓按收盘价估值，不强制平仓"
    cost_bps: float = 15.0
    trade_statistics: str = "交易笔数含期末未平仓持仓；已平仓收益扣除买卖双边费用，未平仓只扣已发生的买入费用"

    def to_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def sma_cross(
    df: pd.DataFrame, fast: int = 20, slow: int = 60, cost_bps: float = 15.0
) -> tuple[BacktestResult, pd.DataFrame]:
    """跑均线交叉回测，返回 (指标汇总, 每日明细表)。"""
    if fast >= slow:
        raise ValueError("快均线周期必须小于慢均线周期")
    if len(df) < slow + 2:
        raise ValueError(f"数据太短：只有 {len(df)} 天，至少需要 {slow + 2} 天")

    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("交易成本必须在 0（含）至 10000（不含）基点之间")
    if not {"open", "close"}.issubset(df.columns):
        raise ValueError("次日开盘成交需要 open 和 close 列，不能用收盘价代替开盘价")
    prices = df[["open", "close"]].astype(float)
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("开盘价和收盘价必须为有限正数")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("行情日期必须升序且不重复")

    close = prices["close"]
    ret = close.pct_change().fillna(0.0)
    signal = (sma(close, fast) > sma(close, slow)).astype(float)
    position = signal.shift(1).fillna(0.0)
    fee = cost_bps / 10_000
    cash, shares = 1.0, 0.0
    entry_equity = None
    trade_returns = []
    values = []
    for target, opening, closing in zip(position, prices["open"], close):
        if target == 1 and shares == 0:
            entry_equity = cash
            # 买入金额 + 买入费用 = 现有现金，不借款。
            shares = cash / (opening * (1 + fee))
            cash = 0.0
        elif target == 0 and shares > 0:
            cash = shares * opening * (1 - fee)
            shares = 0.0
            trade_returns.append(cash / entry_equity - 1)
            entry_equity = None
        # 持仓时隔夜损益自然保留；买入前的跳空不归策略所有。
        values.append(cash + shares * closing)

    equity = pd.Series(values, index=df.index)
    strat_ret = equity.pct_change().fillna(equity.iloc[0] - 1)
    if shares > 0:
        # 保留原有统计口径：期末未平仓的一段持仓也算一笔。
        trade_returns.append(equity.iloc[-1] / entry_equity - 1)
    bh_equity = (1 + ret).cumprod()

    n = len(df)
    total = float(equity.iloc[-1] - 1)
    annual = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1)
    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    trades = len(trade_returns)
    win_rate = round(float((np.array(trade_returns) > 0).mean() * 100), 1) if trades else None

    result = BacktestResult(
        fast=fast,
        slow=slow,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        days=n,
        total_return_pct=round(total * 100, 2),
        annual_return_pct=round(annual * 100, 2),
        sharpe=round(sharpe, 2),
        max_drawdown_pct=round(_max_drawdown(equity) * 100, 2),
        trades=trades,
        win_rate_pct=win_rate,
        exposure_pct=round(float(position.mean() * 100), 1),
        buy_hold_return_pct=round(float(bh_equity.iloc[-1] - 1) * 100, 2),
        buy_hold_max_drawdown_pct=round(_max_drawdown(bh_equity) * 100, 2),
        cost_bps=cost_bps,
    )
    daily = pd.DataFrame(
        {"close": close, "position": position, "strategy": equity, "buy_hold": bh_equity}
    )
    return result, daily


def plot_equity(daily: pd.DataFrame, title: str, path) -> None:
    """画策略净值 vs 买入持有净值，存成 PNG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(daily.index, daily["strategy"], label="Strategy", linewidth=1.6)
    ax.plot(daily.index, daily["buy_hold"], label="Buy & Hold", linewidth=1.2, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Net value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
