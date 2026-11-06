"""均线交叉策略回测（只做多）。

规则：快均线上穿慢均线时满仓买入，下穿时清仓。
防未来函数：用第 t 天收盘后才知道的信号，第 t+1 天才持仓（signal.shift(1)）。
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

    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    signal = (sma(close, fast) > sma(close, slow)).astype(float)  # 当天收盘后的判断
    position = signal.shift(1).fillna(0.0)  # 第二天才持有
    turnover = position.diff().abs().fillna(position.iloc[0])
    strat_ret = position * ret - turnover * cost_bps / 10_000

    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + ret).cumprod()

    n = len(df)
    total = float(equity.iloc[-1] - 1)
    annual = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1)
    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    # 逐笔统计：每段连续持仓算一笔交易
    trade_id = (position.diff() == 1).cumsum().where(position == 1)
    trade_rets = (1 + strat_ret).groupby(trade_id).prod() - 1
    trades = int(len(trade_rets))
    win_rate = round(float((trade_rets > 0).mean() * 100), 1) if trades else None

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
