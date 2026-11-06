"""Di少的短线策略（下跌趋势版）。

规则全部来自策略作者本人，说明书见：
https://claude.ai/artifact/8fQN3XNCLNrFYpXRG5uBvk

这个文件把说明书拆成四块，每块对应一个函数，想改规则就改这里：
1. market_regime   大盘环境：量能低迷、量能极度萎缩、连续下跌
2. entry_signals   选股买点：均线多头 + 5 日上穿 10 日 + 回调不破 20 日线（可选 W 底）
3. _exit_reason    卖出规则：硬止损 10%；情绪小票破 10 日线卖；其他票破 20 日线次日收不回就卖
4. backtest        组合回测：量能低迷时总仓位不超过半仓，单只不超过三成，最多 8 只

数据能直接算的规则在代码里；连板、断板、集合竞价、假拉升这类盘面判断，
暂时写在 ANALYST_NOTES 里交给分析师 Agent 按你的描述判断，以后有数据了再换成代码。

交易约定（防未来函数 + A股 T+1）：第 t 天收盘后根据当天及以前的数据出信号，第 t+1 天开盘价成交。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from trader.tools.indicators import sma

TRADING_DAYS = 252
EMOTION = "情绪"  # 情绪类题材小票
OTHER = "其他"  # 防守板块、科技权重等


@dataclass
class ShortTermParams:
    """说明书里的所有数字都在这里，改数字不用动逻辑。"""

    # 一、大盘环境
    low_volume_amount: float = 1.5e12  # 两市 20 日平均成交额低于 1.5 万亿 = 量能低迷
    amount_window: int = 20
    shrink_ratio: float = 0.7  # 当天成交额比 20 日均值少三成 = 量能极度萎缩
    down_days: int = 3  # 指数连跌几天算"连续下跌"（作者未定，先按 3 天）

    # 四、选股
    pullback_lookback: int = 20  # 最近 20 天收盘都没跌破 20 日线 = 回调不破线
    require_w_bottom: bool = False  # W 底是简化识别，默认不作为硬条件
    w_lookback: int = 60

    # 五、卖出
    stop_loss: float = 0.10  # 亏 10% 直接卖

    # 六、仓位
    max_total_low_volume: float = 0.5  # 量能低迷时最多半仓
    max_total_normal: float = 1.0  # 量能不低迷时（作者未说明，先按可满仓）
    max_single: float = 0.3  # 单只不超过总资金三成
    max_holdings: int = 8

    cost_bps: float = 15.0  # 每次买或卖的成本（佣金 + 印花税 + 滑点的近似）

    def to_dict(self) -> dict:
        return asdict(self)


# 交给分析师 Agent 的盘面判断规则（暂时无法用数据直接计算的部分）
ANALYST_NOTES = """策略作者对下跌趋势中盘面的判断方法（分析时请按这个思路，并说明哪些条件你能用数据验证、哪些不能）：
- 大盘量能低迷（两市成交额约 1.5 万亿）时视为下跌趋势，总仓位不超过半仓。
- 假拉升：盘面拉升以小盘题材股为主；权重科技股盘中拉升速度慢于创业板、科创板，主动性不强，且拉升不带量。出现这种情况大概率还会回落，不追。
- 博科技权重反弹的时机：连续下跌 + 量能极度萎缩（成交额比 20 日均值少三成），同时短线情绪退潮：连板小票集合竞价不及预期、市场最高连板龙头断板、中军和后排表现很差。
- 下跌趋势中选股：均线多头排列、5 日线上穿 10 日线，每一波拉升后的回调不跌破 10 日线或 20 日线，最好走出 W 底；题材偏防守板块或情绪类小票。
- 卖出：情绪小票跌破 10 日线就走；其他票跌破 10 日线后看 20 日线能否承接，跌破 20 日线次日收不回就卖；任何票亏损 10% 直接卖。
"""


# ---------- 一、大盘环境 ----------
def market_regime(market: pd.DataFrame, p: ShortTermParams | None = None) -> pd.DataFrame:
    """输入大盘表（index_close, amount），输出每天的环境判断。"""
    p = p or ShortTermParams()
    amount = market["amount"].astype(float)
    amount_ma = amount.rolling(p.amount_window, min_periods=p.amount_window).mean()
    down = market["index_close"].diff() < 0
    # 连跌天数：遇到上涨就清零
    streak = down.astype(int).groupby((~down).cumsum()).cumsum()

    out = pd.DataFrame(index=market.index)
    out["amount"] = amount
    out["amount_ma"] = amount_ma
    out["low_volume"] = amount_ma < p.low_volume_amount
    out["extreme_shrink"] = amount < amount_ma * p.shrink_ratio
    out["down_streak"] = streak
    # 博科技反弹的"数据部分"条件；情绪退潮那部分需要分析师另行判断
    out["rebound_setup"] = (streak >= p.down_days) & out["extreme_shrink"]
    out["max_total"] = np.where(out["low_volume"], p.max_total_low_volume, p.max_total_normal)
    return out


# ---------- 四、选股买点 ----------
def w_bottom(close: pd.Series, lookback: int = 60) -> pd.Series:
    """简化版 W 底：近 lookback 天里有两个相距至少 5 天、高度相差不超过 3% 的低点，
    两低点之间的高点（颈线）比低点高至少 5%，且当天收盘站上颈线。"""

    def check(a: np.ndarray) -> float:
        i = int(np.argmin(a))
        far = [k for k in range(len(a)) if abs(k - i) >= 5]
        if not far:
            return 0.0
        j = far[int(np.argmin(a[far]))]
        lo, hi = min(i, j), max(i, j)
        neck = a[lo : hi + 1].max()
        ok = (
            abs(a[j] - a[i]) / a[i] <= 0.03
            and neck >= max(a[i], a[j]) * 1.05
            and hi < len(a) - 1
            and a[-1] > neck
        )
        return 1.0 if ok else 0.0

    return close.rolling(lookback, min_periods=lookback).apply(check, raw=True).fillna(0).astype(bool)


def entry_signals(df: pd.DataFrame, p: ShortTermParams | None = None) -> pd.DataFrame:
    """在个股日线上加均线和买入信号列。"""
    p = p or ShortTermParams()
    out = df.copy()
    c = out["close"].astype(float)
    out["ma5"], out["ma10"], out["ma20"] = sma(c, 5), sma(c, 10), sma(c, 20)
    bull = (out["ma5"] > out["ma10"]) & (out["ma10"] > out["ma20"])
    cross = (out["ma5"] > out["ma10"]) & (out["ma5"].shift(1) <= out["ma10"].shift(1))
    above20 = (c >= out["ma20"]).astype(float)
    held = above20.rolling(p.pullback_lookback, min_periods=p.pullback_lookback).min() == 1
    out["bull"] = bull
    out["cross_5_10"] = cross
    out["pullback_held"] = held
    out["buy"] = bull & cross & held
    if p.require_w_bottom:
        out["w_bottom"] = w_bottom(c, p.w_lookback)
        out["buy"] &= out["w_bottom"]
    return out


# ---------- 五、卖出规则 ----------
@dataclass
class _Holding:
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    kind: str
    below20: bool = False  # 已经收盘跌破 20 日线，等第二天看能否收回


def _exit_reason(h: _Holding, row: pd.Series, p: ShortTermParams) -> str | None:
    """看当天收盘后是否要卖，返回卖出原因；不卖返回 None。会更新 h.below20。"""
    close = float(row["close"])
    if close <= h.entry_price * (1 - p.stop_loss):
        return f"亏损 {p.stop_loss:.0%} 止损"
    if h.kind == EMOTION:
        return "情绪票跌破 10 日线" if close < row["ma10"] else None
    if close < row["ma20"]:
        if h.below20:
            return "跌破 20 日线，次日未收回"
        h.below20 = True
    else:
        h.below20 = False
    return None


# ---------- 六、组合回测 ----------
@dataclass
class StrategyResult:
    start: str
    end: str
    days: int
    total_return_pct: float
    annual_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    avg_exposure_pct: float
    trades: int
    win_rate_pct: float | None
    benchmark_return_pct: float
    benchmark_max_drawdown_pct: float
    open_positions: list[str]
    trade_list: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def to_dict(self, max_trades: int = 30) -> dict:
        d = asdict(self)
        d["trade_list"] = d["trade_list"][-max_trades:]  # 给大模型看的只保留最近几笔
        return d


def _max_dd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def backtest(
    stocks: dict[str, pd.DataFrame],
    kinds: dict[str, str] | None = None,
    market: pd.DataFrame | None = None,
    p: ShortTermParams | None = None,
    allowed: pd.DataFrame | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """按说明书跑组合回测。

    stocks: {代码: 日线表}；kinds: {代码: "情绪" 或 "其他"}，未写的按"其他"；
    market: 大盘表（index_close, amount），不给时不限制总仓位、基准用个股等权持有。
    allowed: 可选的"当天能不能买"表（行 = 日期，列 = 代码，值 True/False），用来做每月换股票池；
             表里没有的代码不受限制。已经持有的票不受影响，照常按卖出规则卖。
    返回 (结果汇总, 每日净值表)。
    """
    p = p or ShortTermParams()
    kinds = kinds or {}
    if not stocks:
        raise ValueError("至少要给一只股票")
    sig = {s: entry_signals(df, p) for s, df in stocks.items()}
    dates = sorted(set().union(*(df.index for df in sig.values())))
    regime = market_regime(market, p) if market is not None else None
    cost = p.cost_bps / 10_000

    cash = 1.0
    holdings: dict[str, _Holding] = {}
    last_price: dict[str, float] = {}
    pending_buy: list[str] = []
    pending_sell: dict[str, str] = {}
    max_total = p.max_total_normal
    trades: list[dict] = []
    rows = []

    def value() -> float:
        return cash + sum(h.shares * last_price[s] for s, h in holdings.items())

    for t in dates:
        today = {s: d.loc[t] for s, d in sig.items() if t in d.index}

        # 1) 开盘：先卖后买
        for s, reason in list(pending_sell.items()):
            if s not in today:  # 停牌，顺延
                continue
            h = holdings.pop(s)
            px = float(today[s]["open"])
            cash += h.shares * px * (1 - cost)
            trades.append(
                {
                    "symbol": s, "kind": h.kind,
                    "buy_date": str(h.entry_date.date()), "buy_price": round(h.entry_price, 2),
                    "sell_date": str(t.date()), "sell_price": round(px, 2),
                    "return_pct": round((px * (1 - cost) / h.entry_price - 1) * 100, 2),  # 已含买卖成本
                    "reason": reason,
                }
            )
            del pending_sell[s]
        for s in today:  # 开盘价先更新，用于计算当前仓位
            if s in holdings:
                last_price[s] = float(today[s]["open"])
        for s in pending_buy:
            if s not in today or s in holdings or len(holdings) >= p.max_holdings:
                continue
            equity = value()
            exposure = (equity - cash) / equity
            w = min(p.max_single, max_total - exposure)
            amount = min(w * equity, cash)
            if amount < 0.01 * equity:
                break  # 仓位已满
            px = float(today[s]["open"])
            holdings[s] = _Holding(amount * (1 - cost) / px, px * (1 + cost), t, kinds.get(s, OTHER))
            last_price[s] = px
            cash -= amount
        pending_buy = []

        # 2) 收盘：记净值
        for s in today:
            last_price[s] = float(today[s]["close"])
        equity = value()
        rows.append({"date": t, "equity": equity, "exposure": (equity - cash) / equity, "holdings": len(holdings)})

        # 3) 收盘后：出明天的信号
        for s, h in holdings.items():
            if s in today and s not in pending_sell:
                reason = _exit_reason(h, today[s], p)
                if reason:
                    pending_sell[s] = reason
        if regime is not None and t in regime.index:
            max_total = float(regime.at[t, "max_total"])
        pending_buy = [
            s for s, r in today.items() if s not in holdings and bool(r["buy"]) and _is_allowed(allowed, t, s)
        ]

    daily = pd.DataFrame(rows).set_index("date")
    equity = daily["equity"]
    ret = equity.pct_change().fillna(0.0)

    if market is not None:
        bench = market["index_close"].reindex(daily.index).ffill().bfill()
    else:  # 个股等权买入持有
        closes = pd.DataFrame({s: d["close"] for s, d in stocks.items()}).reindex(daily.index).ffill().bfill()
        bench = (closes / closes.iloc[0]).mean(axis=1)
    bench = bench / bench.iloc[0]
    daily["benchmark"] = bench

    n = len(daily)
    std = ret.std()
    closed = [tr["return_pct"] for tr in trades]
    result = StrategyResult(
        start=str(daily.index[0].date()),
        end=str(daily.index[-1].date()),
        days=n,
        total_return_pct=round((equity.iloc[-1] - 1) * 100, 2),
        annual_return_pct=round((equity.iloc[-1] ** (TRADING_DAYS / n) - 1) * 100, 2),
        sharpe=round(float(ret.mean() / std * np.sqrt(TRADING_DAYS)), 2) if std > 0 else 0.0,
        max_drawdown_pct=round(_max_dd(equity) * 100, 2),
        avg_exposure_pct=round(float(daily["exposure"].mean() * 100), 1),
        trades=len(trades),
        win_rate_pct=round(sum(r > 0 for r in closed) / len(closed) * 100, 1) if closed else None,
        benchmark_return_pct=round((bench.iloc[-1] - 1) * 100, 2),
        benchmark_max_drawdown_pct=round(_max_dd(bench) * 100, 2),
        open_positions=sorted(holdings),
        trade_list=trades,
        params=p.to_dict(),
    )
    return result, daily


def _is_allowed(allowed: pd.DataFrame | None, t: pd.Timestamp, s: str) -> bool:
    if allowed is None or s not in allowed.columns:
        return True
    return t in allowed.index and bool(allowed.at[t, s])


def latest_snapshot(market: pd.DataFrame, p: ShortTermParams | None = None) -> dict:
    """最新一天的大盘环境，给分析师看。"""
    r = market_regime(market, p).iloc[-1]
    return {
        "date": str(r.name.date()),
        "two_market_amount_yi": round(float(r["amount"]) / 1e8, 0),
        "amount_ma20_yi": None if pd.isna(r["amount_ma"]) else round(float(r["amount_ma"]) / 1e8, 0),
        "low_volume": bool(r["low_volume"]),
        "extreme_shrink": bool(r["extreme_shrink"]),
        "down_streak_days": int(r["down_streak"]),
        "rebound_setup_data_part": bool(r["rebound_setup"]),
        "max_total_position": float(r["max_total"]),
    }


def plot(daily: pd.DataFrame, title: str, path) -> None:
    """画策略净值、基准和仓位。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), dpi=120, sharex=True, height_ratios=[3, 1])
    ax1.plot(daily.index, daily["equity"], label="My strategy", linewidth=1.6)
    ax1.plot(daily.index, daily["benchmark"], label="Benchmark", linewidth=1.2, alpha=0.7)
    ax1.set_title(title)
    ax1.set_ylabel("Net value")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.fill_between(daily.index, daily["exposure"], step="post", alpha=0.5)
    ax2.set_ylabel("Exposure")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
