import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_df():
    """造 300 个交易日的行情：先涨后跌再涨，便于产生均线交叉。"""
    idx = pd.bdate_range("2023-01-02", periods=300, name="date")
    t = np.arange(300)
    close = 100 + 20 * np.sin(t / 30) + t * 0.05
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": 1000.0, "amount": 1e6, "pct_chg": 0.0, "turnover": 0.5,
        },
        index=idx,
    )
