import pandas as pd
import pytest

from trader.data.market import get_daily, normalize


def fake_raw(start: str, end: str) -> pd.DataFrame:
    """造一份和 AkShare 同样中文列名的假数据（只含工作日）。"""
    days = pd.bdate_range(start, end)
    n = len(days)
    return pd.DataFrame(
        {
            "日期": days.strftime("%Y-%m-%d"),
            "开盘": range(n),
            "最高": range(n),
            "最低": range(n),
            "收盘": range(n),
            "成交量": range(n),
            "成交额": range(n),
            "涨跌幅": [0.0] * n,
            "换手率": [0.0] * n,
        }
    )


class FakeFetcher:
    def __init__(self):
        self.calls = []

    def __call__(self, symbol, start, end, adjust):
        self.calls.append((start, end))
        return fake_raw(start, end)


def test_normalize_renames_and_indexes():
    df = normalize(fake_raw("2024-01-01", "2024-01-05"))
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing


def test_second_call_uses_cache(tmp_path):
    f = FakeFetcher()
    a = get_daily("600519", "2024-01-01", "2024-03-31", cache_dir=tmp_path, fetcher=f)
    b = get_daily("600519", "2024-01-01", "2024-03-31", cache_dir=tmp_path, fetcher=f)
    assert len(f.calls) == 1
    pd.testing.assert_frame_equal(a, b)


def test_subrange_served_from_cache(tmp_path):
    f = FakeFetcher()
    get_daily("600519", "2024-01-01", "2024-03-31", cache_dir=tmp_path, fetcher=f)
    df = get_daily("600519", "2024-02-01", "2024-02-29", cache_dir=tmp_path, fetcher=f)
    assert len(f.calls) == 1
    assert df.index.min() >= pd.Timestamp("2024-02-01")
    assert df.index.max() <= pd.Timestamp("2024-02-29")


def test_only_missing_range_is_fetched(tmp_path):
    f = FakeFetcher()
    get_daily("600519", "2024-02-01", "2024-02-29", cache_dir=tmp_path, fetcher=f)
    df = get_daily("600519", "2024-01-01", "2024-03-31", cache_dir=tmp_path, fetcher=f)
    assert f.calls[1:] == [("20240101", "20240131"), ("20240301", "20240331")]
    assert not df.index.duplicated().any()
    assert df.index.min() == pd.Timestamp("2024-01-01")
    assert df.index.max() == pd.Timestamp("2024-03-29")


def test_empty_result_is_cached_too(tmp_path):
    calls = []

    def empty(symbol, start, end, adjust):
        calls.append(start)
        return pd.DataFrame()

    get_daily("600519", "2024-02-10", "2024-02-17", cache_dir=tmp_path, fetcher=empty)
    df = get_daily("600519", "2024-02-10", "2024-02-17", cache_dir=tmp_path, fetcher=empty)
    assert df.empty
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["60051", "abcdef", "6005190"])
def test_rejects_bad_symbol(bad, tmp_path):
    with pytest.raises(ValueError):
        get_daily(bad, "2024-01-01", "2024-01-31", cache_dir=tmp_path, fetcher=FakeFetcher())


def test_rejects_reversed_dates(tmp_path):
    with pytest.raises(ValueError):
        get_daily("600519", "2024-02-01", "2024-01-01", cache_dir=tmp_path, fetcher=FakeFetcher())
