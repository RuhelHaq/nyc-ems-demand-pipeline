import pandas as pd
import pytest

from src.data import clean_month, validate_month
from src.download import list_months, month_bounds

BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "RICHMOND / STATEN ISLAND"]
CONFIG = {"cleaning": {"drop_boroughs": ["UNKNOWN"]},
          "forecast": {"boroughs": BOROUGHS, "shift_hours": 8}}
HOURS_IN_JAN = 31 * 24


def raw_sample():
    """A tiny fake API response: 100 calls per borough at midnight Jan 1, plus 1 UNKNOWN call."""
    rows = [{"borough": b, "day": "2025-01-01T00:00:00.000", "hour": "0", "calls": "100"} for b in BOROUGHS]
    rows.append({"borough": "UNKNOWN", "day": "2025-01-01T00:00:00.000", "hour": "0", "calls": "1"})
    return pd.DataFrame(rows)


def test_month_bounds_handles_december():
    assert month_bounds("2025-12") == ("2025-12-01", "2026-01-01")


def test_list_months_includes_both_ends():
    assert len(list_months("2025-01", "2025-12")) == 12


def test_clean_month_fills_every_hour():
    df = clean_month(raw_sample(), "2025-01", CONFIG)
    assert len(df) == HOURS_IN_JAN * len(BOROUGHS)          # every hour x every borough
    assert df["calls"].sum() == 500                          # 5 boroughs x 100 calls
    assert (df["calls"] == 0).sum() == len(df) - 5           # all other hours filled with 0


def test_clean_month_drops_unknown():
    df = clean_month(raw_sample(), "2025-01", CONFIG)
    assert "UNKNOWN" not in set(df["borough"])


def test_validate_month_accepts_clean_data():
    df = clean_month(raw_sample(), "2025-01", CONFIG)
    assert validate_month(df, "2025-01", CONFIG)


def test_validate_month_catches_duplicates():
    df = clean_month(raw_sample(), "2025-01", CONFIG)
    bad = pd.concat([df, df.head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        validate_month(bad, "2025-01", CONFIG)


def test_validate_month_catches_negative_counts():
    df = clean_month(raw_sample(), "2025-01", CONFIG)
    df.loc[0, "calls"] = -5
    with pytest.raises(ValueError):
        validate_month(df, "2025-01", CONFIG)