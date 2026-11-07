import pandas as pd

from src.features import add_history_features, to_shifts


def hourly_sample():
    """10 days of fake hourly data: borough A always 1 call per hour, borough B always 100."""
    hours = pd.date_range("2025-01-01", periods=24 * 10, freq="h")
    a = pd.DataFrame({"timestamp": hours, "borough": "A", "calls": 1})
    b = pd.DataFrame({"timestamp": hours, "borough": "B", "calls": 100})
    return pd.concat([a, b], ignore_index=True)


def test_to_shifts_adds_up_hours():
    shifts = to_shifts(hourly_sample(), 8)
    assert len(shifts) == 10 * 3 * 2                                   # 10 days x 3 shifts x 2 boroughs
    assert (shifts.loc[shifts["borough"] == "A", "calls"] == 8).all()   # 8 hours x 1 call
    assert (shifts.loc[shifts["borough"] == "B", "calls"] == 800).all()


def test_lags_stay_within_borough():
    df = add_history_features(to_shifts(hourly_sample(), 8), 8)
    a = df[df["borough"] == "A"].dropna(subset=["lag_prev_shift"])
    assert (a["lag_prev_shift"] == 8).all()        # A's lag must never be B's 800


def test_rolling_average_does_not_peek_at_current_shift():
    shifts = to_shifts(hourly_sample(), 8)
    last_a = shifts[shifts["borough"] == "A"].index[-1]
    shifts.loc[last_a, "calls"] = 9999             # an extreme value in the shift we 'predict'
    df = add_history_features(shifts, 8)
    assert df.loc[last_a, "roll_7d"] == 8          # the spike must not leak into its own feature