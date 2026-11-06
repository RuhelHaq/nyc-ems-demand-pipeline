import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.download import load_config
from src.data import load_hourly

FEATURE_COLS = [
    "shift_of_day", "day_of_week", "is_weekend", "month", "is_holiday",
    "lag_prev_shift", "lag_day", "lag_week", "roll_7d", "roll_28d",
]


def to_shifts(hourly, shift_hours):
    """Combine hourly counts into shifts (e.g. 8-hour blocks) per borough."""
    df = hourly.copy()
    df["shift_start"] = df["timestamp"].dt.floor(f"{shift_hours}h")
    shifts = df.groupby(["borough", "shift_start"], as_index=False)["calls"].sum()
    return shifts.sort_values(["borough", "shift_start"]).reset_index(drop=True)


def add_calendar_features(df, shift_hours):
    """Add features that are known ahead of time: shift, weekday, month, holidays."""
    df = df.copy()
    ts = df["shift_start"]
    df["shift_of_day"] = ts.dt.hour // shift_hours
    df["day_of_week"] = ts.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.month

    holidays = USFederalHolidayCalendar().holidays(start=ts.min(), end=ts.max())
    df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    return df


def add_history_features(df, shift_hours):
    """Add lag and rolling features, computed separately for each borough."""
    df = df.copy()
    per_day = 24 // shift_hours            # 3 shifts per day when shifts are 8 hours
    by_borough = df.groupby("borough")["calls"]

    df["lag_prev_shift"] = by_borough.shift(1)
    df["lag_day"] = by_borough.shift(per_day)
    df["lag_week"] = by_borough.shift(7 * per_day)

    # shift(1) first, so the average only uses shifts that have already finished
    df["roll_7d"] = by_borough.transform(lambda s: s.shift(1).rolling(7 * per_day).mean())
    df["roll_28d"] = by_borough.transform(lambda s: s.shift(1).rolling(28 * per_day).mean())
    return df


def make_features(hourly, config):
    """Hourly data in, model-ready table out."""
    shift_hours = config["forecast"]["shift_hours"]
    df = to_shifts(hourly, shift_hours)
    df = add_calendar_features(df, shift_hours)
    df = add_history_features(df, shift_hours)
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


if __name__ == "__main__":
    config = load_config()
    hourly = load_hourly(config["data"]["history_start"], config["data"]["replay_end"], config)
    features = make_features(hourly, config)
    print("Shape:", features.shape)
    print("From", features["shift_start"].min(), "to", features["shift_start"].max())
    print("Missing values:", features[FEATURE_COLS].isna().sum().sum())
    print(features.head())