import os

import pandas as pd

from src.download import load_config, list_months, month_bounds


def load_raw_month(month, config):
    """Open one raw monthly file saved by the download step."""
    path = os.path.join(config["data"]["raw_dir"], f"ems_hourly_{month}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{month}: no file at {path}. Run: python -m src.download")
    return pd.read_parquet(path)


def clean_month(df, month, config):
    """Fix types, build one timestamp, drop unwanted boroughs, fill missing hours with 0."""
    df = df.copy()

    # 1. Text -> numbers
    df["calls"] = df["calls"].astype(int)
    df["hour"] = df["hour"].astype(int)

    # 2. Day + hour -> one timestamp (e.g. 2025-01-15 18:00)
    df["timestamp"] = pd.to_datetime(df["day"]) + pd.to_timedelta(df["hour"], unit="h")

    # 3. Drop boroughs listed in the config (e.g. UNKNOWN)
    df = df[~df["borough"].isin(config["cleaning"]["drop_boroughs"])]

    # 4. Build the full grid: every hour of the month x every borough
    start, end = month_bounds(month)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    boroughs = config["forecast"]["boroughs"]
    full_grid = pd.MultiIndex.from_product([hours, boroughs], names=["timestamp", "borough"])

    # 5. Line the data up against the grid; empty slots get 0 calls
    df = (
        df.groupby(["timestamp", "borough"])["calls"].sum()
        .reindex(full_grid, fill_value=0)
        .reset_index()
    )
    return df.sort_values(["borough", "timestamp"]).reset_index(drop=True)


def validate_month(df, month, config):
    """Stop with a clear error if the cleaned month looks wrong."""
    start, end = month_bounds(month)
    n_hours = len(pd.date_range(start, end, freq="h", inclusive="left"))
    boroughs = config["forecast"]["boroughs"]

    if len(df) != n_hours * len(boroughs):
        raise ValueError(f"{month}: expected {n_hours * len(boroughs)} rows, got {len(df)}")

    if df.duplicated(["timestamp", "borough"]).any():
        raise ValueError(f"{month}: duplicate borough-hours found")

    if (df["calls"] < 0).any():
        raise ValueError(f"{month}: negative call counts found")

    totals = df.groupby("borough")["calls"].sum()
    empty = totals[totals == 0].index.tolist()
    if empty:
        raise ValueError(f"{month}: no calls at all for {empty} - check borough names")

    return True


def load_hourly(start_month, end_month, config):
    """Load, clean, and validate every month in a range, then stack them into one table."""
    frames = []
    for month in list_months(start_month, end_month):
        df = clean_month(load_raw_month(month, config), month, config)
        validate_month(df, month, config)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    config = load_config()
    df = load_hourly(config["data"]["history_start"], config["data"]["replay_end"], config)
    print("Shape:", df.shape)
    print("From", df["timestamp"].min(), "to", df["timestamp"].max())
    print(df.groupby("borough")["calls"].sum())