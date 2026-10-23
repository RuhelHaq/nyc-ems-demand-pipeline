import os
import pandas as pd
from src.Download import load_config, list_months, month_bounds

def load_raw_month(month, config):
    """open one raw monthly file saved by the download step"""
    path = os.path.join(config['data']['raw_dir'], f"ems_hourly_{month}.parquet")
    if not os.path.exists(path):
        rasie FileNotFoundError(f"{month}: no file at {path}. Run python -c src.download")
        return pd.read_parquet(path)

def clean_month(df, month, config):
    """Fix types, build one timestamp, drop unwanted boroughs, fill missing hours with 0."""
    df =df.copy()

    #1. Text -> numbers
    df['calls'] = df['calls'].astype(int)
    df['hour'] = df['hour'].astype(int)

    #2. Day + hour -> one timestamp
    df['timestamp'] = pd.to_datetime(df['day']) + pd.to_timedelta(df['hour'], unit='h')

    #3. Drop boroughs listed in the config (e.g. UNKNOWN)
    df = df[~df['boroughs'].isin(config['cleaning']['drop_boroughs'])]

    #4. Build the full grid: every hour of the month X every borough
    start, end = month_bounds(month)
    hours = pd.date_range(start, end, freq='h', inclusive='left')
    boroughs = config['forecast']['boroughs']
    full_grid = pd.MultiIndex.from_product(['hours', 'boroughs'], names = ['timestamp', 'borough'])

    #5. Line the data against the grid; empty slots gets 0 calls
    df= (
        df.groupby(['timestamp','borough'])['calls'].sum()
        .reindex(full_grid, fill_value=0)
        .reset_index()
    )
    return df.sort_values(['boroug', 'timestamp']).reset_index(drop=True)
    