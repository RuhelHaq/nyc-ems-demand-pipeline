import os
import time
import pandas as pd 
import requests
import yaml

def load_config(path="config.yaml"):
    """Read the settings file nad return it as dictionary"""
    with open(path) as f:
        return yaml.safe_load(f)

def list_months(start, end):
    """Return every month from start to end"""
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(m) for m in months]

def month_bounds(month):
    """Return every month from start to end"""
    start = pd.Period(month, freq="M")
    end = start + 1
    return f"{start}-01", f"{end}-01"

def download_month(month, config):
    """Download hourly call counts by borough for one month and save them"""
    raw_dir = config["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"ems_hourly_{month}.parquet")

    # skip months we already have 
    if os.path.exists(path):
        print(f"{month}: already downloaded, skipping")
        return path
    start, end = month_bounds(month)
    params = {
        "$select": "borough, date_trunc_ymd(incident_datetime) AS day, "
                    "date_extract_hh(incident_datetime) AS hour, count(*) AS calls",
         "$where": f"incident_datetime >= '{start}' AND incident_datetime < '{end}'",
         "$group": "borough, day, hour",
         "$limit": 50000,           
    }

    response = requests.get(config['data']['api_url'], params= params, timeout =120)
    response.raise_for_status() # stop with an error if API complained

    df = pd.DataFrame(response.json())
    df.to_parquet(path, index=False)
    print(f"{month}: saved {len(df)} rows")
    return path

if __name__ == "__main__":
    config = load_config()
    months = list_months(config['data']['history_start'], config['data']['replay_end'])
    print(f"Downloading {len(months)} months...")

    for month in months:
        download_month(month, config)
        time.sleep(1) # one request per second
        