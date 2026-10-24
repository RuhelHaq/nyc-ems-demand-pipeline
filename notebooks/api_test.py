import requests
# nyc ems call url
URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"

# Test 1: one raw row , to check the real column names
r = requests.get(URL, params = {"$limit": 1}, timeout = 60)
print('Test 1 - Columns:')
print(list(r.json()[0].keys()))

# Test 2: the data range of whole dataset
r = requests.get(URL, params = {"$select": "min(incident_datetime), max(incident_datetime)"}, timeout=60)
print("\nTest 2 - data range:")
print(r.json())

# Test 3: one month, already counted by borough and hour
params = {
    "$select": "borough, date_trunc_ymd(incident_datetime) AS day, "
                "date_extract_hh(incident_datetime) AS hour, count(*) AS calls",
    "$where": "incident_datetime >= '2025-01-01' AND incident_datetime < '2025-02-01'",
    "$group": "borough, day, hour",
    "$limit": 500000,
}
r = requests.get(URL, params= params, timeout=120)
rows = r.json()
print("\nTest 3 - status:", r.status_code)
print(rows if isinstance(rows, dict) else rows[:3])
print("rows:", len(rows))

# Test 4: check exact borough names 
params = {
    "$select": "borough, count(*) AS calls",
    "$where": "incident_datetime >= '2025-01-01' AND incident_datetime < '2026-01-01'",
    "$group": "borough",
}
r = requests.get(URL, params = params, timeout=120)
print("\nTest 4 - boroughs in 2015:")
print(r.json())