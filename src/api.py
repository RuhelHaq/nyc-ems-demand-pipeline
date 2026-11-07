import re
from datetime import date, datetime, time
from enum import Enum
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.download import load_config
from src.data import load_hourly
from src.features import FEATURE_COLS, to_shifts, add_calendar_features, add_history_features
from src.train import make_X

# ---------- Load once, when the API starts ----------
config = load_config()
SHIFT_HOURS = config["forecast"]["shift_hours"]
BOROUGHS = config["forecast"]["boroughs"]

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
MODEL_URI = f"models:/{config['mlflow']['registered_model']}@champion"
model = mlflow.xgboost.load_model(MODEL_URI)

hourly = load_hourly(config["data"]["history_start"], config["data"]["replay_end"], config)
shifts = to_shifts(hourly, SHIFT_HOURS)

# Dropdown choices for /docs, built from the config
Borough = Enum("Borough", {re.sub(r"\W+", "_", b): b for b in BOROUGHS}, type=str)
Shift = Enum("Shift", {f"starts_{h:02d}": f"{h:02d}:00" for h in range(0, 24, SHIFT_HOURS)}, type=str)


def build_features(shift_df):
    """Same feature steps as the pipeline: calendar + lag/rolling features."""
    shift_df = shift_df.sort_values(["borough", "shift_start"]).reset_index(drop=True)
    df = add_calendar_features(shift_df, SHIFT_HOURS)
    return add_history_features(df, SHIFT_HOURS)


features = build_features(shifts).dropna(subset=FEATURE_COLS)

app = FastAPI(title="NYC EMS Shift Demand API",
              description="Forecasts EMS calls per borough for each shift.")


# ---------- Endpoints ----------
@app.get("/health")
def health():
    """Check the API is running and see which model and data it uses."""
    return {"status": "ok", "model": MODEL_URI, "data_through": str(shifts["shift_start"].max())}


@app.get("/forecast/next")
def forecast_next(borough: Optional[Borough] = None):
    """Forecast the next shift after the latest data. Leave borough empty for all boroughs."""
    next_start = shifts["shift_start"].max() + pd.Timedelta(hours=SHIFT_HOURS)
    empty_rows = pd.DataFrame({"borough": BOROUGHS, "shift_start": next_start, "calls": float("nan")})
    full = build_features(pd.concat([shifts, empty_rows], ignore_index=True))
    rows = full[full["shift_start"] == next_start]
    if borough is not None:
        rows = rows[rows["borough"] == borough.value]
    preds = model.predict(make_X(rows, config))
    return {"shift_start": str(next_start), "shift_hours": SHIFT_HOURS,
            "forecasts": {b: round(float(p), 1) for b, p in zip(rows["borough"], preds)}}


@app.get("/forecast")
def forecast(borough: Borough, day: date, shift: Shift):
    """Look up a past shift: the model's prediction and the actual number of calls."""
    start = pd.Timestamp(datetime.combine(day, time(int(shift.value[:2]))))
    row = features[(features["borough"] == borough.value) & (features["shift_start"] == start)]
    if row.empty:
        raise HTTPException(status_code=404,
                            detail=f"No data for that shift. Data covers {features['shift_start'].min():%Y-%m-%d} "
                                   f"to {features['shift_start'].max():%Y-%m-%d}.")
    pred = float(model.predict(make_X(row, config))[0])
    actual = int(row["calls"].iloc[0])
    return {"borough": borough.value, "shift_start": str(start),
            "predicted_calls": round(pred, 1), "actual_calls": actual,
            "error": round(pred - actual, 1)}


# ---------- Simple dashboard page ----------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    borough_options = "".join(f"<option>{b}</option>" for b in BOROUGHS)
    shift_options = "".join(f"<option>{s.value}</option>" for s in Shift)
    return f"""
<!doctype html><html><head><meta charset="utf-8"><title>NYC EMS Forecast</title>
<style>
 body {{ font-family: Arial, sans-serif; max-width: 640px; margin: 32px auto; padding: 0 16px; color: #222; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; }}
 select, input, button {{ font-size: 15px; padding: 6px; margin: 4px 4px 4px 0; }}
 button {{ background: #1F4E79; color: white; border: 0; border-radius: 6px; padding: 8px 14px; cursor: pointer; }}
 .box {{ background: #f2f6fa; border-radius: 8px; padding: 12px 16px; margin-top: 10px; }}
 table {{ border-collapse: collapse; }} td {{ padding: 4px 12px 4px 0; }}
</style></head><body>
<h1>NYC EMS Shift Demand Forecast</h1>
<p>Model: <code>{MODEL_URI}</code></p>

<h2>Next shift forecast</h2>
<button onclick="nextShift()">Forecast next shift</button>
<div id="next" class="box">Click the button.</div>

<h2>Check a past shift</h2>
<select id="borough">{borough_options}</select>
<input id="day" type="date" value="2026-07-02">
<select id="shift">{shift_options}</select>
<button onclick="lookup()">Look up</button>
<div id="past" class="box">Pick a borough, date, and shift.</div>

<p style="margin-top:28px"><a href="/docs">API docs</a></p>
<script>
async function nextShift() {{
  const r = await fetch('/forecast/next'); const d = await r.json();
  let rows = Object.entries(d.forecasts).map(([b, v]) => `<tr><td>${{b}}</td><td><b>${{v}}</b> calls</td></tr>`).join('');
  document.getElementById('next').innerHTML = `<p>Shift starting <b>${{d.shift_start}}</b></p><table>${{rows}}</table>`;
}}
async function lookup() {{
  const b = encodeURIComponent(document.getElementById('borough').value);
  const day = document.getElementById('day').value;
  const s = encodeURIComponent(document.getElementById('shift').value);
  const r = await fetch(`/forecast?borough=${{b}}&day=${{day}}&shift=${{s}}`); const d = await r.json();
  document.getElementById('past').innerHTML = r.ok
    ? `<p><b>${{d.borough}}</b>, shift starting ${{d.shift_start}}</p>
       <p>Predicted: <b>${{d.predicted_calls}}</b> &nbsp; Actual: <b>${{d.actual_calls}}</b> &nbsp; Error: ${{d.error}}</p>`
    : `<p>${{d.detail}}</p>`;
}}
</script></body></html>"""