import logging
import re

import mlflow
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.download import load_config
from src.data import load_hourly
from src.features import FEATURE_COLS, make_features, make_X

logging.getLogger("cmdstanpy").disabled = True  # quiet Prophet's messages


def split_by_month(df, start_month, end_month):
    """Keep rows whose shift starts within [start_month, end_month], inclusive."""
    start = pd.Period(start_month, freq="M").start_time
    end = pd.Period(end_month, freq="M").end_time
    return df[(df["shift_start"] >= start) & (df["shift_start"] <= end)].copy()



def score(test, preds):
    """MAE overall and per borough, with names MLflow accepts."""
    metrics = {"mae": mean_absolute_error(test["calls"], preds)}
    for b, group in test.assign(pred=preds).groupby("borough"):
        name = "mae_" + re.sub(r"[^A-Za-z0-9]+", "_", b).strip("_")
        metrics[name] = mean_absolute_error(group["calls"], group["pred"])
    return metrics


def predict_baseline(train, test, config):
    """Same shift last week."""
    return test["lag_week"].values


def predict_xgboost(train, test, config):
    """One model for all boroughs."""
    model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                         random_state=config["model"]["random_state"])
    model.fit(make_X(train, config), train["calls"])
    return model.predict(make_X(test, config))


def predict_prophet(train, test, config):
    """One Prophet model per borough."""
    preds = pd.Series(index=test.index, dtype=float)
    for b in config["forecast"]["boroughs"]:
        tr = train[train["borough"] == b].rename(columns={"shift_start": "ds", "calls": "y"})
        te = test[test["borough"] == b]
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m.add_country_holidays(country_name="US")
        m.fit(tr[["ds", "y"]])
        forecast = m.predict(te.rename(columns={"shift_start": "ds"})[["ds"]])
        preds.loc[te.index] = forecast["yhat"].values
    return preds.values


MODELS = {"baseline": predict_baseline, "prophet": predict_prophet, "xgboost": predict_xgboost}


def train_and_log(name, train, test, config):
    """Train one model, score it on the test month, and record it in MLflow."""
    with mlflow.start_run(run_name=name):
        preds = MODELS[name](train, test, config)
        metrics = score(test, preds)
        mlflow.log_params({
            "model": name,
            "shift_hours": config["forecast"]["shift_hours"],
            "train_start": str(train["shift_start"].min()),
            "train_end": str(train["shift_start"].max()),
            "test_month": str(test["shift_start"].min())[:7],
        })
        mlflow.log_metrics(metrics)
    return metrics


if __name__ == "__main__":
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    hourly = load_hourly(config["data"]["history_start"], config["data"]["replay_end"], config)
    df = make_features(hourly, config)
    train = split_by_month(df, config["data"]["history_start"], config["data"]["history_end"])
    test = split_by_month(df, config["data"]["replay_start"], config["data"]["replay_start"])
    print(f"Train rows: {len(train)}, test rows: {len(test)}")

    for name in MODELS:
        metrics = train_and_log(name, train, test, config)
        print(f"{name:9s}  MAE = {metrics['mae']:.1f} calls per shift")