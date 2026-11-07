import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.download import load_config, list_months
from src.data import load_hourly
from src.features import make_features
from src.train import split_by_month, make_X, score
from src.evaluate import ALIAS, fit_xgboost, get_champion_version, champion_mae


def drift_check(history, month_df, threshold):
    """Compare each borough's average calls this month with its previous 90 days."""
    month_start = month_df["shift_start"].min()
    recent = history[history["shift_start"] >= month_start - pd.Timedelta(days=90)]
    before = recent.groupby("borough")["calls"].mean()
    now = month_df.groupby("borough")["calls"].mean()
    change = (now - before) / before
    worst = change.abs().idxmax()
    return float(change[worst]), worst, bool(change.abs().max() > threshold)


def run_month(month, df, config, client):
    """One month of 'production': monitor, check drift, retrain, promote or keep."""
    name = config["mlflow"]["registered_model"]
    month_start = pd.Period(month, freq="M").start_time
    history = df[df["shift_start"] < month_start]          # everything before this month
    test = split_by_month(df, month, month)                 # the month that just "arrived"

    with mlflow.start_run(run_name=f"replay-{month}") as run:
        # 1. Score the current champion on the new month
        champ_version = get_champion_version(client, name)
        champ_mae = champion_mae(name, test, config)

        # 2. Score the baseline (same shift last week)
        base_mae = mean_absolute_error(test["calls"], test["lag_week"])

        # 3. Drift check
        drift, drift_borough, drift_flag = drift_check(history, test, config["replay"]["drift_threshold"])

        # 4. Train a challenger on all data before this month
        model = fit_xgboost(history, config)
        chall_metrics = score(test, model.predict(make_X(test, config)))
        chall_mae = chall_metrics["mae"]
        mlflow.xgboost.log_model(model, name="model", registered_model_name=name)
        chall_version = client.search_model_versions(f"run_id='{run.info.run_id}'")[0].version

        # 5. Promote only if the challenger is better
        promoted = chall_mae < champ_mae
        if promoted:
            client.set_registered_model_alias(name, ALIAS, chall_version)

        mlflow.log_params({"month": month, "champion_version": champ_version,
                           "challenger_version": chall_version, "drift_borough": drift_borough})
        mlflow.log_metrics({"champion_mae": champ_mae, "challenger_mae": chall_mae,
                            "baseline_mae": base_mae, "drift_pct": drift * 100,
                            "drift_flag": int(drift_flag), "promoted": int(promoted)})

    return {"month": month, "champion_v": champ_version, "champion_mae": round(champ_mae, 2),
            "challenger_mae": round(chall_mae, 2), "baseline_mae": round(base_mae, 2),
            "drift_pct": round(drift * 100, 1), "drift_borough": drift_borough,
            "drift_flag": drift_flag, "promoted": promoted}


if __name__ == "__main__":
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    client = MlflowClient()

    hourly = load_hourly(config["data"]["history_start"], config["data"]["replay_end"], config)
    df = make_features(hourly, config)

    results = []
    for month in list_months(config["data"]["replay_start"], config["data"]["replay_end"]):
        row = run_month(month, df, config, client)
        results.append(row)
        print(f"{month}: champion v{row['champion_v']} MAE {row['champion_mae']}, "
              f"challenger {row['challenger_mae']}, baseline {row['baseline_mae']}, "
              f"drift {row['drift_pct']}% ({row['drift_borough']}) "
              f"{'DRIFT ' if row['drift_flag'] else ''}{'-> PROMOTED' if row['promoted'] else ''}")

    summary = pd.DataFrame(results)
    os.makedirs("models", exist_ok=True)
    summary.to_csv("models/replay_summary.csv", index=False)
    print(f"\nPromoted {summary['promoted'].sum()} of {len(summary)} months. "
          f"Average champion MAE {summary['champion_mae'].mean():.2f} vs baseline {summary['baseline_mae'].mean():.2f}.")