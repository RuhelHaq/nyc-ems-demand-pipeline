import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.download import load_config
from src.data import load_hourly
from src.features import make_features
from src.train import split_by_month, make_X, score

ALIAS = "champion"


def fit_xgboost(train, config):
    """Train an XGBoost model on the training rows."""
    model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                         random_state=config["model"]["random_state"])
    model.fit(make_X(train, config), train["calls"])
    return model


def log_challenger(model, train, test, config, client):
    """Score the new model, save it in MLflow, and register it as a new version."""
    name = config["mlflow"]["registered_model"]
    metrics = score(test, model.predict(make_X(test, config)))

    with mlflow.start_run(run_name="xgboost-challenger") as run:
        mlflow.log_params({
            "model": "xgboost",
            "train_end": str(train["shift_start"].max()),
            "test_month": str(test["shift_start"].min())[:7],
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model", registered_model_name=name)

    # Find the version number MLflow just created for this run
    versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
    return versions[0].version, metrics["mae"]


def get_champion_version(client, name):
    """Return the current champion's version number, or None if there isn't one yet."""
    try:
        return client.get_model_version_by_alias(name, ALIAS).version
    except MlflowException:
        return None


def champion_mae(name, test, config):
    """Load the current champion and score it on the same test month."""
    model = mlflow.xgboost.load_model(f"models:/{name}@{ALIAS}")
    return mean_absolute_error(test["calls"], model.predict(make_X(test, config)))


def promote_if_better(client, version, challenger_mae, test, config):
    """Move the 'champion' label to the challenger only if it beats the champion."""
    name = config["mlflow"]["registered_model"]
    current = get_champion_version(client, name)

    if current is None:
        client.set_registered_model_alias(name, ALIAS, version)
        decision = f"v{version} promoted (first champion), MAE {challenger_mae:.2f}"
    else:
        champ = champion_mae(name, test, config)
        if challenger_mae < champ:
            client.set_registered_model_alias(name, ALIAS, version)
            decision = f"v{version} promoted: MAE {challenger_mae:.2f} beat v{current} ({champ:.2f})"
        else:
            decision = f"kept v{current}: challenger v{version} MAE {challenger_mae:.2f} vs champion {champ:.2f}"

    client.set_model_version_tag(name, version, "test_mae", f"{challenger_mae:.2f}")
    return decision


if __name__ == "__main__":
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    client = MlflowClient()

    hourly = load_hourly(config["data"]["history_start"], config["data"]["replay_end"], config)
    df = make_features(hourly, config)
    train = split_by_month(df, config["data"]["history_start"], config["data"]["history_end"])
    test = split_by_month(df, config["data"]["replay_start"], config["data"]["replay_start"])

    model = fit_xgboost(train, config)
    version, mae = log_challenger(model, train, test, config, client)
    print(promote_if_better(client, version, mae, test, config))