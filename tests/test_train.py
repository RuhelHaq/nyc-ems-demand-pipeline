import pandas as pd

from src.features import FEATURE_COLS
from src.train import make_X, score

CONFIG = {"forecast": {"boroughs": ["A", "B"]}}


def sample():
    df = pd.DataFrame({col: [0, 0, 0] for col in FEATURE_COLS})
    df["borough"] = ["A", "B", "A"]
    df["calls"] = [10, 20, 30]
    return df


def test_make_x_adds_one_column_per_borough():
    X = make_X(sample(), CONFIG)
    assert list(X["is_A"]) == [1, 0, 1]
    assert list(X["is_B"]) == [0, 1, 0]
    assert "borough" not in X.columns           # the model only gets numbers


def test_score_is_zero_for_perfect_predictions():
    df = sample()
    metrics = score(df, df["calls"].values)
    assert metrics["mae"] == 0
    assert metrics["mae_A"] == 0 and metrics["mae_B"] == 0


def test_score_measures_average_miss():
    df = sample()
    metrics = score(df, df["calls"].values + 5)  # every prediction 5 too high
    assert metrics["mae"] == 5