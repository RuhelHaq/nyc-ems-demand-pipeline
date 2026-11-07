import shutil

import mlflow

from src.download import load_config

if __name__ == "__main__":
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    uri = f"models:/{config['mlflow']['registered_model']}@champion"
    shutil.rmtree("models/champion", ignore_errors=True)       # remove any old export
    path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path="models/champion")
    print("Exported champion model to", path)
    