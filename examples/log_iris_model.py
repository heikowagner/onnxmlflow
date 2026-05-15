"""End-to-end example: train an Iris model → export ONNX → log_model() → view in MLflow UI."""
from __future__ import annotations

import os
import sys
import tempfile

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import onnxmlflow
from tests.create_test_model import create_iris_model


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name

    try:
        print("Building Iris classifier ONNX model…")
        create_iris_model(model_path)

        mlflow.set_experiment("onnx-viewer-demo")

        with mlflow.start_run() as run:
            viewer = onnxmlflow.log_model(
                model_path,
                artifact_dir="onnx_viewer",
                groups={
                    "Pipeline": {
                        "Preprocessing": ["Normalizer*", "*Identity*"],
                        "Classifier":    ["Linear*", "Cast"],
                    },
                },
                output_groups={
                    "Predictions": ["label", "probabilities"],
                },
            )
            run_id = run.info.run_id

        tracking_uri = mlflow.get_tracking_uri()
        print(f"\nLogged successfully!")
        print(f"  Tracking URI : {tracking_uri}")
        print(f"  Run ID       : {run_id}")
        print(f"  Viewer path  : {viewer}")
        print()
        print("To inspect the interactive viewer:")
        print("  1. mlflow ui --port 5000")
        print("  2. Open http://localhost:5000")
        print("  3. Navigate to experiment 'onnx-viewer-demo' → run → Artifacts → onnx_viewer/viewer.html")
    finally:
        os.unlink(model_path)


if __name__ == "__main__":
    main()
