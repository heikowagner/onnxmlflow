"""Tests for onnxmlflow.logger.log_model()."""
from __future__ import annotations

import os
import sys

import pytest

# Allow imports without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mlflow
import onnxmlflow
from tests.create_test_model import create_iris_model


# ─── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def iris_onnx(tmp_path_factory):
    """Return the path to a freshly built ONNX Iris classifier."""
    tmp = tmp_path_factory.mktemp("models")
    path = str(tmp / "iris_classifier.onnx")
    create_iris_model(path)
    return path


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_log_model_returns_viewer_path(iris_onnx, tmp_path):
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run() as run:
        viewer_path = onnxmlflow.log_model(iris_onnx, artifact_dir="onnx_viewer")

    assert viewer_path == "onnx_viewer/viewer.html"

    client = mlflow.tracking.MlflowClient(tracking_uri)
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "onnx_viewer")}
    assert "onnx_viewer/iris_classifier.onnx" in artifacts
    assert "onnx_viewer/viewer.html" in artifacts


def test_log_model_html_contains_input_metadata(iris_onnx, tmp_path):
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run() as run:
        onnxmlflow.log_model(iris_onnx, artifact_dir="onnx_viewer")

    client = mlflow.tracking.MlflowClient(tracking_uri)
    local_html = client.download_artifacts(run.info.run_id, "onnx_viewer/viewer.html", str(tmp_path))

    html = open(local_html, encoding="utf-8").read()
    # The Iris model input is named "float_input" with type FLOAT
    assert "float_input" in html
    assert "FLOAT" in html
    assert "onnxruntime-web" in html
    # Template placeholders must be replaced by real values
    assert "__MODEL_FILENAME__" not in html
    assert "__INPUTS_META_JSON__" not in html


def test_log_model_raises_on_missing_file(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    with mlflow.start_run():
        with pytest.raises(FileNotFoundError):
            onnxmlflow.log_model("/nonexistent/path/model.onnx")


def test_log_model_raises_without_active_run(iris_onnx):
    mlflow.end_run()  # ensure no run is active
    with pytest.raises(RuntimeError, match="No active MLflow run"):
        onnxmlflow.log_model(iris_onnx)
