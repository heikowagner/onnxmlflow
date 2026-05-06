from __future__ import annotations

import os
import tempfile
from typing import Optional

import onnx
import mlflow

from .html_generator import generate_viewer_html


def log_model(
    model_path: str,
    artifact_dir: str = "onnx_viewer",
    run_id: Optional[str] = None,
) -> str:
    """Log an ONNX model to MLflow together with an interactive HTML viewer.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file on disk.
    artifact_dir:
        Subdirectory inside the run's artifact store where both the model file
        and ``viewer.html`` will be stored.
    run_id:
        Target MLflow run ID.  When *None* the currently active run is used.

    Returns
    -------
    str
        Artifact-relative path of the generated HTML viewer,
        e.g. ``"onnx_viewer/viewer.html"``.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not point to an existing file.
    RuntimeError
        If no MLflow run is active and *run_id* is not provided.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    model_filename = os.path.basename(model_path)
    html_content = generate_viewer_html(model, model_filename, artifact_dir)

    active_run = mlflow.active_run()
    resolved_run_id = run_id or (active_run.info.run_id if active_run else None)
    if resolved_run_id is None:
        raise RuntimeError(
            "No active MLflow run. Call log_model inside a "
            "`with mlflow.start_run():` block or pass run_id explicitly."
        )

    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        html_path = os.path.join(tmp, "viewer.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)

        client.log_artifact(resolved_run_id, model_path, artifact_path=artifact_dir)
        client.log_artifact(resolved_run_id, html_path, artifact_path=artifact_dir)

    return f"{artifact_dir}/viewer.html"
