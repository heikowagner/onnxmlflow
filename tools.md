# Tools & Technology Reference

## Python Runtime

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.9 | `match` statements not used; 3.9 `Optional` still used |

## Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mlflow` | ≥ 2.0 | Experiment tracking, run management, artifact upload/download |
| `onnx` | ≥ 1.14 | Load `.onnx` files, inspect graph I/O, `checker.check_model()` |
| `numpy` | ≥ 1.24 | Transitive; required by onnx and onnxruntime |

## Dev / Test Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 7 | Test runner |
| `scikit-learn` | ≥ 1.2 | Generate Iris fixture model |
| `skl2onnx` | ≥ 1.14 | Convert sklearn estimators to ONNX format |

Install everything:
```bash
pip install -e ".[dev]"
```

## Frontend (viewer.html)

| Library | Version | How loaded | Purpose |
|---------|---------|------------|---------|
| `onnxruntime-web` | 1.20.1 | CDN `<script>` tag | Run ONNX inference in the browser |

CDN URL:
```
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js
```

> To pin or update the version, edit the `<script src=...>` line in
> `src/onnxmlflow/templates/viewer.html`.

## Build System

| Tool | Config file | Purpose |
|------|-------------|---------|
| setuptools ≥ 68 | `pyproject.toml` | Package build backend |
| wheel | `pyproject.toml` | Binary distribution |

Build a wheel:
```bash
pip install build
python -m build
```

## Version Control

| Tool | Notes |
|------|-------|
| git | Standard workflow; `.gitignore` excludes `*.onnx`, `mlruns/`, venvs |

## MLflow Artifact Layout

After `log_model("model.onnx", artifact_dir="onnx_viewer")`:

```
<mlflow-run>/
└── artifacts/
    └── onnx_viewer/
        ├── model.onnx        ← original model file
        └── viewer.html       ← generated interactive viewer
```

The viewer is accessible in the MLflow UI under
**Experiments → Run → Artifacts → onnx_viewer/viewer.html**.

## Key Source Files

| File | Role |
|------|------|
| `src/onnxmlflow/logger.py` | `log_model()` — validates model, calls generator, uploads artifacts |
| `src/onnxmlflow/html_generator.py` | `generate_viewer_html()` — reads ONNX graph metadata, renders template |
| `src/onnxmlflow/templates/viewer.html` | Single-file HTML+JS viewer; uses `__PLACEHOLDER__` strings |
| `tests/create_test_model.py` | `create_iris_model()` — builds a test `.onnx` with skl2onnx |
| `tests/test_logger.py` | pytest suite: artifact upload, HTML content, error cases |
| `examples/log_iris_model.py` | End-to-end runnable demo |
