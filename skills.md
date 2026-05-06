---
name: onnxmlflow
description: "Workflows for the onnxmlflow project. Use when: adding support for new ONNX types, updating the HTML viewer, changing how model metadata is extracted, adding new MLflow logging features, creating test models, debugging artifact uploads, or understanding the template substitution system."
argument-hint: "Describe the change or feature to work on"
---

# onnxmlflow Skills

## What This Project Does

`onnxmlflow` is a Python library that logs `.onnx` model files to MLflow
**and** generates a self-contained `viewer.html` artifact alongside the model.
The viewer uses **ONNX Runtime Web** (JavaScript, loaded from CDN) to run
live inference inside MLflow's artifact viewer iframe — no server-side
compute required.

## Project Layout

```
src/onnxmlflow/
├── __init__.py            public API: log_model()
├── logger.py              orchestrates MLflow artifact upload
├── html_generator.py      reads ONNX metadata → renders HTML template
└── templates/
    └── viewer.html        self-contained HTML+JS viewer

tests/
├── __init__.py
├── create_test_model.py   builds iris_classifier.onnx via skl2onnx
└── test_logger.py         pytest tests for log_model()

examples/
└── log_iris_model.py      end-to-end demo script
```

## Core Workflows

### Log an ONNX model
```python
import mlflow, onnxmlflow

with mlflow.start_run():
    onnxmlflow.log_model("model.onnx", artifact_dir="onnx_viewer")
```

### Create the test fixture model
```bash
python tests/create_test_model.py
```

### Run the full example
```bash
mlflow ui --port 5000 &
python examples/log_iris_model.py
# Open MLflow UI → experiment 'onnx-viewer-demo' → run → Artifacts → onnx_viewer/viewer.html
```

### Run tests
```bash
pip install -e ".[dev]"
pytest
```

## HTML Viewer Architecture

| Placeholder (in template)  | Replaced with                              |
|----------------------------|--------------------------------------------|
| `__MODEL_FILENAME__`       | basename of the `.onnx` file               |
| `__MODEL_ARTIFACT_DIR__`   | artifact subdirectory path                 |
| `__MODEL_NAME__`           | `model.graph.name` or filename stem        |
| `__INPUTS_META_JSON__`     | JSON array `[{name, type, shape}]`         |
| `__OUTPUTS_META_JSON__`    | JSON array `[{name, type, shape}]`         |

Replacements are plain string substitution in `html_generator.py`.

### Model URL in the Browser

MLflow serves the viewer at:
`/get-artifact?path=<dir>/viewer.html&run_uuid=<id>`

The HTML extracts `run_uuid` from `window.location.search` and constructs:
`/get-artifact?path=<dir>/<model>.onnx&run_uuid=<id>`

### ONNX type → HTML input mapping

| ONNX type         | `<input>` rendered          |
|-------------------|-----------------------------|
| FLOAT, DOUBLE     | `type="number" step="any"`  |
| INT*, UINT*       | `type="number" step="1"`    |
| STRING            | `type="text"`               |
| BOOL              | `type="checkbox"`           |

Dynamic / symbolic shape dimensions (e.g. `batch_size`) are resolved to **1**
at inference time.

## Key Dependencies

- `onnx>=1.14` — model loading and metadata inspection
- `mlflow>=2.0` — experiment tracking and artifact storage
- `onnxruntime-web@1.20.1` (CDN) — browser-side ONNX inference

See [tools.md](../tools.md) for the full tech-stack reference.
