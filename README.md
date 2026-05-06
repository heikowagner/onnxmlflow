# onnxmlflow

Log ONNX models to MLflow and get a self-contained **interactive HTML viewer** alongside every run — no extra server, no plugins required.

The viewer runs inside MLflow's built-in artifact UI and uses [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (JavaScript) to execute inference directly in the browser.

![Viewer screenshot placeholder](https://raw.githubusercontent.com/placeholder/onnxmlflow/main/docs/screenshot.png)

---

## Features

- One-call logging: `onnxmlflow.log_model("model.onnx")`
- Viewer reads input/output metadata from the ONNX graph and renders matching form controls automatically:
  - `FLOAT` / `DOUBLE` → `<input type="number">`
  - `INT*` / `UINT*` → `<input type="number" step="1">`
  - `STRING` → `<input type="text">`
  - `BOOL` → `<input type="checkbox">`
- Symbolic batch dimensions (e.g. `batch_size`) resolve to 1 at inference time
- Probability outputs (`Sequence<Map<class, score>>`) render as a sorted table with visual bars
- Model bytes are **base64-embedded** in the HTML — works inside MLflow's sandboxed iframe without any network fetch

---

## Installation

```bash
pip install onnxmlflow
```

For development (includes scikit-learn + skl2onnx for test model generation):

```bash
git clone https://github.com/your-org/onnxmlflow
cd onnxmlflow
pip install -e ".[dev]"
```

---

## Quick start

```python
import mlflow
import onnxmlflow

with mlflow.start_run():
    onnxmlflow.log_model("my_model.onnx", artifact_dir="onnx_viewer")
```

Then open MLflow UI, navigate to the run's **Artifacts** tab, and click `onnx_viewer/viewer.html`.

---

## Running the example

The repo ships with an end-to-end demo that trains a Logistic Regression on the Iris dataset, exports it to ONNX, and logs it to a local MLflow SQLite store.

```bash
# 1. Install dev dependencies
pip install -e ".[dev]"

# 2. Run the example
python examples/log_iris_model.py

# 3. Start the MLflow UI
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db

# 4. Open http://localhost:5001
#    Go to experiment "onnx-viewer-demo" → run → Artifacts → onnx_viewer/viewer.html
```

---

## API

### `onnxmlflow.log_model(model_path, artifact_dir="onnx_viewer", run_id=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | — | Path to the `.onnx` file on disk |
| `artifact_dir` | `str` | `"onnx_viewer"` | Subdirectory in the run's artifact store |
| `run_id` | `str \| None` | `None` | Target run ID; falls back to `mlflow.active_run()` |

**Returns** the artifact-relative path of the viewer, e.g. `"onnx_viewer/viewer.html"`.

**Raises**
- `FileNotFoundError` — if `model_path` does not exist
- `RuntimeError` — if no MLflow run is active and `run_id` is not provided

---

## Artifact layout

After calling `log_model`:

```
<mlflow-run>/artifacts/
└── onnx_viewer/
    ├── my_model.onnx     ← original model file
    └── viewer.html       ← self-contained interactive viewer
```

---

## How the viewer works

| Step | What happens |
|------|-------------|
| Log time (Python) | ONNX graph I/O metadata is extracted and serialised to JSON. The model bytes are base64-encoded. Both are embedded as JavaScript constants in `viewer.html`. |
| Browser load | The page decodes the base64 bytes and passes them to `ort.InferenceSession.create()` — no network request needed. |
| User fills form | Input fields are generated from the embedded metadata. Each field's type matches the ONNX element type. |
| Click "Run Inference" | ORT runs the session with the collected tensor feeds and displays results. Plain tensors show raw values; `Sequence<Map<>>` probability outputs render as a sorted score table. |

---

## Development

```bash
# Run tests
pytest

# Re-create the test fixture model (requires scikit-learn + skl2onnx)
python tests/create_test_model.py
```

---

## Project structure

```
src/onnxmlflow/
├── __init__.py            Public API
├── logger.py              Validates model, generates HTML, uploads artifacts
├── html_generator.py      Reads ONNX graph metadata → fills viewer template
└── templates/
    └── viewer.html        Self-contained HTML + JavaScript viewer

tests/
├── create_test_model.py   Builds iris_classifier.onnx via skl2onnx
└── test_logger.py         pytest suite

examples/
└── log_iris_model.py      End-to-end demo
```

---

## License

[MIT](LICENSE)
