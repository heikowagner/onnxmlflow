from __future__ import annotations

import base64
import json
import os
from typing import Any

import onnx
from onnx import TensorProto

# Reverse mapping: integer element-type code → ONNX DataType name string
# e.g. {1: 'FLOAT', 7: 'INT64', 8: 'STRING', ...}
_ELEM_TYPE_NAME: dict[int, str] = {v: k for k, v in TensorProto.DataType.items()}

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "viewer.html")


def _extract_tensor_meta(value_info_list) -> list[dict[str, Any]]:
    """Return [{name, type, shape}] for a list of ValueInfoProto entries."""
    result: list[dict[str, Any]] = []
    for vi in value_info_list:
        entry: dict[str, Any] = {"name": vi.name, "type": "FLOAT", "shape": []}
        t = vi.type
        if t.HasField("tensor_type"):
            entry["type"] = _ELEM_TYPE_NAME.get(t.tensor_type.elem_type, "FLOAT")
            if t.tensor_type.HasField("shape"):
                for dim in t.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        entry["shape"].append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        # symbolic dim like "batch_size" → keep as string
                        entry["shape"].append(dim.dim_param)
                    else:
                        entry["shape"].append(None)
        result.append(entry)
    return result


def generate_viewer_html(
    model: onnx.ModelProto,
    model_filename: str,
    artifact_dir: str,
    run_id: str = "",
    model_path: str = "",
) -> str:
    """Render the HTML viewer template with model metadata injected in-place."""
    inputs_meta = _extract_tensor_meta(model.graph.input)
    outputs_meta = _extract_tensor_meta(model.graph.output)
    model_name = model.graph.name or os.path.splitext(model_filename)[0]

    # Embed model as base64 so the viewer works inside MLflow's sandboxed iframe
    # without needing a fetch() call.
    if model_path and os.path.isfile(model_path):
        with open(model_path, "rb") as fh:
            model_b64 = base64.b64encode(fh.read()).decode("ascii")
    else:
        model_b64 = base64.b64encode(model.SerializeToString()).decode("ascii")

    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as fh:
        template = fh.read()

    return (
        template
        .replace("__MODEL_FILENAME__", model_filename)
        .replace("__MODEL_ARTIFACT_DIR__", artifact_dir)
        .replace("__MODEL_NAME__", model_name)
        .replace("__RUN_ID__", run_id)
        .replace("__MODEL_B64__", model_b64)
        .replace("__INPUTS_META_JSON__", json.dumps(inputs_meta))
        .replace("__OUTPUTS_META_JSON__", json.dumps(outputs_meta))
    )
