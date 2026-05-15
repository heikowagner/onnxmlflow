from __future__ import annotations

import base64
import json
import os
from typing import Any

import onnx
from onnx import TensorProto

from .model_utils import make_viewer_model

# Reverse mapping: integer element-type code → ONNX DataType name string
# e.g. {1: 'FLOAT', 7: 'INT64', 8: 'STRING', ...}
_ELEM_TYPE_NAME: dict[int, str] = {v: k for k, v in TensorProto.DataType.items()}

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "viewer.html")


def _type_description(type_proto) -> tuple[str, bool]:
    """Return (human-readable type string, is_plain_tensor).

    is_plain_tensor=False signals that tensor.data should NOT be used in JS.
    """
    if type_proto.HasField("tensor_type"):
        name = _ELEM_TYPE_NAME.get(type_proto.tensor_type.elem_type, "FLOAT")
        return name, True
    if type_proto.HasField("sequence_type"):
        inner, _ = _type_description(type_proto.sequence_type.elem_type)
        return f"Sequence<{inner}>", False
    if type_proto.HasField("map_type"):
        key = _ELEM_TYPE_NAME.get(type_proto.map_type.key_type, "?")
        val, _ = _type_description(type_proto.map_type.value_type)
        return f"Map<{key},{val}>", False
    return "unknown", False


def _extract_tensor_meta(value_info_list) -> list[dict[str, Any]]:
    """Return [{name, type, shape, is_tensor}] for a list of ValueInfoProto entries."""
    result: list[dict[str, Any]] = []
    for vi in value_info_list:
        type_str, is_tensor = _type_description(vi.type)
        shape: list = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append(None)
        result.append({"name": vi.name, "type": type_str, "shape": shape, "is_tensor": is_tensor})
    return result


def generate_viewer_html(
    model: onnx.ModelProto,
    model_filename: str,
    artifact_dir: str,
    run_id: str = "",
    model_path: str = "",
) -> str:
    """Render the HTML viewer template with model metadata injected in-place."""
    # Patch the model for the viewer: remove ZipMap so ORT Web gets plain tensors.
    viewer_model = make_viewer_model(model)

    inputs_meta  = _extract_tensor_meta(viewer_model.graph.input)
    outputs_meta = _extract_tensor_meta(viewer_model.graph.output)
    model_name   = viewer_model.graph.name or os.path.splitext(model_filename)[0]

    # Embed the PATCHED model as base64 — no ZipMap, all outputs are plain tensors.
    model_b64 = base64.b64encode(viewer_model.SerializeToString()).decode("ascii")

    # Extract graph nodes for the JS visualisation (no protobuf parsing needed in browser).
    graph_nodes = [
        {"opType": n.op_type, "name": n.name, "inputs": list(n.input), "outputs": list(n.output)}
        for n in viewer_model.graph.node
    ]

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
        .replace("__GRAPH_NODES_JSON__", json.dumps(graph_nodes))
    )
