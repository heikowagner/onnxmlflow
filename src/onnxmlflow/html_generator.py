from __future__ import annotations

import base64
import json
import os
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper

from .model_utils import make_viewer_model, make_scored_model

# Reverse mapping: integer element-type code → ONNX DataType name string
# e.g. {1: 'FLOAT', 7: 'INT64', 8: 'STRING', ...}
_ELEM_TYPE_NAME: dict[int, str] = {v: k for k, v in TensorProto.DataType.items()}

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "viewer.html")

# ort is optional at runtime — used only for pre-computing intermediate tensor values.
try:
    import onnxruntime as _ort
except ImportError:
    _ort = None  # type: ignore

_ORT_DTYPE: dict[str, Any] = {
    "FLOAT": np.float32, "DOUBLE": np.float64,
    "INT8": np.int8,  "INT16": np.int16,  "INT32": np.int32,  "INT64": np.int64,
    "UINT8": np.uint8, "UINT16": np.uint16, "UINT32": np.uint32, "UINT64": np.uint64,
    "BOOL": np.bool_,
}


def _summarize_array(arr: np.ndarray) -> str:
    """Compact summary matching JS summarizeTensor(), for embedding in HTML."""
    if arr.size == 0:
        return ""
    shape_str = "\u00d7".join(str(d) for d in arr.shape) if arr.ndim > 0 else ""
    prefix = f"[{shape_str}]\u00a0" if shape_str else ""
    is_float = np.issubdtype(arr.dtype, np.floating)
    flat = arr.flat
    n = arr.size
    if n == 1:
        v = arr.flat[0]
        return prefix + (f"{v:.4f}" if is_float else str(v))
    if is_float:
        vals = arr.flatten()
        if n <= 5:
            return prefix + ", ".join(f"{v:.3f}" for v in vals)
        mi = int(np.argmax(vals))
        return prefix + f"argmax={mi} ({vals[mi]:.3f})"
    vals = arr.flatten()
    if n <= 5:
        return prefix + ", ".join(str(int(v)) for v in vals)
    return prefix + ", ".join(str(int(v)) for v in vals[:3]) + "\u2026"


def _run_scored_inference(
    scored_model: onnx.ModelProto,
    inputs_meta: list[dict[str, Any]],
) -> dict[str, str]:
    """Run scored_model with zero inputs via onnxruntime; return {tensor_name: summary}."""
    if _ort is None:
        return {}
    try:
        sess = _ort.InferenceSession(
            scored_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feed: dict[str, np.ndarray] = {}
        for inp in inputs_meta:
            dtype = _ORT_DTYPE.get(inp["type"], np.float32)
            shape = [
                1 if (d is None or not isinstance(d, int) or d <= 0) else d
                for d in (inp["shape"] or [1])
            ]
            feed[inp["name"]] = np.zeros(shape, dtype=dtype)
        out_names = [o.name for o in sess.get_outputs()]
        results = sess.run(out_names, feed)
        return {
            name: _summarize_array(np.asarray(arr))
            for name, arr in zip(out_names, results)
            if not isinstance(arr, list) and _summarize_array(np.asarray(arr))
        }
    except Exception:
        return {}


def _compact_stats(arr: np.ndarray) -> str:
    """Return a short human-readable summary for a weight array."""
    if arr.size == 0:
        return "[]"
    shape_str = "×".join(str(d) for d in arr.shape) if arr.ndim > 0 else "scalar"
    if np.issubdtype(arr.dtype, np.floating):
        return f"[{shape_str}] μ={arr.mean():.3f} σ={arr.std():.3f}"
    return f"[{shape_str}]"

def _format_weights(name: str, arr: np.ndarray, n_classes: int = 0, max_lines: int = 6) -> list[str]:
    """Format a weight array as display lines for the graph (one line per row for 2-D).

    For flat 1-D coefficient arrays, pass n_classes to reshape into (n_classes, n_features).
    """
    if arr.size == 0:
        return []
    is_float = np.issubdtype(arr.dtype, np.floating)

    def fmt(v: Any) -> str:
        return f"{float(v):.3f}" if is_float else str(int(v))

    # Attempt to reshape flat coefficients into (n_classes, n_features).
    # Skip if the array IS the class axis (i.e. size == n_classes → it's the bias vector).
    if arr.ndim == 1 and n_classes > 1 and arr.size % n_classes == 0 and arr.size != n_classes:
        arr = arr.reshape(n_classes, arr.size // n_classes)

    if arr.ndim <= 1:
        flat = arr.flatten()
        if flat.size <= 8:
            return [f"{name}: [{', '.join(fmt(v) for v in flat)}]"]
        return [f"{name}[{flat.size}]: \u03bc={float(flat.mean()):.3f} \u03c3={float(flat.std()):.3f}"]

    # 2-D: one line per row, labelled by class index
    lines: list[str] = []
    for i, row in enumerate(arr):
        if len(lines) >= max_lines:
            lines.append(f"  \u2026 ({arr.shape[0] - max_lines} more)")
            break
        if row.size <= 8:
            lines.append(f"{name}[{i}]: [{', '.join(fmt(v) for v in row)}]")
        else:
            lines.append(f"{name}[{i}]: \u03bc={float(row.mean()):.3f} \u03c3={float(row.std()):.3f}")
    return lines

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
    groups: dict | None = None,
    output_groups: dict | None = None,
) -> str:
    """Render the HTML viewer template with model metadata injected in-place."""
    # Patch the model for the viewer: remove ZipMap so ORT Web gets plain tensors.
    viewer_model = make_viewer_model(model)

    inputs_meta  = _extract_tensor_meta(viewer_model.graph.input)
    outputs_meta = _extract_tensor_meta(viewer_model.graph.output)
    model_name   = viewer_model.graph.name or os.path.splitext(model_filename)[0]

    # Embed the PATCHED model as base64 — no ZipMap, all outputs are plain tensors.
    model_b64 = base64.b64encode(viewer_model.SerializeToString()).decode("ascii")

    # Build a lookup of graph initializer arrays and their compact summaries.
    init_arrays: dict[str, np.ndarray] = {}
    init_map: dict[str, str] = {}
    for init in viewer_model.graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
            init_arrays[init.name] = arr
            init_map[init.name] = _compact_stats(arr)
        except Exception:
            pass

    # Extract graph nodes for the JS visualisation.
    # Include a `params` list of compact weight/attribute summaries per node.
    graph_nodes = []
    for n in viewer_model.graph.node:
        node_dict: dict[str, Any] = {
            "opType": n.op_type,
            "name": n.name,
            "inputs": list(n.input),
            "outputs": list(n.output),
        }

        params: list[str] = []
        weights: list[str] = []

        # Collect all float tensors for this node: initializer inputs + attributes.
        # Used together so n_classes inference works across both sources.
        all_floats: dict[str, np.ndarray] = {}

        # 1. Initializer (weight tensor) inputs — compact stats in params, full matrix in weights.
        for inp in n.input:
            if inp in init_arrays:
                arr = init_arrays[inp]
                if arr.dtype.kind != "f" and arr.dtype.kind != "i":
                    continue
                # params.append(f"{inp}: {init_map[inp]}")
                all_floats[inp] = arr.astype(np.float32)


        # 2. Alle Attribute als params aufnehmen (auch String, Int, Maps, Constant)
        for attr in n.attribute:
            try:
                if attr.type == AttributeProto.FLOATS and len(attr.floats) > 0:
                    params.append(f"{attr.name}: [{', '.join(f'{v:.3f}' for v in attr.floats)}]")
                elif attr.type == AttributeProto.INTS and len(attr.ints) > 0:
                    params.append(f"{attr.name}: [{', '.join(str(v) for v in attr.ints)}]")
                elif attr.type == AttributeProto.STRINGS and len(attr.strings) > 0:
                    params.append(f"{attr.name}: [{', '.join(s.decode('utf-8','replace') for s in attr.strings)}]")
                elif attr.type == AttributeProto.STRING and attr.s:
                    params.append(f"{attr.name}: '{attr.s.decode('utf-8','replace')}'")
                elif attr.type == AttributeProto.INT and attr.i is not None:
                    params.append(f"{attr.name}: {attr.i}")
                elif attr.type == AttributeProto.FLOAT and attr.f is not None:
                    params.append(f"{attr.name}: {attr.f:.3f}")
                elif attr.type == AttributeProto.TENSOR and attr.t.ByteSize() > 0:
                    t = numpy_helper.to_array(attr.t)
                    params.append(f"{attr.name}: {t.tolist()}")
            except Exception:
                pass

        # Infer number of classes / output features from bias-like tensors.

        # n_classes nur aus Bias-Vektor bestimmen (wie vorher)
        n_classes = 0
        for bias_name in ("intercepts", "intercept", "bias", "B"):
            if bias_name in all_floats:
                cand = all_floats[bias_name]
                if cand.ndim == 1:
                    n_classes = int(cand.size)
                    break

        # Format weight lines — works for both attribute floats and initializer matrices.
        MAX_WEIGHT_LINES = 8
        for name, arr in all_floats.items():
            if arr.size <= 1:
                continue
            lines = _format_weights(name, arr, n_classes=n_classes)
            weights.extend(lines)
            if len(weights) >= MAX_WEIGHT_LINES:
                weights = weights[:MAX_WEIGHT_LINES]
                break

        if params:
            node_dict["params"] = params[:4]
        if weights:
            node_dict["weights"] = weights[:MAX_WEIGHT_LINES]
        graph_nodes.append(node_dict)

    # Scored model: viewer model with all intermediate tensors exposed so the
    # browser can capture per-node activation values after running inference.
    scored_model = make_scored_model(viewer_model)
    scored_model_b64 = base64.b64encode(scored_model.SerializeToString()).decode("ascii")

    # Pre-compute intermediate tensor values in Python (zero inputs) for immediate
    # graph annotation — independent of whether the browser scored session works.
    initial_tensor_values = _run_scored_inference(scored_model, inputs_meta)

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
        .replace("__GRAPH_GROUPS_JSON__", json.dumps(groups or {}))
        .replace("__GRAPH_OUTPUT_GROUPS_JSON__", json.dumps(output_groups or {}))
        .replace("__SCORED_MODEL_B64__", scored_model_b64)
        .replace("__INITIAL_TENSOR_VALUES_JSON__", json.dumps(initial_tensor_values))
    )
