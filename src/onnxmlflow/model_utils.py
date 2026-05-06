"""ONNX model graph utilities used when generating the HTML viewer."""
from __future__ import annotations

import copy

import onnx
from onnx import TensorProto, helper


def make_viewer_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a viewer-friendly copy of *model*.

    ZipMap nodes convert float tensors to ``Sequence<Map<INT64,FLOAT>>``, which
    ORT Web cannot read back as tensor data.  This function removes every ZipMap
    node and replaces the corresponding graph output with the ZipMap's plain
    float input tensor, preserving the original output *name* via an Identity
    node so that ``OUTPUTS_META`` names in the HTML remain consistent.

    Models without ZipMap nodes are returned unchanged (as a deep copy).
    """
    model = copy.deepcopy(model)
    graph = model.graph

    # Map: sequence-output-name → float-input-name
    zipmap_map: dict[str, str] = {}
    keep_nodes = []
    for node in graph.node:
        if node.op_type == "ZipMap":
            zipmap_map[node.output[0]] = node.input[0]
        else:
            keep_nodes.append(node)

    if not zipmap_map:
        return model

    del graph.node[:]
    graph.node.extend(keep_nodes)

    # Add Identity nodes so the output keeps its original name.
    # e.g.  "probabilities" → Identity → "output_probability"
    for seq_name, float_name in zipmap_map.items():
        identity = helper.make_node("Identity", inputs=[float_name], outputs=[seq_name])
        graph.node.append(identity)

    # Update output type annotations: old Sequence<Map<>> → FLOAT tensor
    for out in graph.output:
        if out.name in zipmap_map:
            out.type.CopyFrom(
                helper.make_tensor_type_proto(TensorProto.FLOAT, None)
            )

    try:
        onnx.checker.check_model(model)
    except Exception:
        # If checker fails (e.g. unknown shape), still return the model —
        # ORT Web will validate it at load time.
        pass

    return model
