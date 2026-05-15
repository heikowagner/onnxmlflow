"""ONNX model graph utilities used when generating the HTML viewer."""
from __future__ import annotations

import copy

import onnx
import onnx.shape_inference
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


def make_scored_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of *model* with all intermediate tensors exposed as graph outputs.

    Used by the HTML viewer's scored session to capture per-node activation
    values during inference and display them on the graph.
    """
    model = copy.deepcopy(model)

    # Use shape inference to obtain type/shape info for intermediate tensors.
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        type_map = {vi.name: vi.type for vi in inferred.graph.value_info}
    except Exception:
        type_map = {}

    graph = model.graph
    declared_outputs = {o.name for o in graph.output}
    known_initializers = {init.name for init in graph.initializer}

    for node in graph.node:
        for out_name in node.output:
            if out_name and out_name not in declared_outputs and out_name not in known_initializers:
                if out_name not in type_map:
                    continue  # skip: ORT Web rejects outputs with no type annotation
                vi = onnx.ValueInfoProto()
                vi.name = out_name
                vi.type.CopyFrom(type_map[out_name])
                graph.output.append(vi)
                declared_outputs.add(out_name)

    try:
        onnx.checker.check_model(model)
    except Exception:
        pass

    return model
