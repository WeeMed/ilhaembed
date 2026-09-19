"""Standard-operator INT8 quantization with bounded activation and vocabulary error."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def quantize_embedding_rows(path, vocab_size, hidden_size):
    """Quantize each vocabulary row independently using standard Gather/Cast/Mul ops."""
    model = onnx.load(path)
    tensors = {t.name: t for t in model.graph.initializer}
    matches = [
        n
        for n in model.graph.node
        if n.op_type == "Gather"
        and n.input[0] in tensors
        and list(tensors[n.input[0]].dims) == [vocab_size, hidden_size]
    ]
    names = {n.input[0] for n in matches}
    if len(names) != 1:
        raise ValueError("Expected one explicit vocabulary embedding table")
    name = next(iter(names))
    weights = numpy_helper.to_array(tensors[name]).astype(np.float32)
    scales = np.maximum(np.max(np.abs(weights), axis=1) / 127.0, 1e-9).astype(
        np.float32
    )
    quantized = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(np.int8)
    qname = name + "_row_int8"
    sname = name + "_row_scale"
    aname = name + "_scale_axis"
    retained = [t for t in model.graph.initializer if t.name != name]
    retained.extend(
        [
            numpy_helper.from_array(quantized, qname),
            numpy_helper.from_array(scales, sname),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), aname),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Gather" and node.input[0] == name:
            out = node.output[0]
            ids = node.input[1]
            nodes.extend(
                [
                    helper.make_node("Gather", [qname, ids], [out + "_int8"], axis=0),
                    helper.make_node(
                        "Cast", [out + "_int8"], [out + "_float"], to=TensorProto.FLOAT
                    ),
                    helper.make_node("Gather", [sname, ids], [out + "_scales"], axis=0),
                    helper.make_node(
                        "Unsqueeze", [out + "_scales", aname], [out + "_scale3d"]
                    ),
                    helper.make_node("Mul", [out + "_float", out + "_scale3d"], [out]),
                ]
            )
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def quantize_tokenwise_activations(path):
    """Keep INT8 matmuls, with one symmetric activation scale per token row."""
    model = onnx.load(path)
    constants = {
        "tw_axes": np.array([-1], dtype=np.int64),
        "tw_qmax": np.array(127, dtype=np.float32),
        "tw_eps": np.array(1e-9, dtype=np.float32),
        "tw_offset": np.array(128, dtype=np.float32),
        "tw_low": np.array(0, dtype=np.float32),
        "tw_high": np.array(255, dtype=np.float32),
    }
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in constants.items()
    )
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "DynamicQuantizeLinear":
            nodes.append(node)
            continue
        x = node.input[0]
        q, scale, zero = node.output
        prefix = q + "_tw"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(128, dtype=np.uint8), zero)
        )
        nodes.extend(
            [
                helper.make_node("Abs", [x], [prefix + "_abs"]),
                helper.make_node(
                    "ReduceMax",
                    [prefix + "_abs", "tw_axes"],
                    [prefix + "_max"],
                    keepdims=1,
                ),
                helper.make_node(
                    "Div", [prefix + "_max", "tw_qmax"], [prefix + "_scale"]
                ),
                helper.make_node("Max", [prefix + "_scale", "tw_eps"], [scale]),
                helper.make_node("Div", [x, scale], [prefix + "_scaled"]),
                helper.make_node("Round", [prefix + "_scaled"], [prefix + "_rounded"]),
                helper.make_node(
                    "Add", [prefix + "_rounded", "tw_offset"], [prefix + "_offset"]
                ),
                helper.make_node(
                    "Clip",
                    [prefix + "_offset", "tw_low", "tw_high"],
                    [prefix + "_clipped"],
                ),
                helper.make_node(
                    "Cast", [prefix + "_clipped"], [q], to=TensorProto.UINT8
                ),
            ]
        )
        count += 1
    if not count:
        raise ValueError("No dynamic activation quantizers were transformed")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Scalar scale annotations are invalid after changing to per-token scales.
    del model.graph.value_info[:]
    for item in [
        model,
        model.graph,
        *model.graph.node,
        *model.graph.input,
        *model.graph.output,
        *model.graph.initializer,
    ]:
        item.doc_string = ""
        if hasattr(item, "metadata_props"):
            item.ClearField("metadata_props")
    onnx.checker.check_model(model)
    onnx.save_model(model, path)
    return count
