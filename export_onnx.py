#!/usr/bin/env python3
import math
import pickle
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import torch
import torch.nn as nn

OPSET = 17
TOL = 1e-4
STEPS = 8
Z_DIM = 256
Z_NORM_TOL = 1e-2
SONIC_CTX = 4
SONIC_QPOS = 36
SONIC_HEIGHT = 0.788740
SONIC_ENC_OBS = 1762
ROWS_TOL = 0.5
ROWS_CHECK = 16

BATCH_ELEMENTWISE = {
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "PRelu",
    "Sub",
    "Where",
    "Xor",
}
BATCH_UNARY = {
    "Abs",
    "Acos",
    "Asin",
    "Atan",
    "Cast",
    "Ceil",
    "Clip",
    "Cos",
    "Dropout",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "IsInf",
    "IsNaN",
    "LeakyRelu",
    "Log",
    "Mish",
    "Neg",
    "Not",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Sigmoid",
    "Sign",
    "Sin",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
    "Trilu",
}
BATCH_REDUCE = {
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
}
BATCH_AXIS_DEFAULT = {
    "ArgMax": 0,
    "ArgMin": 0,
    "LayerNormalization": -1,
    "LogSoftmax": -1,
    "OneHot": -1,
    "Softmax": -1,
    "Split": 0,
    "TopK": -1,
}
BATCH_EINSUM = re.compile(
    r"\.\.\. (\w) (\w) (\w), \.\.\. \1 \2 (\w) -> \.\.\. \1 \3 \4"
)

BFM_ZERO_LATENTS = [
    "move-ego-0-0",
    "move-ego-0-0.3",
    "move-ego-0-0.7",
    "move-ego-90-0.3",
    "move-ego-90-0.7",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "move-ego--90-0.7",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "move-ego-low0.5-0-0",
    "move-ego-low0.6-0-0.7",
]


class RlGym(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.memory = nn.LSTM(47, 64, 1)
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))

    def forward(
        self,
        x,
        h,
        c,
    ):
        out, (hn, cn) = self.memory(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class RoboMimic(nn.Module):
    def __init__(
        self,
        eps,
    ):
        super().__init__()
        self.rnn = nn.LSTM(96, 256, 1)
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        self.register_buffer("mean", torch.zeros(1, 96))
        self.register_buffer("std", torch.ones(1, 96))
        self.eps = eps

    def forward(
        self,
        x,
        h,
        c,
    ):
        x = (x - self.mean) / (self.std + self.eps)
        out, (hn, cn) = self.rnn(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class G1Gym(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.memory = nn.LSTM(90, 64, 1)
        self.actor = nn.Sequential(
            nn.Linear(64, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 27),
        )

    def forward(
        self,
        x,
        h,
        c,
    ):
        out, (hn, cn) = self.memory(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class Mturan33Loco(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        layers = []
        prev = 57
        for h in (512, 256, 128):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 12))
        self.actor = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        return self.actor(x)


class SunnyFolded(nn.Module):
    def __init__(
        self,
        mean,
        std,
        w0,
        b0,
        rest,
    ):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        layers = [nn.Linear(99, 512), nn.ELU()]
        layers[0].weight = nn.Parameter(w0)
        layers[0].bias = nn.Parameter(b0)
        for w, b in rest:
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            layers[-1].weight = nn.Parameter(w)
            layers[-1].bias = nn.Parameter(b)
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers[:-1])

    def forward(
        self,
        x,
    ):
        return self.net(torch.clamp((x - self.mean) / self.std, -100.0, 100.0))


def export_sunny(
    src,
    scan_path,
    dst,
):
    model = onnx.load(src)
    init = {
        i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy())
        for i in model.graph.initializer
    }
    mean, std = init["obs_mean"], init["onnx::Div_71"]

    scan = torch.from_numpy(np.load(scan_path).astype(np.float32))
    norm = torch.clamp((scan - mean[99:]) / std[99:], -100.0, 100.0)
    if float(norm.abs().max()) > TOL:
        raise RuntimeError(
            "export_onnx: sunny's shipped scan is not the graph's own mean, "
            "so the scan branch is not constant and cannot be folded"
        )

    cnn = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ELU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ELU(),
        nn.Flatten(),
        nn.Linear(4576, 64),
        nn.ELU(),
        nn.Linear(64, 32),
        nn.ELU(),
    )
    cnn[0].weight = nn.Parameter(init["scan_cnn.0.weight"])
    cnn[0].bias = nn.Parameter(init["scan_cnn.0.bias"])
    cnn[2].weight = nn.Parameter(init["scan_cnn.2.weight"])
    cnn[2].bias = nn.Parameter(init["scan_cnn.2.bias"])
    cnn[5].weight = nn.Parameter(init["scan_cnn.5.weight"])
    cnn[5].bias = nn.Parameter(init["scan_cnn.5.bias"])
    cnn[7].weight = nn.Parameter(init["scan_cnn.7.weight"])
    cnn[7].bias = nn.Parameter(init["scan_cnn.7.bias"])
    cnn.eval()
    with torch.no_grad():
        feat = cnn(norm.view(1, 1, 13, 11))[0]

    w0 = init["policy_net.0.weight"]
    net = SunnyFolded(
        mean[:99],
        std[:99],
        w0[:, :99].clone(),
        init["policy_net.0.bias"] + w0[:, 99:] @ feat,
        [
            (init["policy_net.2.weight"], init["policy_net.2.bias"]),
            (init["policy_net.4.weight"], init["policy_net.4.bias"]),
            (init["action_net.weight"], init["action_net.bias"]),
        ],
    )
    net.eval()

    example = (torch.randn(4, 99),)
    with torch.no_grad():
        ref = net(*example)

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )

    full = torch.cat([example[0], scan.expand(example[0].shape[0], -1)], dim=1)
    agree(dst, onnx_run(src, ["obs"], (full,)), onnx_run(dst, ["obs"], example))


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "export_onnx: a CUDA device is required; amo was traced on one "
            "and holds constants pinned to cuda:0"
        )


def agree(
    tag,
    ref,
    got,
):
    worst = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    if worst > TOL:
        raise RuntimeError(
            f"export_onnx: {tag} disagrees with its source by {worst:.3e}, "
            f"over the {TOL:.0e} tolerance"
        )


def onnx_run(
    path,
    names,
    tensors,
):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(
        None, {n: t.detach().cpu().numpy() for n, t in zip(names, tensors)}
    )


def export_stateless(
    src,
    dst,
    example,
    in_names,
    out_names,
):
    module = torch.jit.load(src, map_location="cuda")
    module.eval()
    example = [t.to("cuda") for t in example]
    with torch.no_grad():
        ref = module(*example)
    ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)

    torch.onnx.export(
        module,
        tuple(example),
        dst,
        input_names=in_names,
        output_names=out_names,
        opset_version=OPSET,
        dynamo=False,
    )
    agree(
        dst, [r.detach().cpu().numpy() for r in ref], onnx_run(dst, in_names, example)
    )


def export_submodule(
    src,
    dst,
    net,
    prefix,
    obs_dim,
):
    state = torch.load(src, map_location="cpu", weights_only=False)["model"]
    wanted = set(net.state_dict())
    weights = {
        k[len(prefix) :]: v
        for k, v in state.items()
        if k.startswith(prefix) and k[len(prefix) :] in wanted
    }
    net.load_state_dict(weights)
    net.eval()

    example = (torch.randn(1, obs_dim),)
    with torch.no_grad():
        ref = net(*example)

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, [ref.numpy()], onnx_run(dst, ["obs"], example))


def export_recurrent(
    src,
    dst,
    net,
    obs_dim,
    hidden,
    drop,
    rename,
    carries_state=True,
):
    original = torch.jit.load(src, map_location="cpu")
    original.eval()
    state = original.state_dict()

    weights = {k: v for k, v in state.items() if k not in drop}
    for source, target in rename.items():
        weights[target] = state[source]
    net.load_state_dict(weights)
    net.eval()

    with torch.no_grad():
        if carries_state:
            original.hidden_state.zero_()
            original.cell_state.zero_()
        h = torch.zeros(1, 1, hidden)
        c = torch.zeros(1, 1, hidden)
        for _ in range(STEPS):
            x = torch.randn(1, obs_dim)
            ref = original(x)
            got, hn, cn = net(x, h, c)
            agree(f"{dst} rebuild", [ref.numpy()], [got.numpy()])
            if carries_state:
                h, c = hn, cn

    example = (
        torch.randn(1, obs_dim),
        torch.zeros(1, 1, hidden),
        torch.zeros(1, 1, hidden),
    )
    in_names = ["obs", "hidden_in", "cell_in"]
    out_names = ["action", "hidden_out", "cell_out"]
    with torch.no_grad():
        ref = list(net(*example))

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=in_names,
        output_names=out_names,
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, [r.detach().numpy() for r in ref], onnx_run(dst, in_names, example))


def export_float_io(
    src,
    dst,
    example,
):
    model = onnx.load(src)
    graph = model.graph
    wrapped = {}

    for value in graph.input:
        elem = value.type.tensor_type.elem_type
        if elem == onnx.TensorProto.FLOAT:
            continue
        inner = value.name + "__i"
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == value.name:
                    node.input[i] = inner
        graph.node.insert(
            0,
            onnx.helper.make_node(
                "Cast", [value.name], [inner], name="wrap_" + value.name, to=elem
            ),
        )
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        wrapped[value.name] = elem

    for value in graph.output:
        elem = value.type.tensor_type.elem_type
        if elem == onnx.TensorProto.FLOAT:
            continue
        inner = value.name + "__i"
        for node in graph.node:
            for i, name in enumerate(node.output):
                if name == value.name:
                    node.output[i] = inner
            for i, name in enumerate(node.input):
                if name == value.name:
                    node.input[i] = inner
        graph.node.append(
            onnx.helper.make_node(
                "Cast",
                [inner],
                [value.name],
                name="wrap_" + value.name,
                to=onnx.TensorProto.FLOAT,
            )
        )
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        wrapped[value.name] = elem

    if not wrapped:
        raise RuntimeError(f"export_onnx: {src} is already float32 at the boundary")

    missing = sorted({v.name for v in graph.input} - set(example))
    if missing:
        raise RuntimeError(f"export_onnx: {src} has no example input for {missing}")

    onnx.save(model, dst)

    source = {
        name: example[name].astype(onnx.helper.tensor_dtype_to_np_dtype(wrapped[name]))
        if name in wrapped
        else example[name]
        for name in example
    }
    session = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    ref = [np.asarray(r, np.float32) for r in session.run(None, source)]
    session = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    agree(dst, ref, session.run(None, example))


def export_batched(
    src,
    dst,
    batch=STEPS,
):
    model = onnx.load(src)
    graph = model.graph

    rows = {}
    for value in list(graph.input) + list(graph.output):
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if len(shape) != 1 or shape[0] < 1:
            raise RuntimeError(
                f"export_onnx: {src} boundary '{value.name}' is {shape}, not one row"
            )
        rows[value.name] = shape
        value.CopyFrom(
            onnx.helper.make_tensor_value_info(
                value.name,
                value.type.tensor_type.elem_type,
                ["batch"] + shape,
            )
        )
    del graph.value_info[:]
    onnx.save(model, dst)

    rng = np.random.default_rng(0)
    feed = {
        value.name: rng.standard_normal((batch, *rows[value.name])).astype(np.float32)
        for value in graph.input
    }
    one = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    many = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    per_row = [one.run(None, {n: v[r] for n, v in feed.items()}) for r in range(batch)]
    ref = [
        np.stack([per_row[r][k] for r in range(batch)]) for k in range(len(per_row[0]))
    ]
    agree(dst, ref, many.run(None, feed))


def record_rows(
    src,
    feed,
):
    model = onnx.shape_inference.infer_shapes(onnx.load(src), strict_mode=False)
    graph = model.graph
    elem = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    known = {o.name for o in graph.output}
    for node in graph.node:
        for name in node.output:
            if name and name not in known and name in elem:
                graph.output.append(
                    onnx.helper.make_tensor_value_info(name, elem[name], None)
                )
                known.add(name)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, (np.asarray(v) for v in session.run(None, feed))))


class Batcher:
    def __init__(
        self,
        src,
        feed,
    ):
        self.model = onnx.load(src)
        self.graph = self.model.graph
        self.rec = record_rows(src, feed)
        for init in self.graph.initializer:
            self.rec[init.name] = numpy_helper.to_array(init)
        for value in self.graph.input:
            self.rec[value.name] = np.asarray(feed[value.name])
        self.weights = {init.name for init in self.graph.initializer}
        self.folded = set(self.weights)
        self.varying = {value.name for value in self.graph.input}
        self.emitted = set()
        self.nodes = []
        self.inits = []
        self.count = 0
        self.ref = self.graph.input[0].name

    def fresh(
        self,
        tag,
    ):
        self.count += 1
        return f"rows_{tag}_{self.count}"

    def rank(
        self,
        name,
    ):
        return self.rec[name].ndim

    def shape(
        self,
        name,
    ):
        return list(self.rec[name].shape)

    def add(
        self,
        op,
        inputs,
        outputs=None,
        **attrs,
    ):
        if outputs is None:
            outputs = [self.fresh(op.lower())]
        self.nodes.append(onnx.helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs

    def tensor(
        self,
        value,
        tag="c",
    ):
        name = self.fresh(tag)
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def i64(
        self,
        values,
        tag="i",
    ):
        return self.tensor(np.asarray(values, dtype=np.int64), tag)

    def unsqueeze(
        self,
        x,
        axes,
    ):
        if not axes:
            return x
        return self.add("Unsqueeze", [x, self.i64(list(axes), "axes")])

    def pad_rank(
        self,
        x,
        now,
        want,
    ):
        return self.unsqueeze(x, list(range(1, 1 + want - now)))

    def batch_dim(self):
        return self.add(
            "Slice",
            [
                self.add("Shape", [self.ref]),
                self.i64([0], "start"),
                self.i64([1], "end"),
            ],
        )

    def const(
        self,
        name,
    ):
        if name in self.weights or name in self.emitted:
            return name
        self.inits.append(numpy_helper.from_array(self.rec[name], name))
        self.emitted.add(name)
        return name

    def arg(
        self,
        name,
    ):
        return name if name in self.varying else self.const(name)

    def rows(
        self,
        name,
    ):
        if name in self.varying:
            return name
        value = self.rec[name]
        target = self.batch_dim()
        if value.ndim:
            target = self.add(
                "Concat", [target, self.i64(list(value.shape), "shape")], axis=0
            )
        return self.add("Expand", [self.unsqueeze(self.const(name), [0]), target])

    @staticmethod
    def axis(
        axis,
        rank,
    ):
        return axis + rank if axis < 0 else axis

    @staticmethod
    def attr(
        node,
        key,
        default=None,
    ):
        for a in node.attribute:
            if a.name == key:
                return onnx.helper.get_attribute_value(a)
        return default

    @staticmethod
    def attrs(
        node,
        *skip,
    ):
        return {
            a.name: onnx.helper.get_attribute_value(a)
            for a in node.attribute
            if a.name not in skip
        }

    def run(
        self,
        dst,
    ):
        graph = self.graph
        for value in list(graph.input) + list(graph.output):
            if self.shape(value.name)[:1] != [1]:
                raise RuntimeError(f"export_onnx: {value.name} is not one row")
        for node in graph.node:
            inputs = [i for i in node.input if i]
            folded = node.op_type in {"Shape", "Size", "Constant"} or all(
                i in self.folded for i in inputs
            )
            if folded:
                self.folded.update(o for o in node.output if o)
                continue
            self.varying.update(o for o in node.output if o)
            self.rewrite(node)
        inputs, prologue, outputs, epilogue = [], [], [], []
        for value in graph.input:
            inner = value.name + "__row"
            for node in self.nodes:
                for k, name in enumerate(node.input):
                    if name == value.name:
                        node.input[k] = inner
            prologue.append(
                onnx.helper.make_node(
                    "Unsqueeze", [value.name, self.i64([1], "axes")], [inner]
                )
            )
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    value.name,
                    value.type.tensor_type.elem_type,
                    ["batch"] + self.shape(value.name)[1:],
                )
            )
        for value in graph.output:
            inner = value.name + "__row"
            for node in self.nodes:
                for k, name in enumerate(node.output):
                    if name == value.name:
                        node.output[k] = inner
            epilogue.append(
                onnx.helper.make_node(
                    "Squeeze", [inner, self.i64([1], "axes")], [value.name]
                )
            )
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    value.name,
                    value.type.tensor_type.elem_type,
                    ["batch"] + self.shape(value.name)[1:],
                )
            )
        used = set()
        for node in self.nodes:
            used.update(node.input)
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                prologue + self.nodes + epilogue,
                graph.name,
                inputs,
                outputs,
                initializer=[i for i in graph.initializer if i.name in used]
                + self.inits,
            ),
            opset_imports=self.model.opset_import,
        )
        model.ir_version = self.model.ir_version
        onnx.checker.check_model(model)
        onnx.save(model, dst)

    def rewrite(
        self,
        node,
    ):
        op = node.op_type
        if op in BATCH_ELEMENTWISE:
            return self.elementwise(node)
        if op in BATCH_UNARY:
            return self.add(
                op,
                [self.arg(i) if i else "" for i in node.input],
                list(node.output),
                **self.attrs(node),
            )
        if op in BATCH_REDUCE:
            return self.reduce(node)
        if op in BATCH_AXIS_DEFAULT:
            return self.axis_op(node, BATCH_AXIS_DEFAULT[op])
        handler = getattr(self, "op_" + op, None)
        if handler is None:
            raise RuntimeError(f"export_onnx: no batched form of {op} ({node.name})")
        return handler(node)

    def elementwise(
        self,
        node,
    ):
        out = node.output[0]
        want = self.rank(out)
        args = [
            self.pad_rank(i, self.rank(i), want) if i in self.varying else self.const(i)
            for i in node.input
        ]
        self.add(node.op_type, args, [out], **self.attrs(node))

    def reduce(
        self,
        node,
    ):
        x = node.input[0]
        rank = self.rank(x)
        axes = self.attr(node, "axes")
        if axes is None:
            axes = list(range(rank))
        self.add(
            node.op_type,
            [self.arg(x)],
            list(node.output),
            axes=[self.axis(a, rank) + 1 for a in axes],
            **self.attrs(node, "axes"),
        )

    def axis_op(
        self,
        node,
        default,
    ):
        anchor = node.output[0] if node.op_type == "OneHot" else node.input[0]
        axis = self.axis(self.attr(node, "axis", default), self.rank(anchor)) + 1
        self.add(
            node.op_type,
            [self.arg(i) if i else "" for i in node.input],
            list(node.output),
            axis=axis,
            **self.attrs(node, "axis"),
        )

    def op_ReduceSum(
        self,
        node,
    ):
        x = node.input[0]
        rank = self.rank(x)
        if len(node.input) > 1 and node.input[1]:
            axes = self.rec[node.input[1]].tolist()
        else:
            axes = list(range(rank))
        self.add(
            "ReduceSum",
            [
                self.arg(x),
                self.i64([self.axis(int(a), rank) + 1 for a in axes], "axes"),
            ],
            list(node.output),
            **self.attrs(node),
        )

    def op_MatMul(
        self,
        node,
    ):
        a, b = node.input
        out = node.output[0]
        ra, rb = self.rank(a), self.rank(b)
        big = max(self.rank(out), 2)
        squeeze = []
        if a in self.varying:
            if ra == 1:
                a = self.pad_rank(self.unsqueeze(a, [1]), 2, big)
                squeeze.append(big - 1)
            else:
                a = self.pad_rank(a, ra, big)
        else:
            value = self.rec[a]
            if ra == 1:
                value = value.reshape(1, -1)
                squeeze.append(big - 1)
            a = self.tensor(np.ascontiguousarray(value), "w")
        if b in self.varying:
            if rb == 1:
                b = self.pad_rank(self.unsqueeze(b, [2]), 2, big)
                squeeze.append(big)
            else:
                b = self.pad_rank(b, rb, big)
        else:
            value = self.rec[b]
            if rb == 1:
                value = value.reshape(-1, 1)
                squeeze.append(big)
            b = self.tensor(np.ascontiguousarray(value), "w")
        y = self.add("MatMul", [a, b])
        if squeeze:
            self.add("Squeeze", [y, self.i64(squeeze, "axes")], [out])
        else:
            self.add("Identity", [y], [out])

    def op_Gemm(
        self,
        node,
    ):
        a, b = node.input[0], node.input[1]
        c = node.input[2] if len(node.input) > 2 else ""
        if b in self.varying or c in self.varying or self.attr(node, "transA", 0):
            raise RuntimeError(
                f"export_onnx: Gemm {node.name} is not a row times weights"
            )
        w = self.rec[b]
        if self.attr(node, "transB", 0):
            w = w.T
        w = w * self.attr(node, "alpha", 1.0)
        y = self.add("MatMul", [a, self.tensor(np.ascontiguousarray(w), "w")])
        if c:
            bias = self.rec[c] * self.attr(node, "beta", 1.0)
            self.add(
                "Add",
                [y, self.tensor(np.ascontiguousarray(bias), "b")],
                [node.output[0]],
            )
        else:
            self.add("Identity", [y], [node.output[0]])

    def op_Einsum(
        self,
        node,
    ):
        equation = self.attr(node, "equation").decode()
        if BATCH_EINSUM.fullmatch(equation) is None:
            raise RuntimeError(f"export_onnx: Einsum {equation} has no batched form")
        a, b = node.input
        want = self.rank(node.output[0])
        ra = want + 1 if a in self.varying else self.rank(a)
        a = self.pad_rank(a, self.rank(a), want) if a in self.varying else self.const(a)
        b = self.pad_rank(b, self.rank(b), want) if b in self.varying else self.const(b)
        at = self.add("Transpose", [a], perm=list(range(ra - 2)) + [ra - 1, ra - 2])
        self.add("MatMul", [at, b], [node.output[0]])

    def op_Reshape(
        self,
        node,
    ):
        data, shape = node.input
        if shape in self.varying:
            raise RuntimeError(
                f"export_onnx: Reshape {node.name} takes a shape per row"
            )
        out = node.output[0]
        self.add(
            "Reshape",
            [self.arg(data), self.i64([-1] + self.shape(out), "shape")],
            [out],
        )

    def op_Flatten(
        self,
        node,
    ):
        out = node.output[0]
        self.add(
            "Reshape",
            [self.arg(node.input[0]), self.i64([-1] + self.shape(out), "shape")],
            [out],
        )

    def op_Transpose(
        self,
        node,
    ):
        x = node.input[0]
        perm = self.attr(node, "perm") or list(range(self.rank(x)))[::-1]
        self.add(
            "Transpose",
            [self.arg(x)],
            list(node.output),
            perm=[0] + [p + 1 for p in perm],
        )

    def op_Unsqueeze(
        self,
        node,
    ):
        x, axes = node.input
        rank = self.rank(node.output[0])
        axes = [self.axis(int(a), rank) + 1 for a in self.rec[axes].tolist()]
        self.add("Unsqueeze", [self.arg(x), self.i64(axes, "axes")], list(node.output))

    def op_Squeeze(
        self,
        node,
    ):
        x = node.input[0]
        rank = self.rank(x)
        if len(node.input) > 1 and node.input[1]:
            axes = [self.axis(int(a), rank) for a in self.rec[node.input[1]].tolist()]
        else:
            axes = [k for k, d in enumerate(self.shape(x)) if d == 1]
        self.add(
            "Squeeze",
            [self.arg(x), self.i64([a + 1 for a in axes], "axes")],
            list(node.output),
        )

    def op_Concat(
        self,
        node,
    ):
        out = node.output[0]
        axis = self.axis(self.attr(node, "axis"), self.rank(out)) + 1
        self.add("Concat", [self.rows(i) for i in node.input], [out], axis=axis)

    def op_Slice(
        self,
        node,
    ):
        data, *rest = node.input
        if any(r in self.varying for r in rest):
            raise RuntimeError(f"export_onnx: Slice {node.name} takes bounds per row")
        rank = self.rank(data)
        if len(rest) > 2 and rest[2]:
            axes = [self.axis(int(a), rank) + 1 for a in self.rec[rest[2]].tolist()]
        else:
            axes = list(range(1, 1 + self.rec[rest[0]].size))
        args = [
            self.arg(data),
            self.const(rest[0]),
            self.const(rest[1]),
            self.i64(axes, "axes"),
        ]
        if len(rest) > 3 and rest[3]:
            args.append(self.const(rest[3]))
        self.add("Slice", args, list(node.output))

    def op_Gather(
        self,
        node,
    ):
        data, idx = node.input
        out = node.output[0]
        rd, ri = self.rank(data), self.rank(idx)
        axis = self.axis(self.attr(node, "axis", 0), rd)
        if idx not in self.varying:
            self.add("Gather", [data, self.const(idx)], [out], axis=axis + 1)
            return
        if data not in self.varying:
            value = self.rec[data]
            if axis:
                value = np.transpose(
                    value, [axis] + [k for k in range(rd) if k != axis]
                )
            y = self.add(
                "Gather", [self.tensor(np.ascontiguousarray(value), "g"), idx], axis=0
            )
        else:
            if axis:
                data = self.add(
                    "Transpose",
                    [data],
                    perm=[0, axis + 1] + [k + 1 for k in range(rd) if k != axis],
                )
            y = self.add(
                "GatherND", [data, self.unsqueeze(idx, [ri + 1])], batch_dims=1
            )
        if not axis:
            self.add("Identity", [y], [out])
            return
        perm = (
            [0]
            + [1 + ri + j for j in range(axis)]
            + list(range(1, 1 + ri))
            + [1 + ri + j for j in range(axis, rd - 1)]
        )
        self.add("Transpose", [y], [out], perm=perm)

    def op_GatherElements(
        self,
        node,
    ):
        data, idx = node.input
        axis = self.axis(self.attr(node, "axis", 0), self.rank(data)) + 1
        self.add(
            "GatherElements",
            [self.rows(data), self.rows(idx)],
            list(node.output),
            axis=axis,
        )

    def op_ScatterElements(
        self,
        node,
    ):
        axis = self.axis(self.attr(node, "axis", 0), self.rank(node.input[0])) + 1
        self.add(
            "ScatterElements",
            [self.rows(i) for i in node.input],
            list(node.output),
            axis=axis,
            **self.attrs(node, "axis"),
        )

    def op_ScatterND(
        self,
        node,
    ):
        data, idx, upd = node.input
        lead = self.shape(idx)[:-1]
        batch = self.batch_dim()
        span = self.add(
            "Range",
            [
                self.i64(0, "zero"),
                self.add("Squeeze", [batch, self.i64([0], "axes")]),
                self.i64(1, "one"),
            ],
        )
        span = self.add(
            "Reshape", [span, self.i64([-1] + [1] * (len(lead) + 1), "shape")]
        )
        row = self.add(
            "Expand",
            [span, self.add("Concat", [batch, self.i64(lead + [1], "shape")], axis=0)],
        )
        full = self.add("Concat", [row, self.rows(idx)], axis=len(lead) + 1)
        self.add(
            "ScatterND",
            [self.rows(data), full, self.rows(upd)],
            list(node.output),
            **self.attrs(node),
        )

    def op_Expand(
        self,
        node,
    ):
        x, shape = node.input
        if shape in self.varying:
            raise RuntimeError(f"export_onnx: Expand {node.name} takes a shape per row")
        out = node.output[0]
        want = self.rank(out)
        target = [int(s) for s in self.rec[shape].tolist()]
        target = [1] * (1 + want - len(target)) + target
        x = self.pad_rank(x, self.rank(x), want) if x in self.varying else self.rows(x)
        self.add("Expand", [x, self.i64(target, "shape")], [out])

    def op_Tile(
        self,
        node,
    ):
        x, reps = node.input
        if reps in self.varying:
            raise RuntimeError(f"export_onnx: Tile {node.name} takes repeats per row")
        reps = [1] + [int(v) for v in self.rec[reps].tolist()]
        self.add("Tile", [self.arg(x), self.i64(reps, "reps")], list(node.output))

    def op_CumSum(
        self,
        node,
    ):
        x, axis = node.input
        axis = self.axis(int(self.rec[axis]), self.rank(x)) + 1
        self.add(
            "CumSum",
            [self.arg(x), self.i64(axis, "axis")],
            list(node.output),
            **self.attrs(node),
        )

    def op_Conv(
        self,
        node,
    ):
        x = node.input[0]
        out = node.output[0]
        flat = self.add(
            "Reshape", [self.arg(x), self.i64([-1] + self.shape(x)[1:], "shape")]
        )
        y = self.add(
            "Conv", [flat] + [self.const(i) for i in node.input[1:]], **self.attrs(node)
        )
        self.add("Reshape", [y, self.i64([-1] + self.shape(out), "shape")], [out])

    def op_Pad(
        self,
        node,
    ):
        x, pads, *rest = node.input
        if pads in self.varying:
            raise RuntimeError(f"export_onnx: Pad {node.name} takes pads per row")
        pads = [int(v) for v in self.rec[pads].tolist()]
        half = len(pads) // 2
        args = [self.arg(x), self.i64([0] + pads[:half] + [0] + pads[half:], "pads")]
        if rest and rest[0]:
            args.append(self.const(rest[0]))
        self.add("Pad", args, list(node.output), **self.attrs(node))

    def op_Resize(
        self,
        node,
    ):
        x, _, scales, *rest = list(node.input) + ["", ""]
        if not scales or scales in self.varying or rest[0]:
            raise RuntimeError(
                f"export_onnx: Resize {node.name} without constant scales"
            )
        scales = [1.0] + [float(v) for v in self.rec[scales].tolist()]
        self.add(
            "Resize",
            [self.arg(x), "", self.tensor(np.asarray(scales, np.float32), "scales")],
            list(node.output),
            **self.attrs(node),
        )


def export_rows(
    src,
    dst,
    feed,
    rows=ROWS_CHECK,
):
    Batcher(src, feed(1, 0)).run(dst)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    one = ort.InferenceSession(src, options, providers=["CPUExecutionProvider"])
    many = ort.InferenceSession(dst, options, providers=["CPUExecutionProvider"])
    mixed = feed(rows, 1)
    got = many.run(None, mixed)
    for r in range(rows):
        alone = {k: np.repeat(v[r : r + 1], rows, axis=0) for k, v in mixed.items()}
        same = many.run(None, alone)
        for k, (a, b) in enumerate(zip(got, same)):
            if not all(np.array_equal(b[0], b[j]) for j in range(rows)):
                raise RuntimeError(
                    f"export_onnx: {dst} output {k} differs between equal rows"
                )
            if not np.array_equal(a[r], b[r]):
                raise RuntimeError(
                    f"export_onnx: {dst} row {r} depends on its neighbours"
                )
        ref = one.run(None, {k: v[r : r + 1] for k, v in mixed.items()})
        worst = max(float(np.abs(x[0] - y[r]).max()) for x, y in zip(ref, got))
        if worst > ROWS_TOL:
            raise RuntimeError(f"export_onnx: {dst} row {r} is off by {worst:.3e}")


def sonic_planner_feed(
    rows,
    seed,
):
    rng = np.random.default_rng(seed)
    context = np.zeros((rows, SONIC_CTX, SONIC_QPOS), np.float32)
    context[:, :, :2] = rng.standard_normal((rows, SONIC_CTX, 2)) * 0.3
    context[:, :, 2] = SONIC_HEIGHT
    context[:, :, 3] = 1.0
    context[:, :, 7:] = rng.standard_normal((rows, SONIC_CTX, 29)) * 0.1
    allowed = np.zeros((rows, 11), np.float32)
    allowed[:, :6] = 1.0
    return {
        "context_mujoco_qpos": context,
        "mode": rng.integers(0, 2, rows).astype(np.float32),
        "target_vel": rng.uniform(0.0, 1.0, rows).astype(np.float32),
        "movement_direction": rng.standard_normal((rows, 3)).astype(np.float32),
        "facing_direction": rng.standard_normal((rows, 3)).astype(np.float32),
        "random_seed": rng.integers(0, 5000, rows).astype(np.float32),
        "height": np.full(rows, -1.0, np.float32),
        "has_specific_target": np.zeros((rows, 1), np.float32),
        "specific_target_positions": np.zeros((rows, SONIC_CTX, 3), np.float32),
        "specific_target_headings": np.zeros((rows, SONIC_CTX), np.float32),
        "allowed_pred_num_tokens": allowed,
    }


def sonic_encoder_feed(
    rows,
    seed,
):
    rng = np.random.default_rng(seed)
    return {"obs_dict": rng.standard_normal((rows, SONIC_ENC_OBS)).astype(np.float32)}


def export_latents(
    src,
    dst,
    names,
):
    with open(src, "rb") as handle:
        table = pickle.load(handle)

    want = math.sqrt(Z_DIM)
    ref, rows = [], []
    for name in names:
        if name not in table:
            raise RuntimeError(f"export_onnx: {src} has no latent named '{name}'")
        z = table[name]
        while isinstance(z, (list, tuple)):
            z = z[0]
        z = torch.as_tensor(z).detach().reshape(-1).float()
        if z.numel() != Z_DIM:
            raise RuntimeError(
                f"export_onnx: latent '{name}' is {z.numel()} long, expected {Z_DIM}"
            )
        norm = float(z.norm())
        if abs(norm - want) > Z_NORM_TOL:
            raise RuntimeError(
                f"export_onnx: latent '{name}' has norm {norm:.6f}, expected {want:.6f}"
            )
        ref.append(z.numpy())
        rows.append(name + "," + ",".join(f"{v:.9g}" for v in z.tolist()))

    header = "name," + ",".join(f"z{i}" for i in range(Z_DIM))
    with open(dst, "w") as handle:
        handle.write(header + "\n" + "\n".join(rows) + "\n")

    got, back = [], []
    with open(dst) as handle:
        for line in handle:
            cells = line.strip().split(",")
            if not cells or cells[0] == "name":
                continue
            back.append(cells[0])
            got.append(np.asarray(cells[1:], dtype=np.float32))
    if back != list(names):
        raise RuntimeError(f"export_onnx: {dst} did not round-trip its names")
    agree(dst, ref, got)


def main():
    require_cuda()
    torch.manual_seed(0)

    export_stateless(
        "policies/amo/model_adapter.pt",
        "policies/amo/model_adapter.onnx",
        [torch.randn(1, 12)],
        ["input"],
        ["output"],
    )
    export_stateless(
        "policies/amo/model.pt",
        "policies/amo/model.onnx",
        [torch.randn(1, 1043), torch.randn(1, 2325)],
        ["obs_teacher", "extra_hist"],
        ["output"],
    )
    export_recurrent(
        "policies/rl_gym/model.pt",
        "policies/rl_gym/model.onnx",
        RlGym(),
        47,
        64,
        {"hidden_state", "cell_state"},
        {},
    )

    normalizer = torch.jit.load(
        "policies/robomimic/model.pt", map_location="cpu"
    ).normalizer
    export_recurrent(
        "policies/robomimic/model.pt",
        "policies/robomimic/model.onnx",
        RoboMimic(float(normalizer.eps)),
        96,
        256,
        {
            "hidden_state",
            "cell_state",
            "normalizer._mean",
            "normalizer._var",
            "normalizer._std",
            "normalizer.count",
        },
        {"normalizer._mean": "mean", "normalizer._std": "std"},
    )

    export_recurrent(
        "policies/g1_gym/model.pt",
        "policies/g1_gym/model.onnx",
        G1Gym(),
        90,
        64,
        {"hidden_state", "cell_state"},
        {},
    )

    export_recurrent(
        "policies/stepdown/model.pt",
        "policies/stepdown/model.onnx",
        RlGym(),
        47,
        64,
        set(),
        {},
        carries_state=False,
    )

    export_sunny(
        "policies/sunny/model_raw.onnx",
        "policies/sunny/scan_mean.npy",
        "policies/sunny/model.onnx",
    )

    export_submodule(
        "policies/mturan33/model.pt",
        "policies/mturan33/model.onnx",
        Mturan33Loco(),
        "loco_actor.",
        57,
    )

    export_latents(
        "policies/bfm_zero/reward_locomotion.pkl",
        "policies/bfm_zero/latents.csv",
        BFM_ZERO_LATENTS,
    )

    export_batched(
        "policies/mimic_lite/model_roa.onnx",
        "policies/mimic_lite/model.onnx",
    )

    context = np.zeros((1, SONIC_CTX, SONIC_QPOS), np.float32)
    context[:, :, 2] = SONIC_HEIGHT
    context[:, :, 3] = 1.0
    allowed = np.zeros((1, 11), np.float32)
    allowed[0, :6] = 1.0
    export_float_io(
        "policies/sonic/planner_sonic.onnx",
        "policies/sonic/planner_sonic_f32.onnx",
        {
            "context_mujoco_qpos": context,
            "mode": np.zeros(1, np.float32),
            "target_vel": np.full(1, 0.8, np.float32),
            "movement_direction": np.array([[1.0, 0.0, 0.0]], np.float32),
            "facing_direction": np.array([[1.0, 0.0, 0.0]], np.float32),
            "random_seed": np.full(1, 1234.0, np.float32),
            "height": np.full(1, -1.0, np.float32),
            "has_specific_target": np.zeros((1, 1), np.float32),
            "specific_target_positions": np.zeros((1, SONIC_CTX, 3), np.float32),
            "specific_target_headings": np.zeros((1, SONIC_CTX), np.float32),
            "allowed_pred_num_tokens": allowed,
        },
    )
    export_rows(
        "policies/sonic/planner_sonic_f32.onnx",
        "policies/sonic/planner_sonic_rows.onnx",
        sonic_planner_feed,
    )
    export_rows(
        "policies/sonic/model_encoder.onnx",
        "policies/sonic/model_encoder_rows.onnx",
        sonic_encoder_feed,
    )


if __name__ == "__main__":
    main()
