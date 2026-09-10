#!/usr/bin/env python3
import math
import pickle

import numpy as np
import onnx
import onnxruntime as ort
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
    """Retype every non-float graph input and output to float32.

    TensorRT binds this planner through one float32 path, but upstream declares
    four inputs int64 and one output int32. Each of those keeps its own dtype
    inside the graph: the boundary tensor becomes float32 and a Cast named
    wrap_<tensor> converts it back, so nothing but the edge of the graph moves.
    """
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
    """Reopen a one-row graph boundary as a dynamic batch of rows.

    MimicLite exports a single robot: `command` [304] and `policy` [535] in,
    `action` [29] out, with no batch axis at all. The harness drives the whole
    field in one call, so every boundary tensor gains a leading dynamic axis.
    The body is left alone -- every node in the graph is a MatMul, an Add, a
    LayerNormalization over the last axis, a Mish, or a Concat on axis -1, and
    all of those already broadcast over a leading batch.
    """
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


if __name__ == "__main__":
    main()
