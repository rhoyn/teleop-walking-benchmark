#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

OPSET = 17
TOL = 1e-4
STEPS = 8


class RlGym(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.LSTM(47, 64, 1)
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))

    def forward(self, x, h, c):
        out, (hn, cn) = self.memory(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class RoboMimic(nn.Module):
    def __init__(self, eps):
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

    def forward(self, x, h, c):
        x = (x - self.mean) / (self.std + self.eps)
        out, (hn, cn) = self.rnn(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "export_onnx: a CUDA device is required; amo was traced on one "
            "and holds constants pinned to cuda:0"
        )


def agree(tag, ref, got):
    worst = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    if worst > TOL:
        raise RuntimeError(
            f"export_onnx: {tag} disagrees with its source by {worst:.3e}, "
            f"over the {TOL:.0e} tolerance"
        )


def onnx_run(path, names, tensors):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(
        None, {n: t.detach().cpu().numpy() for n, t in zip(names, tensors)}
    )


def export_stateless(src, dst, example, in_names, out_names):
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


def export_recurrent(src, dst, net, obs_dim, hidden, drop, rename):
    original = torch.jit.load(src, map_location="cpu")
    original.eval()
    state = original.state_dict()

    weights = {k: v for k, v in state.items() if k not in drop}
    for source, target in rename.items():
        weights[target] = state[source]
    net.load_state_dict(weights)
    net.eval()

    with torch.no_grad():
        original.hidden_state.zero_()
        original.cell_state.zero_()
        h = torch.zeros(1, 1, hidden)
        c = torch.zeros(1, 1, hidden)
        for _ in range(STEPS):
            x = torch.randn(1, obs_dim)
            ref = original(x)
            got, h, c = net(x, h, c)
            agree(f"{dst} rebuild", [ref.numpy()], [got.numpy()])

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


if __name__ == "__main__":
    main()
