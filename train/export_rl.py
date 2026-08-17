# -*- coding: utf-8 -*-
"""导出 RL 检查点 (.pt) → 部署态 numpy 权重 (rl_net.npz)。

键名与 agent/rl_policy.load_net 一一对应；导出后立即做一次 torch vs numpy
前向一致性抽查（同一随机局面，logits 最大绝对差 < 1e-4 才算导出成功）。
用法: .venv/bin/python train/export_rl.py train/rl_s87_ship1.pt agent/rl_net.npz
"""
import sys, os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "train"))
os.environ.setdefault("CABT_FEAT", "v9")
import rl_selfplay as R  # noqa: E402

pt = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "agent", "rl_net.npz")


class RLNet(nn.Module):
    def __init__(self, d=R.D):
        super().__init__()
        self.emb = nn.EmbeddingBag(R.N_FEAT, d, mode="sum")
        self.sfc = nn.Linear(d + R.N_SCALAR, R.H)
        self.card = nn.Embedding(R.CARD_N, 32)
        self.atk = nn.Embedding(R.ATK_N, 16)
        self.otype = nn.Embedding(R.N_TYPE, 16)
        self.h1 = nn.Linear(R.H + 64, R.H)
        self.out = nn.Linear(R.H, 1)
        self.vout = nn.Linear(R.H, 1)


net = RLNet()
net.load_state_dict(torch.load(pt, map_location="cpu"))
W = R.torch_to_np(net)
np.savez_compressed(out, **W)

# ---- torch vs numpy 一致性抽查 ----
rng = np.random.default_rng(0)
idx = rng.integers(0, R.N_FEAT, 40).tolist()
val = rng.random(40).astype(np.float32).tolist()
sc = rng.random(R.N_SCALAR).astype(np.float32).tolist()
ofs = [(int(rng.integers(0, R.N_TYPE)), int(rng.integers(0, R.CARD_N)), int(rng.integers(0, R.ATK_N)))
       for _ in range(7)]

Wr = {k: np.load(out)[k] for k in np.load(out).files}
Wr["vout_b"] = float(Wr["vout_b"])
np_logits, _ = R.np_forward(Wr, idx, val, sc, ofs)

with torch.no_grad():
    ti = torch.tensor(idx); tv = torch.tensor(val); tsc = torch.tensor(sc)
    e = (net.emb.weight[ti] * tv[:, None]).sum(0)
    srep = torch.relu(net.sfc(torch.cat([e, tsc])))
    tt = torch.tensor([o[0] for o in ofs]); tc = torch.tensor([o[1] for o in ofs])
    ta = torch.tensor([o[2] for o in ofs])
    of = torch.cat([net.otype(tt), net.card(tc), net.atk(ta)], dim=1)
    h = torch.relu(net.h1(torch.cat([srep.expand(len(ofs), -1), of], dim=1)))
    t_logits = net.out(h).squeeze(1).numpy()

diff = float(np.abs(np_logits - t_logits).max())
assert diff < 1e-4, f"torch/numpy 前向不一致 max|Δ|={diff}"
print(f"exported {out} ({os.path.getsize(out)/1e6:.1f} MB)  前向一致性 max|Δ|={diff:.2e} OK")
