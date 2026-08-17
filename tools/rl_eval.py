# -*- coding: utf-8 -*-
"""RL 检查点的判决级评测：贪心(temp=0)对九族真实 field，样本量足够到能下结论。

为什么单独写：训练循环里那个每 25 轮的贪心评测只有 120 局(±9pp)，用来看趋势都嫌抖，
更不能当过闸判据。本脚本默认 600 局(±4pp)，并给 Wilson 置信区间。
对照锚：v1 的树在同一把尺子(arena_eval 九族、opp-pilot 启发式)上是 79.1%。

用法:
  CABT_FEAT=v9 .venv/bin/python tools/rl_eval.py train/rl_s87.pt --games 600 --workers 10
"""
import sys, os, math, argparse
import multiprocessing as mp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "train"))
os.environ.setdefault("CABT_FEAT", "v9")

import rl_selfplay as R


def wilson(w, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="检查点路径, 或字面量 tree=用v1冠军树打锚")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--deck", default=os.path.join(ROOT, "agent", "deck_stardom87.csv"))
    ap.add_argument("--gauntlet", default=os.path.join(ROOT, "gauntlet_train0805"))
    ap.add_argument("--mimic-dirs", default="", help="逗号分隔真实bot目录; 给了就测bot池(mimic_frac=1)")
    a = ap.parse_args()

    import torch
    if a.ckpt == "tree":
        W = "tree"
    elif not os.path.exists(a.ckpt):
        sys.exit(f"!! 检查点不存在: {a.ckpt}（静默回退会产出假结论，直接中止）")

    # 与训练同构的网络定义(仅为读权重)
    import torch.nn as nn

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

    if a.ckpt != "tree":
        net = RLNet()
        net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
        W = R.torch_to_np(net)

    deck = [int(x) for x in open(a.deck) if x.strip()]
    opps = sorted(os.path.join(a.gauntlet, f) for f in os.listdir(a.gauntlet) if f.endswith(".csv"))
    mimics = []
    for d in [x for x in a.mimic_dirs.split(",") if x.strip()]:
        mimics.append((os.path.abspath(os.path.join(d, "main.py")),
                       [int(x) for x in open(os.path.join(d, "deck.csv")) if x.strip()]))
    mf = 1.0 if mimics else 0.0
    per = max(1, a.games // a.workers)
    with mp.Pool(a.workers, initializer=R._init, initargs=(deck, opps, mimics)) as pool:
        # 关镜像; mimics 给了则全部对局打真实bot池
        out = pool.map(R._play, [(W, 7000 + w, per, a.temp, False, mf) for w in range(a.workers)])
    w = sum(o[1] for o in out); n = sum(o[2] for o in out)
    lo, hi = wilson(w, n)
    mde = 1.96 * math.sqrt(2 * 0.25 / max(n, 1)) * 100
    print(f"{a.ckpt}  temp={a.temp}")
    print(f"  纯九族陪练 {100*w/max(n,1):.1f}%  ({w}-{n-w})  Wilson95 [{lo:.1f}, {hi:.1f}]  MDE≈±{mde:.1f}pp")
    print(f"  对照: v1 的树同尺 79.1%  |  随机MAIN+圣经层 ≈26.7%")


if __name__ == "__main__":
    main()
