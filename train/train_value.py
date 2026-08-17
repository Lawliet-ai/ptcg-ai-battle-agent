"""Train value-net v1: state -> P(win). Data from selfplay_gen shards.

Split train/val BY GAME (g%10==0 -> val) so correlated states from one game never leak
across the split. Reports val accuracy/AUC vs. a ported hand-written-eval baseline —
the net must beat the baseline before it earns a slot in the tree.

Usage: python train/train_value.py --data train/data/shard0.jsonl.gz --epochs 4
Saves train/value_net.pt (torch) — runtime export comes later once it proves itself.
"""
import sys, os, gzip, json, argparse, random, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os
if _os.environ.get("CABT_FEAT") == "v10":
    from features import featurize_v10 as featurize, N_FEAT_V9 as N_FEAT, N_SCALAR_V10 as N_SCALAR
    print("[feat] v10 xiaoqi-manual eyes: N_FEAT=%d N_SCALAR=%d" % (N_FEAT, N_SCALAR))
elif _os.environ.get("CABT_FEAT") == "v93":
    from features import featurize_v93 as featurize, N_FEAT_V9 as N_FEAT, N_SCALAR_V93 as N_SCALAR
    print("[feat] v9.3 prize-exposure eyes: N_FEAT=%d N_SCALAR=%d" % (N_FEAT, N_SCALAR))
elif _os.environ.get("CABT_FEAT") == "v9":
    from features import featurize_v9 as featurize, N_FEAT_V9 as N_FEAT, N_SCALAR_V9 as N_SCALAR
    print("[feat] v9 eyes: N_FEAT=%d N_SCALAR=%d" % (N_FEAT, N_SCALAR))
else:
    from features import featurize, N_FEAT, N_SCALAR


class ValueNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_FEAT, d, mode="sum")
        self.fc1 = nn.Linear(d + N_SCALAR, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, idx, off, val, sc):
        e = self.emb(idx, off, per_sample_weights=val)
        h = torch.relu(self.fc1(torch.cat([e, sc], dim=1)))
        h = torch.relu(self.fc2(h))
        return self.out(h).squeeze(1)


def load(paths, max_samples, soft=False, hand_mask=0.0):
    """-> lists of (idx, val, scalars, label, baseline_v), split train/val by game id.
    `paths` is comma-separated; game ids are offset per shard so splits stay clean.
    soft: label = 0.5*z + 0.5*(rv mapped to [0,1]) when the sample carries a search value.
    hand_mask: 以此概率把样本我方hand置None(保留handCount) —— 推理时46.9-58.6%的树叶子
    落在对手回合帧(hand=None), 而产料100%手牌可见; 不增广则这些帧全部分布外
    (猎捕案#5: 网络把'手牌不可见'读成'手牌打空'的伪濒死信号)。掩码只动hand,
    featurize的handCount兜底(features.py v9路径)会接管标量, 嵌入块自然置空 —— 与推理形态逐位一致。"""
    rng = random.Random(20260805)
    tr, va = [], []
    n = 0
    for si, path in enumerate(paths.split(",")):
        with gzip.open(path.strip(), "rt") as fh:
            for line in fh:
                if n >= max_samples:
                    break
                s = json.loads(line)
                cur, me = s["cur"], s["me"]
                if hand_mask > 0 and rng.random() < hand_mask:
                    p = cur["players"][me]
                    if p.get("hand") is not None:
                        p = dict(p, handCount=p.get("handCount", len(p["hand"] or [])), hand=None)
                        cur = dict(cur, players=[p if i == me else q for i, q in enumerate(cur["players"])])
                idx, val, sc = featurize(cur, me)
                b = baseline(cur, me)
                z = float(s["z"])
                if soft and "rv" in s:
                    z = 0.5 * z + 0.5 * (float(s["rv"]) + 1.0) / 2.0
                rec = (idx, val, sc, z, b)
                g = s["g"] + si * 1000003          # offset so shards never collide
                (va if g % 10 == 0 else tr).append(rec)
                n += 1
    return tr, va


def baseline(cur, me):
    """Port of the hand-written positional eval (prize race + material; no KO-threat
    term — that needs attack tables). Good enough to rank states for an AUC baseline."""
    my = cur["players"][me]; op = cur["players"][1 - me]
    v = (len(op.get("prize") or []) - len(my.get("prize") or [])) / 6.0
    tb = 0.0
    ma = (my.get("active") or [None])[0]; oa = (op.get("active") or [None])[0]
    if ma is None: tb -= 0.06
    if oa is None: tb += 0.06
    def hp_sum(p, a):
        return (a["hp"] if a else 0) + sum(b["hp"] for b in (p.get("bench") or []) if b)
    mh, oh = hp_sum(my, ma), hp_sum(op, oa)
    if mh + oh > 0: tb += 0.06 * (mh - oh) / (mh + oh)
    def e_sum(p, a):
        t = len(a.get("energyCards") or []) if a else 0
        return t + sum(len(b.get("energyCards") or []) for b in (p.get("bench") or []) if b)
    me_, oe = e_sum(my, ma), e_sum(op, oa)
    if me_ + oe > 0: tb += 0.04 * (me_ - oe) / (me_ + oe)
    return v + max(-0.16, min(0.16, tb))


def batches(recs, bs, shuffle):
    order = list(range(len(recs)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [recs[j] for j in order[i:i + bs]]
        idx, off, val, pos = [], [], [], 0
        for r in chunk:
            off.append(pos); idx += r[0]; val += r[1]; pos += len(r[0])
        yield (torch.tensor(idx, dtype=torch.long),
               torch.tensor(off, dtype=torch.long),
               torch.tensor(val, dtype=torch.float32),
               torch.tensor([r[2] for r in chunk], dtype=torch.float32),
               torch.tensor([float(r[3]) for r in chunk], dtype=torch.float32))


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data/shard0.jsonl.gz")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-samples", type=int, default=900000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--init", default="", help="warm-start from this checkpoint (league fine-tune)")
    ap.add_argument("--soft", action="store_true",
                    help="AlphaZero-style label: 0.5*outcome + 0.5*root search value (rv)")
    ap.add_argument("--hand-mask", type=float, default=0.0,
                    help="以此概率把样本我方hand置None(对齐推理时对手回合帧分布), 建议0.5")
    ap.add_argument("--out", default="train/value_net.pt")
    a = ap.parse_args()

    t0 = time.time()
    tr, va = load(a.data, a.max_samples, soft=a.soft, hand_mask=a.hand_mask)
    print(f"loaded {len(tr)} train / {len(va)} val in {time.time()-t0:.0f}s soft={a.soft} "
          f"hand_mask={a.hand_mask}", flush=True)

    base_scores = np.array([r[4] for r in va]); labels = (np.array([r[3] for r in va]) > 0.5).astype(float)
    print(f"BASELINE(hand-eval port): val AUC {auc(base_scores, labels):.4f}  "
          f"acc {(np.sign(base_scores) == np.sign(labels*2-1)).mean():.4f}", flush=True)

    model = ValueNet(d=a.d)
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        cur = model.state_dict()
        # 扩列热启动:v9.3 在标量段末尾追加了新特征,fc1 的输入宽度随之变大。
        # 旧权重按列复制到新矩阵的前缀,新列置0 —— 起点与旧网逐位等价,新特征从零学起,
        # 保住 v9.2b 全部知识(K线那次直接 load_state_dict 撞维度崩掉,就是缺这一步)。
        grown = []
        for k, v in ck.items():
            if k in cur and cur[k].shape != v.shape:
                if cur[k].dim() == 2 and cur[k].shape[0] == v.shape[0] and cur[k].shape[1] > v.shape[1]:
                    w = torch.zeros_like(cur[k]); w[:, :v.shape[1]] = v
                    ck[k] = w; grown.append(f"{k}{tuple(v.shape)}->{tuple(w.shape)}")
                else:
                    raise SystemExit(f"[init] 形状不兼容且无法扩列: {k} {tuple(v.shape)} vs {tuple(cur[k].shape)}")
        model.load_state_dict(ck)
        print(f"warm-started from {a.init}" + (f" (扩列: {', '.join(grown)})" if grown else ""), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        model.train(); tl = nb = 0
        for idx, off, val, sc, z in batches(tr, a.bs, True):
            opt.zero_grad()
            loss = lossf(model(idx, off, val, sc), z)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval(); preds = []
        with torch.no_grad():
            for idx, off, val, sc, z in batches(va, a.bs, False):
                preds.append(torch.sigmoid(model(idx, off, val, sc)).numpy())
        p = np.concatenate(preds)
        print(f"ep{ep+1}: train_loss {tl/nb:.4f}  val AUC {auc(p, labels):.4f}  "
              f"acc {((p > .5) == (labels == 1)).mean():.4f}", flush=True)
        torch.save(model.state_dict(), a.out)
    print(f"saved {a.out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
