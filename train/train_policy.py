"""Train the policy head ("选招头"): imitate which MAIN action the deep tree picked.
Purpose: replace the hand-written heuristic softmax as the tree's move-ordering prior —
a learned prior focuses search on tree-approved moves, deepening effective search.

Data: selfplay_tree_gen shards with opts/pick. Split train/val BY GAME.
Metrics: val top-1 hit rate vs (a) random 1/n, (b) the heuristic's own top-1 (how often
policy._score_main's argmax equals the tree's pick — the incumbent prior's hit rate).

Usage: python train/train_policy.py --data train/data/tree_pol1.jsonl.gz --epochs 4
"""
import sys, os, gzip, json, argparse, random, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
import os as _os
if _os.environ.get("CABT_FEAT") == "v9":
    from features import featurize_v9 as featurize, N_FEAT_V9 as N_FEAT, N_SCALAR_V9 as N_SCALAR, CARD_N
    print("[feat] v9 eyes (policy)")
else:
    from features import featurize, N_FEAT, N_SCALAR, CARD_N

N_TYPE = 16
ATK_N = 600            # > max attackId


def opt_feat(o, cur, me):
    """(type, cardId, attackId) resolved: PLAY/ATTACH reference the HAND by index."""
    t, cid, aid, idx = (o + [None] * 4)[:4]
    t = t if isinstance(t, int) and 0 <= t < N_TYPE else 0
    if t in (7, 8) and cid is None and isinstance(idx, int):   # PLAY/ATTACH via hand index
        hand = cur["players"][me].get("hand") or []
        if 0 <= idx < len(hand):
            cid = hand[idx]["id"]
    cid = cid if isinstance(cid, int) and 0 <= cid < CARD_N else 0
    aid = aid if isinstance(aid, int) and 0 <= aid < ATK_N else 0
    return t, cid, aid


class PolicyNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_FEAT, d, mode="sum")     # state side (ValueNet-style)
        self.sfc = nn.Linear(d + N_SCALAR, 96)
        self.card = nn.Embedding(CARD_N, 32)
        self.atk = nn.Embedding(ATK_N, 16)
        self.otype = nn.Embedding(N_TYPE, 16)   # NB: can't be named .type (shadows torch)
        self.h1 = nn.Linear(96 + 64, 96)
        self.out = nn.Linear(96, 1)

    def score(self, srep, t, c, a):
        of = torch.cat([self.otype(t), self.card(c), self.atk(a)], dim=1)
        h = torch.relu(self.h1(torch.cat([srep, of], dim=1)))
        return self.out(h).squeeze(1)

    def state_repr(self, idx, off, val, sc):
        e = self.emb(idx, off, per_sample_weights=val)
        return torch.relu(self.sfc(torch.cat([e, sc], dim=1)))


SOFT = _os.environ.get("CABT_POLICY_SOFT", "1") != "0"   # 学教师分布(默认开)


def load(paths, max_samples):
    tr, va = [], []
    n = 0
    for si, p in enumerate(paths.split(",")):
        with gzip.open(p.strip(), "rt") as fh:
            for line in fh:
                if n >= max_samples:
                    break
                s = json.loads(line)
                opts = s.get("opts") or []
                if len(opts) < 2:
                    continue
                cur, me = s["cur"], s["me"]
                sidx, sval, sc = featurize(cur, me)
                ofs = [opt_feat(o, cur, me) for o in opts]
                # 教师的判断=访问分布(85%样本有),不只是argmax动作。
                # 硬标签只说"选3",分布说"3值55%、0值44%、其余是垃圾"——后者才是判断。
                vis = s.get("vis")
                pi = None
                if vis and SOFT:
                    v = [float(vis.get(str(k), 0.0)) for k in range(len(opts))]
                    tot = sum(v)
                    if tot > 0:
                        pi = [x / tot for x in v]
                rec = (sidx, sval, sc, ofs, s["pick"], pi)
                g = s["g"] + si * 1000003
                (va if g % 10 == 0 else tr).append(rec)
                n += 1
    return tr, va


def batches(recs, bs, shuffle):
    order = list(range(len(recs)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [recs[j] for j in order[i:i + bs]]
        yield chunk


def run_batch(model, chunk):
    """Flatten (state × its options) into one scoring pass; softmax per state."""
    idx, off, val, pos = [], [], [], 0
    for r in chunk:
        off.append(pos); idx += r[0]; val += r[1]; pos += len(r[0])
    srep = model.state_repr(torch.tensor(idx, dtype=torch.long),
                            torch.tensor(off, dtype=torch.long),
                            torch.tensor(val, dtype=torch.float32),
                            torch.tensor([r[2] for r in chunk], dtype=torch.float32))
    rows, ts, cs, as_ = [], [], [], []
    for ri, r in enumerate(chunk):
        for (t, c, a) in r[3]:
            rows.append(ri); ts.append(t); cs.append(c); as_.append(a)
    scores = model.score(srep[torch.tensor(rows)],
                         torch.tensor(ts), torch.tensor(cs), torch.tensor(as_))
    # per-state cross entropy over its own options
    loss = 0.0
    hits = 0
    p0 = 0
    for ri, r in enumerate(chunk):
        k = len(r[3])
        sl = scores[p0:p0 + k]
        if len(r) > 5 and r[5] is not None:
            # AlphaZero式策略蒸馏:对齐教师访问分布(含次优项的相对价值)
            tgt_pi = torch.tensor(r[5], dtype=torch.float32)
            loss = loss + nn.functional.kl_div(
                nn.functional.log_softmax(sl, dim=0), tgt_pi, reduction="sum")
        else:
            tgt = torch.tensor([r[4]])
            loss = loss + nn.functional.cross_entropy(sl.unsqueeze(0), tgt)
        hits += int(sl.argmax().item() == r[4])
        p0 += k
    return loss / len(chunk), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data/tree_pol1.jsonl.gz")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-samples", type=int, default=400000)
    ap.add_argument("--init", default="", help="warm-start (structure from self-play, style from masters)")
    ap.add_argument("--out", default="train/policy_net.pt")
    a = ap.parse_args()

    t0 = time.time()
    tr, va = load(a.data, a.max_samples)
    rnd = sum(1.0 / len(r[3]) for r in va) / max(1, len(va))
    print(f"loaded {len(tr)} train / {len(va)} val in {time.time()-t0:.0f}s | "
          f"随机基线 top1≈{rnd:.3f}", flush=True)

    model = PolicyNet()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
        print(f"warm-started from {a.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        model.train(); tl = nb = 0
        for chunk in batches(tr, a.bs, True):
            opt.zero_grad()
            loss, _ = run_batch(model, chunk)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval(); hits = 0
        with torch.no_grad():
            for chunk in batches(va, a.bs, False):
                _, h = run_batch(model, chunk)
                hits += h
        print(f"ep{ep+1}: loss {tl/nb:.4f}  val top1 {hits/len(va):.4f}", flush=True)
        torch.save(model.state_dict(), a.out)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
