# -*- coding: utf-8 -*-
"""RL 策略的部署态运行时：numpy-only 前向（天梯包不带 torch）。

MAIN(context=0) 决策点用训练出的策略网络贪心出招；其余决策点、以及任何异常，
一律回退到 policy.make_agent（圣经-87 条令层 + 启发式）——与训练时的分工完全一致。

网络权重从 rl_net.npz 加载（与 train/export_rl.py 的键一一对应）。
直接 import featurize_v9（features.py 无外部依赖），不走 CABT_FEAT 环境变量 ——
Kaggle 上没人替我们设环境变量，静默走错特征宽度是最阴的死法。
"""
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from features import featurize_v9, CARD_N            # noqa: E402
import policy as _policy                              # noqa: E402

N_TYPE, ATK_N = 16, 600

# 自检计数器（冒烟测试读取；线上无副作用）
STATS = {"net": 0, "fallback": 0, "error": 0}


def load_net(path):
    d = np.load(path)
    need = ("emb", "sfc_w", "sfc_b", "card", "atk", "otype",
            "h1_w", "h1_b", "out_w", "out_b", "vout_w", "vout_b")
    miss = [k for k in need if k not in d.files]
    if miss:
        raise KeyError(f"rl_net.npz missing keys: {miss}")
    W = {k: d[k] for k in need}
    W["vout_b"] = float(W["vout_b"])
    return W


def _opt_feat(o, cur, me):
    t = o.get("type"); cid = o.get("cardId"); aid = o.get("attackId"); idx = o.get("index")
    t = t if isinstance(t, int) and 0 <= t < N_TYPE else 0
    if t in (7, 8) and cid is None and isinstance(idx, int):
        hand = cur["players"][me].get("hand") or []
        if 0 <= idx < len(hand):
            c = hand[idx]
            cid = c.get("id") if isinstance(c, dict) else c
    cid = cid if isinstance(cid, int) and 0 <= cid < CARD_N else 0
    aid = aid if isinstance(aid, int) and 0 <= aid < ATK_N else 0
    return t, cid, aid


def _forward(W, idx, val, sc, ofs):
    e = (W["emb"][idx] * np.asarray(val, dtype=np.float32)[:, None]).sum(0)
    x = np.concatenate([e, np.asarray(sc, dtype=np.float32)])
    srep = np.maximum(W["sfc_w"] @ x + W["sfc_b"], 0.0)
    t = np.array([o[0] for o in ofs]); c = np.array([o[1] for o in ofs]); a = np.array([o[2] for o in ofs])
    of = np.concatenate([W["otype"][t], W["card"][c], W["atk"][a]], axis=1)
    h = np.maximum(np.concatenate([np.repeat(srep[None, :], len(ofs), 0), of], axis=1)
                   @ W["h1_w"].T + W["h1_b"], 0.0)
    return (h @ W["out_w"].T + W["out_b"]).ravel()


def make_rl_agent(deck_ids, net_path, go_first=False):
    deck = [int(x) for x in deck_ids]
    W = load_net(net_path)                 # 加载失败=启动即炸，validation episode 会当场暴露
    heur = _policy.make_agent(deck, go_first=go_first)

    def agent(obs_dict):
        sel = obs_dict.get("select")
        if sel is None:
            return list(deck)
        try:
            if sel.get("context") != 0:
                STATS["fallback"] += 1
                return heur(obs_dict)
            cur = obs_dict.get("current")
            opts = sel.get("option") or []
            if cur is None or len(opts) < 2:
                STATS["fallback"] += 1
                return heur(obs_dict)
            me = cur["yourIndex"]
            idx, val, sc = featurize_v9(cur, me)
            ofs = [_opt_feat(o, cur, me) for o in opts]
            logits = _forward(W, idx, val, sc, ofs)
            STATS["net"] += 1
            return [int(np.argmax(logits))]
        except Exception:
            STATS["error"] += 1
            try:
                return heur(obs_dict)
            except Exception:
                mn = sel.get("minCount", 0) if sel else 0
                n = len(sel.get("option", [])) if sel else 0
                return list(range(min(mn, n)))

    return agent
