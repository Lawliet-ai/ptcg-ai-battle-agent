"""Runtime value net: numpy forward pass over the exported .npz weights, evaluating a
search-API state (class form) from player `me`'s perspective -> [-1, 1].

Loaded lazily via CABT_VALUE_NET=<path-to-npz> (or explicit load()); if anything fails
the tree silently falls back to the hand-written policy._evaluate — never crashes."""
import os

import numpy as np

from features import (featurize, featurize_v9, featurize_v93, featurize_v10,  # noqa: F401
                      N_SCALAR, N_FEAT_V9, N_SCALAR_V9, N_SCALAR_V93, N_SCALAR_V10)

_CACHE = {}
_HANDFIX = os.environ.get("CABT_HANDFIX", "0") == "1"   # hand=None透传(见_cur_dict注释),默认关


def load(path=None):
    """按路径缓存的多脑加载:同进程可同时驾驶 v8/v9(过闸考试的双脑前提)。"""
    path = path or os.environ.get("CABT_VALUE_NET")
    if not path:
        return None
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        return None
    z = np.load(path)
    W = {k: z[k] for k in ("emb", "w1", "b1", "w2", "b2", "wo", "bo")}
    # v8/v9 双轨:v9 npz 的 emb 行数 = 8*CARD_N(多出场地zone),据此自动选特征器
    W["v9"] = (W["emb"].shape[0] == N_FEAT_V9)
    # 标量段宽度 = fc1输入宽 - 嵌入维;v9.3 与 v9 的 emb 行数相同,只有标量数不同(31→37),
    # 所以必须按这个宽度选特征器,不能只看 emb 行数。
    W["nsc"] = int(W["w1"].shape[1] - W["emb"].shape[1])
    _CACHE[path] = W
    return W


def _poke_dict(p):
    if p is None:
        return None
    return {"id": p.id, "hp": p.hp,
            "energyCards": [{"id": e.id} for e in (p.energyCards or [])],
            "tools": [{"id": t.id} for t in (p.tools or [])]}


def _cur_dict(state, me):
    """Minimal obs['current']-shaped dict from the search-API class state."""
    out = {"players": [], "turn": getattr(state, "turn", 0),
           "firstPlayer": getattr(state, "firstPlayer", 0),
           "stadium": [{"id": s.id} for s in (getattr(state, "stadium", None) or []) if s],
           "supporterPlayed": bool(getattr(state, "supporterPlayed", False)),
           "energyAttached": bool(getattr(state, "energyAttached", False)),
           "stadiumPlayed": bool(getattr(state, "stadiumPlayed", False))}
    for pi in range(2):
        ps = state.players[pi]
        out["players"].append({
            "active": [_poke_dict(p) for p in (ps.active or [])] or [None],
            "bench": [_poke_dict(p) for p in (ps.bench or []) if p],
            # hand=None(对手回合帧,46.9-58.6%的叶子)必须透传None让features的handCount
            # 兜底接管;旧写法洗成[]=喂"手牌打空"伪濒死信号,兜底成死代码。门控默认关。
            "hand": ((([{"id": c.id} for c in ps.hand] if ps.hand is not None else None)
                      if _HANDFIX else [{"id": c.id} for c in (ps.hand or [])])
                     if pi == me else []),
            "discard": [{"id": c.id} for c in (ps.discard or [])],
            "prize": [0] * len(ps.prize or []),
            "deckCount": ps.deckCount, "handCount": ps.handCount,
            "poisoned": bool(getattr(ps, "poisoned", False)),
            "burned": bool(getattr(ps, "burned", False)),
            "asleep": bool(getattr(ps, "asleep", False)),
            "paralyzed": bool(getattr(ps, "paralyzed", False)),
            "confused": bool(getattr(ps, "confused", False)),
        })
    return out


def evaluate_state(state, me, net=None):
    """-> value in [-1,1] from me's perspective, or None if net unavailable.
    net: 指定npz路径(双脑对战);None=环境变量默认。"""
    W = load(net)
    if W is None:
        return None
    try:
        if W.get("nsc", 0) >= N_SCALAR_V10:
            _feat = featurize_v10
        elif W.get("nsc", 0) >= N_SCALAR_V93:
            _feat = featurize_v93
        elif W.get("v9"):
            _feat = featurize_v9
        else:
            _feat = featurize
        idx, val, sc = _feat(_cur_dict(state, me), me)
        e = (W["emb"][idx] * np.asarray(val, dtype=np.float32)[:, None]).sum(0)
        x = np.concatenate([e, np.asarray(sc, dtype=np.float32)])
        h = np.maximum(W["w1"] @ x + W["b1"], 0)
        h = np.maximum(W["w2"] @ h + W["b2"], 0)
        p = 1.0 / (1.0 + np.exp(-(W["wo"] @ h + W["bo"])[0]))
        return float(2.0 * p - 1.0)
    except Exception:
        return None
