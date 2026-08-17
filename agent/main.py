"""Kaggle submission entry point. Defines agent(obs_dict) -> list[int].

This build: T0 Solrock/Lunatone Mega Lucario deck, piloted by the macro-action PUCT
tree (search_agent_tree) with the learned value net at the leaves (value_net.npz,
numpy forward; silently falls back to the hand-written eval if numpy is missing) and
the fuel gauge (spends the per-agent time bank, always keeping a 60s reserve).
"""
import os, sys

# Kaggle exec()s this file with no __file__ defined -> fall back to the fixed path.
try:
    DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DIR = "/kaggle_simulations/agent"
if DIR not in sys.path:
    sys.path.insert(0, DIR)

# Probe: does the ladder image have numpy? (read back via episode agent logs)
try:
    import numpy as _np
    print("[probe] numpy OK", _np.__version__, file=sys.stderr, flush=True)
except Exception as _e:
    print("[probe] numpy MISSING:", repr(_e)[:60], file=sys.stderr, flush=True)

from search_agent_tree import make_tree_agent


def _read_deck():
    for p in (os.path.join(DIR, "deck.csv"), "/kaggle_simulations/agent/deck.csv", "deck.csv"):
        if os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise FileNotFoundError("deck.csv")


DECK = _read_deck()

# v9.2b价值网是海星星棋谱训出来的:只给海星星牌组当脑子;
# 其他牌组(如玛俐长毛巨魔)裸树,net加载器拿不到env会安全返回None走启发式叶子。
if 1030 in DECK:
    os.environ.setdefault("CABT_VALUE_NET", os.path.join(DIR, "value_net.npz"))

# Candidate opponent decks for determinization (recognised from their visible board).
# ABOMASNOW = current ladder king (32% of our matches), extracted from real replays.
ABOMASNOW = (
    [723] * 4 + [722] * 4 + [721] * 2 + [1227] * 4 + [1235] * 4 + [1145] * 4
    + [1205] * 2 + [1158] + [3] * 35
)
MEGA_STARMIE = (
    [3] * 9 + [17] * 4 + [666] * 4 + [1086] * 4 + [1120] * 4 + [1122] * 4
    + [1145] * 4 + [1189] * 4 + [1227] * 4 + [1229] * 4 + [1030] * 3 + [1031] * 3
    + [1097] * 2 + [1223] * 2 + [1225] * 2 + [1121] + [1159] + [1182]
)
GRIMMSNARL = (
    [7] * 10 + [112] * 4 + [646] * 4 + [1086] * 4 + [1152] * 4 + [1219] * 4
    + [1227] * 4 + [1259] * 4 + [647] * 3 + [648] * 3 + [1079] * 3 + [1097] * 3
    + [104] * 2 + [860] * 2 + [1161] * 2 + [1182] * 2 + [1080] + [1231]
)

# Determinization prior pool: the 14 modal lists actually on the ladder (from replay
# post-mortem), ordered by prevalence. Falls back to the 3 hand-curated lists above if
# the generated module is missing. Honest-ruler A/B: 14-pool 60.7% vs 3-stale 58.3%.
try:
    from priors_data import PRIORS
except Exception:
    PRIORS = [ABOMASNOW, MEGA_STARMIE, GRIMMSNARL]

# go_first=True: Carmine ("if you go first, usable turn 1") + aggro tempo (evolve T2).
# adrena=True: 愿增猿卡面数学入威胁账(对手带恶能猿=+30杀伤半径/只+每回合血税)。
# 天梯55%是玛俐系,本地陪练开不出名师级的猿引擎,这笔账的真裁判在天梯。
# go_first: 纯海星星版先攻(Carmine逻辑);闪焰王牌版沿用当初gosecond实测配置=后攻。
_agent = make_tree_agent(DECK, opp_priors=PRIORS,
                         D=3, iters=200, top_m=4, H=10, go_first=(666 not in DECK), budget=6.0,
                         adrena=True)


def agent(obs_dict):
    return _agent(obs_dict)
