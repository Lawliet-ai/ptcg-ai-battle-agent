import re
"""Pure-python featurizer shared by training AND runtime (tree leaf eval) — keeping it
dependency-free avoids train/serve skew and lets the ladder bundle run it without numpy.

State -> (idx, val, scalars):
  idx/val: sparse card-bag features, 7 zones x CARD_N ids (EmbeddingBag input)
  scalars: fixed-length dense vector (prizes, hp, energy, counts, turn)
"""
CARD_N = 1300          # > max cardId (1267); fixed so weights stay valid across runs
N_SCALAR = 16

Z_MY_ACTIVE, Z_MY_BENCH, Z_MY_HAND, Z_MY_DISCARD = 0, 1, 2, 3
Z_OP_ACTIVE, Z_OP_BENCH, Z_OP_DISCARD = 4, 5, 6
N_FEAT = 7 * CARD_N

# --- v9 "new eyes" (07-10): stadium zone + board-texture scalars. Legacy featurize()
# kept byte-identical so v8 weights stay valid; v9 nets use featurize_v9(). ---
Z_STADIUM = 7
N_FEAT_V9 = 8 * CARD_N
N_SCALAR_V9 = 31       # first 16 identical to legacy, then 15 new

# --- v9.3 "prize exposure" (07-22): the engine could see prizes ALREADY TAKEN but never
# how many prizes its own board is OFFERING. 98-loss forensics: 60% of losses had our board
# exposing >=6 prizes (opponent wins by KO'ing what is already on the table), 9% had a lone
# 1-prize body. Both are the same blind spot. megaEx=3 / ex=2 / basic=1. ---
N_SCALAR_V93 = N_SCALAR_V9 + 6


def _prize_of(p):
    """这只宝可梦被击倒时送给对手几张奖励卡(megaEx=3 / ex=2 / 其它=1)。"""
    if not p:
        return 0
    cd = _card(p.get("id"))
    if cd is None:
        return 1
    return 3 if getattr(cd, "megaEx", False) else 2 if getattr(cd, "ex", False) else 1


_CARD_CACHE = None


def _card(cid):
    global _CARD_CACHE
    if _CARD_CACHE is None:
        try:
            from cg.api import all_card_data
            _CARD_CACHE = {c.cardId: c for c in all_card_data()}
        except Exception:
            _CARD_CACHE = {}
    return _CARD_CACHE.get(cid)


def _add_poke(idx, val, zone, p, w):
    if not p:
        return
    idx.append(zone * CARD_N + p["id"]); val.append(w)
    for e in (p.get("energyCards") or []):
        idx.append(zone * CARD_N + e["id"]); val.append(0.5 * w)
    for t in (p.get("tools") or []):
        idx.append(zone * CARD_N + t["id"]); val.append(0.5 * w)


def featurize(cur, me):
    """cur = obs["current"] dict, me = our player index. Returns (idx, val, scalars)."""
    idx, val = [], []
    my = cur["players"][me]; op = cur["players"][1 - me]

    ma = (my.get("active") or [None])[0]
    oa = (op.get("active") or [None])[0]
    _add_poke(idx, val, Z_MY_ACTIVE, ma, 1.0)
    _add_poke(idx, val, Z_OP_ACTIVE, oa, 1.0)
    for b in (my.get("bench") or []):
        _add_poke(idx, val, Z_MY_BENCH, b, 1.0)
    for b in (op.get("bench") or []):
        _add_poke(idx, val, Z_OP_BENCH, b, 1.0)
    for c in (my.get("hand") or []):
        idx.append(Z_MY_HAND * CARD_N + c["id"]); val.append(0.5)
    for c in (my.get("discard") or []):
        idx.append(Z_MY_DISCARD * CARD_N + c["id"]); val.append(0.25)
    for c in (op.get("discard") or []):
        idx.append(Z_OP_DISCARD * CARD_N + c["id"]); val.append(0.25)

    def _e(p):
        return len(p.get("energyCards") or []) if p else 0
    my_bench = [b for b in (my.get("bench") or []) if b]
    op_bench = [b for b in (op.get("bench") or []) if b]
    scalars = [
        len(my.get("prize") or []) / 6.0,
        len(op.get("prize") or []) / 6.0,
        my.get("deckCount", 0) / 60.0,
        op.get("deckCount", 0) / 60.0,
        # 对手回合的帧里 my["hand"] 是 None 而 handCount 仍可见:训练样本100%手牌可见
        # (len(hand)==handCount),推理时却有58.6%的叶子落在这类帧上,旧写法喂进伪0=
        # "我方手牌打空"的濒死信号。取 handCount 兜底只是补回真值,不改变特征语义。
        (len(my["hand"]) if my.get("hand") is not None else my.get("handCount", 0)) / 10.0,
        op.get("handCount", 0) / 10.0,
        (ma["hp"] / 340.0) if ma else 0.0,
        (oa["hp"] / 340.0) if oa else 0.0,
        _e(ma) / 4.0, _e(oa) / 4.0,
        len(my_bench) / 5.0, len(op_bench) / 5.0,
        (sum(_e(b) for b in my_bench) + _e(ma)) / 10.0,
        (sum(_e(b) for b in op_bench) + _e(oa)) / 10.0,
        min(cur.get("turn", 0), 40) / 40.0,
        1.0 if cur.get("firstPlayer", 0) == me else 0.0,
    ]
    return idx, val, scalars


def featurize_v9(cur, me, v93=False):
    """v9新眼睛:legacy的全部特征 + 场地zone + 15个board-texture标量。
    v93=True 时再追加6个"奖励卡暴露"标量(见 N_SCALAR_V93)。
    新标量(17-31):板凳最弱血量(双方)/零能板凳数(抓杀目标)/板凳总血量/场地在场/
    supporterPlayed/energyAttached/stadiumPlayed/双方特殊状态数/奖差/双方弃牌堆规模。"""
    idx, val, scalars = featurize(cur, me)
    my = cur["players"][me]; op = cur["players"][1 - me]

    for s in (cur.get("stadium") or []):
        if s:
            idx.append(Z_STADIUM * CARD_N + s["id"]); val.append(1.0)

    def _hp(p):
        return (p.get("hp") or 0) if p else 0

    def _en(p):
        return len(p.get("energyCards") or []) if p else 0

    my_b = [b for b in (my.get("bench") or []) if b]
    op_b = [b for b in (op.get("bench") or []) if b]

    def _status(p):
        return sum(1 for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")
                   if p.get(k))

    scalars += [
        (min((_hp(b) for b in my_b), default=0)) / 340.0,
        (min((_hp(b) for b in op_b), default=0)) / 340.0,
        sum(1 for b in my_b if _en(b) == 0) / 5.0,
        sum(1 for b in op_b if _en(b) == 0) / 5.0,
        sum(_hp(b) for b in my_b) / 1700.0,
        sum(_hp(b) for b in op_b) / 1700.0,
        1.0 if (cur.get("stadium") or []) else 0.0,
        1.0 if cur.get("supporterPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0,
        1.0 if cur.get("stadiumPlayed") else 0.0,
        _status(my) / 3.0,
        _status(op) / 3.0,
        (len(op.get("prize") or []) - len(my.get("prize") or [])) / 6.0,
        len(my.get("discard") or []) / 60.0,
        len(op.get("discard") or []) / 60.0,
    ]
    if not v93:
        return idx, val, scalars

    # --- v9.3: prize exposure ---
    my_all = ([a for a in (my.get("active") or []) if a]) + my_b
    op_all = ([a for a in (op.get("active") or []) if a]) + op_b
    my_exp = sum(_prize_of(p) for p in my_all)
    op_exp = sum(_prize_of(p) for p in op_all)
    my_left = len(my.get("prize") or [])
    op_left = len(op.get("prize") or [])
    scalars += [
        my_exp / 9.0,                 # 我方摆在桌上的奖励卡总数
        op_exp / 9.0,
        # 关键项:对手只要打光我方现有的怪,够不够拿完他剩下的奖?>=1 表示"场面本身即败因"
        min(my_exp / max(op_left, 1), 2.0) / 2.0,
        min(op_exp / max(my_left, 1), 2.0) / 2.0,
        _prize_of(my.get("active")[0] if my.get("active") else None) / 3.0,   # 前排单体奖励卡
        1.0 if (my_exp and my_exp >= op_left) else 0.0,                        # 硬旗标
    ]
    return idx, val, scalars


def featurize_v93(cur, me):
    """v9.3:v9全部特征 + 6个奖励卡暴露标量(我方/对方摆在桌上的奖励卡总量与危险比)。"""
    return featurize_v9(cur, me, v93=True)


# --- v10 "Lawliet手册特征"(07-22):把20条人工裁决里引擎"看不见的东西"变成可学的输入。
# 不写规则、不接管决策 —— 只补信息,让网络自己学它们值多少分,由树在局面里灵活取舍。
# 覆盖四类盲区:①这一拳的真实总输出(含板凳溅射)②能量是不是回合结束就消失的临时货
# ③前排倒下有没有替补(空板凳的致命度,而非单纯的板凳数)④双方离"能开火"还差几个能量。
N_SCALAR_V10 = N_SCALAR_V93 + 8

_ATK_CACHE = None


def _attacks():
    global _ATK_CACHE
    if _ATK_CACHE is None:
        try:
            from cg.api import all_attack
            _ATK_CACHE = {a.attackId: a for a in all_attack()}
        except Exception:
            _ATK_CACHE = {}
    return _ATK_CACHE


_SPLASH_PAT = re.compile(r"also does (\d+) damage to (?:1 of )?your opponent.s benched")


def _atk_profile(p):
    """(可用招式的最大前排伤害, 该招式的板凳溅射, 还差几个能量才能开火)。
    Lawliet裁决#2:喷射冲击只要1个能量就打前排120+板凳50,引擎此前只看到120。"""
    if not p:
        return 0, 0, 9
    cd = _card(p.get("id"))
    if cd is None:
        return 0, 0, 9
    have = len(p.get("energyCards") or [])
    A = _attacks()
    best_dmg, best_sp, gap = 0, 0, 9
    for aid in (getattr(cd, "attacks", None) or []):
        a = A.get(aid)
        if a is None:
            continue
        cost = len(getattr(a, "energies", None) or [])
        dmg = getattr(a, "damage", 0) or 0
        m = _SPLASH_PAT.search((getattr(a, "text", "") or "").lower())
        sp = int(m.group(1)) if m else 0
        if cost <= have:                       # 现在就能打的招式
            if dmg + sp > best_dmg + best_sp:
                best_dmg, best_sp = dmg, sp
        gap = min(gap, max(0, cost - have))
    return best_dmg, best_sp, gap


def featurize_v10(cur, me):
    idx, val, scalars = featurize_v9(cur, me, v93=True)
    my = cur["players"][me]; op = cur["players"][1 - me]
    ma = (my.get("active") or [None])[0]
    oa = (op.get("active") or [None])[0]
    my_b = [b for b in (my.get("bench") or []) if b]
    op_b = [b for b in (op.get("bench") or []) if b]

    my_dmg, my_sp, my_gap = _atk_profile(ma)
    op_dmg, op_sp, op_gap = _atk_profile(oa)
    # 溅射能白捡几个奖:对手板凳上血量<=溅射值的目标数(Lawliet#2"一回合杀两只")
    sp_kill = sum(1 for b in op_b if (b.get("hp") or 999) <= my_sp) if my_sp else 0
    # 临时能量:回合结束会自弃的(点火能量)。Lawliet#1/#7/#9:"贴了也打不了,下回合就没了"
    tmp_e = 0
    for p in ([ma] + my_b):
        for e in ((p or {}).get("energyCards") or []):
            cd = _card(e.get("id"))
            if cd is None:
                continue
            t = " ".join((getattr(sk, "text", "") or "") for sk in (getattr(cd, "skills", None) or [])).lower()
            if "end of your turn" in t and "discard" in t:
                tmp_e += 1
    # 空板凳致命度:前排倒下有没有人能顶上(0=没有→当场判负,不是"少一只"那么简单)
    has_backup = 1.0 if my_b else 0.0

    scalars += [
        (my_dmg + my_sp) / 340.0,     # 我方这一拳的真实总输出(含溅射)
        (op_dmg + op_sp) / 340.0,     # 对手的
        my_sp / 100.0,                # 溅射量本身
        sp_kill / 5.0,                # 溅射能直接击倒的对手板凳数
        min(my_gap, 3) / 3.0,         # 我方离开火还差几个能量
        min(op_gap, 3) / 3.0,         # 对手离开火还差几个能量(威胁计时器)
        tmp_e / 4.0,                  # 场上临时能量数(虚假的战力)
        has_backup,                   # 有无替补(空板凳=前排一倒即负)
    ]
    return idx, val, scalars
