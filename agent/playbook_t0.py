# -*- coding: utf-8 -*-
"""Lawliet手册·T0月石太阳岩 开局铁律层(强约束)。

开局阶段(前几个回合)的 MAIN 决策不交给树,直接照手册脚本执行:
  ①Mega路卡利欧能进化就进化 ②日月引擎/幕下/二号利欧路第一时间拍板凳
  ③重力山铺场 ④贴能按手册目标序(Mega前排→板凳利欧路→幕下;太阳岩当回合手贴)
  ⑤贴完能量后手里还有富余 → 月之循环弃1抽3
攻击选择永远交给树(审计:斩杀/强化精算已100%/99%达标)。
仅当牌组同时含 Mega路卡利欧/太阳岩/月石 时启用(自动识别)。

`opening_pick(obs_cls, cards) -> list[int] | None`,None=放行给树。
"""
import os, sys

MEGA, RIOLU, SOLROCK, LUNATONE, MAKUHITA, HARIYAMA = 678, 677, 676, 675, 673, 674
GRAVITY, F_ENERGY = 1252, 6
_OPENING_TURNS = 8          # 引擎turn按半回合计 → 双方各前4个回合


def deck_matches(deck_ids):
    s = set(deck_ids)
    return MEGA in s and SOLROCK in s and LUNATONE in s


def _dbg(msg):
    if os.environ.get("CABT_DBG_PB"):
        print(f"[pb] {msg}", file=sys.stderr, flush=True)


def _race_phase(st, my, opp):
    """竞速期判定(Lawliet07-07:该放弃场面把资源砸向奖品竞速的时刻)——任一成立:
    ①任一方奖品≤2(终盘数学题) ②我方牌库≤10(月之循环会抽死自己)
    ③对手只差1奖(补脆皮引擎=给老大指令送活靶)"""
    try:
        if len(my.prize or []) <= 2 or len(opp.prize or []) <= 2:
            return True
        if my.deckCount <= 10:
            return True
    except Exception:
        return False
    return False


def _in_play(my):
    out = []
    for p in (my.active or []):
        if p:
            out.append(p)
    for p in (my.bench or []):
        if p:
            out.append(p)
    return out


def opening_pick(obs, cards):
    """obs = Observation class (MAIN select). Returns scripted pick or None."""
    try:
        st = obs.current
        me = st.yourIndex
        my = st.players[me]
        if st.turn > _OPENING_TURNS:
            return None            # 定版(840局实测):只管开局=51.8最优;管得越宽越差
        hand = my.hand or []
        opts = obs.select.option
        pokes = _in_play(my)
        counts = {}
        for p in pokes:
            counts[p.id] = counts.get(p.id, 0) + 1
        hand_energy = sum(1 for c in hand if c.id == F_ENERGY)

        def hand_id(o):
            i = getattr(o, "index", None)
            return hand[i].id if (i is not None and i < len(hand)) else None

        # ① 进化 Mega 路卡利欧,见到就进
        for i, o in enumerate(opts):
            if o.type == 9 and hand_id(o) == MEGA:
                _dbg(f"evolve Mega t{st.turn}")
                return [i]
        # ② 拍引擎/铺场(按手册理想盘面缺什么补什么;Lawliet修正:先保打手线——
        # 场上没利欧路先拍利欧路,再轮到引擎)
        want = []
        if counts.get(RIOLU, 0) == 0:
            want.append(RIOLU)
        if counts.get(SOLROCK, 0) < 1:
            want.append(SOLROCK)
        if counts.get(LUNATONE, 0) < 2:
            want.append(LUNATONE)
        if counts.get(MAKUHITA, 0) < 1:
            want.append(MAKUHITA)
        if counts.get(RIOLU, 0) < 2:
            want.append(RIOLU)
        for w in want:
            for i, o in enumerate(opts):
                if o.type == 7 and hand_id(o) == w:
                    _dbg(f"bench {w} t{st.turn}")
                    return [i]
        # ③ 重力山(我方无2阶,纯单方面削对面)
        for i, o in enumerate(opts):
            if o.type == 7 and hand_id(o) == GRAVITY:
                _dbg(f"stadium t{st.turn}")
                return [i]
        # ④ 贴能按手册目标序:Mega前排(<2) → 板凳利欧路(<2) → 幕下(<3)
        def _tgt(o):
            try:
                if o.inPlayArea == 4:
                    return (my.active or [None])[0]
                if o.inPlayArea == 5:
                    return my.bench[o.inPlayIndex]
            except Exception:
                return None
            return None
        best_i, best_rank = None, 99
        for i, o in enumerate(opts):
            if o.type != 8 or hand_id(o) != F_ENERGY:
                continue
            t = _tgt(o)
            if t is None:
                continue
            ne = len(t.energyCards or [])
            rank = 99
            if t.id == MEGA and o.inPlayArea == 4 and ne < 2:
                rank = 0
            elif t.id == RIOLU and o.inPlayArea == 4 and ne < 1:
                rank = 1
            elif t.id == RIOLU and o.inPlayArea == 5 and ne < 2:
                rank = 2
            elif t.id == MAKUHITA and ne < 3:
                rank = 3
            if rank < best_rank:
                best_rank, best_i = rank, i
        if best_i is not None and hand_energy >= 1 and st.turn <= _OPENING_TURNS:
            _dbg(f"attach rank{best_rank} t{st.turn}")
            return [best_i]
        # ⑤ 月之循环:贴完(或没得贴)且手里还有能量 → 弃1抽3
        attached = bool(getattr(st, "energyAttached", False))
        if hand_energy >= (1 if attached or best_i is None else 2):
            for i, o in enumerate(opts):
                if o.type == 10:
                    try:
                        src = None
                        if o.area == 4:
                            src = (my.active or [None])[0]
                        elif o.area == 5:
                            src = my.bench[o.index]
                        if src is not None and src.id == LUNATONE:
                            _dbg(f"lunar cycle t{st.turn}")
                            return [i]
                    except Exception:
                        continue
        return None                # 攻击/老大/其他 → 交给树
    except Exception:
        return None


def setup_active_pick(opts, hand):
    """SETUP阶段:手册指定前排=利欧路(其次幕下)。返回index或None。"""
    try:
        order = {RIOLU: 0, MAKUHITA: 1}
        best_i, best_r = None, 99
        for i, o in enumerate(opts):
            idx = getattr(o, "index", None)
            cid = hand[idx].id if (idx is not None and idx < len(hand)) else None
            r = order.get(cid, 99)
            if r < best_r:
                best_r, best_i = r, i
        return best_i if best_r < 99 else None
    except Exception:
        return None


def fetch_pick(obs):
    """检索铁律(补全"半套脚本"的另一半):开局阶段,宝可平板/黄昏球/斗之锣等单张
    检索优先抓还缺的日月引擎件;手上一张能量都没有时让位(先拿能量保运转)。"""
    try:
        st = obs.current
        sel = obs.select
        if sel.maxCount != 1:
            return None
        me = st.yourIndex
        my = st.players[me]
        if st.turn > _OPENING_TURNS:
            return None            # 定版:检索铁律同样只管开局
        hand = my.hand or []
        hand_energy = sum(1 for c in hand if c.id == F_ENERGY)
        opts = sel.option
        if hand_energy == 0 and any(getattr(o, "cardId", None) == F_ENERGY for o in opts):
            return None                     # 没能量先拿能量,引擎转不动全白搭
        seen = {}
        for p in _in_play(my):
            seen[p.id] = seen.get(p.id, 0) + 1
        for c in hand:
            seen[c.id] = seen.get(c.id, 0) + 1
        # Lawliet修正(07-07):先判断有没有基础打手——没有利欧路先找利欧路(保打手线),
        # 有了才去找引擎;多余资源再补第二只月石。
        want = []
        if seen.get(RIOLU, 0) == 0:
            want.append(RIOLU)
        if seen.get(SOLROCK, 0) < 1:
            want.append(SOLROCK)
        if seen.get(LUNATONE, 0) < 1:
            want.append(LUNATONE)
        if seen.get(LUNATONE, 0) == 1:
            want.append(LUNATONE)           # 手册:板凳两只月石(备份)
        for w in want:
            for i, o in enumerate(opts):
                if getattr(o, "cardId", None) == w:
                    _dbg(f"fetch {w} t{st.turn}")
                    return i
        return None
    except Exception:
        return None
