"""Macro-action PUCT search agent (MCTS) for the cabt engine.

A drop-in, stronger sibling of search_agent.py with the same contract:
    make_tree_agent(deck_ids, opp_priors, ...) -> agent(obs_dict) -> list[int].

Design — the official Kiyota RL+MCTS skeleton fused with our heuristic:
  * The tree branches ONLY at our own MAIN decisions. Every sub-decision (targeting,
    discards, coin flips) and the entire opponent turn in between is resolved by the
    heuristic policy._decide — i.e. the opponent is modelled as a fixed reasonable
    player, the SAME regime as the flat-rollout agent, but with structured lookahead
    over our decisions instead of independent playouts. Because every branch node is
    ours, no value negation is needed (unlike Kiyota's two-player tree).
  * Children = the heuristic's shortlisted MAIN candidates (top_m + every attack, whose
    scaling damage the heuristic mis-values). Prior = softmax of their heuristic scores
    (A1, policy._main_prior).
  * Leaf value = a short truncated heuristic rollout then policy._evaluate (A2). A
    terminal leaf scores +1 win / 0 draw / -1 loss from the engine result.
  * PIMC (A4): build an independent tree over D determinized worlds — the unseen cards
    are drawn from OUR remaining deck; the opponent's are guessed from the recognised
    meta prior — and pick the MAIN action with the most total visits across worlds.
  * Never crashes (A6): any failure at any level falls back to the heuristic agent.

This reuses _determinize / _recognize from search_agent.py so the PIMC sampling stays
identical to the flat-rollout agent (apples-to-apples A/B).
"""
import math, random, sys, time

from cg.api import (to_observation_class, search_begin, search_step, search_end,
                    OptionType as OT, SelectContext as SC)
import policy
from search_agent import _determinize, _recognize, _update_prize_cache, _build_sigs

try:                    # learned value net (CABT_VALUE_NET=<npz>); optional — numpy may
    import value_net as _VN   # be absent on the ladder, then we fall back to hand eval
except Exception:
    _VN = None
try:                    # learned policy head (CABT_POLICY_NET=<npz>): move-ordering prior
    import policy_net as _PN
except Exception:
    _PN = None
try:                    # Lawliet手册·T0开局铁律层(牌组匹配才启用)
    import playbook_t0 as _PB
except Exception:
    _PB = None
try:                    # Lawliet手册·海星场地铁律层(顶对手场地/毒猴先手节日广场)
    import playbook_staryu as _PBS
except Exception:
    _PBS = None

_MAIN = 0                 # SelectContext.MAIN
import os as _os
_C_PUCT = float(_os.environ.get("CABT_CPUCT", "0.4"))   # PUCT探索常数(扫参开关)
# 候选层"空板凳保底铺场"开关,与 CABT_BENCH_GUARD(守卫层)分离,便于三臂A/B。
# 风险:多一个候选=固定iters预算下稀释每候选的搜索次数,并摊薄原有先验。
# 07-23三种子144局判决:候选注入 57.6% vs 纯树 48.6%(+9.0,三种子全胜 +2.0/+18.7/+6.2)。
# 与"规则接管"(天梯582/604/676全败)形成对照:病根是视野盲区不是判断力——铺场先验7000
# 排在14支援者(8800)+18道具(8600)之后被 top_m=4 截断,树连评估它的机会都没有。
# 这里只负责让树看见,做不做由搜索自己算。默认开。
_CAND_BENCH = [_os.environ.get("CABT_CAND_BENCH", "1") != "0"]
_CAND_GRIMM = [_os.environ.get("CABT_CAND_GRIMM", "0") == "1"]   # 玛俐名师手册注入包(糖果/老大/印章)
_CAND_BIBLE = [_os.environ.get("CABT_BIBLE", "1") != "0"]        # 圣经注入包(萨尔瓦多T1/小胜救场/牢笼/斗篷)
# 叶子评估分布对齐:走到我方手牌可见的局面再喂价值网(训练100%手牌可见 vs 推理58.6%不可见)
_ALIGN_EVAL = [_os.environ.get("CABT_ALIGN_EVAL", "0") == "1"]
# PIMC时间片按世界均分(旧: D个世界抢一个总deadline, 平均只跑2.76/3)
_PER_WORLD = [_os.environ.get("CABT_PER_WORLD", "1") != "0"]
# Lawliet威胁账:leaf评估叠加"按送奖加权的双向即时兑换威胁"(开关,默认开)
_THREAT = [_os.environ.get("CABT_THREAT", "1") != "0"]
# 能量候选注入:保证"给会开火的攻击手贴能量"被树看见(截断扫描:被top_m截断的56%是贴能)。
# 与铺场注入同构——只加一个精准候选,不抬top_m(抬宽3种子实测-3~-4:稀释搜索+引入垃圾)。
_CAND_FUEL = [_os.environ.get("CABT_CAND_FUEL", "0") == "1"]
# 尸检修正三:斩杀被能量锁住时注入贴能候选(窄版fuel,只在"贴了当回合能杀"时触发)
_CAND_LFUEL = [_os.environ.get("CABT_CAND_LFUEL", "0") == "1"]
# 场地候选注入:冲浪海滩(免费换血)+战斗牢笼(挡板凳放伤,治胡地)被top_m截断34次。场地先验6000<道具/支援者。
_CAND_STADIUM = [_os.environ.get("CABT_CAND_STADIUM", "0") == "1"]
# 退却注入:退却297次被top_m截断(先验极低因弃能量)。只在板凳有显著更强攻击手时注入(Lawliet#9换血)。
_CAND_RETREAT = [_os.environ.get("CABT_CAND_RETREAT", "0") == "1"]
_STEP_CAP = 400           # safety cap while auto-resolving to the next MAIN
_DBG = [0]
_last_err = [None]


# ---------- forward-model helpers ----------
def _terminal_value(cur, me):
    """±1 / 0 from the engine result field (result: player index winner, 2 draw)."""
    if cur.result == 2:
        return 0.0
    return 1.0 if cur.result == me else -1.0


def _decide_pick(o, cards, attacks, go_first, deck):
    """Heuristic pick for the current (non-branch) decision, sanitised to a legal set."""
    sel = o.select
    if sel is None:
        return list(deck)
    p = policy._decide(o, sel, sel.option, sel.context, sel.minCount, sel.maxCount,
                       cards, attacks, go_first)
    p = [i for i in p if 0 <= i < len(sel.option)]
    if len(p) < sel.minCount:
        p = list(range(sel.minCount))
    return p[:max(sel.minCount, sel.maxCount)] if sel.maxCount else p[:sel.minCount]


def _advance(searchId, pick, me, deck, cards, attacks, go_first):
    """Apply `pick`, then auto-resolve every sub-decision and the opponent turn with the
    heuristic until we reach OUR next MAIN decision (a branch node) or the game ends.
    Returns the resulting SearchState."""
    ss = search_step(searchId, pick)
    steps = 0
    while steps < _STEP_CAP:
        o = ss.observation
        cur = o.current
        if cur is None or cur.result != -1:
            return ss
        sel = o.select
        if sel is not None and sel.context == _MAIN and cur.yourIndex == me:
            return ss                                 # our next MAIN → stop, branch here
        ss = search_step(ss.searchId, _decide_pick(o, cards, attacks, go_first, deck))
        steps += 1
    return ss


def _advance_2p(searchId, pick, me, deck, cards, attacks, go_first):
    """双人树版:应用pick后自动解决所有子决策,停在任意一方的下一个MAIN(或终局)。"""
    ss = search_step(searchId, pick)
    steps = 0
    while steps < _STEP_CAP:
        o = ss.observation
        cur = o.current
        if cur is None or cur.result != -1:
            return ss
        sel = o.select
        if sel is not None and sel.context == _MAIN:
            return ss                                 # 任意一方MAIN → 分叉点
        ss = search_step(ss.searchId, _decide_pick(o, cards, attacks, go_first, deck))
        steps += 1
    return ss


def _leaf_value(ss, me, deck, cards, attacks, go_first, H, net=None):
    """Truncated heuristic rollout of H steps then positional eval; ±1/0 if a terminal is
    reached first. Hedges the eval's noise with a few plies of real play (classic UCT)."""
    o = ss.observation
    cur = o.current
    if cur is not None and cur.result != -1:
        return _terminal_value(cur, me)
    steps = 0
    while steps < H:
        o = ss.observation
        cur = o.current
        if cur is None or cur.result != -1:
            break
        ss = search_step(ss.searchId, _decide_pick(o, cards, attacks, go_first, deck))
        steps += 1
    o = ss.observation
    cur = o.current
    # 训练样本100%是"我方视角/手牌可见"(20000/20000),而叶子有58.6%落在对手回合的帧上,
    # 那里 players[me].hand is None → 手牌嵌入与手牌数标量全零 = 训练分布里"手牌打空"的
    # 濒死信号。这里把推演再往前走到我方能看见手牌的局面再交给网络评估,让推理分布对齐训练分布。
    if _ALIGN_EVAL[0] and cur is not None and cur.result == -1:
        extra = 0
        while extra < 12:
            try:
                if cur.players[me].hand is not None:
                    break
            except Exception:
                break
            ss = search_step(ss.searchId, _decide_pick(o, cards, attacks, go_first, deck))
            o = ss.observation
            cur = o.current
            if cur is None or cur.result != -1:
                break
            extra += 1
    if cur is not None and cur.result != -1:
        return _terminal_value(cur, me)
    if _VN is not None and cur is not None:
        v = _VN.evaluate_state(cur, me, net=net)
        if v is not None:
            # Lawliet威胁账:网络评估是静态的,叠一个"即时兑换"动态项(按送奖加权的双向击杀威胁)。
            # 不搜索不训练,直接改评估精度——今日教训:搜索加深无用/网络学不会/直接注入才涨(候选注入+39)。
            if _THREAT[0]:
                v = max(-1.0, min(1.0, v + policy.threat_eval(cur, me, cards, attacks)))
            return v
    ev = policy._evaluate(o, me, cards, attacks)
    if _THREAT[0] and cur is not None:
        ev = max(-1.0, min(1.0, ev + policy.threat_eval(cur, me, cards, attacks)))
    return ev


# ---------- tree ----------
class _Child:
    __slots__ = ("opt", "prob", "node")

    def __init__(self, opt, prob):
        self.opt = opt        # REAL option index of this MAIN action (world-invariant)
        self.prob = prob
        self.node = None


class _Node:
    __slots__ = ("ss", "value", "total", "visit", "children", "terminal", "mine")

    def __init__(self, ss, value, children, terminal, mine=True):
        self.ss = ss
        self.value = value
        self.total = value        # seed with its own estimate (== one self-backprop)
        self.visit = 1
        self.children = children
        self.terminal = terminal
        self.mine = mine


def _shortlist(o, cards, attacks, top_m, pnet=None):
    """Candidate MAIN option indices + priors for a branch node."""
    opts = o.select.option
    n = len(opts)
    scored = sorted(range(n), key=lambda i: -policy._score_main(opts[i], o, cards, attacks))
    cand = scored[:min(top_m, n)]
    for i in range(n):
        if opts[i].type == OT.ATTACK and i not in cand:   # search judges scaling attacks
            cand.append(i)
    # 板凳为空时,保证"铺基础怪"进候选。它的静态分(7000)排在支援者(8800)/物品(8600)之后,
    # 牌组里14张支援者+18张物品会把 top_m 名额占满,于是这个动作从不被展开——树连评估
    # 它的机会都没有(98败复盘:手握海星星连续7回合空板凳)。这里只负责让树看见,
    # 值不值得铺由搜索自己算,与上面"每个ATTACK都进候选"同构。
    if _CAND_BENCH[0]:
        _bi = policy._bench_basic_idx(scored, opts, o, cards, allow_fetch=True)
        if _bi is not None and _bi not in cand:
            cand.append(_bi)
    if _CAND_FUEL[0]:
        _fi = policy._fuel_attacker_idx(scored, opts, o, cards, attacks)
        if _fi is not None and _fi not in cand:
            cand.append(_fi)
    if _CAND_LFUEL[0]:
        _li = policy._lethal_fuel_idx(scored, opts, o, cards, attacks)
        if _li is not None and _li not in cand:
            cand.append(_li)
    if _CAND_STADIUM[0]:
        _si = policy._stadium_idx(scored, opts, o, cards)
        if _si is not None and _si not in cand:
            cand.append(_si)
    if _CAND_RETREAT[0]:
        _ri = policy._retreat_swap_idx(scored, opts, o, cards, attacks)
        if _ri is not None and _ri not in cand:
            cand.append(_ri)
    if _CAND_GRIMM[0]:
        for _gi in policy._grimm_pack_idx(scored, opts, o, cards, attacks):
            if _gi not in cand:
                cand.append(_gi)
    if _CAND_BIBLE[0]:
        for _bi2 in policy._bible_pack_idx(scored, opts, o, cards, attacks):
            if _bi2 not in cand:
                cand.append(_bi2)
    sc = [policy._score_main(opts[i], o, cards, attacks) for i in cand]
    pri = policy._main_prior(sc)
    if _PN is not None:                    # learned prior over the shortlisted candidates
        p = _PN.priors(o.current, o.current.yourIndex, [opts[i] for i in cand], pnet=pnet)
        if p is not None:
            pri = p
    return [_Child(cand[k], pri[k]) for k in range(len(cand))]


def _make_node(ss, me, deck, cards, attacks, go_first, top_m, H, net=None, pnet=None):
    o = ss.observation
    cur = o.current
    if cur is None or cur.result != -1:
        return _Node(ss, _terminal_value(cur, me) if cur else 0.0, [], True)
    children = _shortlist(o, cards, attacks, top_m, pnet=pnet)
    return _Node(ss, _leaf_value(ss, me, deck, cards, attacks, go_first, H, net=net), children, False)


def _make_node_2p(ss, me, deck, cards, attacks, go_first, top_m, H, net=None, pnet=None):
    o = ss.observation
    cur = o.current
    if cur is None or cur.result != -1:
        return _Node(ss, _terminal_value(cur, me) if cur else 0.0, [], True)
    mine = (cur.yourIndex == me)
    children = _shortlist(o, cards, attacks, top_m, pnet=(pnet if mine else None))
    return _Node(ss, _leaf_value(ss, me, deck, cards, attacks, go_first, H, net=net),
                 children, False, mine=mine)


def _select_child(node):
    c = _C_PUCT * math.sqrt(node.visit)
    sign = 1.0 if node.mine else -1.0            # 对手节点:选对我们最坏的
    best, best_u = None, -1e18
    for ch in node.children:
        if ch.node is None:
            q = node.total / node.visit          # first-play urgency: parent average
            vis = 0
        else:
            q = ch.node.total / ch.node.visit
            vis = ch.node.visit
        u = sign * q + c * ch.prob / (1 + vis)
        if u > best_u:
            best_u, best = u, ch
    return best


def _backprop(path, value):
    for nd in path:
        nd.total += value
        nd.visit += 1


def _run_tree(root, me, deck, cards, attacks, go_first, iters, top_m, H, deadline, net=None, pnet=None,
              two_player=False):
    adv = _advance_2p if two_player else _advance
    mk = _make_node_2p if two_player else _make_node
    for _ in range(iters):
        if time.time() > deadline:
            break
        node = root
        path = [root]
        while not node.terminal and node.children:
            ch = _select_child(node)
            if ch.node is None:                       # expand
                try:
                    ss2 = adv(node.ss.searchId, [ch.opt], me, deck, cards, attacks, go_first)
                    ch.node = mk(ss2, me, deck, cards, attacks, go_first, top_m, H, net=net, pnet=pnet)
                    val = ch.node.value
                except Exception as e:
                    _last_err[0] = repr(e)[:90]
                    ch.node = _Node(node.ss, node.value, [], True)
                    val = node.value
                _backprop(path, val)                  # ancestors; child self-seeded
                node = None
                break
            node = ch.node
            path.append(node)
        if node is not None:                          # reached terminal / childless
            _backprop(path, node.value)


def make_tree_agent(deck_ids, opp_priors, D=4, iters=24, top_m=4, H=6,
                    go_first=False, seed=12345, budget=2.0, stats_out=None, net=None,
                    snipe=False, pnet=None, two_player=False, net_map=None, adrena=None,
                    text_dmg=None):
    deck = [int(x) for x in deck_ids]
    if opp_priors and isinstance(opp_priors[0], int):
        opp_priors = [opp_priors]
    priors = [[int(x) for x in p] for p in opp_priors]
    _cd, _ = policy._data()
    sig_sets = _build_sigs(priors)
    rng = random.Random(seed)
    heur = policy.make_agent(deck, go_first=go_first)      # fallback + sub-decisions
    pcache = {}                                             # inferred-prize cache (per game)
    pb_on = bool(_PB and _PB.deck_matches(deck))            # 手册铁律层(T0牌自动启用)
    pbs_on = bool(_PBS and _PBS.deck_matches(deck))         # 海星场地铁律层(带冲浪海滩自动启用)

    adrena_flag = (policy._THREAT_ADRENA[0] if adrena is None else bool(adrena))
    _bmode = "s87" if 861 in deck else "e1"
    td_flag = (policy._TEXT_DMG_ON[0] if text_dmg is None else bool(text_dmg))

    def agent(obs_dict):
        policy._SNIPE[0] = snipe          # per-agent rollout开关(同进程双agent各自切换)
        policy._THREAT_ADRENA[0] = adrena_flag    # 同款逐agent开关:A/B时对手不吃我方的Adrena账
        policy._TEXT_DMG_ON[0] = td_flag  # 同款:文本伤害补账逐agent切换
        policy._BIBLE_MODE[0] = _bmode    # 圣经模式逐agent切换(861在表=s87)
        sel = obs_dict.get("select")
        if stats_out is not None:
            stats_out["last"] = None      # cleared unless a real search runs this call
        if sel is None:
            return list(deck)
        try:
            _update_prize_cache(obs_dict, deck, pcache)
            cards, attacks = policy._data()
            n = len(sel["option"])
            sbi = bool(obs_dict.get("search_begin_input"))
            if pb_on and sel.get("context") in (int(SC.TO_HAND), int(SC.LOOK)) and n >= 1:
                obs_cls0 = to_observation_class(obs_dict)     # 检索铁律:开局优先抓引擎件
                pk = _PB.fetch_pick(obs_cls0)
                if pk is not None:
                    return [pk]
            if pb_on and sel.get("context") == 1 and n >= 1:   # SETUP_ACTIVE:手册=利欧路打头
                obs_cls0 = to_observation_class(obs_dict)
                pk = _PB.setup_active_pick(obs_cls0.select.option,
                                           obs_cls0.current.players[obs_cls0.current.yourIndex].hand or [])
                if pk is not None:
                    return [pk]
            if sel.get("context") != _MAIN or n < 2 or not sbi:
                res = heur(obs_dict)
                if pbs_on and sel.get("context") == _MAIN and n >= 2:
                    obs_cls0 = to_observation_class(obs_dict)   # 场地铁律也管启发式兜底路径
                    pk = _PBS.main_pick(obs_cls0, cards)
                    if pk is not None:
                        return pk
                    v = _PBS.veto_indices(obs_cls0, cards)
                    if v and res and res[0] in v:
                        alt = [i for i in range(n) if i not in v]
                        if alt:
                            alt.sort(key=lambda i: -policy._score_main(
                                obs_cls0.select.option[i], obs_cls0, cards, attacks))
                            return [alt[0]]
                return res

            obs_cls = to_observation_class(obs_dict)
            if pb_on:                                       # 开局铁律:命中脚本则不进树
                pk = _PB.opening_pick(obs_cls, cards)
                if pk is not None:
                    return pk
            if pbs_on:                                      # 场地铁律:顶对手场地/毒猴先手节日广场
                pk = _PBS.main_pick(obs_cls, cards)
                if pk is not None:
                    return pk
            opts = obs_cls.select.option
            scored = sorted(range(n), key=lambda i: -policy._score_main(opts[i], obs_cls, cards, attacks))
            top_score = policy._score_main(opts[scored[0]], obs_cls, cards, attacks)
            if top_score >= 30000:                    # lethal / forced → take it, skip search
                # ...unless the Bench is empty and a Basic can still be benched: benching is free
                # (does not end the turn), so this same lethal is still available on the next
                # decision, while attacking into an empty Bench loses on the spot if the Active
                # trades. Strict-dominance reorder, not a strategy override.
                _bi = (policy._bench_basic_idx(scored, opts, obs_cls, cards, allow_fetch=True)
                       if policy._BENCH_GUARD[0] else None)
                if _bi is not None:
                    return [_bi]
                return [scored[0]]

            # LawlietT1手册(10/10人工裁决一致):板凳为空时先把身体铺出来——直接放基础怪,
            # 或用好伙伴宝芬类道具搜两只上板凳;贴能量/场地/道具全部往后。
            # 这些动作都不结束回合,所以不牺牲本回合任何其它操作。
            # 病根:价值网是自对弈训练的,而自对弈双方都不铺板凳 → 网络学到"空板凳没事"(自举陷阱),
            # 光把铺场塞进候选(_CAND_BENCH)不够,树仍会否决它,故此处直接接管。
            if policy._DEPLOY_FIRST[0]:
                _di = policy._bench_basic_idx(scored, opts, obs_cls, cards, allow_fetch=True)
                if _di is not None:
                    return [_di]
            if pbs_on:                                # 场地铁律否决:别顶自己/别对钢龙先手拍
                _veto = _PBS.veto_indices(obs_cls, cards)
                if _veto and len(_veto) < len(scored):
                    scored = [i for i in scored if i not in _veto]

            me = obs_dict["current"]["yourIndex"]
            opp_guess = _recognize(obs_dict, priors, sig_sets)
            net_use = net
            if net_map and opp_guess is not None:            # MoE:认出对手→换专属脑
                for _pi, _pl in enumerate(priors):
                    if _pl is opp_guess:
                        net_use = net_map.get(_pi, net)
                        break
            # Fuel gauge: the ladder gives a per-agent time BANK (remainingOverageTime,
            # ~600s; hitting 0 = instant loss). Spend it proportionally — always keep a
            # 60s reserve untouched, throttle down as it drains. `budget` acts as the
            # per-decision CAP. Locally (no bank field) the cap is the fixed budget.
            b = budget
            rem = obs_dict.get("remainingOverageTime")
            if isinstance(rem, (int, float)) and rem > 0:
                b = max(0.3, min(budget, (float(rem) - 60.0) / 30.0))
            _t0 = time.time()
            deadline = _t0 + b
            agg = {}                                   # opt -> [total_visits, total_value]
            for _w in range(D):
                if time.time() > deadline:
                    break
                # 时间片按世界均分:旧写法D个世界共用一个总deadline,靠后的世界经常整块被跳过
                # (实测平均只跑2.76/3个世界),PIMC投票残缺且残缺得不均匀,给所有A/B注入方差。
                deadline_w = min(deadline, _t0 + b * (_w + 1) / float(D)) if _PER_WORLD[0] else deadline
                yd, yp, od, op, oh, oa = _determinize(obs_dict, deck, opp_guess, rng, pcache)
                try:
                    root_ss = search_begin(obs_cls, yd, yp, od, op, oh, oa)
                    root = _make_node(root_ss, me, deck, cards, attacks, go_first, top_m, H, net=net_use, pnet=pnet)
                    if root.terminal or not root.children:
                        continue
                    _run_tree(root, me, deck, cards, attacks, go_first,
                              iters, top_m, H, deadline_w, net=net_use, pnet=pnet,
                              two_player=two_player)
                    for ch in root.children:
                        if ch.node is not None:
                            a = agg.setdefault(ch.opt, [0, 0.0])
                            a[0] += ch.node.visit
                            a[1] += ch.node.total
                except Exception as e:
                    _last_err[0] = repr(e)[:90]
                finally:
                    try:
                        search_end()
                    except Exception:
                        pass

            if stats_out is not None:                  # AlphaZero-style soft labels: expose
                tot_v = sum(v[0] for v in agg.values())  # the search's visit ledger + value
                stats_out["last"] = None if not tot_v else {
                    "vis": {int(k): int(v[0]) for k, v in agg.items()},
                    "rv": float(sum(v[1] for v in agg.values()) / tot_v),
                }
            if not agg:                                # search did nothing → heuristic top
                return [scored[0]]
            # most total visits; tie-break by mean value
            best_opt = max(agg, key=lambda k: (agg[k][0], agg[k][1] / max(1, agg[k][0])))
            if _DBG[0] < 8:
                _DBG[0] += 1
                order = sorted(agg.items(), key=lambda kv: -kv[1][0])[:4]
                print(f"[tree] n={n} worlds={D} pick={best_opt} "
                      f"visits={[(k, v[0]) for k, v in order]} err={_last_err[0]}",
                      file=sys.stderr, flush=True)
            # 永不空过:树选了"结束回合",但手上还有能打出伤害的攻击 → 改打。
            # END与ATTACK都交出回合权,ATTACK多一份伤害,严格占优(同"空板凳先铺"逻辑)。
            if policy._NEVER_PASS[0] and best_opt < len(opts) and opts[best_opt].type == OT.END:
                for i in range(len(opts)):
                    if opts[i].type != OT.ATTACK:
                        continue
                    _a = attacks.get(opts[i].attackId)
                    if _a and (_a.damage or 0) > 0:
                        best_opt = i
                        break
            return [best_opt]
        except Exception as e:
            if _DBG[0] < 8:
                _DBG[0] += 1
                print(f"[tree] outer-except -> heuristic: {repr(e)[:90]}", file=sys.stderr, flush=True)
            return heur(obs_dict)

    return agent
