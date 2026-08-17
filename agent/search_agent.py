"""PIMC (determinized Monte-Carlo) search agent for the cabt engine.

At each MAIN decision it shortlists the top candidate actions with the heuristic,
then for each candidate runs K determinized roll-outs (guess the hidden cards,
apply the action, play the game out with the heuristic as the roll-out policy for
both sides) and picks the action with the highest expected win. Non-MAIN
sub-decisions defer to the fast heuristic. Any failure falls back to the
heuristic — the agent never crashes.

`make_search_agent(deck_ids, opp_guess_ids, K, top_m, ...)` -> agent(obs_dict).
`opp_guess_ids` is our prior over the opponent's 60-card list (for the ladder,
pass the meta favourite; for local tests, pass the true opponent list).
"""
import os, random, collections, sys, time

from cg.api import to_observation_class, search_begin, search_step, search_end, OptionType as OT
import policy

_DBG = [0]          # emit a few stderr diagnostics on Kaggle (search running? timing?)
_last_err = [None]  # last search_begin/rollout exception (diagnostic)
_DECISION_BUDGET = 2.0   # seconds; bail a slow PIMC decision to the best-so-far (no timeouts)


# ---------- determinization: fill in the hidden cards ----------
def _pkmn_ids(p, out):
    if not p:
        return
    out.append(p["id"])
    for e in (p.get("energyCards") or []):
        out.append(e["id"])
    for t in (p.get("tools") or []):
        out.append(t["id"])
    for pe in (p.get("preEvolution") or []):
        out.append(pe["id"])


def _board_ids(ps, include_hand):
    out = []
    if include_hand:
        for c in (ps.get("hand") or []):
            out.append(c["id"])
    for p in (ps.get("active") or []):
        _pkmn_ids(p, out)
    for p in ps.get("bench", []):
        _pkmn_ids(p, out)
    for c in ps.get("discard", []):
        out.append(c["id"])
    return out


def _fit(lst, k):
    """Return exactly k ids from lst (truncate or pad by recycling)."""
    if len(lst) >= k:
        return lst[:k]
    out = list(lst)
    i = 0
    while len(out) < k and lst:
        out.append(lst[i % len(lst)]); i += 1
    return out


def _update_prize_cache(obs_dict, our_deck, cache):
    """Lawliet's tabletop trick, engine-grade: whenever a search shows our FULL deck,
    diff it against our 60-list — the leftover multiset is EXACTLY what's locked in
    our prizes. Cached; _determinize then splits deck/prize precisely instead of
    guessing. Cache resets itself when a reused agent starts a new game."""
    try:
        st = obs_dict.get("current") or {}
        me = st.get("yourIndex", 0)
        my = st["players"][me]
        turn = st.get("turn", 0)
        if cache.get("turn", -1) > turn:          # turn went backwards → new game
            cache.pop("P", None)
        cache["turn"] = turn
        dk = (obs_dict.get("select") or {}).get("deck")
        if not dk:
            return
        seen = [c["id"] for c in dk if c]
        if len(seen) != my.get("deckCount", -1):  # partial view (top-N look) → skip
            return
        m = collections.Counter(our_deck)
        for i in _board_ids(my, include_hand=True) + seen:
            if m[i] > 0:
                m[i] -= 1
        left = collections.Counter(m.elements())
        pC = len(my.get("prize") or [])
        n = sum(left.values())
        # n == pC: exact prize set. n == pC+1: prizes + ONE in-limbo card (the trainer
        # being resolved) — a tight SUPERSET; intersecting across sightings converges
        # to the exact set (each sighting's limbo card differs; taken prizes drop out).
        if n == pC:
            cache["P"] = left
        elif n == pC + 1:
            cache["P"] = (cache["P"] & left) if cache.get("P") else left
        else:
            return
        if os.environ.get("CABT_DBG_PRIZE"):
            print(f"[prize] turn={turn} pool={sorted(cache['P'].elements())} "
                  f"(奖{pC}张,池{sum(cache['P'].values())}张)", file=sys.stderr, flush=True)
    except Exception:
        pass


def _determinize(obs_dict, our_deck, opp_guess, rng, pcache=None):
    st = obs_dict["current"]; me = st["yourIndex"]
    my = st["players"][me]; opp = st["players"][1 - me]

    m = collections.Counter(our_deck)
    for i in _board_ids(my, include_hand=True):
        if m[i] > 0:
            m[i] -= 1
    unk_us = [i for i, c in m.items() for _ in range(c)]
    rng.shuffle(unk_us)
    dC, pC = my["deckCount"], len(my["prize"])
    your_deck = your_prize = None
    P = (pcache or {}).get("P")
    if P:                                          # inferred prize multiset available
        pool = list((collections.Counter(unk_us) & P).elements())
        if len(pool) >= pC:
            your_prize = pool if len(pool) == pC else rng.sample(pool, pC)
            rest = list((collections.Counter(unk_us) - collections.Counter(your_prize)).elements())
            rng.shuffle(rest)
            your_deck = _fit(rest, dC)
    if your_deck is None:
        your_deck, your_prize = _fit(unk_us, dC), _fit(unk_us[dC:], pC)

    gm = collections.Counter(opp_guess)
    for i in _board_ids(opp, include_hand=False):
        if gm[i] > 0:
            gm[i] -= 1
    unk_op = [i for i, c in gm.items() for _ in range(c)]
    rng.shuffle(unk_op)
    oD, oH, oP = opp["deckCount"], opp["handCount"], len(opp["prize"])
    opp_deck = _fit(unk_op, oD)
    opp_hand = _fit(unk_op[oD:], oH)
    opp_prize = _fit(unk_op[oD + oH:], oP)

    opp_active = []
    oa = opp.get("active") or []
    if oa and oa[0] is None:                      # face-down active → guess a basic
        opp_active = [opp_guess[0]]
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


_RECOG_V2 = [os.environ.get("CABT_RECOG_V2", "0") == "1"]


def _build_sigs(priors):
    """(独有签名, 全部宝可梦集合) — 独有签名用于精确识别,全集用于同族变体的匹配度打分。"""
    _cd, _ = policy._data()
    psets = [{c for c in p if _cd.get(c) and _cd[c].cardType == 0} for p in priors]
    sigs = [psets[i] - set().union(set(), *[psets[j] for j in range(len(priors)) if j != i])
            for i in range(len(priors))]
    return list(zip(sigs, psets))


def _recognize(obs_dict, priors, sig_sets):
    """Identify the opponent deck from its visible Pokémon.
    旧版只按"该牌组独有的宝可梦"打分 —— 同族变体(路卡两版/铝钢龙两版/凯西两版)的独有集
    互相抵消成空,16套里有5套签名恒空、永远得0分、一律回落prior[0]。而那5套恰好是天梯上
    杀我们最狠的路卡与铝钢龙。v2改为:先按可见宝可梦与该牌组全集的覆盖数打分(同族也能认出),
    独有签名命中作为次级加权,再以牌组宝可梦种类少者(更具体)破平。"""
    if len(priors) == 1:
        return priors[0]
    st = obs_dict.get("current") or {}
    me = st.get("yourIndex", 0)
    opp = st["players"][1 - me]
    vis = set(_board_ids(opp, include_hand=False))
    pairs = [x if isinstance(x, tuple) else (x, x) for x in sig_sets]
    if not _RECOG_V2[0]:
        best_i, best = 0, 0
        for i, (sig, _ps) in enumerate(pairs):
            score = len(vis & sig)
            if score > best:
                best, best_i = score, i
        return priors[best_i]
    best_i, best = 0, ()
    for i, (sig, ps) in enumerate(pairs):
        cover = len(vis & ps)
        if not cover:
            continue
        key = (cover, len(vis & sig), -len(ps))
        if key > best:
            best, best_i = key, i
    return priors[best_i]


def make_search_agent(deck_ids, opp_priors, K=6, top_m=3, rollout_cap=400,
                      go_first=False, seed=12345):
    deck = [int(x) for x in deck_ids]
    # opp_priors = candidate opponent 60-card lists (the known meta decks); we
    # recognise which one we're facing from their visible board each decision.
    if opp_priors and isinstance(opp_priors[0], int):
        opp_priors = [opp_priors]
    priors = [[int(x) for x in p] for p in opp_priors]
    # signature = the Pokémon unique to each prior (used to recognise the opponent)
    _cd, _ = policy._data()
    sig_sets = _build_sigs(priors)
    rng = random.Random(seed)
    heur = policy.make_agent(deck, go_first=go_first)   # fallback + sub-decisions
    pcache = {}                                          # inferred-prize cache (per game)

    def _rollout(searchId, first_pick, cards, attacks):
        """Apply first_pick at the root, then play out heuristically; return 1/0.5/0."""
        me = _rollout.me
        ss = search_step(searchId, first_pick)
        steps = 0
        while steps < rollout_cap:
            o = ss.observation
            cur = o.current
            if cur is not None and cur.result != -1:
                return 1.0 if cur.result == me else 0.0
            sel = o.select
            if sel is None:
                ss = search_step(ss.searchId, list(deck)); steps += 1; continue
            pick = policy._decide(o, sel, sel.option, sel.context,
                                  sel.minCount, sel.maxCount, cards, attacks, go_first)
            pick = [i for i in pick if 0 <= i < len(sel.option)]
            if len(pick) < sel.minCount:
                pick = list(range(sel.minCount))
            ss = search_step(ss.searchId, pick[:max(sel.minCount, sel.maxCount)])
            steps += 1
        return 0.5  # unresolved

    def agent(obs_dict):
        sel = obs_dict.get("select")
        if sel is None:
            return list(deck)
        try:
            _update_prize_cache(obs_dict, deck, pcache)
            cards, attacks = policy._data()
            n = len(sel["option"])
            sbi = bool(obs_dict.get("search_begin_input"))
            # only search real MAIN choices; everything else uses the heuristic
            if sel.get("context") != 0 or n < 2 or not sbi:
                if _DBG[0] < 8 and sel.get("context") == 0:
                    _DBG[0] += 1
                    print(f"[dbg] MAIN n={n} sbi={sbi} -> heuristic", file=sys.stderr, flush=True)
                return heur(obs_dict)

            obs_cls = to_observation_class(obs_dict)
            opts = obs_cls.select.option
            # shortlist candidates by heuristic score...
            scored = sorted(range(n), key=lambda i: -policy._score_main(opts[i], obs_cls, cards, attacks))
            cand = scored[:min(top_m, n)]
            # ...but ALWAYS let search judge every attack (the heuristic mis-values
            # scaling attacks like Myriad Leaf Shower; only a roll-out sees true damage).
            for i in range(n):
                if opts[i].type == OT.ATTACK and i not in cand:
                    cand.append(i)
            top_score = policy._score_main(obs_cls.select.option[cand[0]], obs_cls, cards, attacks)
            if top_score >= 30000:                       # lethal / forced → take it, skip search
                return [cand[0]]

            _rollout.me = obs_dict["current"]["yourIndex"]
            opp_guess = _recognize(obs_dict, priors, sig_sets)
            best_i, best_v = cand[0], -1.0
            t_start = time.time()
            for i in cand:
                if time.time() - t_start > _DECISION_BUDGET:   # never blow the clock
                    break
                tot = 0.0
                for _ in range(K):
                    yd, yp, od, op, oh, oa = _determinize(obs_dict, deck, opp_guess, rng, pcache)
                    try:
                        root = search_begin(obs_cls, yd, yp, od, op, oh, oa)
                        tot += _rollout(root.searchId, [i], cards, attacks)
                    except Exception as e:
                        tot += 0.5
                        _last_err[0] = repr(e)[:90]
                    finally:
                        try:
                            search_end()
                        except Exception:
                            pass
                v = tot / K
                if v > best_v:
                    best_v, best_i = v, i
            if _DBG[0] < 8:
                _DBG[0] += 1
                print(f"[dbg] MAIN n={n} sbi=True SEARCH {len(cand)}cand {time.time()-t_start:.2f}s err={_last_err[0]}",
                      file=sys.stderr, flush=True)
            return [best_i]
        except Exception as e:
            if _DBG[0] < 8:
                _DBG[0] += 1
                print(f"[dbg] outer-except -> heuristic: {repr(e)[:90]}", file=sys.stderr, flush=True)
            return heur(obs_dict)

    return agent
