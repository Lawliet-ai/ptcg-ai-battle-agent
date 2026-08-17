"""Parallel arena evaluation: pit ONE of our decks (piloted by our agent) against a
GAUNTLET of real field decks, many games, seats swapped, across CPU cores. Reports the
win rate vs each opponent and overall — the fitness signal the self-training loop uses.

Opponents are piloted by the fast heuristic (competent + cheap, and identical across
every candidate so relative comparisons are fair). Our side's pilot is configurable:
  heuristic (fastest, for the broad deck search) | flat | tree (accurate, for final check).

Usage:
  python tools/arena_eval.py <our_deck.csv> [--pilot heuristic|flat|tree]
         [--games 40] [--gauntlet gauntlet] [--workers N] [--seed 0]
Local win rates are noisy (unseeded shuffle) — treat as a relative signal, ladder is judge.
"""
import sys, os, json, ctypes, glob, argparse, random, math
import multiprocessing as mp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENGINE = os.path.join(ROOT, "engine")
AGENT = os.path.join(ROOT, "agent")

_G = {}  # per-worker globals


def _read(path):
    return [int(x) for x in open(path) if x.strip()]


_BASIC_ENERGY = None


def _basic_energy_ids():
    global _BASIC_ENERGY
    if _BASIC_ENERGY is None:
        import csv
        s = set()
        with open(os.path.join(ROOT, "data", "EN_Card_Data.csv"), newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) > 4 and row[4].strip() == "Basic Energy":
                    try:
                        s.add(int(row[0]))
                    except ValueError:
                        pass
        _BASIC_ENERGY = s
    return _BASIC_ENERGY


def _lint_deck(ids, label):
    """名单合法性闸。s7a的5×1030曾让引擎在GetBattleData段错误,720局实验静默死在第49局。"""
    from collections import Counter
    errs = []
    if len(ids) != 60:
        errs.append(f"{len(ids)}张≠60")
    be = _basic_energy_ids()
    for cid, n in Counter(ids).items():
        if n > 4 and cid not in be:
            errs.append(f"卡{cid}×{n}>4")
    if errs:
        raise SystemExit(f"[deck-lint] {label} 名单非法: {'; '.join(errs)} — 拒跑")


def _wilson(w, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h) / d, 100 * (c + h) / d)


def _worker_init(our_deck, pilot, opp_priors, opp_pilot="heuristic", tree_cfg=None,
                 opp_tree_cfg=None, net=None, opp_net=None, snipe=False, opp_snipe=False,
                 pnet=None, opp_pnet=None, top_m_our=0, two_player=False, net_map=None,
                 adrena=False, text_dmg=False, opp_text_dmg=False):
    sys.path.insert(0, ENGINE); sys.path.insert(0, AGENT)
    import cg.game as game
    from cg.sim import lib, Battle
    import policy, search_agent, search_agent_tree
    _G["game"] = game; _G["lib"] = lib; _G["Battle"] = Battle
    _G["policy"] = policy; _G["SA"] = search_agent; _G["ST"] = search_agent_tree
    _G["our_deck"] = our_deck; _G["pilot"] = pilot; _G["opp_priors"] = opp_priors
    _G["opp_pilot"] = opp_pilot
    # 尺子必须与上线一致(main.py: D=3 iters=200 H=10 budget=6.0)。历史教训:默认曾是
    # (3,48,6,1.5)=四分之一搜索深度,所有门控判决都不是在出赛配置下测的。
    _G["tree_cfg"] = tree_cfg or (3, 200, 10, 6.0)
    _G["opp_tree_cfg"] = opp_tree_cfg or _G["tree_cfg"]   # 对手树配置独立(A/B时对手锁基线)
    _G["net"] = net; _G["opp_net"] = opp_net              # 双脑:我方/对手价值网npz路径
    _G["snipe"] = snipe; _G["opp_snipe"] = opp_snipe      # 杠杆三rollout开关
    _G["pnet"] = pnet; _G["opp_pnet"] = opp_pnet          # 双选招头
    _G["top_m_our"] = top_m_our
    _G["net_map"] = net_map; _G["two_player"] = two_player
    _G["adrena"] = adrena                                  # 我方独享Adrena账(对手锁False防污染)
    _G["text_dmg"] = text_dmg; _G["opp_text_dmg"] = opp_text_dmg   # 文本伤害补账逐侧开关
    _G["opp_cache"] = {}


def _get_obs():
    sd = _G["lib"].GetBattleData(_G["Battle"].battle_ptr)
    obs = json.loads(sd.json.decode())
    obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return obs, sd.selectPlayer


def _load_mimic(path):
    """加载对手agent。注意:对手notebook在import时会往当前工作目录写deck.csv
    (farmers/lucario950/main.py 无 try 地 Path("deck.csv").write_text(...)),
    会污染仓库根的 deck.csv 并可能被我们自己的相对路径回退读到。这里切到临时目录加载。"""
    key = "mimic:" + path
    if key not in _G:
        import importlib.util, tempfile
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="mimic_")
        try:
            os.chdir(tmp)
            spec = importlib.util.spec_from_file_location("mimic_mod", abs_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            os.chdir(cwd)
        _G[key] = mod.agent
    return _G[key]


def _make(deck, pilot, go_first, priors, cfg_key="tree_cfg", net_key="net", snipe_key="snipe",
          pnet_key="pnet", td_key="text_dmg"):
    if isinstance(pilot, str) and pilot.endswith(".py"):
        return _load_mimic(pilot)
    if pilot == "policy":
        import policy_net as _POL
        return _POL.make_policy_agent(deck, go_first=go_first, pnet=_G.get("pnet_our"))
    if pilot == "heuristic":
        return _G["policy"].make_agent(deck, go_first=go_first, text_dmg=_G.get(td_key, False))
    if pilot == "bible":
        import bible_agent
        return bible_agent.make_bible_agent(deck, go_first=go_first)
    if pilot == "flat":
        _G["SA"]._DECISION_BUDGET = 1.0
        return _G["SA"].make_search_agent(deck, priors, K=3, top_m=3, go_first=go_first)
    if pilot == "tree":
        D, iters, H, budget = _G[cfg_key]
        tm = _G.get("top_m_our") if net_key == "net" and _G.get("top_m_our") else 4
        tp = bool(_G.get("two_player")) if net_key == "net" else False
        ad = None
        if _G.get("adrena"):
            ad = True if net_key == "net" else False
        # MoE net_map只挂我方侧(net_key=="net"),对手锁None防污染。历史教训:这里曾漏传,
        # --net-map整个是空操作,MoE两轮A/B判的都是空转臂。
        nm = _G.get("net_map") if net_key == "net" else None
        return _G["ST"].make_tree_agent(deck, priors, D=D, iters=iters, top_m=tm, H=H,
                                        go_first=go_first, budget=budget, net=_G.get(net_key),
                                        snipe=bool(_G.get(snipe_key)), pnet=_G.get(pnet_key),
                                        two_player=tp, net_map=nm, adrena=ad,
                                        text_dmg=_G.get(td_key, False))
    raise ValueError(pilot)


def _play(task):
    """task = (opp_path, our_seat). Returns (opp_slug, our_win:0/1/-1 unresolved).
    引擎libcg用std::random_device取熵、不可播种——对局不可复现,--seed只定任务顺序。"""
    opp_path, our_seat = task
    game = _G["game"]
    our = _G["our_deck"]; opp = _read(opp_path)
    slug = os.path.splitext(os.path.basename(opp_path))[0]
    us = _make(our, _G["pilot"], go_first=(our_seat == 0), priors=_G["opp_priors"])
    them = _G["opp_cache"].get(opp_path + str(our_seat))
    if them is None:
        them = _make(opp, _G["opp_pilot"], go_first=(our_seat == 1), priors=[opp],
                     cfg_key="opp_tree_cfg", net_key="opp_net", snipe_key="opp_snipe",
                     pnet_key="opp_pnet", td_key="opp_text_dmg")
        _G["opp_cache"][opp_path + str(our_seat)] = them
    d0, d1 = (our, opp) if our_seat == 0 else (opp, our)
    agents = [us, them] if our_seat == 0 else [them, us]
    obs0, sd0 = game.battle_start(d0, d1)
    if obs0 is None or getattr(sd0, "errorType", 0):
        raise RuntimeError(
            f"battle_start拒绝开局 errorType={getattr(sd0, 'errorType', '?')} opp={slug} "
            f"seat={our_seat} — 名单非法或引擎错误(裸调GetBattleData(NULL)会段错误挂死整个实验)")
    obs, sp = _get_obs(); steps = 0
    last_cur = None
    result = -1
    while steps < 6000:
        cur = obs.get("current")
        if cur is not None:
            last_cur = cur
        if cur is not None and cur.get("result", -1) != -1:
            w = cur["result"]
            result = 1 if w == our_seat else (0 if w == 1 - our_seat else -1)
            break
        act = agents[sp](obs)
        obs = game.battle_select(act); obs, sp = _get_obs(); steps += 1
    try:
        if last_cur is not None:
            ps = last_cur["players"]
            mi = our_seat
            myp = ps[mi]
            bodies = len([x for x in (myp.get("active") or []) if x]) +                      len([x for x in (myp.get("bench") or []) if x])
            print(f"[forensics] slug={slug} win={result} turn={last_cur.get('turn')} "
                  f"bodies={bodies} myprize={len(myp.get('prize') or [])} "
                  f"opprize={len(ps[1-mi].get('prize') or [])}",
                  file=sys.stderr, flush=True)
    except Exception:
        pass
    game.battle_finish()
    return (slug, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--pilot", default="heuristic")
    ap.add_argument("--opp-pilot", default="heuristic")
    ap.add_argument("--games", type=int, default=40)   # games per opponent (split across 2 seats)
    ap.add_argument("--gauntlet", default=os.path.join(ROOT, "gauntlet"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=0,
                    help="只播任务顺序;引擎random_device不可播种,对局不可复现,"
                         "两次运行=两组独立样本,不构成配对A/B")
    ap.add_argument("--only", default="")   # comma substring filter of opponent slugs
    ap.add_argument("--tree-cfg", default="")  # "D,iters,H,budget" for --pilot tree
    ap.add_argument("--opp-tree-cfg", default="")  # 对手树配置;缺省=跟随--tree-cfg
    ap.add_argument("--net", default="")       # 我方价值网npz;缺省=CABT_VALUE_NET
    ap.add_argument("--opp-net", default="")   # 对手价值网npz;缺省=CABT_VALUE_NET
    ap.add_argument("--snipe", action="store_true")      # 杠杆三:我方rollout抓板凳
    ap.add_argument("--opp-snipe", action="store_true")  # 对手同
    ap.add_argument("--pnet", default="")      # 我方选招头npz
    ap.add_argument("--opp-pnet", default="") # 对手选招头npz
    ap.add_argument("--top-m", type=int, default=0)  # 我方树候选宽度覆盖(0=默认4)
    ap.add_argument("--net-map", default="")   # MoE: JSON文件{先验idx: npz路径},仅我方
    ap.add_argument("--two-player", action="store_true")  # 我方用双人树
    ap.add_argument("--adrena", action="store_true")      # Adrena威胁账仅我方开(对手锁False)
    ap.add_argument("--text-dmg", action="store_true")      # 文本伤害补账:我方开
    ap.add_argument("--opp-text-dmg", action="store_true")  # 文本伤害补账:对手开(点亮陪练凯西)
    ap.add_argument("--priors", default="gauntlet",
                    help="'gauntlet'=true opponent lists (oracle) | 'main3'=the 3 stale "
                         "lists shipped in main.py (honest, matches the ladder bundle) | "
                         "comma-separated csv paths")
    a = ap.parse_args()
    tree_cfg = None
    if a.tree_cfg:
        p = a.tree_cfg.split(",")
        tree_cfg = (int(p[0]), int(p[1]), int(p[2]), float(p[3]) if len(p) > 3 else 1.5)

    our_deck = _read(a.deck if os.path.isabs(a.deck) else os.path.join(ROOT, a.deck))
    gdir = a.gauntlet if os.path.isabs(a.gauntlet) else os.path.join(ROOT, a.gauntlet)
    opps = sorted(glob.glob(os.path.join(gdir, "*.csv")))
    if a.only:
        subs = a.only.split(",")
        opps = [o for o in opps if any(s in o for s in subs)]
    if not opps:
        print("no gauntlet decks found"); sys.exit(1)
    if a.priors == "gauntlet":            # oracle priors: the true opponent lists
        priors = [_read(o) for o in opps]
    elif a.priors == "main3":             # honest priors: exactly what the ladder bundle has
        sys.path.insert(0, ENGINE); sys.path.insert(0, AGENT)
        import main as _m
        priors = [list(_m.ABOMASNOW), list(_m.MEGA_STARMIE), list(_m.GRIMMSNARL)]
    elif a.priors.endswith(".py"):        # pool file: a module exposing PRIORS (e.g. agent/priors_data.py)
        import importlib.util as _ilu
        _pp = a.priors if os.path.isabs(a.priors) else os.path.join(ROOT, a.priors)
        _spec = _ilu.spec_from_file_location("prior_pool", _pp)
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        priors = [list(x) for x in _mod.PRIORS]
    else:
        priors = [_read(p) for p in a.priors.split(",")]

    _lint_deck(our_deck, a.deck)
    for o in opps:
        _lint_deck(_read(o), os.path.basename(o))

    tasks = []
    for o in opps:
        for g in range(a.games):
            tasks.append((o, g % 2))
    random.Random(a.seed).shuffle(tasks)

    rec = {os.path.splitext(os.path.basename(o))[0]: [0, 0, 0] for o in opps}  # W,L,unresolved
    with mp.Pool(a.workers, initializer=_worker_init,
                 initargs=(our_deck, a.pilot, priors, a.opp_pilot, tree_cfg,
                           tuple(float(x) if i == 3 else int(x) for i, x in enumerate(a.opp_tree_cfg.split(","))) if a.opp_tree_cfg else None,
                           a.net or None, a.opp_net or None, a.snipe, a.opp_snipe,
                           a.pnet or None, a.opp_pnet or None, a.top_m, a.two_player,
                           ({int(k): v for k, v in __import__("json").load(open(a.net_map)).items()} if a.net_map else None),
                           a.adrena, a.text_dmg, a.opp_text_dmg)) as pool:
        for slug, r in pool.imap_unordered(_play, tasks, chunksize=1):
            if r == 1: rec[slug][0] += 1
            elif r == 0: rec[slug][1] += 1
            else: rec[slug][2] += 1

    print(f"\n=== {os.path.basename(a.deck)}  pilot={a.pilot}  vs gauntlet "
          f"({a.games}/opp, {a.workers} workers) ===")
    print("  [诚实条款] 引擎不可播种:两次运行=两组独立样本;\"双种子同向\"不构成配对证据"
          "(真零效应下双正概率25%)")
    tW = tL = 0
    for slug in sorted(rec, key=lambda s: -(rec[s][0] / max(1, rec[s][0] + rec[s][1]))):
        w, l, u = rec[slug]
        tot = w + l
        wr = 100 * w / tot if tot else 0
        lo, hi = _wilson(w, tot)
        tW += w; tL += l
        print(f"  {slug:44s} {wr:5.1f}%  ({w}-{l}{'  未决%d' % u if u else ''})"
              f"  CI[{lo:.0f},{hi:.0f}]")
    ov = 100 * tW / (tW + tL) if (tW + tL) else 0
    lo, hi = _wilson(tW, tW + tL)
    print(f"  {'—— 总胜率 ——':44s} {ov:5.1f}%  ({tW}-{tL})  CI[{lo:.0f},{hi:.0f}]")

    # 曝光覆盖率: 本池对手占天梯真实曝光的份额(对表story/ladder_exposure.json,家族级映射)
    try:
        expo = json.load(open(os.path.join(ROOT, "story", "ladder_exposure.json")))["expo"]
        fam = json.load(open(os.path.join(ROOT, "story", "gauntlet_expo_map.json")))["families"]
        pool_slugs = [os.path.splitext(os.path.basename(o))[0] for o in opps]
        tot_e = sum(expo.values()); cov = 0; unmapped = set(pool_slugs)
        for f in fam.values():
            hit = [s for s in pool_slugs if any(t in s or s in t for t in f["slugs"])]
            if hit:
                cov += sum(expo.get(k, 0) for k in f["archetypes"])
                unmapped -= set(hit)
        pct = 100 * cov / tot_e if tot_e else 0
        warn = "  ⚠️ <70%: 长尾错误(forecast类)本地不可见,别单凭此尺过闸" if pct < 70 else ""
        print(f"  [曝光覆盖] 本池≈天梯真实曝光的 {pct:.0f}%{warn}")
        if unmapped:
            print(f"  [曝光覆盖] 未映射slug(不计入): {', '.join(sorted(unmapped))}")
    except Exception as e:
        print(f"  [曝光覆盖] 对表失败({e}) — 覆盖率未知")


if __name__ == "__main__":
    main()
