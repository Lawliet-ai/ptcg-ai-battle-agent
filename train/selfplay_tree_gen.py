"""League self-play data factory: BOTH seats piloted by the tree+value-net (the current
champion pilot), so the data distribution matches what the tree actually reaches — and
as the net improves, both players of the next generation improve with it.

Opponent mix: 50% T0 mirror, 50% real-field decks (anchor vs narrow-equilibrium drift).
Records EVERY MAIN decision from the deciding player's own perspective (both seats),
so one game yields two players' worth of labelled states.

Usage: CABT_VALUE_NET=agent/value_net.npz python train/selfplay_tree_gen.py \
           --games 2000 --out train/data/tree_g1 --workers 8
"""
import sys, os, json, gzip, ctypes, argparse
import multiprocessing as mp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENGINE, AGENT = os.path.join(ROOT, "engine"), os.path.join(ROOT, "agent")
_G = {}


def _read(p):
    return [int(x) for x in open(p) if x.strip()]


def _init(our_deck, opp_paths, cfg):
    sys.path.insert(0, ENGINE); sys.path.insert(0, AGENT)
    import cg.game as game
    from cg.sim import lib, Battle
    import search_agent_tree as ST
    _G.update(game=game, lib=lib, Battle=Battle, ST=ST, our=our_deck,
              opps=[_read(p) for p in opp_paths], cfg=cfg)


def _obs():
    sd = _G["lib"].GetBattleData(_G["Battle"].battle_ptr)
    o = json.loads(sd.json.decode())
    o["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return o, sd.selectPlayer


def _one(gi):
    game, ST = _G["game"], _G["ST"]
    D, iters, H, budget = _G["cfg"]
    # 50% mirror, 50% field (both piloted by the SAME champion pilot)
    opp = _G["our"] if gi % 2 == 0 else _G["opps"][(gi // 2) % len(_G["opps"])]
    d0, d1 = (_G["our"], opp) if gi % 4 < 2 else (opp, _G["our"])
    st = [{}, {}]                        # per-seat search-stats hooks (soft labels)
    a0 = ST.make_tree_agent(d0, [d1], D=D, iters=iters, top_m=4, H=H,
                            go_first=True, budget=budget, stats_out=st[0])
    a1 = ST.make_tree_agent(d1, [d0], D=D, iters=iters, top_m=4, H=H,
                            go_first=False, budget=budget, stats_out=st[1])
    agents = [a0, a1]
    game.battle_start(d0, d1)
    obs, sp = _obs()
    states, steps, result = [], 0, -1
    while steps < 6000:
        cur = obs.get("current")
        if cur is not None and cur.get("result", -1) != -1:
            result = cur["result"]; break
        sel = obs.get("select")
        act = agents[sp](obs)
        if sel is not None and sel.get("context") == 0 and cur is not None:
            # policy-head data: option tuples + the pick; PLUS AlphaZero soft labels
            # (search visit distribution + root search value) when a real search ran
            opts = [[o.get("type"), o.get("cardId"), o.get("attackId"), o.get("index")]
                    for o in (sel.get("option") or [])]
            rec = {"cur": cur, "me": sp, "t": cur.get("turn", 0),
                   "opts": opts, "pick": act[0] if act else 0}
            s_last = st[sp].get("last")
            if s_last:
                rec["vis"] = s_last["vis"]; rec["rv"] = s_last["rv"]
            states.append(rec)
        obs = game.battle_select(act); obs, sp = _obs(); steps += 1
    game.battle_finish()
    if result not in (0, 1):
        return []
    for s in states:
        s["z"] = 1 if result == s["me"] else 0
        s["g"] = gi
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "data", "tree_g1"))
    ap.add_argument("--deck", default=os.path.join(ROOT, "agent", "deck_lucario_t0.csv"))
    ap.add_argument("--gauntlet", default=os.path.join(ROOT, "gauntlet_core"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cfg", default="2,48,8,0.6")   # fast tree for data throughput
    ap.add_argument("--shard-games", type=int, default=0,
                    help="每N局关闭并重开一个分片(0=单文件旧行为)。单流写法在进程被杀时"
                         "留下未终结的gzip(08-12重启就这么丢了2/3产料);分片=最多只赔N局。")
    a = ap.parse_args()

    cfg = tuple(float(x) if i == 3 else int(x) for i, x in enumerate(a.cfg.split(",")))
    our = _read(a.deck)
    opps = sorted(os.path.join(a.gauntlet, f) for f in os.listdir(a.gauntlet) if f.endswith(".csv"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    n_samples = n_games = 0
    shards, fh, shard_at = [], None, 0

    def _open_shard():
        nonlocal fh, shard_at
        p = (f"{a.out}.p{len(shards):03d}.jsonl.gz" if a.shard_games else f"{a.out}.jsonl.gz")
        shards.append(p); shard_at = 0
        fh = gzip.open(p, "wt")

    def _close_shard():
        # 必须真正close(): flush只刷缓冲, gzip尾部的CRC+长度在close时才写,
        # 少了它整个档就是"unexpected end of file"。
        nonlocal fh
        if fh is not None:
            fh.close(); fh = None

    _open_shard()
    try:
        with mp.Pool(a.workers, initializer=_init, initargs=(our, opps, cfg)) as pool:
            for states in pool.imap_unordered(_one, range(a.games), chunksize=1):
                n_games += 1; shard_at += 1
                for s in states:
                    fh.write(json.dumps(s, separators=(",", ":")) + "\n")
                    n_samples += 1
                if a.shard_games and shard_at >= a.shard_games:
                    _close_shard()
                    print(f"[ckpt] sealed {shards[-1]} @ {n_games} games / {n_samples} samples",
                          flush=True)
                    _open_shard()
                if n_games % 200 == 0:
                    print(f"{n_games}/{a.games} games, {n_samples} samples", flush=True)
    finally:
        _close_shard()          # Ctrl-C / 异常也要封好最后一片
    # 刚好在分片边界收尾时最后一片是空的(0样本), 别让它进训练名单
    shards = [p for p in shards if os.path.getsize(p) > 0 and any(gzip.open(p, "rt"))]
    print(f"DONE {n_games} games -> {n_samples} samples", flush=True)
    print("TRAIN-WITH: " + ",".join(shards), flush=True)


if __name__ == "__main__":
    main()
