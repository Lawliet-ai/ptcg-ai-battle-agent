"""Path-3 data factory: play many fast games (our T0 deck vs the real-field gauntlet,
heuristic pilots both sides for speed) and record every one of OUR MAIN-decision states
with the final game result. This is the training set for the value net that will replace
the tree's hand-written leaf evaluation.

Each sample (jsonl.gz, one per line):
  {"cur": <obs["current"] dict>, "me": <our player index>, "z": 1|0 win/loss, "t": turn}

Usage: python train/selfplay_gen.py --games 20000 --out train/data/shard0 --workers 6
"""
import sys, os, json, gzip, ctypes, argparse, random
import multiprocessing as mp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENGINE, AGENT = os.path.join(ROOT, "engine"), os.path.join(ROOT, "agent")
_G = {}


def _read(p):
    return [int(x) for x in open(p) if x.strip()]


def _init(our_deck, opp_paths):
    sys.path.insert(0, ENGINE); sys.path.insert(0, AGENT)
    import cg.game as game
    from cg.sim import lib, Battle
    import policy
    _G.update(game=game, lib=lib, Battle=Battle, policy=policy,
              our=our_deck, opps=[_read(p) for p in opp_paths])


def _obs():
    sd = _G["lib"].GetBattleData(_G["Battle"].battle_ptr)
    o = json.loads(sd.json.decode())
    o["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return o, sd.selectPlayer


def _one(gi):
    game, policy = _G["game"], _G["policy"]
    opp = _G["opps"][gi % len(_G["opps"])]
    seat = gi % 2
    us = policy.make_agent(_G["our"], go_first=(seat == 0))
    them = policy.make_agent(opp, go_first=(seat == 1))
    d0, d1 = (_G["our"], opp) if seat == 0 else (opp, _G["our"])
    agents = [us, them] if seat == 0 else [them, us]
    game.battle_start(d0, d1)
    obs, sp = _obs()
    states, steps, result = [], 0, -1
    while steps < 6000:
        cur = obs.get("current")
        if cur is not None and cur.get("result", -1) != -1:
            result = cur["result"]; break
        sel = obs.get("select")
        if (sp == seat and sel is not None and sel.get("context") == 0
                and cur is not None):
            states.append({"cur": cur, "t": cur.get("turn", 0)})
        obs = game.battle_select(agents[sp](obs)); obs, sp = _obs(); steps += 1
    game.battle_finish()
    if result not in (0, 1):
        return []
    z = 1 if result == seat else 0
    for s in states:
        s["me"] = seat; s["z"] = z; s["g"] = gi   # game id -> split train/val by GAME
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "data", "shard0"))
    ap.add_argument("--deck", default=os.path.join(ROOT, "agent", "deck_lucario_t0.csv"))
    ap.add_argument("--gauntlet", default=os.path.join(ROOT, "gauntlet_core"))
    ap.add_argument("--mirror", type=int, default=1, help="include T0 mirror as an opponent")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    our = _read(a.deck)
    opps = sorted(os.path.join(a.gauntlet, f) for f in os.listdir(a.gauntlet) if f.endswith(".csv"))
    if a.mirror:
        opps.append(a.deck)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    n_samples = n_games = wins = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(our, opps)) as pool, \
         gzip.open(a.out + ".jsonl.gz", "wt") as fh:
        for states in pool.imap_unordered(_one, range(a.games), chunksize=8):
            n_games += 1
            if states:
                wins += states[0]["z"]
                for s in states:
                    fh.write(json.dumps(s, separators=(",", ":")) + "\n")
                    n_samples += 1
            if n_games % 2000 == 0:
                print(f"{n_games}/{a.games} games, {n_samples} samples, "
                      f"winrate {wins/max(1,n_games):.3f}", flush=True)
    print(f"DONE {n_games} games -> {n_samples} samples -> {a.out}.jsonl.gz", flush=True)


if __name__ == "__main__":
    main()
