# -*- coding: utf-8 -*-
"""带完整记录的对局采集:我方tree vs 对手(mimic .py或heuristic),存成尸检管线同款replay JSON
(steps[[{observation,action}x2]] + rewards + info.TeamNames),death_report/replay_detail直接可用。
用法: .venv/bin/python tools/record_matches.py --opp farmers/lucario950/main.py \
        --opp-deck farmers/lucario950/deck.csv --games 30 --out story/farmer_lucario_games
"""
import sys, os, json, argparse, importlib.util

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "agent"))

import ctypes
import cg.game as game
from cg.sim import lib, Battle
import policy, search_agent_tree


def get_obs():
    sd = lib.GetBattleData(Battle.battle_ptr)
    obs = json.loads(sd.json.decode())
    obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return obs, sd.selectPlayer


def load_mimic(path):
    """同arena_eval:对手模块import时会往cwd写deck.csv,切临时目录加载防污染。"""
    import tempfile
    abs_path = os.path.abspath(path)
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="mimic_")
    try:
        os.chdir(tmp)
        spec = importlib.util.spec_from_file_location("mimic_opp", abs_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod.agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="agent/deck_staryu_s6d.csv")
    ap.add_argument("--opp", required=True)          # 对手main.py路径
    ap.add_argument("--opp-deck", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="agent/value_net_v92b.npz")
    ap.add_argument("--opp-name", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    our = [int(x) for x in open(a.deck) if x.strip()][:60]
    opp = [int(x) for x in open(a.opp_deck) if x.strip()][:60]
    priors_spec = importlib.util.spec_from_file_location("pd", "agent/priors_data.py")
    pm = importlib.util.module_from_spec(priors_spec); priors_spec.loader.exec_module(pm)
    opp_name = a.opp_name or os.path.basename(os.path.dirname(a.opp))

    w = l = 0
    for g in range(a.games):
        seat = g % 2
        us = search_agent_tree.make_tree_agent(our, pm.PRIORS, D=3, iters=200, top_m=4, H=10,
                                               go_first=(seat == 0), budget=6.0,
                                               net=a.net or None, adrena=True)
        them = load_mimic(a.opp)
        agents = [us, them] if seat == 0 else [them, us]
        d0, d1 = (our, opp) if seat == 0 else (opp, our)
        game.battle_start(d0, d1)
        obs, sp = get_obs()
        steps = []
        n = 0
        result = -1
        while n < 6000:
            cur = obs.get("current")
            if cur is not None and cur.get("result", -1) != -1:
                result = cur["result"]
                steps.append([{"observation": obs, "action": None},
                              {"observation": obs, "action": None}])
                break
            act = agents[sp](obs)
            rec = [{"observation": None, "action": None}, {"observation": None, "action": None}]
            rec[sp] = {"observation": obs, "action": act if isinstance(act, list) else list(act)}
            steps.append(rec)
            obs = game.battle_select(act)
            obs, sp = get_obs()
            n += 1
        game.battle_finish()
        if result == -1:
            continue
        my_win = 1 if result == seat else 0
        w += my_win; l += 1 - my_win
        rewards = [0, 0]
        rewards[result] = 1
        names = ["", ""]
        names[seat] = "us"; names[1 - seat] = opp_name
        # steps[1]兼容: 尸检管线从steps[1][a].action读deck
        if len(steps) > 1:
            steps[1][0]["action"] = d0
            steps[1][1]["action"] = d1
        out = {"steps": steps, "rewards": rewards,
               "info": {"TeamNames": names, "EpisodeId": f"local{g:03d}"}}
        tag = "win" if my_win else "loss"
        json.dump(out, open(os.path.join(a.out, f"episode-local{g:03d}{tag}-replay.json"), "w"))
        print(f"game{g}: {'胜' if my_win else '负'} (累计 {w}-{l})", flush=True)
    print(f"DONE {w}-{l} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
