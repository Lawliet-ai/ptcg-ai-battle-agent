# -*- coding: utf-8 -*-
"""夜循环复盘器: 拉指定提交的最新天梯对局, 分胜负落盘+清单。
用法: .venv/bin/python tools/loop_autopsy.py <submissionId> [N=30] [outdir=story/loop_autopsy/<sub>]
产出: outdir/episode-<id>-replay.json + manifest.json(胜负/座位/对手) + 终端胜负摘要。"""
import sys, os, json, time, subprocess, collections

import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
API = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
PY = os.path.join(ROOT, ".venv", "bin", "python")


def eps(sub):
    for _ in range(4):
        try:
            r = requests.post(API, json={"submissionId": int(sub)}, timeout=30)
            if r.status_code == 200:
                return r.json().get("episodes", [])
        except Exception:
            pass
        time.sleep(2)
    return []


def main():
    sub = int(sys.argv[1])
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "story", "loop_autopsy", str(sub))
    os.makedirs(outdir, exist_ok=True)
    episodes = sorted(eps(sub), key=lambda e: -e["id"])
    manifest, wl = [], collections.Counter()
    for e in episodes[:n_max]:
        eid = e["id"]
        ags = e.get("agents", [])
        seat = next((i for i, g in enumerate(ags) if g.get("submissionId") == sub), None)
        if seat is None:
            continue
        opp = ags[1 - seat] if len(ags) == 2 else {}
        fp = os.path.join(outdir, f"episode-{eid}-replay.json")
        if not os.path.exists(fp):
            subprocess.run([PY, "-m", "kaggle", "competitions", "replay", str(eid), "-p", outdir, "-q"],
                           capture_output=True, text=True)
            time.sleep(0.3)
        win = None
        try:
            d = json.load(open(fp))
            rw = d.get("rewards") or [0, 0]
            win = bool(rw[seat] and rw[seat] >= max(rw))
            if rw[0] == rw[1]:
                win = None
        except Exception:
            pass
        wl["W" if win else ("D" if win is None else "L")] += 1
        manifest.append({"eid": eid, "seat": seat, "win": win,
                         "opp_sub": opp.get("submissionId"),
                         "opp_score": opp.get("updatedScore")})
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"))
    print(f"sub={sub}: {wl['W']}W-{wl['L']}L-{wl['D']}D  (拉取{len(manifest)}局→{outdir})")
    losses = [m for m in manifest if m["win"] is False]
    print("败局eid:", " ".join(str(m["eid"]) for m in losses[:15]))


if __name__ == "__main__":
    main()
