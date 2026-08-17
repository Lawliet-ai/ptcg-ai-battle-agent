"""Decode OUR (toolbox = player 0) moves out of a cabt-viewer replay JSON into a
Chinese, turn-by-turn play log — so a human can critique exactly what our agent
did in the game they're watching, with no whose-turn / English ambiguity.

Card names come from the viewer's full Chinese card DB. Usage:
    python tools/replay_cn.py [replay.json]
Default reads the toolbox_vs_grimm.json the viewer is serving.
"""
import sys, os, json

ROOT = os.path.expanduser("~/Desktop/AI/ptcg-abc")
VIEWER = os.path.join(ROOT, "_repos/cabt-viewer")
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cg.api import all_attack, OptionType as OT, SelectContext as SC

# full Chinese card names (1267 cards) from the viewer's generated DB
_rows = json.load(open(os.path.join(VIEWER, "src/lib/cabt/cardData.generated.json")))
_rows = _rows if isinstance(_rows, list) else list(_rows.values())
NAME = {}
for c in _rows:
    for k in ("cardId", "id"):
        if c.get(k) is not None:
            NAME.setdefault(int(c[k]), c.get("name") or f"#{c[k]}")
ATK = {a.attackId: a for a in all_attack()}

ME = 0  # toolbox is player 0 in replay.py


def cn(cid):
    return NAME.get(cid, f"#{cid}") if cid is not None else "?"


def poke_at(p, area, idx):
    try:
        if area == 4:
            return (p.get("active") or [None])[0]
        if area == 5:
            return p["bench"][idx]
    except Exception:
        return None
    return None


def hand_id(p, i):
    h = p.get("hand") or []
    return h[i]["id"] if (i is not None and 0 <= i < len(h)) else None


def describe(opt, obs):
    """One chosen option -> Chinese phrase (None = skip as noise)."""
    cur = obs["current"]; my = cur["players"][ME]
    ctx = obs["select"].get("context"); t = opt.get("type")
    if t == OT.PLAY:
        return f"打出 {cn(hand_id(my, opt.get('index')))}"
    if t == OT.ATTACH:
        tgt = poke_at(my, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        return f"贴 {cn(hand_id(my, opt.get('index')))} → {cn(tgt['id']) if tgt else '?'}"
    if t == OT.ABILITY:
        p = poke_at(my, opt.get("area"), opt.get("index"))
        return f"用特性:{cn(p['id']) if p else '?'}"
    if t == OT.EVOLVE:
        return f"进化 → {cn(opt.get('cardId'))}"
    if t == OT.ATTACK:
        a = ATK.get(opt.get("attackId"))
        if not a:
            return "⚔ 攻击"
        note = ""
        if a.text and "number of cards in your hand" in a.text.lower():
            my_h = len(my.get("hand") or [])
            opp_h = cur["players"][1 - ME].get("handCount")
            if opp_h is not None and my_h != opp_h:
                note = f"   ⚠️手牌{my_h}≠对手{opp_h}→本招0伤(空气)!"
        return f"⚔ {a.name}({a.damage}){note}"
    if t == OT.RETREAT:
        return "撤退"
    # sub-selections worth showing (indented by caller)
    if ctx in (SC.SWITCH, SC.TO_ACTIVE) and opt.get("cardId") is not None:
        pi = opt.get("playerIndex")
        return (f"↳ Boss 拉出对手 {cn(opt['cardId'])}" if (pi is not None and pi != ME)
                else f"↳ 换上 {cn(opt['cardId'])}")
    if ctx in (SC.SETUP_ACTIVE_POKEMON, SC.SETUP_BENCH_POKEMON, SC.TO_BENCH, SC.TO_FIELD) and opt.get("cardId") is not None:
        return f"↳ 布场 {cn(opt['cardId'])}"
    if ctx in (SC.TO_HAND, SC.TO_HAND_ENERGY, SC.LOOK) and opt.get("cardId") is not None:
        return f"↳ 拿到 {cn(opt['cardId'])}"
    if ctx in (SC.DISCARD, SC.DISCARD_ENERGY, SC.DISCARD_ENERGY_CARD, SC.TO_DECK,
               SC.TO_DECK_BOTTOM) and opt.get("cardId") is not None:
        return f"↳ 弃/回库 {cn(opt['cardId'])}"
    return None


def board(cur):
    my = cur["players"][ME]; opp = cur["players"][1 - ME]
    def one(p):
        a = (p.get("active") or [None])[0]
        if not a:
            return "空"
        return f"{cn(a['id'])}({a.get('hp')}血/{len(a.get('energies',[]))}能)"
    return (f"前排 {one(my)} | 对手 {one(opp)} | "
            f"奖 {len(my.get('prize',[]))}-{len(opp.get('prize',[]))}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        VIEWER, "public/game-logs/toolbox_vs_grimm.json")
    d = json.load(open(path))
    steps = d["steps"]

    cur_turn = None
    lines = []
    for s in steps:
        obs = s["observation"]; cur = obs.get("current") or {}
        sel = obs.get("select")
        if not sel or cur.get("yourIndex") != ME:
            continue
        turn = cur.get("turn")
        opts = sel.get("option") or []
        for idx in (s.get("action") or []):
            if not (0 <= idx < len(opts)):
                continue
            phrase = describe(opts[idx], obs)
            if not phrase:
                continue
            if turn != cur_turn:
                cur_turn = turn
                lines.append(f"\n━━━ 第{turn}回合 · 我方 ━━━  [{board(cur)}]")
            indent = "    " if phrase.startswith("↳") else "  · "
            lines.append(f"{indent}{phrase}")

    # final result
    res = None
    for s in reversed(steps):
        c = s["observation"].get("current") or {}
        if c.get("result", -1) != -1:
            res = c["result"]; break
    print("\n".join(lines))
    print(f"\n=== 结果:{'我方(toolbox)胜' if res == ME else '对手胜' if res is not None else '未决'} ===")


if __name__ == "__main__":
    main()
