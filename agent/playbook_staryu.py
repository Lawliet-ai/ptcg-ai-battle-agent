"""海星场地铁律层(Lawliet规则,07-10):
R-A 场上有对手的场地 → 拍我们的场地顶掉,优先冲浪海滩。
R-B 对手是毒猴系 且 场上无场地 → 先手拍节日广场(毒免疫)。
其余情况不干预(树自决)。只在 MAIN 且本回合未用过场地时触发。
"""
from cg.api import OptionType as OT

SURFING_BEACH = 1262
FESTIVAL = 1245
OURS = {SURFING_BEACH, FESTIVAL}
_MUNKIDORI_CACHE = None


def deck_matches(deck):
    import os
    if not os.environ.get("CABT_STADIUM_RULE"):  # 默认关闭:A/B判决-12.6(07-10,规则版56.2 vs 68.8)
        return False                             # 病根=硬编码丢时机判断;开发时显式置1启用
    return 1031 in deck and SURFING_BEACH in deck


def _munkidori_ids(cards):
    global _MUNKIDORI_CACHE
    if _MUNKIDORI_CACHE is None:
        _MUNKIDORI_CACHE = {cid for cid, c in cards.items() if "Munkidori" in (c.name or "")}
    return _MUNKIDORI_CACHE


def _play_option_for(opts, hand, cid):
    for i, o in enumerate(opts):
        if o.type == OT.PLAY and o.index is not None and o.index < len(hand):
            if (hand[o.index].id if hand[o.index] else None) == cid:
                return i
    return None


_METAL_CACHE = None


def _metal_ids(cards):
    global _METAL_CACHE
    if _METAL_CACHE is None:
        _METAL_CACHE = {cid for cid, c in cards.items()
                        if "Duraludon" in (c.name or "") or "Archaludon" in (c.name or "")}
    return _METAL_CACHE


def veto_indices(obs_cls, cards):
    """要从树的候选里剔除的选项(Lawliet规则的'不许做'半边):
    ①场上是我们自己的场地→禁止再拍场地(别顶自己)
    ②无场地在场 且 对手是钢龙系→禁止先手拍场地(等它拍了再顶)"""
    cur = obs_cls.current
    if cur is None:
        return set()
    me = cur.yourIndex
    hand = cur.players[me].hand or []
    opts = obs_cls.select.option
    stadium = cur.stadium or []
    stadium_id = stadium[0].id if stadium and stadium[0] else None
    stadium_opts = set()
    for i, o in enumerate(opts):
        if o.type == OT.PLAY and o.index is not None and o.index < len(hand):
            cid = hand[o.index].id if hand[o.index] else None
            if cid in OURS:
                stadium_opts.add(i)
    if not stadium_opts:
        return set()
    if stadium_id in OURS:
        return stadium_opts
    if stadium_id is None:
        opp = cur.players[1 - me]
        opp_cards = [p for p in (opp.active or []) if p] + [p for p in (opp.bench or []) if p]
        opp_ids = {p.id for p in opp_cards} | {c.id for c in (opp.discard or []) if c}
        if opp_ids & _metal_ids(cards):
            return stadium_opts
    return set()


def main_pick(obs_cls, cards):
    cur = obs_cls.current
    if cur is None or cur.stadiumPlayed:
        return None
    me = cur.yourIndex
    my = cur.players[me]
    opp = cur.players[1 - me]
    hand = my.hand or []
    opts = obs_cls.select.option
    stadium = cur.stadium or []
    stadium_id = stadium[0].id if stadium and stadium[0] else None

    # R-A: 对手场地在场 → 顶掉(冲浪海滩优先)
    if stadium_id is not None and stadium_id not in OURS:
        for cid in (SURFING_BEACH, FESTIVAL):
            i = _play_option_for(opts, hand, cid)
            if i is not None:
                return [i]
        return None

    # R-B: 毒猴局 且 无场地 → 先手节日广场
    if stadium_id is None:
        mk = _munkidori_ids(cards)
        opp_cards = [p for p in (opp.active or []) if p] + [p for p in (opp.bench or []) if p]
        opp_ids = {p.id for p in opp_cards} | {c.id for c in (opp.discard or []) if c}
        if opp_ids & mk:
            i = _play_option_for(opts, hand, FESTIVAL)
            if i is not None:
                return [i]
    return None
