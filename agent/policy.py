"""Generic heuristic policy for the cabt Pokémon-TCG engine.

`make_agent(deck_ids, go_first=False)` returns an `agent(obs_dict) -> list[int]`
that plays ANY deck at a reasonable baseline: it develops the board, respects
energy discipline (no over-fill), takes lethal attacks, and NEVER crashes
(every path falls back to a legal selection). This is the shared scaffolding we
specialise per deck later; it also pilots opponent decks as sparring bots.

Option scoring is driven by the engine's own card/attack data (all_card_data /
all_attack), so it adapts to whatever 60 cards it's handed.
"""
import collections
import math
import os
import re

from cg.api import (to_observation_class, all_card_data, all_attack,
                    OptionType as OT, SelectContext as SC, CardType as CT)

_CARDS = None
_ATTACKS = None
_COLORLESS = 0


def _data():
    global _CARDS, _ATTACKS
    if _CARDS is None:
        _CARDS = {c.cardId: c for c in all_card_data()}
        _ATTACKS = {a.attackId: a for a in all_attack()}
    return _CARDS, _ATTACKS


# ---------- small helpers ----------
def _poke_at(pstate, area, index):
    """Pokémon referenced by an option's (inPlay)area/index on a player's field."""
    try:
        if area == 4:  # ACTIVE
            return pstate.active[0] if pstate.active else None
        if area == 5:  # BENCH
            return pstate.bench[index]
    except Exception:
        return None
    return None


def _max_attack_cost(poke, cards, attacks):
    cd = cards.get(poke.id)
    if not cd:
        return 1
    costs = [len(attacks[a].energies) for a in cd.attacks if a in attacks]
    return max(costs) if costs else 1


_TEXT_DMG_ON = [os.environ.get("CABT_TEXT_DMG", "0") == "1"]
# 文本公式伤害补账: 引擎对这族攻击damage字段恒填0,威胁账全盲(猎捕案#4:凯西唯一胜利
# 手段Powerful Hand在1011局回放出现164局,我们的账里是0伤)。attackId→保守静态估伤;
# 放伤害指示物类(1072)不吃弱点,但我方水系防守面对超系本无×2,静态近似可接受。
_TEXT_DMG = {
    1072: 120,   # Alakazam·Powerful Hand: 2指示物×对手手牌张数,~6张→120
    183:  100,   # Fezandipiti ex·Cruel Arrow: 指定1只100(板凳明文不吃W/R)
    1046: 120,   # Mega Abomasnow ex·Hammer-lanche: 100×翻6张中水能期望~1.2
    1042: 60,    # Kyogre·Riptide: 20×弃牌堆水能,中盘期望~3张
    # 1240(Resentful Refrain)故意不入表: v2实战8W-13L(38%) vs v1 21W-14L(60%)判负。
    # 静态250污染了_best_eff_vs/_live_reach/_lethal的全部前瞻路径——树以为随时能打250,
    # _lethal误触发30000"现在就能击倒", 打不死即灾难。这是"绝对分值插队"病的第四次变形。
    # 只保留_score_main里按真实opp.handCount的动态计价(那里拿得到盘面, 不会说谎)。
}


def _text_dmg(aid):
    return _TEXT_DMG.get(aid, 0) if _TEXT_DMG_ON[0] else 0


def _best_damage(poke, cards, attacks):
    cd = cards.get(poke.id)
    if not cd:
        return 0
    return max((attacks[a].damage or _text_dmg(a) for a in cd.attacks if a in attacks),
               default=0)


def _eff_mult(attacker, defender, cards):
    """Weakness multiplier: 2 if the attacker's type hits the defender's weakness."""
    if not attacker or not defender:
        return 1
    ac = cards.get(attacker.id)
    dc = cards.get(defender.id)
    if ac and dc and dc.weakness is not None and ac.energyType == dc.weakness:
        return 2
    return 1


def _eff_damage(dmg, attacker, defender, cards):
    """Static attack damage after weakness ×2 (the number that decides type-counters)."""
    return dmg * _eff_mult(attacker, defender, cards)


def _attack_damage(atk, my_poke, opp_poke):
    """True printed damage of an attack from the CURRENT board — handles scaling attacks
    whose damage grows with Energy in play (Ogerpon's Myriad Leaf Shower: +30 per Energy on
    BOTH Active Pokémon). Without this the heuristic reads Ogerpon's attack as a flat 30 and
    badly under-values it."""
    if not atk:
        return 0
    dmg = atk.damage or 0
    if not dmg:
        est = _text_dmg(getattr(atk, "attackId", None))
        if est:
            return est          # 文本公式静态估伤,不再走缩放正则
    text = (getattr(atk, "text", "") or "").lower()
    m = re.search(r"does (\d+) more damage for each energy attached to both active", text)
    if m:
        per = int(m.group(1))
        me_e = len(my_poke.energies or []) if my_poke else 0
        op_e = len(opp_poke.energies or []) if opp_poke else 0
        dmg += per * (me_e + op_e)
    return dmg


_SPLASH_RE = re.compile(r"also does (\d+) damage to (?:1 of )?your opponent.s benched")


def _splash_damage(atk):
    """招式对板凳的附带伤害(超级宝石海星ex的喷射冲击:前排120+板凳50,而且只要1个能量)。
    引擎此前只看前排伤害,系统性低估了这类招式。Lawliet裁决#2:'一技能打前排,同时给后排凯西
    打50,一回合杀两只拿两张奖'——卡面核对无误。"""
    if not atk:
        return 0
    m = _SPLASH_RE.search((getattr(atk, "text", "") or "").lower())
    return int(m.group(1)) if m else 0


def _lethal(dmg, attacker, defender, cards):
    """True if effective damage (weakness ×2) KOs the defender at its current HP."""
    if defender is None:
        return False
    return _eff_damage(dmg, attacker, defender, cards) >= defender.hp


def _best_eff_vs(poke, defender, cards, attacks):
    """Best EFFECTIVE damage `poke` can deal `defender` across its attacks (weakness ×2,
    scaling-aware)."""
    cd = cards.get(poke.id) if poke else None
    if not cd:
        return 0
    best = 0
    for a in cd.attacks:
        atk = attacks.get(a)
        if not atk:
            continue
        best = max(best, _eff_damage(_attack_damage(atk, poke, defender), poke, defender, cards))
    return best


def _attack_whiffs(atk, my, opp):
    """True if this attack CURRENTLY does nothing (a known conditional whiff), so taking it
    wastes the turn. Handles Iron Boulder's Adjusted Horn: it deals damage ONLY when your
    hand size equals the opponent's — otherwise it does nothing."""
    text = (getattr(atk, "text", "") or "").lower()
    if "number of cards in your hand" in text:
        mh = len(my.hand) if getattr(my, "hand", None) is not None else None
        oh = getattr(opp, "handCount", None)
        if mh is not None and oh is not None and mh != oh:
            return True
    return False


_WHIFF_BENCH_RE = None


def _whiff_now(atk, atk_side, def_side, cards, stadium=None):
    """通用条件哑火判定:该攻击此刻是否确定无效(只判'确定性+双方可观测'的条件家族,
    掷硬币/弃手牌代价类保守当有效)。卡池26张does-nothing攻击的枚举见战役编年史2026-07-31。
    atk_side/def_side = PlayerState(攻击方/受击方视角)。威胁账用它剔除死威胁——
    杀掉月石后太阳岩的70必须从对手杀伤半径里消失,树才看得见'拆开关'线的价值。"""
    global _WHIFF_BENCH_RE
    text = (getattr(atk, "text", "") or "")
    tl = text.lower()
    if "does nothing" not in tl:
        return False
    try:
        # 家族1: 板凳/在场依赖 "if you don't have X (and Y) on your bench"
        if _WHIFF_BENCH_RE is None:
            import re as _re
            _WHIFF_BENCH_RE = _re.compile(
                r"if you don.t have ([a-z'’ .]+?)(?: and ([a-z'’ .]+?))? on your bench", _re.I)
        m = _WHIFF_BENCH_RE.search(text)
        if m:
            bench_names = {(cards.get(b.id).name.lower() if cards.get(b.id) else "")
                           for b in (atk_side.bench or []) if b}
            for need in (m.group(1), m.group(2)):
                if need and need.strip().lower() not in bench_names:
                    return True
            return False
        # 家族2: 无场地
        if "if there is no stadium in play" in tl:
            return not stadium
        # 家族3: 板凳数下限 (V-Force: 4 or fewer benched -> nothing)
        if "or fewer benched" in tl:
            import re as _re
            m2 = _re.search(r"if you have (\d+) or fewer benched", tl)
            if m2:
                return len([b for b in (atk_side.bench or []) if b]) <= int(m2.group(1))
        # 家族4: 受击方前排状态
        da = def_side.active[0] if def_side.active else None
        if "has no damage counters" in tl and da is not None:
            return (da.hp or 0) >= (cards.get(da.id).hp if cards.get(da.id) else 0)
        if "isn.t a pok" in tl.replace("é", "e") or "isn’t a pokémon {ex}" in tl:
            if da is not None:
                dc = cards.get(da.id)
                return not (dc and (getattr(dc, "ex", False) or getattr(dc, "megaEx", False)))
        if "isn.t burned" in tl or "isn’t burned" in tl:
            return not getattr(def_side, "burned", False)
        # 家族5: 对手奖励数窗口 (Fickle Spitting: exactly 3 or 4 remaining)
        if "exactly 3 or 4 prize cards remaining" in tl:
            return len(def_side.prize or []) not in (3, 4)
        # 家族6: 手牌数恰好N (Seventh Kick; handCount对双方可见)
        if "exactly 7 cards in your hand" in tl:
            hc = getattr(atk_side, "handCount", None)
            if hc is None and getattr(atk_side, "hand", None) is not None:
                hc = len(atk_side.hand)
            return hc is not None and hc != 7
    except Exception:
        return False
    return False


def _live_best_eff(poke, opp_active, my, opp, cards, attacks):
    """Best effective damage `poke` can deal RIGHT NOW — skips attacks that currently whiff,
    so a stuck attacker (Iron Boulder whiffing Adjusted Horn) reads as 0 → we switch it out."""
    cd = cards.get(poke.id) if poke else None
    if not cd:
        return 0
    best = 0
    for aid in cd.attacks:
        a = attacks.get(aid)
        if not a or _attack_whiffs(a, my, opp):
            continue
        best = max(best, _eff_damage(_attack_damage(a, poke, opp_active), poke, opp_active, cards))
    return best


_EVO_FUEL = [os.environ.get("CABT_EVO_FUEL", "1") != "0"]   # 板凳预充能冲进化后招式(A/B +9.3与per_world同包,两对局同向)
_CONSUMER_V2 = [os.environ.get("CABT_CONSUMER_V2", "1") != "0"]   # 睁眼包v2:重写拉人/上前排的消费者(新尺+9/+9)
_deck_evo_map = {}          # 进化前形态名 -> [进化后cardId] (按我方牌组一次性建表)


def _build_evo_map(deck_ids, cards):
    m = {}
    for cid in set(int(x) for x in deck_ids):
        cd = cards.get(cid)
        pre = getattr(cd, "evolvesFrom", None) if cd else None
        if pre:
            m.setdefault(pre, []).append(cid)
    return m


def _live_reach(attacker_ent, def_cd, my, obs, cards, attacks):
    """attacker_ent(场上实体)在当前能量下对def_cd能打出的最大有效伤害;
    本回合还没贴过能量则把手里最大的一贴算进去(老大/换人常在贴能前打出)。"""
    if attacker_ent is None:
        return 0
    ac = cards.get(attacker_ent.id)
    if ac is None:
        return 0
    have = len(getattr(attacker_ent, "energies", None) or [])
    try:
        if not getattr(obs.current, "energyAttached", True):
            bonus = 0
            for h in (my.hand or []):
                bonus = max(bonus, _provides_units(h.id, ac, cards))
            have += bonus
    except Exception:
        pass
    best = 0
    for a in (ac.attacks or []):
        atk = attacks.get(a)
        if not atk or len(atk.energies or []) > have:
            continue
        mult = 2 if (def_cd is not None and def_cd.weakness is not None
                     and ac.energyType == def_cd.weakness) else 1
        best = max(best, (atk.damage or 0) * mult)
    return best


_ZONE_MATH = [os.environ.get("CABT_ZONE_MATH", "0") == "1"]
_NEUTRAL_ZONE = 1247


def _zone_blocks(ac, dc, st):
    """中和地带(1247)卡面: 对手的ex/V攻击对无RuleBox宝可梦伤害为0。
    我方唯一攻击手是超级宝石海星ex, 凯西变体2全队无RuleBox → 场地在场我方输出归零,
    引擎此前不知道这条规则, 会连续打0伤攻击浪费回合(该配对实测15%)。ac=攻击方卡,dc=受击方卡。"""
    if not _ZONE_MATH[0] or ac is None or dc is None:
        return False
    try:
        stad = getattr(st, "stadium", None) or []
        s0 = stad[0] if stad else None
        sid = getattr(s0, "id", None)
        if sid is None and isinstance(s0, dict):
            sid = s0.get("id")
        if sid != _NEUTRAL_ZONE:
            return False
        atk_rb = getattr(ac, "ex", False) or getattr(ac, "megaEx", False)
        def_rb = getattr(dc, "ex", False) or getattr(dc, "megaEx", False)
        return atk_rb and not def_rb
    except Exception:
        return False


_DEATH_FC = [os.environ.get("CABT_DEATH_FORECAST", "0") == "1"]   # 天梯终判569.9 vs v2sight 695(-125)否决; 本地A/B双正(+3.2/+9.4)但上线翻车, 默认关归档
_WALLY, _POTION = 1229, 1112


def _incoming_max(obs, cards, attacks):
    """对手前排下回合对我方前排的最大有效伤害(死亡预报原语)。
    三代案卷三判官独立收敛的头号新病灶: 引擎从不问"我的前排下回合会不会死",
    于是零成本满血奶(小胜)压手到死、能量贴给必死前排陪葬、莉莉艾把救命牌洗走。"""
    try:
        st = obs.current
        me = st.yourIndex
        my = st.players[me]; opp = st.players[1 - me]
        my_a = my.active[0] if my.active else None
        opp_a = opp.active[0] if opp.active else None
        if my_a is None or opp_a is None:
            return 0, None
        eff = _best_eff_vs(opp_a, my_a, cards, attacks)
        # 圣经修正案三(v1.2)·凯西手牌=炮口: Powerful Hand=20×对手手牌数(放指示物, 不吃弱抗),
        # 胡地在对手场上任意位置都算威胁(可promote/换人)。写进眼睛而不是写进嘴——
        # 第0问/戒9/换防全部自动联动, 树自己学会别把Mega递过去。
        if _BIBLE_V4[0]:
            opp_board = [x for x in (opp.active or []) if x] + [b for b in (opp.bench or []) if b]
            if any(p.id == 743 for p in opp_board):
                eff = max(eff, 20 * (getattr(opp, "handCount", 0) or 0))
        return eff, my_a
    except Exception:
        return 0, None


def _build_target(poke, cards, attacks):
    """(colored-need Counter, total-energy-need) for the poke's main (max-damage) attack —
    what we're fuelling toward. Colorless slots accept any energy so are excluded here."""
    cd = cards.get(poke.id) if poke else None
    if not cd:
        return collections.Counter(), 1
    best, bestd = None, -1
    pool = list(cd.attacks)
    # 沿进化线前瞻:板凳上的海星星只有1能的水枪,total_need=1 → _score_main 的
    # "have >= total_need 就 -500" 会拒绝给它贴第2/3个能量,于是永远预充不出
    # 进化后那记3能210的星云光束(名师手册里的"接班人预充能"就是这一手)。
    # 这里把牌组里能从它进化出来的形态的招式也纳入需求上限。
    if _EVO_FUEL[0] and _deck_evo_map:
        for nxt in _deck_evo_map.get(cd.name, ()):
            nc = cards.get(nxt)
            if nc is not None:
                pool.extend(nc.attacks or [])
    for aid in pool:
        a = attacks.get(aid)
        if a and a.damage > bestd:
            bestd, best = a.damage, a
    if best is None:
        return collections.Counter(), 1
    colored = collections.Counter(int(e) for e in best.energies if e != _COLORLESS)
    return colored, len(best.energies)


def _energy_provides(cid, cards):
    """What colour an energy card supplies: an int EnergyType, 'WILD' for special
    energy (Prism etc. → fills any colour), or None if it isn't an energy card."""
    cd = cards.get(cid)
    if not cd:
        return None
    if cd.cardType == CT.SPECIAL_ENERGY:
        return "WILD"
    if cd.cardType == CT.BASIC_ENERGY:
        return int(cd.energyType)
    return None


def _provides_units(cid, tgt_cd, cards):
    """一张能量卡贴到tgt上提供几个单位(点火贴进化怪=3,其余=1;非能量=0)。
    从卡文'provides {X}...'的花括号计数解析,不硬编码卡id。"""
    cd = cards.get(cid)
    if not cd:
        return 0
    if cd.cardType == CT.BASIC_ENERGY:
        return 1
    if cd.cardType == CT.SPECIAL_ENERGY:
        txt = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or []))
        import re as _re
        counts = [len(_re.findall(r"\{[A-Z]\}", seg)) for seg in txt.split("provides")[1:]]
        counts = [c for c in counts if c > 0]
        if not counts:
            return 1
        evolved = tgt_cd is not None and not getattr(tgt_cd, "basic", True)
        return max(counts) if evolved else min(counts)
    return 0


_HAND_RESET = None       # trainer ids that SHUFFLE/DISCARD your whole hand → play them LAST
_ENERGY_FETCH = None     # trainer ids that search/attach Basic Energy → prioritise when starved
_RARE_CANDY = None       # Rare Candy item ids (evolve a Basic straight to Stage 2)


def _card_tags():
    """Classify trainers once from their rules text: hand-reset / energy-fetch / rare-candy."""
    global _HAND_RESET, _ENERGY_FETCH, _RARE_CANDY
    if _HAND_RESET is None:
        cards, _ = _data()
        _HAND_RESET, _ENERGY_FETCH, _RARE_CANDY = set(), set(), set()
        for cid, cd in cards.items():
            if cd.cardType not in (CT.SUPPORTER, CT.ITEM):
                continue
            text = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or [])).lower()
            if "shuffle your hand into your deck" in text or "discard your hand" in text:
                _HAND_RESET.add(cid)
            if "basic energy" in text and ("search your deck" in text or "attach" in text):
                _ENERGY_FETCH.add(cid)
            if (cd.name or "").strip().lower() == "rare candy":
                _RARE_CANDY.add(cid)
    return _HAND_RESET, _ENERGY_FETCH, _RARE_CANDY


_SELF_DISCARD_E = None


def _self_discard_energy(cid, cards):
    """自弃能量(点火能量等):回合结束时会被弃掉。卡面语义,引擎此前不知道。"""
    global _SELF_DISCARD_E
    if _SELF_DISCARD_E is None:
        _SELF_DISCARD_E = set()
        for _cid, cd in cards.items():
            if cd.cardType not in (CT.BASIC_ENERGY, CT.SPECIAL_ENERGY):
                continue
            t = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or [])).lower()
            if "end of your turn" in t and "discard" in t:
                _SELF_DISCARD_E.add(_cid)
    return cid in _SELF_DISCARD_E


def _energy_starved(my, cards, attacks):
    """True when NO in-play Pokémon is within one attach of firing its main attack — i.e.
    energy is the real bottleneck, so digging for / attaching energy is the priority."""
    pokes = []
    if my.active and my.active[0]:
        pokes.append(my.active[0])
    pokes += [b for b in (my.bench or []) if b]
    if not pokes:
        return True
    need_more = min(max(0, _build_target(p, cards, attacks)[1] - len(p.energies or []))
                    for p in pokes)
    return need_more >= 2


# ---------- positional value (search-tree / truncated-rollout leaf) ----------
_THREAT_V3 = [__import__('os').environ.get('CABT_THREAT_V3','0')=='1']  # 威胁账v3:能量就绪度打折(A/B)
_THREAT_V2 = [__import__('os').environ.get('CABT_THREAT_V2','1')!='0']  # 威胁账v2:节奏差+高奖软肉(A/B)
# Adrena记账(反玛俐主刀,但本质是愿增猿卡面数学,对任何带猿牌组成立):
# 带恶能愿增猿每回合可把己方3个伤害指示物搬给对方=+30杀伤半径/只+每回合30血税。
_THREAT_ADRENA = [__import__('os').environ.get('CABT_THREAT_ADRENA','0')=='1']
# 威胁账条件哑火剔除:死条件攻击(月石不在→太阳岩70无效等)不计入对手杀伤半径,
# 拆开关的价值由此进入叶子评估(A/B门控)
_THREAT_WHIFF = [__import__('os').environ.get('CABT_THREAT_WHIFF','0')=='1']
_ADRENA_ID, _DARK_E = 112, 7


def _reach_live(poke, defender, atk_side, def_side, cards, attacks, stadium):
    """威胁账用的杀伤半径:排除当前确定哑火的条件攻击。defender=None时算裸最大伤(板凳目标近似)。"""
    cd = cards.get(poke.id) if poke else None
    if not cd:
        return 0
    best = 0
    for a in cd.attacks:
        atk = attacks.get(a)
        if not atk:
            continue
        if _whiff_now(atk, atk_side, def_side, cards, stadium):
            continue
        if defender is not None:
            best = max(best, _eff_damage(_attack_damage(atk, poke, defender), poke, defender, cards))
        else:
            best = max(best, atk.damage or 0)
    return best


def _adrena_count(field):
    """场上带恶能量的愿增猿数(Adrena-Brain就绪数)"""
    n = 0
    for m in field:
        if m and m.id == _ADRENA_ID and any(int(e) == _DARK_E for e in (m.energies or [])):
            n += 1
    return n


def _prize_worth(poke, cards):
    """击倒这只送对手几张奖(超级ex=3/ex=2/其余=1)。Lawliet:该打谁,看的就是打死送几张奖。"""
    if not poke:
        return 0
    cd = cards.get(poke.id)
    if not cd:
        return 1
    return 3 if getattr(cd, "megaEx", False) else 2 if getattr(cd, "ex", False) else 1


def threat_eval(st, me, cards, attacks):
    """Lawliet的威胁/交换账(当前局面,不搜索、不训练):
    看对手场面→我方这回合能击杀的最高价值目标(送我几张奖) vs 对手下回合能击杀我方的最高价值目标。
    两者之差=这个局面在'即时兑换'维度谁占便宜。归一化成小修正,盖不过奖励卡主导项。"""
    try:
        my = st.players[me]; opp = st.players[1 - me]
        my_a = my.active[0] if my.active else None
        opp_a = opp.active[0] if opp.active else None
        opp_field = ([opp_a] if opp_a else []) + [b for b in (opp.bench or []) if b]
        my_field = ([my_a] if my_a else []) + [b for b in (my.bench or []) if b]
        # v3 能量效率(Lawliet:没能量的大怪不是即时威胁,临时能量不算真战力):
        # 攻击手离主攻开火还差能量时,它的威胁打折——就绪=1.0,差1个=0.5,差2+=0.25。
        def _ready(a):
            if not _THREAT_V3[0] or not a:
                return 1.0
            _, need = _build_target(a, cards, attacks)
            gap = max(0, need - len(a.energies or []))
            return 1.0 if gap == 0 else (0.5 if gap == 1 else 0.25)
        # Adrena就绪数(双向对称;我方牌组没猿时恒0,零影响)
        my_ad = _adrena_count(my_field) if _THREAT_ADRENA[0] else 0
        opp_ad = _adrena_count(opp_field) if _THREAT_ADRENA[0] else 0
        _stad = getattr(st, "stadium", None)
        # 我方前排这一拳(含板凳溅射能力由攻击数据体现)能击杀的对手目标里,送奖最高的
        my_kill = 0
        if my_a:
            for t in opp_field:
                if _THREAT_WHIFF[0]:
                    reach = _reach_live(my_a, t if t is opp_a else None, my, opp, cards, attacks, _stad)
                else:
                    reach = _best_eff_vs(my_a, t, cards, attacks) if t is opp_a else _best_damage(my_a, cards, attacks)
                if _zone_blocks(cards.get(my_a.id), cards.get(t.id), st):
                    reach = 0          # 中和地带: ex打无RuleBox=0伤(威胁账如实记零)
                if reach + 30 * my_ad >= t.hp:
                    my_kill = max(my_kill, _prize_worth(t, cards))
            my_kill *= _ready(my_a)
        # 对手前排下回合能击杀我方目标里,送对手最高的(我方的损失)。
        # 注意:对手侧不做哑火剔除——他的条件(补月石/拍场地/铺满板凳)在他自己回合可修复,
        # "此刻哑火"不等于"下回合无害",剔除会让树敢站进修好条件就死的格子(A/B实证全陪练-3.5)。
        opp_kill = 0
        if opp_a:
            for t in my_field:
                reach = _best_eff_vs(opp_a, t, cards, attacks) if t is my_a else _best_damage(opp_a, cards, attacks)
                if _zone_blocks(cards.get(opp_a.id), cards.get(t.id), st):
                    reach = 0          # 对称: 对面ex打我方无RuleBox(海星星)同样为0
                if reach + 30 * opp_ad >= t.hp:
                    opp_kill = max(opp_kill, _prize_worth(t, cards))
            opp_kill *= _ready(opp_a)
        # 净即时兑换:我方能拿的奖 - 对手能拿的奖,每张奖权重0.04(远小于1/6奖励卡步长)
        tb = 0.04 * (my_kill - opp_kill)
        # Adrena血税:每只就绪猿=每回合30点定向搬伤,是持续性威胁(杀掉猿这笔账立刻消失,
        # 树因此自己会算出"点猴"线,不需要硬编码打谁)
        if _THREAT_ADRENA[0]:
            tb += 0.015 * (min(my_ad, 3) - min(opp_ad, 3))

        if _THREAT_V2[0]:
            # Lawliet"更多看对手场面"扩展:
            # ① 节奏差=谁先开火。我方离主攻能开火差几个能量 vs 对手,快的一方掌握兑换主动权。
            def _fire_gap(a):
                if not a:
                    return 9
                _, need = _build_target(a, cards, attacks)
                return max(0, need - len(a.energies or []))
            gap = _fire_gap(opp_a) - _fire_gap(my_a)     # >0=我方更快开火
            tb += 0.015 * max(-3, min(3, gap))
            # ② 高奖软肉:对手板凳上零能量的高价值ex(老大该抓下来杀的目标),越多我方越有前途
            soft = sum(_prize_worth(b, cards) for b in (opp.bench or [])
                       if b and not (b.energyCards or []) and _prize_worth(b, cards) >= 2)
            tb += 0.02 * min(soft, 3)
        return max(-0.5, min(0.5, tb))
    except Exception:
        return 0.0


def _evaluate(obs, me, cards, attacks):
    """Value of a NON-terminal state from `me`'s perspective, in [-1, 1].

    Dominated by the PRIZE RACE — you win by taking all six of your prizes, so fewer of
    MY prizes left = closer to winning. Immediate-KO threat, board presence, HP material
    and energy development are small tie-breakers, together capped below one prize-step
    (1/6 ≈ 0.167) so they refine leaves of equal prize count but can never flip a clear
    prize lead. This is the leaf estimate the MCTS backs up; terminals are scored ±1 by
    the caller from the engine's `result`."""
    try:
        st = obs.current
        my = st.players[me]; opp = st.players[1 - me]
        my_left = len(my.prize) if my.prize is not None else 6
        opp_left = len(opp.prize) if opp.prize is not None else 6
        if my_left <= 0:
            return 1.0
        if opp_left <= 0:
            return -1.0
        v = (opp_left - my_left) / 6.0                 # dominant term: the prize race

        tb = 0.0
        my_a = my.active[0] if my.active else None
        opp_a = opp.active[0] if opp.active else None
        # who threatens a KO next swing (whiff-aware)
        if my_a and opp_a:
            if _live_best_eff(my_a, opp_a, my, opp, cards, attacks) >= opp_a.hp:
                tb += 0.05
            if _best_eff_vs(opp_a, my_a, cards, attacks) >= my_a.hp:
                tb -= 0.05
        # an empty Active is a crisis (about to be forced to promote / take prizes)
        if my_a is None:
            tb -= 0.06
        if opp_a is None:
            tb += 0.06
        # HP material on board (small)
        my_hp = (my_a.hp if my_a else 0) + sum(b.hp for b in (my.bench or []) if b)
        opp_hp = (opp_a.hp if opp_a else 0) + sum(b.hp for b in (opp.bench or []) if b)
        if my_hp + opp_hp > 0:
            tb += 0.06 * (my_hp - opp_hp) / (my_hp + opp_hp)
        # energy development — are we set up to actually attack?
        my_e = (len(my_a.energies or []) if my_a else 0) + \
               sum(len(b.energies or []) for b in (my.bench or []) if b)
        opp_e = (len(opp_a.energies or []) if opp_a else 0) + \
                sum(len(b.energies or []) for b in (opp.bench or []) if b)
        if my_e + opp_e > 0:
            tb += 0.04 * (my_e - opp_e) / (my_e + opp_e)

        v += max(-0.16, min(0.16, tb))
        return max(-0.99, min(0.99, v))
    except Exception:
        return 0.0


def _main_prior(scores):
    """Softmax the shortlisted MAIN candidate scores into PUCT priors. Scores span a huge
    range (0..30000), so we normalise by the SPREAD (scale-invariant) with temperature
    spread/3 — the top action is favoured but the rest stay explorable, instead of the
    degenerate one-hot a raw softmax would give. Returns a probability list."""
    if not scores:
        return []
    mx = max(scores)
    spread = mx - min(scores)
    if spread <= 1e-9:
        return [1.0 / len(scores)] * len(scores)
    T = spread / 3.0
    ex = [math.exp((s - mx) / T) for s in scores]
    tot = sum(ex)
    return [e / tot for e in ex]


# ---------- MAIN-phase option scoring ----------
_SNIPE = [False]   # 杠杆三:rollout对手会抓板凳(make_tree_agent(snipe=)按agent切换)

# 空板凳守卫(接管式);⚠️天梯判负后默认关(582/604时代),别被旧注释骗
_BENCH_GUARD = [os.environ.get("CABT_BENCH_GUARD", "0") == "1"]
# 自弃能量守卫(点火能量类):默认开
_SELF_DISCARD_GUARD = [os.environ.get("CABT_SDE_GUARD", "0") == "1"]
# 铺场优先(LawlietT1手册:板凳没满时,能把基础怪放上板凳的检索件压过贴能量/场地):默认开
_DEPLOY_FIRST = [os.environ.get("CABT_DEPLOY_FIRST", "0") == "1"]
# 永不空过(有伤害可打就别结束回合):默认开
_NEVER_PASS = [os.environ.get("CABT_NEVER_PASS", "0") == "1"]
# 100败尸检修正一:板凳空时,检索件选卡优先捞基础怪(ep87579621三张高级球打完板凳仍空)
_FETCH_BASIC = [os.environ.get("CABT_FETCH_BASIC", "1") != "0"]   # 平反复开:19败局取证零开火,winray低分系小样本+安置噪声,案卷铁证仍在(ep89080155)
# 100败尸检修正二:对手已到斩杀线(杀我这只就凑满奖)时,promote低奖尸体挡刀而非把胜负点送上前排
# 3种子A/B过闸(80.4→83.5,零负种子),默认开
_SHIELD_PROMOTE = [os.environ.get("CABT_SHIELD_PROMOTE", "1") != "0"]
# 二代尸检修正:进化克制——进化让对手"还需击杀次数"降到≤2时罚分(ep89018754 5:0全Mega阵被两刀翻杀)
_EVOLVE_RESTRAINT = [os.environ.get("CABT_EVOLVE_RESTRAINT", "0") == "1"]   # ⚠️截断bug回退(08-02):罚3000把EVOLVE挤出top_m,树根本看不见而非被劝阻(4blades天梯505病根);液障账重设计走真农夫尺后再上
# 二代尸检修正:老大拉人时can_ko必须用"当前能量下可用的招式+目标当前血量"。
# ep88957979铁证:5:5手握老大,能1的Mega被记成"210 Nebula打得动210血拉帝亚斯"(实际只有120可用),
# 放着70血必杀的1奖胜负手不拉,到手的第6奖让掉,下回合被清场。
_GUST_FIX = [os.environ.get("CABT_GUST_FIX", "1") != "0"]   # v3外科版:仅精确必胜置顶,空载配对零回归,默认开


_DEPLOY_ITEM = None


def _is_deploy_item(cid, cards):
    """能把基础宝可梦直接放上板凳的道具(好伙伴宝芬等)——空板凳时唯一的即时补身体手段。"""
    global _DEPLOY_ITEM
    if _DEPLOY_ITEM is None:
        _DEPLOY_ITEM = set()
        for _cid, cd in cards.items():
            if cd.cardType != CT.ITEM:
                continue
            t = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or [])).lower()
            if "onto your bench" in t and "basic" in t:
                _DEPLOY_ITEM.add(_cid)
    return cid in _DEPLOY_ITEM


def _bench_basic_idx(scored, opts, obs, cards, allow_fetch=False):
    """板凳为空时,返回能立刻补身体的选项:直接放基础怪 >(allow_fetch)好伙伴宝芬类铺场道具。
    两者都不结束回合,所以是严格占优的重排序,不是在替树做战略判断。
    LawlietT1手册(10/10裁决一致):板凳空时先把身体铺出来,贴能量/场地/道具都往后放。"""
    st = obs.current
    my = st.players[st.yourIndex]
    if my.bench:                                  # 板凳有人 → 不干预
        return None
    fetch_i = None
    for i in scored:
        o = opts[i]
        if o.type != OT.PLAY:
            continue
        try:
            cd = cards.get(my.hand[o.index].id) if my.hand else None
        except Exception:
            cd = None
        if not cd:
            continue
        if cd.cardType == CT.POKEMON and cd.basic:
            return i                              # 直接放基础怪最优先
        if allow_fetch and fetch_i is None and _is_deploy_item(cd.cardId, cards):
            fetch_i = i
    if fetch_i is not None:
        return fetch_i
    return None


def _fuel_attacker_idx(scored, opts, obs, cards, attacks):
    """确保"给会真正开火的攻击手贴能量"这个选项能被树看见。
    截断扫描:被 top_m=4 挤出的动作里 56% 是贴能量——贴能选项极多(每宝可梦×每能量),
    先验分中庸,极易被 14 支援者+18 道具挤出候选。Lawliet裁决反复强调"能量贴给要开火的那只"。
    只返回一个候选(前排优先,其次板凳最强攻击手),做不做由搜索自己算;临时能量(点火)
    在本回合无法开火时不注入(Lawliet#1/#7/#9:贴了也打不出,回合结束就没了)。"""
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    oa = opp.active[0] if opp.active else None

    def _tgt_of(o):
        return _poke_at(my, o.inPlayArea, o.inPlayIndex)

    def _wasteful_tmp(o, tgt):
        # 这一贴是临时能量,且贴完仍开不了火 → 是废贴,不注入
        try:
            eid = my.hand[o.index].id if my.hand else None
        except Exception:
            return False
        cd = cards.get(eid)
        if not cd:
            return False
        txt = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or [])).lower()
        if "end of your turn" in txt and "discard" in txt:
            _, total = _build_target(tgt, cards, attacks)
            return (len(tgt.energies or []) + 1) < total
        return False

    best_i, best_key = None, -1.0
    for i in scored:
        o = opts[i]
        if o.type != OT.ATTACH:
            continue
        tgt = _tgt_of(o)
        if tgt is None:
            continue
        colored_need, total = _build_target(tgt, cards, attacks)
        if len(tgt.energies or []) >= total:          # 已满,别过充
            continue
        if _wasteful_tmp(o, tgt):                      # 临时能量废贴,跳过
            continue
        eff = _best_eff_vs(tgt, oa, cards, attacks) if oa else _best_damage(tgt, cards, attacks)
        front = 40 if o.inPlayArea == 4 else 0        # 前排优先(马上能打)
        key = eff + front + _attacker_value(cards.get(tgt.id), attacks)
        if key > best_key:
            best_key, best_i = key, i
    return best_i


def _lethal_fuel_idx(scored, opts, obs, cards, attacks):
    """尸检修正三:确保"贴一手能量、当回合就能击倒对面前排"的贴能候选被树看见。
    与判负的宽泛 _CAND_FUEL 不同,只在斩杀被能量锁住时才注入:前排存在一招
    弱点修正后伤害≥对面前排当前血量、但现有能量开不了火(ep88409843贴一手点火
    Nebula 210当场拿第6奖获胜,引擎没看见)。贴完到底解不解锁由树的真实模拟判定。"""
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    active = my.active[0] if my.active else None
    oa = opp.active[0] if opp.active else None
    if active is None or oa is None:
        return None
    acd, ocd = cards.get(active.id), cards.get(oa.id)
    if not acd or not ocd:
        return None
    have = len(active.energies or [])
    locked_kill = False
    for aid in (acd.attacks or []):
        a = attacks.get(aid)
        if not a:
            continue
        dmg = a.damage or 0
        if ocd.weakness is not None and acd.energyType == ocd.weakness:
            dmg *= 2
        need = len(a.energies or [])
        # 能杀但现在开不了火,且差距在"一贴可能补上"的范围(点火类特殊能量一贴可顶多枚)
        if dmg >= (oa.hp or 9999) and need > have and need <= have + 3:
            locked_kill = True
            break
    if not locked_kill:
        return None
    for i in scored:                          # 先验序里第一个"贴给前排"的选项
        o = opts[i]
        if o.type == OT.ATTACH and o.inPlayArea == 4:
            return i
    return None


def _stadium_idx(scored, opts, obs, cards):
    """确保"打出场地牌"能被树看见。截断扫描:冲浪海滩(20次)+战斗牢笼(14次)被 top_m=4 截断
    ——场地先验仅6000,低于道具/支援者,几乎永不进候选。Lawliet裁决:牢笼"专治胡地(挡板凳放伤)"、
    冲浪海滩"让海星免费换血"。只在场上"还没有我方刚打的这张场地"时注入一个候选,做不做树自己算。"""
    st = obs.current
    my = st.players[st.yourIndex]
    for i in scored:
        o = opts[i]
        if o.type != OT.PLAY:
            continue
        try:
            cd = cards.get(my.hand[o.index].id) if my.hand else None
        except Exception:
            cd = None
        if cd and cd.cardType == CT.STADIUM:
            return i          # 手里最靠前的场地牌;是否覆盖当前场地由搜索评估
    return None


def _retreat_swap_idx(scored, opts, obs, cards, attacks):
    """确保"退却换上更强攻击手"被树看见。截断扫描:退却297次被 top_m=4 截断(先验极低,因退却弃能量)。
    现有逻辑只在前排完全打不动(有效伤害=0)时给退却高分,漏了Lawliet#9:前排打得动但很弱
    (海星星20伤),板凳蹲着强得多的打手(大海星潜力200)。只在板凳最强潜力显著超过前排当前
    有效输出时注入一个退却候选,值不值得弃能量换由搜索自己算。"""
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    active = my.active[0] if my.active else None
    oa = opp.active[0] if opp.active else None
    if active is None or not my.bench:
        return None
    front = _live_best_eff(active, oa, my, opp, cards, attacks)
    bench_pot = max((_best_eff_vs(b, oa, cards, attacks) if oa else _best_damage(b, cards, attacks))
                    for b in my.bench if b)
    if bench_pot < front + 100:               # 板凳没有显著更强的打手 → 不注入
        return None
    for i in scored:
        if opts[i].type == OT.RETREAT:
            return i
    return None


# 玛俐名师手册注入包的卡ID(仅当这些卡在我方选项里出现才生效,海星星等其他牌组零影响)
_G_CANDY, _G_IMP, _G_EX, _G_BOSS, _G_STAMP, _G_MUNKI = 1079, 646, 648, 1182, 1080, 112
_DARK = 7


_BIBLE = [os.environ.get("CABT_BIBLE", "1") != "0"]
_BIBLE_V3X = [os.environ.get("CABT_BIBLE_V3X", "0") == "1"]   # v3三件套(检索路由/芭亚果/海滩), 3h判负隔离
_BIBLE_V4 = [os.environ.get("CABT_BIBLE_V4", "0") == "1"]    # v1.2修正案(硬闸+炮口眼+铺场枝), bible5判负(587)隔离
_BIBLE_MODE = ["e1"]   # 圣经模式: e1 | s87(87号双Mega表, 由make_agent按牌表861自动设置)


def _bible87_pack(opts, obs, cards, attacks, st, me, my, opp, front, hand_ids, vis, _find):
    """圣经-87注入包(师承stardom 68局全量生涯, story/圣经-87-v1.md)。只塞视野, 判断归树。"""
    out = []
    board = [x for x in (my.active or []) if x] + [b for b in (my.bench or []) if b]
    board_ids = {p.id for p in board}
    megas = [p for p in board if p.id in (1031, 861)]
    bench_n = len([b for b in (my.bench or []) if b])
    # ①信号/希尔达提前抓Mega(铁令3): 手无双Mega时
    if 1031 not in hand_ids and 861 not in hand_ids:
        for cid in (1145, 1225):
            i = _find(cid, types=(OT.PLAY,))
            if i is not None and i not in out:
                out.append(i)
        # ②萨尔瓦多: 场有可进化底座(找+进化二合一; T8从零到330的范式)
        if board_ids & {1030, 860}:
            i = _find(1189, types=(OT.PLAY,))
            if i is not None and i not in out:
                out.append(i)
    # ③瓦力救场: 任一Mega带伤且前排将死(圣经红线: 第二只Mega绝不送)
    dmg_mega = [p for p in megas if (p.hp or 330) < (330 if p.id == 1031 else 310)]
    if dmg_mega:
        inc, fa = _incoming_max(obs, cards, attacks)
        if fa is not None and inc > 0 and inc >= (fa.hp or 0):
            i = _find(1229, types=(OT.PLAY,))
            if i is not None and i not in out:
                out.append(i)
    # ④海滩: 场地未立(pivot引擎, stardom 87次pivot vs 12次硬撤)
    if not (getattr(st, "stadium", None) or []):
        i = _find(1262, types=(OT.PLAY,))
        if i is not None and i not in out:
            out.append(i)
    # ⑤老大: 本回合真能击倒板凳目标才拖(19次里7次当回合零出手=白烧全组仅2张的点名权)。
    # 判据必须过"当前能量出得了这一招": _best_eff_vs只看最大印刷伤害, 0能海星ex也被当成能打210。
    # 恶毒猴触发器(112)已删——34局命中0次, meta早换成路卡/胡地/暴雪王。
    if front is not None:
        for b in (opp.bench or []):
            if b and (b.id == 112 or _best_eff_vs(front, b, cards, attacks) >= (b.hp or 999)):
                i = _find(1182, types=(OT.PLAY,))
                if i is not None and i not in out:
                    out.append(i)
                break
    # ⑥铺场兑现: 手有海星星/雪童子/宝芬且板凳有空(身体=双线冗余的血液)
    if bench_n < getattr(my, "benchMax", 5):
        for cid in (1030, 860, 1086):
            i = _find(cid, types=(OT.PLAY,))
            if i is not None and i not in out:
                out.append(i)
    return out


def _bible_pack_idx(scored, opts, obs, cards, attacks):
    """圣经四条高置信线的候选注入(与铺场注入同构: 只让树看见, 值不值得做树自己算):
    (e1模式四线; s87模式走_bible87_pack)
    ①萨尔瓦多后手T1连招(Lawliet终审: 直接进化+出手, 全牌库最快暴力)
    ②小胜的体贴救场(第0问: 前排Mega将死时把满血奶塞进树的视野; forecast的教训=不用绝对分插队)
    ③战斗牢笼 vs 多龙家族(封Phantom Dive放伤, 专项胜率58→71)
    ④英雄斗篷 vs 路卡/雪暴(把330→430, 破对手OHKO线)"""
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    front = my.active[0] if my.active else None
    hand_ids = [c.id for c in (my.hand or [])]
    vis = {p.id for p in ([x for x in (opp.active or []) if x]
                          + [b for b in (opp.bench or []) if b])}
    out = []

    def _find(cid, types=None):
        for i in range(len(opts)):
            o = opts[i]
            if types is not None and o.type not in types:
                continue
            if _resolve_cid(o, obs, None if not hasattr(obs, "select") else obs.select) == cid:
                return i
        return None

    if _BIBLE_MODE[0] == "s87":
        return _bible87_pack(opts, obs, cards, attacks, st, me, my, opp, front, hand_ids, vis, _find)

    # ①萨尔瓦多T1连招: 我的首回合+后攻+前排海星星+手有能量
    if getattr(st, "turn", 99) <= 2 and getattr(st, "firstPlayer", me) != me \
            and front is not None and front.id == 1030 \
            and any(c in (3, 17) for c in hand_ids):
        i = _find(1189, types=(OT.PLAY,))
        if i is not None:
            out.append(i)
    # ②小胜救场: 前排是带伤Mega且下回合会死
    if front is not None and front.id == 1031 and (front.hp or 330) < 330:
        inc, _f = _incoming_max(obs, cards, attacks)
        if inc >= (front.hp or 0) and inc < 330:
            i = _find(1229, types=(OT.PLAY,))
            if i is not None:
                out.append(i)
    # ③牢笼 vs 多龙
    if vis & {119, 120, 121}:
        i = _find(1264, types=(OT.PLAY,))
        if i is not None:
            out.append(i)
    # ④斗篷 vs 路卡/雪暴(场上有Mega才值得)
    if (vis & {678, 676, 674, 723, 721}) and any(
            p.id == 1031 for p in ([x for x in (my.active or []) if x]
                                   + [b for b in (my.bench or []) if b])):
        i = _find(1159, types=(OT.PLAY, OT.ATTACH))
        if i is not None:
            out.append(i)
    # ⑤(v2)贴能开斩杀: 前排Mega贴上这张能就够出致死招 → 注入该贴能候选
    opp_front = opp.active[0] if opp.active else None
    if front is not None and front.id == 1031 and opp_front is not None:
        cur_e = len(front.energies or [])
        for i in range(len(opts)):
            o2 = opts[i]
            if o2.type != OT.ATTACH:
                continue
            cid2 = _resolve_cid(o2, obs, None)
            if cid2 == 17 and cur_e >= 0:          # 点火=进化怪3能, Nebula 210直接可用
                if (opp_front.hp or 999) <= 210:
                    out.append(i); break
            if cid2 == 3 and cur_e == 0:           # 1水=Jetting 120(+弱点240)
                eff = _eff_damage(120, front, opp_front, cards)
                if (opp_front.hp or 999) <= eff:
                    out.append(i); break
    # ⑥(v2)老大拉斩杀: 对手板凳有本回合可击倒目标时注入老大候选
    if front is not None and front.id == 1031:
        for b in (opp.bench or []):
            if b and _best_eff_vs(front, b, cards, attacks) >= (b.hp or 999):
                i = _find(1182, types=(OT.PLAY,))
                if i is not None:
                    out.append(i)
                break
    # ⑦(v2)萨尔瓦多找+进化二合一: 任意回合手无Mega且场有海星星(Lawliet终审T2条款)
    if 1031 not in hand_ids and any(
            p.id == 1030 for p in ([x for x in (my.active or []) if x]
                                   + [b for b in (my.bench or []) if b])):
        i = _find(1189, types=(OT.PLAY,))
        if i is not None and i not in out:
            out.append(i)
    # ⑧(v3)芭亚果 vs 超系家族(凯西/恶毒猴): -60定向甲
    if _BIBLE_V3X[0] and vis & {741, 742, 743, 112, 648}:
        i = _find(1164, types=(OT.PLAY, OT.ATTACH))
        if i is not None and i not in out:
            out.append(i)
    # ⑨(v3)冲浪海滩: 我方场地未立时给树看见(免费换血引擎, 先验6000常被截断)
    if _BIBLE_V3X[0] and not (getattr(st, "stadium", None) or []):
        i = _find(1262, types=(OT.PLAY,))
        if i is not None and i not in out:
            out.append(i)
    # ⑩(v4)铺场兑现枝(修正案一): 17败里4败=攥着身体件看独苗被清。手有海星星/宝芬/超球且
    # 板凳有空 → 全部塞进视野(克制家族除外, 但独苗时豁免作废)。
    if not _BIBLE_V4[0]:
        return out
    bench_n = len([b for b in (my.bench or []) if b])
    bench_max = getattr(my, "benchMax", 5)
    board_n = bench_n + len([x for x in (my.active or []) if x])
    restrict4 = bool(vis & {678, 676, 674, 119, 120, 121})
    if bench_n < bench_max and (not restrict4 or board_n <= 1):
        for cid4 in (1030, 1086, 1121):
            i = _find(cid4, types=(OT.PLAY,))
            if i is not None and i not in out:
                out.append(i)
    return out


def _bible_attach_to(obs, sel, opts, cards, attacks):
    """圣经贴能目标学(ctx22, 非MAIN=无搜索兜底): Mega优先; 前排必死时给板凳Mega; 禁炮灰海星星。"""
    inc, front = _incoming_max(obs, cards, attacks)
    dying = front is not None and inc > 0 and inc >= (front.hp or 0)
    best_i, best = 0, -1
    for i, o in enumerate(opts):
        ent = _resolve_obj(o, obs, sel)
        cid = getattr(ent, "id", None) if ent is not None else getattr(o, "cardId", None)
        s = 10
        # 注: 这张表只列海星线是刻意的。v2/v3实测把861/860提到同级 → 胜率60%跌到38%:
        # 能量在两条Mega线之间分散=两条都不成型。stardom本人的贴能也是海星54%/雪妖女25%,
        # 这套牌的经济只养得起一门主炮。别再"平权"。
        if cid == 1031:
            s = 1000
            if dying and getattr(o, "inPlayArea", None) == 4:
                s = 300            # 必死前排降权, 板凳Mega接班
        elif cid == 1030:
            s = 200
        if s > best:
            best, best_i = s, i
    return [best_i]


def _grimm_pack_idx(scored, opts, obs, cards, attacks):
    """玛俐名师手册(40局1100+分胜局汇纂)三条高置信线的候选注入。只让树看见,值不值得做树自己算:
    ①神奇糖果直跳(24/29局首跳t3-t7;Punk Up落地搜5能=这套牌的tempo心脏)
    ②老大拉对手板凳进斩杀圈的ex(10/40局收官手;圈=180+30x我方带恶能愿增猿数)
    ③不公平印章在对手手肥(>=7张)时引爆(6/40局,卡面已限定只在我方刚被KO后可用)"""
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    out = []
    hand_ids = set()
    for h in (my.hand or []):
        try:
            hand_ids.add(h.id)
        except Exception:
            pass
    board_ids = [m.id for m in ((my.active or []) + (my.bench or [])) if m]
    munki_powered = sum(1 for m in ((my.active or []) + (my.bench or []))
                        if m and m.id == _G_MUNKI and any(int(e) == _DARK for e in (m.energies or [])))
    opp_bench = [m for m in (opp.bench or []) if m]
    kill_line = 180 + 30 * munki_powered

    for i in scored:
        o = opts[i]
        if o.type != OT.PLAY:
            continue
        try:
            cd = cards.get(my.hand[o.index].id) if my.hand else None
        except Exception:
            cd = None
        if not cd:
            continue
        if cd.cardId == _G_CANDY and _G_EX in hand_ids and _G_IMP in board_ids:
            out.append(i)
        elif cd.cardId == _G_BOSS:
            for m in opp_bench:
                mc = cards.get(m.id)
                if mc is not None and (getattr(mc, "ex", False) or getattr(mc, "megaEx", False)) \
                        and (m.hp or 999) <= kill_line:
                    out.append(i)
                    break
        elif cd.cardId == _G_STAMP and (opp.handCount or 0) >= 7:
            out.append(i)
    return out


def _score_main(opt, obs, cards, attacks):
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    active = my.active[0] if my.active else None
    opp_active = opp.active[0] if opp.active else None
    t = opt.type

    # 圣经硬闸3(v1.2)·老大0能禁烧: 全组仅1张, 本回合出不了招就别拖人(2次白烧全在败局)。
    if _BIBLE_V4[0] and t == OT.PLAY and getattr(opt, "cardId", None) == 1182:
        _has_e = active is not None and len(active.energies or []) > 0
        _hand_e = any(getattr(c, "id", 0) in (3, 17) for c in (my.hand or []))
        if not _has_e and not _hand_e:
            return 60

    if t == OT.ATTACK:
        atk = attacks.get(opt.attackId)
        if atk and _attack_whiffs(atk, my, opp):
            return 300        # conditional attack that does NOTHING right now → never whiff a turn on it
        # 中和地带在场且我方ex打无RuleBox目标 → 伤害为0,视同哑火(否则树会连续打0伤浪费回合)
        if active is not None and opp_active is not None and \
                _zone_blocks(cards.get(active.id), cards.get(opp_active.id), st):
            return 300
        dmg = _attack_damage(atk, active, opp_active)       # scaling-aware (Ogerpon grows with energy)
        eff = _eff_damage(dmg, active, opp_active, cards)   # weakness ×2 — the counter-type signal
        if active and _lethal(dmg, active, opp_active, cards):
            return 30000 + eff                               # KO now → always
        if dmg == 0:
            return 500        # status/setup "attack" (Thunder Wave, Recovery Net) — only if nothing else
        sp = _splash_damage(atk)
        if sp:
            # 板凳溅射是白捡的伤害:能点死残血板凳(直接拿奖),或为下回合做斩杀铺垫。
            # 若溅射就能击倒对手某只板凳 → 视同击倒线,大幅加权。
            snipe_kill = any(b and (b.hp or 999) <= sp for b in (opp.bench or []))
            return 4000 + eff + sp + (2000 if snipe_kill else 0)
        return 4000 + eff     # real damage, below development; ranked by EFFECTIVE damage

    if t == OT.ABILITY:
        return 9000  # draw / energy-accel abilities are (almost) free value

    if t == OT.EVOLVE:
        # 进化克制(二代尸检ep89018754: 5:0领先把三只海星星全升Mega=场上挂9奖靶,被两刀0→6翻杀;
        # 一代87942741同病): 进化把1奖身体变3奖靶,若因此让对手"还需击杀次数"降到≤2,罚分。
        # 账是精确的: 对手拿满还差opp_needs张,我方板上按送奖降序贪心数几刀凑满。
        if _EVOLVE_RESTRAINT[0]:
            try:
                evo_cd = cards.get(my.hand[opt.index].id) if my.hand else None
            except Exception:
                evo_cd = None
            if evo_cd is not None and (getattr(evo_cd, "megaEx", False) or getattr(evo_cd, "ex", False)):
                opp_needs = len(opp.prize or [])
                board = [m for m in (my.active or []) + (my.bench or []) if m]
                worths = sorted((_prize_worth(m, cards) for m in board), reverse=True)

                def _kills_to_win(ws):
                    s = 0
                    for k, w in enumerate(ws, 1):
                        s += w
                        if s >= opp_needs:
                            return k
                    return 99
                before = _kills_to_win(worths)
                # 进化=删掉被进化那只(按其进化前身的送奖数,通常1)、加进化后的送奖数
                pre_name = getattr(evo_cd, "evolvesFrom", None)
                pre_w = 1
                for m in board:
                    mc = cards.get(m.id)
                    if mc is not None and pre_name and mc.name == pre_name:
                        pre_w = _prize_worth(m, cards)
                        break
                new_ws = list(worths)
                if pre_w in new_ws:
                    new_ws.remove(pre_w)
                new_ws.append(3 if getattr(evo_cd, "megaEx", False) else 2)
                new_ws.sort(reverse=True)
                after = _kills_to_win(new_ws)
                if after < before and after <= 2:
                    return 3000     # 让对手两刀内收官的进化才罚:仍高于纯状态招,树可推翻
        return 8500

    if t == OT.ATTACH:
        tgt = _poke_at(my, opt.inPlayArea, opt.inPlayIndex)
        if tgt is None:
            return 200
        # 圣经修正案二(v1.2)·点火唯一合法姿势: 目标必须是"本回合能打Nebula的前排进化怪"。
        # 贴板凳=弃牌(回合末自弃), 贴基础怪=只值1{C}, 先手T1贴=纯蒸发。30局13次蒸发11次在败局。
        if _BIBLE_V4[0] and getattr(opt, "cardId", None) == 17:
            cd_t = cards.get(tgt.id)
            evolved = cd_t is not None and (getattr(cd_t, "stage1", False)
                                            or getattr(cd_t, "stage2", False)
                                            or getattr(cd_t, "megaEx", False))
            my_first_as_first = getattr(st, "turn", 99) <= 1 and getattr(st, "firstPlayer", -1) == me
            if my_first_as_first or opt.inPlayArea != 4 or not evolved:
                return 50
        # 死亡预报·消费者3(三代案卷local001): 唯一能量贴给下回合必死的前排=能量陪葬。
        # 前排会死+板凳有身位 → 该目标的贴附重罚, 能量自然流向板凳(预充能已会冲进化线)。
        if _DEATH_FC[0] and opt.inPlayArea == 4:
            inc, fa = _incoming_max(obs, cards, attacks)
            if fa is not None and inc >= (fa.hp or 0) and any(b for b in (my.bench or []) if b):
                return 900          # 低于铺场/进化/板凳贴附, 高于纯垃圾动作
        colored_need, total_need = _build_target(tgt, cards, attacks)
        have = len(tgt.energies or [])
        if have >= total_need:
            return -500  # main attack already fuelled → don't over-fill
        prov = None
        try:
            prov = _energy_provides(my.hand[opt.index].id, cards) if my.hand else None
        except Exception:
            prov = None
        have_colors = collections.Counter(int(e) for e in (tgt.energies or []))
        colored_unmet = colored_need - have_colors           # coloured slots still needing THAT colour
        remaining = total_need - have
        colorless_open = remaining - sum(colored_unmet.values())  # slots that accept ANY energy
        # A wrong-colour energy is NOT waste — it legally fills a colorless slot. It's only
        # waste when every remaining slot is a colour this energy can't provide.
        # 自弃能量(点火能量)贴上后本回合仍打不出攻击 → 回合结束它就没了,纯浪费。
        # LawlietT1裁决 #1/#7/#9:"贴了点火能量你也无法攻击,下一个回合就会被丢掉,会浪费掉"。
        if (_SELF_DISCARD_GUARD[0] and my.hand
                and _self_discard_energy(my.hand[opt.index].id, cards)
                and (have + 1) < total_need):
            return 400
        if prov == "WILD" or prov is None or (not colored_need) or (prov in colored_unmet):
            base = 7000       # pays a required colour (or wildcard / no colour requirement)
        elif colorless_open > 0:
            base = 6800       # off-colour energy filling a colorless slot — fine, marginally after colour-matchers
        else:
            base = 400        # can't fill any remaining slot (only colour slots left it can't pay) → real waste
        # Fuel whoever will actually ATTACK — usually the type-counter, wherever it sits.
        # A dead-weight Active has no special claim on our energy: put it on the bench
        # attacker we're bringing up (and once the Active is fuelled, on the NEXT attacker).
        tgt_eff = (_best_eff_vs(tgt, opp_active, cards, attacks) if opp_active
                   else _best_damage(tgt, cards, attacks))
        anchor = _attacker_value(cards.get(tgt.id), attacks)   # intrinsic attacker quality (Ogerpon/ex high)
        # Fuel the BEST attacker, wherever it sits — a benched Iron Leaves ex (180) must out-pull a
        # front Iron Leaves (100). Role dominates; the Active only gets a tiny tie-break (attacks sooner).
        role = min(max(tgt_eff, anchor) // 2, 400)
        soon = 200 if (total_need - have) <= 1 else 0    # finishing a cost → attacker ready next turn
        front = 25 if (opt.inPlayArea == 4) else 0
        return base + role + soon + front

    if t == OT.PLAY:
        cd = None
        try:
            cd = cards.get(my.hand[opt.index].id) if my.hand else None
        except Exception:
            cd = None
        if cd is None:
            return 5000
        _reset, efetch, _rc = _card_tags()
        # When we can't power an attack, a card that FINDS energy jumps the queue
        # (dig for what we actually need, not another Pokémon — see state27).
        starve_bonus = 500 if (cd.cardId in efetch and _energy_starved(my, cards, attacks)) else 0
        if cd.cardType == CT.SUPPORTER:
            snipe_bonus = 0
            if _DEATH_FC[0] and cd.cardId == _WALLY:
                # 小胜的体贴: 超级ex回满+能量回手。死亡预报: 前排Mega下回合被打死且回满后能活 → 跳队
                inc, fa = _incoming_max(obs, cards, attacks)
                if fa is not None:
                    fc = cards.get(fa.id)
                    if fc is not None and getattr(fc, "megaEx", False):
                        maxhp = fc.hp or 330
                        dying = inc >= (fa.hp or 0)
                        heal_saves = maxhp > inc
                        cheap = len(fa.energies or []) <= 1     # 身上能量少=回手代价低
                        if dying and heal_saves and cheap:
                            snipe_bonus += 1500                  # 免费满血奶救3奖, 压过普通支援者
            if _SNIPE[0] and "Boss" in (cd.name or ""):
                # 假想敌加狠:对面板凳有"我方前排一拳能杀"或"零能量高奖"目标 → 老大跳队
                try:
                    me_i = obs.current.yourIndex
                    _my = obs.current.players[me_i]; _op = obs.current.players[1 - me_i]
                    _act = _my.active[0] if _my.active else None
                    if _act:
                        _ac = cards.get(_act.id)
                        _dmg = max((attacks[a].damage for a in (_ac.attacks if _ac else []) if a in attacks), default=0)
                        for _b in (_op.bench or []):
                            if not _b:
                                continue
                            _bc = cards.get(_b.id)
                            _mult = 2 if (_bc and _ac and _bc.weakness == _ac.energyType) else 1
                            _prize = 3 if (_bc and _bc.megaEx) else 2 if (_bc and _bc.ex) else 1
                            _noe = not (_b.energyCards or [])
                            if (_dmg * _mult >= (_b.hp or 999) and (_prize >= 2 or _noe)) or (_noe and _prize >= 2):
                                snipe_bonus = 700
                                break
                except Exception:
                    pass
            return 8800 + starve_bonus + snipe_bonus   # hand-reset ordering handled in _decide MAIN
        if cd.cardType == CT.ITEM:
            if _DEATH_FC[0] and cd.cardId == _POTION:
                inc, fa = _incoming_max(obs, cards, attacks)
                if fa is not None and inc > 0:
                    hp = fa.hp or 0
                    if inc >= hp and hp + 60 > inc:
                        return 9200 + starve_bonus     # 这60血正好把前排从死亡线拉回 → 跳队
            return 8600 + starve_bonus
        if cd.cardType == CT.STADIUM:
            return 6000
        if cd.cardType == CT.TOOL:
            return 6500
        # Pokémon: don't deploy a body that's WEAK to what's across the table — it just gets
        # OHKO'd for prizes. (Latias ex is weak Darkness → never bench it into a Grimmsnarl/Dark
        # deck; giving up 2 prizes for nothing.)
        # 手册·铺场纪律:日月引擎(太阳岩/月石)从手里第一时间拍上板凳——它们是整套
        # 牌的抽牌引擎(编入前审计:0/30局按手册铺场)。
        if cd.cardId in _ENGINE_PAIR:
            return 8700
        if cd.cardType == CT.POKEMON and cd.weakness is not None:
            opp_pk = ([opp.active[0]] if opp.active and opp.active[0] else []) + list(opp.bench or [])
            opp_types = {cards[p.id].energyType for p in opp_pk if p and cards.get(p.id)}
            if cd.weakness in opp_types:
                return -500 if cd.ex else 1500    # weak ex across the table → NEVER deploy (below END)
        # Iron Leaves ex: HOLD it until we can move enough Energy (from Ogerpon's pile) to swing
        # the same turn. Playing it with nothing to move wastes Rapid Vernier (the whole point).
        if _is_rapid_vernier(cd):
            in_play = (list(my.active or []) + list(my.bench or []))
            movable = sum(len(p.energies or []) for p in in_play if p)
            need = min((len(attacks[a].energies) for a in cd.attacks
                        if a in attacks and attacks[a].damage > 0), default=3)
            if movable == 0:
                return -200    # nothing to move → Rapid Vernier does nothing; playing it is pure waste
            if movable < need:
                return 900     # can't fully burst yet → hold in hand, last resort only
        return 7000

    if t == OT.RETREAT:
        # Manual retreat DISCARDS energy (the retreat cost). Prefer a Switch item, or just
        # take a hit and switch next turn — so keep paid retreat a genuine last resort.
        if active is None:
            return -300
        cur_eff = _live_best_eff(active, opp_active, my, opp, cards, attacks)  # whiff-aware
        if cur_eff == 0:
            return 1500        # Active can't hurt anything right now (dead or whiffing) → switch it out
        return -300

    if t == OT.END:
        return 0

    return 100


# ---------- sub-selection handling ----------
# Lawliet手册(T0月石太阳岩,2026-07-06亲授)的引擎件:太阳岩/月石=抽牌引擎,必须
# 最优先铺上板凳(审计:编入前铺场符合率0%)。按卡id生效,谁的牌里有这两张都受益。
_ENGINE_PAIR = {675, 676}          # Lunatone / Solrock
_HARIYAMA, _MAKUHITA = 674, 673   # 力士纪律:幕下没喂能量前别急进化


def _card_value(cid, cards, attacks):
    """Rough usefulness of a card we might search for / protect from discard."""
    if cid in _ENGINE_PAIR:
        return 190                 # 手册:日月引擎优先级最高(检索/上板凳都排前)
    cd = cards.get(cid)
    if not cd:
        return 1
    if cd.cardType in (CT.BASIC_ENERGY, CT.SPECIAL_ENERGY):
        return 2
    if cd.cardType == CT.POKEMON:
        return 50 + _attacker_value(cd, attacks)  # attackers most valuable (ramp engines rate high)
    if cd.cardType == CT.SUPPORTER:
        return 20
    if cd.cardType == CT.ITEM:
        return 15
    return 10


def _best_damage_id(cd, attacks):
    return max((attacks[a].damage for a in cd.attacks if a in attacks), default=0)


def _attacker_value(cd, attacks):
    """How valuable this card is AS AN ATTACKER for setup/search priority. A ramp/self-fuel
    engine (Ogerpon: Teal Dance self-attaches + draws; Myriad Leaf Shower scales) rates high
    despite a low printed base, so the agent actually sets it up and fuels it instead of
    judging it by its turn-1 damage."""
    if not cd:
        return 0
    base = _best_damage_id(cd, attacks)
    atxt = " ".join((attacks[a].text or "") for a in cd.attacks if a in attacks).lower()
    stxt = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or [])).lower()
    scaling = "more damage for each energy" in atxt
    self_engine = ("attach a basic" in stxt and "draw a card" in stxt)
    if scaling or self_engine:
        return max(base, 210)     # premium anchor — worth building around
    # a Pokémon whose only real attack has a HARD condition (Iron Boulder's hand-parity) is
    # unreliable — stop the greedy heuristic from fixating on its high printed damage.
    if "number of cards in your hand" in atxt:
        return base // 3
    return base


def _is_rapid_vernier(cd):
    """Iron Leaves ex-style burst piece: its whole value is the ON-PLAY 'switch it in + move
    any amount of Energy to it' trigger. Deploying it early (setup, or with no energy to move)
    wastes that — hold it in hand until the burst turn."""
    if not cd:
        return False
    stxt = " ".join((getattr(s, "text", "") or "") for s in (cd.skills or [])).lower()
    return "switch it with your active" in stxt and "move any amount of energy" in stxt


def _keep_value(cid, obs, cards, attacks):
    """How much we want to KEEP a card when paying a discard cost (Ultra Ball etc.) — the
    LOWEST-value cards get pitched. Board-aware: energy and the TYPE-COUNTER attacker are
    precious; an off-type / non-counter Pokémon is the cheap thing to ditch (state 20:
    keep Prism + Boss, pitch the Psychic Iron Boulder, not the other way round)."""
    cd = cards.get(cid)
    if not cd:
        return 0
    opp_weak = None
    try:
        me = obs.current.yourIndex
        opp = obs.current.players[1 - me]
        oa = opp.active[0] if opp.active else None
        oc = cards.get(oa.id) if oa else None
        opp_weak = oc.weakness if oc else None
    except Exception:
        opp_weak = None
    if cd.cardType == CT.SPECIAL_ENERGY:
        return 95                                 # Prism — flexible key fuel
    if cd.cardType == CT.BASIC_ENERGY:
        return 75                                 # energy-hungry deck: keep energy
    if cd.cardType == CT.POKEMON:
        counters = (opp_weak is not None and cd.energyType == opp_weak)
        base = (90 if counters else 35) + _attacker_value(cd, attacks) // 20
        # 睁眼后(SEE_CARDS)这个函数才第一次真正生效,随即暴露:海星星36=全牌组最低,
        # 高级球的弃牌代价第一个扔的就是补位基础怪——而"板凳空没怪可上"正是头号死法(10/46)。
        # 身体越少,手里的基础怪越是命;进化体的多余副本则是真便宜。
        if _KEEP_V2[0]:
            try:
                st = obs.current
                my = st.players[st.yourIndex]
                bodies = len([x for x in (my.active or []) if x]) + \
                         len([x for x in (my.bench or []) if x])
                if getattr(cd, "basic", False):
                    if bodies <= 1:
                        return max(base, 92)      # 独苗:补位基础怪仅次于万能能量
                    if bodies == 2:
                        return max(base, 58)
                else:
                    copies = sum(1 for h in (my.hand or []) if getattr(h, "id", None) == cid)
                    if copies >= 2:
                        return min(base, 30)      # 叠不出去的多余进化体=最廉价的代价
                    return max(base, 62)          # 独一份的赢点,别当零钱花
            except Exception:
                return base
        return base
    if cd.cardType == CT.SUPPORTER:
        return 70                                 # Boss / draw — the engine
    if cd.cardType == CT.ITEM:
        return 50
    return 40


def _opt_card_id(opt):
    return opt.cardId


# ⚠️ 2026-08-03 农夫案卷挖出的地基级bug:引擎对"从牌库/弃牌堆/手牌里挑一张"类选项
# 只给 area+index,cardId 恒为 None,真身藏在 sel.deck / player.discard / player.hand。
# 旧代码只在setup阶段解析过手牌索引,检索类(高级球/宝芬/装置3.0/夜间担架/超级进化信号/莉莉艾)
# 全部 cards.get(None)=None → 所有评分分支失效 → 闭眼抓牌。
# 实测:46局真农夫对局里305次检索决策全盲,头号死法"开局独苗被处决"10/46直接源于此。
_SEE_CARDS = [os.environ.get("CABT_SEE_CARDS", "1") != "0"]   # 真农夫尺A/B: 对路卡8.3→39.6, 对闪焰14.6→20.8, 默认开
_KEEP_V2 = [os.environ.get("CABT_KEEP_V2", "0") == "1"]        # 弃牌保留价值board-aware(睁眼后才显形的缺陷)
_SNIPE_TARGET = [os.environ.get("CABT_SNIPE_TARGET", "1") != "0"]   # 放伤/溅射按真实血量与送奖选目标(v2包成员)
_BOARD_SEE = [os.environ.get("CABT_BOARD_SEE", "0") == "1"]   # ⚠️天梯判负回退: fullsight 547 vs eyesopen 664(-116)。下游(挡刀/拉人/弃牌)是在"恒选第0项"的退化行为上调出来的,给真信息反而更差;要用必须先重做那些消费者         # 解析场上(前排/板凳/看牌区)身份;关=只解析牌库手牌弃牌堆(08-03上午版)
_A_DECK, _A_HAND, _A_DISCARD = 1, 2, 3
_A_ACTIVE, _A_BENCH, _A_PRIZE, _A_STADIUM, _A_LOOKING = 4, 5, 6, 7, 12


def _resolve_obj(o, obs, sel):
    """选项指向的实体(卡或场上宝可梦)。引擎所有非MAIN选项都是 area+index+playerIndex,
    cardId 恒为 None,身份分散在六个不同的地方——这是"闭眼打牌"的完整版图:
      牌库=sel.deck | 手牌/弃牌堆=players[pi].hand/.discard | 看牌区=current.looking
      前排/板凳=players[pi].active/.bench(打谁/拉谁/挡刀/贴能/治疗全在这)
    2026-08-03 上午只补了前三个,场上区域仍瞎(老大242次全选0号、打谁118次全砸0号板凳)。"""
    if not _SEE_CARDS[0]:
        return None
    idx = getattr(o, "index", None)
    if idx is None:
        return None
    try:
        area = getattr(o, "area", None)
        st = obs.current
        pi = getattr(o, "playerIndex", None)
        p = st.players[pi if pi is not None else st.yourIndex]
        if area == _A_DECK:
            src = getattr(sel, "deck", None) or []
        elif area == _A_LOOKING:
            src = getattr(st, "looking", None) or []
        elif area == _A_HAND:
            src = p.hand or []
        elif area == _A_DISCARD:
            src = p.discard or []
        elif area in (_A_ACTIVE, _A_BENCH, _A_PRIZE, _A_STADIUM):
            if not _BOARD_SEE[0]:
                return None
            src = (p.active if area == _A_ACTIVE else
                   p.bench if area == _A_BENCH else
                   p.prize if area == _A_PRIZE else
                   getattr(st, "stadium", None)) or []
        else:
            return None
        if not (0 <= idx < len(src)):
            return None
        ent = src[idx]
        # 弃能量/换能量类(ctx26/30/33):area+index指到宝可梦,真正被选中的是它身上的
        # 第energyIndex张能量卡。不下钻就会把"该弃哪张能量"当成"该弃哪只宝可梦"来算。
        ei = getattr(o, "energyIndex", None)
        if ei is not None:
            cards_on = getattr(ent, "energyCards", None) or []
            if 0 <= ei < len(cards_on):
                return cards_on[ei]
            return None
        return ent
    except Exception:
        return None
    return None


def _resolve_cid(o, obs, sel):
    """选项的真实卡ID。解析不出来时回落原cardId,不改变旧行为。"""
    cid = getattr(o, "cardId", None)
    if cid is not None or not _SEE_CARDS[0]:
        return cid
    ent = _resolve_obj(o, obs, sel)
    return getattr(ent, "id", None) if ent is not None else None


def _decide(obs, sel, opts, ctx, mn, mx, cards, attacks, go_first):
    n = len(opts)

    def _cid(o):
        return _resolve_cid(o, obs, sel)

    # 圣经贴能目标学(非MAIN, 无搜索兜底的睁眼级修复), CABT_BIBLE=1时生效
    if _BIBLE[0] and ctx == SC.ATTACH_TO and getattr(obs, "current", None) is not None:
        try:
            return _bible_attach_to(obs, sel, opts, cards, attacks)
        except Exception:
            pass

    # --- Yes/No decisions ---
    if ctx == SC.IS_FIRST:
        # 圣经修正案五(v1.2): 赢掷币一律选后手——后手T1=萨尔瓦多连招唯一窗口+出招权;
        # 先手T1不能攻不能进化, 全牌库最不值钱的回合。掉分一晚可证伪, 翻回即可。
        if _BIBLE[0] and _BIBLE_MODE[0] == "s87":
            want = OT.YES          # 圣经-87铁令1: 先攻33/33(先攻72% vs 后攻46%)
        else:
            want = OT.NO if _BIBLE_V4[0] else (OT.YES if go_first else OT.NO)
        for i, o in enumerate(opts):
            if o.type == want:
                return [i]
        return [0]
    if ctx == SC.MULLIGAN:
        for i, o in enumerate(opts):
            if o.type == OT.NO:
                return [i]
        return [0]
    if ctx in (SC.ACTIVATE, SC.FIRST_EFFECT, SC.COIN_HEAD, SC.MORE_DEVOLVE):
        want = OT.YES  # activate helpful effects; call heads
        for i, o in enumerate(opts):
            if o.type == want:
                return [i]
        return [0]

    # --- Setup: put our best attacker Active, fill the bench ---
    # setup/bench options reference the Pokémon by HAND INDEX, not cardId → resolve via the hand.
    _me_setup = obs.current.yourIndex
    _hand_setup = obs.current.players[_me_setup].hand or []

    def _setup_cid(o):
        i = o.index
        if i is not None and i < len(_hand_setup):
            return _hand_setup[i].id
        return _opt_card_id(o)

    if ctx == SC.SETUP_ACTIVE_POKEMON:
        # 圣经-87铁令2: 闪焰王牌在手必扣前排(9/9); 次序 海星星/雪童子 > 急冻鸟 > 厄诡椪最后
        if _BIBLE[0] and _BIBLE_MODE[0] == "s87":
            # 戒三(v1.1): 闪焰扣前排只在后攻有益。先攻局内 闪焰1W-5L vs 非闪焰10W-2L(p=0.0128,n=18):
            # 先攻本就白扔一个回合, 再叠一个只会50伤的160血肉盾, 首Mega落地推迟一整个自己回合。
            # 只改先攻分支(约1/3的局), 后攻分支一个字不动——上一轮"强制后攻"就是全局改翻的车。
            _pref = {666: 500, 1030: 300, 860: 300, 414: 150, 117: 10}
            _sc = [(_pref.get(_setup_cid(o) or 0, 100), i) for i, o in enumerate(opts)]
            _sc.sort(reverse=True)
            return [_sc[0][1]]
        best, bi = -1e9, 0
        _lead_tank = bool(__import__("os").environ.get("CABT_LEAD_TANK"))  # C①:满场快攻下最肉优先
        for i, o in enumerate(opts):
            cd = cards.get(_setup_cid(o))
            if _lead_tank and cd:
                v = (cd.hp or 0) * 100          # 血量压倒一切(160闪焰>70海星星)
                if cd.ex and cd.weakness == 7:
                    v -= 100000
                if _is_rapid_vernier(cd):
                    v -= 100000
                if v > best:
                    best, bi = v, i
                continue
            v = _attacker_value(cd, attacks) if cd else 0   # ramp engine (Ogerpon) opens as anchor
            # never LEAD with a 2-prize ex that's weak to Darkness — Grimmsnarl (Dark) is 45% of
            # the meta and we can't see the opponent yet; keep such a tech off the front (Latias ex).
            if cd and cd.ex and cd.weakness == 7:
                v -= 100000
            if _is_rapid_vernier(cd):        # Iron Leaves ex: hold for its burst, never open with it
                v -= 100000
            if v > best:
                best, bi = v, i
        return [bi]
    if ctx in (SC.SETUP_BENCH_POKEMON, SC.TO_BENCH, SC.TO_FIELD):
        order = sorted(range(n), key=lambda i: -_card_value(_setup_cid(opts[i]), cards, attacks))
        # Bench the ordinary attackers, but HOLD Iron Leaves ex in hand (setup wastes Rapid
        # Vernier) — only bench it if forced to meet the minimum count.
        nonburst = [i for i in order if not _is_rapid_vernier(cards.get(_setup_cid(opts[i])))]
        burst = [i for i in order if _is_rapid_vernier(cards.get(_setup_cid(opts[i])))]
        chosen = nonburst[:min(mx, n)]
        if len(chosen) < mn:
            chosen += burst[:mn - len(chosen)]
        return chosen

    # --- MAIN: greedy best action (develop first, then lethal/attack) ---
    if ctx == SC.MAIN:
        me = obs.current.yourIndex
        my = obs.current.players[me]
        reset, _ef, rare = _card_tags()
        scored = sorted(range(n), key=lambda i: -_score_main(opts[i], obs, cards, attacks))
        top = opts[scored[0]]

        def _play_cid(o):
            try:
                return my.hand[o.index].id if my.hand else None
            except Exception:
                return None

        # Deferral 1: never shuffle/discard your hand (Lillie's/Carmine) while you can still
        # DEPLOY a Pokémon, evolve, Rare Candy, or attach — spend those first, or you shuffle
        # irreplaceable pieces (the ex you haven't benched, the attacker you just recovered)
        # back into the deck (states 10, 43, 47, 63).
        if top.type == OT.PLAY and _play_cid(top) in reset:
            for i in scored:
                o = opts[i]
                if o.type in (OT.EVOLVE, OT.ATTACH):
                    return [i]
                if o.type == OT.PLAY:
                    oc = _play_cid(o); ocd = cards.get(oc)
                    if oc in rare or (ocd and ocd.cardType == CT.POKEMON):
                        return [i]

        # Deferral 2: ATTACKING ends the turn — first cash in every FREE resource (abilities like
        # Ogerpon's Teal Dance, and fuelling attaches). Never attack with a free ability / good
        # attach still unused (state 67). Both are once-per-turn, so they clear and the attack
        # follows on the next decision.
        if top.type == OT.ATTACK:
            for i in scored:
                o = opts[i]
                if o.type == OT.ABILITY:
                    return [i]
                if o.type == OT.ATTACH and _score_main(o, obs, cards, attacks) >= 6800:
                    return [i]
            # ...and with an empty Bench, benching a Basic outranks attacking: our attacks are
            # flat damage, benching does not end the turn, so the attack is still there next
            # decision — but if the Active is KO'd with nothing to promote, the game ends on
            # the spot regardless of the prize race (98-loss forensics: the single largest
            # loss mode, incl. games lost while ahead 4-0 on prizes).
            bi = _bench_basic_idx(scored, opts, obs, cards, allow_fetch=True) if _DEPLOY_FIRST[0] else None
            if bi is not None:
                return [bi]
        # 永不空过:END 与 ATTACK 都交出回合权,但 ATTACK 带伤害 → 严格占优。
        # (98败复盘:385个"有伤害可打"的回合里142个整回合没打,其中28次主动选END)
        if _NEVER_PASS[0] and top.type == OT.END:
            for i in scored:
                o = opts[i]
                if o.type != OT.ATTACK:
                    continue
                a = attacks.get(o.attackId)
                if a and (a.damage or 0) > 0 and not _attack_whiffs(a, my, opp):
                    return [i]
        return [scored[0]]

    # --- Damage / snipe: hit the opponent's Active (or lowest-HP target) ---
    if ctx in (SC.DAMAGE, SC.DAMAGE_COUNTER, SC.DAMAGE_COUNTER_ANY, SC.EFFECT_TARGET):
        me = obs.current.yourIndex
        opp = obs.current.players[1 - me]
        opp_active_id = opp.active[0].id if opp.active and opp.active[0] else None

        # 这一族=放伤害指示物/溅射打谁(60局157个决策点)。旧实现比的是 o.cardId==前排id,
        # 而所有选项的 cardId 恒为 None → 永远退化成"对手侧的第一个",注释承诺的
        # lowest-HP 逻辑压根没写。有了实体解析后按真账排:能打死的优先(送奖越多越好),
        # 打不死就打离斩杀最近的。
        dmg_amount = (getattr(sel, "remainDamageCounter", 0) or 0) * 10

        def dmg_key(i):
            o = opts[i]
            is_opp = (o.playerIndex is not None and o.playerIndex != me)
            if not _SNIPE_TARGET[0]:
                return (is_opp, (o.cardId == opp_active_id))
            ent = _resolve_obj(o, obs, sel)
            hp = getattr(ent, "hp", None) if ent is not None else None
            kills = 1 if (hp is not None and dmg_amount >= hp) else 0
            worth = _prize_worth(ent, cards) if (ent is not None and kills) else 0
            near = -(hp if hp is not None else 9999)      # 血越少越靠前
            return (is_opp, kills, worth, near)

        order = sorted(range(n), key=lambda i: dmg_key(i), reverse=True)
        k = max(mn, min(mx, n))
        return order[:k] if k > 0 else []

    # --- Discard costs: pitch the least valuable (energy/dupes first) ---
    if ctx in (SC.DISCARD, SC.DISCARD_ENERGY, SC.DISCARD_ENERGY_CARD,
               SC.DISCARD_CARD_OR_ATTACHED_CARD, SC.DISCARD_TOOL_CARD, SC.TO_DECK,
               SC.TO_DECK_BOTTOM, SC.TO_DECK_ENERGY):
        order = sorted(range(n), key=lambda i: _keep_value(_cid(opts[i]), obs, cards, attacks))
        # discard the minimum required, cheapest-to-keep first
        k = mn if mn > 0 else 0
        return order[:k]

    # --- Choose a Pokémon to bring Active (our Switch/retreat) or to gust (Boss) ---
    if ctx in (SC.TO_ACTIVE, SC.SWITCH):
        me = obs.current.yourIndex
        my = obs.current.players[me]; opp = obs.current.players[1 - me]
        our_active = my.active[0] if my.active else None
        opp_active = opp.active[0] if opp.active else None

        def _eff_ids(atk_id, def_id):
            ac = cards.get(atk_id); dc = cards.get(def_id)
            if not ac or not dc:
                return 0
            raw = max((attacks[a].damage for a in ac.attacks if a in attacks), default=0)
            mult = 2 if (dc.weakness is not None and ac.energyType == dc.weakness) else 1
            return raw * mult

        opp_opts = [i for i, o in enumerate(opts)
                    if o.playerIndex is not None and o.playerIndex != me]
        if opp_opts:                      # Boss's Orders — drag up their bench
            remaining = len(my.prize or [])       # KO-prizes we still need to win
            def gust_key(i):
                o = opts[i]
                cid = _cid(o); cd = cards.get(cid)
                if not cd:
                    return (0, 0, 0, 0, 0)
                prize = 3 if cd.megaEx else 2 if cd.ex else 1
                # 睁眼包v2:拉人要看真实体(当前血量/已囤能量/挂了什么工具),不是看卡面上限。
                # 旧版恒选第0项(242/242实测),因为cd永远解析不出来;现在按"这一刀能否收掉、
                # 送几张奖、它已经囤了多少能量(=威胁成型度)"排序。
                ent = _resolve_obj(o, obs, sel) if _CONSUMER_V2[0] else None
                if ent is not None:
                    cur_hp = ent.hp or cd.hp
                    fuel = len(getattr(ent, "energies", None) or [])
                    tools = len(getattr(ent, "tools", None) or [])
                    reach = _live_reach(our_active, cd, my, obs, cards, attacks)
                    can_ko = 1 if reach >= cur_hp else 0
                    gain = min(prize, remaining) if can_ko else 0
                    wins = 1 if (can_ko and gain >= remaining) else 0
                    if can_ko:
                        # 能收:送奖最多优先,其次它囤的能量越多越该现在拆
                        return (wins, 1, prize * 2000 + fuel * 300 + tools * 200, 0, 0)
                    # 收不掉:拖一个"成型度高且退不回去"的上来收税
                    rc = cd.retreatCost or 0
                    dmg_frac = int(100 * reach / max(1, cur_hp))
                    return (0, 0, fuel * 300 + tools * 200 + rc * 150 + dmg_frac, 0, 0)
                can_ko = 1 if (our_active and _eff_ids(our_active.id, cid) >= cd.hp) else 0
                gain = min(prize, remaining) if can_ko else 0
                wins = 1 if (can_ko and gain >= remaining) else 0   # this KO ends the game → prize map
                # 二代尸检ep88957979(5:5让掉第6奖):老can_ko高估射程(不看能量/当前血量),
                # 平均局面里这歪打正着=拉大血牛搁浅(A/B证实全面改精确反而-2.3);
                # 但收官时刻必须精确——"这一拉真能赢"用当前能量可用招式+目标当前血量校验,置顶压过一切。
                if _GUST_FIX[0] and our_active is not None:
                    tgt = _poke_at(opp, o.inPlayArea, o.inPlayIndex)
                    cur_hp = tgt.hp if (tgt is not None and tgt.hp) else cd.hp
                    ac = cards.get(our_active.id)
                    have = len(our_active.energies or [])
                    if not getattr(obs.current, "energyAttached", True):
                        bonus = 0
                        for h in (my.hand or []):
                            bonus = max(bonus, _provides_units(h.id, ac, cards))
                        have += bonus
                    reach = 0
                    for a in (ac.attacks if ac else []):
                        atk = attacks.get(a)
                        if not atk or len(atk.energies or []) > have:
                            continue
                        mult = 2 if (cd.weakness is not None and ac.energyType == cd.weakness) else 1
                        reach = max(reach, (atk.damage or 0) * mult)
                    if reach >= cur_hp and min(prize, remaining) >= remaining:
                        wins = 2                     # 精确校验的必胜拉人,压过一切假wins
                rc = cd.retreatCost or 0          # can't KO? strand a CLUNKY one → taxes them a Switch/energy
                threat = max((attacks[a].damage for a in cd.attacks if a in attacks), default=0)
                # win now > KO a high-prize win-con > (stuck) strand their heaviest wall
                return (wins, can_ko, prize, rc, threat)
            best = max(opp_opts, key=gust_key)
            return [best]
        # our own Switch/retreat target → bring up the best counter vs their Active, but NEVER
        # promote a body the opponent would OHKO by weakness (don't switch Latias into a Dark deck).
        opp_type = None
        _oc = cards.get(opp_active.id) if opp_active else None
        if _oc:
            opp_type = _oc.energyType

        opp_needs = len(opp.prize or [])       # 对手还差几张奖=他的斩杀线
        my_needs = len(my.prize or [])

        def own_key(i):
            cd = cards.get(_cid(opts[i]))
            if not cd:
                return -1e9
            base = _eff_ids(cd.cardId, opp_active.id) if opp_active else _best_damage_id(cd, attacks)
            # 睁眼包v2:上前排看的是"这只现在能不能开火"和"死了送几张奖",不是卡面伤害上限。
            # 旧版42/42恒选第0项。能付得起费用的攻击手 > 血厚 > 别主动把megaEx(3奖)顶上去。
            if _CONSUMER_V2[0]:
                ent = _resolve_obj(opts[i], obs, sel)
                if ent is not None:
                    fuel = len(getattr(ent, "energies", None) or [])
                    live = _live_reach(ent, cards.get(opp_active.id) if opp_active else None,
                                       my, obs, cards, attacks) if opp_active else 0
                    worth = 3 if getattr(cd, "megaEx", False) else 2 if getattr(cd, "ex", False) else 1
                    base = live * 10 + (ent.hp or 0) - worth * 60
                    if fuel == 0:
                        base -= 400          # 0能量上去=白给一个回合
            if opp_type is not None and cd.weakness == opp_type:
                base -= 100000     # weak to what's in front → it just gets KO'd; keep it benched
            if _SHIELD_PROMOTE[0] and opp_active is not None:
                worth = 3 if getattr(cd, "megaEx", False) else 2 if getattr(cd, "ex", False) else 1
                # 赢在当下压倒一切:这只上去能击倒对面且拿的奖够收官 → 顶格加分
                gain = min(_prize_worth(opp_active, cards), my_needs)
                if _eff_ids(cd.cardId, opp_active.id) >= (opp_active.hp or 9999) and gain >= my_needs:
                    base += 200000
                # 挡刀账(尸检4局实锤ep87826277等):对手杀掉这只就凑满奖时,它就是胜负点,
                # 别送上前排;派1奖尸体消耗,除非场上全是胜负点(同减不改相对序)
                elif worth >= opp_needs:
                    base -= 50000
                # 圣经修正案三(v1.2)·对凯西禁promote Mega不看剩奖: 91757868 T7 promote当场被
                # 20×手牌炮终结。凯西在场+这是Mega → 同罚(有海星星时它自然胜出)。
                elif _BIBLE_V4[0] and getattr(cd, "megaEx", False) and any(
                        p is not None and p.id in (741, 742, 743)
                        for p in ([x for x in (opp.active or []) if x]
                                  + [b for b in (opp.bench or []) if b])):
                    base -= 50000
            return base
        order = sorted(range(n), key=own_key, reverse=True)
        k = max(mn, min(mx, n))
        return order[:k] if k > 0 else []

    # --- Search / draw to hand: grab the most useful; when starved, energy comes first ---
    if ctx in (SC.TO_HAND, SC.TO_HAND_ENERGY, SC.LOOK,
               SC.EVOLVES_FROM, SC.EVOLVES_TO, SC.EVOLVE, SC.ATTACH_FROM, SC.ATTACH_TO,
               SC.HEAL, SC.REMOVE_DAMAGE_COUNTER):
        me = obs.current.yourIndex
        my = obs.current.players[me]; opp = obs.current.players[1 - me]
        oa = opp.active[0] if opp.active else None
        oc = cards.get(oa.id) if oa else None
        opp_weak = oc.weakness if oc else None
        starved = _energy_starved(my, cards, attacks)

        bench_dry = not any(b for b in (my.bench or []))

        def _hand_val(i):
            cid = _cid(opts[i]); base = _card_value(cid, cards, attacks)
            cd = cards.get(cid)
            if cd and cd.cardType == CT.POKEMON and opp_weak is not None and cd.energyType == opp_weak:
                base += 130    # grab the TYPE-COUNTER attacker for THIS opponent (grass vs Grimmsnarl)
            if starved and cd and cd.cardType in (CT.BASIC_ENERGY, CT.SPECIAL_ENERGY):
                base += 100    # energy is the bottleneck → pull it before another Pokémon (state27)
            if _FETCH_BASIC[0] and bench_dry and cd and cd.cardType == CT.POKEMON \
                    and getattr(cd, "basic", False):
                base += 500    # 板凳空=独苗清场在一击之间:检索件先捞能上场的基础怪,进化件都得靠后
            if _BIBLE_V3X[0]:
                # 圣经检索路由(Lawliet终审·超球分流): 手无Mega且场有底座→优先捞1031;
                # 场上身体≤1→优先捞1030(海星星是宝芬管的, 检索位让给刚需)
                hand_ids2 = [c.id for c in (my.hand or [])]
                board2 = [x for x in (my.active or []) if x] + [b for b in (my.bench or []) if b]
                if cid == 1031 and 1031 not in hand_ids2 and any(p.id == 1030 for p in board2):
                    base += 400
                if cid == 1030 and len(board2) <= 1:
                    base += 450
            return base
        order = sorted(range(n), key=lambda i: -_hand_val(i))
        k = max(mn, min(mx, n))
        return order[:k] if k > 0 else []

    # --- Count selections: take the maximum offered (draw more, etc.) ---
    if ctx in (SC.DRAW_COUNT, SC.DAMAGE_COUNTER_COUNT, SC.REMOVE_DAMAGE_COUNTER_COUNT):
        # options are NUMBER; pick the largest number
        best, bi = -1, 0
        for i, o in enumerate(opts):
            v = o.number if o.number is not None else i
            if v > best:
                best, bi = v, i
        return [bi]

    # --- default: legal minimum, preferring our own useful cards ---
    order = sorted(range(n), key=lambda i: -_card_value(_cid(opts[i]), cards, attacks))
    k = max(mn, min(mn if mn > 0 else 1, n))
    return order[:k] if k > 0 else []


def make_agent(deck_ids, go_first=False, text_dmg=None):
    deck = [int(x) for x in deck_ids]
    _bmode = "s87" if 861 in deck else "e1"
    if _EVO_FUEL[0] and not _deck_evo_map:
        try:
            _deck_evo_map.update(_build_evo_map(deck, _data()[0]))
        except Exception:
            pass
    td_flag = _TEXT_DMG_ON[0] if text_dmg is None else bool(text_dmg)

    def agent(obs_dict):
        _TEXT_DMG_ON[0] = td_flag         # 逐agent开关(同_SNIPE样板,防进程级env污染A/B)
        _BIBLE_MODE[0] = _bmode           # 圣经模式逐agent切换(861在表=s87)
        sel = obs_dict.get("select")
        # safe fallback bounds
        mn = sel.get("minCount", 0) if sel else 0
        n = len(sel.get("option", [])) if sel else 0
        try:
            if sel is None:
                return list(deck)
            cards, attacks = _data()
            obs = to_observation_class(obs_dict)
            s = obs.select
            res = _decide(obs, s, s.option, s.context, s.minCount, s.maxCount, cards, attacks, go_first)
            # sanitise: valid, unique, within [minCount, maxCount]
            seen, out = set(), []
            for i in res:
                if isinstance(i, int) and 0 <= i < len(s.option) and i not in seen:
                    seen.add(i); out.append(i)
            if len(out) < s.minCount:
                for i in range(len(s.option)):
                    if i not in seen:
                        out.append(i); seen.add(i)
                        if len(out) >= s.minCount:
                            break
            return out[:max(s.minCount, s.maxCount)] if s.maxCount else out[:s.minCount]
        except Exception:
            return list(range(min(mn, n)))

    return agent


# ---------- eval-variant hook (code-evolution loop) ----------
# CABT_EVAL_VARIANT=<name> swaps _evaluate for a candidate from eval_variants.py;
# the arena judges the variants and only winners survive. Never crashes: unknown
# name / missing module keeps the incumbent.
_ev = os.environ.get("CABT_EVAL_VARIANT")
if _ev:
    try:
        import eval_variants as _EV
        _evaluate = getattr(_EV, _ev, _evaluate)  # noqa: F811
    except Exception:
        pass
