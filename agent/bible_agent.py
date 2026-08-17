# -*- coding: utf-8 -*-
"""圣经agent: story/圣经-E1-v1.md 的机械翻译(sue124架构: 分带打分+matchup切表)。
MAIN决策=圣经大脑; ATTACH_TO=圣经贴能目标学; 其余decision委托policy._decide(睁眼修复齐全)。
分带即优先级(sue124的招牌): 数量级编码动作顺序, 攻击类最后结算(攻击=结束回合)。"""
import os

import policy
from policy import (_data, _eff_damage, _attack_damage, _lethal, _best_eff_vs,
                    _splash_damage, _incoming_max, _resolve_cid, OT, SC)
from cg.api import to_observation_class

STARYU, MEGA = 1030, 1031
W_ENERGY, IGNITION = 3, 17
POFFIN, POTION, ULTRA, GEAR, SIGNAL = 1086, 1112, 1121, 1122, 1145
CAPE, PAYAPA, BOSS, SALV, HILDA, LILLIE, WALLY = 1159, 1164, 1182, 1189, 1225, 1227, 1229
BEACH, CAGE = 1262, 1264
FINDERS = {SIGNAL, HILDA, SALV, ULTRA}          # 找Mega四路由

# 家族识别标记(矿工D牌表逐张核对的独有id)
_FAMILIES = [
    ("lucario",   {678, 676, 674}),             # Mega Lucario ex / Solrock / Hariyama
    ("dreepy",    {119, 120, 121}),             # 多龙线
    ("kadabra",   {741, 742, 743}),             # 凯西线
    ("archaludon", {190}),                      # Archaludon ex
    ("snover",    {723, 721}),                  # Mega Abomasnow / Kyogre
    ("crustle",   {344, 345}),                  # 石居蟹线
    ("kangaskhan", {756}),
    ("marnie",    {112, 648}),                  # 恶毒猴 / 玛俐长毛巨魔
    ("mirror",    {1031}),
]
_RESTRICT_BENCH = {"lucario", "dreepy"}         # Lawliet终审: 家族总开关=零加铺方针


def _family(opp):
    vis = set()
    for p in ([x for x in (opp.active or []) if x] + [b for b in (opp.bench or []) if b]):
        vis.add(p.id)
    for name, sig in _FAMILIES:
        if vis & sig:
            return name
    return "unknown"


def _prizes_left(pl):
    return len(pl.prize or [])


def _ko_prizes(cd):
    if cd is None:
        return 1
    if getattr(cd, "megaEx", False):
        return 3
    return 2 if getattr(cd, "ex", False) else 1


def _board(pl):
    return [x for x in (pl.active or []) if x] + [b for b in (pl.bench or []) if b]


def _hand_ids(pl):
    return [c.id for c in (pl.hand or [])]


def _energy_n(p):
    return len(p.energies or []) if p else 0


class _Brain:
    """每个决策点重建一次的盘面账本。"""

    def __init__(self, obs, cards, attacks, go_first):
        st = obs.current
        self.obs, self.cards, self.attacks = obs, cards, attacks
        self.me = st.yourIndex
        self.my, self.opp = st.players[self.me], st.players[1 - self.me]
        self.turn = getattr(st, "turn", 0)
        self.my_first_turn = self.turn <= 2
        self.go_first = go_first
        self.front = self.my.active[0] if self.my.active else None
        self.opp_front = self.opp.active[0] if self.opp.active else None
        self.hand = _hand_ids(self.my)
        self.board = _board(self.my)
        self.bench = [b for b in (self.my.bench or []) if b]
        self.bench_space = getattr(self.my, "benchMax", 5) - len(self.bench)
        self.family = _family(self.opp)
        self.restrict = self.family in _RESTRICT_BENCH
        self.incoming, _ = _incoming_max(obs, cards, attacks)   # 第0问原语
        self.front_dying = (self.front is not None and self.incoming >= (self.front.hp or 0)
                            and self.incoming > 0)
        self.my_megas = [p for p in self.board if p.id == MEGA]
        self.my_staryu = [p for p in self.board if p.id == STARYU]
        self.staryu_in_hand = STARYU in self.hand
        self.mega_in_hand = MEGA in self.hand
        self.finder_in_hand = bool(set(self.hand) & {SIGNAL, HILDA, SALV})
        self.energy_in_hand = [c for c in self.hand if c in (W_ENERGY, IGNITION)]
        self.to_take = _prizes_left(self.my)      # 我还要拿几张奖

    # ---- 攻击算术(圣经第3节) ----
    def _attack_score(self, opt):
        atk = self.attacks.get(opt.attackId)
        if atk is None or self.front is None or self.opp_front is None:
            return 300
        dmg = _attack_damage(atk, self.front, self.opp_front)
        eff = _eff_damage(dmg, self.front, self.opp_front, self.cards)
        name = (getattr(atk, "name", "") or "")
        opp_cd = self.cards.get(self.opp_front.id)
        # 岩居蟹免疫ex攻击: 只许Nebula(穿透条款)
        if self.family == "crustle" and opp_cd is not None and self.opp_front.id == 345 \
                and "Nebula" not in name:
            return 310
        if _lethal(dmg, self.front, self.opp_front, self.cards):
            gain = _ko_prizes(opp_cd)
            if gain >= self.to_take:
                return 100000 + eff               # 斩杀=终局
            return 30000 + eff + gain * 100       # 戒23: 能兑击倒的招优先
        if dmg == 0:
            return 400
        sp = _splash_damage(atk)
        bonus = 0
        if sp:                                    # 溅射目标学交给policy的_SNIPE_TARGET, 这里给量
            if any(b and (b.hp or 999) <= sp for b in (self.opp.bench or [])):
                bonus += 2500                     # 溅射可点死=白捡奖
        # 戒20: 能打就打——攻击带正分, 但低于所有发展动作(分带=顺序: 攻击最后)
        return 4000 + eff + sp + bonus

    # ---- MAIN打分 ----
    def score(self, opt, sel):
        t = opt.type
        cid = _resolve_cid(opt, self.obs, sel)
        if t == OT.ATTACK:
            return self._attack_score(opt)
        if t == OT.END:
            return 100
        if t == OT.RETREAT:
            # 前排必死且板凳有身位→撤(冲浪海滩的免费换在ABILITY/场地路径, 这里是付费兜底)
            if self.front_dying and self.bench:
                return 13500
            return 200
        if t == OT.EVOLVE:
            return self._evolve_score(cid)
        if t == OT.ABILITY:
            return 8000 if self.front_dying else 3500
        if t == OT.ATTACH:
            return self._attach_score(cid)
        if t == OT.PLAY:
            return self._play_score(cid)
        return 500

    def _evolve_score(self, cid):
        # 戒19: 残血海星星在大威胁下禁原地进化(把1奖升3奖)
        if self.front is not None and self.front.id == STARYU and (self.front.hp or 70) <= 30 \
                and self.incoming >= 270:
            return 350
        # 戒3: 场上最后一只1奖体禁全进化(板凳无海星星且手上无补充)
        if len(self.my_staryu) == 1 and not self.staryu_in_hand and self.my_megas:
            return 900
        return 20000                              # 戒6: 可进化就进化, 330身板是最好的墙

    def _attach_score(self, cid):
        if cid == IGNITION:
            if self.go_first and self.my_first_turn:
                return 150                        # 戒7: 先手T1禁贴点火(回合末蒸发)
            if not self.my_megas:
                return 200                        # 点火贴基础怪=bug级浪费(矿工A硬提醒3)
            return 16000                          # 贴了必须打(攻击分带在后, 同回合自然结算)
        if cid == W_ENERGY:
            if self.front_dying and any(p.id == MEGA for p in self.bench):
                return 15500                      # 戒9: 必死前排不喂, 能量归板凳接班体(目标在ATTACH_TO选)
            return 15000
        return 1000

    def _play_score(self, cid):
        h = self.hand
        # --- 斩杀与救场带 ---
        if cid == BOSS:
            return self._boss_score()
        if cid == WALLY:
            return self._wally_score()
        if cid == POTION:
            # 戒17: 只在+60能挪出死亡线时用
            if self.front is not None and self.front.id == MEGA and self.front_dying \
                    and self.incoming < (self.front.hp or 0) + 60:
                return 19000
            return 300
        # --- 开局与成型带 ---
        if cid == SALV:
            if self.my_first_turn and not self.go_first and self.energy_in_hand \
                    and self.front is not None and self.front.id == STARYU:
                return 28000                      # Lawliet终审: 后手T1直接进化出手=最高优先
            if not self.mega_in_hand and self.my_staryu:
                return 12800                      # 找+进化二合一
            return 2000
        if cid == HILDA:
            if not self.mega_in_hand or not self.energy_in_hand:
                return 12500
            return 1500
        if cid == SIGNAL:
            if not self.mega_in_hand and not any(p.id == MEGA for p in self.bench):
                return 12000
            return 600                            # 已有Mega别捞第2只压手
        if cid == ULTRA:
            # Lawliet终审的分流逻辑
            if not self.board:                    # 独苗都没有=竞技规则不可能, 防御性
                return 11000
            urgent_body = (self.bench_space > 0 and not self.restrict
                           and not self.staryu_in_hand and POFFIN not in h
                           and len(self.board) <= 1)
            if urgent_body:
                return 11000                      # 独苗危险时找海星星(戒2本义)
            if not self.mega_in_hand and not self.finder_in_hand:
                return 11500                      # 前排有海星星+无找EX件→超球找大海星
            return 800                            # 其余情况留手
        if cid == POFFIN:
            if self.restrict:
                return 700                        # 家族总开关: 路卡/多龙零加铺
            if self.bench_space > 0 and not self.staryu_in_hand:
                return 17000                      # 核心铺后场工具
            if self.bench_space > 0 and len(self.my_staryu) < 2:
                return 9000
            return 800
        if cid == GEAR:
            has_supporter = bool(set(h) & {HILDA, SALV, LILLIE, WALLY, BOSS})
            return 10000 if not has_supporter else 900
        if cid == LILLIE:
            usable = set(h) & {W_ENERGY, IGNITION, BOSS, WALLY}
            if usable:
                return 400                        # 戒25: 手有用得上的关键牌禁洗手
            return 13000 if self.to_take == 6 else 8000
        # --- 工具与场地带 ---
        if cid == CAPE:
            pr = {"lucario": 15500, "snover": 15200, "mirror": 15000}
            return pr.get(self.family, 5000) if self.my_megas else 1200
        if cid == PAYAPA:
            return 9000 if self.family in ("kadabra", "marnie") else 2000
        if cid == CAGE:
            return 27000 if self.family == "dreepy" else 3000
        if cid == BEACH:
            return 6000
        # 上场类PLAY(手牌宝可梦上板凳)
        if cid == STARYU:
            if self.restrict:
                return 750                        # 克制家族: 不喂Boss/不给溅射目标
            if self.bench_space > 0:
                return 18000                      # 戒1: 有身位有海星星必铺
            return 500
        if cid == MEGA:
            return 1000                           # Mega不从手上"打出", 走进化路径
        return 2500

    def _boss_score(self):
        # 戒21/22: 只许拉"本回合能击倒的最值钱目标"或拉引擎怪, 0能禁烧
        if self.front is None or _energy_n(self.front) == 0 and not self.energy_in_hand:
            return 250
        best = 0
        my_cd = self.cards.get(self.front.id) if self.front else None
        for b in (self.opp.bench or []):
            if not b:
                continue
            eff = _best_eff_vs(self.front, b, self.cards, self.attacks) if self.front else 0
            if eff >= (b.hp or 999):
                gain = _ko_prizes(self.cards.get(b.id))
                val = 26000 + gain * 500
                if gain >= self.to_take:
                    val = 90000                   # 拖上来打死=终局
                best = max(best, val)
            elif b.id in (112, 676, 666, 860):    # 引擎怪: 恶毒猴/月石/闪焰/雪童子
                best = max(best, 8500)
        return best if best else 250

    def _wally_score(self):
        dmg_mega = [p for p in self.my_megas if (p.hp or 330) < 330]
        if not dmg_mega:
            return 350
        front_is_dmg_mega = self.front is not None and self.front.id == MEGA \
            and (self.front.hp or 330) < 330
        if front_is_dmg_mega and self.front_dying and self.incoming < 330:
            return 25000                          # 救命满血奶
        if front_is_dmg_mega and _energy_n(self.front) == 0:
            return 21000                          # 戒13: 零代价满血禁压手
        if front_is_dmg_mega and _energy_n(self.front) <= 1:
            return 12200                          # 低代价(能量回手会被单卡攻击经济消解)
        return 2200


def _attach_target(obs, sel, cards):
    """ATTACH_TO(ctx22): 贴能目标学。Mega优先; 前排必死时给板凳Mega; 戒12禁炮灰海星星。"""
    st = obs.current
    me = st.yourIndex
    my = st.players[me]
    opts = sel.option
    attacks = _data()[1]
    incoming, front = _incoming_max(obs, cards, attacks)
    front_dying = front is not None and incoming >= (front.hp or 0) and incoming > 0
    best_i, best = 0, -1
    for i, o in enumerate(opts):
        ent = policy._resolve_obj(o, obs, sel) if hasattr(policy, "_resolve_obj") else None
        cid = getattr(ent, "id", None) if ent is not None else _resolve_cid(o, obs, sel)
        s = 10
        if cid == MEGA:
            s = 1000
            if front is not None and front.id == MEGA and front_dying:
                # 必死前排的Mega降权, 板凳Mega升权(戒9)
                is_front = bool(getattr(o, "inPlayArea", None) == 4)
                s = 300 if is_front else 1500
        elif cid == STARYU:
            s = 200                               # 戒12: 尽量不贴炮灰(除非没Mega可贴)
        if s > best:
            best, best_i = s, i
    return [best_i]


def make_bible_agent(deck_ids, go_first=False):
    deck = [int(x) for x in deck_ids]

    def agent(obs_dict):
        policy._TEXT_DMG_ON[0] = True             # 圣经的威胁账必须看得见Powerful Hand族
        policy._SNIPE[0] = policy._SNIPE[0]       # 保持现值(逐agent开关礼仪)
        sel = obs_dict.get("select")
        mn = sel.get("minCount", 0) if sel else 0
        n = len(sel.get("option", [])) if sel else 0
        try:
            if sel is None:
                return list(deck)
            cards, attacks = _data()
            obs = to_observation_class(obs_dict)
            s = obs.select
            ctx, opts = s.context, s.option
            if ctx == SC.MAIN and obs.current is not None:
                brain = _Brain(obs, cards, attacks, go_first)
                scores = [brain.score(o, s) for o in opts]
                res = [max(range(len(opts)), key=lambda i: scores[i])]
            elif ctx == SC.ATTACH_TO and obs.current is not None:
                res = _attach_target(obs, s, cards)
            else:
                res = policy._decide(obs, s, opts, ctx, s.minCount, s.maxCount,
                                     cards, attacks, go_first)
            seen, out = set(), []
            for i in res:
                if isinstance(i, int) and 0 <= i < len(opts) and i not in seen:
                    seen.add(i); out.append(i)
            if len(out) < s.minCount:
                for i in range(len(opts)):
                    if i not in seen:
                        out.append(i); seen.add(i)
                        if len(out) >= s.minCount:
                            break
            return out[:max(s.minCount, s.maxCount)] if s.maxCount else out[:s.minCount]
        except Exception:
            return list(range(min(mn, n)))        # tomatomato式兜底: 永不因报错判负

    return agent
