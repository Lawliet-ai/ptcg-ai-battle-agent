# -*- coding: utf-8 -*-
"""RL 自对弈训练：在 MAIN 决策点用策略梯度直接学出招，去掉 PIMC 搜索。

背景（story/顶端agent研究.md）：榜首及顶端普遍是 RL 策略网络而非搜索；我们自己的实验
（大搜索 -10pp、D=1≥D=4）与社区 3 万局行为统计独立地指向同一结论——这个环境惩罚搜索。
本脚本是那个诊断的对照实验。

设计：
  · 策略＝指针式打分头（复用 train_policy.PolicyNet 的结构）＋ 价值基线头。
  · 只接管 MAIN(context=0)；其余决策点走 policy.make_agent（圣经-87 条令层）。
  · rollout 在 worker 里用 **numpy 前向**（无梯度，快）；更新时在主进程用 torch 重算 log-prob。
  · 对手：50% 镜像（同权重）＋ 50% 九族真实 field（启发式驾驶），与部署分布对齐。
  · 回报：终局 ±1；牌库耗尽判负额外 -0.5（社区一手 reward shaping 经验）。

用法:
  CABT_FEAT=v9 .venv/bin/python train/rl_selfplay.py --iters 400 --games-per-iter 200 \
      --workers 10 --out train/rl_s87.pt
"""
import sys, os, json, ctypes, time, random, argparse, math
import multiprocessing as mp

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "train"))

os.environ.setdefault("CABT_FEAT", "v9")
from features import featurize_v9 as featurize, N_FEAT_V9 as N_FEAT, N_SCALAR_V9 as N_SCALAR, CARD_N  # noqa

N_TYPE, ATK_N, D, H = 16, 600, 64, 96


def opt_feat(o, cur, me):
    """选项 -> (type, cardId, attackId)。PLAY/ATTACH 用手牌下标解析出 cardId。"""
    t = o.get("type"); cid = o.get("cardId"); aid = o.get("attackId"); idx = o.get("index")
    t = t if isinstance(t, int) and 0 <= t < N_TYPE else 0
    if t in (7, 8) and cid is None and isinstance(idx, int):
        hand = cur["players"][me].get("hand") or []
        if 0 <= idx < len(hand):
            c = hand[idx]
            cid = c.get("id") if isinstance(c, dict) else c
    cid = cid if isinstance(cid, int) and 0 <= cid < CARD_N else 0
    aid = aid if isinstance(aid, int) and 0 <= aid < ATK_N else 0
    return t, cid, aid


# ---------------------------------------------------------------- numpy 前向
def np_forward(W, idx, val, sc, ofs):
    """返回每个选项的 logit 与状态价值。W = numpy 权重字典。"""
    e = (W["emb"][idx] * np.asarray(val, dtype=np.float32)[:, None]).sum(0)
    x = np.concatenate([e, np.asarray(sc, dtype=np.float32)])
    srep = np.maximum(W["sfc_w"] @ x + W["sfc_b"], 0.0)
    v = float(np.tanh(W["vout_w"] @ srep + W["vout_b"]))
    t = np.array([o[0] for o in ofs]); c = np.array([o[1] for o in ofs]); a = np.array([o[2] for o in ofs])
    of = np.concatenate([W["otype"][t], W["card"][c], W["atk"][a]], axis=1)
    h = np.maximum(np.concatenate([np.repeat(srep[None, :], len(ofs), 0), of], axis=1) @ W["h1_w"].T + W["h1_b"], 0.0)
    return (h @ W["out_w"].T + W["out_b"]).ravel(), v


def torch_to_np(net):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return {"emb": sd["emb.weight"], "sfc_w": sd["sfc.weight"], "sfc_b": sd["sfc.bias"],
            "card": sd["card.weight"], "atk": sd["atk.weight"], "otype": sd["otype.weight"],
            "h1_w": sd["h1.weight"], "h1_b": sd["h1.bias"],
            "out_w": sd["out.weight"], "out_b": sd["out.bias"],
            "vout_w": sd["vout.weight"].ravel(), "vout_b": float(sd["vout.bias"][0])}


# ---------------------------------------------------------------- worker
_G = {}


def _init(deck, opp_paths, mimics=None):
    import cg.game as game
    from cg.sim import lib, Battle
    import policy as P
    _G.update(game=game, lib=lib, Battle=Battle, P=P, deck=deck,
              opps=[[int(x) for x in open(p) if x.strip()] for p in opp_paths],
              mimics=mimics or [])


def _load_mimic(path):
    """加载真实天梯bot仿制体(farmers/* 与 _repos/ptcg-abc/agents/*)。三坑三解:
    ①import时往cwd写deck.csv→切临时目录; ②部分bot依赖 agents/_base →挂进sys.path;
    ③megastarmie/megastarmie_v2/mewtwo 各带同名policy_base.py→每次加载前从
    sys.modules清掉共享名, 且把该bot自己的目录排在_base之前, 防止串味。"""
    key = "mimic:" + path
    if key not in _G:
        import importlib.util, tempfile
        abs_path = os.path.abspath(path)
        adir = os.path.dirname(abs_path)
        base = os.path.join(os.path.dirname(adir), "_base")
        cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="mimic_")
        added = []
        for m in ("policy_base", "generic_policy", "value_weights"):
            sys.modules.pop(m, None)
        try:
            os.chdir(tmp)
            for pth in (base, adir):                 # adir 后插=优先级更高
                if os.path.isdir(pth) and pth not in sys.path:
                    sys.path.insert(0, pth); added.append(pth)
            spec = importlib.util.spec_from_file_location("mimic_mod_" + str(len(_G)), abs_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            os.chdir(cwd)
            for pth in added:
                if pth in sys.path:
                    sys.path.remove(pth)
        _G[key] = mod.agent
    return _G[key]


def _obs():
    sd = _G["lib"].GetBattleData(_G["Battle"].battle_ptr)
    o = json.loads(sd.json.decode())
    o["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return o, sd.selectPlayer


def _net_agent(W, deck, go_first, rng, temp, record):
    """MAIN 用网络采样，其余走启发式(圣经层)。record 为 None 时不记轨迹(评测/对手用)。"""
    heur = _G["P"].make_agent(deck, go_first=go_first)

    def agent(obs_dict):
        sel = obs_dict.get("select")
        if sel is None:
            return list(deck)
        if sel.get("context") != 0:
            return heur(obs_dict)
        cur = obs_dict.get("current")
        opts = sel.get("option") or []
        if cur is None or len(opts) < 2:
            return heur(obs_dict)
        me = cur["yourIndex"]
        try:
            idx, val, sc = featurize(cur, me)
            ofs = [opt_feat(o, cur, me) for o in opts]
            logits, v = np_forward(W, idx, val, sc, ofs)
        except Exception:
            return heur(obs_dict)
        z = logits / max(temp, 1e-6)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        pick = int(rng.choices(range(len(p)), weights=p, k=1)[0]) if temp > 0 else int(p.argmax())
        if record is not None:
            # 记下当刻双方剩余奖励卡 —— 用于稠密回报塑形(见 _play)。
            ps = cur["players"]
            record.append({"idx": list(map(int, idx)), "val": list(map(float, val)),
                           "sc": list(map(float, sc)), "ofs": ofs, "pick": pick, "v": v,
                           "mp": len(ps[me].get("prize") or []), "op": len(ps[1 - me].get("prize") or [])})
        return [pick]

    return agent


def _play(args):
    W, seed, n_games, temp = args[:4]
    allow_mirror = args[4] if len(args) > 4 else True
    mimic_frac = args[5] if len(args) > 5 else 0.0
    rng = random.Random(seed)
    game, deck, opps = _G["game"], _G["deck"], _G["opps"]
    traj, wins, deckouts = [], 0, 0
    for gi in range(n_games):
        mirror = allow_mirror and (gi % 2 == 0)
        use_mimic = (not mirror) and _G["mimics"] and rng.random() < mimic_frac
        if mirror:
            opp = deck
        elif use_mimic:
            mpath, mdeck = _G["mimics"][rng.randrange(len(_G["mimics"]))]
            opp = mdeck
        else:
            opp = opps[rng.randrange(len(opps))]
        our_seat = gi % 2
        d0, d1 = (deck, opp) if our_seat == 0 else (opp, deck)
        rec = []
        if isinstance(W, str) and W == "tree":
            tk = f"tree:{our_seat}"
            if tk not in _G:
                os.environ["CABT_VALUE_NET"] = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "agent", "value_net_s87.npz")
                import search_agent_tree as ST
                _G[tk] = ST.make_tree_agent(deck, [deck], D=3, iters=200, top_m=4, H=10,
                                            go_first=(our_seat == 0), budget=6.0)
            us = _G[tk]
        else:
            us = _net_agent(W, deck, our_seat == 0, rng, temp, rec)
        if mirror:
            them = _net_agent(W, opp, our_seat == 1, rng, temp, None)
        elif use_mimic:
            them = _load_mimic(mpath)
        else:
            them = _G["P"].make_agent(opp, go_first=(our_seat == 1))
        agents = [us, them] if our_seat == 0 else [them, us]
        try:
            obs0, sd0 = game.battle_start(d0, d1)
            if obs0 is None or getattr(sd0, "errorType", 0):
                continue
            o, sp = _obs()
            steps, result, last = 0, -1, None
            while steps < 6000:
                cur = o.get("current")
                if cur is not None:
                    last = cur
                    if cur.get("result", -1) != -1:
                        result = cur["result"]; break
                o = game.battle_select(agents[sp](o)); o, sp = _obs(); steps += 1
        except Exception:
            # 社区bot在个别局面会崩(IndexError实录); 弃局保锅。天梯上对手崩=我们赢,
            # 本地弃局=保守计(既不算胜也不算负)。
            try:
                game.battle_finish()
            except Exception:
                pass
            continue
        game.battle_finish()
        if result not in (0, 1):
            continue
        if not rec:
            wins += (result == our_seat)
            continue
        won = (result == our_seat)
        wins += won
        z = 1.0 if won else -1.0
        # 牌库耗尽判负额外惩罚(社区一手经验)
        if not won and last is not None:
            try:
                if int(last["players"][our_seat].get("deckCount") or 99) <= 0:
                    z -= 0.5; deckouts += 1
            except Exception:
                pass
        # ---- 稠密回报塑形：终局±1 太稀疏(一局~40个MAIN决策共享同一个信号, 信用分配≈0)。
        # 拿奖/送奖是这个游戏真正的胜负单位(6奖制), 用它给每一步打分。
        # r_t = K*(我方奖减少) - K*(对方奖减少)；G_t = 未来 r 之和 + 终局 z。
        K = float(os.environ.get("CABT_RL_PRIZE_K", "0.15"))
        mp_end = op_end = None
        if last is not None:
            try:
                mp_end = len(last["players"][our_seat].get("prize") or [])
                op_end = len(last["players"][1 - our_seat].get("prize") or [])
            except Exception:
                pass
        rs = []
        for i, r in enumerate(rec):
            nm = rec[i + 1]["mp"] if i + 1 < len(rec) else (mp_end if mp_end is not None else r["mp"])
            no = rec[i + 1]["op"] if i + 1 < len(rec) else (op_end if op_end is not None else r["op"])
            rs.append(K * (r["mp"] - nm) - K * (r["op"] - no))
        g = z
        for i in range(len(rec) - 1, -1, -1):
            g = rs[i] + g
            # 价值头是 tanh(±1 饱和)，回报叠加塑形后可达 ±1.9，超出部分学不到 →
            # 统一除以 2 压回可表示区间（只是尺度，不改变相对次序）。
            rec[i]["z"] = max(-1.0, min(1.0, g / 2.0))
        traj.extend(rec)
    return traj, wins, n_games, deckouts


# ---------------------------------------------------------------- 主训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--games-per-iter", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--deck", default=os.path.join(ROOT, "agent", "deck_stardom87.csv"))
    ap.add_argument("--gauntlet", default=os.path.join(ROOT, "gauntlet_train0805"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "rl_s87.pt"))
    ap.add_argument("--init", default="")
    ap.add_argument("--mimic-dirs", default="",
                    help="逗号分隔的真实bot目录(各含main.py+deck.csv), 如 farmers/lucario950,farmers/meta_a")
    ap.add_argument("--mimic-frac", type=float, default=0.5,
                    help="非镜像局中抽真实bot当对手的概率")
    ap.add_argument("--ckpt-every", type=int, default=20)
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    class RLNet(nn.Module):
        def __init__(self, d=D):
            super().__init__()
            self.emb = nn.EmbeddingBag(N_FEAT, d, mode="sum")
            self.sfc = nn.Linear(d + N_SCALAR, H)
            self.card = nn.Embedding(CARD_N, 32)
            self.atk = nn.Embedding(ATK_N, 16)
            self.otype = nn.Embedding(N_TYPE, 16)
            self.h1 = nn.Linear(H + 64, H)
            self.out = nn.Linear(H, 1)
            self.vout = nn.Linear(H, 1)

        def srep_one(self, idx, val, sc):
            e = (self.emb.weight[idx] * val[:, None]).sum(0)
            return torch.relu(self.sfc(torch.cat([e, sc])))

        def logits_of(self, srep, t, c, aa):
            of = torch.cat([self.otype(t), self.card(c), self.atk(aa)], dim=1)
            h = torch.relu(self.h1(torch.cat([srep.expand(len(t), -1), of], dim=1)))
            return self.out(h).squeeze(1)

    net = RLNet()
    if a.init and os.path.exists(a.init):
        net.load_state_dict(torch.load(a.init, map_location="cpu")); print(f"[init] {a.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    deck = [int(x) for x in open(a.deck) if x.strip()]
    opp_paths = sorted(os.path.join(a.gauntlet, f) for f in os.listdir(a.gauntlet) if f.endswith(".csv"))
    mimics = []
    for d in [x for x in a.mimic_dirs.split(",") if x.strip()]:
        ap_, dp_ = os.path.join(d, "main.py"), os.path.join(d, "deck.csv")
        assert os.path.exists(ap_) and os.path.exists(dp_), f"mimic 目录缺文件: {d}"
        mimics.append((os.path.abspath(ap_), [int(x) for x in open(dp_) if x.strip()]))
    print(f"[setup] deck={len(deck)} 张, 陪练={len(opp_paths)} 副, 真实bot={len(mimics)} 个, "
          f"workers={a.workers}", flush=True)

    per = max(1, a.games_per_iter // a.workers)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(deck, opp_paths, mimics)) as pool:
        for it in range(1, a.iters + 1):
            W = torch_to_np(net)
            tasks = [(W, it * 1000 + w, per, a.temp, True, a.mimic_frac) for w in range(a.workers)]
            out = pool.map(_play, tasks)
            traj = [r for o in out for r in o[0]]
            wins = sum(o[1] for o in out); games = sum(o[2] for o in out)
            deckouts = sum(o[3] for o in out)
            if not traj:
                print(f"it{it}: 无轨迹, 跳过", flush=True); continue

            zs = torch.tensor([r["z"] for r in traj], dtype=torch.float32)
            vs = torch.tensor([r["v"] for r in traj], dtype=torch.float32)
            adv = zs - vs
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

            # ---- 批量更新：逐样本反传是瓶颈(7s/轮里6s在这)，改成整批一次前向反传。
            # EmbeddingBag(offsets) 原生支持"每样本一袋稀疏特征"，正是我们的状态表示形状。
            B = len(traj)
            flat_idx, offs, flat_val, cur_off = [], [], [], 0
            for r in traj:
                offs.append(cur_off); flat_idx += r["idx"]; flat_val += r["val"]; cur_off += len(r["idx"])
            fi = torch.tensor(flat_idx, dtype=torch.long)
            fo = torch.tensor(offs, dtype=torch.long)
            fv = torch.tensor(flat_val, dtype=torch.float32)
            scb = torch.tensor([r["sc"] for r in traj], dtype=torch.float32)
            emb = net.emb(fi, fo, per_sample_weights=fv)
            srep = torch.relu(net.sfc(torch.cat([emb, scb], dim=1)))          # (B,H)
            v = torch.tanh(net.vout(srep)).squeeze(1)

            nop = [len(r["ofs"]) for r in traj]
            maxn = max(nop)
            tt = torch.tensor([o[0] for r in traj for o in r["ofs"]], dtype=torch.long)
            cc = torch.tensor([o[1] for r in traj for o in r["ofs"]], dtype=torch.long)
            aa = torch.tensor([o[2] for r in traj for o in r["ofs"]], dtype=torch.long)
            of = torch.cat([net.otype(tt), net.card(cc), net.atk(aa)], dim=1)
            reps = torch.repeat_interleave(srep, torch.tensor(nop), dim=0)
            lg = net.out(torch.relu(net.h1(torch.cat([reps, of], dim=1)))).squeeze(1)

            # 分段 softmax：填充成 (B,maxn) 矩阵，非法位置置 -inf
            rows = torch.repeat_interleave(torch.arange(B), torch.tensor(nop))
            cols = torch.cat([torch.arange(n) for n in nop])
            pad = torch.full((B, maxn), -1e9)
            pad[rows, cols] = lg
            logp = torch.log_softmax(pad, dim=1)
            picked = logp[torch.arange(B), torch.tensor([r["pick"] for r in traj], dtype=torch.long)]
            pr = logp.exp()
            ent = -torch.where(pad > -1e8, pr * logp, torch.zeros_like(pr)).sum(1)

            loss = -(picked * adv).mean() + 0.5 * ((v - zs) ** 2).mean() - a.ent * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            pl = float((-(picked * adv)).mean()); vl = float(((v - zs) ** 2).mean()); el = float(ent.mean())

            n = len(traj)
            print(f"it{it}: 胜率 {100*wins/max(games,1):5.1f}% ({wins}/{games})  "
                  f"轨迹 {n}  策略损失 {pl:+.3f}  价值损失 {vl:.3f}  熵 {el:.2f}  "
                  f"牌库耗尽 {deckouts}  用时 {time.time()-t0:.0f}s", flush=True)
            if it % a.ckpt_every == 0:
                torch.save(net.state_dict(), a.out)
                # 贪心评测(temp=0)才是真实水平: 训练时是采样, 熵会压低表观胜率。
                # 全部打九族陪练(不打镜像), 与 arena_eval 同口径 —— v1 的树在这把尺子上是 79.1%。
                Wg = torch_to_np(net)
                ev = pool.map(_play, [(Wg, 90000 + it * 10 + w, 12, 0.0, False) for w in range(a.workers)])
                ew = sum(o[1] for o in ev); eg = sum(o[2] for o in ev)
                print(f"[ckpt] {a.out} @ it{it}  贪心评测 {100*ew/max(eg,1):.1f}% ({ew}/{eg})"
                      f"   ← v1树同尺 79.1%", flush=True)
    torch.save(net.state_dict(), a.out)
    print(f"DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
