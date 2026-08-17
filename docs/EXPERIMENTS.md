# Experiment log

Every A/B we ran, including the failures. `p` values are two-proportion z-tests
(`tools/ab_ztest.py`). Ladder numbers are TrueSkill-style ratings; local numbers are win rates
against a fixed sparring pool.

**House rule, enforced without exception: `p > 0.05` is a tie.** No "directionally positive"
exceptions. We retracted a published internal conclusion once for violating this.

---

## 0. First, the instrument

| Measurement | Value |
|---|---|
| Games resolved in a submission's first 8 hours | ≈35 |
| **Minimum detectable win-rate difference (95% CI) at that n** | **±24 pp** |
| Byte-identical submissions, 1 minute apart | 12–18 rating points |
| Same bytes, morning vs afternoon | 732 vs 571 (140 points) |
| New submission cold start | opens ~600, dips to ~500, needs 24h+ |

Everything below is judged against these numbers. Anything smaller than the noise floor is
reported as a tie, however appealing the story.

### The sparring pool, which was wrong for six weeks

| Gauge | Predicts for our tree | Tree's true ladder rate |
|---|---|---|
| Our own heuristics as sparring partners | 79.1% | 55.6% |
| **15-bot pool from real opponent archetypes** | **56.7%** | **55.6%** |

A 37-point divergence, sustained for most of the competition, because we sparred against ourselves.
The fix was to rebuild the instrument, not the agent. It arrived in the final 48 hours.

---

## 1. "Work harder" levers — all null

| Experiment | Result | Verdict |
|---|---|---|
| Deep search D=4 / 400 iters / 4.0s vs D=3 / 96 / 0.7s | 39.6% — deep is **worse** | depth hurts |
| Determinization worlds: D=1 vs D=4 vs D=8 | 40.3% / 34.0% / lower | **fewer worlds is better** |
| Compute moved from hand-inference to lookahead | −4.2 pp | reverted |
| Iterations 200 → 600 (1,200 games/arm) | 30.2% vs 29.7%, p=0.79 | tie |
| Stronger sparring opponents | 61.0% vs 61.0% | zero |
| Imitation learning from master replays | 67.9% vs 75.0% without | worse |
| Policy-head distillation (attempted twice) | 54.6% imitation accuracy, no win-rate gain | a student cannot exceed its teacher |

Seven independent attempts to buy strength with effort. All null. This is the single most consistent
pattern in the log, and it is what eventually pointed us at the paradigm question.

## 2. Knowledge injection — the central controlled experiment

Same insight (bench a Basic when the bench is empty), three deployment forms:

| Form | Local A/B | Ladder |
|---|---|---|
| **Candidate injection** | 57.6% vs 48.6% (144 games, 3 seeds) | **709** |
| Rule override | — | 582 / 604 / 676 |
| Feature encoding | tie | tie |

## 3. Value-net generations

| Gen | Change | Result |
|---|---|---|
| hand eval | — | AUC 0.66 |
| v1 | learned net | AUC 0.82, win rate flat |
| v2–v7 | more data, bigger net, harder sparring | AUC ↑, win rate flat — four-generation plateau |
| **v8** | **soft labels** | **66.7%** — first break |
| **v9** | **fixed encoding** (stadium, bench HP/energy) | **62% over v8** in 300 games, after 21s of training |
| v9.2b | mid-ladder curriculum | passed both gauges, shipped |
| gen2–gen4 | 380k deck-specific samples | 80.2% vs 79.1%, **p=0.558** — ceiling |

## 4. Deck experiments

| Experiment | Result |
|---|---|
| Original Ogerpon toolbox vs the reigning deck | 65% against target, 55% overall, one unfixable hole |
| Deck gauntlet — same pilot, same field, deck as only variable | ours **40%** vs a real tournament list **71%** |
| Programmatic mining of 2,022 cards → 3 original cores | 36.9% / 12.5% / 8.9%, **0–4 vs netdecks** |
| Switching to a real tournament list | **+225 rating in one night** |
| Energy 7 → 9 Water (E1) | +5.4 pp, 1,200 games/arm, **p=0.005**; ladder twins **+55~75** |
| Cutting the 4th Buddy-Buddy Poffin | ladder twins **−214** — the 4th copy is load-bearing |
| Adopting the rank-87 twin-Mega list (first attempt) | **−146** — a stranger's deck driven by a stranger |
| Same list + dedicated value net + extracted playbook | **62% real win rate**, ladder climbing 527→623 |

The last two rows are the same deck. The difference is that the second attempt gave the driver
162k self-play samples on that specific list and a doctrine extracted from its author's own games.

## 5. Playbook amendments — all five negative

After the deck was working, we wrote five amendments of our own from loss forensics. Every one
tested negative on the ladder:

| Amendment | Result |
|---|---|
| Static damage estimate for a hand-scaling attack | **60% → 38%** |
| Equalising energy attachment across both Mega lines | **60% → 38%** |
| Boss's Orders banned on zero-energy turns + 3 more hard gates | 621 vs champion's 713 |
| Restricting Cinderace's setup role to going-second | 50% vs 60% |
| All five combined | 587 |

Two of these deserve their own entry.

**Never hand a search engine a constant.** Mega Froslass's main attack scales with the opponent's
hand size; the engine reports `damage=0` for it, so our scorer classified a ~250-damage gun as a
status move. We supplied a static estimate of 250. The constant reached `_lethal`, the tree's
absolute-priority "KO now" branch started firing on knockouts that did not exist, and win rate fell
from 60% to 38%. *A number that is right on average is a lie at every specific board state.*

**A code fact may be design intent.** Our energy-attachment priority table listed only the Starmie
line; the Froslass line sat at zero energy on 75% of turns. Provably a defect — so we equalised
them. 60% → 38%. The deck's economy funds exactly one attacker, and splitting energy starves both
lines. The original author's lopsided 54%/25% attachment split was the correct bias, and our
correction was the defect.

## 6. RL sprint — the final 72 hours

`train/rl_selfplay.py`: REINFORCE + value baseline + entropy bonus; dense reward shaping on prize
differential; deck-out losses penalised an extra −0.5. Decides MAIN actions directly, **no search**.

| Checkpoint | Training | Local | Ladder |
|---|---|---|---|
| random MAIN + doctrine layer | — | 26.7% | — |
| it125 | minutes | 43.5% | — |
| it322 (probe1) | ~1h | 64.3% | 444 |
| it4000 (probe2) | 3.1h, ~2.4M self-play games | **75.7%** | 413 |
| probe3 (meta-pool trained) | +5h | 54.3% (meta pool) | **607** |
| probe4 (it6000) | +5h | 56.5% (meta pool) | 574 |
| **PIMC tree (our champion)** | one month | 79.1% / 56.7% (meta pool) | ~650 |

**Verdict: policy 52.9% (51 games) vs tree 55.6%, p = 0.81 — indistinguishable**, at **0.3 ms vs
~6 s** per move.

**And the same mistake, a third time.** The policy scored 75.7% on our old sparring pool and
**24.8%** against replicas of real ladder bots — a **50-point transfer gap**. Training 4,000 more
iterations on the old distribution bought nothing against real opponents. Failures 4 (local
benchmark), 5 (value-net curriculum) and this one are one error wearing three costumes: *we kept
building instruments whose distribution did not match deployment.* It took six weeks to see them
as one error rather than three.

---

## Failure catalogue

| # | Failure | Cost | Root cause |
|---|---|---|---|
| 1 | Read 3-hour scores as signal; declared three versions regressions | 1 day, 3 submissions | instrument noise (140-point drift) — all verdicts retracted |
| 2 | Static damage estimate for a scaling attack | 60% → 38% | the constant reached the lethality checker |
| 3 | "Fixed" an energy table that omitted the second line | 60% → 38% | it was design intent |
| 4 | Local benchmark built on our own heuristics | 37-point divergence | sparring ≠ deployment distribution |
| 5 | Value net trained on top-tier decks | ladder stall | curriculum ≠ deployment tier |
| 6 | RL trained on the old sparring pool | 50-point transfer gap | same error, new paradigm |
| 7 | Trusted card metadata (`stage2` / `evolvesFrom` / `basic`) | 3 days counting 8 playable Basics as 10 | metadata does not gate playability — verify with replays |
| 8 | Programmatic deck mining | 0–4 vs netdecks | deck sense is still a human domain |
| 9 | Imitation learning, two ways | both below no-prior baseline | a student cannot exceed its teacher |
| 10 | Bet the project on search | the campaign | this environment penalises search |
