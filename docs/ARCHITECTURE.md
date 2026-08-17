# Architecture

```
observation (imperfect information)
      |
      +-- opponent recognition ------ 16 modal decklists mined from ladder replays;
      |     search_agent.py           matched by signature of revealed Pokémon
      |
      +-- PIMC determinization ------ sample D fully-observable worlds
      |     search_agent.py           (prize inference narrows the deck multiset)
      |
      +-- macro-action PUCT tree ---- branches ONLY at our own MAIN decisions;
      |     search_agent_tree.py      sub-decisions + opponent turn -> heuristic
      |     |
      |     +-- candidate shortlist   <-- playbook knowledge enters HERE
      |     |     policy._score_main + _bible_pack_idx
      |     |
      |     +-- leaf value            truncated rollout -> value_net.evaluate_state
      |
      +-- legal fallback ------------ every path; an exception degrades to a legal
            policy.make_agent         choice, never to a forfeit
```

## 1. Opponent recognition

Determinization is only as good as the deck you sample from. Rather than drawing the opponent's
unknown cards from the whole 2,022-card pool, we mined the **16 modal decklists** actually present on
the ladder from replay data (`agent/priors_data.py`) and match the opponent's revealed Pokémon
against them.

Two versions exist. v1 scored by *unique* signature cards — which silently failed for archetypes
whose signatures cancelled out against same-family variants, sending 5 of 16 priors to a constant
score of zero and defaulting them to `priors[0]`. v2 scores by coverage of the full Pokémon set with
uniqueness as a tiebreak, and identifies 16/16 correctly against v1's 11/16.

**We shipped v1.** Ladder A/B found no measurable difference between them (three separate tests, all
ties), and our rule is that an unproven change does not enter the champion build. The correct
identification simply did not translate into wins at our search depth — the tree's decisions were
dominated by our own board plan, not by which deck the opponent was holding. We report this because
it is a negative result people rarely publish: *a provably better component that buys nothing.*

## 2. PIMC determinization

`_determinize()` builds a concrete world: our unknown cards come from our own remaining decklist
(exact), the opponent's from the recognised prior (approximate).

One refinement earns its keep. Cards can often be **proven** to sit in the prize pool — the engine
reveals enough over a game to constrain it. `_update_prize_cache` maintains that inference, and
determinization places those cards in the prizes *first*, so the deck we search over is the correct
multiset rather than a random split.

**A result that surprised us:** sampling *fewer* worlds was better. D=1 scored 40.3%, D=4 scored
34.0%, D=8 lower still. Averaging over more determinized worlds does not reduce error here; it
blurs the distinctions the search needs to act on. Moving compute from world sampling to lookahead
also cost 4.2 points. Both are consistent with the paradigm finding: in a game this uncertain, more
of either kind of search is not more strength.

## 3. Macro-action PUCT tree

The tree branches **only at our own MAIN decisions**. Every sub-decision (choosing targets,
discarding, coin flips) and the opponent's entire turn is resolved by the heuristic policy.

The consequence is structural: because every branch node belongs to us, **no value negation is
needed**. The opponent is a fixed reasonable player rather than an adversary being minimaxed. This
kept the implementation reliable and let each search cover more of our own decision sequences —
at the cost of never modelling an opponent who adapts.

Configuration shipped: `D=3` worlds, `200` iterations, horizon `10`, `6.0s` budget per move, with a
time bank that always reserves 60 seconds of the per-game allowance.

### The candidate shortlist

```python
scored = sorted(range(n), key=lambda i: -policy._score_main(opts[i], o, cards, attacks))
cand = scored[:min(top_m, n)]                       # top_m = 4
for i in range(n):
    if opts[i].type == OT.ATTACK and i not in cand:
        cand.append(i)                              # the heuristic mis-values scaling attacks
if _CAND_BENCH[0]:
    _bi = policy._bench_basic_idx(...)              # make "bench a Basic" visible
    if _bi is not None and _bi not in cand:
        cand.append(_bi)
```

This is the most important design decision in the agent, and it was discovered by failure.

The shortlist is truncated at `top_m=4` by static heuristic priors. Benching a Basic scored 7000;
the deck's 14 supporters scored 8800 and its 18 items 8600. With four slots, the action was never
expanded — so in a 98-loss review we found games where the agent held a Basic in hand for seven
consecutive turns with an empty bench, and lost to a board wipe. **The tree had not judged the move
badly. It had never seen it.**

Three deployments of that one insight, on the ladder:

| Form | Mechanism | Result |
|---|---|---|
| Candidate injection | force into shortlist; search still judges | **709** |
| Rule override | hard-code the action, bypass search | 582 / 604 / 676 |
| Feature encoding | add to value-net inputs | tie |

Injection widens what the search can consider. Overrides replace its judgment, and lost three times
out of three. Everything in `_bible_pack_idx` follows the injection form for this reason.

## 4. Value network

| | |
|---|---|
| Architecture | sparse embedding (10,400 × 64) + 2-layer MLP, ~0.67M parameters |
| Inference | numpy forward pass — no torch in the submission bundle |
| Label | AlphaZero-style: `0.5 × outcome + 0.5 × root search value` |
| Features (v9) | card-identity zones (hand / active / bench / discard / stadium) + 31 board-texture scalars |
| Fallback | if numpy or the weights are unavailable, the tree falls back to a hand-written eval |

### "Predicting well ≠ playing well"

Four consecutive generations improved validation AUC and produced no win-rate gain. The three fixes
that worked all supplied *truer information* rather than more effort:

**Soft labels.** A final win/loss (±1) is an extremely sparse signal for a position twenty turns from
the end. Blending the search's visit distribution with the root evaluation broke a five-generation
plateau in one training run.

**Vision.** An audit of the feature encoding found it contained **no Stadium, no bench HP, no bench
attached energy, and no knockout-target signal**. The network was not failing to learn; it could not
see the board. Adding a stadium zone and 15 board-texture scalars produced, after **21 seconds of
training**, a network that beat the incumbent 62% over 300 games.

**Curriculum.** The new family dominated locally and stalled on the ladder. Single-variable
diagnosis: training data came from ten top-of-ladder decklists, while our agent actually played
mid-ladder opponents — where it scored 63.1% against the incumbent's 72.6%.

### The ceiling

Three further generations of deck-specific self-play (380,000 samples) produced no measurable gain:
80.2% vs 79.1%, **p = 0.558**. The reason is structural. Our network is a *leaf evaluator* trained on
data generated by our own tree, so its ceiling is our own tree — a bootstrap trap. Training the
policy directly against the environment (`train/rl_selfplay.py`) has no such ceiling, which is why
three days of it matched a month of this.

## 5. Never crashing

Every layer degrades rather than throws:

- `policy.make_agent` wraps every decision; any exception returns `list(range(minCount))`, a legal choice
- `search_agent_tree` falls back to the heuristic agent on any search failure
- deck-list lint (60 cards, ≤4 copies of any non-basic-energy card) runs at three chokepoints
- `battle_start`'s error code is checked rather than passed silently to the next call

Measured across **1,011 recorded ladder episodes: 866 clean double-DONE, exactly 1 INVALID (0.1%)**,
no crashes, no timeouts. A forensic review of the final build's 16 losses found zero mechanical
defects — the losses came from resource starvation, and in 77 of 82 empty-bench decision points
there was simply no Pokémon in hand to play.
