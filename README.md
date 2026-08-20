# PTCG AI Battle Challenge — Team Lawliet

Source for our agent in the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(The Pokémon Company × Kaggle × Matsuo Institute × HEROZ, June–August 2026).

Final Simulation result: **174–182 across 356 ladder episodes (48.9%)** — statistically indistinguishable
from even, around the 43rd percentile of 6,809 teams. We report win counts rather than rating because a
single day's rating on this ladder swings 43 points, which in this part of the field is 600 places. This repository is published as the code companion to our
Strategy writeup, whose central claim is that **this environment structurally penalises search** —
a conclusion we reached by building a search agent for a month and then falsifying it in three days.

If you are here for one thing, make it [`tools/`](tools/). The measurement chain in that directory is
what we would keep if we had to throw the rest away.

---

## What is actually here

```
agent/      the submitted agent — PIMC determinization + macro-action PUCT tree
              + learned value net + playbook candidate injection
train/      self-play data generation, value-net training, and the 72-hour RL sprint
tools/      the measurement apparatus: arena, significance gate, replay forensics, sentinel
gauntlets/  opponent decklists mined from real ladder replays (our sparring pools)
docs/       architecture, the playbook, and the full experiment log
```

**Not included, and why:** the game engine (`cg/`, distributed as a compiled library by the
organisers) is not ours to redistribute. Neither are trained network weights, ladder replay
archives, or the card database — all obtainable from the competition page. Everything here is
code we wrote.

## Running it

You need the competition's engine directory and card data on the competition page, plus Python 3.12
and numpy. The submitted agent does **not** require torch — inference is a numpy forward pass.

```bash
# evaluate a deck against a sparring pool
python tools/arena_eval.py agent/decks/deck_stardom87.csv \
    --pilot tree --net <value_net.npz> --gauntlet gauntlets/real --games 200

# is a difference real, or noise?
python tools/ab_ztest.py <W1> <L1> <W2> <L2>

# pull a submission's real ladder games and split them win/loss for forensics
python tools/loop_autopsy.py <submissionId> 40
```

## The four layers of the agent

| Layer | File | Problem it solves |
|---|---|---|
| Opponent recognition | `agent/search_agent.py` | 16 modal decklists mined from replays; match on revealed Pokémon before sampling |
| PIMC determinization | `agent/search_agent.py` | converts one imperfect-information problem into D perfect-information ones |
| Macro-action PUCT tree | `agent/search_agent_tree.py` | branches **only at our own MAIN decisions**, so no value negation is needed |
| Value net | `agent/value_net.py`, `agent/features.py` | leaf evaluation; numpy inference, AlphaZero-style soft labels |
| Playbook injection | `agent/policy.py` (`_bible_pack_idx`) | human card knowledge enters as *candidates*, never as an override |

The last row is the one that matters. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Three results worth your time

**1. Where knowledge enters beats whether it is correct.** The same insight, deployed three ways:

| Form | Ladder |
|---|---|
| Candidate injection (search still judges) | **709** |
| Rule override (hard-coded) | 582 / 604 / 676 |
| Feature encoding (into the value net) | tie |

The search's shortlist is truncated at top-4 by static priors. The correct action scored 7000 while
the deck's 14 supporters scored 8800 — the search had not judged it badly, it had never seen it.

**2. Never hand a search engine a constant.** An attack whose damage scales with the opponent's hand
size is reported by the engine as `damage=0`. We supplied a static estimate of its median. Win rate
fell **60% → 38%**: the constant reached the lethality checker, and the tree began committing to
knockouts that did not exist. A number that is right on average is a lie at every specific board state.

**3. A three-day policy network matched a one-month search engine.** `train/rl_selfplay.py`
(REINFORCE + value baseline, dense reward shaping on prize differential) decides in **0.3 ms**;
our PIMC tree takes **~6 s**. On the ladder: **52.9% vs 55.6%, p = 0.81** — indistinguishable.

## The measurement apparatus (`tools/`)

The competition's ladder is a noisy instrument, and most of our early mistakes came from not knowing
how noisy:

- a submission resolves **≈35 games in its first 8 hours** → minimum detectable win-rate difference
  is **±24 percentage points**
- byte-identical submissions sent a minute apart settle **12–18 rating points** apart
- the same bytes read **732 in the morning and 571 that afternoon**

So `tools/ab_ztest.py` enforces one house rule without exception: **`p > 0.05` is a tie.** No
"directionally positive" exceptions — we retracted a published internal conclusion once for
violating it. And `tools/sentinel.sh` reports each submission's exact age, because comparing arms of
different ages is how you fool yourself.

`tools/loop_autopsy.py` pulls a submission's real ladder games, splits them win/loss, and stores the
replays — every claim in our writeup about *why* we lost traces back to games this pulled.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the agent in detail, with the reasoning behind each layer
- [`docs/PLAYBOOK.md`](docs/PLAYBOOK.md) — the deck's doctrine, extracted from 68 games of a rank-87 player
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — every A/B we ran, including all the failures

## A note on the comments

Inline comments are largely in Chinese. They are the original engineering record — many carry the
experiment and sample size that motivated the line ("07-23, 3 seeds, 144 games: 57.6% vs 48.6%"), and
translating them would have cost that provenance. The English documentation in `docs/` covers the
same ground for readers who need it.

## License

MIT for our code. Pokémon card data, names, and the game engine belong to their respective owners;
nothing in this repository redistributes them.
