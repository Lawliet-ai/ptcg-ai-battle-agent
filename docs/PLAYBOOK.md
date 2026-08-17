# The deck, and its doctrine

**3 Mega Starmie ex / 3 Mega Froslass ex / 3 Staryu / 3 Snorunt / 2 Cinderace / 1 TR's Articuno /
1 Cornerstone Ogerpon ex · 7 Basic Water + 1 Basic Fighting · 4 Buddy-Buddy Poffin · 4 Wally's
Compassion · 4 Lillie's Determination · 4 Surfing Beach · 3 Ultra Ball / 3 Pokégear / 3 Mega Signal /
3 Hilda · 2 Boss's Orders · 2 Xerosic's Machinations · 1 Salvatore · 1 Cheren · 1 Energy Retrieval ·
1 Hero's Cape (ACE SPEC)**

Decklist: [`agent/decks/deck_stardom87.csv`](../agent/decks/deck_stardom87.csv)

## Why this deck, for an agent

We did not select the strongest deck. We selected the strongest deck **an agent can pilot**:

**1. A shallow decision tree.** Every main attack costs one energy. Attach one, fire. There is no
multi-turn energy plan to get wrong — and cross-turn resource planning is precisely the state
dimension a search agent handles worst. Three-energy systems demand exact allocation several turns
ahead; this deck does not.

**2. Simple is not weak.** Twin 330/310 HP Mega walls, behind a full heal that returns energy to hand.

**3. It attacks the undefended bench.** Jetting Blow deals 120 to the Active *plus 50 to a chosen
benched Pokémon*, and that splash is free — it rides along with the main attack. The bench is the
most fragile real estate on the board: it holds evolution seeds and engine Pokémon at 70–110 HP, and
most decks carry no bench protection at all.

The third property is the deck's hidden engine. In the 68-game career we decoded, **22 of 40 wins
ended with 120+50 sniping rather than a large attack**. Killing an evolution seed deletes the
opponent's *next* Mega before it exists; killing an engine Pokémon cuts his energy supply at the
root; killing a wounded single-prize body converts straight into prize cards.

## The core loop

Because attacks cost one energy, **Wally's Compassion (full heal, energy returned to hand) costs
almost no tempo** — re-attach and fire the same turn:

```
attack -> full heal -> free pivot via Surfing Beach -> attack again
```

This invalidates 200–270 damage per opponent turn while we farm single-prize targets toward a
[1-1-1-1-1] prize map. Each Mega concedes three prizes, so the hard rule is **never surrender the
second one**.

## Structural immunity

Our attackers carry **no Abilities and only basic energy**. That blanks three of the field's most
common tech cards at once: special-energy removal (we run none), Ability-punishing damage (our
attackers have none), and effect-blocking energy (we deal plain damage). This was a construction
choice, not luck.

## Doctrine extracted from 68 games

We decoded the entire submission career of a rank-87 player using this list, decision by decision,
and kept only rules with measurable triggers and zero counterexamples:

| Rule | Evidence |
|---|---|
| Always choose to go first | 33/33 coin flips; 72% win going first vs 46% going second |
| Cinderace to the active slot at setup, always | 9/9 — its Ability only functions during setup, and it is the energy pump |
| First Mega on board by our second turn | 45/67 games |
| Never surrender the second Mega | every loss pattern is [x-3-x]; wins concede at most one |
| Cornerstone Ogerpon is discard fodder | played to the field 0 times in 68 games |
| Lillie's Determination before losing the first prize | draws 8 at six prizes, 6 afterwards |
| Xerosic is mutually exclusive with a Froslass attacker | stripping the opponent's hand also strips our own damage |

These enter the agent as **candidates in the search's shortlist** (`policy._bible_pack_idx`), never
as overrides — see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## The part we got wrong

After the doctrine was working, we wrote five amendments of our own from loss forensics. **All five
tested negative on the ladder** — two of them cost 22 points of win rate each
(see [`EXPERIMENTS.md`](EXPERIMENTS.md) §5).

Human priors are excellent at *generating* hypotheses and worthless at *ratifying* them. Every rule
above survived because a ladder A/B could not kill it, not because it sounded right.

## A note on copying the leaderboard

A community dataset of 74,634 games shows one archetype rising from 0% to 41% of the field in six
days while its win rate against that field fell from a pre-adoption peak of 65% to 48% at peak
popularity. Adoption lags performance by roughly three days.

We copied this list on 8/08 from rank 87. By our 8/13 census, the archetype had vanished from the
top 100 entirely — and our own version was performing better than the original by then, because the
value net and doctrine had been rebuilt around it.
