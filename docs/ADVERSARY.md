# Adversarial ship placement

The solver assumes the enemy places ships **uniformly at random**. The adversary asks
how many extra shots a placement *ruleset* can force if it doesn't. A ruleset is a
pattern for placing ships, both on the board and relative to each other; it is never a
fixed layout. Every game draws a fresh fleet from it.

Code: `adversary.py`. Runner: `benchmarks/adversary_search.py`. Tests:
`tests/test_adversary.py`. Status: experimental research tool; see [STATUS.md](STATUS.md).

## How a ruleset is defined

A policy is a probability distribution over legal fleets:

    P(fleet) ∝ exp( Σ w_k · z_k(fleet) )

Here `z_k` is a placement feature, standardised against uniform placement under the
same rules. A weight of +1 means "prefer fleets about one standard deviation higher in
this feature". All-zero weights are uniform placement.

| Feature | Meaning (raw value, 0..1) |
|---|---|
| `edge` | share of ship squares on the outer ring |
| `ring1` | share of ship squares on the ring just inside the edge |
| `contact` | share of ship pairs that touch, diagonals included (always 0 under no-touch) |
| `parallel` | side-by-side stacks: same-orientation pairs sharing 2+ adjacent squares, per ship |
| `aligned` | share of ships in the majority orientation |
| `spread` | mean distance between ship centres ÷ board diagonal |
| `obscurity` | how unlikely the ship squares look to a uniform-prior solver: 1 − mean prior ÷ max prior |

**Sampling.** Fleets are drawn by Metropolis MCMC (800 moves per fleet by default). The
chain starts from a uniform random legal fleet. Each move tries to put one ship
somewhere uniformly random: illegal moves are rejected (overlap, or touching under the
house rule), and legal ones are accepted with probability min(1, e^{ΔE}). The proposal
is symmetric, so the policy is the chain's stationary distribution. A finite run
from a uniform starting layout is **not an exact draw from that distribution**;
mixing can be slow for extreme weights and tight no-touch constraints. A small-board
test compares feature averages with brute-force enumeration but does not establish
mixing for every fleet or parameter setting.

**Presets:**

| Preset | Weights |
|---|---|
| `uniform` | none |
| `edge_hugger` | edge +2.5 |
| `obscure` | obscurity +2.5 |
| `clustered` | contact +2, spread −2 |
| `spread_out` | spread +2.5, contact −1.5 |
| `parallel_stack` | parallel +2.5, aligned +1.5 |
| `aligned` | aligned +2.5 |

Custom rulesets use the form `"edge=2,spread=-1"`.

## Search and evaluation

1. **Search.** The cross-entropy method tunes all seven weights to maximise the
   solver's average shots. It uses training fleets (`--train-seed`, default 300001)
   and a reduced sample budget per shot. The defaults are 6 iterations × 16 candidates
   × 100 games.
2. **Evaluation.** Uniform placement, every preset, and the searched policies (the
   final CEM mean and the best single candidate) are played against each evaluation
   solver. This uses held-out fleets (`--eval-seed`, default 400001) and the full
   1,500-layout budget. Every policy faces identically seeded solvers.
3. **The report shows:**
   * mean shots, median, 95th percentile, worst game and games needing 60+ shots;
   * the hunt/target split;
   * the difference from uniform, with Welch and Bonferroni-corrected intervals;
   * each policy's average features ("what the pattern looks like");
   * a square-by-square map of where ships end up, for uniform and for the hardest
     policy.

   A JSON file next to the report keeps the weights and every game's shot count.

```bash
python benchmarks/adversary_search.py --jobs 8 --out runs/adversary_standard.md
python benchmarks/adversary_search.py --no-touch --jobs 8 --out runs/adversary_no_touch.md
python benchmarks/adversary_search.py --skip-search --policies uniform "edge=3,parallel=1" \
    --eval-games 1600 --jobs 8 --out runs/adversary_custom.md
```

**House rules and variants:**
* `--no-touch` restricts the adversary to legal no-touch fleets and tells the solver
  the rule.
* `--size` and `--fleet "Name:len,..."` change the board and fleet.
* Only the touching rule the engine implements is supported: no contact, diagonals
  included.

**Defaults:** the search attacks `control`. Evaluation runs against `control` and the
validated `adaptive_parity:margin=0.4`, which shows whether the better baseline is also
more robust.

## Reading the results

* **The adversary is fixed while the solver is not adapting.** A ruleset that beats the
  uniform-prior solver tells you which assumption the solver over-relies on. It doesn't
  tell you the ruleset is good against a solver that learns.
* **The natural follow-up is a robust solver:** one that assumes a mixture of uniform
  and adversarial priors, or that infers the opponent's tendencies during a series of
  games. Benchmark it against both uniform and adversarial fleets.
* **Search results are tuned on the training fleets.** Trust the held-out evaluation
  table, not the search progress table.
* **Finite-chain sensitivity matters.** For candidate policies, repeat the held-out
  evaluation with a larger `--mcmc-steps` value and independent seeds before drawing
  conclusions. Treat results with poor mixing as exploratory.

## Preliminary sanity check (not a benchmark)

This was run while the tool was being built. It used only 150 held-out games per
policy, 800 layouts per shot, the presets only (no search), and `control` as the
solver. It confirms that the tool detects placement effects; the full runs above
should replace these numbers.

| Policy | Mean shots | Δ vs uniform (corrected CI) | What the fleets look like |
|---|---:|---|---|
| `obscure` | 51.3 | +5.6 (+3.3 to +8.0) | ships around the rim and corners (68% of squares on the edge) |
| `spread_out` | 48.7 | +3.0 (+0.6 to +5.4) | ships far apart, rarely touching |
| `edge_hugger` | 48.6 | +2.9 (+0.5 to +5.4) | 78% of ship squares on the edge |
| `uniform` | 45.6 | — | |
| `parallel_stack` | 45.4 | −0.2 (n.s.) | side-by-side stacks; fewer hunt shots, more target shots |
| `clustered` | 44.5 | −1.1 (n.s.) | ships in one area; finding one ship leads to the rest |
| `aligned` | 44.3 | −1.4 (n.s.) | all ships the same orientation |

The uniform-prior solver's main weakness is the rim. Its heat map is lowest at the
edges and corners, so it searches them last, and a ruleset that concentrates ships
there costs it several shots per game.
