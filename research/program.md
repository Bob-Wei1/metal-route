# Autoresearch program — improve the autorouter's corpus completion

You are an autonomous research agent improving this PCB autorouter, in the spirit of
karpathy/`autoresearch`: make ONE change, run a fixed eval, keep it only if it beats the
champion, otherwise revert. Then stop. The loop will call you again for the next one.

## Objective (the one number)

Maximize **`completion_rate`** on the real-board corpus benchmark
(`metalroute bench-corpus` over `benchmarks/corpus/`, 112 real multi-layer boards).
Tiebreaker / must-not-regress: **`fully_routed_boards`**.

- Current champion: see `research/baseline.json` (re-read it every iteration — don't
  trust a number memorized here).
- Starting point: **73.8%** completion (2337/3167 net-segments), **32/112** boards fully
  routed. **830 net-segments are unrouted** — that is the headroom.

## One experiment

1. **Pick ONE hypothesis** — a single, motivated change (see directions below). One change
   per experiment so the signal is attributable.
2. **Make the edit** in the router source (Rust).
3. **Eval:** `bash research/eval.sh exp-NNNN`
   (writes `research/results/exp-NNNN.json`; prints a `RESULT …` line).
4. **Score:** `python3 research/score.py --compare research/baseline.json research/results/exp-NNNN.json`
   — it prints per-group deltas and a final `VERDICT: KEEP` or `VERDICT: DISCARD`.
5. **Record + decide:**
   - **KEEP:** `cp research/results/exp-NNNN.json research/baseline.json`, append a row to
     `research/leaderboard.md`, then `git add -A && git commit -m "exp-NNNN: completion
     A→B%, full X→Y (<one-line what you changed>)"`.
   - **DISCARD:** `git checkout -- <files you edited>` to revert, append a `(discarded)`
     row to `research/leaderboard.md` (so the idea isn't retried blindly). Leave
     `research/baseline.json` untouched.
6. **Stop.** (A build failure or a sweep that scores fewer boards is an automatic discard.)

## Hard rules

- **ONE change per experiment.** Must compile (`eval.sh` fails fast → discard).
- **Must stay deterministic** — routing is byte-identical across runs today; if your
  change makes two evals differ, discard it.
- **Editable surface (free Rust edits):** the router crates — `crates/mr-cpu/`
  (especially `negotiated.rs`), `crates/mr-core/`, `crates/mr-srj/`, and the
  resolution/grid policy (`default_resolution` and `route_problem` in
  `crates/mr-cli/src/lib.rs`).
- **DO NOT TOUCH (these are the referee — changing them is cheating the metric):**
  - `benchmarks/corpus/**` (the test boards)
  - `crates/mr-cli/src/corpus.rs` (the scorer / what "routed" means)
  - `research/eval.sh`, `research/score.py`
  - Do not special-case corpus inputs by name/shape.
- Speed/GPU is **out of scope** — completion only.

## Tuning + algorithm surface (where the gains likely are)

In `crates/mr-cpu/src/negotiated.rs`:
- `MAX_ITERS` (60) — negotiation iterations before legalization; more may resolve
  congested boards (watch runtime).
- `pfac = 1 + iter` ramp — how fast present-congestion pricing grows.
- `SCALE` (16) and the cost model (`history`, present, clearance halo weights).
- rip-up budgets / ordering in legalization; net ordering heuristics.
- via cost / keepout (`ViaModel`) — these boards are multi-layer, so via behavior matters.

Other ideas:
- Resolution / grid-cell sizing (`default_resolution`) — too coarse loses routability,
  too fine blows up the grid; per-board sizing may help dense `srj15` boards.
- Congestion handling on the densest boards (look at which boards fail).

## Diagnose before you guess

Render the gallery to see *which* boards/nets fail and why (overlaps, congestion, vias):

```
scripts/bench-corpus.sh            # SVG gallery + report into benchmarks/runs/<ts>/
open benchmarks/runs/<ts>/index.html
```

Failures are sorted first in each group. The per-board breakdown is in every
`research/results/*.json` (`per_board[]`) — target the boards with the most unrouted nets.
