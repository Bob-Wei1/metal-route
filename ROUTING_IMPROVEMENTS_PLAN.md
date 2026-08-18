# metalroute — Routing Improvements Plan (parallel agent-team handoff)

**Audience:** an orchestrating agent that will run a team of sub-agents (in parallel,
with git-worktree isolation) to implement five routing improvements, then integrate
and benchmark them. This document is self-contained — you do not need any prior
conversation context.

**Repo:** the `metal-route` Cargo workspace. Current `main` HEAD
already contains the prerequisite fix described in §3.

---

## 1. Objective

Improve the `metalroute` CPU PCB autorouter on two axes:
- **DRC**: reduce genuine inter-net clearance violations in the emitted (DRC-verified) output.
- **Completion**: reduce unrouted nets on dense / over-budget / single-layer boards.

The project's standing preference is **strictly-DRC-clean over completion** — a modest
completion loss in exchange for a large DRC reduction is acceptable, **but it must be
quantified and reported**, never hidden. DRC-reduction levers and completion levers are
in tension (see §7 D1↔C1) and must be benchmarked **together**, not in isolation.

---

## 2. System overview

Pipeline: `SimpleRouteJson`/DSN board → non-uniform **Hanan grid** rasterise (`mr-srj`) →
**negotiated-congestion A\*** router, 4-connected orthogonal + through-vias
(`mr-cpu/src/negotiated.rs`) → **beautify** orthogonal staircases into 45° diagonals
(`mr-srj/src/smooth.rs`) → exact-geometry **DRC** oracle (`mr-drc`) verifies copper-to-copper
spacing.

Clearance model: the router enforces inter-net spacing with a grid **halo** that hard-blocks
foreign cells within `clearance + track_w` of routed copper; foreign **pads** are inflated in
the base grid by `block_margin = clearance + track_w/2`; via keepout = `via_pad/2 + clearance + track_w/2`.

Key constants: `DEFAULT_CLEARANCE_MM = 0.15`, `DEFAULT_TRACE_WIDTH = 0.15`,
`VIA_PAD_MM = 0.45` (radius 0.225 ≫ track half-width 0.075) — all in `crates/mr-cli/src/lib.rs`.

### Key files & line anchors (current `main` HEAD)
- `crates/mr-srj/src/lib.rs` — `build_grid_lines` (~300, Hanan lines + fill channels),
  `enforce_budget`/`CELL_BUDGET` (~275/377, drops fill lines when grid too big),
  `rasterize_with_layers` (pad inflation + `block_margin_mm`/`foreign_margin_mm` ~700–830),
  `pad_cells_for_point` (own-pad escape halo + foreign clip ~890–960), `decompose_connections` (~1050).
- `crates/mr-cpu/src/negotiated.rs` — negotiation loop (~500–1100), `route_legal` (~1259),
  `ring_conflict` (via annular-ring guard ~1324), `stamp_owner` (~1520), `for_each_halo_cell` (~1399).
- `crates/mr-core/src/lib.rs` — `ViaModel` (~314), `keepout_mm`.
- `crates/mr-cli/src/lib.rs` — `VIA_PAD_MM` (45), `route_problem` (~289), via keepout (~332),
  `route_dsn_problem` (~856).
- `crates/mr-srj/src/smooth.rs` — `beautify_traces`/`beautify_run`, `legalize_clearance` (~191;
  treats vias + endpoints as IMMOVABLE anchors ~112/168).
- `crates/mr-drc/src/lib.rs` — exact oracle `DrcBoard::check`, `Feature`/`Shape` (~351),
  `feature_gap` (~427). **Do NOT weaken the DRC.**
- `crates/mr-cli/src/drc_board.rs` — solution→DrcBoard + net-identity reconstruction for the corpus DRC.

---

## 3. Prerequisite already done — DO NOT redo (committed on `main`)

The **dominant** clearance violation (routed-segment-vs-foreign-pad) was fixed by clipping
the own-pad **escape halo** (`pad_cells_for_point`) against the foreign pad's *clearance band*
(true geometric margin `min_clearance + track_w/2`), not just the bare pad rect. A new
`min_clearance_mm` param threads through `rasterize_with_layers → decompose_connections →
pad_cells_for_point`. Result: **corpus DRC 3950→1900 (−52%), −15 nets, clean boards 38→39;
DSN fixture 236→126.** This is the new baseline below.

A **via-vs-pad placement guard** inside `ring_conflict` was tried and **REVERTED** — it was
net-negative on the corpus (DRC unchanged, −19 nets). **Lesson: do via-vs-pad as a GRID
RESERVATION (inflate pads), never as a placement-time veto.** See D2.

---

## 4. Current baseline (committed `main` — the numbers to beat)

- **Full corpus** (`benchmarks/corpus`, 112 boards): **DRC 1900**, routed **2708/3167 (85.5%)**,
  **76/112** full boards, **39** clean boards. (~12 min wall, parallel.)
- **Subset** (7 boards, ~1–2 min): **DRC 189**, routed **150/155**. Per board:
  `bugreport16` 10/10 DRC 0 · `bugreport30` 8/12 DRC 18 · `bugreport48` 16/16 DRC 8 ·
  `bugreport64` 22/22 DRC 28 · `sample11` 26/26 DRC 79 · `sample43` 55/56 DRC 50 ·
  `sample55` 13/13 DRC 6.
- **DSN fixture** `bench/fixture_fresh`: **126** clearance violations, 93/141 two-point nets, 33 fully-connected.

### Residual-violation profile (evidence that shapes the work)
Distance-to-rule histogram across 5 dirty boards (rule 0.15 mm): **40% in `[0.10,0.145)`**
(fractional-track near-misses), 24% `[0.05,0.10)`, 20% hard overlap `<0`. Spatially, the
vast majority **hug a foreign PAD** (sample11: 76/79 within ~0.25 mm of a pad); vias are one
contributing cluster, not the majority. DRC is concentrated: top-5 boards = 43% (bugreport50
alone = 373), top-15 = 71%. Unrouted is concentrated too: top-5 boards = 71% of 459 unrouted;
**12 over-budget boards** drop ALL fill lanes (bugreport05: 733×878 features, 0 fill,
87/228 routed). Unroutable-alone (233) is 100% on multi-layer boards; congested (226) splits
82 single-layer / 144 multi-layer.

---

## 5. Validation harness (every agent uses these)

**Build:** `cargo build --release -p mr-cli` → binary `target/release/metalroute`.

**Subset (fast, ~1–2 min)** — use a UNIQUE dir/out per agent to avoid collisions:
```bash
SUB=/tmp/rs_$LEVER; rm -rf $SUB; mkdir -p $SUB/bug-reports $SUB/srj15
for b in bugreport16-d95f38 bugreport64-be7d8f bugreport48-569cfe bugreport30-2174c8; do cp benchmarks/corpus/bug-reports/$b.srj.json $SUB/bug-reports/; done
for s in sample55-region-reroute sample11-region-reroute sample43-region-reroute; do cp benchmarks/corpus/srj15/$s.srj.json $SUB/srj15/; done
./target/release/metalroute bench-corpus --dir $SUB --out /tmp/$LEVER.json
python3 -c "import json;r=json.load(open('/tmp/$LEVER.json'));print('DRC',sum(b.get('drc_violations',0) for b in r['per_board']),'routed',sum(b['nets_routed'] for b in r['per_board']),'/',sum(b['nets_total'] for b in r['per_board']),'|',[(b['board'][-9:],b['nets_routed'],b.get('drc_violations')) for b in r['per_board']])"
```

**Full corpus (slow, ~12 min — INTEGRATION ONLY, not per-agent):**
`./target/release/metalroute bench-corpus --dir benchmarks/corpus --out /tmp/corpus.json`

**Per-violation DRC detail** (one-board dir, stderr): prefix with
`DRC_DEBUG=1 DRC_DEBUG_CAP=400`. Output lines: `class L<layer> @(x,y) nets=(a,b) measured=… required=…`.

**DSN fixture** (~40–90 s): `./target/release/metalroute drc --input bench/fixture_fresh/fixture.dsn --skip-nets=GND --skip-nets=+5VA --skip-nets=-5VA --skip-nets=3V3 --skip-nets=5V`

**Per-board JSON fields:** `drc_violations`, `nets_routed`, `nets_total`,
`unrouted:[{name,reason}]` (reason ∈ congested | unroutable_alone | …), `congestion_peak`,
`grid_w`/`grid_h`/`grid_layers`, `wall_ms`.

**Tests:** `cargo test --workspace` MUST stay green (238 tests currently). Update an assertion
only when a real geometry change legitimately shifts it (e.g. grid size); never weaken a
meaningful one.

---

## 6. Hard constraints (apply to every agent)

1. **Clearance-off byte-identical fast path:** when `clearance_cells == 0` / `min_clearance == 0`
   / `keepout_mm == 0`, behaviour must be byte-identical to before. Gate new logic behind the
   clearance-active condition. The test `rasterize_clearance_zero_is_byte_identical` guards this.
2. **Determinism:** same input → same output (DRC + routed counts) across runs. No `Date/random`.
3. **Tests green:** `cargo test --workspace` passes.
4. **Minimal, focused diff** in the agent's owned function(s) only — so changes integrate cleanly.
5. **Report the full tradeoff:** subset DRC AND completion (per board), before→after, plus any
   test/assertion changes and the reasoning.
6. **Do NOT weaken `mr-drc`** (the oracle). Do NOT run the full corpus per-agent (too slow under
   contention) — subset only; integrator runs the full corpus.

---

## 7. Work items

Folded: **D2 is part of A1** (same code region as D1). Ranked by leverage.

### A1 — D1 (pad-clearance safety band) + D2 (via-class pad reservation)  — `mr-srj/src/lib.rs`
- **Mechanism/evidence:** base grid inflates foreign pads by `block_margin = clearance + track_w/2`
  (= 0.225), which reserves the node *center*; but emitted copper is the segment *between* nodes
  + endpoint snap-back to the exact pad + 45° chamfers — none on the reserved node. → 40% of
  residual violations land in `[0.10,0.145)`, 76/79 sample11 violations hug a pad. Separately,
  vias (pad radius 0.225 ≫ track 0.075) are inflated only by the track-sized margin, so a via can
  land with its pad biting a foreign pad's clearance band (hard overlaps).
- **Change (D1):** widen the foreign-**pad** halo by a safety band so even a snapped/chamfered
  segment stays clear — e.g. pad `block_margin → clearance + track_w` (or `clearance + k·track_w`,
  `k∈[0.5,1]`). Apply the SAME widening to the matching `foreign_margin_mm` escape clip in
  `pad_cells_for_point` so the two stay consistent. Make `k` a named constant with rationale.
- **Change (D2):** inflate pads to `max(track_w/2, VIA_PAD_MM/2)` on via-allowed layers (grid
  reservation), so the grid never offers a via node too close to a foreign pad. This is the SAFE
  form of the reverted placement guard.
- **Sweep & report** `k ∈ {0, 0.5, 1.0}` on the subset: DRC and routed per board. Pick the band
  that maximizes DRC reduction at acceptable completion loss — but **leave it parameterized**, the
  integrator joint-tunes it against C1.
- **Expected:** −700…−1000 DRC corpus-wide; converts many full-but-dirty boards to clean.
- **Tradeoff/risk:** ↓ completion (narrower escape corridors). Medium-low effort. **This is the
  D-side of the D1↔C1 tension — see §8.**
- **SOTA framing:** this is the grid form of Minkowski C-space inflation + a per-class clearance
  matrix (vias inflate more than tracks). Reason about the *swept segment*, not the node.

### A2 — C1 (adaptive / raised `CELL_BUDGET`)  — `mr-srj/src/lib.rs`
- **Mechanism/evidence:** `CELL_BUDGET = 160_000` is a hardcoded const (~275); `enforce_budget`
  (~377) drops fill lines when the grid exceeds it. 12 boards exceed it; the worst drop ALL fill
  lanes (bugreport05: 0 fill, 87/228 routed; bugreport50). With no fill lanes, distant features
  have no channel between them → nets become "unroutable_alone". `sample43` (in the subset) is the
  over-budget test case.
- **Change:** make the budget adaptive (scale with feature-line count / board area) and/or raise it,
  and/or expose as an env/CLI knob. Re-sweep — a prior config had a sharp optimum at 160k under a
  DIFFERENT grid scheme; re-validate against the current Hanan baseline.
- **Report:** subset completion + DRC + resulting grid_w/grid_h + wall_ms (timing noisy under
  contention — note it). Focus on `sample43`.
- **Expected:** +100…150 nets corpus-wide (bugreport05 alone = 141 unrouted).
- **Tradeoff/risk:** bigger grids = slower A\* + more memory (the GPU-memory concern is moot; CPU
  only). Low implementation effort. **This is the C-side of the D1↔C1 tension.**
- **SOTA framing:** the principled alternative is a coarse **global-route + DP layer-assignment**
  pass before the fine grid (so the fine grid runs only locally) — out of scope here, but note it
  as the follow-on if raising the budget hits a memory/time wall.

### A3 — C2 (BGA/LGA escape fanout)  — `mr-srj/src/lib.rs` `build_grid_lines`
- **Mechanism/evidence:** dense regular pad arrays lack enough fill lines between pad rows/cols to
  form an escape lane per pin. Example: `bugreport23-LGA15x4` (2.54 mm pitch, 1.6 mm pads, 2-layer)
  gets only a 47×14 grid → 37/45 unroutable-alone, 0 DRC. Pad-edge gap ~0.94 mm has room for one
  0.45 mm channel, but inner array pins have no path out.
- **Change:** add denser local fill lines between closely-spaced pad edges (sub-pitch escape lanes)
  so inner pins get escape paths; optionally bias inner-pin escape to a less-congested layer via vias.
- **Test target:** `bugreport23-LGA15x4` is NOT in the standard subset — copy it into a one-board
  dir and benchmark it directly (`find benchmarks/corpus -iname '*LGA*'` / `*bugreport23*`).
- **Expected:** +40…80 nets on array-package boards.
- **Tradeoff/risk:** HIGH effort (real escape-routing). **Interacts with C1** — more lanes = more
  cells, can blow the budget; coordinate at integration. If a full fanout pre-pass is too large,
  deliver a minimal "guarantee ≥1 escape lane between adjacent same-orientation pad edges" version.

### A4 — D3 (movable vias in the post-route legaliser)  — `mr-srj/src/smooth.rs`
- **Mechanism/evidence:** `legalize_clearance` (~191) treats `Via` points and endpoints as
  IMMOVABLE (~112/168) and only nudges interior wire vertices. So via-adjacent and endpoint-pinned
  pad grazes are structurally unreachable — 40 full-but-dirty boards retain 480 DRC.
- **Change:** allow nudging a via landing within its own pad / own-net region (and/or shifting it
  one grid node along the trace) when the move **strictly increases** the worst foreign-clearance
  gap on BOTH incident segments — reuse the existing monotone, DRC-validated acceptance gate so it
  can never introduce a short. Re-snap both incident segments to preserve connectivity.
- **Expected:** −150…−300 DRC, concentrated on full-but-dirty boards (`bugreport64`, `sample11`).
- **Tradeoff/risk:** **ZERO completion risk** (post-route, connectivity-preserving). Medium-high
  implementation care (moving a via reshapes two segments). The **safest** lever — good first merge.

### A5 — C3 (single-layer congestion)  — `mr-cpu/src/negotiated.rs`
- **Mechanism/evidence:** 10 single-layer boards route 139/221; 82 of 226 congested nets are
  single-layer — genuine planar contention with no via escape.
- **Change (exploratory):** strengthen negotiation rip-up ordering / history-cost schedule for
  single-layer boards (e.g. better net ordering, stronger history accumulation, targeted reroute of
  the most-contended nets). Do NOT regress multi-layer boards.
- **Test target:** single-layer boards (`layerCount == 1`); find them and benchmark a handful.
- **Expected:** +20…40 nets (planar routing has a hard combinatorial ceiling; a prior optimization
  loop already plateaued here).
- **Tradeoff/risk:** HIGH effort, LOW/ceiling-bound ROI. **If no clean win, report a negative
  result honestly** rather than forcing a regression-prone change.

---

## 8. Orchestration strategy

### Worktree isolation + file ownership
Run each agent in its **own git worktree** (branched from current `main`) so parallel edits cannot
clobber each other. Ownership (chosen to minimize cross-agent conflicts):

| Agent | Lever(s) | Owns (functions, not whole file) |
|---|---|---|
| A1 | D1 + D2 | `mr-srj/lib.rs`: pad inflation in `rasterize_with_layers`, `block_margin_mm`/`foreign_margin_mm`, `pad_cells_for_point` clip |
| A2 | C1 | `mr-srj/lib.rs`: `CELL_BUDGET`, `enforce_budget` |
| A3 | C2 | `mr-srj/lib.rs`: `build_grid_lines` fill-channel logic |
| A4 | D3 | `mr-srj/smooth.rs`: `legalize_clearance` |
| A5 | C3 | `mr-cpu/negotiated.rs`: negotiation loop / cost |

A1, A2, A3 all touch `mr-srj/lib.rs` but in **different functions**, so merges are mostly
auto-resolvable; A4 and A5 are in separate files (clean). Each agent must report its full
`git diff` and commit in its worktree.

### Integration phase (orchestrator owns this — do NOT delegate)
1. Create an integration branch from `main`.
2. Merge / apply agents in this order (safest first): **A4 (D3) → A2 (C1) → A3 (C2) → A1 (D1+D2) → A5 (C3)**.
   Resolve the (function-disjoint) `mr-srj/lib.rs` overlaps.
3. `cargo test --workspace` after each merge.
4. **Joint-tune D1↔C1:** A1 narrows escape corridors (↓completion); A2/C1 adds lanes (↑completion).
   Run the FULL corpus across a small grid of (D1 band `k`, C1 budget) settings and pick the point
   that maximizes DRC reduction subject to **completion ≥ baseline − a stated tolerance** (e.g. not
   below ~2700/3167). Report the chosen point and the tradeoff curve.
5. Final full-corpus + DSN-fixture numbers vs. the §4 baseline; confirm clearance-off byte-identical
   + determinism + all tests green. Commit on a branch for human review (do NOT auto-merge to main —
   the completion/DRC tradeoff is a human judgment call).

### Suggested commit/PR granularity
D3 (zero-risk) can land on its own. D1+D2+C1 should land together (joint-tuned). C2 and C3 land
only if they show a clean positive; otherwise report and shelve.

---

## 9. Gotchas / things already tried

- **Via-vs-pad PLACEMENT guard in `ring_conflict` was net-negative (−19 nets, no DRC gain) and
  reverted.** Do via clearance as GRID RESERVATION (D2), not a placement veto.
- **Beautify is NOT the villain** — it *reduces* violations (raw 1845 → beautify 461 → legalize 421
  on the subset). Don't "fix" beautify; the source is the raw grid/pad geometry.
- The router is **4-connected** (orthogonal); diagonals come only from beautify. Native 45° would
  require an octilinear Hanan grid (±45° lines) — out of scope here; if attempted later, the
  clearance inflation must account for the √2 diagonal-spacing shrink.
- `min_clearance` is **null** on most corpus boards; the caller's default (0.15) is passed via the
  new `min_clearance_mm` param. Use the TRUE clearance, never the `ceil`-rounded `clearance_cells·resolution`.
- Some residual violations are **input pad-vs-pad** (board geometry already < clearance) — unfixable
  by routing; don't chase them.

---

## 10. Definition of done

A reviewed branch where: DRC is materially below 1900 on the full corpus with completion held at or
near baseline (the D1↔C1 joint-tuned tradeoff stated explicitly), the safe D3 win is included, C2/C3
either landed with positive evidence or reported as negative results; `cargo test --workspace` green;
clearance-off byte-identical fast path and determinism preserved; before→after corpus + subset + DSN
fixture numbers reported per the §4 baseline.
