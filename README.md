# metalroute

A GPU-accelerated PCB **maze / global router** for Apple Silicon, written in Rust.
metalroute builds its own routing model and its own **Metal compute kernels** from
scratch, validates them against a CPU oracle, and benchmarks them on the
tscircuit autorouting problem format.

> **Platform:** macOS / Apple Silicon (Metal). The CPU router, adapters, and the
> whole test suite are portable; only the GPU backend (`mr-metal`) requires Metal.
> Linux/Windows users: the contribution is the **algorithm + reproducible
> benchmark**, not a binary.

This is a learning + craft + open-source project: understand PCB place-and-route
**and** GPU data-parallel algorithm design by building a vertical slice by hand.
See [`design.md`](design.md) for the full rationale and
[`the plan`](.claude/plans/) for the parallelized build breakdown.

## What's here

A 12-crate Cargo workspace, built contract-first so the pieces are independently
ownable and swappable behind one `Router` trait.

| Crate | Role |
|-------|------|
| `mr-core` | Contract: `Dims`/`Grid` (canonical row-major mapping), `NetEndpoints`, `BoardRoute`, `TieBreak`, `RouterError`, the `Router` trait. |
| `mr-grid` | Cost-grid construction: obstacle marking + Chebyshev clearance inflation. |
| `mr-fixtures` | Shared golden test cases: ASCII grid format, the 32×32 hand case, the M0 obstacle battery, the tie-break conformance case. |
| `mr-cpu` | **CPU routers:** Lee/Dijkstra, targeted A\*, separable prefix-min sweep, live bounded rip-up, and deterministic PathFinder-style negotiated routing with multilayer vias and physical clearance. |
| `mr-metal` | **Metal GPU kernels (macOS):** batched row/column/via prefix-min sweeps, canonical weighted/zero-cost paths, exact Hanan-edge isolated batches, and `MetalRouter`. |
| `mr-srj` | tscircuit **SimpleRouteJson** I/O: rasterize problems → grid, de-rasterize routes → `pcb_trace` solution soup. |
| `mr-ingest` | KiCad ingest via the `kicad-cruncher` CLI (shell-out + JSON). |
| `mr-oracle` | Route equivalence: equal cost + equal congestion (not bit-identical paths). |
| `mr-bench` | M2 go/no-go speedup projection + timing harness + Criterion bench. |
| `mr-server` | axum `POST /solve` speaking the tscircuit solver protocol. |
| `mr-bridge` | Freerouting handoff via `bed-of-nails` (DSN/SES) — the detailed-routing step. |
| `mr-cli` | The `metalroute` binary: `route`, `project`, `bench`, `handoff`. |

## Quick start

```sh
cargo build --workspace
cargo test  --workspace          # CPU suite + real-GPU Metal tests (on macOS)

# Route a tscircuit SimpleRouteJson problem into a solution soup
metalroute route --input problem.srj.json --out solution.json

# Project the M2 batch-GPU go/no-go for a given board
metalroute project --width 256 --height 256 --nets 500

# Run the local tscircuit-style benchmark and write a negotiated-router report
metalroute bench --out benchmarks/cpu_baseline.json

# Route the vendored real-board corpus + render an SVG gallery (see below)
scripts/bench-corpus.sh                      # -> benchmarks/runs/<ts>-corpus/index.html

# Serve the tscircuit solver protocol (for the official autorouting harness)
mr-server --port 1234            # then: POST /solve {simple_route_json: ...}

# Hand a routed board to Freerouting for detailed routing (needs bed-of-nails)
metalroute handoff --pcb board.kicad_pcb
```

## Results

### Correctness (CPU ↔ GPU oracle)

The CPU router is the correctness oracle. The Metal kernels are graded against it
on the shared fixture battery via `mr-oracle` (equal total cost + equal per-cell
congestion + equal unrouted set, under the shared canonical tie-break):

- M3 GPU wavefront field == CPU BFS field — **pass** (incl. the 32×32 wall, cost 93).
- M4 GPU prefix-min sweep field == CPU sweep field == CPU BFS field — **pass**.
- On the committed fixed fixtures and deterministic fixed-seed stress cases,
  `MetalRouter` and `LeeRouter` return the same canonical results for the covered
  weighted/zero/obstacle grids, multilayer batches, passable-pad overrides, chunk
  boundaries, and concurrent calls — **pass**. This is broad regression coverage,
  not an exhaustive proof over every cost grid.

The Rust source suite now contains **543 test cases** (533 passing and ten
explicitly ignored manual frontier/performance/live/real-board gates), up from
243, plus two passing doctests. The added cases
targeted cross-router contracts, zero-cost cycles, weighted and multilayer ties,
physical clearance on non-uniform grids, actual rip-up displacement, SRJ/DSN
layer and rotation semantics, deterministic DRC/oracle behavior, GPU batching and
memory caps, fixed-fixture and fixed-seed Metal equivalence, exact via-spacing
guards and exemption lookup, cached isolation diagnoses, parallel scheduling
determinism, coherent typed-SRJ rules, authoritative DRC acceptance,
topology-preserving interior and terminal-via repair, bounded
blocker-informed/dihedral legalization portfolios, and exact board-outline masks,
continuous board-edge DRC, legacy-first selection, and product-path parity.
`research/test_score.py` adds 13 workload-identity, aggregate-consistency, and
regression-gate tests for benchmark reports.

### M0 finding (the de-risk gate)

The single largest assumption was that GAMER's separable H-then-V prefix-min sweep
transfers to the PCB obstacle model. It does — **for distances**: the converged
sweep cost field is bit-identical to Lee's BFS across the whole obstacle battery.
The subtlety: a converged *cost* field does **not** carry the canonical *path* for
free — backward greedy descent can choose a longer equal-cost path or cycle on a
zero-cost plateau. CPU search and weighted Metal sweeps therefore label states by
`(cost, hop_count, lower_predecessor)`. Every reconstructed predecessor strictly
decreases hop count. Unit-cost Metal grids use the equivalent distance-only fast
path. (See `crates/mr-cpu/src/sweep.rs` and `crates/mr-metal/src/gpu.rs`.)

### Speed (honest)

Release timings on an M4, 128×128 grid, 64 independent nets. This is not a
like-for-like algorithm benchmark: targeted Lee stops once destinations settle;
the CPU field timing constructs complete source-distance fields but does not
reconstruct paths; Metal computes full fields and reconstructs the requested paths.

| Implementation | statistic | observed latency |
|----------------|-----------|-----------------:|
| CPU `LeeRouter` targeted paths | one timed route/process | 14.95–18.22 ms |
| CPU full distance fields | one timed batch/process | 26.28–27.67 ms |
| Metal batched fields + paths | median of 7 warm batches/process | **3.42–3.75 ms** |

Each range spans three isolated processes; the seven-sample Metal median is the
fourth sorted observation. The observed Lee-to-Metal elapsed-time ratio is
4.0–5.3× and the CPU-full-field-to-Metal ratio is 7.1–8.1×, but neither is a
like-for-like speedup for the reasons above. Cold runtime shader/pipeline setup
remains 22.5–26.5 ms, then a process-global context amortizes it. Unit/obstacle
grids avoid the weighted hop plane; committed fixed and fixed-seed cases check
that weighted and zero-cost grids retain canonical CPU equivalence.

Metal also exposes an exact weighted Hanan-edge isolated-route batch (including
vias, windows, and passable pads) behind a dependency-inverted CPU provider. The
experimental adapter packs each submitted window at its real cropped size and
reconstructs compact paths on the GPU. Matched warm real-board A/Bs remained neutral
or about 1–2% slower across 0.76M–5.26M submitted window-cells, so they did **not**
establish a reliable automatic crossover and the production negotiated router keeps
targeted CPU A* by default. Experimental offload is explicit with
`METALROUTE_EXPERIMENTAL_METAL_ISOLATED=1`; GPU contention or any command failure
immediately takes the exact whole-batch CPU fallback.

A separate research branch (`adba265`, not integrated) measured the remaining
headroom for retiring each ragged Metal field as soon as it converged. Exact
logical-work ratios (retired/current, lower is better) were 1.000/0.899 on
bug05-shaped open/heterogeneous micro workloads and 1.000/0.844 on bug50-shaped
workloads. All exceeded the predeclared 0.69 NO-GO boundary, so per-field
retirement was rejected rather than adding production complexity for modest
best-case savings. This was a synthetic logical-work probe, not a timed
active-retirement implementation or real-board speedup.

### tscircuit-style synthetic benchmark

`metalroute bench` generates ten deterministic 30-net SimpleRouteJson boards.
Against the exact pre-change run, negotiated routing moves from **206/300 (68.7%)
to 216/300 (72.0%)**, with mean routed cost 74.84. The final release sample took
0.653 s in the report and 0.97 s externally, versus the exact 1.680 s baseline
report timer. The deterministic quality result is the primary gate.
Counted self-halo snapshots make clearance-active negotiation parallel and
byte-identical across 1/2/4 Rayon threads; one bounded single-layer coordination
pass recovers the serial router's completion.

> The official `autorouting-dataset benchmark --solver-url` harness drives
> `mr-server`'s `/solve` endpoint directly; when its npm package is available, point
> it at a running `mr-server`. The local `bench` command is the reproducible
> substitute.

### Real-board corpus (`bench-corpus`)

The synthetic `bench` generator is uniformly easy. For *real* routing difficulty,
`metalroute bench-corpus` routes the boards vendored under
[`benchmarks/corpus/`](benchmarks/corpus/MANIFEST.md) — **112 real circuit-derived
problems** from [tscircuit/tscircuit-autorouter](https://github.com/tscircuit/tscircuit-autorouter)
(MIT): the `srj15` multi-net region-reroute set plus 57 bug-report boards (real
designs like arduino-uno, esp32-breakout, LGA15x4). Each file is a pure
SimpleRouteJson, so there's no conversion step.

```bash
scripts/bench-corpus.sh            # build + route all + SVG gallery + report.json
scripts/bench-corpus.sh srj15      # just one sub-corpus
scripts/vendor-corpus.sh           # refresh the fixtures from upstream
```

The current `c32e582` release uses the `negotiated` router on all 112 boards and
enforces every active `outline`/`minBoardEdgeClearance` contract. Its two
independent full-corpus runs have identical timing-stripped semantics:

| corpus | routed nets | fully routed boards |
|--------|------------:|--------------------:|
| `bug-reports` | **2267/2447 (92.6%)** | **36/57** |
| `srj15` | **719/720 (99.9%)** | **54/55** |
| **total** | **2986/3167 (94.3%)** | **90/112** |

The raw release report records total route cost **375,563**, **427** native DRC
findings, **77** clean boards, **71** fully-routed-clean boards, and zero errors.
The preceding 3,000-route portfolio did not enforce board outlines: an exact
identity audit found 37 of those routes unsafe across nine boards, making its
physically valid baseline 2,963 rather than 3,000. The feature-aware comparison is:

| physical metric | pre-outline portfolio | current release | change |
|-----------------|----------------------:|----------------:|-------:|
| Safe routed nets | 2963/3167 | **2986/3167** | **+23** |
| Physically fully routed boards | 89/112 | **90/112** | **+1** |
| Comparable ordinary DRC findings | 451 | **443** | **-8** |
| Clean boards | 74 | **77** | **+3** |
| Fully routed + clean | 70 | **71** | **+1** |

The native-report DRC count and the independent comparable-ordinary count are
different checker views and are intentionally not conflated. The exact audit
found zero remaining outline failures and zero board-edge findings in the current
route identities.

Timing is observational rather than an acceptance gate. The first and repeat
certified runs took 324.27 s and 332.50 s externally, respectively. The repeat
used 2,593.23 s user CPU,
17.52 s system CPU, and 1,385,381,888 bytes maximum RSS. A matched run of the
edge-invalid predecessor took 180.43 s on the same machine, so this release does
not claim a speedup over invalid routing: exact safety and constrained reroutes
carry a measured correctness cost.

The accepted pre-feature baseline at `21a0570` isolates the dense via-spacing
change at `02a7f95` from the earlier improvements. Against that baseline, routed
nets improve
**2965 → 2989**, full boards **91 → 92**, DRC **483 → 443**, clean boards
**68 → 72**, and fully-routed-clean boards **65 → 66**. Route cost is
effectively flat at 377,159 → 377,164 while 24 more nets are retained. Per-group
completion is `bug-reports` 2247/2447 → 2270/2447 and `srj15` 718/720 →
719/720. The per-group DRC result is deliberately stated: `bug-reports` improves
345 → 295 (-50), while `srj15` regresses 138 → 148 (+10); the aggregate gate
still improves by 40 findings.

For that isolated A/B, median and nearest-rank p95 move slightly the other way
(0.081 s → 0.089 s and 61.623 s → 64.224 s), while maximum time improves
100.632 s → 96.231 s, the sum of overlapping per-board timers falls **732.030 s
→ 702.733 s**, and end-to-end external elapsed falls **106.53 s → 99.52 s**.
Ten boards gain 26 routed nets; `bugreport63` loses two while its DRC count improves
2 → 0, for the net +24. DRC counts improve on 12 boards and worsen on six; the
largest increases are `sample12` +20 and `sample16` +3, while `bugreport50` -18
and `bugreport11` -16 are the largest reductions. The isolated baseline and
dense-via report SHA-256 values are respectively
`8e200e19ecaf0867ef212e739c3a37c7a8ec645b543ffde86f2ca79276985903`
and `defe7106f2562e4d89685f7c9f2aab4c6d7ddc43db399ca001085d13a6c22c51`.

The subsequent bounded terminal-via repair keeps the same completion and route
cost while reducing DRC **443 → 434**, raising clean boards **72 → 76**, and
raising fully-routed-clean boards **66 → 70**. It cleans `sample13`, `sample18`,
`sample19`, and `sample23` (two findings each) and removes one of five findings
from `sample44`; no board worsens. The canonical terminal-baseline report SHA-256 is
`8e52c3ab9a17188e28ab73d4d4314774261ea9dbdca3ba0367a4f583ccabf35f`.
Deleting report-level `total_wall_ms`/`nets_per_sec`, every group's
`total_wall_ms`, and every board's `wall_ms`, then serializing compact JSON with
sorted keys (including the trailing newline) gives the same independent-repeat SHA-256,
`c8e11c7d5fef04dbbd3d7659dd1a5d66d80887be51ee916a7f11881dd3ffc1d1`.
That stage's group physical totals are 295 DRC findings/35 clean/29 fully-routed-clean
for `bug-reports` and 139/41/41 for `srj15`.

The pre-outline bounded legalization portfolio then moved **2989 → 3000** routed
nets and **92 → 94** fully routed boards, while DRC improves **434 → 429**,
clean boards hold at 76, and fully-routed-clean boards rise **70 → 71**. Route
cost moves 377,164 → 379,530 with 11 additional routes. Only seven boards
change: `bugreport27`, `28`, `29`, `30`, `36-d4c6c2`, `50`, and `63`; all gain
completion, `bugreport50` also drops five DRC findings, and no board loses a
route or gains a finding.

The blocker-informed candidate records failed-group→blocking-owner edges during
the existing bounded rip-up pass, keeps the graph acyclic, and evaluates at most
one stable topological restart. It is disabled above 192 groups, 384 nets, or
1.5M cells. A separate incomplete-order fallback is limited to 6–16 groups, at
most 32 nets and 250k cells, and samples at most four unique cyclic/dihedral
orders. Both fallbacks may replace the established result only for a strict
route-count gain; equal-completion alternatives leave the original bytes intact.
That pre-outline report SHA-256 is
`70f9c026f7e789cc1f0c6153ae9631b2fd22a64e4221c66c746aa38c199b7cc0`;
its timing-stripped semantic SHA-256 is
`1d507b5e091a3ca6b3c299f206c6eef5608c9112dbca16e7e1faf3e7dda0c6ab`.
Its group physical totals are 290 DRC findings/35 clean/30
fully-routed-clean for `bug-reports` and 139/41/41 for `srj15`.

### Exact board-outline and edge-clearance release

An `outline` now activates exact polygon enforcement; if it omits
`minBoardEdgeClearance`, the producer default is 0.2 mm. An explicit edge
clearance without an outline constrains the declared rectangular bounds. Active
contracts validate as finite simple polygons and fail closed. The one deliberately
narrow normalization removes only zero-area collinear backtracking hairpins (and
the duplicate vertex they expose), allowing the malformed `bugreport55` outline
without silently repairing general self-intersections.

The SRJ layer rasterizes one layer-invariant mask for trace-centre nodes, via
centres, and both directions of every planar edge. Its row/column edge index keeps
the exact continuous predicates while measuring 25–317× faster than the naive
all-edges scan on the committed detailed-outline probes; randomized parity tests
are exact. All CPU routers enforce the dependency-inverted mask. Metal kernels do
not yet carry its node/directed-edge representation, so every public Metal field,
router, and isolated-batch entry point rejects a nonempty board mask; provider
callers rerun the whole batch on CPU instead of accepting an unsafe GPU path.

The product selector is legacy-first. It recreates the exact pre-outline route,
including beautification/legalization/via repair, then checks the final emitted
soup against the active polygon. Edge-clean legacy bytes return unchanged; only
an unsafe result pays for a constrained rerun. The constrained soup must be
edge-safe and non-worse under the complete authoritative DRC profile. CLI
`route`, server `/solve`, and `/api/trace` share that policy. Mask construction
uses the width actually emitted by partial/legacy inputs, and the continuous
checker uses each emitted segment width and via-pad diameter. Zero-length and
singleton wire geometry is treated as copper disks, and zero/sub-epsilon polygon
crossings use the same tolerance in routing and DRC. Board-edge findings use the
reserved `__board_edge__` identity.

The only changed cohort is the nine active-outline boards below. “Invalid” means
a pre-outline routed identity that the independent continuous audit rejects;
safe delta compares the current release with the predecessor after removing those
identities.

| Board | Pre-outline raw | Invalid | Pre-outline safe | Current safe | Safe delta |
|-------|----------------:|--------:|-----------------:|-------------:|-----------:|
| `bugreport09-618e09` | 27 | 3 | 24 | 26 | **+2** |
| `bugreport21-board-outline` | 1 | 1 | 0 | 1 | **+1** |
| `bugreport43-e0f33a` | 16 | 1 | 15 | 17 | **+2** |
| `bugreport46-ac4337` | 91 | 7 | 84 | 89 | **+5** |
| `bugreport47-8ee80e-esp32-breakout` | 29 | 7 | 22 | 27 | **+5** |
| `bugreport48-569cfe` | 16 | 3 | 13 | 15 | **+2** |
| `bugreport49-8536f4` | 146 | 3 | 143 | 147 | **+4** |
| `bugreport50-e1c376` | 303 | 8 | 295 | 295 | 0 |
| `bugreport55-b7c349` | 10 | 4 | 6 | 8 | **+2** |
| **Affected total** | **639** | **37** | **602** | **625** | **+23** |

Active masks also admit one tightly bounded completion-only extension for
17–18 connection groups: Adaptive may evaluate at most four additional
cyclic/dihedral legalizations and one rip-up on boards with at most 32 nets and
250k cells, and only when an individually routable net remains incomplete. The
established Adaptive/ForceSerial public winner is selected first; the extension
can replace it only for a strict final routed-count gain, with board route,
isolation diagnosis, and trace moving atomically. This recovers
`bugreport09` from 25/27 to 26/27 in the constrained path while reducing its
native DRC findings from six to one; an equal-completion `bugreport43` candidate
leaves the established soup unchanged.

The first and repeat raw report SHA-256 values are respectively
`17cb8fcd347f14ed55d5cd037cf484e853cb63ee5e49d9132c339b4e1c7f68f7`
and `6950023a20b0674f6284cae1b894cc17d1c0d65cd380f69b2bb3bc54803c24d7`.
After deleting top-level `total_wall_ms`/`nets_per_sec`, every group
`total_wall_ms`, and every board `wall_ms`, then sorting and compacting JSON with
a trailing newline, both produce semantic SHA-256
`d7f9a956f29211a45d1e4f18a1472886b6b3f9da8108136e566ee6e971843441`.

Two runtime hardenings remain research-only. Speculatively running legacy and
constrained candidates in parallel changed a focused `bugreport46` wall time
14.90 → 13.09 s but consumed 10% more CPU and 6% more peak RSS, so it was a
NO-GO. Dynamic phase-B preferred-path revalidation passed the exact semantic
audit and focused tests, but triggered excessive full-board fallbacks and did not
finish the corpus within 300 s; it is not integrated because its runtime bound is
not acceptable.

The physical rule is feature-aware: via-to-trace centre spacing is via radius +
edge clearance + trace radius, while via-to-via centre spacing is both via radii
plus edge clearance. Committed features stamp exact Euclidean exclusion disks into a
dense group-aware landing guard (`free`, one owning group, or mixed owners), so a
candidate via checks two cells in O(1); same-group copper remains exempt and planar
moves continue to use their ordinary halo. The guard is rebuilt after rip-up and is
not allocated for legacy callers that omit physical via spacing. Against the
equivalent scan implementation on `bugreport50`, route time falls **117.657 s →
44.390 s** with identical timing-stripped semantics (SHA-256
`8803762864a31e1e376eedad3ae409283136195711b5219133fd630b75716b30`).

The principal earlier speed wins remain unique-cell SRJ pad-halo filtering, O(1)
exact Hanan-distance heuristics, and fused Jacobi pricing with planar/via neighbor
enumeration that avoids per-expansion allocation. Pad-ownership-aware smoothing
and one bounded, board-wide DRC-scored via move contribute to the separate
physical-quality work. Interior vias retain the established one-clearance rigid
moves; endpoint-adjacent vias can instead grow a short stationary-terminal
dogleg, quantized from the exact generic-clearance deficit in quarter-clearance
steps. Physical endpoints, trace order, reconstructed net labels, and ordered via
spans are invariant, and only a full-board DRC reduction is accepted.
Legacy unlabeled-pad ownership is inferred only from immutable trace endpoints,
preventing a moved interior via from claiming a foreign pad during repair scoring.
The DRC bridge now preserves the standard physical
layer stack, uses endpoint-side ownership for terminal vias, propagates declared
connectivity across each routed group, and resolves only unambiguous fixed-pad
aliases. The DRC exact-gap fast reject adds a smaller measured 10.8%
microbenchmark improvement.

### Coherent typed SimpleRouteJson subset

The CLI route path and an unoverridden server `/solve` now activate modern
physical fields only when they form one coherent, uniformly enforceable profile.
The supported subset resolves board/per-connection trace width, generic
`minClearance`/`defaultObstacleMargin`, trace↔pad (also used for via↔trace),
via↔pad, optional pad↔pad and drill↔drill clearances, and exact declared
routed via pad/hole diameters. Per-connection width and via-diameter aliases must
agree; when both generic fields are present, the established `minClearance`
value takes precedence over `defaultObstacleMargin`. Via geometry must satisfy
the annular minimum; pads must be finite, connected, unrotated rects or
conservative circle bounds; and every terminal must be covered only by pads
that resolve unambiguously to its electrical group. Partial, mixed-width, or
ambiguous inputs fail closed to the established legacy policy.

Board-outline and `minBoardEdgeClearance` enforcement is now a separate active
contract layered over both coherent typed inputs and partial/legacy profiles; it
is therefore no longer part of the unsupported list. `allowViaInPad` still only
parses and round-trips, while bus and differential-pair declarations remain
outside routing semantics. The pre-outline typed projection certified the SRJ29
AM62L/LPDDR4 frontier fixture at 498×321×8: **28/33** connections routed, cost
**4,841**, and zero findings under the supported typed DRC. Its recorded solution
SHA-256 is
`e5ac13e3621c6f81152d51bd4b745de3128901978838308311fbfffcca79c262`.
The current product additionally enforces that fixture's rectangular outline and
edge clearance when it is routed; via-in-pad, bus, and differential-pair behavior
remains outside the asserted result.
None of the locked 112 corpus boards satisfies the coherent typed-profile gate
(0/112); nine independently activate the board-edge contract. The pre-terminal
typed-rule revision preserves the accepted legacy report's
timing-stripped semantics exactly (SHA-256
`6e1722326a644051f19d0b3fc13b8d2fde2b8f0c0383df061122b5980b442819`).

[Upstream PR 2145](https://github.com/tscircuit/tscircuit-autorouter/pull/2145)
exposed the `bugreport21-board-outline` concave-cutout failure that motivated the
now-shipped exact mask/checker stack. The release closes its former gates:
legacy-byte preservation for already-safe soups, complete CPU and fail-closed
Metal coverage, bidirectional edge masks, final-soup validation, singleton/
zero-length geometry, narrow malformed-outline normalization, and indexed mask
rasterization.

Typed acceptance is authoritative, but bounded via-repair discovery still uses
the generic clearance: pair-only or drill-only findings do not independently
trigger discovery, though a generic-triggered move can reduce them incidentally.
Adversarial inputs with unbounded distinct coincident via representations can
also make the typed DRC broad phase quadratic. JSON compatibility remains
backward-compatible; downstream Rust code using public struct literals must add
the new fields (the workspace crates are still version 0.1).

The run writes a self-contained SVG gallery (`benchmarks/runs/<ts>-corpus/index.html`,
gitignored) rendering obstacles, routed traces, and vias per board — failures
sorted first — so regressions are eyeballable. The SVG renderer is dependency-free
Rust (no Node), so it reproduces from a clean checkout.

## Reusable dependencies (consumed via shell-out)

- [`kicad-cruncher`](.) — KiCad parsing. Note: its `pnp` surface gives component
  placements + layers but **no nets / pad geometry / outline**; routing needs the
  netlist surface (Open Question #2, documented in `mr-ingest`).
- [`bed-of-nails`](.) — Specctra DSN export + SES import + Freerouting orchestration
  (the M5 detailed-routing step).

## Status vs. milestones

M0 (sweep de-risk) ✅ · M1 (CPU Lee) ✅ · M2 (live rip-up + negotiated congestion)
✅ · M3 (Metal wavefront) ✅ · M4 (batched Metal prefix-min sweep) ✅ · M5
(Freerouting handoff seam) ✅. Multilayer vias, non-uniform physical costs,
clearance halos, 45° output beautification, exact DRC, continuous board-outline/
edge-clearance enforcement, and deterministic corpus gates are implemented.

The next architectural step is a shared global candidate portfolio: coarse
hypergraph/corridor planning and fanout, GPU-batched detailed candidate scoring,
then exact DRC acceptance/repair. The production negotiated router is still CPU;
Metal currently accelerates independent shortest-path batches rather than the
dynamic congestion loop itself.

## License

MIT OR Apache-2.0.
