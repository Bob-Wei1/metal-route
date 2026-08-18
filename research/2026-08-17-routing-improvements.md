# Autorouter improvement report — 2026-08-17 (finalized 2026-08-18)

This report records the test expansion, exact before/after measurements, and the
next architecture suggested by current tscircuit work. It deliberately separates
aggregate gains from remaining regression gates.

## Reproducible baseline

Baseline revision: `325d44d`. The final measured candidate is `7aee642`. Both
corpus reports contain the same 112 board IDs and the same per-board net totals.
The archived baseline used the former first-seen layer checker, so its exact route
traces (including all 2,701 retained net tags) were replayed through the corrected
checker at `c6e08b1`. All non-DRC report fields are byte-for-byte identical. Thus
routing and timing remain the original baseline measurement, while the DRC rows
below compare the same checker semantics. The corrected baseline report SHA-256 is
`880d54d4fa34ba26fa70fdb1c9a3172962e87104a4876726a298362fbbfb5e35`;
the final report SHA-256 is
`6543f29685c9cf4b97528b2156473da98ffbb07f7fc246da32f81cdbd01b662d`.
The final synthetic and six-board preflight report hashes are respectively
`4ca110b2b5f8fa14ba274e4c3b8cf0461a46dad4e9a4cd4b2bff34ec17920d65`
and `5e08056ee706c0161db75eba6e6ea28c3d653d1d77a2cd0013e0b5119171fd03`.

```sh
cargo build --locked -p mr-cli --release
time target/release/metalroute bench \
  --out /tmp/metalroute-before-synth.json
# Repeat at the candidate revision with:
time target/release/metalroute bench \
  --out /tmp/metalroute-after-synth.json

time target/release/metalroute bench-corpus \
  --dir benchmarks/corpus \
  --out /tmp/metalroute-before-corpus.json
# Repeat at the candidate revision with:
time target/release/metalroute bench-corpus \
  --dir benchmarks/corpus \
  --out /tmp/metalroute-after-corpus.json
```

Synthetic defaults are ten boards, 30 nets, size 50, eight obstacles, and seed 1.
Corpus defaults include `--router negotiated --max-cells 12000000`.

Corpus `total_wall_ms` is the sum of overlapping per-board route timers. External
`time` is the end-to-end wall clock and is governed by the slowest parallel board.
The commands reproduce the workload and deterministic routing outputs, and
remeasure timing. A checker-comparable baseline DRC requires replaying the archived
baseline traces through the corrected checker as described above; directly
rerunning the baseline revision uses its legacy checker.

## Test expansion

Rust test attributes increased from 243 to 385. The current workspace executes
384 conventional tests, has one explicitly ignored live-tool test, and passes two
doctests. Thirteen Python tests cover the benchmark scorer. Important new families:

- exact shared contracts across Lee, A*, RipUp, and Negotiated routers;
- exhaustive 3×3 A*/Lee equality, zero-cost cycle prevention, and weighted ties;
- multilayer sweep fields, passable pads, malformed/OOB inputs, and cost accounting;
- real rip-up displacement, non-uniform physical clearance, overlapping halos,
  and deterministic 1/2/4-thread negotiation;
- SRJ layer ownership, unknown-layer obstacles, width preservation, and DSN
  arbitrary-angle/multi-shape pad geometry;
- DRC spatial-index equivalence and total deterministic ordering;
- fixed-fixture and deterministic fixed-seed Metal checks for weighted/zero/unit
  dispatch, chunk boundaries, memory limits, pad overrides, multilayer batches,
  and concurrent calls;
- exact benchmark workload identity, DRC, clean-board, error, group, and aggregate
  consistency gates;
- cached isolation diagnoses and bounded scratch reuse under nested concurrent
  routes;
- exact DRC acceptance, compressed via-leg geometry, pad ownership, and bounded
  topology-preserving via repair.

## Implemented changes

### CPU routing

- Targeted A* stops after the canonical equal-cost frontier instead of exhausting
  the reachable board. Board-wide minimum step cost is computed once.
- Search labels are `(cost, hop_count, lower_predecessor)`. Hop count strictly
  decreases during reconstruction, eliminating zero-cost predecessor cycles.
- Sweep reconstruction uses the same minimum-hop shortest-edge graph contract.
- RipUp can now perform a real priority displacement; its previous fresh-route
  control flow made the rip branch unreachable.
- Negotiated routing honors grid weights, real Hanan spacing, short-coordinate
  fallback, own pads, via costs, and physical clearance during both negotiation
  and legalization.
- Clearance-active parallel negotiation subtracts each net's exact counted
  self-halo from the Jacobi occupancy snapshot. It is byte-identical across Rayon
  thread counts. A bounded serial polish is retained only for single-layer boards;
  on multilayer boards it moved otherwise good paths between layers and increased
  downstream DRC.
- Explicit multi-segment connections on 17–179-net, at-most-250k-cell boards get
  one deterministic all-serial candidate. The portfolio keeps more routed nets,
  then lower `u64` grid cost, retaining the adaptive primary on exact ties. The
  unconditional cell bound prevents incomplete routes from creating unbounded
  fallback latency; route and traced-route select the same candidate.
- `route_with_outcome` exposes the isolation-routability bits legalization already
  computes, so the CLI no longer reruns the full router for every failed net.
- Parallel Jacobi search scratch is lazily thread-local: it is reused across
  iterations and nested routes while remaining bounded by executing threads,
  rather than concurrent boards multiplied by inner workers.
- Exact per-gap Hanan costs are prefix-summed once per board, making the admissible
  physical-distance heuristic O(1) per heap operation. A matched `bugreport11` A/B
  was byte-identical and reduced route time by 47.3%.
- The Jacobi hot path folds the immutable congestion snapshot once when at least
  32 dirty nets amortize the board-wide pass, and flat planar/via neighbor
  enumeration avoids per-expansion allocation. A 512-case randomized
  fused/unfused oracle covers weights,
  zero-cost cells, obstacles, windows, pads, nonuniform coordinates, restricted
  vias, saturation, and 1–4 layers. Matched normalized outputs were identical;
  median route time fell 9.5% on `bugreport46`, while one tail `bugreport50` pair
  fell 22.5%.

### Metal routing

- Sources are routed in memory-bounded batches with one process-global Metal
  context. Per-line change flags replace one global atomic per relaxed cell.
- On committed fixed fixtures and deterministic fixed-seed stress cases, weighted
  and zero-cost fields carry minimum-hop labels and match the CPU canonical path.
  Unit/obstacle grids use a distance-only fast path. This is fixed-seed regression
  coverage rather than exhaustive enumeration of cost grids.
- Shared and packed cost planes preserve per-net passable-pad overrides without
  giving up batching.
- Public field results are chunked and capped before host/GPU allocation;
  multiplication overflow, empty-grid source explosions, device limits, and
  32-bit kernel indexing are rejected deterministically.
- An exact edge-aware isolated-route batch supports non-uniform Hanan gaps,
  per-net windows and pads, restricted/zero-cost vias, weighted saturation, and
  canonical minimum-hop ties. A dependency-inverted CPU seam validates results
  and falls back atomically on malformed output or any Metal command failure.
- Real-board crossover testing found no reliable automatic win (five of eight
  representatives regressed), so targeted CPU A* remains the default. macOS users
  can explicitly test the provider with
  `METALROUTE_EXPERIMENTAL_METAL_ISOLATED=1`; a busy GPU lane falls back immediately
  rather than serializing parallel corpus jobs.

### Geometry and measurement

- Grid coordinate fallback, layer-aware pad passability, conservative unknown
  layers, per-vertex widths, strict DSN parsing/rules, and arbitrary rotated pad
  AABBs are now covered and corrected.
- SRJ pad-clearance rasterization de-duplicates overlapping halo cells before its
  expensive foreign-obstacle test. The former `bugreport05` raster/route tail fell
  from 564.226 s to 35.411 s end-to-end with identical normalized output.
- DRC ordering is total and input-order independent; oracle comparison preserves
  duplicate net multiplicity and exact congestion-vector length.
- The solution-to-DRC bridge models compressed via landing legs and seeds the
  canonical physical stack before ingesting geometry. Unknown routed layers map
  to top; empty/all-unknown obstacle layer lists expand to every effective layer;
  mixed lists retain their known, deduplicated layers. Terminal vias own only the
  source/destination endpoint side, while their barrels remain copper on the full
  physical span.
- The exact legalizer's pad features carry connectivity ownership, so smoothing
  can move copper into its own pad while unknown pads remain foreign. Legacy
  unlabeled-pad ownership is inferred only from immutable, layer-matched trace
  endpoints, so an interior via cannot move into a foreign pad and then relabel it
  as its own during scoring. Declared connectivity propagates across the entire
  initial routed identity; fixed-pad aliases gain ownership only when they resolve
  uniquely, and ambiguous aliases stay foreign.
- DRC via traversal normalizes direction and clamps spans before constructing an
  inclusive range. DSN conversion rejects missing or duplicate layer identities
  instead of silently collapsing a via or aliasing it onto another plane.
- Geometry candidates are accepted against every authoritative DRC finding. Fewer
  findings is primary; equal-count candidates must preserve stable finding
  multiplicity and strictly improve 1 nm-quantized severity without worsening a
  rank.
- A final bounded repair considers at most eight implicated, nonterminal, unshared
  vias and eight one-clearance compass moves. It rigidly preserves anchors,
  endpoints, trace order, and via spans, then retains at most one candidate and
  only when the full-board DRC count strictly falls. Under the corrected checker,
  checked-in regressions improve sample11 from 20 to 18 findings and sample25 from
  3 to 0.
- DRC rejects pairs whose copper AABBs already prove enough separation before the
  exact geometry gap calculation; randomized indexed-vs-naive tests preserve exact
  results and the focused microbenchmark improved 10.8%.
- Benchmark reports now label the production router `negotiated` (the baseline's
  `ripup-cpu` label was wrong; it already invoked NegotiatedRouter).

## Results

### Metal microbenchmark

M4, release, 128×128, 64 independent nets, three isolated processes. This is not
a like-for-like algorithm benchmark: targeted Lee stops after destinations settle;
the CPU full-field timing constructs source-distance fields without reconstructing
paths; Metal computes full fields and reconstructs paths. CPU rows are one timed
observation per process. Metal warm p50/7 is the ordinary median—the fourth sorted
observation—of seven post-setup batches in each process.

| Measurement | Result |
|-------------|-------:|
| Metal cold setup | 22.525–26.518 ms |
| Metal warm p50/7 | **3.419–3.746 ms** |
| CPU targeted Lee paths | 14.946–18.217 ms |
| CPU full fields (no path reconstruction) | 26.280–27.666 ms |

The observed Lee-to-Metal elapsed-time ratio is 4.01–5.33× and the
CPU-full-field-to-Metal ratio is 7.06–8.09× after setup is amortized. These ratios
describe the measured operations above; they are not like-for-like algorithmic
speedups.

### Synthetic negotiated benchmark

| Metric | Before | After |
|--------|-------:|------:|
| Routed | 206/300 | **216/300** |
| Completion | 68.67% | **72.00%** |
| Observed report time | 1.680 s | **0.653 s** |
| Mean routed cost | 74.136 | 74.843 |

The completion result and mean cost are deterministic. The final release sample
took 0.97 s externally; report time excludes process startup.

### Exact 112-board corpus

| Metric | Before | After | Change |
|--------|-------:|------:|-------:|
| Routed nets | 2701/3167 | **2729/3167** | +28 |
| Fully routed boards | 78/112 | **91/112** | +13 |
| DRC findings | 1227 | **714** | -513 |
| Clean boards | 49 | **66** | +17 |
| Fully routed + clean | 46 | **63** | +17 |
| Total route cost | 340,055 | **332,354** | -2.26% |
| Median board time | 1.392 s | **0.080 s** | 17.35× faster |
| Nearest-rank P95 | 339.922 s | **41.733 s** | 8.15× faster |
| Maximum board time | 679.474 s | **82.961 s** | 8.19× faster |
| Sum of board timers | 4715.180 s | **546.430 s** | 8.63× faster |
| External elapsed | 715.55 s | **83.04 s** | 8.62× faster |

For the even 112-board sample, median is the arithmetic mean of sorted observations
56 and 57 (one-indexed), not either middle observation. Nearest-rank p95 is sorted
observation 107.

Per group, `srj15` improves 705/720 → 718/720 and 46 → 53 full boards.
`bug-reports` improves 1996/2447 → 2011/2447 and 32 → 38 full boards.
The hardened scorer returns `KEEP`: exact workload identity is unchanged, both
groups improve, errors remain zero, and aggregate DRC, clean-board, and
fully-routed-clean gates all improve. This is not a claim of per-board monotonicity:
among the 47 boards whose finding count changes, 43 improve and four worsen. There
are 18 clean-board gains and one loss, plus 19 fully-routed-clean gains and two
losses.

The legacy baseline report contained 1,493 findings under the old first-seen layer
checker; that number is intentionally not used in the comparison table. Rechecking
the byte-identical baseline routes with the corrected physical-layer/ownership
semantics yields the checker-comparable 1,227 above. On `bugreport05`, routed nets
move 86/228 → 80/228, grid cost 37,109 → 39,288, and findings 15 → 12. The
six-board preflight is 70/113 with 31 findings: bugreport01 and bugreport23 are
clean, bugreport30 has 18, bugreport45 is clean, sample21 has six, and sample22
has seven. Corpus
`total_wall_ms` is a sum of overlapping board timers; external elapsed is the real
end-to-end measure.

## Transferable upstream work

Current upstream reviewed: `tscircuit-autorouter` v0.0.817 (`aee844f`,
2026-08-17).

- [Pipeline7 exact-DRC fast probe and shared spatial state](https://github.com/tscircuit/tscircuit-autorouter/commit/1f6e42b77c85911163051e38420dccb36b06affb)
- [Compact A*, ancestry caching, shared layer queries, dynamic stitch validation, and candidate portfolios](https://github.com/tscircuit/tscircuit-autorouter/commit/c2d9d13df1e8af22626e987082a2bb877682da0f)
- [Balanced tiny-hypergraph routing experiment](https://github.com/tscircuit/tscircuit-autorouter/blob/main/experiments/tiny-hypergraph-balanced-routing.md)
- [Outside-in partial rip](https://github.com/tscircuit/tiny-hypergraph/blob/main/experiments/outside-in-partial-rip.md)
- [Compact port×region candidate state](https://github.com/tscircuit/tiny-hypergraph/blob/main/experiments/compact-candidate-hop-state.md)
- [Exact DRC repair fixtures](https://github.com/tscircuit/high-density-repair03)
- [BGA fanout solver](https://github.com/tscircuit/fanout-solver)
- [Upstream regression thresholds](https://github.com/tscircuit/tscircuit-autorouter/blob/main/scripts/benchmark/detect-benchmark-regressions.ts)

The most credible next architecture is CPU coarse corridor/hypergraph planning,
GPU-batched detailed candidate search/scoring, and exact CPU DRC acceptance with
bounded local repair/partial rip. The production NegotiatedRouter does not yet
offload its dynamic congestion iterations to Metal.

Licensing note: the reviewed core repositories and several fixture sets are MIT.
`srj24` includes Apache-2.0 KiCad-derived inputs and needs retained provenance.
Public repositories without a license (including several SRJ repro repositories)
must not be vendored or redistributed without permission.
