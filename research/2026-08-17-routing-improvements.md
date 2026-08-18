# Autorouter improvement report — 2026-08-17 (finalized 2026-08-18)

This report records the test expansion, exact before/after measurements, and the
next architecture suggested by current tscircuit work. It deliberately separates
aggregate gains from remaining regression gates.

## Reproducible baseline

Historical baseline revision: `325d44d`. The previous measured candidate was
`7aee642`, the isolated dense-via candidate is `02a7f95`, the coherent typed-SRJ
projection is `f6c7fb0`, the accepted terminal-via baseline is `9e5b40a`, and the
accepted pre-outline dependency/dihedral portfolio is `ddafd9a`. The current
measured board-outline release is `c32e582`. Every
corpus report in the comparisons below contains the same 112 board IDs and
per-board net totals.
The archived historical baseline used the former first-seen layer checker, so its
exact route traces (including all 2,701 retained net tags) were replayed through
the corrected checker at `c6e08b1`. All non-DRC report fields remained
byte-for-byte identical. Its checker-comparable SHA-256 is
`880d54d4fa34ba26fa70fdb1c9a3172962e87104a4876726a298362fbbfb5e35`.

The isolated via-spacing A/B uses accepted pre-feature revision `21a0570` as its
baseline and `02a7f95` as its candidate. The corresponding report SHA-256 values
are `8e200e19ecaf0867ef212e739c3a37c7a8ec645b543ffde86f2ca79276985903`
and `defe7106f2562e4d89685f7c9f2aab4c6d7ddc43db399ca001085d13a6c22c51`.
The canonical terminal-via baseline report SHA-256 is
`8e52c3ab9a17188e28ab73d4d4314774261ea9dbdca3ba0367a4f583ccabf35f`;
an independent repeat is
`4749ed5f97fdd1dca69840bccc58f3c9befc02823be5a79ae6f02ef944ade345`.
After deleting `.total_wall_ms`, `.nets_per_sec`, each group `total_wall_ms`, and
each board `wall_ms`, then serializing compact JSON with sorted keys, both
terminal-via reports have SHA-256
`c8e11c7d5fef04dbbd3d7659dd1a5d66d80887be51ee916a7f11881dd3ffc1d1`.
The accepted dependency/dihedral portfolio report has raw SHA-256
`70f9c026f7e789cc1f0c6153ae9631b2fd22a64e4221c66c746aa38c199b7cc0`.
Applying the same timing-field deletion and hashing the sorted compact JSON
including its trailing newline gives semantic SHA-256
`1d507b5e091a3ca6b3c299f206c6eef5608c9112dbca16e7e1faf3e7dda0c6ab`.

The current exact-outline release has independent raw corpus SHA-256 values
`17cb8fcd347f14ed55d5cd037cf484e853cb63ee5e49d9132c339b4e1c7f68f7`
and `6950023a20b0674f6284cae1b894cc17d1c0d65cd380f69b2bb3bc54803c24d7`.
After deleting `.total_wall_ms`, `.nets_per_sec`, each group `total_wall_ms`, and
each board `wall_ms`, then serializing compact sorted JSON with its trailing
newline, both normalize to SHA-256
`d7f9a956f29211a45d1e4f18a1472886b6b3f9da8108136e566ee6e971843441`.

At `f6c7fb0`, none of the locked corpus boards activated the fail-closed typed
profile (0/112). Its raw corpus report SHA-256 is
`e5c7ded949efd41499e9de69c748a1062ca0d823a97aad6d489194b1a2ac786f`,
and its timing-stripped semantics are byte-identical to the accepted legacy run
at SHA-256
`6e1722326a644051f19d0b3fc13b8d2fde2b8f0c0383df061122b5980b442819`.
The earlier corpus, synthetic, and six-board preflight hashes belong to `7aee642`
and are retained only as intermediate provenance:
`6543f29685c9cf4b97528b2156473da98ffbb07f7fc246da32f81cdbd01b662d`,
`4ca110b2b5f8fa14ba274e4c3b8cf0461a46dad4e9a4cd4b2bff34ec17920d65`
and `5e08056ee706c0161db75eba6e6ea28c3d653d1d77a2cd0013e0b5119171fd03`.

The matched `bugreport50` scan and dense-guard runs retain the same normalized
routing semantics. Their shared semantic SHA-256 is
`8803762864a31e1e376eedad3ae409283136195711b5219133fd630b75716b30`
for both: 300/322 nets, cost 46,280, and 15 DRC findings. Route time falls from
117.657 s for per-candidate scans to 44.390 s for the dense guard.

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

Rust test attributes increased from 243 to 543. The current workspace executes
533 tests, has ten explicitly ignored manual frontier/performance/live/real-board
gates, and passes two doctests. Six Criterion route-benchmark smoke cases also pass
but are not counted as tests. Thirteen Python tests cover the benchmark scorer.
Important new families:

- exact shared contracts across Lee, A*, RipUp, and Negotiated routers;
- exhaustive 3×3 A*/Lee equality, zero-cost cycle prevention, and weighted ties;
- multilayer sweep fields, passable pads, malformed/OOB inputs, and cost accounting;
- real rip-up displacement, non-uniform physical clearance, overlapping halos,
  and deterministic 1/2/4-thread negotiation;
- exact Euclidean via-to-trace/via-to-via spacing, group-aware dense landing-guard
  equivalence, strict-boundary behavior, rip rebuilds, and normalized exemptions;
- SRJ layer ownership, unknown-layer obstacles, width preservation, and DSN
  arbitrary-angle/multi-shape pad geometry;
- fail-closed coherent typed-SRJ width, alias, terminal-pad, pair-clearance, via
  geometry, drill-spacing, CLI, and server product-path contracts;
- DRC spatial-index equivalence and total deterministic ordering;
- fixed-fixture and deterministic fixed-seed Metal checks for weighted/zero/unit
  dispatch, chunk boundaries, memory limits, pad overrides, multilayer batches,
  ragged windows, aggregate allocation caps, and concurrent calls;
- exact benchmark workload identity, DRC, clean-board, error, group, and aggregate
  consistency gates;
- cached isolation diagnoses and bounded scratch reuse under nested concurrent
  routes;
- exact DRC acceptance, compressed via-leg geometry, pad ownership, and bounded
  topology-preserving interior/stationary-terminal via repair;
- acyclic blocker dependencies, cap-boundary no-ops, strict-gain selection,
  unique dihedral-order sampling, and real-board completion/DRC portfolio gates;
- exact concave/rectangular board-outline predicates, indexed-vs-naive raster
  parity, bidirectional planar masks, trace/via radii, zero/sub-epsilon crossings,
  singleton wire disks, malformed-contract failure, and narrow collinear-spur
  normalization;
- legacy-first final-soup selection, constrained DRC non-regression, emitted-width
  parity, server `/solve`/`/api/trace` parity, Metal fail-closed fallback, and the
  bounded 17–18-group strict-completion extension.

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
- The SRJ/DSN adapters now pass feature-aware physical centre spacing: via-to-trace
  is via radius + copper edge clearance + trace radius, while via-to-via is both
  via radii + edge clearance. Exact Euclidean checks make a candidate exactly at
  the required distance legal; zero edge clearance still prevents copper overlap.
- Modern SRJ fields project only when one coherent board-wide profile is
  enforceable. Supported inputs resolve uniform board/per-connection trace width,
  generic `minClearance`/`defaultObstacleMargin`, trace↔pad (also governing
  via↔trace), via↔pad, optional pad↔pad and drill↔drill clearances, and exact
  declared routed via pad/hole diameters. Per-connection width and via-diameter
  aliases must agree. When both generic fields are present, the established
  `minClearance` value takes precedence over `defaultObstacleMargin`; via geometry
  must satisfy the annular minimum, and pair-specific obstacle rules must
  conservatively dominate the resolved generic rule.
- The typed gate fails closed unless pads are finite, connected, unrotated rects
  or conservative circle bounds and every routed terminal is covered only by pads
  that resolve unambiguously to its electrical group. Partial rules,
  mixed connection widths, unsupported geometry, bare endpoints, or ambiguous
  aliases retain the byte-stable legacy path.
- A nonempty `outline` or explicit `minBoardEdgeClearance` independently activates
  exact board-edge routing for coherent typed and partial/legacy inputs. An outline
  without a clearance uses the producer's 0.2 mm default; an explicit clearance
  without an outline constrains the declared bounds. Malformed active contracts
  fail closed. Only exact zero-area collinear backtracking hairpins and their
  resulting duplicate vertex are normalized; arbitrary self-intersections remain
  errors.
- The SRJ rasterizer projects exact continuous trace-centre, via-centre, and
  bidirectional planar-edge predicates into a dependency-inverted core mask.
  Trace keepout is edge clearance plus emitted radius; via keepout is edge
  clearance plus routed via-pad radius. Partial profiles build the mask at the
  width they actually emit rather than an unsupported declared width.
- The product selector first runs the exact historical route with only the
  board-edge contract removed, including every postprocessor, then validates the
  final soup. Edge-clean bytes return unchanged. An unsafe route is ineligible and
  triggers one constrained rerun, which must be edge-safe and non-worse under the
  complete authoritative DRC profile. CLI `route`, server `/solve`, and
  `/api/trace` use the same selection contract.
- Legalization pre-stamps committed trace cells and via landings into a dense,
  group-aware via-landing guard. Each cell is free, owned by one connection group,
  or mixed, so a candidate via needs two O(1) tag reads while same-group sharing
  remains legal. The guard is rebuilt after rip-up. Legacy callers with no physical
  via-to-via spacing allocate no guard and keep their previous behavior.
- The dense guard is covered against the former exact per-candidate scan over
  non-uniform/truncated coordinates, multiple layers and groups, mixed overlap,
  strict boundaries, rip rebuilds, and parallel thresholds. On matched
  `bugreport50`, it preserves the scan's semantic digest while cutting route time
  117.657 s → 44.390 s.
- Via-exemption lists are normalized once: already sorted/deduplicated inputs stay
  borrowed, while other inputs are cloned, sorted, and deduplicated. Hot searches
  use binary lookup rather than repeatedly scanning the list.
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
- When the original legalization portfolio leaves an individually routable net
  congested, boards with 6–16 connection groups, at most 32 nets, and at most
  250k cells sample up to four additional unique cyclic/dihedral group orders:
  one-step left/right rotations, reversal, and the opposite reversed rotation.
  These alternatives are completion-only: they can replace the established
  winner only by routing strictly more nets, so equal-completion results preserve
  the original bytes.
- Active board masks extend that completion-only cohort to exactly 17–18 groups,
  still capped at 32 nets and 250k cells. Adaptive alone may evaluate at most four
  cyclic/dihedral legalizations and one rip-up. The established Adaptive/
  ForceSerial winner is chosen first; only a strict final routed-count gain can
  atomically replace its board route, isolation diagnosis, and trace. This avoids
  duplicating the extension in ForceSerial and makes equal-completion cost changes
  inert.
- The bounded rip-up pass records exact failed-group→blocking-owner edges while
  rejecting cycle-closing edges. It can make one stable topological legalization
  restart, again accepted only for a strict completion gain. Dependency collection
  and the restart are disabled above 192 groups, 384 nets, or 1.5M cells; over-cap
  inputs remain inert rather than allocating a partial graph.
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
- Metal kernels do not yet encode exact board trace-node, via-node, and directed
  planar-edge masks. Every public Metal field, router, and isolated-batch entry
  point therefore rejects any nonempty board mask before GPU work; provider
  callers fall back atomically to the complete CPU batch. This is a fail-closed
  boundary, not a claim that the GPU enforces outlines internally.
- An exact edge-aware isolated-route batch supports non-uniform Hanan gaps,
  per-net windows and pads, restricted/zero-cost vias, weighted saturation, and
  canonical minimum-hop ties. A dependency-inverted CPU seam validates results
  and falls back atomically on malformed output or any Metal command failure.
- The isolated adapter packs ragged cropped windows rather than padding every net
  to a batch-wide rectangle, reconstructs canonical paths on the GPU, and reads
  back only compact retained paths. Checked per-chunk aggregate allocation caps
  and call-wide field/path limits reject oversized work before host or Metal
  allocation; exact-boundary and descriptor-heavy regressions pin the limits.
- Real-board crossover testing found no reliable automatic win (five of eight
  representatives regressed), so targeted CPU A* remains the default. macOS users
  can explicitly test the provider with
  `METALROUTE_EXPERIMENTAL_METAL_ISOLATED=1`; a busy GPU lane falls back immediately
  rather than serializing parallel corpus jobs.
- Research commit `adba265` instrumented exact first-no-change passes to estimate
  per-field early-retirement headroom without changing route semantics. On
  bug05-shaped micro workloads the retired/current logical-work ratios were
  1.000000 for open weights and 0.899466 for heterogeneous entry prices; the
  bug50-shaped ratios were 1.000000 and 0.844051. All exceed the predeclared 0.69
  NO-GO threshold. These were synthetic logical-work probes, not real-board
  routing or timings of an active-retirement implementation. The instrumentation
  remains on a separate research branch and is not part of the production router.

### Geometry and measurement

- Board-mask rasterization uses deterministic row/column edge indexes whose
  candidate sets are conservative supersets of every polygon edge that can affect
  an exact predicate. Committed detailed-outline probes measured 25–317× over the
  naive all-edges scan, and randomized indexed-vs-naive masks match exactly.
- Continuous final-soup validation and DRC use actual emitted wire widths and via
  pad diameters. Uncovered singleton wire points and zero-length segments retain
  disk geometry; zero/sub-epsilon cutout crossings follow the same tolerance as
  mask construction. Board-edge violations carry the reserved `__board_edge__`
  pseudo-net so identity/severity comparisons remain deterministic.
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
- A final bounded repair considers at most eight implicated, unshared vias and
  eight compass candidates each. Established interior-via candidates keep their
  exact cap priority and one-clearance rigid move. Endpoint-adjacent vias instead
  grow a stationary-terminal dogleg whose radius rounds the exact generic-rule
  deficit up to a quarter-clearance step. Physical first/last endpoints, trace
  order, reconstructed net labels, and ordered via spans remain invariant; at
  most one candidate survives, only when authoritative full-board DRC strictly
  falls. Pair-specific or drill-only findings may be reduced incidentally by a
  generic-triggered move, but they do not independently trigger discovery.
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

### Synthetic negotiated benchmark (intermediate `7aee642`)

| Metric | Before | After |
|--------|-------:|------:|
| Routed | 206/300 | **216/300** |
| Completion | 68.67% | **72.00%** |
| Observed report time | 1.680 s | **0.653 s** |
| Mean routed cost | 74.136 | 74.843 |

The completion result and mean cost are deterministic. That intermediate release
sample took 0.97 s externally; report time excludes process startup. It was not
remeasured at `02a7f95`, so these numbers are not presented as a current-final A/B.

### Exact 112-board corpus

The current `c32e582` release enforces active outline and board-edge contracts.
Both full runs reproduce this deterministic raw result:

| Metric | Current release |
|--------|----------------:|
| Routed nets | **2986/3167 (94.3%)** |
| Fully routed boards | **90/112** |
| Total route cost | **375,563** |
| Native report DRC findings | **427** |
| Clean boards | **77** |
| Fully routed + clean | **71** |
| Errors | **0** |

Per group, `bug-reports` routes 2267/2447 nets with 36/57 boards full;
`srj15` routes 719/720 with 54/55 boards full. The raw predecessor retained
3,000 nets and 94 boards but did not enforce outlines. An independent continuous
identity audit rejects 37 of those routes across nine boards, so its physically
valid baseline is 2,963 routes and 89 physically full boards. That makes the
feature-aware acceptance table:

| Physical metric | Pre-outline portfolio | `c32e582` | Change |
|-----------------|----------------------:|----------:|-------:|
| Safe routed nets | 2963/3167 | **2986/3167** | **+23** |
| Physically fully routed boards | 89/112 | **90/112** | **+1** |
| Comparable ordinary DRC findings | 451 | **443** | **-8** |
| Clean boards | 74 | **77** | **+3** |
| Fully routed + clean | 70 | **71** | **+1** |

The native report's 427 findings and the audit's 443 comparable ordinary
findings are different checker views and are reported separately. The latter is
used only for a like-for-like predecessor comparison after excluding board-edge
findings. The candidate itself has zero exact outline failures and zero
board-edge findings.

The first certified run took 324.27 s externally. The independent repeat took
332.50 s externally, 2,593.23 s user CPU, 17.52 s system CPU, and
1,385,381,888 bytes maximum RSS. A matched run of the physically invalid
predecessor took 180.43 s on the same machine. Therefore this rollout is a
correctness/quality gain, not a speed claim; exact safety and constrained reroutes
have a measured runtime cost. The two raw and shared semantic hashes are recorded
in “Reproducible baseline” above.

### Isolated feature-aware via-spacing rollout

The accepted pre-feature report at `21a0570` provides the narrower A/B for the
dense-via work:

| Metric | Accepted baseline | Dense-via final | Change |
|--------|------------------:|----------------:|-------:|
| Routed nets | 2965/3167 | **2989/3167** | +24 |
| Fully routed boards | 91/112 | **92/112** | +1 |
| DRC findings | 483 | **443** | -40 |
| Clean boards | 68 | **72** | +4 |
| Fully routed + clean | 65 | **66** | +1 |
| Total route cost | 377,159 | 377,164 | +5 |
| Median board time | **0.080655 s** | 0.088734 s | slightly slower |
| Nearest-rank P95 | **61.623376 s** | 64.224316 s | slightly slower |
| Maximum board time | 100.631944 s | **96.231495 s** | improved |
| Sum of board timers | 732.029698 s | **702.733228 s** | improved |
| External elapsed | 106.53 s | **99.52 s** | improved |

`bug-reports` moves 2247/2447 → 2270/2447 with 38 full boards in both runs;
its DRC count improves 345 → 295. `srj15` moves 718/720 → 719/720 and 53 →
54 full boards, but its DRC count worsens 138 → 148. Thus the per-group physical
result is mixed even though the aggregate DRC gate improves by 40.

This is also not a per-board monotonicity claim. Ten boards gain 26 routes;
`bugreport63` loses two routes while improving from two DRC findings to zero, for
the net +24. DRC improves on 12 boards and worsens on six. The largest regressions
are `sample12` +20 and `sample16` +3 findings; the largest gains are
`bugreport50` -18 and `bugreport11` -16. Median and p95 are slightly slower in the
isolated A/B, while maximum, aggregate timer sum, and external elapsed improve.
The hardened scorer still returns `KEEP` because both group completion rates are
non-regressing and every aggregate acceptance gate passes.

### Coherent typed-SRJ frontier and terminal-via repair

The typed projection is intentionally narrower than the full modern SRJ schema.
It enforces the coherent uniform subset described above: width, generic margin,
trace↔pad/via↔trace, via↔pad, optional pad↔pad and drill↔drill clearances,
and the exact declared routed via pad/hole geometry. Board `outline` and
`minBoardEdgeClearance` now form a separate enforced contract over both coherent
typed inputs and partial/legacy profiles. `allowViaInPad` still only parses and
round-trips; bus and differential-pair declarations remain outside routing
semantics. JSON compatibility is retained, but adding the optional fields is a
source-level change for downstream Rust struct literals (the workspace is still
version 0.1).

The SRJ29 AM62L/LPDDR4 frontier fixture activated that typed subset at the
pre-outline projection. Its no-resolution CLI used a 498×321×8 grid and routed
**28/33** connections at cost **4,841** with zero findings under the supported
typed DRC; configured server `/solve` pinned the same completion/DRC contract.
The recorded CLI solution
SHA-256 is
`e5ac13e3621c6f81152d51bd4b745de3128901978838308311fbfffcca79c262`.
The current product additionally enforces the fixture's rectangular outline and
edge clearance when routed. This is still not a full-SRJ claim: via-in-pad, bus,
and differential-pair constraints are not part of the asserted result.

Relative to the typed corpus report, the bounded terminal-via repair preserves
2989/3167 routed nets, 92 full boards, and cost 377,164 while moving DRC
**443 → 434**, clean boards **72 → 76**, and fully-routed-clean boards
**66 → 70**. `sample13`, `sample18`, `sample19`, and `sample23` each move from
two findings to zero; `sample44` moves from five to four; no board worsens. A
focused repeated `sample13` check moves from a 585.883 ms median to 590.813 ms
(+0.84%) while becoming clean, so this is a physical-quality improvement rather
than a speed claim.

The accepted repair remains opportunistic: pair-only and drill-only findings do
not independently trigger generic-clearance discovery, though a generic-triggered
move may reduce them incidentally. Every accepted candidate is still graded by
authoritative typed full-board DRC. The typed DRC broad phase is near-linear for
ordinary router output, but an
adversarial input with unbounded distinct coincident via representations can make
its physical-site comparisons quadratic.

### Pre-outline bounded dependency and dihedral legalization portfolio

Against the terminal-via baseline, the accepted `ddafd9a` portfolio routes
**2989/3167 → 3000/3167** nets and **92 → 94** boards fully. DRC improves
**434 → 429**, clean boards hold at 76, fully-routed-clean boards rise
**70 → 71**, and cost moves 377,164 → 379,530 with 11 more retained routes.
The scorer returns `KEEP`: workload identity and both group completion gates are
preserved, errors remain zero, and no aggregate physical-quality gate regresses.

Exactly seven timing-stripped board records change:

| Board | Routed | Cost | DRC |
|-------|-------:|-----:|----:|
| `bugreport27-dd3734` | 11/14 → **12/14** | 1,923 → 2,106 | 18 → 18 |
| `bugreport28-18a9ef` | 11/14 → **12/14** | 1,785 → 2,096 | 18 → 18 |
| `bugreport29-7deae8` | 11/14 → **12/14** | 1,785 → 2,096 | 18 → 18 |
| `bugreport30-2174c8` | 11/12 → **12/12** | 1,995 → 2,342 | 18 → 18 |
| `bugreport36-d4c6c2` | 5/8 → **7/8** | 903 → 1,495 | 0 → 0 |
| `bugreport50-e1c376` | 300/322 → **303/322** | 46,280 → 46,600 | 15 → **10** |
| `bugreport63-274be2` | 10/12 → **12/12** | 1,246 → 1,548 | 0 → 0 |

The dependency candidate learns an acyclic failed-group→blocking-owner graph
during the ordinary bounded FIFO rip-up attempt, then evaluates at most one
stable topological restart. It collects no graph above 192 groups, 384 nets, or
1.5M cells. The order candidate runs only when the established portfolio leaves
an individually routable net incomplete and the board has 6–16 groups, at most
32 nets, and at most 250k cells. It samples no more than four deduplicated
cyclic/dihedral orders rather than enumerating all rotations. Both candidates
must strictly increase routed-net count; equal-completion cost or route changes
cannot replace the established result.

### Exact board-outline rollout

The independent route-identity audit found 248 board-edge violations attached to
37 routed identities on the nine active-outline boards. The constrained release
has zero outline failures and zero edge findings. Unaffected board semantics are
preserved; within the affected cohort, every board is non-regressing after the
invalid predecessor identities are removed:

| Board | Pre-outline raw | Invalid | Pre-outline safe | Release safe | Safe delta |
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

The legacy-first selector is central to both compatibility and runtime: it
returns the historical final soup without a constrained solve whenever that soup
is already exact-edge-safe. Only the nine unsafe boards reroute against trace/
via/directed-edge masks, and every constrained soup passes the final continuous
checker and full DRC non-regression gate. This preserves inactive-board topology
and contract-stripped bytes, while malformed active contracts fail before routing.

The 17–18-group extension is deliberately narrower than the existing 6–16-group
portfolio. It is active only with exact board masks, at no more than 32 nets and
250k cells, and only when the established result misses an individually routable
net. At most four legalizations and one rip-up are evaluated on Adaptive; the
established Adaptive/ForceSerial winner remains authoritative on any completion
tie. On `bugreport09`, the constrained route moves 25/27 with six native findings
to 26/27 with one; `bugreport43` pins the equal-completion no-change control.

Two performance/hardening experiments were rejected from production:

- A speculative parallel legacy+constrained selector improved focused
  `bugreport46` wall time 14.90 → 13.09 s, but used 10% more CPU and 6% more
  maximum RSS. The trade did not justify duplicating routing work, so it is a
  NO-GO.
- Dynamic phase-B revalidation of staged preferred paths passed exact semantic
  review and focused tests, including the established via-ring and owner/halo/
  endpoint semantics. In the full composition it caused excessive full-board
  fallbacks and did not finish the corpus within 300 s. It remains a correctness-
  sound research stack, not a production change, because the fallback/runtime
  bound is unacceptable.

The legacy baseline report contained 1,493 findings under the old first-seen layer
checker; that number is intentionally not used in the comparison table. Rechecking
the byte-identical baseline routes with the corrected physical-layer/ownership
semantics yields the checker-comparable 1,227 above. For historical provenance,
the intermediate `7aee642` run moved `bugreport05` 86/228 → 80/228, cost 37,109
→ 39,288, and findings 15 → 12; its six-board preflight was 70/113 with 31
findings. Those intermediate values are not substituted into either current-release
table. Corpus `total_wall_ms` is a sum of overlapping board timers; external
elapsed is the real end-to-end measure.

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
- [Board-outline and edge-clearance enforcement frontier (PR 2145)](https://github.com/tscircuit/tscircuit-autorouter/pull/2145)

PR 2145 exposed the concrete `bugreport21-board-outline` concave-cutout gap that
the current release now closes. The original research stack was expanded through
all release gates: exact trace/via/bidirectional-edge masks, continuous final-soup
DRC, legacy-byte preservation for already-safe routes, fail-closed Metal fallback,
zero-length/singleton semantics, narrow malformed-outline normalization, and an
indexed layer-invariant raster. The dynamic phase-B hardening experiment described
above remains separate because of its unbounded fallback behavior, not because
the shipped board-edge contract depends on it.

The most credible next architecture is CPU coarse corridor/hypergraph planning,
GPU-batched detailed candidate search/scoring, and exact CPU DRC acceptance with
bounded local repair/partial rip. The production NegotiatedRouter does not yet
offload its dynamic congestion iterations to Metal.

Licensing note: the reviewed core repositories and several fixture sets are MIT.
`srj24` includes Apache-2.0 KiCad-derived inputs and needs retained provenance.
Public repositories without a license (including several SRJ repro repositories)
must not be vendored or redistributed without permission.
