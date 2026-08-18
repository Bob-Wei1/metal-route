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

The Rust source suite now contains **385 test cases** (384 passing, one explicitly
ignored live-tool test), up from 243, plus two passing doctests. The added cases
targeted cross-router contracts, zero-cost cycles, weighted and multilayer ties,
physical clearance on non-uniform grids, actual rip-up displacement, SRJ/DSN
layer and rotation semantics, deterministic DRC/oracle behavior, GPU batching and
memory caps, fixed-fixture and fixed-seed Metal equivalence, cached isolation
diagnoses, parallel scheduling determinism, authoritative DRC acceptance, and
topology-preserving via repair.
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
vias, windows, and passable pads) behind a dependency-inverted CPU provider. Its
256×192×2, 48-net median-of-seven warm latency is 16.7–18.9 ms. Real-board A/Bs did
**not** establish a reliable automatic crossover—five of eight representative
boards were slower—so the production negotiated router keeps targeted CPU A* by
default. Experimental offload is explicit with
`METALROUTE_EXPERIMENTAL_METAL_ISOLATED=1`; GPU contention or any command failure
immediately takes the exact whole-batch CPU fallback.

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

Exact baseline run on 2026-08-17 and final/recheck on 2026-08-18 (`negotiated`,
identical 112 boards and settings):

| corpus | before | after | fully routed before → after |
|--------|-------:|------:|----------------------------:|
| `srj15` | 705/720 (97.9%) | **718/720 (99.7%)** | 46 → **53** |
| `bug-reports` | 1996/2447 (81.6%) | **2011/2447 (82.2%)** | 32 → **38** |
| **total** | 2701/3167 (85.3%) | **2729/3167 (86.2%)** | 78 → **91** |

Using the same corrected DRC checker on both route sets, exact-geometry findings
fall **1227 → 714**, algorithmic route cost falls **340,055 → 332,354**,
clean boards rise **49 → 66**, and fully-routed-clean boards rise **46 → 63**.
The hardened scorer returns **`KEEP`**: workload identity is exact,
completion and full-board counts improve in both corpus groups, errors remain
zero, and every aggregate DRC/clean-board gate improves. This is an aggregate
claim: 43 boards improve and four worsen; 18 boards become clean and one becomes
dirty; 19 become fully-routed-clean and two lose that status.

The exact finished-code run improves the headline median and tail summaries:
standard median **1.392 s → 0.080 s** (17.3×), nearest-rank p95
**339.9 s → 41.7 s** (8.15×), sum of overlapping board timers
**4715 s → 546 s** (8.63×), and
external elapsed **715.55 s → 83.04 s** (8.62×). With 112 boards, the reported
median is the arithmetic mean of sorted observations 56 and 57 (one-indexed), not
a lower- or upper-middle value; nearest-rank p95 is observation 107. Peak per-board
time falls 679.5 s → 83.0 s. The principal speed wins are unique-cell SRJ pad-halo
filtering, O(1) exact Hanan-distance heuristics, and fused Jacobi pricing with
planar/via neighbor enumeration that avoids per-expansion allocation; exact A/Bs
retained normalized route output. Pad-ownership-aware smoothing and one bounded,
board-wide DRC-scored via move contribute to the separate physical-quality work.
Legacy unlabeled-pad ownership is inferred only from immutable trace endpoints,
preventing a moved interior via from claiming a foreign pad during repair scoring.
The DRC bridge now preserves the standard physical
layer stack, uses endpoint-side ownership for terminal vias, propagates declared
connectivity across each routed group, and resolves only unambiguous fixed-pad
aliases. The DRC exact-gap fast reject adds a smaller measured 10.8%
microbenchmark improvement.

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
clearance halos, 45° output beautification, exact DRC, and deterministic corpus
gates are implemented.

The next architectural step is a shared global candidate portfolio: coarse
hypergraph/corridor planning and fanout, GPU-batched detailed candidate scoring,
then exact DRC acceptance/repair. The production negotiated router is still CPU;
Metal currently accelerates independent shortest-path batches rather than the
dynamic congestion loop itself.

## License

MIT OR Apache-2.0.
