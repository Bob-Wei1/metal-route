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
| `mr-cpu` | **CPU routers:** Lee/Dijkstra (M1), A\* baseline, the separable prefix-min sweep (M0), and bounded rip-up (M2). |
| `mr-metal` | **Metal GPU kernels (macOS):** M3 wavefront + M4 row/column prefix-min sweep, and `MetalRouter`. |
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

# Run the local tscircuit-style benchmark and write the CPU baseline report
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
congestion + equal unrouted set, under the `LowerCellIdx` tie-break):

- M3 GPU wavefront field == CPU BFS field — **pass** (incl. the 32×32 wall, cost 93).
- M4 GPU prefix-min sweep field == CPU sweep field == CPU BFS field — **pass**.
- `MetalRouter` ≡ `LeeRouter` on the battery + multi-net boards — **pass**.

### M0 finding (the de-risk gate)

The single largest assumption was that GAMER's separable H-then-V prefix-min sweep
transfers to the PCB obstacle model. It does — **for distances**: the converged
sweep cost field is bit-identical to Lee's BFS across the whole obstacle battery.
The subtlety: a converged *cost* field does **not** carry the canonical *path* for
free — backward greedy descent can pick a different equal-cost path. The fix used
in both CPU and GPU is to descend by **lowest `CellIdx` valid predecessor** under
the same `(dist, idx)` ordering the sweep relaxes with, which reproduces the
tie-break path. (See `crates/mr-cpu/src/sweep.rs`.)

### Speed (honest)

D3 batch benchmark, 128×128 grid, 64 independent nets, like-for-like (each net
routed as an independent shortest-path field):

| Router | nets/sec | wall |
|--------|---------:|-----:|
| CPU `LeeRouter` | ~133 | ~481 ms |
| Metal `MetalRouter` | ~185 | ~345 ms |

**Metal is ~1.39× faster than CPU Lee at this scale** — modest, and dominated by
per-net GPU dispatch overhead (a synchronous command buffer per sweep round per
net, no cross-net batching). This is exactly the PCB-scale caveat the design
predicted: PCB grids are small relative to VLSI, so launch/transfer latency eats
most of the parallel win. A real speedup needs batching many nets into one
dispatch — future work.

### tscircuit-style benchmark (CPU baseline)

`metalroute bench` generates a deterministic SimpleRouteJson suite and scores the
CPU router on completion rate, mean trace length, and throughput. Checked-in
baseline ([`benchmarks/cpu_baseline.json`](benchmarks/cpu_baseline.json)):
10 boards, **56% completion**, **357 nets/sec**, M2 **GO** (2.43×).

The honest tension this exposes: **sparse** boards route ~96–100% but are M2
**NO-GO** (too few nets to beat GPU dispatch), while **dense batch** boards are M2
**GO** but the single-layer rip-up router only completes ~40–56% — the regime that
favors the GPU is the regime where CPU completion is hardest. Better net ordering
/ multi-layer routing would lift completion (future work).

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

Current baseline (`negotiated` router, per-board declared layers):

| corpus | boards | net completion | fully routed |
|--------|-------:|---------------:|-------------:|
| `srj15` | 55 | **73.8%** (531/720) | 8/55 |
| `bug-reports` | 57 | **73.8%** (1806/2447) | 24/57 |
| **total** | **112** | **73.8%** (2337/3167) | 32/112 |

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

M0 (sweep de-risk) ✅ · M1 (CPU Lee) ✅ · M2 (rip-up + go/no-go) ✅ · M3 (Metal
wavefront) ✅ · M4 (Metal prefix-min sweep) ✅ · M5 (Freerouting handoff seam) ✅.

Explicitly out of scope so far: multi-layer + vias, 45°/any-angle routing,
cost-based congestion-aware global routing, a GUI.

## License

MIT OR Apache-2.0.
