# metalroute

An experimental PCB global/maze router in Rust, with native Metal compute
kernels for Apple Silicon.

metalroute exists to have fun and push the boundaries of LLM-driven,
benchmark-gated self-improving engineering loops. The project asks a practical
question: how far can a disciplined loop of hypotheses, code changes, tests,
real-board benchmarks, and exact audits move a hard systems problem?

The loop is intentionally simple:

1. Form one concrete routing or performance hypothesis.
2. Make a narrow, attributable change.
3. Run deterministic tests and fixed real-board benchmarks.
4. Audit geometry, performance, and regressions.
5. Keep demonstrated improvements and record the no-go experiments too.

“Self-improving” describes the engineering process around the repository. The
router does not rewrite itself at runtime.

> **Research software:** metalroute is a global/maze-routing laboratory, not a
> production PCB detailed router. Inspect and verify every generated route before
> using it in a real design.

## What it does

- Reads tscircuit SimpleRouteJson and emits `pcb_trace` solution soups.
- Routes multilayer boards with deterministic negotiated congestion, targeted A*,
  bounded rip-up, vias, clearance halos, and legalization portfolios.
- Enforces exact polygonal board outlines and board-edge clearance.
- Runs native geometric DRC and preserves reproducible routing semantics.
- Provides Metal implementations of batched shortest-path field computation.
- Exposes a CLI, a tscircuit-compatible `/solve` server, an interactive route
  visualizer, corpus benchmarks, and a DSN/Freerouting handoff seam.

The default real-board negotiated router currently runs on the CPU. Metal is
proven and useful for bounded independent path batches, but it is not yet the
engine for the dynamic congestion loop.

## Current result

The audited corpus contains 112 real circuit-derived problems vendored from
[tscircuit/tscircuit-autorouter](https://github.com/tscircuit/tscircuit-autorouter).

| Metric | Current audited result |
|---|---:|
| Routed net segments | **2,986 / 3,167 (94.3%)** |
| Fully routed boards | **90 / 112** |
| Total routed cost | **375,563** |
| Native corpus-checker findings | **427** |
| Clean boards | **77 / 112** |
| Fully routed and clean boards | **71 / 112** |
| Corpus errors | **0** |
| Exact outline failures | **0** |
| Exact board-edge findings | **0** |

The predecessor retained more raw routes because it did not enforce board
outlines. After removing its unsafe route identities, the feature-aware comparison
is:

| Physical metric | Pre-outline portfolio | Current | Change |
|---|---:|---:|---:|
| Safe routed net segments | 2,963 | **2,986** | **+23** |
| Physically fully routed boards | 89 | **90** | **+1** |
| Comparable ordinary DRC findings | 451 | **443** | **-8** |
| Clean boards | 74 | **77** | **+3** |
| Fully routed and clean boards | 70 | **71** | **+1** |

The native checker count and comparable-audit count are different views and are
intentionally not conflated. Two independent full-corpus runs produced identical
routing semantics after timing fields were removed.

The Rust workspace contains 556 unit and integration test cases: 546 passing and
ten intentionally ignored manual/live/performance gates. Its two doctests also
pass. Detailed methodology, accepted changes, rejected experiments, hashes, and
checker definitions are in the
[engineering report](research/2026-08-17-routing-improvements.md).

## Metal, without the spin

On an Apple M4, in release mode, with a 128×128 grid and 64 independent nets:

| Measurement | Observed latency |
|---|---:|
| CPU targeted Lee paths | 14.95–18.22 ms |
| CPU complete distance fields | 26.28–27.67 ms |
| Metal batched fields and paths, warm median | **3.42–3.75 ms** |

These are not presented as a like-for-like end-to-end speedup. Targeted Lee stops
early, the CPU field measurement does not reconstruct paths, and Metal computes
complete fields before reconstructing the requested paths.

The production negotiated router remains CPU-first because matched real-board
tests did not establish a reliable crossover for automatic Metal offload. The
experimental isolated provider can be enabled on macOS with:

```sh
METALROUTE_EXPERIMENTAL_METAL_ISOLATED=1 \
  cargo run --locked --release -p mr-cli --bin metalroute -- \
  route --input board.srj.json --out solution.json
```

Unsupported geometry, a busy GPU lane, or a Metal failure falls back atomically
to CPU. Exact board masks currently force that safe fallback because the Metal
kernels do not yet encode their directed-edge representation.

Exact outline enforcement also has a real cost: the two certified corpus runs
took about 324 and 333 seconds on the measurement machine, versus about 180
seconds for the outline-invalid predecessor. This release is a correctness and
physical-quality improvement, not a full-corpus speed claim.

## Freerouting comparison

A pinned three-board smoke test compares byte-identical DAC2020 DSNs with
[Freerouting 2.3.0](https://github.com/freerouting/freerouting/releases/tag/v2.3.0).
Each tool gets one routing worker and three fresh processes; the table reports
median external wall time from process launch through SES creation. Freerouting
then reloads both tools' sessions and supplies the common `U / V` quality count
(unconnected items / violations).

| Fixture | metalroute median | Freerouting median | metalroute U / V | Freerouting U / V | Quality-gated ratio |
|---|---:|---:|---:|---:|---:|
| DAC2020 bm08 | **0.057 s** | 3.556 s | 0 / 24 | **0 / 1** | — |
| DAC2020 bm06 | **10.410 s** | 18.833 s | 4 / 244 | **2 / 8** | — |
| DAC2020 bm07 | 13.692 s | **12.556 s** | 4 / 40 | **3 / 0** | **Freerouting 1.09×** |

No metalroute speedup is established. The reported routing-task count matches on
bm08 (25) and bm07 (86), while bm06 is withheld at 97 vs 98. On bm08,
metalroute's lower raw wall time is not assigned a ratio because its output has
more violations. On bm07, Freerouting is both faster and no worse on the common
quality counts, producing an observed median ratio of **1.09× in Freerouting's
favor** under the gate. bm07's sample spread is high, so treat that number as
directional smoke evidence, not a precise general speedup. Freerouting is no
worse in U and strictly better in V on all three boards.

See the [full samples, hashes, and interpretation](benchmarks/freerouting/results/2026-08-18-m4-pro.md)
and the [reproduction methodology](benchmarks/freerouting/README.md). This is a
three-board, two-layer smoke matrix—not Freerouting's complete benchmark suite.

## Quick start

Requirements:

- Rust 1.93 or newer;
- macOS on Apple Silicon for the Metal backend;
- a platform supported by Rust for the CPU router and non-Metal crates.

```sh
git clone https://github.com/Bob-Wei1/metal-route.git
cd metal-route

cargo build --workspace --locked
cargo test --workspace --locked
```

Route one vendored SimpleRouteJson board:

```sh
cargo run --locked --release -p mr-cli --bin metalroute -- route \
  --input benchmarks/corpus/bug-reports/bugreport21-board-outline.srj.json \
  --out solution.json
```

Run the tscircuit-compatible server:

```sh
cargo run --locked --release -p mr-server -- --port 1234
# POST /solve at http://localhost:1234
```

Run the real-board corpus and generate an SVG gallery:

```sh
scripts/bench-corpus.sh
# output: benchmarks/runs/<timestamp>-corpus/index.html
```

The complete corpus currently takes roughly 5½ minutes and about 1.4 GB peak RSS
on the certification machine. For the interactive visualizer, see
[web/README.md](web/README.md).

## Architecture

```text
SimpleRouteJson / DSN
        │
        ▼
grid, physical costs, pads, vias, exact board mask
        │
        ├── negotiated CPU router — production default
        └── Metal batch field engine — bounded/experimental workloads
        │
        ▼
legalization, exact outline validation, geometric DRC
        │
        ▼
solution soup, CLI, HTTP API, SVG gallery, Freerouting handoff
```

The Cargo workspace has 13 crates:

| Area | Crates |
|---|---|
| Contracts and grid model | `mr-core`, `mr-grid` |
| Routing engines | `mr-cpu`, `mr-metal` |
| Formats and validation | `mr-srj`, `mr-ingest`, `mr-drc`, `mr-oracle` |
| Products and evaluation | `mr-cli`, `mr-server`, `mr-bench`, `mr-fixtures` |
| Detailed-routing bridge | `mr-bridge` |

The shared `Router` contract keeps the CPU oracle, Metal experiments, server, and
benchmark harness independently testable.

## Reproducing the work

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
scripts/bench-corpus.sh
```

Timing varies by machine. Completion, route identity, cost, and DRC semantics are
the deterministic gates. Benchmark inputs and their provenance are documented in
[benchmarks/corpus/MANIFEST.md](benchmarks/corpus/MANIFEST.md).

For a same-DSN, one-worker comparison with Freerouting 2.3.0, including
post-route SES reload and Freerouting DRC, see the
[Freerouting comparison methodology](benchmarks/freerouting/README.md).

## Known limitations

- This is global/maze routing, not a full push-and-shove detailed router.
- Dynamic negotiated congestion is still CPU-bound.
- Exact board masks currently disable Metal routing and use CPU fallback.
- The supported modern SimpleRouteJson physical-rule subset is intentionally
  conservative; via-in-pad, buses, and differential-pair behavior are incomplete.
- Exact outline enforcement improves safety and quality but adds substantial work
  on boards whose unconstrained route crosses an edge.
- Remaining routed output still contains DRC findings and requires inspection.

## Roadmap

- Coarse corridor/hypergraph planning and fanout.
- GPU-batched detailed candidate generation and scoring.
- Exact CPU DRC acceptance with bounded local repair and partial rip-up.
- A measured Metal implementation of more of the congestion loop—or a clearly
  documented no-go result if the hardware crossover is not worthwhile.
- Broader typed SimpleRouteJson rule coverage.
- More hardware and corpus measurements without fixture-specific routing logic.

## Contributing

Issues, experiments, and pull requests are welcome. The most useful change comes
with:

- one stated hypothesis;
- fixed before/after inputs;
- deterministic tests;
- corpus and physical-quality results;
- an explanation of regressions and tradeoffs.

Negative results are valuable here. A cleanly measured no-go prevents the next
LLM or human contributor from repeating the same attractive mistake.

Please do not tune against corpus filenames or modify benchmark inputs, scorers,
or acceptance gates to manufacture an improvement.

## License and fixture provenance

The code is dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.

Vendored benchmark inputs retain their own provenance and licensing. See
[benchmarks/corpus/MANIFEST.md](benchmarks/corpus/MANIFEST.md) and the
[SRJ29 frontier fixture license](benchmarks/frontier/srj29/LICENSE).
