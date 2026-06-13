# Benchmark results

Reproduce the tscircuit harness with `scripts/bench-tscircuit.sh <samples> [cats…]`
(clones `tscircuit/autorouting`, runs `runBenchmark` against `mr-server /solve`;
a sample passes only if the solution clears the harness DRC: every port connected
+ no trace overlapping a pad or another trace). Offline regression coverage of
the same router lives in `crates/mr-cli/tests/srj_fixtures.rs` (real captured
problems, no bun needed).

## tscircuit autorouting harness

| category | start | now | samples |
|---|---:|---:|---:|
| single-trace | 0% | **99%** | 100 |
| distant-single-trace | 0% | **97%** | 100 |
| traces | 0% | **52%** | 100 |
| keyboards | 0% | **25%** | 8 |

The starting 0% was a format/contract gap (every net's endpoint sat on its own
pad obstacle → HTTP 400). Getting from there to here took, in order:

1. **Protocol correctness** — parse obstacle `connectedTo`/`layers`, point
   `layer`, `minTraceWidth`; snap trace endpoints to exact ports; per-net pad
   masking (a net may cross only its own pads). → single-trace/distant ~100%.
2. **Negotiated-congestion router** (PathFinder) replacing priority rip-up:
   route all nets on a soft cost grid (present-sharing + accumulating history),
   converge to cell-disjoint routes, then legalize. → multi-net `traces`.
3. **Order-robust + bounded rip-up legalization** for co-dependent nets.

**Single-layer ceiling.** `traces`/`keyboards` declare `layerCount=1` but many
instances are only routable across layers (the harness's own reference solver
falls back to vias on them). Our router is single-layer, so those cap below 100%.
Multi-layer + vias is the next lever (design.md "Phase 5").

## Real board: bed-of-nails fixture (the clean rig)

`metalroute route-dsn` parses a Specctra `.dsn` → `SimpleRouteJson` → routes with
`NegotiatedRouter`, reporting wall-clock + connectivity. Run on the H-PCB52832
bed-of-nails fixture `test5.dsn`:

```
route-dsn --input test5.dsn --resolution 0.2 \
          --skip-nets=GND --skip-nets=+5VA --skip-nets=-5VA --skip-nets=3V3
```

- Board: **8 layers, 90 components, 480 pads, 103 nets, 139.22 × 165.58 mm**
  (grid 697 × 828 = 577k cells at 0.2 mm).
- Routed signal nets (planes skipped): **39 / 212 two-point segments (18.4%),
  8 original nets fully connected**.
- **Wall-clock: 99.4 s** (single thread). The 5-minute budget is met.

Connectivity is low because this is an 8-layer board forced onto one layer — the
crossings that the original design resolves with vias have nowhere to go. The
deliverable here is the **speed**: routing was infeasible (well over 5 min)
before the perf work and is now ~99 s.

### Speed work that got it under budget

Per-net routing used to cost O(all board cells): each net cloned the full grid
and allocated board-sized `dist`/`pred` arrays, ×nets ×~60 negotiation
iterations. Fixes (`crates/mr-cpu/src/{dijkstra,negotiated}.rs`):

- `SearchBuf`: generation-stamped reusable search buffers — O(1) reset,
  O(explored) work, no per-call allocation.
- `astar_buf`: closure-driven A* over the base grid + global history/present
  arrays — no per-net grid clone.
- **Per-net window**: each net searches only the bounding box of its endpoints
  (+margin), with a full-board retry on failure. Bed-of-nails nets are local, so
  this is the dominant win.

Result: ~8× lower per-net wall time; full board infeasible → ~99 s.
