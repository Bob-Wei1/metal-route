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

## Phase 5: multi-layer routing + vias (landed)

The router is no longer 2D-only. The grid carries a layer axis (`Dims.layers`,
flat index `(l*h+y)*w+x` — identical to `y*w+x` at `layers==1`, so every
single-layer result is byte-identical and the offline fixture suite stays 100%).
The negotiated router takes vertical (via) steps between adjacent layers, gated and
priced by a `ViaModel` (through-hole by default; blind/buried spans honoured when a
DSN declares them). Vias are emitted in the solution soup as `route_type:"via"`
points (`from_layer`/`to_layer`), and `route`/`route-dsn` accept `--layers N`.

**Layer policy.** A problem routes on its declared `layerCount` by default (so
single-layer `traces`/`keyboards` are unchanged and benchmark-legal); `--layers`
overrides the budget for real boards. DSN ingest preserves the real stackup names
(`F.Cu`/`B.Cu`/…) and per-pad layer assignment instead of collapsing to `top`.

**Reproducible in-repo demonstrations** (no external assets):

- Unit/integration: `cargo test --workspace` (158 tests). Key end-to-end test
  `single_layer_wall_blocks_but_second_layer_vias_through` (mr-cli): a net walled
  off on `top` is **0/1 routable on one layer → 1/1 with a top↔bottom via detour**
  once `--layers 2` is granted.
- CLI, single board walled on `top`:
  ```
  route --input wall.json --resolution 1.0            # routed 0/1, grid 10x6x1L
  route --input wall.json --resolution 1.0 --layers 2 # routed 1/1, grid 10x6x2L, 2 vias
  ```
  Emitted route: `top` → via↓(3.5,3.5) → `bottom` → via↑(6.5,3.5) → `top`.
- CLI, 2-layer DSN with a net spanning a top pad to a bottom pad:
  ```
  RESULT route-dsn nets=1 routed=1 conn=100.0% vias=1 wall=0.000s grid=40x40x2L
  ```
  (DSN parsed `layers=2`, pads placed on F.Cu/B.Cu, one via bridges them.)

### Headline: multi-layer lift on a real 8-layer bed-of-nails fixture

The committed `test5.dsn` is *partially routed* (its `(wiring)` carries 175
protected power-plane traces + 174 fanout vias), so it is not a clean from-scratch
measurement. Instead we generate a **fresh, unrouted** fixture from the same DUT
(`H-PCB52832-A1.kicad_pcb`) with the bed-of-nails tool and route its signal nets:

```
bon generate H-PCB52832-A1.kicad_pcb -o fixture_fresh -i ad3_mte_2x15
bon route fixture_fresh --export-only          # writes fixture.dsn: 8 layers,
                                               # poured planes + fanout vias, signal unrouted
route-dsn --input fixture_fresh/fixture.dsn --resolution 0.2 \
          --layers N --skip-nets=GND --skip-nets=3V3 --skip-nets=-5VA --skip-nets=+5VA
```

Board: 8 layers, 76 components, 323 pads, 55 signal nets → **142 two-point
segments**, grid 650 × 758 (492.7k cells/layer) at 0.2 mm.

| `--layers` | connectivity | vias | wall-clock |
|-----------:|-------------:|-----:|-----------:|
| 1 (baseline) | **21.1%** (30/142) | 0 | 47.6 s |
| 2 | **100%** (142/142) | 210 | 0.85 s |
| 8 | **100%** (142/142) | 222 | 0.83 s |

The lever lands exactly as designed: single-layer caps at 21% — the crossings have
nowhere to go — while two layers reach **100%**. Multi-layer is also *faster*: on one
layer the negotiated router burns all 60 iterations + the rip-up budget fighting
unwinnable crossings (47 s); with vias available, nets route on the first pass
(<1 s). The 8-layer grid is 8× the cells but costs no more wall-clock because the
windowed A* explores locally and the thrash is gone. The 8-layer solution spreads
wire across F.Cu/In1.Cu/In2.Cu (≈18.0k/10.3k/1.1k vertices) with all vias on short
adjacent spans — it escapes to inner layers only where a crossing demands it, and
three layers already suffice for this sparse fixture.

(`test5.dsn`'s single-layer **18.4%** below was an earlier, differently-seeded
fixture instance; the fresh board's 21.1% is the like-for-like single-layer point.)

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
