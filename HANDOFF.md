# metalroute — DRC-clean routing + acceleration: handoff

Status as of branch `feat/tscircuit-bench-sota`, HEAD `0e7b077`. Working tree clean;
all changed crates pass `cargo test` + `cargo clippy -D warnings` + `cargo fmt --check`.

## What metalroute is

Rust workspace (12 crates) implementing a GPU-accelerable PCB maze router. CPU
`NegotiatedRouter` (PathFinder-style negotiated congestion) is the quality path;
`mr-metal` has a simple GPU distance-field router. Ingests Specctra DSN, routes signal
nets on signal layers with through-vias, exports SES. Goal of this effort: make routed
boards **DRC-clean** and **fast**.

## What was built this effort (all committed)

1. **`mr-drc` crate** (`36d8e7d`) — native geometric DRC checker. Violation classes:
   copper↔copper **clearance** (per-layer uniform-grid spatial index), **via-through-
   plane** (antipad-aware), **annular-ring**. Deterministic, 16 golden tests. Pure
   geometry; not a full KiCad DRC.
2. **DSN clearance + plane parsing** (`4d2cff9`, in `0e7b077`) — `(rule (clearance N))`,
   `(plane "NET" (polygon LAYER ..))` net↔layer bindings, and a pad-id→net map.
3. **`drc` CLI + builder + baselines** (`f3a1c54`) — `metalroute drc` / `route-dsn --drc`.
   `mr-cli/src/drc.rs::build_drc_board` maps routed signal-grid copper onto the FULL
   physical stackup so through-vias are seen crossing inner planes. Baselines in
   `benchmarks/drc_baseline.json`, narrative in `benchmarks/drc_results.md`.
4. **Plane antipads** (`f2ee406`, `e379bae`) — poured zones relieve foreign through-vias;
   `build_drc_board` models `antipad_radius = drill/2 + plane_antipad` (toggle
   `--no-plane-zones`). **Eliminated via-through-plane 304 → 0.**
5. **Clearance halos + soft clearance + P1/P2** (`f2ee406`, `e379bae`, `0e7b077`):
   - Legalization clearance halo (`mr-cpu` `stamp_owner`/`owner`+`halo`), soft (never
     drops nets). **Finding: legalization-only clearance can't reduce violations** on a
     congested coarse grid — disabled by default.
   - **P1 (negotiation clearance):** `present_halo[]` priced into `route_negotiated`
     (`CLEARANCE_NEG_WEIGHT`) so the negotiation spreads nets. Byte-identical at
     clearance=0.
   - **P2 (pad clearance):** `mr-srj` inflates pads by clearance + expands own-pad
     `passable_pads` (same-net pin access). `route_dsn_problem` wires `clearance_cells`
     + via keepout from the DSN rule.
   - **rayon parallelism:** when clearance active, per-iteration nets route in parallel
     (snapshot + deterministic merge); multi-order legalization runs in parallel.
6. **SOTA research** (`benchmarks/drc_sota_research.md`) — TritonRoute (gridded soft
   object/marker cost) + Freerouting (shape clearance). Validated roadmap below.
7. **Toolchain chore** (`f1877cd`) — rustfmt/clippy fixes for rustc/clippy 1.93.

## DRC results progression (`bench/fixture_fresh/fixture.dsn`, 8 layers, 4 planes)

| Stage | routed | clearance | via-plane | total | runtime |
| --- | --- | --- | --- | --- | --- |
| M1 baseline (no clearance) | 142/142 | 2715 | 304 | 3019 | ~5 s |
| M2 hard legalization halo | 108/142 | 1394 | 0 | 1394 | minutes |
| M2.4 default (clearance off, antipads on) | 142/142 | 2715 | 0 | **2715** | ~5 s |
| M3 P1+P2 (clearance in negotiation) — **20-net subset** | 40/40 | 173 | 0 | 173 | <120 s |
| M3 P1+P2 full board | — | — | — | — | **>5 min (unmeasured)** |

The plane-short elimination (304→0) is the durable win. P1 clearly spreads nets
(subset clearance is low), but the full board is too slow to measure cleanly yet.

## OPEN PROBLEMS (priority order)

1. **[BIGGEST] Full board >5 min — incremental rerouting is disabled when clearance is
   on.** `mr-cpu/src/negotiated.rs`: `incremental = n_nets > 8 && !clearance_active`.
   With clearance active, ALL nets re-route EVERY iteration (×~60 iters). The N agent
   disabled the incremental skip because a net's cost can change when a *neighbour's*
   `present_halo` changes, which the copper-only `prev_overused` set doesn't capture.
   **Fix:** track a per-iteration "halo-dirty" cell set (cells whose `present_halo`
   changed) and re-route a net if its path touches a `prev_overused` OR `prev_halo_dirty`
   cell. This should cut re-routes from 60×142 to congested-only → **minutes → seconds**,
   and is the single highest-leverage change. Keep clearance=0 byte-identical.
2. **rayon saturates only ~3/12 cores** — legalization commits groups sequentially within
   each order. Parallelizing within a legalization pass is hard (sequential ownership
   dependency); the multi-order pass is already parallel. After fix #1 reduces work,
   this matters less. Consider more candidate orders to widen the parallel pass.
3. **Clearance not yet 0** — subset shows 173 (not 0). Needs: tune `CLEARANCE_NEG_WEIGHT`
   (currently `SCALE`=16; try higher), **half-pitch grid** (P4 — diagonal/0.45 mm-via
   geometry the cell halo rounds away), **per-type clearance** (P3 — wire/via/pad), and a
   final **DRC-repair rip-up pass** (P5).
4. **GPU is NOT a toggle here.** `mr-metal::MetalRouter` is the *simple distance-field*
   router — no clearance, no negotiation, no via model. Turning it on discards all DRC
   work and produces overlapping copper. Real Metal acceleration needs NEW kernels:
   batched per-net priced-wavefront (the `wavefront` kernel already does arbitrary
   per-cell costs — feed it the priced grid `SCALE+history+pfac*present+clr`), batched
   across nets to beat per-net dispatch overhead (project D3 saw only 1.39× per-net).
   Do this AFTER fix #1 (which may make GPU unnecessary for this board size).
5. **Plane-antipad fix is a model assumption** — assumes planes are poured zones that
   relieve vias. Validate with the best-effort `kicad-cli pcb drc` cross-check (not yet
   wired; needs the SES→.kicad_pcb import path). `--no-plane-zones` gives the pessimistic
   model (still reports all 304).

## GOTCHAS

- **Process management:** the binary is `metalroute drc` (NOT `metalroute -- drc` — cargo
  consumes the `--`). `pkill -f "metalroute -- drc"` is a no-op; use
  `pkill -9 -f "target/debug/metalroute"`. Earlier runs leaked as 100%-CPU zombies that
  starved measurements. **Run the binary directly under `timeout -s KILL <n>`** (not via
  `cargo run`) so a timeout actually kills the child. (macOS `timeout` is GNU coreutils —
  available here.) Always check `ps aux | grep target/debug/metalroute` after.
- **Memory is ~25–140 MB** — the router is compute-bound, not memory-bound (a few
  grid-sized `Vec`s + per-thread `SearchBuf`). Low RAM + 100% CPU is the expected profile.
- rust-analyzer shows spurious "unsafe"/proc-macro errors (toolchain 1.93 vs RA 5 skew) —
  ignore; trust `cargo`.

## How to run / measure

```
# build once
cargo build -p mr-cli
# DRC on the fixture (run binary directly + hard timeout to avoid zombies):
timeout -s KILL 600 ./target/debug/metalroute drc \
  --input bench/fixture_fresh/fixture.dsn \
  --skip-nets=GND --skip-nets=+5VA --skip-nets=-5VA --skip-nets=3V3 \
  --max-violations 20 --out benchmarks/drc_after.json
# subset for a fast signal: add --max-nets 20
# tests / gates (CI set):
cargo test  -p mr-core -p mr-grid -p mr-fixtures -p mr-cpu -p mr-srj -p mr-ingest -p mr-drc -p mr-oracle -p mr-bench -p mr-server -p mr-bridge -p mr-cli
cargo clippy --all-targets <same -p list> -- -D warnings
cargo fmt --all --check
```

## Key files

- `crates/mr-cpu/src/negotiated.rs` — the router. `route()` negotiation loop (incremental
  skip ~line 380, `present_halo` inc/dec ~410/455, rayon parallel block when
  `clearance_active`), `route_negotiated` (priced cost incl. `present_halo`),
  `for_each_halo_cell`, `stamp_owner`/`free_owner` (legalization halo), `legalize_in_order`
  (parallel multi-order), `route_legal` (soft halo). Consts: `SCALE`, `CLEARANCE_PENALTY`,
  `CLEARANCE_NEG_WEIGHT`.
- `crates/mr-drc/src/lib.rs` — DRC checker (`DrcBoard::check`, classes, spatial index).
- `crates/mr-cli/src/drc.rs` — `build_drc_board` (signal→physical stackup, antipads), `drc`
  subcommand. `crates/mr-cli/src/lib.rs::route_dsn_problem` — wires clearance + builds DrcBoard.
- `crates/mr-srj/src/lib.rs` — `rasterize_with_layers` (pad clearance + own-pad access).
- `crates/mr-ingest/src/dsn.rs` — clearance + plane + pin_nets parsing.
- `crates/mr-metal/src/{lib,gpu}.rs` — GPU wavefront/sweep kernels + simple `MetalRouter`.
- `benchmarks/drc_results.md` (progression), `benchmarks/drc_sota_research.md` (roadmap),
  `benchmarks/drc_*.json` (snapshots).

## Recommended next step

Implement **open problem #1 (incremental rerouting under clearance)** first — it's the
highest-leverage, lowest-risk change and unblocks a clean full-board measurement. Then
tune `CLEARANCE_NEG_WEIGHT` + measure, then decide P3/P4/P5 vs GPU based on the residual.
