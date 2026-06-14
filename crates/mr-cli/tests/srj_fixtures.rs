//! Offline regression suite over REAL tscircuit problems.
//!
//! The JSON fixtures in `tests/fixtures/srj/` are exact `SimpleRouteJson`
//! payloads captured from the tscircuit `autorouting` benchmark
//! (`getSimpleRouteJson`) — the same bytes the live harness POSTs to
//! `mr-server`. This lets us route them through the real pipeline
//! (`rasterize` → `NegotiatedRouter`) and assert connectivity WITHOUT needing
//! bun / the harness checked out, so router regressions are caught by
//! `cargo test`.

use mr_core::{LayerMap, Router};
use mr_cpu::NegotiatedRouter;
use mr_srj::{rasterize_with_layers, SimpleRouteJson};

/// Default solve-layer budget mirrored from `mr_server::DEFAULT_SOLVE_LAYERS`.
/// The live solver routes single-layer-declared problems on this many layers so
/// the negotiated router can resolve crossings with through-vias.
const SOLVE_LAYERS: u32 = 2;

const MIN_RESOLUTION: f64 = 0.1;
const TARGET_CELLS_PER_AXIS: f64 = 200.0;

/// Mirror of `mr_server::choose_resolution` so the fixtures route at the same
/// grid the live solver uses (kept in sync intentionally; the server owns the
/// canonical copy).
fn choose_resolution(srj: &SimpleRouteJson) -> f64 {
    let b = &srj.bounds;
    let span = (b.max_x - b.min_x)
        .max(0.0)
        .max((b.max_y - b.min_y).max(0.0));
    if span <= 0.0 {
        return MIN_RESOLUTION;
    }
    let mut res = (span / TARGET_CELLS_PER_AXIS).max(MIN_RESOLUTION);
    if let Some(w) = srj.min_trace_width {
        if w.is_finite() && w > 0.0 {
            res = res.min((w * 2.0).max(MIN_RESOLUTION));
        }
    }
    res
}

/// Load a fixture, route it single-layer, return `(routed, total)` two-point nets.
fn route_fixture(name: &str) -> (usize, usize) {
    route_fixture_layers(name, 1)
}

/// Load a fixture and route it on `max(layerCount, layers)` layers — the same
/// `rasterize_with_layers` → `NegotiatedRouter` path the live `/solve` handler
/// uses. Returns `(routed, total)` two-point nets.
fn route_fixture_layers(name: &str, layers: u32) -> (usize, usize) {
    let path = format!("{}/tests/fixtures/srj/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let srj: SimpleRouteJson = serde_json::from_str(&text).expect("parse SimpleRouteJson");
    let effective = srj.layer_count.max(layers);
    let problem = rasterize_with_layers(
        &srj,
        choose_resolution(&srj),
        LayerMap::standard(effective),
        0,
    );
    let board = NegotiatedRouter::new()
        .route(&problem.grid, &problem.nets)
        .expect("route");
    (board.results.len(), problem.nets.len())
}

/// Single-net categories must route fully — the simplest possible contract.
#[test]
fn single_net_fixtures_fully_route() {
    for name in [
        "single-trace_seed0.json",
        "single-trace_seed1.json",
        "distant-single-trace_seed0.json",
    ] {
        let (routed, total) = route_fixture(name);
        assert_eq!(routed, total, "{name}: routed {routed}/{total}");
    }
}

/// Multi-net `traces` fixtures known to be single-layer routable: every net
/// must connect (these seeds pass on the live harness).
#[test]
fn traces_fixtures_fully_route() {
    for name in [
        "traces_seed0.json",
        "traces_seed1.json",
        "traces_seed3.json",
    ] {
        let (routed, total) = route_fixture(name);
        assert_eq!(routed, total, "{name}: routed {routed}/{total}");
    }
}

/// The router is deterministic: same problem, same routed count, twice.
#[test]
fn routing_is_deterministic() {
    let a = route_fixture("traces_seed0.json");
    let b = route_fixture("traces_seed0.json");
    assert_eq!(a, b);
}

/// A real `keyboards` problem (3 nets, declares `layerCount=1`, mixes `oval` and
/// `rect` obstacles). Guards that the multi-layer `/solve` path — `rasterize_with_layers`
/// → `NegotiatedRouter` → layered solution — parses real keyboard geometry and routes
/// it on the solver's default 2-layer budget without error (no bun needed).
///
/// Note: the live `keyboards`/`traces` lift from the 2-layer budget is a *harness*
/// effect — crossing traces land on different layers so the continuous renders no
/// longer overlap (`@tscircuit/checks checkEachPcbTraceNonOverlapping`). Our internal
/// per-net routed count is the same at 1 vs 2 layers, so the lift is measured by the
/// live harness (`scripts/bench-tscircuit.sh`), not here. The via *mechanism* itself is
/// proven by `mr_cli`'s `single_layer_wall_blocks_but_second_layer_vias_through`.
#[test]
fn keyboards_fixture_routes_on_default_budget() {
    let (single, total) = route_fixture_layers("keyboards_seed1.json", 1);
    let (multi, total2) = route_fixture_layers("keyboards_seed1.json", SOLVE_LAYERS);
    assert!(total > 0, "fixture must carry nets to route");
    assert_eq!(total, total2, "net count must not depend on layer budget");
    assert!(
        multi >= single,
        "more layers must not route fewer nets: {single} -> {multi} of {total}"
    );
    assert_eq!(
        multi, total,
        "every net should connect at 2 layers: {multi}/{total}"
    );
}

/// The multi-layer budget never *regresses* the single-layer-routable categories
/// — `traces`/single-trace fixtures still fully route when given extra layers.
#[test]
fn multi_layer_does_not_regress_single_layer_fixtures() {
    for name in [
        "single-trace_seed0.json",
        "traces_seed0.json",
        "traces_seed1.json",
        "traces_seed3.json",
    ] {
        let (routed, total) = route_fixture_layers(name, SOLVE_LAYERS);
        assert_eq!(routed, total, "{name}: routed {routed}/{total} at 2 layers");
    }
}
