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
use mr_drc::{DrcBoard, DrcRules, LayerKind, Pad, Segment, ViolationClass};
use mr_srj::{rasterize_with_layers, to_solution_layered, RoutePoint, SimpleRouteJson};

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

/// NATIVE DRC, `track_w > clearance` regime: route a board whose trace width (0.3)
/// exceeds its clearance (0.1) end-to-end through the live pipeline, build a physical
/// `mr_drc::DrcBoard` from the routed copper + pads, and assert the native checker
/// reports ZERO clearance violations.
///
/// This is the real-board analogue of the grid-level invariant in `mr-srj`
/// (`track_gt_clearance_reserves_centreline_margin_zero_free_nodes`): the rasteriser
/// now reserves `clearance + track_w/2` around foreign copper, so the centreline the
/// router lays on a grid node is far enough that the 0.3-wide trace keeps full 0.1
/// clearance to every foreign pad. Before the fix a centred trace could sit 0.233 from
/// a pad edge → copper 0.083 away → a clearance violation; this guards against that.
#[test]
fn track_gt_clearance_routes_drc_clean_native_checker() {
    // Two columns of three different-net pads, inner edges 0.7 apart so the rasteriser
    // places fill lanes between them (the dangerous gap from the unit test), plus a
    // routing target on the far side, forcing the router to thread the channel.
    const SRJ: &str = r#"{
        "minTraceWidth": 0.3,
        "minClearance": 0.1,
        "layerCount": 1,
        "bounds": { "minX": 0, "maxX": 6, "minY": 0, "maxY": 6 },
        "obstacles": [
            { "type": "rect", "center": {"x": 2.0, "y": 2.0}, "width": 0.6, "height": 0.6, "connectedTo": ["a"] },
            { "type": "rect", "center": {"x": 3.3, "y": 2.0}, "width": 0.6, "height": 0.6, "connectedTo": ["b"] },
            { "type": "rect", "center": {"x": 2.0, "y": 4.0}, "width": 0.6, "height": 0.6, "connectedTo": ["a"] },
            { "type": "rect", "center": {"x": 3.3, "y": 4.0}, "width": 0.6, "height": 0.6, "connectedTo": ["b"] }
        ],
        "connections": [
            { "name": "a", "pointsToConnect": [ {"x": 2.0, "y": 2.0}, {"x": 2.0, "y": 4.0} ] },
            { "name": "b", "pointsToConnect": [ {"x": 3.3, "y": 2.0}, {"x": 3.3, "y": 4.0} ] }
        ]
    }"#;
    let srj: SimpleRouteJson = serde_json::from_str(SRJ).unwrap();
    let trace_w = srj.min_trace_width.unwrap();
    let clearance = srj.min_clearance.unwrap();
    let resolution = choose_resolution(&srj);
    // Same clearance_cells the CLI/server derive for the routing pipeline.
    let clearance_cells = (clearance / resolution).ceil() as u32;
    let layers = LayerMap::standard(1);
    let problem = rasterize_with_layers(&srj, resolution, layers.clone(), clearance_cells);
    // Route with the router's OWN negotiation clearance DISABLED (clearance_cells = 0),
    // so DRC-cleanliness rests on the rasteriser's `clearance + track_w/2` grid blocking
    // alone (not the router's separate clearance). The tight geometric guard that the
    // dangerous nodes are gone is the mr-srj unit test
    // `track_gt_clearance_reserves_centreline_margin_zero_free_nodes`; THIS test is the
    // end-to-end confirmation that the live route → physical-board → native checker path
    // reports ZERO clearance violations on a track_w>clearance board.
    let board = NegotiatedRouter::new()
        .route(&problem.grid, &problem.nets)
        .expect("route");
    let traces = to_solution_layered(&board, &problem.mapping, &problem.pin_points, trace_w, &layers);

    // Build a physical DRC board: every routed wire vertex-pair → a Segment, every
    // pad → a Pad (its own net). The net of a trace is the net of the two pads it
    // joins; recover it by matching the trace's endpoints to a pad centre.
    let mut segments: Vec<Segment> = Vec::new();
    let pad_specs = [
        ("a", 2.0, 2.0),
        ("b", 3.3, 2.0),
        ("a", 2.0, 4.0),
        ("b", 3.3, 4.0),
    ];
    let net_at = |x: f64, y: f64| -> Option<&'static str> {
        pad_specs
            .iter()
            .find(|(_, px, py)| (px - x).abs() <= 0.31 && (py - y).abs() <= 0.31)
            .map(|(n, _, _)| *n)
    };
    for tr in &traces {
        // Resolve the trace net from its first wire vertex.
        let first = tr.route.iter().find_map(|p| match p {
            RoutePoint::Wire { x, y, .. } => Some((*x, *y)),
            _ => None,
        });
        let net = first
            .and_then(|(x, y)| net_at(x, y))
            .unwrap_or("unknown")
            .to_string();
        let pts: Vec<(f64, f64, u32)> = tr
            .route
            .iter()
            .filter_map(|p| match p {
                RoutePoint::Wire { x, y, .. } => Some((*x, *y, 0u32)),
                _ => None,
            })
            .collect();
        for w in pts.windows(2) {
            segments.push(Segment {
                net: net.clone(),
                layer: w[0].2,
                a: (w[0].0, w[0].1),
                b: (w[1].0, w[1].1),
                width: trace_w,
            });
        }
    }
    let pads: Vec<Pad> = pad_specs
        .iter()
        .map(|(n, x, y)| Pad {
            net: Some((*n).to_string()),
            layer: 0,
            center: (*x, *y),
            width: 0.6,
            height: 0.6,
        })
        .collect();

    let drc = DrcBoard {
        layers: vec![LayerKind::Signal],
        segments,
        pads,
        vias: Vec::new(),
        rules: DrcRules {
            clearance,
            plane_antipad: 0.0,
            min_annular_ring: 0.0,
        },
    };
    let violations = drc.check();
    let clearance_viol: Vec<_> = violations
        .iter()
        .filter(|v| v.class == ViolationClass::Clearance)
        .collect();
    assert!(
        clearance_viol.is_empty(),
        "native DRC: {} clearance violation(s) on a track_w>clearance board: {:#?}",
        clearance_viol.len(),
        clearance_viol
    );
    // Sanity: the router actually laid copper (otherwise zero violations is vacuous).
    assert!(
        !drc.segments.is_empty(),
        "expected routed copper to DRC-check"
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
