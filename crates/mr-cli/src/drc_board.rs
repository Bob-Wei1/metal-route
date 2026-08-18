//! Bridge from a routed [`SimpleRouteJson`] solution (continuous-space
//! [`PcbTrace`]s) to a physical [`mr_drc::DrcBoard`] so the corpus benchmark can
//! run a real geometric DRC over what it actually drew.
//!
//! This mirrors the logic of [`crate::drc::build_drc_board`] but works from the
//! *emitted solution soup* (traces in board units) rather than the cell-space
//! [`mr_core::BoardRoute`]. It is intentionally simple and conservative:
//!
//! * Every routed layer is treated as [`LayerKind::Signal`] (we have no plane
//!   information in the SRJ solution path, so we never *invent* a plane — which
//!   keeps via-through-plane checks honest rather than fabricating shorts).
//! * Net identity is RECONSTRUCTED from geometry, since a [`PcbTrace`] in the
//!   solution soup carries no net field. Traces that share an exact vertex are
//!   the same electrical net (junction-grouped same-net sub-connections meet at a
//!   shared point — the geometric twin of the DSN path's `r.net.split('#')` base-net
//!   collapse). A union-find over shared vertices assigns one synthetic net name
//!   per connected component, so neither a trace's own corners NOR a sibling
//!   sub-net trips a false clearance violation, while genuinely distinct nets stay
//!   mutually foreign.
//! * Obstacles (pads / keepouts) are tagged with the net of whichever trace
//!   *terminates inside* them (a routed net ends on its own pad), so own-pad
//!   contact is immune. An obstacle no trace lands in is foreign copper / a
//!   keepout and keeps net `None` (conflicts with every net — never hides a short).
//!
//! The result is deterministic: vertex keys are quantised to a fixed grid, union
//! roots and layer indices are assigned by first-seen order over a stable
//! traversal, and the checker itself sorts its output.

use std::collections::BTreeMap;

use mr_drc::{DrcBoard, DrcRules, LayerKind, Pad, Segment, Via};
use mr_srj::{PcbTrace, RoutePoint, SimpleRouteJson};

use crate::{VIA_DRILL_MM, VIA_PAD_MM};

/// Resolve continuous-space layer *names* (e.g. `"top"`, `"inner1"`, `"bottom"`)
/// to dense physical layer indices, assigned in first-seen order for determinism.
#[derive(Default)]
struct LayerIndex {
    map: BTreeMap<String, u32>,
    order: Vec<String>,
}

impl LayerIndex {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.map.get(name) {
            return i;
        }
        let i = self.order.len() as u32;
        self.map.insert(name.to_string(), i);
        self.order.push(name.to_string());
        i
    }

    /// Number of distinct layers seen (at least `min`, so a single-layer board
    /// still reports a one-layer stack).
    fn len(&self, min: usize) -> usize {
        self.order.len().max(min)
    }
}

/// Build a physical [`DrcBoard`] from a routed SRJ solution.
///
/// * `srj` supplies the static obstacles (pads / keepouts).
/// * `traces` is the emitted solution soup (wires + vias) in board units.
/// * `rules` are the DRC constraints to enforce.
/// * `layers` is the routed layer count; it lower-bounds the layer stack so a
///   board with vias spanning unseen layers still reports a sensible stack.
///
/// Pure and deterministic.
/// Reconstruct the per-trace electrical-net label exactly as [`solution_to_drc_board`]
/// (and hence the DRC) sees it, in trace order:
///
/// 1. Start from the router's tag (`PcbTrace::net`, a `g<groupid>`), or a union-find
///    component label over shared vertices for untagged (hand-built) traces.
/// 2. Relabel any trace whose endpoint lands on an SRJ `connectivity_netNNNN` pad to
///    `c<net>`, collapsing the many router sub-groups that solder to one shared junction
///    pad into a single electrical net — the same immunity the DRC grants.
///
/// Exposed so the clearance legaliser can tag traces with the DRC's own net identity
/// before it runs, making its same-net immunity (and its internal violation gate) agree
/// with the authoritative checker rather than the bare router groups.
pub fn reconstruct_net_labels(srj: &SimpleRouteJson, traces: &[PcbTrace]) -> Vec<String> {
    let mut uf = UnionFind::new(traces.len());
    let mut first_at: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for (i, t) in traces.iter().enumerate() {
        for rp in &t.route {
            let (x, y) = match rp {
                RoutePoint::Wire { x, y, .. } => (*x, *y),
                RoutePoint::Via { x, y, .. } => (*x, *y),
            };
            let key = (quantize(x), quantize(y));
            match first_at.get(&key) {
                Some(&j) => uf.union(i, j),
                None => {
                    first_at.insert(key, i);
                }
            }
        }
    }
    let mut net_name: Vec<String> = Vec::with_capacity(traces.len());
    let mut root_name: BTreeMap<usize, String> = BTreeMap::new();
    for (i, t) in traces.iter().enumerate() {
        let name = match &t.net {
            Some(n) => n.clone(),
            None => {
                let r = uf.find(i);
                let n = root_name.len();
                root_name
                    .entry(r)
                    .or_insert_with(|| format!("net#{n}"))
                    .clone()
            }
        };
        net_name.push(name);
    }
    let conn_pads: Vec<(&mr_srj::Obstacle, &str)> = srj
        .obstacles
        .iter()
        .filter_map(|o| mr_srj::obstacle_connectivity_net(o).map(|n| (o, n)))
        .collect();
    let connectivity_at = |x: f64, y: f64| -> Option<String> {
        conn_pads
            .iter()
            .filter(|(o, _)| {
                (x - o.center.x).abs() <= o.width / 2.0 + 1e-6
                    && (y - o.center.y).abs() <= o.height / 2.0 + 1e-6
            })
            .min_by(|(a, _), (b, _)| {
                (a.width * a.height)
                    .partial_cmp(&(b.width * b.height))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, n)| format!("c{n}"))
    };
    for (i, t) in traces.iter().enumerate() {
        let ends = [t.route.first(), t.route.last()];
        let endpoint_net = ends.into_iter().flatten().find_map(|rp| {
            let (x, y) = match rp {
                RoutePoint::Wire { x, y, .. } | RoutePoint::Via { x, y, .. } => (*x, *y),
            };
            connectivity_at(x, y)
        });
        if let Some(c) = endpoint_net {
            net_name[i] = c;
        }
    }
    net_name
}

pub fn solution_to_drc_board(
    srj: &SimpleRouteJson,
    traces: &[PcbTrace],
    rules: DrcRules,
    layers: u32,
) -> DrcBoard {
    let mut idx = LayerIndex::default();
    let mut segments: Vec<Segment> = Vec::new();
    let mut vias: Vec<Via> = Vec::new();
    let mut pads: Vec<Pad> = Vec::new();

    // Reconstruct the per-trace electrical-net identity exactly as the DRC sees it.
    let net_name = reconstruct_net_labels(srj, traces);
    // Trace vertices tagged with their net, for pad-net resolution below.
    let mut tagged_vertices: Vec<(f64, f64, String)> = Vec::new();

    // Traces first (stable: outer index, then route order) so layer indices are
    // deterministic and same-net immunity holds within each net component.
    for (i, t) in traces.iter().enumerate() {
        let net = net_name[i].clone();
        // Record every vertex tagged with this net so an obstacle can be matched
        // to the net that terminates inside it.
        for rp in &t.route {
            let (x, y) = match rp {
                RoutePoint::Wire { x, y, .. } | RoutePoint::Via { x, y, .. } => (*x, *y),
            };
            tagged_vertices.push((x, y, net.clone()));
        }
        // Walk the route, emitting one Segment per consecutive Wire pair on the
        // same layer, and one Via per Via point.
        let mut prev: Option<(f64, f64, f64, u32)> = None; // x, y, width, layer
        for rp in &t.route {
            match rp {
                RoutePoint::Wire { x, y, width, layer } => {
                    let l = idx.intern(layer);
                    if let Some((px, py, pw, pl)) = prev {
                        if pl == l && (px, py) != (*x, *y) {
                            segments.push(Segment {
                                net: net.clone(),
                                layer: l,
                                a: (px, py),
                                b: (*x, *y),
                                // Use the wider of the two endpoints' widths so a
                                // tapering segment is checked at its fattest.
                                width: pw.max(*width),
                            });
                        }
                    }
                    prev = Some((*x, *y, *width, l));
                }
                RoutePoint::Via {
                    x,
                    y,
                    from_layer,
                    to_layer,
                } => {
                    let from = idx.intern(from_layer);
                    let to = idx.intern(to_layer);
                    vias.push(Via {
                        net: net.clone(),
                        center: (*x, *y),
                        pad_diameter: VIA_PAD_MM,
                        drill_diameter: VIA_DRILL_MM,
                        from_layer: from,
                        to_layer: to,
                        antipad_radius: None,
                    });
                    // A via does not break the wire polyline's layer continuity in
                    // the SRJ soup (the next Wire carries its own layer), so we do
                    // NOT update `prev` from a via.
                }
            }
        }
    }

    // Obstacles: one Pad per occupied layer. The pad's net is the net of whichever
    // trace terminates inside it (a routed net ends on its own pad) — so own-pad
    // contact is immune. An obstacle no trace lands in is foreign copper / a keepout
    // and keeps net `None` (conflicts with every net — never hides a real short).
    for o in &srj.obstacles {
        let net = pad_net(o, &tagged_vertices);
        // A pad declaring a connectivity net is labelled `c<net>` to match the
        // traces relabelled above, so the pad is immune to EVERY trace of its own
        // electrical net (including the many sub-nets that share a junction pad). A
        // pad with no connectivity net falls back to the trace-containment tag.
        if o.layers.is_empty() {
            // Unlayered obstacle: place it on layer 0 so it is still checked.
            let l = idx.intern("");
            pads.push(Pad {
                net: net.clone(),
                layer: l,
                center: (o.center.x, o.center.y),
                width: o.width,
                height: o.height,
            });
        } else {
            for layer in &o.layers {
                let l = idx.intern(layer);
                pads.push(Pad {
                    net: net.clone(),
                    layer: l,
                    center: (o.center.x, o.center.y),
                    width: o.width,
                    height: o.height,
                });
            }
        }
    }

    let stack_len = idx.len(layers.max(1) as usize);
    let layer_stack: Vec<LayerKind> = (0..stack_len).map(|_| LayerKind::Signal).collect();

    DrcBoard {
        layers: layer_stack,
        segments,
        pads,
        vias,
        rules,
    }
}

/// Quantise a coordinate to a fixed sub-micron grid so two vertices the router
/// emitted at the "same" junction collide despite float round-trip noise.
fn quantize(v: f64) -> i64 {
    // 1e4 units per mm → 0.1 µm resolution, far finer than any real pitch.
    (v * 10_000.0).round() as i64
}

/// The electrical net of obstacle `o`. Prefer the ground-truth `connectivity_netNNNN`
/// it declares (labelled `c<net>` to match the connectivity-relabelled traces), so a
/// pad is immune to its own net even when several router sub-nets share it. With no
/// declared connectivity net, fall back to the net of whichever tagged trace vertex
/// lands inside the pad rect (inclusive); failing that, `None` (foreign / keepout —
/// conflicts with every net, never hiding a real short).
fn pad_net(o: &mr_srj::Obstacle, tagged: &[(f64, f64, String)]) -> Option<String> {
    if let Some(n) = mr_srj::obstacle_connectivity_net(o) {
        return Some(format!("c{n}"));
    }
    let (hw, hh) = (o.width / 2.0, o.height / 2.0);
    let (cx, cy) = (o.center.x, o.center.y);
    tagged
        .iter()
        .find(|(x, y, _)| (*x - cx).abs() <= hw && (*y - cy).abs() <= hh)
        .map(|(_, _, net)| net.clone())
}

/// Minimal union-find for grouping traces into electrical nets by shared vertices.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Attach the higher-index root to the lower so naming is first-seen stable.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi] = lo;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_drc::ViolationClass;
    use mr_srj::PcbTrace;

    fn srj_no_obstacles() -> SimpleRouteJson {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "obstacles": [],
            "connections": [],
        });
        serde_json::from_value(v).unwrap()
    }

    fn wire(x: f64, y: f64) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width: 0.1,
            layer: "top".into(),
        }
    }

    #[test]
    fn two_nets_crossing_too_close_violates() {
        // Two different traces on the same layer, parallel, copper gap below the
        // 0.2 clearance: centreline gap 0.15, half-widths 0.05 each → 0.05 gap.
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]),
            PcbTrace::new(vec![wire(0.0, 0.15), wire(10.0, 0.15)]),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        let v = board.check();
        let clearance = v
            .iter()
            .filter(|x| x.class == ViolationClass::Clearance)
            .count();
        assert!(
            clearance >= 1,
            "sub-clearance crossing must produce a violation"
        );
    }

    #[test]
    fn clean_solution_has_no_violations() {
        // Same two traces, but well apart (1.0 centreline gap >> clearance).
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]),
            PcbTrace::new(vec![wire(0.0, 1.0), wire(10.0, 1.0)]),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "well-spaced solution must be DRC-clean"
        );
    }

    #[test]
    fn same_trace_corner_is_not_a_violation() {
        // A single trace bending back on itself: same net, so never a clearance hit.
        let srj = srj_no_obstacles();
        let traces = vec![PcbTrace::new(vec![
            wire(0.0, 0.0),
            wire(10.0, 0.0),
            wire(10.0, 0.05),
            wire(0.0, 0.05),
        ])];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "a trace's own corners never conflict"
        );
    }

    #[test]
    fn junction_grouped_subnets_are_one_net() {
        // Two traces meeting at a shared junction vertex (1.0, 0.0) are the same
        // electrical net, so even though they run within clearance they must NOT
        // produce a clearance violation (mirrors the DSN path's `#seg` collapse).
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(1.0, 0.0)]),
            // Shares the (1.0, 0.0) vertex, then runs parallel 0.05 away.
            PcbTrace::new(vec![wire(1.0, 0.0), wire(1.0, 0.05), wire(0.0, 0.05)]),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "sub-nets sharing a junction vertex must be treated as one net"
        );
    }

    #[test]
    fn pad_is_immune_to_the_net_that_terminates_in_it() {
        // A pad obstacle whose net is the trace ending inside it must not conflict
        // with that trace's copper (own-pad contact is legal).
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.6, "height": 0.6, "layers": ["top"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        // Trace starts inside the pad rect at (0,0) and runs out.
        let traces = vec![PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)])];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "a pad must be immune to the net that terminates inside it"
        );
    }

    #[test]
    fn shared_connectivity_pad_immunises_sibling_subnets() {
        // Two separately-grouped traces (distinct router `g` labels) both terminate
        // on ONE junction pad declaring `connectivity_net7`. They run within
        // clearance of each other AT the pad. Because the pad and both traces resolve
        // to the same `c<net>` identity via `connectedTo`, this must NOT violate —
        // it is one electrical net meeting at its own pad.
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 10.0, "minY": -1.0, "maxY": 10.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.6, "height": 0.6, "layers": ["top"],
                "connectedTo": ["pcb_smtpad_0", "connectivity_net7", "source_port_1"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        // Both traces END inside the pad at (0,0) but carry DIFFERENT group labels,
        // and run parallel 0.05 apart (well within clearance) leaving the pad.
        let traces = vec![
            PcbTrace::new(vec![wire(10.0, 0.0), wire(0.0, 0.0)]).with_net("g0"),
            PcbTrace::new(vec![wire(10.0, 0.05), wire(0.0, 0.0)]).with_net("g1"),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "sub-nets sharing a connectivity pad must be one electrical net: {:?}",
            board.check()
        );
    }

    #[test]
    fn distinct_connectivity_nets_still_conflict() {
        // Two traces on DIFFERENT connectivity nets running sub-clearance must STILL
        // violate — the relabel must not blanket-immunise everything.
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 10.0, "minY": -1.0, "maxY": 10.0},
            "obstacles": [
                {"type": "rect", "center": {"x": 0.0, "y": 0.0}, "width": 0.4, "height": 0.4,
                 "layers": ["top"], "connectedTo": ["connectivity_net1"]},
                {"type": "rect", "center": {"x": 0.0, "y": 0.15}, "width": 0.4, "height": 0.4,
                 "layers": ["top"], "connectedTo": ["connectivity_net2"]}
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]).with_net("g0"),
            PcbTrace::new(vec![wire(0.0, 0.15), wire(10.0, 0.15)]).with_net("g1"),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            !board.check().is_empty(),
            "different connectivity nets within clearance must still violate"
        );
    }

    #[test]
    fn via_emits_real_geometry() {
        let srj = srj_no_obstacles();
        let traces = vec![PcbTrace::new(vec![
            RoutePoint::Wire {
                x: 1.0,
                y: 1.0,
                width: 0.1,
                layer: "top".into(),
            },
            RoutePoint::Via {
                x: 1.0,
                y: 1.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            },
            RoutePoint::Wire {
                x: 1.0,
                y: 1.0,
                width: 0.1,
                layer: "bottom".into(),
            },
        ])];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 2);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.vias[0].pad_diameter, VIA_PAD_MM);
        assert_eq!(board.vias[0].drill_diameter, VIA_DRILL_MM);
        // 0.45/0.2 via → annular ring 0.125 > 0.05, so no annular violation.
        assert!(board.check().is_empty());
    }
}
