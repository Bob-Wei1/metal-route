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
//! * Each [`PcbTrace`] gets a synthetic, stable net name (`trace#<i>`). The SRJ
//!   solution soup carries no net identity on a [`PcbTrace`], so per-trace names
//!   preserve same-net immunity *within* a trace (its own corners never trip a
//!   clearance violation) while keeping every distinct trace mutually foreign.
//! * Obstacles (pads / keepouts) have unknown net (`None`), so the checker treats
//!   them as conflicting with every net — conservative, never hides a real short.
//!
//! The result is deterministic: layer indices are assigned by first-seen order
//! over a stable traversal of the traces (then obstacles), and the checker itself
//! sorts its output.

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

    // Traces first (stable: outer index, then route order) so layer indices are
    // deterministic and same-net immunity holds within each trace.
    for (i, t) in traces.iter().enumerate() {
        let net = format!("trace#{i}");
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
                RoutePoint::Via { x, y, from_layer, to_layer } => {
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

    // Obstacles: one Pad per occupied layer. Net is unknown (None) → conservative.
    for o in &srj.obstacles {
        if o.layers.is_empty() {
            // Unlayered obstacle: place it on layer 0 so it is still checked.
            let l = idx.intern("");
            pads.push(Pad {
                net: None,
                layer: l,
                center: (o.center.x, o.center.y),
                width: o.width,
                height: o.height,
            });
        } else {
            for layer in &o.layers {
                let l = idx.intern(layer);
                pads.push(Pad {
                    net: None,
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
        RoutePoint::Wire { x, y, width: 0.1, layer: "top".into() }
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
        let rules = DrcRules { clearance: 0.2, plane_antipad: 0.25, min_annular_ring: 0.05 };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        let v = board.check();
        let clearance = v.iter().filter(|x| x.class == ViolationClass::Clearance).count();
        assert!(clearance >= 1, "sub-clearance crossing must produce a violation");
    }

    #[test]
    fn clean_solution_has_no_violations() {
        // Same two traces, but well apart (1.0 centreline gap >> clearance).
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]),
            PcbTrace::new(vec![wire(0.0, 1.0), wire(10.0, 1.0)]),
        ];
        let rules = DrcRules { clearance: 0.2, plane_antipad: 0.25, min_annular_ring: 0.05 };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(board.check().is_empty(), "well-spaced solution must be DRC-clean");
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
        let rules = DrcRules { clearance: 0.2, plane_antipad: 0.25, min_annular_ring: 0.05 };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(board.check().is_empty(), "a trace's own corners never conflict");
    }

    #[test]
    fn via_emits_real_geometry() {
        let srj = srj_no_obstacles();
        let traces = vec![PcbTrace::new(vec![
            RoutePoint::Wire { x: 1.0, y: 1.0, width: 0.1, layer: "top".into() },
            RoutePoint::Via { x: 1.0, y: 1.0, from_layer: "top".into(), to_layer: "bottom".into() },
            RoutePoint::Wire { x: 1.0, y: 1.0, width: 0.1, layer: "bottom".into() },
        ])];
        let rules = DrcRules { clearance: 0.2, plane_antipad: 0.25, min_annular_ring: 0.05 };
        let board = solution_to_drc_board(&srj, &traces, rules, 2);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.vias[0].pad_diameter, VIA_PAD_MM);
        assert_eq!(board.vias[0].drill_diameter, VIA_DRILL_MM);
        // 0.45/0.2 via → annular ring 0.125 > 0.05, so no annular violation.
        assert!(board.check().is_empty());
    }
}
