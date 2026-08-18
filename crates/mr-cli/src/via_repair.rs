//! Bounded exact-DRC repair for clearance-conflicting interior vias.
//!
//! The continuous clearance legaliser handles movable wire vertices and rigid vias
//! with local monotone nudges. A small residue remains when one interior through-via
//! is boxed against several foreign features. This module tries one deliberately
//! narrow topology-preserving portfolio: a generic-clearance geometry prefilter
//! selects at most eight vias, then moves each by one generic clearance in eight
//! compass directions. Typed pair-specific or drill-only findings can therefore be
//! left unproposed, but every proposed candidate is checked against the authoritative
//! typed full-board DRC; at most one strictly lower-finding candidate is retained.

use mr_drc::{dist, point_rect_gap, point_seg_dist, DrcBoard, DrcRules, Via, Violation};
use mr_srj::{PcbTrace, RoutePoint, SimpleRouteJson};

use crate::{drc_board, drc_candidate_is_better, drc_severity};

const MAX_REPAIR_VIAS: usize = 8;
const GEOMETRY_EPS_MM: f64 = 1e-9;
const LENGTH_QUANTUM_MM: f64 = 1e-6;

// Cardinal directions first, then diagonals. The order is part of deterministic
// tie-breaking; diagonal vectors are normalised so every candidate moves exactly
// one clearance, independent of direction.
const REPAIR_DIRECTIONS: [(f64, f64); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MovableVia {
    drc_via_index: usize,
    trace_index: usize,
    point_index: usize,
}

/// Candidate ordering after the strict authoritative acceptance gate.
///
/// Count remains primary. The aggregate quantised deficit chooses the gentlest exact
/// DRC result at equal count, then extra planar copper discourages gratuitous detours.
/// Stable soup indices and direction finish the total order without float comparison.
type CandidateRank = (usize, u128, u64, usize, usize, usize);

/// Try the bounded via portfolio once, returning the input byte-for-byte when no
/// strictly lower-finding candidate survives structural and authoritative DRC gates.
pub(crate) fn repair_clearance_vias(
    srj: &SimpleRouteJson,
    traces: Vec<PcbTrace>,
    rules: DrcRules,
    layers: u32,
) -> Vec<PcbTrace> {
    if rules.clearance <= 0.0
        || traces.is_empty()
        || !traces.iter().any(|trace| {
            trace
                .route
                .iter()
                .any(|point| matches!(point, RoutePoint::Via { .. }))
        })
    {
        return traces;
    }

    let before_board = drc_board::solution_to_drc_board(srj, &traces, rules, layers);
    let before = drc_board::check_with_srj_rules(srj, &before_board);
    if before.is_empty() {
        return traces;
    }

    let relevant = clearance_violating_vias(&before_board);
    let movable = select_movable_vias(&traces, &before_board, &relevant);
    if movable.is_empty() {
        return traces;
    }

    let before_length = planar_copper_length(&before_board);
    let mut best: Option<(CandidateRank, Vec<PcbTrace>)> = None;

    for candidate_via in movable {
        let via_radius = before_board.vias[candidate_via.drc_via_index].pad_diameter / 2.0;
        let RoutePoint::Via { x, y, .. } =
            traces[candidate_via.trace_index].route[candidate_via.point_index]
        else {
            continue;
        };
        for (direction_index, (dx, dy)) in REPAIR_DIRECTIONS.into_iter().enumerate() {
            let candidate_center = (x + dx * rules.clearance, y + dy * rules.clearance);
            if !via_inside_bounds(candidate_center, via_radius, srj)
                || !candidate_site_is_unique(
                    &traces,
                    candidate_via.trace_index,
                    candidate_via.point_index,
                    candidate_center,
                )
            {
                continue;
            }

            let mut candidate = traces.clone();
            if !move_via_rigid(
                &mut candidate[candidate_via.trace_index],
                candidate_via.point_index,
                candidate_center,
            ) {
                continue;
            }

            let candidate_board = drc_board::solution_to_drc_board(srj, &candidate, rules, layers);
            let candidate_violations = drc_board::check_with_srj_rules(srj, &candidate_board);
            // This repair is intentionally count-only for acceptance. Reuse the
            // authoritative comparator as a second guard so its semantics stay the
            // single source of truth for future result-shape changes.
            if candidate_violations.len() >= before.len()
                || !drc_candidate_is_better(&before, &candidate_violations)
            {
                continue;
            }

            let rank = (
                candidate_violations.len(),
                total_deficit(&candidate_violations),
                added_copper_length(before_length, planar_copper_length(&candidate_board)),
                candidate_via.trace_index,
                candidate_via.point_index,
                direction_index,
            );
            if best.as_ref().is_none_or(|(best_rank, _)| rank < *best_rank) {
                best = Some((rank, candidate));
            }
        }
    }

    best.map_or(traces, |(_, candidate)| candidate)
}

/// Generic-clearance participation prefilter over the public DRC geometry.
/// Pair-specific typed and via-hole thresholds are intentionally not projected
/// here, so this helper may omit a repair opportunity; the authoritative typed
/// full-board gate above still prevents an unsafe candidate from being accepted.
/// Via-through-plane and annular-ring findings are also excluded because
/// translating a via cannot change either invariant.
fn clearance_violating_vias(board: &DrcBoard) -> Vec<usize> {
    let threshold = board.rules.clearance - GEOMETRY_EPS_MM;
    if threshold <= 0.0 {
        return Vec::new();
    }

    board
        .vias
        .iter()
        .enumerate()
        .filter_map(|(via_index, via)| {
            let radius = via.pad_diameter / 2.0;
            let segment_conflict = board.segments.iter().any(|segment| {
                via_present_on_layer(via, segment.layer, board.layers.len())
                    && segment.net != via.net
                    && point_seg_dist(via.center, segment.a, segment.b)
                        - radius
                        - segment.width / 2.0
                        < threshold
            });
            let pad_conflict = board.pads.iter().any(|pad| {
                via_present_on_layer(via, pad.layer, board.layers.len())
                    && pad.net.as_deref() != Some(via.net.as_str())
                    && point_rect_gap(via.center, pad.center, pad.width, pad.height) - radius
                        < threshold
            });
            let via_conflict = board.vias.iter().enumerate().any(|(other_index, other)| {
                other_index != via_index
                    && other.net != via.net
                    && via_spans_overlap(via, other, board.layers.len())
                    && dist(via.center, other.center) - radius - other.pad_diameter / 2.0
                        < threshold
            });
            (segment_conflict || pad_conflict || via_conflict).then_some(via_index)
        })
        .collect()
}

fn via_present_on_layer(via: &Via, layer: u32, layer_count: usize) -> bool {
    let in_physical_stack = layer_count == 0 || layer < layer_count as u32;
    let lo = via.from_layer.min(via.to_layer);
    let hi = via.from_layer.max(via.to_layer);
    in_physical_stack && (lo..=hi).contains(&layer)
}

fn via_spans_overlap(a: &Via, b: &Via, layer_count: usize) -> bool {
    let lo = a
        .from_layer
        .min(a.to_layer)
        .max(b.from_layer.min(b.to_layer));
    let mut hi = a
        .from_layer
        .max(a.to_layer)
        .min(b.from_layer.max(b.to_layer));
    if layer_count != 0 {
        hi = hi.min(layer_count.saturating_sub(1) as u32);
    }
    lo <= hi
}

/// Map DRC vias back to their stable trace/route positions (the bridge emits vias in
/// exactly this order), reject terminal/shared/malformed sites, then enforce the hard
/// eight-via cap before any candidate board is built.
fn select_movable_vias(
    traces: &[PcbTrace],
    board: &DrcBoard,
    relevant: &[usize],
) -> Vec<MovableVia> {
    let route_vias: Vec<(usize, usize)> = traces
        .iter()
        .enumerate()
        .flat_map(|(trace_index, trace)| {
            trace
                .route
                .iter()
                .enumerate()
                .filter_map(move |(point_index, point)| {
                    matches!(point, RoutePoint::Via { .. }).then_some((trace_index, point_index))
                })
        })
        .collect();
    if route_vias.len() != board.vias.len() {
        return Vec::new();
    }

    relevant
        .iter()
        .copied()
        .filter_map(|drc_via_index| {
            let &(trace_index, point_index) = route_vias.get(drc_via_index)?;
            structurally_movable(traces, trace_index, point_index).then_some(MovableVia {
                drc_via_index,
                trace_index,
                point_index,
            })
        })
        .take(MAX_REPAIR_VIAS)
        .collect()
}

fn structurally_movable(traces: &[PcbTrace], trace_index: usize, via_index: usize) -> bool {
    let Some(trace) = traces.get(trace_index) else {
        return false;
    };
    // `via_index == 1` would require moving the first (terminal) wire point.
    if via_index < 2 || via_index + 1 >= trace.route.len() {
        return false;
    }
    let RoutePoint::Via {
        x: via_x,
        y: via_y,
        from_layer,
        to_layer,
    } = &trace.route[via_index]
    else {
        return false;
    };
    let RoutePoint::Wire {
        x: source_x,
        y: source_y,
        layer: source_layer,
        ..
    } = &trace.route[via_index - 1]
    else {
        return false;
    };
    let RoutePoint::Wire {
        x: landing_x,
        y: landing_y,
        layer: landing_layer,
        ..
    } = &trace.route[via_index + 1]
    else {
        return false;
    };
    if source_layer != from_layer
        || landing_layer != to_layer
        || dist((*source_x, *source_y), (*via_x, *via_y)) > GEOMETRY_EPS_MM
    {
        return false;
    }
    let explicit_landing = dist((*landing_x, *landing_y), (*via_x, *via_y)) <= GEOMETRY_EPS_MM;
    if explicit_landing && via_index + 2 == trace.route.len() {
        return false; // rigid translation would move the terminal route point
    }

    // A coincident point outside this via and its two possible landing anchors is a
    // shared electrical junction. Moving just one branch would silently disconnect it.
    !traces
        .iter()
        .enumerate()
        .any(|(other_trace_index, other_trace)| {
            other_trace
                .route
                .iter()
                .enumerate()
                .any(|(other_point_index, point)| {
                    let permitted = other_trace_index == trace_index
                        && (other_point_index == via_index
                            || other_point_index + 1 == via_index
                            || (explicit_landing && other_point_index == via_index + 1));
                    !permitted && dist(route_point_xy(point), (*via_x, *via_y)) <= GEOMETRY_EPS_MM
                })
        })
}

fn candidate_site_is_unique(
    traces: &[PcbTrace],
    trace_index: usize,
    via_index: usize,
    candidate: (f64, f64),
) -> bool {
    let trace = &traces[trace_index];
    let RoutePoint::Via { x, y, .. } = &trace.route[via_index] else {
        return false;
    };
    let explicit_landing =
        dist(route_point_xy(&trace.route[via_index + 1]), (*x, *y)) <= GEOMETRY_EPS_MM;
    !traces
        .iter()
        .enumerate()
        .any(|(other_trace_index, other_trace)| {
            other_trace
                .route
                .iter()
                .enumerate()
                .any(|(other_point_index, point)| {
                    let moves_with_via = other_trace_index == trace_index
                        && (other_point_index == via_index
                            || other_point_index + 1 == via_index
                            || (explicit_landing && other_point_index == via_index + 1));
                    !moves_with_via && dist(route_point_xy(point), candidate) <= GEOMETRY_EPS_MM
                })
        })
}

/// Translate one structurally eligible via, its mandatory source anchor, and an
/// optional explicit destination landing as a rigid unit. A noncoincident next Wire
/// is the canonical compressed destination leg; it stays fixed and the bridge emits
/// the new via-to-wire landing segment.
fn move_via_rigid(trace: &mut PcbTrace, via_index: usize, center: (f64, f64)) -> bool {
    if !center.0.is_finite() || !center.1.is_finite() || via_index >= trace.route.len() {
        return false;
    }
    let RoutePoint::Via { x, y, .. } = trace.route[via_index].clone() else {
        return false;
    };
    let explicit_landing =
        dist(route_point_xy(&trace.route[via_index + 1]), (x, y)) <= GEOMETRY_EPS_MM;

    let RoutePoint::Wire {
        x: source_x,
        y: source_y,
        ..
    } = &mut trace.route[via_index - 1]
    else {
        return false;
    };
    *source_x = center.0;
    *source_y = center.1;
    let RoutePoint::Via { x, y, .. } = &mut trace.route[via_index] else {
        return false;
    };
    *x = center.0;
    *y = center.1;
    if explicit_landing {
        let RoutePoint::Wire { x, y, .. } = &mut trace.route[via_index + 1] else {
            return false;
        };
        *x = center.0;
        *y = center.1;
    }
    trace_structure_is_valid(trace)
}

fn trace_structure_is_valid(trace: &PcbTrace) -> bool {
    trace.route.len() >= 2
        && trace
            .route
            .windows(2)
            .all(|pair| match (&pair[0], &pair[1]) {
                (RoutePoint::Wire { layer: left, .. }, RoutePoint::Wire { layer: right, .. }) => {
                    left == right
                }
                (
                    RoutePoint::Wire {
                        x: wire_x,
                        y: wire_y,
                        layer,
                        ..
                    },
                    RoutePoint::Via {
                        x: via_x,
                        y: via_y,
                        from_layer,
                        ..
                    },
                ) => {
                    layer == from_layer
                        && dist((*wire_x, *wire_y), (*via_x, *via_y)) <= GEOMETRY_EPS_MM
                }
                (RoutePoint::Via { to_layer, .. }, RoutePoint::Wire { layer, .. }) => {
                    to_layer == layer
                }
                (RoutePoint::Via { .. }, RoutePoint::Via { .. }) => false,
            })
}

fn via_inside_bounds(center: (f64, f64), radius: f64, srj: &SimpleRouteJson) -> bool {
    center.0 - radius >= srj.bounds.min_x
        && center.0 + radius <= srj.bounds.max_x
        && center.1 - radius >= srj.bounds.min_y
        && center.1 + radius <= srj.bounds.max_y
}

fn route_point_xy(point: &RoutePoint) -> (f64, f64) {
    match point {
        RoutePoint::Wire { x, y, .. } | RoutePoint::Via { x, y, .. } => (*x, *y),
    }
}

fn planar_copper_length(board: &DrcBoard) -> f64 {
    board
        .segments
        .iter()
        .map(|segment| dist(segment.a, segment.b))
        .sum()
}

fn total_deficit(violations: &[Violation]) -> u128 {
    violations
        .iter()
        .map(|violation| u128::from(drc_severity(violation)))
        .sum()
}

fn added_copper_length(before: f64, candidate: f64) -> u64 {
    let added = (candidate - before).max(0.0);
    if !added.is_finite() {
        return u64::MAX;
    }
    (added / LENGTH_QUANTUM_MM).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_drc::LayerKind;
    use mr_srj::{Bounds, Obstacle, Point};

    fn wire(x: f64, y: f64, layer: &str) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width: 0.15,
            layer: layer.into(),
        }
    }

    fn via(x: f64, y: f64) -> RoutePoint {
        RoutePoint::Via {
            x,
            y,
            from_layer: "top".into(),
            to_layer: "bottom".into(),
        }
    }

    fn compressed_trace(offset: f64) -> PcbTrace {
        PcbTrace::new(vec![
            wire(offset - 2.0, 0.0, "top"),
            wire(offset, 0.0, "top"),
            via(offset, 0.0),
            wire(offset + 1.0, 0.0, "bottom"),
            wire(offset + 2.0, 0.0, "bottom"),
        ])
        .with_net(format!("n{offset}"))
    }

    fn vertical_compressed_trace(x: f64, net: &str) -> PcbTrace {
        PcbTrace::new(vec![
            wire(x, -2.0, "top"),
            wire(x, 0.0, "top"),
            via(x, 0.0),
            wire(x, 2.0, "bottom"),
        ])
        .with_net(net)
    }

    fn open_srj() -> SimpleRouteJson {
        SimpleRouteJson {
            layer_count: 2,
            min_trace_width: Some(0.15),
            min_clearance: Some(0.15),
            physical_rules: mr_srj::SimpleRoutePhysicalRules::default(),
            obstacles: Vec::new(),
            connections: Vec::new(),
            bounds: Bounds {
                min_x: -5.0,
                max_x: 5.0,
                min_y: -5.0,
                max_y: 5.0,
            },
        }
    }

    fn empty_board(via_count: usize) -> DrcBoard {
        DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: Vec::new(),
            pads: Vec::new(),
            vias: (0..via_count)
                .map(|i| Via {
                    net: format!("n{i}"),
                    center: (i as f64 * 10.0, 0.0),
                    pad_diameter: crate::VIA_PAD_MM,
                    drill_diameter: 0.2,
                    from_layer: 0,
                    to_layer: 1,
                    antipad_radius: None,
                })
                .collect(),
            rules: DrcRules {
                clearance: 0.15,
                plane_antipad: 0.2,
                min_annular_ring: 0.1,
            },
        }
    }

    #[test]
    fn rigid_move_preserves_compressed_destination_leg_and_endpoints() {
        let mut trace = compressed_trace(0.0);
        let first = trace.route.first().cloned().unwrap();
        let last = trace.route.last().cloned().unwrap();
        let compressed_landing = trace.route[3].clone();
        assert!(move_via_rigid(&mut trace, 2, (0.1, -0.1)));
        assert_eq!(trace.route.first(), Some(&first));
        assert_eq!(trace.route.last(), Some(&last));
        assert_eq!(route_point_xy(&trace.route[1]), (0.1, -0.1));
        assert_eq!(route_point_xy(&trace.route[2]), (0.1, -0.1));
        assert_eq!(trace.route[3], compressed_landing);
        assert!(trace_structure_is_valid(&trace));
    }

    #[test]
    fn rigid_move_translates_an_explicit_landing_but_not_a_terminal() {
        let mut trace = PcbTrace::new(vec![
            wire(-2.0, 0.0, "top"),
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
            wire(0.0, 0.0, "bottom"),
            wire(2.0, 0.0, "bottom"),
        ]);
        assert!(move_via_rigid(&mut trace, 2, (0.0, 0.15)));
        assert_eq!(route_point_xy(&trace.route[1]), (0.0, 0.15));
        assert_eq!(route_point_xy(&trace.route[2]), (0.0, 0.15));
        assert_eq!(route_point_xy(&trace.route[3]), (0.0, 0.15));

        let terminal_source = PcbTrace::new(vec![
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
            wire(1.0, 0.0, "bottom"),
        ]);
        assert!(!structurally_movable(&[terminal_source], 0, 1));
    }

    #[test]
    fn selection_rejects_shared_sites_and_caps_vias_before_expansion() {
        let traces: Vec<_> = (0..10).map(|i| compressed_trace(i as f64 * 10.0)).collect();
        let board = empty_board(10);
        let selected = select_movable_vias(&traces, &board, &(0..10).collect::<Vec<_>>());
        assert_eq!(selected.len(), MAX_REPAIR_VIAS);
        assert_eq!(selected.len() * REPAIR_DIRECTIONS.len(), 64);

        let mut shared = traces[..2].to_vec();
        shared[1] = compressed_trace(0.0);
        let shared_board = empty_board(2);
        assert!(select_movable_vias(&shared, &shared_board, &[0, 1]).is_empty());
    }

    #[test]
    fn no_clearance_finding_retains_input_byte_for_byte() {
        let mut srj = open_srj();
        srj.connections = vec![mr_srj::Connection {
            name: "n".into(),
            root_connection_name: None,
            rules: mr_srj::ConnectionRules::default(),
            points_to_connect: vec![
                Point {
                    x: -2.0,
                    y: 0.0,
                    layer: Some("top".into()),
                },
                Point {
                    x: 2.0,
                    y: 0.0,
                    layer: Some("bottom".into()),
                },
            ],
        }];
        let traces = vec![compressed_trace(0.0)];
        let rules = DrcRules {
            clearance: 0.15,
            plane_antipad: 0.2,
            min_annular_ring: 0.1,
        };
        assert_eq!(
            repair_clearance_vias(&srj, traces.clone(), rules, 2),
            traces
        );
    }

    #[test]
    fn relevant_vias_are_retained_when_no_exact_candidate_reduces_count() {
        // The two via pads need 0.60 mm centre spacing but start 0.15 mm apart.
        // A one-clearance (0.15 mm) move cannot clear their two-layer via/trace
        // cluster, so the strict count gate must retain the original soup.
        let srj = open_srj();
        let traces = vec![
            vertical_compressed_trace(0.0, "left"),
            vertical_compressed_trace(0.15, "right"),
        ];
        let rules = DrcRules {
            clearance: 0.15,
            plane_antipad: 0.2,
            min_annular_ring: 0.1,
        };
        let board = drc_board::solution_to_drc_board(&srj, &traces, rules, 2);
        assert!(!board.check().is_empty());
        assert_eq!(clearance_violating_vias(&board), [0, 1]);
        assert_eq!(
            repair_clearance_vias(&srj, traces.clone(), rules, 2),
            traces,
            "equal-count candidate geometry must not churn the soup"
        );
    }

    #[test]
    fn interior_via_cannot_claim_a_foreign_legacy_pad() {
        let mut srj = open_srj();
        // The pad sits on an intermediate layer crossed only by the target via.
        // Its legacy input carries no connectivity identity, so no route owns it.
        srj.obstacles = vec![Obstacle {
            kind: "rect".into(),
            center: Point {
                x: 0.19,
                y: 0.0,
                layer: None,
            },
            width: 0.1,
            height: 0.1,
            shape: None,
            ccw_rotation_degrees: None,
            layers: vec!["inner1".into()],
            connected_to: Vec::new(),
        }];
        // Establish deterministic top/inner1/bottom physical layer indices. The
        // two dummy traces are far from the target geometry and the legacy pad.
        let traces = vec![
            PcbTrace::new(vec![wire(-4.0, -4.0, "top"), wire(-3.0, -4.0, "top")])
                .with_net("top-dummy"),
            PcbTrace::new(vec![wire(-4.0, -3.0, "inner1"), wire(-3.0, -3.0, "inner1")])
                .with_net("inner-dummy"),
            compressed_trace(0.0),
        ];
        let rules = DrcRules {
            clearance: 0.15,
            plane_antipad: 0.2,
            min_annular_ring: 0.1,
        };

        let before_board = drc_board::solution_to_drc_board(&srj, &traces, rules, 3);
        assert_eq!(before_board.pads[0].net, None);
        assert_eq!(
            before_board.check().len(),
            1,
            "fixture has one via-pad finding"
        );

        let repaired = repair_clearance_vias(&srj, traces.clone(), rules, 3);
        assert_eq!(
            repaired, traces,
            "entering an unlabelled foreign pad must not erase its only finding by relabelling it"
        );
        let after_board = drc_board::solution_to_drc_board(&srj, &repaired, rules, 3);
        assert_eq!(after_board.pads[0].net, None);
        assert_eq!(after_board.check().len(), 1);
    }
}
