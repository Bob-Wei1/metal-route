//! Bounded exact-DRC repair for clearance-conflicting vias.
//!
//! The continuous clearance legaliser handles movable wire vertices and rigid vias
//! with local monotone nudges. A small residue remains when one through-via is boxed
//! against several foreign features or pinned to a route terminal. This module tries
//! a deliberately narrow topology-preserving portfolio: a generic-clearance prefilter
//! selects at most eight vias, then tries eight compass directions. Interior vias
//! move rigidly by one generic clearance; endpoint-adjacent vias grow a short
//! stationary-terminal dogleg whose radius is chosen from the exact generic-clearance
//! deficit, rounded up to one of four quarter-clearance steps. The physical first/last
//! endpoint stays exact. Typed pair-specific or drill-only findings can therefore be
//! left unproposed, but every proposed candidate is checked against the authoritative
//! typed full-board DRC; at most one strictly lower-finding candidate is retained.

use mr_drc::{dist, point_rect_gap, point_seg_dist, DrcBoard, DrcRules, Via, Violation};
use mr_srj::{PcbTrace, RoutePoint, SimpleRouteJson};

use crate::{drc_board, drc_candidate_is_better, drc_severity};

const MAX_REPAIR_VIAS: usize = 8;
const TERMINAL_STEP_QUANTA: u32 = 4;
const GEOMETRY_EPS_MM: f64 = 1e-9;
const LENGTH_QUANTUM_MM: f64 = 1e-6;

// Cardinal directions first, then diagonals. The order is part of deterministic
// tie-breaking; vectors are normalised so every candidate for a given via uses its
// single selected radius, independent of direction.
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
    kind: ViaMoveKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViaMoveKind {
    InteriorRigid,
    StationarySourceTerminal,
    StationaryDestinationTerminal,
    StationaryDestinationLanding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PhysicalEndpoint {
    x_bits: u64,
    y_bits: u64,
    layer: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TraceTopology {
    net: Option<String>,
    first: Option<PhysicalEndpoint>,
    last: Option<PhysicalEndpoint>,
    via_spans: Vec<(String, String)>,
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
    if !rules.clearance.is_finite()
        || rules.clearance <= 0.0
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

    let before_topology = topology_signature(&traces);
    let before_labels = drc_board::reconstruct_net_labels(srj, &traces, layers);
    let before_length = planar_copper_length(&before_board);
    let mut best: Option<(CandidateRank, Vec<PcbTrace>)> = None;

    for candidate_via in movable {
        let via_radius = before_board.vias[candidate_via.drc_via_index].pad_diameter / 2.0;
        let RoutePoint::Via { x, y, .. } =
            traces[candidate_via.trace_index].route[candidate_via.point_index]
        else {
            continue;
        };
        let step = repair_step(&before_board, candidate_via);
        for (direction_index, (dx, dy)) in REPAIR_DIRECTIONS.into_iter().enumerate() {
            let candidate_center = (x + dx * step, y + dy * step);
            let inside = if candidate_via.kind == ViaMoveKind::InteriorRigid {
                via_inside_bounds(candidate_center, via_radius, srj)
            } else {
                terminal_via_inside_bounds(candidate_center, (x, y), via_radius, srj)
            };
            let unique = candidate_site_is_unique(
                &traces,
                candidate_via.trace_index,
                candidate_via.point_index,
                candidate_via.kind,
                candidate_center,
            );
            if !inside || !unique {
                continue;
            }

            let mut candidate = traces.clone();
            if !move_via(
                &mut candidate[candidate_via.trace_index],
                candidate_via.point_index,
                candidate_via.kind,
                candidate_center,
            ) {
                continue;
            }
            // This is a geometry-only pass. Endpoint-side coordinates/layers,
            // trace order/net tags, via count, and ordered via spans are hard
            // invariants. The endpoint form may canonicalise from a terminal Via
            // to a Wire, but its physical identity remains bit-exact.
            let candidate_topology = topology_signature(&candidate);
            let candidate_labels = drc_board::reconstruct_net_labels(srj, &candidate, layers);
            if candidate_topology != before_topology || candidate_labels != before_labels {
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

/// Pick one compact dogleg radius from the current via's exact generic-clearance
/// deficit. Quarter-clearance quantisation avoids both an ineffectual sub-grid
/// nudge and the new-neighbour collisions caused by always moving a terminal via
/// a full clearance. This still produces exactly one candidate per direction and
/// never exceeds the accepted interior portfolio's one-clearance radius.
fn terminal_repair_step(board: &DrcBoard, via_index: usize) -> f64 {
    let clearance = board.rules.clearance;
    if !clearance.is_finite() || clearance <= 0.0 {
        return 0.0;
    }
    let Some(via) = board.vias.get(via_index) else {
        return clearance;
    };
    let radius = via.pad_diameter / 2.0;
    let mut max_deficit: f64 = 0.0;
    let mut record_gap = |gap: f64| {
        if !gap.is_finite() {
            max_deficit = f64::INFINITY;
        } else if gap < clearance - GEOMETRY_EPS_MM {
            max_deficit = max_deficit.max(clearance - gap);
        }
    };
    for segment in &board.segments {
        if via_present_on_layer(via, segment.layer, board.layers.len()) && segment.net != via.net {
            let gap =
                point_seg_dist(via.center, segment.a, segment.b) - radius - segment.width / 2.0;
            record_gap(gap);
        }
    }
    for pad in &board.pads {
        if via_present_on_layer(via, pad.layer, board.layers.len())
            && pad.net.as_deref() != Some(via.net.as_str())
        {
            let gap = point_rect_gap(via.center, pad.center, pad.width, pad.height) - radius;
            record_gap(gap);
        }
    }
    for (other_index, other) in board.vias.iter().enumerate() {
        if other_index != via_index
            && other.net != via.net
            && via_spans_overlap(via, other, board.layers.len())
        {
            let gap = dist(via.center, other.center) - radius - other.pad_diameter / 2.0;
            record_gap(gap);
        }
    }

    quantized_terminal_step(clearance, max_deficit)
}

fn quantized_terminal_step(clearance: f64, deficit: f64) -> f64 {
    if !clearance.is_finite() || clearance <= 0.0 {
        return 0.0;
    }
    if !deficit.is_finite() {
        return clearance;
    }
    let quantum = clearance / f64::from(TERMINAL_STEP_QUANTA);
    for multiplier in 1..TERMINAL_STEP_QUANTA {
        let boundary = f64::from(multiplier) * quantum;
        if deficit <= boundary + GEOMETRY_EPS_MM {
            return boundary;
        }
    }
    clearance
}

fn repair_step(board: &DrcBoard, via: MovableVia) -> f64 {
    if via.kind == ViaMoveKind::InteriorRigid {
        board.rules.clearance
    } else {
        terminal_repair_step(board, via.drc_via_index)
    }
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
/// exactly this order), reject shared/malformed sites, classify the bounded interior
/// or stationary-terminal rewrite, then enforce the hard eight-via cap before any
/// candidate board is built.
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

    let classified: Vec<_> = relevant
        .iter()
        .copied()
        .filter_map(|drc_via_index| {
            let &(trace_index, point_index) = route_vias.get(drc_via_index)?;
            let kind = movable_via_kind(traces, trace_index, point_index)?;
            Some(MovableVia {
                drc_via_index,
                trace_index,
                point_index,
                kind,
            })
        })
        .collect();

    // Preserve the accepted interior portfolio byte-for-byte: terminal
    // opportunities may consume only slots the old selection left unused.
    let mut selected: Vec<_> = classified
        .iter()
        .copied()
        .filter(|via| via.kind == ViaMoveKind::InteriorRigid)
        .take(MAX_REPAIR_VIAS)
        .collect();
    if selected.len() < MAX_REPAIR_VIAS {
        selected.extend(
            classified
                .iter()
                .copied()
                .filter(|via| via.kind != ViaMoveKind::InteriorRigid)
                .take(MAX_REPAIR_VIAS - selected.len()),
        );
    }
    selected
}

fn movable_via_kind(
    traces: &[PcbTrace],
    trace_index: usize,
    via_index: usize,
) -> Option<ViaMoveKind> {
    let trace = traces.get(trace_index)?;
    if via_index == 0 || via_index >= trace.route.len() {
        return None;
    }
    let RoutePoint::Via {
        x: via_x,
        y: via_y,
        from_layer,
        to_layer,
    } = &trace.route[via_index]
    else {
        return None;
    };
    let RoutePoint::Wire {
        x: source_x,
        y: source_y,
        layer: source_layer,
        ..
    } = &trace.route[via_index - 1]
    else {
        return None;
    };
    if source_layer != from_layer
        || dist((*source_x, *source_y), (*via_x, *via_y)) > GEOMETRY_EPS_MM
    {
        return None;
    }

    let destination = trace.route.get(via_index + 1);
    let explicit_landing = match destination {
        Some(RoutePoint::Wire { x, y, layer, .. }) if layer == to_layer => {
            dist((*x, *y), (*via_x, *via_y)) <= GEOMETRY_EPS_MM
        }
        Some(_) => return None,
        None => false,
    };

    // A coincident point outside this via and its two possible landing anchors is a
    // shared electrical junction. Moving just one branch would silently disconnect it.
    let shared = traces
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
        });
    if shared {
        return None;
    }

    let kind = if via_index == 1 {
        ViaMoveKind::StationarySourceTerminal
    } else if destination.is_none() {
        ViaMoveKind::StationaryDestinationTerminal
    } else if explicit_landing && via_index + 2 == trace.route.len() {
        ViaMoveKind::StationaryDestinationLanding
    } else {
        ViaMoveKind::InteriorRigid
    };
    if kind != ViaMoveKind::InteriorRigid && !trace_structure_is_valid(trace) {
        None
    } else {
        Some(kind)
    }
}

fn candidate_site_is_unique(
    traces: &[PcbTrace],
    trace_index: usize,
    via_index: usize,
    kind: ViaMoveKind,
    candidate: (f64, f64),
) -> bool {
    let trace = &traces[trace_index];
    let RoutePoint::Via { x, y, .. } = &trace.route[via_index] else {
        return false;
    };
    let explicit_landing = trace
        .route
        .get(via_index + 1)
        .is_some_and(|point| dist(route_point_xy(point), (*x, *y)) <= GEOMETRY_EPS_MM);
    !traces
        .iter()
        .enumerate()
        .any(|(other_trace_index, other_trace)| {
            other_trace
                .route
                .iter()
                .enumerate()
                .any(|(other_point_index, point)| {
                    let moves_with_via = if other_trace_index != trace_index {
                        false
                    } else if other_point_index == via_index {
                        true
                    } else {
                        match kind {
                            ViaMoveKind::InteriorRigid => {
                                other_point_index + 1 == via_index
                                    || (explicit_landing && other_point_index == via_index + 1)
                            }
                            ViaMoveKind::StationarySourceTerminal => false,
                            ViaMoveKind::StationaryDestinationTerminal
                            | ViaMoveKind::StationaryDestinationLanding => {
                                other_point_index + 1 == via_index
                            }
                        }
                    };
                    !moves_with_via && dist(route_point_xy(point), candidate) <= GEOMETRY_EPS_MM
                })
        })
}

fn move_via(trace: &mut PcbTrace, via_index: usize, kind: ViaMoveKind, center: (f64, f64)) -> bool {
    if !center.0.is_finite() || !center.1.is_finite() {
        return false;
    }
    let mut moved = trace.clone();
    let changed = match kind {
        ViaMoveKind::InteriorRigid => move_via_rigid(&mut moved, via_index, center),
        ViaMoveKind::StationarySourceTerminal => {
            move_source_terminal_via(&mut moved, via_index, center)
        }
        ViaMoveKind::StationaryDestinationTerminal => {
            move_destination_terminal_via(&mut moved, via_index, center)
        }
        ViaMoveKind::StationaryDestinationLanding => {
            move_destination_landing_via(&mut moved, via_index, center)
        }
    };
    if !changed || !trace_structure_is_valid(&moved) {
        return false;
    }
    *trace = moved;
    true
}

/// Preserve the first Wire endpoint, insert a same-layer source dogleg anchor,
/// and move the Via. If this Via was also the last point, append the stationary
/// destination endpoint on its `to_layer`.
fn move_source_terminal_via(trace: &mut PcbTrace, via_index: usize, center: (f64, f64)) -> bool {
    let Some(RoutePoint::Wire { width, layer, .. }) = trace.route.get(via_index - 1).cloned()
    else {
        return false;
    };
    let Some(RoutePoint::Via {
        x: terminal_x,
        y: terminal_y,
        to_layer,
        ..
    }) = trace.route.get(via_index).cloned()
    else {
        return false;
    };
    let was_destination_terminal = via_index + 1 == trace.route.len();
    trace.route.insert(
        via_index,
        RoutePoint::Wire {
            x: center.0,
            y: center.1,
            width,
            layer,
        },
    );
    let RoutePoint::Via { x, y, .. } = &mut trace.route[via_index + 1] else {
        return false;
    };
    *x = center.0;
    *y = center.1;
    if was_destination_terminal {
        trace.route.push(RoutePoint::Wire {
            x: terminal_x,
            y: terminal_y,
            width,
            layer: to_layer,
        });
    }
    true
}

/// Move a final Via and its nonterminal source anchor, then append a Wire at the
/// old center on `to_layer`. The append is the stationary physical destination.
fn move_destination_terminal_via(
    trace: &mut PcbTrace,
    via_index: usize,
    center: (f64, f64),
) -> bool {
    let Some(RoutePoint::Wire { width, .. }) = trace.route.get(via_index - 1).cloned() else {
        return false;
    };
    let Some(RoutePoint::Via {
        x: terminal_x,
        y: terminal_y,
        to_layer,
        ..
    }) = trace.route.get(via_index).cloned()
    else {
        return false;
    };
    if via_index + 1 != trace.route.len() {
        return false;
    }
    let RoutePoint::Wire { x, y, .. } = &mut trace.route[via_index - 1] else {
        return false;
    };
    *x = center.0;
    *y = center.1;
    let RoutePoint::Via { x, y, .. } = &mut trace.route[via_index] else {
        return false;
    };
    *x = center.0;
    *y = center.1;
    trace.route.push(RoutePoint::Wire {
        x: terminal_x,
        y: terminal_y,
        width,
        layer: to_layer,
    });
    true
}

/// The terminal is already an explicit destination Wire. Move the rigid source
/// anchor and Via but leave that final landing untouched, creating the dogleg.
fn move_destination_landing_via(
    trace: &mut PcbTrace,
    via_index: usize,
    center: (f64, f64),
) -> bool {
    if via_index + 2 != trace.route.len() {
        return false;
    }
    let RoutePoint::Wire { x, y, .. } = &mut trace.route[via_index - 1] else {
        return false;
    };
    *x = center.0;
    *y = center.1;
    let RoutePoint::Via { x, y, .. } = &mut trace.route[via_index] else {
        return false;
    };
    *x = center.0;
    *y = center.1;
    true
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

/// Terminal pads commonly sit one trace-radius inside the routing bounds, so their
/// existing via pad legitimately protrudes beyond that routing envelope. A dogleg
/// may retain (but never increase) the original protrusion on each side.
fn terminal_via_inside_bounds(
    center: (f64, f64),
    original: (f64, f64),
    radius: f64,
    srj: &SimpleRouteJson,
) -> bool {
    let min_x = srj.bounds.min_x.min(original.0 - radius);
    let max_x = srj.bounds.max_x.max(original.0 + radius);
    let min_y = srj.bounds.min_y.min(original.1 - radius);
    let max_y = srj.bounds.max_y.max(original.1 + radius);
    center.0 - radius >= min_x - GEOMETRY_EPS_MM
        && center.0 + radius <= max_x + GEOMETRY_EPS_MM
        && center.1 - radius >= min_y - GEOMETRY_EPS_MM
        && center.1 + radius <= max_y + GEOMETRY_EPS_MM
}

fn route_point_xy(point: &RoutePoint) -> (f64, f64) {
    match point {
        RoutePoint::Wire { x, y, .. } | RoutePoint::Via { x, y, .. } => (*x, *y),
    }
}

fn physical_endpoint(point: &RoutePoint, first: bool) -> PhysicalEndpoint {
    let (x, y) = route_point_xy(point);
    let layer = match point {
        RoutePoint::Wire { layer, .. } => layer.clone(),
        RoutePoint::Via {
            from_layer,
            to_layer,
            ..
        } => {
            if first {
                from_layer.clone()
            } else {
                to_layer.clone()
            }
        }
    };
    PhysicalEndpoint {
        x_bits: x.to_bits(),
        y_bits: y.to_bits(),
        layer,
    }
}

fn topology_signature(traces: &[PcbTrace]) -> Vec<TraceTopology> {
    traces
        .iter()
        .map(|trace| TraceTopology {
            net: trace.net.clone(),
            first: trace
                .route
                .first()
                .map(|point| physical_endpoint(point, true)),
            last: trace
                .route
                .last()
                .map(|point| physical_endpoint(point, false)),
            via_spans: trace
                .route
                .iter()
                .filter_map(|point| match point {
                    RoutePoint::Via {
                        from_layer,
                        to_layer,
                        ..
                    } => Some((from_layer.clone(), to_layer.clone())),
                    RoutePoint::Wire { .. } => None,
                })
                .collect(),
        })
        .collect()
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
    use mr_drc::{LayerKind, Pad, Segment};
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
    fn rigid_move_translates_an_explicit_landing_and_classifies_a_terminal() {
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
        assert_eq!(
            movable_via_kind(&[terminal_source], 0, 1),
            Some(ViaMoveKind::StationarySourceTerminal)
        );
    }

    #[test]
    fn stationary_terminal_doglegs_preserve_physical_endpoints_and_via_spans() {
        let mut source = PcbTrace::new(vec![
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
            wire(1.0, 0.0, "bottom"),
        ])
        .with_net("source");
        let source_signature = topology_signature(std::slice::from_ref(&source));
        assert!(move_via(
            &mut source,
            1,
            ViaMoveKind::StationarySourceTerminal,
            (0.0, 0.15),
        ));
        assert_eq!(
            topology_signature(std::slice::from_ref(&source)),
            source_signature
        );
        assert_eq!(route_point_xy(&source.route[0]), (0.0, 0.0));
        assert_eq!(route_point_xy(&source.route[1]), (0.0, 0.15));
        assert_eq!(route_point_xy(&source.route[2]), (0.0, 0.15));
        assert!(trace_structure_is_valid(&source));

        let mut destination = PcbTrace::new(vec![
            wire(-1.0, 0.0, "top"),
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
        ])
        .with_net("destination");
        let destination_signature = topology_signature(std::slice::from_ref(&destination));
        assert!(move_via(
            &mut destination,
            2,
            ViaMoveKind::StationaryDestinationTerminal,
            (0.0, 0.15),
        ));
        assert_eq!(
            topology_signature(std::slice::from_ref(&destination)),
            destination_signature
        );
        assert_eq!(route_point_xy(&destination.route[1]), (0.0, 0.15));
        assert_eq!(route_point_xy(&destination.route[2]), (0.0, 0.15));
        assert_eq!(route_point_xy(&destination.route[3]), (0.0, 0.0));
        assert!(matches!(
            &destination.route[3],
            RoutePoint::Wire { layer, .. } if layer == "bottom"
        ));
        assert!(trace_structure_is_valid(&destination));

        let srj = open_srj();
        let board = drc_board::solution_to_drc_board(
            &srj,
            &[destination],
            DrcRules {
                clearance: 0.15,
                plane_antipad: 0.2,
                min_annular_ring: 0.1,
            },
            2,
        );
        assert!(board.segments.iter().any(|segment| {
            segment.layer == 1 && segment.a == (0.0, 0.15) && segment.b == (0.0, 0.0)
        }));
    }

    #[test]
    fn two_terminal_via_grows_two_doglegs_without_moving_either_endpoint() {
        let mut trace = PcbTrace::new(vec![wire(0.0, 0.0, "top"), via(0.0, 0.0)]);
        let signature = topology_signature(std::slice::from_ref(&trace));
        assert!(move_via(
            &mut trace,
            1,
            ViaMoveKind::StationarySourceTerminal,
            (0.15, 0.0),
        ));
        assert_eq!(topology_signature(std::slice::from_ref(&trace)), signature);
        assert_eq!(trace.route.len(), 4);
        assert!(trace_structure_is_valid(&trace));
    }

    #[test]
    fn destination_landing_dogleg_keeps_the_explicit_final_wire_stationary() {
        let mut trace = PcbTrace::new(vec![
            wire(-1.0, 0.0, "top"),
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
            wire(0.0, 0.0, "bottom"),
        ]);
        assert_eq!(
            movable_via_kind(std::slice::from_ref(&trace), 0, 2),
            Some(ViaMoveKind::StationaryDestinationLanding)
        );
        let signature = topology_signature(std::slice::from_ref(&trace));
        let last = trace.route.last().cloned();
        assert!(move_via(
            &mut trace,
            2,
            ViaMoveKind::StationaryDestinationLanding,
            (0.0, 0.15),
        ));
        assert_eq!(topology_signature(std::slice::from_ref(&trace)), signature);
        assert_eq!(trace.route.last().cloned(), last);
        assert_eq!(route_point_xy(&trace.route[1]), (0.0, 0.15));
        assert_eq!(route_point_xy(&trace.route[2]), (0.0, 0.15));
    }

    #[test]
    fn candidate_uniqueness_never_excludes_a_stationary_terminal_or_landing() {
        let source = PcbTrace::new(vec![
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
            wire(1.0, 0.0, "bottom"),
        ]);
        assert!(!candidate_site_is_unique(
            std::slice::from_ref(&source),
            0,
            1,
            ViaMoveKind::StationarySourceTerminal,
            (0.0, 0.0),
        ));
        assert!(candidate_site_is_unique(
            std::slice::from_ref(&source),
            0,
            1,
            ViaMoveKind::StationarySourceTerminal,
            (0.2, 0.2),
        ));

        let landing = PcbTrace::new(vec![
            wire(-1.0, 0.0, "top"),
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
            wire(0.0, 0.0, "bottom"),
        ]);
        assert!(!candidate_site_is_unique(
            std::slice::from_ref(&landing),
            0,
            2,
            ViaMoveKind::StationaryDestinationLanding,
            (0.0, 0.0),
        ));

        let final_via = PcbTrace::new(vec![
            wire(0.2, 0.2, "top"),
            wire(-1.0, 0.0, "top"),
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
        ]);
        assert!(!candidate_site_is_unique(
            std::slice::from_ref(&final_via),
            0,
            3,
            ViaMoveKind::StationaryDestinationTerminal,
            (0.2, 0.2),
        ));
        assert!(candidate_site_is_unique(
            std::slice::from_ref(&final_via),
            0,
            3,
            ViaMoveKind::StationaryDestinationTerminal,
            (0.3, 0.3),
        ));
    }

    #[test]
    fn final_via_to_wire_keeps_connectivity_label_and_endpoint_pad_ownership() {
        let srj: SimpleRouteJson = serde_json::from_value(serde_json::json!({
            "layerCount": 2,
            "bounds": {"minX": -2.0, "maxX": 2.0, "minY": -2.0, "maxY": 2.0},
            "obstacles": [
                {"type": "rect", "center": {"x": 0.0, "y": 0.0}, "width": 0.2,
                 "height": 0.2, "layers": ["top"], "connectedTo": ["connectivity_net8"]},
                {"type": "rect", "center": {"x": 0.0, "y": 0.0}, "width": 0.2,
                 "height": 0.2, "layers": ["bottom"], "connectedTo": ["connectivity_net7"]}
            ]
        }))
        .unwrap();
        let mut trace = PcbTrace::new(vec![
            wire(-1.0, 0.0, "top"),
            wire(0.0, 0.0, "top"),
            via(0.0, 0.0),
        ])
        .with_net("g0");
        let before_labels =
            drc_board::reconstruct_net_labels(&srj, std::slice::from_ref(&trace), srj.layer_count);
        assert_eq!(before_labels, ["cconnectivity_net7"]);
        assert!(move_via(
            &mut trace,
            2,
            ViaMoveKind::StationaryDestinationTerminal,
            (0.0, 0.15),
        ));
        assert_eq!(
            drc_board::reconstruct_net_labels(&srj, std::slice::from_ref(&trace), srj.layer_count,),
            before_labels
        );
        let board = drc_board::solution_to_drc_board(
            &srj,
            &[trace],
            DrcRules {
                clearance: 0.15,
                plane_antipad: 0.2,
                min_annular_ring: 0.1,
            },
            2,
        );
        assert!(board
            .pads
            .iter()
            .any(|pad| { pad.layer == 0 && pad.net.as_deref() == Some("cconnectivity_net8") }));
        assert!(board
            .pads
            .iter()
            .any(|pad| { pad.layer == 1 && pad.net.as_deref() == Some("cconnectivity_net7") }));
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
    fn terminal_candidates_never_displace_the_existing_eight_interior_slots() {
        let mut traces = vec![
            PcbTrace::new(vec![
                wire(-20.0, 0.0, "top"),
                via(-20.0, 0.0),
                wire(-19.0, 0.0, "bottom"),
            ]),
            PcbTrace::new(vec![
                wire(-10.0, 0.0, "top"),
                via(-10.0, 0.0),
                wire(-9.0, 0.0, "bottom"),
            ]),
        ];
        traces.extend((0..9).map(|i| compressed_trace(i as f64 * 10.0)));
        let board = empty_board(traces.len());
        let selected = select_movable_vias(&traces, &board, &(0..traces.len()).collect::<Vec<_>>());
        assert_eq!(selected.len(), MAX_REPAIR_VIAS);
        assert!(selected
            .iter()
            .all(|via| via.kind == ViaMoveKind::InteriorRigid));
        assert_eq!(
            selected
                .iter()
                .map(|via| via.drc_via_index)
                .collect::<Vec<_>>(),
            (2..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn terminal_bound_relaxation_retains_but_never_increases_existing_protrusion() {
        let mut srj = open_srj();
        srj.bounds = Bounds {
            min_x: 0.0,
            max_x: 1.0,
            min_y: 0.0,
            max_y: 1.0,
        };
        let radius = 0.2;
        for (original, retained, inward, outward) in [
            ((0.1, 0.5), (0.1, 0.5), (0.2, 0.5), (0.09, 0.5)),
            ((0.9, 0.5), (0.9, 0.5), (0.8, 0.5), (0.91, 0.5)),
            ((0.5, 0.1), (0.5, 0.1), (0.5, 0.2), (0.5, 0.09)),
            ((0.5, 0.9), (0.5, 0.9), (0.5, 0.8), (0.5, 0.91)),
        ] {
            assert!(terminal_via_inside_bounds(retained, original, radius, &srj));
            assert!(terminal_via_inside_bounds(inward, original, radius, &srj));
            assert!(!terminal_via_inside_bounds(outward, original, radius, &srj));
        }

        // A pad that originally fits cannot newly protrude.
        assert!(!terminal_via_inside_bounds(
            (0.19, 0.5),
            (0.2, 0.5),
            radius,
            &srj,
        ));

        // Existing left-side allowance cannot transfer to the right side.
        srj.bounds.max_x = 0.3;
        assert!(!terminal_via_inside_bounds(
            (0.2, 0.5),
            (0.1, 0.5),
            radius,
            &srj,
        ));

        // If the bounds are narrower than the diameter, both protruding sides
        // pin that axis: either movement worsens one side.
        srj.bounds.max_x = 0.2;
        assert!(terminal_via_inside_bounds(
            (0.1, 0.5),
            (0.1, 0.5),
            radius,
            &srj,
        ));
        assert!(!terminal_via_inside_bounds(
            (0.11, 0.5),
            (0.1, 0.5),
            radius,
            &srj,
        ));
        assert!(!terminal_via_inside_bounds(
            (0.09, 0.5),
            (0.1, 0.5),
            radius,
            &srj,
        ));
    }

    #[test]
    fn terminal_step_bins_are_stable_at_every_quarter_boundary() {
        let clearance = 0.2;
        let quantum = clearance / f64::from(TERMINAL_STEP_QUANTA);
        for multiplier in 1..4_u32 {
            let boundary = f64::from(multiplier) * quantum;
            let expected = boundary;
            assert_eq!(
                quantized_terminal_step(clearance, boundary - 2.0 * GEOMETRY_EPS_MM),
                expected
            );
            assert_eq!(quantized_terminal_step(clearance, boundary), expected);
            assert_eq!(
                quantized_terminal_step(clearance, boundary + GEOMETRY_EPS_MM / 2.0),
                expected
            );
            assert_eq!(
                quantized_terminal_step(clearance, boundary + 2.0 * GEOMETRY_EPS_MM),
                f64::from(multiplier + 1) * quantum
            );
        }
        assert_eq!(quantized_terminal_step(clearance, f64::INFINITY), clearance);
        assert_eq!(quantized_terminal_step(clearance, f64::NAN), clearance);
        assert_eq!(quantized_terminal_step(0.0, 0.1), 0.0);
        assert_eq!(quantized_terminal_step(-0.1, 0.1), 0.0);
        assert_eq!(quantized_terminal_step(f64::NAN, 0.1), 0.0);
    }

    #[test]
    fn terminal_step_uses_worst_foreign_overlapping_feature_only() {
        let mut board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Signal,
                LayerKind::Signal,
                LayerKind::Signal,
            ],
            segments: vec![
                // Foreign overlapping-layer segment: 0.10 gap => 0.10 deficit.
                Segment {
                    net: "segment".into(),
                    layer: 0,
                    a: (-1.0, 0.25),
                    b: (1.0, 0.25),
                    width: 0.1,
                },
                // Much worse, but same-net and therefore excluded.
                Segment {
                    net: "target".into(),
                    layer: 0,
                    a: (-1.0, 0.01),
                    b: (1.0, 0.01),
                    width: 0.1,
                },
                // Foreign but outside the target via's 0..1 span.
                Segment {
                    net: "other-layer".into(),
                    layer: 3,
                    a: (-1.0, 0.01),
                    b: (1.0, 0.01),
                    width: 0.1,
                },
            ],
            pads: vec![Pad {
                // 0.08 gap => 0.12 deficit, the maximum included feature.
                net: Some("pad".into()),
                layer: 1,
                center: (0.23, 0.0),
                width: 0.1,
                height: 0.1,
            }],
            vias: vec![
                Via {
                    net: "target".into(),
                    center: (0.0, 0.0),
                    pad_diameter: 0.2,
                    drill_diameter: 0.1,
                    from_layer: 0,
                    to_layer: 1,
                    antipad_radius: None,
                },
                Via {
                    net: "nonoverlap".into(),
                    center: (0.01, 0.0),
                    pad_diameter: 0.2,
                    drill_diameter: 0.1,
                    from_layer: 2,
                    to_layer: 3,
                    antipad_radius: None,
                },
            ],
            rules: DrcRules {
                clearance: 0.2,
                plane_antipad: 0.2,
                min_annular_ring: 0.0,
            },
        };
        assert_eq!(terminal_repair_step(&board, 0), 3.0 * (0.2 / 4.0));
        assert_eq!(
            repair_step(
                &board,
                MovableVia {
                    drc_via_index: 0,
                    trace_index: 0,
                    point_index: 2,
                    kind: ViaMoveKind::InteriorRigid,
                },
            ),
            0.2,
            "the accepted interior radius remains exactly one clearance"
        );

        // An overlapping foreign via dominates and clamps an overlap deficit to 1x.
        board.vias.push(Via {
            net: "overlap".into(),
            center: (0.01, 0.0),
            pad_diameter: 0.2,
            drill_diameter: 0.1,
            from_layer: 1,
            to_layer: 2,
            antipad_radius: None,
        });
        assert_eq!(terminal_repair_step(&board, 0), 0.2);

        // With only the excluded features, no phantom deficit is introduced.
        board.segments.remove(0);
        board.pads.clear();
        board.vias.pop();
        assert_eq!(terminal_repair_step(&board, 0), 0.05);
    }

    #[test]
    fn exact_repair_cleans_source_and_destination_terminal_vias_end_to_end() {
        let srj = open_srj();
        let rules = DrcRules {
            clearance: 0.15,
            plane_antipad: 0.2,
            min_annular_ring: 0.1,
        };
        for target in [
            PcbTrace::new(vec![
                wire(0.0, 0.34, "top"),
                via(0.0, 0.34),
                wire(1.0, 0.34, "bottom"),
            ])
            .with_net("target"),
            PcbTrace::new(vec![
                wire(-2.0, 0.34, "top"),
                wire(0.0, 0.34, "top"),
                via(0.0, 0.34),
            ])
            .with_net("target"),
        ] {
            let traces = vec![
                target,
                PcbTrace::new(vec![wire(-2.0, 0.0, "bottom"), wire(2.0, 0.0, "bottom")])
                    .with_net("foreign"),
            ];
            let before_board =
                drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count);
            let before = before_board.check();
            assert!(!before.is_empty());
            let signature = topology_signature(&traces);
            let labels = drc_board::reconstruct_net_labels(&srj, &traces, srj.layer_count);

            let repaired = repair_clearance_vias(&srj, traces, rules, srj.layer_count);
            let after =
                drc_board::solution_to_drc_board(&srj, &repaired, rules, srj.layer_count).check();
            assert!(
                after.is_empty(),
                "terminal repair left findings: {after:#?}"
            );
            assert!(after.len() < before.len());
            assert_eq!(topology_signature(&repaired), signature);
            assert_eq!(
                drc_board::reconstruct_net_labels(&srj, &repaired, srj.layer_count),
                labels
            );
        }
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
