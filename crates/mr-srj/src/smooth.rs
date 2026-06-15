//! Trace beautifier (Phase 1): turn the router's Manhattan output into
//! hand-routed-looking geometry — straight 45° diagonals instead of staircases
//! and chamfered corners instead of square 90° turns — **without** ever
//! introducing a clearance violation or changing connectivity.
//!
//! ## Why this exists
//!
//! The router walks a strictly 4-connected grid ([`mr_core::Dims::neighbors4`]),
//! so every turn is 90° and every diagonal run degrades into a unit-step
//! staircase. [`crate::to_solution_layered`] then emits one
//! [`RoutePoint::Wire`] vertex per grid cell verbatim. This pass rewrites that
//! geometry after the fact.
//!
//! ## Safety invariant
//!
//! Each candidate edge (a corner-cut shortcut or a 45° chamfer) is validated
//! with the **same exact-distance engine the DRC checker uses**
//! ([`mr_drc::seg_seg_dist`] et al.). A candidate is accepted only if, for every
//! nearby foreign feature, it is either above the required clearance **or no
//! closer than the original geometry already was** (the *monotone non-worsening*
//! rule). This makes the pass incapable of creating or worsening a violation —
//! worst case it leaves a corner square. Trace endpoints (port-snapped) and via
//! landings are fixed anchors and are never moved, so connectivity is preserved
//! and the via count is unchanged.

use std::collections::HashMap;

use mr_drc::{point_seg_dist, seg_rect_gap, seg_seg_dist};

use crate::{Obstacle, PcbTrace, RoutePoint};

/// A 2-D point in continuous coordinates.
type P = (f64, f64);

/// Fraction of the shorter adjacent segment used as the chamfer depth. 0.5 cuts
/// from midpoint to midpoint (maximal 45° chamfer); the cut is additionally
/// capped by [`MAX_CHAMFER_MM`]. On a unit-step staircase this bevels every step
/// to its midpoints, and the resulting equal-slope segments then collinear-merge
/// into a single straight diagonal; on a long L it leaves `straight → 45° → straight`.
const CHAMFER_FRAC: f64 = 0.5;
/// Hard cap (continuous units, ~mm) on a single chamfer's depth, so a long
/// segment is not bevelled into oblivion.
const MAX_CHAMFER_MM: f64 = 1.0;
/// Cross-product magnitude below which three points count as collinear.
const COLLINEAR_EPS: f64 = 1e-9;
/// Numerical slack when comparing a candidate gap to the original gap.
const GAP_EPS: f64 = 1e-9;
/// Via pad radius used for clearance against vias (PcbTrace vias carry no geometry
/// of their own). Matches the real signal-via annular pad: `VIA_PAD_MM / 2.0`
/// (`mr_cli::VIA_PAD_MM == 0.45`, not visible from this crate — keep in sync).
const VIA_RADIUS: f64 = 0.225;

/// A foreign feature a candidate trace edge must stay clear of. The stored gap
/// helpers return the distance from a segment *centre-line* to the feature's
/// *surface*, so the required threshold is uniformly `clearance + half_width`.
enum Feature {
    /// Another trace's wire segment, modelled as a capsule of half-width `half_w`.
    Seg { a: P, b: P, half_w: f64, trace: usize },
    /// A pad / keepout rectangle (axis-aligned, full width/height).
    Rect { c: P, w: f64, h: f64 },
    /// A via barrel, modelled as a circle of radius `r`.
    Circle { c: P, r: f64, trace: usize },
}

impl Feature {
    /// Axis-aligned bounding box `(min_x, min_y, max_x, max_y)` for broad-phase.
    fn bbox(&self) -> (f64, f64, f64, f64) {
        match self {
            Feature::Seg { a, b, half_w, .. } => (
                a.0.min(b.0) - half_w,
                a.1.min(b.1) - half_w,
                a.0.max(b.0) + half_w,
                a.1.max(b.1) + half_w,
            ),
            Feature::Rect { c, w, h, .. } => {
                (c.0 - w / 2.0, c.1 - h / 2.0, c.0 + w / 2.0, c.1 + h / 2.0)
            }
            Feature::Circle { c, r, .. } => (c.0 - r, c.1 - r, c.0 + r, c.1 + r),
        }
    }

    /// Originating trace index, or `None` for board-fixed features (pads).
    fn trace(&self) -> Option<usize> {
        match self {
            Feature::Seg { trace, .. } | Feature::Circle { trace, .. } => Some(*trace),
            Feature::Rect { .. } => None,
        }
    }

    /// Gap from segment `[s0,s1]`'s centre-line to this feature's surface
    /// (negative when overlapping). Compared against `clearance + half_w_self`.
    fn gap(&self, s0: P, s1: P) -> f64 {
        match self {
            Feature::Seg { a, b, half_w, .. } => seg_seg_dist(s0, s1, *a, *b) - half_w,
            Feature::Rect { c, w, h } => seg_rect_gap(s0, s1, *c, *w, *h),
            Feature::Circle { c, r, .. } => point_seg_dist(*c, s0, s1) - r,
        }
    }
}

/// One contiguous single-layer run of wire vertices, bounded by trace endpoints
/// and/or via landings. The first and last points are immovable anchors.
struct Run {
    layer: String,
    width: f64,
    pts: Vec<P>,
}

/// An element of a parsed trace, in original order.
enum Item {
    Run(Run),
    Via(RoutePoint),
}

/// Beautify a whole solution soup: pull staircases taut into diagonals and
/// chamfer square corners, validating every new edge against all other copper
/// and pads. Endpoints, vias, and connectivity are preserved.
///
/// `obstacles` are the problem's pads/keepouts, `clearance` the minimum
/// copper-to-copper spacing (0 if the problem declares none).
pub fn beautify_traces(
    traces: Vec<PcbTrace>,
    obstacles: &[Obstacle],
    clearance: f64,
) -> Vec<PcbTrace> {
    // Build the static feature context ONCE from the *original* geometry, so the
    // result is independent of the order traces are processed and so a beautified
    // (inset) trace is always validated against a conservative original footprint.
    let by_layer = features_by_layer(&traces, obstacles);

    traces
        .into_iter()
        .enumerate()
        .map(|(ti, trace)| beautify_one(ti, trace, &by_layer, clearance))
        .collect()
}

/// Beautify a single trace against the precomputed feature context.
fn beautify_one(
    ti: usize,
    trace: PcbTrace,
    by_layer: &LayerFeatures,
    clearance: f64,
) -> PcbTrace {
    let mut items = parse_items(&trace);
    for item in &mut items {
        if let Item::Run(run) = item {
            beautify_run(ti, run, by_layer, clearance);
        }
    }
    PcbTrace::new(flatten_items(items))
}

/// Split a trace's route into single-layer wire runs separated by via anchors.
fn parse_items(trace: &PcbTrace) -> Vec<Item> {
    let mut items = Vec::new();
    let mut cur: Option<Run> = None;
    for rp in &trace.route {
        match rp {
            RoutePoint::Wire {
                x,
                y,
                width,
                layer,
            } => {
                // A layer change without an intervening via still closes the run.
                if let Some(run) = &cur {
                    if &run.layer != layer {
                        items.push(Item::Run(cur.take().unwrap()));
                    }
                }
                let run = cur.get_or_insert_with(|| Run {
                    layer: layer.clone(),
                    width: *width,
                    pts: Vec::new(),
                });
                run.pts.push((*x, *y));
            }
            via @ RoutePoint::Via { .. } => {
                if let Some(run) = cur.take() {
                    items.push(Item::Run(run));
                }
                items.push(Item::Via(via.clone()));
            }
        }
    }
    if let Some(run) = cur.take() {
        items.push(Item::Run(run));
    }
    items
}

/// Re-emit parsed items as a flat route, restoring `Wire`/`Via` points in order.
fn flatten_items(items: Vec<Item>) -> Vec<RoutePoint> {
    let mut route = Vec::new();
    for item in items {
        match item {
            Item::Run(run) => {
                for (x, y) in run.pts {
                    route.push(RoutePoint::Wire {
                        x,
                        y,
                        width: run.width,
                        layer: run.layer.clone(),
                    });
                }
            }
            Item::Via(v) => route.push(v),
        }
    }
    route
}

/// In-place beautify of one run: merge collinear cell-vertices, run a single
/// DRC-guarded 45° chamfer sweep over the remaining corners, then collinear-merge
/// again so a chamfered staircase's equal-slope segments fuse into one diagonal.
fn beautify_run(ti: usize, run: &mut Run, by_layer: &LayerFeatures, clearance: f64) {
    if run.pts.len() < 3 {
        return; // no interior vertices to touch
    }
    let half_w = run.width / 2.0;
    let feats = by_layer.for_layer(ti, &run.layer);

    collapse_collinear(&mut run.pts);
    chamfer_corners(&mut run.pts, &feats, half_w, clearance);
    collapse_collinear(&mut run.pts);
}

/// Drop interior vertices that are collinear with their neighbours. Always safe
/// (geometry-preserving), so no DRC check is needed.
fn collapse_collinear(pts: &mut Vec<P>) {
    if pts.len() < 3 {
        return;
    }
    let mut out: Vec<P> = Vec::with_capacity(pts.len());
    out.push(pts[0]);
    for i in 1..pts.len() - 1 {
        let a = *out.last().unwrap();
        let b = pts[i];
        let c = pts[i + 1];
        let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if cross.abs() > COLLINEAR_EPS {
            out.push(b); // genuine corner — keep it
        }
    }
    out.push(*pts.last().unwrap());
    *pts = out;
}

/// Replace each remaining interior corner with a pair of 45° chamfer points,
/// shrinking the cut until it is DRC-legal (or skipping the corner entirely).
fn chamfer_corners(pts: &mut Vec<P>, feats: &[&Feature], half_w: f64, clearance: f64) {
    if pts.len() < 3 {
        return;
    }
    let req = clearance + half_w;
    let mut out: Vec<P> = Vec::with_capacity(pts.len() * 2);
    out.push(pts[0]);
    for i in 1..pts.len() - 1 {
        let a = pts[i - 1];
        let pi = pts[i];
        let b = pts[i + 1];
        let la = dist(a, pi);
        let lb = dist(pi, b);
        let base = (la.min(lb) * CHAMFER_FRAC).min(MAX_CHAMFER_MM);
        let orig = [(a, pi), (pi, b)];

        // Try progressively smaller chamfer depths.
        let mut applied = false;
        let mut d = base;
        for _ in 0..4 {
            if d <= GAP_EPS || la <= GAP_EPS || lb <= GAP_EPS {
                break;
            }
            let c1 = lerp(pi, a, d / la); // toward A
            let c2 = lerp(pi, b, d / lb); // toward B
            if accepted((c1, c2), &orig, feats, req) {
                out.push(c1);
                out.push(c2);
                applied = true;
                break;
            }
            d *= 0.5;
        }
        if !applied {
            out.push(pi); // leave the corner square
        }
    }
    out.push(*pts.last().unwrap());
    *pts = out;
}

/// Monotone non-worsening acceptance: a candidate segment is legal if, for every
/// nearby feature, it is at or above the required clearance **or** no closer than
/// the original local geometry already was. This can never introduce or worsen a
/// violation regardless of net ownership (a trace's own pad is touched by both
/// the original and the candidate, so it is accepted).
fn accepted(cand: (P, P), orig: &[(P, P)], feats: &[&Feature], req: f64) -> bool {
    let cb = seg_bbox(cand.0, cand.1, req);
    for f in feats {
        if !bbox_overlap(cb, f.bbox()) {
            continue; // too far to matter
        }
        let g_new = f.gap(cand.0, cand.1);
        if g_new >= req {
            continue;
        }
        let g_orig = orig
            .iter()
            .map(|(o0, o1)| f.gap(*o0, *o1))
            .fold(f64::INFINITY, f64::min);
        if g_new + GAP_EPS >= g_orig {
            continue; // not closer than we already were
        }
        return false;
    }
    true
}

/// Per-layer feature index plus the layer-agnostic features (vias, all-layer pads).
struct LayerFeatures {
    /// Wire-segment features grouped by their layer name.
    segs: HashMap<String, Vec<Feature>>,
    /// Pads tagged with the layers they sit on (empty == all layers).
    pads: Vec<(Vec<String>, Feature)>,
    /// Vias — treated as blocking on every layer (a barrel spans the stackup).
    vias: Vec<Feature>,
}

impl LayerFeatures {
    /// Borrowed feature list relevant to a candidate on `layer` from trace `ti`,
    /// excluding the trace's own copper (self-crossing is handled by the geometry
    /// staying within the original footprint).
    fn for_layer(&self, ti: usize, layer: &str) -> Vec<&Feature> {
        let mut v: Vec<&Feature> = Vec::new();
        if let Some(segs) = self.segs.get(layer) {
            v.extend(segs.iter().filter(|f| f.trace() != Some(ti)));
        }
        v.extend(self.vias.iter().filter(|f| f.trace() != Some(ti)));
        for (layers, pad) in &self.pads {
            if layers.is_empty() || layers.iter().any(|l| l == layer) {
                v.push(pad);
            }
        }
        v
    }
}

/// Build the static feature context from the original traces and the board pads.
fn features_by_layer(traces: &[PcbTrace], obstacles: &[Obstacle]) -> LayerFeatures {
    let mut segs: HashMap<String, Vec<Feature>> = HashMap::new();
    let mut vias: Vec<Feature> = Vec::new();

    for (ti, trace) in traces.iter().enumerate() {
        // Wire segments: consecutive same-layer wire vertices.
        let mut prev: Option<(P, f64, String)> = None;
        for rp in &trace.route {
            match rp {
                RoutePoint::Wire {
                    x,
                    y,
                    width,
                    layer,
                } => {
                    if let Some((pp, pw, pl)) = &prev {
                        if pl == layer {
                            segs.entry(layer.clone()).or_default().push(Feature::Seg {
                                a: *pp,
                                b: (*x, *y),
                                half_w: pw.max(*width) / 2.0,
                                trace: ti,
                            });
                        }
                    }
                    prev = Some(((*x, *y), *width, layer.clone()));
                }
                RoutePoint::Via { x, y, .. } => {
                    vias.push(Feature::Circle {
                        c: (*x, *y),
                        r: VIA_RADIUS,
                        trace: ti,
                    });
                    prev = None; // a via breaks the wire chain
                }
            }
        }
    }

    let pads = obstacles
        .iter()
        .map(|o| {
            (
                o.layers.clone(),
                Feature::Rect {
                    c: (o.center.x, o.center.y),
                    w: o.width,
                    h: o.height,
                },
            )
        })
        .collect();

    LayerFeatures { segs, pads, vias }
}

// --- small geometry helpers -------------------------------------------------

fn dist(a: P, b: P) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Point a fraction `t` of the way from `from` toward `to`.
fn lerp(from: P, to: P, t: f64) -> P {
    (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t)
}

/// Bounding box of segment `[a,b]` inflated by `pad`.
fn seg_bbox(a: P, b: P, pad: f64) -> (f64, f64, f64, f64) {
    (
        a.0.min(b.0) - pad,
        a.1.min(b.1) - pad,
        a.0.max(b.0) + pad,
        a.1.max(b.1) + pad,
    )
}

fn bbox_overlap(p: (f64, f64, f64, f64), q: (f64, f64, f64, f64)) -> bool {
    p.0 <= q.2 && q.0 <= p.2 && p.1 <= q.3 && q.1 <= p.3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;

    fn wire(x: f64, y: f64) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width: 0.1,
            layer: "top".to_string(),
        }
    }

    fn pts_of(t: &PcbTrace) -> Vec<P> {
        t.route
            .iter()
            .filter_map(|rp| match rp {
                RoutePoint::Wire { x, y, .. } => Some((*x, *y)),
                _ => None,
            })
            .collect()
    }

    /// Approximate equality for chamfered (fractional) coordinates.
    fn approx(a: P, b: P) -> bool {
        (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
    }

    /// True if any consecutive segment is a genuine 45° diagonal (|dx| ≈ |dy| > 0).
    fn has_diagonal(p: &[P]) -> bool {
        p.windows(2).any(|w| {
            let (dx, dy) = ((w[1].0 - w[0].0).abs(), (w[1].1 - w[0].1).abs());
            dx > 1e-6 && (dx - dy).abs() < 1e-6
        })
    }

    /// A clean unit staircase with no obstacles becomes a straight 45° diagonal
    /// (the chamfered equal-slope steps collinear-merge), with only short
    /// orthogonal stubs at the fixed endpoints. No 90° zig-zag survives.
    #[test]
    fn staircase_becomes_diagonal() {
        let route = vec![
            wire(0.0, 0.0),
            wire(1.0, 0.0),
            wire(1.0, 1.0),
            wire(2.0, 1.0),
            wire(2.0, 2.0),
            wire(3.0, 2.0),
            wire(3.0, 3.0),
        ];
        let out = beautify_traces(vec![PcbTrace::new(route)], &[], 0.0);
        let p = pts_of(&out[0]);
        assert!(approx(p[0], (0.0, 0.0)) && approx(*p.last().unwrap(), (3.0, 3.0)));
        assert!(p.len() < 7, "staircase should be simplified, got {p:?}");
        assert!(has_diagonal(&p), "result should contain a 45° diagonal: {p:?}");
        // The interior 90° step vertices are gone.
        for v in [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (3.0, 2.0)] {
            assert!(!p.iter().any(|q| approx(*q, v)), "step vertex {v:?} survived");
        }
    }

    /// An isolated 90° corner with room becomes `straight → 45° → straight`: the
    /// square corner is cut, endpoints preserved, and a diagonal segment appears.
    #[test]
    fn corner_gets_chamfered() {
        let route = vec![wire(0.0, 0.0), wire(2.0, 0.0), wire(2.0, 2.0)];
        let out = beautify_traces(vec![PcbTrace::new(route)], &[], 0.0);
        let p = pts_of(&out[0]);
        assert!(!p.iter().any(|q| approx(*q, (2.0, 0.0))), "square corner should be cut");
        assert!(approx(p[0], (0.0, 0.0)) && approx(*p.last().unwrap(), (2.0, 2.0)));
        assert!(has_diagonal(&p), "chamfer should introduce a 45° segment: {p:?}");
    }

    /// Endpoints and vias are immovable; via count is unchanged.
    #[test]
    fn endpoints_and_vias_preserved() {
        let route = vec![
            wire(0.0, 0.0),
            wire(1.0, 0.0),
            wire(1.0, 1.0),
            RoutePoint::Via {
                x: 1.0,
                y: 1.0,
                from_layer: "top".to_string(),
                to_layer: "bottom".to_string(),
            },
            RoutePoint::Wire {
                x: 1.0,
                y: 1.0,
                width: 0.1,
                layer: "bottom".to_string(),
            },
            RoutePoint::Wire {
                x: 2.0,
                y: 1.0,
                width: 0.1,
                layer: "bottom".to_string(),
            },
        ];
        let trace = PcbTrace::new(route);
        let first = trace.route.first().cloned();
        let last = trace.route.last().cloned();
        let out = beautify_traces(vec![trace], &[], 0.0);
        let vias = out[0]
            .route
            .iter()
            .filter(|rp| matches!(rp, RoutePoint::Via { .. }))
            .count();
        assert_eq!(vias, 1, "via count must be unchanged");
        assert_eq!(out[0].route.first().cloned(), first, "start anchor moved");
        assert_eq!(out[0].route.last().cloned(), last, "end anchor moved");
    }

    /// Safety invariant: a pad sitting on the maximal-chamfer line forces the cut
    /// to shrink (or be skipped) so the result never gets closer to the pad than
    /// the clearance. The original L-legs are clear of the pad; we assert every
    /// emitted segment keeps `>= clearance` from it.
    #[test]
    fn chamfer_never_violates_pad_clearance() {
        let clearance = 0.2;
        let route = vec![wire(0.0, 0.0), wire(2.0, 0.0), wire(2.0, 2.0)];
        // Pad centred at (1.5,0.5) — exactly on the full chamfer (1,0)->(2,1), so
        // the maximal cut would slice through it and must be shrunk. The axis legs
        // (y=0 and x=2) sit 0.5 away, comfortably clear.
        let pad = Obstacle {
            kind: "rect".to_string(),
            center: Point { x: 1.5, y: 0.5, layer: None },
            width: 0.2,
            height: 0.2,
            layers: vec!["top".to_string()],
            connected_to: vec![],
        };
        let out = beautify_traces(vec![PcbTrace::new(route)], &[pad], clearance);
        let p = pts_of(&out[0]);
        let half_w = 0.05; // wire width 0.1 / 2
        for w in p.windows(2) {
            let gap = seg_rect_gap(w[0], w[1], (1.5, 0.5), 0.2, 0.2);
            assert!(
                gap + 1e-9 >= clearance + half_w,
                "segment {:?}->{:?} gap {gap} violates clearance",
                w[0],
                w[1]
            );
        }
    }
}
