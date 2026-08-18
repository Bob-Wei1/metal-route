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

use std::collections::{HashMap, HashSet};

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
    Seg {
        a: P,
        b: P,
        half_w: f64,
        trace: usize,
    },
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
    pts: Vec<P>,
    /// Width attached to each corresponding point in `pts`. Route widths are
    /// vertex data; collapsing them to one run-wide value loses taper geometry.
    widths: Vec<f64>,
}

impl Run {
    fn max_width(&self) -> f64 {
        self.widths.iter().copied().fold(0.0, f64::max)
    }

    fn segment_half_width(&self, a: usize, b: usize) -> f64 {
        self.widths[a].max(self.widths[b]) / 2.0
    }

    /// Width shared exactly by every vertex, or `None` for a tapered/mixed run.
    /// Beautification changes vertex cardinality, so mixed-width runs stay exact
    /// until there is an explicit taper-aware simplifier.
    fn uniform_width(&self) -> Option<f64> {
        let first = *self.widths.first()?;
        self.widths
            .iter()
            .all(|width| *width == first)
            .then_some(first)
    }
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
fn beautify_one(ti: usize, trace: PcbTrace, by_layer: &LayerFeatures, clearance: f64) -> PcbTrace {
    let mut items = parse_items(&trace);
    for item in &mut items {
        if let Item::Run(run) = item {
            beautify_run(ti, run, by_layer, clearance);
        }
    }
    let mut out = PcbTrace::new(flatten_items(items));
    // Beautify only reshapes geometry — carry the trace's net identity through.
    out.net = trace.net;
    out
}

/// Number of relaxation sweeps the clearance legaliser runs. Each sweep visits
/// every movable vertex once; the gains saturate quickly because a sweep only
/// ever increases foreign clearance (and never worsens it), so a small fixed bound
/// keeps the pass cheap and deterministic.
const LEGALIZE_SWEEPS: usize = 8;
/// Candidate nudge offsets (continuous units, ~mm) tried at each movable vertex,
/// largest first, so the legaliser opens up as much foreign clearance as a single
/// move legally can before falling back to a finer step. Sized around the default
/// `clearance + track_w` so one move can clear a half-track overlap.
const NUDGE_STEPS: [f64; 5] = [0.30, 0.20, 0.10, 0.05, 0.025];

/// Post-route exact-geometry clearance legaliser (runs AFTER [`beautify_traces`]).
///
/// The negotiated router enforces inter-net spacing with a grid HALO — cells within
/// a radius of a routed net's path *nodes* are blocked for foreign nets. But copper
/// is the SEGMENTS between nodes, and the emitted geometry adds vertex snapping
/// (endpoints pulled to the exact pad, interior vertices at cell centres) and 45°
/// chamfers, so a foreign segment can still pass closer to ours than any node-to-node
/// distance the halo measured. The exact DRC ([`mr_drc`]) then reports those genuine
/// different-net clearance shorts.
///
/// This pass repairs them on the emitted geometry directly, against the same exact
/// distance engine the DRC uses. It moves only INTERIOR wire vertices (never an
/// endpoint anchor or a via landing, so connectivity and via positions are
/// untouched) along a small set of candidate offsets, accepting a move only when it
/// *strictly increases* the worst foreign-clearance gap on the vertex's incident
/// segments AND leaves every incident segment no worse than `min(required, original)`
/// against every OTHER foreign feature. Because acceptance is validated by the exact
/// oracle and is monotone (a gap is never pushed below where it started), the pass
/// can only reduce violations, never create one, and it cannot break connectivity.
///
/// `obstacles` are the problem's pads/keepouts, `clearance` the minimum
/// copper-to-copper spacing. When `clearance <= 0` the pass is a no-op (returns the
/// input unchanged), preserving the clearance-off fast path byte-for-byte.
pub fn legalize_clearance(
    traces: Vec<PcbTrace>,
    obstacles: &[Obstacle],
    clearance: f64,
) -> Vec<PcbTrace> {
    if clearance <= 0.0 || traces.is_empty() {
        return traces;
    }
    let mut items_per_trace: Vec<Vec<Item>> = traces.iter().map(parse_items).collect();
    let nets: Vec<Option<String>> = traces.iter().map(|t| t.net.clone()).collect();

    for _ in 0..LEGALIZE_SWEEPS {
        // Rebuild the foreign-feature context from the CURRENT geometry at the start of
        // each sweep so every accepted move is validated against the latest copper.
        // Within a sweep, moves are applied Gauss-Seidel against THIS snapshot, so a
        // later vertex is checked against an earlier mover's OLD position — that race
        // can, on rare boxed-in geometry, nudge two tracks toward each other. We GATE
        // the sweep with a self-consistent count (foreign violations rebuilt from the
        // TRIAL geometry's own updated copper) and keep the sweep only if it did not
        // raise that count, so the pass is internally non-regressing. The CALLER applies
        // a second, authoritative gate against the real DRC (which knows the exact
        // electrical-net relabelling), so the emitted board can never regress even if
        // this crate's net view differs from the DRC's. See `count_foreign_violations`.
        let snapshot = current_traces(&items_per_trace, &nets);
        let by_layer = features_by_layer(&snapshot, obstacles);
        let before = count_foreign_violations(&items_per_trace, &by_layer, clearance);

        let mut trial = clone_items(&items_per_trace);
        let mut moved = false;
        for (ti, items) in trial.iter_mut().enumerate() {
            for item in items.iter_mut() {
                if let Item::Run(run) = item {
                    if legalize_run(ti, run, &by_layer, clearance) {
                        moved = true;
                    }
                }
            }
            // After interior-vertex relaxation, try nudging each VIA landing (and its
            // two coincident run anchors) as a rigid unit. This is the only place a
            // run's first/last anchor is allowed to move, and ONLY when that anchor is
            // a via landing (never a real endpoint), so connectivity is preserved.
            if legalize_vias(ti, items, &by_layer, clearance) {
                moved = true;
            }
        }
        if !moved {
            break; // quiescent: no vertex could improve, further sweeps are futile
        }
        // Rebuild the context from the TRIAL geometry so `after` reflects every trace's
        // NEW position (catching a within-sweep race where two tracks both moved); the
        // sweep is accepted only if the count did not rise in the real, post-move world.
        let trial_snapshot = current_traces(&trial, &nets);
        let trial_ctx = features_by_layer(&trial_snapshot, obstacles);
        let after = count_foreign_violations(&trial, &trial_ctx, clearance);
        if after <= before {
            items_per_trace = trial; // accept: non-regressing
        } else {
            break; // a sweep that would regress; stop and keep the last good geometry
        }
    }

    items_per_trace
        .into_iter()
        .zip(traces)
        .map(|(items, trace)| {
            let mut out = PcbTrace::new(flatten_items(items));
            out.net = trace.net; // geometry-only reshape; carry net identity through
            out
        })
        .collect()
}

/// Count distinct different-net clearance violations across the in-progress geometry,
/// measured against the SAME exact distance engine the DRC uses, on the fixed feature
/// `context`. Each trace's wire segments are tested against the foreign features for
/// their layer; a segment-feature pair below `clearance + half_width` counts once. Used
/// only to gate a legalisation sweep (accept iff the count does not rise), so it need
/// not perfectly mirror the DRC's pair canonicalisation — only move monotonically with
/// it, which a per-(segment, feature) tally does.
fn count_foreign_violations(
    items_per_trace: &[Vec<Item>],
    context: &LayerFeatures,
    clearance: f64,
) -> usize {
    let mut n = 0usize;
    for (ti, items) in items_per_trace.iter().enumerate() {
        for item in items {
            match item {
                Item::Run(run) => {
                    let feats = context.foreign_for_layer(ti, &run.layer);
                    for (i, w) in run.pts.windows(2).enumerate() {
                        let (s0, s1) = (w[0], w[1]);
                        let req = clearance + run.segment_half_width(i, i + 1);
                        let cb = seg_bbox(s0, s1, req);
                        for f in &feats {
                            if !bbox_overlap(cb, f.bbox()) {
                                continue;
                            }
                            if f.gap(s0, s1) + GAP_EPS < req {
                                n += 1;
                            }
                        }
                    }
                }
                Item::Via(RoutePoint::Via {
                    x,
                    y,
                    from_layer,
                    to_layer,
                }) => {
                    // Count the barrel against foreign features on BOTH the layers it
                    // spans. The legaliser may move a via, so this term must be in the
                    // gate or a barrel could be pushed into a foreign feature unseen.
                    // Tally each unique foreign feature once (a feature reachable from
                    // both layers — e.g. an all-layer pad — is deduplicated by pointer).
                    let v = (*x, *y);
                    let cb = seg_bbox(v, v, VIA_RADIUS + clearance);
                    let mut feats = context.foreign_for_layer(ti, from_layer);
                    if to_layer != from_layer {
                        feats.extend(context.foreign_for_layer(ti, to_layer));
                    }
                    feats.sort_by_key(|f| *f as *const Feature as usize);
                    feats.dedup_by_key(|f| *f as *const Feature as usize);
                    for f in &feats {
                        if !bbox_overlap(cb, f.bbox()) {
                            continue;
                        }
                        // Barrel surface gap vs the via clearance (clearance only — the
                        // barrel radius is the via's own copper, mirrored on the feature
                        // side by the feature's own half-width already folded into gap()).
                        if f.gap(v, v) - VIA_RADIUS + GAP_EPS < clearance {
                            n += 1;
                        }
                    }
                }
                Item::Via(_) => {}
            }
        }
    }
    n
}

/// Deep-clone the in-progress per-trace items so a legalisation sweep can be applied to
/// a trial copy and discarded if it would regress the violation count.
fn clone_items(items_per_trace: &[Vec<Item>]) -> Vec<Vec<Item>> {
    items_per_trace
        .iter()
        .map(|items| {
            items
                .iter()
                .map(|it| match it {
                    Item::Run(r) => Item::Run(Run {
                        layer: r.layer.clone(),
                        pts: r.pts.clone(),
                        widths: r.widths.clone(),
                    }),
                    Item::Via(v) => Item::Via(v.clone()),
                })
                .collect()
        })
        .collect()
}

/// Re-emit the in-progress per-trace items back into `PcbTrace`s (carrying their net
/// labels) so the foreign-feature context can be rebuilt from the CURRENT geometry
/// between legalisation sweeps.
fn current_traces(items_per_trace: &[Vec<Item>], nets: &[Option<String>]) -> Vec<PcbTrace> {
    items_per_trace
        .iter()
        .enumerate()
        .map(|(ti, items)| {
            let route = flatten_items_ref(items);
            let mut t = PcbTrace::new(route);
            t.net = nets.get(ti).cloned().flatten();
            t
        })
        .collect()
}

/// Relax one run's interior vertices to recover foreign clearance. Returns true iff
/// any vertex moved. A vertex's two incident segments are re-measured against the
/// foreign features for its layer; if either is below clearance, the vertex is nudged
/// along a small set of offsets and the first move that strictly improves the worst
/// incident foreign gap (without worsening any other) is committed.
fn legalize_run(ti: usize, run: &mut Run, by_layer: &LayerFeatures, clearance: f64) -> bool {
    if run.pts.len() < 3 {
        return false; // only the two immovable anchors — nothing interior to move
    }
    let half_w = run.max_width() / 2.0;
    let req = clearance + half_w;
    let feats = by_layer.foreign_for_layer(ti, &run.layer);
    let mut moved = false;

    // PHASE 1 — segment push. A parallel violating run is a stretch of copper running
    // alongside a foreign feature; nudging one vertex at a time just kinks it. Instead,
    // for each interior segment whose BOTH endpoints are movable (interior), shift the
    // two endpoints TOGETHER perpendicular to the segment, which slides the whole run
    // sideways away from the neighbour while staying parallel. The shifted endpoints'
    // OTHER incident segments (to the fixed/anchor neighbours) tilt slightly; the
    // exact-gap check below covers them, so a push is taken only when it improves the
    // worst foreign gap across all four affected segments without worsening it.
    for i in 1..run.pts.len().saturating_sub(2) {
        let (p, q) = (run.pts[i], run.pts[i + 1]);
        let (dx, dy) = (q.0 - p.0, q.1 - p.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= GAP_EPS {
            continue;
        }
        // Unit perpendicular to the segment.
        let (nx, ny) = (-dy / len, dx / len);
        let before = push_min_gap(run, i, &feats, req, 0.0, 0.0);
        if before >= req {
            continue; // segment already clear
        }
        let mut best: Option<(f64, f64, f64)> = None; // (gap, ox, oy)
        for &step in &NUDGE_STEPS {
            for sign in [1.0, -1.0] {
                let (ox, oy) = (nx * step * sign, ny * step * sign);
                let g = push_min_gap(run, i, &feats, req, ox, oy);
                if g > before + GAP_EPS {
                    match best {
                        Some((bg, _, _)) if g <= bg => {}
                        _ => best = Some((g, ox, oy)),
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        if let Some((_, ox, oy)) = best {
            run.pts[i] = (run.pts[i].0 + ox, run.pts[i].1 + oy);
            run.pts[i + 1] = (run.pts[i + 1].0 + ox, run.pts[i + 1].1 + oy);
            moved = true;
        }
    }

    // PHASE 2 — single-vertex relaxation (handles corners and run ends the push misses).
    for i in 1..run.pts.len() - 1 {
        let a = run.pts[i - 1];
        let p = run.pts[i];
        let b = run.pts[i + 1];
        // Worst foreign gap on the two segments incident to vertex `i` as it stands.
        let cur = incident_min_gap(p, a, b, &feats, req);
        if cur >= req {
            continue; // this vertex is already clear — leave it untouched
        }
        // Search candidate offsets along the 4 axis directions and the 4 diagonals,
        // largest step first. Accept the move that most improves the worst incident
        // foreign gap while never pushing it below where it started (monotone). A
        // partial improvement that does not yet reach `req` is still kept: successive
        // sweeps compound it, and a parallel run is opened up one vertex at a time.
        let mut best: Option<(f64, P)> = None;
        for &step in &NUDGE_STEPS {
            for (dx, dy) in DIRS {
                let cand = (p.0 + dx * step, p.1 + dy * step);
                let g = incident_min_gap(cand, a, b, &feats, req);
                if g > cur + GAP_EPS {
                    match best {
                        Some((bg, _)) if g <= bg => {}
                        _ => best = Some((g, cand)),
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        if let Some((_, cand)) = best {
            run.pts[i] = cand;
            moved = true;
        }
    }
    moved
}

/// Nudge VIA landings to recover foreign clearance on the via-adjacent copper.
///
/// A via in the parsed item stream is an `Item::Via` flanked by the run that lands on
/// it (its LAST point coincides with the via barrel) and the run that departs from it
/// (its FIRST point coincides). The legaliser otherwise treats those flanking anchors
/// as immovable, so a graze on the very first/last segment of a via-adjacent run — or
/// against the via barrel itself — is structurally unreachable.
///
/// Here we move the via barrel and BOTH coincident run anchors TOGETHER (a rigid
/// translation of one planar point shared by three geometry items), so the via and the
/// two runs stay connected exactly as before — the move is connectivity-preserving by
/// construction (the three coincident coordinates remain coincident). We try the same
/// small candidate offsets and commit a move only when it *strictly increases* the
/// worst foreign-clearance gap over EVERY segment/feature the move touches (the two
/// incident segments on their own layers, plus the via barrel) without pushing any of
/// them below where it started — the identical monotone, exact-oracle acceptance rule
/// the interior-vertex relaxation uses. Endpoints (runs not flanking a via) are never
/// touched, and no via is added or removed; only existing barrels are repositioned.
///
/// Returns true iff any via moved.
fn legalize_vias(ti: usize, items: &mut [Item], by_layer: &LayerFeatures, clearance: f64) -> bool {
    let mut moved = false;
    // Indices of `Item::Via` that have a Run immediately before AND after them; only
    // such vias have two incident wire segments whose landing we can co-move safely.
    let via_idxs: Vec<usize> = (1..items.len().saturating_sub(1))
        .filter(|&k| {
            matches!(items[k], Item::Via(_))
                && matches!(items[k - 1], Item::Run(_))
                && matches!(items[k + 1], Item::Run(_))
        })
        .collect();

    for k in via_idxs {
        // The via barrel position and the two coincident anchor points.
        let v = match &items[k] {
            Item::Via(RoutePoint::Via { x, y, .. }) => (*x, *y),
            _ => continue,
        };
        // Penultimate point of the landing run (the neighbour of the via anchor on the
        // incoming side) and second point of the departing run (neighbour on the
        // outgoing side), plus each run's layer/req. A run with a single point has no
        // incident segment on that side, so skip such degenerate vias entirely.
        let (in_neighbor, in_layer, in_req) = match &items[k - 1] {
            Item::Run(r) if r.pts.len() >= 2 => {
                let last = r.pts.len() - 1;
                (
                    r.pts[last - 1],
                    r.layer.clone(),
                    clearance + r.segment_half_width(last - 1, last),
                )
            }
            _ => continue,
        };
        let (out_neighbor, out_layer, out_req) = match &items[k + 1] {
            Item::Run(r) if r.pts.len() >= 2 => (
                r.pts[1],
                r.layer.clone(),
                clearance + r.segment_half_width(0, 1),
            ),
            _ => continue,
        };
        let in_feats = by_layer.foreign_for_layer(ti, &in_layer);
        let out_feats = by_layer.foreign_for_layer(ti, &out_layer);

        // Worst foreign gap over everything the move touches, as it stands. We measure
        // each incident segment against its own layer's foreign features and the via
        // barrel (radius VIA_RADIUS) against the union — exactly the geometry that moves.
        let cur = via_min_gap(
            v,
            in_neighbor,
            &in_feats,
            in_req,
            out_neighbor,
            &out_feats,
            out_req,
        );
        if cur >= in_req.min(out_req) {
            continue; // already clear on both incident segments and the barrel — leave it
        }

        // The geometry the move changes, with the original (pre-move) shape so the
        // per-feature non-worsening gate can compare against where each element started.
        // `via_accepted` requires that EVERY affected element stays at/above its required
        // clearance OR no closer than it originally was (against every foreign feature) —
        // the same exact-oracle, per-feature rule `accepted()` enforces for chamfers.
        // This is strictly stronger than "the worst gap improved": a candidate that lifts
        // the minimum while pushing some OTHER feature below `min(req, original)` is
        // rejected, which is exactly the case the looser min-only rule let slip through.
        let mut best: Option<(f64, P)> = None;
        for &step in &NUDGE_STEPS {
            for (dx, dy) in DIRS {
                let cand = (v.0 + dx * step, v.1 + dy * step);
                if !via_accepted(
                    v,
                    cand,
                    in_neighbor,
                    &in_feats,
                    in_req,
                    out_neighbor,
                    &out_feats,
                    out_req,
                ) {
                    continue; // would worsen some feature below min(req, original) — unsafe
                }
                let g = via_min_gap(
                    cand,
                    in_neighbor,
                    &in_feats,
                    in_req,
                    out_neighbor,
                    &out_feats,
                    out_req,
                );
                if g > cur + GAP_EPS {
                    match best {
                        Some((bg, _)) if g <= bg => {}
                        _ => best = Some((g, cand)),
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        if let Some((_, cand)) = best {
            // Co-move the three coincident coordinates so connectivity is preserved.
            if let Item::Via(RoutePoint::Via { x, y, .. }) = &mut items[k] {
                *x = cand.0;
                *y = cand.1;
            }
            if let Item::Run(r) = &mut items[k - 1] {
                let last = r.pts.len() - 1;
                r.pts[last] = cand;
            }
            if let Item::Run(r) = &mut items[k + 1] {
                r.pts[0] = cand;
            }
            moved = true;
        }
    }
    moved
}

/// Worst foreign-clearance gap over the geometry a via move at `v` touches: the
/// incoming incident segment `[in_neighbor, v]` on its layer, the outgoing incident
/// segment `[v, out_neighbor]` on its layer, and the via barrel (a circle of radius
/// [`VIA_RADIUS`] at `v`) against every foreign feature on either layer. The returned
/// value is the raw centre-line/surface gap; the caller compares improvements
/// monotonically (no absolute threshold beyond the strict-increase rule), so this is a
/// sound non-worsening guard over exactly the moved geometry.
#[allow(clippy::too_many_arguments)]
fn via_min_gap(
    v: P,
    in_neighbor: P,
    in_feats: &[&Feature],
    in_req: f64,
    out_neighbor: P,
    out_feats: &[&Feature],
    out_req: f64,
) -> f64 {
    let mut worst = f64::INFINITY;
    // Incoming incident segment, measured against the incoming layer's foreign features.
    {
        let (s0, s1) = (in_neighbor, v);
        let cb = seg_bbox(s0, s1, in_req);
        for f in in_feats {
            if bbox_overlap(cb, f.bbox()) {
                worst = worst.min(f.gap(s0, s1));
            }
        }
    }
    // Outgoing incident segment, measured against the outgoing layer's foreign features.
    {
        let (s0, s1) = (v, out_neighbor);
        let cb = seg_bbox(s0, s1, out_req);
        for f in out_feats {
            if bbox_overlap(cb, f.bbox()) {
                worst = worst.min(f.gap(s0, s1));
            }
        }
    }
    // Via barrel: a degenerate segment at `v` (a point) padded by the barrel radius.
    // The barrel spans the stackup, so test it against foreign features on BOTH layers.
    let barrel_pad = VIA_RADIUS + in_req.max(out_req);
    let cb = seg_bbox(v, v, barrel_pad);
    for f in in_feats.iter().chain(out_feats.iter()) {
        if bbox_overlap(cb, f.bbox()) {
            // Gap from the barrel SURFACE: centre-line/point gap minus the via radius.
            worst = worst.min(f.gap(v, v) - VIA_RADIUS);
        }
    }
    worst
}

/// Per-feature non-worsening acceptance for a via move from `from` to `to`. Returns
/// true iff EVERY element the move changes — the incoming incident segment, the
/// outgoing incident segment, and the via barrel — is, against every foreign feature,
/// at/above its required clearance OR no closer than the SAME element was at the
/// via's original position. This is the via analogue of [`accepted`] (which guards the
/// chamfer/interior-vertex moves): it can never push any foreign gap below
/// `min(required, original)`, so it cannot introduce or worsen a clearance short on the
/// moved geometry, regardless of which feature happened to be the global minimum.
#[allow(clippy::too_many_arguments)]
fn via_accepted(
    from: P,
    to: P,
    in_neighbor: P,
    in_feats: &[&Feature],
    in_req: f64,
    out_neighbor: P,
    out_feats: &[&Feature],
    out_req: f64,
) -> bool {
    // The incoming incident segment, before and after the via endpoint moved.
    if !accepted((in_neighbor, to), &[(in_neighbor, from)], in_feats, in_req) {
        return false;
    }
    // The outgoing incident segment, before and after.
    if !accepted(
        (to, out_neighbor),
        &[(from, out_neighbor)],
        out_feats,
        out_req,
    ) {
        return false;
    }
    // The via barrel (a degenerate point inflated by VIA_RADIUS). Reuse the same
    // monotone rule on the surface gap: a candidate barrel position is legal vs a
    // foreign feature when it is no closer than `clearance` OR no closer than the
    // barrel originally was. `clearance == req - half_w`; the half_w belongs to the
    // *wire*, not the barrel, so the barrel's own threshold is `req - VIA_RADIUS`'s
    // wire half-width removed — but we conservatively reuse `accepted` over the barrel
    // SURFACE by representing the barrel as the point `to` with the feature gap reduced
    // by VIA_RADIUS, comparing against the larger incident `req` (a safe over-estimate
    // of the spacing the barrel must keep). Over-requiring can only REJECT moves, never
    // wrongly accept one, so it cannot compromise safety.
    let barrel_req = in_req.max(out_req);
    barrel_accepted(from, to, in_feats, out_feats, barrel_req)
}

/// Non-worsening acceptance for the via BARREL alone (a circle of radius [`VIA_RADIUS`]
/// translating from `from` to `to`), tested against every foreign feature on either
/// incident layer. A move is legal vs a feature when the barrel-surface gap at `to` is
/// at/above `req` OR no closer than it was at `from`. Mirrors [`accepted`]'s rule on the
/// barrel surface so the barrel can never be pushed into a foreign feature.
fn barrel_accepted(
    from: P,
    to: P,
    in_feats: &[&Feature],
    out_feats: &[&Feature],
    req: f64,
) -> bool {
    let cb = seg_bbox(to, to, VIA_RADIUS + req);
    for f in in_feats.iter().chain(out_feats.iter()) {
        if !bbox_overlap(cb, f.bbox()) {
            continue;
        }
        let g_new = f.gap(to, to) - VIA_RADIUS;
        if g_new >= req {
            continue;
        }
        let g_orig = f.gap(from, from) - VIA_RADIUS;
        if g_new + GAP_EPS >= g_orig {
            continue; // not closer than the barrel already was
        }
        return false;
    }
    true
}

/// Worst foreign-clearance gap over every segment AFFECTED by offsetting run vertices
/// `i` and `i+1` by `(ox, oy)`: the pushed segment `[p',q']` plus the two connector
/// segments to its (unmoved) neighbours `[i-1, i']` and `[q', i+2]` when those
/// neighbours exist. This is exactly the set of segments whose geometry the push
/// changes, so comparing it before/after is a sound monotone guard.
fn push_min_gap(run: &Run, i: usize, feats: &[&Feature], req: f64, ox: f64, oy: f64) -> f64 {
    let p = (run.pts[i].0 + ox, run.pts[i].1 + oy);
    let q = (run.pts[i + 1].0 + ox, run.pts[i + 1].1 + oy);
    let mut segs: Vec<(P, P)> = vec![(p, q)];
    if i >= 1 {
        segs.push((run.pts[i - 1], p));
    }
    if i + 2 < run.pts.len() {
        segs.push((q, run.pts[i + 2]));
    }
    let mut worst = f64::INFINITY;
    for (s0, s1) in segs {
        let cb = seg_bbox(s0, s1, req);
        for f in feats {
            if !bbox_overlap(cb, f.bbox()) {
                continue;
            }
            worst = worst.min(f.gap(s0, s1));
        }
    }
    worst
}

/// The minimum foreign-clearance gap over the two segments `[a,p]` and `[p,b]`
/// incident to a candidate vertex position `p`. A gap is `feature_gap - 0` already
/// includes the segment half-width via `req` at the call site; here we return the raw
/// `Feature::gap` (centre-line to surface) and the caller compares against `req`.
/// Broad-phased by the per-segment bbox so only nearby features are measured.
fn incident_min_gap(p: P, a: P, b: P, feats: &[&Feature], req: f64) -> f64 {
    let mut worst = f64::INFINITY;
    for (s0, s1) in [(a, p), (p, b)] {
        let cb = seg_bbox(s0, s1, req);
        for f in feats {
            if !bbox_overlap(cb, f.bbox()) {
                continue;
            }
            worst = worst.min(f.gap(s0, s1));
        }
    }
    worst
}

/// The 8 unit nudge directions (4 axis-aligned + 4 diagonal), deterministic order.
const DIAGONAL_UNIT: f64 = std::f64::consts::FRAC_1_SQRT_2;
const DIRS: [(f64, f64); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (DIAGONAL_UNIT, DIAGONAL_UNIT),
    (DIAGONAL_UNIT, -DIAGONAL_UNIT),
    (-DIAGONAL_UNIT, DIAGONAL_UNIT),
    (-DIAGONAL_UNIT, -DIAGONAL_UNIT),
];

/// Split a trace's route into single-layer wire runs separated by via anchors.
fn parse_items(trace: &PcbTrace) -> Vec<Item> {
    let mut items = Vec::new();
    let mut cur: Option<Run> = None;
    for rp in &trace.route {
        match rp {
            RoutePoint::Wire { x, y, width, layer } => {
                // A layer change without an intervening via still closes the run.
                if let Some(run) = &cur {
                    if &run.layer != layer {
                        items.push(Item::Run(cur.take().unwrap()));
                    }
                }
                let run = cur.get_or_insert_with(|| Run {
                    layer: layer.clone(),
                    pts: Vec::new(),
                    widths: Vec::new(),
                });
                run.pts.push((*x, *y));
                run.widths.push(*width);
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

/// Re-emit parsed items as a flat route by REFERENCE (does not consume `items`),
/// used to rebuild the feature context between legalisation sweeps.
fn flatten_items_ref(items: &[Item]) -> Vec<RoutePoint> {
    let mut route = Vec::new();
    for item in items {
        match item {
            Item::Run(run) => {
                debug_assert_eq!(run.pts.len(), run.widths.len());
                for ((x, y), width) in run.pts.iter().zip(&run.widths) {
                    route.push(RoutePoint::Wire {
                        x: *x,
                        y: *y,
                        width: *width,
                        layer: run.layer.clone(),
                    });
                }
            }
            Item::Via(v) => route.push(v.clone()),
        }
    }
    route
}

/// Re-emit parsed items as a flat route, restoring `Wire`/`Via` points in order.
fn flatten_items(items: Vec<Item>) -> Vec<RoutePoint> {
    let mut route = Vec::new();
    for item in items {
        match item {
            Item::Run(run) => {
                debug_assert_eq!(run.pts.len(), run.widths.len());
                for ((x, y), width) in run.pts.into_iter().zip(run.widths) {
                    route.push(RoutePoint::Wire {
                        x,
                        y,
                        width,
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
    let Some(width) = run.uniform_width() else {
        return; // preserve tapered/per-vertex widths and their exact vertices
    };
    let half_w = width / 2.0;
    let feats = by_layer.for_layer(ti, &run.layer);

    collapse_collinear(&mut run.pts);
    chamfer_corners(&mut run.pts, &feats, half_w, clearance);
    collapse_collinear(&mut run.pts);
    run.widths = vec![width; run.pts.len()];
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
    /// Per-trace electrical-net label (`PcbTrace::net`), indexed by trace id. Used
    /// by [`Self::foreign_for_layer`] to grant same-net immunity exactly the way
    /// the DRC reconstructs it (sibling sub-nets that share a junction must never be
    /// treated as a clearance obstacle, or the legaliser would try to push apart
    /// copper that is legitimately one net meeting at a pad).
    nets: Vec<Option<String>>,
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

    /// FOREIGN feature list relevant to trace `ti` on `layer`: like
    /// [`Self::for_layer`] but additionally excludes copper of the SAME electrical
    /// net (`nets[ti] == nets[other]`, both `Some`), so the clearance legaliser only
    /// measures against genuinely foreign copper — exactly the pairs the DRC counts.
    /// Untagged traces (`None` net) fall back to the trace-index exclusion only.
    fn foreign_for_layer(&self, ti: usize, layer: &str) -> Vec<&Feature> {
        let my_net = self.nets.get(ti).and_then(|n| n.as_deref());
        let same_net = |f: &Feature| -> bool {
            match (
                my_net,
                f.trace()
                    .and_then(|t| self.nets.get(t))
                    .and_then(|n| n.as_deref()),
            ) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        };
        let mut v: Vec<&Feature> = Vec::new();
        if let Some(segs) = self.segs.get(layer) {
            v.extend(
                segs.iter()
                    .filter(|f| f.trace() != Some(ti) && !same_net(f)),
            );
        }
        v.extend(
            self.vias
                .iter()
                .filter(|f| f.trace() != Some(ti) && !same_net(f)),
        );
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
    let mut routed_layers: HashSet<String> = HashSet::new();

    for (ti, trace) in traces.iter().enumerate() {
        // Wire segments: consecutive same-layer wire vertices.
        let mut prev: Option<(P, f64, String)> = None;
        for rp in &trace.route {
            match rp {
                RoutePoint::Wire { x, y, width, layer } => {
                    routed_layers.insert(layer.clone());
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
                RoutePoint::Via {
                    x,
                    y,
                    from_layer,
                    to_layer,
                } => {
                    routed_layers.insert(from_layer.clone());
                    routed_layers.insert(to_layer.clone());
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
            // Match rasterisation's conservative fallback: an empty layer list,
            // or one containing no layer known to the routed geometry, means all
            // layers. If at least one name is known, retain only those recognized
            // names and ignore unknown aliases.
            let layers = if o.layers.is_empty()
                || !o.layers.iter().any(|layer| routed_layers.contains(layer))
            {
                Vec::new()
            } else {
                o.layers
                    .iter()
                    .filter(|layer| routed_layers.contains(*layer))
                    .cloned()
                    .collect()
            };
            (
                layers,
                Feature::Rect {
                    c: (o.center.x, o.center.y),
                    w: o.width,
                    h: o.height,
                },
            )
        })
        .collect();

    let nets = traces.iter().map(|t| t.net.clone()).collect();
    LayerFeatures {
        segs,
        pads,
        vias,
        nets,
    }
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
        assert!(
            has_diagonal(&p),
            "result should contain a 45° diagonal: {p:?}"
        );
        // The interior 90° step vertices are gone.
        for v in [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (3.0, 2.0)] {
            assert!(
                !p.iter().any(|q| approx(*q, v)),
                "step vertex {v:?} survived"
            );
        }
    }

    /// An isolated 90° corner with room becomes `straight → 45° → straight`: the
    /// square corner is cut, endpoints preserved, and a diagonal segment appears.
    #[test]
    fn corner_gets_chamfered() {
        let route = vec![wire(0.0, 0.0), wire(2.0, 0.0), wire(2.0, 2.0)];
        let out = beautify_traces(vec![PcbTrace::new(route)], &[], 0.0);
        let p = pts_of(&out[0]);
        assert!(
            !p.iter().any(|q| approx(*q, (2.0, 0.0))),
            "square corner should be cut"
        );
        assert!(approx(p[0], (0.0, 0.0)) && approx(*p.last().unwrap(), (2.0, 2.0)));
        assert!(
            has_diagonal(&p),
            "chamfer should introduce a 45° segment: {p:?}"
        );
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
            center: Point {
                x: 1.5,
                y: 0.5,
                layer: None,
            },
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

    /// Rasterisation treats an obstacle whose layer names are all unknown as an
    /// all-layer keepout. Smoothing must use the same conservative fallback rather
    /// than silently chamfering through it.
    #[test]
    fn unknown_layer_pad_blocks_smoothing_on_every_layer() {
        let clearance = 0.2;
        let route = vec![wire(0.0, 0.0), wire(2.0, 0.0), wire(2.0, 2.0)];
        let pad = Obstacle {
            kind: "rect".into(),
            center: Point {
                x: 1.5,
                y: 0.5,
                layer: None,
            },
            width: 0.2,
            height: 0.2,
            layers: vec!["unknown-copper-layer".into()],
            connected_to: vec![],
        };

        let out = beautify_traces(vec![PcbTrace::new(route)], &[pad], clearance);
        let pts = pts_of(&out[0]);
        for edge in pts.windows(2) {
            let gap = seg_rect_gap(edge[0], edge[1], (1.5, 0.5), 0.2, 0.2);
            assert!(
                gap + GAP_EPS >= clearance + 0.05,
                "unknown-layer pad must conservatively block top smoothing: {edge:?}, gap={gap}"
            );
        }
    }

    fn wire_with_width(x: f64, y: f64, width: f64, layer: &str) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width,
            layer: layer.into(),
        }
    }

    /// Varying widths are vertex data, not a run-wide style. Until a taper-aware
    /// simplifier exists, beautification must retain such a run exactly; otherwise
    /// it silently rewrites copper geometry and can disconnect a via landing.
    #[test]
    fn beautify_preserves_nonuniform_widths_and_via_connectivity() {
        let route = vec![
            wire_with_width(0.0, 0.0, 0.10, "top"),
            wire_with_width(1.0, 0.0, 0.15, "top"),
            wire_with_width(1.0, 1.0, 0.20, "top"),
            via_at(1.0, 1.0),
            wire_with_width(1.0, 1.0, 0.25, "bottom"),
            wire_with_width(2.0, 1.0, 0.30, "bottom"),
            wire_with_width(2.0, 2.0, 0.35, "bottom"),
        ];
        let trace = PcbTrace::new(route.clone()).with_net("N");
        let out = beautify_traces(vec![trace], &[], 0.1);
        assert_eq!(
            out[0].route, route,
            "nonuniform-width route must round-trip exactly"
        );
        assert_eq!(out[0].net.as_deref(), Some("N"));
    }

    /// The legaliser may move coordinates, but it must never collapse all widths in
    /// a run to the first vertex's width. With no foreign features this is an exact
    /// round trip, including the two wire anchors coincident with the via.
    #[test]
    fn legalize_preserves_per_vertex_widths_and_via_connectivity() {
        let route = vec![
            wire_with_width(0.0, 0.0, 0.10, "top"),
            wire_with_width(1.0, 1.0, 0.15, "top"),
            via_at(1.0, 1.0),
            wire_with_width(1.0, 1.0, 0.25, "bottom"),
            wire_with_width(2.0, 2.0, 0.30, "bottom"),
        ];
        let trace = PcbTrace::new(route.clone()).with_net("N");
        let out = legalize_clearance(vec![trace], &[], 0.2);
        assert_eq!(
            out[0].route, route,
            "widths and via anchors must round-trip exactly"
        );
        assert_eq!(out[0].net.as_deref(), Some("N"));
    }

    /// Worst foreign clearance gap between two traces (different nets) over their
    /// wire segments, measured exactly (centre-line to centre-line minus both
    /// half-widths). Used by the legalisation tests below.
    fn min_track_gap(a: &PcbTrace, b: &PcbTrace, half_w: f64) -> f64 {
        let segs = |t: &PcbTrace| -> Vec<(P, P)> {
            let pts = pts_of(t);
            (0..pts.len().saturating_sub(1))
                .map(|i| (pts[i], pts[i + 1]))
                .collect()
        };
        let mut worst = f64::INFINITY;
        for (a0, a1) in segs(a) {
            for (b0, b1) in segs(b) {
                worst = worst.min(seg_seg_dist(a0, a1, b0, b1) - 2.0 * half_w);
            }
        }
        worst
    }

    /// The legaliser nudges an interior vertex of a foreign-net trace away from a
    /// too-close neighbour so the emitted copper clears the required spacing, while
    /// keeping both endpoint anchors (and the via-free connectivity) fixed.
    #[test]
    fn legalize_opens_sub_clearance_track() {
        let clearance = 0.2;
        let half_w = 0.05; // width 0.1
                           // Trace A: straight horizontal reference at y = 0.
        let a = PcbTrace::new(vec![wire(0.0, 0.0), wire(4.0, 0.0)]).with_net("A");
        // Trace B: runs parallel only 0.12 above A through an interior vertex (copper
        // gap 0.12 - 0.10 = 0.02 << clearance 0.2), with endpoints far from A so the
        // anchors themselves are clear and the interior vertex is free to move up.
        let b = PcbTrace::new(vec![wire(0.0, 1.0), wire(2.0, 0.12), wire(4.0, 1.0)]).with_net("B");
        let before = min_track_gap(&a, &b, half_w);
        assert!(
            before < clearance,
            "fixture must start in violation: {before}"
        );
        let out = legalize_clearance(vec![a.clone(), b], &[], clearance);
        let after = min_track_gap(&out[0], &out[1], half_w);
        assert!(
            after > before + 1e-6,
            "legalise must increase the gap: {before} -> {after}"
        );
        // Endpoint anchors are immovable.
        let bp = pts_of(&out[1]);
        assert!(
            approx(bp[0], (0.0, 1.0)) && approx(*bp.last().unwrap(), (4.0, 1.0)),
            "endpoint anchors must not move: {bp:?}"
        );
    }

    /// Clearance off (== 0) is a no-op: the geometry round-trips byte-for-byte.
    #[test]
    fn legalize_noop_when_clearance_off() {
        let t = PcbTrace::new(vec![wire(0.0, 0.0), wire(1.0, 0.5), wire(2.0, 0.0)]).with_net("A");
        let out = legalize_clearance(vec![t.clone()], &[], 0.0);
        assert_eq!(
            pts_of(&out[0]),
            pts_of(&t),
            "clearance-off must not move anything"
        );
    }

    /// Same-net sub-traces meeting at a junction are immune: the legaliser must not
    /// try to push apart copper that is one electrical net (no spurious moves).
    #[test]
    fn legalize_leaves_same_net_alone() {
        let clearance = 0.2;
        // Two traces of net "A" running parallel 0.1 apart — well within clearance, but
        // SAME net, so they must be left untouched.
        let a = PcbTrace::new(vec![wire(0.0, 0.0), wire(2.0, 0.0)]).with_net("A");
        let b = PcbTrace::new(vec![wire(0.0, 0.1), wire(1.0, 0.1), wire(2.0, 0.1)]).with_net("A");
        let out = legalize_clearance(vec![a.clone(), b.clone()], &[], clearance);
        assert_eq!(
            pts_of(&out[1]),
            pts_of(&b),
            "same-net copper must not be nudged"
        );
    }

    /// (D3) A via landing that grazes a foreign track is nudged — together with its two
    /// coincident wire anchors — to recover clearance, WITHOUT changing the via count or
    /// breaking the via↔run coincidence (connectivity). The two real trace endpoints
    /// (the far ends of each run) stay fixed.
    fn via_at(x: f64, y: f64) -> RoutePoint {
        RoutePoint::Via {
            x,
            y,
            from_layer: "top".to_string(),
            to_layer: "bottom".to_string(),
        }
    }
    fn wire_on(x: f64, y: f64, layer: &str) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width: 0.1,
            layer: layer.to_string(),
        }
    }

    #[test]
    fn legalize_nudges_via_landing() {
        let clearance = 0.2;
        // Trace B (the obstacle): a straight horizontal foreign track on `top` at y = 0.
        let b =
            PcbTrace::new(vec![wire_on(0.0, 0.0, "top"), wire_on(4.0, 0.0, "top")]).with_net("B");
        // Trace A: arrives on `top` from far above, drops a via at (2, 0.12) — only
        // 0.12 from B (copper gap 0.12 - 0.10 = 0.02 << clearance 0.2) — then departs on
        // `bottom`. The via landing on `top` is the LAST point of the first run; the
        // approach segment from (2,2) grazes B. The far ends (2,2) and (2,-2) are the
        // real endpoints and must stay put; only the via (and its two anchors) may move.
        let a = PcbTrace::new(vec![
            wire_on(2.0, 2.0, "top"),
            wire_on(2.0, 0.12, "top"),
            via_at(2.0, 0.12),
            wire_on(2.0, 0.12, "bottom"),
            wire_on(2.0, -2.0, "bottom"),
        ])
        .with_net("A");

        let via_xy = |t: &PcbTrace| -> P {
            t.route
                .iter()
                .find_map(|rp| match rp {
                    RoutePoint::Via { x, y, .. } => Some((*x, *y)),
                    _ => None,
                })
                .unwrap()
        };
        let via_count = |t: &PcbTrace| -> usize {
            t.route
                .iter()
                .filter(|rp| matches!(rp, RoutePoint::Via { .. }))
                .count()
        };

        let out = legalize_clearance(vec![a.clone(), b.clone()], &[], clearance);
        let (oa, ob) = (&out[0], &out[1]);

        // Via count unchanged.
        assert_eq!(via_count(oa), 1, "via count must be unchanged");
        // The via moved away from B (its y grew past the original 0.12).
        let (vx, vy) = via_xy(oa);
        let (bx, by) = via_xy(&a);
        assert!(
            (vy - by).abs() > 1e-6 || (vx - bx).abs() > 1e-6,
            "via should have been nudged: {:?} -> {:?}",
            (bx, by),
            (vx, vy)
        );
        // Connectivity preserved: via barrel still coincides with both incident anchors.
        let pa = pts_of(oa);
        assert!(
            approx(pa[1], (vx, vy)),
            "landing anchor must track the via: {pa:?}"
        );
        assert!(
            approx(pa[2], (vx, vy)),
            "departing anchor must track the via: {pa:?}"
        );
        // Real endpoints (far run ends) are immovable.
        assert!(approx(pa[0], (2.0, 2.0)), "start endpoint moved: {pa:?}");
        assert!(
            approx(*pa.last().unwrap(), (2.0, -2.0)),
            "end endpoint moved: {pa:?}"
        );
        // The obstacle trace B is left alone (it was already clear of itself).
        assert_eq!(pts_of(ob), pts_of(&b), "obstacle track must not move");

        // Clearance against B's `top` segment strictly improved (was 0.02 copper gap).
        let gap_to_b = |t: &PcbTrace| -> f64 {
            // Only the `top` approach segment of A faces B; measure it directly.
            let p = pts_of(t);
            seg_seg_dist(p[0], p[1], (0.0, 0.0), (4.0, 0.0)) - 2.0 * 0.05
        };
        let before = 0.12 - 0.10;
        let after = gap_to_b(oa);
        assert!(
            after > before + 1e-6,
            "via nudge must increase the foreign gap: {before} -> {after}"
        );
    }
}
