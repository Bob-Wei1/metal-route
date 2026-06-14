//! `mr-drc` — a geometric design-rule check for routed boards.
//!
//! This crate is deliberately *pure geometry*: it knows nothing about grids,
//! rasterisation, DSN, or the routers. A caller (today `mr-cli`) builds a
//! [`DrcBoard`] — net-tagged copper features on a physical layer stack, plus the
//! [`DrcRules`] to enforce — and [`DrcBoard::check`] returns every [`Violation`].
//!
//! Scope is intentionally narrow: only the violation classes metalroute can
//! actually produce ([`ViolationClass`]). It is **not** a full KiCad-equivalent
//! DRC (no silkscreen, soldermask, courtyard, zone-fill connectivity, diff-pairs,
//! length matching, footprint checks). The optional `kicad-cli pcb drc` cross-check
//! in the benchmark suite is what validates that we agree with KiCad on the classes
//! we *do* check.
//!
//! All lengths are in the board's continuous units (millimetres for our DSN flow).
//!
//! ## Determinism
//!
//! [`DrcBoard::check`] must return a deterministically ordered `Vec<Violation>`
//! (sort by `class`, then `layer`, then quantised `location`, then `nets`) so the
//! checker can be diffed across runs and against a future GPU implementation via the
//! `mr-oracle` equivalence pattern.
//!
//! ## Performance
//!
//! Clearance checking must be near-linear: bin every feature into a uniform grid of
//! cell size `clearance + max_feature_extent` and only test feature pairs that share
//! or neighbour a bin. A naïve O(n²) sweep is acceptable only for the tiny golden
//! unit tests, never for the full-board path.

use serde::{Deserialize, Serialize};

/// The physical role of a copper layer in the stackup, indexed top (0) → bottom.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// A routable signal layer.
    Signal,
    /// A poured power/ground plane carrying `net` across the whole board area.
    /// A via that crosses this layer shorts to it unless it is the same net or the
    /// via carries a sufficient antipad (see [`Via::antipad_radius`]).
    Plane { net: String },
}

/// A straight copper trace segment on one layer, owned by `net`, drawn `width` wide
/// (so its half-width `width/2` is the clearance-relevant inflation radius).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub net: String,
    pub layer: u32,
    pub a: (f64, f64),
    pub b: (f64, f64),
    pub width: f64,
}

/// An axis-aligned rectangular copper pad on one layer.
///
/// `net == None` means the pad's net is unknown; the checker then treats it as
/// conflicting with *every* other net (conservative — never hides a real short).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    pub net: Option<String>,
    pub layer: u32,
    pub center: (f64, f64),
    pub width: f64,
    pub height: f64,
}

/// A via spanning the inclusive physical layer range `[from_layer, to_layer]`.
///
/// `pad_diameter` is the annular copper pad; `drill_diameter` is the hole. On every
/// layer the barrel passes through, the via presents copper of `pad_diameter` for
/// clearance purposes. `antipad_radius` is the clearance hole carved into any plane
/// it crosses: `None` (or too small) means the plane copper is not relieved and the
/// via shorts to a foreign plane (the M1 baseline state); `Some(r)` large enough
/// clears the short (the M2 fix).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Via {
    pub net: String,
    pub center: (f64, f64),
    pub pad_diameter: f64,
    pub drill_diameter: f64,
    pub from_layer: u32,
    pub to_layer: u32,
    pub antipad_radius: Option<f64>,
}

/// The design-rule constraints to enforce, all in board units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrcRules {
    /// Minimum copper-to-copper spacing between features of different nets.
    pub clearance: f64,
    /// Minimum annular clearance a plane must give a foreign via barrel: a via
    /// crossing a foreign plane is clean iff `antipad_radius >= drill/2 + plane_antipad`.
    pub plane_antipad: f64,
    /// Minimum via annular ring `(pad_diameter - drill_diameter) / 2`.
    pub min_annular_ring: f64,
}

/// The full physical board to check: a layer stack plus every copper feature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrcBoard {
    /// Physical copper layers, index 0 = top. Signal vs. plane (with net).
    pub layers: Vec<LayerKind>,
    pub segments: Vec<Segment>,
    pub pads: Vec<Pad>,
    pub vias: Vec<Via>,
    pub rules: DrcRules,
}

/// The class of a reported violation — only the classes metalroute can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationClass {
    /// Two different-net copper features (track/via-pad/pad) are closer than
    /// `clearance` on the same layer.
    Clearance,
    /// A via barrel crosses a plane of a different net without a sufficient antipad.
    ViaThroughPlane,
    /// A via's annular ring is below `min_annular_ring`.
    AnnularRing,
}

/// One reported violation. `nets` holds the two conflicting nets (for
/// [`ViolationClass::ViaThroughPlane`], `nets.0` is the via net and `nets.1` the
/// plane net; for [`ViolationClass::AnnularRing`], `nets.1` is empty). `measured`
/// is the offending value and `required` the rule it broke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    pub class: ViolationClass,
    pub layer: u32,
    pub location: (f64, f64),
    pub nets: (String, String),
    pub measured: f64,
    pub required: f64,
}

/// Violation counts by class, for baselines and human output.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrcSummary {
    pub total: usize,
    pub clearance: usize,
    pub via_through_plane: usize,
    pub annular_ring: usize,
}

impl DrcSummary {
    /// Tally a violation slice by class.
    pub fn of(violations: &[Violation]) -> Self {
        let mut s = DrcSummary {
            total: violations.len(),
            ..Default::default()
        };
        for v in violations {
            match v.class {
                ViolationClass::Clearance => s.clearance += 1,
                ViolationClass::ViaThroughPlane => s.via_through_plane += 1,
                ViolationClass::AnnularRing => s.annular_ring += 1,
            }
        }
        s
    }

    /// True when the board is DRC-clean.
    pub fn is_clean(&self) -> bool {
        self.total == 0
    }
}

impl DrcBoard {
    /// Run every check and return all violations in deterministic order.
    ///
    /// Stream A implements: clearance (uniform-grid spatial index, per layer,
    /// different-net pairs only), via-through-plane (antipad-aware), annular ring.
    pub fn check(&self) -> Vec<Violation> {
        let mut out = Vec::new();
        self.check_clearance(&mut out);
        self.check_via_through_plane(&mut out);
        self.check_annular_ring(&mut out);
        out.sort_by_key(violation_key);
        out
    }

    /// Clearance: bin every copper feature into a per-layer uniform grid and test
    /// only different-net pairs that share or neighbour a bin (near-linear).
    fn check_clearance(&self, out: &mut Vec<Violation>) {
        // Collect the copper features present on each physical layer. A via pad is
        // present (as a circle) on every layer in its inclusive span.
        let mut by_layer: std::collections::HashMap<u32, Vec<Feature>> =
            std::collections::HashMap::new();
        let layer_count = self.layers.len() as u32;

        for s in &self.segments {
            by_layer
                .entry(s.layer)
                .or_default()
                .push(Feature::segment(s));
        }
        for p in &self.pads {
            by_layer.entry(p.layer).or_default().push(Feature::pad(p));
        }
        for v in &self.vias {
            let (lo, hi) = (v.from_layer.min(v.to_layer), v.from_layer.max(v.to_layer));
            for layer in lo..=hi {
                // Only stamp the via pad on physical layers that actually exist; if
                // the stack is unknown (empty), trust the span as given.
                if layer_count == 0 || layer < layer_count {
                    by_layer.entry(layer).or_default().push(Feature::via(v));
                }
            }
        }

        // Cell size = clearance + 2·(largest feature extent), so any conflicting
        // pair shares or neighbours a bin. A conflicting pair's centroids are at most
        // `clearance + extent_a + extent_b ≤ clearance + 2·max_extent` apart; sizing
        // the cell to that bound keeps their bin indices within ±1 on each axis, so
        // the 3×3 neighbourhood scan below is exhaustive (a `+ max_extent` cell would
        // let a pair land two bins apart and be missed).
        let max_extent = by_layer
            .values()
            .flat_map(|fs| fs.iter())
            .map(Feature::extent)
            .fold(0.0_f64, f64::max);
        let cell = (self.rules.clearance + 2.0 * max_extent).max(f64::MIN_POSITIVE);

        // Stable iteration order over layers for determinism of `out` before sort.
        let mut layers: Vec<u32> = by_layer.keys().copied().collect();
        layers.sort_unstable();

        for layer in layers {
            let feats = &by_layer[&layer];
            // Bin by the feature centroid cell.
            let mut bins: std::collections::HashMap<(i64, i64), Vec<usize>> =
                std::collections::HashMap::new();
            for (i, f) in feats.iter().enumerate() {
                let (cx, cy) = f.centroid();
                let key = ((cx / cell).floor() as i64, (cy / cell).floor() as i64);
                bins.entry(key).or_default().push(i);
            }

            // To report each unordered pair once, only test feature j against i when
            // (i < j). We scan each feature's own bin and the 3x3 neighbourhood.
            for (&(bx, by), idxs) in &bins {
                for &i in idxs {
                    for ndx in -1..=1 {
                        for ndy in -1..=1 {
                            if let Some(neigh) = bins.get(&(bx + ndx, by + ndy)) {
                                for &j in neigh {
                                    if i >= j {
                                        continue;
                                    }
                                    self.test_clearance_pair(layer, &feats[i], &feats[j], out);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn test_clearance_pair(&self, layer: u32, a: &Feature, b: &Feature, out: &mut Vec<Violation>) {
        if !nets_conflict(&a.net, &b.net) {
            return;
        }
        let gap = feature_gap(a, b);
        if gap < self.rules.clearance - EPS {
            let (ax, ay) = a.centroid();
            let (bx, by) = b.centroid();
            // Canonicalise the net pair so the row is independent of which feature
            // happened to get the lower bin index (input-order independence).
            let na = a.net.clone().unwrap_or_default();
            let nb = b.net.clone().unwrap_or_default();
            let nets = if na <= nb { (na, nb) } else { (nb, na) };
            out.push(Violation {
                class: ViolationClass::Clearance,
                layer,
                location: ((ax + bx) * 0.5, (ay + by) * 0.5),
                nets,
                measured: gap,
                required: self.rules.clearance,
            });
        }
    }

    /// Via-through-plane: a via barrel shorts every foreign-net plane in its
    /// inclusive span unless it carries a sufficient antipad.
    fn check_via_through_plane(&self, out: &mut Vec<Violation>) {
        let layer_count = self.layers.len() as u32;
        let required = |v: &Via| v.drill_diameter / 2.0 + self.rules.plane_antipad;
        for v in &self.vias {
            let (lo, hi) = (v.from_layer.min(v.to_layer), v.from_layer.max(v.to_layer));
            for layer in lo..=hi {
                if layer_count != 0 && layer >= layer_count {
                    continue;
                }
                let Some(LayerKind::Plane { net }) = self.layers.get(layer as usize) else {
                    continue;
                };
                if *net == v.net {
                    continue;
                }
                let req = required(v);
                let antipad = v.antipad_radius.unwrap_or(0.0);
                let clean = v.antipad_radius.is_some_and(|r| r >= req - EPS);
                if !clean {
                    out.push(Violation {
                        class: ViolationClass::ViaThroughPlane,
                        layer,
                        location: v.center,
                        nets: (v.net.clone(), net.clone()),
                        measured: antipad,
                        required: req,
                    });
                }
            }
        }
    }

    /// Annular ring: `(pad_diameter - drill_diameter)/2` must meet `min_annular_ring`.
    fn check_annular_ring(&self, out: &mut Vec<Violation>) {
        for v in &self.vias {
            let measured = (v.pad_diameter - v.drill_diameter) / 2.0;
            if measured < self.rules.min_annular_ring - EPS {
                out.push(Violation {
                    class: ViolationClass::AnnularRing,
                    layer: v.from_layer,
                    location: v.center,
                    nets: (v.net.clone(), String::new()),
                    measured,
                    required: self.rules.min_annular_ring,
                });
            }
        }
    }
}

/// Floating-point slack: a gap counts as a violation only when it is strictly
/// below the rule by more than this, so features placed exactly at the rule pass.
const EPS: f64 = 1e-9;

/// Two nets conflict when they are different. A `None` net (unknown pad) is treated
/// as a distinct always-foreign net, so it conflicts with everything — including
/// another `None`.
fn nets_conflict(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x != y,
        _ => true,
    }
}

/// A copper feature reduced to its clearance geometry on one layer.
///
/// Every feature is either a capsule (segment inflated by `inflate`), a circle
/// (via pad: a point inflated by its radius), or an axis-aligned rectangle (pad).
/// Modelling capsules and circles as an inflated core lets `feature_gap` compose
/// most cases as `core_gap - inflate_a - inflate_b`.
#[derive(Clone, Debug)]
struct Feature {
    net: Option<String>,
    shape: Shape,
    /// Radius the core shape is inflated by (half-width / pad radius); rects use 0.
    inflate: f64,
}

#[derive(Clone, Debug)]
enum Shape {
    /// A line core from `a` to `b` (inflated → capsule).
    Segment { a: (f64, f64), b: (f64, f64) },
    /// A point core (inflated → circle).
    Point { c: (f64, f64) },
    /// An axis-aligned rectangle, full width/height about its center.
    Rect { c: (f64, f64), w: f64, h: f64 },
}

impl Feature {
    fn segment(s: &Segment) -> Self {
        Feature {
            net: Some(s.net.clone()),
            shape: Shape::Segment { a: s.a, b: s.b },
            inflate: s.width / 2.0,
        }
    }

    fn pad(p: &Pad) -> Self {
        Feature {
            net: p.net.clone(),
            shape: Shape::Rect {
                c: p.center,
                w: p.width,
                h: p.height,
            },
            inflate: 0.0,
        }
    }

    fn via(v: &Via) -> Self {
        Feature {
            net: Some(v.net.clone()),
            shape: Shape::Point { c: v.center },
            inflate: v.pad_diameter / 2.0,
        }
    }

    /// A representative point for binning and reporting.
    fn centroid(&self) -> (f64, f64) {
        match &self.shape {
            Shape::Segment { a, b } => ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5),
            Shape::Point { c } => *c,
            Shape::Rect { c, .. } => *c,
        }
    }

    /// Half the largest planar reach from the centroid, used to size grid cells so
    /// every conflicting pair lands in neighbouring bins.
    fn extent(&self) -> f64 {
        match &self.shape {
            Shape::Segment { a, b } => dist(*a, *b) / 2.0 + self.inflate,
            Shape::Point { .. } => self.inflate,
            Shape::Rect { w, h, .. } => (w.max(*h)) / 2.0,
        }
    }
}

/// Copper-to-copper gap between two features (negative if they overlap). For the
/// capsule/circle cases this is the core-to-core distance minus both inflations;
/// rectangles are handled by the appropriate point/segment-to-rect routine.
fn feature_gap(a: &Feature, b: &Feature) -> f64 {
    match (&a.shape, &b.shape) {
        // Core ⇄ core distances minus inflation (capsules and circles).
        (Shape::Segment { a: p0, b: p1 }, Shape::Segment { a: q0, b: q1 }) => {
            seg_seg_dist(*p0, *p1, *q0, *q1) - a.inflate - b.inflate
        }
        (Shape::Segment { a: p0, b: p1 }, Shape::Point { c }) => {
            point_seg_dist(*c, *p0, *p1) - a.inflate - b.inflate
        }
        (Shape::Point { c }, Shape::Segment { a: p0, b: p1 }) => {
            point_seg_dist(*c, *p0, *p1) - a.inflate - b.inflate
        }
        (Shape::Point { c: c0 }, Shape::Point { c: c1 }) => dist(*c0, *c1) - a.inflate - b.inflate,
        // Rectangle cases: distance from the rect to the other core, minus the
        // other feature's inflation (the rect itself has inflate == 0).
        (Shape::Rect { c, w, h }, Shape::Point { c: p }) => {
            point_rect_gap(*p, *c, *w, *h) - b.inflate
        }
        (Shape::Point { c: p }, Shape::Rect { c, w, h }) => {
            point_rect_gap(*p, *c, *w, *h) - a.inflate
        }
        (Shape::Rect { c, w, h }, Shape::Segment { a: p0, b: p1 }) => {
            seg_rect_gap(*p0, *p1, *c, *w, *h) - b.inflate
        }
        (Shape::Segment { a: p0, b: p1 }, Shape::Rect { c, w, h }) => {
            seg_rect_gap(*p0, *p1, *c, *w, *h) - a.inflate
        }
        (
            Shape::Rect {
                c: c0,
                w: w0,
                h: h0,
            },
            Shape::Rect {
                c: c1,
                w: w1,
                h: h1,
            },
        ) => rect_rect_gap(*c0, *w0, *h0, *c1, *w1, *h1),
    }
}

/// Euclidean point-to-point distance.
pub fn dist(p: (f64, f64), q: (f64, f64)) -> f64 {
    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
}

/// Distance from point `p` to the segment `[a, b]` (0 if `p` lies on it).
pub fn point_seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len2 = abx * abx + aby * aby;
    if len2 <= f64::MIN_POSITIVE {
        return dist(p, a);
    }
    let t = (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len2).clamp(0.0, 1.0);
    let proj = (a.0 + t * abx, a.1 + t * aby);
    dist(p, proj)
}

/// Shortest distance between two segments `[p0,p1]` and `[q0,q1]` (0 if they cross).
///
/// Standard closest-distance-between-two-segments: minimise the squared distance of
/// `s(t) = p0 + t·d1` and `r(u) = q0 + u·d2` over `t,u ∈ [0,1]`, with the degenerate
/// (point) cases handled explicitly.
pub fn seg_seg_dist(p0: (f64, f64), p1: (f64, f64), q0: (f64, f64), q1: (f64, f64)) -> f64 {
    let d1 = (p1.0 - p0.0, p1.1 - p0.1);
    let d2 = (q1.0 - q0.0, q1.1 - q0.1);
    let r = (p0.0 - q0.0, p0.1 - q0.1);
    let a = d1.0 * d1.0 + d1.1 * d1.1; // |d1|^2
    let e = d2.0 * d2.0 + d2.1 * d2.1; // |d2|^2
    let f = d2.0 * r.0 + d2.1 * r.1;

    // Both segments degenerate to points.
    if a <= f64::MIN_POSITIVE && e <= f64::MIN_POSITIVE {
        return dist(p0, q0);
    }
    let (mut s, t);
    if a <= f64::MIN_POSITIVE {
        // First segment is a point.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.0 * r.0 + d1.1 * r.1;
        if e <= f64::MIN_POSITIVE {
            // Second segment is a point.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.0 * d2.0 + d1.1 * d2.1;
            let denom = a * e - b * b;
            s = if denom > f64::MIN_POSITIVE {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut tt = (b * s + f) / e;
            if tt < 0.0 {
                tt = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if tt > 1.0 {
                tt = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            t = tt;
        }
    }
    let cp = (p0.0 + s * d1.0, p0.1 + s * d1.1);
    let cq = (q0.0 + t * d2.0, q0.1 + t * d2.1);
    dist(cp, cq)
}

/// Gap from a point to an axis-aligned rectangle (center `c`, full `w`×`h`); 0 when
/// the point is inside.
pub fn point_rect_gap(p: (f64, f64), c: (f64, f64), w: f64, h: f64) -> f64 {
    let dx = (c.0 - p.0).abs() - w / 2.0;
    let dy = (c.1 - p.1).abs() - h / 2.0;
    let ox = dx.max(0.0);
    let oy = dy.max(0.0);
    if ox == 0.0 && oy == 0.0 {
        // Inside the rectangle.
        0.0
    } else {
        (ox * ox + oy * oy).sqrt()
    }
}

/// Gap from a segment to an axis-aligned rectangle. Sampled against the rectangle's
/// four edges (and inside-test), which is exact for the segment-vs-AABB case.
pub fn seg_rect_gap(p0: (f64, f64), p1: (f64, f64), c: (f64, f64), w: f64, h: f64) -> f64 {
    // If either endpoint is inside, gap is 0.
    if point_rect_gap(p0, c, w, h) == 0.0 || point_rect_gap(p1, c, w, h) == 0.0 {
        return 0.0;
    }
    let (hw, hh) = (w / 2.0, h / 2.0);
    let corners = [
        (c.0 - hw, c.1 - hh),
        (c.0 + hw, c.1 - hh),
        (c.0 + hw, c.1 + hh),
        (c.0 - hw, c.1 + hh),
    ];
    let mut best = f64::INFINITY;
    // Segment-to-each-edge distance.
    for i in 0..4 {
        let e0 = corners[i];
        let e1 = corners[(i + 1) % 4];
        best = best.min(seg_seg_dist(p0, p1, e0, e1));
    }
    best
}

/// Gap between two axis-aligned rectangles (0 if they overlap or touch).
pub fn rect_rect_gap(c0: (f64, f64), w0: f64, h0: f64, c1: (f64, f64), w1: f64, h1: f64) -> f64 {
    let dx = (c0.0 - c1.0).abs() - (w0 + w1) / 2.0;
    let dy = (c0.1 - c1.1).abs() - (h0 + h1) / 2.0;
    let ox = dx.max(0.0);
    let oy = dy.max(0.0);
    (ox * ox + oy * oy).sqrt()
}

/// Class ordering key: explicit so it does not depend on the enum's derive order.
fn class_order(c: ViolationClass) -> u8 {
    match c {
        ViolationClass::Clearance => 0,
        ViolationClass::ViaThroughPlane => 1,
        ViolationClass::AnnularRing => 2,
    }
}

/// Quantise a coordinate to a fixed grid so float jitter can't reorder ties.
fn quantise(x: f64) -> i64 {
    (x / 1e-6).round() as i64
}

/// Total, stable ordering key for a violation: class, layer, quantised location,
/// then nets. Coordinates are quantised to 1e-6 to avoid float-ordering hazards.
fn violation_key(v: &Violation) -> (u8, u32, i64, i64, String, String) {
    (
        class_order(v.class),
        v.layer,
        quantise(v.location.0),
        quantise(v.location.1),
        v.nets.0.clone(),
        v.nets.1.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> DrcRules {
        DrcRules {
            clearance: 0.2,
            plane_antipad: 0.1,
            min_annular_ring: 0.05,
        }
    }

    fn seg(net: &str, layer: u32, a: (f64, f64), b: (f64, f64), width: f64) -> Segment {
        Segment {
            net: net.to_string(),
            layer,
            a,
            b,
            width,
        }
    }

    fn classes(vs: &[Violation], class: ViolationClass) -> usize {
        vs.iter().filter(|v| v.class == class).count()
    }

    #[test]
    fn parallel_segments_violate_when_too_close() {
        // Two horizontal traces, width 0.1 (half-width 0.05 each). Centreline gap
        // 0.25 → copper gap 0.25 - 0.05 - 0.05 = 0.15 < clearance 0.2 → violation.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 0.25), (10.0, 0.25), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        let v = board.check();
        assert_eq!(classes(&v, ViolationClass::Clearance), 1);
        let c = &v[0];
        assert!(
            (c.measured - 0.15).abs() < 1e-9,
            "measured = {}",
            c.measured
        );
        assert_eq!(c.required, 0.2);
        assert_eq!(c.layer, 0);
    }

    #[test]
    fn segments_exactly_at_clearance_pass() {
        // Centreline gap 0.30 → copper gap 0.30 - 0.05 - 0.05 = 0.20 == clearance.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 0.30), (10.0, 0.30), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(board.check().is_empty(), "at-clearance must not fire");
    }

    #[test]
    fn segments_above_clearance_pass() {
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 1.0), (10.0, 1.0), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(board.check().is_empty());
    }

    #[test]
    fn same_net_touching_is_clean() {
        // Two overlapping segments on the same net → never a clearance violation.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("NET1", 0, (0.0, 0.0), (10.0, 0.0), 0.2),
                seg("NET1", 0, (5.0, 0.0), (5.0, 5.0), 0.2),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(board.check().is_empty(), "same net never conflicts");
    }

    #[test]
    fn different_layers_dont_interact() {
        // Same planar position, zero gap, but different layers → no clearance.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.2),
                seg("B", 1, (0.0, 0.0), (10.0, 0.0), 0.2),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(
            board.check().is_empty(),
            "features on different layers must not conflict"
        );
    }

    #[test]
    fn none_net_pad_conflicts_with_everything() {
        // A net==None pad touching a segment of a real net → violation.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1)],
            pads: vec![Pad {
                net: None,
                layer: 0,
                center: (5.0, 0.1),
                width: 0.2,
                height: 0.2,
            }],
            vias: vec![],
            rules: rules(),
        };
        let v = board.check();
        assert_eq!(classes(&v, ViolationClass::Clearance), 1);
        // None net renders as empty string.
        assert!(v[0].nets.0.is_empty() || v[0].nets.1.is_empty());
    }

    #[test]
    fn via_through_foreign_plane_fires_without_antipad() {
        // Stack: Signal / Plane(GND) / Signal. Via on NET1 spanning 0..=2 crosses
        // the GND plane on layer 1 with no antipad → ViaThroughPlane.
        let board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let v = board.check();
        assert_eq!(classes(&v, ViolationClass::ViaThroughPlane), 1);
        let tp = v
            .iter()
            .find(|x| x.class == ViolationClass::ViaThroughPlane)
            .unwrap();
        assert_eq!(tp.layer, 1);
        assert_eq!(tp.nets, ("NET1".to_string(), "GND".to_string()));
        assert_eq!(tp.measured, 0.0);
        // required = drill/2 + plane_antipad = 0.15 + 0.1 = 0.25.
        assert!((tp.required - 0.25).abs() < 1e-9);
    }

    #[test]
    fn via_through_plane_clean_with_large_antipad() {
        let board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: Some(0.25), // == drill/2 + plane_antipad
            }],
            rules: rules(),
        };
        assert_eq!(
            classes(&board.check(), ViolationClass::ViaThroughPlane),
            0,
            "antipad exactly meeting the rule clears the short"
        );
    }

    #[test]
    fn via_through_same_net_plane_is_clean() {
        let board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "GND".to_string(), // same net as the plane
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        assert_eq!(
            classes(&board.check(), ViolationClass::ViaThroughPlane),
            0,
            "a via on the plane's own net never shorts it"
        );
    }

    #[test]
    fn endpoint_plane_is_a_short() {
        // Via ends ON a foreign plane (layer 0 is the plane). Endpoints are part of
        // the inclusive span, so this must fire.
        let board = DrcBoard {
            layers: vec![
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 1,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let v = board.check();
        let tp: Vec<_> = v
            .iter()
            .filter(|x| x.class == ViolationClass::ViaThroughPlane)
            .collect();
        assert_eq!(tp.len(), 1);
        assert_eq!(tp[0].layer, 0, "endpoint plane on layer 0 shorts");
    }

    #[test]
    fn annular_ring_below_minimum_fires() {
        // (pad - drill)/2 = (0.30 - 0.25)/2 = 0.025 < min 0.05.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.30,
                drill_diameter: 0.25,
                from_layer: 0,
                to_layer: 1,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let v = board.check();
        let ar = v
            .iter()
            .find(|x| x.class == ViolationClass::AnnularRing)
            .unwrap();
        assert!((ar.measured - 0.025).abs() < 1e-9);
        assert_eq!(ar.required, 0.05);
        assert_eq!(ar.layer, 0);
        assert_eq!(ar.nets.1, "");
    }

    #[test]
    fn annular_ring_at_minimum_passes() {
        // (0.40 - 0.30)/2 = 0.05 == min.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.40,
                drill_diameter: 0.30,
                from_layer: 0,
                to_layer: 1,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        assert_eq!(classes(&board.check(), ViolationClass::AnnularRing), 0);
    }

    /// A small mixed board reused by the determinism tests.
    fn mixed_board(segs: Vec<Segment>, pads: Vec<Pad>, vias: Vec<Via>) -> DrcBoard {
        DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: segs,
            pads,
            vias,
            rules: rules(),
        }
    }

    fn sample_features() -> (Vec<Segment>, Vec<Pad>, Vec<Via>) {
        let segs = vec![
            seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
            seg("B", 0, (0.0, 0.25), (10.0, 0.25), 0.1), // clearance vs A
            seg("C", 2, (0.0, 0.0), (5.0, 0.0), 0.1),
        ];
        let pads = vec![Pad {
            net: None,
            layer: 0,
            center: (3.0, 0.1),
            width: 0.1,
            height: 0.1,
        }];
        let vias = vec![
            Via {
                net: "NET1".to_string(),
                center: (7.0, 5.0),
                pad_diameter: 0.30,
                drill_diameter: 0.25, // annular 0.025 < 0.05
                from_layer: 0,
                to_layer: 2, // crosses GND plane on layer 1
                antipad_radius: None,
            },
            Via {
                net: "GND".to_string(),
                center: (8.0, 6.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2, // same net as plane → no short
                antipad_radius: None,
            },
        ];
        (segs, pads, vias)
    }

    #[test]
    fn determinism_same_board_twice() {
        let (s, p, v) = sample_features();
        let b1 = mixed_board(s.clone(), p.clone(), v.clone());
        let b2 = mixed_board(s, p, v);
        assert_eq!(b1.check(), b2.check());
        // Sanity: it actually produced violations across classes.
        let r = b1.check();
        assert!(classes(&r, ViolationClass::Clearance) >= 1);
        assert!(classes(&r, ViolationClass::ViaThroughPlane) >= 1);
        assert!(classes(&r, ViolationClass::AnnularRing) >= 1);
    }

    #[test]
    fn determinism_shuffled_input_same_output() {
        let (s, p, v) = sample_features();
        let baseline = mixed_board(s.clone(), p.clone(), v.clone()).check();

        // Reverse each input vector — order must not change the sorted result.
        let mut s2 = s;
        s2.reverse();
        let mut v2 = v;
        v2.reverse();
        let shuffled = mixed_board(s2, p, v2).check();

        assert_eq!(
            baseline, shuffled,
            "shuffling feature input order must yield identical sorted violations"
        );
    }

    #[test]
    fn each_pair_reported_once() {
        // Two close different-net segments must produce exactly one clearance row,
        // not two (i<j de-duplication).
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 0.1), (10.0, 0.1), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert_eq!(classes(&board.check(), ViolationClass::Clearance), 1);
    }

    #[test]
    fn geometry_helpers_basic() {
        assert!((dist((0.0, 0.0), (3.0, 4.0)) - 5.0).abs() < 1e-12);
        assert!((point_seg_dist((5.0, 3.0), (0.0, 0.0), (10.0, 0.0)) - 3.0).abs() < 1e-12);
        // Parallel segments 2 apart.
        assert!(
            (seg_seg_dist((0.0, 0.0), (10.0, 0.0), (0.0, 2.0), (10.0, 2.0)) - 2.0).abs() < 1e-12
        );
        // Crossing segments → 0.
        assert!(seg_seg_dist((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)) < 1e-12);
        // Point outside a unit rect centred at origin, 1 to the right of the edge.
        assert!((point_rect_gap((1.5, 0.0), (0.0, 0.0), 1.0, 1.0) - 1.0).abs() < 1e-12);
        // Point inside → 0.
        assert!(point_rect_gap((0.0, 0.0), (0.0, 0.0), 1.0, 1.0) < 1e-12);
    }
}
