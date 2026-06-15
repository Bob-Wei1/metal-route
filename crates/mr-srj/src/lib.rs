//! `mr-srj` — tscircuit SimpleRouteJson I/O for metalroute.
//!
//! Two responsibilities:
//!
//! 1. **Input (B3): rasterise.** Parse a [`SimpleRouteJson`] problem (continuous
//!    geometry) and convert it into a cell-space [`RasterizedProblem`]: a
//!    [`mr_core::Grid`] of obstacles, a list of two-point [`mr_core::NetEndpoints`],
//!    and the [`Mapping`] that ties continuous coordinates to grid cells.
//! 2. **Output (B4): de-rasterise.** Turn a routed [`mr_core::BoardRoute`] back
//!    into a tscircuit solution soup ([`PcbTrace`] objects) using the same
//!    [`Mapping`] to recover continuous coordinates from cell centres.
//!
//! ## Coordinate mapping ([`Mapping`])
//!
//! The grid is a **non-uniform / Hanan grid**: instead of a single scalar
//! `resolution`, each axis carries a sorted array of continuous grid-line
//! positions ([`Mapping::x_lines`] / [`Mapping::y_lines`]). A cell *is* a pair of
//! line indices `(x, y)`, and its continuous coordinate sits exactly ON those lines
//! — `cell_center((x,y)) = (x_lines[x], y_lines[y])` (no `+0.5`: nodes are on
//! lines). Grid dimensions are line counts: `dims.w = x_lines.len()`,
//! `dims.h = y_lines.len()`, each floored at 1 so degenerate / zero-area bounds
//! still yield a usable single-cell grid.
//!
//! Each cell owns the half-open continuous region bounded by the midpoints between
//! its line and its neighbours (a 1-D Voronoi partition). [`Mapping::new`] /
//! [`Mapping::with_layers`] still take a scalar `resolution`; they build a
//! *uniform* line set placed at the historical cell centres
//! (`origin + (i+0.5)·res`), so the cell regions, [`Mapping::point_to_cell`] floor
//! mapping, and obstacle coverage are byte-identical to the old uniform grid. Only
//! the internal representation changed.
//!
//! ## Obstacles
//!
//! A `rect` obstacle centred at `(cx, cy)` with `width w` / `height h` covers the
//! continuous box `[cx-w/2, cx+w/2] × [cy-h/2, cy+h/2]`. Every grid cell whose own
//! square area overlaps that box is marked an obstacle (area-overlap, not
//! centre-in-rect), via [`mr_grid::GridBuilder::mark_rect`].
//!
//! ## k-point connection decomposition (plan R8)
//!
//! The router contract is strictly two-point. A connection with `k` points in
//! `pointsToConnect` is decomposed into `k-1` consecutive two-point
//! [`mr_core::NetEndpoints`] in chain order (point[0]→point[1], point[1]→point[2],
//! …). When `k > 2` the resulting nets are named `"<conn.name>#0"`,
//! `"<conn.name>#1"`, …; a plain two-point connection keeps the bare `conn.name`.
//! Connections with fewer than two points produce no nets.

use std::collections::HashMap;

use mr_core::{BoardRoute, CellIdx, Dims, Grid, GridCoords, LayerMap, NetEndpoints};
use mr_grid::GridBuilder;
use serde::{Deserialize, Serialize};

mod smooth;
pub use smooth::beautify_traces;

/// A 2-D point in continuous tscircuit coordinates.
///
/// The real harness attaches extra fields (`layer`, `pcb_port_id`); `layer` is
/// captured for completeness and any other unknown fields are ignored by serde.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub layer: Option<String>,
}

/// Axis-aligned problem bounds in continuous coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

/// A single rectangular obstacle (the only obstacle shape tscircuit emits here).
///
/// `kind` mirrors the JSON `type` field (always `"rect"` today) but is kept as a
/// `String` so unknown shapes still round-trip rather than failing to parse.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Obstacle {
    #[serde(rename = "type")]
    pub kind: String,
    pub center: Point,
    pub width: f64,
    pub height: f64,
    /// Layers this obstacle sits on (e.g. `["top"]`). Empty if unspecified.
    #[serde(default)]
    pub layers: Vec<String>,
    /// IDs of the pads/elements this obstacle is electrically connected to
    /// (the harness emits `connectedTo`). Empty if unspecified.
    #[serde(default, rename = "connectedTo")]
    pub connected_to: Vec<String>,
}

/// One named connection: a list of pads/points that must all be electrically
/// joined. Decomposed into chained two-point nets at rasterisation time (R8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Connection {
    pub name: String,
    pub points_to_connect: Vec<Point>,
}

/// A tscircuit SimpleRouteJson routing problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimpleRouteJson {
    pub layer_count: u32,
    /// Minimum trace width in continuous units (the harness emits `minTraceWidth`).
    /// Drives both the emitted wire width and the rasterisation resolution.
    #[serde(default)]
    pub min_trace_width: Option<f64>,
    /// Minimum copper-to-copper clearance in continuous units. Like
    /// [`Self::min_trace_width`], this is parsed from the DSN `(rule (clearance N))`
    /// rule (the harness emits `minClearance`) and feeds the design-rule check.
    #[serde(default)]
    pub min_clearance: Option<f64>,
    #[serde(default)]
    pub obstacles: Vec<Obstacle>,
    #[serde(default)]
    pub connections: Vec<Connection>,
    pub bounds: Bounds,
}

/// Continuous ↔ grid-cell coordinate conversion. See module docs for the rules.
#[derive(Debug, Clone, PartialEq)]
pub struct Mapping {
    /// Sorted, deduped continuous x positions of the grid lines. Nodes sit ON
    /// these lines; `dims.w == x_lines.len()`. Always non-empty (≥ 1 line).
    pub x_lines: Vec<f64>,
    /// Sorted, deduped continuous y positions of the grid lines. Nodes sit ON
    /// these lines; `dims.h == y_lines.len()`. Always non-empty (≥ 1 line).
    pub y_lines: Vec<f64>,
    /// Grid dimensions (≥ 1×1). `dims.w == x_lines.len()`, `dims.h == y_lines.len()`.
    pub dims: Dims,
}

impl Mapping {
    /// Build a single-layer mapping for `bounds` at the given `resolution`.
    ///
    /// Lines are placed uniformly at the historical cell centres
    /// (`origin + (i+0.5)·res`), with `ceil(span / resolution)` lines per axis,
    /// floored at 1 so a zero- or negative-area bounds still yields a 1×1 grid.
    /// Use [`Mapping::with_layers`] for a multi-layer board; this remains
    /// single-layer so every existing 2D caller is byte-identical.
    pub fn new(bounds: &Bounds, resolution: f64) -> Self {
        Self::with_layers(bounds, resolution, 1)
    }

    /// Build a mapping for `bounds` at `resolution` over `layers` stacked planes.
    ///
    /// The planar (x, y) sizing is identical to [`Mapping::new`]; only the layer
    /// axis grows. `layers == 1` is byte-identical to [`Mapping::new`]. The line
    /// arrays are a *uniform* set placed at the historical cell centres so cell
    /// regions, the floor mapping, and obstacle coverage match the old uniform
    /// grid exactly.
    pub fn with_layers(bounds: &Bounds, resolution: f64, layers: u32) -> Self {
        let span_x = (bounds.max_x - bounds.min_x).max(0.0);
        let span_y = (bounds.max_y - bounds.min_y).max(0.0);
        let w = ((span_x / resolution).ceil() as u32).max(1);
        let h = ((span_y / resolution).ceil() as u32).max(1);
        let x_lines = uniform_lines(bounds.min_x, resolution, w);
        let y_lines = uniform_lines(bounds.min_y, resolution, h);
        Self::from_lines(x_lines, y_lines, layers)
    }

    /// Build a mapping directly from explicit sorted line arrays over `layers`
    /// planes. Empty arrays are floored to a single line at 0.0 so `dims` stays
    /// ≥ 1×1. The caller is responsible for the arrays being sorted ascending and
    /// deduped; [`Mapping::with_layers`] and the grid-line builder satisfy this.
    pub fn from_lines(mut x_lines: Vec<f64>, mut y_lines: Vec<f64>, layers: u32) -> Self {
        if x_lines.is_empty() {
            x_lines.push(0.0);
        }
        if y_lines.is_empty() {
            y_lines.push(0.0);
        }
        let w = x_lines.len() as u32;
        let h = y_lines.len() as u32;
        Self {
            x_lines,
            y_lines,
            dims: Dims::with_layers(w, h, layers),
        }
    }

    /// Continuous coordinate of cell `cell` — the position of its grid lines. The
    /// layer is ignored (every layer shares the same planar geometry), so a cell
    /// and its via-neighbour on another layer map to the same continuous `(x, y)`.
    pub fn cell_center(&self, cell: CellIdx) -> (f64, f64) {
        let (x, y) = self.dims.xy(cell);
        (self.x_lines[x as usize], self.y_lines[y as usize])
    }

    /// Layer of `cell` (0 == top). Always 0 for a single-layer mapping.
    pub fn cell_layer(&self, cell: CellIdx) -> u32 {
        self.dims.layer_of(cell)
    }

    /// Cell on layer 0 containing continuous point `(x, y)`, clamped into the grid.
    /// See [`Mapping::point_to_cell_layer`] to target a specific layer.
    pub fn point_to_cell(&self, point: (f64, f64)) -> CellIdx {
        self.point_to_cell_layer(point, 0)
    }

    /// Cell on `layer` containing continuous point `(x, y)`, clamped into the grid.
    /// The layer is clamped to the last valid layer.
    pub fn point_to_cell_layer(&self, point: (f64, f64), layer: u32) -> CellIdx {
        let (cx, cy) = self.point_to_xy(point);
        let l = layer.min(self.dims.layers.saturating_sub(1));
        self.dims.idx3(cx, cy, l)
    }

    /// Cell `(x, y)` whose Voronoi region contains a continuous point, clamped into
    /// the grid. The region of line `i` is the half-open interval bounded by the
    /// midpoints to its neighbours, so this is the index of the nearest line with
    /// ties resolved toward the lower index (matching the historical `floor` mapping
    /// on a uniform line set).
    fn point_to_xy(&self, point: (f64, f64)) -> (u32, u32) {
        (
            nearest_line(&self.x_lines, point.0),
            nearest_line(&self.y_lines, point.1),
        )
    }

    /// Upper (inclusive) x-cell index touched by a continuous box that ends at `hi`:
    /// the largest region whose lower Voronoi boundary is strictly less than `hi`
    /// (half-open, so a box edge exactly on a cell boundary does not spill into the
    /// next cell). On a uniform line set this equals the old
    /// `ceil((hi-origin)/res) - 1`.
    fn x_cell_upper(&self, hi: f64) -> u32 {
        cell_upper(&self.x_lines, hi)
    }

    /// Upper (inclusive) y-cell index touched by a box ending at `hi`. See
    /// [`Mapping::x_cell_upper`].
    fn y_cell_upper(&self, hi: f64) -> u32 {
        cell_upper(&self.y_lines, hi)
    }
}

/// `count` uniformly spaced lines starting at the first cell centre
/// `origin + 0.5·res` and stepping by `res` — the historical cell-centre
/// positions for a uniform grid. `count` is the caller's already-floored
/// dimension (≥ 1).
fn uniform_lines(origin: f64, res: f64, count: u32) -> Vec<f64> {
    (0..count)
        .map(|i| origin + (i as f64 + 0.5) * res)
        .collect()
}

/// Two grid lines closer than this (continuous units) are treated as coincident
/// and collapsed during dedup, so near-coincident pad/obstacle coordinates do not
/// produce degenerate zero-width cells. Chosen well below any realistic feature
/// pitch (sub-micron on a mm board) so genuinely distinct features survive.
const LINE_EPSILON: f64 = 1e-6;

/// Soft ceiling on the planar cell count (`x_lines.len() · y_lines.len()`). When a
/// problem's Hanan + fill line set would exceed this, fill lines are thinned (see
/// [`build_grid_lines`]) and, if still over, a warning is logged; the historical
/// uniform grid sat around this magnitude, so dense BGAs don't silently blow up the
/// search space.
const CELL_BUDGET: usize = 40_000;

/// Build the non-uniform (Hanan-style) grid-line arrays for `srj` — the per-axis
/// sorted, deduped continuous positions every grid node sits on (plan section 2).
///
/// Per axis the line set is the union of:
/// * every **connection endpoint** coordinate (`points_to_connect`) — this is the
///   line that makes each pad an *exact* grid node, which is the whole point: a
///   pad's `cell_center` then equals its continuous coordinate, so the de-raster
///   `pin_points` snap-back becomes an identity and the pad-exit wiggle disappears;
/// * every **obstacle edge** (`center ± width/2`, `center ± height/2`) so obstacle
///   boundaries align to lines and coverage is exact (no partial-cell obstacles);
/// * the board **bounds** (`min`/`max`), so the grid spans the whole board.
///
/// Then **fill lines** are inserted into any gap between adjacent feature lines that
/// exceeds `track_w + 2·clearance` (the minimum room for a routing channel between
/// two features), subdividing the gap at ~that spacing so distant features still
/// have lanes between them — the classic "track between pins" line. Finally the set
/// is sorted and deduped (lines within [`LINE_EPSILON`] collapse).
///
/// `track_w` / `clearance` are the design-rule track width and copper clearance;
/// `_layers` is accepted for symmetry with the rasteriser (the planar line set is
/// layer-independent) but unused. The fill spacing and resulting cell count are
/// capped against [`CELL_BUDGET`]; an over-budget result is logged but still
/// returned (the router copes, just more slowly).
fn build_grid_lines(
    srj: &SimpleRouteJson,
    _layers: u32,
    track_w: f64,
    clearance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    // Board bounds: the grid must span the whole board.
    xs.push(srj.bounds.min_x);
    xs.push(srj.bounds.max_x);
    ys.push(srj.bounds.min_y);
    ys.push(srj.bounds.max_y);

    // Every endpoint coordinate — the lines that pin pads to exact nodes.
    for conn in &srj.connections {
        for p in &conn.points_to_connect {
            xs.push(p.x);
            ys.push(p.y);
        }
    }

    // Every obstacle edge, so obstacle boxes land exactly on lines.
    for obs in &srj.obstacles {
        xs.push(obs.center.x - obs.width / 2.0);
        xs.push(obs.center.x + obs.width / 2.0);
        ys.push(obs.center.y - obs.height / 2.0);
        ys.push(obs.center.y + obs.height / 2.0);
    }

    // `xs` / `ys` are now the *feature* lines (pads, obstacle edges, bounds): these
    // are essential — dropping one moves a pad off-node and reintroduces the pad-exit
    // wiggle — so the budget enforcement below never touches them.
    sort_dedup(&mut xs);
    sort_dedup(&mut ys);

    // Routing-channel fill lines, computed separately so the budget enforcer can thin
    // them without disturbing the feature lines. Every gap wide enough for a centred,
    // clearance-legal track gets a midpoint lane (plus parallel lanes for wide gaps);
    // gaps too tight for a clearance-legal lane get none (route goes around). See
    // [`fill_gaps`] for the exact per-gap policy.
    let mut x_fill = fill_lines(&xs, track_w, clearance);
    let mut y_fill = fill_lines(&ys, track_w, clearance);

    // ENFORCE the cell budget. The feature lines alone fix one factor of the
    // `x·y` product per axis; we have a fill budget for the other. Thin the fill
    // (coalescing near-coincident lanes, then dropping the least-important — densest-
    // packed — lanes) until `xs·ys <= CELL_BUDGET`, keeping every feature line. If the
    // feature lines alone already blow the budget, that is logged explicitly: no
    // amount of fill thinning can help and we must not silently proceed as if fine.
    enforce_budget(&xs, &ys, &mut x_fill, &mut y_fill);

    xs.append(&mut x_fill);
    ys.append(&mut y_fill);
    sort_dedup(&mut xs);
    sort_dedup(&mut ys);

    (xs, ys)
}

/// Thin the per-axis fill-line sets in place so the final grid honours
/// [`CELL_BUDGET`] (`x_features·y_features` plus retained fill ≤ budget), keeping
/// every feature line.
///
/// Strategy (fill lines are the only thinnable part — feature lines pin pads to
/// nodes and must all survive):
/// 1. Coalesce fill lines that are near-coincident with a feature line or with each
///    other (within [`LINE_EPSILON`]) — these add cells without adding a distinct
///    routing lane.
/// 2. While the combined grid is over budget, drop the *least-important* fill line:
///    the one whose two neighbours (feature-or-fill) are closest together, i.e. the
///    most redundant lane in the most crowded channel. Drop alternately from the axis
///    that currently has more fill, so both axes stay balanced.
/// 3. If after removing *all* fill the feature lines alone still exceed the budget,
///    log that explicitly — the search space is irreducibly large and the caller
///    proceeds knowingly (rather than silently).
fn enforce_budget(
    x_features: &[f64],
    y_features: &[f64],
    x_fill: &mut Vec<f64>,
    y_fill: &mut Vec<f64>,
) {
    // Step 1: drop fill lines coincident with a feature line (sort_dedup later would
    // merge them, but they must not count against the budget while we thin).
    coalesce_fill(x_features, x_fill);
    coalesce_fill(y_features, y_fill);

    let cells = |xf: usize, yf: usize| -> usize {
        (x_features.len() + xf).saturating_mul(y_features.len() + yf)
    };

    if cells(x_fill.len(), y_fill.len()) <= CELL_BUDGET {
        return;
    }

    // Step 3 (early check): feature lines alone over budget — fill thinning is futile.
    let feature_cells = x_features.len().saturating_mul(y_features.len());
    if feature_cells > CELL_BUDGET {
        // Drop all fill (it cannot help) and report honestly.
        let dropped = x_fill.len() + y_fill.len();
        x_fill.clear();
        y_fill.clear();
        eprintln!(
            "mr-srj: Hanan FEATURE lines alone {}×{} = {feature_cells} cells exceed \
             budget {CELL_BUDGET}; dropped all {dropped} fill lines but feature set is \
             irreducible (pads/obstacles) — routing on an over-budget grid. Consider \
             coarser features/clearance.",
            x_features.len(),
            y_features.len(),
        );
        return;
    }

    // Step 2: drop the least-important fill line, alternating axes, until under budget.
    let mut dropped = 0usize;
    while cells(x_fill.len(), y_fill.len()) > CELL_BUDGET {
        // Pick the axis to thin: the one with more fill remaining (keeps balance);
        // if one axis is empty, thin the other.
        let thin_x = if x_fill.is_empty() {
            false
        } else if y_fill.is_empty() {
            true
        } else {
            x_fill.len() >= y_fill.len()
        };
        let (features, fill) = if thin_x {
            (x_features, &mut *x_fill)
        } else {
            (y_features, &mut *y_fill)
        };
        if fill.is_empty() {
            break; // both axes empty — handled by the feature-only check above
        }
        drop_least_important(features, fill);
        dropped += 1;
    }

    let final_cells = cells(x_fill.len(), y_fill.len());
    eprintln!(
        "mr-srj: Hanan grid over budget {CELL_BUDGET}; coalesced/dropped {dropped} fill \
         lines → {}×{} = {final_cells} cells (kept all {}+{} feature lines).",
        x_features.len() + x_fill.len(),
        y_features.len() + y_fill.len(),
        x_features.len(),
        y_features.len(),
    );
}

/// Remove fill lines that are within [`LINE_EPSILON`] of a feature line or of an
/// already-kept fill line — they would merge in `sort_dedup` anyway and so should not
/// count against the budget. `fill` is sorted and deduped against the (sorted)
/// `features` in place.
fn coalesce_fill(features: &[f64], fill: &mut Vec<f64>) {
    sort_dedup(fill);
    fill.retain(|&f| {
        // Keep iff no feature line is within epsilon (binary search the sorted set).
        let i = features.partition_point(|&x| x < f - LINE_EPSILON);
        !(i < features.len() && (features[i] - f).abs() <= LINE_EPSILON)
    });
}

/// Drop the single most-redundant line from sorted `fill`: the fill line sitting in
/// the *tightest* local channel (smallest distance to its nearest neighbour among the
/// merged feature+fill lines). Removing the densest-packed lane sheds a cell while
/// preserving the widest, most useful routing channels. Ties break toward the lower
/// index for determinism.
fn drop_least_important(features: &[f64], fill: &mut Vec<f64>) {
    if fill.is_empty() {
        return;
    }
    // Merge feature+fill once to measure each fill line's neighbour spacing.
    let mut all: Vec<f64> = Vec::with_capacity(features.len() + fill.len());
    all.extend_from_slice(features);
    all.extend_from_slice(fill);
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut best_idx = 0usize; // index into `fill`
    let mut best_gap = f64::INFINITY;
    for (fi, &f) in fill.iter().enumerate() {
        // Neighbour spacing of `f` in the merged set = min distance to the line just
        // below and just above it.
        let pos = all.partition_point(|&x| x < f); // first index >= f (f itself)
        let left = if pos > 0 { f - all[pos - 1] } else { f64::INFINITY };
        // skip past the copy(ies) of f to its upper neighbour
        let mut up = pos + 1;
        while up < all.len() && (all[up] - f).abs() <= LINE_EPSILON {
            up += 1;
        }
        let right = if up < all.len() {
            all[up] - f
        } else {
            f64::INFINITY
        };
        let nn = left.min(right);
        if nn < best_gap {
            best_gap = nn;
            best_idx = fi;
        }
    }
    fill.remove(best_idx);
}

/// Sort `v` ascending and collapse runs of lines closer than [`LINE_EPSILON`] into
/// one (keeping the first). Non-finite values are dropped so they cannot poison the
/// line array. Leaves `v` empty only if every input was non-finite; callers floor
/// the empty case via [`Mapping::from_lines`].
fn sort_dedup(v: &mut Vec<f64>) {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.partial_cmp(b).unwrap()); // finite-only → total order
    v.dedup_by(|a, b| (*a - *b).abs() <= LINE_EPSILON);
}

/// Hard cap on fill lines inserted into any one gap, so a pathologically wide empty
/// board span (e.g. a far-flung connector) cannot explode the line count.
const MAX_FILL_PER_GAP: usize = 256;

/// Compute the routing-channel fill lines for the sorted, deduped `features` line
/// set so a track can run *between* two features without colliding with either. The
/// fill lines are returned separately (NOT merged into `features`) so the budget
/// enforcer can thin them while keeping every feature line.
///
/// The geometry, per gap `g` between two adjacent feature lines (`track_w` = design
/// track width, `clearance` = the copper clearance the rasteriser will actually
/// inflate by, `channel = track_w + 2·clearance` = the minimum gap a centred track
/// needs to keep `clearance` to *both* sides):
///
/// * `g >= channel` — a centred lane keeps ≥ `clearance` to both feature edges, so
///   it is DRC-legal. We always insert the **exact midpoint** (the maximal-clearance
///   position, where the lane is most robust to the non-uniform halo), and for wider
///   gaps additionally subdivide at ~`track_w` pitch so distant features still get
///   multiple parallel routing channels (the "tracks between pins" lines). The free
///   corridor (gap minus the two clearance halos) is what the router actually uses.
/// * `track_w <= g < channel` — a track physically fits but a centred lane cannot
///   keep full `clearance` on *both* sides, so any lane here risks a DRC overlap.
///   We insert **no** lane: the route goes around through wider neighbouring gaps.
/// * `g < track_w` — not even a bare track fits; no lane, route goes around.
fn fill_lines(features: &[f64], track_w: f64, clearance: f64) -> Vec<f64> {
    // `track_w` must be a usable positive spacing; reject non-finite / non-positive.
    if features.len() < 2 || !(track_w.is_finite() && track_w > 0.0) {
        return Vec::new();
    }
    let clearance = if clearance.is_finite() && clearance > 0.0 {
        clearance
    } else {
        0.0
    };
    // Minimum gap for a centred, clearance-legal lane.
    let channel = track_w + 2.0 * clearance;
    // Lane pitch: keep the historical fine fill density (the grid spans open board
    // area at ~`track_w`), which gives the router ample routing room. These are
    // candidate grid nodes, not committed copper, so the router still negotiates real
    // clearance between the tracks that ultimately use them.
    let pitch = track_w.max(LINE_EPSILON);
    let mut fill: Vec<f64> = Vec::new();
    for win in features.windows(2) {
        let (lo, hi) = (win[0], win[1]);
        let gap = hi - lo;
        // No clearance-legal centred lane fits between these two features — a track
        // here could not keep full clearance to both sides, so insert nothing and let
        // the route go around through the wider neighbouring gaps. (Two pad edges this
        // close also have no free corridor between them on the pad layer at all.)
        if gap < channel {
            continue;
        }
        // Subdivide the gap into equal sub-intervals each ≤ `pitch` wide, placing a
        // fill line at every interior boundary. `intervals = ceil(gap/pitch)` is
        // clamped to ≥ 2 so even a gap only just at `channel` gets at least ONE
        // interior lane (when `intervals == 2` that lane is the exact midpoint, the
        // maximal-clearance centred lane a tight-pitch pad-escape route needs). Wider
        // gaps get several parallel lanes so distant features stay finely routable and
        // open board area carries multiple tracks. Lines that land inside a pad's
        // inflated clearance halo are unreachable nodes (the grid marks them blocked) —
        // harmless — while the ones in the free corridor give routing room. The per-gap
        // count is capped so a wide empty span cannot blow up.
        let intervals = ((gap / pitch).ceil() as usize).clamp(2, MAX_FILL_PER_GAP + 1);
        let step = gap / intervals as f64;
        for k in 1..intervals {
            fill.push(lo + step * k as f64);
        }
    }
    fill
}

/// Index of the line in sorted `lines` whose half-open Voronoi region contains
/// `p`. Region `i` is `[ (lines[i-1]+lines[i])/2 , (lines[i]+lines[i+1])/2 )`, so
/// the result is the nearest line with midpoint ties broken toward the lower index.
///
/// A binary search counts the interior region boundaries `≤ p`; that count is the
/// containing region index (each boundary before region `i` is the midpoint
/// `(lines[i-1]+lines[i])/2`). On a uniform cell-centre line set this reproduces
/// `floor((p - origin)/res)` exactly (each region equals the old half-open cell
/// `[origin + i·res, origin + (i+1)·res)`). Non-finite or out-of-range `p` clamps
/// into `[0, len-1]`.
fn nearest_line(lines: &[f64], p: f64) -> u32 {
    debug_assert!(!lines.is_empty());
    if !p.is_finite() {
        // NaN/inf: clamp to the nearest end deterministically.
        return if p > 0.0 { lines.len() as u32 - 1 } else { 0 };
    }
    // Binary search for the number of interior boundaries `<= p`; that count is the
    // index of the containing region. The boundary before region `i` (for i in
    // 1..n) is the midpoint `(lines[i-1]+lines[i])/2`. A region index `idx` is
    // valid iff every such boundary up to `idx` is `<= p`, so we count boundaries
    // satisfying `midpoint <= p` — the predicate is monotone in `i`, so a half-open
    // binary search applies.
    let n = lines.len();
    let (mut lo, mut hi) = (0usize, n - 1);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2); // upper-mid; boundary before region `mid`
        if 0.5 * (lines[mid - 1] + lines[mid]) <= p {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo as u32
}

/// The cell-space form of a [`SimpleRouteJson`] problem: the obstacle grid, the
/// two-point nets to route, and the [`Mapping`] needed to convert results back.
#[derive(Debug, Clone)]
pub struct RasterizedProblem {
    pub grid: Grid,
    pub nets: Vec<NetEndpoints>,
    pub mapping: Mapping,
    /// The board's ordered layer names (index ↔ name). For a single-layer problem
    /// this is `["top"]`. Pass it to [`to_solution_layered`] so emitted vertices
    /// and vias carry the right layer names.
    pub layers: LayerMap,
    /// Exact continuous `(x, y)` of every connection endpoint, keyed by the cell
    /// it rasterised to. Used by [`to_solution`] to snap each trace's first/last
    /// vertex back to the exact port coordinate so connectivity checks
    /// (`distance < 0.01`) pass. If two endpoints collide into one cell only one
    /// survives — the resolution is chosen fine enough that this does not happen.
    pub pin_points: HashMap<CellIdx, (f64, f64)>,
}

/// (B3) Rasterise a continuous tscircuit problem into a cell-space problem.
///
/// See the module docs for the dimension, obstacle-overlap, and k-point
/// decomposition rules.
pub fn rasterize(srj: &SimpleRouteJson, resolution: f64) -> RasterizedProblem {
    // The board's layer axis. `layer_count == 1` yields `["top"]`, so every
    // single-layer construction below collapses onto layer 0 and is byte-identical
    // to the pre-layers path.
    rasterize_with_layers(srj, resolution, LayerMap::standard(srj.layer_count), 0)
}

/// (B3) Rasterise with an explicit [`LayerMap`] — use this when the layer *names*
/// are not the standard `top`/`inner_N`/`bottom` (e.g. a Specctra DSN's `F.Cu` /
/// `B.Cu` stackup), so each [`Point`]/[`Obstacle`]'s named layer resolves to the
/// right grid plane instead of collapsing onto layer 0. The grid is built with
/// `layers.len()` planes; `rasterize` is the standard-naming special case.
///
/// The planar grid is **non-uniform / Hanan**: rather than uniform cells of size
/// `resolution`, [`build_grid_lines`] draws per-axis lines through every pad
/// endpoint and obstacle edge (plus the board bounds and fill channels) so each pad
/// lands on an exact node — this is what removes the pad-exit wiggle (see
/// [`build_grid_lines`] and `trace_route`). `resolution` no longer sizes the cells;
/// it is the fill-channel spacing fallback when the problem omits a track width.
///
/// `clearance_cells` reserves a copper-to-copper clearance halo around every pad.
/// The caller passes it as a count of the *old uniform* cells
/// (`ceil(min_clearance / resolution)`), so its continuous width is
/// `clearance_cells · resolution`; because the Hanan cells are non-uniform, the base
/// grid is grown by that **geometric** distance (see
/// [`mr_grid::GridBuilder::inflate_clearance`]) so a track of another net cannot
/// enter the halo. To preserve own-pad access through that now-inflated region, each
/// net's `passable_pads` is correspondingly expanded to the same geometric halo
/// (line-distance ≤ `clearance_mm` on both axes, on the endpoint's layer) of its own
/// pad cells, so the net can still escape its own pads. `clearance_cells == 0` is a
/// no-op: no inflation, no halo expansion.
pub fn rasterize_with_layers(
    srj: &SimpleRouteJson,
    resolution: f64,
    layers: LayerMap,
    clearance_cells: u32,
) -> RasterizedProblem {
    let layer_count = layers.len();
    // Non-uniform / Hanan grid: build per-axis lines through every pad endpoint and
    // obstacle edge (plus bounds + fill channels) so each pad lands on an exact node
    // — this is what removes the pad-exit wiggle at the source (see `build_grid_lines`
    // and `trace_route`). The design-rule track width / clearance drive the fill
    // channel threshold; both fall back to `resolution` when the problem omits them,
    // so a rule-less fixture still gets sensible routing lanes.
    let track_w = srj.min_trace_width.filter(|w| *w > 0.0).unwrap_or(resolution);
    // Effective copper clearance the rasteriser will actually enforce: the halo width
    // applied below (`clearance_cells · resolution`) — NOT the raw `srj.min_clearance`.
    // The fill-channel policy MUST use this same value, otherwise it would size routing
    // lanes against a clearance the inflation does not match (e.g. when the problem
    // omits a clearance rule but the server still inflates by one trace width), placing
    // "channels" that the inflated pad halos then swallow — a pure-disconnect regression.
    let clearance = (clearance_cells as f64 * resolution).max(0.0);
    // Foreign-copper blocking margin = the **track centreline rule**. A node may host
    // the *centre* of a `track_w`-wide trace; that trace keeps `clearance` to foreign
    // copper only if its centre is at least `clearance + track_w/2` from the foreign
    // edge. The plain `clearance` halo (`inflate_clearance` blocks node *centres* within
    // `clearance`) therefore under-reserves by `track_w/2` whenever `track_w > clearance`
    // — the DEFAULT regime when the problem omits a clearance rule (clearance == 0) yet
    // still has a known trace width. We reserve the extra `track_w/2` so NO routable node
    // (fill OR feature-derived) sits where a centred trace would overlap a pad/obstacle.
    //
    // The extra margin is taken from the *declared* `min_trace_width` only (not the
    // `resolution` fill fallback): when the problem states no track width there is no
    // defined trace to reserve against, and forcing the `resolution`-based fallback here
    // would (a) break the byte-identical contract of `rasterize()`/`clearance_cells == 0`
    // fixtures that omit `minTraceWidth`, and (b) over-block on coarse-resolution boards.
    let track_block_mm = srj
        .min_trace_width
        .filter(|w| *w > 0.0)
        .map_or(0.0, |w| w / 2.0);
    // Total distance every FOREIGN pad/obstacle reserves around itself.
    let block_margin_mm = clearance + track_block_mm;
    let (x_lines, y_lines) = build_grid_lines(srj, layer_count, track_w, clearance);
    let mapping = Mapping::from_lines(x_lines, y_lines, layer_count);
    let mut builder = GridBuilder::new(mapping.dims, 1);

    // Collect every connection endpoint as `(x, y, layer)`. These are the pad
    // centres we must connect; the resolved layer is the named `Point.layer`
    // (default "top" / layer 0). The pad each endpoint sits in IS marked as an
    // obstacle in the base grid (correct DRC model: all pads are obstacles); the
    // router later unmasks each net's own pad cells via `passable_pads` so a net
    // may escape its own pads but cannot run through a foreign net's pad.
    let endpoints: Vec<(f64, f64, u32)> = srj
        .connections
        .iter()
        .flat_map(|conn| {
            conn.points_to_connect
                .iter()
                .map(|p| (p.x, p.y, point_layer(p, &layers)))
        })
        .collect();

    for obs in &srj.obstacles {
        // Continuous box covered by the rect.
        let min_x = obs.center.x - obs.width / 2.0;
        let max_x = obs.center.x + obs.width / 2.0;
        let min_y = obs.center.y - obs.height / 2.0;
        let max_y = obs.center.y + obs.height / 2.0;

        // Lower cell holds the box minimum; upper cell is the last cell whose
        // square overlaps the box. A box edge that lands exactly on a cell
        // boundary does not spill into the next cell (half-open cells).
        let (x0, y0) = mapping.point_to_xy((min_x, min_y));
        let x1 = mapping.x_cell_upper(max_x);
        let y1 = mapping.y_cell_upper(max_y);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        // Place the obstacle on each layer it names; an empty or unknown layer
        // list falls back to ALL layers (so a single-layer "top" fixture blocks
        // layer 0 exactly as before).
        for layer in obstacle_layers(obs, &layers) {
            builder.mark_rect_layer(x0, y0, x1, y1, layer);
        }
    }

    // Record each endpoint's exact continuous coordinate so `to_solution` can
    // snap traces back to the port, keyed by the endpoint's own (layered) cell.
    // Endpoint cells stay obstacles in the base grid; the router unmasks each
    // net's own pad cells per net.
    let mut pin_points: HashMap<CellIdx, (f64, f64)> = HashMap::new();
    for &(px, py, layer) in &endpoints {
        let cell = mapping.point_to_cell_layer((px, py), layer);
        pin_points.insert(cell, (px, py));
    }

    // Via keepout is now enforced in the router's legalization fold (via
    // `ViaModel.keepout`, see `mr_cpu::NegotiatedRouter` / `stamp_owner`), not at
    // rasterise time — committed vias reserve their keepout halo there, so nothing
    // extra is stamped on the base grid here.

    // The continuous grid-line geometry, so the geometric clearance inflation below
    // (and own-pad halo expansion) measure halo widths in real units over the now
    // non-uniform lines, not against an equal-cell assumption.
    let coords = GridCoords::from_lines(mapping.x_lines.clone(), mapping.y_lines.clone());

    // Pad clearance: grow every pad obstacle by the foreign-copper blocking halo so a
    // foreign net's *centred* track cannot overlap it. That halo is the track-centreline
    // distance `block_margin_mm = clearance + track_w/2` (see above) — NOT the bare
    // `clearance`: a node exactly `clearance` from a pad edge would host a centred
    // `track_w` trace whose copper reaches `clearance - track_w/2 < clearance` of the
    // pad (a DRC overlap whenever `track_w > clearance`). The Hanan grid's cells are
    // non-uniform, so we inflate by that geometric distance over `coords` rather than a
    // cell count (matching `mr_grid::inflate_clearance`'s mm contract). Own-pad access
    // through the (now wider) halo is restored below by expanding each net's
    // `passable_pads` to the SAME geometric neighbourhood, so widening the foreign halo
    // does not over-block a net from escaping its own pad. `block_margin_mm == 0`
    // (no clearance rule AND no declared trace width) is a no-op: byte-identical base
    // grid, preserving the `rasterize()` / `clearance_cells == 0` contract.
    if block_margin_mm > 0.0 {
        builder.inflate_clearance(block_margin_mm, &coords);
    }

    let grid = builder.build();
    let nets = decompose_connections(
        &srj.connections,
        &mapping,
        &srj.obstacles,
        &layers,
        block_margin_mm,
    );

    RasterizedProblem {
        grid,
        nets,
        mapping,
        layers,
        pin_points,
    }
}

/// Resolve a [`Point`]'s grid layer from its optional named `layer`, defaulting to
/// layer 0 ("top") when absent or unknown.
fn point_layer(p: &Point, layers: &LayerMap) -> u32 {
    p.layer
        .as_deref()
        .and_then(|name| layers.index_of(name))
        .unwrap_or(0)
}

/// The grid layers an [`Obstacle`] occupies. Each named layer maps via the
/// [`LayerMap`]; an empty list, or a list of only unknown names, falls back to
/// ALL layers. Returned ascending and deduplicated.
fn obstacle_layers(obs: &Obstacle, layers: &LayerMap) -> Vec<u32> {
    let mut out: Vec<u32> = obs
        .layers
        .iter()
        .filter_map(|name| layers.index_of(name))
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        (0..layers.len()).collect()
    } else {
        out
    }
}

/// All grid cells covered by obstacle rect(s) that contain the continuous point
/// `point`, plus the point's own rasterised cell. This is the set of pad cells a
/// net owning this endpoint is allowed to traverse. The cells are returned in
/// ascending [`CellIdx`] order, deduplicated, for deterministic serialisation.
///
/// When `clearance_mm > 0` the set is additionally expanded to the planar
/// **geometric** Chebyshev neighbourhood (line distance `≤ clearance_mm` on both
/// axes, on the endpoint's `layer`) of every own-pad cell — mirroring
/// [`mr_grid::GridBuilder::inflate_clearance`]'s mm-based halo — so the net can still
/// reach and escape its own pad through the inflated clearance halo that now
/// surrounds it. `clearance_mm <= 0` leaves the set unchanged.
fn pad_cells_for_point(
    point: (f64, f64),
    layer: u32,
    mapping: &Mapping,
    obstacles: &[Obstacle],
    clearance_mm: f64,
) -> Vec<CellIdx> {
    let (px, py) = point;
    // Pad cells as (x, y) on this endpoint's layer; we inflate these planar coords
    // before resolving to layered cell indices so the halo stays on `layer`.
    let mut planar: Vec<(u32, u32)> = Vec::new();
    for obs in obstacles {
        let min_x = obs.center.x - obs.width / 2.0;
        let max_x = obs.center.x + obs.width / 2.0;
        let min_y = obs.center.y - obs.height / 2.0;
        let max_y = obs.center.y + obs.height / 2.0;
        if px < min_x || px > max_x || py < min_y || py > max_y {
            continue;
        }
        // Same cell-range logic as the obstacle marking loop.
        let (x0, y0) = mapping.point_to_xy((min_x, min_y));
        let x1 = mapping.x_cell_upper(max_x);
        let y1 = mapping.y_cell_upper(max_y);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                planar.push((x, y));
            }
        }
    }
    // Always include the endpoint's own rasterised cell's planar coordinate.
    let (ex, ey) = mapping.point_to_xy((px, py));
    planar.push((ex, ey));

    // Resolve to layered cells, expanding each pad cell to its *geometric* Chebyshev
    // neighbourhood (every line within `clearance_mm` on both axes) on the endpoint's
    // own layer so the net can traverse the inflated clearance halo around its own
    // pads. The per-axis in-clearance line bands are computed from the line arrays so
    // the halo width is the same continuous distance as `inflate_clearance` no matter
    // how unevenly the Hanan lines fall.
    //
    // CLIP: the clearance-halo expansion must NOT unmask a cell that sits inside a
    // *foreign* pad (an obstacle that does NOT contain this endpoint). On tight-pitch
    // pin columns the own-pad halo otherwise bleeds into the neighbouring pad, letting
    // the net drive straight through it (a DRC overlap). Own-pad cells (resolved
    // above into `planar`) are always kept; only the *expanded* halo ring is clipped.
    // A cell at line position `(lx, ly)` is foreign-blocked iff it lies inside some
    // obstacle rect that does not also contain `(px, py)`.
    let foreign_blocks = |lx: f64, ly: f64| -> bool {
        obstacles.iter().any(|obs| {
            let min_x = obs.center.x - obs.width / 2.0;
            let max_x = obs.center.x + obs.width / 2.0;
            let min_y = obs.center.y - obs.height / 2.0;
            let max_y = obs.center.y + obs.height / 2.0;
            let in_this = lx >= min_x && lx <= max_x && ly >= min_y && ly <= max_y;
            // Foreign iff this cell is inside the rect but the endpoint is not.
            let owns = px >= min_x && px <= max_x && py >= min_y && py <= max_y;
            in_this && !owns
        })
    };
    let own_planar: std::collections::HashSet<(u32, u32)> = planar.iter().copied().collect();
    let mut cells: Vec<CellIdx> = Vec::new();
    let clearance = clearance_mm.max(0.0);
    for &(x, y) in &planar {
        let (x0, x1) = line_band(&mapping.x_lines, x, clearance);
        let (y0, y1) = line_band(&mapping.y_lines, y, clearance);
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                // Keep own-pad cells unconditionally; clip expanded-halo cells that
                // fall inside a foreign pad.
                if !own_planar.contains(&(nx, ny))
                    && foreign_blocks(mapping.x_lines[nx as usize], mapping.y_lines[ny as usize])
                {
                    continue;
                }
                cells.push(mapping.dims.idx3(nx, ny, layer));
            }
        }
    }
    cells.sort_unstable();
    cells.dedup();
    cells
}

/// Inclusive `[lo, hi]` range of indices in sorted `lines` whose position is within
/// `clearance` continuous units of the line at index `seed`. With `clearance == 0`
/// this is just `[seed, seed]`. The in-clearance indices form a contiguous band
/// around `seed` (the lines are sorted), found by walking outward from `seed` until
/// the line distance first exceeds `clearance` — mirroring `mr_grid`'s `line_span`,
/// so the own-pad halo matches the inflated obstacle halo exactly.
fn line_band(lines: &[f64], seed: u32, clearance: f64) -> (u32, u32) {
    let n = lines.len() as u32;
    debug_assert!(n > 0);
    let seed = seed.min(n - 1);
    let pos = lines[seed as usize];
    let mut lo = seed;
    while lo > 0 && (pos - lines[lo as usize - 1]).abs() <= clearance {
        lo -= 1;
    }
    let mut hi = seed;
    while hi + 1 < n && (lines[hi as usize + 1] - pos).abs() <= clearance {
        hi += 1;
    }
    (lo, hi)
}

/// Upper (inclusive) cell index touched by a continuous box that ends at `hi`,
/// over the sorted `lines` of one axis: the largest region whose lower Voronoi
/// boundary is strictly less than `hi`. A box edge that lands exactly on a region
/// boundary does not spill into the next region (half-open cells). On a uniform
/// cell-centre line set this reproduces `ceil((hi-origin)/res) - 1`, clamped into
/// `[0, len-1]`.
///
/// The boundary before region `i` (for i in 1..n) is the midpoint
/// `(lines[i-1]+lines[i])/2`; the predicate `boundary < hi` is monotone in `i`, so
/// a half-open binary search yields the largest region whose boundary still
/// undershoots `hi`.
fn cell_upper(lines: &[f64], hi: f64) -> u32 {
    debug_assert!(!lines.is_empty());
    let n = lines.len();
    if !hi.is_finite() {
        return if hi > 0.0 { n as u32 - 1 } else { 0 };
    }
    let (mut lo, mut hist) = (0usize, n - 1);
    while lo < hist {
        let mid = (lo + hist).div_ceil(2); // boundary before region `mid`
        if 0.5 * (lines[mid - 1] + lines[mid]) < hi {
            lo = mid;
        } else {
            hist = mid - 1;
        }
    }
    lo as u32
}

/// Decompose every connection into chained two-point nets (plan R8).
///
/// Each emitted net's `passable_pads` is the union of the own-pad cells of its
/// src and dst points: the cells of every obstacle rect that contains that
/// endpoint, plus the endpoint's own cell. This lets the router unmask exactly
/// this net's pads while keeping every foreign pad an obstacle.
fn decompose_connections(
    connections: &[Connection],
    mapping: &Mapping,
    obstacles: &[Obstacle],
    layers: &LayerMap,
    clearance_mm: f64,
) -> Vec<NetEndpoints> {
    let mut nets = Vec::new();
    for conn in connections {
        let pts = &conn.points_to_connect;
        if pts.len() < 2 {
            continue;
        }
        let segments = pts.len() - 1;
        for (seg, win) in pts.windows(2).enumerate() {
            let src_layer = point_layer(&win[0], layers);
            let dst_layer = point_layer(&win[1], layers);
            let src = mapping.point_to_cell_layer((win[0].x, win[0].y), src_layer);
            let dst = mapping.point_to_cell_layer((win[1].x, win[1].y), dst_layer);
            let net = if segments == 1 {
                conn.name.clone()
            } else {
                format!("{}#{}", conn.name, seg)
            };
            // Union of the src and dst pad cells (each on its endpoint's layer),
            // sorted + deduped.
            let mut passable_pads = pad_cells_for_point(
                (win[0].x, win[0].y),
                src_layer,
                mapping,
                obstacles,
                clearance_mm,
            );
            passable_pads.extend(pad_cells_for_point(
                (win[1].x, win[1].y),
                dst_layer,
                mapping,
                obstacles,
                clearance_mm,
            ));
            passable_pads.sort_unstable();
            passable_pads.dedup();
            nets.push(NetEndpoints {
                net,
                src,
                dst,
                passable_pads,
            });
        }
    }
    nets
}

// ---------------------------------------------------------------------------
// Output soup (B4)
// ---------------------------------------------------------------------------

/// One element of a routed trace: either a wire segment vertex or a via.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "route_type", rename_all = "snake_case")]
pub enum RoutePoint {
    Wire {
        x: f64,
        y: f64,
        width: f64,
        layer: String,
    },
    Via {
        x: f64,
        y: f64,
        from_layer: String,
        to_layer: String,
    },
}

/// A routed trace in tscircuit solution form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcbTrace {
    #[serde(rename = "type")]
    pub kind: String,
    pub route: Vec<RoutePoint>,
}

impl PcbTrace {
    /// Construct a wire-only `pcb_trace` from the given route points.
    pub fn new(route: Vec<RoutePoint>) -> Self {
        Self {
            kind: "pcb_trace".to_string(),
            route,
        }
    }
}

/// (B4) De-rasterise a routed board into a *single-layer* tscircuit solution soup.
///
/// Backward-compatible entry point: every vertex is emitted on `layer` and no vias
/// are produced. Prefer [`to_solution_layered`] for multi-layer boards — this just
/// delegates to it with a one-element [`LayerMap`] named `layer`, so a single-layer
/// path produces byte-identical output to before vias existed.
pub fn to_solution(
    board: &BoardRoute,
    mapping: &Mapping,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    trace_width: f64,
    layer: &str,
) -> Vec<PcbTrace> {
    let layers = LayerMap::from_names(vec![layer.to_string()]);
    to_solution_layered(board, mapping, pin_points, trace_width, &layers)
}

/// (B4) De-rasterise a routed board into a tscircuit solution soup, emitting vias
/// wherever a path changes layers.
///
/// Each [`mr_core::RouteResult`] becomes one [`PcbTrace`]. Walking the path:
///
/// * Consecutive cells with the *same* planar `(x, y)` but a *different* layer form
///   a vertical (via) run. A maximal contiguous vertical run collapses into ONE
///   [`RoutePoint::Via`] from the run's first layer to its last, named via the
///   [`LayerMap`].
/// * Every planar move emits a [`RoutePoint::Wire`] vertex on its cell's layer.
///
/// The first and last *planar* vertex of every trace are snapped to the exact port
/// coordinate via `pin_points` (keyed by the endpoint cell) so the harness'
/// connectivity check (`distance(port, vertex) < 0.01`) matches. Interior vertices
/// stay at cell centres; an endpoint cell missing from `pin_points` falls back to
/// its cell centre.
///
/// A purely single-layer path emits only `Wire` points on layer 0's name and no
/// vias — byte-identical to [`to_solution`]'s historical output.
pub fn to_solution_layered(
    board: &BoardRoute,
    mapping: &Mapping,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    trace_width: f64,
    layers: &LayerMap,
) -> Vec<PcbTrace> {
    board
        .results
        .iter()
        .map(|result| {
            let route = trace_route(result, mapping, pin_points, trace_width, layers);
            PcbTrace::new(route)
        })
        .collect()
}

/// Build the [`RoutePoint`] route for one path, collapsing vertical (via) runs.
fn trace_route(
    result: &mr_core::RouteResult,
    mapping: &Mapping,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    trace_width: f64,
    layers: &LayerMap,
) -> Vec<RoutePoint> {
    let path = &result.path;
    let last = path.len().saturating_sub(1);
    let mut route: Vec<RoutePoint> = Vec::with_capacity(path.len());
    let mut i = 0;
    while i < path.len() {
        let cell = path[i];
        // Emit a Wire vertex for this cell on its layer. Snap the trace's first
        // and last planar vertex to the exact port coordinate; interior vertices
        // stay at the cell centre.
        let (x, y) = if i == 0 || i == last {
            pin_points
                .get(&cell)
                .copied()
                .unwrap_or_else(|| mapping.cell_center(cell))
        } else {
            mapping.cell_center(cell)
        };
        route.push(RoutePoint::Wire {
            x,
            y,
            width: trace_width,
            layer: layers.name(mapping.dims.layer_of(cell)).to_string(),
        });

        // Detect a maximal vertical run starting at `i`: consecutive cells sharing
        // this cell's planar (x, y) but changing layer. Collapse the whole run into
        // ONE via from the wire we just emitted to the run's final layer, and skip
        // straight to that final cell (its same-(x,y) intermediates carry no wires).
        let (cx, cy) = mapping.dims.xy(cell);
        let mut j = i;
        while j + 1 < path.len() {
            let (nx, ny) = mapping.dims.xy(path[j + 1]);
            if nx == cx && ny == cy && path[j + 1] != path[j] {
                j += 1;
            } else {
                break;
            }
        }
        if j > i {
            let (vx, vy) = mapping.cell_center(cell);
            let from_layer = mapping.dims.layer_of(path[i]);
            let to_layer = mapping.dims.layer_of(path[j]);
            route.push(RoutePoint::Via {
                x: vx,
                y: vy,
                from_layer: layers.name(from_layer).to_string(),
                to_layer: layers.name(to_layer).to_string(),
            });
            // Resume after the run's last cell; that cell's wire is the via's
            // landing and is not re-emitted.
            i = j + 1;
        } else {
            i += 1;
        }
    }
    route
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::RouteResult;


    /// A small but realistic problem: 10×10 board, one rect obstacle, two
    /// connections (one 2-point, one 3-point).
    const SAMPLE: &str = r#"{
        "layerCount": 2,
        "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
        "obstacles": [
            { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
        ],
        "connections": [
            { "name": "VCC", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 1} ] },
            { "name": "GND", "pointsToConnect": [ {"x": 1, "y": 9}, {"x": 5, "y": 9}, {"x": 9, "y": 9} ] }
        ]
    }"#;

    #[test]
    fn deserializes_sample() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(srj.layer_count, 2);
        assert_eq!(srj.obstacles.len(), 1);
        assert_eq!(srj.obstacles[0].kind, "rect");
        assert_eq!(srj.connections.len(), 2);
        assert_eq!(srj.connections[1].points_to_connect.len(), 3);
        assert_eq!(srj.bounds.max_x, 10.0);
    }

    #[test]
    fn rasterize_dims_and_net_count() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        // Hanan grid: per axis the feature lines are bounds {0,10}, endpoints, and
        // obstacle edges {4,6}; then `fill_lines` adds a midpoint lane in every gap
        // ≥ channel (= track_w = resolution = 1.0). With clearance 0 here a unit gap
        // (e.g. 0↔1) now gets its midpoint (0.5) — the `>=` fix that the old `>`
        // threshold dropped — and 3-wide gaps (e.g. 1↔4) get two interior lanes
        // (2,3). x picks up endpoints {1,5,9} → 15 lines; y endpoints are only {1,9}
        // (GND's three pads share y=9) → 13 lines. SAMPLE declares layerCount 2.
        assert_eq!(prob.mapping.dims, Dims::with_layers(15, 13, 2));
        assert_eq!(
            prob.mapping.x_lines,
            vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
        );
        assert_eq!(
            prob.mapping.y_lines,
            vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
        );
        // Every pad endpoint coordinate is still an exact grid line (the whole point of
        // the Hanan grid — fill never displaces a feature line).
        for &c in &[0.0, 1.0, 5.0, 9.0, 10.0, 4.0, 6.0] {
            assert!(prob.mapping.x_lines.iter().any(|&l| (l - c).abs() < 1e-9));
        }
        for &c in &[0.0, 1.0, 9.0, 10.0, 4.0, 6.0] {
            assert!(prob.mapping.y_lines.iter().any(|&l| (l - c).abs() < 1e-9));
        }
        // VCC -> 1 net, GND (3 points) -> 2 nets. Total 3.
        assert_eq!(prob.nets.len(), 3);
    }

    #[test]
    fn rasterize_marks_expected_obstacle_cells() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        // Hanan grid: the obstacle box [4,6] is bounded by NODES at 4 and 6, but the
        // fill lanes 4.5 / 5.5 also fall inside it on the x axis, so the half-open cell
        // mapping covers x lines {4.0,4.5,5.0,5.5,6.0} = cells 5..=9 and y lines
        // {4.0,5.0,6.0} = cells 5..=7. Every cell whose line position lies in [4,6] on
        // both axes is blocked.
        for y in 5..=7 {
            for x in 5..=9 {
                assert!(prob.grid.is_obstacle(d.idx(x, y)), "({x},{y}) blocked");
            }
        }
        // Just outside the box stays passable (lines 3.0 / 7.0 are beyond the edges).
        assert!(!prob.grid.is_obstacle(d.idx(10, 6))); // x line 7.0
        assert!(!prob.grid.is_obstacle(d.idx(4, 6))); // x line 3.0
        assert!(!prob.grid.is_obstacle(d.idx(7, 8))); // y line 7.0
        assert!(!prob.grid.is_obstacle(d.idx(0, 0)));
        // The obstacle names no layers, so it blocks BOTH of SAMPLE's 2 layers: the
        // same block is present on layer 1 at the same planar (x, y).
        assert!(prob.grid.is_obstacle(d.idx3(5, 5, 1)));
        assert!(prob.grid.is_obstacle(d.idx3(9, 7, 1)));
        // 5 x-cells × 3 y-cells = 15 blocked cells per layer × 2 layers.
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 15 * 2);
    }

    #[test]
    fn two_point_connection_keeps_bare_name() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        let vcc: Vec<_> = prob
            .nets
            .iter()
            .filter(|n| n.net.starts_with("VCC"))
            .collect();
        assert_eq!(vcc.len(), 1);
        assert_eq!(vcc[0].net, "VCC");
        // On the Hanan grid the endpoints land on their own feature lines: x line 1.0
        // is index 2 and 9.0 is index 12; y line 1.0 is index 2. So (1,1) -> (2,2) and
        // (9,1) -> (12,2).
        let d = prob.mapping.dims;
        assert_eq!(vcc[0].src, d.idx(2, 2));
        assert_eq!(vcc[0].dst, d.idx(12, 2));
    }

    #[test]
    fn k3_connection_decomposes_into_two_chained_nets() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        let gnd: Vec<_> = prob
            .nets
            .iter()
            .filter(|n| n.net.starts_with("GND"))
            .collect();
        assert_eq!(gnd.len(), 2);
        assert_eq!(gnd[0].net, "GND#0");
        assert_eq!(gnd[1].net, "GND#1");
        let d = prob.mapping.dims;
        // Chain order: (1,9)->(5,9), then (5,9)->(9,9). On the Hanan grid x lines
        // 1.0/5.0/9.0 are indices 2/7/12 and y line 9.0 is index 10.
        assert_eq!(gnd[0].src, d.idx(2, 10));
        assert_eq!(gnd[0].dst, d.idx(7, 10));
        assert_eq!(gnd[1].src, d.idx(7, 10));
        assert_eq!(gnd[1].dst, d.idx(12, 10));
    }

    #[test]
    fn to_solution_emits_cell_center_wire_points() {
        // 10x10 grid at res 1, origin (0,0). 3-cell horizontal path on row 0.
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let path = vec![d.idx(0, 0), d.idx(1, 0), d.idx(2, 0)];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "VCC".into(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let pins = HashMap::new();
        let traces = to_solution(&board, &mapping, &pins, 0.2, "top");
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].kind, "pcb_trace");
        assert_eq!(traces[0].route.len(), 3);
        // Cell centres at res 1, origin 0: (0.5,0.5), (1.5,0.5), (2.5,0.5).
        let expected = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)];
        for (pt, &(ex, ey)) in traces[0].route.iter().zip(expected.iter()) {
            match pt {
                RoutePoint::Wire { x, y, width, layer } => {
                    assert_eq!(*x, ex);
                    assert_eq!(*y, ey);
                    assert_eq!(*width, 0.2);
                    assert_eq!(layer, "top");
                }
                _ => panic!("expected wire point"),
            }
        }
    }

    #[test]
    fn solution_serializes_with_expected_tags() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 4.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path: vec![d.idx(0, 0), d.idx(1, 0)],
                cost: 1,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let pins = HashMap::new();
        let traces = to_solution(&board, &mapping, &pins, 0.1, "top");
        let json = serde_json::to_string(&traces).unwrap();
        assert!(json.contains("\"type\":\"pcb_trace\""), "json: {json}");
        assert!(json.contains("\"route_type\":\"wire\""), "json: {json}");
    }

    #[test]
    fn via_route_point_serializes_snake_case() {
        // Guard the via variant's tag/field naming even though to_solution
        // doesn't emit vias yet.
        let v = RoutePoint::Via {
            x: 1.0,
            y: 2.0,
            from_layer: "top".into(),
            to_layer: "bottom".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("\"route_type\":\"via\""), "json: {json}");
        assert!(json.contains("\"from_layer\":\"top\""), "json: {json}");
        assert!(json.contains("\"to_layer\":\"bottom\""), "json: {json}");
    }

    /// Track B: an obstacle declared on "bottom" only blocks bottom-layer cells,
    /// never the top layer at the same (x, y).
    #[test]
    fn rasterize_obstacle_layer_isolated() {
        let blob = r#"{
            "layerCount": 2,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "layers": ["bottom"], "center": {"x": 5, "y": 5},
                  "width": 2, "height": 2 }
            ],
            "connections": []
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        assert_eq!(d.layers, 2);
        // Hanan lines on the integers 0..=10: rect box [4,6] spans cells x=4..=6,
        // y=4..=6 on the bottom layer (index 1); the top layer stays free.
        for y in 4..=6 {
            for x in 4..=6 {
                assert!(prob.grid.is_obstacle(d.idx3(x, y, 1)), "bottom blocked");
                assert!(!prob.grid.is_obstacle(d.idx3(x, y, 0)), "top free");
            }
        }
        // Exactly the 3x3 block on one layer.
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 9);
    }

    /// Track B: a src Point on "top" resolves to a layer-0 cell; on "bottom" it
    /// resolves to the bottom-layer cell at the same planar (x, y).
    #[test]
    fn rasterize_src_resolves_endpoint_layer() {
        let blob = r#"{
            "layerCount": 2,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [],
            "connections": [
                { "name": "a", "pointsToConnect": [
                    {"x": 1, "y": 1, "layer": "top"},
                    {"x": 9, "y": 9, "layer": "bottom"} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        let net = &prob.nets[0];
        // No obstacles, endpoints (1,1)/(9,9): x lines are bounds {0,10}, endpoints
        // {1,9}, plus fill {0.5, 2..8, 9.5} → 1.0 is index 2 and 9.0 is index 10.
        assert_eq!(net.src, d.idx3(2, 2, 0), "top -> layer 0");
        assert_eq!(net.dst, d.idx3(10, 10, 1), "bottom -> layer 1");
    }

    /// Track B: an obstacle with no `layers` (or unknown names) blocks ALL layers,
    /// so a single-layer "top" fixture is unaffected.
    #[test]
    fn rasterize_obstacle_no_layers_blocks_all() {
        let blob = r#"{
            "layerCount": 3,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
            ],
            "connections": []
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        for l in 0..3 {
            assert!(prob.grid.is_obstacle(d.idx3(4, 4, l)), "layer {l} blocked");
        }
        // Hanan lines on the integers: 3x3 block on each of 3 layers.
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 9 * 3);
    }

    /// Track C: a 2-layer path with exactly one layer transition emits exactly one
    /// Via (with correct from/to names) and Wires on the right layers.
    #[test]
    fn to_solution_layered_emits_one_via() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::with_layers(&bounds, 1.0, 2);
        let d = mapping.dims;
        let layers = LayerMap::standard(2); // ["top","bottom"]
                                            // Move on top to (2,0), via down to bottom, then on bottom to (4,0).
        let path = vec![
            d.idx3(0, 0, 0),
            d.idx3(1, 0, 0),
            d.idx3(2, 0, 0),
            d.idx3(2, 0, 1), // via step
            d.idx3(3, 0, 1),
            d.idx3(4, 0, 1),
        ];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 5,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let pins = HashMap::new();
        let traces = to_solution_layered(&board, &mapping, &pins, 0.2, &layers);
        let route = &traces[0].route;
        // 3 top wires + 1 via + 2 bottom wires = 6.
        assert_eq!(route.len(), 6, "route: {route:?}");
        let vias: Vec<_> = route
            .iter()
            .filter(|p| matches!(p, RoutePoint::Via { .. }))
            .collect();
        assert_eq!(vias.len(), 1, "exactly one via");
        match vias[0] {
            RoutePoint::Via {
                x,
                y,
                from_layer,
                to_layer,
            } => {
                assert_eq!((*x, *y), (2.5, 0.5), "via at the shared cell centre");
                assert_eq!(from_layer, "top");
                assert_eq!(to_layer, "bottom");
            }
            _ => unreachable!(),
        }
        // Wires before the via are on "top"; after are on "bottom".
        let layers_of: Vec<&str> = route
            .iter()
            .filter_map(|p| match p {
                RoutePoint::Wire { layer, .. } => Some(layer.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(layers_of, vec!["top", "top", "top", "bottom", "bottom"]);
    }

    /// Track C: a multi-layer via run (top -> bottom across 3 layers) collapses to
    /// a single Via spanning the full run.
    #[test]
    fn to_solution_layered_collapses_multilayer_via_run() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 4.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::with_layers(&bounds, 1.0, 3);
        let d = mapping.dims;
        let layers = LayerMap::standard(3); // ["top","inner1","bottom"]
        let path = vec![
            d.idx3(0, 0, 0),
            d.idx3(0, 0, 1), // via run start
            d.idx3(0, 0, 2), // via run end (same x,y)
            d.idx3(1, 0, 2),
        ];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 4,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution_layered(&board, &mapping, &HashMap::new(), 0.1, &layers);
        let route = &traces[0].route;
        let vias: Vec<_> = route
            .iter()
            .filter_map(|p| match p {
                RoutePoint::Via {
                    from_layer,
                    to_layer,
                    ..
                } => Some((from_layer.as_str(), to_layer.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(vias, vec![("top", "bottom")], "one via spanning the run");
    }

    /// Track C: a single-layer path through `to_solution` is unchanged — all Wire,
    /// layer "top", no vias (guards byte-identical legacy output).
    #[test]
    fn to_solution_single_layer_unchanged() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let path = vec![d.idx(0, 0), d.idx(1, 0), d.idx(2, 0)];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution(&board, &mapping, &HashMap::new(), 0.2, "top");
        assert_eq!(traces[0].route.len(), 3);
        assert!(traces[0]
            .route
            .iter()
            .all(|p| matches!(p, RoutePoint::Wire { layer, .. } if layer == "top")));
    }

    #[test]
    fn cell_center_and_point_to_cell_roundtrip() {
        let bounds = Bounds {
            min_x: -5.0,
            max_x: 5.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::new(&bounds, 2.0);
        // span 10/2 = 5 wide, 4/2 = 2 tall.
        assert_eq!(mapping.dims, Dims::new(5, 2));
        for i in 0..mapping.dims.len() as u32 {
            let (x, y) = mapping.cell_center(i);
            assert_eq!(mapping.point_to_cell((x, y)), i);
        }
    }

    #[test]
    fn degenerate_bounds_yield_min_grid() {
        let bounds = Bounds {
            min_x: 3.0,
            max_x: 3.0,
            min_y: 3.0,
            max_y: 3.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        assert_eq!(mapping.dims, Dims::new(1, 1));
        // Any point clamps to cell 0.
        assert_eq!(mapping.point_to_cell((100.0, -100.0)), 0);
    }

    /// Phase 1: the uniform constructor populates the line arrays with the historical
    /// cell-centre positions (`origin + (i+0.5)·res`), one per dimension, and
    /// `cell_center` reads straight off them — proving the array model reproduces the
    /// old uniform grid exactly.
    #[test]
    fn uniform_lines_match_old_cell_centres() {
        let bounds = Bounds {
            min_x: -5.0,
            max_x: 5.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::new(&bounds, 2.0);
        assert_eq!(mapping.dims, Dims::new(5, 2));
        assert_eq!(mapping.x_lines.len(), 5);
        assert_eq!(mapping.y_lines.len(), 2);
        // Lines sit at the old centres: origin + (i+0.5)*res.
        assert_eq!(mapping.x_lines, vec![-4.0, -2.0, 0.0, 2.0, 4.0]);
        assert_eq!(mapping.y_lines, vec![1.0, 3.0]);
        // cell_center reads the line positions (no +0.5 on top of them).
        let d = mapping.dims;
        assert_eq!(mapping.cell_center(d.idx(0, 0)), (-4.0, 1.0));
        assert_eq!(mapping.cell_center(d.idx(4, 1)), (4.0, 3.0));
    }

    /// Phase 1: nearest-line `point_to_xy` reproduces the old `floor((p-origin)/res)`
    /// cell assignment on a uniform line set, including the half-open boundary rule
    /// (a point exactly on a cell boundary belongs to the upper cell).
    #[test]
    fn nearest_line_reproduces_floor_mapping() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0); // lines at 0.5,1.5,...,9.5
        let res = 1.0_f64;
        for &p in &[0.0, 0.4, 0.999, 1.0, 1.5, 4.2, 9.49, 9.9, 100.0, -3.0] {
            let expected = {
                let f = ((p - 0.0) / res).floor();
                if !f.is_finite() || f < 0.0 {
                    0
                } else {
                    (f as u32).min(mapping.dims.w - 1)
                }
            };
            assert_eq!(
                mapping.point_to_xy((p, p)).0,
                expected,
                "point_to_xy({p}) must match the old floor mapping"
            );
        }
    }

    /// Phase 1: the line model also works on a *non-uniform* set — `point_to_xy`
    /// picks the nearest line (midpoint ties → lower index) and `cell_upper` honours
    /// the half-open boundary. Guards the generalisation that phases 2–3 rely on.
    #[test]
    fn nonuniform_lines_nearest_and_upper() {
        // Lines at 0, 1, 10 — a wide gap on the right.
        let mapping = Mapping::from_lines(vec![0.0, 1.0, 10.0], vec![0.0, 1.0, 10.0], 1);
        assert_eq!(mapping.dims, Dims::new(3, 3));
        // Boundaries: between 0 and 1 -> 0.5; between 1 and 10 -> 5.5.
        assert_eq!(mapping.point_to_xy((0.4, 0.0)).0, 0);
        assert_eq!(mapping.point_to_xy((0.5, 0.0)).0, 1, "midpoint -> upper index");
        assert_eq!(mapping.point_to_xy((5.4, 0.0)).0, 1);
        assert_eq!(mapping.point_to_xy((5.5, 0.0)).0, 2, "midpoint -> upper index");
        assert_eq!(mapping.point_to_xy((100.0, 0.0)).0, 2, "clamps to last line");
        // cell_upper: a box ending strictly inside region 1 (< 5.5) tops out at 1;
        // ending exactly on a boundary excludes the next region.
        assert_eq!(mapping.x_cell_upper(0.4), 0);
        assert_eq!(mapping.x_cell_upper(0.5), 0, "edge on 0.5 boundary excludes cell 1");
        assert_eq!(mapping.x_cell_upper(0.6), 1);
        assert_eq!(mapping.x_cell_upper(5.5), 1, "edge on 5.5 boundary excludes cell 2");
        assert_eq!(mapping.x_cell_upper(6.0), 2);
    }

    /// (a) Parse the real-harness shape: `minTraceWidth`, obstacle `layers` /
    /// `connectedTo`, and per-point `layer` / `pcb_port_id` (unknown fields are
    /// ignored, not rejected).
    #[test]
    fn parses_real_harness_fields() {
        let blob = r#"{
            "minTraceWidth": 0.1,
            "layerCount": 1,
            "bounds": { "minX": 0, "maxX": 5, "minY": 0, "maxY": 5 },
            "obstacles": [
                { "type": "rect", "layers": ["top"], "center": {"x": 2, "y": 2},
                  "width": 0.38, "height": 0.38, "connectedTo": ["pcb_smtpad_3"] }
            ],
            "connections": [
                { "name": "source_trace_0", "pointsToConnect": [
                    {"x": 1, "y": 1, "layer": "top", "pcb_port_id": "pcb_port_15"},
                    {"x": 4, "y": 4, "layer": "top", "pcb_port_id": "pcb_port_16"} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        assert_eq!(srj.min_trace_width, Some(0.1));
        assert_eq!(srj.obstacles[0].layers, vec!["top".to_string()]);
        assert_eq!(
            srj.obstacles[0].connected_to,
            vec!["pcb_smtpad_3".to_string()]
        );
        let p = &srj.connections[0].points_to_connect[0];
        assert_eq!(p.x, 1.0);
        assert_eq!(p.layer.as_deref(), Some("top"));
    }

    /// (b) An endpoint that sits at the centre of its own pad obstacle leaves the
    /// pad cells marked as obstacles in the BASE grid (correct DRC model), but the
    /// net's `passable_pads` carries the whole pad (and the src cell) so the router
    /// can unmask it. A decoy pad (no endpoint) is in no net's `passable_pads`.
    #[test]
    fn rasterize_pad_is_obstacle_but_in_own_net_passable_pads() {
        // A pad large enough to span >= 3x3 cells centred on the endpoint at
        // (5,5); at res 0.1 a 0.4mm pad spans several cells.
        let blob = r#"{
            "minTraceWidth": 0.1,
            "layerCount": 1,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 0.4, "height": 0.4,
                  "connectedTo": ["pcb_smtpad_0"] },
                { "type": "rect", "center": {"x": 2, "y": 2}, "width": 0.4, "height": 0.4,
                  "connectedTo": ["pcb_smtpad_decoy"] }
            ],
            "connections": [
                { "name": "n", "pointsToConnect": [ {"x": 5, "y": 5}, {"x": 9, "y": 9} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 0.1);
        let d = prob.mapping.dims;
        let src = prob.nets[0].src;
        // Base grid: the pad cell IS an obstacle now.
        assert!(
            prob.grid.is_obstacle(src),
            "pad cell is an obstacle in the base grid"
        );
        // The own pad's cells (and the src cell) are in this net's passable_pads.
        let pads = &prob.nets[0].passable_pads;
        assert!(pads.contains(&src), "src cell must be in passable_pads");
        let (cx, cy) = prob.mapping.point_to_xy((5.0, 5.0));
        // The own pad spans >= 3x3 cells; every one is in passable_pads.
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (x, y) = ((cx as i32 + dx) as u32, (cy as i32 + dy) as u32);
                let cell = d.idx(x, y);
                assert!(
                    prob.grid.is_obstacle(cell),
                    "own pad cell ({x},{y}) is an obstacle in base grid"
                );
                assert!(
                    pads.contains(&cell),
                    "own pad cell ({x},{y}) must be in passable_pads"
                );
            }
        }
        // The decoy pad at (2,2) contains no endpoint -> in NO net's
        // passable_pads, and remains an obstacle.
        let (dx2, dy2) = prob.mapping.point_to_xy((2.0, 2.0));
        let decoy = d.idx(dx2, dy2);
        assert!(prob.grid.is_obstacle(decoy), "decoy pad stays an obstacle");
        for net in &prob.nets {
            assert!(
                !net.passable_pads.contains(&decoy),
                "decoy pad must not be in any net's passable_pads"
            );
        }
    }

    /// A non-connected (decoy) pad must remain an obstacle even when another
    /// net's endpoint lives elsewhere.
    #[test]
    fn rasterize_keeps_decoy_pad_obstacle() {
        let blob = r#"{
            "layerCount": 1,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 1, "height": 1 }
            ],
            "connections": [
                { "name": "n", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 9} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        // Decoy pad centred at (5,5) contains no endpoint -> still blocked. Resolve its
        // cell through the mapping (the Hanan grid does not put it at integer index
        // (5,5) any more).
        let (dx, dy) = prob.mapping.point_to_xy((5.0, 5.0));
        let decoy = prob.mapping.dims.idx(dx, dy);
        assert!(prob.grid.is_obstacle(decoy));
        // And it is in no net's passable_pads.
        for net in &prob.nets {
            assert!(!net.passable_pads.contains(&decoy));
        }
    }

    /// (c) `to_solution` snaps the first and last vertex to the exact port
    /// coordinate carried in `pin_points`, while interior vertices stay at cell
    /// centres.
    #[test]
    fn to_solution_snaps_endpoints_to_exact_ports() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let path = vec![d.idx(0, 0), d.idx(1, 0), d.idx(2, 0)];
        // Exact ports offset from the cell centres (0.5,0.5) / (2.5,0.5).
        let mut pins = HashMap::new();
        pins.insert(d.idx(0, 0), (0.12, 0.34));
        pins.insert(d.idx(2, 0), (2.87, 0.65));
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution(&board, &mapping, &pins, 0.1, "top");
        let r = &traces[0].route;
        match &r[0] {
            RoutePoint::Wire { x, y, .. } => {
                assert_eq!((*x, *y), (0.12, 0.34));
            }
            _ => panic!("wire expected"),
        }
        // Interior vertex stays at the cell centre (1.5, 0.5).
        match &r[1] {
            RoutePoint::Wire { x, y, .. } => {
                assert_eq!((*x, *y), (1.5, 0.5));
            }
            _ => panic!("wire expected"),
        }
        match &r[2] {
            RoutePoint::Wire { x, y, .. } => {
                assert_eq!((*x, *y), (2.87, 0.65));
            }
            _ => panic!("wire expected"),
        }
    }

    /// Pad clearance JSON: two small pads of DIFFERENT nets on a 7×7 board, pad "a"
    /// centred at (1.5,3.5) and pad "b" at (4.5,3.5), each 0.5×0.5. On the Hanan grid
    /// the pad edges + centres become exact lines, so each pad spans a small cell
    /// block centred on its endpoint rather than a single uniform-grid cell.
    const CLEARANCE_SRJ: &str = r#"{
        "layerCount": 1,
        "bounds": { "minX": 0, "maxX": 7, "minY": 0, "maxY": 7 },
        "obstacles": [
            { "type": "rect", "center": {"x": 1.5, "y": 3.5}, "width": 0.5, "height": 0.5,
              "connectedTo": ["pad_a"] },
            { "type": "rect", "center": {"x": 4.5, "y": 3.5}, "width": 0.5, "height": 0.5,
              "connectedTo": ["pad_b"] }
        ],
        "connections": [
            { "name": "a", "pointsToConnect": [ {"x": 1.5, "y": 3.5}, {"x": 1.5, "y": 0.5} ] },
            { "name": "b", "pointsToConnect": [ {"x": 4.5, "y": 3.5}, {"x": 4.5, "y": 0.5} ] }
        ]
    }"#;

    /// `clearance_cells = 1` (→ `clearance_mm = 1·resolution = 1.0` on the Hanan
    /// grid): each pad reserves a geometric clearance halo (foreign tracks can't
    /// enter), while a net's OWN `passable_pads` includes that halo so it can escape.
    /// Asserts the *geometric* invariants — line-distance to the pad cell, not fixed
    /// uniform-grid indices, since the Hanan grid's cells are non-uniform.
    #[test]
    fn rasterize_pad_clearance_reserves_halo_and_unmasks_own() {
        let srj: SimpleRouteJson = serde_json::from_str(CLEARANCE_SRJ).unwrap();
        let prob = rasterize_with_layers(&srj, 1.0, LayerMap::standard(1), 1);
        let d = prob.mapping.dims;
        let clearance = 1.0_f64; // clearance_cells (1) · resolution (1.0)

        // Pad endpoint cells: (1.5,3.5) and (4.5,3.5) land on exact nodes.
        let (ax, ay) = prob.mapping.point_to_xy((1.5, 3.5));
        let (bx, by) = prob.mapping.point_to_xy((4.5, 3.5));
        let pad_a = d.idx(ax, ay);
        let pad_b = d.idx(bx, by);
        // The pads themselves are obstacles in the base grid.
        assert!(prob.grid.is_obstacle(pad_a));
        assert!(prob.grid.is_obstacle(pad_b));

        // Every cell within `clearance` line-distance (both axes) of a pad cell is
        // reserved as an obstacle — the geometric Chebyshev halo. Check pad "a".
        for ny in 0..d.h {
            for nx in 0..d.w {
                let dx = (prob.mapping.x_lines[nx as usize] - prob.mapping.x_lines[ax as usize]).abs();
                let dy = (prob.mapping.y_lines[ny as usize] - prob.mapping.y_lines[ay as usize]).abs();
                if dx <= clearance && dy <= clearance {
                    assert!(
                        prob.grid.is_obstacle(d.idx(nx, ny)),
                        "pad-a halo cell ({nx},{ny}) must be reserved"
                    );
                }
            }
        }

        // Net "a" can escape its OWN pad: its passable_pads includes that same
        // geometric halo (cells that are obstacles in the base grid but unmasked for
        // net "a"). Pick the immediate line neighbours around the pad cell.
        let net_a = prob.nets.iter().find(|n| n.net == "a").unwrap();
        let neighbours = [
            (ax.saturating_sub(1), ay),
            (ax + 1, ay),
            (ax, ay.saturating_sub(1)),
            (ax, ay + 1),
        ];
        for (x, y) in neighbours {
            if x >= d.w || y >= d.h {
                continue;
            }
            let cell = d.idx(x, y);
            assert!(
                net_a.passable_pads.contains(&cell),
                "net a must be able to traverse its own-pad halo at ({x},{y})"
            );
        }
        // Net "a" must NOT be allowed through the foreign pad "b" core.
        assert!(
            !net_a.passable_pads.contains(&pad_b),
            "net a must not be allowed through foreign pad b"
        );
    }

    /// `clearance_cells = 0`: the no-clearance build. Grid + nets match `rasterize`
    /// (no clearance) exactly, and the only obstacles are the two pad blocks with no
    /// inflation halo grown around them.
    #[test]
    fn rasterize_clearance_zero_is_byte_identical() {
        let srj: SimpleRouteJson = serde_json::from_str(CLEARANCE_SRJ).unwrap();
        // `rasterize` (no clearance) and `rasterize_with_layers(.., 0)` must agree.
        let baseline = rasterize(&srj, 1.0);
        let zero = rasterize_with_layers(&srj, 1.0, LayerMap::standard(1), 0);

        assert_eq!(
            baseline.grid.cost, zero.grid.cost,
            "clearance_cells=0 grid must equal the no-clearance grid"
        );
        assert_eq!(
            baseline.nets, zero.nets,
            "clearance_cells=0 nets/passable_pads must equal the no-clearance build"
        );

        // No inflation halo: the obstacle set is exactly the two pad blocks. On the
        // Hanan grid each 0.5×0.5 pad (edges + centre line) spans a 3×3 cell block,
        // so 2 pads = 18 obstacle cells, and they are confined to the pads' rows.
        let d = zero.mapping.dims;
        let obstacles = zero
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(obstacles, 18, "two 3x3 pad blocks, no inflation halo");
        // Confirm there is no halo above/below the pad block: the row two lines above
        // the pad-a top edge is free at the pad's column.
        let (ax, _ay) = zero.mapping.point_to_xy((1.5, 3.5));
        let top_edge_row = zero.mapping.point_to_xy((1.5, 3.25)).1;
        if top_edge_row >= 1 {
            assert!(
                !zero.grid.is_obstacle(d.idx(ax, top_edge_row - 1)),
                "no inflation above the pad block"
            );
        }
    }

    /// Two adjacent pads of DIFFERENT nets where the declared track width EXCEEDS the
    /// clearance (`track_w = 0.3 > clearance = 0.1`) — the regime in which the bare
    /// `clearance` halo under-reserves. Pad "a" at (2.0,2.0) (right edge x=2.3) and pad
    /// "b" at (3.3,2.0) (left edge x=3.0) leave a 0.7 gap between their inner edges.
    /// 0.7 ≥ channel (track_w + 2·clearance = 0.5), so `fill_lines` subdivides it
    /// (intervals = ceil(0.7/0.3) = 3, step ≈ 0.233): the first fill lane lands at
    /// x ≈ 2.533, i.e. 0.233 from pad "a"'s edge — OUTSIDE the bare 0.1 clearance halo
    /// but INSIDE the correct 0.25 (= clearance + track_w/2) centreline margin. That is
    /// exactly the "free lane just outside the clearance-only halo" the review flagged:
    /// a centred 0.3 trace there reaches 0.233 − 0.15 = 0.083 of pad "a" (< 0.1 → DRC
    /// overlap). `minClearance` 0.1, `minTraceWidth` 0.3.
    const TRACK_GT_CLEARANCE_SRJ: &str = r#"{
        "minTraceWidth": 0.3,
        "minClearance": 0.1,
        "layerCount": 1,
        "bounds": { "minX": 0, "maxX": 6, "minY": 0, "maxY": 6 },
        "obstacles": [
            { "type": "rect", "center": {"x": 2.0, "y": 2.0}, "width": 0.6, "height": 0.6,
              "connectedTo": ["pad_a"] },
            { "type": "rect", "center": {"x": 3.3, "y": 2.0}, "width": 0.6, "height": 0.6,
              "connectedTo": ["pad_b"] }
        ],
        "connections": [
            { "name": "a", "pointsToConnect": [ {"x": 2.0, "y": 2.0}, {"x": 2.0, "y": 5.0} ] },
            { "name": "b", "pointsToConnect": [ {"x": 3.3, "y": 2.0}, {"x": 3.3, "y": 5.0} ] }
        ]
    }"#;

    /// REGRESSION (DRC overlap when `track_w > clearance`): the foreign-copper halo
    /// must reserve the **track-centreline** distance `clearance + track_w/2`, not just
    /// `clearance`. Otherwise a *free* (routable) node can sit `clearance` from a pad
    /// edge, and a centred `track_w` trace placed there reaches `clearance - track_w/2`
    /// of the foreign pad — a DRC overlap whenever `track_w > clearance`.
    ///
    /// Assert the physically-correct invariant directly on the rasterised grid: NO
    /// routable (non-obstacle) node lies within `clearance + track_w/2` (line-distance,
    /// both axes) of a FOREIGN pad's copper rect. (A node may be within that distance of
    /// its OWN pad — own-pad escape is allowed via `passable_pads` — so the check skips
    /// nodes that fall inside, or in the reserved margin of, the pad they belong to. We
    /// approximate "foreign" conservatively: a node is offending only if it is free yet
    /// within the margin of a pad rect it is NOT inside.)
    #[test]
    fn track_gt_clearance_reserves_centreline_margin_zero_free_nodes() {
        let srj: SimpleRouteJson = serde_json::from_str(TRACK_GT_CLEARANCE_SRJ).unwrap();
        // Drive the SAME entry the live `/solve` + CLI routing path uses, with a real
        // clearance: clearance_cells = ceil(0.1 / 0.05) = 2 → clearance_mm = 0.1.
        let resolution = 0.05;
        let clearance = 0.1_f64;
        let track_w = 0.3_f64;
        let margin = clearance + track_w / 2.0; // 0.25 — the track-centreline rule
        let clearance_cells = (clearance / resolution).ceil() as u32;
        let prob = rasterize_with_layers(&srj, resolution, LayerMap::standard(1), clearance_cells);
        let d = prob.mapping.dims;

        // Each pad's copper rect (continuous).
        let pads: Vec<(f64, f64, f64, f64)> = srj
            .obstacles
            .iter()
            .map(|o| {
                (
                    o.center.x - o.width / 2.0,
                    o.center.x + o.width / 2.0,
                    o.center.y - o.height / 2.0,
                    o.center.y + o.height / 2.0,
                )
            })
            .collect();

        // Line-distance from a point to a rect's nearest edge (0 if inside).
        let dist_to_rect = |x: f64, y: f64, r: &(f64, f64, f64, f64)| -> f64 {
            let dx = (r.0 - x).max(x - r.1).max(0.0);
            let dy = (r.2 - y).max(y - r.3).max(0.0);
            dx.max(dy) // Chebyshev: the halo is a square (both axes ≤ margin)
        };
        let inside = |x: f64, y: f64, r: &(f64, f64, f64, f64)| -> bool {
            x >= r.0 && x <= r.1 && y >= r.2 && y <= r.3
        };

        let eps = 1e-9;
        let mut offending = 0usize;
        for ny in 0..d.h {
            for nx in 0..d.w {
                let cell = d.idx(nx, ny);
                if prob.grid.is_obstacle(cell) {
                    continue; // reserved node — fine
                }
                let (x, y) = (
                    prob.mapping.x_lines[nx as usize],
                    prob.mapping.y_lines[ny as usize],
                );
                for r in &pads {
                    // Skip the node's OWN pad (escape via passable_pads is permitted).
                    if inside(x, y, r) {
                        continue;
                    }
                    // A free node within `margin` of a FOREIGN pad edge is the bug.
                    if dist_to_rect(x, y, r) < margin - eps {
                        offending += 1;
                    }
                }
            }
        }
        assert_eq!(
            offending, 0,
            "every routable node must be ≥ clearance+track_w/2 ({margin}) from a foreign \
             pad — found {offending} free nodes inside the centreline margin (DRC overlap)"
        );

        // Sanity: the fix must NOT block everything — the board still has routable nodes
        // (otherwise a trivially-empty grid would also pass the loop above).
        let free = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c != mr_core::OBSTACLE)
            .count();
        assert!(free > 0, "fix over-blocked: no routable nodes remain");
    }

    /// The same fixture must also leave each net able to ESCAPE its own pad: the fix
    /// widened the foreign halo, so the own-pad `passable_pads` had to widen to match
    /// (else completion tanks). Assert net "a" can traverse its own-pad neighbourhood.
    #[test]
    fn track_gt_clearance_keeps_own_pad_escapable() {
        let srj: SimpleRouteJson = serde_json::from_str(TRACK_GT_CLEARANCE_SRJ).unwrap();
        let resolution = 0.05;
        let clearance_cells = (0.1_f64 / resolution).ceil() as u32;
        let prob = rasterize_with_layers(&srj, resolution, LayerMap::standard(1), clearance_cells);
        let d = prob.mapping.dims;
        let net_a = prob.nets.iter().find(|n| n.net == "a").unwrap();
        let (ax, ay) = prob.mapping.point_to_xy((2.0, 2.0));
        // The four immediate line neighbours of pad-a's centre cell are inside the
        // (now wider) own-pad halo and must be unmasked for net "a".
        for (x, y) in [
            (ax.saturating_sub(1), ay),
            (ax + 1, ay),
            (ax, ay.saturating_sub(1)),
            (ax, ay + 1),
        ] {
            if x >= d.w || y >= d.h {
                continue;
            }
            assert!(
                net_a.passable_pads.contains(&d.idx(x, y)),
                "net a must traverse its own-pad halo at ({x},{y}) after the widened margin"
            );
        }
        // And never through foreign pad b's core.
        let (bx, by) = prob.mapping.point_to_xy((3.3, 2.0));
        assert!(
            !net_a.passable_pads.contains(&d.idx(bx, by)),
            "net a must not be allowed through foreign pad b"
        );
    }

    /// Phase 3a fixture: an IC pad at an arbitrary sub-grid coordinate plus a target,
    /// a board with a couple of obstacles at off-integer positions, and a real DRC
    /// (track width + clearance). Exercises the Hanan line construction end to end.
    const HANAN_SRJ: &str = r#"{
        "minTraceWidth": 0.15,
        "minClearance": 0.2,
        "layerCount": 1,
        "bounds": { "minX": 0, "maxX": 20, "minY": 0, "maxY": 20 },
        "obstacles": [
            { "type": "rect", "center": {"x": 7.3, "y": 4.1}, "width": 1.2, "height": 0.8 },
            { "type": "rect", "center": {"x": 12.6, "y": 13.9}, "width": 2.0, "height": 1.0 }
        ],
        "connections": [
            { "name": "net1", "pointsToConnect": [ {"x": 3.14, "y": 2.72}, {"x": 16.5, "y": 17.25} ] }
        ]
    }"#;

    /// (a) Every pad-endpoint coordinate maps to a grid node whose `cell_center`
    /// equals the *exact* pad coordinate — the core Hanan invariant that makes the
    /// pin-snap an identity and removes the pad-exit wiggle.
    #[test]
    fn pad_center_maps_to_exact_node() {
        let srj: SimpleRouteJson = serde_json::from_str(HANAN_SRJ).unwrap();
        let prob = rasterize(&srj, 1.0);
        for conn in &srj.connections {
            for p in &conn.points_to_connect {
                let cell = prob.mapping.point_to_cell((p.x, p.y));
                let (cx, cy) = prob.mapping.cell_center(cell);
                assert!(
                    (cx - p.x).abs() <= LINE_EPSILON && (cy - p.y).abs() <= LINE_EPSILON,
                    "pad ({}, {}) must land on an exact node, got cell_center ({cx}, {cy})",
                    p.x,
                    p.y
                );
            }
        }
    }

    /// (c) The Hanan line set passes through every endpoint coordinate and every
    /// obstacle edge (`center ± w/2`, `center ± h/2`), plus the board bounds.
    #[test]
    fn lines_pass_through_endpoints_and_obstacle_edges() {
        let srj: SimpleRouteJson = serde_json::from_str(HANAN_SRJ).unwrap();
        let prob = rasterize(&srj, 1.0);
        let has = |lines: &[f64], v: f64| lines.iter().any(|&l| (l - v).abs() <= LINE_EPSILON);

        // Board bounds.
        assert!(has(&prob.mapping.x_lines, 0.0) && has(&prob.mapping.x_lines, 20.0));
        assert!(has(&prob.mapping.y_lines, 0.0) && has(&prob.mapping.y_lines, 20.0));

        // Every endpoint coordinate is a line on its axis.
        for conn in &srj.connections {
            for p in &conn.points_to_connect {
                assert!(has(&prob.mapping.x_lines, p.x), "x line through endpoint {}", p.x);
                assert!(has(&prob.mapping.y_lines, p.y), "y line through endpoint {}", p.y);
            }
        }

        // Every obstacle edge is a line on its axis.
        for obs in &srj.obstacles {
            for ex in [obs.center.x - obs.width / 2.0, obs.center.x + obs.width / 2.0] {
                assert!(has(&prob.mapping.x_lines, ex), "x line through obstacle edge {ex}");
            }
            for ey in [obs.center.y - obs.height / 2.0, obs.center.y + obs.height / 2.0] {
                assert!(has(&prob.mapping.y_lines, ey), "y line through obstacle edge {ey}");
            }
        }
    }

    /// (b) The first trace segment leaving a pad is collinear with the second — no
    /// dogleg. Because the pad sits on its own node, the `pin_points` snap-back is an
    /// identity, so a path that leaves the pad straight stays straight after
    /// de-rasterisation (the old uniform grid offset the pad from its cell centre,
    /// bending the first segment into an S — that is now gone).
    #[test]
    fn first_segment_leaving_pad_is_collinear() {
        let srj: SimpleRouteJson = serde_json::from_str(HANAN_SRJ).unwrap();
        let prob = rasterize(&srj, 1.0);
        // The src pad cell and its two horizontal neighbours on the same row form a
        // straight exit lane (all share the pad's y line). Build that path manually
        // (mr-srj has no router) and de-rasterise it.
        let src = prob.nets[0].src;
        let (sx, sy) = prob.mapping.dims.xy(src);
        assert!(sx + 2 < prob.mapping.dims.w, "room for a straight exit");
        let d = prob.mapping.dims;
        let path = vec![
            src,
            d.idx(sx + 1, sy),
            d.idx(sx + 2, sy),
        ];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: prob.nets[0].net.clone(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution_layered(
            &board,
            &prob.mapping,
            &prob.pin_points,
            0.15,
            &prob.layers,
        );
        let pts: Vec<(f64, f64)> = traces[0]
            .route
            .iter()
            .map(|p| match p {
                RoutePoint::Wire { x, y, .. } => (*x, *y),
                RoutePoint::Via { x, y, .. } => (*x, *y),
            })
            .collect();
        assert_eq!(pts.len(), 3);
        // First vertex is the exact pad coordinate (snap-back identity).
        let p0 = &srj.connections[0].points_to_connect[0];
        assert!(
            (pts[0].0 - p0.x).abs() <= LINE_EPSILON && (pts[0].1 - p0.y).abs() <= LINE_EPSILON,
            "first vertex must be the exact pad coord, got {:?}",
            pts[0]
        );
        // Collinearity: cross product of (p1-p0) and (p2-p0) is ~0 → no dogleg.
        let cross = (pts[1].0 - pts[0].0) * (pts[2].1 - pts[0].1)
            - (pts[1].1 - pts[0].1) * (pts[2].0 - pts[0].0);
        assert!(
            cross.abs() <= 1e-9,
            "first two segments must be collinear (no pad-exit dogleg), cross = {cross}"
        );
    }

    // ---- FIX 2 (fill coverage) + FIX 3 (cell budget) ----

    /// FIX 2 (A): two adjacent pads whose inner gap comfortably exceeds the routing
    /// channel (`track_w + 2·clearance`) MUST get a fill lane in the free corridor
    /// between them — the previous `gap > channel` (strict) policy could drop the lane
    /// for a gap that exactly equalled the channel, disconnecting the net. The lane
    /// must sit ≥ clearance from both pad edges (so a track on it keeps clearance).
    #[test]
    fn fill_inserts_routable_channel_between_adjacent_pads() {
        let track_w = 0.1;
        let clearance = 0.1;
        let channel = track_w + 2.0 * clearance; // 0.3
                                                  // Two 0.4-wide pads centred at x=0 and x=1.0: inner edges 0.2 and 0.8,
                                                  // gap 0.6 > channel. Free corridor (edges + clearance) is [0.3, 0.7].
        let srj = SimpleRouteJson {
            layer_count: 1,
            min_trace_width: Some(track_w),
            min_clearance: Some(clearance),
            obstacles: vec![
                Obstacle {
                    kind: "rect".into(),
                    center: Point { x: 0.0, y: 0.0, layer: None },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                },
                Obstacle {
                    kind: "rect".into(),
                    center: Point { x: 1.0, y: 0.0, layer: None },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                },
            ],
            connections: vec![],
            bounds: Bounds { min_x: -1.0, max_x: 2.0, min_y: -1.0, max_y: 1.0 },
        };
        let (xs, _ys) = build_grid_lines(&srj, 1, track_w, clearance);
        // At least one fill line falls strictly inside the free corridor [0.3, 0.7]
        // (≥ clearance from both pad edges), so a track of another net can run there.
        let lane = xs.iter().find(|&&x| x > 0.3 + LINE_EPSILON && x < 0.7 - LINE_EPSILON);
        assert!(
            lane.is_some(),
            "expected a routing lane in the [0.3,0.7] corridor between the pads, got {xs:?}"
        );
        // The boundary case: a gap exactly equal to the channel still yields a lane —
        // and there the single subdivision lands on the exact midpoint.
        // A gap only just above the channel still yields a lane (the old `gap > channel`
        // strict skip risked dropping near-boundary channels → disconnect). Pads 0.4
        // wide whose inner edges are 0.32 apart (just over channel 0.30): centres ±0.36.
        let srj2 = SimpleRouteJson {
            obstacles: vec![
                Obstacle {
                    kind: "rect".into(),
                    center: Point { x: -0.36, y: 0.0, layer: None },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                },
                Obstacle {
                    kind: "rect".into(),
                    center: Point { x: 0.36, y: 0.0, layer: None },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                },
            ],
            ..srj.clone()
        };
        let (xs2, _) = build_grid_lines(&srj2, 1, track_w, clearance);
        // Inner edges at -0.16 and 0.16 (gap 0.32 > channel) → a lane in the free
        // corridor [-0.06, 0.06] (≥ clearance from both edges).
        assert!(
            xs2.iter().any(|&x| x.abs() <= 0.06 + LINE_EPSILON),
            "a gap just above the channel must still get a clearance-legal lane, got {xs2:?}"
        );
        assert!(
            channel > 0.0,
            "sanity: channel is the gap that just admits a clearance-legal lane"
        );
    }

    /// FIX 2 (A): two pads closer than `track_w` (no track fits at all) get NO fill
    /// lane between them — a lane there would be a DRC overlap — so the route must go
    /// around. The surrounding (wider) gaps still carry channels.
    #[test]
    fn fill_no_lane_when_pads_too_tight_but_go_around_exists() {
        let track_w = 0.1;
        let clearance = 0.1;
        // Pads 0.4 wide centred at x=0 and x=0.42: inner edges 0.2 and 0.22, gap 0.02
        // (< track_w) — physically unroutable between them.
        let srj = SimpleRouteJson {
            layer_count: 1,
            min_trace_width: Some(track_w),
            min_clearance: Some(clearance),
            obstacles: vec![
                Obstacle {
                    kind: "rect".into(),
                    center: Point { x: 0.0, y: 0.0, layer: None },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                },
                Obstacle {
                    kind: "rect".into(),
                    center: Point { x: 0.42, y: 0.0, layer: None },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                },
            ],
            connections: vec![],
            bounds: Bounds { min_x: -2.0, max_x: 2.0, min_y: -1.0, max_y: 1.0 },
        };
        let (xs, _) = build_grid_lines(&srj, 1, track_w, clearance);
        // No fill line inside the unroutable 0.02-wide gap [0.2, 0.22].
        assert!(
            !xs.iter().any(|&x| x > 0.2 + LINE_EPSILON && x < 0.22 - LINE_EPSILON),
            "must NOT place a lane in a sub-track gap, got {xs:?}"
        );
        // But a go-around lane exists in the wide open board area beyond the pads
        // (e.g. between the right pad edge 0.62 and the +2.0 bound).
        assert!(
            xs.iter().any(|&x| x > 0.62 + LINE_EPSILON && x < 2.0 - LINE_EPSILON),
            "a go-around channel must exist in the open area, got {xs:?}"
        );
    }

    /// FIX 3: a dense pad field whose Hanan + fill grid would exceed [`CELL_BUDGET`]
    /// is brought back UNDER budget by thinning fill lines, while every feature line
    /// (pad edges) is kept. This proves the budget is ENFORCED, not merely warned.
    #[test]
    fn cell_budget_is_enforced_by_thinning_fill() {
        // A field of 0.4mm pads on a 0.5mm pitch, big enough that the fill grid would
        // blow past the budget. Track 0.1 / clearance 0.1 → channel 0.3, so the
        // 0.1-wide inter-pad gaps get no fill but the board interior between rows does.
        let track_w = 0.1;
        let clearance = 0.1;
        let n = 60; // 60×60 pads → ~120 feature lines + dense fill per axis
        let pitch = 0.5;
        let mut obstacles = Vec::new();
        for i in 0..n {
            for j in 0..n {
                obstacles.push(Obstacle {
                    kind: "rect".into(),
                    center: Point {
                        x: i as f64 * pitch,
                        y: j as f64 * pitch,
                        layer: None,
                    },
                    width: 0.4,
                    height: 0.4,
                    layers: vec![],
                    connected_to: vec![],
                });
            }
        }
        let span = (n as f64 - 1.0) * pitch;
        let srj = SimpleRouteJson {
            layer_count: 1,
            min_trace_width: Some(track_w),
            min_clearance: Some(clearance),
            obstacles,
            connections: vec![],
            bounds: Bounds {
                min_x: -1.0,
                max_x: span + 1.0,
                min_y: -1.0,
                max_y: span + 1.0,
            },
        };

        // What the FEATURE lines alone cost (the irreducible floor): pad edges + bounds.
        let mut xf: Vec<f64> = vec![-1.0, span + 1.0];
        for i in 0..n {
            xf.push(i as f64 * pitch - 0.2);
            xf.push(i as f64 * pitch + 0.2);
        }
        sort_dedup(&mut xf);
        let feature_cells = xf.len().saturating_mul(xf.len());
        assert!(
            feature_cells <= CELL_BUDGET,
            "test precondition: feature lines ({}²={feature_cells}) must fit the budget \
             so fill-thinning can succeed",
            xf.len()
        );

        let (xs, ys) = build_grid_lines(&srj, 1, track_w, clearance);
        let cells = xs.len().saturating_mul(ys.len());
        assert!(
            cells <= CELL_BUDGET,
            "budget must be ENFORCED: {}×{} = {cells} cells should be ≤ {CELL_BUDGET}",
            xs.len(),
            ys.len(),
        );
        // Every pad-edge feature line survives the thinning (pads stay on-node).
        for i in 0..n {
            for edge in [i as f64 * pitch - 0.2, i as f64 * pitch + 0.2] {
                assert!(
                    xs.iter().any(|&x| (x - edge).abs() <= LINE_EPSILON),
                    "feature line {edge} must be kept after budget thinning"
                );
            }
        }
    }

    /// FIX 3: when the FEATURE lines alone exceed the budget, thinning fill cannot
    /// help — `build_grid_lines` must still return a usable grid (all features) rather
    /// than panicking or hanging, and the over-budget condition is surfaced (logged).
    #[test]
    fn cell_budget_feature_lines_alone_over_budget_returns_all_features() {
        let track_w = 0.1;
        let clearance = 0.0;
        // 250 pads in a line on each axis with distinct coordinates → ~500 feature
        // lines per axis → 500² = 250k cells from features alone, well over budget.
        let n = 250;
        let mut obstacles = Vec::new();
        for i in 0..n {
            obstacles.push(Obstacle {
                kind: "rect".into(),
                center: Point {
                    x: i as f64,
                    y: i as f64,
                    layer: None,
                },
                width: 0.4,
                height: 0.4,
                layers: vec![],
                connected_to: vec![],
            });
        }
        let srj = SimpleRouteJson {
            layer_count: 1,
            min_trace_width: Some(track_w),
            min_clearance: Some(clearance),
            obstacles,
            connections: vec![],
            bounds: Bounds {
                min_x: -1.0,
                max_x: n as f64,
                min_y: -1.0,
                max_y: n as f64,
            },
        };
        let (xs, ys) = build_grid_lines(&srj, 1, track_w, clearance);
        // Feature lines alone exceed the budget; we still get them all (every pad edge
        // present so pads stay on-node), and fill is fully dropped.
        assert!(xs.len().saturating_mul(ys.len()) > CELL_BUDGET);
        for i in 0..n {
            for edge in [i as f64 - 0.2, i as f64 + 0.2] {
                assert!(
                    xs.iter().any(|&x| (x - edge).abs() <= LINE_EPSILON),
                    "feature line {edge} must always survive"
                );
            }
        }
    }

    /// `enforce_budget` thins fill (keeping every feature line) until the product is
    /// under budget, exercising the coalesce + least-important-drop path directly.
    #[test]
    fn enforce_budget_keeps_features_and_drops_redundant_fill() {
        // Few features, a flood of fill on the x axis only.
        let x_features = vec![0.0, 10.0];
        let y_features = vec![0.0, 1.0, 2.0];
        let mut x_fill: Vec<f64> = (1..=5000).map(|k| k as f64 * 0.001).collect();
        let mut y_fill: Vec<f64> = Vec::new();
        enforce_budget(&x_features, &y_features, &mut x_fill, &mut y_fill);
        let cells = (x_features.len() + x_fill.len()) * (y_features.len() + y_fill.len());
        assert!(
            cells <= CELL_BUDGET,
            "enforce_budget must bring the product under the ceiling, got {cells}"
        );
        // Feature lines are untouched.
        assert_eq!(x_features, vec![0.0, 10.0]);
        assert_eq!(y_features, vec![0.0, 1.0, 2.0]);
    }
}
