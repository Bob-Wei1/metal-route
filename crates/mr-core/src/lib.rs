//! `mr-core` — the metalroute contract crate.
//!
//! This crate holds ONLY shared data types, the [`Router`] trait, the canonical
//! row-major coordinate mapping ([`Dims::idx`] / [`Dims::xy`]), the deterministic
//! [`TieBreak`] rule, and [`RouterError`]. It contains no routing logic.
//!
//! Everything else in the workspace depends on this crate, so its surface is kept
//! deliberately small and stable. Two invariants are load-bearing across crates:
//!
//! 1. **One coordinate mapping.** Cells are addressed by a single row-major
//!    [`CellIdx`] via [`Dims`]. No crate may define its own mapping (see plan R3).
//! 2. **One tie-break.** When several equal-cost expansions compete, every router
//!    (CPU and GPU) resolves the tie identically per [`TieBreak`] (see plan R2).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cost of stepping onto a cell. [`OBSTACLE`] marks an impassable cell.
pub type Cost = u32;

/// A cell address into a row-major grid. `y * width + x`.
pub type CellIdx = u32;

/// Sentinel cost marking an impassable cell.
pub const OBSTACLE: Cost = Cost::MAX;

/// Serde default for [`Dims::layers`]: grids deserialised from pre-layer JSON have
/// no `layers` field and must read back as single-layer.
fn one_layer() -> u32 {
    1
}

/// Grid dimensions plus the *only* sanctioned cell ↔ coordinate mapping.
///
/// The grid is `layers` stacked `w × h` planes. The canonical flat mapping is
/// `idx3(x, y, l) = (l*h + y)*w + x`; for `l == 0` this is exactly the historical
/// 2D mapping `y*w + x`, so every single-layer (`layers == 1`) call site is
/// byte-identical to before layers existed. The planar [`Dims::idx`] / [`Dims::xy`]
/// helpers operate on layer 0 / strip the layer respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dims {
    pub w: u32,
    pub h: u32,
    /// Number of stacked copper planes. Defaults to 1 (single-layer).
    #[serde(default = "one_layer")]
    pub layers: u32,
}

impl Dims {
    /// A single-layer grid. Layer count defaults to 1 so every existing 2D caller
    /// is unchanged.
    pub fn new(w: u32, h: u32) -> Self {
        Self { w, h, layers: 1 }
    }

    /// A multi-layer grid of `layers` stacked `w × h` planes.
    pub fn with_layers(w: u32, h: u32, layers: u32) -> Self {
        Self {
            w,
            h,
            layers: layers.max(1),
        }
    }

    /// Number of cells across every layer.
    pub fn len(&self) -> usize {
        self.w as usize * self.h as usize * self.layers as usize
    }

    /// Number of cells in a single `w × h` plane.
    #[inline]
    pub fn plane(&self) -> usize {
        self.w as usize * self.h as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Canonical row-major index of `(x, y)` on layer 0. Caller guarantees
    /// in-bounds. Equivalent to `idx3(x, y, 0)`.
    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> CellIdx {
        y * self.w + x
    }

    /// Canonical flat index of `(x, y, layer)`: `(layer*h + y)*w + x`. Caller
    /// guarantees in-bounds.
    #[inline]
    pub fn idx3(&self, x: u32, y: u32, layer: u32) -> CellIdx {
        (layer * self.h + y) * self.w + x
    }

    /// Inverse of [`Dims::idx`]: planar `(x, y)` of a cell index, with the layer
    /// stripped. Single-layer callers are unaffected.
    #[inline]
    pub fn xy(&self, i: CellIdx) -> (u32, u32) {
        let plane = self.w * self.h;
        let r = i % plane;
        (r % self.w, r / self.w)
    }

    /// Inverse of [`Dims::idx3`]: `(x, y, layer)` of a cell index.
    #[inline]
    pub fn xyz(&self, i: CellIdx) -> (u32, u32, u32) {
        let plane = self.w * self.h;
        let layer = i / plane;
        let r = i % plane;
        (r % self.w, r / self.w, layer)
    }

    /// Layer of a cell index.
    #[inline]
    pub fn layer_of(&self, i: CellIdx) -> u32 {
        i / (self.w * self.h)
    }

    pub fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.w && y < self.h
    }

    pub fn contains(&self, i: CellIdx) -> bool {
        (i as usize) < self.len()
    }

    /// 4-connected *same-layer* neighbours of `i`, returned in ascending
    /// [`CellIdx`] order so that iteration order is identical everywhere (anchors
    /// the tie-break). Layer-crossing moves are vias — see [`Dims::via_neighbors`].
    pub fn neighbors4(&self, i: CellIdx) -> Vec<CellIdx> {
        let (x, y, l) = self.xyz(i);
        let mut v = Vec::with_capacity(4);
        if y > 0 {
            v.push(self.idx3(x, y - 1, l));
        }
        if x > 0 {
            v.push(self.idx3(x - 1, y, l));
        }
        if x + 1 < self.w {
            v.push(self.idx3(x + 1, y, l));
        }
        if y + 1 < self.h {
            v.push(self.idx3(x, y + 1, l));
        }
        v.sort_unstable();
        v
    }

    /// The adjacent-layer neighbours of `i` at the same `(x, y)` — the cells a via
    /// step may reach. Empty for a single-layer grid. Returned in ascending
    /// [`CellIdx`] order (lower layer first).
    pub fn via_neighbors(&self, i: CellIdx) -> Vec<CellIdx> {
        if self.layers <= 1 {
            return Vec::new();
        }
        let (x, y, l) = self.xyz(i);
        let mut v = Vec::with_capacity(2);
        if l > 0 {
            v.push(self.idx3(x, y, l - 1));
        }
        if l + 1 < self.layers {
            v.push(self.idx3(x, y, l + 1));
        }
        v
    }
}

/// The continuous (mm) positions of a board's grid lines, one sorted array per
/// planar axis — the geometry a router needs to price a move by its real length
/// rather than by a uniform unit hop.
///
/// A cell `(x, y)` sits ON the lines `(x_lines[x], y_lines[y])` (this is the
/// non-uniform / Hanan model: line index == cell coordinate, line position == node
/// coordinate). The geometric length of a planar A* step `a -> b` between two
/// 4-neighbour cells is then `|x_lines[bx] - x_lines[ax]| + |y_lines[by] -
/// y_lines[ay]|` — exactly one of the two terms is non-zero for an orthogonal step.
///
/// `dims.w == x_lines.len()` and `dims.h == y_lines.len()` is the load-bearing
/// invariant; [`mr_srj::Mapping`](../mr_srj/struct.Mapping.html) builds the arrays
/// and satisfies it. A router given no coords falls back to
/// [`GridCoords::uniform`], where every step has unit length, so the geometric cost
/// of a step is a constant `COST_SCALE` — byte-identical to the pre-geometric
/// uniform-hop behaviour.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GridCoords {
    /// Sorted continuous x positions of the grid lines; `len() == dims.w`.
    pub x_lines: Vec<f64>,
    /// Sorted continuous y positions of the grid lines; `len() == dims.h`.
    pub y_lines: Vec<f64>,
}

impl GridCoords {
    /// Build from explicit per-axis line arrays (e.g. an
    /// [`mr_srj::Mapping`](../mr_srj/struct.Mapping.html)'s `x_lines` / `y_lines`).
    /// The caller guarantees they are sorted ascending and match the routed grid's
    /// `dims.w` / `dims.h`.
    pub fn from_lines(x_lines: Vec<f64>, y_lines: Vec<f64>) -> Self {
        Self { x_lines, y_lines }
    }

    /// Uniform unit-spaced coords for `dims`: line `i` sits at continuous position
    /// `i`. Every planar step then has geometric length `1.0`, so a router's
    /// geometric step cost is the constant `COST_SCALE` — reproducing the historical
    /// uniform-hop pricing exactly. This is the default when a router is given no
    /// real geometry.
    pub fn uniform(dims: Dims) -> Self {
        Self {
            x_lines: (0..dims.w).map(|i| i as f64).collect(),
            y_lines: (0..dims.h).map(|i| i as f64).collect(),
        }
    }

    /// Continuous x of column `x`. Falls back to `x` itself (unit spacing) when the
    /// array is shorter than the grid — a defensive guard so a coords/grid size
    /// mismatch degrades to uniform pricing rather than panicking.
    #[inline]
    pub fn x_of(&self, x: u32) -> f64 {
        self.x_lines.get(x as usize).copied().unwrap_or(x as f64)
    }

    /// Continuous y of row `y`. See [`GridCoords::x_of`].
    #[inline]
    pub fn y_of(&self, y: u32) -> f64 {
        self.y_lines.get(y as usize).copied().unwrap_or(y as f64)
    }

    /// Geometric (Manhattan) distance in continuous units between two cells, using
    /// their planar `(x, y)` line positions and ignoring the layer (every layer
    /// shares the planar geometry). For two 4-neighbour cells this is the length of
    /// the single orthogonal step between them.
    #[inline]
    pub fn manhattan_len(&self, dims: Dims, a: CellIdx, b: CellIdx) -> f64 {
        let (ax, ay) = dims.xy(a);
        let (bx, by) = dims.xy(b);
        (self.x_of(ax) - self.x_of(bx)).abs() + (self.y_of(ay) - self.y_of(by)).abs()
    }
}

/// The ordered names of a board's copper layers, index ↔ name. Layer 0 is the top
/// copper, `len()-1` the bottom; inner layers sit between. This is the single place
/// that maps the routing grid's integer layer axis to the layer *names* the
/// tscircuit/DSN world speaks in (`"top"`, `"inner1"`, …, `"bottom"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerMap {
    names: Vec<String>,
}

impl LayerMap {
    /// Build from an explicit ordered list of layer names (e.g. parsed from a DSN
    /// stackup). Empty input is promoted to a single `"top"` layer.
    pub fn from_names(mut names: Vec<String>) -> Self {
        if names.is_empty() {
            names.push("top".to_string());
        }
        Self { names }
    }

    /// The standard tscircuit naming for `count` layers: `["top"]`,
    /// `["top","bottom"]`, or `["top","inner1",…,"inner{n-2}","bottom"]`.
    pub fn standard(count: u32) -> Self {
        let count = count.max(1);
        if count == 1 {
            return Self::from_names(vec!["top".to_string()]);
        }
        let mut names = Vec::with_capacity(count as usize);
        names.push("top".to_string());
        for i in 1..count - 1 {
            names.push(format!("inner{i}"));
        }
        names.push("bottom".to_string());
        Self { names }
    }

    /// Number of layers.
    pub fn len(&self) -> u32 {
        self.names.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Name of layer `idx`, or `"top"` if out of range (defensive).
    pub fn name(&self, idx: u32) -> &str {
        self.names
            .get(idx as usize)
            .map(String::as_str)
            .unwrap_or("top")
    }

    /// Index of a named layer, if present.
    pub fn index_of(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|n| n == name).map(|i| i as u32)
    }
}

/// The fabrication class of a via span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViaClass {
    /// Spans the full stackup (top ↔ bottom).
    Through,
    /// Touches exactly one outer layer (top or bottom) but not both.
    Blind,
    /// Entirely between inner layers.
    Buried,
}

/// Which layer transitions a via may make, what each adjacent step costs, and the
/// keepout a placed via requires. The router treats a layer change as a vertical
/// A* step between adjacent layers; this model gates which steps are legal and
/// prices them. Contiguous vertical runs are later classified into a [`ViaClass`]
/// span for output and DRC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViaModel {
    /// Total layers the model is defined over.
    pub layers: u32,
    /// Cost of a single adjacent-layer step, in the same fixed-point units as a
    /// router's planar step cost. A via that crosses `k` layers pays `k` steps.
    pub step_cost: Cost,
    /// Keepout radius (continuous mm) a placed via reserves around itself on the
    /// layers it passes through — the annular-ring radius the router keeps foreign
    /// copper/halo clear of. `0.0` means "use the board clearance".
    pub keepout_mm: f64,
    /// Legal adjacent steps as inclusive `(lo, hi)` layer pairs with `hi == lo+1`.
    /// `None` means every adjacent step is legal (the through-hole default).
    allowed_steps: Option<Vec<(u32, u32)>>,
}

impl ViaModel {
    /// Default via cost: roughly ten planar steps at the negotiated router's
    /// `SCALE` of 16. Vias are cheap enough to use but never free.
    pub const DEFAULT_STEP_COST: Cost = 160;

    /// Through-hole model over `layers`: every adjacent step legal, default cost,
    /// no extra keepout (`keepout_mm == 0.0` preserves the byte-identical
    /// clearance-off fast path).
    pub fn through_hole(layers: u32) -> Self {
        Self {
            layers: layers.max(1),
            step_cost: Self::DEFAULT_STEP_COST,
            keepout_mm: 0.0,
            allowed_steps: None,
        }
    }

    /// A model permitting only the given inclusive adjacent `(lo, hi)` steps
    /// (`hi == lo+1`). Steps outside the set are forbidden — this is how blind /
    /// buried-only stackups are expressed.
    pub fn with_allowed_steps(layers: u32, step_cost: Cost, steps: Vec<(u32, u32)>) -> Self {
        Self {
            layers: layers.max(1),
            step_cost,
            keepout_mm: 0.0,
            allowed_steps: Some(steps),
        }
    }

    /// Whether a single step between `a` and `b` is a legal via move: the layers
    /// must be adjacent and (if a restricted set is configured) listed.
    pub fn is_step_legal(&self, a: u32, b: u32) -> bool {
        if a.abs_diff(b) != 1 || a >= self.layers || b >= self.layers {
            return false;
        }
        match &self.allowed_steps {
            None => true,
            Some(steps) => {
                let lo = a.min(b);
                let hi = a.max(b);
                steps.iter().any(|&(s, e)| s == lo && e == hi)
            }
        }
    }

    /// Classify a via that spans `[lo, hi]` (inclusive layer indices) over a stack
    /// of `layers`.
    pub fn classify_span(lo: u32, hi: u32, layers: u32) -> ViaClass {
        let touches_top = lo == 0;
        let touches_bottom = hi == layers.saturating_sub(1);
        match (touches_top, touches_bottom) {
            (true, true) => ViaClass::Through,
            (false, false) => ViaClass::Buried,
            _ => ViaClass::Blind,
        }
    }
}

/// A cost grid: row-major `cost`, indexed by [`CellIdx`]. `OBSTACLE` = blocked.
///
/// This is the canonical board representation passed to every [`Router`]. The
/// `mr-grid` crate owns the *construction* of grids (rasterisation, clearance
/// inflation); this type is just the shared data so the trait can name it without
/// creating a dependency cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid {
    pub dims: Dims,
    pub cost: Vec<Cost>,
    /// Layer-local cells where planar copper remains legal but a via annular pad
    /// would violate a static obstacle's clearance. Empty is the legacy all-legal
    /// representation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub via_forbidden: Vec<bool>,
    /// Dependency-inverted board-geometry mask. Empty means no board-outline
    /// contract. Each byte uses [`Grid::BOARD_*`] bits to distinguish trace-node,
    /// via-centre, and directed planar-edge exclusions from ordinary pad obstacles;
    /// own-pad exemptions can therefore never reopen the physical board boundary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub board_constraint: Vec<u8>,
}

impl Grid {
    pub const BOARD_TRACE_NODE: u8 = 1 << 0;
    pub const BOARD_VIA_NODE: u8 = 1 << 1;
    pub const BOARD_EDGE_NEG_Y: u8 = 1 << 2;
    pub const BOARD_EDGE_NEG_X: u8 = 1 << 3;
    pub const BOARD_EDGE_POS_X: u8 = 1 << 4;
    pub const BOARD_EDGE_POS_Y: u8 = 1 << 5;

    /// A grid of `dims` with every cell initialised to `fill`.
    pub fn filled(dims: Dims, fill: Cost) -> Self {
        Self {
            cost: vec![fill; dims.len()],
            dims,
            via_forbidden: Vec::new(),
            board_constraint: Vec::new(),
        }
    }

    #[inline]
    pub fn cost_at(&self, i: CellIdx) -> Cost {
        self.cost[i as usize]
    }

    #[inline]
    pub fn is_obstacle(&self, i: CellIdx) -> bool {
        self.cost[i as usize] == OBSTACLE
    }

    /// Whether a via annular pad may not occupy this layer-local cell. Planar
    /// routing deliberately ignores this mask.
    #[inline]
    pub fn is_via_forbidden(&self, i: CellIdx) -> bool {
        self.via_forbidden.get(i as usize).copied().unwrap_or(false)
    }

    /// Whether the board polygon forbids a trace centre at `i`. Unlike a base
    /// obstacle this is never exempted for an own-pad cell.
    #[inline]
    pub fn is_board_forbidden(&self, i: CellIdx) -> bool {
        self.board_constraint
            .get(i as usize)
            .is_some_and(|mask| mask & Self::BOARD_TRACE_NODE != 0)
    }

    /// Whether the board polygon forbids a via annulus centred at `i`.
    #[inline]
    pub fn is_board_via_forbidden(&self, i: CellIdx) -> bool {
        self.board_constraint
            .get(i as usize)
            .is_some_and(|mask| mask & Self::BOARD_VIA_NODE != 0)
    }

    /// Whether the exact continuous board polygon forbids the adjacent planar
    /// step `u -> v`. Non-planar/non-adjacent pairs conservatively return `true`
    /// when a board mask is active; callers should only ask about planar neighbours.
    #[inline]
    pub fn is_board_planar_step_forbidden(&self, u: CellIdx, v: CellIdx) -> bool {
        if self.board_constraint.is_empty() {
            return false;
        }
        if !self.dims.contains(u) || !self.dims.contains(v) {
            return true;
        }
        let (ux, uy, ul) = self.dims.xyz(u);
        let (vx, vy, vl) = self.dims.xyz(v);
        if ul != vl {
            return true;
        }
        let (source_bit, destination_bit) = if ux == vx && uy == vy + 1 {
            (Self::BOARD_EDGE_NEG_Y, Self::BOARD_EDGE_POS_Y)
        } else if uy == vy && ux == vx + 1 {
            (Self::BOARD_EDGE_NEG_X, Self::BOARD_EDGE_POS_X)
        } else if uy == vy && ux + 1 == vx {
            (Self::BOARD_EDGE_POS_X, Self::BOARD_EDGE_NEG_X)
        } else if ux == vx && uy + 1 == vy {
            (Self::BOARD_EDGE_POS_Y, Self::BOARD_EDGE_NEG_Y)
        } else {
            return true;
        };
        let Some(&source_mask) = self.board_constraint.get(u as usize) else {
            return true;
        };
        let Some(&destination_mask) = self.board_constraint.get(v as usize) else {
            return true;
        };
        source_mask & source_bit != 0 || destination_mask & destination_bit != 0
    }

    pub fn has_board_constraints(&self) -> bool {
        !self.board_constraint.is_empty()
    }

    #[inline]
    pub fn set(&mut self, i: CellIdx, c: Cost) {
        self.cost[i as usize] = c;
    }

    /// True when `cost` and an optional via mask match `dims`.
    pub fn is_well_formed(&self) -> bool {
        self.cost.len() == self.dims.len()
            && (self.via_forbidden.is_empty() || self.via_forbidden.len() == self.dims.len())
            && (self.board_constraint.is_empty() || self.board_constraint.len() == self.dims.len())
    }
}

/// One net to route: a named source/target pair of cells.
///
/// Multi-terminal nets are decomposed into pairs upstream (see plan R8); the
/// router contract is strictly two-point.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NetEndpoints {
    pub net: String,
    pub src: CellIdx,
    pub dst: CellIdx,
    /// Cells this net is permitted to traverse even though they are obstacles in
    /// the base grid — namely this net's *own* pad cells. In the base grid ALL
    /// pads are obstacles; a router unmasks each net's `passable_pads` in its
    /// per-net working grid so the net can escape its own pads but cannot run
    /// through a foreign net's pad. Defaults to empty (no pads), which preserves
    /// behaviour for every construction site that does not set it.
    #[serde(default)]
    pub passable_pads: Vec<CellIdx>,
    /// Raw cells of this net's endpoint-owned pad cores where a via may override
    /// the grid's static via halo. Unlike `passable_pads`, this excludes the broad
    /// trace escape corridor and is clipped against every foreign via halo.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub via_passable_pads: Vec<CellIdx>,
}

/// A routed net: the ordered path of cells from src to dst and its total cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteResult {
    pub net: String,
    pub path: Vec<CellIdx>,
    pub cost: Cost,
}

/// The full result of routing a board.
///
/// `congestion` is per-cell (length == `dims.len()`): how many routed nets occupy
/// each cell. Two results are considered equal by the oracle when total path cost
/// and `congestion` match — not when paths are bit-identical (ties exist).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardRoute {
    pub results: Vec<RouteResult>,
    pub unrouted: Vec<String>,
    pub congestion: Vec<u32>,
    /// The router's ground-truth electrical-net group id for each entry of
    /// `results`, aligned 1:1. Nets sharing a group id were permitted to share /
    /// abut copper (no clearance enforced between them); distinct group ids are
    /// mutually foreign. Empty when a construction site does not track grouping
    /// (e.g. single-net routers, hand-built test literals) — consumers must fall
    /// back to geometric reconstruction in that case. `#[serde(default)]` keeps
    /// older serialized boards readable.
    #[serde(default)]
    pub groups: Vec<u32>,
}

impl BoardRoute {
    /// Sum of every routed net's cost — the headline number the oracle compares.
    pub fn total_cost(&self) -> u64 {
        self.results.iter().map(|r| r.cost as u64).sum()
    }

    /// Build the per-cell congestion vector from a set of routed paths.
    pub fn congestion_from(dims: Dims, results: &[RouteResult]) -> Vec<u32> {
        let mut c = vec![0u32; dims.len()];
        for r in results {
            // Congestion is route occupancy, not visit frequency. Router-produced
            // shortest paths are normally simple, but the contract also accepts
            // hand-built/deserialised results; a loop in one such path must not make
            // that single route look like multiple nets at the repeated cell. Keep
            // the defensive set proportional to this path, not to the whole board:
            // production grids can contain millions of cells.
            let mut seen = std::collections::HashSet::with_capacity(r.path.len());
            for &cell in &r.path {
                let cell = cell as usize;
                if seen.insert(cell) {
                    c[cell] += 1;
                }
            }
        }
        c
    }
}

/// A replayable trace of what the negotiated router did while routing a board,
/// for step-by-step animation in the visualiser webapp.
///
/// It is produced ONLY by `mr_cpu::NegotiatedRouter::route_traced`; the normal
/// `Router::route` path builds nothing and stays byte-identical. Every cell is a
/// [`CellIdx`] into [`RouteTrace::dims`] — the client maps cells to continuous
/// coordinates itself using the board's Hanan line arrays (which the server ships
/// alongside this trace), so the router never has to serialise geometry.
///
/// This type lives in `mr-core` (not `mr-cpu`) deliberately: it crosses the
/// router→server boundary like [`BoardRoute`], references the same [`CellIdx`] /
/// [`Dims`] contract types, and keeps `mr-cpu` free of a serde dependency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTrace {
    /// The grid the trace is addressed against (same `dims` as the routed grid).
    pub dims: Dims,
    /// Per-net static metadata, indexed by net index (aligned with the input
    /// `nets` slice). `len() == number of nets`.
    pub nets: Vec<TracedNet>,
    /// Number of distinct connection groups the router formed.
    pub n_groups: usize,
    /// One entry per negotiation iteration that was recorded (`<= MAX_ITERS`;
    /// unchanged iterations may be pruned). Each is a frame the client renders.
    pub iterations: Vec<IterSnapshot>,
    /// The legalization phase result. `None` only if the router errored before
    /// reaching it (which surfaces as `Err` from `route_traced`, so in practice
    /// this is always `Some` on success).
    pub legalization: Option<LegalizationTrace>,
}

/// Static per-net information for the trace: identity, endpoints, group, and the
/// net's "alone path" (routed by itself on the empty grid) for an ideal-route /
/// ratsnest overlay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedNet {
    pub net: String,
    pub src: CellIdx,
    pub dst: CellIdx,
    /// Connection-group id (nets sharing a group may legally share copper).
    pub group: u32,
    /// The path this net takes alone on the base grid (no other nets present);
    /// empty when the net is individually unroutable. Computed once during
    /// legalization.
    pub alone_path: Vec<CellIdx>,
}

/// One negotiation-iteration boundary: the routed state at the end of an iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterSnapshot {
    /// Zero-based iteration index.
    pub iter: u32,
    /// The present-penalty factor for this iteration (`== 1 + iter`): congestion
    /// sharing gets steadily more expensive as this grows.
    pub pfac: u32,
    /// Current routed path per net (empty == unrouted this iteration), indexed by
    /// net index. This is the frame's geometry.
    pub paths: Vec<Vec<CellIdx>>,
    /// Cells over-used (occupied by >=2 distinct groups) at the end of this
    /// iteration — the cells whose history was just bumped, i.e. the hot spots the
    /// negotiation is trying to relieve.
    pub overused_cells: Vec<CellIdx>,
    /// `false` on the final iteration (converged: cell-disjoint across groups).
    pub any_overuse: bool,
}

/// The legalization phase: the group order chosen, what each candidate order
/// achieved, and the final committed per-net result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalizationTrace {
    /// The group order (a permutation of `0..n_groups`) the multi-order pass
    /// selected as best.
    pub chosen_order: Vec<usize>,
    /// Per-candidate-order evaluation, in candidate order: how many nets each
    /// order routed and its total unit cost. Lets the client show "the router
    /// tried these orders and kept this one".
    pub candidates: Vec<CandidateEval>,
    /// Final committed path per net (`None` == dropped/unrouted), indexed by net.
    pub committed: Vec<Option<Vec<CellIdx>>>,
}

/// One candidate group-order's legalization outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEval {
    pub order: Vec<usize>,
    pub routed: usize,
    pub total_cost: Cost,
}

/// The deterministic tie-break shared by every router so CPU and GPU agree.
///
/// A parallel prefix-min does NOT preserve a sequential tie-break for free
/// (plan M0/R2): implementations must *demonstrate* they reproduce this rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TieBreak {
    /// Among equal-cost choices, prefer the one with the lower [`CellIdx`].
    #[default]
    LowerCellIdx,
}

/// Errors a [`Router`] may surface. Per-net "no path" during multi-net routing is
/// reported via [`BoardRoute::unrouted`], not as an error; these variants are for
/// contract violations and unrecoverable conditions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RouterError {
    #[error("net `{net}`: endpoint out of bounds or on an obstacle")]
    InvalidEndpoint { net: String },
    #[error("grid is malformed: cost length does not match dims")]
    MalformedGrid,
    #[error("rip-up exhausted after {passes} passes ({unrouted} nets unrouted)")]
    RipUpExhausted { passes: u32, unrouted: usize },
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
}

/// The single seam every routing implementation (M1/M2 CPU, M3/M4 Metal) shares,
/// so they are drop-in swappable behind benchmarks, the CLI, and the HTTP server.
pub trait Router {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R3 guard: the canonical mapping round-trips for every cell of several grids.
    #[test]
    fn idx_xy_roundtrip() {
        for &(w, h) in &[(1u32, 1u32), (1, 7), (7, 1), (32, 32), (13, 5), (5, 13)] {
            let d = Dims::new(w, h);
            for i in 0..d.len() as u32 {
                let (x, y) = d.xy(i);
                assert!(
                    d.in_bounds(x, y),
                    "{i} -> ({x},{y}) out of bounds for {d:?}"
                );
                assert_eq!(d.idx(x, y), i, "idx∘xy mismatch for {d:?} at {i}");
            }
            for y in 0..h {
                for x in 0..w {
                    let i = d.idx(x, y);
                    assert_eq!(d.xy(i), (x, y), "xy∘idx mismatch for {d:?} at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn neighbors_are_ascending_and_in_bounds() {
        let d = Dims::new(4, 4);
        let center = d.idx(2, 2);
        let n = d.neighbors4(center);
        assert_eq!(n, vec![d.idx(2, 1), d.idx(1, 2), d.idx(3, 2), d.idx(2, 3)]);
        assert!(n.windows(2).all(|w| w[0] < w[1]), "neighbours must ascend");
        // corner has exactly two neighbours
        assert_eq!(d.neighbors4(d.idx(0, 0)).len(), 2);
    }

    #[test]
    fn idx3_equals_idx_on_layer0() {
        // The 3D mapping must reduce to the historical 2D mapping on layer 0 so
        // single-layer call sites are byte-identical.
        let d = Dims::with_layers(13, 5, 4);
        for y in 0..5 {
            for x in 0..13 {
                assert_eq!(d.idx3(x, y, 0), d.idx(x, y));
            }
        }
    }

    #[test]
    fn idx3_xyz_roundtrip_across_layers() {
        let d = Dims::with_layers(7, 3, 5);
        assert_eq!(d.len(), 7 * 3 * 5);
        for i in 0..d.len() as u32 {
            let (x, y, l) = d.xyz(i);
            assert!(x < 7 && y < 3 && l < 5, "{i} -> ({x},{y},{l}) oob");
            assert_eq!(d.idx3(x, y, l), i);
            assert_eq!(d.xy(i), (x, y), "xy must strip the layer");
            assert_eq!(d.layer_of(i), l);
        }
    }

    #[test]
    fn neighbors4_stay_on_layer() {
        let d = Dims::with_layers(4, 4, 3);
        let c = d.idx3(2, 2, 1);
        let n = d.neighbors4(c);
        assert_eq!(
            n,
            vec![
                d.idx3(2, 1, 1),
                d.idx3(1, 2, 1),
                d.idx3(3, 2, 1),
                d.idx3(2, 3, 1)
            ]
        );
        assert!(n.iter().all(|&v| d.layer_of(v) == 1));
        assert!(n.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn via_neighbors_cross_layers_only() {
        // Single layer: no vias.
        assert!(Dims::new(4, 4).via_neighbors(5).is_empty());
        let d = Dims::with_layers(4, 4, 3);
        // Middle layer reaches both neighbours, same (x,y).
        let mid = d.idx3(1, 1, 1);
        assert_eq!(d.via_neighbors(mid), vec![d.idx3(1, 1, 0), d.idx3(1, 1, 2)]);
        // Top layer reaches only up.
        assert_eq!(d.via_neighbors(d.idx3(1, 1, 0)), vec![d.idx3(1, 1, 1)]);
        // Bottom layer reaches only down.
        assert_eq!(d.via_neighbors(d.idx3(1, 1, 2)), vec![d.idx3(1, 1, 1)]);
    }

    #[test]
    fn layermap_standard_naming() {
        assert_eq!(LayerMap::standard(1).name(0), "top");
        let two = LayerMap::standard(2);
        assert_eq!((two.name(0), two.name(1)), ("top", "bottom"));
        let four = LayerMap::standard(4);
        assert_eq!(
            (four.name(0), four.name(1), four.name(2), four.name(3)),
            ("top", "inner1", "inner2", "bottom")
        );
        assert_eq!(four.index_of("inner2"), Some(2));
        assert_eq!(four.index_of("nope"), None);
    }

    #[test]
    fn viamodel_through_hole_legal_steps_and_classes() {
        let v = ViaModel::through_hole(8);
        assert!(v.is_step_legal(0, 1));
        assert!(v.is_step_legal(7, 6));
        assert!(!v.is_step_legal(0, 2), "non-adjacent is not a single step");
        assert!(!v.is_step_legal(7, 8), "out of range");
        assert_eq!(ViaModel::classify_span(0, 7, 8), ViaClass::Through);
        assert_eq!(ViaModel::classify_span(0, 2, 8), ViaClass::Blind);
        assert_eq!(ViaModel::classify_span(5, 7, 8), ViaClass::Blind);
        assert_eq!(ViaModel::classify_span(3, 5, 8), ViaClass::Buried);
    }

    #[test]
    fn viamodel_restricted_steps_forbid_others() {
        // Blind/buried-only stack: only the 1-2 step is drilled.
        let v = ViaModel::with_allowed_steps(4, 100, vec![(1, 2)]);
        assert!(v.is_step_legal(1, 2));
        assert!(v.is_step_legal(2, 1));
        assert!(!v.is_step_legal(0, 1));
        assert!(!v.is_step_legal(2, 3));
    }

    #[test]
    fn dims_deserialises_without_layers_as_single_layer() {
        // Pre-layer JSON has no `layers` field.
        let d: Dims = serde_json::from_str(r#"{"w":4,"h":3}"#).unwrap();
        assert_eq!(d, Dims::new(4, 3));
        assert_eq!(d.layers, 1);
    }

    #[test]
    fn congestion_counts_overlaps() {
        let d = Dims::new(3, 1);
        let results = vec![
            RouteResult {
                net: "a".into(),
                path: vec![0, 1],
                cost: 2,
            },
            RouteResult {
                net: "b".into(),
                path: vec![1, 2],
                cost: 2,
            },
        ];
        assert_eq!(BoardRoute::congestion_from(d, &results), vec![1, 2, 1]);
    }

    /// `congestion` counts occupying routes, not the number of times a malformed
    /// or deliberately loopy path happens to visit a cell.
    #[test]
    fn congestion_counts_each_route_once_per_cell() {
        let d = Dims::new(4, 1);
        let results = vec![
            RouteResult {
                net: "loopy".into(),
                path: vec![0, 1, 2, 1, 0, 1],
                cost: 5,
            },
            RouteResult {
                net: "other".into(),
                path: vec![1, 2, 3, 2],
                cost: 3,
            },
        ];
        assert_eq!(BoardRoute::congestion_from(d, &results), vec![1, 2, 2, 1]);
    }

    #[test]
    fn small_multilayer_mappings_and_neighbours_match_reference() {
        for layers in 1..=4 {
            for h in 1..=5 {
                for w in 1..=5 {
                    let d = Dims::with_layers(w, h, layers);
                    for l in 0..layers {
                        for y in 0..h {
                            for x in 0..w {
                                let i = d.idx3(x, y, l);
                                assert_eq!(d.xyz(i), (x, y, l));
                                assert_eq!(d.xy(i), (x, y));
                                assert_eq!(d.layer_of(i), l);
                                assert!(d.contains(i));

                                let mut expected_planar = Vec::new();
                                if x > 0 {
                                    expected_planar.push(d.idx3(x - 1, y, l));
                                }
                                if x + 1 < w {
                                    expected_planar.push(d.idx3(x + 1, y, l));
                                }
                                if y > 0 {
                                    expected_planar.push(d.idx3(x, y - 1, l));
                                }
                                if y + 1 < h {
                                    expected_planar.push(d.idx3(x, y + 1, l));
                                }
                                expected_planar.sort_unstable();
                                let actual_planar = d.neighbors4(i);
                                assert_eq!(
                                    actual_planar, expected_planar,
                                    "at ({x},{y},{l}) in {d:?}"
                                );
                                assert!(actual_planar.windows(2).all(|pair| pair[0] < pair[1]));

                                let mut expected_vias = Vec::new();
                                if l > 0 {
                                    expected_vias.push(d.idx3(x, y, l - 1));
                                }
                                if l + 1 < layers {
                                    expected_vias.push(d.idx3(x, y, l + 1));
                                }
                                assert_eq!(d.via_neighbors(i), expected_vias);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gridcoords_uniform_metric_is_symmetric_and_layer_invariant() {
        let d = Dims::with_layers(6, 4, 3);
        let coords = GridCoords::uniform(d);
        assert_eq!(coords.x_lines.len(), d.w as usize);
        assert_eq!(coords.y_lines.len(), d.h as usize);
        for a in 0..d.plane() as u32 {
            for b in 0..d.plane() as u32 {
                let ab = coords.manhattan_len(d, a, b);
                let ba = coords.manhattan_len(d, b, a);
                assert_eq!(ab, ba);
                let (ax, ay) = d.xy(a);
                let (bx, by) = d.xy(b);
                assert_eq!(ab, ax.abs_diff(bx) as f64 + ay.abs_diff(by) as f64);
                assert_eq!(
                    ab,
                    coords.manhattan_len(d, d.idx3(ax, ay, 2), d.idx3(bx, by, 1)),
                    "the planar metric must ignore layer"
                );
            }
        }
    }

    #[test]
    fn serde_defaults_new_collection_fields() {
        let net: NetEndpoints = serde_json::from_str(r#"{"net":"n","src":1,"dst":2}"#).unwrap();
        assert!(net.passable_pads.is_empty());
        assert!(net.via_passable_pads.is_empty());

        let grid: Grid = serde_json::from_str(r#"{"dims":{"w":2,"h":1},"cost":[1,1]}"#).unwrap();
        assert!(grid.via_forbidden.is_empty());

        let net = NetEndpoints {
            net: "roundtrip".into(),
            src: 0,
            dst: 1,
            passable_pads: vec![1, 0],
            via_passable_pads: vec![1],
        };
        assert_eq!(
            serde_json::from_str::<NetEndpoints>(&serde_json::to_string(&net).unwrap()).unwrap(),
            net
        );
        let grid = Grid {
            dims: Dims::new(2, 1),
            cost: vec![1, 1],
            via_forbidden: vec![false, true],
            board_constraint: vec![
                Grid::BOARD_EDGE_POS_X,
                Grid::BOARD_EDGE_NEG_X | Grid::BOARD_VIA_NODE,
            ],
        };
        assert_eq!(
            serde_json::from_str::<Grid>(&serde_json::to_string(&grid).unwrap()).unwrap(),
            grid
        );
        assert!(grid.is_board_planar_step_forbidden(0, 1));
        assert!(grid.is_board_planar_step_forbidden(1, 0));
        assert!(grid.is_board_via_forbidden(1));

        let board: BoardRoute =
            serde_json::from_str(r#"{"results":[],"unrouted":[],"congestion":[0,0]}"#).unwrap();
        assert!(board.groups.is_empty());
    }

    #[test]
    fn board_planar_edge_masks_are_symmetric_from_either_endpoint() {
        let dims = Dims::new(2, 2);
        for (u, v, source_bit, destination_bit) in [
            (
                dims.idx(0, 0),
                dims.idx(1, 0),
                Grid::BOARD_EDGE_POS_X,
                Grid::BOARD_EDGE_NEG_X,
            ),
            (
                dims.idx(0, 0),
                dims.idx(0, 1),
                Grid::BOARD_EDGE_POS_Y,
                Grid::BOARD_EDGE_NEG_Y,
            ),
        ] {
            for (cell, bit) in [(u, source_bit), (v, destination_bit)] {
                let mut grid = Grid::filled(dims, 1);
                grid.board_constraint = vec![0; dims.len()];
                grid.board_constraint[cell as usize] = bit;
                assert!(grid.is_board_planar_step_forbidden(u, v));
                assert!(grid.is_board_planar_step_forbidden(v, u));
            }
        }
    }

    #[test]
    fn total_cost_accumulates_wider_than_cost() {
        let board = BoardRoute {
            results: vec![
                RouteResult {
                    net: "a".into(),
                    path: vec![],
                    cost: u32::MAX,
                },
                RouteResult {
                    net: "b".into(),
                    path: vec![],
                    cost: u32::MAX,
                },
            ],
            unrouted: vec![],
            congestion: vec![],
            groups: vec![],
        };
        assert_eq!(board.total_cost(), 2 * u32::MAX as u64);
    }

    #[test]
    fn layermap_empty_input_and_zero_standard_are_canonical_single_layer() {
        for map in [LayerMap::from_names(vec![]), LayerMap::standard(0)] {
            assert_eq!(map.len(), 1);
            assert!(!map.is_empty());
            assert_eq!(map.name(0), "top");
            assert_eq!(map.index_of("top"), Some(0));
        }
    }

    #[test]
    fn viamodel_legality_is_symmetric_and_adjacent_only() {
        let model = ViaModel::with_allowed_steps(5, 17, vec![(0, 1), (2, 3)]);
        for a in 0..=5 {
            for b in 0..=5 {
                assert_eq!(model.is_step_legal(a, b), model.is_step_legal(b, a));
                let expected = matches!((a.min(b), a.max(b)), (0, 1) | (2, 3)) && a < 5 && b < 5;
                assert_eq!(model.is_step_legal(a, b), expected, "step {a}->{b}");
            }
        }
    }
}
