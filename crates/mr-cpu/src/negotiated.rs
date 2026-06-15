//! `NegotiatedRouter` (Phase 2) — PathFinder-style negotiated-congestion routing.
//!
//! The [`RipUpRouter`](crate::RipUpRouter) routes nets sequentially with a strict
//! priority rule: a lower-index net is never displaced for a higher-index one. Two
//! nets competing for crossing corridors therefore often leave one permanently
//! unrouted even when a disjoint solution exists. This router instead lets every
//! net route greedily on its own *congestion-priced* copy of the grid, then makes
//! shared cells progressively more expensive until the routes separate
//! (negotiated congestion, à la Nair/McMurchie PathFinder).
//!
//! ## Cost model (fixed-point integers, no floats)
//!
//! Base passable cell costs [`SCALE`]. The price net `i` pays to step onto cell
//! `c` is
//!
//! ```text
//! cost(c) = SCALE + history[c] + pfac * SCALE * occ_excl_i(c)
//! ```
//!
//! where `history[c]` is a permanent per-cell aversion accumulated over iterations
//! for cells that were over-used, `occ_excl_i(c)` is how many *other* nets
//! currently occupy `c`, and `pfac` (the present-penalty factor) grows each
//! iteration so sharing becomes steadily more expensive. Costs are capped strictly
//! below [`OBSTACLE`] so a priced cell is never confused with an impassable one.
//!
//! ## Connection groups
//!
//! A multi-terminal connection is decomposed upstream into chained sub-nets named
//! `"<conn>#0"`, `"<conn>#1"`, … These are electrically one net and are *allowed*
//! to share cells. Overuse is therefore measured across connection **groups** (the
//! name prefix before `'#'`): a cell is over-used only when ≥2 distinct groups
//! occupy it.
//!
//! ## Convergence and legalization
//!
//! The negotiation loop runs at most [`MAX_ITERS`] iterations; it stops early when
//! no cell is over-used. Because convergence is not guaranteed within the bound, a
//! final legalization pass commits nets group-by-group, marking already-committed
//! foreign-group cells as hard obstacles and rerouting once if needed. This makes
//! the returned [`BoardRoute`] cell-disjoint across groups even if negotiation did
//! not fully settle. The router NEVER loops unbounded.

use std::collections::HashMap;

use rayon::prelude::*;

use mr_core::{
    BoardRoute, CellIdx, Cost, Grid, GridCoords, NetEndpoints, RouteResult, Router, RouterError,
    ViaModel, OBSTACLE,
};

use crate::dijkstra::{astar_buf, edge_cost, SearchBuf, COST_SCALE};

/// Fixed-point cost scale: the base cost of one unit of planar travel.
///
/// Numerically equal to [`COST_SCALE`] (`16`): a planar step of unit geometric
/// length costs `SCALE`, and on a non-uniform grid a step of length `len` costs
/// `round(len * COST_SCALE)` (see [`edge_cost`]). All congestion penalties below are
/// expressed in multiples of `SCALE`, so they keep their strength *relative to the
/// geometric base step* regardless of the board's physical pitch. On a uniform grid
/// every step has length 1, so the base step is exactly `SCALE` and the whole cost
/// model is byte-identical to the pre-geometric router.
pub const SCALE: Cost = COST_SCALE as Cost;

/// Historical soft clearance penalty (TritonRoute `objCost`-style). Inter-net
/// clearance is now a STRICTLY HARD constraint in the committing legalization pass
/// ([`route_legal`] hard-blocks foreign halo cells), so a net that cannot reach its
/// dst clearance-cleanly is left unrouted/congested rather than violating spacing.
/// This constant is retained for API stability and reference; `route_legal` no
/// longer prices it. The iterative NEGOTIATION phase still uses a separate SOFT
/// pressure ([`CLEARANCE_NEG_WEIGHT`]) to spread nets so legalization can succeed.
#[allow(dead_code)]
pub const CLEARANCE_PENALTY: Cost = 16 * SCALE;

/// Soft clearance weight priced into the NEGOTIATION search (TritonRoute `objCost`
/// analog). Distinct from [`CLEARANCE_PENALTY`] (a legalization-only cost): during
/// negotiation each net additionally pays `pfac * CLEARANCE_NEG_WEIGHT * present_halo[c]`
/// to ENTER a cell `c` that lies inside *another* net's clearance / via-keepout halo
/// (see `present_halo` in [`NegotiatedRouter::route`]). Because the cost scales with
/// `pfac` (which grows each iteration), routing within a neighbour's spacing becomes
/// steadily more expensive, so the negotiation SPREADS nets apart by clearance over
/// iterations rather than relying on legalization alone to enforce it.
///
/// Tunable. Starts at [`SCALE`] (`= 16`): one unit of clearance overlap is priced
/// like one unit of direct cell sharing (`pfac * SCALE * present[c]`), a deliberately
/// soft starting point that the per-iteration `pfac` ramp amplifies. Like every other
/// negotiation cost it is capped strictly below [`OBSTACLE`] so a penalized cell is
/// never confused with an impassable one. When `clearance_cells == 0` AND
/// `via_model.keepout_mm == 0.0` the halo field stays all-zero and this term contributes
/// nothing, keeping the default router byte-identical to the pre-clearance behaviour.
pub const CLEARANCE_NEG_WEIGHT: Cost = SCALE;

/// Maximum negotiation iterations before falling through to legalization.
pub const MAX_ITERS: u32 = 60;

/// Net-count threshold above which the negotiation loop routes its nets in
/// PARALLEL (the snapshot-based Jacobi merge) even when clearance is inactive.
/// Below it, small boards keep the sequential Gauss-Seidel path so the
/// deterministic unit tests and single-layer fixtures stay byte-identical. Real
/// many-net boards (e.g. tscircuit `keyboards`, 40–70 nets) are otherwise
/// single-threaded and dominate wall-clock; routing them across all cores is the
/// difference between ~85 s and a few seconds. The parallel path is deterministic
/// (index-ordered merge) regardless of net count.
const PARALLEL_NEGOTIATION_THRESHOLD: usize = 16;


/// Bounded rip-up-and-reroute budget multipliers (see [`ripup_legalize`]). The
/// global budget caps the total number of rip-up operations; the per-net cap
/// bounds how many times any single net may be displaced. Both guarantee
/// termination — once either is hit, the stage stops ripping and keeps what it has.
const RIPUP_GLOBAL_BUDGET_PER_NET: usize = 20;
const RIPUP_PER_NET_CAP_EXTRA: usize = 4;

/// Per-net committed paths from one legalization pass, in input net order: `Some`
/// when the net was placed, `None` when it could not be (dropped/unrouted).
type Committed = Vec<Option<Vec<CellIdx>>>;

/// A rectangular search window in cell coordinates (inclusive bounds). The
/// per-net A* search is restricted to this box so the explored area scales with a
/// net's local region instead of the whole board — the dominant speedup for the
/// local nets of a bed-of-nails board. A full-board window covers `0..w, 0..h`.
#[derive(Clone, Copy)]
struct Window {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl Window {
    /// The whole board.
    fn full(dims: mr_core::Dims) -> Self {
        Window {
            x0: 0,
            y0: 0,
            x1: dims.w.saturating_sub(1),
            y1: dims.h.saturating_sub(1),
        }
    }

    /// True when cell `c` lies inside the window.
    #[inline]
    fn contains(&self, dims: mr_core::Dims, c: CellIdx) -> bool {
        let (x, y) = dims.xy(c);
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// Build the per-net search window: the bounding box of `{src, dst}` plus every
/// cell in `pads` (the net's own passable pads must be reachable), expanded by a
/// margin so the net can detour locally. `margin = max(16, ceil(0.30 * max(bbox_w,
/// bbox_h)))` cells, then clamped to the board.
fn net_window(dims: mr_core::Dims, src: CellIdx, dst: CellIdx, pads: &[CellIdx]) -> Window {
    let (sx, sy) = dims.xy(src);
    let (dx, dy) = dims.xy(dst);
    let mut x0 = sx.min(dx);
    let mut y0 = sy.min(dy);
    let mut x1 = sx.max(dx);
    let mut y1 = sy.max(dy);
    // The window must include all of the net's own pad cells.
    for &p in pads {
        let (px, py) = dims.xy(p);
        x0 = x0.min(px);
        y0 = y0.min(py);
        x1 = x1.max(px);
        y1 = y1.max(py);
    }
    let bbox_w = x1 - x0;
    let bbox_h = y1 - y0;
    let span = bbox_w.max(bbox_h);
    // ceil(0.30 * span) == (3*span + 9) / 10.
    let margin = 16u32.max((3 * span).div_ceil(10));
    Window {
        x0: x0.saturating_sub(margin),
        y0: y0.saturating_sub(margin),
        x1: (x1 + margin).min(dims.w.saturating_sub(1)),
        y1: (y1 + margin).min(dims.h.saturating_sub(1)),
    }
}

/// O(1)-membership set over board cells using the same generation-stamp trick as
/// [`crate::dijkstra::SearchBuf`]: a cell is a member iff `stamp[c] == gen`. Reset
/// is O(1) (bump `gen`); marking the current net's pads is O(pads). Replaces the
/// per-net `Vec::contains` / `HashSet` membership in the hot loop.
struct PadSet {
    stamp: Vec<u32>,
    gen: u32,
}

impl PadSet {
    fn new(n: usize) -> Self {
        PadSet {
            stamp: vec![0; n],
            gen: 0,
        }
    }

    /// Clear all membership in O(1) and load `pads` as the current set in
    /// O(pads). Handles `gen` wraparound by zeroing the stamps.
    fn load(&mut self, pads: &[CellIdx]) {
        match self.gen.checked_add(1) {
            Some(g) => self.gen = g,
            None => {
                self.stamp.iter_mut().for_each(|s| *s = 0);
                self.gen = 1;
            }
        }
        for &p in pads {
            self.stamp[p as usize] = self.gen;
        }
    }

    #[inline]
    fn contains(&self, c: CellIdx) -> bool {
        self.stamp[c as usize] == self.gen
    }
}

/// PathFinder-style negotiated-congestion router. Default multi-net backend.
///
/// `via_model` gates and prices layer changes. When `None` (the default), a
/// [`ViaModel::through_hole`] over the grid's layer count is synthesised at route
/// time, so a multi-layer board uses unrestricted through-hole vias unless the
/// caller supplies a restricted model via [`NegotiatedRouter::with_via_model`]. On
/// a single-layer board the model is inert (no via neighbours exist).
#[derive(Debug, Default, Clone)]
pub struct NegotiatedRouter {
    via_model: Option<ViaModel>,
    /// Continuous grid-line geometry used to price a planar step by its real length
    /// (`round(len * COST_SCALE)`) instead of a uniform unit hop. `None` (the
    /// default) means "no geometry supplied": the search falls back to
    /// [`GridCoords::uniform`] over the grid's `dims`, where every step has unit
    /// length and the base step is exactly [`SCALE`] — byte-identical to the
    /// pre-geometric router. On a non-uniform / Hanan grid the caller supplies the
    /// board's line arrays via [`NegotiatedRouter::with_coords`].
    coords: Option<GridCoords>,
    /// Geometric clearance distance (continuous units, e.g. mm) reserved around
    /// committed copper during legalization to keep *other* nets away — a net's
    /// committed track owns not just its path cells but a halo extending this far on
    /// each cell's own layer, measured over the [`coords`](NegotiatedRouter#structfield.coords)
    /// line arrays, so foreign nets must stay at least this far from it (minimum
    /// spacing). `0.0` = disabled, i.e. the original behaviour (only the path cells
    /// are owned). The halo never overwrites a cell already owned by another group
    /// and never claims a base-obstacle cell, so it cannot wall a foreign net off
    /// from its own pad.
    ///
    /// On a [`GridCoords::uniform`] grid (unit-spaced lines) a clearance of `n` is
    /// byte-identical to the former `n`-cell Chebyshev radius, so
    /// [`with_clearance_cells`](NegotiatedRouter::with_clearance_cells) (which sets
    /// this to `n as f64`) reproduces the pre-geometric tests exactly. On a
    /// non-uniform / Hanan grid the geometric distance is what keeps real
    /// copper-to-copper spacing (a cell count there spans a variable physical width).
    clearance_mm: f64,
}

impl NegotiatedRouter {
    pub fn new() -> Self {
        Self {
            via_model: None,
            coords: None,
            clearance_mm: 0.0,
        }
    }

    /// Use an explicit [`ViaModel`] (e.g. a blind/buried stackup) instead of the
    /// default through-hole model. Builder-style; returns the configured router.
    pub fn with_via_model(mut self, vm: ViaModel) -> Self {
        self.via_model = Some(vm);
        self
    }

    /// Supply the board's continuous grid-line geometry so planar steps are priced
    /// by their real length (`round(len * COST_SCALE)`) rather than as uniform unit
    /// hops. Builder-style; returns the configured router. Without it the router
    /// uses [`GridCoords::uniform`] and is byte-identical to the pre-geometric
    /// behaviour. See the [`coords`](NegotiatedRouter#structfield.coords) field.
    pub fn with_coords(mut self, coords: GridCoords) -> Self {
        self.coords = Some(coords);
        self
    }

    /// Set the geometric clearance distance (continuous units, e.g. mm) reserved
    /// around committed copper, so distinct nets keep at least this much spacing,
    /// measured over the supplied [`coords`](NegotiatedRouter::with_coords) line
    /// arrays. Builder-style; returns the configured router. `0.0` (the default)
    /// disables the halo and reproduces the byte-identical pre-clearance behaviour.
    /// See the [`clearance_mm`](NegotiatedRouter#structfield.clearance_mm) field.
    pub fn with_clearance_mm(mut self, mm: f64) -> Self {
        self.clearance_mm = mm.max(0.0);
        self
    }

    /// Back-compat shim for callers/tests that express clearance as a cell count on a
    /// uniform grid: a radius of `n` cells equals a geometric distance of `n` over
    /// [`GridCoords::uniform`]'s unit-spaced lines, so this is exactly
    /// [`with_clearance_mm`](NegotiatedRouter::with_clearance_mm)`(n as f64)` and
    /// stays byte-identical to the former Chebyshev halo on a uniform grid. On a
    /// non-uniform grid prefer `with_clearance_mm` with the real distance.
    pub fn with_clearance_cells(mut self, n: u32) -> Self {
        self.clearance_mm = n as f64;
        self
    }
}

/// The connection group of a net name: the prefix before the first `'#'`. Chained
/// sub-nets of one connection share a group and may legally share cells.
fn group_of(name: &str) -> &str {
    match name.find('#') {
        Some(i) => &name[..i],
        None => name,
    }
}

/// Cost of a path on the ORIGINAL unit grid: number of steps (cells excluding the
/// source), matching how [`LeeRouter`](crate::LeeRouter) and
/// [`RipUpRouter`](crate::RipUpRouter) report cost. Never the inflated congestion
/// price.
fn unit_cost(path: &[CellIdx]) -> Cost {
    path.len().saturating_sub(1) as Cost
}

/// Lower bound on the planar A* cost between two cells in fixed-point [`Cost`] units,
/// plus the layer distance priced at the cheapest legal via step (`min_via_cost`).
///
/// # Admissibility (why a *sum of per-gap* `edge_cost`, not one `edge_cost` of the sum)
///
/// The search pays a planar step `u -> v` a base of `edge_cost(gap_uv) =
/// round(COST_SCALE * gap_uv)` (plus non-negative congestion). So the planar base paid
/// along ANY real path from `a` to `b` is `Σ_steps round(COST_SCALE * g_i)` over its
/// per-step line gaps `g_i >= 0`. A heuristic is admissible iff it never exceeds that
/// minimum.
///
/// The previous form `edge_cost(manhattan_len(a, b)) = round(COST_SCALE * Σ_axis Δ)`
/// rounds the AGGREGATE length once. Because `round(Σ x_i)` can exceed `Σ round(x_i)`
/// (round-of-sum > sum-of-rounds — e.g. two gaps of `1.5/16` each round to `0`, but
/// their sum `3/16` rounds to `0`… and conversely `0.5/16` twice rounds to `0+0=0`
/// while their sum `1/16` rounds to `1`), the old heuristic could OVERESTIMATE the
/// per-step total the search actually pays on non-integer line spacings, making A*
/// inadmissible and letting `astar_buf`'s "break on pop dst" return a non-optimal path.
///
/// The fix sums `edge_cost` over EACH grid-line gap along the two straight axis legs
/// from `a` to `b`: `Σ_{x lines in [ax,bx)} edge_cost(gap) + Σ_{y lines in [ay,by)}
/// edge_cost(gap)`. This is EXACTLY the planar base the search pays on a monotone
/// (staircase) path between the cells, and any non-monotone path covers a superset of
/// gaps (it must retrace), so it can only cost more. Hence this sum is `<=` the planar
/// base of every real path — a guaranteed lower bound, in the same fixed-point units
/// as the edge cost (the admissibility requirement for A* optimality). The via term is
/// already admissible (`>= min_via_cost` per layer change) and is preserved as-is.
///
/// On a uniform grid every gap is `1.0`, `edge_cost(1.0) == COST_SCALE`, and the sum is
/// `COST_SCALE * (dx + dy)` — byte-identical to the historical `(dx + dy) * SCALE` and
/// to the old `edge_cost(manhattan_len)` there (the round-of-sum / sum-of-rounds gap
/// only opens on non-integer spacings). The layer term is always 0 on a single-layer
/// grid.
fn manhattan_scaled(
    dims: mr_core::Dims,
    coords: &GridCoords,
    a: CellIdx,
    b: CellIdx,
    min_via_cost: Cost,
) -> Cost {
    let (ax, ay) = dims.xy(a);
    let (bx, by) = dims.xy(b);
    let planar = axis_leg_cost(&coords.x_lines, ax, bx)
        .saturating_add(axis_leg_cost(&coords.y_lines, ay, by));
    let dl = dims.layer_of(a).abs_diff(dims.layer_of(b));
    planar.saturating_add(dl.saturating_mul(min_via_cost))
}

/// Sum of per-gap `edge_cost`s along one axis from line index `i` to line index `j`
/// (order-independent) — i.e. `Σ edge_cost(|lines[k+1] - lines[k]|)` over every adjacent
/// line gap strictly between the two indices. This is precisely the base cost the search
/// accrues stepping straight along that axis, the per-step rounding the aggregate-length
/// heuristic failed to account for. Out-of-range / mismatched indices fall back to unit
/// spacing via [`GridCoords::x_of`]-style `.get(...).unwrap_or` semantics, mirrored here
/// so a coords/grid size mismatch degrades to uniform pricing rather than panicking.
#[inline]
fn axis_leg_cost(lines: &[f64], i: u32, j: u32) -> Cost {
    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
    let mut total: Cost = 0;
    for k in lo..hi {
        let a = lines.get(k as usize).copied().unwrap_or(k as f64);
        let b = lines.get(k as usize + 1).copied().unwrap_or((k + 1) as f64);
        total = total.saturating_add(edge_cost((b - a).abs()));
    }
    total
}

impl Router for NegotiatedRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        for net in nets {
            // An endpoint is invalid only if out of bounds, or it sits on an
            // obstacle that is NOT one of this net's own (passable) pad cells.
            let endpoint_invalid = |c: CellIdx| {
                !grid.dims.contains(c) || (grid.is_obstacle(c) && !net.passable_pads.contains(&c))
            };
            if endpoint_invalid(net.src) || endpoint_invalid(net.dst) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
        }

        let dims = grid.dims;
        let n_cells = dims.len();
        let n_nets = nets.len();

        // Effective via model: the caller's, or a through-hole model over the grid's
        // layer count. On a single-layer grid it is inert (no via neighbours exist),
        // so the search stays byte-identical to the planar router.
        let via_model = self
            .via_model
            .clone()
            .unwrap_or_else(|| ViaModel::through_hole(dims.layers));

        // Effective grid geometry: the caller's line arrays, or a uniform unit grid
        // over `dims`. The uniform fallback prices every planar step at exactly
        // `SCALE`, so a router given no coords is byte-identical to the pre-geometric
        // behaviour. Referenced by `&` in every per-net search closure below.
        let coords = self
            .coords
            .clone()
            .unwrap_or_else(|| GridCoords::uniform(dims));

        // Reusable search workspace and own-pad membership set, sized once to the
        // board and reused across every per-net search (no per-net O(n) work).
        let mut buf = SearchBuf::new(n_cells);
        let mut pad_set = PadSet::new(n_cells);

        // Connection group id per net. Cells owned by one group are HARD obstacles
        // for every OTHER group during legalization, while a group's own sub-nets may
        // share copper — so nets that must overlap have to live in the SAME group.
        //
        // Two effects union nets into one group:
        //  1. NAME: sub-nets of one multi-point connection share a `group_of` key
        //     (`<conn>#<seg>` -> `<conn>`), interned by first appearance.
        //  2. GEOMETRY: nets that share an endpoint CELL (src or dst) are the same
        //     electrical net meeting at a junction — e.g. the MST edges of a multi-pad
        //     net, delivered as separately-named 2-point connections that touch at a
        //     shared branch pad. Without merging them they are forced cell-disjoint and
        //     the second edge cannot even start from the branch cell the first one
        //     committed, abandoning a net that routes fine alone. On the non-uniform
        //     Hanan grid every distinct pad coordinate gets its own grid line, so a
        //     shared src/dst CELL means the points are identical — this can never fuse
        //     two genuinely-distinct nets into a short.
        let mut group_ids: Vec<usize> = vec![0; n_nets];
        {
            // Start from the name-based interning, then union by shared endpoint cell.
            let mut parent: Vec<usize> = (0..n_nets).collect();
            fn find(parent: &mut [usize], x: usize) -> usize {
                let mut r = x;
                while parent[r] != r {
                    r = parent[r];
                }
                let mut c = x; // path-halving keeps repeated finds near-flat
                while parent[c] != r {
                    let next = parent[c];
                    parent[c] = r;
                    c = next;
                }
                r
            }
            let union = |parent: &mut [usize], a: usize, b: usize| {
                let (ra, rb) = (find(parent, a), find(parent, b));
                if ra != rb {
                    // Union toward the lower index for deterministic roots.
                    if ra < rb {
                        parent[rb] = ra;
                    } else {
                        parent[ra] = rb;
                    }
                }
            };
            // (1) name groups.
            let mut by_name: HashMap<&str, usize> = HashMap::new();
            for (i, net) in nets.iter().enumerate() {
                match by_name.get(group_of(&net.net)) {
                    Some(&j) => union(&mut parent, i, j),
                    None => {
                        by_name.insert(group_of(&net.net), i);
                    }
                }
            }
            // (2) shared-endpoint-cell junctions.
            let mut by_cell: HashMap<CellIdx, usize> = HashMap::new();
            for (i, net) in nets.iter().enumerate() {
                for &c in &[net.src, net.dst] {
                    match by_cell.get(&c) {
                        Some(&j) => union(&mut parent, i, j),
                        None => {
                            by_cell.insert(c, i);
                        }
                    }
                }
            }
            // Dense group ids by first appearance of each union root (deterministic).
            let mut root_to_g: HashMap<usize, usize> = HashMap::new();
            for i in 0..n_nets {
                let r = find(&mut parent, i);
                let next = root_to_g.len();
                group_ids[i] = *root_to_g.entry(r).or_insert(next);
            }
        }

        // Persistent congestion state.
        let mut history: Vec<u32> = vec![0; n_cells];
        let mut present: Vec<u32> = vec![0; n_cells];
        // Per-cell count of how many *other* nets' clearance footprints (planar
        // clearance halo + via keepout) cover this cell — the negotiation analog of
        // TritonRoute's `objCost`. Maintained EXACTLY parallel to `present`: when a
        // net's path is removed from `present` its halo footprint is removed from
        // `present_halo`, and when the new path is added to `present` its footprint is
        // added here (see the inc/dec sites in the loop below, both via
        // `for_each_halo_cell`). So during net i's search `present_halo` excludes net i
        // = every OTHER net's clearance footprint, exactly like `present`. When
        // `clearance_cells == 0 && via_model.keepout_mm == 0.0` the footprint is empty so
        // this stays all-zero and contributes nothing (byte-identical default).
        let mut present_halo: Vec<u32> = vec![0; n_cells];
        // Whether the clearance mechanism is active at all. Drives both the
        // `present_halo` pricing and the incremental-skip gating below.
        let clearance_active = self.clearance_mm > 0.0 || via_model.keepout_mm > 0.0;
        // Current routed path per net (empty == not currently routed).
        let mut paths: Vec<Vec<CellIdx>> = vec![Vec::new(); n_nets];

        // Per-net search window (bbox of endpoints+pads, expanded by a margin),
        // precomputed once: the negotiation search is restricted to this box so
        // explored area scales with the net's local region, not the whole board.
        let windows: Vec<Window> = nets
            .iter()
            .map(|net| net_window(dims, net.src, net.dst, &net.passable_pads))
            .collect();

        // Incremental rerouting (large boards only): after the first iteration,
        // only a net that is unrouted OR whose path touches a cell that was
        // over-used last iteration needs rerouting — a "happy" net's priced costs
        // are unchanged (history only grows on over-used cells), so its optimal
        // path is unchanged. This cuts later iterations from O(all nets) to
        // O(congested nets). Gated to large net counts so the small deterministic
        // unit tests keep their exact full-reroute behaviour.
        //
        // CLEARANCE GATING: the quiescence test below skips a routed net whose path
        // avoids every `prev_overused` cell, on the reasoning that such a net's priced
        // costs are unchanged. That reasoning is incomplete once clearance is active: a
        // net's priced cost can also change because a *neighbour's* halo
        // (`present_halo`) shifted onto its path, which `prev_overused` (a copper-overuse
        // set) does not capture. The clearance (Jacobi) branch therefore augments the
        // skip with a second signal — `halo_dirty`, the set of cells whose `present_halo`
        // value actually CHANGED during the previous iteration's merge — and reroutes a
        // net whose path touches a `prev_overused` OR a `halo_dirty` cell. A *delta* (not
        // an absolute `present_halo > 0`) is required because `for_each_halo_cell` stamps
        // the clearance box including its center, so every routed net self-halos its own
        // path cells; only genuine external movement should force a reroute. The
        // sequential (clearance-off) branch keeps the original copper-only skip and stays
        // byte-identical (`present_halo`/`halo_dirty` remain all-zero / empty there).
        let incremental = n_nets > 8 && !clearance_active;
        // Route nets in parallel (snapshot-based Jacobi merge) when clearance is
        // active (it REQUIRES the merge to keep `present_halo` consistent) or when the
        // board is large enough that the sequential per-net loop dominates wall-clock.
        // Small non-clearance boards keep the sequential Gauss-Seidel path so the
        // deterministic tests and single-layer fixtures stay byte-identical. For the
        // large non-clearance case the parallel branch is already correct: `present_halo`
        // stays all-zero (so its pricing/`halo_dirty` contribute nothing) and the
        // incremental dirty set is driven by `prev_overused`, which `incremental` (true
        // for `n_nets > 8`) maintains below.
        let use_parallel = clearance_active || n_nets > PARALLEL_NEGOTIATION_THRESHOLD;
        // `overused` from the previous iteration (cell -> was it over-used). Empty
        // before the first iteration (everything reroutes).
        let mut prev_overused: Vec<bool> = vec![false; n_cells];
        let mut prev_overused_cells: Vec<CellIdx> = Vec::new();
        // `halo_dirty[c]` == did `present_halo[c]` change during the previous iteration's
        // merge (clearance branch only). Read by the dirty-net test, repopulated by the
        // merge. List-reset (no O(all cells) memset). `halo_delta` is per-iteration merge
        // scratch: the net change to `present_halo[c]` this iteration; a cell is dirty iff
        // its delta ends non-zero (a net rerouted to the SAME path cancels to zero and is
        // correctly NOT marked). Both stay empty/zero when clearance is inactive.
        let mut halo_dirty: Vec<bool> = vec![false; n_cells];
        let mut halo_dirty_cells: Vec<CellIdx> = Vec::new();
        let mut halo_delta: Vec<i32> = vec![0; n_cells];
        let mut halo_touched_cells: Vec<CellIdx> = Vec::new();

        // Per-iteration overuse scratch, allocated once and cleared incrementally
        // (via the touched-cell lists) so no iteration pays an O(all cells) memset.
        let mut first_group: Vec<i64> = vec![-1; n_cells];
        let mut overused: Vec<bool> = vec![false; n_cells];

        for iter in 0..MAX_ITERS {
            let pfac: u32 = 1 + iter;

            if use_parallel {
                // ---- Snapshot-based (Jacobi-style) PARALLEL negotiation ----
                //
                // Every DIRTY net reroutes against a READ-ONLY snapshot of
                // `present`/`present_halo`/`history` (occupancy from the END of the
                // previous iteration); clean nets keep their path and their existing
                // contribution to the occupancy maps. Dirty nets are routed in parallel
                // with rayon and MERGED back SEQUENTIALLY in net-index order, so the
                // outcome is independent of thread scheduling (deterministic). This is a
                // standard PathFinder variant (Jacobi vs Gauss-Seidel) and may converge
                // to a different-but-equivalent route than the sequential path. This
                // branch runs for the clearance path and for large non-clearance boards
                // (`n_nets > PARALLEL_NEGOTIATION_THRESHOLD`); small non-clearance boards
                // stay on the sequential path, so their byte-identical output is unchanged.
                //
                // DIRTY SET: a net needs rerouting when it is unrouted, or its path
                // touches a cell that was over-used (`prev_overused`) or whose
                // `present_halo` changed (`halo_dirty`) during the previous iteration.
                // Anything else is unaffected (its priced costs are unchanged) and is
                // skipped — cutting later iterations from O(all nets) to O(congested).
                // Gated to >8 nets so the small deterministic tests keep full reroute.
                let dirty: Vec<usize> = (0..n_nets)
                    .filter(|&i| {
                        n_nets <= 8
                            || iter == 0
                            || paths[i].is_empty()
                            || paths[i]
                                .iter()
                                .any(|&c| prev_overused[c as usize] || halo_dirty[c as usize])
                    })
                    .collect();
                // Reset the previous iteration's `halo_dirty` set now that we have
                // consumed it; the merge below repopulates it for the next iteration.
                for &c in &halo_dirty_cells {
                    halo_dirty[c as usize] = false;
                }
                halo_dirty_cells.clear();
                let clearance_mm = self.clearance_mm;
                // Borrow the snapshots immutably for the duration of the parallel map.
                // Each worker thread keeps its OWN reusable `SearchBuf` + `PadSet`
                // scratch via `map_init` (never shared across threads). The closure
                // captures only shared `&` references (all `Sync`): `grid`, the three
                // occupancy slices, `nets`, `windows`, `via_model`.
                let present_snap: &[u32] = &present;
                let halo_snap: &[u32] = &present_halo;
                let history_snap: &[u32] = &history;
                let nets_ref = nets;
                let windows_ref = &windows;
                let via_ref = &via_model;
                let coords_ref = &coords;
                let mut routed_paths: Vec<(usize, Option<Vec<CellIdx>>)> = dirty
                    .par_iter()
                    .map_init(
                        || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                        |(buf, pad_set), &i| {
                            let net = &nets_ref[i];
                            pad_set.load(&net.passable_pads);
                            // Route within the net's window; on failure, retry once on
                            // the full board so the occasional global net still
                            // completes. Pure read-only search over the snapshots.
                            let routed = route_negotiated(
                                buf,
                                grid,
                                coords_ref,
                                pad_set,
                                present_snap,
                                halo_snap,
                                history_snap,
                                pfac,
                                net.src,
                                net.dst,
                                windows_ref[i],
                                via_ref,
                            )
                            .or_else(|| {
                                route_negotiated(
                                    buf,
                                    grid,
                                    coords_ref,
                                    pad_set,
                                    present_snap,
                                    halo_snap,
                                    history_snap,
                                    pfac,
                                    net.src,
                                    net.dst,
                                    Window::full(dims),
                                    via_ref,
                                )
                            });
                            (i, routed.map(|(p, _)| p))
                        },
                    )
                    .collect();
                // Deterministic INCREMENTAL merge: process the rerouted (dirty) nets in
                // net-index order, each subtracting its OLD path/halo and adding its NEW
                // path/halo from the shared maps (clean nets are left untouched). Counts
                // are commutative so the result is identical regardless of scheduling and
                // matches a full rebuild. `halo_delta` accumulates the net change to
                // `present_halo`; a cell whose delta ends non-zero is recorded into the
                // next iteration's `halo_dirty` set (a reroute to the SAME path cancels
                // to zero and is correctly NOT marked, which is what lets the dirty set
                // shrink to the congested region and the iteration cost collapse).
                routed_paths.sort_unstable_by_key(|(i, _)| *i);
                for (i, path) in routed_paths {
                    // Remove the old path's copper + halo (the net's prior contribution).
                    for &c in &paths[i] {
                        present[c as usize] = present[c as usize].saturating_sub(1);
                    }
                    for_each_halo_cell(dims, &coords, grid, &paths[i], clearance_mm, &via_model, |c| {
                        present_halo[c as usize] = present_halo[c as usize].saturating_sub(1);
                        halo_delta[c as usize] -= 1;
                        halo_touched_cells.push(c);
                    });
                    match path {
                        Some(path) => {
                            for &c in &path {
                                present[c as usize] = present[c as usize].saturating_add(1);
                            }
                            for_each_halo_cell(
                                dims,
                                &coords,
                                grid,
                                &path,
                                clearance_mm,
                                &via_model,
                                |c| {
                                    present_halo[c as usize] =
                                        present_halo[c as usize].saturating_add(1);
                                    halo_delta[c as usize] += 1;
                                    halo_touched_cells.push(c);
                                },
                            );
                            paths[i] = path;
                        }
                        None => {
                            // Unrouted this iteration: contributes nothing to occupancy.
                            paths[i].clear();
                        }
                    }
                }
                // Finalize the halo-dirty set for the next iteration: any touched cell
                // whose net delta is non-zero changed and is marked dirty; reset the
                // delta scratch via the touched list (no O(all cells) memset). Touched
                // cells may repeat — the delta read + reset makes the marking idempotent.
                for &c in &halo_touched_cells {
                    let d = &mut halo_delta[c as usize];
                    if *d != 0 {
                        *d = 0;
                        if !halo_dirty[c as usize] {
                            halo_dirty[c as usize] = true;
                            halo_dirty_cells.push(c);
                        }
                    }
                }
                halo_touched_cells.clear();
            } else {
                // ---- Sequential incremental negotiation (clearance INACTIVE) ----
                // Verbatim pre-parallel behaviour; kept byte-identical. `clearance_active`
                // is false here, so `present_halo` stays all-zero and contributes nothing.
                for i in 0..n_nets {
                    let net = &nets[i];

                    // Skip quiescent nets after the first iteration: a routed net whose
                    // path avoids every previously-over-used cell keeps its path.
                    if incremental && iter > 0 && !paths[i].is_empty() {
                        let touches_overuse = paths[i].iter().any(|&c| prev_overused[c as usize]);
                        if !touches_overuse {
                            continue;
                        }
                    }

                    // Remove this net's old path from `present` before pricing, and its
                    // clearance footprint from `present_halo` in lockstep (saturating), so
                    // both maps exclude net i during its own search. `for_each_halo_cell`
                    // is a no-op when clearance is inactive, so `present_halo` stays 0.
                    for &c in &paths[i] {
                        present[c as usize] = present[c as usize].saturating_sub(1);
                    }
                    if clearance_active {
                        for_each_halo_cell(
                            dims,
                            &coords,
                            grid,
                            &paths[i],
                            self.clearance_mm,
                            &via_model,
                            |c| {
                                present_halo[c as usize] =
                                    present_halo[c as usize].saturating_sub(1);
                            },
                        );
                    }
                    paths[i].clear();

                    pad_set.load(&net.passable_pads);

                    // Route within the net's window; on failure, retry once on the
                    // full board so the occasional global net still completes.
                    let routed = route_negotiated(
                        &mut buf,
                        grid,
                        &coords,
                        &pad_set,
                        &present,
                        &present_halo,
                        &history,
                        pfac,
                        net.src,
                        net.dst,
                        windows[i],
                        &via_model,
                    )
                    .or_else(|| {
                        route_negotiated(
                            &mut buf,
                            grid,
                            &coords,
                            &pad_set,
                            &present,
                            &present_halo,
                            &history,
                            pfac,
                            net.src,
                            net.dst,
                            Window::full(dims),
                            &via_model,
                        )
                    });

                    if let Some((path, _)) = routed {
                        for &c in &path {
                            present[c as usize] = present[c as usize].saturating_add(1);
                        }
                        if clearance_active {
                            for_each_halo_cell(
                                dims,
                                &coords,
                                grid,
                                &path,
                                self.clearance_mm,
                                &via_model,
                                |c| {
                                    present_halo[c as usize] =
                                        present_halo[c as usize].saturating_add(1);
                                },
                            );
                        }
                        paths[i] = path;
                    }
                    // else: leave unrouted this iteration (no contribution to present).
                }
            }

            // Overuse across GROUPS: a cell is over-used iff ≥2 distinct groups
            // occupy it. Track the first group seen per cell; a second distinct
            // group flags overuse. We only touch the cells the current paths cover,
            // so the scan is O(total path length), not O(all cells): `first_group`
            // and `overused` are cleared via the touched-cell list, not a memset.
            let mut overused_cells: Vec<CellIdx> = Vec::new();
            let mut any_overuse = false;
            for i in 0..n_nets {
                let g = group_ids[i] as i64;
                for &c in &paths[i] {
                    let slot = &mut first_group[c as usize];
                    if *slot < 0 {
                        *slot = g;
                    } else if *slot != g && !overused[c as usize] {
                        overused[c as usize] = true;
                        overused_cells.push(c);
                        any_overuse = true;
                    }
                }
            }

            // Bump history on the over-used cells.
            for &c in &overused_cells {
                history[c as usize] = history[c as usize].saturating_add(SCALE);
            }

            // Clear the per-iteration scratch for the cells we touched (O(touched),
            // not O(all cells)): `first_group` via the path cells, `overused` via
            // the over-used list.
            for path in paths.iter().take(n_nets) {
                for &c in path {
                    first_group[c as usize] = -1;
                }
            }
            for &c in &overused_cells {
                overused[c as usize] = false;
            }

            // Roll this iteration's over-used set into `prev_overused` for the next
            // iteration's quiescence test (list-based reset to avoid an O(n) clear).
            if incremental {
                for &c in &prev_overused_cells {
                    prev_overused[c as usize] = false;
                }
                for &c in &overused_cells {
                    prev_overused[c as usize] = true;
                }
                std::mem::swap(&mut prev_overused_cells, &mut overused_cells);
            }

            if !any_overuse {
                break; // converged: cell-disjoint across groups
            }
        }
        // ---- Order-robust legalization ----
        //
        // Legalization commits whole connection groups, one at a time, in a chosen
        // group order; each group's cells become hard obstacles for later groups.
        // The first-committed group can take a corridor that strands a later one,
        // so the order matters: a cell-disjoint solution may exist only under a
        // different order. We therefore run the same commit logic for several
        // candidate group orders and keep the result that routes the most nets.

        // Distinct group ids in first-appearance order (group_ids are interned by
        // first appearance, so the set {0..n_groups-1} is exactly that order).
        let n_groups = group_ids.iter().map(|&g| g + 1).max().unwrap_or(0);

        // Per-net "alone-path" length: the net routed by itself on the base grid
        // (own pads unmasked, no other nets present). A net that is individually
        // unroutable contributes 0 and can never be committed. This doubles as the
        // ordering difficulty metric.
        let mut alone_len: Vec<Cost> = vec![0; n_nets];
        // Full alone-path cells per net, used by the rip-up stage to find which
        // committed foreign-group nets a stranded net's natural route would cross.
        let mut alone_path: Vec<Vec<CellIdx>> = vec![Vec::new(); n_nets];
        {
            // No foreign occupancy here; the alone-path is the net by itself on the
            // base grid (route within the window first, full board on failure). Each
            // net's search is INDEPENDENT and pure over the read-only grid, so they run
            // in PARALLEL with rayon — per-thread `SearchBuf`/`PadSet` scratch via
            // `map_init`, results collected over the indexed range so they land in net
            // order regardless of scheduling (byte-identical to the old serial loop).
            // This loop was previously the dominant single-threaded phase of a full
            // route (every net, plus the occasional expensive full-board fallback).
            let no_owner: Vec<i64> = Vec::new();
            let no_halo: Vec<i64> = Vec::new();
            let alone: Vec<(Cost, Vec<CellIdx>)> = (0..n_nets)
                .into_par_iter()
                .map_init(
                    || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                    |(buf, pad_set), i| {
                        let net = &nets[i];
                        pad_set.load(&net.passable_pads);
                        let routed = route_legal(
                            buf, grid, &coords, pad_set, &no_owner, &no_halo, -1, net.src,
                            net.dst, windows[i], &via_model, self.clearance_mm,
                        )
                        .or_else(|| {
                            route_legal(
                                buf,
                                grid,
                                &coords,
                                pad_set,
                                &no_owner,
                                &no_halo,
                                -1,
                                net.src,
                                net.dst,
                                Window::full(dims),
                                &via_model,
                                self.clearance_mm,
                            )
                        });
                        match routed {
                            Some((path, _)) => (unit_cost(&path), path),
                            None => (0, Vec::new()),
                        }
                    },
                )
                .collect();
            for (i, (len, path)) in alone.into_iter().enumerate() {
                alone_len[i] = len;
                alone_path[i] = path;
            }
        }

        // Per-group alone-path length = sum over the group's nets.
        let mut group_alone: Vec<Cost> = vec![0; n_groups];
        for i in 0..n_nets {
            group_alone[group_ids[i]] = group_alone[group_ids[i]].saturating_add(alone_len[i]);
        }

        // Candidate group orders (each a permutation of 0..n_groups).
        let base_order: Vec<usize> = (0..n_groups).collect();
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        // 1. First-appearance / input order.
        candidates.push(base_order.clone());
        // 2. Ascending by alone-path length (stable; ties keep input order).
        {
            let mut o = base_order.clone();
            o.sort_by_key(|&g| group_alone[g]);
            candidates.push(o);
        }
        // 3. Descending by alone-path length (stable; ties keep input order).
        {
            let mut o = base_order.clone();
            o.sort_by_key(|&g| std::cmp::Reverse(group_alone[g]));
            candidates.push(o);
        }
        // 4. For few groups, exhaustively try every order (≤ 5! = 120).
        //
        // The cap is deliberately at 5, not the group count itself: each extra order is
        // a FULL `legalize_in_order` pass (routes every net), so the candidate count
        // multiplies the legalization cost. The factorial makes 6 groups = 720 passes
        // and 7 = 5040 — on real multi-net boards (e.g. tscircuit `keyboards`) that is
        // ~15–30 s of (already parallel, ~9-core) work per solve for NO completion gain:
        // the 3 heuristic orders above plus the rip-up stage already recover the same
        // routable nets the exhaustive search finds. Capping 6–7 group boards to the
        // heuristics cuts those solves ~18× (≈30 s → ≈1.7 s) with identical routed
        // counts. (An earlier experiment ADDING random orders beyond exhaustive also
        // REGRESSED DRC quality — the `(routed, unit_cost)` selection metric is not
        // clearance-aligned — so more orders is never the answer; fewer is.)
        if n_groups <= 5 {
            for perm in permutations(&base_order) {
                candidates.push(perm);
            }
        }

        // Evaluate each candidate IN PARALLEL and keep the best. The candidate orders
        // are independent (`legalize_in_order` is a pure function of its inputs), so
        // rayon distributes them across cores; each worker uses its OWN `SearchBuf` +
        // `PadSet` scratch via `map_init` (never shared). Results are collected with
        // their candidate INDEX, then the winner is picked by a fully deterministic,
        // scheduling-independent fold over the indexed results.
        let evaluated: Vec<(usize, usize, Cost, Committed)> = candidates
            .par_iter()
            .enumerate()
            .map_init(
                || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                |(buf, pad_set), (idx, order)| {
                    let committed = legalize_in_order(
                        grid,
                        &coords,
                        buf,
                        pad_set,
                        nets,
                        &group_ids,
                        &paths,
                        &windows,
                        order,
                        n_cells,
                        &via_model,
                        self.clearance_mm,
                    );
                    let routed = committed.iter().filter(|c| c.is_some()).count();
                    let total_cost: Cost = committed
                        .iter()
                        .filter_map(|c| c.as_ref())
                        .map(|p| unit_cost(p))
                        .fold(0, |a, b| a.saturating_add(b));
                    (idx, routed, total_cost, committed)
                },
            )
            .collect();

        // Deterministic pick — IDENTICAL to the old sequential criterion: most nets
        // routed, then lowest total unit cost, then lexicographically lowest group
        // ORDER (`order < bo`). Iterating the indexed results in candidate-index
        // order (sequential scan) reproduces the exact tie-break path the sequential
        // loop took (it processed `candidates` in order with the same `better` test),
        // so the chosen `committed` is byte-identical regardless of how rayon
        // scheduled the parallel evaluation.
        let mut best: Option<(usize, Cost, &Vec<usize>)> = None;
        let mut best_idx: Option<usize> = None;
        for (idx, routed, total_cost, _committed) in &evaluated {
            let order = &candidates[*idx];
            let better = match &best {
                None => true,
                Some((br, bc, bo)) => {
                    *routed > *br
                        || (*routed == *br && *total_cost < *bc)
                        || (*routed == *br && *total_cost == *bc && order < bo)
                }
            };
            if better {
                best = Some((*routed, *total_cost, order));
                best_idx = Some(*idx);
            }
        }
        let (best_routed, best_order, multi_committed) = match best_idx {
            Some(idx) => {
                let routed = best.as_ref().map(|b| b.0).unwrap_or(0);
                // Reclaim the winning committed vec by index without cloning.
                let committed = evaluated
                    .into_iter()
                    .find(|(i, _, _, _)| *i == idx)
                    .map(|(_, _, _, c)| c)
                    .unwrap_or_else(|| vec![None; n_nets]);
                (routed, candidates[idx].clone(), committed)
            }
            None => (0, base_order.clone(), vec![None; n_nets]),
        };

        // ---- Bounded rip-up-and-reroute legalization (final stage) ----
        //
        // The multi-order pass commits whole groups in some fixed order and never
        // displaces an already-committed group. That fails when net A's shortest
        // route blocks B *and* B's shortest route blocks A: no single commit order
        // works, you must displace a committed net. This stage seeds a work-queue
        // with the BEST multi-order result and, for any net that cannot route
        // around the committed cells, rips up the cheapest committed blocker(s),
        // re-places the stranded net, and re-enqueues the ripped nets — bounded by
        // a global rip budget so it always terminates. The result is used only if
        // it routes strictly more nets than the multi-order pass (never a regress).
        let committed = if best_routed < n_nets {
            let rip = ripup_legalize(
                grid,
                &coords,
                &mut buf,
                &mut pad_set,
                nets,
                &group_ids,
                &alone_path,
                &windows,
                &best_order,
                &multi_committed,
                n_cells,
                &via_model,
                self.clearance_mm,
            );
            let rip_routed = rip.iter().filter(|c| c.is_some()).count();
            if rip_routed > best_routed {
                rip
            } else {
                multi_committed
            }
        } else {
            multi_committed
        };
        // Assemble in input net order for determinism. Carry the router's actual
        // electrical-net group id alongside each routed net (aligned 1:1 with
        // `results`) so downstream DRC can grant same-net copper the exact same
        // immunity the router permitted, rather than re-deriving grouping post-hoc.
        let mut results: Vec<RouteResult> = Vec::new();
        let mut groups: Vec<u32> = Vec::new();
        let mut unrouted: Vec<String> = Vec::new();
        for (i, net) in nets.iter().enumerate() {
            match &committed[i] {
                Some(path) => {
                    results.push(RouteResult {
                        net: net.net.clone(),
                        path: path.clone(),
                        cost: unit_cost(path),
                    });
                    groups.push(group_ids[i] as u32);
                }
                None => unrouted.push(net.net.clone()),
            }
        }

        let congestion = BoardRoute::congestion_from(dims, &results);
        Ok(BoardRoute {
            results,
            unrouted,
            congestion,
            groups,
        })
    }
}

/// Route one net for the negotiation phase using on-the-fly congestion pricing —
/// no grid clone. The price of the planar move `u -> v` is
/// `edge_cost(len(u,v)) + history[v] + pfac*SCALE*present[v]` (+ the clearance halo
/// term), capped strictly below [`OBSTACLE`] (`present` already excludes this net's
/// own occupancy). The planar base is the move's GEOMETRIC length from `coords`
/// rather than a uniform `SCALE`, so steps over a wide channel cost more than steps
/// over a fine pitch; on a uniform grid every step is length 1 and the base is
/// exactly `SCALE` (byte-identical). A cell is blocked iff it is a base obstacle that
/// is NOT one of the net's own pads, or it lies outside `window`. The own endpoints
/// are forced passable. Returns the windowed shortest path and its (priced) cost, or
/// `None`.
#[allow(clippy::too_many_arguments)]
fn route_negotiated(
    buf: &mut SearchBuf,
    base: &Grid,
    coords: &GridCoords,
    pads: &PadSet,
    present: &[u32],
    present_halo: &[u32],
    history: &[u32],
    pfac: u32,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    // Price to MOVE onto cell `c` over a base step cost `base_cost` (the planar
    // geometric edge length for a planar move, or the via `step_cost` for a layer
    // change): the base plus permanent history, the present-congestion penalty, and
    // the clearance penalty (TritonRoute `objCost` analog) —
    // `pfac * CLEARANCE_NEG_WEIGHT * present_halo[c]`, i.e. how many OTHER nets'
    // clearance footprints cover `c`, scaled by the growing present-factor so the
    // negotiation spreads nets apart over iterations. Capped below OBSTACLE. The
    // congestion weights are multiples of `SCALE`, so they keep their strength
    // relative to the geometric base step at any pitch. `present_halo` is all-zero
    // when clearance is inactive, so that term vanishes (byte-identical default).
    let priced_with_base = |c: CellIdx, base_cost: u64| -> Cost {
        let ci = c as usize;
        let priced = base_cost
            .saturating_add(history[ci] as u64)
            .saturating_add((pfac as u64) * (SCALE as u64) * (present[ci] as u64))
            .saturating_add(
                (pfac as u64) * (CLEARANCE_NEG_WEIGHT as u64) * (present_halo[ci] as u64),
            );
        priced.min(OBSTACLE as u64 - 1) as Cost
    };
    // Edge-aware planar base: the geometric length of the move `u -> v`, in the same
    // fixed-point units as the heuristic. On a uniform grid this is the constant
    // `SCALE`, so `cost_fn` reduces to the historical `priced_with_base(v, SCALE)`.
    let cost_fn = |u: CellIdx, v: CellIdx| -> Cost {
        priced_with_base(v, edge_cost(coords.manhattan_len(dims, u, v)) as u64)
    };
    let blocked_fn = |c: CellIdx| -> bool {
        if !window.contains(dims, c) {
            return true;
        }
        if c == src || c == dst {
            return false;
        }
        base.is_obstacle(c) && !pads.contains(c)
    };
    // A via step `u -> v` (adjacent layers, same x,y) is allowed iff the model
    // permits that layer transition; it is priced like a cell but with the via's
    // `step_cost` as the base instead of the planar `SCALE`, so vias negotiate the
    // destination cell's congestion exactly as planar moves do. `blocked_fn` is
    // applied to `v` by `astar_buf` first, so an out-of-window / foreign-pad target
    // already blocks the via.
    let via_step = |u: CellIdx, v: CellIdx| -> Option<Cost> {
        if via_model.is_step_legal(dims.layer_of(u), dims.layer_of(v)) {
            Some(priced_with_base(v, via_model.step_cost as u64))
        } else {
            None
        }
    };
    let h = |c: CellIdx| manhattan_scaled(dims, coords, c, dst, via_model.step_cost);
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h, via_step)
}

/// Route one net for legalization / rip-up using on-the-fly costs — no grid clone.
///
/// Clearance AND copper are BOTH HARD blocks here (this is the committing pass; a
/// route it returns is guaranteed clearance-clean, so a net that cannot reach `dst`
/// without violating clearance returns `None` and is reported unrouted/congested by
/// the caller — never silently committed in violation):
///   * `owner` — committed COPPER. A cell owned by a group other than `own_group`
///     is a HARD obstacle (two distinct nets may never overlap). Cells owned by
///     `own_group` (siblings) and free cells are passable at their base cost.
///   * `halo`  — foreign clearance / via-keepout halo. A cell with `halo[c]` a
///     foreign group (`>= 0 && != own_group`) that is NOT also this net's own copper
///     is a HARD obstacle too — the legalizer must not place copper inside another
///     net's required spacing. Own-group halo costs nothing (same-net override).
///   * `src`/`dst` stay forced-passable (a net's own pads must remain reachable).
///
/// Every passable step is priced by its GEOMETRIC length (`edge_cost` from `coords`)
/// rather than the grid's per-cell value, so the planar base ignores whether a cell
/// is an unmasked own-pad or ordinary copper — its endpoints are always enterable and
/// the search is confined to `window`. `owner`/`halo` may each be empty to mean "no
/// owners / no halo" (the alone-path case). When `halo` is empty the foreign-halo
/// hard block and the via annular-ring guard are both inert, so the clearance-off
/// fast path is byte-identical to the pre-clearance router. Returns the windowed
/// shortest path and its cost, or `None`.
#[allow(clippy::too_many_arguments)]
fn route_legal(
    buf: &mut SearchBuf,
    base: &Grid,
    coords: &GridCoords,
    pads: &PadSet,
    owner: &[i64],
    halo: &[i64],
    own_group: i64,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
    clearance: f64,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    let has_owner = !owner.is_empty();
    let has_halo = !halo.is_empty();
    // Edge-aware planar base: the geometric length of the move `u -> v` in fixed-point
    // units (same units as the heuristic), replacing the old uniform per-cell base.
    // On a uniform grid every step is length 1, so the base is the constant `SCALE`
    // (the legalizer reports `unit_cost(path)` — path length — so this magnitude only
    // affects the path CHOICE, never the emitted cost). Clearance is now HARD (handled
    // in `blocked_fn`), so `cost_fn` is the pure geometric base.
    let cost_fn =
        |u: CellIdx, v: CellIdx| -> Cost { edge_cost(coords.manhattan_len(dims, u, v)) };
    let blocked_fn = |c: CellIdx| -> bool {
        if !window.contains(dims, c) {
            return true;
        }
        // Foreign-group COPPER cells are hard obstacles, even at this net's
        // endpoints — distinct nets may never overlap.
        if has_owner {
            let o = owner[c as usize];
            if o >= 0 && o != own_group {
                return true;
            }
        }
        // Endpoints (this net's own pads) stay reachable. Checked AFTER foreign copper
        // (two nets may never share a cell) but BEFORE the foreign-halo block, so a pad
        // sitting in another net's halo can still be entered to terminate the route.
        if c == src || c == dst {
            return false;
        }
        // Foreign-group HALO is now a HARD obstacle: committing copper inside another
        // net's required spacing is a clearance violation, so this pass refuses it. A
        // foreign halo cell that is ALSO this net's own copper (`owner == own_group`)
        // is not blocked. When `halo` is empty this branch is inert (fast path).
        if has_halo {
            let h = halo[c as usize];
            let own_copper = has_owner && owner[c as usize] == own_group;
            if h >= 0 && h != own_group && !own_copper {
                return true;
            }
        }
        base.is_obstacle(c) && !pads.contains(c)
    };
    // Via annular-ring radius: a placed via's pad reserves `max(clearance, keepout)`
    // around itself on both spanned layers. When this is <= 0 (clearance-off fast
    // path) the guard is skipped entirely and behaviour is byte-identical.
    let via_r = clearance.max(via_model.keepout_mm);
    let ring_guard = has_halo && via_r > 0.0;
    // True iff a via landing at `(cx, cy)` on `layer` would put its annular ring over
    // foreign copper or a foreign halo cell — in which case the via step is rejected.
    // The endpoint cells (src/dst) are exempt so a via may still land on the net's own
    // pad. Scans the geometric `geom_box` band of radius `via_r` over `coords`.
    let ring_conflict = |cx: u32, cy: u32, layer: u32| -> bool {
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, via_r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, via_r);
        for ny in y0..y1 {
            for nx in x0..x1 {
                let n = dims.idx3(nx, ny, layer);
                if n == src || n == dst {
                    continue;
                }
                let ni = n as usize;
                let o = if has_owner { owner[ni] } else { -1 };
                if o >= 0 && o != own_group {
                    return true; // ring overlaps foreign copper
                }
                let hh = halo[ni];
                if hh >= 0 && hh != own_group && !(o == own_group) {
                    return true; // ring overlaps a foreign halo cell
                }
            }
        }
        false
    };
    // A via step is legal per the model; it costs the via's `step_cost` (foreign
    // owners / endpoints are already rejected by `blocked_fn` on the destination).
    // Additionally reject the step when the via's annular ring at `v` would overlap
    // foreign copper/halo on EITHER spanned layer.
    let via_step = |u: CellIdx, v: CellIdx| -> Option<Cost> {
        let (lu, lv) = (dims.layer_of(u), dims.layer_of(v));
        if !via_model.is_step_legal(lu, lv) {
            return None;
        }
        if ring_guard {
            let (vx, vy, _) = dims.xyz(v);
            if ring_conflict(vx, vy, lu) || ring_conflict(vx, vy, lv) {
                return None;
            }
        }
        Some(via_model.step_cost)
    };
    let h = |c: CellIdx| manhattan_scaled(dims, coords, c, dst, via_model.step_cost);
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h, via_step)
}

/// Enumerate every cell in a `path`'s SOFT clearance footprint, invoking `visit`
/// once per cell (cells may repeat across overlapping halos — the callers use
/// saturating inc/dec or first-claim guards, so repeats are intended/idempotent).
///
/// The footprint is exactly the set of cells [`stamp_owner`] considers for `halo`
/// (so the negotiation `present_halo` field and the legalization `halo` map share
/// one shape):
///   * **Planar clearance halo.** For each path cell, on that cell's OWN layer, the
///     [`geom_box`] band of geometric radius `r = clearance` (continuous units over
///     `coords`; the `(2r+1)x(2r+1)` Chebyshev box on a uniform grid). Base-obstacle
///     cells are SKIPPED (a halo never claims an obstacle / foreign pad).
///   * **Via keepout halo.** At each via (two consecutive path cells sharing `(x,y)`
///     but differing in layer), on BOTH spanned layers, the geometric box of radius
///     `max(clearance, via_model.keepout_mm)`. Base-obstacle cells are SKIPPED.
///
/// When `clearance == 0.0` AND `via_model.keepout_mm == 0.0` every radius is 0, so
/// `visit` is never called and the footprint is empty — the property that keeps the
/// clearance-off router byte-identical.
///
/// NON-UNIFORM GRID NOTE: the halo radius is a *geometric distance* (`clearance`,
/// continuous units) measured over `coords`, NOT a cell count, so on a Hanan grid
/// it spans the same physical width everywhere regardless of local line density —
/// matching [`mr_grid::GridBuilder::inflate_clearance`]'s hard-obstacle model. On a
/// [`GridCoords::uniform`] grid (unit-spaced lines) a `clearance` of `n` reproduces
/// the former `n`-cell Chebyshev box exactly (byte-identical). The via keepout is
/// likewise geometric (`via_model.keepout_mm` continuous units).
///
/// NOTE: unlike [`stamp_owner`] this does NOT skip cells already owned/claimed (it
/// has no ownership view); it visits the geometric footprint and leaves
/// owner/first-claim policy to the caller. `present_halo` wants the raw geometric
/// count (every other net's halo contributes), so this is the right shape for it,
/// and `stamp_owner` layers its own `owner == -1 && halo == -1` guard on top.
fn for_each_halo_cell(
    dims: mr_core::Dims,
    coords: &GridCoords,
    base: &Grid,
    path: &[CellIdx],
    clearance: f64,
    via_model: &ViaModel,
    mut visit: impl FnMut(CellIdx),
) {
    // Visit the planar geometric box of radius `r` (continuous units) around
    // `(cx, cy)` on `layer`, skipping base obstacles (a halo never claims an
    // obstacle / foreign pad). The per-axis index band is computed over the `coords`
    // line arrays, so on a uniform grid this is the `(2r+1)x(2r+1)` Chebyshev box and
    // on a Hanan grid it is the set of lines within `r` continuous units.
    let box_cells = |cx: u32, cy: u32, layer: u32, r: f64, visit: &mut dyn FnMut(CellIdx)| {
        if r <= 0.0 {
            return;
        }
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, r);
        for ny in y0..y1 {
            for nx in x0..x1 {
                let n = dims.idx3(nx, ny, layer);
                if !base.is_obstacle(n) {
                    visit(n);
                }
            }
        }
    };

    // Planar clearance halo around each path cell, on that cell's own layer.
    for &c in path {
        let (cx, cy, cl) = dims.xyz(c);
        box_cells(cx, cy, cl, clearance, &mut visit);
    }

    // Via keepout: a via is a consecutive same-(x,y), layer-changing step. At each
    // via (x,y) visit the larger of the planar clearance and the via keepout on both
    // layers the via spans.
    let via_r = clearance.max(via_model.keepout_mm);
    if via_r > 0.0 {
        for w in path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                box_cells(ax, ay, al, via_r, &mut visit);
                box_cells(bx, by, bl, via_r, &mut visit);
            }
        }
    }
}

/// Half-open `[lo, hi)` range of line indices in `lines` (sorted ascending,
/// non-empty, length `count`) within `r` continuous units of the seed line at index
/// `seed`. The in-clearance indices form a contiguous band (lines are sorted), so we
/// walk outward from `seed` and stop at the first line strictly farther than `r`.
///
/// This is the negotiation-halo twin of `mr_grid`'s `line_span`: on a
/// [`GridCoords::uniform`] grid (unit-spaced lines) `geom_box(.., seed, r)` returns
/// `[seed - floor(r), seed + floor(r) + 1)`, i.e. the former Chebyshev cell box of
/// radius `r`, keeping the uniform path byte-identical. Only `lines[..count]` is
/// consulted so a defensive coords array longer than `dims` never reads past the grid.
fn geom_box(lines: &[f64], count: u32, seed: u32, r: f64) -> (u32, u32) {
    let n = (lines.len() as u32).min(count);
    if n == 0 {
        return (0, 0);
    }
    let seed = seed.min(n - 1);
    let pos = lines[seed as usize];
    let mut lo = seed;
    while lo > 0 && (pos - lines[(lo - 1) as usize]).abs() <= r {
        lo -= 1;
    }
    let mut hi = seed + 1;
    while hi < n && (lines[hi as usize] - pos).abs() <= r {
        hi += 1;
    }
    (lo, hi)
}

/// Fold a committed `path` into the ownership maps, separating HARD copper from the
/// SOFT clearance halo so a net is never dropped merely for failing to honour
/// spacing. This is the single place both legalizers commit copper.
///
/// Two parallel maps are written (both indexed by cell, `-1` == free):
///   * `owner` — the committed PATH cells (the actual copper). A foreign group's
///     `owner` cell is a HARD block: two distinct nets must never overlap.
///   * `halo`  — the clearance / via-keepout cells around the copper. A foreign
///     group's `halo` cell is a HARD block in [`route_legal`] (the committing pass):
///     copper may never be placed inside another net's required spacing, so a net
///     that cannot route clear is left unrouted/congested rather than violating.
///
/// Exact stamping rule, applied for the committed `path` belonging to `group`:
///
/// 1. **Path cells (copper).** `owner[c] = group` for every cell `c` on the path,
///    unconditionally (matches the pre-clearance behaviour — the path always wins
///    its own cells).
/// 2. **Planar clearance halo.** For each path cell, on that cell's OWN layer,
///    visit every cell `n` within geometric distance `clearance` over `coords` (the
///    [`geom_box`] band; on a uniform grid this is the `(2r+1)x(2r+1)` Chebyshev
///    box). Set `halo[n] = group` ONLY IF `owner[n] == -1`
///    (the cell is not real copper — a halo never overwrites copper ownership) AND
///    `halo[n] == -1` (still free halo) AND `!base.is_obstacle(n)`. Overlap
///    tie-break: the FIRST group to claim a free halo cell keeps it (`halo[n] == -1`
///    guard); a later group's halo never overwrites it. A base obstacle / foreign
///    pad is never claimed (claiming it could deter access to that net's own pad,
///    though here the penalty is only soft).
/// 3. **Via keepout.** A via is detected as two consecutive path cells sharing the
///    same `(x, y)` but differing in layer. At each such `(x, y)`, on *every* layer
///    the via spans, stamp a halo of radius `max(clearance, via_model.keepout_mm)`
///    under the identical rule — a via pad is wider than a track, so it reserves a
///    larger neighbourhood.
///
/// CRITICAL: when `clearance == 0.0` AND `via_model.keepout_mm == 0.0` this marks
/// *exactly* the path cells into `owner` and writes NOTHING into `halo` (a radius-0
/// box is skipped; the via halo radius is likewise 0). `halo` stays all `-1`, so
/// [`route_legal`]'s soft cost adds nothing and the default router is byte-identical
/// to the pre-clearance implementation. `clearance` is a geometric distance over
/// `coords` (continuous units); on a uniform grid `n` reproduces the former `n`-cell
/// Chebyshev halo — see [`geom_box`].
#[allow(clippy::too_many_arguments)]
fn stamp_owner(
    owner: &mut [i64],
    halo: &mut [i64],
    base: &Grid,
    dims: mr_core::Dims,
    coords: &GridCoords,
    path: &[CellIdx],
    group: i64,
    clearance: f64,
    via_model: &ViaModel,
) {
    // Stamp a planar geometric halo of radius `r` (continuous units, over `coords`)
    // around `(cx, cy)` on `layer` into the `halo` map, claiming only cells that are
    // not real copper (`owner == -1`), not yet claimed by any group's halo
    // (`halo == -1`, first-claim-wins), and not base obstacles. The centre cell is
    // included, but a path cell already has `owner == group`, so the `owner == -1`
    // guard skips it (its halo entry stays free — own-group halo is irrelevant since
    // own-group cells cost nothing).
    let stamp_halo = |owner: &[i64], halo: &mut [i64], cx: u32, cy: u32, layer: u32, r: f64| {
        if r <= 0.0 {
            return;
        }
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, r);
        for ny in y0..y1 {
            for nx in x0..x1 {
                let n = dims.idx3(nx, ny, layer);
                if owner[n as usize] == -1 && halo[n as usize] == -1 && !base.is_obstacle(n) {
                    halo[n as usize] = group;
                }
            }
        }
    };

    // 1: own the path cells (copper) unconditionally.
    for &c in path {
        owner[c as usize] = group;
    }
    // 2: stamp the planar clearance halo around each path cell.
    for &c in path {
        let (cx, cy, cl) = dims.xyz(c);
        stamp_halo(owner, halo, cx, cy, cl, clearance);
    }

    // 3: via keepout. A via is a consecutive same-(x,y), layer-changing step. At
    // each via (x,y) stamp the larger of the planar clearance and the via keepout
    // on every layer the via spans (both endpoints' layers).
    let via_r = clearance.max(via_model.keepout_mm);
    if via_r > 0.0 {
        for w in path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                stamp_halo(owner, halo, ax, ay, al, via_r);
                stamp_halo(owner, halo, bx, by, bl, via_r);
            }
        }
    }
}

/// Inverse of [`stamp_owner`] for the rip-up stage: free every cell in a committed
/// `path`'s footprint (the copper path cells in `owner` AND the clearance /
/// via-keepout halo cells in `halo`) that is currently owned by `group`. Cells
/// owned by another group (a foreign group's copper, or a halo cell the first-claim
/// tie-break awarded to another group) are left untouched, so freeing is idempotent
/// and never releases a cell another group depends on. The footprint must be
/// scanned identically to `stamp_owner` so no halo cell leaks across a rip.
fn free_owner(
    owner: &mut [i64],
    halo: &mut [i64],
    dims: mr_core::Dims,
    coords: &GridCoords,
    path: &[CellIdx],
    group: i64,
    clearance: f64,
    via_model: &ViaModel,
) {
    // Release halo cells this group claimed (only `halo[n] == group`; first-claim
    // tie-break may have awarded an overlapping cell to another group, which we must
    // not clear). Copper path cells are released separately below. The scan mirrors
    // `stamp_owner`'s geometric `geom_box` exactly so no halo cell leaks across a rip.
    let clear_halo = |halo: &mut [i64], cx: u32, cy: u32, layer: u32, r: f64| {
        if r <= 0.0 {
            return;
        }
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, r);
        for ny in y0..y1 {
            for nx in x0..x1 {
                let n = dims.idx3(nx, ny, layer);
                if halo[n as usize] == group {
                    halo[n as usize] = -1;
                }
            }
        }
    };

    // Path cells: clear the copper still owned by `group` (siblings may have re-owned
    // an overlap to the same `group`, so this stays idempotent).
    for &c in path {
        if owner[c as usize] == group {
            owner[c as usize] = -1;
        }
    }
    for &c in path {
        let (cx, cy, cl) = dims.xyz(c);
        clear_halo(halo, cx, cy, cl, clearance);
    }

    let via_r = clearance.max(via_model.keepout_mm);
    if via_r > 0.0 {
        for w in path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                clear_halo(halo, ax, ay, al, via_r);
                clear_halo(halo, bx, by, bl, via_r);
            }
        }
    }
}

/// Commit all connection groups in the given `group_order`, returning the
/// per-net committed path (or `None` if that net could not be placed).
///
/// Each group is committed as a unit: its members are placed (reusing the
/// negotiated `paths[i]` when that path already avoids every foreign-group cell,
/// otherwise rerouting once around the committed `occupied` set), then the
/// group's cells are folded into `occupied` so later groups treat them as hard
/// obstacles. Sibling sub-nets of the same group never block one another.
///
/// `group_order` must be a permutation of the distinct group ids. `work` is used
/// as scratch and is left in an arbitrary state. The result is in net-index
/// order and is fully deterministic given the inputs and order.
#[allow(clippy::too_many_arguments)]
fn legalize_in_order(
    grid: &Grid,
    coords: &GridCoords,
    buf: &mut SearchBuf,
    pad_set: &mut PadSet,
    nets: &[NetEndpoints],
    group_ids: &[usize],
    paths: &[Vec<CellIdx>],
    windows: &[Window],
    group_order: &[usize],
    n_cells: usize,
    via_model: &ViaModel,
    clearance: f64,
) -> Committed {
    let dims = grid.dims;
    let n_nets = nets.len();
    // Owning group per committed COPPER cell, or -1 for free: a cell is a foreign
    // HARD obstacle for net i iff its owner is a group other than i's.
    let mut owner: Vec<i64> = vec![-1; n_cells];
    // Owning group per committed clearance-HALO cell, or -1 for free: a foreign
    // halo cell is a HARD block in `route_legal` (the committing pass), so copper is
    // never placed inside another net's spacing — an unroutable net is dropped.
    let mut halo: Vec<i64> = vec![-1; n_cells];
    let mut committed: Committed = vec![None; n_nets];

    // Net indices committed by the group currently being placed; their paths are
    // folded into `owner` (with clearance halos) only after the whole group
    // commits, so sibling sub-nets never block each other. Tracked as a list
    // (cleared per group) to avoid an O(n_cells) sweep.
    let mut group_members: Vec<usize> = Vec::new();

    // ---- Serial-equivalent staging for within-candidate parallelism ----
    //
    // Groups commit sequentially because each group's copper blocks later ones. But
    // two groups whose search regions are spatially DISJOINT can never affect each
    // other's *windowed* route, so they can be routed in PARALLEL and committed in any
    // order with the identical result. We assign each group a stage = 1 + the max
    // stage of any earlier-order group it spatially conflicts with (bounding boxes,
    // inflated by the clearance/via-keepout radius). Groups in the same stage are
    // mutually disjoint; we route all their nets in parallel against the owner/halo
    // from prior stages, then commit per group in `group_order` (deterministic).
    //
    // Windowed routes are byte-identical to the sequential router (a disjoint group's
    // cells are never in this net's window, so its commit order is irrelevant). The
    // rare full-board FALLBACK reads outside the window, so those nets are handled
    // SERIALLY in stage order (phase B) — deterministic, and they only differ from the
    // pure-sequential order in the uncommon global-net case, which the best-of-orders
    // pass and rip-up downstream absorb.
    let n_groups_total = group_ids.iter().map(|&g| g + 1).max().unwrap_or(0);
    let mut group_nets: Vec<Vec<usize>> = vec![Vec::new(); n_groups_total];
    for i in 0..n_nets {
        group_nets[group_ids[i]].push(i);
    }
    // Group bounding boxes (union of member windows). `None` = group with no nets.
    let gbox: Vec<Option<(u32, u32, u32, u32)>> = (0..n_groups_total)
        .map(|g| {
            let mut members = group_nets[g].iter();
            let first = members.next()?;
            let w = &windows[*first];
            let mut b = (w.x0, w.y0, w.x1, w.y1);
            for &i in members {
                let w = &windows[i];
                b.0 = b.0.min(w.x0);
                b.1 = b.1.min(w.y0);
                b.2 = b.2.max(w.x1);
                b.3 = b.3.max(w.y1);
            }
            Some(b)
        })
        .collect();
    // Conservative cell-count inflation of the per-group bbox for the staging
    // (which groups may legalize in the same parallel stage) heuristic ONLY: a
    // too-small value merely co-schedules groups the per-stage in-order hard-blocking
    // commit still resolves correctly, so a geometric-distance approximation is safe
    // here. `clearance.ceil()` matches the old cell count exactly on a uniform grid.
    let infl = clearance.max(via_model.keepout_mm).ceil() as u32;
    let conflict = |a: usize, b: usize| -> bool {
        match (gbox[a], gbox[b]) {
            (Some(a), Some(b)) => {
                let ax0 = a.0.saturating_sub(infl);
                let ay0 = a.1.saturating_sub(infl);
                let (ax1, ay1) = (a.2 + infl, a.3 + infl);
                let bx0 = b.0.saturating_sub(infl);
                let by0 = b.1.saturating_sub(infl);
                let (bx1, by1) = (b.2 + infl, b.3 + infl);
                !(ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0)
            }
            _ => false,
        }
    };
    let mut stage: Vec<usize> = vec![0; n_groups_total];
    for (oi, &g) in group_order.iter().enumerate() {
        let mut s = 0;
        for &g2 in &group_order[..oi] {
            if conflict(g, g2) {
                s = s.max(stage[g2] + 1);
            }
        }
        stage[g] = s;
    }
    let max_stage = group_order.iter().map(|&g| stage[g]).max().unwrap_or(0);

    for s in 0..=max_stage {
        let batch: Vec<usize> = group_order
            .iter()
            .copied()
            .filter(|&g| stage[g] == s)
            .collect();
        if batch.is_empty() {
            continue;
        }
        // Nets of this stage in (group_order-within-batch, input) order, matching the
        // phase-B iteration below so the parallel results line up by position.
        let batch_nets: Vec<usize> = batch
            .iter()
            .flat_map(|&g| group_nets[g].iter().copied())
            .collect();
        // Phase A (parallel): clean-reuse or WINDOWED route against owner/halo from
        // prior stages. Per-thread scratch via `map_init`; reads only `&` snapshots.
        let owner_ref: &[i64] = &owner;
        let halo_ref: &[i64] = &halo;
        let coords_ref = coords;
        let mut phase_a: Vec<Option<Vec<CellIdx>>> = batch_nets
            .par_iter()
            .map_init(
                || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                |(b, ps), &i| {
                    let gi = group_ids[i] as i64;
                    let net = &nets[i];
                    let cur = &paths[i];
                    let clean = !cur.is_empty()
                        && cur.iter().all(|&c| {
                            let o = owner_ref[c as usize];
                            let h = halo_ref[c as usize];
                            (o < 0 || o == gi) && (h < 0 || h == gi)
                        });
                    if clean {
                        Some(cur.clone())
                    } else {
                        ps.load(&net.passable_pads);
                        route_legal(
                            b, grid, coords_ref, ps, owner_ref, halo_ref, gi, net.src, net.dst,
                            windows[i], via_model, clearance,
                        )
                        .map(|(p, _)| p)
                    }
                },
            )
            .collect();
        // Phase B (serial, group_order): commit; nets that failed the windowed route
        // get the full-board fallback now, against owner incl. earlier same-stage
        // groups. Stamp each group's owner/halo only after the whole group commits.
        let mut k = 0;
        for &g in &batch {
            let gi = g as i64;
            for &i in &group_nets[g] {
                let chosen = match std::mem::take(&mut phase_a[k]) {
                    Some(p) => Some(p),
                    None => {
                        pad_set.load(&nets[i].passable_pads);
                        route_legal(
                            buf,
                            grid,
                            coords,
                            pad_set,
                            &owner,
                            &halo,
                            gi,
                            nets[i].src,
                            nets[i].dst,
                            Window::full(dims),
                            via_model,
                            clearance,
                        )
                        .map(|(p, _)| p)
                    }
                };
                if let Some(path) = chosen {
                    committed[i] = Some(path);
                    group_members.push(i);
                }
                k += 1;
            }
            for &i in &group_members {
                if let Some(path) = &committed[i] {
                    stamp_owner(
                        &mut owner,
                        &mut halo,
                        grid,
                        dims,
                        coords,
                        path,
                        gi,
                        clearance,
                        via_model,
                    );
                }
            }
            group_members.clear();
        }
    }

    committed
}

/// Bounded rip-up-and-reroute legalization.
///
/// Produces a cell-disjoint-across-groups commit that, unlike [`legalize_in_order`],
/// may *displace* an already-committed net to make room for a stranded one. This
/// solves the co-dependent case where net A's natural route blocks B and B's blocks
/// A simultaneously — no static commit order works, you must rip up a committed net.
///
/// Algorithm (work-queue):
/// 1. Seed the queue with every net in `seed_group_order` (the BEST multi-order
///    order, so only the genuinely-hard leftovers cost extra work). Within a group,
///    nets are queued in input order.
/// 2. Pop net `i`. Build a grid where cells owned by OTHER groups are hard obstacles
///    (own pads unmasked), then route `i`. If it routes, commit it (mark its cells
///    owned by `i`'s group). If not, find the committed OTHER-group nets whose cells
///    lie on `i`'s alone path (the blockers); rip up the blocker(s) with the
///    smallest committed path length first (cheapest to re-place), free their cells,
///    commit `i`, and re-enqueue the ripped nets (bumping their rip count).
/// 3. Bounds: a global rip budget and a per-net rip cap; when exhausted, stop
///    ripping and place whatever still fits. ALWAYS terminates.
///
/// Determinism: every tie is broken by `(group_id, net index, cell idx)`; no RNG.
/// Sibling sub-nets of one connection (same group) never block each other because
/// occupancy is tracked per *group*, never per net.
#[allow(clippy::too_many_arguments)]
fn ripup_legalize(
    grid: &Grid,
    coords: &GridCoords,
    buf: &mut SearchBuf,
    pad_set: &mut PadSet,
    nets: &[NetEndpoints],
    group_ids: &[usize],
    alone_path: &[Vec<CellIdx>],
    windows: &[Window],
    seed_group_order: &[usize],
    seed_committed: &Committed,
    n_cells: usize,
    via_model: &ViaModel,
    clearance: f64,
) -> Committed {
    let dims = grid.dims;
    let n_nets = nets.len();

    // A net needs the (expensive) full-board fallback only if its own alone-path
    // genuinely leaves its window. Nets whose alone-path fits the window never gain
    // anything from a full-board search (which, when it fails, explores the whole
    // board) — so we suppress the fallback for them. Nets with an EMPTY alone-path
    // are individually unroutable on the base grid and can never commit at all.
    let needs_full: Vec<bool> = (0..n_nets)
        .map(|i| {
            !alone_path[i].is_empty()
                && alone_path[i].iter().any(|&c| !windows[i].contains(dims, c))
        })
        .collect();
    let routable: Vec<bool> = (0..n_nets).map(|i| !alone_path[i].is_empty()).collect();

    // SEED from the best multi-order legalization instead of re-routing every net
    // from scratch: the parallel multi-order pass already committed most nets, so we
    // start from its solution and only work the genuinely-stranded residue. This
    // skips the (serial, ~O(n_nets) `route_legal`) clean-route pass that otherwise
    // dominated this stage. Owner/halo below are stamped to match the seed. The result
    // can only ADD nets (via rips), so it never routes fewer than the seed.
    let mut committed: Committed = seed_committed.clone();
    // Owning group per committed COPPER cell, or -1 for free: a foreign-group copper
    // cell is a HARD obstacle for net i. Owning group per clearance-HALO cell, or -1:
    // a foreign-group halo cell is ALSO a HARD block in `route_legal`, so a stranded
    // net that cannot route clear is left unrouted rather than violating spacing.
    let mut owner: Vec<i64> = vec![-1; n_cells];
    let mut halo: Vec<i64> = vec![-1; n_cells];
    // Stamp the seeded commits into owner/halo so the residue routes against them.
    for (i, slot) in committed.iter().enumerate() {
        if let Some(path) = slot {
            stamp_owner(
                &mut owner,
                &mut halo,
                grid,
                dims,
                coords,
                path,
                group_ids[i] as i64,
                clearance,
                via_model,
            );
        }
    }

    let mut rip_count: Vec<usize> = vec![0; n_nets];
    let global_budget = RIPUP_GLOBAL_BUDGET_PER_NET * n_nets.max(1);
    let per_net_cap = n_nets + RIPUP_PER_NET_CAP_EXTRA;
    let mut rips_done: usize = 0;

    // FIFO work-queue of net indices to (re)place. Seed in best-order, groups in
    // the seed order and nets within a group in input order, for determinism.
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for &g in seed_group_order {
        for (i, &gi) in group_ids.iter().enumerate() {
            if gi == g {
                queue.push_back(i);
            }
        }
    }
    // ROUND-BASED NO-PROGRESS EARLY-EXIT. The rip-up loop otherwise grinds until the
    // global rip budget (thousands of rips) is exhausted, because a few mutually-
    // contended nets ping-pong (A rips B, re-enqueued B rips A, …) — each cycle an
    // expensive *failing* A* that commits nothing net-new. We mark round boundaries
    // with a sentinel pushed to the back of the queue: re-enqueues during a round
    // land BEHIND the sentinel (next round). When the sentinel surfaces we compare
    // the committed count to the best seen; a round that did not strictly increase it
    // cannot make progress on an identical later pass, so we stop. Productive rounds
    // (which place a net) strictly raise the count, so there are at most `n_nets` of
    // them → fast, bounded termination. The `accept-if-better` gate at the call site
    // means stopping early can at worst fall back to `multi_committed` (never worse).
    // Deterministic: the sentinel and count are pure functions of queue contents.
    const ROUND_MARK: usize = usize::MAX;
    // How many consecutive break-even rounds (committed count unchanged) to tolerate
    // before giving up. A genuine rescue rips a blocker in one round (count breaks
    // even: −victim +stranded) and re-places the blocker in the NEXT round (count
    // rises), so we must allow at least one break-even round; oscillating ping-pong
    // also breaks even, so we cap it. Two is the minimum that admits a one-round
    // rip→replace gap while still bounding the ping-pong.
    const MAX_EVEN_ROUNDS: usize = 2;
    queue.push_back(ROUND_MARK);
    let mut best_count = committed.iter().filter(|c| c.is_some()).count();
    let mut even_rounds = 0usize;

    // Free every cell currently owned by group `g` — both the committed path cells
    // AND their clearance / via-keepout halo (via `free_owner`, the symmetric
    // inverse of `stamp_owner`) so no reserved cell leaks across a rip. Reset
    // committed entries here.
    let free_group_cells = |owner: &mut [i64],
                            halo: &mut [i64],
                            committed: &mut Committed,
                            group_ids: &[usize],
                            g: usize| {
        for i in 0..committed.len() {
            if group_ids[i] == g {
                if let Some(path) = committed[i].take() {
                    free_owner(
                        owner,
                        halo,
                        dims,
                        coords,
                        &path,
                        g as i64,
                        clearance,
                        via_model,
                    );
                }
            }
        }
    };

    while let Some(i) = queue.pop_front() {
        // Round boundary: stop unless this round committed strictly more nets.
        if i == ROUND_MARK {
            if queue.is_empty() {
                break;
            }
            let cur = committed.iter().filter(|c| c.is_some()).count();
            if cur > best_count {
                // Progress: a new high-water of committed nets. Keep going.
                best_count = cur;
                even_rounds = 0;
                queue.push_back(ROUND_MARK);
                continue;
            }
            if cur == best_count && even_rounds + 1 < MAX_EVEN_ROUNDS {
                // Break-even: could be a rip awaiting its re-placement next round.
                // Allow a bounded number before concluding it is unproductive churn.
                even_rounds += 1;
                queue.push_back(ROUND_MARK);
                continue;
            }
            // Net loss (rips freed more than they placed) or too many break-even
            // rounds: unproductive. Stop; the accept-if-better gate keeps the seed.
            break;
        }
        // If already committed (e.g. re-enqueued then satisfied earlier), skip.
        if committed[i].is_some() {
            continue;
        }
        // Individually unroutable on the base grid: it can never commit here, and
        // any search (especially a full-board one) would only fail expensively.
        if !routable[i] {
            continue;
        }
        let net = &nets[i];
        let g = group_ids[i];
        let gi = g as i64;

        // Route within the net's window (cells owned by OTHER groups are hard
        // obstacles), falling back to the full board only for nets whose natural
        // route genuinely leaves the window — so a stranded global net is never
        // ripped-around prematurely, without the whole-board failure cost for the
        // many purely-local nets.
        pad_set.load(&net.passable_pads);
        let routed = route_legal(
            buf, grid, coords, pad_set, &owner, &halo, gi, net.src, net.dst, windows[i],
            via_model, clearance,
        )
        .or_else(|| {
            if needs_full[i] {
                route_legal(
                    buf,
                    grid,
                    coords,
                    pad_set,
                    &owner,
                    &halo,
                    gi,
                    net.src,
                    net.dst,
                    Window::full(dims),
                    via_model,
                    clearance,
                )
            } else {
                None
            }
        });

        if let Some((path, _)) = routed {
            stamp_owner(
                &mut owner,
                &mut halo,
                grid,
                dims,
                coords,
                &path,
                gi,
                clearance,
                via_model,
            );
            committed[i] = Some(path);
            continue;
        }

        // Stranded. Find committed OTHER-group blockers on i's alone path.
        if rips_done >= global_budget || rip_count[i] >= per_net_cap {
            // Budget exhausted for this net: leave it unrouted.
            continue;
        }

        // Collect the distinct blocker groups whose owned cells lie on i's alone
        // path. (Sibling cells of i's own group are never blockers.)
        let mut blocker_groups: Vec<usize> = Vec::new();
        for &c in &alone_path[i] {
            let o = owner[c as usize];
            if o >= 0 && o as usize != g {
                let bg = o as usize;
                if !blocker_groups.contains(&bg) {
                    blocker_groups.push(bg);
                }
            }
        }

        if blocker_groups.is_empty() {
            // Nothing committed is in the way along the natural route, yet it still
            // would not route — it is genuinely unroutable given current commits.
            continue;
        }

        // Rip the blocker group whose total committed path length is smallest
        // (cheapest to re-place); tie-break by lowest group id for determinism.
        blocker_groups.sort_by_key(|&bg| {
            let len: Cost = (0..n_nets)
                .filter(|&j| group_ids[j] == bg)
                .filter_map(|j| committed[j].as_ref())
                .map(|p| unit_cost(p))
                .fold(0, |a, b| a.saturating_add(b));
            (len, bg)
        });

        let victim = blocker_groups[0];

        // Re-enqueue every (currently committed) net of the victim group, in input
        // order, bumping their rip count, then free the group's cells and commit i.
        for j in 0..n_nets {
            if group_ids[j] == victim && committed[j].is_some() {
                rip_count[j] += 1;
                queue.push_back(j);
            }
        }
        free_group_cells(&mut owner, &mut halo, &mut committed, group_ids, victim);
        rips_done += 1;

        // Re-route i now that the victim's cells are free.
        let rerouted = route_legal(
            buf, grid, coords, pad_set, &owner, &halo, gi, net.src, net.dst, windows[i],
            via_model, clearance,
        )
        .or_else(|| {
            if needs_full[i] {
                route_legal(
                    buf,
                    grid,
                    coords,
                    pad_set,
                    &owner,
                    &halo,
                    gi,
                    net.src,
                    net.dst,
                    Window::full(dims),
                    via_model,
                    clearance,
                )
            } else {
                None
            }
        });
        if let Some((path, _)) = rerouted {
            stamp_owner(
                &mut owner,
                &mut halo,
                grid,
                dims,
                coords,
                &path,
                gi,
                clearance,
                via_model,
            );
            committed[i] = Some(path);
        } else {
            // Still cannot route even after the rip: re-enqueue i once (its rip
            // count is bounded by per_net_cap, so this terminates).
            rip_count[i] += 1;
            queue.push_back(i);
        }
    }

    committed
}

/// All permutations of `items`, generated in a deterministic order (Heap-free
/// lexicographic-by-construction recursion: index 0 fixed first, then 1, …).
/// Used only for small group counts (≤ 7), so allocating the full set is cheap.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::with_capacity(items.len());
    let mut used: Vec<bool> = vec![false; items.len()];
    permute_rec(items, &mut used, &mut cur, &mut out);
    out
}

fn permute_rec(
    items: &[usize],
    used: &mut [bool],
    cur: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if cur.len() == items.len() {
        out.push(cur.clone());
        return;
    }
    for i in 0..items.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        cur.push(items[i]);
        permute_rec(items, used, cur, out);
        cur.pop();
        used[i] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::Dims;
    use mr_grid::GridBuilder;

    fn net(name: &str, src: CellIdx, dst: CellIdx) -> NetEndpoints {
        NetEndpoints {
            net: name.into(),
            src,
            dst,
            passable_pads: Vec::new(),
        }
    }

    fn disjoint(a: &[CellIdx], b: &[CellIdx]) -> bool {
        let sa: std::collections::HashSet<_> = a.iter().copied().collect();
        b.iter().all(|c| !sa.contains(c))
    }

    /// Two nets whose individually-shortest straight paths cross in the middle.
    /// The naive (independent) router would conflict on the centre cell; the
    /// negotiated router routes BOTH on cell-disjoint paths.
    #[test]
    fn crossing_nets_route_disjoint() {
        // 5x5 open grid. The endpoints are pulled in one cell from the border so a
        // disjoint crossing solution genuinely exists (each net can detour into the
        // free margin), unlike the degenerate 3x3 where the only routes collide.
        //   A goes (2,1) -> (2,3)  (vertical, centre column)
        //   B goes (1,2) -> (3,2)  (horizontal, centre row)
        // Both straight paths pass through the centre cell (2,2).
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(2, 1), dims.idx(2, 3));
        let b = net("b", dims.idx(1, 2), dims.idx(3, 2));

        // Sanity: the naive shortest paths really do share the centre cell, so a
        // disjoint solution is non-trivial.
        let centre = dims.idx(2, 2);
        let fa = crate::LeeRouter::route_one(&grid, a.src, a.dst).unwrap().0;
        let fb = crate::LeeRouter::route_one(&grid, b.src, b.dst).unwrap().0;
        assert!(
            fa.contains(&centre) && fb.contains(&centre),
            "precondition: naive paths collide at centre"
        );

        let br = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(br.unrouted.is_empty(), "both nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);

        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        assert!(
            disjoint(pa, pb),
            "paths must be cell-disjoint: {pa:?} {pb:?}"
        );
        assert_eq!(pa.first().copied(), Some(a.src));
        assert_eq!(pa.last().copied(), Some(a.dst));
        assert_eq!(pb.first().copied(), Some(b.src));
        assert_eq!(pb.last().copied(), Some(b.dst));
    }

    /// `BoardRoute::groups` is exported aligned 1:1 with `results`, carries the
    /// router's ground-truth electrical-net group id, and gives `#`-sibling nets the
    /// SAME id while a foreign net gets a different one. The DRC relies on this to
    /// grant same-net copper exactly the immunity the router permitted.
    #[test]
    fn route_exports_aligned_group_ids() {
        let dims = Dims::new(8, 8);
        let grid = GridBuilder::new(dims, 1).build();
        // `g#0` and `g#1` share the `group_of` prefix `g` → one group. `foreign` is
        // its own group.
        let s0 = net("g#0", dims.idx(0, 0), dims.idx(7, 0));
        let s1 = net("g#1", dims.idx(0, 2), dims.idx(7, 2));
        let f = net("foreign", dims.idx(0, 5), dims.idx(7, 5));
        let br = NegotiatedRouter::new()
            .route(&grid, &[s0, s1, f])
            .unwrap();
        assert!(br.unrouted.is_empty(), "all nets must route: {br:?}");
        assert_eq!(br.groups.len(), br.results.len(), "groups align 1:1 with results");
        // Map back from results (which are in input order here) to assert grouping.
        let g_of = |name: &str| {
            let i = br.results.iter().position(|r| r.net == name).unwrap();
            br.groups[i]
        };
        assert_eq!(g_of("g#0"), g_of("g#1"), "`#`-siblings share a group id");
        assert_ne!(g_of("g#0"), g_of("foreign"), "a foreign net is a distinct group");
    }

    /// A net must detour around a foreign net's pad (a hard obstacle it does not
    /// own). Both route, cell-disjoint, and the detouring net avoids the pad.
    #[test]
    fn detours_around_foreign_pad() {
        // 7x3 open grid; A's pad is the 2x1 block (3,0),(3,1) (obstacle in base).
        let dims = Dims::new(7, 3);
        let grid = GridBuilder::new(dims, 1).mark_rect(3, 0, 3, 1).build();

        let a_pad: Vec<CellIdx> = vec![dims.idx(3, 0), dims.idx(3, 1)];
        let net_a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(3, 0),
            dst: dims.idx(3, 1),
            passable_pads: a_pad.clone(),
        };
        let net_b = NetEndpoints {
            net: "b".into(),
            src: dims.idx(0, 1),
            dst: dims.idx(6, 1),
            passable_pads: Vec::new(),
        };

        let br = NegotiatedRouter::new()
            .route(&grid, &[net_a, net_b])
            .unwrap();
        assert!(br.unrouted.is_empty(), "both nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);

        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        for c in pb {
            assert!(!a_pad.contains(c), "B must route around A's pad; cell {c}");
        }
        assert!(disjoint(pa, pb), "paths must be cell-disjoint");
    }

    /// An over-constrained single corridor that only one net can use: exactly one
    /// routes, one is unrouted, and the router terminates (no panic / hang).
    #[test]
    fn over_constrained_corridor_one_unrouted() {
        // 3x3 with walls top and bottom of the centre row -> the only path between
        // the left and right centre cells is the single corridor (0,1)-(1,1)-(2,1).
        let dims = Dims::new(3, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_cell(1, 0);
        b.mark_cell(1, 2);
        let grid = b.build();
        // DISTINCT endpoints (so they are not junction-merged into one net), but both
        // can only cross left<->right through the lone corridor cell (1,1): net a runs
        // the centre row, net b drops in from the bottom corners — both must take
        // (1,1), so exactly one routes.
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 2), dims.idx(2, 2)),
        ];
        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(
            br.results.len() + br.unrouted.len(),
            2,
            "every net accounted for"
        );
        assert_eq!(br.unrouted.len(), 1, "both cannot fit the single corridor");
        assert_eq!(br.results.len(), 1);
    }

    /// Determinism: routing the same problem twice yields identical results.
    #[test]
    fn deterministic_results() {
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 0), dims.idx(4, 4)),
            net("b", dims.idx(4, 0), dims.idx(0, 4)),
            net("c", dims.idx(0, 2), dims.idx(4, 2)),
        ];
        let br1 = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        let br2 = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(br1.results, br2.results);
        assert_eq!(br1.unrouted, br2.unrouted);
        assert_eq!(br1.congestion, br2.congestion);
    }

    /// Two chained sub-nets of one connection ("X#0","X#1") share a middle cell.
    /// They are the same group, so the shared cell is NOT overuse and both route.
    #[test]
    fn same_connection_subnets_may_share() {
        // 3x1 line: X#0 routes (0,0)->(1,0); X#1 routes (1,0)->(2,0). They share
        // the middle cell (1,0). Being one connection, that is legal.
        let dims = Dims::new(3, 1);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("X#0", dims.idx(0, 0), dims.idx(1, 0)),
            net("X#1", dims.idx(1, 0), dims.idx(2, 0)),
        ];
        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert!(br.unrouted.is_empty(), "both sub-nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);
        let mid = dims.idx(1, 0);
        assert!(br.results[0].path.contains(&mid));
        assert!(br.results[1].path.contains(&mid));
        // The shared middle cell shows congestion 2 — allowed within one group.
        assert_eq!(br.congestion[mid as usize], 2);
    }

    /// Co-dependent two-net board: BOTH nets must deviate from their greedy path
    /// for a cell-disjoint solution to exist. A vertical wall on column x=2 blocks
    /// rows y=2,3,4, so the only crossing is via rows 0–1. Net A's greedy route is
    /// straight across row 1 — which walls B off from crossing — and net B's greedy
    /// route detours up *through A's endpoints*, stranding A. Neither greedy commit
    /// order works; the disjoint solution forces A up onto row 0 AND B to cross at
    /// (2,1) clear of A's endpoints. The full router must route BOTH.
    #[test]
    fn co_dependent_both_must_deviate() {
        let dims = Dims::new(5, 5);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(2, 2);
        gb.mark_cell(2, 3);
        gb.mark_cell(2, 4);
        let grid = gb.build();

        let a = net("a", dims.idx(0, 1), dims.idx(4, 1));
        let b = net("b", dims.idx(0, 3), dims.idx(4, 3));

        // Precondition: each net's *greedy* path blocks the other. Routing A alone
        // (straight row 1) leaves B with no disjoint route, and vice versa.
        let ga = crate::LeeRouter::route_one(&grid, a.src, a.dst).unwrap().0;
        let gb_path = crate::LeeRouter::route_one(&grid, b.src, b.dst).unwrap().0;
        assert!(
            ga.iter().any(|c| gb_path.contains(c)) || {
                // B's greedy uses A's endpoints, so even where the cell sets don't
                // literally overlap the greedy commit still strands the other.
                ga.contains(&a.src) && gb_path.contains(&a.src)
            },
            "precondition: greedy paths are in mutual conflict"
        );

        let br = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "both co-dependent nets must route: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        assert!(
            disjoint(pa, pb),
            "paths must be cell-disjoint: {pa:?} {pb:?}"
        );
        assert_eq!(pa.first().copied(), Some(a.src));
        assert_eq!(pa.last().copied(), Some(a.dst));
        assert_eq!(pb.first().copied(), Some(b.src));
        assert_eq!(pb.last().copied(), Some(b.dst));
    }

    /// Bounded rip-up actually rescues a net that the multi-order pass alone leaves
    /// unrouted. On this 6x6 board with a single wall at (4,1), the negotiation +
    /// multi-order legalization commits two of the three nets and strands the
    /// third; only displacing an already-committed net (rip-up) frees a corridor
    /// so all three route. We assert the full router routes ALL THREE — which is
    /// only reachable through the rip-up stage (verified empirically against a
    /// rip-up-disabled build during development).
    #[test]
    fn ripup_rescues_third_net() {
        let dims = Dims::new(6, 6);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(4, 1);
        let grid = gb.build();

        let nets = vec![
            net("a", dims.idx(3, 0), dims.idx(4, 3)),
            net("b", dims.idx(0, 4), dims.idx(5, 4)),
            net("c", dims.idx(2, 1), dims.idx(5, 3)),
        ];

        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert!(
            br.unrouted.is_empty(),
            "rip-up must route all three nets: {br:?}"
        );
        assert_eq!(br.results.len(), 3);

        // Cell-disjoint across the three (distinct) groups.
        for i in 0..br.results.len() {
            for j in (i + 1)..br.results.len() {
                assert!(
                    disjoint(&br.results[i].path, &br.results[j].path),
                    "group paths must be cell-disjoint: {} vs {}",
                    br.results[i].net,
                    br.results[j].net
                );
            }
        }
        // Endpoints honoured.
        for (r, n) in br.results.iter().zip(nets.iter()) {
            assert_eq!(r.path.first().copied(), Some(n.src));
            assert_eq!(r.path.last().copied(), Some(n.dst));
        }
    }

    /// Rip-up must terminate and never panic even when no disjoint solution exists.
    /// Re-uses the over-constrained single corridor (two DISTINCT-endpoint nets that
    /// both must cross through the lone corridor cell): rip-up may ping-pong the
    /// corridor but is bounded by its global/per-net budgets, so it stops and returns
    /// the best partial (one net), never regressing below the multi-order result.
    #[test]
    fn ripup_terminates_when_unsolvable() {
        let dims = Dims::new(3, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_cell(1, 0);
        b.mark_cell(1, 2);
        let grid = b.build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 2), dims.idx(2, 2)),
        ];
        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(br.results.len() + br.unrouted.len(), 2);
        assert_eq!(br.unrouted.len(), 1, "single corridor fits only one net");
        // Determinism across repeated runs (rip-up tie-breaks must be stable).
        let br2 = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(br.results, br2.results);
        assert_eq!(br.unrouted, br2.unrouted);
    }

    /// The deterministic permutation generator yields every permutation exactly
    /// once, in a fixed (lexicographic-by-construction) order.
    #[test]
    fn permutations_are_complete_and_ordered() {
        let perms = permutations(&[0, 1, 2]);
        assert_eq!(
            perms,
            vec![
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0],
            ]
        );
        // 4 distinct items -> 24 unique permutations.
        let p4 = permutations(&[0, 1, 2, 3]);
        assert_eq!(p4.len(), 24);
        let uniq: std::collections::HashSet<_> = p4.iter().cloned().collect();
        assert_eq!(uniq.len(), 24);
    }

    /// A net whose optimal path must leave its bbox+margin window still routes,
    /// via the full-board retry. We build a tall thin board with a wall that forces
    /// a long detour far outside the src/dst bounding box, then assert the net
    /// routes and its path actually exits the window (proving the window alone
    /// would have failed and the full-board retry rescued it).
    #[test]
    fn net_routes_via_full_board_retry_when_path_leaves_window() {
        // 40 wide, 6 tall. src=(0,0), dst=(2,2): a tiny bbox in the top-left corner,
        // so the window (margin = max(16, ceil(0.3*2)) = 16) spans x in [0,18]. Row
        // y=1 is walled for x in 0..=37, leaving the ONLY top<->bottom crossing at
        // x=38,39 — far outside the window. The net must run right along y=0 to x=39,
        // drop to y=2, then run left to dst. The windowed search alone fails; the
        // full-board retry rescues it.
        let dims = Dims::new(40, 6);
        let mut gb = GridBuilder::new(dims, 1);
        for x in 0..=37 {
            gb.mark_cell(x, 1);
        }
        let grid = gb.build();

        let a = net("a", dims.idx(0, 0), dims.idx(2, 2));

        // The window for this net: bbox {(0,0),(2,2)} expanded by margin 16 ->
        // x in [0,18]. The forced crossing at x=38/39 lies well outside it.
        let win = net_window(dims, a.src, a.dst, &[]);
        assert!(
            win.x1 < 38,
            "precondition: gap column is outside the window"
        );

        let br = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "net must route via the full-board retry: {br:?}"
        );
        assert_eq!(br.results.len(), 1);
        let path = &br.results[0].path;
        assert_eq!(path.first().copied(), Some(a.src));
        assert_eq!(path.last().copied(), Some(a.dst));
        // The path must leave the window (reach the far-right gap column) — proof
        // that the windowed search alone would have failed.
        assert!(
            path.iter().any(|&c| {
                let (x, _) = dims.xy(c);
                x > win.x1
            }),
            "optimal path must exit the window (reach the far gap)"
        );
    }

    /// Order-robust legalization: a board where committing net A's group first
    /// strands net B, but committing B's group first routes both. The full router
    /// must find the good order and route BOTH nets.
    ///
    /// 5x5 open grid:
    ///   A: (0,2) -> (4,2)  horizontal across the middle row
    ///   B: (2,1) -> (2,3)  vertical across the middle column
    /// If A claims the whole middle row first, it walls off rows y<2 from y>2 and
    /// B (which must get from y=1 to y=3) is stranded. If B claims the middle
    /// column first, A simply detours one row up/down and crosses elsewhere.
    #[test]
    fn legalization_is_order_robust() {
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();

        // Net order is [A, B], i.e. the FAILING first-appearance order.
        let a = net("a", dims.idx(0, 2), dims.idx(4, 2));
        let b = net("b", dims.idx(2, 1), dims.idx(2, 3));

        // Direct check of the legalization primitive: hand-crafted negotiated
        // paths that collide on the centre cell (2,2). One commit order strands a
        // net; the other routes both — proving the order genuinely matters.
        let n_cells = dims.len();
        let group_ids = vec![0usize, 1usize];
        let nets_ab = [a.clone(), b.clone()];
        let crafted = vec![
            // A: full middle row.
            vec![
                dims.idx(0, 2),
                dims.idx(1, 2),
                dims.idx(2, 2),
                dims.idx(3, 2),
                dims.idx(4, 2),
            ],
            // B: full middle column.
            vec![dims.idx(2, 1), dims.idx(2, 2), dims.idx(2, 3)],
        ];
        let mut buf = SearchBuf::new(n_cells);
        let mut pad_set = PadSet::new(n_cells);
        let windows: Vec<Window> = nets_ab
            .iter()
            .map(|net| net_window(dims, net.src, net.dst, &net.passable_pads))
            .collect();

        // Single-layer board: the via model is inert (no via neighbours exist).
        let via_model = ViaModel::through_hole(dims.layers);
        // Abstract Dims-only grid: uniform unit coords reproduce unit-hop pricing.
        let coords = GridCoords::uniform(dims);

        // Order [0,1] = A first: B should be stranded.
        let c_ab = legalize_in_order(
            &grid,
            &coords,
            &mut buf,
            &mut pad_set,
            &nets_ab,
            &group_ids,
            &crafted,
            &windows,
            &[0, 1],
            n_cells,
            &via_model,
            0.0,
        );
        assert!(c_ab[0].is_some(), "A commits in A-first order");
        assert!(
            c_ab[1].is_none(),
            "B must be stranded when A claims the middle row first"
        );

        // Order [1,0] = B first: both route.
        let c_ba = legalize_in_order(
            &grid,
            &coords,
            &mut buf,
            &mut pad_set,
            &nets_ab,
            &group_ids,
            &crafted,
            &windows,
            &[1, 0],
            n_cells,
            &via_model,
            0.0,
        );
        assert!(
            c_ba[0].is_some() && c_ba[1].is_some(),
            "both route when B claims the middle column first"
        );

        // The full router must pick the good order and route BOTH nets even though
        // first-appearance order [A, B] alone would strand B.
        let br = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "order-robust router must route both nets: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        assert!(
            disjoint(pa, pb),
            "paths must be cell-disjoint: {pa:?} {pb:?}"
        );
    }

    /// A 2-layer board whose ONLY planar corridor on layer 0 is fully walled off
    /// between src and dst. The net is unroutable on layer 0 alone but routes by
    /// via-ing up to layer 1, crossing the (open) wall column there, and via-ing
    /// back down. The path must therefore touch BOTH layers.
    #[test]
    fn vias_route_around_a_full_layer0_wall() {
        // 3 wide, 1 tall, 2 layers. A wall at (1,0) on layer 0 splits the single
        // row into two halves; there is no layer-0 path from (0,0) to (2,0).
        // `GridBuilder::mark_cell` marks layer 0 only, so layer 1 is fully open.
        let dims = Dims::with_layers(3, 1, 2);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(1, 0); // wall on layer 0 at x=1
        let grid = gb.build();

        // Precondition: layer 0 alone has no path (the wall is the whole corridor).
        assert!(grid.is_obstacle(dims.idx3(1, 0, 0)));
        assert!(
            !grid.is_obstacle(dims.idx3(1, 0, 1)),
            "layer 1 must be open"
        );

        let a = net("a", dims.idx3(0, 0, 0), dims.idx3(2, 0, 0));
        let br = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "net must route by changing layers: {br:?}"
        );
        assert_eq!(br.results.len(), 1);

        let path = &br.results[0].path;
        assert_eq!(path.first().copied(), Some(a.src));
        assert_eq!(path.last().copied(), Some(a.dst));
        // The path uses both layers (a via up, cross, a via back down).
        assert!(
            path.iter().any(|&c| dims.layer_of(c) == 0),
            "path must touch layer 0: {path:?}"
        );
        assert!(
            path.iter().any(|&c| dims.layer_of(c) == 1),
            "path must via onto layer 1 to cross the wall: {path:?}"
        );
        // Every step is either a same-layer 4-neighbour or a same-(x,y) via step.
        for w in path.windows(2) {
            let planar = dims.neighbors4(w[0]).contains(&w[1]);
            let via = dims.via_neighbors(w[0]).contains(&w[1]);
            assert!(planar || via, "illegal step {} -> {}", w[0], w[1]);
        }
    }

    /// Same wall as above, but a restricted [`ViaModel`] forbids the only layer
    /// transition (0<->1). With no legal via and no layer-0 corridor, the net is
    /// unroutable.
    #[test]
    fn forbidden_via_step_leaves_net_unrouted() {
        let dims = Dims::with_layers(3, 1, 2);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(1, 0);
        let grid = gb.build();

        // Two layers but the model permits NO adjacent step (empty allow-list), so
        // the 0<->1 transition is illegal and the wall cannot be bypassed.
        let vm = ViaModel::with_allowed_steps(2, ViaModel::DEFAULT_STEP_COST, Vec::new());
        let a = net("a", dims.idx3(0, 0, 0), dims.idx3(2, 0, 0));
        let br = NegotiatedRouter::new()
            .with_via_model(vm)
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        assert_eq!(br.results.len(), 0, "no route exists without a legal via");
        assert_eq!(br.unrouted, vec!["a".to_string()]);
    }

    /// `layers == 1` regression: a known single-layer board must route exactly as
    /// it did before vias existed (same path cost). Vias add nothing because
    /// `via_neighbors` is empty on a single-layer grid.
    #[test]
    fn single_layer_routes_identically_with_vias_available() {
        // 5x5 open grid, straight Manhattan net: the optimal cost is the bbox
        // Manhattan distance (4) regardless of which monotone path is taken.
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(0, 0), dims.idx(4, 0));

        // Default router (synthesises a through-hole model over 1 layer) and an
        // explicit through-hole model must both produce the same single-layer route.
        let br_default = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        let br_vm = NegotiatedRouter::new()
            .with_via_model(ViaModel::through_hole(1))
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();

        assert!(br_default.unrouted.is_empty());
        assert_eq!(br_default.results.len(), 1);
        assert_eq!(br_default.results[0].cost, 4, "straight 5-wide run costs 4");
        // Every cell stays on layer 0 (there is no other layer).
        assert!(br_default.results[0]
            .path
            .iter()
            .all(|&c| dims.layer_of(c) == 0));
        // The via model makes no difference on a single-layer board.
        assert_eq!(br_default.results, br_vm.results);
        assert_eq!(br_default.congestion, br_vm.congestion);
    }

    /// True iff any cell of `a` lies within the 8-neighbourhood (Chebyshev
    /// radius one) of any cell of `b` on the same layer — i.e. the two paths touch
    /// or sit adjacent. With `clearance_cells >= 1` two distinct nets must not.
    fn within_one(dims: Dims, a: &[CellIdx], b: &[CellIdx]) -> bool {
        let sb: std::collections::HashSet<_> = b.iter().copied().collect();
        a.iter().any(|&c| {
            let (x, y, l) = dims.xyz(c);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx as u32 >= dims.w || ny as u32 >= dims.h {
                        continue;
                    }
                    if sb.contains(&dims.idx3(nx as u32, ny as u32, l)) {
                        return true;
                    }
                }
            }
            false
        })
    }

    /// Clearance is now a SOFT cost: the halo makes spacing PREFERRED, not forced.
    /// On a roomy board where a clearance-legal route exists at modest extra cost,
    /// the router routes BOTH nets and keeps their committed copper Chebyshev-clear
    /// (no shared 8-neighbour cell). Each net still owns its copper hard-disjointly.
    ///
    /// Endpoints are pulled away from each other so a clearance-legal detour is
    /// genuinely available: A runs straight along row 1, B's endpoints sit on row 4
    /// (three rows below A) so B can run straight on row 4, well clear of A's halo —
    /// the cheapest route is already the clearance-legal one.
    #[test]
    fn clearance_halo_prefers_spacing_when_room() {
        // 8 wide, 6 tall, wide open. A on row 1, B on row 4 — there is plenty of
        // room for both to stay ≥2 rows apart.
        let dims = Dims::new(8, 6);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(0, 1), dims.idx(7, 1));
        let b = net("b", dims.idx(0, 4), dims.idx(7, 4));

        // clearance 1: both route AND, because the board is roomy, the router prefers
        // the clearance-legal layout — the two paths share no 8-neighbour cell.
        let br = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "soft clearance: both nets must still route on a roomy board: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let (pa, pb) = (&br.results[0].path, &br.results[1].path);
        assert!(disjoint(pa, pb), "copper must stay hard-disjoint");
        assert!(
            !within_one(dims, pa, pb),
            "soft clearance: on a roomy board the router PREFERS a clearance-legal \
             layout (no shared 8-neighbour cell): {pa:?} {pb:?}"
        );
    }

    /// HARD clearance: `route_legal` (the committing pass) never commits copper into
    /// a foreign net's clearance halo. On a congested fixture where the only route
    /// home for net B would have to cross A's halo, B is now left UNROUTED rather than
    /// silently violating clearance — the accepted trade. Whatever DOES route stays
    /// clearance-clean (no path cell within the halo radius of a foreign net).
    #[test]
    fn hard_clearance_drops_net_rather_than_violate() {
        // 3 wide, 5 tall. Net A runs straight down column 1, rows 0..=3 — a vertical
        // wall of copper that leaves only the single cell (1,4) free in column 1. Net
        // B must cross from (0,2) to (2,2): the ONLY copper-free crossing of column 1
        // is through (1,4), which is Chebyshev-adjacent to A's copper at (1,3) — i.e.
        // squarely inside A's clearance halo. With a HARD halo (radius 1) that cell is
        // blocked for B, walling it off from its target. B MUST be dropped, not routed
        // through the halo. (Under the old soft cost B routed through at a violation.)
        let dims = Dims::new(3, 5);
        let a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(1, 0),
            dst: dims.idx(1, 3),
            passable_pads: Vec::new(),
        };
        let grid = GridBuilder::new(dims, 1).build();
        let b = net("b", dims.idx(0, 2), dims.idx(2, 2));

        let br = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // Exactly one net must be left unrouted: B cannot reach its target without
        // violating A's clearance, so the hard constraint drops it (congested).
        assert!(
            !br.unrouted.is_empty(),
            "hard clearance must leave the congested net unrouted, not violate: {br:?}"
        );

        // Whatever routed is clearance-clean: no committed path cell sits within the
        // clearance radius of a DIFFERENT net's copper.
        for r1 in &br.results {
            for r2 in &br.results {
                if r1.net == r2.net {
                    continue;
                }
                assert!(
                    !within_one(dims, &r1.path, &r2.path),
                    "committed copper must never violate clearance: {r1:?} vs {r2:?}"
                );
            }
        }
    }

    /// Empty-clearance fast path is UNCHANGED by the hardening: on the exact same
    /// congested fixture, with `clearance = 0` and the default (keepout 0) via model,
    /// the halo is empty so `route_legal`'s foreign-halo hard block and via-ring guard
    /// are both inert — BOTH nets route (B passes through (1,4) freely), reproducing
    /// the pre-clearance behaviour byte-identically.
    #[test]
    fn hard_clearance_fast_path_unchanged_when_clearance_zero() {
        let dims = Dims::new(3, 5);
        let a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(1, 0),
            dst: dims.idx(1, 3),
            passable_pads: Vec::new(),
        };
        let grid = GridBuilder::new(dims, 1).build();
        let b = net("b", dims.idx(0, 2), dims.idx(2, 2));

        let br = NegotiatedRouter::new()
            .with_clearance_cells(0)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // No halo at all → both nets route (B is NOT dropped); copper stays disjoint.
        assert!(
            br.unrouted.is_empty(),
            "clearance-0 fast path: both nets must route (halo inert): {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let ra = br.results.iter().find(|r| r.net == "a").expect("A routes");
        let rb = br.results.iter().find(|r| r.net == "b").expect("B routes");
        assert!(
            disjoint(&ra.path, &rb.path),
            "distinct nets must never overlap copper: {ra:?} {rb:?}"
        );
    }

    /// Via keepout: a placed via reserves an annular ring around itself. Soft pressure
    /// steers nets apart during negotiation; the committing legalization pass enforces
    /// the ring HARD (no foreign copper may sit inside it). On a 2-layer board whose
    /// only layer-0 corridor is walled, net A must via up at a chokepoint to cross.
    /// With `via_model.keepout_mm = 1.0` and ample room for B to via up at a DIFFERENT
    /// row, the router routes BOTH nets and keeps B's committed copper out of A's
    /// reserved via neighbourhood. We assert both route and B's copper shares no cell
    /// of A's via 8-neighbourhood.
    #[test]
    fn via_keepout_prefers_clear_neighbourhood_when_room() {
        // 5 wide, 5 tall, 2 layers. Wall the WHOLE of layer 0 column x=2 so the only
        // way past x=2 on layer 0 is to via up to layer 1, cross, and via back. The
        // extra height (5 rows) gives B room to via at a row well clear of A's via.
        let dims = Dims::with_layers(5, 5, 2);
        let mut gb = GridBuilder::new(dims, 1);
        for y in 0..5 {
            gb.mark_cell(2, y);
        }
        let grid = gb.build();
        // Sanity: layer 0 column x=2 is fully walled; layer 1 is open there.
        for y in 0..5 {
            assert!(grid.is_obstacle(dims.idx3(2, y, 0)));
            assert!(!grid.is_obstacle(dims.idx3(2, y, 1)));
        }

        // A crosses the wall along row 0 (top edge); B crosses along row 4 (bottom
        // edge) — four rows apart, so each can via clear of the other's keepout.
        let a = net("a", dims.idx3(0, 0, 0), dims.idx3(4, 0, 0));
        let b = net("b", dims.idx3(0, 4, 0), dims.idx3(4, 4, 0));

        let vm = {
            let mut m = ViaModel::through_hole(2);
            // Uniform unit grid: 1.0 mm keepout == the former 1-cell Chebyshev box.
            m.keepout_mm = 1.0;
            m
        };
        let br = NegotiatedRouter::new()
            .with_via_model(vm)
            .with_clearance_cells(0)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // Both route (each can always via across; soft keepout never drops a net).
        assert!(
            br.unrouted.is_empty(),
            "soft via keepout: both nets must route: {br:?}"
        );
        let ra = br
            .results
            .iter()
            .find(|r| r.net == "a")
            .expect("A must route");
        let rb = br
            .results
            .iter()
            .find(|r| r.net == "b")
            .expect("B must route");
        // Collect A's via (x,y) chokepoints: consecutive same-(x,y), layer-changing.
        let mut via_neigh: std::collections::HashSet<CellIdx> = std::collections::HashSet::new();
        for w in ra.path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                // 8-neighbourhood on BOTH via layers (excluding the via cells).
                for &l in &[al, bl] {
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let nx = ax as i64 + dx;
                            let ny = ay as i64 + dy;
                            if nx < 0 || ny < 0 || nx as u32 >= dims.w || ny as u32 >= dims.h {
                                continue;
                            }
                            let nc = dims.idx3(nx as u32, ny as u32, l);
                            if !grid.is_obstacle(nc) {
                                via_neigh.insert(nc);
                            }
                        }
                    }
                }
            }
        }
        assert!(
            !via_neigh.is_empty(),
            "A must place at least one via: {ra:?}"
        );

        // With room to spare, the router PREFERS keeping B's copper out of A's
        // reserved via neighbourhood (the soft keepout steers B away).
        for &c in &rb.path {
            assert!(
                !via_neigh.contains(&c) || ra.path.contains(&c),
                "soft via keepout (roomy board): B's copper should avoid A's reserved \
                 via neighbourhood at cell {c}: {rb:?}"
            );
        }
    }

    /// Regression: `clearance_cells = 0` reproduces an existing golden result. We
    /// re-run the exact `crossing_nets_route_disjoint` scenario through the default
    /// (clearance 0) router and an explicit `with_clearance_cells(0)` router and
    /// assert both match the known-good disjoint outcome byte-for-byte.
    #[test]
    fn clearance_zero_reproduces_golden() {
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(2, 1), dims.idx(2, 3));
        let b = net("b", dims.idx(1, 2), dims.idx(3, 2));

        let br_default = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        let br_zero = NegotiatedRouter::new()
            .with_clearance_cells(0)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // Byte-identical: explicit clearance 0 == the default.
        assert_eq!(br_default.results, br_zero.results);
        assert_eq!(br_default.unrouted, br_zero.unrouted);
        assert_eq!(br_default.congestion, br_zero.congestion);

        // And it reproduces the golden disjoint result.
        assert!(br_zero.unrouted.is_empty());
        assert_eq!(br_zero.results.len(), 2);
        assert!(disjoint(&br_zero.results[0].path, &br_zero.results[1].path));
    }

    /// Determinism under parallelism: a clearance-ACTIVE route (which now takes the
    /// snapshot-based rayon-parallel negotiation path) must produce a byte-identical
    /// [`BoardRoute`] across two runs. The deterministic snapshot + index-ordered
    /// merge make the result independent of how rayon schedules the parallel net
    /// searches, so two runs of the same problem agree exactly.
    #[test]
    fn clearance_active_route_is_deterministic_under_parallelism() {
        // A handful of nets on a roomy board so several route in parallel each
        // iteration. clearance_cells = 1 makes `clearance_active` true → parallel path.
        let dims = Dims::new(12, 12);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 0), dims.idx(11, 11)),
            net("b", dims.idx(11, 0), dims.idx(0, 11)),
            net("c", dims.idx(0, 5), dims.idx(11, 5)),
            net("d", dims.idx(5, 0), dims.idx(5, 11)),
            net("e", dims.idx(0, 9), dims.idx(11, 2)),
        ];

        let router = NegotiatedRouter::new().with_clearance_cells(1);
        let br1 = router.route(&grid, &nets).unwrap();
        let br2 = router.route(&grid, &nets).unwrap();

        // Byte-identical across runs despite parallel scheduling.
        assert_eq!(br1.results, br2.results);
        assert_eq!(br1.unrouted, br2.unrouted);
        assert_eq!(br1.congestion, br2.congestion);
    }

    /// Parallel + clearance still routes all nets on a roomy board (a
    /// timing-independent correctness check on the parallel negotiation path). Three
    /// well-separated nets with clearance active must all route, cell-disjoint.
    #[test]
    fn clearance_active_parallel_routes_all_on_roomy_board() {
        let dims = Dims::new(10, 10);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(9, 1)),
            net("b", dims.idx(0, 5), dims.idx(9, 5)),
            net("c", dims.idx(0, 8), dims.idx(9, 8)),
        ];

        let br = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route(&grid, &nets)
            .unwrap();

        assert!(
            br.unrouted.is_empty(),
            "parallel clearance route must place all nets on a roomy board: {br:?}"
        );
        assert_eq!(br.results.len(), 3);
        for i in 0..br.results.len() {
            for j in (i + 1)..br.results.len() {
                assert!(
                    disjoint(&br.results[i].path, &br.results[j].path),
                    "distinct nets must be cell-disjoint"
                );
            }
        }
    }

    // ---- heuristic admissibility (lower-bound) property ----------------------
    //
    // The search pays, per planar step `u -> v`, a base of `edge_cost(gap_uv)` (plus
    // non-negative congestion/via terms). So the planar base paid along any real path
    // is the SUM of per-step roundings `Σ edge_cost(g_i)`. The heuristic must never
    // exceed that, or A* (with `astar_buf`'s break-on-pop-dst) can return a non-optimal
    // path. The old `edge_cost(manhattan_len(a, b))` rounded the AGGREGATE length once,
    // and round-of-sum can exceed sum-of-rounds, so it could overestimate on
    // non-integer line spacings. These tests pin the lower-bound property the new
    // per-axis summed form (`manhattan_scaled`) guarantees, and exhibit a concrete
    // spacing where the OLD form violated it.

    /// What the search actually pays in planar base cost along a sequence of cells:
    /// `Σ edge_cost(manhattan_len(step))` over consecutive pairs — the exact per-step
    /// rounding `astar_buf`'s `cost_fn` accrues (congestion-free).
    fn summed_path_base(dims: Dims, coords: &GridCoords, path: &[CellIdx]) -> Cost {
        path.windows(2)
            .map(|w| edge_cost(coords.manhattan_len(dims, w[0], w[1])))
            .fold(0u32, Cost::saturating_add)
    }

    /// The OLD (inadmissible) heuristic: a single rounding of the aggregate length.
    fn old_manhattan_planar(dims: Dims, coords: &GridCoords, a: CellIdx, b: CellIdx) -> Cost {
        edge_cost(coords.manhattan_len(dims, a, b))
    }

    #[test]
    fn heuristic_is_lower_bound_collinear_and_l_shaped() {
        // Sweep a range of non-integer gap patterns on a single layer (no via term),
        // and assert the heuristic never exceeds the per-step summed path base for
        // both a straight (collinear) path and an L-shaped path between the corners.
        let gap_sets: &[&[f64]] = &[
            &[0.5, 0.5, 0.5, 0.5],          // halves: round-of-sum vs sum-of-rounds
            &[0.03125, 0.03125, 0.03125],   // 0.5/16 each: each rounds to 0, sum doesn't
            &[1.5, 2.5, 0.5, 3.5],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[0.46875, 0.46875, 0.46875],   // 7.5/16 each
            &[2.0, 2.0, 2.0],               // integers: must match exactly (uniform-like)
        ];
        for xs in gap_sets {
            for ys in gap_sets {
                let x_lines = cumsum(xs);
                let y_lines = cumsum(ys);
                let dims = Dims::new(x_lines.len() as u32, y_lines.len() as u32);
                let coords = GridCoords::from_lines(x_lines.clone(), y_lines.clone());
                let a = dims.idx(0, 0);
                let b = dims.idx(dims.w - 1, dims.h - 1);

                let h = manhattan_scaled(dims, &coords, a, b, 0);

                // Collinear leg along x (then y handled by the y-only pair below).
                let x_only = manhattan_scaled(dims, &coords, dims.idx(0, 0), dims.idx(dims.w - 1, 0), 0);
                let x_path: Vec<CellIdx> = (0..dims.w).map(|x| dims.idx(x, 0)).collect();
                assert!(
                    x_only <= summed_path_base(dims, &coords, &x_path),
                    "collinear-x heuristic must be a lower bound: h={x_only} > path"
                );

                // L-shaped path: straight along x to the far column, then up in y.
                let mut l_path: Vec<CellIdx> = (0..dims.w).map(|x| dims.idx(x, 0)).collect();
                l_path.extend((1..dims.h).map(|y| dims.idx(dims.w - 1, y)));
                let l_cost = summed_path_base(dims, &coords, &l_path);
                assert!(
                    h <= l_cost,
                    "L-shaped heuristic must be a lower bound: h={h} > path={l_cost} \
                     (xs={xs:?}, ys={ys:?})"
                );

                // The other L (up first, then across) — same corners, also a lower bound.
                let mut l_path2: Vec<CellIdx> = (0..dims.h).map(|y| dims.idx(0, y)).collect();
                l_path2.extend((1..dims.w).map(|x| dims.idx(x, dims.h - 1)));
                let l_cost2 = summed_path_base(dims, &coords, &l_path2);
                assert!(
                    h <= l_cost2,
                    "L-shaped(2) heuristic must be a lower bound: h={h} > path={l_cost2}"
                );
            }
        }
    }

    #[test]
    fn old_heuristic_could_overestimate_but_new_does_not() {
        // Concrete witness: three gaps of 0.4/16 mm each. Per-step the search pays
        // edge_cost(0.4/16) = round(0.4) = 0 per step → 0 total. The aggregate is
        // 1.2/16 → edge_cost = round(1.2) = 1. So the OLD heuristic returns 1 > 0,
        // overestimating (inadmissible: round-of-sum > sum-of-rounds). The new summed
        // heuristic returns 0 == 0.
        let g = 0.4 / COST_SCALE;
        let x_lines = cumsum(&[g, g, g]); // 4 lines, 3 gaps
        let dims = Dims::new(x_lines.len() as u32, 1);
        let coords = GridCoords::from_lines(x_lines, vec![0.0]);
        let a = dims.idx(0, 0);
        let b = dims.idx(dims.w - 1, 0);
        let path: Vec<CellIdx> = (0..dims.w).map(|x| dims.idx(x, 0)).collect();
        let summed = summed_path_base(dims, &coords, &path);

        let old = old_manhattan_planar(dims, &coords, a, b);
        let new = manhattan_scaled(dims, &coords, a, b, 0);

        assert_eq!(summed, 0, "search pays 0 per-step here");
        assert!(
            old > summed,
            "old heuristic must overestimate in this witness: old={old} summed={summed}"
        );
        assert!(
            new <= summed,
            "new heuristic must stay a lower bound: new={new} summed={summed}"
        );
    }

    #[test]
    fn heuristic_uniform_grid_byte_identical() {
        // On a uniform unit grid the new summed heuristic must equal both the old
        // aggregate form and the historical (dx + dy) * SCALE, preserving byte-identity.
        let dims = Dims::new(7, 5);
        let coords = GridCoords::uniform(dims);
        for &(ax, ay, bx, by) in &[(0u32, 0u32, 6u32, 4u32), (3, 1, 3, 4), (0, 2, 6, 2)] {
            let a = dims.idx(ax, ay);
            let b = dims.idx(bx, by);
            let new = manhattan_scaled(dims, &coords, a, b, 0);
            let old = old_manhattan_planar(dims, &coords, a, b);
            let expected = (ax.abs_diff(bx) + ay.abs_diff(by)) * SCALE;
            assert_eq!(new, old, "uniform: new must equal old aggregate form");
            assert_eq!(new, expected, "uniform: new must equal (dx+dy)*SCALE");
        }
    }

    /// Prefix sums starting at 0.0 → a sorted line array of `gaps.len() + 1` lines.
    fn cumsum(gaps: &[f64]) -> Vec<f64> {
        let mut lines = Vec::with_capacity(gaps.len() + 1);
        let mut acc = 0.0;
        lines.push(acc);
        for &g in gaps {
            acc += g;
            lines.push(acc);
        }
        lines
    }
}
