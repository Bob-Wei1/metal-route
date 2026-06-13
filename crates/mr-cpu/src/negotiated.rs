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

use mr_core::{
    BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError, ViaModel,
    OBSTACLE,
};

use crate::dijkstra::{astar_buf, SearchBuf};

/// Fixed-point cost scale: the base cost of stepping onto a passable cell.
pub const SCALE: Cost = 16;

/// Soft clearance penalty (TritonRoute `objCost`-style): the extra price a net pays
/// to step onto a cell that lies in *another* group's committed clearance halo
/// during legalization. Clearance is a SOFT cost, not a hard block — a net must
/// never be DROPPED merely because the only route home crosses a foreign net's
/// spacing halo. Instead it routes through the halo at this high penalty (a recorded
/// clearance violation), preserving connectivity. Foreign COPPER (committed path
/// cells) remains a HARD block; two distinct nets must never overlap.
///
/// Chosen large relative to [`SCALE`] (`16 * SCALE`) so the A* search strongly
/// prefers any clearance-legal detour but will violate as a last resort. The priced
/// cost is always capped strictly below [`OBSTACLE`] so a penalized cell is never
/// confused with an impassable one.
pub const CLEARANCE_PENALTY: Cost = 16 * SCALE;

/// Maximum negotiation iterations before falling through to legalization.
pub const MAX_ITERS: u32 = 60;

/// Free-cell cost used when unmasking a net's own pad cells (mirrors the base grid
/// convention used by `GridBuilder`, which fills passable cells with cost 1).
const FREE_COST: Cost = 1;

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
    /// Planar Chebyshev clearance radius (in cells) reserved around committed
    /// copper during legalization to keep *other* nets away — a net's committed
    /// track owns not just its path cells but a halo of this radius on each cell's
    /// own layer, so foreign nets must stay at least this far from it (minimum
    /// spacing). `0` = disabled, i.e. today's behaviour (only the path cells are
    /// owned). The halo never overwrites a cell already owned by another group and
    /// never claims a base-obstacle cell, so it cannot wall a foreign net off from
    /// its own pad.
    clearance_cells: u32,
}

impl NegotiatedRouter {
    pub fn new() -> Self {
        Self {
            via_model: None,
            clearance_cells: 0,
        }
    }

    /// Use an explicit [`ViaModel`] (e.g. a blind/buried stackup) instead of the
    /// default through-hole model. Builder-style; returns the configured router.
    pub fn with_via_model(mut self, vm: ViaModel) -> Self {
        self.via_model = Some(vm);
        self
    }

    /// Set the planar Chebyshev clearance radius (in cells) reserved around
    /// committed copper, so distinct nets keep a minimum spacing of `n` cells.
    /// Builder-style; returns the configured router. `0` (the default) disables the
    /// halo and reproduces the byte-identical pre-clearance behaviour. See the
    /// [`clearance_cells`](NegotiatedRouter#structfield.clearance_cells) field.
    pub fn with_clearance_cells(mut self, n: u32) -> Self {
        self.clearance_cells = n;
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

/// Manhattan distance between two cells, scaled by [`SCALE`], plus the layer
/// distance priced at the cheapest legal via step (`min_via_cost`). Admissible
/// for the 3D per-cell cost model: every planar step costs at least `SCALE` and
/// every layer change costs at least `min_via_cost`, so reaching `b` from `a`
/// requires at least `|dx|+|dy|` planar steps and `|layer(a)-layer(b)|` via
/// steps. The estimate therefore never overestimates and keeps A* optimal while
/// pruning the frontier. On a single-layer grid the layer term is always 0.
fn manhattan_scaled(dims: mr_core::Dims, a: CellIdx, b: CellIdx, min_via_cost: Cost) -> Cost {
    let (ax, ay) = dims.xy(a);
    let (bx, by) = dims.xy(b);
    let dx = ax.abs_diff(bx);
    let dy = ay.abs_diff(by);
    let dl = dims.layer_of(a).abs_diff(dims.layer_of(b));
    (dx + dy)
        .saturating_mul(SCALE)
        .saturating_add(dl.saturating_mul(min_via_cost))
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

        // Reusable search workspace and own-pad membership set, sized once to the
        // board and reused across every per-net search (no per-net O(n) work).
        let mut buf = SearchBuf::new(n_cells);
        let mut pad_set = PadSet::new(n_cells);

        // Connection group id per net (interned, deterministic by first appearance).
        let mut group_ids: Vec<usize> = vec![0; n_nets];
        {
            let mut seen: HashMap<&str, usize> = HashMap::new();
            for (i, net) in nets.iter().enumerate() {
                let g = group_of(&net.net);
                let next = seen.len();
                let id = *seen.entry(g).or_insert(next);
                group_ids[i] = id;
            }
        }

        // Persistent congestion state.
        let mut history: Vec<u32> = vec![0; n_cells];
        let mut present: Vec<u32> = vec![0; n_cells];
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
        let incremental = n_nets > 8;
        // `overused` from the previous iteration (cell -> was it over-used). Empty
        // before the first iteration (everything reroutes).
        let mut prev_overused: Vec<bool> = vec![false; n_cells];
        let mut prev_overused_cells: Vec<CellIdx> = Vec::new();

        // Per-iteration overuse scratch, allocated once and cleared incrementally
        // (via the touched-cell lists) so no iteration pays an O(all cells) memset.
        let mut first_group: Vec<i64> = vec![-1; n_cells];
        let mut overused: Vec<bool> = vec![false; n_cells];

        for iter in 0..MAX_ITERS {
            let pfac: u32 = 1 + iter;

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

                // Remove this net's old path from `present` before pricing.
                for &c in &paths[i] {
                    present[c as usize] = present[c as usize].saturating_sub(1);
                }
                paths[i].clear();

                pad_set.load(&net.passable_pads);

                // Route within the net's window; on failure, retry once on the
                // full board so the occasional global net still completes.
                let routed = route_negotiated(
                    &mut buf, grid, &pad_set, &present, &history, pfac, net.src, net.dst,
                    windows[i], &via_model,
                )
                .or_else(|| {
                    route_negotiated(
                        &mut buf,
                        grid,
                        &pad_set,
                        &present,
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
                    paths[i] = path;
                }
                // else: leave unrouted this iteration (no contribution to present).
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
            // base grid. Route within the window first, full board on failure.
            let no_owner: Vec<i64> = Vec::new();
            let no_halo: Vec<i64> = Vec::new();
            for i in 0..n_nets {
                let net = &nets[i];
                pad_set.load(&net.passable_pads);
                let routed = route_legal(
                    &mut buf, grid, &pad_set, &no_owner, &no_halo, -1, net.src, net.dst,
                    windows[i], &via_model,
                )
                .or_else(|| {
                    route_legal(
                        &mut buf,
                        grid,
                        &pad_set,
                        &no_owner,
                        &no_halo,
                        -1,
                        net.src,
                        net.dst,
                        Window::full(dims),
                        &via_model,
                    )
                });
                if let Some((path, _)) = routed {
                    alone_len[i] = unit_cost(&path);
                    alone_path[i] = path;
                }
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
        // 4. For few groups, exhaustively try every order (≤ 7! = 5040).
        if n_groups <= 7 {
            for perm in permutations(&base_order) {
                candidates.push(perm);
            }
        }

        // Evaluate each candidate and keep the best. "Best" = most nets routed,
        // then lowest total unit cost, then lexicographically lowest group order
        // (for determinism). The group order is carried alongside the result so it
        // can serve as the final tie-break directly.
        let mut best: Option<(usize, Cost, Vec<usize>, Committed)> = None;
        for order in &candidates {
            let committed = legalize_in_order(
                grid,
                &mut buf,
                &mut pad_set,
                nets,
                &group_ids,
                &paths,
                &windows,
                order,
                n_cells,
                &via_model,
                self.clearance_cells,
            );
            let routed = committed.iter().filter(|c| c.is_some()).count();
            let total_cost: Cost = committed
                .iter()
                .filter_map(|c| c.as_ref())
                .map(|p| unit_cost(p))
                .fold(0, |a, b| a.saturating_add(b));
            let better = match &best {
                None => true,
                Some((br, bc, bo, _)) => {
                    routed > *br
                        || (routed == *br && total_cost < *bc)
                        || (routed == *br && total_cost == *bc && order < bo)
                }
            };
            if better {
                best = Some((routed, total_cost, order.clone(), committed));
            }
        }

        let (best_routed, best_order, multi_committed) = best
            .map(|(r, _, o, c)| (r, o, c))
            .unwrap_or_else(|| (0, base_order.clone(), vec![None; n_nets]));

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
                &mut buf,
                &mut pad_set,
                nets,
                &group_ids,
                &alone_path,
                &windows,
                &best_order,
                n_cells,
                &via_model,
                self.clearance_cells,
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

        // Assemble in input net order for determinism.
        let mut results: Vec<RouteResult> = Vec::new();
        let mut unrouted: Vec<String> = Vec::new();
        for (i, net) in nets.iter().enumerate() {
            match &committed[i] {
                Some(path) => results.push(RouteResult {
                    net: net.net.clone(),
                    path: path.clone(),
                    cost: unit_cost(path),
                }),
                None => unrouted.push(net.net.clone()),
            }
        }

        let congestion = BoardRoute::congestion_from(dims, &results);
        Ok(BoardRoute {
            results,
            unrouted,
            congestion,
        })
    }
}

/// Route one net for the negotiation phase using on-the-fly congestion pricing —
/// no grid clone. The cost to ENTER cell `c` is
/// `SCALE + history[c] + pfac*SCALE*present[c]`, capped strictly below [`OBSTACLE`]
/// (`present` already excludes this net's own occupancy). A cell is blocked iff it
/// is a base obstacle that is NOT one of the net's own pads, or it lies outside
/// `window`. The own endpoints are forced passable. Returns the windowed shortest
/// path and its (priced) cost, or `None`.
#[allow(clippy::too_many_arguments)]
fn route_negotiated(
    buf: &mut SearchBuf,
    base: &Grid,
    pads: &PadSet,
    present: &[u32],
    history: &[u32],
    pfac: u32,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    // Price to ENTER cell `c`: planar base `SCALE`, plus permanent history and the
    // present-congestion penalty, capped below OBSTACLE. Vias reuse this same
    // congestion (history + present of the destination cell) but substitute the
    // planar `SCALE` base with the via's `step_cost` so layer changes also
    // negotiate congestion — see `via_priced` below.
    let priced_with_base = |c: CellIdx, base_cost: u64| -> Cost {
        let ci = c as usize;
        let priced = base_cost
            .saturating_add(history[ci] as u64)
            .saturating_add((pfac as u64) * (SCALE as u64) * (present[ci] as u64));
        priced.min(OBSTACLE as u64 - 1) as Cost
    };
    let cost_fn = |c: CellIdx| -> Cost { priced_with_base(c, SCALE as u64) };
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
    let h = |c: CellIdx| manhattan_scaled(dims, c, dst, via_model.step_cost);
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h, via_step)
}

/// Route one net for legalization / rip-up using on-the-fly costs — no grid clone.
///
/// Clearance is a SOFT cost, COPPER a HARD block:
///   * `owner` — committed COPPER. A cell owned by a group other than `own_group`
///     is a HARD obstacle (two distinct nets may never overlap). Cells owned by
///     `own_group` (siblings) and free cells are passable at their base cost.
///   * `halo`  — foreign clearance / via-keepout halo. A cell with `halo[c]` a
///     foreign group (`>= 0 && != own_group`) AND `owner[c] < 0` (not real copper)
///     is passable but priced [`CLEARANCE_PENALTY`] above its base cost (a recorded
///     clearance violation), so the search prefers a clearance-legal route but will
///     enter the halo as a last resort rather than leave the net unrouted. Own-group
///     halo costs nothing (same-net override). The penalized cost is capped strictly
///     below [`OBSTACLE`].
///
/// The net's own pads are unmasked to [`FREE_COST`], its endpoints are always
/// enterable, and the search is confined to `window`. `owner`/`halo` may each be
/// empty to mean "no owners / no halo" (the alone-path case). Returns the windowed
/// shortest path and its (possibly penalized) cost, or `None`.
#[allow(clippy::too_many_arguments)]
fn route_legal(
    buf: &mut SearchBuf,
    base: &Grid,
    pads: &PadSet,
    owner: &[i64],
    halo: &[i64],
    own_group: i64,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    let has_owner = !owner.is_empty();
    let has_halo = !halo.is_empty();
    // Base passable cost of cell `c` (own pads / obstacle-endpoints unmasked to
    // FREE_COST), before any soft clearance penalty.
    let base_cost = |ci: CellIdx| -> Cost {
        if ci == src || ci == dst {
            if base.is_obstacle(ci) {
                FREE_COST
            } else {
                base.cost_at(ci)
            }
        } else if base.is_obstacle(ci) {
            // Reachable here only when it is one of the net's own pads (else
            // blocked_fn rejected it); unmask to FREE_COST.
            FREE_COST
        } else {
            base.cost_at(ci)
        }
    };
    let cost_fn = |c: CellIdx| -> Cost {
        let base_c = base_cost(c);
        // Soft clearance penalty: entering a foreign group's halo cell that is not
        // real copper. Own-group halo (and own copper) costs nothing extra. Capped
        // strictly below OBSTACLE so a penalized cell is never an impassable one.
        if has_halo {
            let ci = c as usize;
            let h = halo[ci];
            let is_copper = has_owner && owner[ci] >= 0;
            if h >= 0 && h != own_group && !is_copper {
                return (base_c as u64)
                    .saturating_add(CLEARANCE_PENALTY as u64)
                    .min(OBSTACLE as u64 - 1) as Cost;
            }
        }
        base_c
    };
    let blocked_fn = |c: CellIdx| -> bool {
        if !window.contains(dims, c) {
            return true;
        }
        // Foreign-group COPPER cells are hard obstacles, even at this net's
        // endpoints — distinct nets may never overlap. Foreign HALO is NOT blocked
        // here (it is priced softly in `cost_fn` instead).
        if has_owner {
            let o = owner[c as usize];
            if o >= 0 && o != own_group {
                return true;
            }
        }
        if c == src || c == dst {
            return false;
        }
        base.is_obstacle(c) && !pads.contains(c)
    };
    // A via step is legal per the model; it costs the via's `step_cost` (foreign
    // owners / endpoints are already rejected by `blocked_fn` on the destination).
    let via_step = |u: CellIdx, v: CellIdx| -> Option<Cost> {
        if via_model.is_step_legal(dims.layer_of(u), dims.layer_of(v)) {
            Some(via_model.step_cost)
        } else {
            None
        }
    };
    let h = |c: CellIdx| manhattan_scaled(dims, c, dst, via_model.step_cost);
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h, via_step)
}

/// Fold a committed `path` into the ownership maps, separating HARD copper from the
/// SOFT clearance halo so a net is never dropped merely for failing to honour
/// spacing. This is the single place both legalizers commit copper.
///
/// Two parallel maps are written (both indexed by cell, `-1` == free):
///   * `owner` — the committed PATH cells (the actual copper). A foreign group's
///     `owner` cell is a HARD block: two distinct nets must never overlap.
///   * `halo`  — the clearance / via-keepout cells around the copper. A foreign
///     group's `halo` cell is a SOFT cost ([`CLEARANCE_PENALTY`]) in
///     [`route_legal`], never a hard block, so a net may route through it as a last
///     resort (a recorded clearance violation) instead of going unrouted.
///
/// Exact stamping rule, applied for the committed `path` belonging to `group`:
///
/// 1. **Path cells (copper).** `owner[c] = group` for every cell `c` on the path,
///    unconditionally (matches the pre-clearance behaviour — the path always wins
///    its own cells).
/// 2. **Planar clearance halo.** For each path cell, on that cell's OWN layer,
///    visit every cell `n` within Chebyshev radius `clearance_cells` (the
///    `(2r+1)x(2r+1)` planar box). Set `halo[n] = group` ONLY IF `owner[n] == -1`
///    (the cell is not real copper — a halo never overwrites copper ownership) AND
///    `halo[n] == -1` (still free halo) AND `!base.is_obstacle(n)`. Overlap
///    tie-break: the FIRST group to claim a free halo cell keeps it (`halo[n] == -1`
///    guard); a later group's halo never overwrites it. A base obstacle / foreign
///    pad is never claimed (claiming it could deter access to that net's own pad,
///    though here the penalty is only soft).
/// 3. **Via keepout.** A via is detected as two consecutive path cells sharing the
///    same `(x, y)` but differing in layer. At each such `(x, y)`, on *every* layer
///    the via spans, stamp a halo of radius `max(clearance_cells, via_model.keepout)`
///    under the identical rule — a via pad is wider than a track, so it reserves a
///    larger neighbourhood.
///
/// CRITICAL: when `clearance_cells == 0` AND `via_model.keepout == 0` this marks
/// *exactly* the path cells into `owner` and writes NOTHING into `halo` (a radius-0
/// box is skipped; the via halo radius is likewise 0). `halo` stays all `-1`, so
/// [`route_legal`]'s soft cost adds nothing and the default router is byte-identical
/// to the pre-clearance implementation.
#[allow(clippy::too_many_arguments)]
fn stamp_owner(
    owner: &mut [i64],
    halo: &mut [i64],
    base: &Grid,
    dims: mr_core::Dims,
    path: &[CellIdx],
    group: i64,
    clearance_cells: u32,
    via_model: &ViaModel,
) {
    // Stamp a planar Chebyshev halo of radius `r` around `(cx, cy)` on `layer` into
    // the `halo` map, claiming only cells that are not real copper (`owner == -1`),
    // not yet claimed by any group's halo (`halo == -1`, first-claim-wins), and not
    // base obstacles. The centre cell is included, but a path cell already has
    // `owner == group`, so the `owner == -1` guard skips it (its halo entry stays
    // free — own-group halo is irrelevant since own-group cells cost nothing).
    let stamp_halo = |owner: &[i64], halo: &mut [i64], cx: u32, cy: u32, layer: u32, r: u32| {
        if r == 0 {
            return;
        }
        let x0 = cx.saturating_sub(r);
        let y0 = cy.saturating_sub(r);
        let x1 = (cx + r).min(dims.w.saturating_sub(1));
        let y1 = (cy + r).min(dims.h.saturating_sub(1));
        for ny in y0..=y1 {
            for nx in x0..=x1 {
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
        stamp_halo(owner, halo, cx, cy, cl, clearance_cells);
    }

    // 3: via keepout. A via is a consecutive same-(x,y), layer-changing step. At
    // each via (x,y) stamp the larger of the planar clearance and the via keepout
    // on every layer the via spans (both endpoints' layers).
    let via_r = clearance_cells.max(via_model.keepout);
    if via_r > 0 {
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
    path: &[CellIdx],
    group: i64,
    clearance_cells: u32,
    via_model: &ViaModel,
) {
    // Release halo cells this group claimed (only `halo[n] == group`; first-claim
    // tie-break may have awarded an overlapping cell to another group, which we must
    // not clear). Copper path cells are released separately below.
    let clear_halo = |halo: &mut [i64], cx: u32, cy: u32, layer: u32, r: u32| {
        if r == 0 {
            return;
        }
        let x0 = cx.saturating_sub(r);
        let y0 = cy.saturating_sub(r);
        let x1 = (cx + r).min(dims.w.saturating_sub(1));
        let y1 = (cy + r).min(dims.h.saturating_sub(1));
        for ny in y0..=y1 {
            for nx in x0..=x1 {
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
        clear_halo(halo, cx, cy, cl, clearance_cells);
    }

    let via_r = clearance_cells.max(via_model.keepout);
    if via_r > 0 {
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
    buf: &mut SearchBuf,
    pad_set: &mut PadSet,
    nets: &[NetEndpoints],
    group_ids: &[usize],
    paths: &[Vec<CellIdx>],
    windows: &[Window],
    group_order: &[usize],
    n_cells: usize,
    via_model: &ViaModel,
    clearance_cells: u32,
) -> Committed {
    let dims = grid.dims;
    let n_nets = nets.len();
    // Owning group per committed COPPER cell, or -1 for free: a cell is a foreign
    // HARD obstacle for net i iff its owner is a group other than i's.
    let mut owner: Vec<i64> = vec![-1; n_cells];
    // Owning group per committed clearance-HALO cell, or -1 for free: a foreign
    // halo cell is a SOFT cost (CLEARANCE_PENALTY) in `route_legal`, never a hard
    // block, so a net routes through it as a last resort rather than being dropped.
    let mut halo: Vec<i64> = vec![-1; n_cells];
    let mut committed: Committed = vec![None; n_nets];

    // Net indices committed by the group currently being placed; their paths are
    // folded into `owner` (with clearance halos) only after the whole group
    // commits, so sibling sub-nets never block each other. Tracked as a list
    // (cleared per group) to avoid an O(n_cells) sweep.
    let mut group_members: Vec<usize> = Vec::new();

    for &g in group_order {
        let gi = g as i64;
        // Members of this group, in input net order for determinism.
        for i in 0..n_nets {
            if group_ids[i] != g {
                continue;
            }
            let net = &nets[i];

            // Prefer the negotiated path only if it avoids every foreign-group cell —
            // both committed copper (`owner`) AND clearance halo (`halo`). A path that
            // would sit in a foreign halo is rerouted via `route_legal`, which prefers
            // clearance-legal cells (soft penalty) but still routes through the halo as
            // a last resort rather than dropping the net. Endpoints are not exempt:
            // distinct groups may never share copper.
            let cur = &paths[i];
            let clean = !cur.is_empty()
                && cur.iter().all(|&c| {
                    let o = owner[c as usize];
                    let h = halo[c as usize];
                    (o < 0 || o == gi) && (h < 0 || h == gi)
                });

            let chosen = if clean {
                Some(cur.clone())
            } else {
                pad_set.load(&net.passable_pads);
                route_legal(
                    buf, grid, pad_set, &owner, &halo, gi, net.src, net.dst, windows[i],
                    via_model,
                )
                .or_else(|| {
                    route_legal(
                        buf,
                        grid,
                        pad_set,
                        &owner,
                        &halo,
                        gi,
                        net.src,
                        net.dst,
                        Window::full(dims),
                        via_model,
                    )
                })
                .map(|(p, _)| p)
            };

            if let Some(path) = chosen {
                committed[i] = Some(path);
                group_members.push(i);
            }
        }

        // Fold this group's committed paths into the owner map (path cells + the
        // clearance/via-keepout halo) only now that the whole group is placed, so
        // a sibling's halo never blocked another sibling's route. Same-group halos
        // are idempotent (they re-own to `gi`). Reset the per-group scratch.
        for &i in &group_members {
            if let Some(path) = &committed[i] {
                stamp_owner(
                    &mut owner,
                    &mut halo,
                    grid,
                    dims,
                    path,
                    gi,
                    clearance_cells,
                    via_model,
                );
            }
        }
        group_members.clear();
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
    buf: &mut SearchBuf,
    pad_set: &mut PadSet,
    nets: &[NetEndpoints],
    group_ids: &[usize],
    alone_path: &[Vec<CellIdx>],
    windows: &[Window],
    seed_group_order: &[usize],
    n_cells: usize,
    via_model: &ViaModel,
    clearance_cells: u32,
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

    let mut committed: Committed = vec![None; n_nets];
    // Owning group per committed COPPER cell, or -1 for free: a foreign-group copper
    // cell is a HARD obstacle for net i. Owning group per clearance-HALO cell, or -1:
    // a foreign-group halo cell is a SOFT cost (CLEARANCE_PENALTY) in `route_legal`,
    // never a hard block, so a stranded net routes through the halo as a last resort.
    let mut owner: Vec<i64> = vec![-1; n_cells];
    let mut halo: Vec<i64> = vec![-1; n_cells];

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
                        &path,
                        g as i64,
                        clearance_cells,
                        via_model,
                    );
                }
            }
        }
    };

    while let Some(i) = queue.pop_front() {
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
            buf, grid, pad_set, &owner, &halo, gi, net.src, net.dst, windows[i], via_model,
        )
        .or_else(|| {
            if needs_full[i] {
                route_legal(
                    buf,
                    grid,
                    pad_set,
                    &owner,
                    &halo,
                    gi,
                    net.src,
                    net.dst,
                    Window::full(dims),
                    via_model,
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
                &path,
                gi,
                clearance_cells,
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
            buf, grid, pad_set, &owner, &halo, gi, net.src, net.dst, windows[i], via_model,
        )
        .or_else(|| {
            if needs_full[i] {
                route_legal(
                    buf,
                    grid,
                    pad_set,
                    &owner,
                    &halo,
                    gi,
                    net.src,
                    net.dst,
                    Window::full(dims),
                    via_model,
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
                &path,
                gi,
                clearance_cells,
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
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 1), dims.idx(2, 1)),
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
    /// Re-uses the over-constrained single corridor (two same-endpoint nets in
    /// distinct groups): rip-up may ping-pong the corridor but is bounded by its
    /// global/per-net budgets, so it stops and returns the best partial (one net),
    /// never regressing below the multi-order result.
    #[test]
    fn ripup_terminates_when_unsolvable() {
        let dims = Dims::new(3, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_cell(1, 0);
        b.mark_cell(1, 2);
        let grid = b.build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 1), dims.idx(2, 1)),
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

        // Order [0,1] = A first: B should be stranded.
        let c_ab = legalize_in_order(
            &grid,
            &mut buf,
            &mut pad_set,
            &nets_ab,
            &group_ids,
            &crafted,
            &windows,
            &[0, 1],
            n_cells,
            &via_model,
            0,
        );
        assert!(c_ab[0].is_some(), "A commits in A-first order");
        assert!(
            c_ab[1].is_none(),
            "B must be stranded when A claims the middle row first"
        );

        // Order [1,0] = B first: both route.
        let c_ba = legalize_in_order(
            &grid,
            &mut buf,
            &mut pad_set,
            &nets_ab,
            &group_ids,
            &crafted,
            &windows,
            &[1, 0],
            n_cells,
            &via_model,
            0,
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

    /// SOFT clearance recovers connectivity: with a HARD halo net B would be DROPPED
    /// (no clearance-legal path to its target), but with the soft penalty B now
    /// ROUTES — through A's clearance halo, at a recorded violation. B's copper still
    /// shares NO cell with A's copper (distinct nets never overlap), yet at least one
    /// B cell lies within `clearance_cells` of an A cell (it entered the halo).
    #[test]
    fn soft_clearance_recovers_dropped_net() {
        // 3 wide, 5 tall. Net A runs straight down column 1, rows 0..=3 — a vertical
        // wall of copper that leaves only the single cell (1,4) free in column 1. Net
        // B must cross from (0,2) to (2,2): the ONLY copper-free crossing of column 1
        // is through (1,4), which is Chebyshev-adjacent to A's copper at (1,3) — i.e.
        // squarely inside A's clearance halo. With a HARD halo (radius 1) that cell
        // (and its approaches) would be blocked, walling B off from its target and
        // DROPPING it. With the SOFT penalty B routes through the halo instead (a
        // recorded violation), so connectivity survives.
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

        // Both nets route — B is NOT dropped despite needing to enter A's halo.
        assert!(
            br.unrouted.is_empty(),
            "soft clearance must recover connectivity for B: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let ra = br.results.iter().find(|r| r.net == "a").expect("A routes");
        let rb = br.results.iter().find(|r| r.net == "b").expect("B routes");

        // B's copper is still hard-disjoint from A's copper.
        assert!(
            disjoint(&ra.path, &rb.path),
            "distinct nets must never overlap copper: {ra:?} {rb:?}"
        );

        // B did enter A's clearance halo: some B cell is within Chebyshev radius
        // `clearance_cells` (1) of an A cell — i.e. the recovered route is a
        // clearance violation, exactly what the soft cost permits as a last resort.
        assert!(
            within_one(dims, &rb.path, &ra.path),
            "B must route THROUGH A's clearance halo (a recorded violation): {rb:?}"
        );
    }

    /// Via keepout is now a SOFT cost too: a placed via's neighbourhood is PREFERRED
    /// clear, not forced. On a 2-layer board whose only layer-0 corridor is walled,
    /// net A must via up at a chokepoint to cross. With `via_model.keepout = 1` and
    /// ample room for B to via up at a DIFFERENT row, the router routes BOTH nets and
    /// prefers to keep B's committed copper out of A's reserved via neighbourhood —
    /// because a clearance-legal route is available at modest extra cost. We assert
    /// both route and B's copper shares no cell of A's via 8-neighbourhood.
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
            m.keepout = 1;
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
        assert!(!via_neigh.is_empty(), "A must place at least one via: {ra:?}");

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
}
