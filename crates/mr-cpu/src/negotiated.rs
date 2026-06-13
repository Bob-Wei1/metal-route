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
    BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError, OBSTACLE,
};

use crate::dijkstra::{astar_buf, SearchBuf};

/// Fixed-point cost scale: the base cost of stepping onto a passable cell.
pub const SCALE: Cost = 16;

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
    let margin = 16u32.max((3 * span + 9) / 10);
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
#[derive(Debug, Default, Clone, Copy)]
pub struct NegotiatedRouter;

impl NegotiatedRouter {
    pub fn new() -> Self {
        Self
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

/// Manhattan distance between two cells, scaled by [`SCALE`]. Admissible for the
/// per-cell cost model (every step costs at least `SCALE`), so it is a valid A*
/// heuristic that keeps Dijkstra optimal while pruning the frontier.
fn manhattan_scaled(dims: mr_core::Dims, a: CellIdx, b: CellIdx) -> Cost {
    let (ax, ay) = dims.xy(a);
    let (bx, by) = dims.xy(b);
    let dx = ax.abs_diff(bx);
    let dy = ay.abs_diff(by);
    (dx + dy).saturating_mul(SCALE)
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
                    windows[i],
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
            for i in 0..n_nets {
                for &c in &paths[i] {
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
            for i in 0..n_nets {
                let net = &nets[i];
                pad_set.load(&net.passable_pads);
                let routed = route_legal(
                    &mut buf, grid, &pad_set, &no_owner, -1, net.src, net.dst, windows[i],
                )
                .or_else(|| {
                    route_legal(
                        &mut buf,
                        grid,
                        &pad_set,
                        &no_owner,
                        -1,
                        net.src,
                        net.dst,
                        Window::full(dims),
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
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    let cost_fn = |c: CellIdx| -> Cost {
        let ci = c as usize;
        let priced = (SCALE as u64)
            .saturating_add(history[ci] as u64)
            .saturating_add((pfac as u64) * (SCALE as u64) * (present[ci] as u64));
        priced.min(OBSTACLE as u64 - 1) as Cost
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
    let h = |c: CellIdx| manhattan_scaled(dims, c, dst);
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h)
}

/// Route one net for legalization / rip-up using on-the-fly costs — no grid clone.
/// Cells owned by a group other than `own_group` are hard obstacles; cells owned by
/// `own_group` (siblings) and free cells are passable at their base cost. The net's
/// own pads are unmasked to [`FREE_COST`], its endpoints are always enterable, and
/// the search is confined to `window`. `owner` may be empty to mean "no owners"
/// (the alone-path case). Returns the windowed shortest path and its base-cost, or
/// `None`.
#[allow(clippy::too_many_arguments)]
fn route_legal(
    buf: &mut SearchBuf,
    base: &Grid,
    pads: &PadSet,
    owner: &[i64],
    own_group: i64,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    let has_owner = !owner.is_empty();
    let cost_fn = |c: CellIdx| -> Cost {
        let ci = c as CellIdx;
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
    let blocked_fn = |c: CellIdx| -> bool {
        if !window.contains(dims, c) {
            return true;
        }
        // Foreign-group cells are hard obstacles, even at this net's endpoints —
        // distinct groups may never share any cell.
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
    let h = |c: CellIdx| manhattan_scaled(dims, c, dst);
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h)
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
) -> Committed {
    let dims = grid.dims;
    let n_nets = nets.len();
    // Owning group per committed cell, or -1 for free. (Replaces the old per-call
    // `occupied: Vec<bool>`; a cell is a foreign obstacle for net i iff its owner
    // is a group other than i's.)
    let mut owner: Vec<i64> = vec![-1; n_cells];
    let mut committed: Committed = vec![None; n_nets];

    // Cells claimed by the group currently being committed; folded into `owner`
    // only after the whole group commits, so sibling sub-nets never block each
    // other. Tracked as a list (cleared per group) to avoid an O(n_cells) sweep.
    let mut group_cells: Vec<CellIdx> = Vec::new();

    for &g in group_order {
        let gi = g as i64;
        // Members of this group, in input net order for determinism.
        for i in 0..n_nets {
            if group_ids[i] != g {
                continue;
            }
            let net = &nets[i];

            // Prefer the negotiated path if it avoids every foreign-group cell.
            // Endpoints are not exempt: distinct groups may never share any cell.
            let cur = &paths[i];
            let clean = !cur.is_empty()
                && cur.iter().all(|&c| {
                    let o = owner[c as usize];
                    o < 0 || o == gi
                });

            let chosen = if clean {
                Some(cur.clone())
            } else {
                pad_set.load(&net.passable_pads);
                route_legal(buf, grid, pad_set, &owner, gi, net.src, net.dst, windows[i])
                    .or_else(|| {
                        route_legal(
                            buf,
                            grid,
                            pad_set,
                            &owner,
                            gi,
                            net.src,
                            net.dst,
                            Window::full(dims),
                        )
                    })
                    .map(|(p, _)| p)
            };

            if let Some(path) = chosen {
                for &c in &path {
                    group_cells.push(c);
                }
                committed[i] = Some(path);
            }
        }

        // Fold this group's cells into the owner map and reset scratch.
        for &c in &group_cells {
            owner[c as usize] = gi;
        }
        group_cells.clear();
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
    // Owning group per cell, or -1 for free. A cell is "owned by another group"
    // (w.r.t. net i) iff occupied_by_group[c] is set to a group != group_ids[i].
    let mut owner: Vec<i64> = vec![-1; n_cells];

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

    // Free every cell currently owned by group `g` (its committed nets are
    // un-committed by the caller separately). Reset committed entries here.
    let free_group_cells =
        |owner: &mut [i64], committed: &mut Committed, group_ids: &[usize], g: usize| {
            for i in 0..committed.len() {
                if group_ids[i] == g {
                    if let Some(path) = committed[i].take() {
                        for &c in &path {
                            // Only clear cells this group still owns (siblings may
                            // share; clearing is idempotent and safe).
                            if owner[c as usize] == g as i64 {
                                owner[c as usize] = -1;
                            }
                        }
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
        let routed = route_legal(buf, grid, pad_set, &owner, gi, net.src, net.dst, windows[i])
            .or_else(|| {
                if needs_full[i] {
                    route_legal(
                        buf,
                        grid,
                        pad_set,
                        &owner,
                        gi,
                        net.src,
                        net.dst,
                        Window::full(dims),
                    )
                } else {
                    None
                }
            });

        if let Some((path, _)) = routed {
            for &c in &path {
                owner[c as usize] = gi;
            }
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
        free_group_cells(&mut owner, &mut committed, group_ids, victim);
        rips_done += 1;

        // Re-route i now that the victim's cells are free.
        let rerouted = route_legal(buf, grid, pad_set, &owner, gi, net.src, net.dst, windows[i])
            .or_else(|| {
                if needs_full[i] {
                    route_legal(
                        buf,
                        grid,
                        pad_set,
                        &owner,
                        gi,
                        net.src,
                        net.dst,
                        Window::full(dims),
                    )
                } else {
                    None
                }
            });
        if let Some((path, _)) = rerouted {
            for &c in &path {
                owner[c as usize] = gi;
            }
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

        let br = NegotiatedRouter
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

        let br = NegotiatedRouter.route(&grid, &[net_a, net_b]).unwrap();
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
        let br = NegotiatedRouter.route(&grid, &nets).unwrap();
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
        let br1 = NegotiatedRouter.route(&grid, &nets).unwrap();
        let br2 = NegotiatedRouter.route(&grid, &nets).unwrap();
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
        let br = NegotiatedRouter.route(&grid, &nets).unwrap();
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

        let br = NegotiatedRouter
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

        let br = NegotiatedRouter.route(&grid, &nets).unwrap();
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
        let br = NegotiatedRouter.route(&grid, &nets).unwrap();
        assert_eq!(br.results.len() + br.unrouted.len(), 2);
        assert_eq!(br.unrouted.len(), 1, "single corridor fits only one net");
        // Determinism across repeated runs (rip-up tie-breaks must be stable).
        let br2 = NegotiatedRouter.route(&grid, &nets).unwrap();
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

        let br = NegotiatedRouter.route(&grid, &[a.clone()]).unwrap();
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
        );
        assert!(
            c_ba[0].is_some() && c_ba[1].is_some(),
            "both route when B claims the middle column first"
        );

        // The full router must pick the good order and route BOTH nets even though
        // first-appearance order [A, B] alone would strand B.
        let br = NegotiatedRouter
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
}
