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

use crate::dijkstra::{dijkstra, reconstruct_path};

/// Fixed-point cost scale: the base cost of stepping onto a passable cell.
pub const SCALE: Cost = 16;

/// Maximum negotiation iterations before falling through to legalization.
pub const MAX_ITERS: u32 = 40;

/// Free-cell cost used when unmasking a net's own pad cells (mirrors the base grid
/// convention used by `GridBuilder`, which fills passable cells with cost 1).
const FREE_COST: Cost = 1;

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

        // Per-cell membership of each net's own passable pads, for fast lookup.
        let pad_sets: Vec<std::collections::HashSet<CellIdx>> = nets
            .iter()
            .map(|net| net.passable_pads.iter().copied().collect())
            .collect();

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

        // Reusable cost buffer for the per-net effective grid.
        let mut work = grid.clone();

        for iter in 0..MAX_ITERS {
            let pfac: u32 = 1 + iter;

            for i in 0..n_nets {
                let net = &nets[i];

                // Remove this net's old path from `present` before pricing.
                for &c in &paths[i] {
                    present[c as usize] = present[c as usize].saturating_sub(1);
                }
                paths[i].clear();

                build_effective_grid(&mut work, grid, &pad_sets[i], &present, &history, pfac);

                let h = |c: CellIdx| manhattan_scaled(dims, c, net.dst);
                let field = dijkstra(&work, net.src, h);
                if let Some(path) = reconstruct_path(&field.pred, net.src, net.dst, &field.dist) {
                    for &c in &path {
                        present[c as usize] = present[c as usize].saturating_add(1);
                    }
                    paths[i] = path;
                }
                // else: leave unrouted this iteration (no contribution to present).
            }

            // Overuse across GROUPS: a cell is over-used iff ≥2 distinct groups
            // occupy it. Track the first group seen per cell; a second distinct
            // group flags overuse.
            let mut first_group: Vec<i64> = vec![-1; n_cells];
            let mut overused: Vec<bool> = vec![false; n_cells];
            let mut any_overuse = false;
            for i in 0..n_nets {
                let g = group_ids[i] as i64;
                for &c in &paths[i] {
                    let slot = &mut first_group[c as usize];
                    if *slot < 0 {
                        *slot = g;
                    } else if *slot != g && !overused[c as usize] {
                        overused[c as usize] = true;
                        any_overuse = true;
                    }
                }
            }

            if !any_overuse {
                break; // converged: cell-disjoint across groups
            }

            for (c, &over) in overused.iter().enumerate() {
                if over {
                    history[c] = history[c].saturating_add(SCALE);
                }
            }
        }

        // ---- Final legalization: commit group-by-group, cell-disjoint across
        // groups. Cells used within a group never block its own members. ----
        let mut occupied: Vec<bool> = vec![false; n_cells];
        let mut committed: Vec<Option<Vec<CellIdx>>> = vec![None; n_nets];

        // Deterministic commit order: by (group id, net index). Group ids are
        // assigned in first-appearance order, so this is stable input order.
        let mut order: Vec<usize> = (0..n_nets).collect();
        order.sort_by_key(|&i| (group_ids[i], i));

        // Track which cells the current group itself owns, so sibling sub-nets of
        // the same connection may overlap freely without being blocked.
        let mut group_cells: Vec<bool> = vec![false; n_cells];
        let mut cur_group: Option<usize> = None;

        for &i in &order {
            let g = group_ids[i];
            if cur_group != Some(g) {
                // Starting a new group: fold the previous group's cells into the
                // global `occupied` set and reset the per-group scratch.
                if cur_group.is_some() {
                    for c in 0..n_cells {
                        if group_cells[c] {
                            occupied[c] = true;
                            group_cells[c] = false;
                        }
                    }
                }
                cur_group = Some(g);
            }

            let net = &nets[i];

            // Does the negotiated path avoid every foreign-group cell? Endpoints
            // are not exempt: two distinct groups may never share any cell,
            // including pads (each net's pads are its own distinct cells).
            let cur = &paths[i];
            let clean = !cur.is_empty() && cur.iter().all(|&c| !occupied[c as usize]);

            let chosen = if clean {
                Some(cur.clone())
            } else {
                // Reroute once on a grid where foreign-group cells are hard
                // obstacles; own pads are unmasked.
                build_legal_grid(&mut work, grid, &pad_sets[i], &occupied, net.src, net.dst);
                let h = |c: CellIdx| manhattan_scaled(dims, c, net.dst);
                let field = dijkstra(&work, net.src, h);
                reconstruct_path(&field.pred, net.src, net.dst, &field.dist)
            };

            if let Some(path) = chosen {
                for &c in &path {
                    group_cells[c as usize] = true;
                }
                committed[i] = Some(path);
            }
        }

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

/// Overwrite `work`'s cost grid in place with net `i`'s congestion-priced view of
/// `base`. `present` already excludes net `i`'s own occupancy (its path was
/// decremented before this call), so `present[c]` is `occ_excl_i(c)` directly.
fn build_effective_grid(
    work: &mut Grid,
    base: &Grid,
    pads: &std::collections::HashSet<CellIdx>,
    present: &[u32],
    history: &[u32],
    pfac: u32,
) {
    for c in 0..base.dims.len() {
        let ci = c as CellIdx;
        let cost = if base.is_obstacle(ci) && !pads.contains(&ci) {
            OBSTACLE
        } else {
            let occ = present[c];
            let priced = (SCALE as u64)
                .saturating_add(history[c] as u64)
                .saturating_add((pfac as u64) * (SCALE as u64) * (occ as u64));
            // Cap strictly below OBSTACLE so a priced cell is still passable.
            priced.min(OBSTACLE as u64 - 1) as Cost
        };
        work.cost[c] = cost;
    }
}

/// Overwrite `work` for the legalization reroute of one net: foreign-group cells
/// (`occupied`) become hard obstacles, the net's own pads are unmasked to
/// [`FREE_COST`], and the net's own endpoints are always kept passable.
fn build_legal_grid(
    work: &mut Grid,
    base: &Grid,
    pads: &std::collections::HashSet<CellIdx>,
    occupied: &[bool],
    src: CellIdx,
    dst: CellIdx,
) {
    for c in 0..base.dims.len() {
        let ci = c as CellIdx;
        let cost = if occupied[c] {
            // Foreign-group cells are hard obstacles — even if they are this net's
            // declared endpoints, since distinct groups may not share any cell.
            OBSTACLE
        } else if ci == src || ci == dst {
            // Own endpoints must be enterable even when they are (own) pad
            // obstacles in the base grid.
            if base.is_obstacle(ci) {
                FREE_COST
            } else {
                base.cost_at(ci)
            }
        } else if base.is_obstacle(ci) {
            if pads.contains(&ci) {
                FREE_COST
            } else {
                OBSTACLE
            }
        } else {
            base.cost_at(ci)
        };
        work.cost[c] = cost;
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
}
