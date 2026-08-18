//! `RipUpRouter` (A3 / M2) — sequential routing with bounded rip-up-on-collision.
//!
//! Nets are provisionally placed from lowest to highest priority (reverse input
//! order), then resolved one at a time. Each net is routed with
//! [`LeeRouter`] over a working grid in which the cells already occupied by
//! *other* committed nets are treated as obstacles. When a net cannot be routed
//! because the corridor it needs is occupied, it rips up the conflicting
//! committed net(s) and is retried; the ripped-up nets are re-queued.
//!
//! ## Priority / anti-oscillation
//!
//! Earlier nets (lower index) have priority: placing them after provisional
//! lower-priority routes gives them something real to displace. A net may only rip
//! up committed nets of **higher** index than itself. A lower-index net is never displaced for a
//! higher-index one. This makes the outcome deterministic and prevents two nets
//! from endlessly ripping each other up. A net blocked only by un-displaceable
//! (lower-index) committed nets is left unrouted.
//!
//! The whole process is bounded to `K == 20` passes ([`MAX_PASSES`]). On
//! exhaustion the still-unrouted nets are reported in [`BoardRoute::unrouted`] and
//! the router returns normally — it NEVER loops unbounded.

use std::collections::HashMap;

use mr_core::{
    BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError, OBSTACLE,
};

use crate::lee::LeeRouter;

/// Maximum number of routing passes before giving up (M2 bound K).
pub const MAX_PASSES: u32 = 20;

/// Sequential net router with bounded rip-up-on-collision.
#[derive(Debug, Default, Clone, Copy)]
pub struct RipUpRouter;

impl RipUpRouter {
    pub fn new() -> Self {
        Self
    }
}

/// A net's currently-committed route (path + cost).
struct Placed {
    path: Vec<CellIdx>,
    cost: Cost,
}

/// Free-cell cost used when unmasking a net's own pad cells. The base grid uses
/// cost 1 for passable cells (see `GridBuilder`); there is no direct accessor, so
/// we use the same constant the rest of the pipeline treats as a free cell.
const FREE_COST: Cost = 1;

/// The base grid with `net`'s own pad cells (`passable_pads`) unmasked back to
/// free cells. This is the grid on which the net would route if it were the only
/// net on the board: all foreign pads stay obstacles, only this net's pads open.
fn own_grid(grid: &Grid, net: &NetEndpoints) -> Grid {
    let mut g = grid.clone();
    for &c in &net.passable_pads {
        g.set(c, FREE_COST);
    }
    g
}

/// Build a working grid where every committed net other than `ni` occupies
/// obstacle cells. The current net's own pad cells (`passable_pads`) are unmasked
/// FIRST so the net can escape its own pads; committed nets' paths are then marked
/// as obstacles, so a committed path still wins on any (abnormal) overlap.
fn working_grid(
    grid: &Grid,
    ni: usize,
    net: &NetEndpoints,
    placed: &HashMap<usize, Placed>,
) -> Grid {
    let mut work = own_grid(grid, net);
    for (&oi, p) in placed {
        if oi == ni {
            continue;
        }
        for &c in &p.path {
            if c == net.src || c == net.dst {
                continue;
            }
            work.set(c, OBSTACLE);
        }
    }
    work
}

impl RipUpRouter {
    /// Implementation plus a displacement count used by the adversarial tests to
    /// prove the router's namesake rip-up path is live.
    fn route_with_stats(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
    ) -> Result<(BoardRoute, usize), RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        for net in nets {
            if net.passable_pads.iter().any(|&c| !grid.dims.contains(c)) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            // An endpoint is invalid only if out of bounds, or it sits on an
            // obstacle that is NOT one of this net's own (passable) pad cells.
            // Sitting on one's own pad obstacle is valid (the router unmasks it).
            let endpoint_invalid = |c: CellIdx| {
                !grid.dims.contains(c) || (grid.is_obstacle(c) && !net.passable_pads.contains(&c))
            };
            if endpoint_invalid(net.src) || endpoint_invalid(net.dst) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
        }

        let mut placed: HashMap<usize, Placed> = HashMap::new();
        // Nets we have permanently given up on (blocked by un-displaceable nets or
        // unroutable even on the empty grid). Never retried.
        let mut abandoned: Vec<bool> = vec![false; nets.len()];
        let mut rip_count = 0usize;

        let mut passes = 0u32;
        loop {
            if passes >= MAX_PASSES {
                break;
            }
            passes += 1;

            // Route low-priority nets first (descending index), then let an earlier
            // / higher-priority net displace them when necessary.  Processing in
            // ascending order made the first legal rip impossible: no higher-index
            // route could ever exist when a lower index was examined.
            let pending: Vec<usize> = (0..nets.len())
                .rev()
                .filter(|i| !placed.contains_key(i) && !abandoned[*i])
                .collect();
            if pending.is_empty() {
                break;
            }

            let mut progressed = false;

            for &ni in &pending {
                if placed.contains_key(&ni) {
                    continue; // routed earlier this pass via a re-attempt
                }
                let net = &nets[ni];
                let work = working_grid(grid, ni, net, &placed);

                if let Some((path, cost)) = LeeRouter::route_one(&work, net.src, net.dst) {
                    placed.insert(ni, Placed { path, cost });
                    progressed = true;
                    continue;
                }

                // Blocked. Can we route at all on the empty grid? The base grid
                // now has this net's pads as obstacles, so check on a per-net grid
                // with only THIS net's own pads unmasked (no committed nets).
                let free_grid = own_grid(grid, net);
                let Some((free_path, _)) = LeeRouter::route_one(&free_grid, net.src, net.dst)
                else {
                    // Unroutable regardless of other nets: give up permanently.
                    abandoned[ni] = true;
                    continue;
                };

                // Find committed nets of HIGHER index that overlap our free path —
                // these we may rip up (priority rule).
                let mut blockers: Vec<usize> = placed
                    .iter()
                    .filter(|(&oi, _)| oi > ni)
                    .filter(|(_, p)| p.path.iter().any(|c| free_path.contains(c)))
                    .map(|(&oi, _)| oi)
                    .collect();
                blockers.sort_unstable();

                if blockers.is_empty() {
                    // Only lower-index (un-displaceable) nets block us, or no
                    // committed net overlaps our shortest path yet a path exists on
                    // the working grid was not found — give up permanently to keep
                    // the process bounded and deterministic.
                    abandoned[ni] = true;
                    continue;
                }

                for b in &blockers {
                    placed.remove(b);
                }
                rip_count += blockers.len();
                progressed = true;

                // Commit the higher-priority net immediately in the space it just
                // freed.  Deferring it to the next descending pass allowed each
                // victim to reclaim the corridor first and caused a bounded but
                // pointless ping-pong.
                let retry = working_grid(grid, ni, net, &placed);
                if let Some((path, cost)) = LeeRouter::route_one(&retry, net.src, net.dst) {
                    placed.insert(ni, Placed { path, cost });
                } else {
                    // A lower-priority committed route still blocks it; that route
                    // is intentionally not displaceable under the priority rule.
                    abandoned[ni] = true;
                }
            }

            if !progressed {
                break;
            }
        }

        // Assemble in input net order for determinism.
        let mut results: Vec<RouteResult> = Vec::new();
        let mut unrouted: Vec<String> = Vec::new();
        for (ni, net) in nets.iter().enumerate() {
            match placed.get(&ni) {
                Some(p) => results.push(RouteResult {
                    net: net.net.clone(),
                    path: p.path.clone(),
                    cost: p.cost,
                }),
                None => unrouted.push(net.net.clone()),
            }
        }

        let congestion = BoardRoute::congestion_from(grid.dims, &results);
        Ok((
            BoardRoute {
                results,
                unrouted,
                congestion,
                groups: Vec::new(),
            },
            rip_count,
        ))
    }
}

impl Router for RipUpRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        self.route_with_stats(grid, nets).map(|(board, _)| board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::{Dims, NetEndpoints};
    use mr_grid::GridBuilder;

    fn net(name: &str, src: CellIdx, dst: CellIdx) -> NetEndpoints {
        NetEndpoints {
            net: name.into(),
            src,
            dst,
            passable_pads: Vec::new(),
        }
    }

    #[test]
    fn two_nonconflicting_nets_both_route() {
        // 5x2 open grid; two horizontal nets on separate rows — no conflict.
        let dims = Dims::new(5, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 0), dims.idx(4, 0)),
            net("b", dims.idx(0, 1), dims.idx(4, 1)),
        ];
        let br = RipUpRouter.route(&grid, &nets).unwrap();
        assert!(br.unrouted.is_empty(), "both nets should route");
        assert_eq!(br.results.len(), 2);
        assert!(
            br.congestion.iter().all(|&c| c <= 1),
            "non-conflicting nets must not overlap"
        );
    }

    #[test]
    fn two_nets_compete_for_single_corridor() {
        // A single 1-wide horizontal corridor (row 1) walled top and bottom. Both
        // nets must traverse the same interior corridor cells, so only one fits.
        //   # # # # #     row 0  (wall)
        //   . . . . .     row 1  (the only corridor)
        //   # # # # #     row 2  (wall)
        let dims = Dims::new(5, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_rect(0, 0, 4, 0);
        b.mark_rect(0, 2, 4, 2);
        let grid = b.build();
        // Net a: full corridor left->right. Net b: shares interior cells (2,1)/(3,1)
        // so it cannot coexist with a.
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(4, 1)),
            net("b", dims.idx(2, 1), dims.idx(4, 1)),
        ];

        let br = RipUpRouter.route(&grid, &nets).unwrap();

        assert!(
            !br.results.is_empty(),
            "at least one net must route through the corridor"
        );
        assert_eq!(
            br.results.len() + br.unrouted.len(),
            2,
            "every net accounted for"
        );
        assert!(
            !br.unrouted.is_empty(),
            "both cannot fit the single corridor"
        );
        // Priority rule: lower-index net 'a' wins, 'b' loses.
        assert_eq!(br.results[0].net, "a");
        assert_eq!(br.unrouted, vec!["b".to_string()]);

        // Determinism: identical outcome on a second run.
        let br2 = RipUpRouter.route(&grid, &nets).unwrap();
        assert_eq!(br.unrouted, br2.unrouted);
        assert_eq!(
            br.results.iter().map(|r| &r.net).collect::<Vec<_>>(),
            br2.results.iter().map(|r| &r.net).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn contention_executes_real_displacement_and_preserves_priority() {
        let dims = Dims::new(5, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_rect(0, 0, 4, 0);
        b.mark_rect(0, 2, 4, 2);
        let grid = b.build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(4, 1)),
            net("b", dims.idx(2, 1), dims.idx(4, 1)),
        ];
        let (board, rips) = RipUpRouter.route_with_stats(&grid, &nets).unwrap();
        assert!(rips >= 1, "the higher-priority net must actually rip b up");
        assert_eq!(
            board
                .results
                .iter()
                .map(|r| r.net.as_str())
                .collect::<Vec<_>>(),
            vec!["a"]
        );
        assert_eq!(board.unrouted, vec!["b"]);
    }

    #[test]
    fn repeated_contention_is_byte_deterministic() {
        let dims = Dims::new(7, 5);
        let grid = GridBuilder::new(dims, 1).mark_cell(3, 2).build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(6, 3)),
            net("b", dims.idx(0, 3), dims.idx(6, 1)),
            net("c", dims.idx(1, 0), dims.idx(5, 4)),
        ];
        let first = RipUpRouter.route(&grid, &nets).unwrap();
        for _ in 0..32 {
            assert_eq!(RipUpRouter.route(&grid, &nets).unwrap(), first);
        }
    }

    #[test]
    fn pass_bound_is_k_and_router_terminates() {
        // Unsatisfiable competition: both nets want the same single passage. The
        // router must terminate (no hang), account for all nets, and the documented
        // bound is K == 20.
        let dims = Dims::new(3, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_cell(1, 0);
        b.mark_cell(1, 2);
        let grid = b.build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 1), dims.idx(2, 1)),
        ];
        let br = RipUpRouter.route(&grid, &nets).unwrap();
        assert_eq!(br.results.len() + br.unrouted.len(), 2);
        assert_eq!(MAX_PASSES, 20);
    }

    /// Per-net pad masking: net A's pad lies directly on net B's shortest straight
    /// path. B must route AROUND A's pad (B's path contains none of A's pad cells),
    /// and A must still route (escaping its own pad). Both routed, paths disjoint.
    #[test]
    fn net_routes_around_foreign_pad() {
        // 7x3 open grid. A's pad is a 2x1 vertical block on column 3, rows 0..=1
        // (cells (3,0),(3,1)) — an obstacle in the base grid. It sits on B's
        // straight path across row 1, but row 2 stays open so B can detour. A's
        // endpoints are inside its own pad so A can escape it.
        let dims = Dims::new(7, 3);
        let grid = GridBuilder::new(dims, 1)
            .mark_rect(3, 0, 3, 1) // A's pad: cells (3,0),(3,1) (obstacle in base grid)
            .build();

        // A owns those two pad cells; route A vertically within them.
        let a_pad: Vec<CellIdx> = vec![dims.idx(3, 0), dims.idx(3, 1)];
        let net_a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(3, 0),
            dst: dims.idx(3, 1),
            passable_pads: a_pad.clone(),
        };
        // B runs across row 1; its straight path would cross A's pad cell (3,1),
        // but B does not own that pad so it must dip down to row 2 and around.
        let net_b = NetEndpoints {
            net: "b".into(),
            src: dims.idx(0, 1),
            dst: dims.idx(6, 1),
            passable_pads: Vec::new(),
        };
        let nets = vec![net_a, net_b];

        let br = RipUpRouter.route(&grid, &nets).unwrap();
        assert!(br.unrouted.is_empty(), "both nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);

        let a_path = &br.results[0].path;
        let b_path = &br.results[1].path;

        // B's path must contain NONE of A's pad cells.
        for c in b_path {
            assert!(
                !a_pad.contains(c),
                "B must route around A's pad; cell {c} is A's pad"
            );
        }
        // A must escape its own pad (it routes within the unmasked pad cells).
        assert!(a_path.contains(&dims.idx(3, 0)));
        assert!(a_path.contains(&dims.idx(3, 1)));

        // The two paths are cell-disjoint.
        for c in b_path {
            assert!(
                !a_path.contains(c),
                "paths must be cell-disjoint; shared cell {c}"
            );
        }
    }
}
