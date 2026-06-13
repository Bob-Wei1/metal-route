//! `RipUpRouter` (A3 / M2) — sequential routing with bounded rip-up-on-collision.
//!
//! Nets are routed one at a time (in the order given). Each net is routed with
//! [`LeeRouter`] over a working grid in which the cells already occupied by
//! *other* committed nets are treated as obstacles. When a net cannot be routed
//! because the corridor it needs is occupied, it rips up the conflicting
//! committed net(s) and is retried; the ripped-up nets are re-queued.
//!
//! ## Priority / anti-oscillation
//!
//! Earlier nets (lower index) have priority: a net may only rip up committed nets
//! of **higher** index than itself. A lower-index net is never displaced for a
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

/// Build a working grid where every committed net other than `ni` occupies
/// obstacle cells (except the current net's own endpoints, which stay passable).
fn working_grid(
    grid: &Grid,
    ni: usize,
    net: &NetEndpoints,
    placed: &HashMap<usize, Placed>,
) -> Grid {
    let mut work = grid.clone();
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

impl Router for RipUpRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        for net in nets {
            if !grid.dims.contains(net.src)
                || !grid.dims.contains(net.dst)
                || grid.is_obstacle(net.src)
                || grid.is_obstacle(net.dst)
            {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
        }

        let mut placed: HashMap<usize, Placed> = HashMap::new();
        // Nets we have permanently given up on (blocked by un-displaceable nets or
        // unroutable even on the empty grid). Never retried.
        let mut abandoned: Vec<bool> = vec![false; nets.len()];

        let mut passes = 0u32;
        loop {
            if passes >= MAX_PASSES {
                break;
            }
            passes += 1;

            // Deterministic order: ascending index. A net is "pending" when it is
            // neither placed nor abandoned.
            let pending: Vec<usize> = (0..nets.len())
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

                // Blocked. Can we route at all on the empty grid?
                let Some((free_path, _)) = LeeRouter::route_one(grid, net.src, net.dst) else {
                    // Unroutable regardless of other nets: give up permanently.
                    abandoned[ni] = true;
                    continue;
                };

                // Find committed nets of HIGHER index that overlap our free path —
                // these we may rip up (priority rule).
                let blockers: Vec<usize> = placed
                    .iter()
                    .filter(|(&oi, _)| oi > ni)
                    .filter(|(_, p)| p.path.iter().any(|c| free_path.contains(c)))
                    .map(|(&oi, _)| oi)
                    .collect();

                if blockers.is_empty() {
                    // Only lower-index (un-displaceable) nets block us, or no
                    // committed net overlaps our shortest path yet a path exists on
                    // the working grid was not found — give up permanently to keep
                    // the process bounded and deterministic.
                    abandoned[ni] = true;
                    continue;
                }

                for b in blockers {
                    placed.remove(&b);
                }
                progressed = true;
                // `ni` will be retried next pass (it is now pending again).
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
        Ok(BoardRoute {
            results,
            unrouted,
            congestion,
        })
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
}
