//! `mr-cpu` — the CPU routers and the M0 GPU de-risk spike.
//!
//! This crate implements the routing algorithms behind the [`mr_core::Router`]
//! contract:
//!
//! * [`LeeRouter`] (A1 / M1) — Lee's wavefront as Dijkstra single-source shortest
//!   path; routes each net independently.
//! * [`AStarRouter`] (A4) — A* with a Manhattan heuristic; same cost & contract,
//!   equal total cost to [`LeeRouter`].
//! * [`RipUpRouter`] (A3 / M2) — sequential routing with bounded
//!   rip-up-on-collision (`K == 20` passes).
//! * [`sweep`] (A2 / M0) — the separable H/V prefix-min distance field
//!   ([`sweep_distance_field`]) validated against the Dijkstra field
//!   ([`bfs_distance_field`]). This is the GPU de-risk spike.
//!
//! ## Shared conventions
//!
//! * A routed net's `cost` is the sum of `grid.cost_at(cell)` over the path
//!   **excluding** the source cell — so on a unit-cost grid `cost == path.len() - 1`.
//! * Tie-break is [`mr_core::TieBreak::LowerCellIdx`]: equal-cost paths prefer fewer
//!   hops and then the lower predecessor cell. The hop key prevents predecessor
//!   cycles on zero-cost plateaus while keeping reconstruction deterministic.
//! * The grid is 4-connected; obstacle cells (`cost == mr_core::OBSTACLE`) are
//!   impassable.

mod astar;
mod dijkstra;
mod lee;
mod negotiated;
mod ripup;
pub mod sweep;

pub use astar::AStarRouter;
pub use lee::LeeRouter;
pub use negotiated::{
    IsolatedRouteProvider, IsolatedRouteRequest, IsolatedRouteWindow, NegotiatedOutcome,
    NegotiatedRouter, MAX_ITERS, SCALE,
};
// Re-export the trace contract types (defined in `mr-core`) so callers of
// `NegotiatedRouter::route_traced` get them from one place.
pub use mr_core::{CandidateEval, IterSnapshot, LegalizationTrace, RouteTrace, TracedNet};
pub use ripup::{RipUpRouter, MAX_PASSES};
pub use sweep::{bfs_distance_field, sweep_distance_field};

#[cfg(test)]
mod contract_tests {
    use super::*;
    use mr_core::{Dims, Grid, NetEndpoints, Router, RouterError, OBSTACLE};
    use mr_grid::GridBuilder;

    fn net(name: &str, src: u32, dst: u32) -> NetEndpoints {
        NetEndpoints {
            net: name.into(),
            src,
            dst,
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        }
    }

    fn routers() -> Vec<(&'static str, Box<dyn Router>)> {
        vec![
            ("lee", Box::new(LeeRouter)),
            ("astar", Box::new(AStarRouter)),
            ("ripup", Box::new(RipUpRouter)),
            ("negotiated", Box::new(NegotiatedRouter::new())),
        ]
    }

    #[test]
    fn every_router_unmasks_own_pad_endpoints() {
        let dims = Dims::new(2, 1);
        let grid = GridBuilder::new(dims, 1).mark_rect(0, 0, 1, 0).build();
        let n = NetEndpoints {
            net: "pad-net".into(),
            src: 0,
            dst: 1,
            passable_pads: vec![0, 1],
            via_passable_pads: Vec::new(),
        };
        for (name, router) in routers() {
            let board = router.route(&grid, std::slice::from_ref(&n)).unwrap();
            assert!(board.unrouted.is_empty(), "{name}: own pads must open");
            assert_eq!(board.results[0].path, vec![0, 1], "{name}");
            assert_eq!(board.results[0].cost, 1, "{name}");
        }
    }

    #[test]
    fn every_router_rejects_out_of_bounds_pad_without_panicking() {
        let dims = Dims::new(2, 1);
        let grid = GridBuilder::new(dims, 1).build();
        let n = NetEndpoints {
            net: "bad-pad".into(),
            src: 0,
            dst: 1,
            passable_pads: vec![dims.len() as u32 + 17],
            via_passable_pads: Vec::new(),
        };
        for (name, router) in routers() {
            assert_eq!(
                router.route(&grid, std::slice::from_ref(&n)),
                Err(RouterError::InvalidEndpoint {
                    net: "bad-pad".into()
                }),
                "{name}"
            );
        }
    }

    #[test]
    fn every_router_honors_weighted_grid_costs() {
        // The two-hop top route enters a cost-100 cell; the four-hop lower detour
        // is the unique cheapest route and must also determine reported cost.
        let dims = Dims::new(3, 2);
        let mut grid = GridBuilder::new(dims, 1).build();
        grid.set(dims.idx(1, 0), 100);
        let n = net("weighted", dims.idx(0, 0), dims.idx(2, 0));
        for (name, router) in routers() {
            let board = router.route(&grid, std::slice::from_ref(&n)).unwrap();
            let result = &board.results[0];
            assert_eq!(result.path, vec![0, 3, 4, 5, 2], "{name}");
            assert_eq!(result.cost, 4, "{name}: cost is the enter-cost sum");
        }
    }

    #[test]
    fn public_shortest_path_routers_are_cycle_free_on_zero_cost_plateau() {
        // Regression: predecessor-only equal-cost rewrites produced
        // 3 -> 0 -> 1 -> 0 here and hung forever during reconstruction.
        let dims = Dims::new(3, 2);
        let grid = Grid::filled(dims, 0);
        let n = net("zero", 5, 3);
        let expected = vec![5, 4, 3];
        for (name, router) in [
            ("lee", Box::new(LeeRouter) as Box<dyn Router>),
            ("astar", Box::new(AStarRouter) as Box<dyn Router>),
        ] {
            let board = router.route(&grid, std::slice::from_ref(&n)).unwrap();
            assert_eq!(board.results[0].path, expected, "{name}");
            assert_eq!(board.results[0].cost, 0, "{name}");
        }
    }

    #[test]
    fn every_router_handles_zero_length_net() {
        let dims = Dims::new(3, 3);
        let grid = GridBuilder::new(dims, 1).build();
        let n = net("point", dims.idx(1, 1), dims.idx(1, 1));
        for (name, router) in routers() {
            let board = router.route(&grid, std::slice::from_ref(&n)).unwrap();
            assert_eq!(board.results[0].path, vec![dims.idx(1, 1)], "{name}");
            assert_eq!(board.results[0].cost, 0, "{name}");
        }
    }

    #[test]
    fn every_router_validates_grid_before_indexing_endpoints() {
        let dims = Dims::new(2, 2);
        let malformed = Grid {
            dims,
            cost: vec![1],
            via_forbidden: Vec::new(),
        };
        let n = net("n", 999, 1000);
        for (name, router) in routers() {
            assert_eq!(
                router.route(&malformed, std::slice::from_ref(&n)),
                Err(RouterError::MalformedGrid),
                "{name}"
            );
        }
    }

    #[test]
    fn every_router_rejects_malformed_via_mask_length() {
        let dims = Dims::new(2, 2);
        let malformed = Grid {
            dims,
            cost: vec![1; dims.len()],
            via_forbidden: vec![false; dims.len() - 1],
        };
        let n = net("n", 0, 1);
        for (name, router) in routers() {
            assert_eq!(
                router.route(&malformed, std::slice::from_ref(&n)),
                Err(RouterError::MalformedGrid),
                "{name}"
            );
        }
    }

    #[test]
    fn every_router_rejects_foreign_obstacle_endpoint() {
        let dims = Dims::new(2, 1);
        let mut grid = Grid::filled(dims, 1);
        grid.set(1, OBSTACLE);
        let n = net("foreign", 0, 1);
        for (name, router) in routers() {
            assert!(
                matches!(
                    router.route(&grid, std::slice::from_ref(&n)),
                    Err(RouterError::InvalidEndpoint { .. })
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn board_route_accounting_and_congestion_are_exact() {
        let dims = Dims::new(5, 3);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 0), dims.idx(4, 0)),
            net("b", dims.idx(0, 2), dims.idx(4, 2)),
        ];
        for (name, router) in routers() {
            let board = router.route(&grid, &nets).unwrap();
            assert!(board.unrouted.is_empty(), "{name}");
            assert_eq!(
                board
                    .results
                    .iter()
                    .map(|r| r.net.as_str())
                    .collect::<Vec<_>>(),
                vec!["a", "b"],
                "{name}: results retain input order"
            );
            assert_eq!(
                board.congestion,
                mr_core::BoardRoute::congestion_from(dims, &board.results),
                "{name}"
            );
            assert_eq!(
                board.results.iter().map(|r| r.cost).sum::<u32>(),
                8,
                "{name}"
            );
        }
    }
}
