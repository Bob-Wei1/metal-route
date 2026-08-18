//! `LeeRouter` (A1 / M1) — Lee's wavefront expansion as single-source shortest
//! path.
//!
//! Lee's algorithm is a BFS wavefront; to honour per-cell costs we implement it as
//! Dijkstra (a binary heap). On the uniform-cost grids in the fixtures this is
//! exactly BFS. Each net is routed **independently** — other nets are ignored.
//! A net whose target is unreachable is reported in [`BoardRoute::unrouted`] and
//! produces no [`RouteResult`].

use mr_core::{BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError};

use crate::dijkstra::{astar_buf, SearchBuf};

/// Lee/Dijkstra single-source router. Routes every net independently.
#[derive(Debug, Default, Clone, Copy)]
pub struct LeeRouter;

impl LeeRouter {
    pub fn new() -> Self {
        Self
    }

    /// Route a single net against `grid`, treating it in isolation. Returns the
    /// path (`src..=dst`) and its cost, or `None` when `dst` is unreachable.
    ///
    /// `cost` = sum of `cost_at` over the path **excluding** the source.
    pub(crate) fn route_one(
        grid: &Grid,
        src: CellIdx,
        dst: CellIdx,
    ) -> Option<(Vec<CellIdx>, Cost)> {
        let mut buf = SearchBuf::new(grid.dims.len());
        Self::route_one_with_buf(&mut buf, grid, src, dst, &[])
    }

    /// Targeted Lee/Dijkstra search with reusable scratch and per-net pad
    /// unmasking.  An obstacle listed in `passable_pads` is entered at the normal
    /// free-cell cost (`1`); ordinary weighted cells retain their grid cost.
    fn route_one_with_buf(
        buf: &mut SearchBuf,
        grid: &Grid,
        src: CellIdx,
        dst: CellIdx,
        passable_pads: &[CellIdx],
    ) -> Option<(Vec<CellIdx>, Cost)> {
        astar_buf(
            buf,
            grid.dims,
            src,
            dst,
            |_u, v| {
                if grid.is_obstacle(v) && passable_pads.contains(&v) {
                    1
                } else {
                    grid.cost_at(v)
                }
            },
            |c| grid.is_obstacle(c) && !passable_pads.contains(&c),
            |_| 0,
            |_, _| None,
        )
    }
}

impl Router for LeeRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        let mut results = Vec::new();
        let mut unrouted = Vec::new();
        let mut buf = SearchBuf::new(grid.dims.len());
        for net in nets {
            if net.passable_pads.iter().any(|&c| !grid.dims.contains(c)) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            if !grid.dims.contains(net.src)
                || !grid.dims.contains(net.dst)
                || (grid.is_obstacle(net.src) && !net.passable_pads.contains(&net.src))
                || (grid.is_obstacle(net.dst) && !net.passable_pads.contains(&net.dst))
            {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            match Self::route_one_with_buf(&mut buf, grid, net.src, net.dst, &net.passable_pads) {
                Some((path, cost)) => results.push(RouteResult {
                    net: net.net.clone(),
                    path,
                    cost,
                }),
                None => unrouted.push(net.net.clone()),
            }
        }
        let congestion = BoardRoute::congestion_from(grid.dims, &results);
        Ok(BoardRoute {
            results,
            unrouted,
            congestion,
            groups: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::OBSTACLE;
    use mr_fixtures::{hand_32x32_wall, obstacle_battery, tie_break_2x2};

    /// Helper: assert a path is contiguous (4-neighbours), endpoint-anchored, and
    /// avoids obstacles.
    fn assert_valid_path(grid: &Grid, path: &[CellIdx], src: CellIdx, dst: CellIdx) {
        assert_eq!(path.first().copied(), Some(src), "path must start at src");
        assert_eq!(path.last().copied(), Some(dst), "path must end at dst");
        for &c in path {
            assert_ne!(grid.cost_at(c), OBSTACLE, "path cell {c} is an obstacle");
        }
        for w in path.windows(2) {
            let neigh = grid.dims.neighbors4(w[0]);
            assert!(
                neigh.contains(&w[1]),
                "cells {} and {} are not 4-neighbours",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn hand_32x32_wall_costs_93_and_valid() {
        let f = hand_32x32_wall();
        let br = LeeRouter.route(&f.grid, &f.nets).unwrap();
        assert!(br.unrouted.is_empty());
        assert_eq!(br.results.len(), 1);
        let r = &br.results[0];
        assert_eq!(r.cost, 93);
        assert_valid_path(&f.grid, &r.path, f.nets[0].src, f.nets[0].dst);
        assert_eq!(
            r.cost as usize,
            r.path.len() - 1,
            "unit grid: cost == moves"
        );
    }

    #[test]
    fn tie_break_2x2_exact_path() {
        let f = tie_break_2x2();
        let br = LeeRouter.route(&f.grid, &f.nets).unwrap();
        let r = &br.results[0];
        assert_eq!(r.path, vec![0, 1, 3], "LowerCellIdx pins [0,1,3]");
        assert_eq!(r.cost, 2);
        assert_eq!(f.expected_path.unwrap(), r.path);
    }

    #[test]
    fn obstacle_battery_pinned_costs_match() {
        for f in obstacle_battery() {
            if let Some(expected) = f.expected_total_cost {
                let br = LeeRouter.route(&f.grid, &f.nets).unwrap();
                assert_eq!(
                    br.total_cost(),
                    expected,
                    "{}: expected total cost {expected}",
                    f.name
                );
            }
        }
    }

    #[test]
    fn blocked_gap_target_unrouted() {
        let f = obstacle_battery()
            .into_iter()
            .find(|f| f.name == "blocked_gap")
            .expect("blocked_gap fixture present");
        let br = LeeRouter.route(&f.grid, &f.nets).unwrap();
        assert!(br.results.is_empty(), "enclosed target yields no result");
        assert_eq!(br.unrouted, vec!["n0".to_string()]);
    }
}
