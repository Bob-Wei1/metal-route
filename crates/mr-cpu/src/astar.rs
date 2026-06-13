//! `AStarRouter` (A4) — A* with a Manhattan-distance heuristic.
//!
//! Same cost semantics, tie-break, and [`Router`] contract as [`crate::LeeRouter`]:
//! each net is routed independently and unreachable targets land in
//! [`BoardRoute::unrouted`]. The Manhattan heuristic is admissible and consistent
//! on a 4-connected unit-cost grid, so A* returns the same optimal cost as Lee
//! (paths may differ when ties exist, but total cost is identical).

use mr_core::{BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError};

use crate::dijkstra::{dijkstra, reconstruct_path};

/// A* router with Manhattan heuristic. Routes every net independently.
#[derive(Debug, Default, Clone, Copy)]
pub struct AStarRouter;

impl AStarRouter {
    pub fn new() -> Self {
        Self
    }

    /// Manhattan distance between two cells under the canonical mapping.
    fn manhattan(grid: &Grid, a: CellIdx, b: CellIdx) -> Cost {
        let (ax, ay) = grid.dims.xy(a);
        let (bx, by) = grid.dims.xy(b);
        ax.abs_diff(bx) + ay.abs_diff(by)
    }

    pub(crate) fn route_one(
        grid: &Grid,
        src: CellIdx,
        dst: CellIdx,
    ) -> Option<(Vec<CellIdx>, Cost)> {
        let field = dijkstra(grid, src, |c| Self::manhattan(grid, c, dst));
        let path = reconstruct_path(&field.pred, src, dst, &field.dist)?;
        Some((path, field.dist[dst as usize]))
    }
}

impl Router for AStarRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        let mut results = Vec::new();
        let mut unrouted = Vec::new();
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
            match Self::route_one(grid, net.src, net.dst) {
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LeeRouter;
    use mr_fixtures::obstacle_battery;

    #[test]
    fn astar_total_cost_equals_lee_on_battery() {
        for f in obstacle_battery() {
            let lee = LeeRouter.route(&f.grid, &f.nets).unwrap();
            let astar = AStarRouter.route(&f.grid, &f.nets).unwrap();
            assert_eq!(
                astar.total_cost(),
                lee.total_cost(),
                "{}: A* cost must equal Lee cost",
                f.name
            );
            assert_eq!(
                astar.unrouted, lee.unrouted,
                "{}: same nets unrouted",
                f.name
            );
        }
    }
}
