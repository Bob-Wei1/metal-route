//! `AStarRouter` (A4) — A* with a Manhattan-distance heuristic.
//!
//! Same cost semantics, tie-break, and [`Router`] contract as [`crate::LeeRouter`]:
//! each net is routed independently and unreachable targets land in
//! [`BoardRoute::unrouted`]. The Manhattan heuristic is admissible and consistent
//! on a 4-connected positive-cost grid (and falls back to `h=0` when zero-cost
//! cells exist), so A* returns the same optimal canonical path and cost as Lee.

use mr_core::{BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError};

use crate::dijkstra::{astar_buf, SearchBuf};

/// A* router with Manhattan heuristic. Routes every net independently.
#[derive(Debug, Default, Clone, Copy)]
pub struct AStarRouter;

impl AStarRouter {
    pub fn new() -> Self {
        Self
    }

    /// Manhattan distance between two cells under the canonical mapping, in CELL
    /// units (one per hop). [`AStarRouter`] prices a step by `grid.cost_at(v)` (the
    /// abstract per-cell grid cost, `1` on a unit grid), NOT by the geometric
    /// `COST_SCALE` used by [`crate::NegotiatedRouter`]: it has no continuous
    /// geometry ([`mr_core::GridCoords`]) to draw lengths from. So the heuristic
    /// stays in those same cell units as a unit lower bound (or zero when a zero-cost
    /// cell exists), keeping it admissible on weighted grids and preserving the
    /// A*-equals-Lee invariant. The
    /// geometric, length-aware heuristic lives in `negotiated.rs::manhattan_scaled`.
    fn manhattan(grid: &Grid, a: CellIdx, b: CellIdx, min_step: Cost) -> Cost {
        let (ax, ay) = grid.dims.xy(a);
        let (bx, by) = grid.dims.xy(b);
        (ax.abs_diff(bx) + ay.abs_diff(by)).saturating_mul(min_step)
    }

    /// Per-board lower bound for one planar step. The current grid contract makes
    /// this either zero (a passable zero-cost cell exists) or one. Keep the scan
    /// outside the per-net hot path.
    fn min_step(grid: &Grid) -> Cost {
        grid.cost
            .iter()
            .copied()
            .filter(|&c| c != mr_core::OBSTACLE)
            .min()
            .unwrap_or(1)
            .min(1)
    }

    #[allow(dead_code)]
    pub(crate) fn route_one(
        grid: &Grid,
        src: CellIdx,
        dst: CellIdx,
    ) -> Option<(Vec<CellIdx>, Cost)> {
        let mut buf = SearchBuf::new(grid.dims.len());
        Self::route_one_with_buf(&mut buf, grid, src, dst, &[], Self::min_step(grid))
    }

    fn route_one_with_buf(
        buf: &mut SearchBuf,
        grid: &Grid,
        src: CellIdx,
        dst: CellIdx,
        passable_pads: &[CellIdx],
        min_step: Cost,
    ) -> Option<(Vec<CellIdx>, Cost)> {
        astar_buf(
            buf,
            grid.dims,
            src,
            dst,
            |u, v| {
                if grid.is_board_planar_step_forbidden(u, v) {
                    return mr_core::OBSTACLE;
                }
                if grid.is_obstacle(v) && passable_pads.contains(&v) {
                    1
                } else {
                    grid.cost_at(v)
                }
            },
            |c| grid.is_board_forbidden(c) || (grid.is_obstacle(c) && !passable_pads.contains(&c)),
            |c| Self::manhattan(grid, c, dst, min_step),
            |_, _| None,
        )
    }
}

impl Router for AStarRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        let mut results = Vec::new();
        let mut unrouted = Vec::new();
        let mut buf = SearchBuf::new(grid.dims.len());
        // One O(n_cells) scan per board, not once for every net.
        let min_step = Self::min_step(grid);
        for net in nets {
            if net.passable_pads.iter().any(|&c| !grid.dims.contains(c)) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            if !grid.dims.contains(net.src)
                || !grid.dims.contains(net.dst)
                || grid.is_board_forbidden(net.src)
                || grid.is_board_forbidden(net.dst)
                || (grid.is_obstacle(net.src) && !net.passable_pads.contains(&net.src))
                || (grid.is_obstacle(net.dst) && !net.passable_pads.contains(&net.dst))
            {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            match Self::route_one_with_buf(
                &mut buf,
                grid,
                net.src,
                net.dst,
                &net.passable_pads,
                min_step,
            ) {
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
    use crate::LeeRouter;
    use mr_core::{Dims, NetEndpoints};
    use mr_fixtures::obstacle_battery;
    use mr_grid::GridBuilder;

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

    #[test]
    fn astar_matches_lee_canonical_path_on_asymmetric_tie() {
        // Equal-cost detours start through cells 0 and 5.  Heuristic order reaches
        // predecessor 5 first, but the shared LowerCellIdx contract requires 0.
        let dims = Dims::new(4, 3);
        let grid = GridBuilder::new(dims, 1).mark_cell(2, 1).build();
        let n = NetEndpoints {
            net: "tie".into(),
            src: dims.idx(0, 1),
            dst: dims.idx(3, 1),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };
        let lee = LeeRouter.route(&grid, std::slice::from_ref(&n)).unwrap();
        let astar = AStarRouter.route(&grid, std::slice::from_ref(&n)).unwrap();
        assert_eq!(lee.results[0].path, vec![4, 0, 1, 2, 3, 7]);
        assert_eq!(astar, lee);
    }

    #[test]
    fn astar_equals_lee_on_every_3x3_obstacle_mask_and_endpoint_pair() {
        let dims = Dims::new(3, 3);
        for src in 0..dims.len() as u32 {
            for dst in 0..dims.len() as u32 {
                if src == dst {
                    continue;
                }
                let candidates: Vec<_> = (0..dims.len() as u32)
                    .filter(|&c| c != src && c != dst)
                    .collect();
                for mask in 0usize..(1usize << candidates.len()) {
                    let mut builder = GridBuilder::new(dims, 1);
                    for (bit, &c) in candidates.iter().enumerate() {
                        if mask & (1 << bit) != 0 {
                            let (x, y) = dims.xy(c);
                            builder.mark_cell(x, y);
                        }
                    }
                    let grid = builder.build();
                    let n = NetEndpoints {
                        net: "n".into(),
                        src,
                        dst,
                        passable_pads: Vec::new(),
                        via_passable_pads: Vec::new(),
                    };
                    let lee = LeeRouter.route(&grid, std::slice::from_ref(&n)).unwrap();
                    let astar = AStarRouter.route(&grid, std::slice::from_ref(&n)).unwrap();
                    assert_eq!(astar, lee, "src={src} dst={dst} mask={mask:#09b}");
                }
            }
        }
    }
}
