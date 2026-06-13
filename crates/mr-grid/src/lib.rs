//! `mr-grid` — construction of [`mr_core::Grid`] cost grids in *cell space*.
//!
//! Continuous-geometry rasterisation (floats → cells) lives in `mr-srj`; this
//! crate works purely in integer cell coordinates: it marks obstacle cells and
//! inflates them by a clearance radius. The output is the canonical
//! [`mr_core::Grid`] every router consumes.

use mr_core::{CellIdx, Cost, Dims, Grid, OBSTACLE};

/// Builds a [`Grid`] by stamping obstacle cells and (optionally) inflating them.
///
/// Typical use:
/// ```
/// use mr_grid::GridBuilder;
/// use mr_core::Dims;
/// let grid = GridBuilder::new(Dims::new(8, 8), 1)
///     .mark_rect(3, 3, 4, 4)   // inclusive cell rectangle
///     .inflate_clearance(1)    // grow obstacles by 1 cell (Chebyshev)
///     .build();
/// assert!(grid.is_well_formed());
/// ```
#[derive(Debug, Clone)]
pub struct GridBuilder {
    dims: Dims,
    base_cost: Cost,
    /// `true` == obstacle. Length == `dims.len()`.
    blocked: Vec<bool>,
}

impl GridBuilder {
    /// New builder over `dims`; every cell starts passable at `base_cost`.
    pub fn new(dims: Dims, base_cost: Cost) -> Self {
        debug_assert_ne!(base_cost, OBSTACLE, "base_cost must not equal OBSTACLE");
        Self {
            blocked: vec![false; dims.len()],
            dims,
            base_cost,
        }
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Mark a single cell as an obstacle. Out-of-bounds is ignored.
    pub fn mark_cell(&mut self, x: u32, y: u32) -> &mut Self {
        if self.dims.in_bounds(x, y) {
            let i = self.dims.idx(x, y) as usize;
            self.blocked[i] = true;
        }
        self
    }

    /// Mark an inclusive cell rectangle `[x0,x1] × [y0,y1]` as obstacles.
    /// Coordinates are clamped to the grid; order of corners does not matter.
    pub fn mark_rect(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> &mut Self {
        let (lo_x, hi_x) = (x0.min(x1), x0.max(x1).min(self.dims.w.saturating_sub(1)));
        let (lo_y, hi_y) = (y0.min(y1), y0.max(y1).min(self.dims.h.saturating_sub(1)));
        if self.dims.is_empty() {
            return self;
        }
        for y in lo_y..=hi_y {
            for x in lo_x..=hi_x {
                let i = self.dims.idx(x, y) as usize;
                self.blocked[i] = true;
            }
        }
        self
    }

    /// Grow every obstacle by `radius` cells using **Chebyshev** distance (a
    /// square halo), computed against the obstacle set as it stands *now* so the
    /// inflation does not compound. `radius == 0` is a no-op.
    pub fn inflate_clearance(&mut self, radius: u32) -> &mut Self {
        if radius == 0 || self.dims.is_empty() {
            return self;
        }
        let seeds: Vec<CellIdx> = (0..self.dims.len() as u32)
            .filter(|&i| self.blocked[i as usize])
            .collect();
        let r = radius as i64;
        for seed in seeds {
            let (sx, sy) = self.dims.xy(seed);
            let (sx, sy) = (sx as i64, sy as i64);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (sx + dx, sy + dy);
                    if nx < 0 || ny < 0 || nx >= self.dims.w as i64 || ny >= self.dims.h as i64 {
                        continue;
                    }
                    let i = self.dims.idx(nx as u32, ny as u32) as usize;
                    self.blocked[i] = true;
                }
            }
        }
        self
    }

    /// Materialise the [`Grid`]: blocked cells become [`OBSTACLE`], the rest
    /// `base_cost`.
    pub fn build(&self) -> Grid {
        let cost = self
            .blocked
            .iter()
            .map(|&b| if b { OBSTACLE } else { self.base_cost })
            .collect();
        Grid {
            dims: self.dims,
            cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_rect_obstacle() {
        let g = GridBuilder::new(Dims::new(5, 5), 1)
            .mark_rect(1, 1, 2, 3)
            .build();
        assert!(g.is_well_formed());
        assert!(g.is_obstacle(g.dims.idx(1, 1)));
        assert!(g.is_obstacle(g.dims.idx(2, 3)));
        assert!(!g.is_obstacle(g.dims.idx(0, 0)));
        assert!(!g.is_obstacle(g.dims.idx(3, 3)));
        // count == 2 wide * 3 tall
        let n = g.cost.iter().filter(|&&c| c == OBSTACLE).count();
        assert_eq!(n, 6);
    }

    #[test]
    fn clearance_inflation_marks_expected_cells() {
        // single obstacle at center of 5x5, inflate by 1 => 3x3 block of obstacles
        let d = Dims::new(5, 5);
        let g = GridBuilder::new(d, 1)
            .mark_cell(2, 2)
            .inflate_clearance(1)
            .build();
        let obstacles = g.cost.iter().filter(|&&c| c == OBSTACLE).count();
        assert_eq!(
            obstacles, 9,
            "1-cell Chebyshev halo of one cell is a 3x3 block"
        );
        for y in 1..=3 {
            for x in 1..=3 {
                assert!(g.is_obstacle(d.idx(x, y)), "({x},{y}) should be obstacle");
            }
        }
        assert!(!g.is_obstacle(d.idx(0, 2)));
    }

    #[test]
    fn inflation_does_not_compound() {
        // two cells two apart, inflate by 1: halos must NOT bridge the gap.
        let d = Dims::new(7, 1);
        let g = GridBuilder::new(d, 1)
            .mark_cell(1, 0)
            .mark_cell(5, 0)
            .inflate_clearance(1)
            .build();
        // cell 3 (the middle) stays passable
        assert!(!g.is_obstacle(d.idx(3, 0)));
        // each obstacle becomes a run of 3 -> 6 total
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 6);
    }
}
