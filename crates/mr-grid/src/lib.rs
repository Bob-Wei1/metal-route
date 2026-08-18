//! `mr-grid` — construction of [`mr_core::Grid`] cost grids in *cell space*.
//!
//! Continuous-geometry rasterisation (floats → cells) lives in `mr-srj`; this
//! crate works purely in integer cell coordinates: it marks obstacle cells and
//! inflates them by a clearance radius. The output is the canonical
//! [`mr_core::Grid`] every router consumes.

use mr_core::{CellIdx, Cost, Dims, Grid, GridCoords, OBSTACLE};

/// Builds a [`Grid`] by stamping obstacle cells and (optionally) inflating them.
///
/// Typical use:
/// ```
/// use mr_grid::GridBuilder;
/// use mr_core::{Dims, GridCoords};
/// let dims = Dims::new(8, 8);
/// let coords = GridCoords::uniform(dims); // unit cells: 1mm clearance == 1 cell
/// let grid = GridBuilder::new(dims, 1)
///     .mark_rect(3, 3, 4, 4)            // inclusive cell rectangle
///     .inflate_clearance(1.0, &coords) // grow obstacles by 1mm (geometric)
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

    /// Mark a single cell on layer 0 as an obstacle. Out-of-bounds is ignored.
    ///
    /// This is the historical 2D entry point: on a single-layer grid layer 0 *is*
    /// the whole grid, so the behaviour is byte-identical to before layers existed.
    /// For an explicit layer use [`Self::mark_cell_layer`].
    pub fn mark_cell(&mut self, x: u32, y: u32) -> &mut Self {
        self.mark_cell_layer(x, y, 0)
    }

    /// Mark a single cell on `layer` as an obstacle. Out-of-bounds (in x, y, or
    /// layer) is ignored.
    pub fn mark_cell_layer(&mut self, x: u32, y: u32, layer: u32) -> &mut Self {
        if self.dims.in_bounds(x, y) && layer < self.dims.layers {
            let i = self.dims.idx3(x, y, layer) as usize;
            self.blocked[i] = true;
        }
        self
    }

    /// Clear a single cell on layer 0 back to passable (the inverse of
    /// [`Self::mark_cell`]). Out-of-bounds is ignored. Used to guarantee a routing
    /// endpoint is never left on an obstacle even when an obstacle rect overlaps it.
    pub fn clear_cell(&mut self, x: u32, y: u32) -> &mut Self {
        self.clear_cell_layer(x, y, 0)
    }

    /// Clear a single cell on `layer` back to passable. Out-of-bounds is ignored.
    pub fn clear_cell_layer(&mut self, x: u32, y: u32, layer: u32) -> &mut Self {
        if self.dims.in_bounds(x, y) && layer < self.dims.layers {
            let i = self.dims.idx3(x, y, layer) as usize;
            self.blocked[i] = false;
        }
        self
    }

    /// Mark an inclusive cell rectangle `[x0,x1] × [y0,y1]` on layer 0 as
    /// obstacles. Coordinates are clamped to the grid; order of corners does not
    /// matter. See [`Self::mark_rect_layer`] for an explicit layer.
    pub fn mark_rect(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> &mut Self {
        self.mark_rect_layer(x0, y0, x1, y1, 0)
    }

    /// Mark an inclusive cell rectangle `[x0,x1] × [y0,y1]` on `layer` as
    /// obstacles. Coordinates are clamped to the grid; order of corners does not
    /// matter. An out-of-range `layer` is ignored.
    pub fn mark_rect_layer(&mut self, x0: u32, y0: u32, x1: u32, y1: u32, layer: u32) -> &mut Self {
        if self.dims.is_empty() || layer >= self.dims.layers {
            return self;
        }
        let (lo_x, hi_x) = (x0.min(x1), x0.max(x1).min(self.dims.w.saturating_sub(1)));
        let (lo_y, hi_y) = (y0.min(y1), y0.max(y1).min(self.dims.h.saturating_sub(1)));
        for y in lo_y..=hi_y {
            for x in lo_x..=hi_x {
                let i = self.dims.idx3(x, y, layer) as usize;
                self.blocked[i] = true;
            }
        }
        self
    }

    /// Grow every obstacle by `clearance` *continuous units* (mm) using
    /// **geometric Chebyshev** distance (a square halo measured against the grid
    /// line positions in `coords`, not against a cell count), computed against the
    /// obstacle set as it stands *now* so the inflation does not compound.
    /// `clearance <= 0` is a no-op.
    ///
    /// On a **non-uniform** (Hanan) grid the cells are unequal, so a fixed cell
    /// radius would over-inflate where lines are dense and under-inflate where they
    /// are sparse. Instead, for each blocked seed cell at line position
    /// `(coords.x_of(sx), coords.y_of(sy))`, a neighbour `(nx, ny)` is marked when
    /// its line distance from the seed is within `clearance` on *both* axes —
    /// i.e. `|x_of(nx) - x_of(sx)| <= clearance && |y_of(ny) - y_of(sy)| <=
    /// clearance`. This reproduces the old square-halo shape but in continuous
    /// units. On a [`GridCoords::uniform`] grid (unit cells) a `clearance` of `n`
    /// is byte-identical to the former `n`-cell Chebyshev radius.
    ///
    /// Inflation is purely *planar*: each obstacle cell grows its halo only within
    /// its own layer plane and never bleeds clearance onto an adjacent layer (a
    /// via's keepout is a separate concern — see `mr-srj`). On a single-layer grid
    /// this is byte-identical to before layers existed.
    ///
    /// The scan window per axis is bounded by walking outward from the seed line
    /// until the line distance first exceeds `clearance`, so cost is proportional
    /// to the number of lines actually within clearance — not to the whole grid.
    pub fn inflate_clearance(&mut self, clearance: f64, coords: &GridCoords) -> &mut Self {
        if clearance <= 0.0 || self.dims.is_empty() {
            return self;
        }
        let seeds: Vec<CellIdx> = (0..self.dims.len() as u32)
            .filter(|&i| self.blocked[i as usize])
            .collect();
        for seed in seeds {
            let (sx, sy, sl) = self.dims.xyz(seed);
            // Per-axis half-open ranges of line indices within `clearance` of the
            // seed line. `line_span` walks outward and stops at the first line that
            // falls outside clearance, so the window is exactly the in-clearance
            // band (deterministic: a purely positional, integer-indexed scan).
            let sxp = coords.x_of(sx);
            let syp = coords.y_of(sy);

            // `GridCoords::{x,y}_of` deliberately degrade missing coordinate
            // entries to unit-spaced positions. A truncated array is not guaranteed
            // to remain sorted at the explicit/fallback boundary, so the bounded
            // `line_span` walk is not sound there. This malformed-input defensive
            // path scans the plane directly and applies the documented getters;
            // normal well-formed mappings keep the fast bounded window below.
            if coords.x_lines.len() < self.dims.w as usize
                || coords.y_lines.len() < self.dims.h as usize
            {
                for ny in 0..self.dims.h {
                    if (coords.y_of(ny) - syp).abs() > clearance {
                        continue;
                    }
                    for nx in 0..self.dims.w {
                        if (coords.x_of(nx) - sxp).abs() <= clearance {
                            let i = self.dims.idx3(nx, ny, sl) as usize;
                            self.blocked[i] = true;
                        }
                    }
                }
                continue;
            }

            let (x0, x1) = line_span(&coords.x_lines, self.dims.w, sx, sxp, clearance);
            let (y0, y1) = line_span(&coords.y_lines, self.dims.h, sy, syp, clearance);
            for ny in y0..y1 {
                for nx in x0..x1 {
                    // Stay on the seed's own layer plane: clearance must not bleed
                    // across layers.
                    let i = self.dims.idx3(nx, ny, sl) as usize;
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
            via_forbidden: Vec::new(),
        }
    }
}

/// Half-open `[lo, hi)` range of line indices in `lines` (a sorted ascending,
/// non-empty coordinate array of length `count`) whose position is within
/// `clearance` continuous units of the seed line at index `seed` / position `pos`.
///
/// `lines` are sorted, so the in-clearance indices form a contiguous band around
/// `seed`; we walk outward from `seed` in each direction and stop at the first
/// line strictly farther than `clearance`. Only `lines[..count]` is consulted, so
/// a coords array longer than `dims` (defensive) never reads past the grid.
fn line_span(lines: &[f64], count: u32, seed: u32, pos: f64, clearance: f64) -> (u32, u32) {
    let n = (lines.len() as u32).min(count);
    if n == 0 {
        return (0, 0);
    }
    let seed = seed.min(n - 1);
    let mut lo = seed;
    while lo > 0 && (pos - lines[(lo - 1) as usize]).abs() <= clearance {
        lo -= 1;
    }
    let mut hi = seed + 1;
    while hi < n && (lines[hi as usize] - pos).abs() <= clearance {
        hi += 1;
    }
    (lo, hi)
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
            .inflate_clearance(1.0, &GridCoords::uniform(d))
            .build();
        let obstacles = g.cost.iter().filter(|&&c| c == OBSTACLE).count();
        assert_eq!(
            obstacles, 9,
            "1mm halo on a unit grid is a 3x3 block (Chebyshev parity)"
        );
        for y in 1..=3 {
            for x in 1..=3 {
                assert!(g.is_obstacle(d.idx(x, y)), "({x},{y}) should be obstacle");
            }
        }
        assert!(!g.is_obstacle(d.idx(0, 2)));
    }

    #[test]
    fn mark_rect_layer_isolates_layers() {
        // A 2-layer grid: marking a rect on the bottom layer must not touch the
        // top layer, and vice versa.
        let d = Dims::with_layers(5, 5, 2);
        let g = GridBuilder::new(d, 1)
            .mark_rect_layer(1, 1, 2, 2, 1) // bottom layer only
            .build();
        // Bottom (layer 1) is blocked.
        assert!(g.is_obstacle(d.idx3(1, 1, 1)));
        assert!(g.is_obstacle(d.idx3(2, 2, 1)));
        // Top (layer 0) at the same (x,y) is untouched.
        assert!(!g.is_obstacle(d.idx3(1, 1, 0)));
        assert!(!g.is_obstacle(d.idx3(2, 2, 0)));
        // Exactly the 2x2 block on one layer.
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 4);
    }

    #[test]
    fn default_methods_target_layer0() {
        // The historical 2D methods address layer 0 on a multi-layer grid.
        let d = Dims::with_layers(4, 4, 3);
        let g = GridBuilder::new(d, 1).mark_cell(2, 2).build();
        assert!(g.is_obstacle(d.idx3(2, 2, 0)));
        assert!(!g.is_obstacle(d.idx3(2, 2, 1)));
        assert!(!g.is_obstacle(d.idx3(2, 2, 2)));
    }

    #[test]
    fn inflation_does_not_bleed_across_layers() {
        // An obstacle on layer 0 inflated by 1 produces a 3x3 halo on layer 0
        // only; layer 1 stays entirely passable.
        let d = Dims::with_layers(5, 5, 2);
        let g = GridBuilder::new(d, 1)
            .mark_cell_layer(2, 2, 0)
            .inflate_clearance(1.0, &GridCoords::uniform(d))
            .build();
        // 3x3 on layer 0.
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 9);
        for y in 1..=3 {
            for x in 1..=3 {
                assert!(g.is_obstacle(d.idx3(x, y, 0)));
                assert!(!g.is_obstacle(d.idx3(x, y, 1)), "({x},{y}) must not bleed");
            }
        }
    }

    #[test]
    fn inflation_does_not_compound() {
        // two cells two apart, inflate by 1: halos must NOT bridge the gap.
        let d = Dims::new(7, 1);
        let g = GridBuilder::new(d, 1)
            .mark_cell(1, 0)
            .mark_cell(5, 0)
            .inflate_clearance(1.0, &GridCoords::uniform(d))
            .build();
        // cell 3 (the middle) stays passable
        assert!(!g.is_obstacle(d.idx(3, 0)));
        // each obstacle becomes a run of 3 -> 6 total
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 6);
    }

    #[test]
    fn geometric_clearance_respects_nonuniform_spacing() {
        // 1x5 column of lines at non-uniform y positions. Seed the obstacle at the
        // middle line (y=2, pos 5.0) and inflate by 2mm. Only lines within 2mm on
        // the y axis should be marked: y=1 (4.0, |Δ|=1) and y=3 (6.0, |Δ|=1) are
        // in; y=0 (0.0, |Δ|=5) and y=4 (10.0, |Δ|=5) are out — a fixed 1-cell
        // radius would (correctly here) also give 3, but a 2-cell radius would
        // wrongly reach y=0/y=4. Geometric distance keeps it tight.
        let d = Dims::new(1, 5);
        let coords = GridCoords::from_lines(vec![0.0], vec![0.0, 4.0, 5.0, 6.0, 10.0]);
        let g = GridBuilder::new(d, 1)
            .mark_cell(0, 2)
            .inflate_clearance(2.0, &coords)
            .build();
        assert!(!g.is_obstacle(d.idx(0, 0)), "y=0 (5mm away) stays passable");
        assert!(g.is_obstacle(d.idx(0, 1)), "y=1 (1mm away) blocked");
        assert!(g.is_obstacle(d.idx(0, 2)), "seed blocked");
        assert!(g.is_obstacle(d.idx(0, 3)), "y=3 (1mm away) blocked");
        assert!(!g.is_obstacle(d.idx(0, 4)), "y=4 (5mm away) stays passable");
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 3);
    }

    #[test]
    fn geometric_clearance_reaches_far_lines_when_dense() {
        // Densely packed lines (0.5mm apart): a 1mm clearance now spans TWO cells
        // either side, where a 1-cell radius would under-inflate. Seed at x=3
        // (pos 1.5); within 1mm are x in [1..=5] (pos 0.5..2.5).
        let d = Dims::new(7, 1);
        let xs: Vec<f64> = (0..7).map(|i| i as f64 * 0.5).collect();
        let coords = GridCoords::from_lines(xs, vec![0.0]);
        let g = GridBuilder::new(d, 1)
            .mark_cell(3, 0)
            .inflate_clearance(1.0, &coords)
            .build();
        for x in 1..=5 {
            assert!(
                g.is_obstacle(d.idx(x, 0)),
                "x={x} within 1mm must be blocked"
            );
        }
        assert!(!g.is_obstacle(d.idx(0, 0)), "x=0 (1.5mm) stays passable");
        assert!(!g.is_obstacle(d.idx(6, 0)), "x=6 (1.5mm) stays passable");
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 5);
    }

    #[test]
    fn geometric_clearance_boundary_is_inclusive() {
        // A line exactly `clearance` away IS within the halo (<= comparison).
        let d = Dims::new(3, 1);
        let coords = GridCoords::from_lines(vec![0.0, 1.0, 2.0], vec![0.0]);
        let g = GridBuilder::new(d, 1)
            .mark_cell(1, 0)
            .inflate_clearance(1.0, &coords)
            .build();
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 3);
    }

    #[test]
    fn geometric_clearance_zero_is_noop() {
        let d = Dims::new(5, 5);
        let coords = GridCoords::uniform(d);
        let g = GridBuilder::new(d, 1)
            .mark_cell(2, 2)
            .inflate_clearance(0.0, &coords)
            .build();
        assert_eq!(g.cost.iter().filter(|&&c| c == OBSTACLE).count(), 1);
    }

    #[test]
    fn truncated_coordinate_arrays_use_documented_uniform_fallback() {
        let d = Dims::new(5, 1);
        for coords in [
            GridCoords::from_lines(vec![], vec![]),
            GridCoords::from_lines(vec![0.0, 1.0], vec![0.0]),
        ] {
            let g = GridBuilder::new(d, 1)
                .mark_cell(2, 0)
                .inflate_clearance(1.0, &coords)
                .build();
            let blocked: Vec<u32> = (0..d.w).filter(|&x| g.is_obstacle(d.idx(x, 0))).collect();
            assert_eq!(blocked, vec![1, 2, 3]);
        }
    }

    #[test]
    fn geometric_inflation_matches_slow_reference_on_small_nonuniform_grids() {
        let xs = vec![-1.0, -0.4, 0.0, 0.15, 1.7];
        let ys = vec![-2.0, -0.25, 0.5, 2.25];
        let d = Dims::with_layers(xs.len() as u32, ys.len() as u32, 3);
        let coords = GridCoords::from_lines(xs, ys);
        let seeds = [(0, 0, 0), (3, 1, 0), (2, 2, 1), (4, 3, 2)];

        for clearance in [0.0, 0.14, 0.15, 0.6, 1.75, 10.0] {
            let mut builder = GridBuilder::new(d, 7);
            for &(x, y, l) in &seeds {
                builder.mark_cell_layer(x, y, l);
            }
            builder.inflate_clearance(clearance, &coords);
            let grid = builder.build();

            for l in 0..d.layers {
                for y in 0..d.h {
                    for x in 0..d.w {
                        let expected = seeds.iter().any(|&(sx, sy, sl)| {
                            sl == l
                                && (coords.x_of(x) - coords.x_of(sx)).abs() <= clearance
                                && (coords.y_of(y) - coords.y_of(sy)).abs() <= clearance
                        });
                        assert_eq!(
                            grid.is_obstacle(d.idx3(x, y, l)),
                            expected,
                            "clearance={clearance}, cell=({x},{y},{l})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rectangle_corner_order_clipping_and_clear_are_consistent() {
        let d = Dims::with_layers(4, 3, 2);
        let forward = GridBuilder::new(d, 1)
            .mark_rect_layer(1, 1, 99, 99, 1)
            .build();
        let reverse = GridBuilder::new(d, 1)
            .mark_rect_layer(99, 99, 1, 1, 1)
            .build();
        assert_eq!(forward, reverse);
        assert_eq!(
            forward.cost.iter().filter(|&&c| c == OBSTACLE).count(),
            3 * 2
        );

        let cleared = GridBuilder::new(d, 1)
            .mark_cell_layer(2, 1, 1)
            .clear_cell_layer(2, 1, 1)
            .mark_cell_layer(500, 500, 500)
            .clear_cell_layer(500, 500, 500)
            .build();
        assert!(cleared.cost.iter().all(|&c| c == 1));
    }

    #[test]
    fn inflation_is_monotone_in_clearance() {
        let d = Dims::new(6, 5);
        let coords = GridCoords::from_lines(
            vec![0.0, 0.2, 0.9, 1.0, 2.8, 5.0],
            vec![-1.0, 0.0, 0.1, 1.5, 4.0],
        );
        let build = |clearance| {
            GridBuilder::new(d, 1)
                .mark_cell(2, 2)
                .mark_cell(5, 4)
                .inflate_clearance(clearance, &coords)
                .build()
        };
        let small = build(0.25);
        let large = build(1.0);
        for i in 0..d.len() as u32 {
            assert!(!small.is_obstacle(i) || large.is_obstacle(i));
        }
    }
}
