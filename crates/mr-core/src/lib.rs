//! `mr-core` — the metalroute contract crate.
//!
//! This crate holds ONLY shared data types, the [`Router`] trait, the canonical
//! row-major coordinate mapping ([`Dims::idx`] / [`Dims::xy`]), the deterministic
//! [`TieBreak`] rule, and [`RouterError`]. It contains no routing logic.
//!
//! Everything else in the workspace depends on this crate, so its surface is kept
//! deliberately small and stable. Two invariants are load-bearing across crates:
//!
//! 1. **One coordinate mapping.** Cells are addressed by a single row-major
//!    [`CellIdx`] via [`Dims`]. No crate may define its own mapping (see plan R3).
//! 2. **One tie-break.** When several equal-cost expansions compete, every router
//!    (CPU and GPU) resolves the tie identically per [`TieBreak`] (see plan R2).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cost of stepping onto a cell. [`OBSTACLE`] marks an impassable cell.
pub type Cost = u32;

/// A cell address into a row-major grid. `y * width + x`.
pub type CellIdx = u32;

/// Sentinel cost marking an impassable cell.
pub const OBSTACLE: Cost = Cost::MAX;

/// Grid dimensions plus the *only* sanctioned cell ↔ coordinate mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dims {
    pub w: u32,
    pub h: u32,
}

impl Dims {
    pub fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.w as usize * self.h as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Canonical row-major index of `(x, y)`. Caller guarantees in-bounds.
    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> CellIdx {
        y * self.w + x
    }

    /// Inverse of [`Dims::idx`]: `(x, y)` of a cell index.
    #[inline]
    pub fn xy(&self, i: CellIdx) -> (u32, u32) {
        (i % self.w, i / self.w)
    }

    pub fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.w && y < self.h
    }

    pub fn contains(&self, i: CellIdx) -> bool {
        (i as usize) < self.len()
    }

    /// 4-connected neighbours of `i`, returned in ascending [`CellIdx`] order so
    /// that iteration order is identical everywhere (anchors the tie-break).
    pub fn neighbors4(&self, i: CellIdx) -> Vec<CellIdx> {
        let (x, y) = self.xy(i);
        let mut v = Vec::with_capacity(4);
        if y > 0 {
            v.push(self.idx(x, y - 1));
        }
        if x > 0 {
            v.push(self.idx(x - 1, y));
        }
        if x + 1 < self.w {
            v.push(self.idx(x + 1, y));
        }
        if y + 1 < self.h {
            v.push(self.idx(x, y + 1));
        }
        v.sort_unstable();
        v
    }
}

/// A cost grid: row-major `cost`, indexed by [`CellIdx`]. `OBSTACLE` = blocked.
///
/// This is the canonical board representation passed to every [`Router`]. The
/// `mr-grid` crate owns the *construction* of grids (rasterisation, clearance
/// inflation); this type is just the shared data so the trait can name it without
/// creating a dependency cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid {
    pub dims: Dims,
    pub cost: Vec<Cost>,
}

impl Grid {
    /// A grid of `dims` with every cell initialised to `fill`.
    pub fn filled(dims: Dims, fill: Cost) -> Self {
        Self {
            cost: vec![fill; dims.len()],
            dims,
        }
    }

    #[inline]
    pub fn cost_at(&self, i: CellIdx) -> Cost {
        self.cost[i as usize]
    }

    #[inline]
    pub fn is_obstacle(&self, i: CellIdx) -> bool {
        self.cost[i as usize] == OBSTACLE
    }

    #[inline]
    pub fn set(&mut self, i: CellIdx, c: Cost) {
        self.cost[i as usize] = c;
    }

    /// True when `cost.len()` matches `dims`.
    pub fn is_well_formed(&self) -> bool {
        self.cost.len() == self.dims.len()
    }
}

/// One net to route: a named source/target pair of cells.
///
/// Multi-terminal nets are decomposed into pairs upstream (see plan R8); the
/// router contract is strictly two-point.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NetEndpoints {
    pub net: String,
    pub src: CellIdx,
    pub dst: CellIdx,
    /// Cells this net is permitted to traverse even though they are obstacles in
    /// the base grid — namely this net's *own* pad cells. In the base grid ALL
    /// pads are obstacles; a router unmasks each net's `passable_pads` in its
    /// per-net working grid so the net can escape its own pads but cannot run
    /// through a foreign net's pad. Defaults to empty (no pads), which preserves
    /// behaviour for every construction site that does not set it.
    #[serde(default)]
    pub passable_pads: Vec<CellIdx>,
}

/// A routed net: the ordered path of cells from src to dst and its total cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteResult {
    pub net: String,
    pub path: Vec<CellIdx>,
    pub cost: Cost,
}

/// The full result of routing a board.
///
/// `congestion` is per-cell (length == `dims.len()`): how many routed nets occupy
/// each cell. Two results are considered equal by the oracle when total path cost
/// and `congestion` match — not when paths are bit-identical (ties exist).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardRoute {
    pub results: Vec<RouteResult>,
    pub unrouted: Vec<String>,
    pub congestion: Vec<u32>,
}

impl BoardRoute {
    /// Sum of every routed net's cost — the headline number the oracle compares.
    pub fn total_cost(&self) -> u64 {
        self.results.iter().map(|r| r.cost as u64).sum()
    }

    /// Build the per-cell congestion vector from a set of routed paths.
    pub fn congestion_from(dims: Dims, results: &[RouteResult]) -> Vec<u32> {
        let mut c = vec![0u32; dims.len()];
        for r in results {
            for &cell in &r.path {
                c[cell as usize] += 1;
            }
        }
        c
    }
}

/// The deterministic tie-break shared by every router so CPU and GPU agree.
///
/// A parallel prefix-min does NOT preserve a sequential tie-break for free
/// (plan M0/R2): implementations must *demonstrate* they reproduce this rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TieBreak {
    /// Among equal-cost choices, prefer the one with the lower [`CellIdx`].
    #[default]
    LowerCellIdx,
}

/// Errors a [`Router`] may surface. Per-net "no path" during multi-net routing is
/// reported via [`BoardRoute::unrouted`], not as an error; these variants are for
/// contract violations and unrecoverable conditions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RouterError {
    #[error("net `{net}`: endpoint out of bounds or on an obstacle")]
    InvalidEndpoint { net: String },
    #[error("grid is malformed: cost length does not match dims")]
    MalformedGrid,
    #[error("rip-up exhausted after {passes} passes ({unrouted} nets unrouted)")]
    RipUpExhausted { passes: u32, unrouted: usize },
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
}

/// The single seam every routing implementation (M1/M2 CPU, M3/M4 Metal) shares,
/// so they are drop-in swappable behind benchmarks, the CLI, and the HTTP server.
pub trait Router {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R3 guard: the canonical mapping round-trips for every cell of several grids.
    #[test]
    fn idx_xy_roundtrip() {
        for &(w, h) in &[(1u32, 1u32), (1, 7), (7, 1), (32, 32), (13, 5), (5, 13)] {
            let d = Dims::new(w, h);
            for i in 0..d.len() as u32 {
                let (x, y) = d.xy(i);
                assert!(
                    d.in_bounds(x, y),
                    "{i} -> ({x},{y}) out of bounds for {d:?}"
                );
                assert_eq!(d.idx(x, y), i, "idx∘xy mismatch for {d:?} at {i}");
            }
            for y in 0..h {
                for x in 0..w {
                    let i = d.idx(x, y);
                    assert_eq!(d.xy(i), (x, y), "xy∘idx mismatch for {d:?} at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn neighbors_are_ascending_and_in_bounds() {
        let d = Dims::new(4, 4);
        let center = d.idx(2, 2);
        let n = d.neighbors4(center);
        assert_eq!(n, vec![d.idx(2, 1), d.idx(1, 2), d.idx(3, 2), d.idx(2, 3)]);
        assert!(n.windows(2).all(|w| w[0] < w[1]), "neighbours must ascend");
        // corner has exactly two neighbours
        assert_eq!(d.neighbors4(d.idx(0, 0)).len(), 2);
    }

    #[test]
    fn congestion_counts_overlaps() {
        let d = Dims::new(3, 1);
        let results = vec![
            RouteResult {
                net: "a".into(),
                path: vec![0, 1],
                cost: 2,
            },
            RouteResult {
                net: "b".into(),
                path: vec![1, 2],
                cost: 2,
            },
        ];
        assert_eq!(BoardRoute::congestion_from(d, &results), vec![1, 2, 1]);
    }
}
