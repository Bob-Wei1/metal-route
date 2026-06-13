//! `mr-fixtures` — the single source of shared routing test cases.
//!
//! Both the CPU routers (M0/M1/M2) and the Metal validation (M3/M4) are graded
//! against the *same* fixtures defined here, so "CPU == GPU" is a comparison over
//! one battery. A fixture optionally pins:
//!
//! * `expected_total_cost` — a hand-computed shortest-path cost, and/or
//! * `expected_path` — the exact path of `nets[0]` under [`TieBreak::LowerCellIdx`],
//!   used to prove the tie-break is reproduced (plan R2).
//!
//! The "golden-grid format" is a tiny ASCII grid ([`parse_ascii`]):
//! `.` open, `#` obstacle, `S` source, `T` target.

use mr_core::{CellIdx, Dims, Grid, NetEndpoints};
use mr_grid::GridBuilder;

/// A self-describing routing test case.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: &'static str,
    pub grid: Grid,
    pub nets: Vec<NetEndpoints>,
    /// Hand-computed total shortest-path cost, when known.
    pub expected_total_cost: Option<u64>,
    /// Exact expected path of `nets[0]` under the shared tie-break, when pinned.
    pub expected_path: Option<Vec<CellIdx>>,
}

/// Parse the ASCII golden-grid format into a single-net [`Fixture`].
///
/// Lines are rows (top to bottom = y ascending); the first line sets the width.
/// `S` marks the source, `T` the target, `#` an obstacle, `.`/space open.
///
/// ```
/// let f = mr_fixtures::parse_ascii("ascii", "S.#\n..T", None);
/// assert_eq!(f.grid.dims, mr_core::Dims::new(3, 2));
/// assert_eq!(f.nets[0].src, 0);
/// assert_eq!(f.nets[0].dst, 5);
/// ```
pub fn parse_ascii(name: &'static str, s: &str, expected_total_cost: Option<u64>) -> Fixture {
    let rows: Vec<&str> = s.lines().filter(|l| !l.is_empty()).collect();
    let h = rows.len() as u32;
    let w = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0) as u32;
    let dims = Dims::new(w, h);
    let mut b = GridBuilder::new(dims, 1);
    let mut src = None;
    let mut dst = None;
    for (y, row) in rows.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            let (x, y) = (x as u32, y as u32);
            match ch {
                '#' => {
                    b.mark_cell(x, y);
                }
                'S' => src = Some(dims.idx(x, y)),
                'T' => dst = Some(dims.idx(x, y)),
                _ => {}
            }
        }
    }
    let nets = match (src, dst) {
        (Some(src), Some(dst)) => vec![NetEndpoints {
            net: "n0".into(),
            src,
            dst,
        }],
        _ => Vec::new(),
    };
    Fixture {
        name,
        grid: b.build(),
        nets,
        expected_total_cost,
        expected_path: None,
    }
}

/// M1 hand case: 32×32, one net top-left → top-right, blocked by a vertical wall
/// at column 15 (rows 0..=30) with a single gap at the bottom (y=31).
///
/// The only crossing is `(15, 31)`, so any path must descend to y=31, cross, and
/// climb back to y=0. Hand-computed minimum cost (4-connected, unit cost):
/// horizontal 31 + descend 31 + ascend 31 = **93** moves.
pub fn hand_32x32_wall() -> Fixture {
    let dims = Dims::new(32, 32);
    let mut b = GridBuilder::new(dims, 1);
    b.mark_rect(15, 0, 15, 30); // wall with a gap at y=31
    Fixture {
        name: "hand_32x32_wall",
        grid: b.build(),
        nets: vec![NetEndpoints {
            net: "n0".into(),
            src: dims.idx(0, 0),
            dst: dims.idx(31, 0),
        }],
        expected_total_cost: Some(93),
        expected_path: None,
    }
}

/// Open 2×2 with a genuine tie: `(0,0) → (1,1)` has two equal-cost paths. Under
/// [`mr_core::TieBreak::LowerCellIdx`] the first step is to the lower-index
/// neighbour `(1,0)=1`, pinning the path to `[0, 1, 3]`.
pub fn tie_break_2x2() -> Fixture {
    let dims = Dims::new(2, 2);
    let grid = GridBuilder::new(dims, 1).build();
    Fixture {
        name: "tie_break_2x2",
        grid,
        nets: vec![NetEndpoints {
            net: "n0".into(),
            src: dims.idx(0, 0),
            dst: dims.idx(1, 1),
        }],
        expected_total_cost: Some(2),
        expected_path: Some(vec![dims.idx(0, 0), dims.idx(1, 0), dims.idx(1, 1)]),
    }
}

/// The M0 battery: assorted small obstacle/clearance grids on which the separable
/// H/V prefix-min sweep must reproduce Lee's BFS distances. Costs are pinned where
/// hand-computable; the rest are validated by sweep-vs-BFS agreement in `mr-cpu`.
pub fn obstacle_battery() -> Vec<Fixture> {
    vec![
        parse_ascii("open_5x5", "S....\n.....\n.....\n.....\n....T", Some(8)),
        parse_ascii("notch", "S.#..\n..#..\n..#..\n..#..\n....T", Some(8)),
        parse_ascii("spiral", "S....\n####.\n...#.\n.#.#.\n.#..T", None),
        parse_ascii("corridor", "S#...\n.#.#.\n.#.#.\n.#.#.\n...#T", None),
        parse_ascii("blocked_gap", "S....\n.###.\n.#T#.\n.###.\n.....", None),
        hand_32x32_wall(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_parses_endpoints_and_obstacles() {
        let f = parse_ascii("t", "S.#\n..T", None);
        assert_eq!(f.grid.dims, Dims::new(3, 2));
        assert_eq!(f.nets[0].src, 0);
        assert_eq!(f.nets[0].dst, 5);
        assert!(f.grid.is_obstacle(f.grid.dims.idx(2, 0)));
        assert!(f.grid.is_well_formed());
    }

    #[test]
    fn hand_case_geometry_is_correct() {
        let f = hand_32x32_wall();
        assert_eq!(f.expected_total_cost, Some(93));
        let d = f.grid.dims;
        // wall present at (15, 0..=30), gap open at (15, 31)
        assert!(f.grid.is_obstacle(d.idx(15, 0)));
        assert!(f.grid.is_obstacle(d.idx(15, 30)));
        assert!(!f.grid.is_obstacle(d.idx(15, 31)));
        // endpoints are passable
        assert_ne!(f.grid.cost_at(f.nets[0].src), mr_core::OBSTACLE);
        assert_ne!(f.grid.cost_at(f.nets[0].dst), mr_core::OBSTACLE);
    }

    #[test]
    fn tie_break_fixture_pins_lower_index_path() {
        let f = tie_break_2x2();
        assert_eq!(f.expected_path, Some(vec![0, 1, 3]));
    }

    #[test]
    fn battery_is_nonempty_and_well_formed() {
        let battery = obstacle_battery();
        assert!(battery.len() >= 5);
        for f in &battery {
            assert!(f.grid.is_well_formed(), "{}", f.name);
            assert_eq!(f.nets.len(), 1, "{} should be single-net", f.name);
        }
    }
}
