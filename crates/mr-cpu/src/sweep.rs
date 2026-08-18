//! `sweep` (A2 / M0) — the de-risk spike for a GPU-friendly, separable
//! prefix-min distance field, validated against a sequential Dijkstra field.
//!
//! # The separable sweep
//!
//! Instead of a frontier-ordered wavefront (BFS/Dijkstra), this computes a
//! single-source distance field by repeating *separable* 1-D prefix-min passes:
//!
//! * a row sweep left→right then right→left, and
//! * a column sweep up→down then down→up,
//!
//! each performing the relaxation `dist[n] = min(dist[n], dist[prev] + cost(n))`
//! along its line. The four passes are iterated until a full round changes
//! nothing. Each pass is embarrassingly parallel across the *independent* lines
//! (rows for a row sweep, columns for a column sweep), which is exactly the shape
//! that maps onto a GPU — hence M0.
//!
//! Because every pass only ever lowers a `dist` value toward a valid neighbour
//! cost and the grid is 4-connected with non-negative costs, the iteration is a
//! monotone label-correcting method that converges to the true single-source
//! shortest-path distances — identical to the Dijkstra field. The tests assert
//! this exact equality on the whole obstacle battery.
//!
//! # M0 tie-break finding
//!
//! A distance field alone does **not** carry a tie-break: ties are a property of
//! *path reconstruction*, not of the converged costs. We first compute minimum hop
//! counts over the graph of shortest-cost edges, then descend from the target by
//! `(cost, hop_count, lower predecessor)`. The hop key makes zero-cost plateaus
//! acyclic while the lower [`CellIdx`] predecessor pins deterministic ties.
//!
//! On `tie_break_2x2` this greedy descent reproduces the BFS path `[0, 1, 3]` —
//! see `tests::sweep_tie_break_reproduces_path`. More generally, every valid
//! predecessor of a cell has the same shortest distance and one fewer canonical
//! hop, so backward lowest-index selection matches the CPU shortest-path label.
//! The exhaustive 3×3 mask test pins this equivalence for every source/target pair.

use std::collections::VecDeque;

use mr_core::{CellIdx, Cost, Grid};

use crate::dijkstra::dijkstra;

/// Relax one step: returns the candidate distance for cell `n` arriving from a
/// neighbour with distance `prev_dist`, or `Cost::MAX` if either side is blocked /
/// unreachable.
#[inline]
fn relax(grid: &Grid, n: CellIdx, prev_dist: Cost) -> Cost {
    if prev_dist == Cost::MAX || grid.is_obstacle(n) {
        return Cost::MAX;
    }
    prev_dist.saturating_add(grid.cost_at(n))
}

/// Single-source distance field via the separable H/V prefix-min sweep.
///
/// Returns `dist` indexed by [`CellIdx`]; unreachable cells (and an obstacle
/// source) are `Cost::MAX`. Converges to the same values as
/// [`bfs_distance_field`].
pub fn sweep_distance_field(grid: &Grid, src: CellIdx) -> Vec<Cost> {
    let dims = grid.dims;
    let (w, h) = (dims.w, dims.h);
    let mut dist = vec![Cost::MAX; dims.len()];
    if dims.is_empty() || !dims.contains(src) || grid.is_obstacle(src) {
        return dist;
    }
    dist[src as usize] = 0;
    // Four-neighbour routing never crosses layers, so only the source plane can
    // become reachable.  Selecting it explicitly is both the multilayer contract
    // and avoids doing identical no-op work on every other plane.
    let layer = dims.layer_of(src);

    loop {
        let mut changed = false;

        // Row sweeps: each row is an independent line.
        for y in 0..h {
            // left -> right
            for x in 1..w {
                let cur = dims.idx3(x, y, layer);
                let prev = dims.idx3(x - 1, y, layer);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
            // right -> left
            for x in (0..w.saturating_sub(1)).rev() {
                let cur = dims.idx3(x, y, layer);
                let prev = dims.idx3(x + 1, y, layer);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
        }

        // Column sweeps: each column is an independent line.
        for x in 0..w {
            // up -> down
            for y in 1..h {
                let cur = dims.idx3(x, y, layer);
                let prev = dims.idx3(x, y - 1, layer);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
            // down -> up
            for y in (0..h.saturating_sub(1)).rev() {
                let cur = dims.idx3(x, y, layer);
                let prev = dims.idx3(x, y + 1, layer);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    dist
}

/// Single-source distance field via Dijkstra — the reference the sweep is graded
/// against. Unreachable cells are `Cost::MAX`.
pub fn bfs_distance_field(grid: &Grid, src: CellIdx) -> Vec<Cost> {
    if !grid.dims.contains(src) {
        return vec![Cost::MAX; grid.dims.len()];
    }
    dijkstra(grid, src, |_| 0).dist
}

/// Reconstruct the canonical path `src..=dst` from a converged distance field.
///
/// Zero-cost edges mean cost alone does not decrease while walking backwards and
/// naïve lowest-index descent can cycle. We therefore BFS the graph of
/// shortest-cost edges to compute the minimum hop count for every cell, then walk
/// backwards by `(cost, hop_count, lower predecessor)`. Hop count decreases at
/// every step, making reconstruction cycle-proof. Returns `None` when `dst` is
/// unreachable or the supplied field is inconsistent with the grid.
pub fn path_from_field(
    grid: &Grid,
    dist: &[Cost],
    src: CellIdx,
    dst: CellIdx,
) -> Option<Vec<CellIdx>> {
    if dist.len() != grid.dims.len() || !grid.dims.contains(src) || !grid.dims.contains(dst) {
        return None;
    }
    if dist[dst as usize] == Cost::MAX {
        return None;
    }

    // Minimum hops over only edges that preserve the supplied shortest-cost
    // labels. This converts a possibly cyclic zero-cost shortest-path graph into
    // an acyclic predecessor relation without changing primary path cost.
    let mut hops = vec![u32::MAX; grid.dims.len()];
    let mut queue = VecDeque::new();
    hops[src as usize] = 0;
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        let hu = hops[u as usize];
        let du = dist[u as usize];
        if du == Cost::MAX {
            continue;
        }
        for v in grid.dims.neighbors4(u) {
            if grid.is_obstacle(v)
                || du.saturating_add(grid.cost_at(v)) != dist[v as usize]
                || hops[v as usize] != u32::MAX
            {
                continue;
            }
            hops[v as usize] = hu.saturating_add(1);
            queue.push_back(v);
        }
    }
    if hops[dst as usize] == u32::MAX {
        return None;
    }

    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        // A valid canonical predecessor preserves cost and has exactly one fewer
        // hop. `neighbors4` is ascending, so the first match is the lowest index.
        let need = dist[cur as usize];
        let step = grid.cost_at(cur);
        let need_hops = hops[cur as usize].checked_sub(1)?;
        let mut next = None;
        for p in grid.dims.neighbors4(cur) {
            let dp = dist[p as usize];
            if dp != Cost::MAX && dp.saturating_add(step) == need && hops[p as usize] == need_hops {
                next = Some(p);
                break;
            }
        }
        let p = next?;
        path.push(p);
        cur = p;
    }
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::Dims;
    use mr_fixtures::{obstacle_battery, tie_break_2x2};
    use mr_grid::GridBuilder;

    #[test]
    fn sweep_field_equals_bfs_field_on_battery() {
        for f in obstacle_battery() {
            let src = f.nets[0].src;
            let sweep = sweep_distance_field(&f.grid, src);
            let bfs = bfs_distance_field(&f.grid, src);
            assert_eq!(sweep.len(), bfs.len(), "{}", f.name);
            for i in 0..sweep.len() {
                assert_eq!(
                    sweep[i], bfs[i],
                    "{}: cell {i} sweep={} bfs={} (Cost::MAX==unreachable)",
                    f.name, sweep[i], bfs[i]
                );
            }
        }
    }

    /// M0 finding: greedy descent over the sweep field reproduces the BFS
    /// tie-break path on the symmetric 2×2 case. See the module docs for the
    /// general caveat — this is a *backward* lowest-index rule and is not the same
    /// mechanism as the forward sequential tie-break.
    #[test]
    fn sweep_tie_break_reproduces_path() {
        let f = tie_break_2x2();
        let src = f.nets[0].src;
        let dst = f.nets[0].dst;
        let dist = sweep_distance_field(&f.grid, src);
        let path = path_from_field(&f.grid, &dist, src, dst).unwrap();
        assert_eq!(
            path,
            vec![0, 1, 3],
            "greedy descent reproduces BFS tie-break path on 2x2"
        );
        assert_eq!(Some(path), f.expected_path);
    }

    #[test]
    fn sweep_matches_bfs_from_every_layer() {
        let dims = Dims::with_layers(4, 3, 3);
        let mut builder = GridBuilder::new(dims, 1);
        builder.mark_cell_layer(1, 1, 0);
        builder.mark_cell_layer(2, 0, 1);
        builder.mark_cell_layer(0, 2, 2);
        let grid = builder.build();
        for src in 0..dims.len() as u32 {
            if grid.is_obstacle(src) {
                continue;
            }
            assert_eq!(
                sweep_distance_field(&grid, src),
                bfs_distance_field(&grid, src),
                "source {src} on layer {}",
                dims.layer_of(src)
            );
        }
    }

    #[test]
    fn sweep_and_greedy_path_match_lee_exhaustively_on_3x3_masks() {
        let dims = Dims::new(3, 3);
        for mask in 0usize..(1usize << dims.len()) {
            let mut builder = GridBuilder::new(dims, 1);
            for c in 0..dims.len() as u32 {
                if mask & (1 << c) != 0 {
                    let (x, y) = dims.xy(c);
                    builder.mark_cell(x, y);
                }
            }
            let grid = builder.build();
            for src in 0..dims.len() as u32 {
                if grid.is_obstacle(src) {
                    continue;
                }
                let sweep = sweep_distance_field(&grid, src);
                let bfs = bfs_distance_field(&grid, src);
                assert_eq!(sweep, bfs, "mask={mask:#011b} src={src}");
                for dst in 0..dims.len() as u32 {
                    if grid.is_obstacle(dst) {
                        continue;
                    }
                    let got = path_from_field(&grid, &sweep, src, dst);
                    let lee = crate::LeeRouter::route_one(&grid, src, dst).map(|x| x.0);
                    assert_eq!(got, lee, "mask={mask:#011b} src={src} dst={dst}");
                }
            }
        }
    }

    #[test]
    fn sweep_matches_bfs_on_weighted_and_zero_cost_cells() {
        let dims = Dims::new(5, 4);
        let mut grid = GridBuilder::new(dims, 1).build();
        for (cell, cost) in [(1, 7), (2, 0), (7, 3), (13, 11), (18, 0)] {
            grid.set(cell, cost);
        }
        grid.set(8, mr_core::OBSTACLE);
        for src in 0..dims.len() as u32 {
            if !grid.is_obstacle(src) {
                assert_eq!(
                    sweep_distance_field(&grid, src),
                    bfs_distance_field(&grid, src)
                );
            }
        }
    }

    #[test]
    fn path_from_field_is_cycle_free_on_zero_cost_plateau() {
        // Regression: cost-only lowest-index descent looped 3 -> 0 -> 1 -> 0.
        let dims = Dims::new(3, 2);
        let grid = Grid::filled(dims, 0);
        let src = 5;
        let dst = 3;
        let dist = sweep_distance_field(&grid, src);
        let expected = vec![5, 4, 3];
        for _ in 0..8 {
            assert_eq!(
                path_from_field(&grid, &dist, src, dst),
                Some(expected.clone())
            );
        }
        assert_eq!(
            path_from_field(&grid, &dist, src, dst),
            crate::LeeRouter::route_one(&grid, src, dst).map(|(path, _)| path)
        );
    }

    #[test]
    fn public_field_helpers_handle_out_of_range_source() {
        let dims = Dims::new(3, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let expected = vec![Cost::MAX; dims.len()];
        assert_eq!(sweep_distance_field(&grid, 99), expected);
        assert_eq!(bfs_distance_field(&grid, 99), expected);
        assert_eq!(path_from_field(&grid, &[0], 0, 1), None);
    }
}
