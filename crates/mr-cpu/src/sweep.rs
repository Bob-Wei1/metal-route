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
//! *path reconstruction*, not of the converged costs. We reconstruct a path by
//! greedy descent from the target, at each step choosing the valid predecessor
//! (a neighbour `p` with `dist[p] + cost(cur) == dist[cur]`) with the lowest
//! [`CellIdx`].
//!
//! On `tie_break_2x2` this greedy descent **does** reproduce the BFS path
//! `[0, 1, 3]` — see [`tests::sweep_tie_break_reproduces_path`]. However this is a
//! *backward* lowest-index rule and is NOT the same mechanism as the forward
//! lowest-index successor rule the Dijkstra router uses. They coincide here only
//! because the 2×2 case is symmetric. In general a backward greedy descent over a
//! parallel field is **not guaranteed** to reproduce the forward sequential
//! tie-break path (it can pick a different but equal-cost path). The robust fix
//! for GPU/CPU path agreement is to carry the tie-break into the field itself —
//! e.g. break distance ties by storing, per cell, the lowest-index predecessor
//! during the sweep (a parent field updated under the same `(dist, idx)` ordering)
//! rather than recovering it afterwards. This is the load-bearing M0 caveat for
//! R2: the cost field parallelises trivially; the canonical *path* does not, and
//! must be made explicit.

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
    if dims.is_empty() || grid.is_obstacle(src) {
        return dist;
    }
    dist[src as usize] = 0;

    loop {
        let mut changed = false;

        // Row sweeps: each row is an independent line (parallelisable across rows).
        for y in 0..h {
            // left -> right
            for x in 1..w {
                let cur = dims.idx(x, y);
                let prev = dims.idx(x - 1, y);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
            // right -> left
            for x in (0..w.saturating_sub(1)).rev() {
                let cur = dims.idx(x, y);
                let prev = dims.idx(x + 1, y);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
        }

        // Column sweeps: each column is an independent line (parallelisable).
        for x in 0..w {
            // up -> down
            for y in 1..h {
                let cur = dims.idx(x, y);
                let prev = dims.idx(x, y - 1);
                let cand = relax(grid, cur, dist[prev as usize]);
                if cand < dist[cur as usize] {
                    dist[cur as usize] = cand;
                    changed = true;
                }
            }
            // down -> up
            for y in (0..h.saturating_sub(1)).rev() {
                let cur = dims.idx(x, y);
                let prev = dims.idx(x, y + 1);
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
    dijkstra(grid, src, |_| 0).dist
}

/// Reconstruct a path `src..=dst` from a converged distance field by greedy
/// descent, breaking ties by lowest [`CellIdx`] predecessor. Returns `None` when
/// `dst` is unreachable. Used by the M0 tie-break test.
pub fn path_from_field(
    grid: &Grid,
    dist: &[Cost],
    src: CellIdx,
    dst: CellIdx,
) -> Option<Vec<CellIdx>> {
    if dist[dst as usize] == Cost::MAX {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        // A valid predecessor p satisfies dist[p] + cost(cur) == dist[cur].
        // neighbors4 is ascending, so the first match is the lowest-index one.
        let need = dist[cur as usize];
        let step = grid.cost_at(cur);
        let mut next = None;
        for p in grid.dims.neighbors4(cur) {
            let dp = dist[p as usize];
            if dp != Cost::MAX && dp.saturating_add(step) == need {
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
    use mr_fixtures::{obstacle_battery, tie_break_2x2};

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
}
