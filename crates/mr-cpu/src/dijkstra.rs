//! Shared Dijkstra single-source machinery used by [`crate::LeeRouter`],
//! [`crate::AStarRouter`], and the BFS reference field in [`crate::sweep`].
//!
//! All routers in this crate share one deterministic tie-break
//! ([`mr_core::TieBreak::LowerCellIdx`]): equal-cost expansions are resolved in
//! favour of the lower [`CellIdx`]. Two mechanisms enforce it:
//!
//! 1. The priority queue is keyed by `(dist, cell)` so that among entries of equal
//!    distance the lower cell index is popped first.
//! 2. Predecessors are **first-writer-wins**: once a cell has a predecessor it is
//!    never overwritten, even by a later equal-distance relaxation. Because the
//!    first writer is always reached along the lowest-index frontier, the
//!    reconstructed path is the canonical one.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use mr_core::{CellIdx, Cost, Grid};

/// Sentinel for "no predecessor / unreachable".
pub(crate) const NO_PRED: CellIdx = CellIdx::MAX;

/// Result of a single-source Dijkstra expansion.
pub(crate) struct DijkstraField {
    /// `dist[i]` is the cheapest cost to reach cell `i` (sum of `cost_at` over the
    /// path excluding the source). `Cost::MAX` for unreachable cells.
    pub dist: Vec<Cost>,
    /// First-writer-wins predecessor of each cell; [`NO_PRED`] when none.
    pub pred: Vec<CellIdx>,
}

/// Run Dijkstra from `src` over `grid`, honouring per-cell costs and the
/// [`mr_core::TieBreak::LowerCellIdx`] rule. Obstacles are impassable.
///
/// `heuristic` lets callers turn this into A*: it must be an admissible,
/// consistent estimate of the remaining cost from a cell to the goal (return `0`
/// for plain Dijkstra/Lee). The priority is `dist + heuristic(cell)`.
pub(crate) fn dijkstra<H>(grid: &Grid, src: CellIdx, heuristic: H) -> DijkstraField
where
    H: Fn(CellIdx) -> Cost,
{
    let n = grid.dims.len();
    let mut dist = vec![Cost::MAX; n];
    let mut pred = vec![NO_PRED; n];

    // Source must be passable to start; an obstacle source yields an empty field.
    if grid.is_obstacle(src) {
        return DijkstraField { dist, pred };
    }

    dist[src as usize] = 0;
    // Heap entries: (Reverse(priority), Reverse(cell)) so we pop the smallest
    // priority first, and among ties the smallest cell index first.
    let mut heap: BinaryHeap<(Reverse<Cost>, Reverse<CellIdx>)> = BinaryHeap::new();
    heap.push((Reverse(heuristic(src)), Reverse(src)));

    while let Some((Reverse(prio), Reverse(u))) = heap.pop() {
        let du = dist[u as usize];
        // Stale entry: a better distance was already committed.
        if prio.saturating_sub(heuristic(u)) > du {
            continue;
        }
        for v in grid.dims.neighbors4(u) {
            if grid.is_obstacle(v) {
                continue;
            }
            let step = grid.cost_at(v);
            let nd = du.saturating_add(step);
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                // First-writer-wins: only set predecessor on a strict improvement.
                pred[v as usize] = u;
                let np = nd.saturating_add(heuristic(v));
                heap.push((Reverse(np), Reverse(v)));
            }
        }
    }

    DijkstraField { dist, pred }
}

/// Reconstruct the path `src..=dst` from a predecessor array, or `None` when
/// `dst` is unreachable. The returned path starts at `src` and ends at `dst`.
pub(crate) fn reconstruct_path(
    pred: &[CellIdx],
    src: CellIdx,
    dst: CellIdx,
    dist: &[Cost],
) -> Option<Vec<CellIdx>> {
    if dist[dst as usize] == Cost::MAX {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        let p = pred[cur as usize];
        if p == NO_PRED {
            // src unreachable from dst chain (shouldn't happen if dist finite).
            return None;
        }
        path.push(p);
        cur = p;
    }
    path.reverse();
    Some(path)
}
