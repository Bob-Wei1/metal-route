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

use mr_core::{CellIdx, Cost, Dims, Grid};

/// Sentinel for "no predecessor / unreachable".
pub(crate) const NO_PRED: CellIdx = CellIdx::MAX;

/// Fixed-point scale converting a continuous (mm) geometric length into the integer
/// [`Cost`] units the search adds up. One unit of geometric length costs
/// `COST_SCALE`; a uniform unit-spaced grid (see [`mr_core::GridCoords::uniform`])
/// therefore prices each planar step at exactly `COST_SCALE`.
///
/// Chosen equal to the negotiated router's base [`crate::SCALE`] (`16`) so that on a
/// uniform grid a planar step still costs `16` — the congestion penalties layered in
/// `negotiated.rs` keep their existing relative strength and the default router stays
/// byte-identical. Large enough that distinct geometric lengths round to distinct
/// integers at typical board pitches; small enough that long paths stay well below
/// the `u32` ceiling ([`saturating_add`](u32::saturating_add) guards the rare
/// overflow regardless).
pub const COST_SCALE: f64 = 16.0;

/// Round a continuous geometric `length` to fixed-point [`Cost`] units
/// (`round(length * COST_SCALE)`). Deterministic (round-half-away-from-zero via
/// [`f64::round`]) and saturating at the `u32` range so a degenerate/huge length can
/// never wrap. Negative or non-finite lengths floor to `0`.
#[inline]
pub(crate) fn edge_cost(length: f64) -> Cost {
    let scaled = (length * COST_SCALE).round();
    if !scaled.is_finite() || scaled <= 0.0 {
        0
    } else if scaled >= Cost::MAX as f64 {
        Cost::MAX
    } else {
        scaled as Cost
    }
}

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

/// Reusable A* search workspace sized once to the board's cell count.
///
/// The key trick is the per-cell `stamp`: a cell's `dist`/`pred` entries are only
/// valid when `stamp[c] == gen`. Bumping `gen` invalidates *every* cell in O(1)
/// (stale entries read as `dist = Cost::MAX` / `pred = NO_PRED`), so a fresh search
/// never pays an O(n) memset — it only touches the cells it actually explores. This
/// is the core that lets [`crate::NegotiatedRouter`] run thousands of windowed
/// searches over a 500k-cell board without re-clearing the board each time.
pub(crate) struct SearchBuf {
    dist: Vec<Cost>,
    pred: Vec<CellIdx>,
    stamp: Vec<u32>,
    gen: u32,
}

impl SearchBuf {
    /// Allocate a workspace for a grid of `n` cells. Allocated once and reused.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            dist: vec![Cost::MAX; n],
            pred: vec![NO_PRED; n],
            stamp: vec![0; n],
            gen: 0,
        }
    }

    /// Begin a fresh search: invalidate all prior per-cell state in O(1) by bumping
    /// the generation. On the rare `u32` wraparound, clear the stamps so no stale
    /// entry can masquerade as current.
    #[inline]
    fn begin(&mut self) {
        match self.gen.checked_add(1) {
            Some(g) => self.gen = g,
            None => {
                self.stamp.iter_mut().for_each(|s| *s = 0);
                self.gen = 1;
            }
        }
    }

    /// Current distance of `c`, treating stale (different-generation) cells as
    /// unreached (`Cost::MAX`).
    #[inline]
    fn dist_of(&self, c: CellIdx) -> Cost {
        if self.stamp[c as usize] == self.gen {
            self.dist[c as usize]
        } else {
            Cost::MAX
        }
    }

    /// Record an improved distance/predecessor for `c`, marking it current.
    #[inline]
    fn set(&mut self, c: CellIdx, d: Cost, p: CellIdx) {
        let ci = c as usize;
        self.dist[ci] = d;
        self.pred[ci] = p;
        self.stamp[ci] = self.gen;
    }

    /// Predecessor of `c` if it is a current (this-generation) cell, else
    /// [`NO_PRED`].
    #[inline]
    fn pred_of(&self, c: CellIdx) -> CellIdx {
        if self.stamp[c as usize] == self.gen {
            self.pred[c as usize]
        } else {
            NO_PRED
        }
    }
}

/// Allocation-reusing A* over the abstract grid described by closures, never
/// cloning a grid and never doing an O(n) reset. Costs and passability are
/// supplied on the fly:
///
/// - `cost_fn(u, v)` is the cost of the planar EDGE from `u` to its 4-neighbour `v`
///   (the price of the move, added once per step). It is edge-aware so a router can
///   price a step by its real geometric length (`round(len * COST_SCALE)`) on a
///   non-uniform grid, not just by a per-cell constant. On a uniform grid the length
///   is the same for every step, so this reduces to a per-cell enter cost exactly
///   like [`dijkstra`]'s `grid.cost_at(v)`. Mirrors `via_step(u, v)`'s shape.
/// - `blocked_fn(c)` marks `c` impassable. The search never relaxes a blocked
///   neighbour; combined with a per-net window this keeps explored area local.
/// - `heuristic(c)` is an admissible remaining-cost estimate to the goal (return
///   `0` for plain Dijkstra). Priority is `dist + heuristic`.
///
/// Layer changes (vias) are driven by `via_step(u, v)`: for each adjacent-layer
/// cell `v` at the same `(x, y)` as the cell `u` being expanded (see
/// [`mr_core::Dims::via_neighbors`]), it returns `Some(cost)` when a via step
/// `u -> v` is legal and what it costs to take, or `None` when no via may be
/// drilled there. Unlike a planar move the via's price is the returned `cost`
/// itself (NOT `cost_fn(v)`); `blocked_fn(v)` is still honoured first so a via
/// can never land on a foreign pad or an out-of-window cell. On a single-layer
/// grid `via_neighbors` is empty, so the search is byte-identical to the planar
/// case regardless of `via_step`.
///
/// Tie-break matches [`dijkstra`] exactly: the heap is keyed `(Reverse(prio),
/// Reverse(cell))` (lowest priority then lowest cell index first), and
/// predecessors are first-writer-wins on strict improvement. Returns the path
/// `src..=dst` and its summed enter-cost, or `None` when `dst` is unreachable.
/// Work is O(explored), not O(n).
#[allow(clippy::too_many_arguments)]
pub(crate) fn astar_buf<C, B, H, V>(
    buf: &mut SearchBuf,
    dims: Dims,
    src: CellIdx,
    dst: CellIdx,
    cost_fn: C,
    blocked_fn: B,
    heuristic: H,
    via_step: V,
) -> Option<(Vec<CellIdx>, Cost)>
where
    C: Fn(CellIdx, CellIdx) -> Cost,
    B: Fn(CellIdx) -> bool,
    H: Fn(CellIdx) -> Cost,
    V: Fn(CellIdx, CellIdx) -> Option<Cost>,
{
    buf.begin();

    // A blocked source (or destination) can never be entered/reached.
    if blocked_fn(src) {
        return None;
    }

    buf.set(src, 0, NO_PRED);
    let mut heap: BinaryHeap<(Reverse<Cost>, Reverse<CellIdx>)> = BinaryHeap::new();
    heap.push((Reverse(heuristic(src)), Reverse(src)));

    while let Some((Reverse(prio), Reverse(u))) = heap.pop() {
        let du = buf.dist_of(u);
        // Stale entry: a better distance was already committed.
        if prio.saturating_sub(heuristic(u)) > du {
            continue;
        }
        if u == dst {
            break;
        }
        // Inline the 4-neighbour enumeration in ascending CellIdx order (up, left,
        // right, down) to avoid allocating + sorting a Vec on every expansion — the
        // single hottest op in the negotiation loop. The neighbours stay on `u`'s
        // own layer `l` (via `idx3`); on a single-layer grid `l == 0` and `idx3`
        // reduces to the historical `idx`, so this path is byte-identical there.
        let (x, y, l) = dims.xyz(u);
        let mut relax = |v: CellIdx| {
            if blocked_fn(v) {
                return;
            }
            // Edge-aware: price the move `u -> v`, not just entering `v`.
            let step = cost_fn(u, v);
            let nd = du.saturating_add(step);
            if nd < buf.dist_of(v) {
                // First-writer-wins: only set predecessor on a strict improvement.
                buf.set(v, nd, u);
                let np = nd.saturating_add(heuristic(v));
                heap.push((Reverse(np), Reverse(v)));
            }
        };
        if y > 0 {
            relax(dims.idx3(x, y - 1, l));
        }
        if x > 0 {
            relax(dims.idx3(x - 1, y, l));
        }
        if x + 1 < dims.w {
            relax(dims.idx3(x + 1, y, l));
        }
        if y + 1 < dims.h {
            relax(dims.idx3(x, y + 1, l));
        }
        // Via (layer-changing) moves AFTER the four planar ones. `via_neighbors`
        // is empty on a single-layer grid (so this is a no-op there) and otherwise
        // returns the adjacent-layer cells at the same (x, y), lower layer first.
        // A via step is priced by `via_step` itself (its returned cost), not by
        // `cost_fn(v)`, but still honours `blocked_fn(v)` first so it can never
        // land on a foreign pad or an out-of-window cell. Tie-break / first-writer
        // semantics are identical to the planar relax.
        for v in dims.via_neighbors(u) {
            if blocked_fn(v) {
                continue;
            }
            if let Some(step) = via_step(u, v) {
                let nd = du.saturating_add(step);
                if nd < buf.dist_of(v) {
                    buf.set(v, nd, u);
                    let np = nd.saturating_add(heuristic(v));
                    heap.push((Reverse(np), Reverse(v)));
                }
            }
        }
    }

    let total = buf.dist_of(dst);
    if total == Cost::MAX {
        return None;
    }
    // Reconstruct src..=dst from the stamped predecessors.
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        let p = buf.pred_of(cur);
        if p == NO_PRED {
            return None;
        }
        path.push(p);
        cur = p;
    }
    path.reverse();
    Some((path, total))
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

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::Dims;
    use mr_grid::GridBuilder;

    /// `astar_buf` must return the same cost as `dijkstra`+`reconstruct_path`, and
    /// a valid shortest path, across several src/dst pairs on an obstacle grid.
    #[test]
    fn astar_buf_matches_dijkstra_on_obstacle_grid() {
        // 8x8 grid with a few obstacle cells forming a partial wall.
        let dims = Dims::new(8, 8);
        let mut gb = GridBuilder::new(dims, 1);
        for y in 0..6 {
            gb.mark_cell(4, y);
        }
        gb.mark_cell(2, 5);
        gb.mark_cell(6, 2);
        let grid = gb.build();

        let pairs = [
            (dims.idx(0, 0), dims.idx(7, 7)),
            (dims.idx(0, 7), dims.idx(7, 0)),
            (dims.idx(1, 1), dims.idx(6, 6)),
            (dims.idx(3, 0), dims.idx(5, 7)),
            (dims.idx(0, 3), dims.idx(7, 3)),
        ];

        let mut buf = SearchBuf::new(dims.len());
        for &(src, dst) in &pairs {
            // Reference: plain Dijkstra (heuristic 0) + reconstruct.
            let field = dijkstra(&grid, src, |_| 0);
            let ref_path = reconstruct_path(&field.pred, src, dst, &field.dist);

            // Under test: astar_buf with the same cost model. Use an admissible
            // Manhattan heuristic (unit step cost == 1 here).
            let h = |c: CellIdx| {
                let (ax, ay) = dims.xy(c);
                let (bx, by) = dims.xy(dst);
                (ax.abs_diff(bx) + ay.abs_diff(by)) as Cost
            };
            let got = astar_buf(
                &mut buf,
                dims,
                src,
                dst,
                |_u, v| grid.cost_at(v),
                |c| grid.is_obstacle(c),
                h,
                // Single-layer grid: no via neighbours, so this is never called.
                |_, _| None,
            );

            match (ref_path, got) {
                (None, None) => {}
                (Some(rp), Some((gp, gcost))) => {
                    let ref_cost = field.dist[dst as usize];
                    assert_eq!(gcost, ref_cost, "cost mismatch {src}->{dst}");
                    // Returned path is a valid shortest path: endpoints correct,
                    // contiguous 4-neighbours, same summed enter-cost.
                    assert_eq!(gp.first().copied(), Some(src));
                    assert_eq!(gp.last().copied(), Some(dst));
                    let mut walked: Cost = 0;
                    for w in gp.windows(2) {
                        assert!(dims.neighbors4(w[0]).contains(&w[1]), "non-adjacent step");
                        assert!(!grid.is_obstacle(w[1]), "path enters obstacle");
                        walked = walked.saturating_add(grid.cost_at(w[1]));
                    }
                    assert_eq!(walked, gcost, "summed path cost mismatch");
                    // Reference path has the same length (both are shortest).
                    assert_eq!(gp.len(), rp.len(), "path length mismatch {src}->{dst}");
                }
                (a, b) => panic!("reachability mismatch {src}->{dst}: {a:?} vs {b:?}"),
            }
        }
    }
}
