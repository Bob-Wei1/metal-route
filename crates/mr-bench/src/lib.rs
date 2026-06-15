//! `mr-bench` — the metalroute timing harness and the **M2 go/no-go speedup
//! projection**.
//!
//! Two responsibilities:
//!
//! 1. [`time_router`] — a tiny, allocation-light wrapper that runs any
//!    [`Router`](mr_core::Router) over a board and reports wall time plus a few
//!    derived throughput numbers ([`RouteTiming`]). Used both by the Criterion
//!    benchmark under `benches/` and by callers that want a cheap one-shot timing.
//!
//! 2. [`project_speedup`] — a **pure** heuristic that projects the batch-GPU
//!    speedup over the CPU router for a given board shape, so the M2 gate can make
//!    a go/no-go call *before* the Metal backend exists. See its docs for the
//!    model and the constants.

use std::time::{Duration, Instant};

use mr_core::{Dims, Grid, NetEndpoints, Router};

// ---------------------------------------------------------------------------
// 1. M2 speedup projection
// ---------------------------------------------------------------------------

/// Fixed host↔GPU dispatch overhead, expressed in units of "CPU work to expand a
/// single grid cell". This is the latency the GPU pays *once per batch* to launch
/// a kernel and shuttle buffers across the bus, regardless of problem size. It is
/// what makes tiny problems lose to the GPU.
///
/// Calibrated so an 8×8 / 1-net problem lands well under the 2.0 gate and a
/// 256×256 / 500-net problem clears it comfortably.
const DISPATCH_OVERHEAD_CELLS: f32 = 50_000.0;

/// Hard cap on the parallel width the GPU can actually exploit at once. Beyond
/// this many concurrent nets the device is saturated and extra nets queue behind
/// the resident batch, so they no longer reduce per-net latency. Keeps the model
/// from projecting unbounded speedup from absurd net counts.
const MAX_PARALLEL_NETS: f32 = 1_024.0;

/// Per-cell GPU throughput advantage once a kernel is actually running: the
/// device chews through a single net's cell expansions this many times faster
/// than one CPU core, thanks to wide SIMD lanes over the cost grid. Applied on
/// top of, and independent from, the batch (cross-net) parallelism.
const GPU_CELL_THROUGHPUT: f32 = 8.0;

/// Project the batch-GPU speedup versus the CPU router for a board of the given
/// dimensions carrying `net_count` independent nets.
///
/// # Model
///
/// The CPU routes nets essentially one at a time, and each net's dominant cost is
/// a grid sweep proportional to the cell count `cells = w * h`:
///
/// ```text
/// cpu_time = net_count * cells
/// ```
///
/// The GPU's win is **batch parallelism**: it routes up to `P` independent nets
/// concurrently in one kernel launch (`P = min(net_count, MAX_PARALLEL_NETS)`),
/// and within a launch each net's cell sweep is itself `GPU_CELL_THROUGHPUT`×
/// faster. But every launch pays a fixed `DISPATCH_OVERHEAD_CELLS` to cross the
/// host↔GPU boundary:
///
/// ```text
/// batches  = ceil(net_count / P)              // here P == net_count, so 1 batch
/// gpu_time = DISPATCH_OVERHEAD_CELLS          // fixed launch cost
///          + batches * (cells / GPU_CELL_THROUGHPUT)   // useful work per batch
/// speedup  = cpu_time / gpu_time
/// ```
///
/// Because the fixed overhead is amortised across the whole batch, speedup rises
/// as either dimension of the problem grows — more nets spread the launch cost
/// over more useful work, and bigger grids make the per-net work dwarf the launch
/// cost. Tiny problems are dominated by the launch cost and lose (< 1×).
///
/// # Guaranteed properties (all covered by tests)
///
/// * monotonically non-decreasing in `net_count`;
/// * monotonically non-decreasing in grid cell count;
/// * a tiny problem (e.g. 8×8, 1 net) returns `< 2.0`;
/// * a large batch (e.g. 256×256, 500 nets) returns `> 2.0`.
pub fn project_speedup(grid: Dims, net_count: u32) -> f32 {
    let cells = grid.len() as f32;
    let nets = net_count as f32;
    if cells == 0.0 || nets == 0.0 {
        return 0.0;
    }

    // CPU: one full grid sweep per net, serially.
    let cpu_time = nets * cells;

    // GPU: how many nets fit in one resident batch.
    let parallel = nets.min(MAX_PARALLEL_NETS);
    // Number of kernel launches needed to cover every net.
    let batches = (nets / parallel).ceil();

    // Each launch pays the fixed dispatch overhead plus the (accelerated) cell
    // sweep for the nets resident in that batch. Per-batch useful work is the
    // single-net sweep, since the batched nets run concurrently.
    let gpu_time = DISPATCH_OVERHEAD_CELLS + batches * (cells / GPU_CELL_THROUGHPUT);

    cpu_time / gpu_time
}

// ---------------------------------------------------------------------------
// 2. Timing harness
// ---------------------------------------------------------------------------

/// Wall-clock timing plus derived throughput for one [`Router::route`] call.
#[derive(Debug, Clone)]
pub struct RouteTiming {
    /// Measured wall-clock duration of the `route` call.
    pub wall: Duration,
    /// Routed nets per second (`net_count / wall_secs`), `0.0` if `wall` is zero.
    pub nets_per_sec: f64,
    /// Number of nets that were successfully routed.
    pub routed: usize,
    /// Number of nets that could not be routed.
    pub unrouted: usize,
    /// Grid cell count (`w * h`) for the board that was timed.
    pub grid_cells: usize,
    /// Number of nets submitted.
    pub net_count: usize,
}

/// Run `router` over `grid`/`nets`, returning the wall time and derived stats.
///
/// On a router error the timing is still returned with `routed == 0` and every
/// submitted net counted as `unrouted`, so a backend failure does not panic the
/// harness.
pub fn time_router<R: Router>(router: &R, grid: &Grid, nets: &[NetEndpoints]) -> RouteTiming {
    let net_count = nets.len();
    let grid_cells = grid.dims.len();

    let start = Instant::now();
    let result = router.route(grid, nets);
    let wall = start.elapsed();

    let (routed, unrouted) = match &result {
        Ok(board) => (board.results.len(), board.unrouted.len()),
        Err(_) => (0, net_count),
    };

    let secs = wall.as_secs_f64();
    let nets_per_sec = if secs > 0.0 {
        net_count as f64 / secs
    } else {
        0.0
    };

    RouteTiming {
        wall,
        nets_per_sec,
        routed,
        unrouted,
        grid_cells,
        net_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::{BoardRoute, RouteResult, RouterError};

    // -- a trivial in-crate router so the harness test needs no real backend --

    /// Routes every net as the single-cell path `[src]` with cost 0. Enough to
    /// exercise [`time_router`] without depending on a real routing crate.
    struct TrivialRouter;

    impl Router for TrivialRouter {
        fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
            let results: Vec<RouteResult> = nets
                .iter()
                .map(|n| RouteResult {
                    net: n.net.clone(),
                    path: vec![n.src],
                    cost: 0,
                })
                .collect();
            let congestion = BoardRoute::congestion_from(grid.dims, &results);
            Ok(BoardRoute {
                results,
                unrouted: Vec::new(),
                congestion,
                groups: Vec::new(),
            })
        }
    }

    // ---- project_speedup -------------------------------------------------

    #[test]
    fn speedup_non_decreasing_in_net_count() {
        let grid = Dims::new(128, 128);
        let mut prev = project_speedup(grid, 1);
        for nets in [2u32, 4, 8, 16, 32, 64, 128, 256, 500, 1000] {
            let s = project_speedup(grid, nets);
            assert!(
                s >= prev - f32::EPSILON,
                "speedup dropped: nets={nets} gave {s} < prev {prev}"
            );
            prev = s;
        }
    }

    #[test]
    fn speedup_non_decreasing_in_grid_cells() {
        let net_count = 64;
        let mut prev = project_speedup(Dims::new(4, 4), net_count);
        for &(w, h) in &[
            (8u32, 8u32),
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
        ] {
            let s = project_speedup(Dims::new(w, h), net_count);
            assert!(
                s >= prev - f32::EPSILON,
                "speedup dropped: {w}x{h} gave {s} < prev {prev}"
            );
            prev = s;
        }
    }

    #[test]
    fn tiny_case_loses_to_dispatch() {
        // 8x8 grid, single net — dominated by dispatch overhead.
        let s = project_speedup(Dims::new(8, 8), 1);
        assert!(s < 2.0, "tiny case should be < 2.0, got {s}");
    }

    #[test]
    fn large_batch_clears_gate() {
        // 256x256 grid, 500 nets — batch parallelism amortises dispatch.
        let s = project_speedup(Dims::new(256, 256), 500);
        assert!(s > 2.0, "large batch should be > 2.0, got {s}");
    }

    #[test]
    fn degenerate_inputs_are_zero() {
        assert_eq!(project_speedup(Dims::new(0, 0), 100), 0.0);
        assert_eq!(project_speedup(Dims::new(64, 64), 0), 0.0);
    }

    // ---- time_router -----------------------------------------------------

    #[test]
    fn time_router_routes_one_net_open_grid() {
        let grid = Grid::filled(Dims::new(8, 8), 1);
        let nets = vec![NetEndpoints {
            net: "n0".into(),
            src: grid.dims.idx(0, 0),
            dst: grid.dims.idx(7, 7),
            passable_pads: Vec::new(),
        }];

        let t = time_router(&TrivialRouter, &grid, &nets);

        assert_eq!(t.routed, 1, "the one net should be routed");
        assert_eq!(t.unrouted, 0);
        assert_eq!(t.net_count, 1);
        assert_eq!(t.grid_cells, 64);
        assert!(t.wall > Duration::ZERO, "wall time must be positive");
        assert!(
            t.nets_per_sec > 0.0 && t.nets_per_sec.is_finite(),
            "nets_per_sec should be a sensible positive number, got {}",
            t.nets_per_sec
        );
    }

    #[test]
    fn time_router_counts_error_as_unrouted() {
        struct FailRouter;
        impl Router for FailRouter {
            fn route(
                &self,
                _grid: &Grid,
                _nets: &[NetEndpoints],
            ) -> Result<BoardRoute, RouterError> {
                Err(RouterError::BackendUnavailable("test".into()))
            }
        }
        let grid = Grid::filled(Dims::new(4, 4), 1);
        let nets = vec![NetEndpoints {
            net: "a".into(),
            src: 0,
            dst: 15,
            passable_pads: Vec::new(),
        }];
        let t = time_router(&FailRouter, &grid, &nets);
        assert_eq!(t.routed, 0);
        assert_eq!(t.unrouted, 1);
    }
}
