//! `mr-metal` — the GPU heart of metalroute (plan M3/M4).
//!
//! This crate ports the single-source distance-field computation behind the
//! routers onto Apple-Silicon GPUs via Metal compute kernels (the `metal`
//! crate). The CPU implementation in `mr-cpu` is the correctness ORACLE: every
//! field this crate produces must equal [`mr_cpu::bfs_distance_field`] /
//! [`mr_cpu::sweep_distance_field`] element-wise, and routed boards must be
//! `mr_oracle::are_equivalent` to [`mr_cpu::LeeRouter`].
//!
//! Two kernels are implemented, both atomic-free with ping-pong buffers:
//!
//! * **M3 — naive wavefront** ([`metal_wavefront_field`]). Each iteration relaxes
//!   every cell against its 4 neighbours: `new[i] = min(old[i], min_n(old[n] +
//!   cost(i)))`. Obstacles stay `Cost::MAX`. Iterates until a change-flag buffer
//!   reports no change.
//!
//! * **M4 — separable H/V prefix-min sweep** ([`metal_sweep_field`]). One kernel
//!   owns a row and runs a serial L→R then R→L prefix-min; another owns a column
//!   and runs U→D then D→U. H/V passes alternate until convergence. This mirrors
//!   [`mr_cpu::sweep_distance_field`] exactly.
//!
//! ## Tie-break / parent-in-field (the M0 finding)
//!
//! Per the M0 caveat (see `mr-cpu/src/sweep.rs`), a converged *cost* field does
//! not carry the canonical tie-break path; that is a property of reconstruction.
//! [`MetalRouter`] reconstructs the path by **backward greedy descent choosing the
//! lowest-[`CellIdx`] valid predecessor** — identical to
//! [`mr_cpu::sweep::path_from_field`]. The oracle requires equal cost + equal
//! congestion (not bit-identical paths); this reconstruction reproduces the
//! [`mr_core::TieBreak::LowerCellIdx`] path on the fixtures and is verified to be
//! `mr_oracle::are_equivalent` to [`mr_cpu::LeeRouter`] in the tests.
//!
//! On non-macOS targets the public surface still exists, but every entry point
//! returns [`RouterError::BackendUnavailable`] so the workspace compiles
//! everywhere.

use mr_core::{BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError};

#[cfg(target_os = "macos")]
mod gpu;

/// Single-source distance field via the naive atomic-free wavefront kernel (M3).
///
/// Returns `dist` indexed by [`CellIdx`]; unreachable cells (and an obstacle
/// source) are `Cost::MAX`. Equal to [`mr_cpu::bfs_distance_field`].
#[cfg(target_os = "macos")]
pub fn metal_wavefront_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    gpu::wavefront_field(grid, src)
}

/// Non-macOS fallback: the Metal backend is unavailable.
#[cfg(not(target_os = "macos"))]
pub fn metal_wavefront_field(_grid: &Grid, _src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// Single-source distance field via the separable H/V prefix-min sweep kernels
/// (M4). Returns `dist` indexed by [`CellIdx`]; unreachable cells are
/// `Cost::MAX`. Equal to [`mr_cpu::sweep_distance_field`] and
/// [`mr_cpu::bfs_distance_field`].
#[cfg(target_os = "macos")]
pub fn metal_sweep_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    gpu::sweep_field(grid, src)
}

/// Non-macOS fallback: the Metal backend is unavailable.
#[cfg(not(target_os = "macos"))]
pub fn metal_sweep_field(_grid: &Grid, _src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// A [`Router`] backed by the Metal GPU sweep kernel.
///
/// For each net it computes the distance field from `src` on the GPU, then
/// reconstructs the path to `dst` by lowest-index backward descent so the result
/// is `mr_oracle::are_equivalent` to [`mr_cpu::LeeRouter`]. Unreachable nets land
/// in [`BoardRoute::unrouted`].
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalRouter;

impl MetalRouter {
    pub fn new() -> Self {
        Self
    }
}

/// Reconstruct the path `src..=dst` from a converged distance field by greedy
/// descent, breaking ties by lowest [`CellIdx`] predecessor. Mirrors
/// `mr_cpu::sweep::path_from_field`. Returns `None` when `dst` is unreachable.
fn path_from_field(grid: &Grid, dist: &[Cost], src: CellIdx, dst: CellIdx) -> Option<Vec<CellIdx>> {
    if dist[dst as usize] == Cost::MAX {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
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

impl Router for MetalRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        let mut results = Vec::new();
        let mut unrouted = Vec::new();
        for net in nets {
            if !grid.dims.contains(net.src)
                || !grid.dims.contains(net.dst)
                || grid.is_obstacle(net.src)
                || grid.is_obstacle(net.dst)
            {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            let dist = metal_sweep_field(grid, net.src)?;
            match path_from_field(grid, &dist, net.src, net.dst) {
                Some(path) => results.push(RouteResult {
                    net: net.net.clone(),
                    path,
                    cost: dist[net.dst as usize],
                }),
                None => unrouted.push(net.net.clone()),
            }
        }
        let congestion = BoardRoute::congestion_from(grid.dims, &results);
        Ok(BoardRoute {
            results,
            unrouted,
            congestion,
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use mr_cpu::{bfs_distance_field, sweep_distance_field, LeeRouter, RipUpRouter};
    use mr_fixtures::{hand_32x32_wall, obstacle_battery, tie_break_2x2};
    use mr_grid::GridBuilder;
    use std::time::Instant;

    // ---- M3: naive wavefront field == CPU BFS field --------------------------

    #[test]
    fn wavefront_field_equals_bfs_on_battery() {
        for f in obstacle_battery() {
            let src = f.nets[0].src;
            let gpu = metal_wavefront_field(&f.grid, src).unwrap();
            let cpu = bfs_distance_field(&f.grid, src);
            assert_eq!(gpu.len(), cpu.len(), "{}", f.name);
            for i in 0..gpu.len() {
                assert_eq!(
                    gpu[i], cpu[i],
                    "{}: cell {i} gpu={} bfs={} (Cost::MAX==unreachable)",
                    f.name, gpu[i], cpu[i]
                );
            }
        }
    }

    #[test]
    fn wavefront_field_equals_bfs_on_hand_wall() {
        let f = hand_32x32_wall();
        let src = f.nets[0].src;
        let gpu = metal_wavefront_field(&f.grid, src).unwrap();
        let cpu = bfs_distance_field(&f.grid, src);
        assert_eq!(gpu, cpu);
        // Sanity: the pinned corner cost is 93.
        assert_eq!(gpu[f.nets[0].dst as usize], 93);
    }

    // ---- M4: separable sweep field == CPU sweep == CPU BFS -------------------

    #[test]
    fn sweep_field_equals_cpu_on_battery() {
        for f in obstacle_battery() {
            let src = f.nets[0].src;
            let gpu = metal_sweep_field(&f.grid, src).unwrap();
            let cpu_sweep = sweep_distance_field(&f.grid, src);
            let cpu_bfs = bfs_distance_field(&f.grid, src);
            assert_eq!(gpu.len(), cpu_bfs.len(), "{}", f.name);
            for i in 0..gpu.len() {
                assert_eq!(
                    gpu[i], cpu_sweep[i],
                    "{}: cell {i} gpu_sweep={} cpu_sweep={}",
                    f.name, gpu[i], cpu_sweep[i]
                );
                assert_eq!(
                    gpu[i], cpu_bfs[i],
                    "{}: cell {i} gpu_sweep={} cpu_bfs={}",
                    f.name, gpu[i], cpu_bfs[i]
                );
            }
        }
    }

    #[test]
    fn sweep_field_equals_cpu_on_hand_wall() {
        let f = hand_32x32_wall();
        let src = f.nets[0].src;
        let gpu = metal_sweep_field(&f.grid, src).unwrap();
        assert_eq!(gpu, bfs_distance_field(&f.grid, src));
        assert_eq!(gpu[f.nets[0].dst as usize], 93);
    }

    // ---- Router: GPU == CPU under the oracle ---------------------------------

    /// A hand-built multi-net grid: an open 6x6 with several independent nets.
    fn multi_net_grid() -> (Grid, Vec<NetEndpoints>) {
        let dims = mr_core::Dims::new(6, 6);
        let mut b = GridBuilder::new(dims, 1);
        // A short central wall to force some detours / shared congestion.
        b.mark_rect(2, 1, 2, 3); // column x=2, rows 1..=3 (inclusive corners)
        let grid = b.build();
        let nets = vec![
            NetEndpoints {
                net: "a".into(),
                src: dims.idx(0, 0),
                dst: dims.idx(5, 0),
                passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "b".into(),
                src: dims.idx(0, 5),
                dst: dims.idx(5, 5),
                passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "c".into(),
                src: dims.idx(0, 2),
                dst: dims.idx(5, 3),
                passable_pads: Vec::new(),
            },
        ];
        (grid, nets)
    }

    #[test]
    fn router_equivalent_to_lee_on_hand_wall() {
        let f = hand_32x32_wall();
        let gpu = MetalRouter.route(&f.grid, &f.nets).unwrap();
        let cpu = LeeRouter.route(&f.grid, &f.nets).unwrap();
        assert!(
            mr_oracle::are_equivalent(&gpu, &cpu),
            "discrepancies: {:?}",
            mr_oracle::compare(&gpu, &cpu)
        );
        assert_eq!(gpu.results[0].cost, 93);
    }

    #[test]
    fn router_equivalent_to_lee_on_tie_break_2x2() {
        let f = tie_break_2x2();
        let gpu = MetalRouter.route(&f.grid, &f.nets).unwrap();
        let cpu = LeeRouter.route(&f.grid, &f.nets).unwrap();
        assert!(
            mr_oracle::are_equivalent(&gpu, &cpu),
            "discrepancies: {:?}",
            mr_oracle::compare(&gpu, &cpu)
        );
        // The tie-break path is pinned to [0, 1, 3].
        assert_eq!(gpu.results[0].path, vec![0, 1, 3]);
    }

    #[test]
    fn router_equivalent_to_lee_on_multi_net() {
        let (grid, nets) = multi_net_grid();
        let gpu = MetalRouter.route(&grid, &nets).unwrap();
        let cpu = LeeRouter.route(&grid, &nets).unwrap();
        assert!(
            mr_oracle::are_equivalent(&gpu, &cpu),
            "discrepancies: {:?}",
            mr_oracle::compare(&gpu, &cpu)
        );
    }

    #[test]
    fn router_equivalent_to_lee_on_battery() {
        for f in obstacle_battery() {
            let gpu = MetalRouter.route(&f.grid, &f.nets).unwrap();
            let cpu = LeeRouter.route(&f.grid, &f.nets).unwrap();
            assert!(
                mr_oracle::are_equivalent(&gpu, &cpu),
                "{}: discrepancies: {:?}",
                f.name,
                mr_oracle::compare(&gpu, &cpu)
            );
        }
    }

    // ---- D3: CPU vs Metal batch benchmark ------------------------------------

    /// Build a large open grid with `n` independent random-ish nets (deterministic).
    fn bench_board(side: u32, n_nets: usize) -> (Grid, Vec<NetEndpoints>) {
        let dims = mr_core::Dims::new(side, side);
        // A few scattered vertical walls (each leaving a gap) so routing detours.
        let mut b = GridBuilder::new(dims, 1);
        let y_hi = (side / 2).max(1) - 1; // wall spans rows 1..=y_hi, leaving y=0 open
        for k in 0..(side / 8) {
            let x = (2 + k * 5).min(side - 2);
            b.mark_rect(x, 1, x, y_hi); // inclusive corners: single column
        }
        let grid = b.build();
        // Deterministic endpoint generation that avoids obstacles.
        let mut nets = Vec::with_capacity(n_nets);
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut pick = |grid: &Grid| -> CellIdx {
            loop {
                let i = (next() % dims.len() as u64) as CellIdx;
                if !grid.is_obstacle(i) {
                    return i;
                }
            }
        };
        for k in 0..n_nets {
            let src = pick(&grid);
            let dst = pick(&grid);
            nets.push(NetEndpoints {
                net: format!("n{k}"),
                src,
                dst,
                passable_pads: Vec::new(),
            });
        }
        (grid, nets)
    }

    /// D3 deliverable: honest CPU-vs-Metal throughput on PCB-scale grids.
    /// Run with: `cargo test -p mr-metal -- --nocapture batch_benchmark`
    #[test]
    fn batch_benchmark_cpu_vs_metal() {
        let side = 128u32;
        let n_nets = 64usize;
        let (grid, nets) = bench_board(side, n_nets);

        // Apples-to-apples: LeeRouter and MetalRouter both route each net
        // INDEPENDENTLY (one shortest-path field per net), so they do equal work.
        let t_lee = Instant::now();
        let lee = LeeRouter.route(&grid, &nets).unwrap();
        let lee_dt = t_lee.elapsed();

        // RipUpRouter is the sequential production CPU router (collision-aware);
        // reported for context only — it does MORE work (rip-up passes) and may
        // leave nets unrouted, so it is not a like-for-like throughput compare.
        let t_rip = Instant::now();
        let rip = RipUpRouter.route(&grid, &nets).unwrap();
        let rip_dt = t_rip.elapsed();

        // Metal: MetalRouter (one GPU sweep field per net) — same work as Lee.
        let t_gpu = Instant::now();
        let gpu = MetalRouter.route(&grid, &nets).unwrap();
        let gpu_dt = t_gpu.elapsed();

        let nps = |dt: std::time::Duration| n_nets as f64 / dt.as_secs_f64();

        println!("\n=== D3 batch benchmark ({side}x{side} grid, {n_nets} independent nets) ===");
        println!(
            "CPU  LeeRouter (indep):   {:>10.3?}  {:>9.1} nets/sec  ({} routed)",
            lee_dt,
            nps(lee_dt),
            lee.results.len()
        );
        println!(
            "Metal MetalRouter (indep):{:>10.3?}  {:>9.1} nets/sec  ({} routed)",
            gpu_dt,
            nps(gpu_dt),
            gpu.results.len()
        );
        println!(
            "CPU  RipUpRouter (ctx):   {:>10.3?}  {:>9.1} nets/sec  ({} routed, collision-aware)",
            rip_dt,
            nps(rip_dt),
            rip.results.len()
        );

        // Headline: like-for-like Lee (CPU) vs MetalRouter (GPU).
        let speedup = lee_dt.as_secs_f64() / gpu_dt.as_secs_f64();
        if speedup >= 1.0 {
            println!("Headline (Lee vs Metal, equal work): Metal is {speedup:.2}x FASTER");
        } else {
            println!(
                "Headline (Lee vs Metal, equal work): CPU is {:.2}x FASTER (expected: per-net GPU dispatch overhead dominates at PCB scale)",
                1.0 / speedup
            );
        }
    }
}
