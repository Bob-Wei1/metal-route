//! Criterion benchmark: time a CPU [`Router`] over a couple of shared fixtures.
//!
//! This bench measures the routing throughput the M2 gate's CPU baseline is
//! projected against by [`mr_bench::project_speedup`]. It runs via
//! `cargo bench -p mr-bench`; it is `harness = false` and does NOT run under
//! `cargo test`.
//!
//! NOTE: the metalroute CPU router (`mr_cpu::LeeRouter`) lands in Wave 1. Until
//! its public type exists, this bench times an in-bench stand-in [`StubRouter`]
//! so the harness compiles and runs today. Swap the `router` binding below for
//! `mr_cpu::LeeRouter::default()` (and drop `StubRouter`) once Wave 1 is in.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mr_bench::time_router;
use mr_core::{BoardRoute, Grid, NetEndpoints, RouteResult, Router, RouterError};
use mr_fixtures::{obstacle_battery, Fixture};

// Touch the mr-cpu crate so the dev-dependency is exercised; replace with the
// real `LeeRouter` once Wave 1 exports it.
use mr_cpu as _;

/// Placeholder router used until `mr_cpu::LeeRouter` exists. Emits a one-cell
/// path per net. Realistic timing comes from swapping this for the real router.
#[derive(Default)]
struct StubRouter;

impl Router for StubRouter {
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
        })
    }
}

fn bench_route(c: &mut Criterion) {
    let router = StubRouter;

    // The shared battery already ends with `hand_32x32_wall`, covering both a
    // spread of small obstacle grids and the larger 32x32 case.
    let cases: Vec<Fixture> = obstacle_battery();

    let mut group = c.benchmark_group("route");
    for fx in &cases {
        let id = BenchmarkId::from_parameter(fx.name);
        group.bench_with_input(id, fx, |b, fx| {
            b.iter(|| {
                let t = time_router(&router, &fx.grid, &fx.nets);
                std::hint::black_box(t);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_route);
criterion_main!(benches);
