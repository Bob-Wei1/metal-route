//! Criterion benchmark: time a CPU [`Router`] over a couple of shared fixtures.
//!
//! This bench measures the routing throughput the M2 gate's CPU baseline is
//! projected against by [`mr_bench::project_speedup`]. It runs via
//! `cargo bench -p mr-bench`; it is `harness = false` and does NOT run under
//! `cargo test`.
//!
//! Times the real CPU baseline (`mr_cpu::LeeRouter`) — the throughput the M2
//! gate's projection (`mr_bench::project_speedup`) is measured against.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mr_bench::time_router;
use mr_cpu::LeeRouter;
use mr_fixtures::{obstacle_battery, Fixture};

fn bench_route(c: &mut Criterion) {
    let router = LeeRouter;

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
