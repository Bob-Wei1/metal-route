//! Local, reproducible tscircuit-style benchmark (plan B5 / Wave 2).
//!
//! The official `autorouting-dataset benchmark --solver-url` harness drives our
//! `mr-server` `/solve` endpoint over the network; at time of writing the npm
//! package is unpublished, so this module generates a deterministic suite of
//! [`SimpleRouteJson`] problems locally and scores our CPU router on the same
//! metrics the official harness reports: **completion rate, trace length, and
//! throughput (nets/sec)**. The numbers are written to a JSON report so they are
//! reproducible from a clean checkout — the honest CPU baseline the M2 go/no-go
//! gate is decided against.

use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use crate::{project, route_problem, Projection, RouterKind};
use mr_srj::SimpleRouteJson;

/// Arguments for the `bench` subcommand.
#[derive(Debug, clap::Parser)]
pub struct BenchArgs {
    /// Number of boards (independent problems) to generate and route.
    #[arg(long, default_value_t = 10)]
    pub boards: u32,

    /// Connections (2-point nets) per board — the batch-parallelism workload.
    #[arg(long, default_value_t = 30)]
    pub nets: u32,

    /// Board side length in continuous units (square board, bounds 0..size).
    #[arg(long, default_value_t = 50.0)]
    pub size: f64,

    /// Rectangular obstacles per board.
    #[arg(long, default_value_t = 8)]
    pub obstacles: u32,

    /// Deterministic seed (board `i` uses `seed + i`).
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// Cell size; defaults to the same bounds-derived heuristic as `route`.
    #[arg(long)]
    pub resolution: Option<f64>,

    /// Write the JSON report here (also printed to stdout if omitted).
    #[arg(long)]
    pub out: Option<std::path::PathBuf>,
}

/// Per-board benchmark record.
#[derive(Debug, Clone, Serialize)]
pub struct BoardReport {
    pub seed: u64,
    pub grid_w: u32,
    pub grid_h: u32,
    pub nets_total: usize,
    pub nets_routed: usize,
    pub total_cost: u64,
    pub wall_ms: f64,
}

/// Aggregate benchmark report — the checked-in CPU baseline.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub router: String,
    pub boards: usize,
    pub nets_total: usize,
    pub nets_routed: usize,
    /// Fraction of submitted nets that produced a routed trace.
    pub completion_rate: f64,
    /// Mean routed-net cost (trace length in grid cells).
    pub mean_trace_cost: f64,
    pub total_wall_ms: f64,
    /// Throughput across all boards.
    pub nets_per_sec: f64,
    /// M2 projection evaluated at the median board's grid + net count.
    pub m2_projected_speedup: f32,
    pub m2_verdict: String,
    pub per_board: Vec<BoardReport>,
}

/// A tiny deterministic LCG (no `rand` dependency, fully reproducible).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    /// Uniform-ish f64 in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() % 100_000) as f64 / 100_000.0 * (hi - lo)
    }
}

/// Generate one deterministic SimpleRouteJson problem.
///
/// Net endpoints are sampled clear of every obstacle (rejection sampling with a
/// margin, falling back to the obstacle-free edge band) so generated problems are
/// always valid input — an endpoint inside an obstacle is genuinely unroutable
/// and the router rejects it, which is not what we want to measure here.
pub fn generate_problem(seed: u64, size: f64, n_obstacles: u32, n_nets: u32) -> SimpleRouteJson {
    let mut rng = Lcg(seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493));

    // (cx, cy, half_w, half_h)
    let rects: Vec<(f64, f64, f64, f64)> = (0..n_obstacles)
        .map(|_| {
            let cx = rng.uniform(0.15 * size, 0.85 * size);
            let cy = rng.uniform(0.15 * size, 0.85 * size);
            let w = rng.uniform(0.04 * size, 0.16 * size);
            let h = rng.uniform(0.04 * size, 0.16 * size);
            (cx, cy, w / 2.0, h / 2.0)
        })
        .collect();

    let obstacles: Vec<_> = rects
        .iter()
        .map(|&(cx, cy, hw, hh)| {
            json!({ "type": "rect", "center": {"x": cx, "y": cy},
                    "width": hw * 2.0, "height": hh * 2.0 })
        })
        .collect();

    // Margin keeps endpoints a cell or two away from obstacle edges.
    let margin = (size / 64.0).max(0.5);
    let in_obstacle = |x: f64, y: f64| {
        rects
            .iter()
            .any(|&(cx, cy, hw, hh)| (x - cx).abs() <= hw + margin && (y - cy).abs() <= hh + margin)
    };
    let sample_point = |rng: &mut Lcg| {
        for _ in 0..32 {
            let x = rng.uniform(0.02 * size, 0.98 * size);
            let y = rng.uniform(0.02 * size, 0.98 * size);
            if !in_obstacle(x, y) {
                return (x, y);
            }
        }
        // Fallback: the edge band is outside every obstacle (centres are in
        // [0.15,0.85]·size with half-extent < 0.08·size).
        (
            rng.uniform(0.01 * size, 0.04 * size),
            rng.uniform(0.01 * size, 0.96 * size),
        )
    };

    let connections: Vec<_> = (0..n_nets)
        .map(|i| {
            let (ax, ay) = sample_point(&mut rng);
            let (bx, by) = sample_point(&mut rng);
            json!({ "name": format!("n{i}"),
                    "pointsToConnect": [ {"x": ax, "y": ay}, {"x": bx, "y": by} ] })
        })
        .collect();

    let value = json!({
        "layerCount": 1,
        "bounds": { "minX": 0.0, "maxX": size, "minY": 0.0, "maxY": size },
        "obstacles": obstacles,
        "connections": connections,
    });
    serde_json::from_value(value).expect("generated SimpleRouteJson is always valid")
}

/// Run the full benchmark suite on the CPU router and produce the report.
pub fn run_suite(args: &BenchArgs) -> Result<BenchReport> {
    let mut per_board = Vec::with_capacity(args.boards as usize);
    let mut nets_total = 0usize;
    let mut nets_routed = 0usize;
    let mut cost_sum = 0u64;
    let mut wall_total = 0.0f64;

    for i in 0..args.boards {
        let seed = args.seed + i as u64;
        let srj = generate_problem(seed, args.size, args.obstacles, args.nets);

        let t0 = Instant::now();
        let (_traces, summary) = route_problem(&srj, args.resolution, RouterKind::Ripup)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        nets_total += summary.total;
        nets_routed += summary.routed;
        cost_sum += summary.total_cost;
        wall_total += wall_ms;

        per_board.push(BoardReport {
            seed,
            grid_w: summary.grid_w,
            grid_h: summary.grid_h,
            nets_total: summary.total,
            nets_routed: summary.routed,
            total_cost: summary.total_cost,
            wall_ms,
        });
    }

    // M2 projection at the median board (sorted by grid cell count).
    let mut by_cells = per_board.clone();
    by_cells.sort_by_key(|b| b.grid_w as u64 * b.grid_h as u64);
    let med = &by_cells[by_cells.len() / 2];
    let proj: Projection = project(med.grid_w, med.grid_h, args.nets);

    let nets_per_sec = if wall_total > 0.0 {
        nets_routed as f64 / (wall_total / 1000.0)
    } else {
        0.0
    };

    Ok(BenchReport {
        router: "ripup-cpu".into(),
        boards: per_board.len(),
        nets_total,
        nets_routed,
        completion_rate: if nets_total > 0 {
            nets_routed as f64 / nets_total as f64
        } else {
            0.0
        },
        mean_trace_cost: if nets_routed > 0 {
            cost_sum as f64 / nets_routed as f64
        } else {
            0.0
        },
        total_wall_ms: wall_total,
        nets_per_sec,
        m2_projected_speedup: proj.speedup,
        m2_verdict: if proj.go { "GO".into() } else { "NO-GO".into() },
        per_board,
    })
}

/// Execute the `bench` subcommand: run the suite, write the report.
pub fn run_bench(args: &BenchArgs) -> Result<BenchReport> {
    let report = run_suite(args)?;
    let json = serde_json::to_string_pretty(&report)?;
    match &args.out {
        Some(path) => std::fs::write(path, &json)?,
        None => println!("{json}"),
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_problem_is_deterministic_and_well_formed() {
        let a = generate_problem(7, 30.0, 8, 60);
        let b = generate_problem(7, 30.0, 8, 60);
        assert_eq!(a.connections.len(), 60);
        assert_eq!(a.obstacles.len(), 8);
        // determinism: same seed -> identical first endpoint
        assert_eq!(
            a.connections[0].points_to_connect[0].x,
            b.connections[0].points_to_connect[0].x
        );
    }

    #[test]
    fn suite_routes_most_nets_and_reports_metrics() {
        // A deliberately SPARSE suite: few nets on a large board route reliably,
        // so this asserts pipeline correctness rather than rip-up saturation
        // behaviour (which is density-dependent and reported, not asserted).
        let args = BenchArgs {
            boards: 4,
            nets: 6,
            size: 50.0,
            obstacles: 3,
            seed: 1,
            resolution: Some(0.8),
            out: None,
        };
        let r = run_suite(&args).unwrap();
        assert_eq!(r.boards, 4);
        assert_eq!(r.nets_total, 24);
        assert!(
            r.completion_rate > 0.8,
            "sparse board completion {} too low",
            r.completion_rate
        );
        assert!(r.nets_per_sec > 0.0);
        assert!(r.mean_trace_cost > 0.0);
        assert!(r.m2_verdict == "GO" || r.m2_verdict == "NO-GO");
    }
}
