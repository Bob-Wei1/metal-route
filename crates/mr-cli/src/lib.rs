//! `mr-cli` — library half of the `metalroute` user-facing CLI.
//!
//! The real work lives here as plain functions so it can be unit/integration
//! tested without spawning a process; [`main`](../main.rs) is a thin clap
//! dispatcher over [`Cli`].
//!
//! Subcommands:
//!
//! * [`run_route`] — read a tscircuit [`SimpleRouteJson`](mr_srj::SimpleRouteJson)
//!   problem, rasterise it, route it (Lee or rip-up), and emit the routed
//!   solution soup as JSON.
//! * [`run_project`] — print the M2 [`project_speedup`](mr_bench::project_speedup)
//!   projection plus a GO / NO-GO verdict at the ~2× gate.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

pub mod bench;
use mr_core::Router;
use mr_cpu::{LeeRouter, RipUpRouter};
use mr_srj::{rasterize, to_solution, SimpleRouteJson};

/// The ~2× speedup threshold the M2 go/no-go gate uses.
pub const GO_NO_GO_THRESHOLD: f32 = 2.0;

/// Default trace width (continuous units) for emitted `pcb_trace` wires.
const DEFAULT_TRACE_WIDTH: f64 = 0.15;

/// Default routing layer name for emitted wires (single-layer for now).
const DEFAULT_LAYER: &str = "top";

/// `metalroute` — a PCB autorouter CLI.
#[derive(Debug, Parser)]
#[command(name = "metalroute", version, about = "metalroute PCB autorouter")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Route a SimpleRouteJson problem into a tscircuit solution soup.
    Route(RouteArgs),
    /// Print the M2 batch-GPU speedup projection and a GO/NO-GO verdict.
    Project(ProjectArgs),
    /// Run the local tscircuit-style benchmark suite and write a CPU baseline report.
    Bench(bench::BenchArgs),
    /// Hand a board to Freerouting (via bed-of-nails) for detailed routing (M5).
    Handoff(HandoffArgs),
}

/// Arguments for the `handoff` subcommand (M5 Freerouting bridge).
#[derive(Debug, Parser)]
pub struct HandoffArgs {
    /// Path to the `.kicad_pcb` to route.
    #[arg(long)]
    pub pcb: PathBuf,

    /// Freerouting optimization passes.
    #[arg(long, default_value_t = 20)]
    pub passes: u32,

    /// Timeout in seconds.
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,

    /// The bed-of-nails command to invoke.
    #[arg(long, default_value = "bon")]
    pub bon_command: String,
}

impl From<&HandoffArgs> for mr_bridge::BridgeConfig {
    fn from(a: &HandoffArgs) -> Self {
        mr_bridge::BridgeConfig {
            freerouting_passes: a.passes,
            timeout_s: a.timeout,
            bon_command: a.bon_command.clone(),
        }
    }
}

/// Core `handoff` logic over an injectable runner (so tests can mock the
/// subprocess). Shells out to bed-of-nails to drive Freerouting.
pub fn handoff_with<R: mr_bridge::CommandRunner>(
    runner: &R,
    args: &HandoffArgs,
) -> Result<mr_bridge::RunOutput> {
    let cfg = mr_bridge::BridgeConfig::from(args);
    let pcb = args.pcb.to_string_lossy();
    mr_bridge::handoff(runner, &pcb, &cfg).context("Freerouting handoff failed")
}

/// Execute the `handoff` subcommand against the real system.
pub fn run_handoff(args: &HandoffArgs) -> Result<mr_bridge::RunOutput> {
    handoff_with(&mr_bridge::SystemRunner, args)
}

/// Which CPU router backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum RouterKind {
    /// Lee/Dijkstra single-source router; each net routed independently.
    Lee,
    /// Sequential router with bounded rip-up-on-collision.
    #[default]
    Ripup,
}

/// Arguments for the `route` subcommand.
#[derive(Debug, Parser)]
pub struct RouteArgs {
    /// Path to the input SimpleRouteJson file.
    #[arg(long)]
    pub input: PathBuf,

    /// Cell size in continuous units. Defaults to a value derived from bounds.
    #[arg(long)]
    pub resolution: Option<f64>,

    /// Routing backend.
    #[arg(long, value_enum, default_value_t = RouterKind::default())]
    pub router: RouterKind,

    /// Output path for the solution soup JSON. Defaults to stdout.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// Arguments for the `project` subcommand.
#[derive(Debug, Parser)]
pub struct ProjectArgs {
    /// Board width in grid cells.
    #[arg(long)]
    pub width: u32,

    /// Board height in grid cells.
    #[arg(long)]
    pub height: u32,

    /// Number of independent nets.
    #[arg(long)]
    pub nets: u32,
}

/// One-line summary of a completed `route` run (also printed to stderr).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// Nets that produced a routed trace.
    pub routed: usize,
    /// Total nets submitted (after k-point decomposition).
    pub total: usize,
    /// Sum of routed-net costs.
    pub total_cost: u64,
    /// Grid width in cells.
    pub grid_w: u32,
    /// Grid height in cells.
    pub grid_h: u32,
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "routed {}/{} nets, total cost {}, grid {}x{}",
            self.routed, self.total, self.total_cost, self.grid_w, self.grid_h
        )
    }
}

/// Result of a `project` run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    /// Projected batch-GPU speedup over the CPU router.
    pub speedup: f32,
    /// Whether the projection clears the ~2× go/no-go gate.
    pub go: bool,
}

impl std::fmt::Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let verdict = if self.go { "GO" } else { "NO-GO" };
        write!(
            f,
            "projected speedup {:.3}x (gate {:.1}x): {}",
            self.speedup, GO_NO_GO_THRESHOLD, verdict
        )
    }
}

/// Choose a sensible cell size when the user does not pass `--resolution`.
///
/// Targets roughly [`TARGET_CELLS_PER_AXIS`] cells along the larger span so the
/// grid is detailed enough to route but not pathologically large. Degenerate /
/// zero-area bounds fall back to `1.0`.
fn default_resolution(srj: &SimpleRouteJson) -> f64 {
    /// Cells we aim to fit along the longer board axis at the default resolution.
    const TARGET_CELLS_PER_AXIS: f64 = 64.0;

    let b = &srj.bounds;
    let span = (b.max_x - b.min_x).max(b.max_y - b.min_y);
    if span.is_finite() && span > 0.0 {
        span / TARGET_CELLS_PER_AXIS
    } else {
        1.0
    }
}

/// Parse a [`SimpleRouteJson`] from raw bytes.
pub fn parse_srj(bytes: &[u8]) -> Result<SimpleRouteJson> {
    serde_json::from_slice(bytes).context("failed to parse SimpleRouteJson input")
}

/// Core `route` logic operating on an already-parsed problem.
///
/// Returns the routed solution soup as a serde JSON value alongside the
/// [`Summary`]. Pulled out from [`run_route`] so tests can drive it without
/// touching the filesystem.
pub fn route_problem(
    srj: &SimpleRouteJson,
    resolution: Option<f64>,
    router: RouterKind,
) -> Result<(Vec<mr_srj::PcbTrace>, Summary)> {
    let resolution = resolution.unwrap_or_else(|| default_resolution(srj));
    anyhow::ensure!(
        resolution.is_finite() && resolution > 0.0,
        "resolution must be a finite positive number, got {resolution}"
    );

    let problem = rasterize(srj, resolution);
    let total = problem.nets.len();

    let board = match router {
        RouterKind::Lee => LeeRouter::new().route(&problem.grid, &problem.nets),
        RouterKind::Ripup => RipUpRouter::new().route(&problem.grid, &problem.nets),
    }
    .context("router failed")?;

    let traces = to_solution(
        &board,
        &problem.mapping,
        &problem.pin_points,
        DEFAULT_TRACE_WIDTH,
        DEFAULT_LAYER,
    );

    let summary = Summary {
        routed: board.results.len(),
        total,
        total_cost: board.total_cost(),
        grid_w: problem.mapping.dims.w,
        grid_h: problem.mapping.dims.h,
    };

    Ok((traces, summary))
}

/// Execute the `route` subcommand: read the input file, route, write the
/// solution JSON to `--out` (or stdout), and return the [`Summary`].
///
/// The caller is responsible for printing the summary (to stderr).
pub fn run_route(args: &RouteArgs) -> Result<Summary> {
    let bytes = std::fs::read(&args.input)
        .with_context(|| format!("failed to read input file {}", args.input.display()))?;
    let srj = parse_srj(&bytes)?;

    let (traces, summary) = route_problem(&srj, args.resolution, args.router)?;

    let json = serde_json::to_string_pretty(&traces).context("failed to serialise solution")?;

    match &args.out {
        Some(path) => std::fs::write(path, json)
            .with_context(|| format!("failed to write output file {}", path.display()))?,
        None => {
            use std::io::Write;
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(json.as_bytes())?;
            stdout.write_all(b"\n")?;
        }
    }

    Ok(summary)
}

/// Core `project` logic: project the speedup and apply the go/no-go gate.
pub fn project(width: u32, height: u32, nets: u32) -> Projection {
    let speedup = mr_bench::project_speedup(mr_core::Dims::new(width, height), nets);
    Projection {
        speedup,
        go: speedup >= GO_NO_GO_THRESHOLD,
    }
}

/// Execute the `project` subcommand.
pub fn run_project(args: &ProjectArgs) -> Result<Projection> {
    Ok(project(args.width, args.height, args.nets))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "layerCount": 2,
        "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
        "obstacles": [
            { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
        ],
        "connections": [
            { "name": "VCC", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 1} ] },
            { "name": "GND", "pointsToConnect": [ {"x": 1, "y": 9}, {"x": 9, "y": 9} ] }
        ]
    }"#;

    #[test]
    fn route_problem_routes_two_nets() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        let (traces, summary) = route_problem(&srj, Some(1.0), RouterKind::Ripup).unwrap();

        // Two 2-point connections -> two nets, both routable on this open board.
        assert_eq!(summary.total, 2);
        assert_eq!(summary.routed, 2);
        assert_eq!(summary.grid_w, 10);
        assert_eq!(summary.grid_h, 10);
        assert!(summary.total_cost > 0);

        assert_eq!(traces.len(), 2);
        for t in &traces {
            assert_eq!(t.kind, "pcb_trace");
            assert!(!t.route.is_empty());
        }
    }

    #[test]
    fn route_problem_lee_backend_also_routes() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        let (traces, summary) = route_problem(&srj, Some(1.0), RouterKind::Lee).unwrap();
        assert_eq!(summary.routed, 2);
        assert!(!traces.is_empty());
    }

    #[test]
    fn default_resolution_targets_reasonable_grid() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        // span 10 / 64 -> small cells; grid should be well-formed and non-trivial.
        let (_traces, summary) = route_problem(&srj, None, RouterKind::Ripup).unwrap();
        assert!(summary.grid_w >= 10 && summary.grid_h >= 10);
    }

    #[test]
    fn default_resolution_handles_degenerate_bounds() {
        let degenerate = r#"{
            "layerCount": 1,
            "bounds": { "minX": 3, "maxX": 3, "minY": 3, "maxY": 3 },
            "connections": [],
            "obstacles": []
        }"#;
        let srj = parse_srj(degenerate.as_bytes()).unwrap();
        assert_eq!(default_resolution(&srj), 1.0);
    }

    #[test]
    fn rejects_non_positive_resolution() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        assert!(route_problem(&srj, Some(0.0), RouterKind::Ripup).is_err());
        assert!(route_problem(&srj, Some(-1.0), RouterKind::Ripup).is_err());
    }

    #[test]
    fn parse_srj_rejects_garbage() {
        assert!(parse_srj(b"not json").is_err());
    }

    #[test]
    fn project_large_batch_is_go() {
        let p = project(256, 256, 500);
        assert!(p.go, "large batch should be GO, got {}", p.speedup);
        assert!(p.speedup > GO_NO_GO_THRESHOLD);
    }

    #[test]
    fn project_tiny_single_net_is_no_go() {
        let p = project(8, 8, 1);
        assert!(!p.go, "tiny single net should be NO-GO, got {}", p.speedup);
        assert!(p.speedup < GO_NO_GO_THRESHOLD);
    }

    #[test]
    fn summary_display_is_one_line() {
        let s = Summary {
            routed: 2,
            total: 3,
            total_cost: 42,
            grid_w: 10,
            grid_h: 8,
        };
        let text = s.to_string();
        assert!(!text.contains('\n'));
        assert!(text.contains("2/3"));
        assert!(text.contains("10x8"));
    }

    #[test]
    fn handoff_builds_expected_bon_invocation() {
        let args = HandoffArgs {
            pcb: PathBuf::from("board.kicad_pcb"),
            passes: 12,
            timeout: 300,
            bon_command: "bon".into(),
        };
        let runner = mr_bridge::MockRunner::ok();
        let out = handoff_with(&runner, &args).unwrap();
        assert!(out.status_ok);
        let (program, argv) = runner.last().expect("invocation recorded");
        assert_eq!(program, "bon");
        assert!(argv.contains(&"board.kicad_pcb".to_string()));
        assert!(argv.contains(&"12".to_string()));
        assert!(argv.contains(&"300".to_string()));
    }

    #[test]
    fn handoff_propagates_backend_failure() {
        let args = HandoffArgs {
            pcb: PathBuf::from("b.kicad_pcb"),
            passes: 20,
            timeout: 600,
            bon_command: "bon".into(),
        };
        let runner = mr_bridge::MockRunner::failing("freerouting crashed");
        assert!(handoff_with(&runner, &args).is_err());
    }

    #[test]
    fn projection_display_shows_verdict() {
        assert!(project(256, 256, 500).to_string().contains("GO"));
        assert!(project(8, 8, 1).to_string().contains("NO-GO"));
    }
}
