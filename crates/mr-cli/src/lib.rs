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
pub mod corpus;
pub mod drc;
use std::collections::HashMap;

use mr_core::{BoardRoute, CellIdx, Grid, GridCoords, LayerMap, NetEndpoints, Router, ViaModel};
use serde::{Deserialize, Serialize};
use mr_cpu::{LeeRouter, NegotiatedRouter, RipUpRouter};
use mr_ingest::dsn::{dsn_to_ingest, DsnIngest, ParseStats};
use mr_srj::{rasterize_with_layers, to_solution_layered, Mapping, RoutePoint, SimpleRouteJson};

/// The ~2× speedup threshold the M2 go/no-go gate uses.
pub const GO_NO_GO_THRESHOLD: f32 = 2.0;

/// Default trace width (continuous units) for emitted `pcb_trace` wires.
const DEFAULT_TRACE_WIDTH: f64 = 0.15;

/// Signal-via geometry (bon's default): 0.45 mm annular pad over a 0.2 mm drill.
/// Shared by the SES exporter and the DRC builder so both agree on via copper.
pub const VIA_PAD_MM: f64 = 0.45;
pub const VIA_DRILL_MM: f64 = 0.2;

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
    /// Route a Specctra `.dsn` board with the CPU router and report connectivity.
    RouteDsn(RouteDsnArgs),
    /// Route a Specctra `.dsn` board, run the native DRC checker, and report (and
    /// optionally write) a violation report.
    Drc(drc::DrcArgs),
    /// Route the vendored corpus of real circuit-derived boards (`benchmarks/corpus/`),
    /// report per-board completion, and optionally render an SVG gallery.
    BenchCorpus(corpus::CorpusArgs),
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
    Ripup,
    /// PathFinder-style negotiated-congestion router (cell-disjoint across groups).
    #[default]
    Negotiated,
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

    /// Number of copper layers to route on. Defaults to the problem's declared
    /// `layerCount`. An override lets you grant extra layers to a board that
    /// declares fewer (only the `negotiated` backend places vias between layers).
    #[arg(long)]
    pub layers: Option<u32>,

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

/// Why a net was left unrouted, diagnosed by re-routing it in isolation on the
/// base grid (all other nets absent).
///
/// This is the headline diagnostic: it separates failures the *algorithm* could
/// in principle fix (contention) from failures rooted in the grid itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnroutedReason {
    /// The net has no path even on an otherwise-empty board at this resolution —
    /// a geometry/resolution limit (e.g. a pad walled off by neighbours, or a
    /// gap too narrow to fit a cell), not contention. Points at resolution levers.
    UnroutableAlone,
    /// The net routes fine in isolation; the multi-net router lost it to
    /// congestion (other nets' committed copper + clearance). Points at the
    /// routing algorithm (net ordering, rip-up, global planning).
    Congested,
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
    /// Number of copper layers routed on.
    pub grid_layers: u32,
    /// Each unrouted net's name paired with its diagnosed [`UnroutedReason`].
    /// Empty on a fully-routed board.
    pub unrouted: Vec<(String, UnroutedReason)>,
}

/// Rendering-side diagnostics for one routed board: the per-cell congestion field
/// plus the non-uniform grid-line coordinates needed to place each cell back in
/// continuous board space (so the gallery can draw a faithful heatmap on the
/// Hanan grid). Kept separate from [`Summary`] because the `f64` line arrays are
/// not `Eq`, and because only the corpus gallery consumes them.
#[derive(Debug, Clone, Default)]
pub struct RouteDiagnostics {
    /// Per-cell occupancy (length == `grid_w * grid_h * grid_layers`): how many
    /// routed nets pass through each cell. Summed across layers for the heatmap.
    pub congestion: Vec<u32>,
    /// Sorted x grid-line coordinates (continuous board units); `len() == grid_w`.
    pub x_lines: Vec<f64>,
    /// Sorted y grid-line coordinates (continuous board units); `len() == grid_h`.
    pub y_lines: Vec<f64>,
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "routed {}/{} nets, total cost {}, grid {}x{}x{}L",
            self.routed, self.total, self.total_cost, self.grid_w, self.grid_h, self.grid_layers
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
    layers: Option<u32>,
) -> Result<(Vec<mr_srj::PcbTrace>, Summary, RouteDiagnostics)> {
    let resolution = resolution.unwrap_or_else(|| default_resolution(srj));
    anyhow::ensure!(
        resolution.is_finite() && resolution > 0.0,
        "resolution must be a finite positive number, got {resolution}"
    );

    // Effective layer count: the override if given, else the problem's declaration.
    // Standard tscircuit naming (top/inner_N/bottom) applies for SimpleRouteJson.
    let layer_count = layers.unwrap_or(srj.layer_count).max(1);
    let layer_map = LayerMap::standard(layer_count);
    // SimpleRouteJson route_problem does not apply pad clearance (that is the DSN
    // pipeline's concern, which carries a min-clearance rule); pass 0.
    let problem = rasterize_with_layers(srj, resolution, layer_map, 0);
    let total = problem.nets.len();

    // Only the negotiated backend places vias; give it a through-hole model over
    // the routed stackup. Lee/Ripup route per-layer with no layer changes.
    let via_model = ViaModel::through_hole(problem.mapping.dims.layers);
    // The board's continuous grid-line geometry, so the negotiated router prices
    // planar steps by their real length. On a uniform grid this is byte-identical to
    // the unit-hop fallback; on a non-uniform / Hanan grid it makes the cost track
    // the true pitch.
    let coords =
        GridCoords::from_lines(problem.mapping.x_lines.clone(), problem.mapping.y_lines.clone());
    let board = match router {
        RouterKind::Lee => LeeRouter::new().route(&problem.grid, &problem.nets),
        RouterKind::Ripup => RipUpRouter::new().route(&problem.grid, &problem.nets),
        RouterKind::Negotiated => NegotiatedRouter::new()
            .with_via_model(via_model.clone())
            .with_coords(coords.clone())
            .route(&problem.grid, &problem.nets),
    }
    .context("router failed")?;

    let traces = to_solution_layered(
        &board,
        &problem.mapping,
        &problem.pin_points,
        DEFAULT_TRACE_WIDTH,
        &problem.layers,
    );
    // Beautify the emitted geometry: pull staircases into diagonals and chamfer
    // square corners. DRC-validated against all other copper/pads, so it never
    // changes connectivity or introduces a clearance violation.
    let traces = mr_srj::beautify_traces(traces, &srj.obstacles, srj.min_clearance.unwrap_or(0.0));

    // Diagnose every unrouted net: was it impossible at this resolution, or just
    // lost to congestion? Cheap — re-routes only the failed nets, one at a time.
    let unrouted =
        classify_unrouted(router, &problem.grid, &problem.nets, &board.unrouted, &via_model, &coords);

    let summary = Summary {
        routed: board.results.len(),
        total,
        total_cost: board.total_cost(),
        grid_w: problem.mapping.dims.w,
        grid_h: problem.mapping.dims.h,
        grid_layers: problem.mapping.dims.layers,
        unrouted,
    };

    let diagnostics = RouteDiagnostics {
        congestion: board.congestion,
        x_lines: problem.mapping.x_lines.clone(),
        y_lines: problem.mapping.y_lines.clone(),
    };

    Ok((traces, summary, diagnostics))
}

/// Diagnose every unrouted net by re-routing it **alone** on the base grid.
///
/// Each name in `unrouted` is submitted as the sole net to the same backend that
/// routed the board, so its own pads are unmasked and full capability (vias, for
/// the negotiated backend) is available — exactly the conditions the multi-net
/// router had, minus every other net's copper. If the net routes in isolation the
/// original failure was contention ([`UnroutedReason::Congested`]); if it still
/// can't route, the grid itself blocks it ([`UnroutedReason::UnroutableAlone`]).
///
/// Runs sequentially: the corpus harness already fans boards out across cores, so
/// nesting rayon here would only oversubscribe. The per-net count is small.
fn classify_unrouted(
    router: RouterKind,
    grid: &Grid,
    nets: &[NetEndpoints],
    unrouted: &[String],
    via_model: &ViaModel,
    coords: &GridCoords,
) -> Vec<(String, UnroutedReason)> {
    let by_name: HashMap<&str, &NetEndpoints> = nets.iter().map(|n| (n.net.as_str(), n)).collect();

    unrouted
        .iter()
        .map(|name| {
            let routes_alone = by_name.get(name.as_str()).is_some_and(|net| {
                let solo = std::slice::from_ref(*net);
                let res = match router {
                    RouterKind::Lee => LeeRouter::new().route(grid, solo),
                    RouterKind::Ripup => RipUpRouter::new().route(grid, solo),
                    RouterKind::Negotiated => NegotiatedRouter::new()
                        .with_via_model(via_model.clone())
                        .with_coords(coords.clone())
                        .route(grid, solo),
                };
                matches!(res, Ok(b) if b.unrouted.is_empty() && !b.results.is_empty())
            });
            let reason = if routes_alone {
                UnroutedReason::Congested
            } else {
                UnroutedReason::UnroutableAlone
            };
            (name.clone(), reason)
        })
        .collect()
}

/// Execute the `route` subcommand: read the input file, route, write the
/// solution JSON to `--out` (or stdout), and return the [`Summary`].
///
/// The caller is responsible for printing the summary (to stderr).
pub fn run_route(args: &RouteArgs) -> Result<Summary> {
    let bytes = std::fs::read(&args.input)
        .with_context(|| format!("failed to read input file {}", args.input.display()))?;
    let srj = parse_srj(&bytes)?;

    let (traces, summary, _diag) = route_problem(&srj, args.resolution, args.router, args.layers)?;

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

// ---------------------------------------------------------------------------
// route-dsn: route a real Specctra DSN board end-to-end
// ---------------------------------------------------------------------------

/// Target cells across the longer board span when deriving a default resolution
/// for `route-dsn` (mirrors `mr_server::choose_resolution`'s policy).
const DSN_TARGET_CELLS_PER_AXIS: f64 = 200.0;

/// Floor on the derived cell size (mm) for `route-dsn`.
const DSN_MIN_RESOLUTION: f64 = 0.1;

/// Arguments for the `route-dsn` subcommand.
#[derive(Debug, Parser)]
pub struct RouteDsnArgs {
    /// Path to the input Specctra `.dsn` file.
    #[arg(long)]
    pub input: PathBuf,

    /// Cell size in mm. Defaults to a value derived from the board bounds.
    #[arg(long)]
    pub resolution: Option<f64>,

    /// Output path for the routed solution (JSON `pcb_trace` soup).
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Skip nets whose name contains this substring (repeatable). Useful for
    /// plane nets (GND, +5VA, 3V3, ...) a single-layer router can't sanely route.
    #[arg(long = "skip-nets")]
    pub skip_nets: Vec<String>,

    /// Cap the number of original nets routed (for quick smoke tests).
    #[arg(long)]
    pub max_nets: Option<usize>,

    /// Number of signal layers to route on. Defaults to all `(type signal)`
    /// layers in the DSN stackup; a smaller value uses the top-N signal layers.
    #[arg(long)]
    pub layers: Option<u32>,

    /// Also write a Specctra session (`.ses`) of the routed copper to this path,
    /// ready to import back onto the source board (`bon route DIR --import-ses`).
    #[arg(long)]
    pub ses: Option<PathBuf>,

    /// After routing, run the native DRC checker and print a violation summary
    /// (clearance, via-through-plane, annular-ring) to stderr.
    #[arg(long, default_value_t = false)]
    pub drc: bool,
}

/// Resolution policy for `route-dsn`: honour a finite positive override, else
/// derive from bounds, floored at [`DSN_MIN_RESOLUTION`] and capped at ~2 trace
/// widths so traces can fit between pads.
fn dsn_resolution(srj: &SimpleRouteJson, override_res: Option<f64>) -> f64 {
    if let Some(r) = override_res {
        if r.is_finite() && r > 0.0 {
            return r;
        }
    }
    let b = &srj.bounds;
    let max_span = (b.max_x - b.min_x)
        .max(0.0)
        .max((b.max_y - b.min_y).max(0.0));
    if max_span <= 0.0 {
        return DSN_MIN_RESOLUTION;
    }
    let mut res = (max_span / DSN_TARGET_CELLS_PER_AXIS).max(DSN_MIN_RESOLUTION);
    if let Some(w) = srj.min_trace_width {
        if w.is_finite() && w > 0.0 {
            res = res.min((w * 2.0).max(DSN_MIN_RESOLUTION));
        }
    }
    res
}

/// Connectivity + timing report for a `route-dsn` run.
#[derive(Debug, Clone, PartialEq)]
pub struct DsnReport {
    /// Parse stats from the DSN ingest.
    pub stats: ParseStats,
    /// Resolution (mm) used to rasterise.
    pub resolution: f64,
    /// Grid width in cells.
    pub grid_w: u32,
    /// Grid height in cells.
    pub grid_h: u32,
    /// Number of copper layers routed on.
    pub grid_layers: u32,
    /// Total two-point nets submitted (after k-point decomposition + filtering).
    pub total_nets: usize,
    /// Two-point nets that produced a routed trace.
    pub routed_nets: usize,
    /// Number of vias placed across all routed traces (layer changes).
    pub vias: usize,
    /// Original (pre-decomposition) connections that routed fully (all segments).
    pub fully_connected: usize,
    /// Original connections submitted (after skip/cap filtering).
    pub original_nets: usize,
    /// Wall-clock seconds spent inside the router.
    pub wall_s: f64,
}

impl DsnReport {
    /// Connectivity percentage = routed two-point nets / total two-point nets.
    pub fn connectivity_pct(&self) -> f64 {
        if self.total_nets == 0 {
            0.0
        } else {
            self.routed_nets as f64 / self.total_nets as f64 * 100.0
        }
    }

    /// Nets routed per wall-clock second (0 if no measurable time).
    pub fn nets_per_sec(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.routed_nets as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// The scrape-friendly one-line `RESULT` summary.
    pub fn result_line(&self) -> String {
        format!(
            "RESULT route-dsn nets={} routed={} conn={:.1}% vias={} wall={:.3}s grid={}x{}x{}L",
            self.total_nets,
            self.routed_nets,
            self.connectivity_pct(),
            self.vias,
            self.wall_s,
            self.grid_w,
            self.grid_h,
            self.grid_layers,
        )
    }
}

/// Drop interior points that are collinear with their neighbours, so a straight
/// cell-by-cell run collapses to its two endpoints (and each corner is kept).
/// Turns thousands of unit-step vertices into a handful of real segments.
fn simplify_collinear(pts: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let mut out = vec![pts[0]];
    for i in 1..pts.len() - 1 {
        let a = *out.last().unwrap();
        let b = pts[i];
        let c = pts[i + 1];
        // Cross product of (b-a) and (c-b); zero ⇒ b lies on the a→c line.
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        if cross != 0 {
            out.push(b);
        }
    }
    out.push(*pts.last().unwrap());
    out
}

/// Build a Specctra session (`.ses`) from a routed board, ready to import back
/// onto the source KiCad PCB (e.g. `bon route DIR --import-ses`).
///
/// Coordinates are the inverse of the DSN ingest: a continuous-mm value becomes
/// `round(mm * units_per_mm)` in the DSN's own raw units, and the y-sign carried
/// through ingest is preserved (the importer re-negates it into KiCad's y-down
/// frame). Tracks are grouped under their base net name (the `#seg`
/// decomposition suffix is stripped); a path's layer changes become `(via ...)`
/// entries between `(wire ...)` runs. Via dimensions are encoded in the padstack
/// name (`Via[..]_<size_um>:<drill_um>_um`) per the importer's convention.
#[allow(clippy::too_many_arguments)]
fn board_to_ses(
    design_name: &str,
    board: &BoardRoute,
    mapping: &Mapping,
    layers: &LayerMap,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    units_per_mm: f64,
    unit: &str,
    divisor: f64,
    trace_width_mm: f64,
) -> String {
    let dims = mapping.dims;
    let to_raw = |mm: f64| (mm * units_per_mm).round() as i64;
    let width_raw = to_raw(trace_width_mm);
    // Signal via: 0.45 mm pad / 0.2 mm drill (bon's default), encoded for the
    // importer's `Via[..]_<size_um>:<drill_um>_um` regex.
    const VIA_NAME: &str = "Via[0-7]_450:200_um";
    let via_pad_raw = to_raw(VIA_PAD_MM);

    // Endpoint vertices snap to the exact port; interior vertices use cell centres.
    let point = |cell: CellIdx, endpoint: bool| -> (f64, f64) {
        if endpoint {
            if let Some(p) = pin_points.get(&cell) {
                return *p;
            }
        }
        mapping.cell_center(cell)
    };

    // Group routed segments by base net name, preserving first-seen order.
    let mut nets: Vec<(String, Vec<&[CellIdx]>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for r in &board.results {
        let base = r.net.split('#').next().unwrap_or(&r.net).to_string();
        let i = *index.entry(base.clone()).or_insert_with(|| {
            nets.push((base.clone(), Vec::new()));
            nets.len() - 1
        });
        nets[i].1.push(r.path.as_slice());
    }

    let mut out = String::new();
    out.push_str(&format!("(session \"{design_name}.ses\"\n"));
    out.push_str(&format!("  (base_design \"{design_name}.dsn\")\n"));
    out.push_str("  (routes\n");
    out.push_str(&format!("    (resolution {unit} {divisor})\n"));
    out.push_str("    (library_out\n");
    out.push_str(&format!(
        "      (padstack \"{VIA_NAME}\" (shape (circle F.Cu {via_pad_raw})))\n"
    ));
    out.push_str("    )\n");
    out.push_str("    (network_out\n");

    for (net, paths) in &nets {
        out.push_str(&format!("      (net \"{net}\"\n"));
        for path in paths {
            if path.is_empty() {
                continue;
            }
            let last = path.len() - 1;
            let mut cur_layer = dims.layer_of(path[0]);
            let p0 = point(path[0], true);
            let mut run: Vec<(i64, i64)> = vec![(to_raw(p0.0), to_raw(p0.1))];
            let flush = |out: &mut String, layer: u32, run: &[(i64, i64)]| {
                let run = simplify_collinear(run);
                if run.len() < 2 {
                    return;
                }
                out.push_str(&format!(
                    "        (wire (path {} {width_raw}",
                    layers.name(layer)
                ));
                for (x, y) in &run {
                    out.push_str(&format!(" {x} {y}"));
                }
                out.push_str(") (type route))\n");
            };
            for k in 1..path.len() {
                let l = dims.layer_of(path[k]);
                if l == cur_layer {
                    let (x, y) = point(path[k], k == last);
                    run.push((to_raw(x), to_raw(y)));
                } else {
                    // Layer change: a via at the shared (x, y) of path[k-1]/path[k].
                    // We emit only the via padstack + position, tagged with its net
                    // (it sits inside this `(net ...)` block). Plane antipads are NOT
                    // emitted as explicit geometry: on import, KiCad's zone fill
                    // reliefs a foreign-net via automatically (the via's net is
                    // preserved, so each poured plane carves its own antipad). The DRC
                    // model in `drc::build_drc_board` mirrors that relief; the
                    // `kicad-cli` cross-check validates the two agree.
                    let (vx, vy) = mapping.cell_center(path[k]);
                    let (vrx, vry) = (to_raw(vx), to_raw(vy));
                    flush(&mut out, cur_layer, &run);
                    out.push_str(&format!("        (via \"{VIA_NAME}\" {vrx} {vry})\n"));
                    cur_layer = l;
                    let (x, y) = point(path[k], k == last);
                    run = vec![(to_raw(x), to_raw(y))];
                }
            }
            flush(&mut out, cur_layer, &run);
        }
        out.push_str("      )\n");
    }

    out.push_str("    )\n  )\n)\n");
    out
}

/// Count the vias (layer-change points) across a routed solution soup.
fn count_vias(traces: &[mr_srj::PcbTrace]) -> usize {
    traces
        .iter()
        .flat_map(|t| t.route.iter())
        .filter(|p| matches!(p, RoutePoint::Via { .. }))
        .count()
}

/// Restrict a parsed stackup to the top `n` layers, rebuilding a through-hole via
/// model over them. `None` (or `n >= len`) keeps the full stackup and its model.
///
/// Retained as the canonical stackup-restriction helper; `route_dsn_problem`
/// currently inlines the equivalent truncation over the signal-layer list.
#[allow(dead_code)]
fn apply_layer_override(
    layer_map: LayerMap,
    via_model: ViaModel,
    n: Option<u32>,
) -> (LayerMap, ViaModel) {
    match n {
        Some(n) if n >= 1 && n < layer_map.len() => {
            let names: Vec<String> = (0..n).map(|i| layer_map.name(i).to_string()).collect();
            (LayerMap::from_names(names), ViaModel::through_hole(n))
        }
        _ => (layer_map, via_model),
    }
}

/// Core `route-dsn` logic: convert a parsed DSN to a problem, route it, and build
/// a [`DsnReport`]. Returns the report plus the routed solution soup.
///
/// `skip_nets` drops any connection whose name contains one of the substrings;
/// `max_nets` caps the number of (post-skip) original connections routed.
///
/// `model_plane_antipads` is forwarded to [`drc::build_drc_board`]: `true` models
/// the poured-zone relief on foreign through-vias (the realistic default), `false`
/// treats planes as bare copper so every crossing shorts.
#[allow(clippy::too_many_arguments)]
pub fn route_dsn_problem(
    ingest: DsnIngest,
    design_name: &str,
    resolution: Option<f64>,
    skip_nets: &[String],
    max_nets: Option<usize>,
    layers: Option<u32>,
    model_plane_antipads: bool,
) -> Result<(DsnReport, Vec<mr_srj::PcbTrace>, String, mr_drc::DrcBoard)> {
    let units_per_mm = ingest.units_per_mm();
    let res_unit = ingest.resolution_unit.clone();
    let res_divisor = ingest.resolution_divisor;
    let DsnIngest {
        mut srj,
        signal_layers,
        stats,
        layer_map: physical_layers,
        planes,
        pin_nets,
        ..
    } = ingest;
    // Filter connections: drop skipped substrings, then cap.
    if !skip_nets.is_empty() {
        srj.connections
            .retain(|c| !skip_nets.iter().any(|s| c.name.contains(s.as_str())));
    }
    if let Some(cap) = max_nets {
        srj.connections.truncate(cap);
    }
    let original_nets = srj.connections.len();

    let resolution = dsn_resolution(&srj, resolution);
    anyhow::ensure!(
        resolution.is_finite() && resolution > 0.0,
        "resolution must be finite and positive, got {resolution}"
    );

    // Route signal nets on the SIGNAL layers only (never on a poured power plane);
    // vias bridge adjacent signal layers as through-vias. `--layers` caps how many
    // signal layers are used. The via model is through-hole over those layers.
    let mut signal_layers = signal_layers;
    if let Some(n) = layers {
        let n = (n as usize).clamp(1, signal_layers.len().max(1));
        signal_layers.truncate(n);
    }
    let layer_map = LayerMap::from_names(signal_layers);
    // Clearance enforcement (M3): the DSN `(rule (clearance N))` is now honoured in
    // cell space. `clearance_cells = ceil(min_clearance / resolution)` is the
    // copper-to-copper halo width in cells. It is enforced in two places:
    //   * the negotiation search (`with_clearance_cells`, the parallel agent's side)
    //     keeps tracks of different nets that many cells apart; and
    //   * pad rasterisation (`rasterize_with_layers`, this crate's `mr-srj`) reserves
    //     the same halo around every pad while still letting each net escape its own
    //     pads via `passable_pads`.
    // Committed vias likewise reserve a keepout halo (`ViaModel.keepout`) sized for
    // the via pad plus clearance. This supersedes the M2.4 "disabled legalization
    // halo" experiment: clearance now lives in the negotiation phase + the pad/via
    // grid, not a post-hoc legalization fold. Plane-antipad modelling (the via-
    // through-plane fix) is independent and stays on.
    let clearance_cells = if stats.min_clearance_mm > 0.0 && resolution > 0.0 {
        (stats.min_clearance_mm / resolution).ceil() as u32
    } else {
        0
    };
    let mut via_model = ViaModel::through_hole(layer_map.len());
    via_model.keepout = if resolution > 0.0 {
        ((VIA_PAD_MM / 2.0 + stats.min_clearance_mm) / resolution).ceil() as u32
    } else {
        0
    };
    let problem = rasterize_with_layers(&srj, resolution, layer_map, clearance_cells);
    let total_nets = problem.nets.len();
    let grid_w = problem.mapping.dims.w;
    let grid_h = problem.mapping.dims.h;
    let grid_layers = problem.mapping.dims.layers;

    // Continuous grid-line geometry: prices planar steps by real length (uniform-grid
    // byte-identical, Hanan-grid pitch-aware).
    let coords =
        GridCoords::from_lines(problem.mapping.x_lines.clone(), problem.mapping.y_lines.clone());
    let start = std::time::Instant::now();
    let board = NegotiatedRouter::new()
        .with_via_model(via_model)
        // Geometric clearance over the (possibly non-uniform) line arrays — the same
        // mm budget the rasteriser inflated foreign pads by — so inter-net spacing is
        // a real distance, not a cell count that varies with local pitch.
        .with_clearance_mm(stats.min_clearance_mm)
        .with_coords(coords)
        .route(&problem.grid, &problem.nets)
        .context("router failed")?;
    let wall_s = start.elapsed().as_secs_f64();

    let routed_nets = board.results.len();

    // A connection is fully connected iff every one of its k-1 chained segments
    // routed. Segment net names are `<conn>` (k==2) or `<conn>#<seg>` (k>2).
    let routed_names: std::collections::HashSet<&str> =
        board.results.iter().map(|r| r.net.as_str()).collect();
    let mut fully_connected = 0usize;
    for conn in &srj.connections {
        let segments = conn.points_to_connect.len().saturating_sub(1);
        if segments == 0 {
            continue;
        }
        let all = if segments == 1 {
            routed_names.contains(conn.name.as_str())
        } else {
            (0..segments)
                .all(|seg| routed_names.contains(format!("{}#{}", conn.name, seg).as_str()))
        };
        if all {
            fully_connected += 1;
        }
    }

    let trace_width = srj.min_trace_width.unwrap_or(DEFAULT_TRACE_WIDTH);
    let traces = to_solution_layered(
        &board,
        &problem.mapping,
        &problem.pin_points,
        trace_width,
        &problem.layers,
    );
    // Beautify the emitted JSON soup: 45° chamfers + diagonalized staircases,
    // DRC-validated against all other copper/pads so it never changes connectivity
    // or introduces a violation. (The .ses below is still built from cell-space
    // `board`, so KiCad reimport stays on the routed grid.)
    let traces = mr_srj::beautify_traces(traces, &srj.obstacles, stats.min_clearance_mm);
    let vias = count_vias(&traces);

    let ses = board_to_ses(
        design_name,
        &board,
        &problem.mapping,
        &problem.layers,
        &problem.pin_points,
        units_per_mm,
        &res_unit,
        res_divisor,
        trace_width,
    );

    // Build the physical DRC model: routed copper on the SIGNAL grid, mapped onto
    // the FULL stackup so a through-via's barrel is seen crossing the inner planes.
    let drc_board = drc::build_drc_board(
        &board,
        &problem.mapping,
        &problem.layers,
        &physical_layers,
        &planes,
        &srj.obstacles,
        &pin_nets,
        trace_width,
        drc::default_rules(stats.min_clearance_mm),
        model_plane_antipads,
    );

    let report = DsnReport {
        stats,
        resolution,
        grid_w,
        grid_h,
        grid_layers,
        total_nets,
        routed_nets,
        vias,
        fully_connected,
        original_nets,
        wall_s,
    };

    Ok((report, traces, ses, drc_board))
}

/// Execute the `route-dsn` subcommand: read + parse the DSN, route it, optionally
/// write the solution, and return the [`DsnReport`]. The caller prints it.
pub fn run_route_dsn(args: &RouteDsnArgs) -> Result<DsnReport> {
    let text = std::fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read DSN file {}", args.input.display()))?;
    let ingest = dsn_to_ingest(&text).context("failed to convert DSN to problem")?;

    let design_name = args
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fixture".to_string());

    let (report, traces, ses, drc_board) = route_dsn_problem(
        ingest,
        &design_name,
        args.resolution,
        &args.skip_nets,
        args.max_nets,
        args.layers,
        // route-dsn assumes poured-zone planes (the realistic default); the `drc`
        // subcommand exposes `--no-plane-zones` to opt into the bare-copper model.
        true,
    )?;

    if let Some(path) = &args.out {
        let json = serde_json::to_string_pretty(&traces).context("failed to serialise solution")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write output file {}", path.display()))?;
    }

    if let Some(path) = &args.ses {
        std::fs::write(path, &ses)
            .with_context(|| format!("failed to write SES file {}", path.display()))?;
    }

    if args.drc {
        let summary = mr_drc::DrcSummary::of(&drc_board.check());
        eprintln!(
            "DRC: {} violation(s) — {} clearance, {} via-through-plane, {} annular-ring",
            summary.total, summary.clearance, summary.via_through_plane, summary.annular_ring,
        );
    }

    Ok(report)
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
        let (traces, summary, _diag) =
            route_problem(&srj, Some(1.0), RouterKind::Ripup, None).unwrap();

        // Two 2-point connections -> two nets, both routable on this open board.
        assert_eq!(summary.total, 2);
        assert_eq!(summary.routed, 2);
        // Non-uniform / Hanan grid (Phase 3): lines fall on the bounds, every pad
        // endpoint, every obstacle edge ({4,6}), plus fill channels. `fill_lines` now
        // adds a midpoint lane in every gap ≥ channel (the coverage fix), so a unit gap
        // (e.g. 0↔1) gets its midpoint 0.5 and 3-wide gaps (e.g. 1↔4) get 2,3 — the
        // union is {0,0.5,1,2,3,4,5,6,7,8,9,9.5,10} = 13 lines per axis.
        assert_eq!(summary.grid_w, 13);
        assert_eq!(summary.grid_h, 13);
        assert!(summary.total_cost > 0);

        assert_eq!(traces.len(), 2);
        for t in &traces {
            assert_eq!(t.kind, "pcb_trace");
            assert!(!t.route.is_empty());
        }
    }

    /// A net whose only corridor is walled off on the top layer. The wall sits on
    /// `"top"` only, so on a single layer the net cannot route; granting a second
    /// layer lets the negotiated router via down, cross, and via back up.
    const TWO_LAYER_WALL: &str = r#"{
        "layerCount": 1,
        "minTraceWidth": 0.1,
        "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 6 },
        "obstacles": [
            { "type": "rect", "layers": ["top"], "center": {"x": 5, "y": 3}, "width": 2, "height": 6 }
        ],
        "connections": [
            { "name": "SIG", "pointsToConnect": [
                {"x": 1, "y": 3, "layer": "top"},
                {"x": 9, "y": 3, "layer": "top"}
            ] }
        ]
    }"#;

    #[test]
    fn single_layer_wall_blocks_but_second_layer_vias_through() {
        let srj = parse_srj(TWO_LAYER_WALL.as_bytes()).unwrap();

        // Declared single layer: the top-layer wall is impassable.
        let (traces1, s1, _d1) =
            route_problem(&srj, Some(1.0), RouterKind::Negotiated, None).unwrap();
        assert_eq!(s1.grid_layers, 1);
        assert_eq!(s1.routed, 0, "net must be unroutable on one layer");
        assert_eq!(count_vias(&traces1), 0);

        // Grant a second layer: the net routes and must change layers (>=2 vias:
        // down before the wall, back up after).
        let (traces2, s2, _d2) =
            route_problem(&srj, Some(1.0), RouterKind::Negotiated, Some(2)).unwrap();
        assert_eq!(s2.grid_layers, 2);
        assert_eq!(s2.routed, 1, "net should route once a second layer exists");
        assert!(
            count_vias(&traces2) >= 2,
            "a top->bottom->top detour needs at least two vias, got {}",
            count_vias(&traces2)
        );
        // The emitted via names must come from the standard stackup.
        let via = traces2
            .iter()
            .flat_map(|t| &t.route)
            .find_map(|p| match p {
                RoutePoint::Via {
                    from_layer,
                    to_layer,
                    ..
                } => Some((from_layer.clone(), to_layer.clone())),
                _ => None,
            })
            .expect("at least one via");
        assert!(
            matches!(
                (via.0.as_str(), via.1.as_str()),
                ("top", "bottom") | ("bottom", "top")
            ),
            "via should span top<->bottom, got {via:?}"
        );
    }

    #[test]
    fn route_problem_lee_backend_also_routes() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        let (traces, summary, _diag) =
            route_problem(&srj, Some(1.0), RouterKind::Lee, None).unwrap();
        assert_eq!(summary.routed, 2);
        assert!(!traces.is_empty());
    }

    #[test]
    fn default_resolution_targets_reasonable_grid() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        // span 10 / 64 -> small cells; grid should be well-formed and non-trivial.
        let (_traces, summary, _diag) =
            route_problem(&srj, None, RouterKind::Ripup, None).unwrap();
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
        assert!(route_problem(&srj, Some(0.0), RouterKind::Ripup, None).is_err());
        assert!(route_problem(&srj, Some(-1.0), RouterKind::Ripup, None).is_err());
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
            grid_layers: 1,
            unrouted: Vec::new(),
        };
        let text = s.to_string();
        assert!(!text.contains('\n'));
        assert!(text.contains("2/3"));
        assert!(text.contains("10x8x1L"));
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

    /// A small synthetic DSN: 2 components on a 20x20mm board, one 2-pin net far
    /// from any obstacle, so the negotiated router should route it.
    const SYNTH_DSN: &str = r#"
    (pcb "rt.dsn"
      (parser (string_quote "))
      (resolution mm 1000)
      (structure
        (layer F.Cu (type signal))
        (boundary (path pcb 0 0 0 20000 0 20000 20000 0 20000 0 0))
        (rule (width 150))
      )
      (placement
        (component "img" (place A 3000 3000 front 0))
        (component "img" (place B 17000 17000 front 0))
      )
      (library
        (image "img" (pin "ps" 1 0 0))
        (padstack "ps" (shape (circle F.Cu 600 0 0)))
      )
      (network (net "N1" (pins A-1 B-1)))
    )
    "#;

    #[test]
    fn route_dsn_round_trip_routes_synthetic_board() {
        let ingest = dsn_to_ingest(SYNTH_DSN).unwrap();
        assert_eq!(ingest.srj.connections.len(), 1);
        let (report, traces, ses, _drc) =
            route_dsn_problem(ingest, "synth", Some(0.5), &[], None, None, true).unwrap();
        // The SES is well-formed and names the routed net.
        assert!(ses.contains("(session"));
        assert!(ses.contains("(net \"N1\""));
        assert_eq!(report.total_nets, 1);
        assert_eq!(report.routed_nets, 1, "open board net should route");
        assert_eq!(report.fully_connected, 1);
        assert!((report.connectivity_pct() - 100.0).abs() < 1e-9);
        assert_eq!(traces.len(), 1);
        assert!(report.result_line().contains("conn=100.0%"));
    }

    #[test]
    fn route_dsn_skip_and_cap_filter_connections() {
        // Two nets; skip one by substring, cap should also apply.
        let dsn = r#"
        (pcb "f.dsn"
          (parser (string_quote "))
          (resolution mm 1000)
          (structure
            (layer F.Cu (type signal))
            (boundary (path pcb 0 0 0 20000 0 20000 20000 0 20000 0 0))
          )
          (placement
            (component "img" (place A 3000 3000 front 0))
            (component "img" (place B 17000 17000 front 0))
            (component "img" (place C 3000 17000 front 0))
          )
          (library
            (image "img" (pin "ps" 1 0 0))
            (padstack "ps" (shape (circle F.Cu 600 0 0)))
          )
          (network
            (net "SIGNAL" (pins A-1 B-1))
            (net "GND" (pins A-1 C-1))
          )
        )
        "#;
        let ingest = dsn_to_ingest(dsn).unwrap();
        assert_eq!(ingest.srj.connections.len(), 2);
        // Skip GND -> only SIGNAL remains.
        let (report, _, _, _) = route_dsn_problem(
            ingest,
            "f",
            Some(0.5),
            &["GND".to_string()],
            None,
            None,
            true,
        )
        .unwrap();
        assert_eq!(report.original_nets, 1);
    }
}
