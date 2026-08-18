//! `mr-cli` — library half of the `metalroute` user-facing CLI.
//!
//! The real work lives here as plain functions so it can be unit/integration
//! tested without spawning a process; [`main`](../main.rs) is a thin clap
//! dispatcher over [`Cli`].
//!
//! Subcommands:
//!
//! * [`run_route`] — read a tscircuit [`SimpleRouteJson`]
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
mod drc_board;
mod via_repair;
use std::collections::HashMap;

use mr_core::{BoardRoute, CellIdx, Grid, GridCoords, LayerMap, NetEndpoints, Router, ViaModel};
use mr_cpu::{LeeRouter, NegotiatedRouter, RipUpRouter};
use mr_ingest::dsn::{dsn_to_ingest, DsnIngest, ParseStats};
use mr_srj::{
    rasterize_with_layers, rasterize_with_uniform_physical_rules, to_solution_layered, Mapping,
    RoutePoint, SimpleRouteJson,
};
use serde::{Deserialize, Serialize};

#[cfg(target_os = "macos")]
use mr_core::RouterError;
#[cfg(target_os = "macos")]
use mr_cpu::{IsolatedRouteProvider, IsolatedRouteRequest};
#[cfg(target_os = "macos")]
use mr_metal::{MetalEdgeCosts, MetalIsolatedRoute, MetalWindow};

/// The ~2× speedup threshold the M2 go/no-go gate uses.
pub const GO_NO_GO_THRESHOLD: f32 = 2.0;

/// Default trace width (continuous units) for emitted `pcb_trace` wires.
const DEFAULT_TRACE_WIDTH: f64 = 0.15;

/// Default copper-to-copper clearance (mm) applied when a board declares no
/// `minClearance`. The router and the DRC checker MUST agree on this default, else
/// the router enforces a different spacing than the checker verifies.
pub const DEFAULT_CLEARANCE_MM: f64 = 0.15;

/// Signal-via geometry (bon's default): 0.45 mm annular pad over a 0.2 mm drill.
/// Shared by the SES exporter and the DRC builder so both agree on via copper.
pub const VIA_PAD_MM: f64 = 0.45;
pub const VIA_DRILL_MM: f64 = 0.2;

/// Resolution used when comparing the severity of two authoritative DRC result
/// sets. One nanometre is far below the router's useful geometric resolution but
/// large enough to keep sub-nanometre distance jitter from turning a tie into an
/// accepted geometry change.
const DRC_SCORE_QUANTUM_MM: f64 = 1e-6;

/// Reserved pseudo-net used by the exact outline checker. Geometry transforms may
/// improve ordinary copper DRC, but not by introducing one from an edge-clean
/// baseline or worsening the aggregate profile of pre-existing findings.
const BOARD_EDGE_NET: &str = "__board_edge__";

/// Numerical tolerance shared by exact outline checks and the board-edge repair
/// gate.
const BOARD_EDGE_GEOMETRY_EPS_MM: f64 = 1e-9;

/// Stable identity available in the public DRC result. Location is deliberately
/// excluded: moving a vertex or via changes the checker-reported feature centroid,
/// even when it is the same physical finding being improved. Because `Violation`
/// carries no feature ids, multiple physical pairs sharing this key are necessarily
/// compared as a severity multiset rather than matched one by one.
type DrcFindingIdentity = (u8, u32, String, String);

fn drc_class_order(class: mr_drc::ViolationClass) -> u8 {
    match class {
        mr_drc::ViolationClass::Clearance => 0,
        mr_drc::ViolationClass::ViaThroughPlane => 1,
        mr_drc::ViolationClass::AnnularRing => 2,
    }
}

fn drc_finding_identity(violation: &mr_drc::Violation) -> DrcFindingIdentity {
    (
        drc_class_order(violation.class),
        violation.layer,
        violation.nets.0.clone(),
        violation.nets.1.clone(),
    )
}

fn drc_severity(violation: &mr_drc::Violation) -> u64 {
    let deficit = violation.required - violation.measured;
    if !deficit.is_finite() {
        return if deficit.is_sign_positive() {
            u64::MAX
        } else {
            0
        };
    }
    if deficit <= 0.0 {
        return 0;
    }
    (deficit / DRC_SCORE_QUANTUM_MM).round() as u64
}

/// Quantised DRC severity profiles grouped by stable finding identity, worst first.
/// Findings are already authoritative: `DrcBoard::check` has removed same-net pairs,
/// including all known-net immunity. In particular, an empty serialized net name
/// means "unknown", not "same net", so every returned finding remains in this map.
fn drc_severity_profiles(
    violations: &[mr_drc::Violation],
) -> std::collections::BTreeMap<DrcFindingIdentity, Vec<u64>> {
    let mut profiles = std::collections::BTreeMap::<DrcFindingIdentity, Vec<u64>>::new();
    for violation in violations {
        profiles
            .entry(drc_finding_identity(violation))
            .or_default()
            .push(drc_severity(violation));
    }
    for severity in profiles.values_mut() {
        severity.sort_unstable_by(|a, b| b.cmp(a));
    }
    profiles
}

fn board_edge_deficit(violation: &mr_drc::Violation) -> f64 {
    let deficit = violation.required - violation.measured;
    if deficit.is_nan() || deficit == f64::INFINITY {
        f64::INFINITY
    } else if deficit <= 0.0 || deficit == f64::NEG_INFINITY {
        0.0
    } else {
        deficit
    }
}

fn board_edge_deficit_profiles(
    violations: &[mr_drc::Violation],
) -> std::collections::BTreeMap<DrcFindingIdentity, Vec<f64>> {
    let mut profiles = std::collections::BTreeMap::<DrcFindingIdentity, Vec<f64>>::new();
    for violation in violations.iter().filter(|violation| {
        violation.nets.0 == BOARD_EDGE_NET || violation.nets.1 == BOARD_EDGE_NET
    }) {
        profiles
            .entry(drc_finding_identity(violation))
            .or_default()
            .push(board_edge_deficit(violation));
    }
    for deficits in profiles.values_mut() {
        deficits.sort_unstable_by(|a, b| b.total_cmp(a));
    }
    profiles
}

fn board_edge_deficit_is_not_worse(after: f64, before: f64) -> bool {
    match (after.is_infinite(), before.is_infinite()) {
        (true, false) => false,
        (_, true) => true,
        (false, false) => after <= before || after - before <= BOARD_EDGE_GEOMETRY_EPS_MM,
    }
}

/// Board-edge safety is a constraint above the ordinary total-finding score. An
/// edge-clean baseline therefore guarantees an edge-clean candidate. When the
/// baseline already contains edge findings, the public `Violation` has no feature
/// id, so this can only require each retained identity/multiplicity deficit rank to
/// be no worse; it does not claim physical-feature correspondence. This aggregate
/// gate still prevents a transform from trading copper findings for a more severe
/// outline profile under the same public identities.
fn board_edge_findings_are_not_worse(
    before: &[mr_drc::Violation],
    candidate: &[mr_drc::Violation],
) -> bool {
    let before = board_edge_deficit_profiles(before);
    let candidate = board_edge_deficit_profiles(candidate);
    for (identity, candidate_severity) in &candidate {
        let Some(before_severity) = before.get(identity) else {
            return false;
        };
        if candidate_severity.len() > before_severity.len() {
            return false;
        }
        if candidate_severity
            .iter()
            .zip(before_severity)
            .any(|(&after, &before)| !board_edge_deficit_is_not_worse(after, before))
        {
            return false;
        }
    }
    true
}

/// Whether a candidate is an unambiguous authoritative DRC improvement.
///
/// Fewer findings remains the primary objective. At equal count, identities and
/// multiplicities must be unchanged, no worst-first severity rank may worsen, and at
/// least one rank must improve. The strict equal-count rule rejects new violations,
/// equal-score substitutions, and severity trade-offs instead of accepting geometry
/// churn on count alone.
fn drc_candidate_is_not_worse(
    before: &[mr_drc::Violation],
    candidate: &[mr_drc::Violation],
) -> bool {
    if !board_edge_findings_are_not_worse(before, candidate) {
        return false;
    }
    if candidate.len() != before.len() {
        return candidate.len() < before.len();
    }

    let before = drc_severity_profiles(before);
    let candidate = drc_severity_profiles(candidate);
    if before.keys().ne(candidate.keys()) {
        return false;
    }
    for (identity, before_severity) in &before {
        let candidate_severity = &candidate[identity];
        if candidate_severity.len() != before_severity.len() {
            return false;
        }
        for (after, before) in candidate_severity.iter().zip(before_severity) {
            if after > before {
                return false;
            }
        }
    }
    true
}

fn drc_candidate_is_better(before: &[mr_drc::Violation], candidate: &[mr_drc::Violation]) -> bool {
    drc_candidate_is_not_worse(before, candidate)
        && (candidate.len() < before.len()
            || drc_severity_profiles(before) != drc_severity_profiles(candidate))
}

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

/// Which router backend to use.
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

/// Explicit opt-in for the experimental Metal isolated-net provider.
///
/// Warm cropped/ragged real-board A/Bs remained neutral or slower through 5.26M
/// submitted window-cells. Keep the proven CPU path as the default until a
/// request-aware crossover rule has repeated positive evidence.
#[cfg(target_os = "macos")]
const METAL_ISOLATED_ENV: &str = "METALROUTE_EXPERIMENTAL_METAL_ISOLATED";

/// Coarse pre-request work floor after the user has explicitly opted in. The
/// negotiated router has not built its per-net windows at this decision point;
/// this only avoids obviously tiny experiments and is not an automatic crossover.
#[cfg(target_os = "macos")]
const METAL_ISOLATED_MIN_FIELD_WORK: usize = 1_000_000;

/// Corpus routing can invoke `route_problem` concurrently. At most one isolated
/// batch may use Metal; a contending board immediately takes the exact CPU fallback
/// instead of waiting behind the GPU. This bounds combined GPU/shared-host working
/// state to one mr-metal chunk (at most 16M field-cells).
#[cfg(target_os = "macos")]
static METAL_ISOLATED_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_os = "macos")]
fn metal_isolated_opted_in() -> bool {
    matches!(
        std::env::var(METAL_ISOLATED_ENV).as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

#[cfg(target_os = "macos")]
fn use_metal_isolated_provider(n_cells: usize, n_nets: usize, opted_in: bool) -> bool {
    opted_in
        && n_cells
            .checked_mul(n_nets)
            .is_none_or(|work| work >= METAL_ISOLATED_MIN_FIELD_WORK)
}

#[cfg(target_os = "macos")]
trait MetalIsolatedBackend {
    fn route_batch(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
        windows: &[MetalWindow],
        edges: MetalEdgeCosts<'_>,
    ) -> std::result::Result<Vec<Option<MetalIsolatedRoute>>, RouterError>;
}

#[cfg(target_os = "macos")]
struct SystemMetalIsolatedBackend;

#[cfg(target_os = "macos")]
impl MetalIsolatedBackend for SystemMetalIsolatedBackend {
    fn route_batch(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
        windows: &[MetalWindow],
        edges: MetalEdgeCosts<'_>,
    ) -> std::result::Result<Vec<Option<MetalIsolatedRoute>>, RouterError> {
        mr_metal::metal_route_isolated_batch_ragged(grid, nets, windows, edges)
    }
}

/// Adapt one CPU provider request to mr-metal. The first solve uses the CPU's
/// normal windows; only its unreachable entries are cloned into a full-board
/// retry. Any error or alignment mismatch aborts the entire provider result so
/// NegotiatedRouter performs its documented whole-batch CPU fallback.
#[cfg(target_os = "macos")]
fn metal_isolated_paths_with(
    backend: &impl MetalIsolatedBackend,
    request: IsolatedRouteRequest<'_>,
) -> std::result::Result<Vec<Option<Vec<CellIdx>>>, RouterError> {
    if request.windows.len() != request.nets.len() {
        return Err(RouterError::BackendUnavailable(
            "Metal isolated request windows are not net-aligned".into(),
        ));
    }

    let windows: Vec<MetalWindow> = request
        .windows
        .iter()
        .map(|window| MetalWindow {
            x0: window.x0,
            y0: window.y0,
            x1: window.x1,
            y1: window.y1,
        })
        .collect();
    let edges = MetalEdgeCosts {
        x: request.x_edge_costs,
        y: request.y_edge_costs,
        vias: request.via_edge_costs,
    };
    let first = backend.route_batch(request.grid, request.nets, &windows, edges)?;
    if first.len() != request.nets.len() {
        return Err(RouterError::BackendUnavailable(
            "Metal isolated result is not net-aligned".into(),
        ));
    }

    let mut paths: Vec<Option<Vec<CellIdx>>> = first
        .into_iter()
        .map(|route| route.map(|route| route.path))
        .collect();
    let retry_indices: Vec<usize> = paths
        .iter()
        .enumerate()
        .filter_map(|(index, path)| path.is_none().then_some(index))
        .collect();
    if retry_indices.is_empty() {
        return Ok(paths);
    }

    let retry_nets: Vec<NetEndpoints> = retry_indices
        .iter()
        .map(|&index| request.nets[index].clone())
        .collect();
    let retry_windows = vec![MetalWindow::full(request.grid.dims); retry_nets.len()];
    let retry = backend.route_batch(request.grid, &retry_nets, &retry_windows, edges)?;
    if retry.len() != retry_indices.len() {
        return Err(RouterError::BackendUnavailable(
            "Metal isolated retry result is not net-aligned".into(),
        ));
    }
    for (index, route) in retry_indices.into_iter().zip(retry) {
        paths[index] = route.map(|route| route.path);
    }
    Ok(paths)
}

#[cfg(target_os = "macos")]
struct MetalIsolatedProvider;

#[cfg(target_os = "macos")]
impl IsolatedRouteProvider for MetalIsolatedProvider {
    fn route_isolated_batch(
        &self,
        request: IsolatedRouteRequest<'_>,
    ) -> std::result::Result<Vec<Option<Vec<CellIdx>>>, RouterError> {
        let _guard = METAL_ISOLATED_MUTEX.try_lock().map_err(|_| {
            RouterError::BackendUnavailable("Metal isolated batch is busy or unavailable".into())
        })?;
        metal_isolated_paths_with(&SystemMetalIsolatedBackend, request)
    }
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
    if !board_edge_contract_is_active(srj) {
        return route_problem_impl(srj, resolution, router, layers, true);
    }

    let via_pad_mm = routed_via_pad_diameter_mm(srj);
    let effective_layer_count = layers.unwrap_or(srj.layer_count).max(1);
    // Validate the active contract before doing any expensive routing. An empty
    // soup is sufficient to parse/normalize the polygon; malformed outlines fail
    // closed rather than accidentally taking the legacy branch.
    mr_srj::solution_respects_board_outline(srj, &[], via_pad_mm, effective_layer_count)
        .context("invalid board-edge contract")?;

    // Preserve the exact pre-outline product whenever its FINAL emitted soup is
    // already safe. Clearing only the outline contract recreates the historical
    // raster topology and runs the identical beautify/legalize/via-repair pipeline.
    let mut legacy_srj = srj.clone();
    legacy_srj.physical_rules.outline.clear();
    legacy_srj.physical_rules.min_board_edge_clearance = None;
    let legacy = route_problem_impl(&legacy_srj, resolution, router, layers, true)?;
    if mr_srj::solution_respects_board_outline(srj, &legacy.0, via_pad_mm, effective_layer_count)
        .context("failed to validate legacy route against board edge")?
    {
        return Ok(legacy);
    }

    // Only an edge-invalid legacy result pays for a constrained rerun. Validate
    // after every postprocessor, then refuse both edge-unsafe output and a worse
    // complete authoritative DRC profile. The unsafe legacy soup is not an
    // eligible fallback: eliminating its board-edge findings may expose a smaller
    // number of ordinary findings (bugreport49), which is still a net improvement.
    let constrained = route_problem_impl(srj, resolution, router, layers, true)?;
    anyhow::ensure!(
        mr_srj::solution_respects_board_outline(
            srj,
            &constrained.0,
            via_pad_mm,
            effective_layer_count,
        )
        .context("failed to validate constrained route against board edge")?,
        "constrained route violates the board-edge contract"
    );
    let legacy_drc = check_srj_solution(srj, &legacy.0, effective_layer_count);
    let constrained_drc = check_srj_solution(srj, &constrained.0, effective_layer_count);
    anyhow::ensure!(
        drc_candidate_is_not_worse(&legacy_drc, &constrained_drc),
        "constrained board-edge reroute regresses authoritative DRC"
    );
    Ok(constrained)
}

fn board_edge_contract_is_active(srj: &SimpleRouteJson) -> bool {
    !srj.physical_rules.outline.is_empty() || srj.physical_rules.min_board_edge_clearance.is_some()
}

/// Match partial/legacy board-edge raster geometry to the width that this CLI
/// actually emits. The caller must resolve the typed physical profile from the
/// original SRJ before using this clone: changing `minTraceWidth` can otherwise
/// make an incoherent profile coherent and silently broaden product semantics.
fn legacy_board_edge_raster_input(srj: &SimpleRouteJson) -> std::borrow::Cow<'_, SimpleRouteJson> {
    if board_edge_contract_is_active(srj) && srj.min_trace_width != Some(DEFAULT_TRACE_WIDTH) {
        let mut constrained = srj.clone();
        constrained.min_trace_width = Some(DEFAULT_TRACE_WIDTH);
        std::borrow::Cow::Owned(constrained)
    } else {
        std::borrow::Cow::Borrowed(srj)
    }
}

fn routed_via_pad_diameter_mm(srj: &SimpleRouteJson) -> f64 {
    srj.uniform_physical_rules()
        .map(|rules| rules.via_pad_diameter_mm)
        .unwrap_or(VIA_PAD_MM)
}

/// Build and check the exact continuous board emitted for an SRJ route. Coherent
/// supported typed projections use their trace↔pad, via↔pad, pad↔pad, and via
/// geometry; legacy/partial inputs retain the historical single-clearance checker.
pub fn check_srj_solution(
    srj: &SimpleRouteJson,
    traces: &[mr_srj::PcbTrace],
    layers: u32,
) -> Vec<mr_drc::Violation> {
    let clearance = srj
        .min_clearance
        .or_else(|| {
            srj.uniform_physical_rules()
                .map(|rules| rules.obstacle_margin_mm)
        })
        .unwrap_or(DEFAULT_CLEARANCE_MM);
    let board =
        drc_board::solution_to_drc_board(srj, traces, drc::default_rules(clearance), layers);
    drc_board::check_with_srj_rules(srj, &board)
}

/// Keep a topology-preserving geometry candidate only when the authoritative SRJ
/// checker proves it cannot add a finding, substitute a different feature pair, or
/// worsen any existing deficit. Used before later repair stages establish their
/// own baseline.
fn select_nonworsening_srj_geometry(
    srj: &SimpleRouteJson,
    original: Vec<mr_srj::PcbTrace>,
    candidate: Vec<mr_srj::PcbTrace>,
    layers: u32,
) -> Vec<mr_srj::PcbTrace> {
    let before = check_srj_solution(srj, &original, layers);
    let after = check_srj_solution(srj, &candidate, layers);
    if drc_candidate_is_not_worse(&before, &after) {
        candidate
    } else {
        original
    }
}

fn route_problem_impl(
    srj: &SimpleRouteJson,
    resolution: Option<f64>,
    router: RouterKind,
    layers: Option<u32>,
    repair_vias: bool,
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
    // The coherent supported SRJ physical-rule projection is used only when it
    // resolves to one uniform trace width. Partial or mixed-width inputs
    // deliberately retain the established constants until the core router can
    // price feature-pair widths.
    let physical = srj.uniform_physical_rules();
    let trace_width = physical
        .map(|rules| rules.trace_width_mm)
        .unwrap_or(DEFAULT_TRACE_WIDTH);
    let via_pad_mm = physical
        .map(|rules| rules.via_pad_diameter_mm)
        .unwrap_or(VIA_PAD_MM);
    // Clearance enforcement on the SimpleRouteJson path (mirrors the DSN pipeline).
    // `min_clearance` is the copper-to-copper edge gap; `DEFAULT_TRACE_WIDTH` is the
    // width every emitted trace carries (see `to_solution_layered` below).
    //   * PADS are inflated at rasterise time by `clearance + track_w/2` (a foreign
    //     trace's centreline must clear the pad EDGE by `clearance`) — pass
    //     `clearance_cells` so `rasterize_with_layers` reserves that halo.
    //   * TRACE-vs-TRACE spacing is the negotiated router's clearance halo (set below).
    //     Its radius is a CENTRELINE-to-foreign-centreline distance, so it must be
    //     `clearance + track_w` (own half-width + clearance + foreign half-width), not
    //     the bare `clearance` — else two centred tracks `clearance` apart overlap
    //     copper by `track_w`.
    let min_clearance = srj
        .min_clearance
        .or_else(|| physical.map(|rules| rules.obstacle_margin_mm))
        .unwrap_or(DEFAULT_CLEARANCE_MM)
        .max(0.0);
    let clearance_cells = if min_clearance > 0.0 && resolution > 0.0 {
        (min_clearance / resolution).ceil() as u32
    } else {
        0
    };
    // Zero edge clearance still forbids copper overlap, so both trace half-widths
    // remain part of the centreline rule even when `min_clearance == 0`.
    let trace_halo_mm = min_clearance + trace_width;
    // Two foreign via pads need both radii plus the copper edge clearance. This is
    // wider than the via-to-track keepout below by the difference between a via and
    // trace radius, so the negotiated router tracks it as a separate centre rule.
    let via_spacing_mm = physical.map_or(VIA_PAD_MM + min_clearance, |rules| {
        (rules.via_pad_diameter_mm + rules.obstacle_margin_mm)
            .max(rules.via_hole_diameter_mm + rules.via_hole_to_hole_clearance_mm.unwrap_or(0.0))
    });
    let via_hole_spacing_mm = physical
        .and_then(|rules| {
            rules
                .via_hole_to_hole_clearance_mm
                .map(|clearance| rules.via_hole_diameter_mm + clearance)
        })
        .unwrap_or(0.0);
    // D2: thread the real signal-via pad diameter so the rasteriser can reserve a
    // via-class halo around foreign pads on via-allowed (multi-layer) stackups.
    let problem = match physical {
        Some(rules) => rasterize_with_uniform_physical_rules(srj, resolution, layer_map, rules),
        None => {
            // `physical` was deliberately selected from the untouched input.
            // Normalize only the SRJ used to build the active board mask/Hanan
            // geometry so it matches the fixed-width legacy soup we emit.
            let raster_srj = legacy_board_edge_raster_input(srj);
            rasterize_with_layers(
                &raster_srj,
                resolution,
                layer_map,
                clearance_cells,
                min_clearance,
                via_pad_mm,
            )
        }
    };
    let total = problem.nets.len();

    // Only the negotiated backend places vias; give it a through-hole model over
    // the routed stackup. Lee/Ripup route per-layer with no layer changes.
    // The via's annular-ring keepout (centreline-to-foreign-centreline, in mm) is the
    // via pad radius + clearance + the foreign trace's half-width, so a committed via's
    // copper keeps full `clearance` from a foreign track's copper.
    let mut via_model = ViaModel::through_hole(problem.mapping.dims.layers);
    let via_to_trace_clearance = physical
        .map(|rules| rules.trace_to_pad_clearance_mm)
        .unwrap_or(min_clearance);
    via_model.keepout_mm = via_pad_mm / 2.0 + via_to_trace_clearance + trace_width / 2.0;
    // The board's continuous grid-line geometry, so the negotiated router prices
    // planar steps by their real length. On a uniform grid this is byte-identical to
    // the unit-hop fallback; on a non-uniform / Hanan grid it makes the cost track
    // the true pitch.
    let coords = GridCoords::from_lines(
        problem.mapping.x_lines.clone(),
        problem.mapping.y_lines.clone(),
    );
    let (board, negotiated_alone) = match router {
        RouterKind::Lee => LeeRouter::new()
            .route(&problem.grid, &problem.nets)
            .map(|board| (board, None)),
        RouterKind::Ripup => RipUpRouter::new()
            .route(&problem.grid, &problem.nets)
            .map(|board| (board, None)),
        RouterKind::Negotiated => {
            let negotiated = NegotiatedRouter::new()
                .with_via_model(via_model.clone())
                .with_clearance_mm(trace_halo_mm)
                .with_via_spacing_mm(via_spacing_mm)
                .with_via_hole_spacing_mm(via_hole_spacing_mm)
                .with_committed_via_to_trace_guard(physical.is_some())
                .with_coords(coords.clone());
            #[cfg(target_os = "macos")]
            let outcome = if use_metal_isolated_provider(
                problem.grid.dims.len(),
                total,
                metal_isolated_opted_in(),
            ) {
                negotiated.route_with_isolated_provider(
                    &problem.grid,
                    &problem.nets,
                    &MetalIsolatedProvider,
                )
            } else {
                negotiated.route_with_outcome(&problem.grid, &problem.nets)
            };
            #[cfg(not(target_os = "macos"))]
            let outcome = negotiated.route_with_outcome(&problem.grid, &problem.nets);
            outcome.map(|outcome| (outcome.board, Some(outcome.alone_routable)))
        }
    }
    .context("router failed")?;

    let traces = to_solution_layered(
        &board,
        &problem.mapping,
        &problem.pin_points,
        trace_width,
        &problem.layers,
    );
    // Beautify the emitted geometry: pull staircases into diagonals and chamfer
    // square corners. A typed profile may require a larger trace↔pad gap than its
    // generic obstacle margin, so establish the authoritative baseline on the RAW
    // soup and retain the beautified candidate only when the complete supported
    // projection is non-worsening. Legacy inputs keep their established path.
    let beautified = mr_srj::beautify_traces(traces.clone(), &srj.obstacles, min_clearance);
    let board_edge_active = !srj.physical_rules.outline.is_empty()
        || srj.physical_rules.min_board_edge_clearance.is_some();
    let traces = if physical.is_some() || board_edge_active {
        select_nonworsening_srj_geometry(srj, traces, beautified, layer_count)
    } else {
        beautified
    };
    // Exact-geometry clearance legalisation: the grid halo guards NODE positions, but
    // copper is the segments between nodes (plus endpoint snapping and 45° chamfers),
    // so the emitted geometry can still hold genuine different-net clearance shorts the
    // exact DRC reports. This pass nudges interior wire vertices to recover spacing,
    // validated against the same exact distance engine — monotone (never worsens a
    // gap) so it only removes violations and never breaks connectivity. No-op when
    // `min_clearance <= 0` (clearance-off fast path stays byte-identical).
    //
    // AUTHORITATIVE GATE: the legaliser lives in `mr-srj` and reconstructs net identity
    // from `PcbTrace::net` (the router's `g<group>` labels), which can differ from the
    // DRC's `c<connectivity_net>` relabelling at shared junction pads. So we gate the
    // pass against the REAL DRC here: fewer complete authoritative findings wins;
    // equal-count geometry is kept only for a strict, quantised same-identity severity
    // improvement. This prevents equal-count regression or churn regardless of any
    // net-view mismatch.
    let traces = if min_clearance > 0.0 {
        let layers = layers.unwrap_or(srj.layer_count).max(1);
        // Tag each trace with the DRC's own electrical-net identity (`c<net>` at shared
        // connectivity pads, else the router group) so the legaliser's same-net immunity
        // and its internal gate agree with the authoritative checker — otherwise it would
        // try to push apart copper the DRC considers one net, or miss real foreign pairs.
        let labels = drc_board::reconstruct_net_labels(srj, &traces, layers);
        let traces: Vec<_> = traces
            .into_iter()
            .zip(labels)
            .map(|(mut t, n)| {
                t.net = Some(n);
                t
            })
            .collect();
        let before = check_srj_solution(srj, &traces, layers);
        let legalised = mr_srj::legalize_clearance(traces.clone(), &srj.obstacles, min_clearance);
        let after = check_srj_solution(srj, &legalised, layers);
        if drc_candidate_is_better(&before, &after) {
            legalised
        } else {
            traces
        }
    } else {
        traces
    };
    // A bounded topology-preserving follow-up for the subset the vertex legaliser
    // cannot reach: an interior via that still participates in an exact clearance
    // finding. One pass evaluates at most 8 vias × 8 one-clearance translations,
    // rigidly carrying its source/explicit landing anchors. Only a strictly lower
    // authoritative full-board finding count is retained.
    let traces = if repair_vias && min_clearance > 0.0 {
        via_repair::repair_clearance_vias(
            srj,
            traces,
            drc::default_rules(min_clearance),
            layer_count,
        )
    } else {
        traces
    };

    // Diagnose every unrouted net: was it impossible at this resolution, or just
    // lost to congestion? Negotiated routing reuses the exact isolation result its
    // legalization phase already computed; the other backends retain solo reroutes.
    let unrouted = classify_unrouted(
        router,
        &problem.grid,
        &problem.nets,
        &board.unrouted,
        &via_model,
        &coords,
        negotiated_alone.as_deref(),
    );

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
/// `negotiated_alone`, when present, is aligned with `nets` and is the exact result
/// already computed by [`NegotiatedRouter`]. Other backends (and tests exercising
/// the fallback) run sequentially: the corpus harness already fans boards out
/// across cores, so nesting rayon here would only oversubscribe.
fn classify_unrouted(
    router: RouterKind,
    grid: &Grid,
    nets: &[NetEndpoints],
    unrouted: &[String],
    via_model: &ViaModel,
    coords: &GridCoords,
    negotiated_alone: Option<&[bool]>,
) -> Vec<(String, UnroutedReason)> {
    let by_name: HashMap<&str, (usize, &NetEndpoints)> = nets
        .iter()
        .enumerate()
        .map(|(i, net)| (net.net.as_str(), (i, net)))
        .collect();

    unrouted
        .iter()
        .map(|name| {
            let routes_alone = by_name.get(name.as_str()).is_some_and(|&(i, net)| {
                if router == RouterKind::Negotiated {
                    if let Some(cached) = negotiated_alone {
                        return cached.get(i).copied().unwrap_or(false);
                    }
                }
                let solo = std::slice::from_ref(net);
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
    // Committed vias likewise reserve a keepout halo (`ViaModel.keepout_mm`, in mm)
    // sized for the via pad plus clearance, and the committing legalization pass
    // enforces it HARD. This supersedes the M2.4 "disabled legalization
    // halo" experiment: clearance now lives in the negotiation phase + the pad/via
    // grid, not a post-hoc legalization fold. Plane-antipad modelling (the via-
    // through-plane fix) is independent and stays on.
    let clearance_cells = if stats.min_clearance_mm > 0.0 && resolution > 0.0 {
        (stats.min_clearance_mm / resolution).ceil() as u32
    } else {
        0
    };
    // The width every emitted trace carries — the router's clearance halo is a
    // CENTRELINE-to-foreign-centreline distance, so it must budget both half-widths.
    let trace_w = srj.min_trace_width.unwrap_or(DEFAULT_TRACE_WIDTH);
    let mut via_model = ViaModel::through_hole(layer_map.len());
    // Via annular-ring keepout in CONTINUOUS mm (the unit the router's halo code
    // expects): the via pad radius + clearance + the foreign trace's half-width, so a
    // committed via's copper keeps full `clearance` from a foreign track's copper. No
    // `/resolution` cell conversion — that was a unit bug on the non-uniform Hanan grid.
    via_model.keepout_mm = VIA_PAD_MM / 2.0 + stats.min_clearance_mm + trace_w / 2.0;
    let via_spacing_mm = VIA_PAD_MM + stats.min_clearance_mm.max(0.0);
    // D2: thread the real signal-via pad diameter for the via-class foreign-pad halo.
    let problem = rasterize_with_layers(
        &srj,
        resolution,
        layer_map,
        clearance_cells,
        stats.min_clearance_mm,
        VIA_PAD_MM,
    );
    let total_nets = problem.nets.len();
    let grid_w = problem.mapping.dims.w;
    let grid_h = problem.mapping.dims.h;
    let grid_layers = problem.mapping.dims.layers;

    // Continuous grid-line geometry: prices planar steps by real length (uniform-grid
    // byte-identical, Hanan-grid pitch-aware).
    let coords = GridCoords::from_lines(
        problem.mapping.x_lines.clone(),
        problem.mapping.y_lines.clone(),
    );
    let start = std::time::Instant::now();
    let board = NegotiatedRouter::new()
        .with_via_model(via_model)
        // Geometric trace-vs-trace clearance over the (possibly non-uniform) line
        // arrays. The halo radius is centreline-to-foreign-centreline, so it is
        // `clearance + track_w` (own half-width + clearance + foreign half-width) — the
        // bare clearance under-blocks and lets two centred tracks overlap by `track_w`.
        .with_clearance_mm(stats.min_clearance_mm + trace_w)
        .with_via_spacing_mm(via_spacing_mm)
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
    // Exact-geometry clearance legalisation (see the SRJ path): nudge interior wire
    // vertices to recover any different-net spacing the node-based grid halo could not
    // guarantee on the emitted segments. Monotone and connectivity-preserving; a no-op
    // when clearance is off.
    let traces = mr_srj::legalize_clearance(traces, &srj.obstacles, stats.min_clearance_mm);
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
    )?;

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

    fn clearance_violation(measured: f64, nets: (&str, &str)) -> mr_drc::Violation {
        mr_drc::Violation {
            class: mr_drc::ViolationClass::Clearance,
            layer: 0,
            location: (0.0, 0.0),
            nets: (nets.0.into(), nets.1.into()),
            measured,
            required: 0.2,
        }
    }

    #[test]
    fn authoritative_drc_comparator_counts_unknown_net_findings() {
        // `("", "")` is the serialized representation of two independent unknown
        // pads. The DRC intentionally considers them foreign, so the gate must not
        // mistake the equal strings for a same-net pair and filter the finding out.
        let unknown_pair = clearance_violation(0.05, ("", ""));
        assert!(drc_candidate_is_better(&[unknown_pair], &[]));
        assert!(!drc_candidate_is_better(
            &[],
            &[clearance_violation(0.05, ("", ""))]
        ));
    }

    #[test]
    fn authoritative_drc_comparator_requires_strict_equal_count_improvement() {
        let before = [
            clearance_violation(0.05, ("A", "B")),
            clearance_violation(0.15, ("C", "D")),
        ];

        // A uniformly smaller deficit is a useful equal-count repair.
        let improved = [
            clearance_violation(0.06, ("A", "B")),
            clearance_violation(0.16, ("C", "D")),
        ];
        assert!(drc_candidate_is_better(&before, &improved));
        let mut moved_improvement = improved.clone();
        moved_improvement[0].location = (0.25, -0.10);
        assert!(
            drc_candidate_is_better(&before, &moved_improvement),
            "the same net-pair finding may move while its severity improves"
        );

        // A new/worse worst finding is not hidden by improving the other one.
        let traded = [
            clearance_violation(0.04, ("A", "B")),
            clearance_violation(0.19, ("C", "D")),
        ];
        assert!(!drc_candidate_is_better(&before, &traded));

        // Replacing findings with a different class/layer/net identity is not an
        // authoritative improvement, even when the replacement is milder.
        let substituted = [
            clearance_violation(0.06, ("E", "F")),
            clearance_violation(0.16, ("G", "H")),
        ];
        assert!(!drc_candidate_is_better(&before, &substituted));
    }

    #[test]
    fn authoritative_drc_comparator_keeps_fewer_findings_primary() {
        let before = [
            clearance_violation(0.15, ("A", "B")),
            clearance_violation(0.16, ("C", "D")),
        ];

        let new_identity = [clearance_violation(0.19, ("E", "F"))];
        assert!(
            drc_candidate_is_better(&before, &new_identity),
            "a lower authoritative finding count remains the primary objective"
        );

        let more_severe = [clearance_violation(0.10, ("A", "B"))];
        assert!(
            drc_candidate_is_better(&before, &more_severe),
            "count reduction wins before severity tie-breaking"
        );
    }

    #[test]
    fn authoritative_drc_comparator_never_trades_board_edge_safety_for_count() {
        let ordinary_before = [
            clearance_violation(0.05, ("A", "B")),
            clearance_violation(0.06, ("C", "D")),
        ];
        let introduced_edge = [clearance_violation(0.19, ("trace", BOARD_EDGE_NET))];
        assert!(
            !drc_candidate_is_better(&ordinary_before, &introduced_edge),
            "fewer total findings cannot introduce a board-edge violation"
        );

        let before = [
            clearance_violation(0.15, ("trace", BOARD_EDGE_NET)),
            clearance_violation(0.05, ("A", "B")),
        ];
        let worsened_edge = [clearance_violation(0.10, ("trace", BOARD_EDGE_NET))];
        assert!(
            !drc_candidate_is_better(&before, &worsened_edge),
            "removing an ordinary finding cannot worsen the retained board edge"
        );

        let improved_edge = [clearance_violation(0.16, ("trace", BOARD_EDGE_NET))];
        assert!(drc_candidate_is_better(&before, &improved_edge));
        assert!(drc_candidate_is_better(&before, &[]));
    }

    #[test]
    fn board_edge_comparator_uses_checker_tolerance_not_ordinary_quantum() {
        let before = [
            clearance_violation(0.1, ("trace", BOARD_EDGE_NET)),
            clearance_violation(0.05, ("A", "B")),
        ];
        let sub_quantum_worsening = [clearance_violation(
            0.1 - 0.4 * DRC_SCORE_QUANTUM_MM,
            ("trace", BOARD_EDGE_NET),
        )];
        assert_eq!(
            drc_severity(&before[0]),
            drc_severity(&sub_quantum_worsening[0]),
            "ordinary DRC intentionally rounds this sub-nanometre change to a tie"
        );
        assert!(!board_edge_findings_are_not_worse(
            &before,
            &sub_quantum_worsening
        ));
        assert!(
            !drc_candidate_is_better(&before, &sub_quantum_worsening),
            "removing an ordinary finding cannot hide a board-edge worsening below 1e-6 mm"
        );

        let checker_tolerance_jitter = [clearance_violation(
            0.1 - BOARD_EDGE_GEOMETRY_EPS_MM,
            ("trace", BOARD_EDGE_NET),
        )];
        assert!(board_edge_findings_are_not_worse(
            &before,
            &checker_tolerance_jitter
        ));
        assert!(drc_candidate_is_better(&before, &checker_tolerance_jitter));
    }

    #[test]
    fn board_edge_comparator_normalizes_nonfinite_deficits_deterministically() {
        let finite = [clearance_violation(0.1, ("trace", BOARD_EDGE_NET))];
        let mut nan = clearance_violation(0.1, ("trace", BOARD_EDGE_NET));
        nan.measured = f64::NAN;
        let mut infinite_deficit = clearance_violation(0.1, ("trace", BOARD_EDGE_NET));
        infinite_deficit.measured = f64::NEG_INFINITY;

        assert!(!board_edge_findings_are_not_worse(
            &finite,
            std::slice::from_ref(&nan)
        ));
        assert!(!board_edge_findings_are_not_worse(
            &finite,
            std::slice::from_ref(&infinite_deficit)
        ));
        assert!(board_edge_findings_are_not_worse(
            std::slice::from_ref(&nan),
            std::slice::from_ref(&nan)
        ));
        assert!(board_edge_findings_are_not_worse(
            std::slice::from_ref(&infinite_deficit),
            &finite
        ));
    }

    #[test]
    fn unsafe_edge_findings_are_ineligible_even_if_safe_route_has_ordinary_findings() {
        let legacy: Vec<_> = (0..33)
            .map(|index| clearance_violation(0.0, (&format!("n{index}"), "__board_edge__")))
            .collect();
        let constrained: Vec<_> = (0..4)
            .map(|index| clearance_violation(0.1, (&format!("a{index}"), "foreign")))
            .collect();
        assert!(
            drc_candidate_is_not_worse(&legacy, &constrained),
            "33 unsafe edge findings to four ordinary findings is a full-profile improvement"
        );
    }

    #[test]
    fn authoritative_drc_comparator_quantises_float_jitter() {
        let before = [clearance_violation(0.100_000_0, ("A", "B"))];
        let sub_nanometre_change = [clearance_violation(0.100_000_4, ("A", "B"))];
        assert!(!drc_candidate_is_better(&before, &sub_nanometre_change));

        let material_change = [clearance_violation(0.100_002_0, ("A", "B"))];
        assert!(drc_candidate_is_better(&before, &material_change));
    }

    #[test]
    fn typed_geometry_gate_rejects_a_new_trace_to_pad_violation() {
        let srj: SimpleRouteJson = serde_json::from_value(serde_json::json!({
            "layerCount": 1,
            "minTraceWidth": 0.1,
            "nominalTraceWidth": 0.1,
            "defaultObstacleMargin": 0.04,
            "minTraceToPadEdgeClearance": 0.1,
            "minViaEdgeToPadEdgeClearance": 0.1,
            "minViaHoleDiameter": 0.2,
            "minViaPadDiameter": 0.4,
            "bounds": {"minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [{
                "type": "rect", "shape": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.2, "height": 0.2, "layers": ["top"],
                "connectedTo": ["foreign_pad"]
            }]
        }))
        .unwrap();
        assert!(srj.uniform_physical_rules().is_some());
        let trace_at = |x| {
            mr_srj::PcbTrace::new(vec![
                RoutePoint::Wire {
                    x,
                    y: -0.8,
                    width: 0.1,
                    layer: "top".into(),
                },
                RoutePoint::Wire {
                    x,
                    y: 0.8,
                    width: 0.1,
                    layer: "top".into(),
                },
            ])
            .with_net("trace")
        };
        let original = vec![trace_at(0.3)];
        let candidate = vec![trace_at(0.2)];
        assert!(check_srj_solution(&srj, &original, 1).is_empty());
        assert_eq!(check_srj_solution(&srj, &candidate, 1).len(), 1);
        assert_eq!(
            select_nonworsening_srj_geometry(&srj, original.clone(), candidate, 1),
            original,
            "a generic-margin geometry transform must not establish a worse typed baseline"
        );
    }

    #[cfg(target_os = "macos")]
    use std::{
        cell::{Cell, RefCell},
        collections::VecDeque,
    };

    #[cfg(target_os = "macos")]
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RecordedMetalCall {
        nets: Vec<String>,
        windows: Vec<MetalWindow>,
        x_edges: Vec<mr_core::Cost>,
        y_edges: Vec<mr_core::Cost>,
        via_edges: Vec<Option<mr_core::Cost>>,
    }

    #[cfg(target_os = "macos")]
    struct ScriptedMetalBackend {
        responses:
            RefCell<VecDeque<std::result::Result<Vec<Option<MetalIsolatedRoute>>, RouterError>>>,
        calls: RefCell<Vec<RecordedMetalCall>>,
    }

    #[cfg(target_os = "macos")]
    impl ScriptedMetalBackend {
        fn new(
            responses: impl IntoIterator<
                Item = std::result::Result<Vec<Option<MetalIsolatedRoute>>, RouterError>,
            >,
        ) -> Self {
            Self {
                responses: RefCell::new(responses.into_iter().collect()),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    #[cfg(target_os = "macos")]
    impl MetalIsolatedBackend for ScriptedMetalBackend {
        fn route_batch(
            &self,
            _grid: &Grid,
            nets: &[NetEndpoints],
            windows: &[MetalWindow],
            edges: MetalEdgeCosts<'_>,
        ) -> std::result::Result<Vec<Option<MetalIsolatedRoute>>, RouterError> {
            self.calls.borrow_mut().push(RecordedMetalCall {
                nets: nets.iter().map(|net| net.net.clone()).collect(),
                windows: windows.to_vec(),
                x_edges: edges.x.to_vec(),
                y_edges: edges.y.to_vec(),
                via_edges: edges.vias.to_vec(),
            });
            self.responses
                .borrow_mut()
                .pop_front()
                .expect("one scripted response per Metal call")
        }
    }

    #[cfg(target_os = "macos")]
    #[derive(Default)]
    struct RecordingSystemMetalProvider {
        calls: Cell<usize>,
        succeeded: Cell<bool>,
    }

    #[cfg(target_os = "macos")]
    impl IsolatedRouteProvider for RecordingSystemMetalProvider {
        fn route_isolated_batch(
            &self,
            request: IsolatedRouteRequest<'_>,
        ) -> std::result::Result<Vec<Option<Vec<CellIdx>>>, RouterError> {
            self.calls.set(self.calls.get() + 1);
            let result = MetalIsolatedProvider.route_isolated_batch(request);
            self.succeeded.set(result.is_ok());
            result
        }
    }

    #[cfg(target_os = "macos")]
    struct AdapterFixture {
        grid: Grid,
        nets: Vec<NetEndpoints>,
        windows: Vec<mr_cpu::IsolatedRouteWindow>,
        x_edges: Vec<mr_core::Cost>,
        y_edges: Vec<mr_core::Cost>,
        via_edges: Vec<Option<mr_core::Cost>>,
    }

    #[cfg(target_os = "macos")]
    impl AdapterFixture {
        fn request(&self) -> IsolatedRouteRequest<'_> {
            IsolatedRouteRequest {
                grid: &self.grid,
                nets: &self.nets,
                windows: &self.windows,
                x_edge_costs: &self.x_edges,
                y_edge_costs: &self.y_edges,
                via_edge_costs: &self.via_edges,
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn adapter_fixture() -> AdapterFixture {
        let dims = mr_core::Dims::new(5, 3);
        let grid = Grid::filled(dims, 1);
        let nets = (0..3)
            .map(|row| NetEndpoints {
                net: format!("n{row}"),
                src: dims.idx(0, row),
                dst: dims.idx(4, row),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            })
            .collect();
        let windows = vec![
            mr_cpu::IsolatedRouteWindow {
                x0: 0,
                y0: 0,
                x1: 4,
                y1: 0,
            },
            mr_cpu::IsolatedRouteWindow {
                x0: 0,
                y0: 1,
                x1: 4,
                y1: 1,
            },
            mr_cpu::IsolatedRouteWindow {
                x0: 0,
                y0: 2,
                x1: 4,
                y1: 2,
            },
        ];
        AdapterFixture {
            grid,
            nets,
            windows,
            x_edges: vec![11, 13, 17, 19],
            y_edges: vec![23, 29],
            via_edges: Vec::new(),
        }
    }

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

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_field_work_gate_has_exact_boundary_and_overflow_policy() {
        assert!(!use_metal_isolated_provider(usize::MAX, 2, false));
        assert!(!use_metal_isolated_provider(999_999, 1, true));
        assert!(use_metal_isolated_provider(1_000_000, 1, true));
        assert!(use_metal_isolated_provider(2_000, 500, true));
        assert!(!use_metal_isolated_provider(2_000, 499, true));
        assert!(use_metal_isolated_provider(usize::MAX, 2, true));
        assert!(!use_metal_isolated_provider(1_000_000, 0, true));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_provider_falls_back_instead_of_waiting_for_a_busy_gpu_lane() {
        let _held = METAL_ISOLATED_MUTEX.lock().unwrap();
        let dims = mr_core::Dims::new(1, 1);
        let grid = Grid::filled(dims, 1);
        let nets: [NetEndpoints; 0] = [];
        let windows: [mr_cpu::IsolatedRouteWindow; 0] = [];
        let edges: [mr_core::Cost; 0] = [];
        let vias: [Option<mr_core::Cost>; 0] = [];

        let result = MetalIsolatedProvider.route_isolated_batch(IsolatedRouteRequest {
            grid: &grid,
            nets: &nets,
            windows: &windows,
            x_edge_costs: &edges,
            y_edge_costs: &edges,
            via_edge_costs: &vias,
        });
        assert!(matches!(result, Err(RouterError::BackendUnavailable(_))));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_adapter_retries_only_missing_entries_and_restores_input_alignment() {
        let fixture = adapter_fixture();
        let first = vec![
            None,
            Some(MetalIsolatedRoute {
                path: vec![5, 6, 7, 8, 9],
                search_cost: 2,
            }),
            None,
        ];
        let retry = vec![
            Some(MetalIsolatedRoute {
                path: vec![0, 1, 2, 3, 4],
                search_cost: 1,
            }),
            Some(MetalIsolatedRoute {
                path: vec![10, 11, 12, 13, 14],
                search_cost: 3,
            }),
        ];
        let backend = ScriptedMetalBackend::new([Ok(first), Ok(retry)]);
        let got = metal_isolated_paths_with(&backend, fixture.request()).unwrap();

        assert_eq!(
            got,
            [
                Some(vec![0, 1, 2, 3, 4]),
                Some(vec![5, 6, 7, 8, 9]),
                Some(vec![10, 11, 12, 13, 14]),
            ]
        );
        let calls = backend.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].nets, ["n0", "n1", "n2"]);
        assert_eq!(calls[1].nets, ["n0", "n2"]);
        assert_eq!(
            calls[0].windows,
            fixture
                .windows
                .iter()
                .map(|window| MetalWindow {
                    x0: window.x0,
                    y0: window.y0,
                    x1: window.x1,
                    y1: window.y1,
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(calls[1].windows, [MetalWindow::full(fixture.grid.dims); 2]);
        for call in calls.iter() {
            assert_eq!(call.x_edges, fixture.x_edges);
            assert_eq!(call.y_edges, fixture.y_edges);
            assert_eq!(call.via_edges, fixture.via_edges);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_adapter_rejects_misaligned_results() {
        let fixture = adapter_fixture();
        let backend = ScriptedMetalBackend::new([Ok(vec![None; fixture.nets.len() - 1])]);
        let error = metal_isolated_paths_with(&backend, fixture.request()).unwrap_err();
        assert!(matches!(error, RouterError::BackendUnavailable(_)));
        assert_eq!(backend.calls.borrow().len(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_adapter_propagates_retry_error_without_partial_output() {
        let fixture = adapter_fixture();
        let backend = ScriptedMetalBackend::new([
            Ok(vec![
                Some(MetalIsolatedRoute {
                    path: vec![0, 1, 2, 3, 4],
                    search_cost: 1,
                }),
                None,
                None,
            ]),
            Err(RouterError::BackendUnavailable(
                "scripted retry failure".into(),
            )),
        ]);
        let error = metal_isolated_paths_with(&backend, fixture.request()).unwrap_err();
        assert!(error.to_string().contains("scripted retry failure"));
        let calls = backend.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].nets, ["n1", "n2"]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ragged_metal_adapter_preserves_exact_output_and_cpu_fallbacks() {
        let dims = mr_core::Dims::with_layers(9, 7, 3);
        let coords = GridCoords::from_lines(
            vec![0.0, 0.2, 0.7, 1.9, 2.0, 4.5, 4.7, 8.0, 8.1],
            vec![0.0, 0.1, 1.0, 1.0, 2.7, 6.0, 6.2],
        );
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx3(2, 3, 0), 0);
        grid.set(dims.idx3(3, 3, 1), 7);
        for (x, y, layer) in [(0, 0, 0), (8, 0, 2), (4, 3, 0), (4, 3, 1)] {
            grid.set(dims.idx3(x, y, layer), mr_core::OBSTACLE);
        }
        let nets = vec![
            NetEndpoints {
                net: "weighted".into(),
                src: dims.idx3(0, 3, 0),
                dst: dims.idx3(8, 3, 2),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "own-pads".into(),
                src: dims.idx3(0, 0, 0),
                dst: dims.idx3(8, 0, 2),
                passable_pads: vec![dims.idx3(0, 0, 0), dims.idx3(8, 0, 2)],
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "zero-length".into(),
                src: dims.idx3(2, 3, 0),
                dst: dims.idx3(2, 3, 0),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        let via_model = ViaModel::with_allowed_steps(3, 37, vec![(0, 1), (1, 2)]);
        let router = NegotiatedRouter::new()
            .with_coords(coords)
            .with_via_model(via_model);
        let cpu = router.route_with_outcome(&grid, &nets).unwrap();
        let provider = RecordingSystemMetalProvider::default();
        let accelerated = router
            .route_with_isolated_provider(&grid, &nets, &provider)
            .unwrap();
        assert_eq!(provider.calls.get(), 1);
        assert!(
            provider.succeeded.get(),
            "the exact-output comparison must exercise a successful ragged Metal batch"
        );
        assert_eq!(accelerated, cpu);

        // The compact request intentionally does not encode per-net via-pad
        // exemptions yet. A selected static-mask contract must therefore fail
        // closed in mr-metal and make NegotiatedRouter rerun the complete
        // isolated batch on CPU, preserving its exact output.
        let mut masked_grid = grid.clone();
        masked_grid.via_forbidden = vec![false; dims.len()];
        let cpu = router.route_with_outcome(&masked_grid, &nets).unwrap();
        let provider = RecordingSystemMetalProvider::default();
        let fallback = router
            .route_with_isolated_provider(&masked_grid, &nets, &provider)
            .unwrap();
        assert_eq!(provider.calls.get(), 1);
        assert!(
            !provider.succeeded.get(),
            "a selected static via mask must fail the ragged request closed"
        );
        assert_eq!(fallback, cpu);

        // Exact board masks include directed planar-edge permissions that the
        // Metal kernels do not carry yet. The public backend rejects even an
        // otherwise all-zero selected mask, and the negotiated router reruns the
        // complete isolated batch on CPU without changing its result.
        let mut outlined_grid = grid.clone();
        outlined_grid.board_constraint = vec![0; dims.len()];
        outlined_grid.board_constraint[dims.idx3(1, 1, 0) as usize] |= Grid::BOARD_EDGE_POS_X;
        let cpu = router.route_with_outcome(&outlined_grid, &nets).unwrap();
        let provider = RecordingSystemMetalProvider::default();
        let fallback = router
            .route_with_isolated_provider(&outlined_grid, &nets, &provider)
            .unwrap();
        assert_eq!(provider.calls.get(), 1);
        assert!(
            !provider.succeeded.get(),
            "an exact board mask must fail the ragged request closed"
        );
        assert_eq!(fallback, cpu);
    }

    #[test]
    fn route_problem_routes_two_nets() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        let (traces, summary, _diag) =
            route_problem(&srj, Some(1.0), RouterKind::Ripup, None).unwrap();

        // Two 2-point connections -> two nets, both routable on this open board.
        assert_eq!(summary.total, 2);
        assert_eq!(summary.routed, 2);
        // Non-uniform / Hanan grid (Phase 3): lines fall on the bounds {0,10}, every
        // pad endpoint {1,9}, every obstacle edge {4,6}, plus fill channels. The
        // regular fill needs `track_w + 2·clearance`; here `clearance` is the coarse
        // ceil-rounded inflation (clearance_cells·resolution = 1·1.0 = 1.0 with
        // track_w = 1.0 → coarse channel 3.0), so the 2.0-wide obstacle gap [4,6] gets
        // NO regular lane. The BGA/LGA escape pass (lever C2) sizes a lane against the
        // TRUE rule (default clearance 0.15 → escape channel 1.3 ≤ 2.0), so it inserts
        // one midpoint escape lane at 5.0 on each axis: 10 → 11 lines per axis. (Before
        // the escape pass this was 10.) The escape lane is reachable only via a net's
        // own-pad escape halo, so it adds routing room without admitting foreign shorts.
        assert_eq!(summary.grid_w, 11);
        assert_eq!(summary.grid_h, 11);
        assert!(summary.total_cost > 0);

        assert_eq!(traces.len(), 2);
        for t in &traces {
            assert_eq!(t.kind, "pcb_trace");
            assert!(!t.route.is_empty());
        }
    }

    #[test]
    fn zero_clearance_srj_still_prevents_via_pad_overlap() {
        for (gap, expected_routed) in [(0.449, 1), (0.45, 2)] {
            let json = format!(
                r#"{{
                    "layerCount": 2,
                    "minTraceWidth": 0.15,
                    "minClearance": 0.0,
                    "bounds": {{"minX": 0.0, "maxX": {gap}, "minY": 0.0, "maxY": 0.0}},
                    "connections": [
                        {{"name": "a", "pointsToConnect": [
                            {{"x": 0.0, "y": 0.0, "layer": "top"}},
                            {{"x": 0.0, "y": 0.0, "layer": "bottom"}}
                        ]}},
                        {{"name": "b", "pointsToConnect": [
                            {{"x": {gap}, "y": 0.0, "layer": "top"}},
                            {{"x": {gap}, "y": 0.0, "layer": "bottom"}}
                        ]}}
                    ]
                }}"#
            );
            let srj = parse_srj(json.as_bytes()).expect("parse zero-clearance fixture");
            let (traces, summary, _) =
                route_problem(&srj, Some(0.05), RouterKind::Negotiated, None)
                    .expect("route zero-clearance fixture");
            assert_eq!(
                summary.routed, expected_routed,
                "unexpected physical via result at {gap} mm: {summary:?}"
            );
            let violations =
                drc_board::solution_to_drc_board(&srj, &traces, drc::default_rules(0.0), 2).check();
            assert!(
                violations.is_empty(),
                "zero-clearance route still cannot overlap copper: {violations:#?}"
            );
        }
    }

    /// Real-board regression for pad-aware exact-geometry repair. The two vias in
    /// bugreport01 initially sit 0.125 mm from foreign pads under a 0.150 mm rule;
    /// each can clear them by moving farther into its own connectivity-labelled pad.
    #[test]
    fn bugreport01_own_pad_vias_legalize_drc_clean() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/corpus/bug-reports/bugreport01-be84eb.srj.json"
        );
        let bytes = std::fs::read(path).expect("read checked-in bugreport01 fixture");
        let srj = parse_srj(&bytes).expect("parse bugreport01");
        let (traces, summary, _) =
            route_problem(&srj, None, RouterKind::Negotiated, None).expect("route bugreport01");

        assert_eq!((summary.routed, summary.total), (12, 12));
        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let violations =
            drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count).check();
        assert!(
            violations.is_empty(),
            "bugreport01 must be DRC-clean after own-pad-aware via repair: {violations:#?}"
        );
    }

    /// Regression for the historical direct segment through bugreport21's
    /// bottom-open concave cutout. The production route must remain fully connected
    /// while every emitted trace capsule and via disk passes authoritative edge DRC.
    #[test]
    fn bugreport21_routes_around_concave_board_cutout() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/corpus/bug-reports/bugreport21-board-outline.srj.json"
        );
        let bytes = std::fs::read(path).expect("read checked-in bugreport21 fixture");
        let srj = parse_srj(&bytes).expect("parse bugreport21");
        let (traces, summary, _) =
            route_problem(&srj, None, RouterKind::Negotiated, None).expect("route bugreport21");

        assert_eq!((summary.routed, summary.total), (1, 1));
        let violations = check_srj_solution(&srj, &traces, srj.layer_count);
        assert!(
            violations.is_empty(),
            "bugreport21 route must clear the concave board edge: {violations:#?}"
        );
        assert!(traces
            .iter()
            .flat_map(|trace| trace.route.windows(2))
            .all(|pair| match (&pair[0], &pair[1]) {
                (
                    RoutePoint::Wire {
                        x: ax,
                        y: ay,
                        layer: al,
                        ..
                    },
                    RoutePoint::Wire {
                        x: bx,
                        y: by,
                        layer: bl,
                        ..
                    },
                ) if al == bl => !(*ay < 2.0 && *by < 2.0 && *ax < -2.0 && *bx > 2.0),
                _ => true,
            }));
    }

    fn bugreport(name: &str) -> SimpleRouteJson {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/corpus/bug-reports")
            .join(name);
        let bytes = std::fs::read(path).expect("read checked-in bugreport fixture");
        parse_srj(&bytes).expect("parse bugreport fixture")
    }

    fn without_board_edge(mut srj: SimpleRouteJson) -> SimpleRouteJson {
        srj.physical_rules.outline.clear();
        srj.physical_rules.min_board_edge_clearance = None;
        srj
    }

    fn assert_route_output_identical(
        actual: &(Vec<mr_srj::PcbTrace>, Summary, RouteDiagnostics),
        expected: &(Vec<mr_srj::PcbTrace>, Summary, RouteDiagnostics),
        context: &str,
    ) {
        assert_eq!(actual.0, expected.0, "{context}: trace soup changed");
        assert_eq!(actual.1, expected.1, "{context}: summary changed");
        assert_eq!(
            actual.2.congestion, expected.2.congestion,
            "{context}: congestion changed"
        );
        assert_eq!(
            actual.2.x_lines, expected.2.x_lines,
            "{context}: x Hanan lines changed"
        );
        assert_eq!(
            actual.2.y_lines, expected.2.y_lines,
            "{context}: y Hanan lines changed"
        );
    }

    #[test]
    fn partial_width_outline_retry_matches_emitted_width_without_touching_inactive_legacy() {
        let mut canonical = bugreport("bugreport21-board-outline.srj.json");
        canonical.min_trace_width = Some(DEFAULT_TRACE_WIDTH);
        assert!(canonical.uniform_physical_rules().is_none());
        let expected_constrained =
            route_problem_impl(&canonical, None, RouterKind::Negotiated, None, true).unwrap();

        for declared_width in [0.05, 0.30] {
            let mut active = canonical.clone();
            active.min_trace_width = Some(declared_width);
            assert!(active.uniform_physical_rules().is_none());
            let constrained_input = legacy_board_edge_raster_input(&active);
            assert!(matches!(&constrained_input, std::borrow::Cow::Owned(_)));
            assert_eq!(
                constrained_input.min_trace_width,
                Some(DEFAULT_TRACE_WIDTH),
                "active partial masks must use emitted width, not {declared_width}"
            );

            let selected = route_problem(&active, None, RouterKind::Negotiated, None).unwrap();
            assert_route_output_identical(
                &selected,
                &expected_constrained,
                &format!("active partial width {declared_width}"),
            );
            assert!(selected
                .0
                .iter()
                .flat_map(|trace| &trace.route)
                .all(|point| {
                    !matches!(
                        point,
                        RoutePoint::Wire { width, .. }
                            if (*width - DEFAULT_TRACE_WIDTH).abs() > f64::EPSILON
                    )
                }));
            assert!(mr_srj::solution_respects_board_outline(
                &active,
                &selected.0,
                VIA_PAD_MM,
                active.layer_count.max(1),
            )
            .unwrap());

            let inactive = without_board_edge(active);
            assert!(matches!(
                legacy_board_edge_raster_input(&inactive),
                std::borrow::Cow::Borrowed(_)
            ));
            let historical =
                route_problem_impl(&inactive, None, RouterKind::Negotiated, None, true).unwrap();
            let public = route_problem(&inactive, None, RouterKind::Negotiated, None).unwrap();
            assert_route_output_identical(
                &public,
                &historical,
                &format!("inactive partial width {declared_width}"),
            );
        }
    }

    fn width_only_incoherent_typed_outline() -> SimpleRouteJson {
        serde_json::from_value(serde_json::json!({
            "layerCount": 2,
            "minTraceWidth": 0.30,
            "nominalTraceWidth": DEFAULT_TRACE_WIDTH,
            "defaultObstacleMargin": 0.04,
            "minTraceToPadEdgeClearance": 0.07,
            "minViaEdgeToPadEdgeClearance": 0.09,
            "minViaHoleEdgeToViaHoleEdgeClearance": 0.10,
            "minPadEdgeToPadEdgeClearance": 0.11,
            "minViaHoleDiameter": 0.20,
            "minViaPadDiameter": 0.40,
            "bounds": {"minX": -9.0, "maxX": 9.0, "minY": -7.0, "maxY": 7.0},
            "outline": [
                {"x": -8.0, "y": -6.0}, {"x": -2.0, "y": -6.0},
                {"x": -2.0, "y": 2.0}, {"x": 2.0, "y": 2.0},
                {"x": 2.0, "y": -6.0}, {"x": 8.0, "y": -6.0},
                {"x": 8.0, "y": 6.0}, {"x": -8.0, "y": 6.0}
            ],
            "obstacles": [
                {
                    "type": "rect", "shape": "rect",
                    "center": {"x": -4.49, "y": -4.0},
                    "width": 0.54, "height": 0.64, "layers": ["top"],
                    "connectedTo": ["n"]
                },
                {
                    "type": "rect", "shape": "rect",
                    "center": {"x": 5.51, "y": -4.0},
                    "width": 0.54, "height": 0.64, "layers": ["top"],
                    "connectedTo": ["n"]
                }
            ],
            "connections": [{
                "name": "n", "nominalTraceWidth": DEFAULT_TRACE_WIDTH,
                "pointsToConnect": [
                    {"x": -4.49, "y": -4.0, "layer": "top"},
                    {"x": 5.51, "y": -4.0, "layer": "top"}
                ]
            }]
        }))
        .unwrap()
    }

    #[test]
    fn raster_width_normalization_cannot_activate_an_incoherent_typed_profile() {
        let active = width_only_incoherent_typed_outline();
        assert!(
            active.uniform_physical_rules().is_none(),
            "the original minimum/nominal mismatch must retain legacy semantics"
        );
        let raster_input = legacy_board_edge_raster_input(&active);
        assert!(matches!(&raster_input, std::borrow::Cow::Owned(_)));
        assert!(
            raster_input.uniform_physical_rules().is_some(),
            "fixture must expose why physical selection cannot be recomputed from the raster clone"
        );
        assert_eq!(
            routed_via_pad_diameter_mm(&active),
            VIA_PAD_MM,
            "the incoherent original keeps legacy via geometry"
        );

        let mut legacy_reference = active.clone();
        let outline = legacy_reference.physical_rules.outline.clone();
        let edge_clearance = legacy_reference.physical_rules.min_board_edge_clearance;
        legacy_reference.physical_rules = Default::default();
        legacy_reference.physical_rules.outline = outline;
        legacy_reference.physical_rules.min_board_edge_clearance = edge_clearance;
        for connection in &mut legacy_reference.connections {
            connection.rules = Default::default();
        }
        assert!(legacy_reference.uniform_physical_rules().is_none());

        let inactive = without_board_edge(active.clone());
        let legacy =
            route_problem_impl(&inactive, Some(0.5), RouterKind::Negotiated, None, true).unwrap();
        assert!(
            !mr_srj::solution_respects_board_outline(
                &active,
                &legacy.0,
                VIA_PAD_MM,
                active.layer_count,
            )
            .unwrap(),
            "fixture must force the constrained product branch"
        );

        let selected = route_problem(&active, Some(0.5), RouterKind::Negotiated, None).unwrap();
        let expected =
            route_problem(&legacy_reference, Some(0.5), RouterKind::Negotiated, None).unwrap();
        assert_route_output_identical(
            &selected,
            &expected,
            "raster-only normalization must preserve legacy clearance/via semantics",
        );
        assert!(selected
            .0
            .iter()
            .flat_map(|trace| &trace.route)
            .all(|point| {
                !matches!(
                    point,
                    RoutePoint::Wire { width, .. }
                        if (*width - DEFAULT_TRACE_WIDTH).abs() > f64::EPSILON
                )
            }));
        assert!(mr_srj::solution_respects_board_outline(
            &active,
            &selected.0,
            VIA_PAD_MM,
            active.layer_count,
        )
        .unwrap());
    }

    #[test]
    fn bugreport21_rejects_legacy_crossing_then_selects_clean_constrained_route() {
        let srj = bugreport("bugreport21-board-outline.srj.json");
        let legacy_srj = without_board_edge(srj.clone());
        let (legacy_traces, _, _) =
            route_problem_impl(&legacy_srj, None, RouterKind::Negotiated, None, true).unwrap();
        assert!(
            !mr_srj::solution_respects_board_outline(
                &srj,
                &legacy_traces,
                VIA_PAD_MM,
                srj.layer_count.max(1),
            )
            .unwrap(),
            "the historical direct path must be rejected by exact final-soup validation"
        );

        let constrained =
            route_problem_impl(&srj, None, RouterKind::Negotiated, None, true).unwrap();
        assert!(mr_srj::solution_respects_board_outline(
            &srj,
            &constrained.0,
            VIA_PAD_MM,
            srj.layer_count.max(1),
        )
        .unwrap());
        let selected = route_problem(&srj, None, RouterKind::Negotiated, None).unwrap();
        assert_eq!(
            serde_json::to_vec(&selected.0).unwrap(),
            serde_json::to_vec(&constrained.0).unwrap(),
            "edge-invalid legacy geometry must select the deterministic constrained rerun"
        );
        let legacy_drc = check_srj_solution(&srj, &legacy_traces, srj.layer_count);
        let selected_drc = check_srj_solution(&srj, &selected.0, srj.layer_count);
        assert!(drc_candidate_is_not_worse(&legacy_drc, &selected_drc));
    }

    #[test]
    #[ignore = "affected real-board regression; run explicitly in release"]
    fn bugreport49_accepts_the_safe_full_drc_improvement() {
        let srj = bugreport("bugreport49-8536f4.srj.json");
        let legacy_srj = without_board_edge(srj.clone());
        let legacy =
            route_problem_impl(&legacy_srj, None, RouterKind::Negotiated, None, true).unwrap();
        assert!(!mr_srj::solution_respects_board_outline(
            &srj,
            &legacy.0,
            routed_via_pad_diameter_mm(&srj),
            srj.layer_count.max(1),
        )
        .unwrap());

        let selected = route_problem(&srj, None, RouterKind::Negotiated, None)
            .expect("an unsafe legacy soup must not block a safer constrained improvement");
        assert!(mr_srj::solution_respects_board_outline(
            &srj,
            &selected.0,
            routed_via_pad_diameter_mm(&srj),
            srj.layer_count.max(1),
        )
        .unwrap());
        let legacy_drc = check_srj_solution(&srj, &legacy.0, srj.layer_count);
        let selected_drc = check_srj_solution(&srj, &selected.0, srj.layer_count);
        assert!(drc_candidate_is_not_worse(&legacy_drc, &selected_drc));
    }

    fn assert_edge_clean_fixture_preserves_exact_legacy_route(name: &str) {
        let srj = bugreport(name);
        let legacy_srj = without_board_edge(srj.clone());
        let legacy = route_problem_impl(&legacy_srj, None, RouterKind::Negotiated, None, true)
            .expect("route legacy topology");
        let via_pad = routed_via_pad_diameter_mm(&srj);
        assert!(
            mr_srj::solution_respects_board_outline(
                &srj,
                &legacy.0,
                via_pad,
                srj.layer_count.max(1),
            )
            .unwrap(),
            "fixture precondition changed: {name} legacy output is no longer edge-clean"
        );

        let selected = route_problem(&srj, None, RouterKind::Negotiated, None)
            .expect("route legacy-first portfolio");
        assert_eq!(
            serde_json::to_vec(&selected.0).unwrap(),
            serde_json::to_vec(&legacy.0).unwrap(),
            "{name} must preserve exact serialized legacy trace bytes"
        );
        assert_eq!(selected.1, legacy.1, "{name} summary changed");
        assert_eq!(selected.2.congestion, legacy.2.congestion);
        assert_eq!(selected.2.x_lines, legacy.2.x_lines);
        assert_eq!(selected.2.y_lines, legacy.2.y_lines);
    }

    #[test]
    #[ignore = "affected real-board byte-regression cohort; run explicitly in release"]
    fn bugreports27_33_35_preserve_edge_clean_legacy_bytes() {
        for name in [
            "bugreport27-dd3734.srj.json",
            "bugreport33-213d45.srj.json",
            "bugreport35-191db9.srj.json",
        ] {
            assert_edge_clean_fixture_preserves_exact_legacy_route(name);
        }
    }

    #[test]
    #[ignore = "affected real-board regression; run explicitly in release"]
    fn bugreport55_normalized_outline_routes_without_a_hard_error() {
        let srj = bugreport("bugreport55-b7c349.srj.json");
        let legacy_srj = without_board_edge(srj.clone());
        let legacy = route_problem_impl(&legacy_srj, None, RouterKind::Negotiated, None, true)
            .expect("route bugreport55 legacy topology");
        let legacy_drc = check_srj_solution(&srj, &legacy.0, srj.layer_count);
        let (traces, summary, _) = route_problem(&srj, None, RouterKind::Negotiated, None)
            .expect("the proven collinear backtracking spur is normalized");
        let selected_drc = check_srj_solution(&srj, &traces, srj.layer_count);
        let edge_count = |findings: &[mr_drc::Violation]| {
            findings
                .iter()
                .filter(|finding| {
                    finding.nets.0 == "__board_edge__" || finding.nets.1 == "__board_edge__"
                })
                .count()
        };
        assert_eq!(
            (legacy.1.routed, legacy.1.total, legacy.1.total_cost),
            (10, 10, 1461)
        );
        assert_eq!((legacy_drc.len(), edge_count(&legacy_drc)), (23, 11));
        assert_eq!(
            (summary.routed, summary.total, summary.total_cost),
            (8, 10, 833),
            "board-edge safety deliberately costs two bug55 routes"
        );
        assert_eq!((selected_drc.len(), edge_count(&selected_drc)), (12, 0));
        assert!(mr_srj::solution_respects_board_outline(
            &srj,
            &traces,
            routed_via_pad_diameter_mm(&srj),
            srj.layer_count.max(1),
        )
        .unwrap());
        assert!(edge_count(&legacy_drc) > 0);
        assert_eq!(edge_count(&selected_drc), 0);
        assert!(drc_candidate_is_not_worse(&legacy_drc, &selected_drc));
    }

    #[test]
    fn route_problem_rejects_a_non_collinear_bowtie_before_routing() {
        let srj: SimpleRouteJson = serde_json::from_value(serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "outline": [
                {"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0}, {"x": 10.0, "y": 0.0}
            ]
        }))
        .unwrap();
        let error = route_problem(&srj, None, RouterKind::Negotiated, None).unwrap_err();
        assert!(error.to_string().contains("invalid board-edge contract"));
    }

    /// Coarse bounds-derived fill spacing must not inflate the physical clearance
    /// rule by fourfold. bugreport23's 0.15 mm rule previously became one 0.618 mm
    /// fill interval, stranding 35 of 45 connections even in isolation. The exact
    /// Hanan halo should recover useful routes without spending DRC correctness.
    #[test]
    fn bugreport23_exact_clearance_recovers_solo_routes_without_drc_regression() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/corpus/bug-reports/bugreport23-LGA15x4.srj.json"
        );
        let bytes = std::fs::read(path).expect("read checked-in bugreport23 fixture");
        let srj = parse_srj(&bytes).expect("parse bugreport23");
        let (traces, summary, _) =
            route_problem_impl(&srj, None, RouterKind::Negotiated, None, false)
                .expect("route bugreport23 without via repair");

        let solo_unroutable = summary
            .unrouted
            .iter()
            .filter(|(_, reason)| *reason == UnroutedReason::UnroutableAlone)
            .count();
        assert!(
            summary.routed >= 26,
            "exact clearance should recover the observed 26/45 completion floor from the \
             rounded-halo baseline 6/45; got {}/45",
            summary.routed
        );
        assert_eq!(
            solo_unroutable, 0,
            "the rounded-halo baseline had 35 nets unable to route even in isolation"
        );

        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let violations =
            drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count).check();
        assert!(
            violations.is_empty(),
            "completion recovery must retain bugreport23's zero-DRC baseline: {violations:#?}"
        );
    }

    #[test]
    fn rasterized_through_pad_routes_its_owned_masked_via_without_repair() {
        let srj: SimpleRouteJson = serde_json::from_value(serde_json::json!({
            "layerCount": 2,
            "minClearance": 0.15,
            "minTraceWidth": 0.15,
            "bounds": { "minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0 },
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.4, "height": 0.4, "layers": ["top", "bottom"]
            }],
            "connections": [{
                "name": "through",
                "pointsToConnect": [
                    {"x": 0.0, "y": 0.0, "layer": "top"},
                    {"x": 0.0, "y": 0.0, "layer": "bottom"}
                ]
            }]
        }))
        .unwrap();
        let (traces, summary, _) =
            route_problem_impl(&srj, Some(1.0), RouterKind::Negotiated, None, false)
                .expect("route a real rasterized through pad");
        assert_eq!((summary.routed, summary.total), (1, 1));
        assert!(traces[0]
            .route
            .iter()
            .any(|point| matches!(point, RoutePoint::Via { .. })));
    }

    type EndpointViaSignature = (
        Option<String>,
        RoutePoint,
        RoutePoint,
        Vec<(String, String)>,
    );

    fn endpoint_and_via_span_signature(traces: &[mr_srj::PcbTrace]) -> Vec<EndpointViaSignature> {
        traces
            .iter()
            .map(|trace| {
                let spans = trace
                    .route
                    .iter()
                    .filter_map(|point| match point {
                        RoutePoint::Via {
                            from_layer,
                            to_layer,
                            ..
                        } => Some((from_layer.clone(), to_layer.clone())),
                        RoutePoint::Wire { .. } => None,
                    })
                    .collect();
                (
                    trace.net.clone(),
                    trace.route.first().cloned().expect("non-empty trace"),
                    trace.route.last().cloned().expect("non-empty trace"),
                    spans,
                )
            })
            .collect()
    }

    fn assert_real_via_repair(
        relative_path: &str,
        expected_routed: usize,
        expected_total: usize,
        expected_before: usize,
        expected_after: usize,
    ) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative_path);
        let bytes = std::fs::read(&path).expect("read checked-in via-repair fixture");
        let srj = parse_srj(&bytes).expect("parse via-repair fixture");
        let (before_traces, summary, _) =
            route_problem_impl(&srj, None, RouterKind::Negotiated, None, false)
                .expect("route before bounded via repair");
        assert_eq!(
            (summary.routed, summary.total),
            (expected_routed, expected_total)
        );

        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let before =
            drc_board::solution_to_drc_board(&srj, &before_traces, rules, srj.layer_count).check();
        assert_eq!(before.len(), expected_before, "unexpected fixture baseline");
        let signature = endpoint_and_via_span_signature(&before_traces);

        let repaired =
            via_repair::repair_clearance_vias(&srj, before_traces, rules, srj.layer_count);
        let after =
            drc_board::solution_to_drc_board(&srj, &repaired, rules, srj.layer_count).check();
        assert_eq!(after.len(), expected_after, "unexpected repaired DRC count");
        assert_eq!(
            drc_candidate_is_better(&before, &after),
            expected_after < expected_before
        );
        assert_eq!(endpoint_and_via_span_signature(&repaired), signature);
    }

    /// Dense four-layer fixture: feature-aware spacing routes all 26 nets with nine
    /// accepted-rollout findings. None admits a beneficial rigid via translation,
    /// and all routed endpoints/spans stay fixed.
    #[test]
    fn sample11_combined_clearance_guards_are_repair_stable() {
        assert_real_via_repair(
            "benchmarks/corpus/srj15/sample11-region-reroute.srj.json",
            26,
            26,
            9,
            9,
        );
    }

    /// Small independent fixture: the combined guards eliminate all findings during
    /// routing, so bounded post-route repair is a deterministic no-op.
    #[test]
    fn sample25_combined_clearance_guards_are_repair_stable() {
        assert_real_via_repair(
            "benchmarks/corpus/srj15/sample25-region-reroute.srj.json",
            5,
            5,
            0,
            0,
        );
    }

    /// The final exact-mask rollout routes this board completely. Feature-aware
    /// dynamic via spacing must preserve 12/12 while staying no worse than the
    /// accepted rollout's five DRC findings (the combined route currently has 3).
    #[test]
    fn bug62_feature_aware_route_preserves_full_board_and_improves_drc() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/corpus/bug-reports/bugreport62-0f6ca4.srj.json"
        );
        let bytes = std::fs::read(path).expect("read checked-in bugreport62 fixture");
        let srj = parse_srj(&bytes).expect("parse bugreport62");
        let (traces, summary, _) =
            route_problem(&srj, None, RouterKind::Negotiated, None).expect("route bugreport62");
        assert_eq!((summary.routed, summary.total), (12, 12));

        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let violations =
            drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count).check();
        assert!(
            violations.len() <= 3,
            "combined route must retain the observed DRC<=3 improvement: {violations:#?}"
        );
    }

    /// An alternate deterministic legalization order can retain the rollout's full
    /// completion without restoring either of its two invalid clearance findings.
    #[test]
    fn bug63_order_portfolio_routes_full_and_clean() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/corpus/bug-reports/bugreport63-274be2.srj.json"
        );
        let bytes = std::fs::read(path).expect("read checked-in bugreport63 fixture");
        let srj = parse_srj(&bytes).expect("parse bugreport63");
        let (traces, summary, _) =
            route_problem(&srj, None, RouterKind::Negotiated, None).expect("route bugreport63");
        assert_eq!((summary.routed, summary.total), (12, 12));

        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let violations =
            drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count).check();
        assert!(
            violations.is_empty(),
            "portfolio result must stay DRC-clean: {violations:#?}"
        );
    }

    /// A dependency-guided legalization restart must recover one additional route
    /// without worsening this board's existing exact-clearance findings.
    #[test]
    fn bug28_dependency_portfolio_recovers_one_route_without_drc_regression() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/corpus/bug-reports/bugreport28-18a9ef.srj.json"
        );
        let bytes = std::fs::read(path).expect("read checked-in bugreport28 fixture");
        let srj = parse_srj(&bytes).expect("parse bugreport28");
        let (traces, summary, _) =
            route_problem(&srj, None, RouterKind::Negotiated, None).expect("route bugreport28");
        assert_eq!((summary.routed, summary.total), (12, 14));

        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let violations =
            drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count).check();
        assert_eq!(
            violations.len(),
            18,
            "the retained route must not worsen exact DRC: {violations:#?}"
        );
    }

    fn assert_dihedral_order_fixture(
        relative_path: &str,
        expected_routed: usize,
        expected_total: usize,
        expected_drc: usize,
    ) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative_path);
        let bytes = std::fs::read(path).expect("read checked-in order fixture");
        let srj = parse_srj(&bytes).expect("parse order fixture");
        let (traces, summary, _) =
            route_problem(&srj, None, RouterKind::Negotiated, None).expect("route order fixture");
        assert_eq!(
            (summary.routed, summary.total),
            (expected_routed, expected_total)
        );

        let rules = drc::default_rules(srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM));
        let violations =
            drc_board::solution_to_drc_board(&srj, &traces, rules, srj.layer_count).check();
        assert_eq!(
            violations.len(),
            expected_drc,
            "the retained route must preserve exact DRC: {violations:#?}"
        );
    }

    /// Starting the reversed traversal at the opposite claimant recovers one more
    /// route without changing this board's fixed exact-clearance findings.
    #[test]
    fn bug27_dihedral_order_recovers_one_route_without_drc_regression() {
        assert_dihedral_order_fixture(
            "benchmarks/corpus/bug-reports/bugreport27-dd3734.srj.json",
            12,
            14,
            18,
        );
    }

    /// The opposite reversed traversal closes the final congestion gap while
    /// retaining the board's accepted exact-clearance count.
    #[test]
    fn bug30_dihedral_order_routes_full_without_drc_regression() {
        assert_dihedral_order_fixture(
            "benchmarks/corpus/bug-reports/bugreport30-2174c8.srj.json",
            12,
            12,
            18,
        );
    }

    /// Trying the opposite cyclic claimant recovers another route and keeps the
    /// previously clean board free of exact DRC findings.
    #[test]
    fn bug36_dihedral_order_recovers_one_route_and_stays_clean() {
        assert_dihedral_order_fixture(
            "benchmarks/corpus/bug-reports/bugreport36-d4c6c2.srj.json",
            7,
            8,
            0,
        );
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
    fn negotiated_cached_unrouted_classification_matches_solo_routes() {
        let dims = mr_core::Dims::new(5, 5);
        let mut grid = Grid::filled(dims, 1);
        for (x, y) in [(2, 1), (1, 2), (3, 2), (2, 3)] {
            grid.set(dims.idx(x, y), mr_core::OBSTACLE);
        }
        let nets = vec![
            NetEndpoints {
                net: "open".into(),
                src: dims.idx(0, 4),
                dst: dims.idx(4, 4),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "blocked".into(),
                src: dims.idx(0, 0),
                dst: dims.idx(2, 2),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "zero".into(),
                src: dims.idx(4, 0),
                dst: dims.idx(4, 0),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        let coords = GridCoords::uniform(dims);
        let via_model = ViaModel::through_hole(dims.layers);
        let outcome = NegotiatedRouter::new()
            .with_via_model(via_model.clone())
            .with_coords(coords.clone())
            .route_with_outcome(&grid, &nets)
            .unwrap();
        assert_eq!(outcome.alone_routable, [true, false, true]);

        let names: Vec<String> = nets.iter().map(|net| net.net.clone()).collect();
        let cached = classify_unrouted(
            RouterKind::Negotiated,
            &grid,
            &nets,
            &names,
            &via_model,
            &coords,
            Some(&outcome.alone_routable),
        );
        let solo = classify_unrouted(
            RouterKind::Negotiated,
            &grid,
            &nets,
            &names,
            &via_model,
            &coords,
            None,
        );
        assert_eq!(cached, solo);
        assert_eq!(
            cached,
            vec![
                ("open".into(), UnroutedReason::Congested),
                ("blocked".into(), UnroutedReason::UnroutableAlone),
                ("zero".into(), UnroutedReason::Congested),
            ]
        );
    }

    #[test]
    fn default_resolution_targets_reasonable_grid() {
        let srj = parse_srj(SAMPLE.as_bytes()).unwrap();
        // span 10 / 64 -> small cells; grid should be well-formed and non-trivial.
        let (_traces, summary, _diag) = route_problem(&srj, None, RouterKind::Ripup, None).unwrap();
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
