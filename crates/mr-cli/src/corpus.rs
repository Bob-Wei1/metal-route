//! `bench-corpus` — route a corpus of *real* SimpleRouteJson boards and report
//! per-board completion, with an optional self-contained SVG gallery.
//!
//! Unlike [`crate::bench`] (which generates synthetic random-obstacle problems),
//! this routes the real circuit-derived boards vendored under
//! `benchmarks/corpus/` (see that directory's `MANIFEST.md`). Every `*.srj.json`
//! beneath the corpus root is routed; the immediate sub-directory name groups
//! boards in the report (`srj15`, `bug-reports`, …).
//!
//! The SVG renderer is deliberately dependency-free Rust (no Node / `circuit-to-svg`)
//! so the gallery reproduces from a clean checkout with just `cargo run`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::Serialize;

use crate::drc::default_rules;
use crate::drc_board::solution_to_drc_board;
use crate::{default_resolution, parse_srj, route_problem, RouteDiagnostics, RouterKind, UnroutedReason};
use mr_drc::Violation;
use mr_srj::{Obstacle, PcbTrace, RoutePoint, SimpleRouteJson};

/// Copper-to-copper clearance (mm) assumed when a board omits `minClearance` —
/// the same default the DSN ingest uses, so the corpus DRC mirrors `route-dsn`.
const DEFAULT_CLEARANCE_MM: f64 = 0.15;

/// Arguments for the `bench-corpus` subcommand.
#[derive(Debug, clap::Parser)]
pub struct CorpusArgs {
    /// Corpus root. Every `*.srj.json` beneath it (recursively) is routed.
    #[arg(long, default_value = "benchmarks/corpus")]
    pub dir: PathBuf,

    /// Routing backend (real boards are multi-layer, so `negotiated` by default).
    #[arg(long, value_enum, default_value_t = RouterKind::Negotiated)]
    pub router: RouterKind,

    /// Override the routed layer count (defaults to each board's `layerCount`).
    #[arg(long)]
    pub layers: Option<u32>,

    /// Cell size; defaults to the same bounds-derived heuristic as `route`.
    #[arg(long)]
    pub resolution: Option<f64>,

    /// Write the JSON report here (a summary always goes to stderr).
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// If set, render one SVG per board + an `index.html` gallery into this dir.
    #[arg(long)]
    pub svg_out: Option<PathBuf>,

    /// Skip (and flag) any board whose predicted grid exceeds this many cells —
    /// a guard so one pathological board can't hang the whole sweep.
    #[arg(long, default_value_t = 12_000_000)]
    pub max_cells: u64,
}

/// One unrouted net and the diagnosed reason it failed.
#[derive(Debug, Clone, Serialize)]
pub struct UnroutedNet {
    pub name: String,
    pub reason: UnroutedReason,
}

/// Count of unrouted nets by [`UnroutedReason`] — the headline diagnostic that
/// says whether failures are resolution-bound (`unroutable_alone`) or the
/// algorithm leaving routable nets on the table (`congested`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct ReasonHistogram {
    /// Nets with no path even alone on the base grid (a grid/resolution limit).
    pub unroutable_alone: usize,
    /// Nets that route fine alone but were lost to congestion (an algorithm limit).
    pub congested: usize,
}

impl ReasonHistogram {
    fn add(&mut self, reason: UnroutedReason) {
        match reason {
            UnroutedReason::UnroutableAlone => self.unroutable_alone += 1,
            UnroutedReason::Congested => self.congested += 1,
        }
    }
}

/// Per-board result.
#[derive(Debug, Clone, Serialize)]
pub struct BoardResult {
    pub corpus: String,
    pub board: String,
    pub nets_total: usize,
    pub nets_routed: usize,
    pub total_cost: u64,
    pub grid_w: u32,
    pub grid_h: u32,
    pub grid_layers: u32,
    pub wall_ms: f64,
    /// `None` on success; a message when the board was skipped or errored.
    pub error: Option<String>,
    /// Each unrouted net with its diagnosed failure reason (empty on a full board).
    pub unrouted: Vec<UnroutedNet>,
    /// Peak per-cell occupancy across routed copper — a congestion proxy.
    pub congestion_peak: u32,
    /// Total geometric DRC violations in the routed solution (0 on skip/error).
    /// The benchmark previously NEVER ran DRC, so clearance/via shorts were
    /// produced but invisible; this makes them a first-class, serialized metric.
    pub drc_violations: usize,
    /// DRC violations broken down by [`mr_drc::ViolationClass`] (debug name →
    /// count), e.g. `{"Clearance": 3, "AnnularRing": 1}`. Empty when clean.
    pub drc_by_class: BTreeMap<String, usize>,
}

impl BoardResult {
    fn completion(&self) -> f64 {
        if self.nets_total == 0 {
            0.0
        } else {
            self.nets_routed as f64 / self.nets_total as f64
        }
    }
}

/// Per-corpus (sub-directory) aggregate.
#[derive(Debug, Clone, Serialize)]
pub struct CorpusGroup {
    pub name: String,
    pub boards: usize,
    pub nets_total: usize,
    pub nets_routed: usize,
    pub completion_rate: f64,
    pub fully_routed_boards: usize,
    pub total_wall_ms: f64,
    /// Why this group's unrouted nets failed (congestion vs. resolution).
    pub unrouted_reasons: ReasonHistogram,
}

/// The full corpus report — the checked-in real-board baseline.
#[derive(Debug, Clone, Serialize)]
pub struct CorpusReport {
    pub router: String,
    pub boards: usize,
    pub nets_total: usize,
    pub nets_routed: usize,
    pub completion_rate: f64,
    pub fully_routed_boards: usize,
    pub total_wall_ms: f64,
    pub nets_per_sec: f64,
    /// Corpus-wide split of unrouted nets by failure reason — the headline that
    /// drives the diagnose-then-fix decision.
    pub unrouted_reasons: ReasonHistogram,
    pub groups: Vec<CorpusGroup>,
    pub per_board: Vec<BoardResult>,
}

/// Recursively collect every `*.srj.json` under `root`, sorted for determinism.
fn collect_srj(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .with_context(|| format!("failed to read corpus dir {}", dir.display()))?;
        for entry in entries {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".srj.json"))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The corpus group a board belongs to: its first path component under `root`.
/// Boards sitting directly in `root` (so `root` is itself a leaf corpus, e.g.
/// `--dir benchmarks/corpus/srj15`) are grouped under `root`'s own dir name.
fn group_of(root: &Path, file: &Path) -> String {
    let root_name = || {
        root.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    };
    let Ok(rel) = file.strip_prefix(root) else {
        return root_name();
    };
    let comps: Vec<_> = rel.components().collect();
    if comps.len() <= 1 {
        root_name()
    } else {
        comps[0].as_os_str().to_string_lossy().into_owned()
    }
}

/// Predicted grid cell count, used by the `max_cells` guard.
fn predicted_cells(srj: &SimpleRouteJson, resolution: Option<f64>, layers: Option<u32>) -> u64 {
    let res = resolution.unwrap_or_else(|| default_resolution(srj));
    if !(res.is_finite() && res > 0.0) {
        return u64::MAX;
    }
    let w = ((srj.bounds.max_x - srj.bounds.min_x) / res).ceil().max(1.0) as u64;
    let h = ((srj.bounds.max_y - srj.bounds.min_y) / res).ceil().max(1.0) as u64;
    let l = layers.unwrap_or(srj.layer_count).max(1) as u64;
    w.saturating_mul(h).saturating_mul(l)
}

/// Tally a violation slice by its [`mr_drc::ViolationClass`] debug name.
fn drc_class_breakdown(violations: &[Violation]) -> BTreeMap<String, usize> {
    let mut by_class = BTreeMap::new();
    for v in violations {
        *by_class.entry(format!("{:?}", v.class)).or_insert(0) += 1;
    }
    by_class
}

/// Route one board, returning its [`BoardResult`], the traces (for SVG), the
/// route diagnostics, and the geometric DRC violations found in the solution.
fn route_board(
    corpus: &str,
    board: &str,
    srj: &SimpleRouteJson,
    args: &CorpusArgs,
) -> (BoardResult, Vec<PcbTrace>, RouteDiagnostics, Vec<Violation>) {
    let nets_total: usize = srj
        .connections
        .iter()
        .map(|c| c.points_to_connect.len().saturating_sub(1).max(0))
        .sum();

    let cells = predicted_cells(srj, args.resolution, args.layers);
    if cells > args.max_cells {
        return (
            BoardResult {
                corpus: corpus.to_string(),
                board: board.to_string(),
                nets_total,
                nets_routed: 0,
                total_cost: 0,
                grid_w: 0,
                grid_h: 0,
                grid_layers: 0,
                wall_ms: 0.0,
                error: Some(format!("skipped: predicted {cells} cells > max-cells")),
                unrouted: Vec::new(),
                congestion_peak: 0,
                drc_violations: 0,
                drc_by_class: BTreeMap::new(),
            },
            Vec::new(),
            RouteDiagnostics::default(),
            Vec::new(),
        );
    }

    let t0 = Instant::now();
    let routed = route_problem(srj, args.resolution, args.router, args.layers);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    match routed {
        Ok((traces, summary, diag)) => {
            let unrouted = summary
                .unrouted
                .iter()
                .map(|(name, reason)| UnroutedNet { name: name.clone(), reason: *reason })
                .collect();
            let congestion_peak = diag.congestion.iter().copied().max().unwrap_or(0);

            // Run a real geometric DRC over what the router actually drew. The
            // benchmark never did this before, so clearance/via shorts were
            // produced but invisible. Build the physical board from the emitted
            // solution and check it.
            let clearance = srj.min_clearance.unwrap_or(DEFAULT_CLEARANCE_MM);
            let rules = default_rules(clearance);
            let layers = args.layers.unwrap_or(srj.layer_count);
            let violations = solution_to_drc_board(srj, &traces, rules, layers).check();
            let drc_by_class = drc_class_breakdown(&violations);

            (
                BoardResult {
                    corpus: corpus.to_string(),
                    board: board.to_string(),
                    nets_total: summary.total,
                    nets_routed: summary.routed,
                    total_cost: summary.total_cost,
                    grid_w: summary.grid_w,
                    grid_h: summary.grid_h,
                    grid_layers: summary.grid_layers,
                    wall_ms,
                    error: None,
                    unrouted,
                    congestion_peak,
                    drc_violations: violations.len(),
                    drc_by_class,
                },
                traces,
                diag,
                violations,
            )
        }
        Err(e) => (
            BoardResult {
                corpus: corpus.to_string(),
                board: board.to_string(),
                nets_total,
                nets_routed: 0,
                total_cost: 0,
                grid_w: 0,
                grid_h: 0,
                grid_layers: 0,
                wall_ms,
                error: Some(format!("error: {e}")),
                unrouted: Vec::new(),
                congestion_peak: 0,
                drc_violations: 0,
                drc_by_class: BTreeMap::new(),
            },
            Vec::new(),
            RouteDiagnostics::default(),
            Vec::new(),
        ),
    }
}

/// Run the whole corpus and produce the report (rendering the gallery if asked).
pub fn run_corpus(args: &CorpusArgs) -> Result<CorpusReport> {
    let root = &args.dir;
    anyhow::ensure!(
        root.is_dir(),
        "corpus dir {} does not exist (run scripts/vendor-corpus.sh first)",
        root.display()
    );
    let files = collect_srj(root)?;
    anyhow::ensure!(!files.is_empty(), "no *.srj.json found under {}", root.display());

    if let Some(svg_dir) = &args.svg_out {
        std::fs::create_dir_all(svg_dir)
            .with_context(|| format!("failed to create svg-out dir {}", svg_dir.display()))?;
    }

    // Route every board IN PARALLEL — boards are fully independent (no shared
    // state), so this is an embarrassingly parallel fan-out across cores and the
    // dominant wall-clock win for the eval loop. Side effects (SVG writes, the
    // per-board log) are hoisted OUT of the parallel section and replayed in file
    // order afterwards, so the report and the SVG gallery stay deterministic
    // regardless of completion order. Each board's own router may itself use rayon
    // internally; nested rayon parallelism composes via work-stealing.
    let outputs: Vec<(BoardResult, Option<String>)> = files
        .par_iter()
        .map(|file| {
            let corpus = group_of(root, file);
            let board = file
                .file_name()
                .map(|n| n.to_string_lossy().trim_end_matches(".srj.json").to_string())
                .unwrap_or_default();

            let err_result = |msg: String| BoardResult {
                corpus: corpus.clone(),
                board: board.clone(),
                nets_total: 0,
                nets_routed: 0,
                total_cost: 0,
                grid_w: 0,
                grid_h: 0,
                grid_layers: 0,
                wall_ms: 0.0,
                error: Some(msg),
                unrouted: Vec::new(),
                congestion_peak: 0,
                drc_violations: 0,
                drc_by_class: BTreeMap::new(),
            };

            let bytes = match std::fs::read(file) {
                Ok(b) => b,
                Err(e) => return (err_result(format!("read failed: {e}")), None),
            };
            let srj = match parse_srj(&bytes) {
                Ok(s) => s,
                Err(e) => return (err_result(format!("parse failed: {e}")), None),
            };

            let (result, traces, diag, violations) = route_board(&corpus, &board, &srj, args);
            // Render SVG content here (CPU-bound, parallel-safe); the actual file
            // write happens sequentially below to keep I/O ordered and fallible.
            let svg = args
                .svg_out
                .as_ref()
                .map(|_| render_svg(&srj, &traces, &result, &diag, &violations));
            (result, svg)
        })
        .collect();

    let mut per_board = Vec::with_capacity(files.len());
    let mut gallery: Vec<(BoardResult, String)> = Vec::new();
    for (result, svg) in outputs {
        if let (Some(svg_dir), Some(svg)) = (&args.svg_out, svg) {
            let svg_name = format!("{}__{}.svg", result.corpus, result.board);
            let svg_path = svg_dir.join(svg_name.replace('/', "_"));
            std::fs::write(&svg_path, &svg)
                .with_context(|| format!("failed to write {}", svg_path.display()))?;
            gallery.push((
                result.clone(),
                svg_path.file_name().unwrap().to_string_lossy().into_owned(),
            ));
        }
        eprintln!(
            "  {:<14} {:<44} {:>3}/{:<3} nets  {:>6.0}ms{}",
            result.corpus,
            result.board,
            result.nets_routed,
            result.nets_total,
            result.wall_ms,
            result
                .error
                .as_deref()
                .map(|e| format!("  [{e}]"))
                .unwrap_or_default(),
        );
        per_board.push(result);
    }

    let report = aggregate(args.router, per_board);

    if let (Some(svg_dir), false) = (&args.svg_out, gallery.is_empty()) {
        let html = render_gallery(&report, &gallery);
        std::fs::write(svg_dir.join("index.html"), html)?;
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(svg_dir.join("summary.json"), json)?;
    }

    if let Some(out) = &args.out {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(out, json)
            .with_context(|| format!("failed to write report {}", out.display()))?;
    }

    Ok(report)
}

/// Fold per-board results into the grouped + overall report.
fn aggregate(router: RouterKind, per_board: Vec<BoardResult>) -> CorpusReport {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, CorpusGroup> = BTreeMap::new();
    let (mut nets_total, mut nets_routed, mut wall, mut fully) = (0usize, 0usize, 0.0f64, 0usize);

    let mut reasons = ReasonHistogram::default();
    for b in &per_board {
        let g = groups.entry(b.corpus.clone()).or_insert_with(|| CorpusGroup {
            name: b.corpus.clone(),
            boards: 0,
            nets_total: 0,
            nets_routed: 0,
            completion_rate: 0.0,
            fully_routed_boards: 0,
            total_wall_ms: 0.0,
            unrouted_reasons: ReasonHistogram::default(),
        });
        g.boards += 1;
        g.nets_total += b.nets_total;
        g.nets_routed += b.nets_routed;
        g.total_wall_ms += b.wall_ms;
        for u in &b.unrouted {
            g.unrouted_reasons.add(u.reason);
            reasons.add(u.reason);
        }
        let full = b.error.is_none() && b.nets_total > 0 && b.nets_routed == b.nets_total;
        if full {
            g.fully_routed_boards += 1;
            fully += 1;
        }
        nets_total += b.nets_total;
        nets_routed += b.nets_routed;
        wall += b.wall_ms;
    }
    for g in groups.values_mut() {
        g.completion_rate = if g.nets_total > 0 {
            g.nets_routed as f64 / g.nets_total as f64
        } else {
            0.0
        };
    }

    CorpusReport {
        router: format!("{router:?}").to_lowercase(),
        boards: per_board.len(),
        nets_total,
        nets_routed,
        completion_rate: if nets_total > 0 {
            nets_routed as f64 / nets_total as f64
        } else {
            0.0
        },
        fully_routed_boards: fully,
        total_wall_ms: wall,
        nets_per_sec: if wall > 0.0 {
            nets_routed as f64 / (wall / 1000.0)
        } else {
            0.0
        },
        unrouted_reasons: reasons,
        groups: groups.into_values().collect(),
        per_board,
    }
}

// ---------------------------------------------------------------------------
// Self-contained SVG rendering
// ---------------------------------------------------------------------------

/// A distinct, readable color per net index (golden-angle hue walk).
fn net_color(i: usize) -> String {
    let hue = (i as f64 * 137.508) % 360.0;
    format!("hsl({:.0} 80% 55%)", hue)
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Render a board (obstacles + routed traces) to a standalone SVG string.
///
/// PCB space is y-up; SVG is y-down, so the drawing is wrapped in a flip
/// transform. Trace widths and coordinates are therefore in board units.
fn render_svg(
    srj: &SimpleRouteJson,
    traces: &[PcbTrace],
    res: &BoardResult,
    diag: &RouteDiagnostics,
    violations: &[Violation],
) -> String {
    let b = &srj.bounds;
    let (w, h) = (b.max_x - b.min_x, b.max_y - b.min_y);
    let (w, h) = (if w > 0.0 { w } else { 1.0 }, if h > 0.0 { h } else { 1.0 });
    // A small margin so edge traces aren't clipped.
    let m = (w.max(h)) * 0.04;
    let stroke = (w.max(h)) * 0.003;

    let mut s = String::new();
    s.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.3} {:.3} {:.3} {:.3}" font-family="monospace">"##,
        b.min_x - m,
        b.min_y - m,
        w + 2.0 * m,
        h + 2.0 * m,
    ));
    // Flip Y about the board's vertical midline.
    s.push_str(&format!(
        r##"<g transform="translate(0 {:.3}) scale(1 -1)">"##,
        b.min_y + b.max_y
    ));

    // Board background.
    s.push_str(&format!(
        r##"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="#0b1020"/>"##,
        b.min_x, b.min_y, w, h
    ));

    // Congestion / utilisation heatmap (beneath copper) — one blocky Hanan cell
    // per occupied grid node, warmer where more nets pass through. Shows how full
    // the board is around a failed net; chokepoints glow.
    s.push_str(&congestion_overlay(diag, res));

    // Obstacles (pads / keepouts) — gray, dimmer on inner layers.
    for o in &srj.obstacles {
        s.push_str(&obstacle_rect(o, stroke));
    }

    // Routed traces — one color per net.
    for (i, t) in traces.iter().enumerate() {
        s.push_str(&trace_path(t, &net_color(i), stroke));
    }

    // Ratsnest of UNROUTED nets — straight red lines between the endpoints the
    // router failed to connect, so failures are explicit, not just implied by a
    // missing trace. Dashed = `Congested` (routable alone), solid = impossible.
    s.push_str(&ratsnest_overlay(srj, res, stroke));

    // Connection endpoints — small white dots so unrouted nets are visible.
    for c in &srj.connections {
        for p in &c.points_to_connect {
            s.push_str(&format!(
                r##"<circle cx="{:.3}" cy="{:.3}" r="{:.3}" fill="#fff" fill-opacity="0.7"/>"##,
                p.x,
                p.y,
                stroke * 2.0
            ));
        }
    }

    // DRC violations — conspicuous red markers ON TOP of all copper, one per
    // violation at its reported location. A hollow red ring plus an X so a
    // clearance short / via violation is impossible to miss against the copper.
    s.push_str(&drc_overlay(violations, stroke));

    s.push_str("</g>");

    // Caption (outside the flip so text is upright).
    let pct = res.completion() * 100.0;
    let color = if res.nets_routed == res.nets_total && res.error.is_none() {
        "#3fb950"
    } else {
        "#f85149"
    };
    // Append the DRC violation count so the schematic SVG is honest about the
    // clearance/via shorts the router produced (previously invisible).
    let drc_suffix = if res.drc_violations > 0 {
        format!(" — {} DRC", res.drc_violations)
    } else {
        String::new()
    };
    let label = res
        .error
        .as_deref()
        .map(|e| format!("{} — {}{}", res.board, e, drc_suffix))
        .unwrap_or_else(|| {
            format!(
                "{} — {}/{} nets ({:.0}%) {:.0}ms{}",
                res.board, res.nets_routed, res.nets_total, pct, res.wall_ms, drc_suffix
            )
        });
    s.push_str(&format!(
        r##"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{color}" fill-opacity="0.9"/>"##,
        b.min_x - m,
        b.min_y - m,
        w + 2.0 * m,
        (h + 2.0 * m) * 0.07,
    ));
    s.push_str(&format!(
        r##"<text x="{:.3}" y="{:.3}" font-size="{:.3}" fill="#fff">{}</text>"##,
        b.min_x - m + stroke * 4.0,
        b.min_y - m + (h + 2.0 * m) * 0.05,
        (h + 2.0 * m) * 0.045,
        esc(&label),
    ));
    s.push_str("</svg>");
    s
}

fn obstacle_rect(o: &Obstacle, _stroke: f64) -> String {
    let x = o.center.x - o.width / 2.0;
    let y = o.center.y - o.height / 2.0;
    // Pads on the top copper a touch brighter than inner/other layers.
    let fill = if o.layers.iter().any(|l| l == "top") {
        "#3a4663"
    } else {
        "#2a3350"
    };
    format!(
        r##"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{fill}" fill-opacity="0.85"/>"##,
        x, y, o.width, o.height
    )
}

/// Render a `pcb_trace` as connected wire segments + via markers.
fn trace_path(t: &PcbTrace, color: &str, stroke: f64) -> String {
    let mut out = String::new();
    let mut pts: Vec<(f64, f64, f64)> = Vec::new(); // x, y, width
    let flush = |pts: &mut Vec<(f64, f64, f64)>, out: &mut String| {
        if pts.len() >= 2 {
            let width = pts.iter().map(|p| p.2).fold(0.0_f64, f64::max).max(stroke);
            let d: Vec<String> = pts
                .iter()
                .enumerate()
                .map(|(i, p)| format!("{}{:.3} {:.3}", if i == 0 { "M" } else { "L" }, p.0, p.1))
                .collect();
            out.push_str(&format!(
                r##"<path d="{}" fill="none" stroke="{color}" stroke-width="{width:.3}" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.95"/>"##,
                d.join(" ")
            ));
        }
        pts.clear();
    };

    for rp in &t.route {
        match rp {
            RoutePoint::Wire { x, y, width, .. } => pts.push((*x, *y, *width)),
            RoutePoint::Via { x, y, .. } => {
                // A via continues the polyline AND draws a marker at its REAL
                // geometry: an annular copper pad of diameter VIA_PAD_MM with a
                // drill dot of diameter VIA_DRILL_MM (board units), not an
                // arbitrary stroke-scaled blob.
                pts.push((*x, *y, stroke));
                out.push_str(&format!(
                    r##"<circle cx="{:.3}" cy="{:.3}" r="{:.4}" fill="#ffd33d" stroke="#000" stroke-width="{:.4}"/>"##,
                    x,
                    y,
                    crate::VIA_PAD_MM / 2.0,
                    stroke * 0.4
                ));
                out.push_str(&format!(
                    r##"<circle cx="{:.3}" cy="{:.3}" r="{:.4}" fill="#0b1020"/>"##,
                    x,
                    y,
                    crate::VIA_DRILL_MM / 2.0
                ));
            }
        }
    }
    flush(&mut pts, &mut out);
    out
}

/// Render the per-cell occupancy field as translucent Hanan cells. Each occupied
/// grid node `(x, y)` — summed across layers — becomes a rect spanning its Voronoi
/// half-gaps to neighbouring grid lines, coloured warmer (amber → red) with
/// occupancy. Returns empty if the diagnostics geometry doesn't match the grid.
fn congestion_overlay(diag: &RouteDiagnostics, res: &BoardResult) -> String {
    let (xs, ys) = (&diag.x_lines, &diag.y_lines);
    let (w, h, layers) =
        (res.grid_w as usize, res.grid_h as usize, res.grid_layers.max(1) as usize);
    if w == 0 || h == 0 || xs.len() != w || ys.len() != h || diag.congestion.len() != w * h * layers
    {
        return String::new();
    }
    // Fold occupancy across layers onto the planar (x, y) grid: index `i` lives at
    // planar node `i % (w*h)` (== y*w + x) regardless of layer.
    let plane = w * h;
    let mut planar = vec![0u32; plane];
    for (i, &c) in diag.congestion.iter().enumerate() {
        planar[i % plane] += c;
    }
    let peak = planar.iter().copied().max().unwrap_or(0).max(1) as f64;
    // A cell's continuous extent: midpoints to neighbouring lines, clamped to the
    // first/last line at the board edge.
    let edge = |lines: &[f64], k: usize| -> (f64, f64) {
        let lo = if k == 0 { lines[0] } else { 0.5 * (lines[k - 1] + lines[k]) };
        let hi = if k + 1 == lines.len() { lines[k] } else { 0.5 * (lines[k] + lines[k + 1]) };
        (lo, hi)
    };
    let mut out = String::new();
    for y in 0..h {
        let (y0, y1) = edge(ys, y);
        for x in 0..w {
            let c = planar[y * w + x];
            if c == 0 {
                continue;
            }
            let (x0, x1) = edge(xs, x);
            let t = c as f64 / peak;
            let op = 0.10 + 0.45 * t;
            let hue = 40.0 - 40.0 * t; // amber (low) -> red (high)
            out.push_str(&format!(
                r##"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="hsl({:.0} 90% 55%)" fill-opacity="{:.3}"/>"##,
                x0,
                y0,
                (x1 - x0).max(0.0),
                (y1 - y0).max(0.0),
                hue,
                op
            ));
        }
    }
    out
}

/// Draw straight "ratsnest" lines for every UNROUTED net segment, mapping each
/// failed sub-net name back to its connection segment (the same `<conn>#<seg>`
/// decomposition the rasteriser uses). Dashed coral = `Congested` (routable in
/// isolation, lost to contention); solid bright red = `UnroutableAlone`, so the
/// two failure classes read apart at a glance.
fn ratsnest_overlay(srj: &SimpleRouteJson, res: &BoardResult, stroke: f64) -> String {
    if res.unrouted.is_empty() {
        return String::new();
    }
    let reason: std::collections::HashMap<&str, UnroutedReason> =
        res.unrouted.iter().map(|u| (u.name.as_str(), u.reason)).collect();
    let mut out = String::new();
    for conn in &srj.connections {
        let pts = &conn.points_to_connect;
        if pts.len() < 2 {
            continue;
        }
        let segments = pts.len() - 1;
        for (seg, win) in pts.windows(2).enumerate() {
            let name =
                if segments == 1 { conn.name.clone() } else { format!("{}#{}", conn.name, seg) };
            let Some(r) = reason.get(name.as_str()) else {
                continue;
            };
            let (color, dash) = match r {
                UnroutedReason::Congested => (
                    "#ff7b72",
                    format!(r##" stroke-dasharray="{:.3} {:.3}""##, stroke * 3.0, stroke * 2.0),
                ),
                UnroutedReason::UnroutableAlone => ("#ff2d2d", String::new()),
            };
            out.push_str(&format!(
                r##"<line x1="{:.3}" y1="{:.3}" x2="{:.3}" y2="{:.3}" stroke="{}" stroke-width="{:.3}" stroke-opacity="0.95" stroke-linecap="round"{}/>"##,
                win[0].x,
                win[0].y,
                win[1].x,
                win[1].y,
                color,
                stroke * 1.6,
                dash
            ));
        }
    }
    out
}

/// Overlay every geometric DRC violation as a conspicuous red marker at its
/// reported `location` (board units), drawn ON TOP of all copper. Each is a
/// hollow red ring with an X through it so a clearance short or via violation is
/// impossible to miss. Drawn inside the y-flip group, so it shares the copper
/// coordinate frame.
fn drc_overlay(violations: &[Violation], stroke: f64) -> String {
    if violations.is_empty() {
        return String::new();
    }
    let r = stroke * 4.0;
    let mut out = String::new();
    for v in violations {
        let (cx, cy) = v.location;
        out.push_str(&format!(
            r##"<circle cx="{cx:.3}" cy="{cy:.3}" r="{r:.3}" fill="none" stroke="#ff2d2d" stroke-width="{:.3}" stroke-opacity="0.95"/>"##,
            stroke * 1.2
        ));
        // An X through the ring.
        out.push_str(&format!(
            r##"<path d="M{:.3} {:.3} L{:.3} {:.3} M{:.3} {:.3} L{:.3} {:.3}" stroke="#ff2d2d" stroke-width="{:.3}" stroke-opacity="0.95" stroke-linecap="round"/>"##,
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            cx - r,
            cy + r,
            cx + r,
            cy - r,
            stroke * 1.2
        ));
    }
    out
}

/// Render the `index.html` gallery: overall + per-corpus stats, then every board
/// (failures first within each group) as an embedded SVG with a pass/fail badge.
fn render_gallery(report: &CorpusReport, gallery: &[(BoardResult, String)]) -> String {
    let mut h = String::new();
    h.push_str(
        r##"<!doctype html><meta charset="utf-8"><title>metalroute real-board corpus</title>
<style>
 body{background:#0b1020;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;padding:24px}
 h1{font-size:20px} h2{margin-top:32px;border-bottom:1px solid #30363d;padding-bottom:6px}
 .stat{display:inline-block;margin-right:24px;font-size:14px;color:#8b949e}
 .stat b{color:#e6edf3;font-size:18px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:14px}
 .card{background:#11162a;border:1px solid #30363d;border-radius:8px;overflow:hidden}
 .card img{width:100%;display:block;background:#0b1020}
 .badge{padding:6px 10px;font-size:12px;display:flex;justify-content:space-between}
 .pass{border-top:3px solid #3fb950} .fail{border-top:3px solid #f85149}
 .pct.ok{color:#3fb950} .pct.bad{color:#f85149}
</style>
"##,
    );
    h.push_str(&format!(
        "<h1>metalroute — real-board corpus ({} router)</h1>",
        esc(&report.router)
    ));
    h.push_str(&format!(
        r##"<div><span class="stat"><b>{}</b> boards</span><span class="stat"><b>{:.1}%</b> net completion ({}/{})</span><span class="stat"><b>{}</b> fully routed</span><span class="stat"><b>{:.0}</b> nets/sec</span><span class="stat"><b>{:.1}s</b> total</span></div>"##,
        report.boards,
        report.completion_rate * 100.0,
        report.nets_routed,
        report.nets_total,
        report.fully_routed_boards,
        report.nets_per_sec,
        report.total_wall_ms / 1000.0,
    ));
    // Headline diagnostic: of all unrouted nets, how many were lost to congestion
    // (algorithm could fix) vs. impossible at this resolution (grid limit)?
    h.push_str(&format!(
        r##"<div style="margin-top:8px"><span class="stat" style="color:#ff7b72"><b>{}</b> congested (routable alone)</span><span class="stat" style="color:#ff2d2d"><b>{}</b> unroutable alone (resolution-bound)</span></div>"##,
        report.unrouted_reasons.congested,
        report.unrouted_reasons.unroutable_alone,
    ));

    let by_name: std::collections::HashMap<&str, &(BoardResult, String)> =
        gallery.iter().map(|e| (e.0.board.as_str(), e)).collect();
    let _ = by_name;

    for g in &report.groups {
        h.push_str(&format!(
            r##"<h2>{} — {:.1}% ({}/{} nets, {}/{} boards full) · <span style="color:#ff7b72">{} congested</span> / <span style="color:#ff2d2d">{} unroutable-alone</span></h2>"##,
            esc(&g.name),
            g.completion_rate * 100.0,
            g.nets_routed,
            g.nets_total,
            g.fully_routed_boards,
            g.boards,
            g.unrouted_reasons.congested,
            g.unrouted_reasons.unroutable_alone,
        ));
        // Failures first, then by board name.
        let mut items: Vec<&(BoardResult, String)> =
            gallery.iter().filter(|(b, _)| b.corpus == g.name).collect();
        items.sort_by(|a, b| {
            a.0.completion()
                .partial_cmp(&b.0.completion())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.board.cmp(&b.0.board))
        });
        h.push_str(r##"<div class="grid">"##);
        for (res, file) in items {
            let full = res.error.is_none() && res.nets_total > 0 && res.nets_routed == res.nets_total;
            let cls = if full { "pass" } else { "fail" };
            let pcls = if full { "ok" } else { "bad" };
            let right = res
                .error
                .as_deref()
                .map(esc)
                .unwrap_or_else(|| format!("{:.0}ms", res.wall_ms));
            // Per-board failure split, shown only when the board has unrouted nets.
            let cong =
                res.unrouted.iter().filter(|u| u.reason == UnroutedReason::Congested).count();
            let alone = res.unrouted.len() - cong;
            let reason_line = if res.unrouted.is_empty() {
                String::new()
            } else {
                format!(
                    r##"<div class="badge" style="border-top:1px solid #30363d"><span style="color:#ff7b72">{cong} congested</span><span style="color:#ff2d2d">{alone} unroutable-alone</span></div>"##
                )
            };
            h.push_str(&format!(
                r##"<div class="card {cls}"><img loading="lazy" src="{}"><div class="badge"><span>{}</span><span class="pct {pcls}">{}/{} ({:.0}%)</span><span>{}</span></div>{}</div>"##,
                esc(file),
                esc(&res.board),
                res.nets_routed,
                res.nets_total,
                res.completion() * 100.0,
                right,
                reason_line,
            ));
        }
        h.push_str("</div>");
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_srj::PcbTrace;

    fn tiny_srj() -> SimpleRouteJson {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "obstacles": [{"type":"rect","center":{"x":5.0,"y":5.0},"width":2.0,"height":2.0,"layers":["top"]}],
            "connections": [{"name":"n0","pointsToConnect":[{"x":1.0,"y":1.0},{"x":9.0,"y":9.0}]}],
        });
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn renders_valid_svg_with_obstacles_and_traces() {
        let srj = tiny_srj();
        let traces = vec![PcbTrace::new(vec![
            RoutePoint::Wire { x: 1.0, y: 1.0, width: 0.15, layer: "top".into() },
            RoutePoint::Wire { x: 9.0, y: 9.0, width: 0.15, layer: "top".into() },
        ])];
        let res = BoardResult {
            corpus: "t".into(),
            board: "tiny".into(),
            nets_total: 1,
            nets_routed: 1,
            total_cost: 16,
            grid_w: 10,
            grid_h: 10,
            grid_layers: 1,
            wall_ms: 1.0,
            error: None,
            unrouted: Vec::new(),
            congestion_peak: 0,
            drc_violations: 0,
            drc_by_class: BTreeMap::new(),
        };
        let svg = render_svg(&srj, &traces, &res, &RouteDiagnostics::default(), &[]);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<rect")); // obstacle + board bg
        assert!(svg.contains("<path")); // the routed trace
    }

    #[test]
    fn render_svg_draws_ratsnest_and_heatmap_for_failures() {
        let srj = tiny_srj(); // single connection "n0" between (1,1) and (9,9)
        let traces: Vec<PcbTrace> = Vec::new();
        let res = BoardResult {
            corpus: "t".into(),
            board: "tiny".into(),
            nets_total: 1,
            nets_routed: 0,
            total_cost: 0,
            grid_w: 2,
            grid_h: 2,
            grid_layers: 1,
            wall_ms: 1.0,
            error: None,
            // "n0" is a single-segment connection, so its sub-net keeps the bare name.
            unrouted: vec![UnroutedNet {
                name: "n0".into(),
                reason: UnroutedReason::Congested,
            }],
            congestion_peak: 2,
            drc_violations: 0,
            drc_by_class: BTreeMap::new(),
        };
        // 2x2 grid lines spanning the board; one busy cell to shade.
        let diag = RouteDiagnostics {
            x_lines: vec![0.0, 10.0],
            y_lines: vec![0.0, 10.0],
            congestion: vec![2, 0, 0, 0],
        };
        let svg = render_svg(&srj, &traces, &res, &diag, &[]);
        // Ratsnest line for the unrouted net, dashed because it's Congested.
        assert!(svg.contains("<line"), "expected a ratsnest line");
        assert!(svg.contains("stroke-dasharray"), "Congested ratsnest must be dashed");
        // Heatmap cell for the occupied node (hsl warm fill from congestion_overlay).
        assert!(svg.contains("hsl("), "expected a congestion heatmap cell");
    }

    #[test]
    fn predicted_cells_scales_with_layers() {
        let srj = tiny_srj();
        let one = predicted_cells(&srj, Some(1.0), Some(1));
        let four = predicted_cells(&srj, Some(1.0), Some(4));
        assert_eq!(four, one * 4);
    }
}
