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

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::{default_resolution, parse_srj, route_problem, RouterKind};
use mr_srj::{Obstacle, PcbTrace, RoutePoint, SimpleRouteJson};

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

/// Route one board, returning its [`BoardResult`] plus the traces (for SVG).
fn route_board(
    corpus: &str,
    board: &str,
    srj: &SimpleRouteJson,
    args: &CorpusArgs,
) -> (BoardResult, Vec<PcbTrace>) {
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
            },
            Vec::new(),
        );
    }

    let t0 = Instant::now();
    let routed = route_problem(srj, args.resolution, args.router, args.layers);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    match routed {
        Ok((traces, summary)) => (
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
            },
            traces,
        ),
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
            },
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

    let mut per_board = Vec::with_capacity(files.len());
    let mut gallery: Vec<(BoardResult, String)> = Vec::new();

    for file in &files {
        let corpus = group_of(root, file);
        let board = file
            .file_name()
            .map(|n| n.to_string_lossy().trim_end_matches(".srj.json").to_string())
            .unwrap_or_default();

        let bytes = match std::fs::read(file) {
            Ok(b) => b,
            Err(e) => {
                per_board.push(BoardResult {
                    corpus,
                    board,
                    nets_total: 0,
                    nets_routed: 0,
                    total_cost: 0,
                    grid_w: 0,
                    grid_h: 0,
                    grid_layers: 0,
                    wall_ms: 0.0,
                    error: Some(format!("read failed: {e}")),
                });
                continue;
            }
        };
        let srj = match parse_srj(&bytes) {
            Ok(s) => s,
            Err(e) => {
                per_board.push(BoardResult {
                    corpus,
                    board,
                    nets_total: 0,
                    nets_routed: 0,
                    total_cost: 0,
                    grid_w: 0,
                    grid_h: 0,
                    grid_layers: 0,
                    wall_ms: 0.0,
                    error: Some(format!("parse failed: {e}")),
                });
                continue;
            }
        };

        let (result, traces) = route_board(&corpus, &board, &srj, args);

        if let Some(svg_dir) = &args.svg_out {
            let svg = render_svg(&srj, &traces, &result);
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

    for b in &per_board {
        let g = groups.entry(b.corpus.clone()).or_insert_with(|| CorpusGroup {
            name: b.corpus.clone(),
            boards: 0,
            nets_total: 0,
            nets_routed: 0,
            completion_rate: 0.0,
            fully_routed_boards: 0,
            total_wall_ms: 0.0,
        });
        g.boards += 1;
        g.nets_total += b.nets_total;
        g.nets_routed += b.nets_routed;
        g.total_wall_ms += b.wall_ms;
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
fn render_svg(srj: &SimpleRouteJson, traces: &[PcbTrace], res: &BoardResult) -> String {
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

    // Obstacles (pads / keepouts) — gray, dimmer on inner layers.
    for o in &srj.obstacles {
        s.push_str(&obstacle_rect(o, stroke));
    }

    // Routed traces — one color per net.
    for (i, t) in traces.iter().enumerate() {
        s.push_str(&trace_path(t, &net_color(i), stroke));
    }

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

    s.push_str("</g>");

    // Caption (outside the flip so text is upright).
    let pct = res.completion() * 100.0;
    let color = if res.nets_routed == res.nets_total && res.error.is_none() {
        "#3fb950"
    } else {
        "#f85149"
    };
    let label = res
        .error
        .as_deref()
        .map(|e| format!("{} — {}", res.board, e))
        .unwrap_or_else(|| {
            format!(
                "{} — {}/{} nets ({:.0}%) {:.0}ms",
                res.board, res.nets_routed, res.nets_total, pct, res.wall_ms
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
                // A via continues the polyline AND draws a marker.
                pts.push((*x, *y, stroke));
                out.push_str(&format!(
                    r##"<circle cx="{:.3}" cy="{:.3}" r="{:.3}" fill="#ffd33d" stroke="#000" stroke-width="{:.3}"/>"##,
                    x,
                    y,
                    stroke * 2.5,
                    stroke * 0.5
                ));
            }
        }
    }
    flush(&mut pts, &mut out);
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

    let by_name: std::collections::HashMap<&str, &(BoardResult, String)> =
        gallery.iter().map(|e| (e.0.board.as_str(), e)).collect();
    let _ = by_name;

    for g in &report.groups {
        h.push_str(&format!(
            r##"<h2>{} — {:.1}% ({}/{} nets, {}/{} boards full)</h2>"##,
            esc(&g.name),
            g.completion_rate * 100.0,
            g.nets_routed,
            g.nets_total,
            g.fully_routed_boards,
            g.boards,
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
            h.push_str(&format!(
                r##"<div class="card {cls}"><img loading="lazy" src="{}"><div class="badge"><span>{}</span><span class="pct {pcls}">{}/{} ({:.0}%)</span><span>{}</span></div></div>"##,
                esc(file),
                esc(&res.board),
                res.nets_routed,
                res.nets_total,
                res.completion() * 100.0,
                right,
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
        };
        let svg = render_svg(&srj, &traces, &res);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<rect")); // obstacle + board bg
        assert!(svg.contains("<path")); // the routed trace
    }

    #[test]
    fn predicted_cells_scales_with_layers() {
        let srj = tiny_srj();
        let one = predicted_cells(&srj, Some(1.0), Some(1));
        let four = predicted_cells(&srj, Some(1.0), Some(4));
        assert_eq!(four, one * 4);
    }
}
