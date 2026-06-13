//! The `drc` subcommand and the bridge from a routed board to [`mr_drc`].
//!
//! [`build_drc_board`] turns the router's output (copper on the *signal* grid) into
//! a physical [`mr_drc::DrcBoard`] over the *full* stackup, so a through-via's
//! barrel is correctly seen crossing the inner power planes it physically drills.
//! [`run_drc`] routes a `.dsn`, runs the checker, and reports/writes the result.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use mr_core::{BoardRoute, LayerMap};
use mr_drc::{DrcBoard, DrcRules, DrcSummary, LayerKind, Pad, Segment, Via, Violation};
use mr_ingest::dsn::{dsn_to_ingest, PlaneDef};
use mr_srj::{Mapping, Obstacle};

use crate::{VIA_DRILL_MM, VIA_PAD_MM};

/// Default plane antipad (relief radius) a via must carve in a foreign plane (mm).
pub const DEFAULT_PLANE_ANTIPAD_MM: f64 = 0.25;
/// Default minimum via annular ring (mm). The 0.45/0.2 via gives a 0.125 mm ring,
/// comfortably above this, so a well-formed via never trips the annular rule.
pub const DEFAULT_MIN_ANNULAR_RING_MM: f64 = 0.05;

/// Standard DRC rules at the given copper-to-copper `clearance` (mm), with the
/// default plane antipad and annular-ring minimums.
pub fn default_rules(clearance: f64) -> DrcRules {
    DrcRules {
        clearance,
        plane_antipad: DEFAULT_PLANE_ANTIPAD_MM,
        min_annular_ring: DEFAULT_MIN_ANNULAR_RING_MM,
    }
}

/// Build a physical [`DrcBoard`] from a routed board.
///
/// * `board` / `mapping` / `signal_layers` describe the routed copper in the
///   *signal* grid the router worked in.
/// * `physical_layers` is the *full* DSN stackup (signal + plane layers); routed
///   signal-layer indices are resolved to physical indices by name so a via's span
///   covers every physical layer (including planes) it really drills through.
/// * `planes` binds each poured plane to the copper layer it fills; `obstacles` +
///   `pin_nets` are the static pads and their nets.
#[allow(clippy::too_many_arguments)]
pub fn build_drc_board(
    board: &BoardRoute,
    mapping: &Mapping,
    signal_layers: &LayerMap,
    physical_layers: &LayerMap,
    planes: &[PlaneDef],
    obstacles: &[Obstacle],
    pin_nets: &HashMap<String, String>,
    trace_width: f64,
    rules: DrcRules,
) -> DrcBoard {
    // physical layer index -> plane net (only for layers a plane fills).
    let mut plane_nets: HashMap<u32, String> = HashMap::new();
    for p in planes {
        if let Some(idx) = physical_layers.index_of(&p.layer) {
            plane_nets.insert(idx, p.net.clone());
        }
    }
    let layers: Vec<LayerKind> = (0..physical_layers.len())
        .map(|i| match plane_nets.get(&i) {
            Some(net) => LayerKind::Plane { net: net.clone() },
            None => LayerKind::Signal,
        })
        .collect();

    // Resolve a signal-grid layer index to its physical-stackup index by name.
    let sig_to_phys = |sl: u32| {
        physical_layers
            .index_of(signal_layers.name(sl))
            .unwrap_or(sl)
    };

    let dims = mapping.dims;
    let mut segments = Vec::new();
    let mut vias = Vec::new();

    for r in &board.results {
        // Chained sub-nets of one connection share a base net (strip the `#seg`).
        let net = r.net.split('#').next().unwrap_or(&r.net).to_string();
        let path = &r.path;

        // Planar copper: one segment per consecutive same-layer move.
        for w in path.windows(2) {
            let (xa, ya, la) = dims.xyz(w[0]);
            let (xb, yb, lb) = dims.xyz(w[1]);
            if la == lb && (xa, ya) != (xb, yb) {
                segments.push(Segment {
                    net: net.clone(),
                    layer: sig_to_phys(la),
                    a: mapping.cell_center(w[0]),
                    b: mapping.cell_center(w[1]),
                    width: trace_width,
                });
            }
        }

        // Vias: collapse each maximal vertical run (same x,y, changing layer) into
        // one through-via spanning the physical layers of its first and last cells.
        let mut i = 0;
        while i < path.len() {
            let (cx, cy, _) = dims.xyz(path[i]);
            let mut j = i;
            while j + 1 < path.len() {
                let (nx, ny, _) = dims.xyz(path[j + 1]);
                if nx == cx && ny == cy && path[j + 1] != path[j] {
                    j += 1;
                } else {
                    break;
                }
            }
            if j > i {
                let l0 = dims.xyz(path[i]).2;
                let l1 = dims.xyz(path[j]).2;
                let (p0, p1) = (sig_to_phys(l0), sig_to_phys(l1));
                vias.push(Via {
                    net: net.clone(),
                    center: mapping.cell_center(path[i]),
                    pad_diameter: VIA_PAD_MM,
                    drill_diameter: VIA_DRILL_MM,
                    from_layer: p0.min(p1),
                    to_layer: p0.max(p1),
                    // M1 baseline: vias carry no antipad, so they short the planes
                    // they cross — exactly the state we are measuring. M2 sets this.
                    antipad_radius: None,
                });
                i = j + 1;
            } else {
                i += 1;
            }
        }
    }

    // Static pads: one rect per obstacle, per physical layer it occupies, tagged
    // with its net (via the pad's `REF-PIN` id) or `None` when unknown.
    let mut pads = Vec::new();
    for obs in obstacles {
        let net = obs
            .connected_to
            .first()
            .and_then(|id| pin_nets.get(id))
            .cloned();
        for layer_name in &obs.layers {
            if let Some(idx) = physical_layers.index_of(layer_name) {
                pads.push(Pad {
                    net: net.clone(),
                    layer: idx,
                    center: (obs.center.x, obs.center.y),
                    width: obs.width,
                    height: obs.height,
                });
            }
        }
    }

    DrcBoard {
        layers,
        segments,
        pads,
        vias,
        rules,
    }
}

/// Arguments for the `drc` subcommand: the same routing knobs as `route-dsn`, plus
/// an optional JSON report path.
#[derive(Debug, Parser)]
pub struct DrcArgs {
    /// Path to the input Specctra `.dsn` file.
    #[arg(long)]
    pub input: PathBuf,

    /// Cell size in mm. Defaults to a value derived from the board bounds.
    #[arg(long)]
    pub resolution: Option<f64>,

    /// Skip nets whose name contains this substring (repeatable).
    #[arg(long = "skip-nets")]
    pub skip_nets: Vec<String>,

    /// Cap the number of original nets routed.
    #[arg(long)]
    pub max_nets: Option<usize>,

    /// Number of signal layers to route on (defaults to all signal layers).
    #[arg(long)]
    pub layers: Option<u32>,

    /// Write the DRC report (summary + a sample of violations) as JSON to this path.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Cap how many individual violations the JSON report lists (the summary always
    /// reflects the true total). Keeps a committed baseline small and stable.
    #[arg(long)]
    pub max_violations: Option<usize>,
}

/// A serialisable DRC run report: routing context plus the violation summary and
/// list. Written by `drc --out` and used to record baselines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrcRunReport {
    pub design: String,
    pub total_nets: usize,
    pub routed_nets: usize,
    pub fully_connected: usize,
    pub vias: usize,
    pub clearance_mm: f64,
    pub summary: DrcSummary,
    pub violations: Vec<Violation>,
}

/// Execute the `drc` subcommand: route the `.dsn`, build the physical board, run the
/// checker, optionally write the JSON report, and return it.
pub fn run_drc(args: &DrcArgs) -> Result<DrcRunReport> {
    let text = std::fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read DSN file {}", args.input.display()))?;
    let ingest = dsn_to_ingest(&text).context("failed to convert DSN to problem")?;
    let clearance_mm = ingest.stats.min_clearance_mm;

    let design = args
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fixture".to_string());

    let (report, _traces, _ses, drc_board) = crate::route_dsn_problem(
        ingest,
        &design,
        args.resolution,
        &args.skip_nets,
        args.max_nets,
        args.layers,
    )?;

    let violations = drc_board.check();
    let summary = DrcSummary::of(&violations);

    // The summary keeps the true total; the listed violations may be capped so a
    // committed baseline stays small.
    let violations = match args.max_violations {
        Some(n) if n < violations.len() => violations[..n].to_vec(),
        _ => violations,
    };

    let run = DrcRunReport {
        design,
        total_nets: report.total_nets,
        routed_nets: report.routed_nets,
        fully_connected: report.fully_connected,
        vias: report.vias,
        clearance_mm,
        summary,
        violations,
    };

    if let Some(path) = &args.out {
        let json = serde_json::to_string_pretty(&run).context("failed to serialise DRC report")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write DRC report {}", path.display()))?;
    }

    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::{Dims, RouteResult};

    /// A through-via that drills from the top signal layer to the bottom signal
    /// layer must be flagged crossing the inner GND plane between them.
    #[test]
    fn via_through_inner_plane_is_flagged() {
        // Physical stack: top(signal) / inner1(GND plane) / bottom(signal).
        let physical = LayerMap::from_names(vec![
            "top".to_string(),
            "inner1".to_string(),
            "bottom".to_string(),
        ]);
        // The router only sees the two signal layers.
        let signal = LayerMap::from_names(vec!["top".to_string(), "bottom".to_string()]);
        let planes = vec![PlaneDef {
            net: "GND".to_string(),
            layer: "inner1".to_string(),
        }];

        // 2-layer signal grid; a one-net via run from signal layer 0 to layer 1.
        let dims = Dims::with_layers(4, 4, 2);
        let bounds = mr_srj::Bounds {
            min_x: 0.0,
            max_x: 4.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::with_layers(&bounds, 1.0, 2);
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "SIG".to_string(),
                path: vec![dims.idx3(1, 1, 0), dims.idx3(1, 1, 1)],
                cost: 1,
            }],
            unrouted: vec![],
            congestion: vec![],
        };

        let drc = build_drc_board(
            &board,
            &mapping,
            &signal,
            &physical,
            &planes,
            &[],
            &HashMap::new(),
            0.15,
            default_rules(0.15),
        );

        // One through-via spanning physical 0..2.
        assert_eq!(drc.vias.len(), 1);
        assert_eq!((drc.vias[0].from_layer, drc.vias[0].to_layer), (0, 2));

        let violations = drc.check();
        let summary = DrcSummary::of(&violations);
        assert_eq!(
            summary.via_through_plane, 1,
            "via must short the foreign GND plane it drills through"
        );
        // Same-net via would NOT short its own plane:
        let mut same_net = board.clone();
        same_net.results[0].net = "GND".to_string();
        let drc2 = build_drc_board(
            &same_net,
            &mapping,
            &signal,
            &physical,
            &planes,
            &[],
            &HashMap::new(),
            0.15,
            default_rules(0.15),
        );
        assert_eq!(
            DrcSummary::of(&drc2.check()).via_through_plane,
            0,
            "a via on the plane's own net is a legal connection"
        );
    }
}
