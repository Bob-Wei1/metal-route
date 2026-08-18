//! The `drc` subcommand and the bridge from a routed board to [`mr_drc`].
//!
//! [`build_drc_board`] turns the router's output (copper on the *signal* grid) into
//! a physical [`mr_drc::DrcBoard`] over the *full* stackup, so a through-via's
//! barrel is correctly seen crossing the inner power planes it physically drills.
//! [`run_drc`] routes a `.dsn`, runs the checker, and reports/writes the result.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use mr_core::{BoardRoute, LayerMap};
use mr_drc::{DrcBoard, DrcRules, DrcSummary, LayerKind, Pad, Segment, Via, Violation};
use mr_ingest::dsn::{dsn_to_ingest, PlaneDef};
use mr_srj::{Mapping, Obstacle};

#[cfg(test)]
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
///   `pin_nets` are the static pads and their nets. `via_pad_diameter` and
///   `via_drill_diameter` are the geometry selected from the DSN's declared via
///   padstack (or the caller's documented fallback when none was declared).
///
/// `model_plane_antipads` selects how a through-via crossing a *foreign* plane is
/// modelled:
///
/// * `true` (the realistic default for **poured-zone** boards): the DSN planes are
///   `(plane "NET" (polygon ...))` poured zones, and a poured zone automatically
///   reliefs (carves an antipad around) a foreign through-via by the zone's
///   clearance. We model that by giving each via an `antipad_radius` of exactly
///   `via_drill_diameter/2 + rules.plane_antipad` — the relief the zone fill provides —
///   so a well-formed via crossing a foreign plane is *not* reported as a short.
///   IMPORTANT / honesty: this antipad is a *model* of the zone-fill relief, not
///   geometry we emit. It physically exists only if the fabrication / zone-fill
///   actually reliefs the via. The `kicad-cli pcb drc` cross-check (a later
///   milestone) is what validates that the imported board really reliefs these
///   vias. This is a deliberate model correction, not a silent silencing of a
///   real short.
/// * `false` (the pessimistic "planes are bare copper" model): every via carries
///   `antipad_radius: None`, so every via crossing a foreign plane shorts to it.
///   Use `--no-plane-zones` when the planes are *not* poured (solid copper pours
///   with no relief) and you want that worst case.
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
    via_pad_diameter: f64,
    via_drill_diameter: f64,
    rules: DrcRules,
    model_plane_antipads: bool,
) -> Result<DrcBoard> {
    anyhow::ensure!(
        mapping.dims.layers == signal_layers.len(),
        "routing grid has {} layers but signal layer map has {}",
        mapping.dims.layers,
        signal_layers.len()
    );

    // Layer names are the identity used to project the routed signal grid onto
    // the physical stack. Duplicate names make that projection ambiguous:
    // `LayerMap::index_of` would pick the first occurrence and silently collapse
    // two physical layers (and a via between them) onto one. Reject either map
    // before resolving indices so malformed DSNs fail closed.
    for (role, layer_map) in [("signal", signal_layers), ("physical", physical_layers)] {
        let mut seen = HashSet::with_capacity(layer_map.len() as usize);
        for layer in 0..layer_map.len() {
            let name = layer_map.name(layer);
            anyhow::ensure!(
                seen.insert(name),
                "{role} layer map contains duplicate name {name:?}"
            );
        }
    }

    // Resolve the complete signal-to-physical mapping up front and fail closed.
    // Falling back to the signal index can silently alias a missing name onto an
    // unrelated plane or signal layer in the physical stack.
    let signal_to_physical: Vec<u32> = (0..signal_layers.len())
        .map(|sl| {
            let name = signal_layers.name(sl);
            physical_layers
                .index_of(name)
                .with_context(|| format!("signal layer {name:?} is absent from physical stack"))
        })
        .collect::<Result<_>>()?;

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
    let sig_to_phys = |sl: u32| signal_to_physical[sl as usize];

    let dims = mapping.dims;
    let mut segments = Vec::new();
    let mut vias = Vec::new();

    // For a poured-zone board, the zone fill reliefs a foreign through-via by the
    // zone clearance. We model that relief as an antipad that exactly meets the
    // rule (`drill/2 + plane_antipad`), so a well-formed via does NOT short a
    // foreign plane. `None` keeps the pessimistic bare-copper model where it does.
    let via_antipad: Option<f64> = if model_plane_antipads {
        Some(via_drill_diameter / 2.0 + rules.plane_antipad)
    } else {
        None
    };

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
        // one emitted through-via. `route-dsn` accepts only a full-stack via (or
        // generates its legacy full-stack fallback), so DRC must stamp the copper
        // and barrel across the complete physical stack even when the router used
        // only one adjacent hop to change signal layers.
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
                vias.push(Via {
                    net: net.clone(),
                    center: mapping.cell_center(path[i]),
                    pad_diameter: via_pad_diameter,
                    drill_diameter: via_drill_diameter,
                    from_layer: 0,
                    to_layer: physical_layers.len().saturating_sub(1),
                    // Poured-zone relief (or `None` for the bare-copper model); see
                    // `model_plane_antipads` on `build_drc_board`.
                    antipad_radius: via_antipad,
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

    Ok(DrcBoard {
        layers,
        segments,
        pads,
        vias,
        rules,
    })
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

    /// Treat planes as bare copper: do NOT model the antipad (relief) a poured zone
    /// gives a foreign through-via. With this set, every via crossing a foreign
    /// plane is reported as a short (the pessimistic worst case). Leave it off for
    /// poured-zone boards, where the zone fill reliefs the via.
    #[arg(long, default_value_t = false)]
    pub no_plane_zones: bool,
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
        !args.no_plane_zones,
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

    /// Shared fixture: a top→bottom through-via on net `via_net` that drills the
    /// inner GND plane. `model_plane_antipads` toggles the poured-zone relief model.
    fn inner_plane_via_board(via_net: &str, model_plane_antipads: bool) -> DrcBoard {
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
                net: via_net.to_string(),
                path: vec![dims.idx3(1, 1, 0), dims.idx3(1, 1, 1)],
                cost: 1,
            }],
            unrouted: vec![],
            congestion: vec![],
            groups: vec![],
        };

        build_drc_board(
            &board,
            &mapping,
            &signal,
            &physical,
            &planes,
            &[],
            &HashMap::new(),
            0.15,
            VIA_PAD_MM,
            VIA_DRILL_MM,
            default_rules(0.15),
            model_plane_antipads,
        )
        .expect("fixture layer maps must be compatible")
    }

    #[test]
    fn missing_signal_layer_in_physical_stack_fails_closed() {
        let bounds = mr_srj::Bounds {
            min_x: 0.0,
            max_x: 2.0,
            min_y: 0.0,
            max_y: 2.0,
        };
        let mapping = Mapping::with_layers(&bounds, 1.0, 2);
        let signal = LayerMap::from_names(vec!["top".to_string(), "bottom".to_string()]);
        let physical = LayerMap::from_names(vec!["top".to_string(), "inner1".to_string()]);
        let board = BoardRoute {
            results: vec![],
            unrouted: vec![],
            congestion: vec![],
            groups: vec![],
        };

        let err = build_drc_board(
            &board,
            &mapping,
            &signal,
            &physical,
            &[],
            &[],
            &HashMap::new(),
            0.15,
            VIA_PAD_MM,
            VIA_DRILL_MM,
            default_rules(0.15),
            true,
        )
        .expect_err("a missing physical signal layer must not be silently aliased");
        assert!(
            err.to_string().contains("bottom") && err.to_string().contains("absent"),
            "unexpected error: {err:#}"
        );
    }

    fn empty_board() -> BoardRoute {
        BoardRoute {
            results: vec![],
            unrouted: vec![],
            congestion: vec![],
            groups: vec![],
        }
    }

    fn empty_mapping(layers: u32) -> Mapping {
        Mapping::with_layers(
            &mr_srj::Bounds {
                min_x: 0.0,
                max_x: 2.0,
                min_y: 0.0,
                max_y: 2.0,
            },
            1.0,
            layers,
        )
    }

    #[test]
    fn duplicate_signal_layer_name_fails_closed() {
        let signal = LayerMap::from_names(vec!["top".to_string(), "top".to_string()]);
        let physical = LayerMap::from_names(vec!["top".to_string(), "bottom".to_string()]);

        let err = build_drc_board(
            &empty_board(),
            &empty_mapping(2),
            &signal,
            &physical,
            &[],
            &[],
            &HashMap::new(),
            0.15,
            VIA_PAD_MM,
            VIA_DRILL_MM,
            default_rules(0.15),
            true,
        )
        .expect_err("duplicate signal identities must not collapse onto one layer");
        assert!(
            err.to_string().contains("signal")
                && err.to_string().contains("duplicate")
                && err.to_string().contains("top"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn duplicate_physical_layer_name_fails_closed() {
        let signal = LayerMap::from_names(vec!["top".to_string()]);
        let physical = LayerMap::from_names(vec!["top".to_string(), "top".to_string()]);

        let err = build_drc_board(
            &empty_board(),
            &empty_mapping(1),
            &signal,
            &physical,
            &[],
            &[],
            &HashMap::new(),
            0.15,
            VIA_PAD_MM,
            VIA_DRILL_MM,
            default_rules(0.15),
            true,
        )
        .expect_err("duplicate physical identities must not hide a stack layer");
        assert!(
            err.to_string().contains("physical")
                && err.to_string().contains("duplicate")
                && err.to_string().contains("top"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn distinct_signal_to_physical_mapping_is_accepted() {
        let signal = LayerMap::from_names(vec!["top".to_string(), "bottom".to_string()]);
        let physical = LayerMap::from_names(vec![
            "top".to_string(),
            "inner1".to_string(),
            "bottom".to_string(),
        ]);

        let board = build_drc_board(
            &empty_board(),
            &empty_mapping(2),
            &signal,
            &physical,
            &[],
            &[],
            &HashMap::new(),
            0.15,
            VIA_PAD_MM,
            VIA_DRILL_MM,
            default_rules(0.15),
            true,
        )
        .expect("a one-to-one signal projection must remain valid");
        assert_eq!(board.layers.len(), 3);
    }

    #[test]
    fn adjacent_inner_route_hop_stamps_through_via_on_full_physical_stack() {
        let physical = LayerMap::from_names(vec![
            "top".to_string(),
            "inner1".to_string(),
            "inner2".to_string(),
            "bottom".to_string(),
        ]);
        let signal = physical.clone();
        let mapping = empty_mapping(4);
        let dims = mapping.dims;
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "SIG".to_string(),
                // The router changes only between the two inner grid layers. The
                // emitted hardware is nevertheless the supported through-via.
                path: vec![dims.idx3(1, 1, 1), dims.idx3(1, 1, 2)],
                cost: 1,
            }],
            unrouted: vec![],
            congestion: vec![],
            groups: vec![],
        };

        let drc = build_drc_board(
            &board,
            &mapping,
            &signal,
            &physical,
            &[],
            &[],
            &HashMap::new(),
            0.15,
            VIA_PAD_MM,
            VIA_DRILL_MM,
            default_rules(0.15),
            true,
        )
        .unwrap();
        assert_eq!(drc.vias.len(), 1);
        assert_eq!(
            (drc.vias[0].from_layer, drc.vias[0].to_layer),
            (0, 3),
            "DRC must model the full physical drill, not just the route-hop span"
        );
    }

    /// Without the poured-zone model (bare copper), a foreign through-via shorts the
    /// inner GND plane it drills.
    #[test]
    fn via_through_inner_plane_is_flagged() {
        let drc = inner_plane_via_board("SIG", false);

        // One through-via spanning physical 0..2.
        assert_eq!(drc.vias.len(), 1);
        assert_eq!((drc.vias[0].from_layer, drc.vias[0].to_layer), (0, 2));
        assert_eq!(
            drc.vias[0].antipad_radius, None,
            "bare-copper model carries no antipad"
        );

        assert_eq!(
            DrcSummary::of(&drc.check()).via_through_plane,
            1,
            "via must short the foreign GND plane it drills through"
        );
    }

    /// With the poured-zone model, the same foreign through-via carries the zone's
    /// relief antipad and is NOT a short.
    #[test]
    fn via_through_inner_plane_clean_with_poured_zone() {
        let drc = inner_plane_via_board("SIG", true);

        assert_eq!(drc.vias.len(), 1);
        // Antipad exactly meets the rule: drill/2 + plane_antipad.
        let expected = VIA_DRILL_MM / 2.0 + DEFAULT_PLANE_ANTIPAD_MM;
        let got = drc.vias[0].antipad_radius.expect("poured zone reliefs via");
        assert!((got - expected).abs() < 1e-12, "antipad = {got}");

        assert_eq!(
            DrcSummary::of(&drc.check()).via_through_plane,
            0,
            "a poured zone reliefs the foreign through-via, so it is not a short"
        );
    }

    /// A via on the plane's own net is a legal connection regardless of the model.
    #[test]
    fn same_net_via_on_own_plane_never_flagged() {
        for model in [false, true] {
            let drc = inner_plane_via_board("GND", model);
            assert_eq!(
                DrcSummary::of(&drc.check()).via_through_plane,
                0,
                "a via on the plane's own net is a legal connection (model = {model})"
            );
        }
    }
}
