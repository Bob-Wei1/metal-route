//! `mr-ingest` — adapter for the `kicad-cruncher` Python CLI.
//!
//! This crate shells out to `kicad-cruncher pnp <board> --format json`, parses the
//! resulting JSON (schema `wn.kicad_cruncher.pnp.v1`), and produces a small board
//! IR ([`BoardIr`]) the rest of the router can later rasterise into a grid.
//!
//! # What the `pnp` JSON actually provides
//!
//! The captured contract (see `tests/fixtures/pnp_sample.json`, real CLI output)
//! is **component placement**, not per-pad copper:
//!
//! ```json
//! {
//!   "schema": "wn.kicad_cruncher.pnp.v1",
//!   "units": "mm",
//!   "placement_count": 7,
//!   "placements": [
//!     { "designator": "FID1", "layer": "top", "center_x": 40.0,
//!       "center_y": 25.0, "rotation": 0.0, "units": "mm", ... }
//!   ]
//! }
//! ```
//!
//! Each placement is a footprint *origin*, so we map one placement -> one [`Pad`].
//!
//! # Design Open Question #2 — known gaps in this source
//!
//! The `pnp` command does **not** expose everything the router ultimately needs:
//!
//! - **Net association is absent.** PnP is a manufacturing artifact; it carries no
//!   net name per placement. [`Pad::net`] is therefore always `None` here. Nets
//!   must come from a different `kicad-cruncher` surface (or a netlist export).
//! - **No per-pad geometry.** We get the footprint placement point, not individual
//!   pad locations/sizes. The IR pad is the component origin, not a copper pad.
//! - **No board outline.** PnP carries no Edge.Cuts geometry, so
//!   [`BoardIr::outline`] is always empty from this source.
//!
//! These are flagged so a later wave can pick a richer ingest path.
//!
//! # Specctra DSN ingest ([`dsn`])
//!
//! The [`dsn`] module is a self-contained path that parses a Specctra `.dsn`
//! S-expression file (component placement + per-pad geometry + netlist, all in
//! one file) directly into an [`mr_srj::SimpleRouteJson`] routing problem. Unlike
//! the `pnp` source above it carries everything the router needs, so it is the
//! preferred "bed-of-nails test rig" ingest path.

pub mod dsn;

use mr_core as _; // contract crate; BoardIr is intentionally self-contained for now.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Errors surfaced while ingesting a `kicad-cruncher` PnP run.
#[derive(Debug)]
pub enum IngestError {
    /// The `kicad-cruncher` process could not be spawned.
    Spawn(std::io::Error),
    /// The CLI ran but exited non-zero. Carries exit code (if any) and stderr.
    CliFailed { code: Option<i32>, stderr: String },
    /// The CLI emitted JSON we could not deserialize into the expected schema.
    Parse(serde_json::Error),
    /// JSON parsed, but declared an unexpected `schema` marker.
    UnexpectedSchema(String),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::Spawn(e) => write!(f, "failed to spawn kicad-cruncher: {e}"),
            IngestError::CliFailed { code, stderr } => {
                write!(f, "kicad-cruncher exited with {code:?}: {stderr}")
            }
            IngestError::Parse(e) => write!(f, "failed to parse pnp JSON: {e}"),
            IngestError::UnexpectedSchema(s) => write!(f, "unexpected pnp schema marker: {s}"),
        }
    }
}

impl std::error::Error for IngestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IngestError::Spawn(e) => Some(e),
            IngestError::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for IngestError {
    fn from(e: serde_json::Error) -> Self {
        IngestError::Parse(e)
    }
}

/// The `schema` marker the `pnp --format json` output is expected to carry.
const PNP_SCHEMA: &str = "wn.kicad_cruncher.pnp.v1";

// ---------------------------------------------------------------------------
// serde structs mirroring the captured JSON (schema wn.kicad_cruncher.pnp.v1)
// ---------------------------------------------------------------------------

/// Top-level `pnp --format json` document.
///
/// Only the fields the IR needs are bound; richer per-placement fields
/// (`parameters`, `canonical_fields`, `field_sources`, etc.) are ignored.
#[derive(Debug, Clone, Deserialize)]
struct PnpDocument {
    schema: String,
    #[serde(default)]
    units: Option<String>,
    placements: Vec<PnpPlacement>,
}

/// One component placement entry from the `placements` array.
#[derive(Debug, Clone, Deserialize)]
struct PnpPlacement {
    designator: String,
    layer: String,
    center_x: f64,
    center_y: f64,
    #[serde(default)]
    rotation: f64,
}

// ---------------------------------------------------------------------------
// The board IR
// ---------------------------------------------------------------------------

/// A single placeable element. With the `pnp` source this is a component
/// placement origin (see crate docs); `net` is always `None` from this source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    /// Net name, if known. Always `None` from the `pnp` source (Open Question #2).
    pub net: Option<String>,
    /// Reference designator, e.g. `"FID1"`, `"U3"`.
    pub designator: String,
    /// X position in `BoardIr::units` (KiCad default mm).
    pub x: f64,
    /// Y position in `BoardIr::units`.
    pub y: f64,
    /// Placement layer, e.g. `"top"` or `"bottom"`.
    pub layer: String,
    /// Placement rotation in degrees.
    pub rotation: f64,
}

/// The small board IR produced by this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardIr {
    /// One pad per component placement.
    pub pads: Vec<Pad>,
    /// Distinct layers referenced by `pads`, in first-seen order.
    pub layers: Vec<String>,
    /// Board outline polygon. Always empty from the `pnp` source (Open Question #2).
    pub outline: Vec<(f64, f64)>,
    /// Coordinate units reported by the CLI (e.g. `"mm"`), if present.
    pub units: Option<String>,
}

/// Parse a `kicad-cruncher pnp --format json` document into a [`BoardIr`].
///
/// Returns [`IngestError::Parse`] on malformed JSON and
/// [`IngestError::UnexpectedSchema`] if the `schema` marker is not the expected
/// `wn.kicad_cruncher.pnp.v1`.
pub fn parse_pnp_json(s: &str) -> Result<BoardIr, IngestError> {
    let doc: PnpDocument = serde_json::from_str(s)?;
    if doc.schema != PNP_SCHEMA {
        return Err(IngestError::UnexpectedSchema(doc.schema));
    }

    let mut layers: Vec<String> = Vec::new();
    let mut pads: Vec<Pad> = Vec::with_capacity(doc.placements.len());
    for p in doc.placements {
        if !layers.iter().any(|l| l == &p.layer) {
            layers.push(p.layer.clone());
        }
        pads.push(Pad {
            net: None, // pnp carries no net association (Open Question #2)
            designator: p.designator,
            x: p.center_x,
            y: p.center_y,
            layer: p.layer,
            rotation: p.rotation,
        });
    }

    Ok(BoardIr {
        pads,
        layers,
        outline: Vec::new(), // pnp carries no Edge.Cuts outline (Open Question #2)
        units: doc.units,
    })
}

/// Shell out to `kicad-cruncher pnp <board_path> --format json` and parse the
/// result into a [`BoardIr`].
///
/// The CLI does **not** stream JSON to stdout (stdout carries log lines only);
/// `--format json -o <dir>` writes a single `*_pnp.json` file into the output
/// directory. We therefore run it against a freshly-created temp directory, read
/// the produced file, and clean up. Requires `kicad-cruncher` on `PATH`. The live
/// path is covered by an `#[ignore]`d integration test so the offline suite stays
/// hermetic.
pub fn run_kicad_cruncher(board_path: &str) -> Result<BoardIr, IngestError> {
    let out_dir = make_temp_dir().map_err(IngestError::Spawn)?;

    let result = (|| {
        let output = Command::new("kicad-cruncher")
            .args(["pnp", board_path, "--format", "json", "-o"])
            .arg(&out_dir)
            .output()
            .map_err(IngestError::Spawn)?;

        if !output.status.success() {
            return Err(IngestError::CliFailed {
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let json_path = find_pnp_json(&out_dir).ok_or_else(|| IngestError::CliFailed {
            code: output.status.code(),
            stderr: format!(
                "kicad-cruncher produced no *_pnp.json in {}",
                out_dir.display()
            ),
        })?;
        let contents = std::fs::read_to_string(&json_path).map_err(IngestError::Spawn)?;
        parse_pnp_json(&contents)
    })();

    // Best-effort cleanup; ignore errors so they don't mask the real result.
    let _ = std::fs::remove_dir_all(&out_dir);
    result
}

/// Create a unique temp directory for one CLI invocation.
fn make_temp_dir() -> std::io::Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut dir = std::env::temp_dir();
    dir.push(format!("mr-ingest-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Find the single `*_pnp.json` the CLI writes into `dir`.
fn find_pnp_json(dir: &PathBuf) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_pnp.json"))
        {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_FIXTURE: &str = include_str!("../tests/fixtures/pnp_sample.json");
    const SYNTH_MULTILAYER: &str = include_str!("../tests/fixtures/pnp_multilayer_synthetic.json");

    #[test]
    fn parses_real_fixture_pad_count_and_layers() {
        let ir = parse_pnp_json(REAL_FIXTURE).expect("real fixture should parse");
        // datum.kicad_pcb: 7 fiducial placements, all on the top layer.
        assert_eq!(ir.pads.len(), 7, "expected 7 pads from datum fixture");
        assert_eq!(ir.layers, vec!["top".to_string()]);
        assert_eq!(ir.units.as_deref(), Some("mm"));
        // No outline / nets from pnp (Open Question #2).
        assert!(ir.outline.is_empty());
        assert!(ir.pads.iter().all(|p| p.net.is_none()));
    }

    #[test]
    fn maps_placement_fields_into_pad() {
        let ir = parse_pnp_json(REAL_FIXTURE).unwrap();
        let fid1 = ir
            .pads
            .iter()
            .find(|p| p.designator == "FID1")
            .expect("FID1 present");
        assert_eq!(fid1.x, 40.0);
        assert_eq!(fid1.y, 25.0);
        assert_eq!(fid1.layer, "top");
        assert_eq!(fid1.rotation, 0.0);
    }

    #[test]
    fn dedups_layers_in_first_seen_order() {
        let ir = parse_pnp_json(SYNTH_MULTILAYER).expect("synthetic fixture should parse");
        assert_eq!(ir.pads.len(), 3);
        // U1 is top, then R1/C1 bottom -> first-seen order [top, bottom].
        assert_eq!(ir.layers, vec!["top".to_string(), "bottom".to_string()]);
        let r1 = ir.pads.iter().find(|p| p.designator == "R1").unwrap();
        assert_eq!(r1.layer, "bottom");
        assert_eq!(r1.rotation, 180.0);
    }

    #[test]
    fn malformed_json_returns_error_not_panic() {
        let err = parse_pnp_json("{ this is not valid json ");
        assert!(matches!(err, Err(IngestError::Parse(_))));
    }

    #[test]
    fn valid_json_wrong_shape_returns_parse_error() {
        // Well-formed JSON but missing required `placements` / `schema`.
        let err = parse_pnp_json(r#"{"hello": "world"}"#);
        assert!(matches!(err, Err(IngestError::Parse(_))));
    }

    #[test]
    fn unexpected_schema_marker_is_rejected() {
        let bad = r#"{"schema":"wn.kicad_cruncher.pnp.v999","placements":[]}"#;
        let err = parse_pnp_json(bad);
        assert!(matches!(err, Err(IngestError::UnexpectedSchema(s)) if s.contains("v999")));
    }

    #[test]
    fn empty_placements_yields_empty_ir() {
        let doc = r#"{"schema":"wn.kicad_cruncher.pnp.v1","units":"mm","placements":[]}"#;
        let ir = parse_pnp_json(doc).unwrap();
        assert!(ir.pads.is_empty());
        assert!(ir.layers.is_empty());
    }

    /// Live integration test against the installed CLI + a real board. Ignored so
    /// the offline suite stays hermetic; set `MR_KICAD_TEST_BOARD` and run with:
    /// `cargo test -p mr-ingest -- --ignored`
    #[test]
    #[ignore = "requires kicad-cruncher on PATH and a real board file"]
    fn run_kicad_cruncher_live() {
        let board = std::env::var("MR_KICAD_TEST_BOARD")
            .expect("set MR_KICAD_TEST_BOARD to a real .kicad_pcb path");
        let ir = run_kicad_cruncher(&board).expect("live kicad-cruncher run should succeed");
        assert!(!ir.pads.is_empty(), "expected at least one placement");
        assert!(ir.layers.contains(&"top".to_string()));
    }
}
