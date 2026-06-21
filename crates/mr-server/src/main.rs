//! `mr-server` binary — serves the tscircuit `/solve` HTTP endpoint backed by the
//! CPU rip-up router.
//!
//! Usage:
//!
//! ```text
//! mr-server [--port <PORT>] [--solve-layers <N>] [--clearance <mm>]
//! ```
//!
//! `--port` (or `-p`) defaults to `1234`, the port the tscircuit
//! `autorouting-dataset benchmark --solver-url http://localhost:1234` harness
//! expects. `--solve-layers` (or `-l`, env `MR_SOLVE_LAYERS`) is the routing
//! layer budget: every problem is routed on `max(layerCount, N)` layers so the
//! negotiated router can resolve crossings with through-vias even when a problem
//! declares a single layer (the harness checks only connectivity + non-overlap).
//! Defaults to [`mr_server::DEFAULT_SOLVE_LAYERS`]. `--clearance` (or `-c`, env
//! `MR_CLEARANCE`) is the copper clearance budget in mm (trace↔trace and
//! trace↔pad); unset = auto (one trace width), `0` = off. The router backend is a
//! [`mr_cpu::NegotiatedRouter`] (PathFinder-style negotiated-congestion routing
//! over Dijkstra), injected behind `Arc<dyn Router>` so a Metal backend can
//! replace it later without touching this file.

use std::net::SocketAddr;
use std::sync::Arc;

use std::path::PathBuf;

use mr_cpu::NegotiatedRouter;
use mr_server::{RouterFactory, DEFAULT_SOLVE_LAYERS};

const DEFAULT_PORT: u16 = 1234;

/// Default corpus directory (relative to the working dir), scanned by `/api/boards`.
const DEFAULT_CORPUS_DIR: &str = "benchmarks/corpus";

/// Default directory of the built SPA, served as the static fallback.
const DEFAULT_WEB_DIR: &str = "web/dist";

/// Parsed CLI options.
struct Opts {
    port: u16,
    solve_layers: u32,
    /// Clearance budget in continuous units: `None` = auto (one trace width),
    /// `Some(mm)` = fixed (`Some(0.0)` = clearance off).
    clearance_mm: Option<f64>,
    /// Root of the board corpus for the visualiser API.
    corpus_dir: PathBuf,
    /// Directory of the built SPA to serve as the static fallback.
    web_dir: PathBuf,
}

/// Parse `--port`/`-p`, `--solve-layers`/`-l`, `--clearance`/`-c`, `--corpus-dir`,
/// `--web-dir` from CLI args. The `default_*` params are used when the corresponding
/// flag is absent (main passes the env-var values).
fn parse_opts(
    args: impl Iterator<Item = String>,
    default_layers: u32,
    default_clearance: Option<f64>,
    default_corpus_dir: PathBuf,
    default_web_dir: PathBuf,
) -> Result<Opts, String> {
    let mut port = DEFAULT_PORT;
    let mut solve_layers = default_layers;
    let mut clearance_mm = default_clearance;
    let mut corpus_dir = default_corpus_dir;
    let mut web_dir = default_web_dir;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" | "-p" => {
                let val = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                port = val.parse().map_err(|_| format!("invalid port: {val:?}"))?;
            }
            other if other.starts_with("--port=") => {
                let val = &other["--port=".len()..];
                port = val.parse().map_err(|_| format!("invalid port: {val:?}"))?;
            }
            "--solve-layers" | "-l" => {
                let val = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                solve_layers = parse_layers(&val)?;
            }
            other if other.starts_with("--solve-layers=") => {
                let val = &other["--solve-layers=".len()..];
                solve_layers = parse_layers(val)?;
            }
            "--clearance" | "-c" => {
                let val = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                clearance_mm = Some(parse_clearance(&val)?);
            }
            other if other.starts_with("--clearance=") => {
                let val = &other["--clearance=".len()..];
                clearance_mm = Some(parse_clearance(val)?);
            }
            "--corpus-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                corpus_dir = PathBuf::from(val);
            }
            other if other.starts_with("--corpus-dir=") => {
                corpus_dir = PathBuf::from(&other["--corpus-dir=".len()..]);
            }
            "--web-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                web_dir = PathBuf::from(val);
            }
            other if other.starts_with("--web-dir=") => {
                web_dir = PathBuf::from(&other["--web-dir=".len()..]);
            }
            "-h" | "--help" => {
                return Err("usage: mr-server [--port <PORT>] [--solve-layers <N>] \
                     [--clearance <mm>] [--corpus-dir <DIR>] [--web-dir <DIR>]"
                    .to_string());
            }
            other => return Err(format!("unexpected argument: {other:?}")),
        }
    }
    Ok(Opts {
        port,
        solve_layers,
        clearance_mm,
        corpus_dir,
        web_dir,
    })
}

/// Read `MR_CORPUS_DIR` / `MR_WEB_DIR`, falling back to the compiled defaults.
fn env_dir(var: &str, default: &str) -> PathBuf {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => PathBuf::from(default),
    }
}

/// Parse a clearance budget in mm, requiring a finite value `>= 0` (`0` = off).
fn parse_clearance(val: &str) -> Result<f64, String> {
    let v: f64 = val
        .parse()
        .map_err(|_| format!("invalid clearance: {val:?}"))?;
    if !v.is_finite() || v < 0.0 {
        return Err("clearance must be a finite value >= 0".to_string());
    }
    Ok(v)
}

/// Read the `MR_CLEARANCE` env var. Absent/empty => `None` (auto = one trace width).
fn env_clearance() -> Result<Option<f64>, String> {
    match std::env::var("MR_CLEARANCE") {
        Ok(v) if !v.is_empty() => Ok(Some(parse_clearance(&v)?)),
        _ => Ok(None),
    }
}

/// Parse a layer budget, requiring `N >= 1`.
fn parse_layers(val: &str) -> Result<u32, String> {
    let n: u32 = val
        .parse()
        .map_err(|_| format!("invalid solve-layers: {val:?}"))?;
    if n < 1 {
        return Err("solve-layers must be >= 1".to_string());
    }
    Ok(n)
}

/// Read the `MR_SOLVE_LAYERS` env var, falling back to [`DEFAULT_SOLVE_LAYERS`].
fn env_solve_layers() -> Result<u32, String> {
    match std::env::var("MR_SOLVE_LAYERS") {
        Ok(v) if !v.is_empty() => parse_layers(&v),
        _ => Ok(DEFAULT_SOLVE_LAYERS),
    }
}

#[tokio::main]
async fn main() {
    // Skip argv[0] (the program name).
    let default_corpus = env_dir("MR_CORPUS_DIR", DEFAULT_CORPUS_DIR);
    let default_web = env_dir("MR_WEB_DIR", DEFAULT_WEB_DIR);
    let parsed = env_solve_layers().and_then(|dl| {
        env_clearance().and_then(|dc| {
            parse_opts(
                std::env::args().skip(1),
                dl,
                dc,
                default_corpus,
                default_web,
            )
        })
    });
    let opts = match parsed {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    // Backend factory: a `NegotiatedRouter` at the per-problem clearance budget,
    // fed the problem's non-uniform (Hanan) grid coords so its geometric A* cost +
    // heuristic match the real cell spacing.
    let make_router: RouterFactory = Arc::new(|mm, coords| {
        Box::new(
            NegotiatedRouter::new()
                .with_clearance_mm(mm)
                .with_coords(coords),
        )
    });
    let addr = SocketAddr::from(([0, 0, 0, 0], opts.port));
    let clr = match opts.clearance_mm {
        Some(0.0) => "off".to_string(),
        Some(mm) => format!("{mm} units"),
        None => "auto (1 trace width)".to_string(),
    };
    println!(
        "mr-server listening on http://{addr} (POST /solve, GET /health, \
         GET /api/boards, POST /api/trace), routing on >= {} layers, clearance {clr}, \
         corpus {}, web {}",
        opts.solve_layers,
        opts.corpus_dir.display(),
        opts.web_dir.display(),
    );

    if let Err(e) = mr_server::serve(
        addr,
        make_router,
        opts.solve_layers,
        opts.clearance_mm,
        opts.corpus_dir,
        opts.web_dir,
    )
    .await
    {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Opts, String> {
        // Default clearance `None` (auto) mirrors `MR_CLEARANCE` being unset.
        parse_opts(
            args.iter().map(|s| s.to_string()),
            DEFAULT_SOLVE_LAYERS,
            None,
            PathBuf::from(DEFAULT_CORPUS_DIR),
            PathBuf::from(DEFAULT_WEB_DIR),
        )
    }

    #[test]
    fn default_port_when_no_args() {
        let o = parse(&[]).unwrap();
        assert_eq!(o.port, DEFAULT_PORT);
        assert_eq!(o.solve_layers, DEFAULT_SOLVE_LAYERS);
    }

    #[test]
    fn parses_port_flag_forms() {
        assert_eq!(parse(&["--port", "8080"]).unwrap().port, 8080);
        assert_eq!(parse(&["-p", "9000"]).unwrap().port, 9000);
        assert_eq!(parse(&["--port=4321"]).unwrap().port, 4321);
    }

    #[test]
    fn parses_solve_layers_flag_forms() {
        assert_eq!(parse(&["--solve-layers", "4"]).unwrap().solve_layers, 4);
        assert_eq!(parse(&["-l", "3"]).unwrap().solve_layers, 3);
        assert_eq!(parse(&["--solve-layers=2"]).unwrap().solve_layers, 2);
    }

    #[test]
    fn parses_clearance_flag_forms() {
        assert_eq!(parse(&[]).unwrap().clearance_mm, None); // auto
        assert_eq!(
            parse(&["--clearance", "0.2"]).unwrap().clearance_mm,
            Some(0.2)
        );
        assert_eq!(parse(&["-c", "0"]).unwrap().clearance_mm, Some(0.0));
        assert_eq!(
            parse(&["--clearance=0.15"]).unwrap().clearance_mm,
            Some(0.15)
        );
    }

    #[test]
    fn rejects_bad_clearance() {
        assert!(parse(&["--clearance", "-1"]).is_err());
        assert!(parse(&["--clearance", "x"]).is_err());
        assert!(parse(&["--clearance"]).is_err());
    }

    #[test]
    fn rejects_bad_port() {
        assert!(parse(&["--port", "notanumber"]).is_err());
        assert!(parse(&["--port"]).is_err());
        assert!(parse(&["--bogus"]).is_err());
    }

    #[test]
    fn rejects_bad_solve_layers() {
        assert!(parse(&["--solve-layers", "0"]).is_err());
        assert!(parse(&["--solve-layers", "x"]).is_err());
        assert!(parse(&["--solve-layers"]).is_err());
    }
}
