//! `mr-server` binary — serves the tscircuit `/solve` HTTP endpoint backed by the
//! CPU rip-up router.
//!
//! Usage:
//!
//! ```text
//! mr-server [--port <PORT>]
//! ```
//!
//! `--port` (or `-p`) defaults to `1234`, the port the tscircuit
//! `autorouting-dataset benchmark --solver-url http://localhost:1234` harness
//! expects. The router backend is a [`mr_cpu::RipUpRouter`] (sequential rip-up
//! routing over Lee's wavefront), injected behind `Arc<dyn Router>` so a Metal
//! backend can replace it later without touching this file.

use std::net::SocketAddr;
use std::sync::Arc;

use mr_cpu::RipUpRouter;
use mr_core::Router;

const DEFAULT_PORT: u16 = 1234;

/// Parse `--port`/`-p` from CLI args, falling back to [`DEFAULT_PORT`].
fn parse_port(args: impl Iterator<Item = String>) -> Result<u16, String> {
    let mut port = DEFAULT_PORT;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" | "-p" => {
                let val = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                port = val
                    .parse()
                    .map_err(|_| format!("invalid port: {val:?}"))?;
            }
            other if other.starts_with("--port=") => {
                let val = &other["--port=".len()..];
                port = val
                    .parse()
                    .map_err(|_| format!("invalid port: {val:?}"))?;
            }
            "-h" | "--help" => {
                return Err("usage: mr-server [--port <PORT>]".to_string());
            }
            other => return Err(format!("unexpected argument: {other:?}")),
        }
    }
    Ok(port)
}

#[tokio::main]
async fn main() {
    // Skip argv[0] (the program name).
    let port = match parse_port(std::env::args().skip(1)) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let router: Arc<dyn Router + Send + Sync> = Arc::new(RipUpRouter::new());
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("mr-server listening on http://{addr} (POST /solve, GET /health)");

    if let Err(e) = mr_server::serve(addr, router).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<u16, String> {
        parse_port(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn default_port_when_no_args() {
        assert_eq!(parse(&[]).unwrap(), DEFAULT_PORT);
    }

    #[test]
    fn parses_port_flag_forms() {
        assert_eq!(parse(&["--port", "8080"]).unwrap(), 8080);
        assert_eq!(parse(&["-p", "9000"]).unwrap(), 9000);
        assert_eq!(parse(&["--port=4321"]).unwrap(), 4321);
    }

    #[test]
    fn rejects_bad_port() {
        assert!(parse(&["--port", "notanumber"]).is_err());
        assert!(parse(&["--port"]).is_err());
        assert!(parse(&["--bogus"]).is_err());
    }
}
