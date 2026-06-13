//! `mr-bridge` — hand the global routing solution to Freerouting for detailed
//! routing.
//!
//! Detailed routing is delegated to the existing Python `bed-of-nails` tool
//! (Specctra DSN export -> Freerouting -> SES import). This crate owns only the
//! *invocation seam*: it builds the argument vector that drives
//! `bed_of_nails.apply_routing` and shells out to it.
//!
//! The subprocess is hidden behind the [`CommandRunner`] trait so the handoff
//! logic is unit-testable without ever launching Java or Python. [`SystemRunner`]
//! is the real implementation ([`std::process::Command`]); [`MockRunner`] records
//! the last invocation for tests.
//!
//! M5 wires this into the workspace; for now the focus is the seam, not the DSN
//! export (the Python side owns DSN/SES).

use std::cell::RefCell;

use mr_core::RouterError;
use serde::{Deserialize, Serialize};

/// Configuration for the Freerouting handoff.
///
/// The defaults mirror `bed_of_nails.route.apply_routing` (`passes=20`,
/// `timeout_s=600`) and assume the `bon` console-script is on `PATH`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// `-mp` passes handed to Freerouting (`apply_routing(passes=...)`).
    pub freerouting_passes: u32,
    /// Wall-clock budget for the whole routing run, seconds
    /// (`apply_routing(timeout_s=...)`).
    pub timeout_s: u64,
    /// The bed-of-nails entrypoint command (the console script, on `PATH`).
    pub bon_command: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            freerouting_passes: 20,
            timeout_s: 600,
            bon_command: "bon".to_string(),
        }
    }
}

/// The captured result of running a subprocess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutput {
    /// True when the process exited successfully (exit status 0).
    pub status_ok: bool,
    pub stdout: String,
    pub stderr: String,
}

/// The seam between [`handoff`] and the operating system.
///
/// Implemented by [`SystemRunner`] in production and [`MockRunner`] in tests so
/// the invocation logic can be exercised without Java/Python/Freerouting.
pub trait CommandRunner {
    /// Run `program` with `args`, capturing stdout/stderr and success.
    fn run(&self, program: &str, args: &[String]) -> std::io::Result<RunOutput>;
}

/// The real runner: spawns `program` via [`std::process::Command`] and captures
/// its output.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemRunner;

impl CommandRunner for SystemRunner {
    fn run(&self, program: &str, args: &[String]) -> std::io::Result<RunOutput> {
        let output = std::process::Command::new(program).args(args).output()?;
        Ok(RunOutput {
            status_ok: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// A single recorded invocation: `(program, args)`.
pub type Invocation = (String, Vec<String>);

/// A test runner that records the last invocation and returns a canned
/// [`RunOutput`].
///
/// `last()` returns the most recent `(program, args)` pair, or `None` if `run`
/// has not been called.
#[derive(Debug)]
pub struct MockRunner {
    response: RunOutput,
    last: RefCell<Option<Invocation>>,
}

impl MockRunner {
    /// A runner whose `run` returns `response` and records each invocation.
    pub fn new(response: RunOutput) -> Self {
        Self {
            response,
            last: RefCell::new(None),
        }
    }

    /// A runner that always reports success with empty output.
    pub fn ok() -> Self {
        Self::new(RunOutput {
            status_ok: true,
            stdout: String::new(),
            stderr: String::new(),
        })
    }

    /// A runner that always reports failure, with `stderr` set to `stderr`.
    pub fn failing(stderr: impl Into<String>) -> Self {
        Self::new(RunOutput {
            status_ok: false,
            stdout: String::new(),
            stderr: stderr.into(),
        })
    }

    /// The most recent `(program, args)` passed to `run`, if any.
    pub fn last(&self) -> Option<Invocation> {
        self.last.borrow().clone()
    }
}

impl CommandRunner for MockRunner {
    fn run(&self, program: &str, args: &[String]) -> std::io::Result<RunOutput> {
        *self.last.borrow_mut() = Some((program.to_string(), args.to_vec()));
        Ok(self.response.clone())
    }
}

/// Build the `(program, args)` that invoke `bed_of_nails` detailed routing for
/// `pcb_path`.
///
/// Mirrors `apply_routing(pcb_path, passes=cfg.freerouting_passes,
/// timeout_s=cfg.timeout_s)` as a CLI call:
///
/// ```text
/// bon route <pcb_path> --passes <N> --timeout <T>
/// ```
///
/// This is the seam the tests assert against.
pub fn build_apply_routing_args(pcb_path: &str, cfg: &BridgeConfig) -> (String, Vec<String>) {
    let args = vec![
        "route".to_string(),
        pcb_path.to_string(),
        "--passes".to_string(),
        cfg.freerouting_passes.to_string(),
        "--timeout".to_string(),
        cfg.timeout_s.to_string(),
    ];
    (cfg.bon_command.clone(), args)
}

/// Hand `pcb_path` off to bed-of-nails for detailed routing via `runner`.
///
/// Builds the invocation with [`build_apply_routing_args`], runs it, and returns
/// the captured [`RunOutput`]. A non-ok exit status is mapped to
/// [`RouterError::BackendUnavailable`] (carrying the captured `stderr`); an
/// `io::Error` from spawning is likewise reported as
/// [`RouterError::BackendUnavailable`].
pub fn handoff<R: CommandRunner>(
    runner: &R,
    pcb_path: &str,
    cfg: &BridgeConfig,
) -> Result<RunOutput, RouterError> {
    let (program, args) = build_apply_routing_args(pcb_path, cfg);
    let output = runner.run(&program, &args).map_err(|e| {
        RouterError::BackendUnavailable(format!("failed to spawn `{program}`: {e}"))
    })?;
    if !output.status_ok {
        return Err(RouterError::BackendUnavailable(format!(
            "`{program}` exited with failure: {}",
            output.stderr.trim()
        )));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_for_default_config() {
        let cfg = BridgeConfig::default();
        let (program, args) = build_apply_routing_args("board.kicad_pcb", &cfg);
        assert_eq!(program, "bon");
        assert_eq!(
            args,
            vec![
                "route".to_string(),
                "board.kicad_pcb".to_string(),
                "--passes".to_string(),
                "20".to_string(),
                "--timeout".to_string(),
                "600".to_string(),
            ]
        );
    }

    #[test]
    fn args_for_custom_passes_and_timeout() {
        let cfg = BridgeConfig {
            freerouting_passes: 50,
            timeout_s: 120,
            bon_command: "bon".to_string(),
        };
        let (program, args) = build_apply_routing_args("/tmp/x.kicad_pcb", &cfg);
        assert_eq!(program, "bon");
        assert_eq!(
            args,
            vec![
                "route".to_string(),
                "/tmp/x.kicad_pcb".to_string(),
                "--passes".to_string(),
                "50".to_string(),
                "--timeout".to_string(),
                "120".to_string(),
            ]
        );
    }

    #[test]
    fn custom_bon_command_is_used_as_program() {
        let cfg = BridgeConfig {
            bon_command: "/opt/venv/bin/bon".to_string(),
            ..Default::default()
        };
        let (program, _args) = build_apply_routing_args("b.kicad_pcb", &cfg);
        assert_eq!(program, "/opt/venv/bin/bon");
    }

    #[test]
    fn handoff_runs_and_records_expected_invocation() {
        let runner = MockRunner::ok();
        let cfg = BridgeConfig::default();
        let out = handoff(&runner, "board.kicad_pcb", &cfg).expect("handoff should succeed");
        assert!(out.status_ok);

        let (program, args) = runner.last().expect("runner should have recorded a call");
        assert_eq!(program, "bon");
        assert_eq!(
            args,
            vec![
                "route".to_string(),
                "board.kicad_pcb".to_string(),
                "--passes".to_string(),
                "20".to_string(),
                "--timeout".to_string(),
                "600".to_string(),
            ]
        );
    }

    #[test]
    fn handoff_maps_failure_to_backend_unavailable() {
        let runner = MockRunner::failing("java not found on PATH");
        let cfg = BridgeConfig::default();
        let err = handoff(&runner, "board.kicad_pcb", &cfg).expect_err("non-ok status must error");
        match err {
            RouterError::BackendUnavailable(msg) => {
                assert!(
                    msg.contains("java not found on PATH"),
                    "stderr should be propagated, got: {msg}"
                );
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
        // even on failure the invocation is recorded
        assert!(runner.last().is_some());
    }

    #[test]
    fn default_config_values() {
        let cfg = BridgeConfig::default();
        assert_eq!(cfg.freerouting_passes, 20);
        assert_eq!(cfg.timeout_s, 600);
        assert_eq!(cfg.bon_command, "bon");
    }

    #[test]
    fn config_roundtrips_through_json() {
        let cfg = BridgeConfig {
            freerouting_passes: 7,
            timeout_s: 42,
            bon_command: "bon".to_string(),
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: BridgeConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }
}
