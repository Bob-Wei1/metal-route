//! `metalroute` — the user-facing PCB autorouter CLI.
//!
//! A thin dispatcher over [`mr_cli`]; all real logic lives in the library so it
//! is unit/integration testable without spawning a process.

use anyhow::Result;
use clap::Parser;
use mr_cli::{bench::run_bench, run_handoff, run_project, run_route, Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Route(args) => {
            let summary = run_route(args)?;
            // Solution JSON goes to --out/stdout; the summary goes to stderr so
            // it never pollutes a piped solution.
            eprintln!("{summary}");
        }
        Command::Project(args) => {
            let projection = run_project(args)?;
            println!("{projection}");
        }
        Command::Bench(args) => {
            let report = run_bench(args)?;
            eprintln!(
                "bench: {} boards, {}/{} nets routed ({:.1}% completion), {:.0} nets/sec, M2 {} ({:.2}x)",
                report.boards,
                report.nets_routed,
                report.nets_total,
                report.completion_rate * 100.0,
                report.nets_per_sec,
                report.m2_verdict,
                report.m2_projected_speedup,
            );
        }
        Command::Handoff(args) => {
            let out = run_handoff(args)?;
            eprintln!("handoff ok={} ({} bytes stdout)", out.status_ok, out.stdout.len());
        }
    }
    Ok(())
}
