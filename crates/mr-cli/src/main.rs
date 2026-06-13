//! `metalroute` — the user-facing PCB autorouter CLI.
//!
//! A thin dispatcher over [`mr_cli`]; all real logic lives in the library so it
//! is unit/integration testable without spawning a process.

use anyhow::Result;
use clap::Parser;
use mr_cli::{run_project, run_route, Cli, Command};

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
    }
    Ok(())
}
