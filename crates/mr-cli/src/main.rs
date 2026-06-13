//! `metalroute` — the user-facing PCB autorouter CLI.
//!
//! A thin dispatcher over [`mr_cli`]; all real logic lives in the library so it
//! is unit/integration testable without spawning a process.

use anyhow::Result;
use clap::Parser;
use mr_cli::{
    bench::run_bench, drc::run_drc, run_handoff, run_project, run_route, run_route_dsn, Cli,
    Command,
};

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
            eprintln!(
                "handoff ok={} ({} bytes stdout)",
                out.status_ok,
                out.stdout.len()
            );
        }
        Command::RouteDsn(args) => {
            let r = run_route_dsn(args)?;
            // Human-readable report (stderr so it never pollutes a piped solution).
            eprintln!(
                "DSN parsed: layers={} components={} pads={} nets={} (skipped {} <2-pin), board {:.2}x{:.2} mm, min_trace_width {:.3} mm",
                r.stats.layers,
                r.stats.components,
                r.stats.pads,
                r.stats.nets,
                r.stats.nets_skipped_small,
                r.stats.board_w_mm,
                r.stats.board_h_mm,
                r.stats.min_trace_width_mm,
            );
            eprintln!(
                "Routing {} original net(s) at resolution {:.4} mm -> grid {}x{} ({} cells)",
                r.original_nets,
                r.resolution,
                r.grid_w,
                r.grid_h,
                (r.grid_w as u64) * (r.grid_h as u64),
            );
            eprintln!(
                "Routed {}/{} two-point nets (connectivity {:.1}%), {} original net(s) fully connected",
                r.routed_nets,
                r.total_nets,
                r.connectivity_pct(),
                r.fully_connected,
            );
            eprintln!(
                "Wall-clock {:.3} s ({:.0} nets/sec)",
                r.wall_s,
                r.nets_per_sec(),
            );
            // Scrape-friendly one-liner on stdout.
            println!("{}", r.result_line());
        }
        Command::Drc(args) => {
            let r = run_drc(args)?;
            eprintln!(
                "DRC {}: routed {}/{} two-point nets, {} fully-connected, {} vias",
                r.design, r.routed_nets, r.total_nets, r.fully_connected, r.vias,
            );
            eprintln!(
                "DRC violations: {} total — {} clearance, {} via-through-plane, {} annular-ring (clearance rule {:.3} mm)",
                r.summary.total,
                r.summary.clearance,
                r.summary.via_through_plane,
                r.summary.annular_ring,
                r.clearance_mm,
            );
            // Scrape-friendly one-liner on stdout.
            println!(
                "drc design={} violations={} clearance={} via_through_plane={} annular_ring={}",
                r.design,
                r.summary.total,
                r.summary.clearance,
                r.summary.via_through_plane,
                r.summary.annular_ring,
            );
        }
    }
    Ok(())
}
