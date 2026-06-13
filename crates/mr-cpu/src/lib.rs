//! `mr-cpu` — the CPU routers and the M0 GPU de-risk spike.
//!
//! This crate implements the routing algorithms behind the [`mr_core::Router`]
//! contract:
//!
//! * [`LeeRouter`] (A1 / M1) — Lee's wavefront as Dijkstra single-source shortest
//!   path; routes each net independently.
//! * [`AStarRouter`] (A4) — A* with a Manhattan heuristic; same cost & contract,
//!   equal total cost to [`LeeRouter`].
//! * [`RipUpRouter`] (A3 / M2) — sequential routing with bounded
//!   rip-up-on-collision (`K == 20` passes).
//! * [`sweep`] (A2 / M0) — the separable H/V prefix-min distance field
//!   ([`sweep_distance_field`]) validated against the Dijkstra field
//!   ([`bfs_distance_field`]). This is the GPU de-risk spike.
//!
//! ## Shared conventions
//!
//! * A routed net's `cost` is the sum of `grid.cost_at(cell)` over the path
//!   **excluding** the source cell — so on a unit-cost grid `cost == path.len() - 1`.
//! * Tie-break is [`mr_core::TieBreak::LowerCellIdx`]: neighbours are processed in
//!   ascending [`mr_core::CellIdx`] order with first-writer-wins predecessors, so
//!   path reconstruction is deterministic.
//! * The grid is 4-connected; obstacle cells (`cost == mr_core::OBSTACLE`) are
//!   impassable.

mod astar;
mod dijkstra;
mod lee;
mod negotiated;
mod ripup;
pub mod sweep;

pub use astar::AStarRouter;
pub use lee::LeeRouter;
pub use negotiated::{NegotiatedRouter, MAX_ITERS, SCALE};
pub use ripup::{RipUpRouter, MAX_PASSES};
pub use sweep::{bfs_distance_field, sweep_distance_field};
