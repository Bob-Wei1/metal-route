//! `NegotiatedRouter` (Phase 2) — PathFinder-style negotiated-congestion routing.
//!
//! The [`RipUpRouter`](crate::RipUpRouter) routes nets sequentially with a strict
//! priority rule: a lower-index net is never displaced for a higher-index one. Two
//! nets competing for crossing corridors therefore often leave one permanently
//! unrouted even when a disjoint solution exists. This router instead lets every
//! net route greedily on its own *congestion-priced* copy of the grid, then makes
//! shared cells progressively more expensive until the routes separate
//! (negotiated congestion, à la Nair/McMurchie PathFinder).
//!
//! ## Cost model (fixed-point integers, no floats)
//!
//! Base passable cell costs [`SCALE`]. The price net `i` pays to step onto cell
//! `c` is
//!
//! ```text
//! cost(c) = SCALE + history[c] + pfac * SCALE * occ_excl_i(c)
//! ```
//!
//! where `history[c]` is a permanent per-cell aversion accumulated over iterations
//! for cells that were over-used, `occ_excl_i(c)` is how many *other* nets
//! currently occupy `c`, and `pfac` (the present-penalty factor) grows each
//! iteration so sharing becomes steadily more expensive. Costs are capped strictly
//! below [`OBSTACLE`] so a priced cell is never confused with an impassable one.
//!
//! ## Connection groups
//!
//! A multi-terminal connection is decomposed upstream into chained sub-nets named
//! `"<conn>#0"`, `"<conn>#1"`, … These are electrically one net and are *allowed*
//! to share cells. Overuse is therefore measured across connection **groups** (the
//! name prefix before `'#'`): a cell is over-used only when ≥2 distinct groups
//! occupy it.
//!
//! ## Convergence and legalization
//!
//! The negotiation loop runs at most [`MAX_ITERS`] iterations; it stops early when
//! no cell is over-used. Because convergence is not guaranteed within the bound, a
//! final legalization pass commits nets group-by-group, marking already-committed
//! foreign-group cells as hard obstacles and rerouting once if needed. This makes
//! the returned [`BoardRoute`] cell-disjoint across groups even if negotiation did
//! not fully settle. The router NEVER loops unbounded.

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{HashMap, HashSet},
};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use mr_core::{
    BoardRoute, CandidateEval, CellIdx, Cost, Dims, Grid, GridCoords, IterSnapshot,
    LegalizationTrace, NetEndpoints, RouteResult, RouteTrace, Router, RouterError, TracedNet,
    ViaModel, OBSTACLE,
};

use crate::dijkstra::{astar_buf, edge_cost, SearchBuf, COST_SCALE};

/// Fixed-point cost scale: the base cost of one unit of planar travel.
///
/// Numerically equal to `COST_SCALE` (`16`): a planar step of unit geometric
/// length costs `SCALE`, and on a non-uniform grid a step of length `len` costs
/// `round(len * COST_SCALE)` (see `edge_cost`). All congestion penalties below are
/// expressed in multiples of `SCALE`, so they keep their strength *relative to the
/// geometric base step* regardless of the board's physical pitch. On a uniform grid
/// every step has length 1, so the base step is exactly `SCALE` and the whole cost
/// model is byte-identical to the pre-geometric router.
pub const SCALE: Cost = COST_SCALE as Cost;

/// Historical soft clearance penalty (TritonRoute `objCost`-style). Inter-net
/// clearance is now a STRICTLY HARD constraint in the committing legalization pass
/// ([`route_legal`] hard-blocks foreign halo cells), so a net that cannot reach its
/// dst clearance-cleanly is left unrouted/congested rather than violating spacing.
/// This constant is retained for API stability and reference; `route_legal` no
/// longer prices it. The iterative NEGOTIATION phase still uses a separate SOFT
/// pressure ([`CLEARANCE_NEG_WEIGHT`]) to spread nets so legalization can succeed.
#[allow(dead_code)]
pub const CLEARANCE_PENALTY: Cost = 16 * SCALE;

/// Soft clearance weight priced into the NEGOTIATION search (TritonRoute `objCost`
/// analog). Distinct from [`CLEARANCE_PENALTY`] (a legalization-only cost): during
/// negotiation each net additionally pays `pfac * CLEARANCE_NEG_WEIGHT * present_halo[c]`
/// to ENTER a cell `c` that lies inside *another* net's clearance / via-keepout halo
/// (see `present_halo` in [`NegotiatedRouter::route`]). Because the cost scales with
/// `pfac` (which grows each iteration), routing within a neighbour's spacing becomes
/// steadily more expensive, so the negotiation SPREADS nets apart by clearance over
/// iterations rather than relying on legalization alone to enforce it.
///
/// Tunable. Starts at [`SCALE`] (`= 16`): one unit of clearance overlap is priced
/// like one unit of direct cell sharing (`pfac * SCALE * present[c]`), a deliberately
/// soft starting point that the per-iteration `pfac` ramp amplifies. Like every other
/// negotiation cost it is capped strictly below [`OBSTACLE`] so a penalized cell is
/// never confused with an impassable one. When `clearance_cells == 0` AND
/// `via_model.keepout_mm == 0.0` the halo field stays all-zero and this term contributes
/// nothing, keeping the default router byte-identical to the pre-clearance behaviour.
pub const CLEARANCE_NEG_WEIGHT: Cost = SCALE;

// The fused Jacobi self stamp stores copper occupancy and clearance multiplicity
// in one counter. That is algebraically exact only while both terms have the same
// coefficient; fail at compile time rather than silently changing route prices if
// either tuning constant diverges later.
const _: () = assert!(CLEARANCE_NEG_WEIGHT == SCALE);

/// Maximum negotiation iterations before falling through to legalization.
pub const MAX_ITERS: u32 = 60;

/// Net-count threshold above which the negotiation loop routes its nets in
/// PARALLEL (the snapshot-based Jacobi merge) even when clearance is inactive.
/// Below it, small boards keep the sequential Gauss-Seidel path so the
/// deterministic unit tests and single-layer fixtures stay byte-identical. Real
/// many-net boards (e.g. tscircuit `keyboards`, 40–70 nets) are otherwise
/// single-threaded and dominate wall-clock; routing them across all cores is the
/// difference between ~85 s and a few seconds. The parallel path is deterministic
/// (index-ordered merge) regardless of net count.
const PARALLEL_NEGOTIATION_THRESHOLD: usize = 16;

/// A full-grid congestion field pays off only when enough independent searches
/// amortize its O(cells) construction. Smaller dirty batches keep the unfused path.
const FUSED_JACOBI_MIN_DIRTY_NETS: usize = 32;

/// Bounded deterministic portfolio: grouped medium-sized boards receive one
/// additional all-serial negotiation candidate. Tiny boards already use serial
/// negotiation, while both a net-count bound and an unconditional cell-count bound
/// prevent the fallback from turning a difficult route into a latency multiplier.
const PORTFOLIO_MIN_NETS: usize = PARALLEL_NEGOTIATION_THRESHOLD + 1;
const PORTFOLIO_MAX_NETS: usize = 179;
const PORTFOLIO_CELL_CAP: usize = 250_000;

/// Completion-only legalization-order fallback. Three independent hard bounds keep
/// up to four extra legalization passes from multiplying latency on large boards.
const ORDER_PORTFOLIO_MIN_GROUPS: usize = 6;
const ORDER_PORTFOLIO_MAX_GROUPS: usize = 16;
const ORDER_PORTFOLIO_MAX_NETS: usize = 32;
const ORDER_PORTFOLIO_CELL_CAP: usize = 250_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NegotiationMode {
    Adaptive,
    ForceSerial,
}

/// Bounded rip-up-and-reroute budget multipliers (see [`ripup_legalize`]). The
/// global budget caps the total number of rip-up operations; the per-net cap
/// bounds how many times any single net may be displaced. Both guarantee
/// termination — once either is hit, the stage stops ripping and keeps what it has.
const RIPUP_GLOBAL_BUDGET_PER_NET: usize = 20;
const RIPUP_PER_NET_CAP_EXTRA: usize = 4;

/// The blocker-informed restart stores and probes an O(groups) dependency graph,
/// then pays for one more whole legalization pass over every net and grid cell.
/// Bound all three input axes while retaining the measured bugreport50 recovery.
const GUIDED_RESTART_MAX_GROUPS: usize = 192;
const GUIDED_RESTART_MAX_NETS: usize = 384;
const GUIDED_RESTART_MAX_CELLS: usize = 1_500_000;
const _: () = assert!(GUIDED_RESTART_MAX_GROUPS >= 168);
const _: () = assert!(GUIDED_RESTART_MAX_NETS >= 322);
const _: () = assert!(GUIDED_RESTART_MAX_CELLS >= 1_294_488);

/// Per-net committed paths from one legalization pass, in input net order: `Some`
/// when the net was placed, `None` when it could not be (dropped/unrouted).
type Committed = Vec<Option<Vec<CellIdx>>>;

/// Match the native DRC's geometric tolerance: a pair exactly at required spacing
/// is legal, and only a centre distance below `required - epsilon` conflicts.
const VIA_SPACING_EPS_MM: f64 = 1e-9;

/// tscircuit treats same-net via centres at most 0.005 mm apart as one physical
/// stacked drill site. Keep this distinct from [`VIA_SPACING_EPS_MM`], which is
/// only the numerical tolerance at a clearance boundary.
const VIA_SAME_LOCATION_MM: f64 = 5e-3;

/// Clearance-map sentinels. A single-owner cell can be ignored by that owner;
/// a mixed cell is covered by two or more groups and is therefore foreign to
/// every group. Rebuilding the map after a rip recovers the precise remaining
/// owner when one side of an overlap disappears.
const HALO_FREE: i64 = -1;
const HALO_MIXED: i64 = -2;
const VIA_TAG_FREE: i32 = -1;
const VIA_TAG_MIXED: i32 = -2;

#[inline]
fn halo_is_foreign(value: i64, own_group: i64) -> bool {
    value == HALO_MIXED || (value >= 0 && value != own_group)
}

#[inline]
fn via_tag_is_foreign(value: i32, own_group: i64) -> bool {
    value == VIA_TAG_MIXED || (value >= 0 && i64::from(value) != own_group)
}

#[inline]
fn via_group_tag(group: i64) -> i32 {
    i32::try_from(group)
        .ok()
        .filter(|&group| group >= 0)
        .unwrap_or(VIA_TAG_MIXED)
}

/// Dense, group-aware feature-pair exclusion fields.
///
/// Each cell records the sole group whose routed features forbid a foreign via
/// centre there, [`VIA_TAG_FREE`] when no group does, or [`VIA_TAG_MIXED`] when several
/// groups' exact Euclidean disks overlap. A candidate via therefore needs only two
/// tag reads (one per touched layer). An optional second field contains only
/// committed-via dilations and is queried by planar moves before endpoint/pad
/// exemptions; unlike the ordinary halo, it is stamped through base obstacles so
/// an own-pad escape cannot bypass a foreign via's keepout.
struct ViaGuard {
    tags: Vec<i32>,
    committed_via_tags: Vec<i32>,
    /// Net-independent drill-spacing disk on the planar grid. The inner
    /// `VIA_SAME_LOCATION_MM` coincidence disk is excluded and recorded in the
    /// group-aware field so one same-group stacked via may be reused while a
    /// foreign-net drill there remains blocked.
    hole_blocked: Vec<bool>,
    hole_centres: Vec<i32>,
    via_trace_mm: f64,
    via_via_mm: f64,
    via_hole_mm: f64,
}

impl ViaGuard {
    /// An unoccupied field for isolated searches. Physical feature-aware mode is
    /// retained through `via_via_mm`, but no dense allocation is needed because
    /// there are no committed foreign features to query.
    fn unoccupied(via_trace_mm: f64, via_via_mm: f64, via_hole_mm: f64) -> Self {
        Self {
            tags: Vec::new(),
            committed_via_tags: Vec::new(),
            hole_blocked: Vec::new(),
            hole_centres: Vec::new(),
            via_trace_mm,
            via_via_mm,
            via_hole_mm,
        }
    }

    /// A committed-ownership field. Legacy callers that omit physical via-via
    /// spacing keep an empty field and use the historical halo-ring guard.
    #[cfg(test)]
    fn dense(dims: Dims, via_trace_mm: f64, via_via_mm: f64) -> Self {
        Self::dense_with_planar_via_guard(dims, via_trace_mm, via_via_mm, 0.0, false)
    }

    /// Enable the committed-via-only field for physical-rule callers that need
    /// symmetric via↔trace enforcement. The compatibility constructor above keeps
    /// all established legacy routes byte-identical.
    fn dense_with_planar_via_guard(
        dims: Dims,
        via_trace_mm: f64,
        via_via_mm: f64,
        via_hole_mm: f64,
        protect_planar_from_vias: bool,
    ) -> Self {
        let tags = if dims.layers > 1 && via_via_mm > VIA_SPACING_EPS_MM {
            vec![VIA_TAG_FREE; dims.len()]
        } else {
            Vec::new()
        };
        let committed_via_tags =
            if protect_planar_from_vias && dims.layers > 1 && via_trace_mm > VIA_SPACING_EPS_MM {
                vec![VIA_TAG_FREE; dims.len()]
            } else {
                Vec::new()
            };
        let plane = (dims.w as usize).saturating_mul(dims.h as usize);
        let (hole_blocked, hole_centres) = if dims.layers > 1 && via_hole_mm > VIA_SPACING_EPS_MM {
            (vec![false; plane], vec![VIA_TAG_FREE; plane])
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            tags,
            committed_via_tags,
            hole_blocked,
            hole_centres,
            via_trace_mm,
            via_via_mm,
            via_hole_mm,
        }
    }

    #[inline]
    fn feature_aware(&self) -> bool {
        self.via_via_mm > VIA_SPACING_EPS_MM
    }

    #[inline]
    fn conflicts_cell(&self, cell: CellIdx, own_group: i64) -> bool {
        self.tags
            .get(cell as usize)
            .is_some_and(|&tag| via_tag_is_foreign(tag, own_group))
    }

    #[inline]
    fn conflicts_step(&self, u: CellIdx, v: CellIdx, own_group: i64) -> bool {
        self.conflicts_cell(u, own_group) || self.conflicts_cell(v, own_group)
    }

    #[inline]
    fn conflicts_planar_cell(&self, cell: CellIdx, own_group: i64) -> bool {
        self.committed_via_tags
            .get(cell as usize)
            .is_some_and(|&tag| via_tag_is_foreign(tag, own_group))
    }

    #[inline]
    fn conflicts_hole_cell(&self, dims: Dims, cell: CellIdx, own_group: i64) -> bool {
        if self.hole_blocked.is_empty() {
            return false;
        }
        let planar = (cell % (dims.w * dims.h)) as usize;
        self.hole_blocked[planar] || via_tag_is_foreign(self.hole_centres[planar], own_group)
    }

    fn clear(&mut self) {
        self.tags.fill(VIA_TAG_FREE);
        self.committed_via_tags.fill(VIA_TAG_FREE);
        self.hole_blocked.fill(false);
        self.hole_centres.fill(VIA_TAG_FREE);
    }

    /// Stamp the exact union of this path's feature dilations. Every path cell
    /// contributes the via-to-trace radius on its own layer; cells touched by a
    /// layer-changing step contribute the wider via-to-via radius instead. Each path
    /// cell is visited once, so a multi-step vertical run stamps its shared middle
    /// layer exactly once.
    fn stamp_path(&mut self, dims: Dims, coords: &GridCoords, path: &[CellIdx], group: i64) {
        if self.tags.is_empty()
            && self.committed_via_tags.is_empty()
            && self.hole_blocked.is_empty()
        {
            return;
        }

        for (i, &cell) in path.iter().enumerate() {
            let (cx, cy, layer) = dims.xyz(cell);
            let is_via_neighbour = |other: CellIdx| {
                let (ox, oy, other_layer) = dims.xyz(other);
                ox == cx && oy == cy && other_layer != layer
            };
            // Inspect each path cell once. A shared middle layer in a multi-step
            // vertical run has both a via predecessor and successor, but still
            // contributes only this single (wider) disk.
            let touches_via = (i > 0 && is_via_neighbour(path[i - 1]))
                || (i + 1 < path.len() && is_via_neighbour(path[i + 1]));
            if !self.tags.is_empty() {
                let radius = if touches_via {
                    self.via_trace_mm.max(self.via_via_mm)
                } else {
                    self.via_trace_mm
                };
                Self::stamp_disk(&mut self.tags, dims, coords, cell, group, radius);
            }
            if touches_via && !self.committed_via_tags.is_empty() {
                Self::stamp_disk(
                    &mut self.committed_via_tags,
                    dims,
                    coords,
                    cell,
                    group,
                    self.via_trace_mm,
                );
            }
        }
        self.stamp_holes_path(dims, coords, path, group);
    }

    fn stamp_holes_path(&mut self, dims: Dims, coords: &GridCoords, path: &[CellIdx], group: i64) {
        if self.hole_blocked.is_empty() {
            return;
        }
        let mut last_planar = None;
        for step in path.windows(2) {
            let (ax, ay, al) = dims.xyz(step[0]);
            let (bx, by, bl) = dims.xyz(step[1]);
            if ax == bx && ay == by && al != bl {
                let planar = ay * dims.w + ax;
                if last_planar != Some(planar) {
                    self.stamp_hole(dims, coords, step[0], group);
                    last_planar = Some(planar);
                }
            } else {
                last_planar = None;
            }
        }
    }

    /// Stamp one physical drill site independently of electrical net and layer
    /// span. The open outer disk is net-independent. Its <=0.005 mm coincidence
    /// core is group-aware so sibling records at one physical same-net site may be
    /// reused, matching the producer/checker convention.
    fn stamp_hole(&mut self, dims: Dims, coords: &GridCoords, center: CellIdx, group: i64) {
        let (cx, cy, _) = dims.xyz(center);
        let center_planar = (cy * dims.w + cx) as usize;
        let group = via_group_tag(group);

        let threshold = (self.via_hole_mm - VIA_SPACING_EPS_MM).max(0.0);
        let threshold_sq = threshold * threshold;
        let center_x = coords.x_of(cx);
        let center_y = coords.y_of(cy);
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, self.via_hole_mm);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, self.via_hole_mm);
        for ny in y0..y1 {
            let dy = coords.y_of(ny) - center_y;
            for nx in x0..x1 {
                let dx = coords.x_of(nx) - center_x;
                let distance_sq = dx * dx + dy * dy;
                if distance_sq >= threshold_sq {
                    continue;
                }
                let planar = (ny * dims.w + nx) as usize;
                if distance_sq <= VIA_SAME_LOCATION_MM * VIA_SAME_LOCATION_MM {
                    self.hole_centres[planar] = match self.hole_centres[planar] {
                        VIA_TAG_FREE => group,
                        existing if existing == group && group != VIA_TAG_MIXED => existing,
                        _ => VIA_TAG_MIXED,
                    };
                } else {
                    self.hole_blocked[planar] = true;
                }
            }
        }

        // A very small caller-provided spacing can make the clearance disk empty;
        // the physical centre still needs ownership so a foreign via cannot reuse
        // the exact site while a same-group stacked representation can.
        if self.hole_centres[center_planar] == VIA_TAG_FREE {
            self.hole_centres[center_planar] = group;
        }
    }

    /// Stamp one exact Euclidean open disk. The same epsilon and strict comparison
    /// as the native DRC keep a pair exactly at required spacing legal. Targets are
    /// not filtered by base obstacles: a via-passable terminal still must respect a
    /// nearby committed feature.
    fn stamp_disk(
        tags: &mut [i32],
        dims: Dims,
        coords: &GridCoords,
        center: CellIdx,
        group: i64,
        radius: f64,
    ) {
        if radius <= VIA_SPACING_EPS_MM {
            return;
        }
        let threshold = (radius - VIA_SPACING_EPS_MM).max(0.0);
        let threshold_sq = threshold * threshold;
        let (cx, cy, layer) = dims.xyz(center);
        let center_x = coords.x_of(cx);
        let center_y = coords.y_of(cy);
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, radius);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, radius);

        for ny in y0..y1 {
            if !geom_line_within(&coords.y_lines, dims.h, cy, ny, radius) {
                continue;
            }
            let dy = coords.y_of(ny) - center_y;
            for nx in x0..x1 {
                if !geom_line_within(&coords.x_lines, dims.w, cx, nx, radius) {
                    continue;
                }
                let dx = coords.x_of(nx) - center_x;
                if dx * dx + dy * dy >= threshold_sq {
                    continue;
                }
                let cell = dims.idx3(nx, ny, layer) as usize;
                let group = via_group_tag(group);
                tags[cell] = match tags[cell] {
                    VIA_TAG_FREE => group,
                    existing if existing == group && group != VIA_TAG_MIXED => existing,
                    _ => VIA_TAG_MIXED,
                };
            }
        }
    }
}

/// Inclusive planar search rectangle for one isolated-net request.
///
/// The same rectangle applies on every layer. A provider must first solve inside
/// this normal per-net window, then retry the full board only when that solve is
/// unreachable; see [`IsolatedRouteProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsolatedRouteWindow {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl IsolatedRouteWindow {
    /// The whole board.
    pub fn full(dims: Dims) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: dims.w.saturating_sub(1),
            y1: dims.h.saturating_sub(1),
        }
    }

    /// True when cell `c` lies inside the window.
    #[inline]
    fn contains(&self, dims: mr_core::Dims, c: CellIdx) -> bool {
        let (x, y) = dims.xy(c);
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// Borrowed, already-rounded static inputs for a batch of isolated-net solves.
///
/// `x_edge_costs[k]` and `y_edge_costs[k]` price the planar gaps `k <-> k+1`;
/// their lengths are `w-1` and `h-1`. `via_edge_costs[k]` prices adjacent layer
/// transition `k <-> k+1`, with `None` forbidding it, and has length `layers-1`.
/// Providers multiply these edge prices by the destination cell's enter weight,
/// applying the same own-pad obstacle override and `OBSTACLE - 1` step cap as the
/// CPU isolated search.
#[derive(Debug, Clone, Copy)]
pub struct IsolatedRouteRequest<'a> {
    pub grid: &'a Grid,
    pub nets: &'a [NetEndpoints],
    pub windows: &'a [IsolatedRouteWindow],
    pub x_edge_costs: &'a [Cost],
    pub y_edge_costs: &'a [Cost],
    pub via_edge_costs: &'a [Option<Cost>],
}

/// Dependency-inverted accelerator for NegotiatedRouter's independent
/// "route each net alone" batch.
///
/// This is a **trusted canonical-path contract**. For every input net, the provider
/// must return exactly the path selected by the CPU isolated search under labels
/// `(search cost, minimum hops, lower predecessor cell)`: solve in the supplied
/// normal window first, and retry on the full board only when the windowed solve is
/// unreachable. `None` means unreachable even after that retry. Results are aligned
/// with `request.nets` and the operation is all-or-error.
///
/// The router defensively checks batch shape and basic path legality, but those
/// checks cannot prove optimality, canonical tie-breaking, or that a full-board
/// path was used only after window failure. Any provider error or detectable invalid
/// entry discards the **entire** batch and transparently reruns every isolated search
/// on the CPU; no partial provider result is ever mixed with CPU results.
pub trait IsolatedRouteProvider {
    /// Solve the aligned isolated-net batch described by `request`.
    fn route_isolated_batch(
        &self,
        request: IsolatedRouteRequest<'_>,
    ) -> Result<Vec<Option<Vec<CellIdx>>>, RouterError>;
}

type Window = IsolatedRouteWindow;

/// Build the per-net search window: the bounding box of `{src, dst}` plus every
/// cell in `pads` (the net's own passable pads must be reachable), expanded by a
/// margin so the net can detour locally. `margin = max(16, ceil(0.30 * max(bbox_w,
/// bbox_h)))` cells, then clamped to the board.
fn net_window(dims: mr_core::Dims, src: CellIdx, dst: CellIdx, pads: &[CellIdx]) -> Window {
    let (sx, sy) = dims.xy(src);
    let (dx, dy) = dims.xy(dst);
    let mut x0 = sx.min(dx);
    let mut y0 = sy.min(dy);
    let mut x1 = sx.max(dx);
    let mut y1 = sy.max(dy);
    // The window must include all of the net's own pad cells.
    for &p in pads {
        let (px, py) = dims.xy(p);
        x0 = x0.min(px);
        y0 = y0.min(py);
        x1 = x1.max(px);
        y1 = y1.max(py);
    }
    let bbox_w = x1 - x0;
    let bbox_h = y1 - y0;
    let span = bbox_w.max(bbox_h);
    // ceil(0.30 * span) == (3*span + 9) / 10.
    let margin = 16u32.max((3 * span).div_ceil(10));
    Window {
        x0: x0.saturating_sub(margin),
        y0: y0.saturating_sub(margin),
        x1: (x1 + margin).min(dims.w.saturating_sub(1)),
        y1: (y1 + margin).min(dims.h.saturating_sub(1)),
    }
}

/// O(1)-membership set over board cells using the same generation-stamp trick as
/// [`crate::dijkstra::SearchBuf`]: a cell is a member iff `stamp[c] == gen`. Reset
/// is O(1) (bump `gen`); marking the current net's pads is O(pads). Replaces the
/// per-net `Vec::contains` / `HashSet` membership in the hot loop.
struct PadSet {
    stamp: Vec<u32>,
    gen: u32,
}

impl PadSet {
    fn new(n: usize) -> Self {
        PadSet {
            stamp: vec![0; n],
            gen: 0,
        }
    }

    /// Clear all membership in O(1) and load `pads` as the current set in
    /// O(pads). Handles `gen` wraparound by zeroing the stamps.
    fn load(&mut self, pads: &[CellIdx]) {
        match self.gen.checked_add(1) {
            Some(g) => self.gen = g,
            None => {
                self.stamp.iter_mut().for_each(|s| *s = 0);
                self.gen = 1;
            }
        }
        for &p in pads {
            self.stamp[p as usize] = self.gen;
        }
    }

    #[inline]
    fn contains(&self, c: CellIdx) -> bool {
        self.stamp[c as usize] == self.gen
    }
}

/// Reusable per-cell multiplicity map with O(1) logical reset.
///
/// A clearance footprint deliberately visits cells more than once when halos from
/// several path segments overlap. Jacobi negotiation must subtract the current
/// net's exact old multiplicity from the shared `present_halo` snapshot; a Boolean
/// membership set would leave residual self-cost. Generation stamps avoid an
/// O(n_cells) clear for every net search.
struct CountedCellSet {
    count: Vec<u32>,
    stamp: Vec<u32>,
    gen: u32,
}

/// Per-thread scratch for snapshot-based Jacobi negotiation. The eight full-grid
/// `u32` planes are lazily allocated only on threads that execute a search, then
/// reused across iterations and concurrent routes on the same Rayon worker.
struct JacobiScratch {
    cells: usize,
    buf: SearchBuf,
    pad_set: PadSet,
    own_path: PadSet,
    own_halo: CountedCellSet,
}

impl JacobiScratch {
    fn new(n_cells: usize) -> Self {
        Self {
            cells: n_cells,
            buf: SearchBuf::new(n_cells),
            pad_set: PadSet::new(n_cells),
            own_path: PadSet::new(n_cells),
            own_halo: CountedCellSet::new(n_cells),
        }
    }
}

thread_local! {
    /// At most one Jacobi scratch allocation per live execution thread. In
    /// particular, nested corpus `par_iter` routes do not each eagerly allocate a
    /// full inner pool: all concurrent routes collectively use at most one slot per
    /// Rayon worker, sized to the largest board that worker has processed.
    static JACOBI_SCRATCH: RefCell<Option<JacobiScratch>> = const { RefCell::new(None) };
}

fn with_jacobi_scratch<R>(
    n_cells: usize,
    on_allocate: impl FnOnce(),
    use_scratch: impl FnOnce(&mut JacobiScratch) -> R,
) -> R {
    JACOBI_SCRATCH.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_none_or(|scratch| scratch.cells < n_cells) {
            on_allocate();
            *slot = Some(JacobiScratch::new(n_cells));
        }
        use_scratch(slot.as_mut().expect("Jacobi scratch was initialized"))
    })
}

impl CountedCellSet {
    fn new(n: usize) -> Self {
        Self {
            count: vec![0; n],
            stamp: vec![0; n],
            gen: 0,
        }
    }

    /// Logically clear the map in O(1). On generation wrap, reset only the stamps;
    /// stale counts remain harmless until their cell is stamped again.
    fn clear(&mut self) {
        match self.gen.checked_add(1) {
            Some(g) => self.gen = g,
            None => {
                self.stamp.fill(0);
                self.gen = 1;
            }
        }
    }

    #[inline]
    fn increment(&mut self, c: CellIdx) {
        let ci = c as usize;
        if self.stamp[ci] == self.gen {
            self.count[ci] = self.count[ci].saturating_add(1);
        } else {
            self.stamp[ci] = self.gen;
            self.count[ci] = 1;
        }
    }

    #[inline]
    fn count(&self, c: CellIdx) -> u32 {
        let ci = c as usize;
        if self.stamp[ci] == self.gen {
            self.count[ci]
        } else {
            0
        }
    }
}

/// PathFinder-style negotiated-congestion router. Default multi-net backend.
///
/// `via_model` gates and prices layer changes. When `None` (the default), a
/// [`ViaModel::through_hole`] over the grid's layer count is synthesised at route
/// time, so a multi-layer board uses unrestricted through-hole vias unless the
/// caller supplies a restricted model via [`NegotiatedRouter::with_via_model`]. On
/// a single-layer board the model is inert (no via neighbours exist).
#[derive(Debug, Default, Clone)]
pub struct NegotiatedRouter {
    via_model: Option<ViaModel>,
    /// Continuous grid-line geometry used to price a planar step by its real length
    /// (`round(len * COST_SCALE)`) instead of a uniform unit hop. `None` (the
    /// default) means "no geometry supplied": the search falls back to
    /// [`GridCoords::uniform`] over the grid's `dims`, where every step has unit
    /// length and the base step is exactly [`SCALE`] — byte-identical to the
    /// pre-geometric router. On a non-uniform / Hanan grid the caller supplies the
    /// board's line arrays via [`NegotiatedRouter::with_coords`].
    coords: Option<GridCoords>,
    /// Geometric clearance distance (continuous units, e.g. mm) reserved around
    /// committed copper during legalization to keep *other* nets away — a net's
    /// committed track owns not just its path cells but a halo extending this far on
    /// each cell's own layer, measured over the [`coords`](NegotiatedRouter#structfield.coords)
    /// line arrays, so foreign nets must stay at least this far from it (minimum
    /// spacing). `0.0` = disabled, i.e. the original behaviour (only the path cells
    /// are owned). The halo never overwrites a cell already owned by another group
    /// and never claims a base-obstacle cell, so it cannot wall a foreign net off
    /// from its own pad.
    ///
    /// On a [`GridCoords::uniform`] grid (unit-spaced lines) a clearance of `n` is
    /// byte-identical to the former `n`-cell Chebyshev radius, so
    /// [`with_clearance_cells`](NegotiatedRouter::with_clearance_cells) (which sets
    /// this to `n as f64`) reproduces the pre-geometric tests exactly. On a
    /// non-uniform / Hanan grid the geometric distance is what keeps real
    /// copper-to-copper spacing (a cell count there spans a variable physical width).
    clearance_mm: f64,
    /// Minimum centre-to-centre distance between vias of different connection
    /// groups on any layer both vias touch. This is deliberately separate from
    /// [`ViaModel::keepout_mm`]: the latter is a via-to-track centreline distance,
    /// while two circular via pads require both pad radii plus copper clearance.
    /// `0.0` disables the dedicated via-via guard.
    via_spacing_mm: f64,
    /// Net-independent drill-centre spacing. Unlike annular copper spacing, this
    /// fabrication rule also applies within one electrical group and one path.
    via_hole_spacing_mm: f64,
    /// Whether committed vias also stamp a dedicated through-obstacle field queried
    /// by planar moves. Typed feature-pair callers enable this; the default keeps
    /// legacy routing byte-identical.
    committed_via_to_trace_guard: bool,
    #[cfg(test)]
    jacobi_scratch_probe: Option<std::sync::Arc<AtomicUsize>>,
}

/// Lightweight result of a negotiated route with the isolation diagnosis the
/// router already computes for legalization.
///
/// `alone_routable[i]` is aligned with the input `nets[i]` and is true exactly
/// when that net can route by itself on the base grid using this router's geometry
/// and via model. Unlike [`RouteTrace`], this carries no paths or per-iteration
/// snapshots, so callers can classify failed nets without rerunning searches or
/// paying visualisation-trace memory costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiatedOutcome {
    pub board: BoardRoute,
    pub alone_routable: Vec<bool>,
}

impl NegotiatedRouter {
    pub fn new() -> Self {
        Self {
            via_model: None,
            coords: None,
            clearance_mm: 0.0,
            via_spacing_mm: 0.0,
            via_hole_spacing_mm: 0.0,
            committed_via_to_trace_guard: false,
            #[cfg(test)]
            jacobi_scratch_probe: None,
        }
    }

    #[cfg(test)]
    fn with_jacobi_scratch_probe(mut self, probe: std::sync::Arc<AtomicUsize>) -> Self {
        self.jacobi_scratch_probe = Some(probe);
        self
    }

    /// Use an explicit [`ViaModel`] (e.g. a blind/buried stackup) instead of the
    /// default through-hole model. Builder-style; returns the configured router.
    pub fn with_via_model(mut self, vm: ViaModel) -> Self {
        self.via_model = Some(vm);
        self
    }

    /// Supply the board's continuous grid-line geometry so planar steps are priced
    /// by their real length (`round(len * COST_SCALE)`) rather than as uniform unit
    /// hops. Builder-style; returns the configured router. Without it the router
    /// uses [`GridCoords::uniform`] and is byte-identical to the pre-geometric
    /// behaviour. See the [`coords`](NegotiatedRouter#structfield.coords) field.
    pub fn with_coords(mut self, coords: GridCoords) -> Self {
        self.coords = Some(coords);
        self
    }

    /// Set the geometric clearance distance (continuous units, e.g. mm) reserved
    /// around committed copper, so distinct nets keep at least this much spacing,
    /// measured over the supplied [`coords`](NegotiatedRouter::with_coords) line
    /// arrays. Builder-style; returns the configured router. `0.0` (the default)
    /// disables the halo and reproduces the byte-identical pre-clearance behaviour.
    /// See the [`clearance_mm`](NegotiatedRouter#structfield.clearance_mm) field.
    pub fn with_clearance_mm(mut self, mm: f64) -> Self {
        self.clearance_mm = mm.max(0.0);
        self
    }

    /// Set the minimum geometric centre-to-centre spacing between foreign vias.
    ///
    /// Via-to-track clearance remains governed by [`ViaModel::keepout_mm`]. Keeping
    /// the two radii independent avoids globally widening the ordinary copper halo
    /// just to account for the second via pad radius. Non-finite and non-positive
    /// values disable this guard, preserving the clearance-off fast path.
    pub fn with_via_spacing_mm(mut self, mm: f64) -> Self {
        self.via_spacing_mm = if mm.is_finite() { mm.max(0.0) } else { 0.0 };
        self
    }

    /// Set the net-independent minimum drill-centre spacing. Exact same-group
    /// centre reuse remains one physical stacked via; every distinct centre obeys
    /// this distance, including two layer changes in a single candidate path.
    pub fn with_via_hole_spacing_mm(mut self, mm: f64) -> Self {
        self.via_hole_spacing_mm = if mm.is_finite() { mm.max(0.0) } else { 0.0 };
        self
    }

    /// Symmetrically enforce [`ViaModel::keepout_mm`] when a committed via is routed
    /// before a foreign planar trace. The dedicated field is stamped through static
    /// obstacle cells and queried before endpoint/pad exemptions. Disabled by
    /// default for compatibility; coherent typed physical-rule pipelines enable it.
    pub fn with_committed_via_to_trace_guard(mut self, enabled: bool) -> Self {
        self.committed_via_to_trace_guard = enabled;
        self
    }

    /// Back-compat shim for callers/tests that express clearance as a cell count on a
    /// uniform grid: a radius of `n` cells equals a geometric distance of `n` over
    /// [`GridCoords::uniform`]'s unit-spaced lines, so this is exactly
    /// [`with_clearance_mm`](NegotiatedRouter::with_clearance_mm)`(n as f64)` and
    /// stays byte-identical to the former Chebyshev halo on a uniform grid. On a
    /// non-uniform grid prefer `with_clearance_mm` with the real distance.
    pub fn with_clearance_cells(mut self, n: u32) -> Self {
        self.clearance_mm = n as f64;
        self
    }
}

/// The connection group of a net name: the prefix before the first `'#'`. Chained
/// sub-nets of one connection share a group and may legally share cells.
fn group_of(name: &str) -> &str {
    match name.find('#') {
        Some(i) => &name[..i],
        None => name,
    }
}

/// Dense deterministic connection groups used by legalization and route traces.
///
/// Name-related sub-nets share a group, as do separately named edges meeting at
/// the same endpoint cell. Keeping this construction in one place ensures both
/// routing variants apply identical electrical ownership semantics.
fn connection_group_ids(nets: &[NetEndpoints]) -> Vec<usize> {
    let n_nets = nets.len();
    let mut group_ids = vec![0; n_nets];
    let mut parent: Vec<usize> = (0..n_nets).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut child = x;
        while parent[child] != root {
            let next = parent[child];
            parent[child] = root;
            child = next;
        }
        root
    }

    let union = |parent: &mut [usize], a: usize, b: usize| {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            // Union toward the lower index for deterministic roots.
            if ra < rb {
                parent[rb] = ra;
            } else {
                parent[ra] = rb;
            }
        }
    };

    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (i, net) in nets.iter().enumerate() {
        match by_name.get(group_of(&net.net)) {
            Some(&j) => union(&mut parent, i, j),
            None => {
                by_name.insert(group_of(&net.net), i);
            }
        }
    }

    let mut by_cell: HashMap<CellIdx, usize> = HashMap::new();
    for (i, net) in nets.iter().enumerate() {
        for &cell in &[net.src, net.dst] {
            match by_cell.get(&cell) {
                Some(&j) => union(&mut parent, i, j),
                None => {
                    by_cell.insert(cell, i);
                }
            }
        }
    }

    // Intern roots by first appearance, yielding dense stable ids.
    let mut root_to_group: HashMap<usize, usize> = HashMap::new();
    for (i, group_id) in group_ids.iter_mut().enumerate() {
        let root = find(&mut parent, i);
        let next = root_to_group.len();
        *group_id = *root_to_group.entry(root).or_insert(next);
    }
    group_ids
}

/// Whether the input explicitly contains multiple routed segments of one named
/// connection. Shared endpoint cells also merge electrical groups during
/// legalization, but are deliberately not a portfolio trigger: unrelated fixtures
/// can quantize endpoints onto the same coarse cell, and treating that coincidence
/// as a retry signal would broaden the bounded cohort substantially.
fn has_named_subnet_group(nets: &[NetEndpoints]) -> bool {
    let mut seen = HashMap::new();
    nets.iter()
        .any(|net| seen.insert(group_of(&net.net), ()).is_some())
}

fn should_try_serial_candidate(
    dims: Dims,
    n_nets: usize,
    has_named_group: bool,
    primary: &BoardRoute,
) -> bool {
    debug_assert!(primary.results.len() <= n_nets);
    (PORTFOLIO_MIN_NETS..=PORTFOLIO_MAX_NETS).contains(&n_nets)
        && has_named_group
        && dims.len() <= PORTFOLIO_CELL_CAP
}

fn should_try_diversified_orders(
    n_groups: usize,
    n_nets: usize,
    n_cells: usize,
    original_best_routed: usize,
    alone_routable: usize,
) -> bool {
    (ORDER_PORTFOLIO_MIN_GROUPS..=ORDER_PORTFOLIO_MAX_GROUPS).contains(&n_groups)
        && n_nets <= ORDER_PORTFOLIO_MAX_NETS
        && n_cells <= ORDER_PORTFOLIO_CELL_CAP
        && original_best_routed < alone_routable
}

/// Four deterministic samples from the cyclic/dihedral order family. They change
/// both the first claimant and traversal direction without paying for every rotation.
/// Orders already covered by the primary portfolio, or duplicated within this
/// fallback, are omitted while preserving proposal order.
fn diversified_fallback_orders(
    base_order: &[usize],
    primary_orders: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    if base_order.len() < 2 {
        return Vec::new();
    }

    let mut left = base_order.to_vec();
    left.rotate_left(1);
    let mut right = base_order.to_vec();
    right.rotate_right(1);
    let mut reverse = base_order.to_vec();
    reverse.reverse();
    let mut opposite_reverse = reverse.clone();
    opposite_reverse.rotate_left(base_order.len() / 2);

    let mut fallback = Vec::with_capacity(4);
    for order in [left, right, reverse, opposite_reverse] {
        if !primary_orders.contains(&order) && !fallback.contains(&order) {
            fallback.push(order);
        }
    }
    fallback
}

/// More routed nets wins; for equal completion, lower caller-grid cost wins. An
/// exact tie deliberately keeps the primary candidate to minimize route churn.
fn serial_candidate_is_better(primary: &BoardRoute, serial: &BoardRoute) -> bool {
    serial.results.len() > primary.results.len()
        || (serial.results.len() == primary.results.len()
            && serial.total_cost() < primary.total_cost())
}

fn empty_route_trace(dims: Dims) -> RouteTrace {
    RouteTrace {
        dims,
        nets: Vec::new(),
        n_groups: 0,
        iterations: Vec::new(),
        legalization: None,
    }
}

/// Cell-hop length used only as an internal deterministic legalization/rip-up
/// difficulty metric. Public `RouteResult::cost` is computed by [`grid_path_cost`]
/// and honors the caller's weighted grid.
fn unit_cost(path: &[CellIdx]) -> Cost {
    path.len().saturating_sub(1) as Cost
}

/// Contract cost of a committed path on the caller's grid.  Ordinary passable
/// cells retain their configured weight; an obstacle explicitly owned by this
/// net through `passable_pads` is unmasked at the canonical free-cell cost `1`.
fn grid_path_cost(grid: &Grid, net: &NetEndpoints, path: &[CellIdx]) -> Cost {
    path.iter().skip(1).fold(0, |acc, &c| {
        let step = if grid.is_obstacle(c) && net.passable_pads.contains(&c) {
            1
        } else {
            grid.cost_at(c)
        };
        acc.saturating_add(step)
    })
}

/// Sum the public per-net grid costs for one legalization candidate without
/// collapsing distinct large candidates at the [`Cost`] ceiling.
fn committed_grid_cost(grid: &Grid, nets: &[NetEndpoints], committed: &Committed) -> u64 {
    committed
        .iter()
        .enumerate()
        .filter_map(|(i, path)| {
            path.as_ref()
                .map(|path| grid_path_cost(grid, &nets[i], path) as u64)
        })
        .fold(0u64, u64::saturating_add)
}

#[inline]
fn legalization_candidate_is_better(
    routed: usize,
    total_cost: u64,
    order: &[usize],
    best_routed: usize,
    best_cost: u64,
    best_order: &[usize],
) -> bool {
    routed > best_routed
        || (routed == best_routed && total_cost < best_cost)
        || (routed == best_routed && total_cost == best_cost && order < best_order)
}

/// Diversified fallback orders form a completion-only portfolio: they may replace
/// the original-order winner only when they route strictly more nets. Equal-routed
/// cost wins stay with the original portfolio because cost is not DRC-aligned.
#[inline]
fn diversified_candidate_is_better(
    routed: usize,
    total_cost: u64,
    order: &[usize],
    original_best_routed: usize,
    best_routed: usize,
    best_cost: u64,
    best_order: &[usize],
) -> bool {
    routed > original_best_routed
        && legalization_candidate_is_better(
            routed,
            total_cost,
            order,
            best_routed,
            best_cost,
            best_order,
        )
}

/// Convert a computed passable step price into the `u32` search domain without
/// producing [`OBSTACLE`], which `astar_buf` reserves as its unreachable sentinel.
#[inline]
fn passable_search_cost(cost: u64) -> Cost {
    cost.min((OBSTACLE - 1) as u64) as Cost
}

/// Prefix sums of the exact per-gap fixed-point costs used by the A* heuristic.
///
/// A route can evaluate its heuristic millions of times. Walking every intervening
/// grid-line gap for each evaluation makes a Hanan-grid heuristic O(axis distance)
/// per heap operation. These prefixes preserve that exact sum in O(1). `u64` is
/// sufficient for the unsaturated sum: an axis has at most `u32::MAX - 1` gaps and
/// every gap costs at most `u32::MAX`, whose product is below `u64::MAX`.
struct ManhattanCosts {
    x_prefix: Vec<u64>,
    y_prefix: Vec<u64>,
}

impl ManhattanCosts {
    fn new(dims: Dims, coords: &GridCoords) -> Self {
        Self {
            x_prefix: axis_cost_prefix(&coords.x_lines, dims.w),
            y_prefix: axis_cost_prefix(&coords.y_lines, dims.h),
        }
    }
}

#[inline]
fn axis_gap_cost(lines: &[f64], k: u32) -> Cost {
    let a = lines.get(k as usize).copied().unwrap_or(k as f64);
    let b = lines.get(k as usize + 1).copied().unwrap_or((k + 1) as f64);
    edge_cost((b - a).abs())
}

/// Lower bound on the planar A* cost between two cells in fixed-point [`Cost`] units,
/// plus the layer distance priced at the cheapest legal via step (`min_via_cost`).
///
/// # Admissibility (why a *sum of per-gap* `edge_cost`, not one `edge_cost` of the sum)
///
/// The search pays a planar step `u -> v` a base of `edge_cost(gap_uv) =
/// round(COST_SCALE * gap_uv)` (plus non-negative congestion). So the planar base paid
/// along ANY real path from `a` to `b` is `Σ_steps round(COST_SCALE * g_i)` over its
/// per-step line gaps `g_i >= 0`. A heuristic is admissible iff it never exceeds that
/// minimum.
///
/// The previous form `edge_cost(manhattan_len(a, b)) = round(COST_SCALE * Σ_axis Δ)`
/// rounds the AGGREGATE length once. Because `round(Σ x_i)` can exceed `Σ round(x_i)`
/// (round-of-sum > sum-of-rounds — e.g. `0.5/16` twice rounds per gap to `0+0=0`
/// while their sum `1/16` rounds to `1`), the old heuristic could OVERESTIMATE the
/// per-step total the search actually pays on non-integer line spacings, making A*
/// inadmissible and letting `astar_buf`'s "break on pop dst" return a non-optimal path.
///
/// The fix sums `edge_cost` over EACH grid-line gap along the two straight axis legs
/// from `a` to `b`: `Σ_{x lines in [ax,bx)} edge_cost(gap) + Σ_{y lines in [ay,by)}
/// edge_cost(gap)`. This is EXACTLY the planar base the search pays on a monotone
/// (staircase) path between the cells, and any non-monotone path covers a superset of
/// gaps (it must retrace), so it can only cost more. Hence this sum is `<=` the planar
/// base of every real path — a guaranteed lower bound, in the same fixed-point units
/// as the edge cost (the admissibility requirement for A* optimality). The via term is
/// already admissible (`>= min_via_cost` per layer change) and is preserved as-is.
///
/// On a uniform grid every gap is `1.0`, `edge_cost(1.0) == COST_SCALE`, and the sum is
/// `COST_SCALE * (dx + dy)` — byte-identical to the historical `(dx + dy) * SCALE` and
/// to the old `edge_cost(manhattan_len)` there (the round-of-sum / sum-of-rounds gap
/// only opens on non-integer spacings). The layer term is always 0 on a single-layer
/// grid.
fn manhattan_scaled(
    dims: Dims,
    costs: &ManhattanCosts,
    a: CellIdx,
    b: CellIdx,
    min_via_cost: Cost,
) -> Cost {
    let (ax, ay) = dims.xy(a);
    let (bx, by) = dims.xy(b);
    let planar = axis_leg_cost(&costs.x_prefix, ax, bx).saturating_add(axis_leg_cost(
        &costs.y_prefix,
        ay,
        by,
    ));
    let dl = dims.layer_of(a).abs_diff(dims.layer_of(b));
    planar.saturating_add(dl.saturating_mul(min_via_cost))
}

/// Build exact unsaturated prefix sums of per-gap `edge_cost`s for one axis.
///
/// Short/mismatched coordinate arrays deliberately use the same index-as-position
/// fallback as [`GridCoords::x_of`], so malformed geometry retains the previous
/// robust unit-spacing behavior rather than panicking.
fn axis_cost_prefix(lines: &[f64], count: u32) -> Vec<u64> {
    let mut prefix = Vec::with_capacity((count as usize).max(1));
    prefix.push(0);
    for k in 0..count.saturating_sub(1) {
        let next = prefix[prefix.len() - 1] + axis_gap_cost(lines, k) as u64;
        prefix.push(next);
    }
    prefix
}

fn axis_edge_costs(lines: &[f64], count: u32) -> Vec<Cost> {
    (0..count.saturating_sub(1))
        .map(|k| axis_gap_cost(lines, k))
        .collect()
}

/// Exact per-gap interval sum from a precomputed prefix, with the same `Cost`
/// saturation as repeated [`Cost::saturating_add`]. Callers pass in-bounds cell
/// coordinates, so both prefix indices exist even for short coordinate arrays.
#[inline]
fn axis_leg_cost(prefix: &[u64], i: u32, j: u32) -> Cost {
    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
    let total = prefix[hi as usize] - prefix[lo as usize];
    total.min(Cost::MAX as u64) as Cost
}

fn provider_path_is_valid(
    grid: &Grid,
    net: &NetEndpoints,
    via_model: &ViaModel,
    path: &[CellIdx],
) -> bool {
    let dims = grid.dims;
    if path.is_empty()
        || path.len() > dims.len()
        || path.first() != Some(&net.src)
        || path.last() != Some(&net.dst)
    {
        return false;
    }

    // Accelerator output is untrusted at this boundary. Large rasterized pads can
    // contain many thousands of cells, so validating every obstacle step with a
    // linear `Vec::contains` would make this defense-in-depth pass quadratic.
    let passable_pads: HashSet<CellIdx> = net.passable_pads.iter().copied().collect();
    let via_passable_pads: HashSet<CellIdx> = net.via_passable_pads.iter().copied().collect();
    let mut seen = HashSet::with_capacity(path.len());
    for &cell in path {
        if !dims.contains(cell)
            || grid.is_board_forbidden(cell)
            || (grid.is_obstacle(cell) && !passable_pads.contains(&cell))
            || !seen.insert(cell)
        {
            return false;
        }
    }

    path.windows(2).all(|step| {
        let (ax, ay, al) = dims.xyz(step[0]);
        let (bx, by, bl) = dims.xyz(step[1]);
        if al == bl {
            ((ax.abs_diff(bx) == 1 && ay == by) || (ay.abs_diff(by) == 1 && ax == bx))
                && !grid.is_board_planar_step_forbidden(step[0], step[1])
        } else {
            ax == bx
                && ay == by
                && via_model.is_step_legal(al, bl)
                && !grid.is_board_via_forbidden(step[0])
                && !grid.is_board_via_forbidden(step[1])
                && (!grid.is_via_forbidden(step[0]) || via_passable_pads.contains(&step[0]))
                && (!grid.is_via_forbidden(step[1]) || via_passable_pads.contains(&step[1]))
        }
    })
}

/// Validate the public pad lists without imposing an ordering contract on serde or
/// manually constructed inputs. Via exemptions must be an in-bounds subset of the
/// ordinary own-pad cells, so they can never bypass base obstacle ownership.
fn net_pad_lists_are_valid(dims: Dims, net: &NetEndpoints) -> bool {
    if net.passable_pads.iter().any(|&cell| !dims.contains(cell))
        || net
            .via_passable_pads
            .iter()
            .any(|&cell| !dims.contains(cell))
    {
        return false;
    }
    if net.via_passable_pads.is_empty() {
        return true;
    }
    let passable: HashSet<CellIdx> = net.passable_pads.iter().copied().collect();
    net.via_passable_pads
        .iter()
        .all(|cell| passable.contains(cell))
}

/// Normalize public/serde via-exemption lists once per top-level route. Rasterized
/// SRJ inputs are already canonical and stay borrowed; unsorted or duplicate lists
/// from other callers are cloned, sorted, and deduplicated without changing the
/// public [`NetEndpoints`] ordering contract.
fn normalize_via_exemptions(nets: &[NetEndpoints]) -> Cow<'_, [NetEndpoints]> {
    let canonical = nets.iter().all(|net| {
        net.via_passable_pads
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    });
    if canonical {
        return Cow::Borrowed(nets);
    }

    let mut normalized = nets.to_vec();
    for net in &mut normalized {
        net.via_passable_pads.sort_unstable();
        net.via_passable_pads.dedup();
    }
    Cow::Owned(normalized)
}

/// Membership in a normalized via-exemption list. Kept generic so the test oracle
/// can count comparisons while exercising the same standard-library binary search.
#[inline]
fn sorted_contains<T: Ord>(sorted: &[T], needle: &T) -> bool {
    sorted.binary_search(needle).is_ok()
}

/// Invoke one trusted provider batch and defensively validate its shape and paths.
/// `None` means either no usable provider result or invalid routing input; the normal
/// route variant then performs its existing validation and CPU isolated searches.
fn provider_alone_paths(
    router: &NegotiatedRouter,
    grid: &Grid,
    nets: &[NetEndpoints],
    provider: &dyn IsolatedRouteProvider,
) -> Option<Vec<Vec<CellIdx>>> {
    if !grid.is_well_formed() {
        return None;
    }
    let dims = grid.dims;
    for net in nets {
        if !net_pad_lists_are_valid(dims, net)
            || !dims.contains(net.src)
            || !dims.contains(net.dst)
            || grid.is_board_forbidden(net.src)
            || grid.is_board_forbidden(net.dst)
            || (grid.is_obstacle(net.src) && !net.passable_pads.contains(&net.src))
            || (grid.is_obstacle(net.dst) && !net.passable_pads.contains(&net.dst))
        {
            return None;
        }
    }

    let via_model = router
        .via_model
        .clone()
        .unwrap_or_else(|| ViaModel::through_hole(dims.layers));
    let coords = router
        .coords
        .clone()
        .unwrap_or_else(|| GridCoords::uniform(dims));
    let windows: Vec<IsolatedRouteWindow> = nets
        .iter()
        .map(|net| net_window(dims, net.src, net.dst, &net.passable_pads))
        .collect();
    let x_edge_costs = axis_edge_costs(&coords.x_lines, dims.w);
    let y_edge_costs = axis_edge_costs(&coords.y_lines, dims.h);
    let via_edge_costs: Vec<Option<Cost>> = (0..dims.layers.saturating_sub(1))
        .map(|layer| {
            via_model
                .is_step_legal(layer, layer + 1)
                .then_some(via_model.step_cost)
        })
        .collect();

    let provided = provider
        .route_isolated_batch(IsolatedRouteRequest {
            grid,
            nets,
            windows: &windows,
            x_edge_costs: &x_edge_costs,
            y_edge_costs: &y_edge_costs,
            via_edge_costs: &via_edge_costs,
        })
        .ok()?;
    if provided.len() != nets.len() {
        return None;
    }

    let mut paths = Vec::with_capacity(nets.len());
    for (net, path) in nets.iter().zip(provided) {
        match path {
            Some(path) if provider_path_is_valid(grid, net, &via_model, &path) => paths.push(path),
            Some(_) => return None,
            None => paths.push(Vec::new()),
        }
    }
    Some(paths)
}

impl NegotiatedRouter {
    /// Route one deterministic negotiation variant, optionally recording a
    /// [`RouteTrace`] for the visualiser.
    ///
    /// Every capture point below only *reads* loop state (it never feeds a value
    /// back into the search), and all captures run on the main thread between rayon
    /// sections, so traced and untraced executions of this variant are identical.
    fn route_variant(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
        group_ids: &[usize],
        negotiation_mode: NegotiationMode,
        provided_alone: Option<&[Vec<CellIdx>]>,
        mut recorder: Option<&mut RouteTrace>,
    ) -> Result<NegotiatedOutcome, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        for net in nets {
            if !net_pad_lists_are_valid(grid.dims, net) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            // An endpoint is invalid only if out of bounds, or it sits on an
            // obstacle that is NOT one of this net's own (passable) pad cells.
            let endpoint_invalid = |c: CellIdx| {
                !grid.dims.contains(c)
                    || grid.is_board_forbidden(c)
                    || (grid.is_obstacle(c) && !net.passable_pads.contains(&c))
            };
            if endpoint_invalid(net.src) || endpoint_invalid(net.dst) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
        }

        let dims = grid.dims;
        let n_cells = dims.len();
        let n_nets = nets.len();
        // Heuristic admissibility only needs to know whether a zero enter-cost
        // exists. Compute this once per board; routing can invoke A* thousands of
        // times, so scanning `grid.cost` inside each search is prohibitive.
        let has_zero_cost = grid.cost.contains(&0);

        // Effective via model: the caller's, or a through-hole model over the grid's
        // layer count. On a single-layer grid it is inert (no via neighbours exist),
        // so the search stays byte-identical to the planar router.
        let via_model = self
            .via_model
            .clone()
            .unwrap_or_else(|| ViaModel::through_hole(dims.layers));

        // Effective grid geometry: the caller's line arrays, or a uniform unit grid
        // over `dims`. The uniform fallback prices every planar step at exactly
        // `SCALE`, so a router given no coords is byte-identical to the pre-geometric
        // behaviour. Referenced by `&` in every per-net search closure below.
        let coords = self
            .coords
            .clone()
            .unwrap_or_else(|| GridCoords::uniform(dims));
        // Exact per-axis fixed-point prefix sums make every A* heuristic lookup
        // O(1), while preserving the former sum-of-rounded-gaps value exactly.
        let heuristic_costs = ManhattanCosts::new(dims, &coords);

        // Reusable search workspace and own-pad membership set, sized once to the
        // board and reused across every per-net search (no per-net O(n) work).
        let mut buf = SearchBuf::new(n_cells);
        let mut pad_set = PadSet::new(n_cells);

        debug_assert_eq!(group_ids.len(), n_nets);

        // TRACE: static board + per-net metadata, now that grouping is known. The
        // per-net `alone_path` is filled later (during legalization). No-op for the
        // production route (`recorder` is `None`).
        if let Some(rec) = &mut recorder {
            rec.dims = dims;
            rec.n_groups = group_ids.iter().map(|&g| g + 1).max().unwrap_or(0);
            rec.nets = nets
                .iter()
                .zip(group_ids.iter())
                .map(|(net, &group)| TracedNet {
                    net: net.net.clone(),
                    src: net.src,
                    dst: net.dst,
                    group: group as u32,
                    alone_path: Vec::new(),
                })
                .collect();
        }

        // Persistent congestion state.
        let mut history: Vec<u32> = vec![0; n_cells];
        let mut present: Vec<u32> = vec![0; n_cells];
        // Per-cell count of all currently routed nets' clearance footprints (planar
        // clearance halo + via keepout) — the negotiation analog of TritonRoute's
        // `objCost`. Maintained EXACTLY parallel to `present`: when a net's path is
        // removed its halo footprint is removed, and when the new path is added its
        // footprint is added here (both via `for_each_halo_cell`). Sequential search
        // removes its old footprint in place; parallel Jacobi search subtracts its
        // exact old multiplicity through a per-worker [`CountedCellSet`]. Thus net i
        // prices only every OTHER net's clearance footprint. When
        // `clearance_cells == 0 && via_model.keepout_mm == 0.0` the footprint is empty so
        // this stays all-zero and contributes nothing (byte-identical default).
        let mut present_halo: Vec<u32> = vec![0; n_cells];
        // Whether the clearance mechanism is active at all. Drives both the
        // `present_halo` pricing and the incremental-skip gating below.
        let clearance_active = self.clearance_mm > 0.0
            || via_model.keepout_mm > 0.0
            || self.via_spacing_mm > 0.0
            || self.via_hole_spacing_mm > 0.0;
        // Current routed path per net (empty == not currently routed).
        let mut paths: Vec<Vec<CellIdx>> = vec![Vec::new(); n_nets];

        // Per-net search window (bbox of endpoints+pads, expanded by a margin),
        // precomputed once: the negotiation search is restricted to this box so
        // explored area scales with the net's local region, not the whole board.
        let windows: Vec<Window> = nets
            .iter()
            .map(|net| net_window(dims, net.src, net.dst, &net.passable_pads))
            .collect();

        // Incremental rerouting (large boards only): after the first iteration,
        // only a net that is unrouted OR whose path touches a cell that was
        // over-used last iteration needs rerouting — a "happy" net's priced costs
        // are unchanged (history only grows on over-used cells), so its optimal
        // path is unchanged. This cuts later iterations from O(all nets) to
        // O(congested nets). Gated to large net counts so the small deterministic
        // unit tests keep their exact full-reroute behaviour.
        //
        // Clearance-active searches are deliberately not incrementally skipped:
        // moving a neighbour changes `present_halo` even when direct copper overuse
        // did not change, so every net is reconsidered in the exact sequential path.
        let incremental = n_nets > 8 && !clearance_active;
        // Large boards use snapshot-based Jacobi negotiation. Each worker subtracts
        // both the net's own copper and its exact counted halo multiplicity, so the
        // snapshot prices the same foreign occupancy as the serial path. Small boards
        // retain Gauss-Seidel to avoid parallel overhead.
        let use_parallel = negotiation_mode == NegotiationMode::Adaptive
            && n_nets > PARALLEL_NEGOTIATION_THRESHOLD;
        // `overused` from the previous iteration (cell -> was it over-used). Empty
        // before the first iteration (everything reroutes).
        let mut prev_overused: Vec<bool> = vec![false; n_cells];
        let mut prev_overused_cells: Vec<CellIdx> = Vec::new();
        // Per-iteration overuse scratch, allocated once and cleared incrementally
        // (via the touched-cell lists) so no iteration pays an O(all cells) memset.
        let mut first_group: Vec<i64> = vec![-1; n_cells];
        let mut overused: Vec<bool> = vec![false; n_cells];
        // Lazily allocated on the first sufficiently large Jacobi batch, then
        // cleared and refilled so every later iteration reuses the same capacity.
        let mut shared_congestion: Vec<u64> = Vec::new();

        for iter in 0..MAX_ITERS {
            let pfac: u32 = 1 + iter;

            if use_parallel {
                // ---- Snapshot-based (Jacobi-style) PARALLEL negotiation ----
                //
                // Every selected net reroutes against a READ-ONLY snapshot of
                // `present`/`present_halo`/`history` (occupancy from the END of the
                // previous iteration); clean nets keep their path and their existing
                // contribution to the occupancy maps. Dirty nets are routed in parallel
                // with rayon and MERGED back SEQUENTIALLY in net-index order, so the
                // outcome is independent of thread scheduling (deterministic). This is a
                // standard PathFinder variant (Jacobi vs Gauss-Seidel) and may converge
                // to a different-but-equivalent route than the sequential path. This
                // branch runs for the clearance path and for large non-clearance boards
                // (`n_nets > PARALLEL_NEGOTIATION_THRESHOLD`); small non-clearance boards
                // stay on the sequential path, so their byte-identical output is unchanged.
                //
                // CLEARANCE-ACTIVE: route every net every iteration. A moved net can
                // lower halo cost away from another net's current path, so an on-path
                // dirty test cannot prove that the latter's optimum is unchanged.
                // CLEARANCE-FREE: retain the proven copper-overuse incremental set.
                let dirty: Vec<usize> = if clearance_active {
                    (0..n_nets).collect()
                } else {
                    (0..n_nets)
                        .filter(|&i| {
                            iter == 0
                                || paths[i].is_empty()
                                || paths[i].iter().any(|&c| prev_overused[c as usize])
                        })
                        .collect()
                };
                let clearance_mm = self.clearance_mm;
                // Borrow the snapshots immutably for the duration of the parallel map.
                // Each executing thread lazily acquires its OWN reusable `SearchBuf`,
                // pad/path stamps, and counted halo stamp from thread-local storage.
                let present_snap: &[u32] = &present;
                let halo_snap: &[u32] = &present_halo;
                let history_snap: &[u32] = &history;
                // Every net in a large Jacobi batch sees the same immutable
                // congestion snapshot. Fold only those shared congestion terms
                // once; history remains a distinct addend in `route_negotiated`,
                // so defensive self-subtraction can never consume it.
                let fuse_jacobi_pricing = dirty.len() >= FUSED_JACOBI_MIN_DIRTY_NETS;
                if fuse_jacobi_pricing {
                    let present_factor = (pfac as u64) * (SCALE as u64);
                    let halo_factor = (pfac as u64) * (CLEARANCE_NEG_WEIGHT as u64);
                    shared_congestion.clear();
                    shared_congestion.extend(present_snap.iter().zip(halo_snap).map(
                        |(&present, &halo)| {
                            (present_factor * present as u64)
                                .saturating_add(halo_factor * halo as u64)
                        },
                    ));
                }
                let shared_congestion_ref =
                    fuse_jacobi_pricing.then_some(shared_congestion.as_slice());
                let nets_ref = nets;
                let windows_ref = &windows;
                let via_ref = &via_model;
                let coords_ref = &coords;
                let mut routed_paths: Vec<(usize, Option<Vec<CellIdx>>)> = dirty
                    .par_iter()
                    .map(|&i| {
                        with_jacobi_scratch(
                            n_cells,
                            || {
                                #[cfg(test)]
                                if let Some(probe) = &self.jacobi_scratch_probe {
                                    probe.fetch_add(1, Ordering::Relaxed);
                                }
                            },
                            |scratch| {
                                let JacobiScratch {
                                    buf,
                                    pad_set,
                                    own_path,
                                    own_halo,
                                    ..
                                } = scratch;
                                let net = &nets_ref[i];
                                pad_set.load(&net.passable_pads);
                                if !fuse_jacobi_pricing {
                                    own_path.load(&paths[i]);
                                }
                                own_halo.clear();
                                for_each_halo_cell(
                                    dims,
                                    coords_ref,
                                    grid,
                                    &paths[i],
                                    clearance_mm,
                                    via_ref,
                                    |c| own_halo.increment(c),
                                );
                                if fuse_jacobi_pricing {
                                    // Copper and halo self-pricing have the same SCALE
                                    // factor (compile-time asserted above). Count both
                                    // in one stamp only for an amortizing fused batch.
                                    for &c in &paths[i] {
                                        own_halo.increment(c);
                                    }
                                }
                                let own_path_ref = (!fuse_jacobi_pricing).then_some(&*own_path);
                                // Route within the net's window; on failure, retry once on
                                // the full board so the occasional global net still completes.
                                // Pure read-only search over the snapshots.
                                let routed = route_negotiated(
                                    buf,
                                    grid,
                                    coords_ref,
                                    &heuristic_costs,
                                    pad_set,
                                    &net.via_passable_pads,
                                    own_path_ref,
                                    Some(own_halo),
                                    shared_congestion_ref,
                                    present_snap,
                                    halo_snap,
                                    history_snap,
                                    pfac,
                                    net.src,
                                    net.dst,
                                    windows_ref[i],
                                    via_ref,
                                    has_zero_cost,
                                )
                                .or_else(|| {
                                    route_negotiated(
                                        buf,
                                        grid,
                                        coords_ref,
                                        &heuristic_costs,
                                        pad_set,
                                        &net.via_passable_pads,
                                        own_path_ref,
                                        Some(own_halo),
                                        shared_congestion_ref,
                                        present_snap,
                                        halo_snap,
                                        history_snap,
                                        pfac,
                                        net.src,
                                        net.dst,
                                        Window::full(dims),
                                        via_ref,
                                        has_zero_cost,
                                    )
                                });
                                (i, routed.map(|(p, _)| p))
                            },
                        )
                    })
                    .collect();
                // Deterministic INCREMENTAL merge: process the rerouted (dirty) nets in
                // net-index order, each subtracting its OLD path/halo and adding its NEW
                // path/halo from the shared maps (clean nets are left untouched). Counts
                // are commutative so the result is identical regardless of scheduling and
                // matches a full rebuild. Clearance-active routing deliberately keeps
                // every net dirty, so no lossy on-path halo-dirty approximation is used.
                routed_paths.sort_unstable_by_key(|(i, _)| *i);
                for (i, path) in routed_paths {
                    // Remove the old path's copper + halo (the net's prior contribution).
                    for &c in &paths[i] {
                        present[c as usize] = present[c as usize].saturating_sub(1);
                    }
                    for_each_halo_cell(
                        dims,
                        &coords,
                        grid,
                        &paths[i],
                        clearance_mm,
                        &via_model,
                        |c| {
                            present_halo[c as usize] = present_halo[c as usize].saturating_sub(1);
                        },
                    );
                    match path {
                        Some(path) => {
                            for &c in &path {
                                present[c as usize] = present[c as usize].saturating_add(1);
                            }
                            for_each_halo_cell(
                                dims,
                                &coords,
                                grid,
                                &path,
                                clearance_mm,
                                &via_model,
                                |c| {
                                    present_halo[c as usize] =
                                        present_halo[c as usize].saturating_add(1);
                                },
                            );
                            paths[i] = path;
                        }
                        None => {
                            // Unrouted this iteration: contributes nothing to occupancy.
                            paths[i].clear();
                        }
                    }
                }
            } else {
                // ---- Sequential negotiation (small boards) ----
                // Clearance-free boards may incrementally skip quiescent nets; with
                // clearance active `incremental` is false and every net reroutes after
                // its exact old copper + halo footprint is removed in place.
                for i in 0..n_nets {
                    let net = &nets[i];

                    // Skip quiescent nets after the first iteration: a routed net whose
                    // path avoids every previously-over-used cell keeps its path.
                    if incremental && iter > 0 && !paths[i].is_empty() {
                        let touches_overuse = paths[i].iter().any(|&c| prev_overused[c as usize]);
                        if !touches_overuse {
                            continue;
                        }
                    }

                    // Remove this net's old path from `present` before pricing, and its
                    // clearance footprint from `present_halo` in lockstep (saturating), so
                    // both maps exclude net i during its own search. `for_each_halo_cell`
                    // is a no-op when clearance is inactive, so `present_halo` stays 0.
                    for &c in &paths[i] {
                        present[c as usize] = present[c as usize].saturating_sub(1);
                    }
                    if clearance_active {
                        for_each_halo_cell(
                            dims,
                            &coords,
                            grid,
                            &paths[i],
                            self.clearance_mm,
                            &via_model,
                            |c| {
                                present_halo[c as usize] =
                                    present_halo[c as usize].saturating_sub(1);
                            },
                        );
                    }
                    paths[i].clear();

                    pad_set.load(&net.passable_pads);

                    // Route within the net's window; on failure, retry once on the
                    // full board so the occasional global net still completes.
                    let routed = route_negotiated(
                        &mut buf,
                        grid,
                        &coords,
                        &heuristic_costs,
                        &pad_set,
                        &net.via_passable_pads,
                        None,
                        None,
                        None,
                        &present,
                        &present_halo,
                        &history,
                        pfac,
                        net.src,
                        net.dst,
                        windows[i],
                        &via_model,
                        has_zero_cost,
                    )
                    .or_else(|| {
                        route_negotiated(
                            &mut buf,
                            grid,
                            &coords,
                            &heuristic_costs,
                            &pad_set,
                            &net.via_passable_pads,
                            None,
                            None,
                            None,
                            &present,
                            &present_halo,
                            &history,
                            pfac,
                            net.src,
                            net.dst,
                            Window::full(dims),
                            &via_model,
                            has_zero_cost,
                        )
                    });

                    if let Some((path, _)) = routed {
                        for &c in &path {
                            present[c as usize] = present[c as usize].saturating_add(1);
                        }
                        if clearance_active {
                            for_each_halo_cell(
                                dims,
                                &coords,
                                grid,
                                &path,
                                self.clearance_mm,
                                &via_model,
                                |c| {
                                    present_halo[c as usize] =
                                        present_halo[c as usize].saturating_add(1);
                                },
                            );
                        }
                        paths[i] = path;
                    }
                    // else: leave unrouted this iteration (no contribution to present).
                }
            }

            // Overuse across GROUPS: a cell is over-used iff ≥2 distinct groups
            // occupy it. Track the first group seen per cell; a second distinct
            // group flags overuse. We only touch the cells the current paths cover,
            // so the scan is O(total path length), not O(all cells): `first_group`
            // and `overused` are cleared via the touched-cell list, not a memset.
            let mut overused_cells: Vec<CellIdx> = Vec::new();
            let mut any_overuse = false;
            for i in 0..n_nets {
                let g = group_ids[i] as i64;
                for &c in &paths[i] {
                    let slot = &mut first_group[c as usize];
                    if *slot < 0 {
                        *slot = g;
                    } else if *slot != g && !overused[c as usize] {
                        overused[c as usize] = true;
                        overused_cells.push(c);
                        any_overuse = true;
                    }
                }
            }

            // Bump history on the over-used cells.
            for &c in &overused_cells {
                history[c as usize] = history[c as usize].saturating_add(SCALE);
            }

            // TRACE: snapshot this iteration's settled state. Taken HERE — after the
            // overuse scan + history bump, but BEFORE the `incremental` block below
            // `mem::swap`s `overused_cells` out — so `overused_cells` still holds this
            // iteration's set. A read-only clone; it does not touch any loop state.
            if let Some(rec) = &mut recorder {
                rec.iterations.push(IterSnapshot {
                    iter,
                    pfac,
                    paths: paths.clone(),
                    overused_cells: overused_cells.clone(),
                    any_overuse,
                });
            }

            // Clear the per-iteration scratch for the cells we touched (O(touched),
            // not O(all cells)): `first_group` via the path cells, `overused` via
            // the over-used list.
            for path in paths.iter().take(n_nets) {
                for &c in path {
                    first_group[c as usize] = -1;
                }
            }
            for &c in &overused_cells {
                overused[c as usize] = false;
            }

            // Roll this iteration's over-used set into `prev_overused` for the next
            // iteration's quiescence test (list-based reset to avoid an O(n) clear).
            if incremental {
                for &c in &prev_overused_cells {
                    prev_overused[c as usize] = false;
                }
                for &c in &overused_cells {
                    prev_overused[c as usize] = true;
                }
                std::mem::swap(&mut prev_overused_cells, &mut overused_cells);
            }

            if !any_overuse {
                break; // converged: cell-disjoint across groups
            }
        }

        // One bounded Gauss-Seidel polish after parallel clearance negotiation on
        // single-layer boards. Jacobi restores throughput but can hand legalization
        // a less coordinated set of simultaneously-chosen paths; a serial sweep
        // lets each net observe earlier polished paths immediately. On multilayer
        // boards, however, this grid-local greedy pass can exchange planar spacing
        // for extra vias/layer changes and worsen the downstream geometric DRC even
        // while its internal halo score improves, so preserve the Jacobi seed there.
        // This is deliberately exactly one pass over all nets: off-path halo
        // decreases make a narrower dirty set unsound, while the hard bound keeps
        // large-board latency predictable.
        if use_parallel && clearance_active && dims.layers == 1 {
            let pfac = MAX_ITERS;
            for i in 0..n_nets {
                let net = &nets[i];
                for &c in &paths[i] {
                    present[c as usize] = present[c as usize].saturating_sub(1);
                }
                for_each_halo_cell(
                    dims,
                    &coords,
                    grid,
                    &paths[i],
                    self.clearance_mm,
                    &via_model,
                    |c| {
                        present_halo[c as usize] = present_halo[c as usize].saturating_sub(1);
                    },
                );
                paths[i].clear();
                pad_set.load(&net.passable_pads);

                let routed = route_negotiated(
                    &mut buf,
                    grid,
                    &coords,
                    &heuristic_costs,
                    &pad_set,
                    &net.via_passable_pads,
                    None,
                    None,
                    None,
                    &present,
                    &present_halo,
                    &history,
                    pfac,
                    net.src,
                    net.dst,
                    windows[i],
                    &via_model,
                    has_zero_cost,
                )
                .or_else(|| {
                    route_negotiated(
                        &mut buf,
                        grid,
                        &coords,
                        &heuristic_costs,
                        &pad_set,
                        &net.via_passable_pads,
                        None,
                        None,
                        None,
                        &present,
                        &present_halo,
                        &history,
                        pfac,
                        net.src,
                        net.dst,
                        Window::full(dims),
                        &via_model,
                        has_zero_cost,
                    )
                });
                if let Some((path, _)) = routed {
                    for &c in &path {
                        present[c as usize] = present[c as usize].saturating_add(1);
                    }
                    for_each_halo_cell(
                        dims,
                        &coords,
                        grid,
                        &path,
                        self.clearance_mm,
                        &via_model,
                        |c| {
                            present_halo[c as usize] = present_halo[c as usize].saturating_add(1);
                        },
                    );
                    paths[i] = path;
                }
            }
        }
        // ---- Order-robust legalization ----
        //
        // Legalization commits whole connection groups, one at a time, in a chosen
        // group order; each group's cells become hard obstacles for later groups.
        // The first-committed group can take a corridor that strands a later one,
        // so the order matters: a cell-disjoint solution may exist only under a
        // different order. We therefore run the same commit logic for several
        // candidate group orders and keep the result that routes the most nets.

        // Distinct group ids in first-appearance order (group_ids are interned by
        // first appearance, so the set {0..n_groups-1} is exactly that order).
        let n_groups = group_ids.iter().map(|&g| g + 1).max().unwrap_or(0);

        // Per-net "alone-path" length: the net routed by itself on the base grid
        // (own pads unmasked, no other nets present). A net that is individually
        // unroutable contributes 0 and can never be committed. This doubles as the
        // ordering difficulty metric.
        let mut alone_len: Vec<Cost> = vec![0; n_nets];
        // Full alone-path cells per net, used by the rip-up stage to find which
        // committed foreign-group nets a stranded net's natural route would cross.
        let mut alone_path: Vec<Vec<CellIdx>> = vec![Vec::new(); n_nets];
        if let Some(provided) = provided_alone {
            debug_assert_eq!(provided.len(), n_nets);
            for (i, path) in provided.iter().enumerate() {
                if !path.is_empty() {
                    alone_len[i] = grid_path_cost(grid, &nets[i], path);
                    alone_path[i] = path.clone();
                }
            }
        } else {
            // No foreign occupancy here; the alone-path is the net by itself on the
            // base grid (route within the window first, full board on failure). Each
            // net's search is INDEPENDENT and pure over the read-only grid, so they run
            // in PARALLEL with rayon — per-thread `SearchBuf`/`PadSet` scratch via
            // `map_init`, results collected over the indexed range so they land in net
            // order regardless of scheduling (byte-identical to the old serial loop).
            // This loop was previously the dominant single-threaded phase of a full
            // route (every net, plus the occasional expensive full-board fallback).
            let no_owner: Vec<i64> = Vec::new();
            let no_halo: Vec<i64> = Vec::new();
            let no_via_guard = ViaGuard::unoccupied(
                via_model.keepout_mm,
                self.via_spacing_mm,
                self.via_hole_spacing_mm,
            );
            let alone: Vec<(Cost, Vec<CellIdx>)> = (0..n_nets)
                .into_par_iter()
                .map_init(
                    || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                    |(buf, pad_set), i| {
                        let net = &nets[i];
                        pad_set.load(&net.passable_pads);
                        let routed = route_legal(
                            buf,
                            grid,
                            &coords,
                            &heuristic_costs,
                            pad_set,
                            &net.via_passable_pads,
                            &no_owner,
                            &no_halo,
                            &no_via_guard,
                            -1,
                            net.src,
                            net.dst,
                            windows[i],
                            &via_model,
                            self.clearance_mm,
                            has_zero_cost,
                        )
                        .or_else(|| {
                            route_legal(
                                buf,
                                grid,
                                &coords,
                                &heuristic_costs,
                                pad_set,
                                &net.via_passable_pads,
                                &no_owner,
                                &no_halo,
                                &no_via_guard,
                                -1,
                                net.src,
                                net.dst,
                                Window::full(dims),
                                &via_model,
                                self.clearance_mm,
                                has_zero_cost,
                            )
                        });
                        match routed {
                            Some((path, _)) => (grid_path_cost(grid, net, &path), path),
                            None => (0, Vec::new()),
                        }
                    },
                )
                .collect();
            for (i, (len, path)) in alone.into_iter().enumerate() {
                alone_len[i] = len;
                alone_path[i] = path;
            }
        }

        // Per-group alone-path length = sum over the group's nets.
        let mut group_alone: Vec<Cost> = vec![0; n_groups];
        for i in 0..n_nets {
            group_alone[group_ids[i]] = group_alone[group_ids[i]].saturating_add(alone_len[i]);
        }

        // Candidate group orders (each a permutation of 0..n_groups).
        let base_order: Vec<usize> = (0..n_groups).collect();
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        // 1. First-appearance / input order.
        candidates.push(base_order.clone());
        // 2. Ascending by alone-path length (stable; ties keep input order).
        {
            let mut o = base_order.clone();
            o.sort_by_key(|&g| group_alone[g]);
            candidates.push(o);
        }
        // 3. Descending by alone-path length (stable; ties keep input order).
        {
            let mut o = base_order.clone();
            o.sort_by_key(|&g| std::cmp::Reverse(group_alone[g]));
            candidates.push(o);
        }
        // 4. For few groups, exhaustively try every order (≤ 5! = 120).
        //
        // The cap is deliberately at 5, not the group count itself: each extra order is
        // a FULL `legalize_in_order` pass (routes every net), so the candidate count
        // multiplies the legalization cost. The factorial makes 6 groups = 720 passes
        // and 7 = 5040 — on real multi-net boards (e.g. tscircuit `keyboards`) that is
        // ~15–30 s of (already parallel, ~9-core) work per solve for NO completion gain:
        // the 3 heuristic orders above plus the rip-up stage already recover the same
        // routable nets the exhaustive search finds. Capping 6–7 group boards to the
        // heuristics cuts those solves ~18× (≈30 s → ≈1.7 s) with identical routed
        // counts. An earlier experiment adding random orders beyond exhaustive also
        // regressed DRC quality because equal-completion cost selection is not
        // clearance-aligned. The bounded deterministic fallback below is therefore
        // completion-only: it cannot replace an equal-routed original winner.
        if n_groups <= 5 {
            for perm in permutations(&base_order) {
                candidates.push(perm);
            }
        }

        // Evaluate each candidate IN PARALLEL. The candidate orders
        // are independent (`legalize_in_order` is a pure function of its inputs), so
        // rayon distributes them across cores; each worker uses its OWN `SearchBuf` +
        // `PadSet` scratch via `map_init` (never shared). Results are collected with
        // their candidate INDEX, then the winner is picked by a fully deterministic,
        // scheduling-independent fold over the indexed results.
        let evaluate_orders = |orders: &[Vec<usize>], index_offset: usize| {
            orders
                .par_iter()
                .enumerate()
                .map_init(
                    || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                    |(buf, pad_set), (idx, order)| {
                        let committed = legalize_in_order(
                            grid,
                            &coords,
                            &heuristic_costs,
                            buf,
                            pad_set,
                            nets,
                            group_ids,
                            &paths,
                            &windows,
                            order,
                            n_cells,
                            &via_model,
                            self.clearance_mm,
                            self.via_spacing_mm,
                            self.via_hole_spacing_mm,
                            self.committed_via_to_trace_guard,
                            has_zero_cost,
                        );
                        let routed = committed.iter().filter(|c| c.is_some()).count();
                        let total_cost = committed_grid_cost(grid, nets, &committed);
                        (index_offset + idx, routed, total_cost, committed)
                    },
                )
                .collect::<Vec<_>>()
        };
        let mut evaluated = evaluate_orders(&candidates, 0);
        let original_candidate_count = candidates.len();
        let original_best_routed = evaluated
            .iter()
            .map(|(_, routed, _, _)| *routed)
            .max()
            .unwrap_or(0);
        let alone_routable = alone_path.iter().filter(|path| !path.is_empty()).count();

        // Medium grouped boards occasionally need one different first claimant for a
        // physical via bottleneck. Only pay for these deterministic alternatives when
        // the original portfolio left an individually-routable net congested. Small
        // boards already exhaust every order; large boards stay behind the hard cap.
        if should_try_diversified_orders(
            n_groups,
            n_nets,
            n_cells,
            original_best_routed,
            alone_routable,
        ) {
            let fallback_orders = diversified_fallback_orders(&base_order, &candidates);

            let offset = candidates.len();
            evaluated.extend(evaluate_orders(&fallback_orders, offset));
            candidates.extend(fallback_orders);
        }

        // Deterministic pick: most nets routed, then lowest true `u64` aggregate
        // caller-grid cost, then lexicographically lowest group ORDER (`order < bo`).
        // Iterating the indexed results in candidate-index order makes the chosen
        // `committed` independent of how rayon scheduled the parallel evaluation.
        let mut best: Option<(usize, u64, &Vec<usize>)> = None;
        let mut best_idx: Option<usize> = None;
        for (idx, routed, total_cost, _committed) in &evaluated {
            let order = &candidates[*idx];
            let better = match &best {
                None => true,
                Some((br, bc, bo)) if *idx >= original_candidate_count => {
                    diversified_candidate_is_better(
                        *routed,
                        *total_cost,
                        order,
                        original_best_routed,
                        *br,
                        *bc,
                        bo,
                    )
                }
                Some((br, bc, bo)) => {
                    legalization_candidate_is_better(*routed, *total_cost, order, *br, *bc, bo)
                }
            };
            if better {
                best = Some((*routed, *total_cost, order));
                best_idx = Some(*idx);
            }
        }
        // TRACE: capture every candidate order's (routed, cost) BEFORE `evaluated` is
        // consumed by the `best_idx` match below. Sorted by candidate index so the
        // list is deterministic and aligns with `candidates`. Read-only.
        let traced_candidates: Option<Vec<CandidateEval>> = recorder.as_ref().map(|_| {
            let mut idxs: Vec<&(usize, usize, u64, Committed)> = evaluated.iter().collect();
            idxs.sort_unstable_by_key(|(idx, _, _, _)| *idx);
            idxs.into_iter()
                .map(|(idx, routed, total_cost, _)| CandidateEval {
                    order: candidates[*idx].clone(),
                    routed: *routed,
                    // RouteTrace's stable public schema stores a `Cost`; selection
                    // above has already used the full `u64` value.
                    total_cost: (*total_cost).min(Cost::MAX as u64) as Cost,
                })
                .collect()
        });
        let (best_routed, best_order, multi_committed) = match best_idx {
            Some(idx) => {
                let routed = best.as_ref().map(|b| b.0).unwrap_or(0);
                // Reclaim the winning committed vec by index without cloning.
                let committed = evaluated
                    .into_iter()
                    .find(|(i, _, _, _)| *i == idx)
                    .map(|(_, _, _, c)| c)
                    .unwrap_or_else(|| vec![None; n_nets]);
                (routed, candidates[idx].clone(), committed)
            }
            None => (0, base_order.clone(), vec![None; n_nets]),
        };

        // ---- Bounded rip-up-and-reroute legalization (final stage) ----
        //
        // The multi-order pass commits whole groups in some fixed order and never
        // displaces an already-committed group. That fails when net A's shortest
        // route blocks B *and* B's shortest route blocks A: no single commit order
        // works, you must displace a committed net. This stage seeds a work-queue
        // with the BEST multi-order result and, for any net that cannot route
        // around the committed cells, rips up the cheapest committed blocker(s),
        // re-places the stranded net, and re-enqueues the ripped nets — bounded by
        // a global rip budget so it always terminates. The result is used only if
        // it routes strictly more nets than the multi-order pass (never a regress).
        let committed = if best_routed < n_nets {
            let rip = ripup_legalize(
                grid,
                &coords,
                &heuristic_costs,
                &mut buf,
                &mut pad_set,
                nets,
                group_ids,
                &alone_path,
                &paths,
                &windows,
                &best_order,
                &multi_committed,
                n_cells,
                &via_model,
                self.clearance_mm,
                self.via_spacing_mm,
                self.via_hole_spacing_mm,
                self.committed_via_to_trace_guard,
                has_zero_cost,
            );
            let rip_routed = rip.iter().filter(|c| c.is_some()).count();
            if rip_routed > best_routed {
                rip
            } else {
                multi_committed
            }
        } else {
            multi_committed
        };
        // TRACE: legalization result — the per-net alone paths (ratsnest / ideal
        // overlay), the chosen group order, every candidate order's score, and the
        // final committed routes. No-op for the production route.
        if let Some(rec) = &mut recorder {
            for (i, ap) in alone_path.iter().enumerate() {
                if let Some(tn) = rec.nets.get_mut(i) {
                    tn.alone_path = ap.clone();
                }
            }
            rec.legalization = Some(LegalizationTrace {
                chosen_order: best_order.clone(),
                candidates: traced_candidates.unwrap_or_default(),
                committed: committed.clone(),
            });
        }
        // Assemble in input net order for determinism. Carry the router's actual
        // electrical-net group id alongside each routed net (aligned 1:1 with
        // `results`) so downstream DRC can grant same-net copper the exact same
        // immunity the router permitted, rather than re-deriving grouping post-hoc.
        let mut results: Vec<RouteResult> = Vec::new();
        let mut groups: Vec<u32> = Vec::new();
        let mut unrouted: Vec<String> = Vec::new();
        for (i, net) in nets.iter().enumerate() {
            match &committed[i] {
                Some(path) => {
                    results.push(RouteResult {
                        net: net.net.clone(),
                        path: path.clone(),
                        cost: grid_path_cost(grid, net, path),
                    });
                    groups.push(group_ids[i] as u32);
                }
                None => unrouted.push(net.net.clone()),
            }
        }

        let congestion = BoardRoute::congestion_from(dims, &results);
        Ok(NegotiatedOutcome {
            board: BoardRoute {
                results,
                unrouted,
                congestion,
                groups,
            },
            alone_routable: alone_path.iter().map(|path| !path.is_empty()).collect(),
        })
    }

    /// Route the adaptive primary and, for the bounded grouped-board cohort, an
    /// all-serial candidate. Candidate selection is deterministic and monotonic in
    /// completion; an exact score tie retains the primary route.
    fn route_impl(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
        provider: Option<&dyn IsolatedRouteProvider>,
        recorder: Option<&mut RouteTrace>,
    ) -> Result<NegotiatedOutcome, RouterError> {
        // Every negotiation/legalization search performs via-mask membership in its
        // hot neighbour loop. Canonicalize once here so all variants and retries use
        // logarithmic lookup while preserving unsorted public/serde inputs.
        let normalized_nets = normalize_via_exemptions(nets);
        let nets = normalized_nets.as_ref();
        let group_ids = connection_group_ids(nets);
        let has_named_group = has_named_subnet_group(nets);
        // The isolated batch is board-static: compute it at most once and share it
        // with both portfolio variants. A rejected provider response is represented
        // as `None`, which preserves the existing all-CPU path in each variant.
        let provided_alone =
            provider.and_then(|provider| provider_alone_paths(self, grid, nets, provider));
        let capture_trace = recorder.is_some();
        let mut primary_trace = empty_route_trace(grid.dims);
        let primary_recorder = if capture_trace {
            Some(&mut primary_trace)
        } else {
            None
        };
        let primary = self.route_variant(
            grid,
            nets,
            &group_ids,
            NegotiationMode::Adaptive,
            provided_alone.as_deref(),
            primary_recorder,
        )?;

        let mut selected = primary;
        let mut selected_trace = primary_trace;
        if should_try_serial_candidate(grid.dims, nets.len(), has_named_group, &selected.board) {
            let mut serial_trace = empty_route_trace(grid.dims);
            let serial_recorder = if capture_trace {
                Some(&mut serial_trace)
            } else {
                None
            };
            let serial = self.route_variant(
                grid,
                nets,
                &group_ids,
                NegotiationMode::ForceSerial,
                provided_alone.as_deref(),
                serial_recorder,
            )?;
            if serial_candidate_is_better(&selected.board, &serial.board) {
                selected = serial;
                selected_trace = serial_trace;
            }
        }

        if let Some(output_trace) = recorder {
            *output_trace = selected_trace;
        }
        Ok(selected)
    }

    /// Route without a visualisation trace while returning the per-input-net
    /// isolation result already computed by legalization.
    ///
    /// `outcome.board` is byte-identical to [`Router::route`] for the same inputs.
    pub fn route_with_outcome(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
    ) -> Result<NegotiatedOutcome, RouterError> {
        self.route_impl(grid, nets, None, None)
    }

    /// Route while sourcing the board-static isolated-net batch from `provider`.
    ///
    /// The provider is invoked at most once, including when the bounded portfolio
    /// evaluates both adaptive and serial negotiation variants. If the provider
    /// errors or returns a malformed batch, the router transparently falls back to
    /// the same CPU isolated searches used by [`Self::route_with_outcome`].
    pub fn route_with_isolated_provider(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
        provider: &dyn IsolatedRouteProvider,
    ) -> Result<NegotiatedOutcome, RouterError> {
        self.route_impl(grid, nets, Some(provider), None)
    }

    /// Route a board and additionally return a [`RouteTrace`] recording each
    /// negotiation iteration and the legalization phase, for step-by-step animation
    /// in the visualiser. The returned [`BoardRoute`] is identical to what
    /// [`Router::route`] would produce for the same inputs (the recorder only reads
    /// loop state); only the extra trace is the difference.
    pub fn route_traced(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
    ) -> Result<(BoardRoute, RouteTrace), RouterError> {
        // Construct the trace explicitly with the real `dims` (it has no meaningful
        // `Default`); the capture points fill the rest.
        let mut trace = empty_route_trace(grid.dims);
        let outcome = self.route_impl(grid, nets, None, Some(&mut trace))?;
        Ok((outcome.board, trace))
    }

    /// Traced counterpart of [`Self::route_with_isolated_provider`].
    ///
    /// Recording is observational only: for the same provider batch, this method's
    /// board is identical to the untraced provider-aware method.
    pub fn route_traced_with_isolated_provider(
        &self,
        grid: &Grid,
        nets: &[NetEndpoints],
        provider: &dyn IsolatedRouteProvider,
    ) -> Result<(BoardRoute, RouteTrace), RouterError> {
        let mut trace = empty_route_trace(grid.dims);
        let outcome = self.route_impl(grid, nets, Some(provider), Some(&mut trace))?;
        Ok((outcome.board, trace))
    }
}

impl Router for NegotiatedRouter {
    /// The production route delegates to the deterministic bounded portfolio
    /// without allocating a visualisation trace.
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        self.route_with_outcome(grid, nets)
            .map(|outcome| outcome.board)
    }
}

/// Route one net for the negotiation phase using on-the-fly congestion pricing —
/// no grid clone. The price of the planar move `u -> v` is
/// `edge_cost(len(u,v)) + history[v] + pfac*SCALE*present[v]` (+ the clearance halo
/// term), capped strictly below [`OBSTACLE`] (`present` already excludes this net's
/// own occupancy). The planar base is the move's GEOMETRIC length from `coords`
/// rather than a uniform `SCALE`, so steps over a wide channel cost more than steps
/// over a fine pitch; on a uniform grid every step is length 1 and the base is
/// exactly `SCALE` (byte-identical). A cell is blocked iff it is a base obstacle that
/// is NOT one of the net's own pads, or it lies outside `window`. The own endpoints
/// are forced passable. Returns the windowed shortest path and its (priced) cost, or
/// `None`.
#[allow(clippy::too_many_arguments)]
fn route_negotiated(
    buf: &mut SearchBuf,
    base: &Grid,
    coords: &GridCoords,
    heuristic_costs: &ManhattanCosts,
    pads: &PadSet,
    via_passable_pads: &[CellIdx],
    own_path: Option<&PadSet>,
    own_halo: Option<&CountedCellSet>,
    shared_congestion: Option<&[u64]>,
    present: &[u32],
    present_halo: &[u32],
    history: &[u32],
    pfac: u32,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
    has_zero_cost: bool,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    // Price to MOVE onto cell `c` over a base step cost `base_cost` (the planar
    // geometric edge length for a planar move, or the via `step_cost` for a layer
    // change): the base plus permanent history, the present-congestion penalty, and
    // the clearance penalty (TritonRoute `objCost` analog) —
    // `pfac * CLEARANCE_NEG_WEIGHT * present_halo[c]`, i.e. how many OTHER nets'
    // clearance footprints cover `c`, scaled by the growing present-factor so the
    // negotiation spreads nets apart over iterations. Capped below OBSTACLE. The
    // congestion weights are multiples of `SCALE`, so they keep their strength
    // relative to the geometric base step at any pitch. `present_halo` is all-zero
    // when clearance is inactive, so that term vanishes (byte-identical default).
    let priced_with_base = |c: CellIdx, base_cost: u64| -> Cost {
        let ci = c as usize;
        let self_present = own_path.is_some_and(|set| set.contains(c)) as u32;
        let self_halo = own_halo.map_or(0, |set| set.count(c));
        let priced = if let Some(shared) = shared_congestion {
            let foreign_congestion = shared[ci]
                .saturating_sub((pfac as u64) * (SCALE as u64) * self_present as u64)
                .saturating_sub((pfac as u64) * (CLEARANCE_NEG_WEIGHT as u64) * self_halo as u64);
            base_cost
                .saturating_add(history[ci] as u64)
                .saturating_add(foreign_congestion)
        } else {
            let foreign_present = present[ci].saturating_sub(self_present);
            let foreign_halo = present_halo[ci].saturating_sub(self_halo);
            base_cost
                .saturating_add(history[ci] as u64)
                .saturating_add((pfac as u64) * (SCALE as u64) * foreign_present as u64)
                .saturating_add((pfac as u64) * (CLEARANCE_NEG_WEIGHT as u64) * foreign_halo as u64)
        };
        passable_search_cost(priced)
    };
    // Edge-aware planar base: the geometric length of the move `u -> v`, in the same
    // fixed-point units as the heuristic. On a uniform grid this is the constant
    // `SCALE`, so `cost_fn` reduces to the historical `priced_with_base(v, SCALE)`.
    let enter_weight = |c: CellIdx| -> Cost {
        if base.is_obstacle(c) && pads.contains(c) {
            1
        } else {
            base.cost_at(c)
        }
    };
    let cost_fn = |u: CellIdx, v: CellIdx| -> Cost {
        if base.is_board_planar_step_forbidden(u, v) {
            return OBSTACLE;
        }
        let geometric = edge_cost(coords.manhattan_len(dims, u, v)) as u64;
        priced_with_base(v, geometric.saturating_mul(enter_weight(v) as u64))
    };
    let blocked_fn = |c: CellIdx| -> bool {
        if !window.contains(dims, c) {
            return true;
        }
        if base.is_board_forbidden(c) {
            return true;
        }
        if c == src || c == dst {
            return false;
        }
        base.is_obstacle(c) && !pads.contains(c)
    };
    // A via step `u -> v` (adjacent layers, same x,y) is allowed iff the model
    // permits that layer transition; it is priced like a cell but with the via's
    // `step_cost` as the base instead of the planar `SCALE`, so vias negotiate the
    // destination cell's congestion exactly as planar moves do. `blocked_fn` is
    // applied to `v` by `astar_buf` first, so an out-of-window / foreign-pad target
    // already blocks the via.
    let via_step = |u: CellIdx, v: CellIdx| -> Option<Cost> {
        if via_model.is_step_legal(dims.layer_of(u), dims.layer_of(v))
            && !base.is_board_via_forbidden(u)
            && !base.is_board_via_forbidden(v)
            && (!base.is_via_forbidden(u) || sorted_contains(via_passable_pads, &u))
            && (!base.is_via_forbidden(v) || sorted_contains(via_passable_pads, &v))
        {
            Some(priced_with_base(
                v,
                (via_model.step_cost as u64).saturating_mul(enter_weight(v) as u64),
            ))
        } else {
            None
        }
    };
    let h = |c: CellIdx| {
        if has_zero_cost {
            0
        } else {
            manhattan_scaled(dims, heuristic_costs, c, dst, via_model.step_cost)
        }
    };
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h, via_step)
}

/// Route one net for legalization / rip-up using on-the-fly costs — no grid clone.
///
/// Clearance AND copper are BOTH HARD blocks here (this is the committing pass; a
/// route it returns is guaranteed clearance-clean, so a net that cannot reach `dst`
/// without violating clearance returns `None` and is reported unrouted/congested by
/// the caller — never silently committed in violation):
///   * `owner` — committed COPPER. A cell owned by a group other than `own_group`
///     is a HARD obstacle (two distinct nets may never overlap). Cells owned by
///     `own_group` (siblings) and free cells are passable at their base cost.
///   * `halo`  — foreign clearance / via-keepout halo. A cell owned by a foreign
///     group, or [`HALO_MIXED`] because multiple groups cover it, is a HARD obstacle
///     unless it is already this group's copper. Own-group halo costs nothing.
///   * `via_guard` — dense per-layer feature dilation for candidate vias plus an
///     optional committed-via-only field for planar steps. The latter is consulted
///     by planar `cost_fn(u,v)` before endpoint/pad exemptions and therefore closes
///     the own-pad escape ordering gap without applying trace spacing to via moves.
///   * `src`/`dst` stay forced-passable (a net's own pads must remain reachable).
///
/// Every passable step is priced by its GEOMETRIC length (`edge_cost` from `coords`)
/// rather than the grid's per-cell value, so the planar base ignores whether a cell
/// is an unmasked own-pad or ordinary copper — its endpoints are always enterable and
/// the search is confined to `window`. `owner`/`halo` may each be empty to mean "no
/// owners / no halo" (the alone-path case). A disabled [`ViaGuard`] selects the
/// legacy generic-halo via guard for low-level callers that do not supply physical
/// via geometry; production physical pipelines use the feature-aware dense field.
/// Returns the windowed shortest path and its cost, or `None`.
#[allow(clippy::too_many_arguments)]
fn route_legal_once(
    buf: &mut SearchBuf,
    base: &Grid,
    coords: &GridCoords,
    heuristic_costs: &ManhattanCosts,
    pads: &PadSet,
    via_passable_pads: &[CellIdx],
    owner: &[i64],
    halo: &[i64],
    via_guard: &ViaGuard,
    local_via_forbidden_xy: &[u32],
    own_group: i64,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
    clearance: f64,
    has_zero_cost: bool,
) -> Option<(Vec<CellIdx>, Cost)> {
    let dims = base.dims;
    let has_owner = !owner.is_empty();
    let has_halo = !halo.is_empty();
    // Edge-aware planar base: the geometric length of the move `u -> v` in fixed-point
    // units (same units as the heuristic), replacing the old uniform per-cell base.
    // On a uniform grid every step is length 1, so the base is the constant `SCALE`
    // (the legalizer reports `unit_cost(path)` — path length — so this magnitude only
    // affects the path CHOICE, never the emitted cost). Clearance is now HARD (handled
    // in `blocked_fn`), so `cost_fn` is the pure weighted geometric base, capped below
    // `OBSTACLE` because that exact value is the search's unreachable sentinel.
    let enter_weight = |c: CellIdx| -> Cost {
        if base.is_obstacle(c) && pads.contains(c) {
            1
        } else {
            base.cost_at(c)
        }
    };
    let cost_fn = |u: CellIdx, v: CellIdx| -> Cost {
        if base.is_board_planar_step_forbidden(u, v)
            || via_guard.conflicts_planar_cell(u, own_group)
            || via_guard.conflicts_planar_cell(v, own_group)
        {
            return OBSTACLE;
        }
        passable_search_cost(
            (edge_cost(coords.manhattan_len(dims, u, v)) as u64)
                .saturating_mul(enter_weight(v) as u64),
        )
    };
    let blocked_fn = |c: CellIdx| -> bool {
        if !window.contains(dims, c) {
            return true;
        }
        if base.is_board_forbidden(c) {
            return true;
        }
        // Foreign-group COPPER cells are hard obstacles, even at this net's
        // endpoints — distinct nets may never overlap.
        if has_owner {
            let o = owner[c as usize];
            if o >= 0 && o != own_group {
                return true;
            }
        }
        // Endpoints (this net's own pads) stay reachable. Checked AFTER foreign copper
        // (two nets may never share a cell) but BEFORE the foreign-halo block, so a pad
        // sitting in another net's halo can still be entered to terminate the route.
        if c == src || c == dst {
            return false;
        }
        // Foreign-group HALO is now a HARD obstacle: committing copper inside another
        // net's required spacing is a clearance violation, so this pass refuses it. A
        // foreign halo cell that is ALSO this net's own copper (`owner == own_group`)
        // is not blocked. When `halo` is empty this branch is inert (fast path).
        if has_halo {
            let h = halo[c as usize];
            let own_copper = has_owner && owner[c as usize] == own_group;
            if halo_is_foreign(h, own_group) && !own_copper {
                return true;
            }
        }
        base.is_obstacle(c) && !pads.contains(c)
    };
    // Via annular-ring radius: a placed via's pad reserves `max(clearance, keepout)`
    // around itself on both spanned layers. When this is <= 0 (clearance-off fast
    // path) the guard is skipped entirely and behaviour is byte-identical.
    let via_r = clearance.max(via_model.keepout_mm);
    let feature_aware_via_guard = via_guard.feature_aware();
    // Compatibility for callers that have not supplied enough geometry to
    // distinguish vias from traces. Production physical pipelines opt into the
    // feature-aware branch and never scan an already-inflated halo. Endpoint
    // identity is not an exemption here: a dynamic foreign owner/halo must win even
    // when the scanned ring happens to cover this net's terminal cell.
    let legacy_ring_conflict = |cx: u32, cy: u32, layer: u32| -> bool {
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, via_r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, via_r);
        for ny in y0..y1 {
            if !geom_line_within(&coords.y_lines, dims.h, cy, ny, via_r) {
                continue;
            }
            for nx in x0..x1 {
                if !geom_line_within(&coords.x_lines, dims.w, cx, nx, via_r) {
                    continue;
                }
                let cell = dims.idx3(nx, ny, layer);
                let cell_owner = if has_owner { owner[cell as usize] } else { -1 };
                if cell_owner >= 0 && cell_owner != own_group {
                    return true;
                }
                if has_halo
                    && halo_is_foreign(halo[cell as usize], own_group)
                    && cell_owner != own_group
                {
                    return true;
                }
            }
        }
        false
    };
    // A via step is legal per the model; it costs the via's `step_cost` (foreign
    // owners / endpoints are already rejected by `blocked_fn` on the destination).
    // Additionally reject the step when the via's annular ring at `v` would overlap
    // a foreign copper centre on either spanned layer. Do NOT scan foreign halo cells
    // here: they are already inflated around their source copper, and scanning around
    // them again would add the radii twice. Planar/destination moves still honour the
    // ordinary halo through `blocked_fn` above.
    let via_step = |u: CellIdx, v: CellIdx| -> Option<Cost> {
        let (lu, lv) = (dims.layer_of(u), dims.layer_of(v));
        if !via_model.is_step_legal(lu, lv) {
            return None;
        }
        if base.is_board_via_forbidden(u) || base.is_board_via_forbidden(v) {
            return None;
        }
        let plane = dims.w * dims.h;
        if local_via_forbidden_xy.binary_search(&(u % plane)).is_ok()
            || via_guard.conflicts_hole_cell(dims, u, own_group)
        {
            return None;
        }
        if (base.is_via_forbidden(u) && !sorted_contains(via_passable_pads, &u))
            || (base.is_via_forbidden(v) && !sorted_contains(via_passable_pads, &v))
        {
            return None;
        }
        let (vx, vy, _) = dims.xyz(v);
        if feature_aware_via_guard {
            if via_guard.conflicts_step(u, v, own_group) {
                return None;
            }
        } else if via_r > 0.0
            && has_halo
            && (legacy_ring_conflict(vx, vy, lu) || legacy_ring_conflict(vx, vy, lv))
        {
            return None;
        }
        Some(passable_search_cost(
            (via_model.step_cost as u64).saturating_mul(enter_weight(v) as u64),
        ))
    };
    let h = |c: CellIdx| {
        if has_zero_cost {
            0
        } else {
            manhattan_scaled(dims, heuristic_costs, c, dst, via_model.step_cost)
        }
    };
    astar_buf(buf, dims, src, dst, cost_fn, blocked_fn, h, via_step)
}

/// Return the first pair of distinct physical via XYs in one path whose drill
/// centres violate the net-independent rule. Same-net centres within the producer's
/// 0.005 mm coincidence tolerance are one stacked via and are ignored.
fn first_self_via_hole_conflict(
    dims: Dims,
    coords: &GridCoords,
    path: &[CellIdx],
    spacing_mm: f64,
) -> Option<(u32, u32)> {
    if spacing_mm <= VIA_SPACING_EPS_MM {
        return None;
    }
    let mut centres: Vec<u32> = Vec::new();
    for step in path.windows(2) {
        let (ax, ay, al) = dims.xyz(step[0]);
        let (bx, by, bl) = dims.xyz(step[1]);
        if ax == bx && ay == by && al != bl {
            let planar = ay * dims.w + ax;
            if centres.last().copied() != Some(planar) {
                centres.push(planar);
            }
        }
    }
    let threshold = (spacing_mm - VIA_SPACING_EPS_MM).max(0.0);
    let threshold_sq = threshold * threshold;
    for i in 0..centres.len() {
        let a = centres[i];
        let (ax, ay) = (a % dims.w, a / dims.w);
        for &b in &centres[i + 1..] {
            let (bx, by) = (b % dims.w, b / dims.w);
            let dx = coords.x_of(ax) - coords.x_of(bx);
            let dy = coords.y_of(ay) - coords.y_of(by);
            let distance_sq = dx * dx + dy * dy;
            if distance_sq <= VIA_SAME_LOCATION_MM * VIA_SAME_LOCATION_MM {
                continue;
            }
            if distance_sq < threshold_sq {
                return Some((a, b));
            }
        }
    }
    None
}

/// Route once, then repair a rare intra-path drill-spacing conflict by a bounded,
/// deterministic branch over forbidding either offending via XY. Every accepted
/// path passes the exact centre-distance predicate; exhausting the eight-search
/// portfolio safely leaves the net unrouted instead of emitting a fabrication
/// violation. The first valid route is unchanged byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn route_legal(
    buf: &mut SearchBuf,
    base: &Grid,
    coords: &GridCoords,
    heuristic_costs: &ManhattanCosts,
    pads: &PadSet,
    via_passable_pads: &[CellIdx],
    owner: &[i64],
    halo: &[i64],
    via_guard: &ViaGuard,
    own_group: i64,
    src: CellIdx,
    dst: CellIdx,
    window: Window,
    via_model: &ViaModel,
    clearance: f64,
    has_zero_cost: bool,
) -> Option<(Vec<CellIdx>, Cost)> {
    let route_with = |buf: &mut SearchBuf, forbidden: &[u32]| {
        route_legal_once(
            buf,
            base,
            coords,
            heuristic_costs,
            pads,
            via_passable_pads,
            owner,
            halo,
            via_guard,
            forbidden,
            own_group,
            src,
            dst,
            window,
            via_model,
            clearance,
            has_zero_cost,
        )
    };

    let first = route_with(buf, &[])?;
    let Some((a, b)) =
        first_self_via_hole_conflict(base.dims, coords, &first.0, via_guard.via_hole_mm)
    else {
        return Some(first);
    };

    const MAX_SELF_HOLE_SEARCHES: usize = 8;
    let mut queue = std::collections::VecDeque::from([vec![b], vec![a]]);
    let mut seen = std::collections::BTreeSet::new();
    let mut searches = 1;
    while searches < MAX_SELF_HOLE_SEARCHES {
        let Some(forbidden) = queue.pop_front() else {
            break;
        };
        if !seen.insert(forbidden.clone()) {
            continue;
        }
        searches += 1;
        let Some(candidate) = route_with(buf, &forbidden) else {
            continue;
        };
        let Some((a, b)) =
            first_self_via_hole_conflict(base.dims, coords, &candidate.0, via_guard.via_hole_mm)
        else {
            return Some(candidate);
        };
        for centre in [b, a] {
            let mut branch = forbidden.clone();
            match branch.binary_search(&centre) {
                Ok(_) => continue,
                Err(index) => branch.insert(index, centre),
            }
            if !seen.contains(&branch) {
                queue.push_back(branch);
            }
        }
    }
    None
}

/// Enumerate every cell in a `path`'s SOFT clearance footprint, invoking `visit`
/// once per cell (cells may repeat across overlapping halos — the callers use
/// saturating inc/dec or first-claim guards, so repeats are intended/idempotent).
///
/// The footprint is exactly the set of cells [`stamp_owner`] considers for `halo`
/// (so the negotiation `present_halo` field and the legalization `halo` map share
/// one shape):
///   * **Planar clearance halo.** For each path cell, on that cell's OWN layer, the
///     [`geom_box`] band of geometric radius `r = clearance` (continuous units over
///     `coords`; the `(2r+1)x(2r+1)` Chebyshev box on a uniform grid). Base-obstacle
///     cells are SKIPPED (a halo never claims an obstacle / foreign pad).
///   * **Via keepout halo.** At each via (two consecutive path cells sharing `(x,y)`
///     but differing in layer), on BOTH spanned layers, the geometric box of radius
///     `max(clearance, via_model.keepout_mm)`. Base-obstacle cells are SKIPPED.
///
/// When `clearance == 0.0` AND `via_model.keepout_mm == 0.0` every radius is 0, so
/// `visit` is never called and the footprint is empty — the property that keeps the
/// clearance-off router byte-identical.
///
/// NON-UNIFORM GRID NOTE: the halo radius is a *geometric distance* (`clearance`,
/// continuous units) measured over `coords`, NOT a cell count, so on a Hanan grid
/// it spans the same physical width everywhere regardless of local line density —
/// matching [`mr_grid::GridBuilder::inflate_clearance`]'s hard-obstacle model. On a
/// [`GridCoords::uniform`] grid (unit-spaced lines) a `clearance` of `n` reproduces
/// the former `n`-cell Chebyshev box exactly (byte-identical). The via keepout is
/// likewise geometric (`via_model.keepout_mm` continuous units).
///
/// NOTE: unlike [`stamp_owner`] this does NOT skip cells already owned/claimed (it
/// has no ownership view); it visits the geometric footprint and leaves
/// owner/first-claim policy to the caller. `present_halo` wants the raw geometric
/// count (every other net's halo contributes), so this is the right shape for it,
/// and `stamp_owner` layers its own `owner == -1 && halo == -1` guard on top.
fn for_each_halo_cell(
    dims: mr_core::Dims,
    coords: &GridCoords,
    base: &Grid,
    path: &[CellIdx],
    clearance: f64,
    via_model: &ViaModel,
    mut visit: impl FnMut(CellIdx),
) {
    // Visit the planar geometric box of radius `r` (continuous units) around
    // `(cx, cy)` on `layer`, skipping base obstacles (a halo never claims an
    // obstacle / foreign pad). The per-axis index band is computed over the `coords`
    // line arrays, so on a uniform grid this is the `(2r+1)x(2r+1)` Chebyshev box and
    // on a Hanan grid it is the set of lines within `r` continuous units.
    let box_cells = |cx: u32, cy: u32, layer: u32, r: f64, visit: &mut dyn FnMut(CellIdx)| {
        if r <= 0.0 {
            return;
        }
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, r);
        for ny in y0..y1 {
            if !geom_line_within(&coords.y_lines, dims.h, cy, ny, r) {
                continue;
            }
            for nx in x0..x1 {
                if !geom_line_within(&coords.x_lines, dims.w, cx, nx, r) {
                    continue;
                }
                let n = dims.idx3(nx, ny, layer);
                if !base.is_obstacle(n) {
                    visit(n);
                }
            }
        }
    };

    // Planar clearance halo around each path cell, on that cell's own layer.
    for &c in path {
        let (cx, cy, cl) = dims.xyz(c);
        box_cells(cx, cy, cl, clearance, &mut visit);
    }

    // Via keepout: a via is a consecutive same-(x,y), layer-changing step. At each
    // via (x,y) visit the larger of the planar clearance and the via keepout on both
    // layers the via spans.
    let via_r = clearance.max(via_model.keepout_mm);
    if via_r > 0.0 {
        for w in path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                box_cells(ax, ay, al, via_r, &mut visit);
                box_cells(bx, by, bl, via_r, &mut visit);
            }
        }
    }
}

/// Half-open candidate range containing every line index within `r` continuous
/// units of `seed`.
///
/// Complete coordinate arrays are sorted, so their matches form one contiguous band
/// and the normal path walks outward from `seed`. A truncated array uses
/// [`GridCoords::x_of`]'s documented index fallback, which can be non-monotonic at
/// the explicit/fallback boundary; in that defensive case the only sound candidate
/// range is the full axis. Callers apply [`geom_line_within`] inside the returned
/// range, making the fallback an exact full scan rather than over-inflation.
///
/// On a [`GridCoords::uniform`] grid this returns
/// `[seed - floor(r), seed + floor(r) + 1)`, preserving the historical Chebyshev box.
fn geom_box(lines: &[f64], count: u32, seed: u32, r: f64) -> (u32, u32) {
    if count == 0 {
        return (0, 0);
    }
    if lines.len() < count as usize {
        return (0, count);
    }
    let seed = seed.min(count - 1);
    let pos = lines[seed as usize];
    let mut lo = seed;
    while lo > 0 && (pos - lines[(lo - 1) as usize]).abs() <= r {
        lo -= 1;
    }
    let mut hi = seed + 1;
    while hi < count && (lines[hi as usize] - pos).abs() <= r {
        hi += 1;
    }
    (lo, hi)
}

/// Exact membership predicate paired with [`geom_box`]. Missing coordinates use
/// the same unit-position fallback as [`GridCoords::x_of`] / `y_of`.
#[inline]
fn geom_line_within(lines: &[f64], count: u32, seed: u32, candidate: u32, r: f64) -> bool {
    if count == 0 || candidate >= count {
        return false;
    }
    let at = |i: u32| lines.get(i as usize).copied().unwrap_or(i as f64);
    (at(candidate) - at(seed.min(count - 1))).abs() <= r
}

/// True when a reused/precomputed path would violate either exact feature guard:
/// planar edges check both endpoints against committed vias, and layer-changing
/// steps check both endpoints against the candidate-via feature union. Fresh
/// searches apply the identical predicates in `cost_fn` / `via_step`.
fn path_has_foreign_via_conflict(
    dims: Dims,
    coords: &GridCoords,
    via_guard: &ViaGuard,
    path: &[CellIdx],
    own_group: i64,
) -> bool {
    first_self_via_hole_conflict(dims, coords, path, via_guard.via_hole_mm).is_some()
        || path.windows(2).any(|step| {
            let (ax, ay, al) = dims.xyz(step[0]);
            let (bx, by, bl) = dims.xyz(step[1]);
            if al == bl {
                via_guard.conflicts_planar_cell(step[0], own_group)
                    || via_guard.conflicts_planar_cell(step[1], own_group)
            } else {
                ax == bx
                    && ay == by
                    && (via_guard.conflicts_hole_cell(dims, step[0], own_group)
                        || (via_guard.feature_aware()
                            && via_guard.conflicts_step(step[0], step[1], own_group)))
            }
        })
}

/// Fold a committed `path` into the ownership maps, separating HARD copper from the
/// SOFT clearance halo so a net is never dropped merely for failing to honour
/// spacing. This is the single place both legalizers commit copper.
///
/// Three dense maps are written:
///   * `owner` — the committed PATH cells (the actual copper). A foreign group's
///     `owner` cell is a HARD block: two distinct nets must never overlap.
///   * `halo`  — the clearance / via-keepout cells around the copper. A foreign
///     group's `halo` cell is a HARD block in [`route_legal`] (the committing pass):
///     copper may never be placed inside another net's required spacing, so a net
///     that cannot route clear is left unrouted/congested rather than violating.
///   * `via_guard` — exact Euclidean feature dilation used only by candidate vias.
///     Track cells contribute the via-to-trace radius and via centres contribute
///     the wider via-to-via radius without widening the ordinary planar halo.
///
/// Exact stamping rule, applied for the committed `path` belonging to `group`:
///
/// 1. **Path cells (copper).** `owner[c] = group` for every cell `c` on the path,
///    unconditionally (matches the pre-clearance behaviour — the path always wins
///    its own cells).
/// 2. **Planar clearance halo.** For each path cell, on that cell's OWN layer,
///    visit every cell `n` within geometric distance `clearance` over `coords` (the
///    [`geom_box`] band; on a uniform grid this is the `(2r+1)x(2r+1)` Chebyshev
///    box). If the cell is not copper or a base obstacle, set a free halo cell to
///    `group`, leave an existing same-group claim alone, or mark a cross-group
///    overlap [`HALO_MIXED`]. A mixed cell is foreign to every group; otherwise a
///    rerouted first claimant could incorrectly enter the second claimant's halo.
/// 3. **Via keepout.** A via is detected as two consecutive path cells sharing the
///    same `(x, y)` but differing in layer. At each such `(x, y)`, on *every* layer
///    the via spans, stamp a halo of radius `max(clearance, via_model.keepout_mm)`
///    under the identical rule — a via pad is wider than a track, so it reserves a
///    larger neighbourhood.
/// 4. **Via guard.** Stamp the path's exact via-to-trace / via-to-via union into
///    `via_guard`; repeated middle layers in a through-via run collapse to one disk.
///
/// CRITICAL: when `clearance == 0.0` AND `via_model.keepout_mm == 0.0` this marks
/// *exactly* the path cells into `owner` and writes NOTHING into `halo` (a radius-0
/// box is skipped; the via halo radius is likewise 0). `halo` stays all `-1`, so
/// [`route_legal`]'s soft cost adds nothing and the default router is byte-identical
/// to the pre-clearance implementation. `clearance` is a geometric distance over
/// `coords` (continuous units); on a uniform grid `n` reproduces the former `n`-cell
/// Chebyshev halo — see [`geom_box`].
#[allow(clippy::too_many_arguments)]
fn stamp_owner(
    owner: &mut [i64],
    halo: &mut [i64],
    via_guard: &mut ViaGuard,
    base: &Grid,
    dims: mr_core::Dims,
    coords: &GridCoords,
    path: &[CellIdx],
    group: i64,
    clearance: f64,
    via_model: &ViaModel,
) {
    // Stamp a planar geometric halo of radius `r` (continuous units, over `coords`)
    // around `(cx, cy)` on `layer` into the `halo` map, claiming only cells that are
    // not real copper (`owner == -1`) and not base obstacles. Cross-group overlap
    // becomes `HALO_MIXED`, while repeated same-group claims remain that group. The centre cell is
    // included, but a path cell already has `owner == group`, so the `owner == -1`
    // guard skips it (its halo entry stays free — own-group halo is irrelevant since
    // own-group cells cost nothing).
    let stamp_halo = |owner: &[i64], halo: &mut [i64], cx: u32, cy: u32, layer: u32, r: f64| {
        if r <= 0.0 {
            return;
        }
        let (x0, x1) = geom_box(&coords.x_lines, dims.w, cx, r);
        let (y0, y1) = geom_box(&coords.y_lines, dims.h, cy, r);
        for ny in y0..y1 {
            if !geom_line_within(&coords.y_lines, dims.h, cy, ny, r) {
                continue;
            }
            for nx in x0..x1 {
                if !geom_line_within(&coords.x_lines, dims.w, cx, nx, r) {
                    continue;
                }
                let n = dims.idx3(nx, ny, layer);
                let ni = n as usize;
                if owner[ni] == -1 && !base.is_obstacle(n) {
                    halo[ni] = match halo[ni] {
                        HALO_FREE => group,
                        existing if existing == group => existing,
                        _ => HALO_MIXED,
                    };
                }
            }
        }
    };

    // 1: own the path cells (copper) unconditionally.
    for &c in path {
        owner[c as usize] = group;
    }
    // 2: stamp the planar clearance halo around each path cell.
    for &c in path {
        let (cx, cy, cl) = dims.xyz(c);
        stamp_halo(owner, halo, cx, cy, cl, clearance);
    }

    // 3: via keepout. A via is a consecutive same-(x,y), layer-changing step. At
    // each via (x,y) stamp the larger of the planar clearance and the via keepout
    // on every layer the via spans (both endpoints' layers).
    let via_r = clearance.max(via_model.keepout_mm);
    if via_r > 0.0 {
        for w in path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                stamp_halo(owner, halo, ax, ay, al, via_r);
                stamp_halo(owner, halo, bx, by, bl, via_r);
            }
        }
    }

    // Dedicated via guards are independent of the ordinary via-to-track halo above.
    // The field is absent for legacy callers with no physical via-via spacing.
    via_guard.stamp_path(dims, coords, path, group);
}

/// Rebuild legalization ownership after a rip.
///
/// A clearance cell may belong to multiple groups. Incrementally deleting one
/// group's scalar halo label would either leak its reservation or erase another
/// group's overlapping reservation. A deterministic net-index rebuild is exact,
/// naturally reduces [`HALO_MIXED`] to the sole remaining group, and rip-up is rare
/// enough that the linear pass is cheaper than a per-cell owner set.
#[allow(clippy::too_many_arguments)]
fn rebuild_owner_maps(
    owner: &mut [i64],
    halo: &mut [i64],
    via_guard: &mut ViaGuard,
    base: &Grid,
    coords: &GridCoords,
    committed: &Committed,
    group_ids: &[usize],
    clearance: f64,
    via_model: &ViaModel,
) {
    owner.fill(-1);
    halo.fill(HALO_FREE);
    via_guard.clear();
    for (i, path) in committed.iter().enumerate() {
        if let Some(path) = path {
            stamp_owner(
                owner,
                halo,
                via_guard,
                base,
                base.dims,
                coords,
                path,
                group_ids[i] as i64,
                clearance,
                via_model,
            );
        }
    }
}

/// Commit all connection groups in the given `group_order`, returning the
/// per-net committed path (or `None` if that net could not be placed).
///
/// Each group is committed as a unit: its members are placed (reusing the
/// negotiated `paths[i]` when that path already avoids every foreign-group cell,
/// otherwise rerouting once around the committed `occupied` set), then the
/// group's cells are folded into `occupied` so later groups treat them as hard
/// obstacles. Sibling sub-nets of the same group never block one another.
///
/// `group_order` must be a permutation of the distinct group ids. `work` is used
/// as scratch and is left in an arbitrary state. The result is in net-index
/// order and is fully deterministic given the inputs and order.
#[allow(clippy::too_many_arguments)]
fn legalize_in_order(
    grid: &Grid,
    coords: &GridCoords,
    heuristic_costs: &ManhattanCosts,
    buf: &mut SearchBuf,
    pad_set: &mut PadSet,
    nets: &[NetEndpoints],
    group_ids: &[usize],
    paths: &[Vec<CellIdx>],
    windows: &[Window],
    group_order: &[usize],
    n_cells: usize,
    via_model: &ViaModel,
    clearance: f64,
    via_spacing_mm: f64,
    via_hole_spacing_mm: f64,
    protect_planar_from_vias: bool,
    has_zero_cost: bool,
) -> Committed {
    let dims = grid.dims;
    let n_nets = nets.len();
    // Owning group per committed COPPER cell, or -1 for free: a cell is a foreign
    // HARD obstacle for net i iff its owner is a group other than i's.
    let mut owner: Vec<i64> = vec![-1; n_cells];
    // Owning group per committed clearance-HALO cell, or -1 for free: a foreign
    // halo cell is a HARD block in `route_legal` (the committing pass), so copper is
    // never placed inside another net's spacing — an unroutable net is dropped.
    let mut halo: Vec<i64> = vec![HALO_FREE; n_cells];
    // Exact feature-aware exclusion field for candidate via centres, separate from
    // the ordinary planar/via-to-track halo above.
    // The feature-aware field models via-pad-edge ↔ trace-edge spacing exactly.
    // Ordinary trace↔trace clearance remains in `halo`; taking the maximum here
    // would incorrectly widen via spacing whenever routed vias are narrower than
    // routed traces.
    let via_trace_keepout_mm = via_model.keepout_mm;
    let mut via_guard = ViaGuard::dense_with_planar_via_guard(
        dims,
        via_trace_keepout_mm,
        via_spacing_mm,
        via_hole_spacing_mm,
        protect_planar_from_vias,
    );
    let mut committed: Committed = vec![None; n_nets];

    // Net indices committed by the group currently being placed; their paths are
    // folded into `owner` (with clearance halos) only after the whole group
    // commits, so sibling sub-nets never block each other. Tracked as a list
    // (cleared per group) to avoid an O(n_cells) sweep.
    let mut group_members: Vec<usize> = Vec::new();

    // ---- Serial-equivalent staging for within-candidate parallelism ----
    //
    // Groups commit sequentially because each group's copper blocks later ones. But
    // two groups whose search regions are spatially DISJOINT can never affect each
    // other's *windowed* route, so they can be routed in PARALLEL and committed in any
    // order with the identical result. We assign each group a stage = 1 + the max
    // stage of any earlier-order group it spatially conflicts with (bounding boxes,
    // inflated by the clearance/via-keepout radius). Groups in the same stage are
    // mutually disjoint; we route all their nets in parallel against the ownership
    // snapshot from prior stages, then commit per group in `group_order`
    // (deterministic).
    //
    // Windowed routes are byte-identical to the sequential router (a disjoint group's
    // cells are never in this net's window, so its commit order is irrelevant). The
    // rare full-board FALLBACK reads outside the window, so those nets are handled
    // SERIALLY in stage order (phase B) — deterministic, and they only differ from the
    // pure-sequential order in the uncommon global-net case, which the best-of-orders
    // pass and rip-up downstream absorb.
    let n_groups_total = group_ids.iter().map(|&g| g + 1).max().unwrap_or(0);
    let mut group_nets: Vec<Vec<usize>> = vec![Vec::new(); n_groups_total];
    for i in 0..n_nets {
        group_nets[group_ids[i]].push(i);
    }
    // Group bounding boxes (union of member windows). `None` = group with no nets.
    let gbox: Vec<Option<(u32, u32, u32, u32)>> = (0..n_groups_total)
        .map(|g| {
            let mut members = group_nets[g].iter();
            let first = members.next()?;
            let w = &windows[*first];
            let mut b = (w.x0, w.y0, w.x1, w.y1);
            for &i in members {
                let w = &windows[i];
                b.0 = b.0.min(w.x0);
                b.1 = b.1.min(w.y0);
                b.2 = b.2.max(w.x1);
                b.3 = b.3.max(w.y1);
            }
            Some(b)
        })
        .collect();
    // Stage conflicts are geometric, not cell-count based.  On a dense Hanan grid
    // `ceil(0.5 mm) == 1 cell` can be far smaller than 0.5 mm (dozens of lines),
    // which previously co-scheduled clearance-conflicting groups and accepted both
    // precomputed paths without revalidation.  Expand each window's physical box
    // by the actual clearance / via radius instead.
    let infl = clearance
        .max(via_model.keepout_mm)
        .max(via_spacing_mm)
        .max(via_hole_spacing_mm);
    let conflict = |a: usize, b: usize| -> bool {
        match (gbox[a], gbox[b]) {
            (Some(a), Some(b)) => {
                let ax0 = coords.x_of(a.0).min(coords.x_of(a.2)) - infl;
                let ay0 = coords.y_of(a.1).min(coords.y_of(a.3)) - infl;
                let ax1 = coords.x_of(a.0).max(coords.x_of(a.2)) + infl;
                let ay1 = coords.y_of(a.1).max(coords.y_of(a.3)) + infl;
                let bx0 = coords.x_of(b.0).min(coords.x_of(b.2)) - infl;
                let by0 = coords.y_of(b.1).min(coords.y_of(b.3)) - infl;
                let bx1 = coords.x_of(b.0).max(coords.x_of(b.2)) + infl;
                let by1 = coords.y_of(b.1).max(coords.y_of(b.3)) + infl;
                !(ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0)
            }
            _ => false,
        }
    };
    let mut stage: Vec<usize> = vec![0; n_groups_total];
    for (oi, &g) in group_order.iter().enumerate() {
        let mut s = 0;
        for &g2 in &group_order[..oi] {
            if conflict(g, g2) {
                s = s.max(stage[g2] + 1);
            }
        }
        stage[g] = s;
    }
    let max_stage = group_order.iter().map(|&g| stage[g]).max().unwrap_or(0);

    for s in 0..=max_stage {
        let batch: Vec<usize> = group_order
            .iter()
            .copied()
            .filter(|&g| stage[g] == s)
            .collect();
        if batch.is_empty() {
            continue;
        }
        // Nets of this stage in (group_order-within-batch, input) order, matching the
        // phase-B iteration below so the parallel results line up by position.
        let batch_nets: Vec<usize> = batch
            .iter()
            .flat_map(|&g| group_nets[g].iter().copied())
            .collect();
        // Phase A (parallel): clean-reuse or WINDOWED route against the ownership
        // snapshot from prior stages. Per-thread scratch via `map_init`; reads only
        // `&` snapshots.
        let owner_ref: &[i64] = &owner;
        let halo_ref: &[i64] = &halo;
        let via_guard_ref = &via_guard;
        let coords_ref = coords;
        let mut phase_a: Vec<Option<Vec<CellIdx>>> = batch_nets
            .par_iter()
            .map_init(
                || (SearchBuf::new(n_cells), PadSet::new(n_cells)),
                |(b, ps), &i| {
                    let gi = group_ids[i] as i64;
                    let net = &nets[i];
                    let cur = &paths[i];
                    let clean = !cur.is_empty()
                        && cur.iter().all(|&c| {
                            let o = owner_ref[c as usize];
                            let h = halo_ref[c as usize];
                            (o < 0 || o == gi) && !halo_is_foreign(h, gi)
                        })
                        && !path_has_foreign_via_conflict(dims, coords_ref, via_guard_ref, cur, gi);
                    if clean {
                        Some(cur.clone())
                    } else {
                        ps.load(&net.passable_pads);
                        route_legal(
                            b,
                            grid,
                            coords_ref,
                            heuristic_costs,
                            ps,
                            &net.via_passable_pads,
                            owner_ref,
                            halo_ref,
                            via_guard_ref,
                            gi,
                            net.src,
                            net.dst,
                            windows[i],
                            via_model,
                            clearance,
                            has_zero_cost,
                        )
                        .map(|(p, _)| p)
                    }
                },
            )
            .collect();
        // Phase B (serial, group_order): commit; nets that failed the windowed route
        // get the full-board fallback now, against owner incl. earlier same-stage
        // groups. Stamp each group's ownership only after the whole group commits.
        let mut k = 0;
        for &g in &batch {
            let gi = g as i64;
            for &i in &group_nets[g] {
                let chosen = match std::mem::take(&mut phase_a[k]) {
                    Some(p) if !path_has_foreign_via_conflict(dims, coords, &via_guard, &p, gi) => {
                        Some(p)
                    }
                    Some(_) | None => {
                        pad_set.load(&nets[i].passable_pads);
                        route_legal(
                            buf,
                            grid,
                            coords,
                            heuristic_costs,
                            pad_set,
                            &nets[i].via_passable_pads,
                            &owner,
                            &halo,
                            &via_guard,
                            gi,
                            nets[i].src,
                            nets[i].dst,
                            Window::full(dims),
                            via_model,
                            clearance,
                            has_zero_cost,
                        )
                        .map(|(p, _)| p)
                    }
                };
                if let Some(path) = chosen {
                    // Drill spacing is a fabrication rule, not an electrical-net
                    // rule. Stamp only hole sites immediately so the next sibling
                    // path is validated against them, while copper owner/halo maps
                    // still wait for the whole group as required for shared routing.
                    via_guard.stamp_holes_path(dims, coords, &path, gi);
                    committed[i] = Some(path);
                    group_members.push(i);
                }
                k += 1;
            }
            for &i in &group_members {
                if let Some(path) = &committed[i] {
                    stamp_owner(
                        &mut owner,
                        &mut halo,
                        &mut via_guard,
                        grid,
                        dims,
                        coords,
                        path,
                        gi,
                        clearance,
                        via_model,
                    );
                }
            }
            group_members.clear();
        }
    }

    committed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyInsert {
    Inserted,
    Existing,
    WouldCycle,
}

/// Add `from -> to` only when doing so preserves an acyclic dependency graph.
/// The edge means `from` must be legalized before its blocking owner `to`.
fn add_acyclic_dependency(
    dependencies: &mut [Vec<usize>],
    from: usize,
    to: usize,
) -> DependencyInsert {
    if dependencies[from].contains(&to) {
        return DependencyInsert::Existing;
    }
    let mut seen = vec![false; dependencies.len()];
    let mut stack = vec![to];
    while let Some(node) = stack.pop() {
        if node == from {
            return DependencyInsert::WouldCycle;
        }
        if seen[node] {
            continue;
        }
        seen[node] = true;
        stack.extend(dependencies[node].iter().copied());
    }
    dependencies[from].push(to);
    dependencies[from].sort_unstable();
    DependencyInsert::Inserted
}

/// Stable topological order for the observed failed-group-to-blocker graph.
/// Unconstrained ties retain the winning seed order and then the group id.
fn dependency_guided_order(seed_order: &[usize], dependencies: &[Vec<usize>]) -> Vec<usize> {
    let n_groups = dependencies.len();
    let mut rank = vec![usize::MAX; n_groups];
    for (i, &group) in seed_order.iter().enumerate() {
        rank[group] = i;
    }
    let mut indegree = vec![0usize; n_groups];
    for targets in dependencies {
        for &target in targets {
            indegree[target] += 1;
        }
    }
    let mut emitted = vec![false; n_groups];
    let mut order = Vec::with_capacity(n_groups);
    while order.len() < n_groups {
        let next = (0..n_groups)
            .filter(|&group| !emitted[group] && indegree[group] == 0)
            .min_by_key(|&group| (rank[group], group));
        let Some(group) = next else {
            debug_assert!(false, "dependency insertion must preserve a DAG");
            order.extend(seed_order.iter().copied().filter(|&g| !emitted[g]));
            break;
        };
        emitted[group] = true;
        order.push(group);
        for &target in &dependencies[group] {
            indegree[target] -= 1;
        }
    }
    order
}

fn dependency_guided_restart_order(
    seed_order: &[usize],
    dependencies: &[Vec<usize>],
) -> Option<Vec<usize>> {
    if dependencies.len() != seed_order.len()
        || !dependencies.iter().any(|targets| !targets.is_empty())
    {
        return None;
    }
    let order = dependency_guided_order(seed_order, dependencies);
    (order != seed_order).then_some(order)
}

#[inline]
fn guided_candidate_is_better(guided: &Committed, current: &Committed) -> bool {
    guided.iter().filter(|path| path.is_some()).count()
        > current.iter().filter(|path| path.is_some()).count()
}

#[inline]
fn should_collect_guided_dependencies(n_groups: usize, n_nets: usize, n_cells: usize) -> bool {
    n_groups <= GUIDED_RESTART_MAX_GROUPS
        && n_nets <= GUIDED_RESTART_MAX_NETS
        && n_cells <= GUIDED_RESTART_MAX_CELLS
}

/// Bounded rip-up-and-reroute legalization.
///
/// Produces a cell-disjoint-across-groups commit that, unlike [`legalize_in_order`],
/// may *displace* an already-committed net to make room for a stranded one. This
/// solves the co-dependent case where net A's natural route blocks B and B's blocks
/// A simultaneously — no static commit order works, you must rip up a committed net.
///
/// Algorithm (work-queue):
/// 1. Seed the queue with every net in `seed_group_order` (the BEST multi-order
///    order, so only the genuinely-hard leftovers cost extra work). Within a group,
///    nets are queued in input order.
/// 2. Pop net `i`. Build a grid where cells owned by OTHER groups are hard obstacles
///    (own pads unmasked), then route `i`. If it routes, commit it (mark its cells
///    owned by `i`'s group). If not, find the committed OTHER-group nets whose cells
///    lie on `i`'s alone path (the blockers); rip up the blocker(s) with the
///    smallest committed path length first (cheapest to re-place), free their cells,
///    commit `i`, and re-enqueue the ripped nets (bumping their rip count).
/// 3. Bounds: a global rip budget and a per-net rip cap; when exhausted, stop
///    ripping and place whatever still fits. ALWAYS terminates.
///
/// Determinism: every tie is broken by `(group_id, net index, cell idx)`; no RNG.
/// Sibling sub-nets of one connection (same group) never block each other because
/// occupancy is tracked per *group*, never per net.
#[allow(clippy::too_many_arguments)]
fn ripup_legalize(
    grid: &Grid,
    coords: &GridCoords,
    heuristic_costs: &ManhattanCosts,
    buf: &mut SearchBuf,
    pad_set: &mut PadSet,
    nets: &[NetEndpoints],
    group_ids: &[usize],
    alone_path: &[Vec<CellIdx>],
    negotiated_paths: &[Vec<CellIdx>],
    windows: &[Window],
    seed_group_order: &[usize],
    seed_committed: &Committed,
    n_cells: usize,
    via_model: &ViaModel,
    clearance: f64,
    via_spacing_mm: f64,
    via_hole_spacing_mm: f64,
    protect_planar_from_vias: bool,
    has_zero_cost: bool,
) -> Committed {
    let dims = grid.dims;
    let n_nets = nets.len();
    let n_groups = group_ids.iter().copied().max().map_or(0, |g| g + 1);
    let collect_dependencies = should_collect_guided_dependencies(n_groups, n_nets, n_cells);
    let mut dependencies = if collect_dependencies {
        vec![Vec::<usize>::new(); n_groups]
    } else {
        Vec::new()
    };

    // A net needs the (expensive) full-board fallback only if its own alone-path
    // genuinely leaves its window. Nets whose alone-path fits the window never gain
    // anything from a full-board search (which, when it fails, explores the whole
    // board) — so we suppress the fallback for them. Nets with an EMPTY alone-path
    // are individually unroutable on the base grid and can never commit at all.
    let needs_full: Vec<bool> = (0..n_nets)
        .map(|i| {
            !alone_path[i].is_empty()
                && alone_path[i].iter().any(|&c| !windows[i].contains(dims, c))
        })
        .collect();
    let routable: Vec<bool> = (0..n_nets).map(|i| !alone_path[i].is_empty()).collect();

    // SEED from the best multi-order legalization instead of re-routing every net
    // from scratch: the parallel multi-order pass already committed most nets, so we
    // start from its solution and only work the genuinely-stranded residue. This
    // skips the (serial, ~O(n_nets) `route_legal`) clean-route pass that otherwise
    // dominated this stage. Owner/halo below are stamped to match the seed. The result
    // can only ADD nets (via rips), so it never routes fewer than the seed.
    let mut committed: Committed = seed_committed.clone();
    // Owning group per committed COPPER cell, or -1 for free: a foreign-group copper
    // cell is a HARD obstacle for net i. Owning group per clearance-HALO cell, or -1:
    // a foreign-group halo cell is ALSO a HARD block in `route_legal`, so a stranded
    // net that cannot route clear is left unrouted rather than violating spacing.
    let mut owner: Vec<i64> = vec![-1; n_cells];
    let mut halo: Vec<i64> = vec![HALO_FREE; n_cells];
    let mut via_guard = ViaGuard::dense_with_planar_via_guard(
        dims,
        via_model.keepout_mm,
        via_spacing_mm,
        via_hole_spacing_mm,
        protect_planar_from_vias,
    );
    // Stamp the seeded commits into all ownership maps so the residue routes
    // against them.
    rebuild_owner_maps(
        &mut owner,
        &mut halo,
        &mut via_guard,
        grid,
        coords,
        &committed,
        group_ids,
        clearance,
        via_model,
    );

    let mut rip_count: Vec<usize> = vec![0; n_nets];
    let global_budget = RIPUP_GLOBAL_BUDGET_PER_NET * n_nets.max(1);
    let per_net_cap = n_nets + RIPUP_PER_NET_CAP_EXTRA;
    let mut rips_done: usize = 0;

    // FIFO work-queue of net indices to (re)place. Seed in best-order, groups in
    // the seed order and nets within a group in input order, for determinism.
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for &g in seed_group_order {
        for (i, &gi) in group_ids.iter().enumerate() {
            if gi == g {
                queue.push_back(i);
            }
        }
    }
    // ROUND-BASED NO-PROGRESS EARLY-EXIT. The rip-up loop otherwise grinds until the
    // global rip budget (thousands of rips) is exhausted, because a few mutually-
    // contended nets ping-pong (A rips B, re-enqueued B rips A, …) — each cycle an
    // expensive *failing* A* that commits nothing net-new. We mark round boundaries
    // with a sentinel pushed to the back of the queue: re-enqueues during a round
    // land BEHIND the sentinel (next round). When the sentinel surfaces we compare
    // the committed count to the best seen; a round that did not strictly increase it
    // cannot make progress on an identical later pass, so we stop. Productive rounds
    // (which place a net) strictly raise the count, so there are at most `n_nets` of
    // them → fast, bounded termination. The `accept-if-better` gate at the call site
    // means stopping early can at worst fall back to `multi_committed` (never worse).
    // Deterministic: the sentinel and count are pure functions of queue contents.
    const ROUND_MARK: usize = usize::MAX;
    // How many consecutive break-even rounds (committed count unchanged) to tolerate
    // before giving up. A genuine rescue rips a blocker in one round (count breaks
    // even: −victim +stranded) and re-places the blocker in the NEXT round (count
    // rises), so we must allow at least one break-even round; oscillating ping-pong
    // also breaks even, so we cap it. Two is the minimum that admits a one-round
    // rip→replace gap while still bounding the ping-pong.
    const MAX_EVEN_ROUNDS: usize = 2;
    queue.push_back(ROUND_MARK);
    let mut best_count = committed.iter().filter(|c| c.is_some()).count();
    let mut even_rounds = 0usize;

    // Free every committed path in group `g`, then rebuild all ownership maps.
    // Rebuilding is necessary because a halo cell can be shared by several groups:
    // deleting a first-owner scalar in place would lose a surviving reservation.
    let free_group_cells = |owner: &mut [i64],
                            halo: &mut [i64],
                            via_guard: &mut ViaGuard,
                            committed: &mut Committed,
                            group_ids: &[usize],
                            g: usize| {
        for i in 0..committed.len() {
            if group_ids[i] == g {
                committed[i] = None;
            }
        }
        rebuild_owner_maps(
            owner, halo, via_guard, grid, coords, committed, group_ids, clearance, via_model,
        );
    };

    while let Some(i) = queue.pop_front() {
        // Round boundary: stop unless this round committed strictly more nets.
        if i == ROUND_MARK {
            if queue.is_empty() {
                break;
            }
            let cur = committed.iter().filter(|c| c.is_some()).count();
            if cur > best_count {
                // Progress: a new high-water of committed nets. Keep going.
                best_count = cur;
                even_rounds = 0;
                queue.push_back(ROUND_MARK);
                continue;
            }
            if cur == best_count && even_rounds + 1 < MAX_EVEN_ROUNDS {
                // Break-even: could be a rip awaiting its re-placement next round.
                // Allow a bounded number before concluding it is unproductive churn.
                even_rounds += 1;
                queue.push_back(ROUND_MARK);
                continue;
            }
            // Net loss (rips freed more than they placed) or too many break-even
            // rounds: unproductive. Stop; the accept-if-better gate keeps the seed.
            break;
        }
        // If already committed (e.g. re-enqueued then satisfied earlier), skip.
        if committed[i].is_some() {
            continue;
        }
        // Individually unroutable on the base grid: it can never commit here, and
        // any search (especially a full-board one) would only fail expensively.
        if !routable[i] {
            continue;
        }
        let net = &nets[i];
        let g = group_ids[i];
        let gi = g as i64;

        // Route within the net's window (cells owned by OTHER groups are hard
        // obstacles), falling back to the full board only for nets whose natural
        // route genuinely leaves the window — so a stranded global net is never
        // ripped-around prematurely, without the whole-board failure cost for the
        // many purely-local nets.
        pad_set.load(&net.passable_pads);
        let routed = route_legal(
            buf,
            grid,
            coords,
            heuristic_costs,
            pad_set,
            &net.via_passable_pads,
            &owner,
            &halo,
            &via_guard,
            gi,
            net.src,
            net.dst,
            windows[i],
            via_model,
            clearance,
            has_zero_cost,
        )
        .or_else(|| {
            if needs_full[i] {
                route_legal(
                    buf,
                    grid,
                    coords,
                    heuristic_costs,
                    pad_set,
                    &net.via_passable_pads,
                    &owner,
                    &halo,
                    &via_guard,
                    gi,
                    net.src,
                    net.dst,
                    Window::full(dims),
                    via_model,
                    clearance,
                    has_zero_cost,
                )
            } else {
                None
            }
        });

        if let Some((path, _)) = routed {
            stamp_owner(
                &mut owner,
                &mut halo,
                &mut via_guard,
                grid,
                dims,
                coords,
                &path,
                gi,
                clearance,
                via_model,
            );
            committed[i] = Some(path);
            continue;
        }

        // Stranded. Find committed OTHER-group blockers on i's alone path.
        if rips_done >= global_budget || rip_count[i] >= per_net_cap {
            // Budget exhausted for this net: leave it unrouted.
            continue;
        }

        // Collect the distinct blocker groups whose owned cells lie on i's alone
        // path. (Sibling cells of i's own group are never blockers.)
        let mut blocker_groups: Vec<usize> = Vec::new();
        for &c in &alone_path[i] {
            let o = owner[c as usize];
            if o >= 0 && o as usize != g {
                let bg = o as usize;
                if !blocker_groups.contains(&bg) {
                    blocker_groups.push(bg);
                }
            }
        }

        if blocker_groups.is_empty() {
            // Nothing committed is in the way along the natural route, yet it still
            // would not route — it is genuinely unroutable given current commits.
            continue;
        }

        // Rip the blocker group whose total committed path length is smallest
        // (cheapest to re-place); tie-break by lowest group id for determinism.
        blocker_groups.sort_by_key(|&bg| {
            let len: Cost = (0..n_nets)
                .filter(|&j| group_ids[j] == bg)
                .filter_map(|j| committed[j].as_ref())
                .map(|p| unit_cost(p))
                .fold(0, |a, b| a.saturating_add(b));
            (len, bg)
        });

        // Learn every exact failed-group -> blocking-owner dependency. Reject an
        // edge that would close a cycle so the graph can safely drive one stable
        // topological restart after the ordinary bounded FIFO attempt.
        if collect_dependencies {
            for &blocker in &blocker_groups {
                let _ = add_acyclic_dependency(&mut dependencies, g, blocker);
            }
        }

        let victim = blocker_groups[0];

        // Re-enqueue every (currently committed) net of the victim group, in input
        // order, bumping their rip count, then free the group's cells and commit i.
        for j in 0..n_nets {
            if group_ids[j] == victim && committed[j].is_some() {
                rip_count[j] += 1;
                queue.push_back(j);
            }
        }
        free_group_cells(
            &mut owner,
            &mut halo,
            &mut via_guard,
            &mut committed,
            group_ids,
            victim,
        );
        rips_done += 1;

        // Re-route i now that the victim's cells are free.
        let rerouted = route_legal(
            buf,
            grid,
            coords,
            heuristic_costs,
            pad_set,
            &net.via_passable_pads,
            &owner,
            &halo,
            &via_guard,
            gi,
            net.src,
            net.dst,
            windows[i],
            via_model,
            clearance,
            has_zero_cost,
        )
        .or_else(|| {
            if needs_full[i] {
                route_legal(
                    buf,
                    grid,
                    coords,
                    heuristic_costs,
                    pad_set,
                    &net.via_passable_pads,
                    &owner,
                    &halo,
                    &via_guard,
                    gi,
                    net.src,
                    net.dst,
                    Window::full(dims),
                    via_model,
                    clearance,
                    has_zero_cost,
                )
            } else {
                None
            }
        });
        if let Some((path, _)) = rerouted {
            stamp_owner(
                &mut owner,
                &mut halo,
                &mut via_guard,
                grid,
                dims,
                coords,
                &path,
                gi,
                clearance,
                via_model,
            );
            committed[i] = Some(path);
        } else {
            // Still cannot route even after the rip: re-enqueue i once (its rip
            // count is bounded by per_net_cap, so this terminates).
            rip_count[i] += 1;
            queue.push_back(i);
        }
    }

    // One bounded global restart through the existing legalization primitive.
    // It is evaluated only when blocker telemetry changed the winning seed order,
    // and can replace the FIFO result only on a strict completion gain.
    if let Some(guided_order) = dependency_guided_restart_order(seed_group_order, &dependencies) {
        let guided = legalize_in_order(
            grid,
            coords,
            heuristic_costs,
            buf,
            pad_set,
            nets,
            group_ids,
            negotiated_paths,
            windows,
            &guided_order,
            n_cells,
            via_model,
            clearance,
            via_spacing_mm,
            via_hole_spacing_mm,
            protect_planar_from_vias,
            has_zero_cost,
        );
        if guided_candidate_is_better(&guided, &committed) {
            committed = guided;
        }
    }

    committed
}

/// All permutations of `items`, generated in a deterministic order (Heap-free
/// lexicographic-by-construction recursion: index 0 fixed first, then 1, …).
/// Used only for small group counts (≤ 7), so allocating the full set is cheap.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::with_capacity(items.len());
    let mut used: Vec<bool> = vec![false; items.len()];
    permute_rec(items, &mut used, &mut cur, &mut out);
    out
}

fn permute_rec(
    items: &[usize],
    used: &mut [bool],
    cur: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if cur.len() == items.len() {
        out.push(cur.clone());
        return;
    }
    for i in 0..items.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        cur.push(items[i]);
        permute_rec(items, used, cur, out);
        cur.pop();
        used[i] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_grid::GridBuilder;

    enum MockProviderReply {
        Paths(Vec<Option<Vec<CellIdx>>>),
        Error,
    }

    struct MockProvider {
        calls: AtomicUsize,
        reply: MockProviderReply,
    }

    impl MockProvider {
        fn paths(paths: Vec<Option<Vec<CellIdx>>>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                reply: MockProviderReply::Paths(paths),
            }
        }

        fn error() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                reply: MockProviderReply::Error,
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl IsolatedRouteProvider for MockProvider {
        fn route_isolated_batch(
            &self,
            _request: IsolatedRouteRequest<'_>,
        ) -> Result<Vec<Option<Vec<CellIdx>>>, RouterError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match &self.reply {
                MockProviderReply::Paths(paths) => Ok(paths.clone()),
                MockProviderReply::Error => Err(RouterError::BackendUnavailable("mock".into())),
            }
        }
    }

    fn net(name: &str, src: CellIdx, dst: CellIdx) -> NetEndpoints {
        NetEndpoints {
            net: name.into(),
            src,
            dst,
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        }
    }

    fn disjoint(a: &[CellIdx], b: &[CellIdx]) -> bool {
        let sa: std::collections::HashSet<_> = a.iter().copied().collect();
        b.iter().all(|c| !sa.contains(c))
    }

    fn board_with_costs(costs: &[Cost], n_total: usize) -> BoardRoute {
        let results = costs
            .iter()
            .enumerate()
            .map(|(i, &cost)| RouteResult {
                net: format!("n{i}"),
                path: Vec::new(),
                cost,
            })
            .collect();
        BoardRoute {
            results,
            unrouted: (costs.len()..n_total).map(|i| format!("n{i}")).collect(),
            congestion: Vec::new(),
            groups: Vec::new(),
        }
    }

    fn assert_trace_eq(actual: &RouteTrace, expected: &RouteTrace) {
        assert_eq!(actual.dims, expected.dims);
        assert_eq!(actual.n_groups, expected.n_groups);
        assert_eq!(actual.nets.len(), expected.nets.len());
        for (a, b) in actual.nets.iter().zip(&expected.nets) {
            assert_eq!(
                (&a.net, a.src, a.dst, a.group, &a.alone_path),
                (&b.net, b.src, b.dst, b.group, &b.alone_path)
            );
        }
        assert_eq!(actual.iterations.len(), expected.iterations.len());
        for (a, b) in actual.iterations.iter().zip(&expected.iterations) {
            assert_eq!(
                (a.iter, a.pfac, a.any_overuse),
                (b.iter, b.pfac, b.any_overuse)
            );
            assert_eq!(a.paths, b.paths);
            assert_eq!(a.overused_cells, b.overused_cells);
        }
        let a = actual.legalization.as_ref().unwrap();
        let b = expected.legalization.as_ref().unwrap();
        assert_eq!(a.chosen_order, b.chosen_order);
        assert_eq!(a.committed, b.committed);
        assert_eq!(a.candidates.len(), b.candidates.len());
        for (a, b) in a.candidates.iter().zip(&b.candidates) {
            assert_eq!(
                (&a.order, a.routed, a.total_cost),
                (&b.order, b.routed, b.total_cost)
            );
        }
    }

    fn provider_paths_from_trace(trace: &RouteTrace) -> Vec<Option<Vec<CellIdx>>> {
        trace
            .nets
            .iter()
            .map(|net| (!net.alone_path.is_empty()).then(|| net.alone_path.clone()))
            .collect()
    }

    #[test]
    fn isolated_provider_matches_cpu_for_reachable_unreachable_and_zero_hop_nets() {
        let dims = Dims::new(5, 5);
        let mut builder = GridBuilder::new(dims, 1);
        for (x, y) in [(2, 1), (1, 2), (3, 2), (2, 3)] {
            builder.mark_cell(x, y);
        }
        let grid = builder.build();
        let nets = vec![
            net("open", dims.idx(0, 4), dims.idx(4, 4)),
            net("blocked", dims.idx(0, 0), dims.idx(2, 2)),
            net("zero", dims.idx(4, 0), dims.idx(4, 0)),
        ];
        let router = NegotiatedRouter::new();
        let cpu_outcome = router.route_with_outcome(&grid, &nets).unwrap();
        let (cpu_board, cpu_trace) = router.route_traced(&grid, &nets).unwrap();
        assert_eq!(cpu_outcome.board, cpu_board);
        assert_eq!(cpu_outcome.alone_routable, [true, false, true]);
        assert_eq!(cpu_trace.nets[1].alone_path, Vec::<CellIdx>::new());
        assert_eq!(cpu_trace.nets[2].alone_path, vec![nets[2].src]);

        let provider = MockProvider::paths(provider_paths_from_trace(&cpu_trace));
        let accelerated = router
            .route_with_isolated_provider(&grid, &nets, &provider)
            .unwrap();
        assert_eq!(accelerated, cpu_outcome);

        let (traced_board, accelerated_trace) = router
            .route_traced_with_isolated_provider(&grid, &nets, &provider)
            .unwrap();
        assert_eq!(traced_board, cpu_board);
        assert_trace_eq(&accelerated_trace, &cpu_trace);
        assert_eq!(provider.calls(), 2, "one batch call per public route call");
    }

    #[test]
    fn isolated_provider_rejection_falls_back_for_the_whole_batch() {
        let dims = Dims::new(5, 5);
        let grid = Grid::filled(dims, 1);
        let nets = vec![
            net("top", dims.idx(0, 0), dims.idx(4, 0)),
            net("bottom", dims.idx(0, 4), dims.idx(4, 4)),
        ];
        let router = NegotiatedRouter::new();
        let (expected_board, expected_trace) = router.route_traced(&grid, &nets).unwrap();

        let providers = [
            MockProvider::error(),
            MockProvider::paths(vec![Some(vec![nets[0].src, nets[0].dst])]),
            // The first entry is a legal but deliberately non-canonical detour;
            // the second jumps four cells and invalidates the batch. If fallback
            // were per-entry, the trace would expose the detour for `top`.
            MockProvider::paths(vec![
                Some(vec![0, 5, 6, 7, 8, 9, 4]),
                Some(vec![nets[1].src, nets[1].dst]),
            ]),
        ];

        for provider in &providers {
            let (board, trace) = router
                .route_traced_with_isolated_provider(&grid, &nets, provider)
                .unwrap();
            assert_eq!(board, expected_board);
            assert_trace_eq(&trace, &expected_trace);
            assert_eq!(provider.calls(), 1);
        }
    }

    #[test]
    fn isolated_provider_path_validation_covers_obstacles_repeats_and_vias() {
        let planar_dims = Dims::new(3, 2);
        let mut planar = Grid::filled(planar_dims, 1);
        planar.set(planar_dims.idx(1, 0), OBSTACLE);
        let planar_net = net("p", planar_dims.idx(0, 0), planar_dims.idx(2, 0));
        let through = ViaModel::through_hole(1);
        assert!(!provider_path_is_valid(
            &planar,
            &planar_net,
            &through,
            &[planar_net.src, planar_dims.idx(1, 0), planar_net.dst]
        ));
        assert!(!provider_path_is_valid(
            &Grid::filled(planar_dims, 1),
            &planar_net,
            &through,
            &[
                planar_net.src,
                planar_dims.idx(0, 1),
                planar_net.src,
                planar_net.dst
            ]
        ));

        let via_dims = Dims::with_layers(1, 1, 2);
        let via_grid = Grid::filled(via_dims, 1);
        let via_net = net("v", via_dims.idx3(0, 0, 0), via_dims.idx3(0, 0, 1));
        let forbidden = ViaModel::with_allowed_steps(2, 7, Vec::new());
        assert!(!provider_path_is_valid(
            &via_grid,
            &via_net,
            &forbidden,
            &[via_net.src, via_net.dst]
        ));

        let mut masked_grid = via_grid.clone();
        masked_grid.via_forbidden = vec![true; via_dims.len()];
        let owned_via = NetEndpoints {
            passable_pads: vec![via_net.dst, via_net.src],
            // Deliberately unsorted to pin the public input contract.
            via_passable_pads: vec![via_net.dst, via_net.src],
            ..via_net.clone()
        };
        assert!(provider_path_is_valid(
            &masked_grid,
            &owned_via,
            &ViaModel::through_hole(2),
            &[owned_via.src, owned_via.dst]
        ));
        let layer_local_only = NetEndpoints {
            via_passable_pads: vec![via_net.src],
            ..owned_via
        };
        assert!(!provider_path_is_valid(
            &masked_grid,
            &layer_local_only,
            &ViaModel::through_hole(2),
            &[layer_local_only.src, layer_local_only.dst]
        ));
    }

    #[test]
    fn unsorted_large_via_exemption_list_is_normalized_once_and_routes_identically() {
        let dims = Dims::with_layers(1, 1, 2);
        let mut grid = Grid::filled(dims, 1);
        grid.via_forbidden = vec![true; dims.len()];
        let src = dims.idx3(0, 0, 0);
        let dst = dims.idx3(0, 0, 1);

        let mut repeated_unsorted = Vec::with_capacity(80_000);
        for _ in 0..40_000 {
            repeated_unsorted.extend([dst, src]);
        }
        let unsorted = NetEndpoints {
            net: "via".into(),
            src,
            dst,
            passable_pads: vec![src, dst],
            via_passable_pads: repeated_unsorted,
        };
        let normalized = normalize_via_exemptions(std::slice::from_ref(&unsorted));
        assert!(matches!(normalized, Cow::Owned(_)));
        assert_eq!(normalized[0].via_passable_pads, [src, dst]);

        let canonical = NetEndpoints {
            via_passable_pads: vec![src, dst],
            ..unsorted.clone()
        };
        let router = NegotiatedRouter::new();
        let expected = router.route(&grid, &[canonical]).unwrap();
        let actual = router.route(&grid, &[unsorted]).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.results[0].path, [src, dst]);
    }

    #[test]
    fn via_exemption_lookup_is_logarithmic_by_comparison_count() {
        #[derive(Clone)]
        struct CountedCell {
            value: CellIdx,
            comparisons: std::sync::Arc<AtomicUsize>,
        }

        impl PartialEq for CountedCell {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == std::cmp::Ordering::Equal
            }
        }
        impl Eq for CountedCell {}
        impl PartialOrd for CountedCell {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for CountedCell {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.comparisons.fetch_add(1, Ordering::Relaxed);
                self.value.cmp(&other.value)
            }
        }

        let comparisons = std::sync::Arc::new(AtomicUsize::new(0));
        let cells: Vec<_> = (0..80_000)
            .map(|value| CountedCell {
                value,
                comparisons: comparisons.clone(),
            })
            .collect();
        let needle = CountedCell {
            value: 79_999,
            comparisons: comparisons.clone(),
        };
        assert!(sorted_contains(&cells, &needle));
        let count = comparisons.load(Ordering::Relaxed);
        assert!(
            count <= 18,
            "80k-entry lookup must remain logarithmic, got {count} comparisons"
        );
    }

    struct InspectingFallbackProvider {
        calls: AtomicUsize,
    }

    impl IsolatedRouteProvider for InspectingFallbackProvider {
        fn route_isolated_batch(
            &self,
            request: IsolatedRouteRequest<'_>,
        ) -> Result<Vec<Option<Vec<CellIdx>>>, RouterError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(request.grid.dims, Dims::with_layers(4, 3, 3));
            assert_eq!(request.nets.len(), 1);
            assert_eq!(
                request.windows,
                [IsolatedRouteWindow::full(request.grid.dims)]
            );
            // Short coordinate arrays use GridCoords' exact index fallback on
            // each endpoint of each gap, not a separate uniform approximation.
            assert_eq!(request.x_edge_costs, [144, 16, 16]);
            assert_eq!(request.y_edge_costs, [16, 16]);
            assert_eq!(request.via_edge_costs, [Some(7), None]);
            Err(RouterError::BackendUnavailable(
                "inspect then fallback".into(),
            ))
        }
    }

    #[test]
    fn isolated_provider_request_uses_exact_short_coordinate_and_via_costs() {
        let dims = Dims::with_layers(4, 3, 3);
        let grid = Grid::filled(dims, 1);
        let nets = vec![net("n", dims.idx3(0, 0, 0), dims.idx3(3, 2, 0))];
        let router = NegotiatedRouter::new()
            .with_coords(GridCoords::from_lines(vec![10.0], Vec::new()))
            .with_via_model(ViaModel::with_allowed_steps(3, 7, vec![(0, 1)]));
        let provider = InspectingFallbackProvider {
            calls: AtomicUsize::new(0),
        };

        let expected = router.route_with_outcome(&grid, &nets).unwrap();
        let actual = router
            .route_with_isolated_provider(&grid, &nets, &provider)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn serial_portfolio_score_is_completion_then_u64_cost_with_primary_ties() {
        let primary = board_with_costs(&[u32::MAX, u32::MAX], 3);
        let fewer = board_with_costs(&[1], 3);
        let equal_but_cheaper = board_with_costs(&[u32::MAX, u32::MAX - 1], 3);
        let equal_exact = primary.clone();
        let more = board_with_costs(&[u32::MAX, u32::MAX, u32::MAX], 3);

        assert!(!serial_candidate_is_better(&primary, &fewer));
        assert!(serial_candidate_is_better(&primary, &equal_but_cheaper));
        assert!(!serial_candidate_is_better(&primary, &equal_exact));
        assert!(serial_candidate_is_better(&primary, &more));
        assert_eq!(primary.total_cost(), 2 * u32::MAX as u64);
    }

    #[test]
    fn legalization_candidate_score_distinguishes_totals_above_cost_max() {
        let dims = Dims::new(3, 2);
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx(1, 0), 100);
        grid.set(dims.idx(2, 0), Cost::MAX / 2);
        let nets = vec![net("a", 0, 2), net("b", 0, 2)];

        let costly: Committed = vec![Some(vec![0, 1, 2]), Some(vec![0, 1, 2])];
        let cheaper: Committed = vec![Some(vec![0, 3, 4, 5, 2]), Some(vec![0, 3, 4, 5, 2])];
        let costly_total = committed_grid_cost(&grid, &nets, &costly);
        let cheaper_total = committed_grid_cost(&grid, &nets, &cheaper);

        assert!(cheaper_total > Cost::MAX as u64);
        assert!(cheaper_total < costly_total);
        // The stable trace field must narrow both totals, but candidate selection
        // must still prefer the genuinely cheaper later (lexicographically larger)
        // order using the full-width totals.
        assert_eq!(costly_total.min(Cost::MAX as u64) as Cost, Cost::MAX);
        assert_eq!(cheaper_total.min(Cost::MAX as u64) as Cost, Cost::MAX);
        assert!(legalization_candidate_is_better(
            2,
            cheaper_total,
            &[1, 0],
            2,
            costly_total,
            &[0, 1],
        ));
    }

    #[test]
    fn diversified_candidate_requires_strict_completion_gain() {
        let original_order = [0, 1, 2];
        let fallback_order = [1, 2, 0];

        assert!(!diversified_candidate_is_better(
            2,
            1,
            &fallback_order,
            2,
            2,
            10,
            &original_order,
        ));
        assert!(diversified_candidate_is_better(
            3,
            100,
            &fallback_order,
            2,
            2,
            10,
            &original_order,
        ));
    }

    #[test]
    fn diversified_fallback_orders_are_bounded_unique_dihedral_samples() {
        let base: Vec<usize> = (0..8).collect();
        let primary = vec![base.clone()];
        let expected = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 0],
            vec![7, 0, 1, 2, 3, 4, 5, 6],
            vec![7, 6, 5, 4, 3, 2, 1, 0],
            vec![3, 2, 1, 0, 7, 6, 5, 4],
        ];
        assert_eq!(diversified_fallback_orders(&base, &primary), expected);

        for n_groups in ORDER_PORTFOLIO_MIN_GROUPS..=ORDER_PORTFOLIO_MAX_GROUPS {
            let base: Vec<usize> = (0..n_groups).collect();
            let orders = diversified_fallback_orders(&base, std::slice::from_ref(&base));
            assert!(orders.len() <= 4);
            for (i, order) in orders.iter().enumerate() {
                assert!(
                    !orders[..i].contains(order),
                    "fallback orders must be unique"
                );
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, base, "every fallback must remain a permutation");
            }
        }

        assert!(diversified_fallback_orders(&[], &[]).is_empty());
        assert!(diversified_fallback_orders(&[0], &[vec![0]]).is_empty());
        assert!(diversified_fallback_orders(&base, &expected).is_empty());
    }

    #[test]
    fn dependency_graph_deduplicates_edges_and_rejects_cycles() {
        let mut dependencies = vec![Vec::new(); 4];
        assert_eq!(
            add_acyclic_dependency(&mut dependencies, 0, 2),
            DependencyInsert::Inserted
        );
        assert_eq!(
            add_acyclic_dependency(&mut dependencies, 0, 2),
            DependencyInsert::Existing
        );
        assert_eq!(
            add_acyclic_dependency(&mut dependencies, 2, 3),
            DependencyInsert::Inserted
        );
        let before = dependencies.clone();
        assert_eq!(
            add_acyclic_dependency(&mut dependencies, 3, 0),
            DependencyInsert::WouldCycle
        );
        assert_eq!(
            dependencies, before,
            "rejected edges must not mutate the DAG"
        );
        assert_eq!(
            add_acyclic_dependency(&mut dependencies, 1, 1),
            DependencyInsert::WouldCycle
        );
    }

    #[test]
    fn dependency_order_respects_edges_and_preserves_seed_ties() {
        let seed = vec![0, 1, 2, 3];
        let mut dependencies = vec![Vec::new(); seed.len()];
        assert_eq!(
            add_acyclic_dependency(&mut dependencies, 2, 0),
            DependencyInsert::Inserted
        );
        assert_eq!(
            dependency_guided_order(&seed, &dependencies),
            vec![1, 2, 0, 3],
            "unconstrained ready groups retain seed rank while 2 precedes 0"
        );
        assert_eq!(
            dependency_guided_order(&[3, 2, 1, 0], &vec![Vec::new(); 4]),
            vec![3, 2, 1, 0],
            "an empty dependency graph is byte-order inert"
        );
    }

    #[test]
    fn guided_restart_is_bounded_and_requires_strict_completion_gain() {
        assert!(should_collect_guided_dependencies(
            GUIDED_RESTART_MAX_GROUPS,
            GUIDED_RESTART_MAX_NETS,
            GUIDED_RESTART_MAX_CELLS,
        ));
        assert!(should_collect_guided_dependencies(168, 322, 1_294_488));

        let just_over_cap = [
            (
                GUIDED_RESTART_MAX_GROUPS + 1,
                GUIDED_RESTART_MAX_NETS,
                GUIDED_RESTART_MAX_CELLS,
            ),
            (
                GUIDED_RESTART_MAX_GROUPS,
                GUIDED_RESTART_MAX_NETS + 1,
                GUIDED_RESTART_MAX_CELLS,
            ),
            (
                GUIDED_RESTART_MAX_GROUPS,
                GUIDED_RESTART_MAX_NETS,
                GUIDED_RESTART_MAX_CELLS + 1,
            ),
        ];
        for (n_groups, n_nets, n_cells) in just_over_cap {
            assert!(!should_collect_guided_dependencies(
                n_groups, n_nets, n_cells
            ));
            let seed: Vec<usize> = (0..n_groups).collect();
            assert_eq!(
                dependency_guided_restart_order(&seed, &[]),
                None,
                "an over-cap route carries no dependency table and must remain inert"
            );
        }

        let one = vec![Some(vec![0]), None];
        let equal_but_different = vec![None, Some(vec![1])];
        let two = vec![Some(vec![0]), Some(vec![1])];
        assert!(!guided_candidate_is_better(&equal_but_different, &one));
        assert!(!guided_candidate_is_better(&one, &one));
        assert!(guided_candidate_is_better(&two, &one));
    }

    #[test]
    fn diversified_order_trigger_is_progress_and_resource_bounded() {
        let eligible = |groups, nets, cells, routed, alone| {
            should_try_diversified_orders(groups, nets, cells, routed, alone)
        };

        assert!(eligible(
            ORDER_PORTFOLIO_MIN_GROUPS,
            ORDER_PORTFOLIO_MAX_NETS,
            ORDER_PORTFOLIO_CELL_CAP,
            5,
            6,
        ));
        assert!(!eligible(
            ORDER_PORTFOLIO_MIN_GROUPS - 1,
            ORDER_PORTFOLIO_MAX_NETS,
            ORDER_PORTFOLIO_CELL_CAP,
            5,
            6,
        ));
        assert!(!eligible(
            ORDER_PORTFOLIO_MAX_GROUPS + 1,
            ORDER_PORTFOLIO_MAX_NETS,
            ORDER_PORTFOLIO_CELL_CAP,
            5,
            6,
        ));
        assert!(!eligible(
            ORDER_PORTFOLIO_MIN_GROUPS,
            ORDER_PORTFOLIO_MAX_NETS + 1,
            ORDER_PORTFOLIO_CELL_CAP,
            5,
            6,
        ));
        assert!(!eligible(
            ORDER_PORTFOLIO_MIN_GROUPS,
            ORDER_PORTFOLIO_MAX_NETS,
            ORDER_PORTFOLIO_CELL_CAP + 1,
            5,
            6,
        ));
        assert!(!eligible(
            ORDER_PORTFOLIO_MIN_GROUPS,
            ORDER_PORTFOLIO_MAX_NETS,
            ORDER_PORTFOLIO_CELL_CAP,
            6,
            6,
        ));
    }

    #[test]
    fn equal_completion_diversification_preserves_original_board_bytes() {
        let dims = Dims::new(7, 7);
        let mut grid = Grid::filled(dims, OBSTACLE);
        for x in 0..7 {
            grid.set(dims.idx(x, 3), 1);
        }
        for y in 0..7 {
            grid.set(dims.idx(3, y), 1);
        }
        for &(x, y) in &[(0, 0), (6, 0), (0, 6), (6, 6)] {
            grid.set(dims.idx(x, y), 1);
        }
        let nets = vec![
            net("a", dims.idx(0, 3), dims.idx(6, 3)),
            net("b", dims.idx(3, 0), dims.idx(3, 6)),
            net("c", dims.idx(0, 0), dims.idx(0, 0)),
            net("d", dims.idx(6, 0), dims.idx(6, 0)),
            net("e", dims.idx(0, 6), dims.idx(0, 6)),
            net("f", dims.idx(6, 6), dims.idx(6, 6)),
        ];
        let results = vec![
            RouteResult {
                net: "a".to_owned(),
                path: (0..7).map(|x| dims.idx(x, 3)).collect(),
                cost: 6,
            },
            RouteResult {
                net: "c".to_owned(),
                path: vec![dims.idx(0, 0)],
                cost: 0,
            },
            RouteResult {
                net: "d".to_owned(),
                path: vec![dims.idx(6, 0)],
                cost: 0,
            },
            RouteResult {
                net: "e".to_owned(),
                path: vec![dims.idx(0, 6)],
                cost: 0,
            },
            RouteResult {
                net: "f".to_owned(),
                path: vec![dims.idx(6, 6)],
                cost: 0,
            },
        ];
        let expected = BoardRoute {
            congestion: BoardRoute::congestion_from(dims, &results),
            results,
            unrouted: vec!["b".to_owned()],
            groups: vec![0, 2, 3, 4, 5],
        };

        let actual = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn serial_portfolio_trigger_enforces_group_net_and_cell_bounds() {
        let incomplete = |n| board_with_costs(&[1], n);
        let complete = |n| board_with_costs(&vec![1; n], n);
        let at_cap = Dims::new(500, 500);
        let above_cap = Dims::new(501, 500);

        assert!(!should_try_serial_candidate(
            at_cap,
            PORTFOLIO_MIN_NETS - 1,
            true,
            &incomplete(PORTFOLIO_MIN_NETS - 1)
        ));
        // An incomplete primary must not bypass the unconditional cell cap.
        assert!(!should_try_serial_candidate(
            above_cap,
            PORTFOLIO_MIN_NETS,
            true,
            &incomplete(PORTFOLIO_MIN_NETS)
        ));
        assert!(should_try_serial_candidate(
            at_cap,
            PORTFOLIO_MAX_NETS,
            true,
            &incomplete(PORTFOLIO_MAX_NETS)
        ));
        assert!(!should_try_serial_candidate(
            at_cap,
            PORTFOLIO_MAX_NETS + 1,
            true,
            &incomplete(PORTFOLIO_MAX_NETS + 1)
        ));
        assert!(!should_try_serial_candidate(
            at_cap,
            PORTFOLIO_MIN_NETS,
            false,
            &incomplete(PORTFOLIO_MIN_NETS)
        ));
        assert!(should_try_serial_candidate(
            at_cap,
            PORTFOLIO_MIN_NETS,
            true,
            &complete(PORTFOLIO_MIN_NETS)
        ));
        assert!(!should_try_serial_candidate(
            above_cap,
            PORTFOLIO_MIN_NETS,
            true,
            &complete(PORTFOLIO_MIN_NETS)
        ));

        let named_group = vec![net("g#0", 0, 1), net("g#1", 2, 3)];
        let shared_cell_only = vec![net("a", 0, 1), net("b", 1, 2)];
        assert!(has_named_subnet_group(&named_group));
        assert!(!has_named_subnet_group(&shared_cell_only));
    }

    /// Two nets whose individually-shortest straight paths cross in the middle.
    /// The naive (independent) router would conflict on the centre cell; the
    /// negotiated router routes BOTH on cell-disjoint paths.
    #[test]
    fn crossing_nets_route_disjoint() {
        // 5x5 open grid. The endpoints are pulled in one cell from the border so a
        // disjoint crossing solution genuinely exists (each net can detour into the
        // free margin), unlike the degenerate 3x3 where the only routes collide.
        //   A goes (2,1) -> (2,3)  (vertical, centre column)
        //   B goes (1,2) -> (3,2)  (horizontal, centre row)
        // Both straight paths pass through the centre cell (2,2).
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(2, 1), dims.idx(2, 3));
        let b = net("b", dims.idx(1, 2), dims.idx(3, 2));

        // Sanity: the naive shortest paths really do share the centre cell, so a
        // disjoint solution is non-trivial.
        let centre = dims.idx(2, 2);
        let fa = crate::LeeRouter::route_one(&grid, a.src, a.dst).unwrap().0;
        let fb = crate::LeeRouter::route_one(&grid, b.src, b.dst).unwrap().0;
        assert!(
            fa.contains(&centre) && fb.contains(&centre),
            "precondition: naive paths collide at centre"
        );

        let br = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(br.unrouted.is_empty(), "both nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);

        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        assert!(
            disjoint(pa, pb),
            "paths must be cell-disjoint: {pa:?} {pb:?}"
        );
        assert_eq!(pa.first().copied(), Some(a.src));
        assert_eq!(pa.last().copied(), Some(a.dst));
        assert_eq!(pb.first().copied(), Some(b.src));
        assert_eq!(pb.last().copied(), Some(b.dst));
    }

    /// `BoardRoute::groups` is exported aligned 1:1 with `results`, carries the
    /// router's ground-truth electrical-net group id, and gives `#`-sibling nets the
    /// SAME id while a foreign net gets a different one. The DRC relies on this to
    /// grant same-net copper exactly the immunity the router permitted.
    #[test]
    fn route_exports_aligned_group_ids() {
        let dims = Dims::new(8, 8);
        let grid = GridBuilder::new(dims, 1).build();
        // `g#0` and `g#1` share the `group_of` prefix `g` → one group. `foreign` is
        // its own group.
        let s0 = net("g#0", dims.idx(0, 0), dims.idx(7, 0));
        let s1 = net("g#1", dims.idx(0, 2), dims.idx(7, 2));
        let f = net("foreign", dims.idx(0, 5), dims.idx(7, 5));
        let br = NegotiatedRouter::new().route(&grid, &[s0, s1, f]).unwrap();
        assert!(br.unrouted.is_empty(), "all nets must route: {br:?}");
        assert_eq!(
            br.groups.len(),
            br.results.len(),
            "groups align 1:1 with results"
        );
        // Map back from results (which are in input order here) to assert grouping.
        let g_of = |name: &str| {
            let i = br.results.iter().position(|r| r.net == name).unwrap();
            br.groups[i]
        };
        assert_eq!(g_of("g#0"), g_of("g#1"), "`#`-siblings share a group id");
        assert_ne!(
            g_of("g#0"),
            g_of("foreign"),
            "a foreign net is a distinct group"
        );
    }

    /// A net must detour around a foreign net's pad (a hard obstacle it does not
    /// own). Both route, cell-disjoint, and the detouring net avoids the pad.
    #[test]
    fn detours_around_foreign_pad() {
        // 7x3 open grid; A's pad is the 2x1 block (3,0),(3,1) (obstacle in base).
        let dims = Dims::new(7, 3);
        let grid = GridBuilder::new(dims, 1).mark_rect(3, 0, 3, 1).build();

        let a_pad: Vec<CellIdx> = vec![dims.idx(3, 0), dims.idx(3, 1)];
        let net_a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(3, 0),
            dst: dims.idx(3, 1),
            passable_pads: a_pad.clone(),
            via_passable_pads: Vec::new(),
        };
        let net_b = NetEndpoints {
            net: "b".into(),
            src: dims.idx(0, 1),
            dst: dims.idx(6, 1),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };

        let br = NegotiatedRouter::new()
            .route(&grid, &[net_a, net_b])
            .unwrap();
        assert!(br.unrouted.is_empty(), "both nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);

        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        for c in pb {
            assert!(!a_pad.contains(c), "B must route around A's pad; cell {c}");
        }
        assert!(disjoint(pa, pb), "paths must be cell-disjoint");
    }

    /// An over-constrained single corridor that only one net can use: exactly one
    /// routes, one is unrouted, and the router terminates (no panic / hang).
    #[test]
    fn over_constrained_corridor_one_unrouted() {
        // 3x3 with walls top and bottom of the centre row -> the only path between
        // the left and right centre cells is the single corridor (0,1)-(1,1)-(2,1).
        let dims = Dims::new(3, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_cell(1, 0);
        b.mark_cell(1, 2);
        let grid = b.build();
        // DISTINCT endpoints (so they are not junction-merged into one net), but both
        // can only cross left<->right through the lone corridor cell (1,1): net a runs
        // the centre row, net b drops in from the bottom corners — both must take
        // (1,1), so exactly one routes.
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 2), dims.idx(2, 2)),
        ];
        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(
            br.results.len() + br.unrouted.len(),
            2,
            "every net accounted for"
        );
        assert_eq!(br.unrouted.len(), 1, "both cannot fit the single corridor");
        assert_eq!(br.results.len(), 1);
    }

    /// Determinism: routing the same problem twice yields identical results.
    #[test]
    fn deterministic_results() {
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 0), dims.idx(4, 4)),
            net("b", dims.idx(4, 0), dims.idx(0, 4)),
            net("c", dims.idx(0, 2), dims.idx(4, 2)),
        ];
        let br1 = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        let br2 = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(br1.results, br2.results);
        assert_eq!(br1.unrouted, br2.unrouted);
        assert_eq!(br1.congestion, br2.congestion);
    }

    #[test]
    fn lightweight_outcome_matches_route_and_solo_routability() {
        let router = NegotiatedRouter::new();
        let mut saw_routable = false;
        let mut saw_unroutable = false;

        for fixture in mr_fixtures::obstacle_battery() {
            let outcome = router
                .route_with_outcome(&fixture.grid, &fixture.nets)
                .unwrap();
            assert_eq!(
                outcome.board,
                router.route(&fixture.grid, &fixture.nets).unwrap(),
                "{}: outcome board must equal the Router API",
                fixture.name
            );

            let solo: Vec<bool> = fixture
                .nets
                .iter()
                .map(|net| {
                    let board = router
                        .route(&fixture.grid, std::slice::from_ref(net))
                        .unwrap();
                    !board.results.is_empty() && board.unrouted.is_empty()
                })
                .collect();
            assert_eq!(
                outcome.alone_routable, solo,
                "{}: cached isolation result must equal a real solo route",
                fixture.name
            );
            saw_routable |= solo.iter().any(|&routable| routable);
            saw_unroutable |= solo.iter().any(|&routable| !routable);
        }

        assert!(saw_routable && saw_unroutable);

        let dims = Dims::new(1, 1);
        let zero = vec![net("zero", 0, 0)];
        let outcome = router
            .route_with_outcome(&Grid::filled(dims, 1), &zero)
            .unwrap();
        assert_eq!(outcome.alone_routable, [true]);
        assert_eq!(outcome.board.results[0].cost, 0);
    }

    /// Two chained sub-nets of one connection ("X#0","X#1") share a middle cell.
    /// They are the same group, so the shared cell is NOT overuse and both route.
    #[test]
    fn same_connection_subnets_may_share() {
        // 3x1 line: X#0 routes (0,0)->(1,0); X#1 routes (1,0)->(2,0). They share
        // the middle cell (1,0). Being one connection, that is legal.
        let dims = Dims::new(3, 1);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("X#0", dims.idx(0, 0), dims.idx(1, 0)),
            net("X#1", dims.idx(1, 0), dims.idx(2, 0)),
        ];
        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert!(br.unrouted.is_empty(), "both sub-nets must route: {br:?}");
        assert_eq!(br.results.len(), 2);
        let mid = dims.idx(1, 0);
        assert!(br.results[0].path.contains(&mid));
        assert!(br.results[1].path.contains(&mid));
        // The shared middle cell shows congestion 2 — allowed within one group.
        assert_eq!(br.congestion[mid as usize], 2);
    }

    /// Co-dependent two-net board: BOTH nets must deviate from their greedy path
    /// for a cell-disjoint solution to exist. A vertical wall on column x=2 blocks
    /// rows y=2,3,4, so the only crossing is via rows 0–1. Net A's greedy route is
    /// straight across row 1 — which walls B off from crossing — and net B's greedy
    /// route detours up *through A's endpoints*, stranding A. Neither greedy commit
    /// order works; the disjoint solution forces A up onto row 0 AND B to cross at
    /// (2,1) clear of A's endpoints. The full router must route BOTH.
    #[test]
    fn co_dependent_both_must_deviate() {
        let dims = Dims::new(5, 5);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(2, 2);
        gb.mark_cell(2, 3);
        gb.mark_cell(2, 4);
        let grid = gb.build();

        let a = net("a", dims.idx(0, 1), dims.idx(4, 1));
        let b = net("b", dims.idx(0, 3), dims.idx(4, 3));

        // Precondition: each net's *greedy* path blocks the other. Routing A alone
        // (straight row 1) leaves B with no disjoint route, and vice versa.
        let ga = crate::LeeRouter::route_one(&grid, a.src, a.dst).unwrap().0;
        let gb_path = crate::LeeRouter::route_one(&grid, b.src, b.dst).unwrap().0;
        assert!(
            ga.iter().any(|c| gb_path.contains(c)) || {
                // B's greedy uses A's endpoints, so even where the cell sets don't
                // literally overlap the greedy commit still strands the other.
                ga.contains(&a.src) && gb_path.contains(&a.src)
            },
            "precondition: greedy paths are in mutual conflict"
        );

        let br = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "both co-dependent nets must route: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        assert!(
            disjoint(pa, pb),
            "paths must be cell-disjoint: {pa:?} {pb:?}"
        );
        assert_eq!(pa.first().copied(), Some(a.src));
        assert_eq!(pa.last().copied(), Some(a.dst));
        assert_eq!(pb.first().copied(), Some(b.src));
        assert_eq!(pb.last().copied(), Some(b.dst));
    }

    /// Bounded rip-up actually rescues a net that the multi-order pass alone leaves
    /// unrouted. On this 6x6 board with a single wall at (4,1), the negotiation +
    /// multi-order legalization commits two of the three nets and strands the
    /// third; only displacing an already-committed net (rip-up) frees a corridor
    /// so all three route. We assert the full router routes ALL THREE — which is
    /// only reachable through the rip-up stage (verified empirically against a
    /// rip-up-disabled build during development).
    #[test]
    fn ripup_rescues_third_net() {
        let dims = Dims::new(6, 6);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(4, 1);
        let grid = gb.build();

        let nets = vec![
            net("a", dims.idx(3, 0), dims.idx(4, 3)),
            net("b", dims.idx(0, 4), dims.idx(5, 4)),
            net("c", dims.idx(2, 1), dims.idx(5, 3)),
        ];

        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert!(
            br.unrouted.is_empty(),
            "rip-up must route all three nets: {br:?}"
        );
        assert_eq!(br.results.len(), 3);

        // Cell-disjoint across the three (distinct) groups.
        for i in 0..br.results.len() {
            for j in (i + 1)..br.results.len() {
                assert!(
                    disjoint(&br.results[i].path, &br.results[j].path),
                    "group paths must be cell-disjoint: {} vs {}",
                    br.results[i].net,
                    br.results[j].net
                );
            }
        }
        // Endpoints honoured.
        for (r, n) in br.results.iter().zip(nets.iter()) {
            assert_eq!(r.path.first().copied(), Some(n.src));
            assert_eq!(r.path.last().copied(), Some(n.dst));
        }
    }

    /// Rip-up must terminate and never panic even when no disjoint solution exists.
    /// Re-uses the over-constrained single corridor (two DISTINCT-endpoint nets that
    /// both must cross through the lone corridor cell): rip-up may ping-pong the
    /// corridor but is bounded by its global/per-net budgets, so it stops and returns
    /// the best partial (one net), never regressing below the multi-order result.
    #[test]
    fn ripup_terminates_when_unsolvable() {
        let dims = Dims::new(3, 3);
        let mut b = GridBuilder::new(dims, 1);
        b.mark_cell(1, 0);
        b.mark_cell(1, 2);
        let grid = b.build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(2, 1)),
            net("b", dims.idx(0, 2), dims.idx(2, 2)),
        ];
        let br = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(br.results.len() + br.unrouted.len(), 2);
        assert_eq!(br.unrouted.len(), 1, "single corridor fits only one net");
        // Determinism across repeated runs (rip-up tie-breaks must be stable).
        let br2 = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(br.results, br2.results);
        assert_eq!(br.unrouted, br2.unrouted);
    }

    /// The deterministic permutation generator yields every permutation exactly
    /// once, in a fixed (lexicographic-by-construction) order.
    #[test]
    fn permutations_are_complete_and_ordered() {
        let perms = permutations(&[0, 1, 2]);
        assert_eq!(
            perms,
            vec![
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0],
            ]
        );
        // 4 distinct items -> 24 unique permutations.
        let p4 = permutations(&[0, 1, 2, 3]);
        assert_eq!(p4.len(), 24);
        let uniq: std::collections::HashSet<_> = p4.iter().cloned().collect();
        assert_eq!(uniq.len(), 24);
    }

    /// A net whose optimal path must leave its bbox+margin window still routes,
    /// via the full-board retry. We build a tall thin board with a wall that forces
    /// a long detour far outside the src/dst bounding box, then assert the net
    /// routes and its path actually exits the window (proving the window alone
    /// would have failed and the full-board retry rescued it).
    #[test]
    fn net_routes_via_full_board_retry_when_path_leaves_window() {
        // 40 wide, 6 tall. src=(0,0), dst=(2,2): a tiny bbox in the top-left corner,
        // so the window (margin = max(16, ceil(0.3*2)) = 16) spans x in [0,18]. Row
        // y=1 is walled for x in 0..=37, leaving the ONLY top<->bottom crossing at
        // x=38,39 — far outside the window. The net must run right along y=0 to x=39,
        // drop to y=2, then run left to dst. The windowed search alone fails; the
        // full-board retry rescues it.
        let dims = Dims::new(40, 6);
        let mut gb = GridBuilder::new(dims, 1);
        for x in 0..=37 {
            gb.mark_cell(x, 1);
        }
        let grid = gb.build();

        let a = net("a", dims.idx(0, 0), dims.idx(2, 2));

        // The window for this net: bbox {(0,0),(2,2)} expanded by margin 16 ->
        // x in [0,18]. The forced crossing at x=38/39 lies well outside it.
        let win = net_window(dims, a.src, a.dst, &[]);
        assert!(
            win.x1 < 38,
            "precondition: gap column is outside the window"
        );

        let br = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "net must route via the full-board retry: {br:?}"
        );
        assert_eq!(br.results.len(), 1);
        let path = &br.results[0].path;
        assert_eq!(path.first().copied(), Some(a.src));
        assert_eq!(path.last().copied(), Some(a.dst));
        // The path must leave the window (reach the far-right gap column) — proof
        // that the windowed search alone would have failed.
        assert!(
            path.iter().any(|&c| {
                let (x, _) = dims.xy(c);
                x > win.x1
            }),
            "optimal path must exit the window (reach the far gap)"
        );
    }

    /// Order-robust legalization: a board where committing net A's group first
    /// strands net B, but committing B's group first routes both. The full router
    /// must find the good order and route BOTH nets.
    ///
    /// 5x5 open grid:
    ///   A: (0,2) -> (4,2)  horizontal across the middle row
    ///   B: (2,1) -> (2,3)  vertical across the middle column
    /// If A claims the whole middle row first, it walls off rows y<2 from y>2 and
    /// B (which must get from y=1 to y=3) is stranded. If B claims the middle
    /// column first, A simply detours one row up/down and crosses elsewhere.
    #[test]
    fn legalization_is_order_robust() {
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();

        // Net order is [A, B], i.e. the FAILING first-appearance order.
        let a = net("a", dims.idx(0, 2), dims.idx(4, 2));
        let b = net("b", dims.idx(2, 1), dims.idx(2, 3));

        // Direct check of the legalization primitive: hand-crafted negotiated
        // paths that collide on the centre cell (2,2). One commit order strands a
        // net; the other routes both — proving the order genuinely matters.
        let n_cells = dims.len();
        let group_ids = vec![0usize, 1usize];
        let nets_ab = [a.clone(), b.clone()];
        let crafted = vec![
            // A: full middle row.
            vec![
                dims.idx(0, 2),
                dims.idx(1, 2),
                dims.idx(2, 2),
                dims.idx(3, 2),
                dims.idx(4, 2),
            ],
            // B: full middle column.
            vec![dims.idx(2, 1), dims.idx(2, 2), dims.idx(2, 3)],
        ];
        let mut buf = SearchBuf::new(n_cells);
        let mut pad_set = PadSet::new(n_cells);
        let windows: Vec<Window> = nets_ab
            .iter()
            .map(|net| net_window(dims, net.src, net.dst, &net.passable_pads))
            .collect();

        // Single-layer board: the via model is inert (no via neighbours exist).
        let via_model = ViaModel::through_hole(dims.layers);
        // Abstract Dims-only grid: uniform unit coords reproduce unit-hop pricing.
        let coords = GridCoords::uniform(dims);
        let heuristic_costs = ManhattanCosts::new(dims, &coords);

        // Order [0,1] = A first: B should be stranded.
        let c_ab = legalize_in_order(
            &grid,
            &coords,
            &heuristic_costs,
            &mut buf,
            &mut pad_set,
            &nets_ab,
            &group_ids,
            &crafted,
            &windows,
            &[0, 1],
            n_cells,
            &via_model,
            0.0,
            0.0,
            0.0,
            false,
            false,
        );
        assert!(c_ab[0].is_some(), "A commits in A-first order");
        assert!(
            c_ab[1].is_none(),
            "B must be stranded when A claims the middle row first"
        );

        // Order [1,0] = B first: both route.
        let c_ba = legalize_in_order(
            &grid,
            &coords,
            &heuristic_costs,
            &mut buf,
            &mut pad_set,
            &nets_ab,
            &group_ids,
            &crafted,
            &windows,
            &[1, 0],
            n_cells,
            &via_model,
            0.0,
            0.0,
            0.0,
            false,
            false,
        );
        assert!(
            c_ba[0].is_some() && c_ba[1].is_some(),
            "both route when B claims the middle column first"
        );

        // The full router must pick the good order and route BOTH nets even though
        // first-appearance order [A, B] alone would strand B.
        let br = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "order-robust router must route both nets: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let pa = &br.results[0].path;
        let pb = &br.results[1].path;
        assert!(
            disjoint(pa, pb),
            "paths must be cell-disjoint: {pa:?} {pb:?}"
        );
    }

    /// A 2-layer board whose ONLY planar corridor on layer 0 is fully walled off
    /// between src and dst. The net is unroutable on layer 0 alone but routes by
    /// via-ing up to layer 1, crossing the (open) wall column there, and via-ing
    /// back down. The path must therefore touch BOTH layers.
    #[test]
    fn vias_route_around_a_full_layer0_wall() {
        // 3 wide, 1 tall, 2 layers. A wall at (1,0) on layer 0 splits the single
        // row into two halves; there is no layer-0 path from (0,0) to (2,0).
        // `GridBuilder::mark_cell` marks layer 0 only, so layer 1 is fully open.
        let dims = Dims::with_layers(3, 1, 2);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(1, 0); // wall on layer 0 at x=1
        let grid = gb.build();

        // Precondition: layer 0 alone has no path (the wall is the whole corridor).
        assert!(grid.is_obstacle(dims.idx3(1, 0, 0)));
        assert!(
            !grid.is_obstacle(dims.idx3(1, 0, 1)),
            "layer 1 must be open"
        );

        let a = net("a", dims.idx3(0, 0, 0), dims.idx3(2, 0, 0));
        let br = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "net must route by changing layers: {br:?}"
        );
        assert_eq!(br.results.len(), 1);

        let path = &br.results[0].path;
        assert_eq!(path.first().copied(), Some(a.src));
        assert_eq!(path.last().copied(), Some(a.dst));
        // The path uses both layers (a via up, cross, a via back down).
        assert!(
            path.iter().any(|&c| dims.layer_of(c) == 0),
            "path must touch layer 0: {path:?}"
        );
        assert!(
            path.iter().any(|&c| dims.layer_of(c) == 1),
            "path must via onto layer 1 to cross the wall: {path:?}"
        );
        // Every step is either a same-layer 4-neighbour or a same-(x,y) via step.
        for w in path.windows(2) {
            let planar = dims.neighbors4(w[0]).contains(&w[1]);
            let via = dims.via_neighbors(w[0]).contains(&w[1]);
            assert!(planar || via, "illegal step {} -> {}", w[0], w[1]);
        }
    }

    /// Same wall as above, but a restricted [`ViaModel`] forbids the only layer
    /// transition (0<->1). With no legal via and no layer-0 corridor, the net is
    /// unroutable.
    #[test]
    fn forbidden_via_step_leaves_net_unrouted() {
        let dims = Dims::with_layers(3, 1, 2);
        let mut gb = GridBuilder::new(dims, 1);
        gb.mark_cell(1, 0);
        let grid = gb.build();

        // Two layers but the model permits NO adjacent step (empty allow-list), so
        // the 0<->1 transition is illegal and the wall cannot be bypassed.
        let vm = ViaModel::with_allowed_steps(2, ViaModel::DEFAULT_STEP_COST, Vec::new());
        let a = net("a", dims.idx3(0, 0, 0), dims.idx3(2, 0, 0));
        let br = NegotiatedRouter::new()
            .with_via_model(vm)
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        assert_eq!(br.results.len(), 0, "no route exists without a legal via");
        assert_eq!(br.unrouted, vec!["a".to_string()]);
    }

    /// `layers == 1` regression: a known single-layer board must route exactly as
    /// it did before vias existed (same path cost). Vias add nothing because
    /// `via_neighbors` is empty on a single-layer grid.
    #[test]
    fn single_layer_routes_identically_with_vias_available() {
        // 5x5 open grid, straight Manhattan net: the optimal cost is the bbox
        // Manhattan distance (4) regardless of which monotone path is taken.
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(0, 0), dims.idx(4, 0));

        // Default router (synthesises a through-hole model over 1 layer) and an
        // explicit through-hole model must both produce the same single-layer route.
        let br_default = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();
        let br_vm = NegotiatedRouter::new()
            .with_via_model(ViaModel::through_hole(1))
            .route(&grid, std::slice::from_ref(&a))
            .unwrap();

        assert!(br_default.unrouted.is_empty());
        assert_eq!(br_default.results.len(), 1);
        assert_eq!(br_default.results[0].cost, 4, "straight 5-wide run costs 4");
        // Every cell stays on layer 0 (there is no other layer).
        assert!(br_default.results[0]
            .path
            .iter()
            .all(|&c| dims.layer_of(c) == 0));
        // The via model makes no difference on a single-layer board.
        assert_eq!(br_default.results, br_vm.results);
        assert_eq!(br_default.congestion, br_vm.congestion);
    }

    /// True iff any cell of `a` lies within the 8-neighbourhood (Chebyshev
    /// radius one) of any cell of `b` on the same layer — i.e. the two paths touch
    /// or sit adjacent. With `clearance_cells >= 1` two distinct nets must not.
    fn within_one(dims: Dims, a: &[CellIdx], b: &[CellIdx]) -> bool {
        let sb: std::collections::HashSet<_> = b.iter().copied().collect();
        a.iter().any(|&c| {
            let (x, y, l) = dims.xyz(c);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx as u32 >= dims.w || ny as u32 >= dims.h {
                        continue;
                    }
                    if sb.contains(&dims.idx3(nx as u32, ny as u32, l)) {
                        return true;
                    }
                }
            }
            false
        })
    }

    /// Clearance is now a SOFT cost: the halo makes spacing PREFERRED, not forced.
    /// On a roomy board where a clearance-legal route exists at modest extra cost,
    /// the router routes BOTH nets and keeps their committed copper Chebyshev-clear
    /// (no shared 8-neighbour cell). Each net still owns its copper hard-disjointly.
    ///
    /// Endpoints are pulled away from each other so a clearance-legal detour is
    /// genuinely available: A runs straight along row 1, B's endpoints sit on row 4
    /// (three rows below A) so B can run straight on row 4, well clear of A's halo —
    /// the cheapest route is already the clearance-legal one.
    #[test]
    fn clearance_halo_prefers_spacing_when_room() {
        // 8 wide, 6 tall, wide open. A on row 1, B on row 4 — there is plenty of
        // room for both to stay ≥2 rows apart.
        let dims = Dims::new(8, 6);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(0, 1), dims.idx(7, 1));
        let b = net("b", dims.idx(0, 4), dims.idx(7, 4));

        // clearance 1: both route AND, because the board is roomy, the router prefers
        // the clearance-legal layout — the two paths share no 8-neighbour cell.
        let br = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        assert!(
            br.unrouted.is_empty(),
            "soft clearance: both nets must still route on a roomy board: {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let (pa, pb) = (&br.results[0].path, &br.results[1].path);
        assert!(disjoint(pa, pb), "copper must stay hard-disjoint");
        assert!(
            !within_one(dims, pa, pb),
            "soft clearance: on a roomy board the router PREFERS a clearance-legal \
             layout (no shared 8-neighbour cell): {pa:?} {pb:?}"
        );
    }

    /// HARD clearance: `route_legal` (the committing pass) never commits copper into
    /// a foreign net's clearance halo. On a congested fixture where the only route
    /// home for net B would have to cross A's halo, B is now left UNROUTED rather than
    /// silently violating clearance — the accepted trade. Whatever DOES route stays
    /// clearance-clean (no path cell within the halo radius of a foreign net).
    #[test]
    fn hard_clearance_drops_net_rather_than_violate() {
        // 3 wide, 5 tall. Net A runs straight down column 1, rows 0..=3 — a vertical
        // wall of copper that leaves only the single cell (1,4) free in column 1. Net
        // B must cross from (0,2) to (2,2): the ONLY copper-free crossing of column 1
        // is through (1,4), which is Chebyshev-adjacent to A's copper at (1,3) — i.e.
        // squarely inside A's clearance halo. With a HARD halo (radius 1) that cell is
        // blocked for B, walling it off from its target. B MUST be dropped, not routed
        // through the halo. (Under the old soft cost B routed through at a violation.)
        let dims = Dims::new(3, 5);
        let a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(1, 0),
            dst: dims.idx(1, 3),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };
        let grid = GridBuilder::new(dims, 1).build();
        let b = net("b", dims.idx(0, 2), dims.idx(2, 2));

        let br = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // Exactly one net must be left unrouted: B cannot reach its target without
        // violating A's clearance, so the hard constraint drops it (congested).
        assert!(
            !br.unrouted.is_empty(),
            "hard clearance must leave the congested net unrouted, not violate: {br:?}"
        );

        // Whatever routed is clearance-clean: no committed path cell sits within the
        // clearance radius of a DIFFERENT net's copper.
        for r1 in &br.results {
            for r2 in &br.results {
                if r1.net == r2.net {
                    continue;
                }
                assert!(
                    !within_one(dims, &r1.path, &r2.path),
                    "committed copper must never violate clearance: {r1:?} vs {r2:?}"
                );
            }
        }
    }

    /// Empty-clearance fast path is UNCHANGED by the hardening: on the exact same
    /// congested fixture, with `clearance = 0` and the default (keepout 0) via model,
    /// the halo is empty so `route_legal`'s foreign-halo hard block and via-ring guard
    /// are both inert — BOTH nets route (B passes through (1,4) freely), reproducing
    /// the pre-clearance behaviour byte-identically.
    #[test]
    fn hard_clearance_fast_path_unchanged_when_clearance_zero() {
        let dims = Dims::new(3, 5);
        let a = NetEndpoints {
            net: "a".into(),
            src: dims.idx(1, 0),
            dst: dims.idx(1, 3),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };
        let grid = GridBuilder::new(dims, 1).build();
        let b = net("b", dims.idx(0, 2), dims.idx(2, 2));

        let br = NegotiatedRouter::new()
            .with_clearance_cells(0)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // No halo at all → both nets route (B is NOT dropped); copper stays disjoint.
        assert!(
            br.unrouted.is_empty(),
            "clearance-0 fast path: both nets must route (halo inert): {br:?}"
        );
        assert_eq!(br.results.len(), 2);
        let ra = br.results.iter().find(|r| r.net == "a").expect("A routes");
        let rb = br.results.iter().find(|r| r.net == "b").expect("B routes");
        assert!(
            disjoint(&ra.path, &rb.path),
            "distinct nets must never overlap copper: {ra:?} {rb:?}"
        );
    }

    /// A top-layer SMD endpoint may own the top cell without owning coincident
    /// bottom copper. Via-forbidden exemptions are layer-local pad membership, not
    /// a blanket exemption merely because one side of the transition is a terminal.
    #[test]
    fn via_forbidden_endpoint_exemption_is_layer_local() {
        let dims = Dims::with_layers(2, 1, 2);
        let mut gb = GridBuilder::new(dims, 1);
        // Force the only possible layer change to occur at x=0: top x=1 is closed.
        gb.mark_cell_layer(1, 0, 0);
        let mut grid = gb.build();
        let top_src = dims.idx3(0, 0, 0);
        let bottom_under_src = dims.idx3(0, 0, 1);
        let bottom_dst = dims.idx3(1, 0, 1);
        grid.via_forbidden = vec![false; dims.len()];
        grid.via_forbidden[top_src as usize] = true;
        grid.via_forbidden[bottom_under_src as usize] = true;

        let top_smd = NetEndpoints {
            net: "top_smd".into(),
            src: top_src,
            dst: bottom_dst,
            passable_pads: vec![top_src],
            via_passable_pads: vec![top_src],
        };
        let blocked = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&top_smd))
            .unwrap();
        assert_eq!(blocked.unrouted, ["top_smd"]);

        let through_pad = NetEndpoints {
            net: "through".into(),
            passable_pads: vec![top_src, bottom_under_src],
            // Deliberately unsorted: public/serde input has no ordering contract.
            via_passable_pads: vec![bottom_under_src, top_src],
            ..top_smd
        };
        let allowed = NegotiatedRouter::new()
            .route(&grid, std::slice::from_ref(&through_pad))
            .unwrap();
        assert!(allowed.unrouted.is_empty());

        let invalid_subset = NetEndpoints {
            net: "bad-subset".into(),
            passable_pads: vec![top_src],
            via_passable_pads: vec![bottom_under_src],
            ..through_pad.clone()
        };
        assert!(matches!(
            NegotiatedRouter::new().route(&grid, &[invalid_subset]),
            Err(RouterError::InvalidEndpoint { .. })
        ));
        let invalid_index = NetEndpoints {
            net: "bad-index".into(),
            via_passable_pads: vec![dims.len() as CellIdx],
            ..through_pad
        };
        assert!(matches!(
            NegotiatedRouter::new().route(&grid, &[invalid_index]),
            Err(RouterError::InvalidEndpoint { .. })
        ));
    }

    /// Via keepout: a placed via reserves an annular ring around itself. Soft pressure
    /// steers nets apart during negotiation; the committing legalization pass enforces
    /// the ring HARD (no foreign copper may sit inside it). On a 2-layer board whose
    /// only layer-0 corridor is walled, net A must via up at a chokepoint to cross.
    /// With `via_model.keepout_mm = 1.0` and ample room for B to via up at a DIFFERENT
    /// row, the router routes BOTH nets and keeps B's committed copper out of A's
    /// reserved via neighbourhood. We assert both route and B's copper shares no cell
    /// of A's via 8-neighbourhood.
    #[test]
    fn via_keepout_prefers_clear_neighbourhood_when_room() {
        // 5 wide, 5 tall, 2 layers. Wall the WHOLE of layer 0 column x=2 so the only
        // way past x=2 on layer 0 is to via up to layer 1, cross, and via back. The
        // extra height (5 rows) gives B room to via at a row well clear of A's via.
        let dims = Dims::with_layers(5, 5, 2);
        let mut gb = GridBuilder::new(dims, 1);
        for y in 0..5 {
            gb.mark_cell(2, y);
        }
        let grid = gb.build();
        // Sanity: layer 0 column x=2 is fully walled; layer 1 is open there.
        for y in 0..5 {
            assert!(grid.is_obstacle(dims.idx3(2, y, 0)));
            assert!(!grid.is_obstacle(dims.idx3(2, y, 1)));
        }

        // A crosses the wall along row 0 (top edge); B crosses along row 4 (bottom
        // edge) — four rows apart, so each can via clear of the other's keepout.
        let a = net("a", dims.idx3(0, 0, 0), dims.idx3(4, 0, 0));
        let b = net("b", dims.idx3(0, 4, 0), dims.idx3(4, 4, 0));

        let vm = {
            let mut m = ViaModel::through_hole(2);
            // Uniform unit grid: 1.0 mm keepout == the former 1-cell Chebyshev box.
            m.keepout_mm = 1.0;
            m
        };
        let br = NegotiatedRouter::new()
            .with_via_model(vm)
            .with_clearance_cells(0)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // Both route (each can always via across; soft keepout never drops a net).
        assert!(
            br.unrouted.is_empty(),
            "soft via keepout: both nets must route: {br:?}"
        );
        let ra = br
            .results
            .iter()
            .find(|r| r.net == "a")
            .expect("A must route");
        let rb = br
            .results
            .iter()
            .find(|r| r.net == "b")
            .expect("B must route");
        // Collect A's via (x,y) chokepoints: consecutive same-(x,y), layer-changing.
        let mut via_neigh: std::collections::HashSet<CellIdx> = std::collections::HashSet::new();
        for w in ra.path.windows(2) {
            let (ax, ay, al) = dims.xyz(w[0]);
            let (bx, by, bl) = dims.xyz(w[1]);
            if ax == bx && ay == by && al != bl {
                // 8-neighbourhood on BOTH via layers (excluding the via cells).
                for &l in &[al, bl] {
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let nx = ax as i64 + dx;
                            let ny = ay as i64 + dy;
                            if nx < 0 || ny < 0 || nx as u32 >= dims.w || ny as u32 >= dims.h {
                                continue;
                            }
                            let nc = dims.idx3(nx as u32, ny as u32, l);
                            if !grid.is_obstacle(nc) {
                                via_neigh.insert(nc);
                            }
                        }
                    }
                }
            }
        }
        assert!(
            !via_neigh.is_empty(),
            "A must place at least one via: {ra:?}"
        );

        // With room to spare, the router PREFERS keeping B's copper out of A's
        // reserved via neighbourhood (the soft keepout steers B away).
        for &c in &rb.path {
            assert!(
                !via_neigh.contains(&c) || ra.path.contains(&c),
                "soft via keepout (roomy board): B's copper should avoid A's reserved \
                 via neighbourhood at cell {c}: {rb:?}"
            );
        }
    }

    fn production_via_model(layers: u32) -> ViaModel {
        let mut model = ViaModel::through_hole(layers);
        // 0.225 via radius + 0.15 clearance + 0.075 trace radius.
        model.keepout_mm = 0.45;
        model
    }

    /// Sparse Hanan regression: there is no intermediate line for a generic halo
    /// stamp to occupy. The 0.45 mm via-to-track radius therefore permits the second
    /// centre at x=0.55 today, but two 0.45 mm via pads leave only 0.10 mm edge gap;
    /// the dedicated 0.60 mm via-centre rule must reject one foreign-group via.
    #[test]
    fn sparse_via_pair_obeys_wider_center_spacing() {
        let dims = Dims::with_layers(2, 1, 2);
        let grid = Grid::filled(dims, 1);
        let coords = GridCoords::from_lines(vec![0.0, 0.55], vec![0.0]);
        let nets = vec![
            net("a", dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)),
            net("b", dims.idx3(1, 0, 0), dims.idx3(1, 0, 1)),
        ];

        let legacy = NegotiatedRouter::new()
            .with_via_model(production_via_model(2))
            .with_clearance_mm(0.30)
            .with_coords(coords.clone())
            .route(&grid, &nets)
            .unwrap();
        assert_eq!(
            legacy.results.len(),
            2,
            "the 0.45 mm rule alone misses this pair"
        );

        let guarded = NegotiatedRouter::new()
            .with_via_model(production_via_model(2))
            .with_clearance_mm(0.30)
            .with_via_spacing_mm(0.60)
            .with_coords(coords)
            .route(&grid, &nets)
            .unwrap();
        assert_eq!(guarded.results.len(), 1);
        assert_eq!(guarded.unrouted.len(), 1);
    }

    /// Zero copper clearance still forbids two physical via pads from overlapping:
    /// their centres need one full pad diameter (0.45 mm), while exact tangency is
    /// legal. This keeps the rule active instead of treating clearance zero as a
    /// feature-off signal.
    #[test]
    fn zero_clearance_via_pads_obey_diameter_boundary() {
        for (gap, expected_routed) in [(0.449, 1), (0.45, 2)] {
            let dims = Dims::with_layers(2, 1, 2);
            let grid = Grid::filled(dims, 1);
            let coords = GridCoords::from_lines(vec![0.0, gap], vec![0.0]);
            let nets = vec![
                net("a", dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)),
                net("b", dims.idx3(1, 0, 0), dims.idx3(1, 0, 1)),
            ];
            let board = NegotiatedRouter::new()
                .with_via_model(ViaModel::through_hole(2))
                .with_clearance_mm(0.0)
                .with_via_spacing_mm(0.45)
                .with_coords(coords)
                .route(&grid, &nets)
                .unwrap();
            assert_eq!(
                board.results.len(),
                expected_routed,
                "unexpected zero-clearance result at {gap} mm: {board:?}"
            );
        }
    }

    /// With zero edge clearance a 0.45 mm via and 0.15 mm trace still need their
    /// centres 0.225 + 0.075 = 0.300 mm apart. Keep this independent from the wider
    /// 0.45 mm zero-clearance via-pair rule.
    #[test]
    fn zero_clearance_via_trace_obeys_sum_of_radii() {
        for (gap, expected_routed) in [(0.299, 1), (0.301, 2)] {
            let dims = Dims::with_layers(2, 3, 2);
            let grid = Grid::filled(dims, 1);
            let coords = GridCoords::from_lines(vec![0.0, gap], vec![0.0, 1.0, 2.0]);
            let nets = vec![
                net("via", dims.idx3(0, 1, 0), dims.idx3(0, 1, 1)),
                net("trace", dims.idx3(1, 0, 0), dims.idx3(1, 2, 0)),
            ];
            let mut via_model = ViaModel::through_hole(2);
            via_model.keepout_mm = 0.30;
            let board = NegotiatedRouter::new()
                .with_via_model(via_model)
                .with_clearance_mm(0.15)
                .with_via_spacing_mm(0.45)
                .with_coords(coords)
                .route(&grid, &nets)
                .unwrap();
            assert_eq!(
                board.results.len(),
                expected_routed,
                "unexpected zero-clearance via/trace result at {gap} mm: {board:?}"
            );
        }
    }

    /// Dense-line boundary regression. Intermediate halo cells at 0.1..=0.4 must
    /// not be inflated a second time by the candidate-via guard. Centres exactly
    /// 0.60 mm apart are DRC-legal and both direct vias must commit.
    #[test]
    fn dense_via_pair_exactly_at_spacing_is_not_overblocked() {
        let dims = Dims::with_layers(7, 1, 2);
        let grid = Grid::filled(dims, 1);
        let coords = GridCoords::from_lines(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![0.0]);
        let nets = vec![
            net("a", dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)),
            net("b", dims.idx3(6, 0, 0), dims.idx3(6, 0, 1)),
        ];
        let board = NegotiatedRouter::new()
            .with_via_model(production_via_model(2))
            .with_clearance_mm(0.30)
            .with_via_spacing_mm(0.60)
            .with_coords(coords)
            .route(&grid, &nets)
            .unwrap();

        assert!(
            board.unrouted.is_empty(),
            "at-boundary pair must route: {board:?}"
        );
        assert_eq!(board.results[0].path.len(), 2);
        assert_eq!(board.results[1].path.len(), 2);
    }

    /// The broad-phase box is not the clearance shape: (0.5,0.4) lies inside a
    /// 0.60-square in both axes but outside the 0.60 Euclidean circle.
    #[test]
    fn diagonal_via_pair_uses_euclidean_distance() {
        let dims = Dims::with_layers(2, 2, 2);
        let grid = Grid::filled(dims, 1);
        let coords = GridCoords::from_lines(vec![0.0, 0.5], vec![0.0, 0.4]);
        let nets = vec![
            net("a", dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)),
            net("b", dims.idx3(1, 1, 0), dims.idx3(1, 1, 1)),
        ];
        let board = NegotiatedRouter::new()
            .with_via_model(production_via_model(2))
            .with_clearance_mm(0.30)
            .with_via_spacing_mm(0.60)
            .with_coords(coords)
            .route(&grid, &nets)
            .unwrap();
        assert!(
            board.unrouted.is_empty(),
            "Euclidean-legal diagonal pair: {board:?}"
        );
    }

    /// A trace centre 0.50 mm from a via is legal under the 0.45 mm via-to-track
    /// rule even though another via at the same centre would not satisfy 0.60 mm.
    /// This pins that the new rule is feature-aware rather than a global halo widen.
    #[test]
    fn via_center_spacing_does_not_widen_via_trace_keepout() {
        let dims = Dims::with_layers(2, 2, 2);
        let grid = Grid::filled(dims, 1);
        let coords = GridCoords::from_lines(vec![0.0, 0.5], vec![0.0, 1.0]);
        let nets = vec![
            net("via", dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)),
            net("trace", dims.idx3(1, 0, 0), dims.idx3(1, 1, 0)),
        ];
        let board = NegotiatedRouter::new()
            .with_via_model(production_via_model(2))
            .with_clearance_mm(0.30)
            .with_via_spacing_mm(0.60)
            .with_coords(coords)
            .route(&grid, &nets)
            .unwrap();
        assert!(
            board.unrouted.is_empty(),
            "via/trace 0.50 mm pair is legal: {board:?}"
        );
    }

    #[test]
    fn via_center_spacing_is_deterministic_through_parallel_threshold() {
        let dims = Dims::with_layers(17, 1, 2);
        let grid = Grid::filled(dims, 1);
        let coords = GridCoords::from_lines((0..17).map(|x| x as f64).collect(), vec![0.0]);
        let nets: Vec<_> = (0..17)
            .map(|x| net(&format!("n{x}"), dims.idx3(x, 0, 0), dims.idx3(x, 0, 1)))
            .collect();
        let router = NegotiatedRouter::new()
            .with_via_model(production_via_model(2))
            .with_via_spacing_mm(0.60)
            .with_coords(coords);
        let expected = router.route(&grid, &nets).unwrap();
        assert_eq!(expected.results.len(), nets.len());
        for _ in 0..12 {
            assert_eq!(router.route(&grid, &nets).unwrap(), expected);
        }
    }

    /// Regression: `clearance_cells = 0` reproduces an existing golden result. We
    /// re-run the exact `crossing_nets_route_disjoint` scenario through the default
    /// (clearance 0) router and an explicit `with_clearance_cells(0)` router and
    /// assert both match the known-good disjoint outcome byte-for-byte.
    #[test]
    fn clearance_zero_reproduces_golden() {
        let dims = Dims::new(5, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let a = net("a", dims.idx(2, 1), dims.idx(2, 3));
        let b = net("b", dims.idx(1, 2), dims.idx(3, 2));

        let br_default = NegotiatedRouter::new()
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();
        let br_zero = NegotiatedRouter::new()
            .with_clearance_cells(0)
            .route(&grid, &[a.clone(), b.clone()])
            .unwrap();

        // Byte-identical: explicit clearance 0 == the default.
        assert_eq!(br_default.results, br_zero.results);
        assert_eq!(br_default.unrouted, br_zero.unrouted);
        assert_eq!(br_default.congestion, br_zero.congestion);

        // And it reproduces the golden disjoint result.
        assert!(br_zero.unrouted.is_empty());
        assert_eq!(br_zero.results.len(), 2);
        assert!(disjoint(&br_zero.results[0].path, &br_zero.results[1].path));
    }

    #[test]
    fn counted_halo_stamp_removes_exact_parallel_self_cost() {
        // A radius-1 footprint visits central cells once for each neighbouring path
        // segment. Boolean membership would subtract only one and leave a large,
        // artificial self-penalty in a Jacobi snapshot.
        let dims = Dims::new(7, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let coords = GridCoords::uniform(dims);
        let heuristic_costs = ManhattanCosts::new(dims, &coords);
        let via_model = ViaModel::through_hole(1);
        let src = dims.idx(1, 2);
        let dst = dims.idx(5, 2);
        let old_path: Vec<_> = (1..=5).map(|x| dims.idx(x, 2)).collect();
        let mut present = vec![0u32; dims.len()];
        for &c in &old_path {
            present[c as usize] += 1;
        }
        let mut present_halo = vec![0u32; dims.len()];
        for_each_halo_cell(dims, &coords, &grid, &old_path, 1.0, &via_model, |c| {
            present_halo[c as usize] += 1
        });

        let mut pads = PadSet::new(dims.len());
        pads.load(&[]);
        let mut own_path = PadSet::new(dims.len());
        own_path.load(&old_path);
        let mut own_halo = CountedCellSet::new(dims.len());
        own_halo.clear();
        for_each_halo_cell(dims, &coords, &grid, &old_path, 1.0, &via_model, |c| {
            own_halo.increment(c)
        });
        let centre = dims.idx(3, 2);
        assert!(present_halo[centre as usize] > 1, "fixture needs overlap");
        for c in 0..dims.len() as CellIdx {
            assert_eq!(own_halo.count(c), present_halo[c as usize], "cell {c}");
        }

        let zeros = vec![0u32; dims.len()];
        let mut with_snapshot_buf = SearchBuf::new(dims.len());
        let with_snapshot = route_negotiated(
            &mut with_snapshot_buf,
            &grid,
            &coords,
            &heuristic_costs,
            &pads,
            &[],
            Some(&own_path),
            Some(&own_halo),
            None,
            &present,
            &present_halo,
            &zeros,
            7,
            src,
            dst,
            Window::full(dims),
            &via_model,
            false,
        );

        // The Jacobi fast path folds the immutable board-wide terms once and
        // combines this net's copper + halo multiplicity into one counted stamp.
        // It must be exactly equivalent to the unfused pricing above.
        let pfac = 7u32;
        let shared_congestion: Vec<u64> = present
            .iter()
            .zip(&present_halo)
            .map(|(&p, &h)| {
                (pfac as u64) * (SCALE as u64) * p as u64
                    + (pfac as u64) * (CLEARANCE_NEG_WEIGHT as u64) * h as u64
            })
            .collect();
        let mut combined_self = CountedCellSet::new(dims.len());
        combined_self.clear();
        for_each_halo_cell(dims, &coords, &grid, &old_path, 1.0, &via_model, |c| {
            combined_self.increment(c)
        });
        for &c in &old_path {
            combined_self.increment(c);
        }
        let mut fused_buf = SearchBuf::new(dims.len());
        let fused = route_negotiated(
            &mut fused_buf,
            &grid,
            &coords,
            &heuristic_costs,
            &pads,
            &[],
            None,
            Some(&combined_self),
            Some(&shared_congestion),
            &present,
            &present_halo,
            &zeros,
            pfac,
            src,
            dst,
            Window::full(dims),
            &via_model,
            false,
        );
        assert_eq!(fused, with_snapshot, "fused Jacobi pricing must be exact");

        let mut empty_buf = SearchBuf::new(dims.len());
        let without_snapshot = route_negotiated(
            &mut empty_buf,
            &grid,
            &coords,
            &heuristic_costs,
            &pads,
            &[],
            None,
            None,
            None,
            &zeros,
            &zeros,
            &zeros,
            7,
            src,
            dst,
            Window::full(dims),
            &via_model,
            false,
        );
        assert_eq!(
            with_snapshot, without_snapshot,
            "a net's old copper and repeated halo visits must contribute zero self-cost"
        );
    }

    #[test]
    fn fused_jacobi_pricing_matches_unfused_randomized() {
        struct Rng(u64);

        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.0
            }

            fn below(&mut self, bound: u32) -> u32 {
                ((self.next() >> 32) as u32) % bound.max(1)
            }
        }

        fn random_axis(rng: &mut Rng, count: u32) -> Vec<f64> {
            let mut out = Vec::with_capacity(count as usize);
            let mut value = 0.0;
            for _ in 0..count {
                out.push(value);
                value += (rng.below(40) + 1) as f64 / 10.0;
            }
            out
        }

        let mut rng = Rng(0x4d59_5df4_d0f3_3173);
        let mut saw_zero = false;
        let mut saw_weighted = false;
        let mut saw_obstacle = false;
        let mut saw_multilayer = false;
        let mut saw_restricted_via = false;
        let mut saw_partial_window = false;
        let mut saw_pad = false;
        let mut saw_high_count = false;

        for case in 0..512 {
            let dims = Dims::with_layers(2 + rng.below(6), 2 + rng.below(6), 1 + rng.below(4));
            saw_multilayer |= dims.layers > 1;

            let mut grid = Grid::filled(dims, 1);
            for cell in 0..dims.len() as CellIdx {
                let cost = match rng.below(12) {
                    0 | 1 => {
                        saw_obstacle = true;
                        OBSTACLE
                    }
                    2 => {
                        saw_zero = true;
                        0
                    }
                    3 | 4 => 1,
                    _ => {
                        saw_weighted = true;
                        2 + rng.below(2_000)
                    }
                };
                grid.set(cell, cost);
            }

            let src = rng.below(dims.len() as u32);
            let dst = rng.below(dims.len() as u32);
            let (sx, sy) = dims.xy(src);
            let (dx, dy) = dims.xy(dst);
            let min_x = sx.min(dx);
            let min_y = sy.min(dy);
            let max_x = sx.max(dx);
            let max_y = sy.max(dy);
            let window = Window {
                x0: rng.below(min_x + 1),
                y0: rng.below(min_y + 1),
                x1: max_x + rng.below(dims.w - max_x),
                y1: max_y + rng.below(dims.h - max_y),
            };
            saw_partial_window |= window != Window::full(dims);

            let mut pad_cells = Vec::new();
            for cell in 0..dims.len() as CellIdx {
                if window.contains(dims, cell)
                    && (grid.is_obstacle(cell) || rng.below(19) == 0)
                    && rng.below(5) == 0
                {
                    pad_cells.push(cell);
                }
            }
            if grid.is_obstacle(src) {
                pad_cells.push(src);
            }
            if grid.is_obstacle(dst) {
                pad_cells.push(dst);
            }
            pad_cells.sort_unstable();
            pad_cells.dedup();
            saw_pad |= !pad_cells.is_empty();
            let mut pads = PadSet::new(dims.len());
            pads.load(&pad_cells);

            let coords = if case % 3 == 0 {
                GridCoords::uniform(dims)
            } else {
                GridCoords::from_lines(random_axis(&mut rng, dims.w), random_axis(&mut rng, dims.h))
            };
            let heuristic_costs = ManhattanCosts::new(dims, &coords);

            let via_model = if dims.layers > 1 && case % 2 == 0 {
                saw_restricted_via = true;
                let allowed = (0..dims.layers - 1)
                    .filter(|_| rng.below(2) == 0)
                    .map(|layer| (layer, layer + 1))
                    .collect();
                ViaModel::with_allowed_steps(
                    dims.layers,
                    1 + rng.below(ViaModel::DEFAULT_STEP_COST * 3),
                    allowed,
                )
            } else {
                let mut model = ViaModel::through_hole(dims.layers);
                model.step_cost = 1 + rng.below(ViaModel::DEFAULT_STEP_COST * 3);
                model
            };

            let pfac = 1 + rng.below(MAX_ITERS);
            let mut present = vec![0u32; dims.len()];
            let mut present_halo = vec![0u32; dims.len()];
            let mut history = vec![0u32; dims.len()];
            let mut own_cells = Vec::new();
            let mut own_halo_counts = vec![0u32; dims.len()];

            for cell in 0..dims.len() as CellIdx {
                let ci = cell as usize;
                let self_present = u32::from(rng.below(7) == 0);
                if self_present != 0 {
                    own_cells.push(cell);
                }
                let self_halo = if rng.below(31) == 0 {
                    saw_high_count = true;
                    1_000_000 + rng.below(1_000_000)
                } else {
                    rng.below(9)
                };
                own_halo_counts[ci] = self_halo;

                let high_present = rng.below(23) == 0;
                let high_halo = rng.below(23) == 0;
                saw_high_count |= high_present || high_halo;
                let foreign_present = if high_present {
                    u32::MAX - self_present - rng.below(4_096)
                } else {
                    rng.below(32)
                };
                let foreign_halo = if high_halo {
                    u32::MAX - self_halo - rng.below(4_096)
                } else {
                    rng.below(64)
                };
                present[ci] = foreign_present + self_present;
                present_halo[ci] = foreign_halo + self_halo;
                history[ci] = if rng.below(29) == 0 {
                    u32::MAX - rng.below(4_096)
                } else {
                    rng.below(100_000)
                };
            }

            let mut own_path = PadSet::new(dims.len());
            own_path.load(&own_cells);
            let mut own_halo = CountedCellSet::new(dims.len());
            let mut combined_self = CountedCellSet::new(dims.len());
            own_halo.clear();
            combined_self.clear();
            for cell in 0..dims.len() as CellIdx {
                let ci = cell as usize;
                let halo = own_halo_counts[ci];
                let copper = u32::from(own_cells.binary_search(&cell).is_ok());
                if halo != 0 {
                    own_halo.stamp[ci] = own_halo.gen;
                    own_halo.count[ci] = halo;
                }
                let combined = halo + copper;
                if combined != 0 {
                    combined_self.stamp[ci] = combined_self.gen;
                    combined_self.count[ci] = combined;
                }
            }

            let present_factor = (pfac as u64) * (SCALE as u64);
            let halo_factor = (pfac as u64) * (CLEARANCE_NEG_WEIGHT as u64);
            let shared_congestion: Vec<u64> = present
                .iter()
                .zip(&present_halo)
                .map(|(&p, &h)| (present_factor * p as u64).saturating_add(halo_factor * h as u64))
                .collect();
            let has_zero_cost = grid.cost.contains(&0);

            let mut unfused_buf = SearchBuf::new(dims.len());
            let unfused = route_negotiated(
                &mut unfused_buf,
                &grid,
                &coords,
                &heuristic_costs,
                &pads,
                &[],
                Some(&own_path),
                Some(&own_halo),
                None,
                &present,
                &present_halo,
                &history,
                pfac,
                src,
                dst,
                window,
                &via_model,
                has_zero_cost,
            );
            let mut fused_buf = SearchBuf::new(dims.len());
            let fused = route_negotiated(
                &mut fused_buf,
                &grid,
                &coords,
                &heuristic_costs,
                &pads,
                &[],
                None,
                Some(&combined_self),
                Some(&shared_congestion),
                &present,
                &present_halo,
                &history,
                pfac,
                src,
                dst,
                window,
                &via_model,
                has_zero_cost,
            );
            assert_eq!(fused, unfused, "randomized pricing mismatch in case {case}");
        }

        assert!(saw_zero && saw_weighted && saw_obstacle);
        assert!(saw_multilayer && saw_restricted_via && saw_partial_window);
        assert!(saw_pad && saw_high_count);
    }

    /// Determinism with active clearance: repeated runs must produce a byte-identical
    /// [`BoardRoute`].
    #[test]
    fn clearance_active_route_is_deterministic() {
        // A handful of nets on a roomy board with the halo cost model active.
        let dims = Dims::new(12, 12);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 0), dims.idx(11, 11)),
            net("b", dims.idx(11, 0), dims.idx(0, 11)),
            net("c", dims.idx(0, 5), dims.idx(11, 5)),
            net("d", dims.idx(5, 0), dims.idx(5, 11)),
            net("e", dims.idx(0, 9), dims.idx(11, 2)),
        ];

        let router = NegotiatedRouter::new().with_clearance_cells(1);
        let br1 = router.route(&grid, &nets).unwrap();
        let br2 = router.route(&grid, &nets).unwrap();

        // Byte-identical across runs despite parallel scheduling.
        assert_eq!(br1.results, br2.results);
        assert_eq!(br1.unrouted, br2.unrouted);
        assert_eq!(br1.congestion, br2.congestion);
    }

    /// Three well-separated nets with clearance active must all route cell-disjointly.
    #[test]
    fn clearance_active_routes_all_on_roomy_board() {
        let dims = Dims::new(10, 10);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(0, 1), dims.idx(9, 1)),
            net("b", dims.idx(0, 5), dims.idx(9, 5)),
            net("c", dims.idx(0, 8), dims.idx(9, 8)),
        ];

        let br = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route(&grid, &nets)
            .unwrap();

        assert!(
            br.unrouted.is_empty(),
            "clearance route must place all nets on a roomy board: {br:?}"
        );
        assert_eq!(br.results.len(), 3);
        for i in 0..br.results.len() {
            for j in (i + 1)..br.results.len() {
                assert!(
                    disjoint(&br.results[i].path, &br.results[j].path),
                    "distinct nets must be cell-disjoint"
                );
            }
        }
    }

    #[test]
    fn nonuniform_stage_enforces_geometric_clearance() {
        // The windows are 38 cell indices apart, but on this dense Hanan axis the
        // copper is only 0.38 mm apart.  Cell-count staging used to co-schedule the
        // groups and commit both despite a 0.5 mm hard clearance.
        let dims = Dims::new(80, 1);
        let grid = GridBuilder::new(dims, 1).build();
        let coords =
            GridCoords::from_lines((0..dims.w).map(|i| i as f64 * 0.01).collect(), vec![0.0]);
        let nets = vec![net("a", 0, 2), net("b", 40, 42)];
        let board = NegotiatedRouter::new()
            .with_coords(coords.clone())
            .with_clearance_mm(0.5)
            .route(&grid, &nets)
            .unwrap();
        assert_eq!(
            board.results.len(),
            1,
            "both straight 1-D routes are too close"
        );
        assert_eq!(board.unrouted.len(), 1);
    }

    #[test]
    fn short_coordinate_arrays_fall_back_to_uniform_for_clearance() {
        let dims = Dims::new(3, 5);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(1, 0), dims.idx(1, 3)),
            net("b", dims.idx(0, 2), dims.idx(2, 2)),
        ];
        let uniform = NegotiatedRouter::new()
            .with_coords(GridCoords::uniform(dims))
            .with_clearance_cells(1)
            .route(&grid, &nets)
            .unwrap();
        let defensive = NegotiatedRouter::new()
            .with_coords(GridCoords::default())
            .with_clearance_cells(1)
            .route(&grid, &nets)
            .unwrap();
        assert_eq!(
            defensive, uniform,
            "missing lines use x_of/y_of unit fallback"
        );
    }

    #[test]
    fn roomy_clearance_crossing_converges_before_iteration_cap() {
        let dims = Dims::new(9, 9);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = vec![
            net("a", dims.idx(4, 1), dims.idx(4, 7)),
            net("b", dims.idx(1, 4), dims.idx(7, 4)),
        ];
        let (_board, trace) = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .route_traced(&grid, &nets)
            .unwrap();
        assert!(
            trace.iterations.len() < MAX_ITERS as usize,
            "a two-net open board must not oscillate for all {MAX_ITERS} iterations"
        );
        assert!(!trace.iterations.last().unwrap().any_overuse);
    }

    #[test]
    fn large_parallel_route_is_identical_across_rayon_pool_sizes() {
        let dims = Dims::new(24, 24);
        let grid = GridBuilder::new(dims, 1).build();
        let nets: Vec<_> = (0..18u32)
            .map(|i| net(&format!("n{i}"), dims.idx(0, i + 2), dims.idx(23, i + 2)))
            .collect();
        let route_in = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| NegotiatedRouter::new().route(&grid, &nets).unwrap())
        };
        let one = route_in(1);
        assert_eq!(route_in(2), one);
        assert_eq!(route_in(4), one);
    }

    #[test]
    fn parallel_clearance_route_is_identical_across_rayon_pool_sizes() {
        // More than PARALLEL_NEGOTIATION_THRESHOLD nets forces the Jacobi path.
        // Two central nets collide, ensuring a second iteration where each worker
        // must subtract its old counted halo; the other nets are separated fillers.
        let dims = Dims::new(40, 40);
        let grid = GridBuilder::new(dims, 1).build();
        let mut nets = vec![
            net("cross-v", dims.idx(25, 12), dims.idx(25, 28)),
            net("cross-h", dims.idx(17, 20), dims.idx(33, 20)),
        ];
        nets.extend((0..15u32).map(|i| {
            let y = i * 2;
            net(&format!("filler-{i}"), dims.idx(0, y), dims.idx(4, y))
        }));

        let route_in = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    NegotiatedRouter::new()
                        .with_clearance_cells(1)
                        .route_traced(&grid, &nets)
                        .unwrap()
                })
        };
        let one = route_in(1);
        assert!(
            one.1.iterations.len() > 1,
            "fixture must exercise old-path self-halo subtraction"
        );
        let assert_same = |got: (BoardRoute, RouteTrace)| {
            assert_eq!(got.0, one.0);
            assert_eq!(got.1.iterations.len(), one.1.iterations.len());
            for (a, b) in got.1.iterations.iter().zip(&one.1.iterations) {
                assert_eq!(
                    (a.iter, a.pfac, a.any_overuse),
                    (b.iter, b.pfac, b.any_overuse)
                );
                assert_eq!(a.paths, b.paths);
                assert_eq!(a.overused_cells, b.overused_cells);
            }
        };
        assert_same(route_in(2));
        assert_same(route_in(4));
    }

    const MULTILAYER_PORTFOLIO_ENDPOINTS: [(u32, u32, u32, u32, u32, u32); 17] = [
        (9, 0, 0, 0, 10, 0),
        (6, 13, 1, 13, 0, 1),
        (0, 3, 1, 13, 9, 1),
        (0, 4, 1, 10, 13, 0),
        (7, 0, 1, 4, 13, 1),
        (3, 0, 0, 8, 13, 0),
        (0, 13, 1, 13, 4, 1),
        (8, 0, 1, 3, 13, 1),
        (2, 0, 1, 13, 13, 1),
        (0, 8, 0, 13, 7, 0),
        (0, 4, 0, 10, 13, 1),
        (0, 6, 0, 13, 7, 1),
        (4, 13, 0, 3, 0, 1),
        (13, 2, 1, 2, 13, 0),
        (2, 13, 0, 13, 0, 0),
        (12, 0, 0, 0, 10, 0),
        (4, 0, 0, 12, 13, 0),
    ];

    fn multilayer_portfolio_nets(dims: Dims, grouped_first_pair: bool) -> Vec<NetEndpoints> {
        MULTILAYER_PORTFOLIO_ENDPOINTS
            .into_iter()
            .enumerate()
            .map(|(i, (sx, sy, sl, dx, dy, dl))| {
                let name = if grouped_first_pair && i < 2 {
                    format!("group#{i}")
                } else {
                    format!("n{i}")
                };
                net(&name, dims.idx3(sx, sy, sl), dims.idx3(dx, dy, dl))
            })
            .collect()
    }

    #[test]
    fn multilayer_parallel_clearance_preserves_better_jacobi_seed() {
        // This two-layer, 17-net fixture crosses the parallel threshold. The exact
        // Jacobi result legalizes five nets, while running the former unconditional
        // one-pass Gauss-Seidel polish first legalizes only four; the bounded order
        // portfolio now retains a sixth. Pin both the better outcome and cross-pool
        // determinism: multilayer routing must pass the Jacobi seed directly to
        // legalization instead of applying that polish.
        let dims = Dims::with_layers(14, 14, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = multilayer_portfolio_nets(dims, false);
        let group_ids = connection_group_ids(&nets);
        assert!(nets.len() > PARALLEL_NEGOTIATION_THRESHOLD);

        let route_in = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    NegotiatedRouter::new()
                        .with_clearance_cells(1)
                        .route_variant(
                            &grid,
                            &nets,
                            &group_ids,
                            NegotiationMode::Adaptive,
                            None,
                            None,
                        )
                        .unwrap()
                        .board
                })
        };
        let one = route_in(1);
        assert_eq!(
            one.results
                .iter()
                .map(|result| result.net.as_str())
                .collect::<Vec<_>>(),
            ["n0", "n1", "n2", "n3", "n15", "n16"],
            "the unpolished Jacobi seed plus bounded order portfolio must retain six nets"
        );
        assert_eq!(route_in(2), one);
        assert_eq!(route_in(4), one);
    }

    #[test]
    fn jacobi_thread_local_scratch_bounds_concurrent_routes() {
        let dims = Dims::with_layers(14, 14, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = multilayer_portfolio_nets(dims, false);
        let constructions = std::sync::Arc::new(AtomicUsize::new(0));
        let router = NegotiatedRouter::new()
            .with_clearance_cells(1)
            .with_jacobi_scratch_probe(constructions.clone());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let outcomes: Vec<_> = pool.install(|| {
            (0..8)
                .into_par_iter()
                .map(|_| router.route_traced(&grid, &nets).unwrap())
                .collect()
        });

        assert!(
            outcomes[0].1.iterations.len() > 1,
            "fixture must exercise scratch reuse across iterations"
        );
        for (board, trace) in &outcomes[1..] {
            assert_eq!(board, &outcomes[0].0);
            assert_trace_eq(trace, &outcomes[0].1);
        }
        let allocated = constructions.load(Ordering::Relaxed);
        assert!(
            (1..=4).contains(&allocated),
            "eight concurrent routes on four workers must allocate at most four \
             thread-local slots total, got {allocated}"
        );
    }

    #[test]
    fn serial_portfolio_prefers_more_complete_primary_with_identical_trace() {
        let dims = Dims::with_layers(14, 14, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = multilayer_portfolio_nets(dims, true);
        let group_ids = connection_group_ids(&nets);
        let router = NegotiatedRouter::new().with_clearance_cells(1);

        let mut primary_trace = empty_route_trace(dims);
        let primary = router
            .route_variant(
                &grid,
                &nets,
                &group_ids,
                NegotiationMode::Adaptive,
                None,
                Some(&mut primary_trace),
            )
            .unwrap()
            .board;
        let serial = router
            .route_variant(
                &grid,
                &nets,
                &group_ids,
                NegotiationMode::ForceSerial,
                None,
                None,
            )
            .unwrap()
            .board;
        assert_eq!((primary.results.len(), primary.total_cost()), (6, 129));
        assert_eq!((serial.results.len(), serial.total_cost()), (5, 88));
        assert!(!serial_candidate_is_better(&primary, &serial));
        assert_eq!(router.route(&grid, &nets).unwrap(), primary);

        let route_in = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| router.route_traced(&grid, &nets).unwrap())
        };
        let one = route_in(1);
        assert_eq!(one.0, primary);
        assert_trace_eq(&one.1, &primary_trace);
        for threads in [2, 4] {
            let got = route_in(threads);
            assert_eq!(got.0, one.0);
            assert_trace_eq(&got.1, &one.1);
        }
    }

    #[test]
    fn isolated_provider_batch_is_reused_across_bounded_portfolio_variants() {
        let dims = Dims::with_layers(14, 14, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let nets = multilayer_portfolio_nets(dims, true);
        let router = NegotiatedRouter::new().with_clearance_cells(1);

        // This bounded-portfolio fixture evaluates both the adaptive primary and
        // ForceSerial alternate before selecting the more complete primary.
        let (expected_board, expected_trace) = router.route_traced(&grid, &nets).unwrap();
        assert_eq!(
            (expected_board.results.len(), expected_board.total_cost()),
            (6, 129)
        );
        let provider = MockProvider::paths(provider_paths_from_trace(&expected_trace));
        let (board, trace) = router
            .route_traced_with_isolated_provider(&grid, &nets, &provider)
            .unwrap();

        assert_eq!(board, expected_board);
        assert_trace_eq(&trace, &expected_trace);
        assert_eq!(
            provider.calls(),
            1,
            "primary and serial variants must borrow one precomputed batch"
        );
    }

    #[test]
    fn serial_portfolio_selects_diversified_serial_when_it_routes_more() {
        let dims = Dims::with_layers(14, 14, 2);
        let grid = GridBuilder::new(dims, 1).build();
        let endpoints = [
            (97, 381),
            (176, 32),
            (271, 326),
            (169, 20),
            (131, 284),
            (110, 261),
            (213, 253),
            (128, 13),
            (115, 388),
            (86, 264),
            (356, 233),
            (77, 58),
            (236, 151),
            (380, 215),
            (62, 156),
            (104, 107),
            (276, 218),
        ];
        let nets: Vec<_> = endpoints
            .into_iter()
            .enumerate()
            .map(|(i, (src, dst))| {
                let name = if i < 2 {
                    format!("group#{i}")
                } else {
                    format!("n{i}")
                };
                net(&name, src, dst)
            })
            .collect();
        let group_ids = connection_group_ids(&nets);
        let router = NegotiatedRouter::new().with_clearance_cells(1);
        let primary = router
            .route_variant(
                &grid,
                &nets,
                &group_ids,
                NegotiationMode::Adaptive,
                None,
                None,
            )
            .unwrap()
            .board;
        let mut serial_trace = empty_route_trace(dims);
        let serial = router
            .route_variant(
                &grid,
                &nets,
                &group_ids,
                NegotiationMode::ForceSerial,
                None,
                Some(&mut serial_trace),
            )
            .unwrap()
            .board;

        assert_eq!((primary.results.len(), primary.total_cost()), (9, 120));
        assert_eq!((serial.results.len(), serial.total_cost()), (10, 117));
        assert!(serial_candidate_is_better(&primary, &serial));
        let (board, trace) = router.route_traced(&grid, &nets).unwrap();
        assert_eq!(board, serial);
        assert_trace_eq(&trace, &serial_trace);
    }

    #[test]
    fn geometric_clearance_boundary_is_inclusive() {
        let dims = Dims::new(6, 1);
        let coords = GridCoords::from_lines(vec![0.0, 0.2, 0.5, 1.0, 1.51, 2.1], vec![0.0]);
        assert_eq!(geom_box(&coords.x_lines, dims.w, 3, 0.5), (2, 4));
        assert_eq!(geom_box(&coords.x_lines, dims.w, 3, 0.51), (2, 5));
    }

    #[test]
    fn truncated_coordinates_scan_every_geometric_witness() {
        let dims = Dims::new(4, 1);
        // `x_of` positions are [0, 100, 2, 3]: the documented unit fallback after
        // the explicit prefix is intentionally non-monotonic.  Around seed x=3 at
        // radius 3, x=0,2,3 are in range while x=1 is not.  An outward monotonic
        // walk stops at x=1 and used to omit the disconnected x=0 witness.
        let coords = GridCoords::from_lines(vec![0.0, 100.0], vec![0.0]);
        let grid = Grid::filled(dims, 1);
        let mut visited = Vec::new();
        for_each_halo_cell(
            dims,
            &coords,
            &grid,
            &[dims.idx(3, 0)],
            3.0,
            &ViaModel::through_hole(dims.layers),
            |c| visited.push(c),
        );
        visited.sort_unstable();
        visited.dedup();
        assert_eq!(
            visited,
            vec![dims.idx(0, 0), dims.idx(2, 0), dims.idx(3, 0)]
        );
    }

    #[test]
    fn maximum_passable_weight_is_not_confused_with_obstacle() {
        let dims = Dims::new(2, 1);
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx(1, 0), OBSTACLE - 1);
        let nets = vec![net("high", dims.idx(0, 0), dims.idx(1, 0))];

        // Exercise the legalization search directly: multiplying the destination's
        // valid weight by the uniform geometric scale must stay below the sentinel.
        let mut buf = SearchBuf::new(dims.len());
        let mut pads = PadSet::new(dims.len());
        pads.load(&[]);
        let coords = GridCoords::uniform(dims);
        let heuristic_costs = ManhattanCosts::new(dims, &coords);
        let legal = route_legal(
            &mut buf,
            &grid,
            &coords,
            &heuristic_costs,
            &pads,
            &[],
            &[],
            &[],
            &ViaGuard::unoccupied(0.0, 0.0, 0.0),
            -1,
            dims.idx(0, 0),
            dims.idx(1, 0),
            Window::full(dims),
            &ViaModel::through_hole(dims.layers),
            0.0,
            false,
        );
        assert_eq!(legal.map(|(path, _)| path), Some(vec![0, 1]));

        let routed = NegotiatedRouter::new().route(&grid, &nets).unwrap();
        assert_eq!(routed.results.len(), 1);
        assert_eq!(routed.results[0].path, vec![dims.idx(0, 0), dims.idx(1, 0)]);
        assert_eq!(routed.results[0].cost, OBSTACLE - 1);

        grid.set(dims.idx(1, 0), OBSTACLE);
        assert!(matches!(
            NegotiatedRouter::new().route(&grid, &nets),
            Err(RouterError::InvalidEndpoint { .. })
        ));
    }

    // ---- heuristic admissibility (lower-bound) property ----------------------
    //
    // The search pays, per planar step `u -> v`, a base of `edge_cost(gap_uv)` (plus
    // non-negative congestion/via terms). So the planar base paid along any real path
    // is the SUM of per-step roundings `Σ edge_cost(g_i)`. The heuristic must never
    // exceed that, or A* (with `astar_buf`'s break-on-pop-dst) can return a non-optimal
    // path. The old `edge_cost(manhattan_len(a, b))` rounded the AGGREGATE length once,
    // and round-of-sum can exceed sum-of-rounds, so it could overestimate on
    // non-integer line spacings. These tests pin the lower-bound property the new
    // per-axis summed form (`manhattan_scaled`) guarantees, and exhibit a concrete
    // spacing where the OLD form violated it.

    /// What the search actually pays in planar base cost along a sequence of cells:
    /// `Σ edge_cost(manhattan_len(step))` over consecutive pairs — the exact per-step
    /// rounding `astar_buf`'s `cost_fn` accrues (congestion-free).
    fn summed_path_base(dims: Dims, coords: &GridCoords, path: &[CellIdx]) -> Cost {
        path.windows(2)
            .map(|w| edge_cost(coords.manhattan_len(dims, w[0], w[1])))
            .fold(0u32, Cost::saturating_add)
    }

    /// The OLD (inadmissible) heuristic: a single rounding of the aggregate length.
    fn old_manhattan_planar(dims: Dims, coords: &GridCoords, a: CellIdx, b: CellIdx) -> Cost {
        edge_cost(coords.manhattan_len(dims, a, b))
    }

    /// Former O(axis-distance) implementation retained as a test oracle for the
    /// prefix lookup. This mirrors the coordinate fallback and saturation exactly.
    fn scanned_axis_leg_cost(lines: &[f64], i: u32, j: u32) -> Cost {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        let mut total: Cost = 0;
        for k in lo..hi {
            let a = lines.get(k as usize).copied().unwrap_or(k as f64);
            let b = lines.get(k as usize + 1).copied().unwrap_or((k + 1) as f64);
            total = total.saturating_add(edge_cost((b - a).abs()));
        }
        total
    }

    #[test]
    fn heuristic_prefix_exactly_matches_gap_scan_and_saturation() {
        // Complete, truncated, empty, non-monotonic defensive input, and totals
        // that saturate `Cost`. Only indices inside the routed dimension are
        // queried; missing line positions use the documented index fallback.
        let cases: &[(u32, &[f64])] = &[
            (6, &[0.0, 0.03125, 0.5, 2.25, 2.25, 9.0]),
            (7, &[10.0, 10.125]),
            (5, &[]),
            (6, &[0.0, 4.0, -2.0]),
            (5, &[0.0, 1.0e20, 2.0e20, 3.0e20, 4.0e20]),
        ];
        for &(count, lines) in cases {
            let prefix = axis_cost_prefix(lines, count);
            assert_eq!(prefix.len(), count as usize);
            for i in 0..count {
                for j in 0..count {
                    assert_eq!(
                        axis_leg_cost(&prefix, i, j),
                        scanned_axis_leg_cost(lines, i, j),
                        "count={count} lines={lines:?} interval=({i},{j})"
                    );
                }
            }
        }

        assert_eq!(axis_cost_prefix(&[], 0), vec![0]);
    }

    #[test]
    fn heuristic_is_lower_bound_collinear_and_l_shaped() {
        // Sweep a range of non-integer gap patterns on a single layer (no via term),
        // and assert the heuristic never exceeds the per-step summed path base for
        // both a straight (collinear) path and an L-shaped path between the corners.
        let gap_sets: &[&[f64]] = &[
            &[0.5, 0.5, 0.5, 0.5],        // halves: round-of-sum vs sum-of-rounds
            &[0.03125, 0.03125, 0.03125], // 0.5/16 each: each rounds to 0, sum doesn't
            &[1.5, 2.5, 0.5, 3.5],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[0.46875, 0.46875, 0.46875], // 7.5/16 each
            &[2.0, 2.0, 2.0],             // integers: must match exactly (uniform-like)
        ];
        for xs in gap_sets {
            for ys in gap_sets {
                let x_lines = cumsum(xs);
                let y_lines = cumsum(ys);
                let dims = Dims::new(x_lines.len() as u32, y_lines.len() as u32);
                let coords = GridCoords::from_lines(x_lines.clone(), y_lines.clone());
                let heuristic_costs = ManhattanCosts::new(dims, &coords);
                let a = dims.idx(0, 0);
                let b = dims.idx(dims.w - 1, dims.h - 1);

                let h = manhattan_scaled(dims, &heuristic_costs, a, b, 0);

                // Collinear leg along x (then y handled by the y-only pair below).
                let x_only = manhattan_scaled(
                    dims,
                    &heuristic_costs,
                    dims.idx(0, 0),
                    dims.idx(dims.w - 1, 0),
                    0,
                );
                let x_path: Vec<CellIdx> = (0..dims.w).map(|x| dims.idx(x, 0)).collect();
                assert!(
                    x_only <= summed_path_base(dims, &coords, &x_path),
                    "collinear-x heuristic must be a lower bound: h={x_only} > path"
                );

                // L-shaped path: straight along x to the far column, then up in y.
                let mut l_path: Vec<CellIdx> = (0..dims.w).map(|x| dims.idx(x, 0)).collect();
                l_path.extend((1..dims.h).map(|y| dims.idx(dims.w - 1, y)));
                let l_cost = summed_path_base(dims, &coords, &l_path);
                assert!(
                    h <= l_cost,
                    "L-shaped heuristic must be a lower bound: h={h} > path={l_cost} \
                     (xs={xs:?}, ys={ys:?})"
                );

                // The other L (up first, then across) — same corners, also a lower bound.
                let mut l_path2: Vec<CellIdx> = (0..dims.h).map(|y| dims.idx(0, y)).collect();
                l_path2.extend((1..dims.w).map(|x| dims.idx(x, dims.h - 1)));
                let l_cost2 = summed_path_base(dims, &coords, &l_path2);
                assert!(
                    h <= l_cost2,
                    "L-shaped(2) heuristic must be a lower bound: h={h} > path={l_cost2}"
                );
            }
        }
    }

    #[test]
    fn old_heuristic_could_overestimate_but_new_does_not() {
        // Concrete witness: three gaps of 0.4/16 mm each. Per-step the search pays
        // edge_cost(0.4/16) = round(0.4) = 0 per step → 0 total. The aggregate is
        // 1.2/16 → edge_cost = round(1.2) = 1. So the OLD heuristic returns 1 > 0,
        // overestimating (inadmissible: round-of-sum > sum-of-rounds). The new summed
        // heuristic returns 0 == 0.
        let g = 0.4 / COST_SCALE;
        let x_lines = cumsum(&[g, g, g]); // 4 lines, 3 gaps
        let dims = Dims::new(x_lines.len() as u32, 1);
        let coords = GridCoords::from_lines(x_lines, vec![0.0]);
        let heuristic_costs = ManhattanCosts::new(dims, &coords);
        let a = dims.idx(0, 0);
        let b = dims.idx(dims.w - 1, 0);
        let path: Vec<CellIdx> = (0..dims.w).map(|x| dims.idx(x, 0)).collect();
        let summed = summed_path_base(dims, &coords, &path);

        let old = old_manhattan_planar(dims, &coords, a, b);
        let new = manhattan_scaled(dims, &heuristic_costs, a, b, 0);

        assert_eq!(summed, 0, "search pays 0 per-step here");
        assert!(
            old > summed,
            "old heuristic must overestimate in this witness: old={old} summed={summed}"
        );
        assert!(
            new <= summed,
            "new heuristic must stay a lower bound: new={new} summed={summed}"
        );
    }

    #[test]
    fn heuristic_uniform_grid_byte_identical() {
        // On a uniform unit grid the new summed heuristic must equal both the old
        // aggregate form and the historical (dx + dy) * SCALE, preserving byte-identity.
        let dims = Dims::new(7, 5);
        let coords = GridCoords::uniform(dims);
        let heuristic_costs = ManhattanCosts::new(dims, &coords);
        for &(ax, ay, bx, by) in &[(0u32, 0u32, 6u32, 4u32), (3, 1, 3, 4), (0, 2, 6, 2)] {
            let a = dims.idx(ax, ay);
            let b = dims.idx(bx, by);
            let new = manhattan_scaled(dims, &heuristic_costs, a, b, 0);
            let old = old_manhattan_planar(dims, &coords, a, b);
            let expected = (ax.abs_diff(bx) + ay.abs_diff(by)) * SCALE;
            assert_eq!(new, old, "uniform: new must equal old aggregate form");
            assert_eq!(new, expected, "uniform: new must equal (dx+dy)*SCALE");
        }
    }

    #[test]
    fn overlapping_halo_survives_owner_rip_deterministically() {
        // With clearance 1, copper at x=1 and x=3 is legal (distance 2), but both
        // halos cover x=2.  The overlap must block both owners while both remain;
        // after ripping group 0 it must become group 1's ordinary halo, not free.
        let dims = Dims::new(5, 1);
        let grid = GridBuilder::new(dims, 1).build();
        let coords = GridCoords::uniform(dims);
        let via_model = ViaModel::through_hole(1);
        let mut committed: Committed = vec![Some(vec![dims.idx(1, 0)]), Some(vec![dims.idx(3, 0)])];
        let group_ids = vec![0, 1];
        let mut owner = vec![-1; dims.len()];
        let mut halo = vec![HALO_FREE; dims.len()];
        let mut via_guard = ViaGuard::dense(dims, 1.0, 0.0);

        rebuild_owner_maps(
            &mut owner,
            &mut halo,
            &mut via_guard,
            &grid,
            &coords,
            &committed,
            &group_ids,
            1.0,
            &via_model,
        );
        let overlap = dims.idx(2, 0) as usize;
        assert_eq!(halo[overlap], HALO_MIXED);
        assert!(halo_is_foreign(halo[overlap], 0));
        assert!(halo_is_foreign(halo[overlap], 1));

        committed[0] = None;
        rebuild_owner_maps(
            &mut owner,
            &mut halo,
            &mut via_guard,
            &grid,
            &coords,
            &committed,
            &group_ids,
            1.0,
            &via_model,
        );
        assert_eq!(halo[overlap], 1, "surviving reservation must transfer");
        assert!(halo_is_foreign(halo[overlap], 0));
        assert!(!halo_is_foreign(halo[overlap], 1));
    }

    #[test]
    fn multistep_via_guard_rebuilds_once_per_layer_after_rip() {
        let dims = Dims::with_layers(3, 1, 3);
        let grid = Grid::filled(dims, 1);
        let coords = GridCoords::from_lines(vec![0.0, 0.5, 1.0], vec![0.0]);
        let via_model = ViaModel::through_hole(3);
        let path0 = vec![dims.idx3(0, 0, 0), dims.idx3(0, 0, 1), dims.idx3(0, 0, 2)];
        let path1 = vec![dims.idx3(2, 0, 0), dims.idx3(2, 0, 1), dims.idx3(2, 0, 2)];
        let overlap: Vec<_> = (0..3).map(|layer| dims.idx3(1, 0, layer)).collect();
        let mut committed: Committed = vec![Some(path0.clone()), Some(path1.clone())];
        let group_ids = vec![0, 1];
        let mut owner = vec![-1; dims.len()];
        let mut halo = vec![HALO_FREE; dims.len()];
        let mut via_guard = ViaGuard::dense(dims, 0.0, 0.60);

        rebuild_owner_maps(
            &mut owner,
            &mut halo,
            &mut via_guard,
            &grid,
            &coords,
            &committed,
            &group_ids,
            0.0,
            &via_model,
        );
        assert_eq!(
            via_guard
                .tags
                .iter()
                .filter(|&&tag| tag != VIA_TAG_FREE)
                .count(),
            9,
            "each layer has two centres and their mixed overlap"
        );
        assert!(path0.iter().all(|&cell| via_guard.tags[cell as usize] == 0));
        assert!(path1.iter().all(|&cell| via_guard.tags[cell as usize] == 1));
        assert!(overlap
            .iter()
            .all(|&cell| via_guard.tags[cell as usize] == VIA_TAG_MIXED));
        assert!(!path_has_foreign_via_conflict(
            dims,
            &coords,
            &via_guard,
            &path0[..2],
            0,
        ));
        assert!(path_has_foreign_via_conflict(
            dims,
            &coords,
            &via_guard,
            &path1[..2],
            0,
        ));
        assert!(path_has_foreign_via_conflict(
            dims,
            &coords,
            &via_guard,
            &overlap[..2],
            0,
        ));

        committed[0] = None;
        rebuild_owner_maps(
            &mut owner,
            &mut halo,
            &mut via_guard,
            &grid,
            &coords,
            &committed,
            &group_ids,
            0.0,
            &via_model,
        );
        assert_eq!(
            via_guard
                .tags
                .iter()
                .filter(|&&tag| tag != VIA_TAG_FREE)
                .count(),
            6
        );
        assert!(path0
            .iter()
            .all(|&cell| via_guard.tags[cell as usize] == VIA_TAG_FREE));
        assert!(path1.iter().all(|&cell| via_guard.tags[cell as usize] == 1));
        assert!(overlap
            .iter()
            .all(|&cell| via_guard.tags[cell as usize] == 1));
    }

    #[test]
    fn dense_via_guard_matches_brute_force_feature_union() {
        // The truncated x coordinates deliberately trigger geom_box's defensive
        // full-axis path: x positions become [0, 100, 2, 3, 4]. The oracle below
        // knows nothing about geom_box and checks every cell directly in Euclidean
        // space, pinning completeness as well as exact diagonal/boundary semantics.
        let dims = Dims::with_layers(5, 3, 2);
        let coords = GridCoords::from_lines(vec![0.0, 100.0], vec![0.0, 0.4, 0.8]);
        let trace_mm = 0.55;
        let via_mm = 0.85;
        let via_path = vec![dims.idx3(3, 1, 0), dims.idx3(3, 1, 1)];
        let trace_path = [dims.idx3(2, 0, 0), dims.idx3(3, 0, 0)];
        let features = [(&via_path[..], 0_i64), (&trace_path[..], 1_i64)];

        let mut guard = ViaGuard::dense(dims, trace_mm, via_mm);
        for &(path, group) in &features {
            guard.stamp_path(dims, &coords, path, group);
        }

        for candidate in 0..dims.len() as CellIdx {
            let (cx, cy, cl) = dims.xyz(candidate);
            let candidate_x = coords.x_of(cx);
            let candidate_y = coords.y_of(cy);
            let mut covering_groups = Vec::new();
            for &(path, group) in &features {
                for (i, &feature) in path.iter().enumerate() {
                    let (fx, fy, fl) = dims.xyz(feature);
                    if fl != cl {
                        continue;
                    }
                    let is_via = (i > 0 && {
                        let (px, py, pl) = dims.xyz(path[i - 1]);
                        px == fx && py == fy && pl != fl
                    }) || (i + 1 < path.len() && {
                        let (nx, ny, nl) = dims.xyz(path[i + 1]);
                        nx == fx && ny == fy && nl != fl
                    });
                    let radius = if is_via {
                        trace_mm.max(via_mm)
                    } else {
                        trace_mm
                    };
                    let threshold = (radius - VIA_SPACING_EPS_MM).max(0.0);
                    let dx = coords.x_of(fx) - candidate_x;
                    let dy = coords.y_of(fy) - candidate_y;
                    if dx * dx + dy * dy < threshold * threshold
                        && !covering_groups.contains(&group)
                    {
                        covering_groups.push(group);
                    }
                }
            }
            let expected = match covering_groups.as_slice() {
                [] => VIA_TAG_FREE,
                [group] => *group as i32,
                _ => VIA_TAG_MIXED,
            };
            assert_eq!(guard.tags[candidate as usize], expected, "cell {candidate}");
        }

        // Repeating one sibling group's stamp is idempotent; a foreign feature at
        // the same centre makes the tag mixed, so neither group may ignore it.
        let centre = via_path[0];
        guard.clear();
        guard.stamp_path(dims, &coords, &via_path, 0);
        guard.stamp_path(dims, &coords, &via_path, 0);
        assert_eq!(guard.tags[centre as usize], 0);
        guard.stamp_path(dims, &coords, &via_path, 1);
        assert_eq!(guard.tags[centre as usize], VIA_TAG_MIXED);
        assert!(guard.conflicts_cell(centre, 0));
        assert!(guard.conflicts_cell(centre, 1));
    }

    #[test]
    fn committed_via_guard_is_symmetric_exact_and_cannot_be_bypassed_by_an_own_pad() {
        // One deliberately non-uniform row gives us both a point 0.5 mm inside
        // the open 0.6 mm keepout and another exactly 0.6 mm from the feature.
        let dims = Dims::with_layers(6, 1, 2);
        let coords = GridCoords::from_lines(vec![0.0, 0.5, 1.0, 1.5, 1.6, 2.1], vec![0.0]);
        let heuristic_costs = ManhattanCosts::new(dims, &coords);
        let mut via_model = ViaModel::through_hole(dims.layers);
        via_model.keepout_mm = 0.6;
        let foreign_via = [dims.idx3(2, 0, 0), dims.idx3(2, 0, 1)];
        let inside = dims.idx3(3, 0, 0);
        let outside = dims.idx3(4, 0, 0);

        // Stamp a committed via through a base obstacle. Its neighbouring own-pad
        // endpoint remains blocked before the feature-aware flag is enabled, but
        // the enabled field wins in planar `cost_fn` before endpoint exemptions.
        let mut builder = GridBuilder::new(dims, 1);
        builder.mark_cell_layer(3, 0, 0);
        let grid = builder.build();
        let mut pads = PadSet::new(dims.len());
        pads.load(&[inside]);
        let mut enabled = ViaGuard::dense_with_planar_via_guard(dims, 0.6, 0.8, 0.0, true);
        enabled.stamp_path(dims, &coords, &foreign_via, 0);
        assert_eq!(enabled.committed_via_tags[inside as usize], 0);
        assert!(path_has_foreign_via_conflict(
            dims,
            &coords,
            &enabled,
            &[outside, inside],
            1
        ));

        let route = |guard: &ViaGuard| {
            route_legal(
                &mut SearchBuf::new(dims.len()),
                &grid,
                &coords,
                &heuristic_costs,
                &pads,
                &[],
                &[],
                &[],
                guard,
                1,
                outside,
                inside,
                Window::full(dims),
                &via_model,
                0.0,
                false,
            )
        };
        assert!(route(&enabled).is_none());

        let mut disabled = ViaGuard::dense_with_planar_via_guard(dims, 0.6, 0.8, 0.0, false);
        disabled.stamp_path(dims, &coords, &foreign_via, 0);
        assert!(disabled.committed_via_tags.is_empty());
        assert_eq!(route(&disabled).unwrap().0, [outside, inside]);

        // Symmetric direction: a committed planar trace goes into the ordinary
        // candidate-via field. A foreign via 0.5 mm away is rejected, while a via
        // exactly 0.6 mm away is legal because the DRC boundary is inclusive.
        let empty = Grid::filled(dims, 1);
        let trace = [dims.idx3(2, 0, 0), dims.idx3(1, 0, 0)];
        let mut trace_guard = ViaGuard::dense_with_planar_via_guard(dims, 0.6, 0.8, 0.0, true);
        trace_guard.stamp_path(dims, &coords, &trace, 0);
        let via_route = |x: u32| {
            let src = dims.idx3(x, 0, 0);
            let dst = dims.idx3(x, 0, 1);
            route_legal(
                &mut SearchBuf::new(dims.len()),
                &empty,
                &coords,
                &heuristic_costs,
                &PadSet::new(dims.len()),
                &[],
                &[],
                &[],
                &trace_guard,
                1,
                src,
                dst,
                Window {
                    x0: x,
                    y0: 0,
                    x1: x,
                    y1: 0,
                },
                &via_model,
                0.0,
                false,
            )
        };
        assert!(via_route(3).is_none(), "0.5 mm is inside the open guard");
        assert_eq!(
            via_route(4).unwrap().0,
            [dims.idx3(4, 0, 0), dims.idx3(4, 0, 1)],
            "the exact 0.6 mm boundary is legal"
        );

        // A planar-only feature never enters the committed-via-only field, so it
        // does not accidentally widen trace↔trace spacing.
        assert!(trace_guard
            .committed_via_tags
            .iter()
            .all(|&tag| tag == VIA_TAG_FREE));
        assert_eq!(
            route_legal(
                &mut SearchBuf::new(dims.len()),
                &empty,
                &coords,
                &heuristic_costs,
                &PadSet::new(dims.len()),
                &[],
                &[],
                &[],
                &trace_guard,
                1,
                dims.idx3(3, 0, 0),
                dims.idx3(4, 0, 0),
                Window::full(dims),
                &via_model,
                0.0,
                false,
            )
            .unwrap()
            .0,
            [dims.idx3(3, 0, 0), dims.idx3(4, 0, 0)]
        );

        assert_eq!(
            enabled.tags.len() * std::mem::size_of::<i32>()
                + enabled.committed_via_tags.len() * std::mem::size_of::<i32>(),
            dims.len() * std::mem::size_of::<i64>(),
            "two i32 fields retain the former one-i64-field memory ceiling"
        );
    }

    #[test]
    fn via_hole_guard_is_net_independent_but_reuses_same_group_coincident_sites() {
        let dims = Dims::with_layers(6, 1, 2);
        let coords = GridCoords::from_lines(
            vec![0.0, 0.004, 0.005, 0.005_001, 0.199_999, 0.2],
            vec![0.0],
        );
        let centre = dims.idx3(0, 0, 0);
        let mut guard = ViaGuard::dense_with_planar_via_guard(dims, 0.3, 0.4, 0.2, true);
        let plane = (dims.w * dims.h) as usize;
        assert_eq!(
            guard.hole_centres.len() * std::mem::size_of::<i32>(),
            plane * std::mem::size_of::<i32>()
        );
        assert_eq!(guard.hole_blocked.len(), plane);
        let word_bits = usize::BITS as usize;
        assert!(guard.hole_blocked.capacity() >= plane);
        assert!(guard.hole_blocked.capacity() <= plane.div_ceil(word_bits) * word_bits);
        let combined_allocated_bytes = (guard.tags.capacity()
            + guard.committed_via_tags.capacity()
            + guard.hole_centres.capacity())
            * std::mem::size_of::<i32>()
            + guard.hole_blocked.capacity().div_ceil(8);
        let combined_bound_bytes = dims.len() * 2 * std::mem::size_of::<i32>()
            + plane * std::mem::size_of::<i32>()
            + plane.div_ceil(word_bits) * word_bits / 8;
        assert!(
            combined_allocated_bytes <= combined_bound_bytes,
            "the complete feature-aware guard is bounded by two per-layer i32 maps plus one planar i32 map and one planar bitset"
        );
        guard.stamp_hole(dims, &coords, centre, 7);

        let cell = |x| dims.idx3(x, 0, 1);
        assert!(!guard.conflicts_hole_cell(dims, cell(1), 7));
        assert!(!guard.conflicts_hole_cell(dims, cell(2), 7));
        assert!(
            guard.conflicts_hole_cell(dims, cell(1), 8),
            "coincident drill reuse is same-group only"
        );
        assert!(
            guard.conflicts_hole_cell(dims, cell(3), 7),
            "a distinct same-group drill just beyond 0.005 mm is blocked"
        );
        assert!(guard.conflicts_hole_cell(dims, cell(4), 7));
        assert!(
            !guard.conflicts_hole_cell(dims, cell(5), 7),
            "the exact required-spacing boundary is legal"
        );

        let two_via_path = |x| {
            vec![
                dims.idx3(0, 0, 0),
                dims.idx3(0, 0, 1),
                dims.idx3(x, 0, 1),
                dims.idx3(x, 0, 0),
            ]
        };
        assert_eq!(
            first_self_via_hole_conflict(dims, &coords, &two_via_path(2), 0.2),
            None,
            "one path may represent a stacked site at the 0.005 mm tolerance"
        );
        assert!(first_self_via_hole_conflict(dims, &coords, &two_via_path(3), 0.2).is_some());
        assert!(first_self_via_hole_conflict(dims, &coords, &two_via_path(4), 0.2).is_some());
        assert_eq!(
            first_self_via_hole_conflict(dims, &coords, &two_via_path(5), 0.2),
            None
        );
    }

    #[test]
    fn same_group_siblings_enforce_distinct_holes_but_reuse_coincident_sites() {
        let committed_at = |separation: f64| {
            let dims = Dims::with_layers(2, 1, 2);
            let coords = GridCoords::from_lines(vec![0.0, separation], vec![0.0]);
            let grid = Grid::filled(dims, OBSTACLE);
            let paths = vec![
                vec![dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)],
                vec![dims.idx3(1, 0, 0), dims.idx3(1, 0, 1)],
            ];
            let nets: Vec<_> = paths
                .iter()
                .enumerate()
                .map(|(index, path)| NetEndpoints {
                    net: format!("shared#{index}"),
                    src: path[0],
                    dst: path[1],
                    passable_pads: path.clone(),
                    via_passable_pads: path.clone(),
                })
                .collect();
            let windows = vec![Window::full(dims); 2];
            legalize_in_order(
                &grid,
                &coords,
                &ManhattanCosts::new(dims, &coords),
                &mut SearchBuf::new(dims.len()),
                &mut PadSet::new(dims.len()),
                &nets,
                &[0, 0],
                &paths,
                &windows,
                &[0],
                dims.len(),
                &ViaModel::through_hole(dims.layers),
                0.0,
                0.0,
                0.2,
                false,
                false,
            )
        };

        assert_eq!(
            committed_at(VIA_SAME_LOCATION_MM)
                .iter()
                .filter(|path| path.is_some())
                .count(),
            2,
            "same-group records within the producer tolerance reuse one drill site"
        );
        let outside = committed_at(VIA_SAME_LOCATION_MM + 1e-6);
        assert!(outside[0].is_some());
        assert!(
            outside[1].is_none(),
            "the immediately stamped first sibling hole must block a distinct forced site"
        );
    }

    #[test]
    fn route_legal_deterministically_branches_away_from_a_self_hole_conflict() {
        let dims = Dims::with_layers(4, 1, 2);
        let coords = GridCoords::from_lines(vec![0.0, 0.1, 0.2, 0.5], vec![0.0]);
        let mut builder = GridBuilder::new(dims, 1);
        builder.mark_cell_layer(1, 0, 0);
        let grid = builder.build();
        let src = dims.idx3(0, 0, 0);
        let dst = dims.idx3(2, 0, 0);
        let heuristic = ManhattanCosts::new(dims, &coords);
        let mut pads = PadSet::new(dims.len());
        pads.load(&[]);
        let guard = ViaGuard::unoccupied(0.0, 0.0, 0.3);
        let via_model = ViaModel::through_hole(dims.layers);

        let raw = route_legal_once(
            &mut SearchBuf::new(dims.len()),
            &grid,
            &coords,
            &heuristic,
            &pads,
            &[],
            &[],
            &[],
            &guard,
            &[],
            0,
            src,
            dst,
            Window::full(dims),
            &via_model,
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(
            first_self_via_hole_conflict(dims, &coords, &raw.0, 0.3),
            Some((0, 2)),
            "the unconstrained shortest staircase uses the close landing: {:?}",
            raw.0
        );

        let repaired = || {
            route_legal(
                &mut SearchBuf::new(dims.len()),
                &grid,
                &coords,
                &heuristic,
                &pads,
                &[],
                &[],
                &[],
                &guard,
                0,
                src,
                dst,
                Window::full(dims),
                &via_model,
                0.0,
                false,
            )
            .unwrap()
            .0
        };
        let first = repaired();
        assert!(first_self_via_hole_conflict(dims, &coords, &first, 0.3).is_none());
        assert!(first.windows(2).any(|step| {
            let (ax, _, al) = dims.xyz(step[0]);
            let (bx, _, bl) = dims.xyz(step[1]);
            ax == 3 && bx == 3 && al != bl
        }));
        assert_eq!(repaired(), first, "the bounded branch is deterministic");
    }

    #[test]
    fn disabled_or_single_layer_via_guard_stays_unallocated() {
        let dims = Dims::with_layers(100, 100, 4);
        let coords = GridCoords::uniform(dims);
        let path = [dims.idx3(50, 50, 0), dims.idx3(50, 50, 1)];
        let mut guard = ViaGuard::dense(dims, 0.45, 0.0);
        assert!(guard.tags.is_empty());
        assert!(guard.hole_blocked.is_empty());
        assert!(guard.hole_centres.is_empty());
        assert!(!guard.feature_aware());
        guard.stamp_path(dims, &coords, &path, 0);
        assert!(
            guard.tags.is_empty(),
            "disabled mode must not allocate lazily"
        );
        assert!(!path_has_foreign_via_conflict(
            dims, &coords, &guard, &path, 1
        ));

        let planar_dims = Dims::new(100, 100);
        let mut planar = ViaGuard::dense_with_planar_via_guard(planar_dims, 0.45, 0.60, 0.75, true);
        assert!(planar.tags.is_empty());
        assert!(planar.committed_via_tags.is_empty());
        assert!(planar.hole_blocked.is_empty());
        assert!(planar.hole_centres.is_empty());
        planar.stamp_path(planar_dims, &GridCoords::uniform(planar_dims), &[0, 1], 0);
        assert!(
            planar.tags.is_empty(),
            "single-layer routing must never allocate a guard it cannot query"
        );
    }

    /// Prefix sums starting at 0.0 → a sorted line array of `gaps.len() + 1` lines.
    fn cumsum(gaps: &[f64]) -> Vec<f64> {
        let mut lines = Vec::with_capacity(gaps.len() + 1);
        let mut acc = 0.0;
        lines.push(acc);
        for &g in gaps {
            acc += g;
            lines.push(acc);
        }
        lines
    }

    /// The load-bearing instrumentation invariant: enabling the trace recorder must
    /// NOT change routing. `route` and `route_traced().0` must produce an IDENTICAL
    /// `BoardRoute` on every fixture (single-net battery + multi-net contention
    /// cases), and the trace must be internally consistent with that result.
    #[test]
    fn route_traced_matches_route_byte_for_byte() {
        // Build a battery: the shared single-net fixtures plus multi-net cases that
        // actually exercise the negotiation loop AND legalization.
        let mut cases: Vec<(String, Grid, Vec<NetEndpoints>)> = Vec::new();
        for f in mr_fixtures::obstacle_battery() {
            cases.push((f.name.to_string(), f.grid, f.nets));
        }
        {
            // Crossing nets (negotiation must separate them).
            let dims = Dims::new(5, 5);
            let grid = GridBuilder::new(dims, 1).build();
            cases.push((
                "crossing".into(),
                grid,
                vec![
                    net("a", dims.idx(2, 1), dims.idx(2, 3)),
                    net("b", dims.idx(1, 2), dims.idx(3, 2)),
                ],
            ));
        }
        {
            // Three nets, one foreign — exercises grouping + multi-order legalization.
            let dims = Dims::new(8, 8);
            let grid = GridBuilder::new(dims, 1).build();
            cases.push((
                "grouped".into(),
                grid,
                vec![
                    net("g#0", dims.idx(0, 0), dims.idx(7, 0)),
                    net("g#1", dims.idx(0, 2), dims.idx(7, 2)),
                    net("foreign", dims.idx(0, 5), dims.idx(7, 5)),
                ],
            ));
        }

        for (name, grid, nets) in &cases {
            let router = NegotiatedRouter::new();
            let plain = router.route(grid, nets).unwrap();
            let (traced_board, trace) = router.route_traced(grid, nets).unwrap();
            assert_eq!(
                plain, traced_board,
                "fixture `{name}`: route_traced must yield an identical BoardRoute"
            );

            // Trace self-consistency.
            assert_eq!(trace.dims, grid.dims, "fixture `{name}`: trace dims");
            assert_eq!(
                trace.nets.len(),
                nets.len(),
                "fixture `{name}`: one TracedNet per input net"
            );
            let leg = trace
                .legalization
                .as_ref()
                .unwrap_or_else(|| panic!("fixture `{name}`: legalization recorded"));
            assert_eq!(
                leg.committed.len(),
                nets.len(),
                "fixture `{name}`: committed has one slot per net"
            );
            // The final committed routes in the trace must match the BoardRoute: each
            // routed net's committed path equals its RouteResult path.
            for (i, ep) in nets.iter().enumerate() {
                let in_board = traced_board.results.iter().find(|r| r.net == ep.net);
                match (&leg.committed[i], in_board) {
                    (Some(path), Some(r)) => assert_eq!(
                        path, &r.path,
                        "fixture `{name}`: committed path matches result for `{}`",
                        ep.net
                    ),
                    (None, None) => {}
                    other => panic!(
                        "fixture `{name}`: committed/result disagree for `{}`: {other:?}",
                        ep.net
                    ),
                }
            }
            // If the loop ran at all, the last recorded iteration is the converged
            // one (no overuse) OR the loop hit MAX_ITERS.
            if let Some(last) = trace.iterations.last() {
                assert!(
                    !last.any_overuse || last.iter == MAX_ITERS - 1,
                    "fixture `{name}`: trace should end converged or at MAX_ITERS"
                );
                // Every snapshot has one path slot per net.
                for snap in &trace.iterations {
                    assert_eq!(snap.paths.len(), nets.len());
                }
            }
        }
    }
}
