//! `mr-metal` — the GPU heart of metalroute (plan M3/M4).
//!
//! This crate ports the single-source distance-field computation behind the
//! routers onto Apple-Silicon GPUs via Metal compute kernels (the `metal`
//! crate). The CPU implementation in `mr-cpu` is the correctness ORACLE: every
//! field this crate produces must equal `mr_cpu::bfs_distance_field` /
//! `mr_cpu::sweep_distance_field` element-wise, and routed boards must be
//! `mr_oracle::are_equivalent` to `mr_cpu::LeeRouter`.
//!
//! Two kernels are implemented, both atomic-free with ping-pong buffers:
//!
//! * **M3 — naive wavefront** ([`metal_wavefront_field`]). Each iteration relaxes
//!   every cell against its 4 neighbours: `new[i] = min(old[i], min_n(old[n] +
//!   cost(i)))`. Obstacles stay `Cost::MAX`. Iterates until a change-flag buffer
//!   reports no change.
//!
//! * **M4 — separable H/V prefix-min sweep** ([`metal_sweep_field`]). One kernel
//!   owns a row and runs a serial L→R then R→L prefix-min; another owns a column
//!   and runs U→D then D→U. H/V passes alternate until convergence. This mirrors
//!   `mr_cpu::sweep_distance_field` exactly.
//! * **Batched M4** ([`metal_sweep_fields`]). Independent source fields are packed
//!   into one buffer and dispatched together, amortising shader compilation,
//!   command submission, and CPU↔GPU synchronization across a board's nets.
//!
//! ## Canonical tie-breaking
//!
//! A converged cost field alone cannot distinguish a short weighted path from a
//! longer equal-cost path, and zero-cost plateaus make strict cost descent
//! impossible. Weighted/zero-cost routing therefore carries a minimum-hop field
//! alongside distance. Unit-cost routing uses `hops == distance` and skips that
//! buffer entirely. [`MetalRouter`] reconstructs through predecessors that reduce
//! hops by one, choosing the lowest [`CellIdx`] when several remain. This is the
//! same `(cost, hops, predecessor)` contract as the CPU routers and is cycle-proof.
//!
//! On non-macOS targets the public surface still exists, but every entry point
//! returns [`RouterError::BackendUnavailable`] so the workspace compiles
//! everywhere.

use mr_core::{BoardRoute, CellIdx, Cost, Grid, NetEndpoints, RouteResult, Router, RouterError};

type FlatBatchedSweep = (Vec<Cost>, Option<Vec<u32>>);

/// Inclusive planar rectangle used by [`metal_route_isolated_batch`].
///
/// The same rectangle applies on every layer. Cells outside it are hard blocked,
/// including passable-pad cells; callers that mirror NegotiatedRouter should submit
/// its normal per-net window first and retry only failed nets with [`Self::full`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalWindow {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl MetalWindow {
    /// A window covering every planar cell in `dims`.
    pub fn full(dims: mr_core::Dims) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: dims.w.saturating_sub(1),
            y1: dims.h.saturating_sub(1),
        }
    }
}

#[cfg(target_os = "macos")]
impl MetalWindow {
    fn is_valid(self, dims: mr_core::Dims) -> bool {
        dims.w > 0
            && dims.h > 0
            && self.x0 <= self.x1
            && self.y0 <= self.y1
            && self.x1 < dims.w
            && self.y1 < dims.h
    }

    fn contains(self, dims: mr_core::Dims, cell: CellIdx) -> bool {
        let (x, y) = dims.xy(cell);
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// Already-rounded edge prices for an exact isolated-route Metal solve.
///
/// Lengths must be `w-1`, `h-1`, and `layers-1`, respectively. `vias[k]` is the
/// price of adjacent layer transition `k <-> k+1`; `None` forbids that transition.
/// Prices are supplied by the caller so this crate does not duplicate the CPU
/// router's floating-point Hanan rounding contract.
#[derive(Debug, Clone, Copy)]
pub struct MetalEdgeCosts<'a> {
    pub x: &'a [Cost],
    pub y: &'a [Cost],
    pub vias: &'a [Option<Cost>],
}

/// One exact independent path returned by [`metal_route_isolated_batch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalIsolatedRoute {
    pub path: Vec<CellIdx>,
    /// Fixed-point search cost, including destination enter weights and excluding
    /// the source, exactly as used to choose the path.
    pub search_cost: Cost,
}

#[cfg(target_os = "macos")]
mod gpu;

/// Single-source distance field via the naive atomic-free wavefront kernel (M3).
///
/// Returns `dist` indexed by [`CellIdx`]; unreachable cells (and an obstacle
/// source) are `Cost::MAX`. Equal to `mr_cpu::bfs_distance_field`.
#[cfg(target_os = "macos")]
pub fn metal_wavefront_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    gpu::wavefront_field(grid, src)
}

/// Non-macOS fallback: the Metal backend is unavailable.
#[cfg(not(target_os = "macos"))]
pub fn metal_wavefront_field(_grid: &Grid, _src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// Single-source distance field via the separable H/V prefix-min sweep kernels
/// (M4). Returns `dist` indexed by [`CellIdx`]; unreachable cells are
/// `Cost::MAX`. Equal to `mr_cpu::sweep_distance_field` and
/// `mr_cpu::bfs_distance_field`.
#[cfg(target_os = "macos")]
pub fn metal_sweep_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    gpu::sweep_field(grid, src)
}

/// Non-macOS fallback: the Metal backend is unavailable.
#[cfg(not(target_os = "macos"))]
pub fn metal_sweep_field(_grid: &Grid, _src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// Compute independent distance fields for every source in one batched Metal
/// solve. Results follow `sources` order and are identical to calling
/// [`metal_sweep_field`] separately, but share compilation and command dispatch.
/// GPU work is internally chunked; requests whose retained nested-vector result
/// would exceed the documented safety bound return an error before allocation.
#[cfg(target_os = "macos")]
pub fn metal_sweep_fields(grid: &Grid, sources: &[CellIdx]) -> Result<Vec<Vec<Cost>>, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::MalformedGrid);
    }
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let n = grid.dims.len();
    validate_public_result_shape(n, sources.len())?;
    if n == 0 {
        return Ok(vec![Vec::new(); sources.len()]);
    }

    let mut fields = Vec::with_capacity(sources.len());
    for source_batch in sources.chunks(fields_per_batch(n)) {
        let flat = gpu::sweep_fields_flat(grid, source_batch, &grid.cost, false)?;
        fields.extend(flat.dist.chunks_exact(n).map(|field| field.to_vec()));
    }
    Ok(fields)
}

/// Non-macOS fallback: the Metal backend is unavailable.
#[cfg(not(target_os = "macos"))]
pub fn metal_sweep_fields(
    _grid: &Grid,
    _sources: &[CellIdx],
) -> Result<Vec<Vec<Cost>>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

#[cfg(target_os = "macos")]
fn metal_sweep_fields_flat_with_costs(
    grid: &Grid,
    sources: &[CellIdx],
    costs: &[Cost],
    track_hops: bool,
) -> Result<FlatBatchedSweep, RouterError> {
    let flat = gpu::sweep_fields_flat(grid, sources, costs, track_hops)?;
    Ok((flat.dist, flat.hops))
}

#[cfg(not(target_os = "macos"))]
fn metal_sweep_fields_flat_with_costs(
    _grid: &Grid,
    _sources: &[CellIdx],
    _costs: &[Cost],
    _track_hops: bool,
) -> Result<FlatBatchedSweep, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// A [`Router`] backed by the Metal GPU sweep kernel.
///
/// It computes nets' distance fields in memory-bounded GPU batches, then
/// reconstructs each path from GPU-computed distance and minimum-hop labels so
/// the result follows the same canonical contract as `mr_cpu::LeeRouter`.
/// Per-net passable pad cells are represented by a packed cost plane, preserving
/// the shared router contract without giving up batching. Unreachable nets land in
/// [`BoardRoute::unrouted`].
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalRouter;

impl MetalRouter {
    pub fn new() -> Self {
        Self
    }
}

/// Bound temporary GPU working sets while leaving enough independent fields to
/// saturate Apple GPUs. Each field needs a distance plane and, when pad costs
/// differ, a cost plane. A batch is capped at 16M cells (64 MiB per such
/// buffer); M4 also carries a hop-count plane. A single larger field runs alone
/// and is separately checked against the device's reported maximum buffer
/// length.
const MAX_BATCH_CELLS: usize = 16 * 1024 * 1024;
const MAX_FIELDS_PER_BATCH: usize = 256;
/// The nested-vector API must retain every completed field at once. Bound that
/// returned allocation independently from the temporary GPU batch so an
/// accidental source count cannot grow until the process is killed by the OS.
/// A single larger field remains allowed when the caller already owns such a grid.
#[cfg(target_os = "macos")]
const MAX_PUBLIC_RESULT_CELLS: usize = 64 * 1024 * 1024;
#[cfg(target_os = "macos")]
const MAX_PUBLIC_FIELDS: usize = 1_000_000;

#[cfg(target_os = "macos")]
fn validate_public_result_shape(cells_per_field: usize, fields: usize) -> Result<(), RouterError> {
    let total = cells_per_field
        .checked_mul(fields)
        .ok_or_else(|| RouterError::BackendUnavailable("Metal result is too large".into()))?;
    if fields > MAX_PUBLIC_FIELDS || total > MAX_PUBLIC_RESULT_CELLS.max(cells_per_field) {
        return Err(RouterError::BackendUnavailable(format!(
            "Metal result would contain {fields} fields / {total} cells; use smaller source chunks"
        )));
    }
    Ok(())
}

fn fields_per_batch(cells_per_field: usize) -> usize {
    if cells_per_field == 0 {
        return 1;
    }
    (MAX_BATCH_CELLS / cells_per_field).clamp(1, MAX_FIELDS_PER_BATCH)
}

#[cfg(target_os = "macos")]
#[inline]
fn isolated_step_cost(edge: Cost, enter_weight: Cost) -> Cost {
    ((edge as u64).saturating_mul(enter_weight as u64)).min((Cost::MAX - 1) as u64) as Cost
}

/// Reconstruct an edge-aware path from exact distance/minimum-hop fields.
/// Candidate predecessors are visited in ascending CellIdx order: lower layer,
/// planar up/left/right/down, then upper layer.
#[cfg(target_os = "macos")]
fn path_from_edge_field(
    grid: &Grid,
    costs: &[Cost],
    dist: &[Cost],
    hops: &[u32],
    edges: MetalEdgeCosts<'_>,
    src: CellIdx,
    dst: CellIdx,
) -> Option<Vec<CellIdx>> {
    let dims = grid.dims;
    if costs.len() != dims.len()
        || dist.len() != dims.len()
        || hops.len() != dims.len()
        || !dims.contains(src)
        || !dims.contains(dst)
        || dist[dst as usize] == Cost::MAX
    {
        return None;
    }

    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        let need_dist = dist[cur as usize];
        let need_hops = hops[cur as usize];
        if need_hops == 0 || need_hops == u32::MAX {
            return None;
        }
        let enter_weight = costs[cur as usize];
        let (x, y, layer) = dims.xyz(cur);
        let mut next = None;
        let mut consider = |p: CellIdx, edge: Option<Cost>| {
            if next.is_some() {
                return;
            }
            let Some(edge) = edge else { return };
            let dp = dist[p as usize];
            let hp = hops[p as usize];
            let step = isolated_step_cost(edge, enter_weight);
            if dp != Cost::MAX
                && dp.saturating_add(step) == need_dist
                && hp.saturating_add(1) == need_hops
            {
                next = Some(p);
            }
        };

        // Every lower-layer cell index precedes every cell on this layer.
        if layer > 0 {
            consider(
                dims.idx3(x, y, layer - 1),
                edges.vias.get((layer - 1) as usize).copied().flatten(),
            );
        }
        if y > 0 {
            consider(
                dims.idx3(x, y - 1, layer),
                edges.y.get((y - 1) as usize).copied(),
            );
        }
        if x > 0 {
            consider(
                dims.idx3(x - 1, y, layer),
                edges.x.get((x - 1) as usize).copied(),
            );
        }
        if x + 1 < dims.w {
            consider(dims.idx3(x + 1, y, layer), edges.x.get(x as usize).copied());
        }
        if y + 1 < dims.h {
            consider(dims.idx3(x, y + 1, layer), edges.y.get(y as usize).copied());
        }
        // Every upper-layer cell index follows every cell on this layer.
        if layer + 1 < dims.layers {
            consider(
                dims.idx3(x, y, layer + 1),
                edges.vias.get(layer as usize).copied().flatten(),
            );
        }

        cur = next?;
        path.push(cur);
        if path.len() > dims.len() {
            return None;
        }
    }
    path.reverse();
    Some(path)
}

/// Route independent nets with the exact static cost/constraint subset used by
/// NegotiatedRouter's "alone path" phase.
///
/// This is an all-or-error batch operation. It applies each net's own obstacle-pad
/// overrides and inclusive window, multiplies caller-supplied Hanan/via edge prices
/// by the destination cell's enter weight with a `Cost::MAX - 1` per-step cap, and
/// returns canonical `(cost, minimum hops, lower predecessor)` paths in input order.
/// `None` means the net is unreachable inside its submitted window.
///
/// The function deliberately does **not** perform the CPU router's full-board retry:
/// submit a second batch containing only the `None` entries with
/// [`MetalWindow::full`]. On [`RouterError::BackendUnavailable`] the caller must run
/// the existing CPU isolated searches; no partial Metal result is returned or safe
/// to use. Dynamic congestion, committed-owner/halo constraints, and via ring guards
/// are outside this API and must remain on the CPU.
#[cfg(target_os = "macos")]
pub fn metal_route_isolated_batch(
    grid: &Grid,
    nets: &[NetEndpoints],
    windows: &[MetalWindow],
    edges: MetalEdgeCosts<'_>,
) -> Result<Vec<Option<MetalIsolatedRoute>>, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::MalformedGrid);
    }
    // The packed Metal request does not yet carry per-net via-pad exemptions, so
    // accepting a static via mask here could return a path the CPU boundary must
    // reject. Preserve the all-or-fallback contract until those permissions are
    // represented in the kernel input.
    if !grid.via_forbidden.is_empty() {
        return Err(RouterError::BackendUnavailable(
            "isolated Metal routing does not support static via masks".into(),
        ));
    }
    let dims = grid.dims;
    let expected_x = (dims.w as usize).saturating_sub(1);
    let expected_y = (dims.h as usize).saturating_sub(1);
    let expected_vias = (dims.layers as usize).saturating_sub(1);
    if windows.len() != nets.len()
        || edges.x.len() != expected_x
        || edges.y.len() != expected_y
        || edges.vias.len() != expected_vias
    {
        return Err(RouterError::BackendUnavailable(
            "isolated batch shape does not match grid/nets".into(),
        ));
    }
    for (net, &window) in nets.iter().zip(windows) {
        if !window.is_valid(dims) {
            return Err(RouterError::BackendUnavailable(format!(
                "net `{}` has an invalid isolated-route window",
                net.net
            )));
        }
        if net.passable_pads.iter().any(|&c| !dims.contains(c))
            || net
                .via_passable_pads
                .iter()
                .any(|c| !dims.contains(*c) || !net.passable_pads.contains(c))
            || !dims.contains(net.src)
            || !dims.contains(net.dst)
            || (grid.is_obstacle(net.src) && !net.passable_pads.contains(&net.src))
            || (grid.is_obstacle(net.dst) && !net.passable_pads.contains(&net.dst))
        {
            return Err(RouterError::InvalidEndpoint {
                net: net.net.clone(),
            });
        }
    }
    if nets.is_empty() {
        return Ok(Vec::new());
    }

    let n = dims.len();
    let batch_fields = fields_per_batch(n);
    // Every later chunk is no larger than the first. Preflight that maximum
    // packed shape, including device limits for all GPU buffers, before making
    // any batch-sized host allocation.
    gpu::preflight_packed_edge_batch(
        grid,
        nets.len().min(batch_fields),
        edges.x.len(),
        edges.y.len(),
        edges.vias.len(),
    )?;
    let via_edges: Vec<Cost> = edges
        .vias
        .iter()
        .map(|edge| edge.unwrap_or_default())
        .collect();
    let via_allowed: Vec<u32> = edges
        .vias
        .iter()
        .map(|edge| u32::from(edge.is_some()))
        .collect();
    let mut output = Vec::with_capacity(nets.len());

    for batch_start in (0..nets.len()).step_by(batch_fields) {
        let batch_end = (batch_start + batch_fields).min(nets.len());
        let net_batch = &nets[batch_start..batch_end];
        let window_batch = &windows[batch_start..batch_end];
        // The maximum chunk was preflighted above; keep the exact checked size
        // here instead of recomputing capacity with saturating arithmetic.
        let packed_len = gpu::checked_batch_cells(n, net_batch.len())?;
        let mut packed_costs = Vec::with_capacity(packed_len);
        for (net, &window) in net_batch.iter().zip(window_batch) {
            let offset = packed_costs.len();
            packed_costs.extend((0..n).map(|i| {
                let cell = i as CellIdx;
                if window.contains(dims, cell) {
                    grid.cost[i]
                } else {
                    Cost::MAX
                }
            }));
            for &pad in &net.passable_pads {
                if window.contains(dims, pad) && packed_costs[offset + pad as usize] == Cost::MAX {
                    packed_costs[offset + pad as usize] = 1;
                }
            }
        }

        let sources: Vec<CellIdx> = net_batch.iter().map(|net| net.src).collect();
        let flat = gpu::sweep_fields_flat_edges(
            grid,
            &sources,
            &packed_costs,
            edges.x,
            edges.y,
            &via_edges,
            &via_allowed,
        )?;
        let hops = flat
            .hops
            .as_ref()
            .expect("edge-aware sweep always returns hop labels");
        for (field, net) in net_batch.iter().enumerate() {
            let field_start = field * n;
            let field_end = field_start + n;
            let dist = &flat.dist[field_start..field_end];
            let hop_field = &hops[field_start..field_end];
            let costs = &packed_costs[field_start..field_end];
            if dist[net.dst as usize] == Cost::MAX {
                output.push(None);
                continue;
            }
            let path = path_from_edge_field(grid, costs, dist, hop_field, edges, net.src, net.dst)
                .ok_or_else(|| {
                    RouterError::BackendUnavailable(
                        "edge-aware Metal path reconstruction failed".into(),
                    )
                })?;
            output.push(Some(MetalIsolatedRoute {
                path,
                search_cost: dist[net.dst as usize],
            }));
        }
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct RaggedRouteStats {
    packed_cells: usize,
    full_field_cells: usize,
    readback_cells: usize,
    chunks: usize,
}

/// Experimental cropped/ragged implementation of
/// [`metal_route_isolated_batch`].
///
/// This has the same exact weighted, via, window, passable-pad, and canonical
/// path contract as the established full-field entry point. Its implementation
/// packs each field at the submitted window's real size and reconstructs paths on
/// the GPU, so distance/hop planes never become host vectors. The surface remains
/// hidden until a negotiated-router integration can select it conservatively.
/// A nonempty [`Grid::via_forbidden`] mask currently fails closed because the
/// compact request does not yet encode each net's `via_passable_pads` exemptions.
/// Any malformed shape, resource failure, shader error, or invalid compact result
/// rejects the whole call with [`RouterError::BackendUnavailable`]; callers must
/// then rerun the complete batch on the CPU.
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn metal_route_isolated_batch_ragged(
    grid: &Grid,
    nets: &[NetEndpoints],
    windows: &[MetalWindow],
    edges: MetalEdgeCosts<'_>,
) -> Result<Vec<Option<MetalIsolatedRoute>>, RouterError> {
    ragged_isolated_batch_impl(grid, nets, windows, edges).map(|(routes, _)| routes)
}

#[cfg(target_os = "macos")]
fn ragged_isolated_batch_impl(
    grid: &Grid,
    nets: &[NetEndpoints],
    windows: &[MetalWindow],
    edges: MetalEdgeCosts<'_>,
) -> Result<(Vec<Option<MetalIsolatedRoute>>, RaggedRouteStats), RouterError> {
    let unavailable = |message: String| RouterError::BackendUnavailable(message);
    if !grid.is_well_formed() {
        return Err(unavailable(
            "ragged isolated batch has a malformed grid".into(),
        ));
    }
    // This compact request does not yet carry per-net via-pad exemptions. Even
    // an all-false, nonempty mask means the caller selected the static-mask
    // contract, so fail the whole batch closed until both arrays are packed.
    if !grid.via_forbidden.is_empty() {
        return Err(unavailable(
            "ragged isolated Metal routing does not support static via masks".into(),
        ));
    }
    let dims = grid.dims;
    let expected_x = (dims.w as usize).saturating_sub(1);
    let expected_y = (dims.h as usize).saturating_sub(1);
    let expected_vias = (dims.layers as usize).saturating_sub(1);
    if windows.len() != nets.len()
        || edges.x.len() != expected_x
        || edges.y.len() != expected_y
        || edges.vias.len() != expected_vias
    {
        return Err(unavailable(
            "ragged isolated batch shape does not match grid/nets".into(),
        ));
    }
    for (net, &window) in nets.iter().zip(windows) {
        if !window.is_valid(dims) {
            return Err(unavailable(format!(
                "net `{}` has an invalid ragged-route window",
                net.net
            )));
        }
        if net.passable_pads.iter().any(|&cell| !dims.contains(cell))
            || net
                .via_passable_pads
                .iter()
                .any(|cell| !dims.contains(*cell) || !net.passable_pads.contains(cell))
            || !dims.contains(net.src)
            || !dims.contains(net.dst)
            || (grid.is_obstacle(net.src) && !net.passable_pads.contains(&net.src))
            || (grid.is_obstacle(net.dst) && !net.passable_pads.contains(&net.dst))
        {
            return Err(unavailable(format!(
                "net `{}` has an invalid ragged-route endpoint",
                net.net
            )));
        }
    }
    if nets.is_empty() {
        return Ok((Vec::new(), RaggedRouteStats::default()));
    }

    let via_edges: Vec<Cost> = edges
        .vias
        .iter()
        .map(|edge| edge.unwrap_or_default())
        .collect();
    let via_allowed: Vec<u32> = edges
        .vias
        .iter()
        .map(|edge| u32::from(edge.is_some()))
        .collect();
    let mut output = Vec::with_capacity(nets.len());
    let mut stats = RaggedRouteStats {
        full_field_cells: dims.len().checked_mul(nets.len()).ok_or_else(|| {
            unavailable("ragged isolated full-field comparison is too large".into())
        })?,
        ..RaggedRouteStats::default()
    };

    let window_cells = |window: MetalWindow| -> Result<usize, RouterError> {
        let w = (window.x1 - window.x0 + 1) as usize;
        let h = (window.y1 - window.y0 + 1) as usize;
        w.checked_mul(h)
            .and_then(|plane| plane.checked_mul(dims.layers as usize))
            .ok_or_else(|| unavailable("ragged isolated window is too large".into()))
    };

    let mut batch_start = 0usize;
    while batch_start < nets.len() {
        let mut batch_end = batch_start;
        let mut batch_cells = 0usize;
        while batch_end < nets.len() && batch_end - batch_start < MAX_FIELDS_PER_BATCH {
            let next = window_cells(windows[batch_end])?;
            let combined = batch_cells
                .checked_add(next)
                .ok_or_else(|| unavailable("ragged isolated batch is too large".into()))?;
            if batch_end > batch_start && combined > MAX_BATCH_CELLS {
                break;
            }
            batch_cells = combined;
            batch_end += 1;
        }

        let net_batch = &nets[batch_start..batch_end];
        let window_batch = &windows[batch_start..batch_end];
        let mut fields = Vec::with_capacity(net_batch.len());
        let mut packed_costs = Vec::with_capacity(batch_cells);
        for (net, &window) in net_batch.iter().zip(window_batch) {
            let w = window.x1 - window.x0 + 1;
            let h = window.y1 - window.y0 + 1;
            let plane = w.checked_mul(h).ok_or_else(|| {
                unavailable(format!("net `{}` ragged window is too large", net.net))
            })?;
            let cell_offset = u32::try_from(packed_costs.len())
                .map_err(|_| unavailable("ragged isolated batch exceeds 32-bit indexing".into()))?;
            for layer in 0..dims.layers {
                for y in window.y0..=window.y1 {
                    for x in window.x0..=window.x1 {
                        packed_costs.push(grid.cost[dims.idx3(x, y, layer) as usize]);
                    }
                }
            }
            let packed_cell = |global: CellIdx| -> u32 {
                if !window.contains(dims, global) {
                    return u32::MAX;
                }
                let (x, y, layer) = dims.xyz(global);
                cell_offset + layer * plane + (y - window.y0) * w + x - window.x0
            };
            for &pad in &net.passable_pads {
                let local = packed_cell(pad);
                if local != u32::MAX && packed_costs[local as usize] == Cost::MAX {
                    packed_costs[local as usize] = 1;
                }
            }
            fields.push(gpu::RaggedField {
                cell_offset,
                w,
                h,
                layers: dims.layers,
                x0: window.x0,
                y0: window.y0,
                src: packed_cell(net.src),
                dst: packed_cell(net.dst),
            });
        }
        if packed_costs.len() != batch_cells {
            return Err(unavailable(
                "ragged isolated packing disagrees with its preflight".into(),
            ));
        }

        let ragged = gpu::sweep_ragged_paths(
            grid,
            &fields,
            &packed_costs,
            edges.x,
            edges.y,
            &via_edges,
            &via_allowed,
        )?;
        if ragged.paths.len() != net_batch.len() || ragged.search_costs.len() != net_batch.len() {
            return Err(unavailable(
                "ragged isolated GPU result shape is invalid".into(),
            ));
        }
        stats.packed_cells = stats
            .packed_cells
            .checked_add(ragged.packed_cells)
            .ok_or_else(|| unavailable("ragged packed-cell statistic overflowed".into()))?;
        stats.readback_cells = stats
            .readback_cells
            .checked_add(ragged.readback_cells)
            .ok_or_else(|| unavailable("ragged readback statistic overflowed".into()))?;
        stats.chunks += 1;

        for (((path, &search_cost), net), &window) in ragged
            .paths
            .into_iter()
            .zip(&ragged.search_costs)
            .zip(net_batch)
            .zip(window_batch)
        {
            let Some(path) = path else {
                if search_cost != Cost::MAX {
                    return Err(unavailable(
                        "unrouted ragged Metal field returned a finite cost".into(),
                    ));
                }
                output.push(None);
                continue;
            };
            if path.first() != Some(&net.src)
                || path.last() != Some(&net.dst)
                || path.len() > window_cells(window)?
                || path.iter().any(|&cell| !window.contains(dims, cell))
            {
                return Err(unavailable(
                    "ragged Metal compact path failed endpoint/window validation".into(),
                ));
            }
            let mut unique = std::collections::HashSet::with_capacity(path.len());
            if path.iter().any(|&cell| !unique.insert(cell)) {
                return Err(unavailable(
                    "ragged Metal compact path contains a cycle".into(),
                ));
            }
            let mut checked_cost: Cost = 0;
            for pair in path.windows(2) {
                let (u, v) = (pair[0], pair[1]);
                let (ux, uy, ul) = dims.xyz(u);
                let (vx, vy, vl) = dims.xyz(v);
                let edge = if ul != vl {
                    if ux != vx || uy != vy || ul.abs_diff(vl) != 1 {
                        return Err(unavailable(
                            "ragged Metal compact path has a non-adjacent via".into(),
                        ));
                    }
                    edges.vias[ul.min(vl) as usize].ok_or_else(|| {
                        unavailable("ragged Metal compact path used a forbidden via".into())
                    })?
                } else if ux.abs_diff(vx) == 1 && uy == vy {
                    edges.x[ux.min(vx) as usize]
                } else if uy.abs_diff(vy) == 1 && ux == vx {
                    edges.y[uy.min(vy) as usize]
                } else {
                    return Err(unavailable(
                        "ragged Metal compact path has non-adjacent planar cells".into(),
                    ));
                };
                let enter = if grid.is_obstacle(v) && net.passable_pads.contains(&v) {
                    1
                } else {
                    grid.cost_at(v)
                };
                if enter == Cost::MAX {
                    return Err(unavailable(
                        "ragged Metal compact path entered an obstacle".into(),
                    ));
                }
                checked_cost = checked_cost.saturating_add(isolated_step_cost(edge, enter));
            }
            if checked_cost != search_cost {
                return Err(unavailable(
                    "ragged Metal compact path cost failed validation".into(),
                ));
            }
            output.push(Some(MetalIsolatedRoute { path, search_cost }));
        }
        batch_start = batch_end;
    }

    if output.len() != nets.len() {
        return Err(unavailable(
            "ragged isolated batch did not produce every result".into(),
        ));
    }
    Ok((output, stats))
}

/// Non-macOS fallback. Callers must retry the complete batch on the CPU.
#[cfg(not(target_os = "macos"))]
#[doc(hidden)]
pub fn metal_route_isolated_batch_ragged(
    _grid: &Grid,
    _nets: &[NetEndpoints],
    _windows: &[MetalWindow],
    _edges: MetalEdgeCosts<'_>,
) -> Result<Vec<Option<MetalIsolatedRoute>>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// Non-macOS fallback. Callers must retry the same batch on the CPU.
#[cfg(not(target_os = "macos"))]
pub fn metal_route_isolated_batch(
    _grid: &Grid,
    _nets: &[NetEndpoints],
    _windows: &[MetalWindow],
    _edges: MetalEdgeCosts<'_>,
) -> Result<Vec<Option<MetalIsolatedRoute>>, RouterError> {
    Err(RouterError::BackendUnavailable(
        "Metal compute is only available on macOS".into(),
    ))
}

/// Reconstruct `src..=dst` from converged distance and minimum-hop fields.
/// Every predecessor must preserve shortest cost and reduce hop count by one;
/// ties choose the lowest [`CellIdx`], matching the CPU router contract.
fn path_from_field_with_hops(
    grid: &Grid,
    costs: &[Cost],
    dist: &[Cost],
    hops: &[u32],
    src: CellIdx,
    dst: CellIdx,
) -> Option<Vec<CellIdx>> {
    if costs.len() != grid.dims.len()
        || dist.len() != grid.dims.len()
        || hops.len() != grid.dims.len()
        || !grid.dims.contains(src)
        || !grid.dims.contains(dst)
    {
        return None;
    }
    if dist[dst as usize] == Cost::MAX {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        let need_dist = dist[cur as usize];
        let need_hops = hops[cur as usize];
        if need_hops == 0 || need_hops == u32::MAX {
            return None;
        }
        let step = costs[cur as usize];
        let mut next = None;
        let mut consider = |p: CellIdx| {
            if next.is_some() {
                return;
            }
            let dp = dist[p as usize];
            let hp = hops[p as usize];
            if dp != Cost::MAX
                && dp.saturating_add(step) == need_dist
                && hp.saturating_add(1) == need_hops
            {
                next = Some(p);
            }
        };
        // Same ascending CellIdx order as Dims::neighbors4, without allocating a
        // Vec at every reconstructed step (hot for large batches).
        let (x, y, layer) = grid.dims.xyz(cur);
        if y > 0 {
            consider(grid.dims.idx3(x, y - 1, layer));
        }
        if x > 0 {
            consider(grid.dims.idx3(x - 1, y, layer));
        }
        if x + 1 < grid.dims.w {
            consider(grid.dims.idx3(x + 1, y, layer));
        }
        if y + 1 < grid.dims.h {
            consider(grid.dims.idx3(x, y + 1, layer));
        }
        let p = next?;
        path.push(p);
        if path.len() > grid.dims.len() {
            return None;
        }
        cur = p;
    }
    path.reverse();
    Some(path)
}

/// Unit-cost specialization: every finite distance is also the minimum hop
/// count, so canonical reconstruction needs no separately carried hop plane.
fn path_from_unit_field(
    grid: &Grid,
    dist: &[Cost],
    src: CellIdx,
    dst: CellIdx,
) -> Option<Vec<CellIdx>> {
    if dist.len() != grid.dims.len()
        || !grid.dims.contains(src)
        || !grid.dims.contains(dst)
        || dist[dst as usize] == Cost::MAX
    {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        let need = dist[cur as usize];
        if need == 0 {
            return None;
        }
        let mut next = None;
        let mut consider = |p: CellIdx| {
            if next.is_none()
                && dist[p as usize] != Cost::MAX
                && dist[p as usize].saturating_add(1) == need
            {
                next = Some(p);
            }
        };
        let (x, y, layer) = grid.dims.xyz(cur);
        if y > 0 {
            consider(grid.dims.idx3(x, y - 1, layer));
        }
        if x > 0 {
            consider(grid.dims.idx3(x - 1, y, layer));
        }
        if x + 1 < grid.dims.w {
            consider(grid.dims.idx3(x + 1, y, layer));
        }
        if y + 1 < grid.dims.h {
            consider(grid.dims.idx3(x, y + 1, layer));
        }
        cur = next?;
        path.push(cur);
    }
    path.reverse();
    Some(path)
}

fn has_unit_costs(grid: &Grid) -> bool {
    grid.cost
        .iter()
        .all(|&cost| cost == 1 || cost == mr_core::OBSTACLE)
}

impl Router for MetalRouter {
    fn route(&self, grid: &Grid, nets: &[NetEndpoints]) -> Result<BoardRoute, RouterError> {
        if !grid.is_well_formed() {
            return Err(RouterError::MalformedGrid);
        }
        if nets.is_empty() {
            return Ok(BoardRoute {
                results: Vec::new(),
                unrouted: Vec::new(),
                congestion: vec![0; grid.dims.len()],
                groups: Vec::new(),
            });
        }
        let n = grid.dims.len();
        // Validate the whole request before submitting any work, matching the
        // all-or-error behaviour of the CPU routers.
        for net in nets {
            if net.passable_pads.iter().any(|&c| !grid.dims.contains(c)) {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
            if !grid.dims.contains(net.src)
                || !grid.dims.contains(net.dst)
                || (grid.is_obstacle(net.src) && !net.passable_pads.contains(&net.src))
                || (grid.is_obstacle(net.dst) && !net.passable_pads.contains(&net.dst))
            {
                return Err(RouterError::InvalidEndpoint {
                    net: net.net.clone(),
                });
            }
        }

        let mut results = Vec::new();
        let mut unrouted = Vec::new();
        let unit_costs = has_unit_costs(grid);
        for net_batch in nets.chunks(fields_per_batch(n)) {
            let sources: Vec<_> = net_batch.iter().map(|net| net.src).collect();
            let needs_cost_overrides = net_batch
                .iter()
                .any(|net| net.passable_pads.iter().any(|&pad| grid.is_obstacle(pad)));
            let mut packed_costs = Vec::new();
            let (flat_dist, flat_hops) = if needs_cost_overrides {
                packed_costs.reserve(n.saturating_mul(net_batch.len()));
                for net in net_batch {
                    let offset = packed_costs.len();
                    packed_costs.extend_from_slice(&grid.cost);
                    for &pad in &net.passable_pads {
                        if packed_costs[offset + pad as usize] == mr_core::OBSTACLE {
                            packed_costs[offset + pad as usize] = 1;
                        }
                    }
                }
                metal_sweep_fields_flat_with_costs(grid, &sources, &packed_costs, !unit_costs)?
            } else {
                metal_sweep_fields_flat_with_costs(grid, &sources, &grid.cost, !unit_costs)?
            };
            for (field, net) in net_batch.iter().enumerate() {
                let dist = &flat_dist[field * n..(field + 1) * n];
                let costs = if needs_cost_overrides {
                    &packed_costs[field * n..(field + 1) * n]
                } else {
                    &grid.cost
                };
                let path = if unit_costs {
                    path_from_unit_field(grid, dist, net.src, net.dst)
                } else {
                    let hops = flat_hops
                        .as_ref()
                        .expect("weighted routing requested hop tracking");
                    let hops = &hops[field * n..(field + 1) * n];
                    path_from_field_with_hops(grid, costs, dist, hops, net.src, net.dst)
                };
                match path {
                    Some(path) => results.push(RouteResult {
                        net: net.net.clone(),
                        path,
                        cost: dist[net.dst as usize],
                    }),
                    None => unrouted.push(net.net.clone()),
                }
            }
        }
        let congestion = BoardRoute::congestion_from(grid.dims, &results);
        Ok(BoardRoute {
            results,
            unrouted,
            congestion,
            groups: Vec::new(),
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use mr_cpu::{
        bfs_distance_field, sweep_distance_field, LeeRouter, NegotiatedRouter, RipUpRouter, SCALE,
    };
    use mr_fixtures::{hand_32x32_wall, obstacle_battery, tie_break_2x2};
    use mr_grid::GridBuilder;
    use std::time::Instant;

    fn fixed_edge(length: f64) -> Cost {
        let scaled = (length * SCALE as f64).round();
        if !scaled.is_finite() || scaled <= 0.0 {
            0
        } else if scaled >= Cost::MAX as f64 {
            Cost::MAX
        } else {
            scaled as Cost
        }
    }

    fn edge_vectors(
        dims: mr_core::Dims,
        coords: &mr_core::GridCoords,
        vias: &mr_core::ViaModel,
    ) -> (Vec<Cost>, Vec<Cost>, Vec<Option<Cost>>) {
        let x = (0..dims.w.saturating_sub(1))
            .map(|i| fixed_edge((coords.x_of(i + 1) - coords.x_of(i)).abs()))
            .collect();
        let y = (0..dims.h.saturating_sub(1))
            .map(|i| fixed_edge((coords.y_of(i + 1) - coords.y_of(i)).abs()))
            .collect();
        let via = (0..dims.layers.saturating_sub(1))
            .map(|layer| {
                vias.is_step_legal(layer, layer + 1)
                    .then_some(vias.step_cost)
            })
            .collect();
        (x, y, via)
    }

    fn test_window(dims: mr_core::Dims, net: &NetEndpoints) -> MetalWindow {
        let (sx, sy) = dims.xy(net.src);
        let (dx, dy) = dims.xy(net.dst);
        let mut x0 = sx.min(dx);
        let mut y0 = sy.min(dy);
        let mut x1 = sx.max(dx);
        let mut y1 = sy.max(dy);
        for &pad in &net.passable_pads {
            let (px, py) = dims.xy(pad);
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
        }
        let span = (x1 - x0).max(y1 - y0);
        let margin = 16u32.max((3 * span).div_ceil(10));
        MetalWindow {
            x0: x0.saturating_sub(margin),
            y0: y0.saturating_sub(margin),
            x1: (x1 + margin).min(dims.w.saturating_sub(1)),
            y1: (y1 + margin).min(dims.h.saturating_sub(1)),
        }
    }

    fn path_search_cost(
        grid: &Grid,
        net: &NetEndpoints,
        path: &[CellIdx],
        x_edges: &[Cost],
        y_edges: &[Cost],
        via_edges: &[Option<Cost>],
    ) -> Cost {
        path.windows(2).fold(0, |total, pair| {
            let (u, v) = (pair[0], pair[1]);
            let (ux, uy, ul) = grid.dims.xyz(u);
            let (vx, vy, vl) = grid.dims.xyz(v);
            let edge = if ul != vl {
                via_edges[ul.min(vl) as usize].expect("path used a forbidden via")
            } else if ux != vx {
                x_edges[ux.min(vx) as usize]
            } else {
                debug_assert_ne!(uy, vy);
                y_edges[uy.min(vy) as usize]
            };
            let enter = if grid.is_obstacle(v) && net.passable_pads.contains(&v) {
                1
            } else {
                grid.cost_at(v)
            };
            total.saturating_add(isolated_step_cost(edge, enter))
        })
    }

    fn assert_isolated_matches_cpu(
        grid: &Grid,
        coords: &mr_core::GridCoords,
        via_model: &mr_core::ViaModel,
        nets: &[NetEndpoints],
    ) {
        let windows: Vec<_> = nets.iter().map(|net| test_window(grid.dims, net)).collect();
        let (x, y, vias) = edge_vectors(grid.dims, coords, via_model);
        let gpu = metal_route_isolated_batch(
            grid,
            nets,
            &windows,
            MetalEdgeCosts {
                x: &x,
                y: &y,
                vias: &vias,
            },
        )
        .unwrap();
        let ragged = metal_route_isolated_batch_ragged(
            grid,
            nets,
            &windows,
            MetalEdgeCosts {
                x: &x,
                y: &y,
                vias: &vias,
            },
        )
        .unwrap();
        assert_eq!(ragged, gpu, "ragged and full-field Metal paths");
        let (_, trace) = NegotiatedRouter::new()
            .with_coords(coords.clone())
            .with_via_model(via_model.clone())
            .route_traced(grid, nets)
            .unwrap();
        assert_eq!(gpu.len(), trace.nets.len());
        for (i, (got, cpu)) in gpu.iter().zip(&trace.nets).enumerate() {
            if cpu.alone_path.is_empty() {
                assert!(got.is_none(), "net {i} should be unreachable");
            } else {
                let got = got
                    .as_ref()
                    .unwrap_or_else(|| panic!("net {i} should route"));
                assert_eq!(got.path, cpu.alone_path, "net {i} canonical path");
                assert_eq!(
                    got.search_cost,
                    path_search_cost(grid, &nets[i], &got.path, &x, &y, &vias),
                    "net {i} search cost"
                );
            }
        }
    }

    // ---- M3: naive wavefront field == CPU BFS field --------------------------

    #[test]
    fn wavefront_field_equals_bfs_on_battery() {
        for f in obstacle_battery() {
            let src = f.nets[0].src;
            let gpu = metal_wavefront_field(&f.grid, src).unwrap();
            let cpu = bfs_distance_field(&f.grid, src);
            assert_eq!(gpu.len(), cpu.len(), "{}", f.name);
            for i in 0..gpu.len() {
                assert_eq!(
                    gpu[i], cpu[i],
                    "{}: cell {i} gpu={} bfs={} (Cost::MAX==unreachable)",
                    f.name, gpu[i], cpu[i]
                );
            }
        }
    }

    #[test]
    fn wavefront_field_equals_bfs_on_hand_wall() {
        let f = hand_32x32_wall();
        let src = f.nets[0].src;
        let gpu = metal_wavefront_field(&f.grid, src).unwrap();
        let cpu = bfs_distance_field(&f.grid, src);
        assert_eq!(gpu, cpu);
        // Sanity: the pinned corner cost is 93.
        assert_eq!(gpu[f.nets[0].dst as usize], 93);
    }

    // ---- M4: separable sweep field == CPU sweep == CPU BFS -------------------

    #[test]
    fn sweep_field_equals_cpu_on_battery() {
        for f in obstacle_battery() {
            let src = f.nets[0].src;
            let gpu = metal_sweep_field(&f.grid, src).unwrap();
            let cpu_sweep = sweep_distance_field(&f.grid, src);
            let cpu_bfs = bfs_distance_field(&f.grid, src);
            assert_eq!(gpu.len(), cpu_bfs.len(), "{}", f.name);
            for i in 0..gpu.len() {
                assert_eq!(
                    gpu[i], cpu_sweep[i],
                    "{}: cell {i} gpu_sweep={} cpu_sweep={}",
                    f.name, gpu[i], cpu_sweep[i]
                );
                assert_eq!(
                    gpu[i], cpu_bfs[i],
                    "{}: cell {i} gpu_sweep={} cpu_bfs={}",
                    f.name, gpu[i], cpu_bfs[i]
                );
            }
        }
    }

    #[test]
    fn sweep_field_equals_cpu_on_hand_wall() {
        let f = hand_32x32_wall();
        let src = f.nets[0].src;
        let gpu = metal_sweep_field(&f.grid, src).unwrap();
        assert_eq!(gpu, bfs_distance_field(&f.grid, src));
        assert_eq!(gpu[f.nets[0].dst as usize], 93);
    }

    #[test]
    fn field_entry_points_handle_empty_invalid_and_obstacle_sources() {
        let empty = Grid::filled(mr_core::Dims::new(0, 0), 1);
        assert!(metal_wavefront_field(&empty, 0).unwrap().is_empty());
        assert!(metal_sweep_field(&empty, 0).unwrap().is_empty());

        let dims = mr_core::Dims::new(4, 3);
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx(1, 1), mr_core::OBSTACLE);
        let unreachable = vec![Cost::MAX; dims.len()];
        assert_eq!(metal_wavefront_field(&grid, 999).unwrap(), unreachable);
        assert_eq!(metal_sweep_field(&grid, 999).unwrap(), unreachable);
        assert_eq!(
            metal_wavefront_field(&grid, dims.idx(1, 1)).unwrap(),
            unreachable
        );
        assert_eq!(
            metal_sweep_field(&grid, dims.idx(1, 1)).unwrap(),
            unreachable
        );
    }

    #[test]
    fn batch_sizing_is_memory_bounded_and_router_preserves_cross_batch_order() {
        assert_eq!(fields_per_batch(0), 1);
        assert_eq!(fields_per_batch(1), MAX_FIELDS_PER_BATCH);
        assert_eq!(fields_per_batch(MAX_BATCH_CELLS), 1);
        assert_eq!(fields_per_batch(MAX_BATCH_CELLS + 1), 1);

        let grid = Grid::filled(mr_core::Dims::new(1, 1), 1);
        let nets: Vec<_> = (0..MAX_FIELDS_PER_BATCH + 1)
            .map(|i| NetEndpoints {
                net: format!("batch-{i:03}"),
                src: 0,
                dst: 0,
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            })
            .collect();
        let routed = MetalRouter.route(&grid, &nets).unwrap();
        assert!(routed.unrouted.is_empty());
        assert_eq!(routed.results.len(), nets.len());
        for (i, result) in routed.results.iter().enumerate() {
            assert_eq!(result.net, format!("batch-{i:03}"));
            assert_eq!(result.path, vec![0]);
            assert_eq!(result.cost, 0);
        }

        // The public field API uses the same bounded chunking rather than
        // constructing one unbounded `cells * sources` allocation.
        let sources = vec![0; MAX_FIELDS_PER_BATCH + 1];
        let fields = metal_sweep_fields(&grid, &sources).unwrap();
        assert_eq!(fields, vec![vec![0]; sources.len()]);

        // Shape rejection is arithmetic-only and occurs before total-sized host
        // vectors or Metal buffers are allocated.
        assert!(matches!(
            gpu::checked_batch_cells(u32::MAX as usize, 2),
            Err(RouterError::BackendUnavailable(_))
        ));
        assert!(matches!(
            validate_public_result_shape(128 * 128, 200_000),
            Err(RouterError::BackendUnavailable(_))
        ));
        assert!(validate_public_result_shape(128 * 128, 64).is_ok());
        assert!(validate_public_result_shape(0, MAX_PUBLIC_FIELDS).is_ok());
        assert!(matches!(
            validate_public_result_shape(0, MAX_PUBLIC_FIELDS + 1),
            Err(RouterError::BackendUnavailable(_))
        ));

        // Empty grids still return one empty Vec per source, so the public call
        // must enforce the field-count cap before taking that allocation path.
        let empty = Grid::filled(mr_core::Dims::new(0, 1), 1);
        let too_many_sources = vec![0; MAX_PUBLIC_FIELDS + 1];
        assert!(matches!(
            metal_sweep_fields(&empty, &too_many_sources),
            Err(RouterError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn distance_only_and_hop_carry_dispatches_are_explicit() {
        let dims = mr_core::Dims::new(4, 3);
        let mut unit = Grid::filled(dims, 1);
        unit.set(dims.idx(2, 1), mr_core::OBSTACLE);
        assert!(has_unit_costs(&unit));

        let unit_flat = gpu::sweep_fields_flat(&unit, &[0, 11], &unit.cost, false).unwrap();
        assert!(unit_flat.hops.is_none());
        for (field, &src) in unit_flat.dist.chunks_exact(dims.len()).zip(&[0, 11]) {
            assert_eq!(field, bfs_distance_field(&unit, src));
        }

        let mut weighted = unit.clone();
        weighted.set(dims.idx(1, 0), 7);
        assert!(!has_unit_costs(&weighted));
        let weighted_flat =
            gpu::sweep_fields_flat(&weighted, &[0, 11], &weighted.cost, true).unwrap();
        let hop_flat = weighted_flat.hops.as_ref().expect("hop plane requested");
        assert_eq!(hop_flat.len(), weighted_flat.dist.len());
        for (field, &src) in weighted_flat.dist.chunks_exact(dims.len()).zip(&[0, 11]) {
            assert_eq!(field, bfs_distance_field(&weighted, src));
        }

        let zero = Grid::filled(dims, 0);
        assert!(!has_unit_costs(&zero));
    }

    #[test]
    fn batched_weighted_multilayer_fields_match_cpu_and_preserve_order() {
        let dims = mr_core::Dims::with_layers(9, 5, 3);
        let mut grid = Grid::filled(dims, 1);
        // Distinct weights and obstacles on every plane catch both accidental
        // unit-cost routing and cross-layer row/column wraparound.
        for i in 0..dims.len() {
            let cell = i as CellIdx;
            let (x, y, layer) = dims.xyz(cell);
            let v = ((x * 7 + y * 11 + layer * 13) % 9) + 1;
            grid.set(cell, v);
            if (x + 2 * y + 3 * layer) % 17 == 7 {
                grid.set(cell, mr_core::OBSTACLE);
            }
        }
        let sources = vec![
            dims.idx3(0, 0, 0),
            dims.idx3(8, 4, 2),
            dims.idx3(4, 2, 1),
            dims.idx3(0, 0, 0), // duplicate source pins order/dup semantics
        ];
        for &src in &sources {
            grid.set(src, 1);
        }

        let gpu = metal_sweep_fields(&grid, &sources).unwrap();
        assert_eq!(gpu.len(), sources.len());
        for (field, &src) in gpu.iter().zip(&sources) {
            assert_eq!(field, &bfs_distance_field(&grid, src), "source {src}");
            assert_eq!(field, &sweep_distance_field(&grid, src), "source {src}");
        }
        assert_eq!(gpu[0], gpu[3], "duplicate sources must duplicate fields");

        // The wavefront kernel uses the same packed layer indexing, exercised on
        // a non-zero source layer where the old implementation failed.
        let src = sources[1];
        assert_eq!(
            metal_wavefront_field(&grid, src).unwrap(),
            bfs_distance_field(&grid, src)
        );
    }

    #[test]
    fn deterministic_stress_batch_matches_cpu_reference() {
        let dims = mr_core::Dims::with_layers(17, 11, 2);
        let mut grid = Grid::filled(dims, 1);
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for i in 0..dims.len() {
            let r = next();
            grid.cost[i] = if r % 13 == 0 {
                mr_core::OBSTACLE
            } else {
                1 + (r % 31) as Cost
            };
        }
        let mut sources = Vec::new();
        while sources.len() < 24 {
            let src = (next() % dims.len() as u64) as CellIdx;
            if !grid.is_obstacle(src) {
                sources.push(src);
            }
        }

        let first = metal_sweep_fields(&grid, &sources).unwrap();
        let second = metal_sweep_fields(&grid, &sources).unwrap();
        assert_eq!(first, second, "batched GPU solve must be deterministic");
        for (field, &src) in first.iter().zip(&sources) {
            assert_eq!(field, &bfs_distance_field(&grid, src), "source {src}");
        }

        let nets: Vec<_> = sources
            .iter()
            .enumerate()
            .map(|(i, &src)| NetEndpoints {
                net: format!("stress-{i:02}"),
                src,
                dst: sources[(i * 7 + 5) % sources.len()],
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            })
            .collect();
        let gpu_routes = MetalRouter.route(&grid, &nets).unwrap();
        let cpu_routes = LeeRouter.route(&grid, &nets).unwrap();
        assert_eq!(gpu_routes.results, cpu_routes.results);
        assert_eq!(gpu_routes.unrouted, cpu_routes.unrouted);
        assert_eq!(gpu_routes.congestion, cpu_routes.congestion);
    }

    #[test]
    fn isolated_batch_matches_cpu_on_hanan_weights_zero_overflow_and_pads() {
        let dims = mr_core::Dims::new(9, 7);
        let coords = mr_core::GridCoords::from_lines(
            vec![0.0, 0.2, 0.7, 1.9, 2.0, 4.5, 4.7, 8.0, 8.1],
            vec![0.0, 0.1, 1.0, 1.0, 2.7, 6.0, 6.2],
        );
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx(2, 3), 0);
        grid.set(dims.idx(3, 3), 7);
        grid.set(dims.idx(1, 6), Cost::MAX - 1);
        for (x, y) in [(0, 0), (8, 0), (3, 4), (5, 4), (4, 3), (4, 5)] {
            grid.set(dims.idx(x, y), mr_core::OBSTACLE);
        }
        let nets = vec![
            NetEndpoints {
                net: "weighted".into(),
                src: dims.idx(0, 3),
                dst: dims.idx(8, 3),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "own-obstacle-pads".into(),
                src: dims.idx(0, 0),
                dst: dims.idx(8, 0),
                passable_pads: vec![dims.idx(0, 0), dims.idx(8, 0)],
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "max-minus-one-enter".into(),
                src: dims.idx(0, 6),
                dst: dims.idx(1, 6),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "zero-length".into(),
                src: dims.idx(2, 3),
                dst: dims.idx(2, 3),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "enclosed".into(),
                src: dims.idx(0, 4),
                dst: dims.idx(4, 4),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        assert_isolated_matches_cpu(&grid, &coords, &mr_core::ViaModel::through_hole(1), &nets);
    }

    #[test]
    fn isolated_batch_matches_cpu_on_restricted_and_extreme_vias() {
        let dims = mr_core::Dims::with_layers(6, 5, 4);
        let coords = mr_core::GridCoords::from_lines(
            vec![0.0, 0.25, 1.0, 2.75, 2.8, 7.0],
            vec![0.0, 0.3, 1.2, 1.25, 4.0],
        );
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx3(5, 4, 1), mr_core::OBSTACLE);
        let restricted = mr_core::ViaModel::with_allowed_steps(4, 37, vec![(0, 1), (2, 3)]);
        let nets = vec![
            NetEndpoints {
                net: "lower-span".into(),
                src: dims.idx3(0, 0, 0),
                dst: dims.idx3(5, 4, 1),
                passable_pads: vec![dims.idx3(5, 4, 1)],
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "upper-span".into(),
                src: dims.idx3(5, 0, 2),
                dst: dims.idx3(0, 4, 3),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "forbidden-middle-span".into(),
                src: dims.idx3(2, 2, 0),
                dst: dims.idx3(2, 2, 3),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        assert_isolated_matches_cpu(&grid, &coords, &restricted, &nets);

        // An allowed MAX-priced via into a zero-weight cell has exact step cost 0
        // after multiplication, while the same edge into weight 1 costs MAX-1.
        for layer in 1..dims.layers {
            grid.set(dims.idx3(3, 1, layer), 0);
        }
        let extreme =
            mr_core::ViaModel::with_allowed_steps(4, Cost::MAX, vec![(0, 1), (1, 2), (2, 3)]);
        let extreme_net = [NetEndpoints {
            net: "zero-weight-max-vias".into(),
            src: dims.idx3(3, 1, 0),
            dst: dims.idx3(3, 1, 3),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        }];
        assert_isolated_matches_cpu(&grid, &coords, &extreme, &extreme_net);
    }

    #[test]
    fn isolated_batch_matches_cpu_minimum_hops_and_lower_predecessor_ties() {
        let hop_dims = mr_core::Dims::new(3, 2);
        let mut hop_grid = Grid::filled(hop_dims, 1);
        // The direct two-edge route and the four-edge route around the top both
        // cost 4*SCALE. The secondary label must choose the two-edge route.
        hop_grid.cost = vec![1, 1, 1, 1, 3, 1];
        let hop_net = [NetEndpoints {
            net: "minimum-hops".into(),
            src: hop_dims.idx(0, 1),
            dst: hop_dims.idx(2, 1),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        }];
        let uniform_hop = mr_core::GridCoords::uniform(hop_dims);
        assert_isolated_matches_cpu(
            &hop_grid,
            &uniform_hop,
            &mr_core::ViaModel::through_hole(1),
            &hop_net,
        );
        let (x, y, vias) =
            edge_vectors(hop_dims, &uniform_hop, &mr_core::ViaModel::through_hole(1));
        let got = metal_route_isolated_batch(
            &hop_grid,
            &hop_net,
            &[MetalWindow::full(hop_dims)],
            MetalEdgeCosts {
                x: &x,
                y: &y,
                vias: &vias,
            },
        )
        .unwrap();
        assert_eq!(got[0].as_ref().unwrap().path, [3, 4, 5]);

        let pred_dims = mr_core::Dims::new(4, 3);
        let mut pred_grid = Grid::filled(pred_dims, 1);
        pred_grid.set(pred_dims.idx(2, 1), mr_core::OBSTACLE);
        let pred_net = [NetEndpoints {
            net: "lower-predecessor".into(),
            src: pred_dims.idx(0, 1),
            dst: pred_dims.idx(3, 1),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        }];
        let uniform_pred = mr_core::GridCoords::uniform(pred_dims);
        assert_isolated_matches_cpu(
            &pred_grid,
            &uniform_pred,
            &mr_core::ViaModel::through_hole(1),
            &pred_net,
        );
        let (x, y, vias) = edge_vectors(
            pred_dims,
            &uniform_pred,
            &mr_core::ViaModel::through_hole(1),
        );
        let got = metal_route_isolated_batch(
            &pred_grid,
            &pred_net,
            &[MetalWindow::full(pred_dims)],
            MetalEdgeCosts {
                x: &x,
                y: &y,
                vias: &vias,
            },
        )
        .unwrap();
        assert_eq!(got[0].as_ref().unwrap().path, [4, 0, 1, 2, 3, 7]);
    }

    #[test]
    fn isolated_batch_deterministic_stress_matches_cpu_alone_paths() {
        let dims = mr_core::Dims::with_layers(7, 5, 3);
        let coords = mr_core::GridCoords::from_lines(
            vec![0.0, 0.1, 0.8, 0.85, 2.5, 4.0, 4.2],
            vec![0.0, 0.4, 0.45, 3.0, 3.2],
        );
        for case in 0..12u64 {
            let mut state = 0xA076_1D64_78BD_642Fu64 ^ case;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut grid = Grid::filled(dims, 1);
            for cost in &mut grid.cost {
                let r = next();
                *cost = if r % 11 == 0 {
                    mr_core::OBSTACLE
                } else if r % 7 == 0 {
                    0
                } else {
                    1 + (r % 19) as Cost
                };
            }
            let mut nets = Vec::new();
            for i in 0..8 {
                let src = (next() % dims.len() as u64) as CellIdx;
                let dst = (next() % dims.len() as u64) as CellIdx;
                let mut pads = Vec::new();
                if grid.is_obstacle(src) {
                    pads.push(src);
                }
                if grid.is_obstacle(dst) && dst != src {
                    pads.push(dst);
                }
                nets.push(NetEndpoints {
                    net: format!("case-{case}-net-{i}"),
                    src,
                    dst,
                    passable_pads: pads,
                    via_passable_pads: Vec::new(),
                });
            }
            let via_model = if case % 2 == 0 {
                mr_core::ViaModel::with_allowed_steps(3, 23, vec![(0, 1), (1, 2)])
            } else {
                mr_core::ViaModel::with_allowed_steps(3, 0, vec![(1, 2)])
            };
            assert_isolated_matches_cpu(&grid, &coords, &via_model, &nets);
        }
    }

    #[test]
    fn ragged_batch_matches_full_fields_on_adversarial_variable_windows() {
        let dims = mr_core::Dims::with_layers(13, 11, 3);
        let x: Vec<_> = (0..dims.w - 1).map(|i| (i * 17) % 41).collect();
        let y: Vec<_> = (0..dims.h - 1).map(|i| (i * 23) % 37).collect();
        let via_patterns = [
            vec![Some(0), Some(Cost::MAX)],
            vec![Some(29), None],
            vec![None, Some(7)],
        ];
        for case in 0..18u64 {
            let mut state = 0xD1B5_4A32_D192_ED03u64 ^ case.wrapping_mul(0x9E37_79B9);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut grid = Grid::filled(dims, 1);
            for enter in &mut grid.cost {
                let r = next();
                *enter = if r % 13 == 0 {
                    Cost::MAX
                } else if r % 7 == 0 {
                    0
                } else {
                    1 + (r % 31) as Cost
                };
            }
            let mut nets = Vec::new();
            let mut windows = Vec::new();
            for net_i in 0..24u32 {
                let w = 2 + (next() % 12) as u32;
                let h = 2 + (next() % 10) as u32;
                let x0 = (next() % (dims.w - w + 1) as u64) as u32;
                let y0 = (next() % (dims.h - h + 1) as u64) as u32;
                let window = MetalWindow {
                    x0,
                    y0,
                    x1: x0 + w - 1,
                    y1: y0 + h - 1,
                };
                let src = dims.idx3(
                    x0 + (next() % w as u64) as u32,
                    y0 + (next() % h as u64) as u32,
                    (next() % dims.layers as u64) as u32,
                );
                let dst = dims.idx3(
                    x0 + (next() % w as u64) as u32,
                    y0 + (next() % h as u64) as u32,
                    (next() % dims.layers as u64) as u32,
                );
                let mut pads = Vec::new();
                if grid.is_obstacle(src) {
                    pads.push(src);
                }
                if grid.is_obstacle(dst) && dst != src {
                    pads.push(dst);
                }
                // Exercise own obstacle pads away from endpoints as well.
                let pad = dims.idx3(x0, y0, net_i % dims.layers);
                if grid.is_obstacle(pad) && !pads.contains(&pad) {
                    pads.push(pad);
                }
                nets.push(NetEndpoints {
                    net: format!("ragged-{case:02}-{net_i:02}"),
                    src,
                    dst,
                    passable_pads: pads,
                    via_passable_pads: Vec::new(),
                });
                windows.push(window);
            }
            let vias = &via_patterns[case as usize % via_patterns.len()];
            let edges = MetalEdgeCosts { x: &x, y: &y, vias };
            let full = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
            let first = metal_route_isolated_batch_ragged(&grid, &nets, &windows, edges).unwrap();
            let second = metal_route_isolated_batch_ragged(&grid, &nets, &windows, edges).unwrap();
            assert_eq!(first, full, "case {case}: ragged vs full-field");
            assert_eq!(second, first, "case {case}: deterministic compact output");
        }
    }

    #[test]
    fn ragged_batch_compacts_fields_and_readback_across_chunk_boundary() {
        let dims = mr_core::Dims::with_layers(64, 48, 2);
        let grid = Grid::filled(dims, 1);
        let nets: Vec<_> = (0..=MAX_FIELDS_PER_BATCH)
            .map(|i| {
                let x0 = ((i * 7) % 52) as u32;
                let y0 = ((i * 11) % 36) as u32;
                NetEndpoints {
                    net: format!("compact-{i:03}"),
                    src: dims.idx3(x0, y0, i as u32 % 2),
                    dst: dims.idx3(x0 + 11, y0 + 11, (i as u32 + 1) % 2),
                    passable_pads: Vec::new(),
                    via_passable_pads: Vec::new(),
                }
            })
            .collect();
        let windows: Vec<_> = nets
            .iter()
            .map(|net| {
                let (x0, y0) = dims.xy(net.src);
                MetalWindow {
                    x0,
                    y0,
                    x1: x0 + 11,
                    y1: y0 + 11,
                }
            })
            .collect();
        let x = vec![13; dims.w as usize - 1];
        let y = vec![17; dims.h as usize - 1];
        let vias = [Some(31)];
        let edges = MetalEdgeCosts {
            x: &x,
            y: &y,
            vias: &vias,
        };
        let full = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
        let (ragged, stats) = ragged_isolated_batch_impl(&grid, &nets, &windows, edges).unwrap();
        assert_eq!(ragged, full);
        assert_eq!(stats.chunks, 2);
        assert_eq!(stats.packed_cells, 12 * 12 * 2 * nets.len());
        assert_eq!(stats.full_field_cells, dims.len() * nets.len());
        assert!(stats.packed_cells * 10 < stats.full_field_cells);
        assert!(stats.readback_cells * 10 < stats.packed_cells);
    }

    #[test]
    fn ragged_batch_maps_all_validation_failures_to_backend_unavailable() {
        let dims = mr_core::Dims::new(3, 2);
        let net = NetEndpoints {
            net: "bad".into(),
            src: 0,
            dst: 5,
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };
        let edges = MetalEdgeCosts {
            x: &[SCALE, SCALE],
            y: &[SCALE],
            vias: &[],
        };
        let mut malformed = Grid::filled(dims, 1);
        malformed.cost.pop();
        assert!(matches!(
            metal_route_isolated_batch_ragged(
                &malformed,
                std::slice::from_ref(&net),
                &[MetalWindow::full(dims)],
                edges,
            ),
            Err(RouterError::BackendUnavailable(_))
        ));
        let grid = Grid::filled(dims, 1);
        assert!(matches!(
            metal_route_isolated_batch_ragged(&grid, std::slice::from_ref(&net), &[], edges,),
            Err(RouterError::BackendUnavailable(_))
        ));
        let invalid = NetEndpoints { src: 99, ..net };
        assert!(matches!(
            metal_route_isolated_batch_ragged(&grid, &[invalid], &[MetalWindow::full(dims)], edges,),
            Err(RouterError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn ragged_batch_rejects_any_nonempty_static_via_mask() {
        let dims = mr_core::Dims::with_layers(3, 2, 2);
        let mut grid = Grid::filled(dims, 1);
        // Nonempty selects the static via-mask contract even when no bit is set.
        grid.via_forbidden = vec![false; dims.len()];
        let src = dims.idx3(0, 0, 0);
        let net = NetEndpoints {
            net: "masked-via".into(),
            src,
            dst: dims.idx3(2, 1, 1),
            passable_pads: vec![src],
            via_passable_pads: vec![src],
        };
        let err = metal_route_isolated_batch_ragged(
            &grid,
            &[net],
            &[MetalWindow::full(dims)],
            MetalEdgeCosts {
                x: &[SCALE, SCALE],
                y: &[SCALE],
                vias: &[Some(SCALE)],
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            RouterError::BackendUnavailable(message)
                if message == "ragged isolated Metal routing does not support static via masks"
        ));
    }

    #[test]
    fn isolated_batch_honours_cpu_window_before_full_retry() {
        let dims = mr_core::Dims::new(50, 40);
        let mut grid = Grid::filled(dims, 1);
        for y in 4..36 {
            grid.set(dims.idx(21, y), mr_core::OBSTACLE);
        }
        let dst = dims.idx(22, 20);
        grid.set(dst, mr_core::OBSTACLE);
        let net = NetEndpoints {
            net: "windowed-pad".into(),
            src: dims.idx(20, 20),
            dst,
            passable_pads: vec![dst],
            via_passable_pads: Vec::new(),
        };
        let coords = mr_core::GridCoords::from_lines(
            (0..dims.w).map(|x| x as f64 * 0.17).collect(),
            (0..dims.h).map(|y| y as f64 * 0.11).collect(),
        );
        let via_model = mr_core::ViaModel::through_hole(1);
        assert_isolated_matches_cpu(&grid, &coords, &via_model, std::slice::from_ref(&net));

        let window = test_window(dims, &net);
        assert_ne!(window, MetalWindow::full(dims));
        let (x, y, vias) = edge_vectors(dims, &coords, &via_model);
        let got = metal_route_isolated_batch(
            &grid,
            std::slice::from_ref(&net),
            &[window],
            MetalEdgeCosts {
                x: &x,
                y: &y,
                vias: &vias,
            },
        )
        .unwrap()[0]
            .as_ref()
            .unwrap()
            .path
            .clone();
        assert!(got.iter().all(|&cell| window.contains(dims, cell)));
    }

    #[test]
    fn isolated_batch_reports_window_failure_then_routes_full_retry() {
        let dims = mr_core::Dims::new(5, 3);
        let mut grid = Grid::filled(dims, 1);
        grid.set(dims.idx(2, 1), mr_core::OBSTACLE);
        let net = NetEndpoints {
            net: "retry".into(),
            src: dims.idx(0, 1),
            dst: dims.idx(4, 1),
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };
        let x = vec![SCALE; dims.w as usize - 1];
        let y = vec![SCALE; dims.h as usize - 1];
        let edges = MetalEdgeCosts {
            x: &x,
            y: &y,
            vias: &[],
        };
        let narrow = MetalWindow {
            x0: 0,
            y0: 1,
            x1: 4,
            y1: 1,
        };
        let first = metal_route_isolated_batch(&grid, std::slice::from_ref(&net), &[narrow], edges)
            .unwrap();
        assert_eq!(first, [None]);

        let retry = metal_route_isolated_batch(
            &grid,
            std::slice::from_ref(&net),
            &[MetalWindow::full(dims)],
            edges,
        )
        .unwrap();
        assert!(retry[0].is_some());
        let cpu = NegotiatedRouter::new()
            .route_traced(&grid, std::slice::from_ref(&net))
            .unwrap();
        assert_eq!(retry[0].as_ref().unwrap().path, cpu.1.nets[0].alone_path);
    }

    #[test]
    fn isolated_batch_is_deterministic_across_internal_chunk_boundary() {
        let dims = mr_core::Dims::new(5, 5);
        let grid = Grid::filled(dims, 1);
        let windows = vec![MetalWindow::full(dims); MAX_FIELDS_PER_BATCH + 1];
        let nets: Vec<_> = (0..=MAX_FIELDS_PER_BATCH)
            .map(|i| NetEndpoints {
                net: format!("chunk-{i:03}"),
                src: if i % 2 == 0 {
                    dims.idx(0, 0)
                } else {
                    dims.idx(0, 4)
                },
                dst: if i % 2 == 0 {
                    dims.idx(4, 4)
                } else {
                    dims.idx(4, 0)
                },
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            })
            .collect();
        let x = vec![SCALE; dims.w as usize - 1];
        let y = vec![SCALE; dims.h as usize - 1];
        let edges = MetalEdgeCosts {
            x: &x,
            y: &y,
            vias: &[],
        };
        let first = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
        let second = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), MAX_FIELDS_PER_BATCH + 1);
        for (i, route) in first.iter().enumerate() {
            let route = route.as_ref().unwrap();
            assert_eq!(route.path.first(), Some(&nets[i].src));
            assert_eq!(route.path.last(), Some(&nets[i].dst));
            assert_eq!(route.search_cost, 8 * SCALE);
        }
    }

    #[test]
    fn isolated_batch_rejects_misaligned_shapes_without_partial_output() {
        let dims = mr_core::Dims::new(3, 2);
        let grid = Grid::filled(dims, 1);
        let net = NetEndpoints {
            net: "n".into(),
            src: 0,
            dst: 5,
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };
        assert!(matches!(
            metal_route_isolated_batch(
                &grid,
                std::slice::from_ref(&net),
                &[],
                MetalEdgeCosts {
                    x: &[SCALE, SCALE],
                    y: &[SCALE],
                    vias: &[],
                },
            ),
            Err(RouterError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn isolated_batch_falls_back_before_gpu_for_static_via_mask() {
        let dims = mr_core::Dims::with_layers(1, 1, 2);
        let mut grid = Grid::filled(dims, 1);
        grid.via_forbidden = vec![true; dims.len()];
        let net = NetEndpoints {
            net: "masked-via".into(),
            src: dims.idx3(0, 0, 0),
            dst: dims.idx3(0, 0, 1),
            passable_pads: vec![dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)],
            via_passable_pads: vec![dims.idx3(0, 0, 0), dims.idx3(0, 0, 1)],
        };
        assert!(matches!(
            metal_route_isolated_batch(
                &grid,
                std::slice::from_ref(&net),
                &[MetalWindow::full(dims)],
                MetalEdgeCosts {
                    x: &[],
                    y: &[],
                    vias: &[Some(SCALE)],
                },
            ),
            Err(RouterError::BackendUnavailable(message))
                if message.contains("static via masks")
        ));
    }

    // ---- Router: GPU == CPU under the oracle ---------------------------------

    /// A hand-built multi-net grid: an open 6x6 with several independent nets.
    fn multi_net_grid() -> (Grid, Vec<NetEndpoints>) {
        let dims = mr_core::Dims::new(6, 6);
        let mut b = GridBuilder::new(dims, 1);
        // A short central wall to force some detours / shared congestion.
        b.mark_rect(2, 1, 2, 3); // column x=2, rows 1..=3 (inclusive corners)
        let grid = b.build();
        let nets = vec![
            NetEndpoints {
                net: "a".into(),
                src: dims.idx(0, 0),
                dst: dims.idx(5, 0),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "b".into(),
                src: dims.idx(0, 5),
                dst: dims.idx(5, 5),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "c".into(),
                src: dims.idx(0, 2),
                dst: dims.idx(5, 3),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        (grid, nets)
    }

    #[test]
    fn router_equivalent_to_lee_on_hand_wall() {
        let f = hand_32x32_wall();
        let gpu = MetalRouter.route(&f.grid, &f.nets).unwrap();
        let cpu = LeeRouter.route(&f.grid, &f.nets).unwrap();
        assert!(
            mr_oracle::are_equivalent(&gpu, &cpu),
            "discrepancies: {:?}",
            mr_oracle::compare(&gpu, &cpu)
        );
        assert_eq!(gpu.results[0].cost, 93);
    }

    #[test]
    fn router_equivalent_to_lee_on_tie_break_2x2() {
        let f = tie_break_2x2();
        let gpu = MetalRouter.route(&f.grid, &f.nets).unwrap();
        let cpu = LeeRouter.route(&f.grid, &f.nets).unwrap();
        assert!(
            mr_oracle::are_equivalent(&gpu, &cpu),
            "discrepancies: {:?}",
            mr_oracle::compare(&gpu, &cpu)
        );
        // The tie-break path is pinned to [0, 1, 3].
        assert_eq!(gpu.results[0].path, vec![0, 1, 3]);
    }

    #[test]
    fn router_equivalent_to_lee_on_multi_net() {
        let (grid, nets) = multi_net_grid();
        let gpu = MetalRouter.route(&grid, &nets).unwrap();
        let cpu = LeeRouter.route(&grid, &nets).unwrap();
        assert!(
            mr_oracle::are_equivalent(&gpu, &cpu),
            "discrepancies: {:?}",
            mr_oracle::compare(&gpu, &cpu)
        );
    }

    #[test]
    fn router_equivalent_to_lee_on_battery() {
        for f in obstacle_battery() {
            let gpu = MetalRouter.route(&f.grid, &f.nets).unwrap();
            let cpu = LeeRouter.route(&f.grid, &f.nets).unwrap();
            assert!(
                mr_oracle::are_equivalent(&gpu, &cpu),
                "{}: discrepancies: {:?}",
                f.name,
                mr_oracle::compare(&gpu, &cpu)
            );
        }
    }

    #[test]
    fn router_honours_passable_pads_and_rejects_out_of_range_pad_cells() {
        let dims = mr_core::Dims::new(7, 3);
        let mut grid = Grid::filled(dims, 1);
        let src = dims.idx(0, 1);
        let dst = dims.idx(6, 1);
        grid.set(src, mr_core::OBSTACLE);
        grid.set(dst, mr_core::OBSTACLE);
        // A foreign pad remains blocked, forcing the same detour on CPU and GPU.
        grid.set(dims.idx(3, 1), mr_core::OBSTACLE);
        let net = NetEndpoints {
            net: "pad-net".into(),
            src,
            dst,
            passable_pads: vec![src, dst],
            via_passable_pads: Vec::new(),
        };
        let gpu = MetalRouter
            .route(&grid, std::slice::from_ref(&net))
            .unwrap();
        let cpu = LeeRouter.route(&grid, std::slice::from_ref(&net)).unwrap();
        assert_eq!(gpu.results, cpu.results);
        assert!(gpu.results[0].path.contains(&src));
        assert!(gpu.results[0].path.contains(&dst));
        assert!(!gpu.results[0].path.contains(&dims.idx(3, 1)));

        let invalid = NetEndpoints {
            passable_pads: vec![dims.len() as CellIdx],
            ..net
        };
        assert!(matches!(
            MetalRouter.route(&grid, &[invalid]),
            Err(RouterError::InvalidEndpoint { .. })
        ));
    }

    #[test]
    fn router_matches_lee_on_weighted_multilayer_batch() {
        let dims = mr_core::Dims::with_layers(8, 6, 2);
        let mut grid = Grid::filled(dims, 1);
        for layer in 0..dims.layers {
            for y in 0..dims.h {
                for x in 0..dims.w {
                    let c = dims.idx3(x, y, layer);
                    grid.set(c, 1 + ((x + 3 * y + 5 * layer) % 7));
                }
            }
            grid.set(dims.idx3(3, 2, layer), mr_core::OBSTACLE);
            grid.set(dims.idx3(3, 3, layer), mr_core::OBSTACLE);
        }
        let nets = vec![
            NetEndpoints {
                net: "top".into(),
                src: dims.idx3(0, 2, 0),
                dst: dims.idx3(7, 2, 0),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "bottom".into(),
                src: dims.idx3(7, 3, 1),
                dst: dims.idx3(0, 3, 1),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        let gpu = MetalRouter.route(&grid, &nets).unwrap();
        let cpu = LeeRouter.route(&grid, &nets).unwrap();
        assert_eq!(gpu.results, cpu.results);
        assert_eq!(gpu.unrouted, cpu.unrouted);
        assert_eq!(gpu.congestion, cpu.congestion);
    }

    #[test]
    fn router_matches_lee_without_cycles_on_zero_cost_plateau() {
        let dims = mr_core::Dims::new(3, 2);
        let grid = Grid::filled(dims, 0);
        let nets = vec![
            NetEndpoints {
                net: "plateau".into(),
                src: 5,
                dst: 3,
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
            NetEndpoints {
                net: "reverse".into(),
                src: 3,
                dst: 5,
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            },
        ];
        let cpu = LeeRouter.route(&grid, &nets).unwrap();
        let gpu = MetalRouter.route(&grid, &nets).unwrap();
        assert_eq!(gpu.results, cpu.results);
        for route in &gpu.results {
            assert!(route.path.len() <= dims.len());
            let unique: std::collections::BTreeSet<_> = route.path.iter().collect();
            assert_eq!(unique.len(), route.path.len(), "path must be simple");
        }
    }

    #[test]
    fn router_matches_lee_minimum_hops_on_weighted_cost_tie() {
        let dims = mr_core::Dims::new(3, 2);
        let mut grid = Grid::filled(dims, 1);
        // The direct bottom-row path and the four-edge path around the top row
        // both cost four. Canonical routing minimizes hops before predecessor.
        grid.cost = vec![1, 1, 1, 1, 3, 1];
        let net = NetEndpoints {
            net: "weighted-tie".into(),
            src: 3,
            dst: 5,
            passable_pads: Vec::new(),
            via_passable_pads: Vec::new(),
        };

        let cpu = LeeRouter.route(&grid, std::slice::from_ref(&net)).unwrap();
        let gpu = MetalRouter
            .route(&grid, std::slice::from_ref(&net))
            .unwrap();
        assert_eq!(cpu.results[0].path, vec![3, 4, 5]);
        assert_eq!(cpu.results[0].cost, 4);
        assert_eq!(gpu.results, cpu.results);
        assert_eq!(gpu.congestion, cpu.congestion);
    }

    // ---- D3: CPU vs Metal batch benchmark ------------------------------------

    /// End-to-end full-field vs ragged/compact work at the observed bug05 and
    /// bug50 board dimensions, net counts, and aggregate window-area ratios.
    ///
    /// Run with:
    /// `cargo test -p mr-metal --release ragged_representative_window_benchmark -- --ignored --nocapture`
    #[test]
    #[ignore = "large representative Metal microbenchmark"]
    fn ragged_representative_window_benchmark() {
        let cases = [
            (
                "bug05-shaped",
                mr_core::Dims::with_layers(733, 878, 2),
                228usize,
                220u32,
                25u32,
                360u32,
                20u32,
            ),
            (
                "bug50-shaped",
                mr_core::Dims::with_layers(702, 461, 4),
                322usize,
                170u32,
                15u32,
                175u32,
                15u32,
            ),
        ];
        for (name, dims, net_count, base_w, step_w, base_h, step_h) in cases {
            let grid = Grid::filled(dims, 1);
            let mut nets = Vec::with_capacity(net_count);
            let mut windows = Vec::with_capacity(net_count);
            for i in 0..net_count as u32 {
                let w = base_w + (i % 5) * step_w;
                let h = base_h + (i % 3) * step_h;
                let x0 = (i * 97) % (dims.w - w + 1);
                let y0 = (i * 53) % (dims.h - h + 1);
                let window = MetalWindow {
                    x0,
                    y0,
                    x1: x0 + w - 1,
                    y1: y0 + h - 1,
                };
                nets.push(NetEndpoints {
                    net: format!("{name}-{i:03}"),
                    src: dims.idx3(x0 + 2, y0 + 2, i % dims.layers),
                    dst: dims.idx3(window.x1 - 2, window.y1 - 2, (i + 1) % dims.layers),
                    passable_pads: Vec::new(),
                    via_passable_pads: Vec::new(),
                });
                windows.push(window);
            }
            let x: Vec<_> = (0..dims.w - 1).map(|i| 8 + (i % 7) * 3).collect();
            let y: Vec<_> = (0..dims.h - 1).map(|i| 9 + (i % 5) * 5).collect();
            let vias = vec![Some(160); dims.layers.saturating_sub(1) as usize];
            let edges = MetalEdgeCosts {
                x: &x,
                y: &y,
                vias: &vias,
            };

            let expected = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
            let (ragged, stats) =
                ragged_isolated_batch_impl(&grid, &nets, &windows, edges).unwrap();
            assert_eq!(ragged, expected);
            let mut full_samples = Vec::with_capacity(3);
            let mut ragged_samples = Vec::with_capacity(3);
            for sample in 0..3 {
                if sample % 2 == 0 {
                    let started = Instant::now();
                    let got =
                        metal_route_isolated_batch_ragged(&grid, &nets, &windows, edges).unwrap();
                    ragged_samples.push(started.elapsed());
                    assert_eq!(got, expected);
                    let started = Instant::now();
                    let got = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
                    full_samples.push(started.elapsed());
                    assert_eq!(got, expected);
                } else {
                    let started = Instant::now();
                    let got = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
                    full_samples.push(started.elapsed());
                    assert_eq!(got, expected);
                    let started = Instant::now();
                    let got =
                        metal_route_isolated_batch_ragged(&grid, &nets, &windows, edges).unwrap();
                    ragged_samples.push(started.elapsed());
                    assert_eq!(got, expected);
                }
            }
            full_samples.sort_unstable();
            ragged_samples.sort_unstable();
            let full = full_samples[1];
            let ragged = ragged_samples[1];
            println!(
                "\n=== {name}: {}x{}x{}, {net_count} fields ===\n\
                 field cells: {} -> {} ({:.2}x smaller)\n\
                 host readback u32s: {} -> {} ({:.1}x smaller)\n\
                 warm end-to-end p50/3: full {full:.3?}; ragged {ragged:.3?}; {:.2}x speedup",
                dims.w,
                dims.h,
                dims.layers,
                stats.full_field_cells,
                stats.packed_cells,
                stats.full_field_cells as f64 / stats.packed_cells as f64,
                stats.full_field_cells * 2,
                stats.readback_cells,
                stats.full_field_cells as f64 * 2.0 / stats.readback_cells as f64,
                full.as_secs_f64() / ragged.as_secs_f64(),
            );
        }
    }

    /// End-to-end timing for the exact weighted/Hanan/window/via isolated API.
    /// Includes host mask packing, shared-buffer creation, kernel convergence,
    /// distance+hop readback, and canonical CPU reconstruction.
    #[test]
    fn isolated_batch_benchmark_end_to_end() {
        let dims = mr_core::Dims::with_layers(256, 192, 2);
        let mut grid = Grid::filled(dims, 1);
        for layer in 0..dims.layers {
            for x in (24..dims.w - 8).step_by(31) {
                for y in 8..dims.h - 8 {
                    if y % 47 != 0 {
                        grid.set(dims.idx3(x, y, layer), mr_core::OBSTACLE);
                    }
                }
            }
        }
        let nets: Vec<_> = (0..48u32)
            .map(|i| NetEndpoints {
                net: format!("isolated-{i:02}"),
                src: dims.idx3(0, (i * 37) % dims.h, i % 2),
                dst: dims.idx3(dims.w - 1, (i * 67 + 11) % dims.h, (i + 1) % 2),
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            })
            .collect();
        let windows: Vec<_> = nets.iter().map(|net| test_window(dims, net)).collect();
        let x: Vec<Cost> = (0..dims.w - 1).map(|i| 8 + (i % 7) * 3).collect();
        let y: Vec<Cost> = (0..dims.h - 1).map(|i| 9 + (i % 5) * 5).collect();
        let vias = [Some(160)];
        let edges = MetalEdgeCosts {
            x: &x,
            y: &y,
            vias: &vias,
        };

        let setup_started = Instant::now();
        let expected = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
        let setup = setup_started.elapsed();
        let mut samples = Vec::with_capacity(7);
        for _ in 0..7 {
            let started = Instant::now();
            let got = metal_route_isolated_batch(&grid, &nets, &windows, edges).unwrap();
            samples.push(started.elapsed());
            assert_eq!(got, expected);
        }
        samples.sort_unstable();
        let p50 = samples[samples.len() / 2];
        let routed = expected.iter().filter(|route| route.is_some()).count();
        println!(
            "\n=== exact isolated Metal batch (256x192x2, 48 nets) ===\n\
             setup: {setup:.3?}; warm p50/7: {p50:.3?}; {:.1} nets/sec; {routed} routed",
            nets.len() as f64 / p50.as_secs_f64(),
        );
    }

    /// Build a large open grid with `n` independent random-ish nets (deterministic).
    fn bench_board(side: u32, n_nets: usize) -> (Grid, Vec<NetEndpoints>) {
        let dims = mr_core::Dims::new(side, side);
        // A few scattered vertical walls (each leaving a gap) so routing detours.
        let mut b = GridBuilder::new(dims, 1);
        let y_hi = (side / 2).max(1) - 1; // wall spans rows 1..=y_hi, leaving y=0 open
        for k in 0..(side / 8) {
            let x = (2 + k * 5).min(side - 2);
            b.mark_rect(x, 1, x, y_hi); // inclusive corners: single column
        }
        let grid = b.build();
        // Deterministic endpoint generation that avoids obstacles.
        let mut nets = Vec::with_capacity(n_nets);
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut pick = |grid: &Grid| -> CellIdx {
            loop {
                let i = (next() % dims.len() as u64) as CellIdx;
                if !grid.is_obstacle(i) {
                    return i;
                }
            }
        };
        for k in 0..n_nets {
            let src = pick(&grid);
            let dst = pick(&grid);
            nets.push(NetEndpoints {
                net: format!("n{k}"),
                src,
                dst,
                passable_pads: Vec::new(),
                via_passable_pads: Vec::new(),
            });
        }
        (grid, nets)
    }

    /// D3 deliverable: honest CPU-vs-Metal throughput on PCB-scale grids.
    /// Run with: `cargo test -p mr-metal --release batch_benchmark -- --nocapture`
    #[test]
    fn batch_benchmark_cpu_vs_metal() {
        let side = 128u32;
        let n_nets = 64usize;
        let (grid, nets) = bench_board(side, n_nets);

        // End-to-end CPU route baseline. Lee is target-directed, whereas the
        // current Metal sweep computes complete source fields; report a separate
        // full-field CPU comparison below rather than calling these equal work.
        let t_lee = Instant::now();
        let lee = LeeRouter.route(&grid, &nets).unwrap();
        let lee_dt = t_lee.elapsed();

        let t_cpu_fields = Instant::now();
        let cpu_fields: Vec<_> = nets
            .iter()
            .map(|net| bfs_distance_field(&grid, net.src))
            .collect();
        std::hint::black_box(&cpu_fields);
        let cpu_fields_dt = t_cpu_fields.elapsed();

        // RipUpRouter is the sequential production CPU router (collision-aware);
        // reported for context only — it does MORE work (rip-up passes) and may
        // leave nets unrouted, so it is not a like-for-like throughput compare.
        let t_rip = Instant::now();
        let rip = RipUpRouter.route(&grid, &nets).unwrap();
        let rip_dt = t_rip.elapsed();

        // The setup sample compiles MSL only when this process has not already
        // initialized the global Metal context. Use a median of seven subsequent
        // samples for stable long-lived-worker throughput.
        let t_gpu_setup = Instant::now();
        let gpu_setup = MetalRouter.route(&grid, &nets).unwrap();
        let gpu_setup_dt = t_gpu_setup.elapsed();
        let mut gpu_samples = Vec::with_capacity(7);
        let mut gpu = None;
        for _ in 0..7 {
            let started = Instant::now();
            let sample = MetalRouter.route(&grid, &nets).unwrap();
            gpu_samples.push(started.elapsed());
            assert_eq!(gpu_setup, sample, "repeated GPU routes must match");
            gpu = Some(sample);
        }
        gpu_samples.sort_unstable();
        let gpu_warm_dt = gpu_samples[gpu_samples.len() / 2];
        let gpu = gpu.expect("at least one measured GPU sample");

        let nps = |dt: std::time::Duration| n_nets as f64 / dt.as_secs_f64();

        println!("\n=== D3 batch benchmark ({side}x{side} grid, {n_nets} independent nets) ===");
        println!(
            "CPU  LeeRouter (indep):   {:>10.3?}  {:>9.1} nets/sec  ({} routed)",
            lee_dt,
            nps(lee_dt),
            lee.results.len()
        );
        println!(
            "CPU  full fields (indep): {:>10.3?}  {:>9.1} fields/sec",
            cpu_fields_dt,
            nps(cpu_fields_dt),
        );
        println!(
            "Metal batch (setup):      {:>10.3?}  {:>9.1} nets/sec  ({} routed)",
            gpu_setup_dt,
            nps(gpu_setup_dt),
            gpu.results.len()
        );
        println!(
            "Metal batch (warm p50/7): {:>10.3?}  {:>9.1} nets/sec  ({} routed)",
            gpu_warm_dt,
            nps(gpu_warm_dt),
            gpu.results.len()
        );
        println!(
            "CPU  RipUpRouter (ctx):   {:>10.3?}  {:>9.1} nets/sec  ({} routed, collision-aware)",
            rip_dt,
            nps(rip_dt),
            rip.results.len()
        );

        let route_speedup = lee_dt.as_secs_f64() / gpu_warm_dt.as_secs_f64();
        if route_speedup >= 1.0 {
            println!(
                "End-to-end (targeted Lee vs warm Metal): Metal is {route_speedup:.2}x FASTER"
            );
        } else {
            println!(
                "End-to-end (targeted Lee vs warm Metal): CPU is {:.2}x FASTER",
                1.0 / route_speedup
            );
        }
        let field_speedup = cpu_fields_dt.as_secs_f64() / gpu_warm_dt.as_secs_f64();
        if field_speedup >= 1.0 {
            println!(
                "CPU distance fields vs Metal end-to-end: Metal is {field_speedup:.2}x FASTER"
            );
        } else {
            println!(
                "CPU distance fields vs Metal end-to-end: CPU is {:.2}x FASTER",
                1.0 / field_speedup
            );
        }
    }
}
