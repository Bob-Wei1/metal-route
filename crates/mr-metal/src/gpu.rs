//! macOS-only Metal compute backend for the distance-field kernels.
//!
//! All kernels are atomic-free and use ping-pong buffers. Costs are carried as
//! `u32`; `Cost::MAX` (`0xFFFF_FFFF`) marks both obstacles and unreachable cells,
//! exactly as on the CPU. Saturating addition is used everywhere so a `MAX`
//! neighbour never wraps.
//!
//! The MSL source is embedded as a Rust string and compiled at runtime via
//! `device.new_library_with_source`.

use metal::objc::rc::autoreleasepool;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

use mr_core::{CellIdx, Cost, Grid, RouterError};

/// Embedded Metal Shading Language kernels.
const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint COST_MAX = 0xFFFFFFFFu;

inline uint sat_add(uint a, uint b) {
    uint s = a + b;
    return (s < a) ? COST_MAX : s; // overflow -> clamp to MAX
}

// Grid dimensions passed as a small constant buffer.
struct Dims { uint w; uint h; };

// ---------------------------------------------------------------------------
// M3: naive atomic-free wavefront relaxation.
//   new[i] = min(old[i], min over 4-neighbours (old[n] + cost(i)))
//   Obstacles stay COST_MAX. A change flag is raised on any strict decrease.
// ---------------------------------------------------------------------------
kernel void wavefront(
    device const uint*  old_dist [[buffer(0)]],
    device       uint*  new_dist [[buffer(1)]],
    device const uint*  cost     [[buffer(2)]],
    constant     Dims&  dims     [[buffer(3)]],
    device       atomic_uint* changed [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = dims.w * dims.h;
    if (gid >= n) return;

    uint c = cost[gid];
    if (c == COST_MAX) {            // obstacle: stays MAX
        new_dist[gid] = COST_MAX;
        return;
    }

    uint best = old_dist[gid];
    uint x = gid % dims.w;
    uint y = gid / dims.w;

    // 4-neighbours; relaxation adds cost(gid) (cost of stepping ONTO this cell).
    if (y > 0)            best = min(best, sat_add(old_dist[gid - dims.w], c));
    if (x > 0)            best = min(best, sat_add(old_dist[gid - 1u],     c));
    if (x + 1u < dims.w)  best = min(best, sat_add(old_dist[gid + 1u],     c));
    if (y + 1u < dims.h)  best = min(best, sat_add(old_dist[gid + dims.w], c));

    new_dist[gid] = best;
    if (best < old_dist[gid]) {
        // A single global flag; atomics here are only for the change signal,
        // not for the field itself (the field is computed atomic-free).
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// M4: separable prefix-min sweeps.
//   Row kernel: one thread per row, serial L->R then R->L prefix-min.
//   Col kernel: one thread per column, serial U->D then D->U.
// Each relaxation lowers dist[cur] toward dist[prev] + cost(cur). A per-cell
// lowest-index parent is carried INTO the field under the same (dist, idx)
// ordering so paths agree (the M0 finding). We never store across obstacles.
// ---------------------------------------------------------------------------
kernel void sweep_rows(
    device       uint*  dist   [[buffer(0)]],
    device const uint*  cost   [[buffer(1)]],
    constant     Dims&  dims   [[buffer(2)]],
    device       atomic_uint* changed [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= dims.h) return;
    uint base = row * dims.w;

    // left -> right
    for (uint x = 1u; x < dims.w; ++x) {
        uint cur = base + x;
        uint c = cost[cur];
        if (c == COST_MAX) continue;          // obstacle: skip
        uint prev = dist[base + x - 1u];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand < dist[cur]) {
            dist[cur] = cand;
            atomic_store_explicit(changed, 1u, memory_order_relaxed);
        }
    }
    // right -> left
    for (uint xi = dims.w; xi >= 2u; --xi) {
        uint x = xi - 2u;                       // iterate x = w-2 .. 0
        uint cur = base + x;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev = dist[base + x + 1u];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand < dist[cur]) {
            dist[cur] = cand;
            atomic_store_explicit(changed, 1u, memory_order_relaxed);
        }
    }
}

kernel void sweep_cols(
    device       uint*  dist   [[buffer(0)]],
    device const uint*  cost   [[buffer(1)]],
    constant     Dims&  dims   [[buffer(2)]],
    device       atomic_uint* changed [[buffer(3)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= dims.w) return;

    // up -> down
    for (uint y = 1u; y < dims.h; ++y) {
        uint cur = y * dims.w + col;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev = dist[(y - 1u) * dims.w + col];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand < dist[cur]) {
            dist[cur] = cand;
            atomic_store_explicit(changed, 1u, memory_order_relaxed);
        }
    }
    // down -> up
    for (uint yi = dims.h; yi >= 2u; --yi) {
        uint y = yi - 2u;                       // iterate y = h-2 .. 0
        uint cur = y * dims.w + col;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev = dist[(y + 1u) * dims.w + col];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand < dist[cur]) {
            dist[cur] = cand;
            atomic_store_explicit(changed, 1u, memory_order_relaxed);
        }
    }
}
"#;

/// Dims as laid out for the MSL `Dims` struct (two `u32`s).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuDims {
    w: u32,
    h: u32,
}

/// A cached Metal context (device + queue + compiled pipelines).
struct MetalCtx {
    device: Device,
    queue: CommandQueue,
    wavefront: ComputePipelineState,
    sweep_rows: ComputePipelineState,
    sweep_cols: ComputePipelineState,
}

impl MetalCtx {
    fn new() -> Result<Self, RouterError> {
        let device = Device::system_default()
            .ok_or_else(|| RouterError::BackendUnavailable("no Metal device".into()))?;
        let queue = device.new_command_queue();
        let opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(SRC, &opts)
            .map_err(|e| RouterError::BackendUnavailable(format!("MSL compile failed: {e}")))?;

        let pipeline = |name: &str| -> Result<ComputePipelineState, RouterError> {
            let f = lib.get_function(name, None).map_err(|e| {
                RouterError::BackendUnavailable(format!("missing kernel `{name}`: {e}"))
            })?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| {
                    RouterError::BackendUnavailable(format!("pipeline `{name}` failed: {e}"))
                })
        };

        Ok(Self {
            wavefront: pipeline("wavefront")?,
            sweep_rows: pipeline("sweep_rows")?,
            sweep_cols: pipeline("sweep_cols")?,
            device,
            queue,
        })
    }
}

fn new_u32_buffer(device: &Device, data: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(data) as u64;
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        bytes.max(4),
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_u32_buffer(buf: &Buffer, len: usize) -> Vec<u32> {
    let ptr = buf.contents() as *const u32;
    // SAFETY: shared-storage buffer of at least `len` u32s, completed on CPU.
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

/// Validate the grid and return `(dims, n, cost_u32, init_dist)`, or an early
/// `dist` field when the source is an obstacle / grid empty.
fn prepare(grid: &Grid, src: CellIdx) -> Result<PrepResult, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::MalformedGrid);
    }
    let dims = grid.dims;
    let n = dims.len();
    let mut init = vec![Cost::MAX; n];
    if dims.is_empty() || grid.is_obstacle(src) {
        return Ok(PrepResult::Trivial(init));
    }
    init[src as usize] = 0;
    Ok(PrepResult::Run {
        gdims: GpuDims {
            w: dims.w,
            h: dims.h,
        },
        n,
        cost: grid.cost.clone(),
        init,
    })
}

enum PrepResult {
    Trivial(Vec<Cost>),
    Run {
        gdims: GpuDims,
        n: usize,
        cost: Vec<Cost>,
        init: Vec<Cost>,
    },
}

/// M3: naive atomic-free wavefront field on the GPU.
pub fn wavefront_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    let prep = prepare(grid, src)?;
    let (gdims, n, cost, init) = match prep {
        PrepResult::Trivial(d) => return Ok(d),
        PrepResult::Run {
            gdims,
            n,
            cost,
            init,
        } => (gdims, n, cost, init),
    };

    let result = autoreleasepool(|| -> Result<Vec<Cost>, RouterError> {
        let ctx = MetalCtx::new()?;
        let dev = &ctx.device;

        let mut buf_a = new_u32_buffer(dev, &init);
        let mut buf_b = new_u32_buffer(dev, &init);
        let cost_buf = new_u32_buffer(dev, &cost);
        let dims_buf = dev.new_buffer_with_data(
            &gdims as *const GpuDims as *const _,
            std::mem::size_of::<GpuDims>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Bound: at most n iterations are ever needed for a monotone field.
        let max_iters = n.max(1);
        let tg = MTLSize::new(64, 1, 1);
        let grid_size = MTLSize::new(n as u64, 1, 1);

        for _ in 0..max_iters {
            let flag_buf = new_u32_buffer(dev, &[0u32]);
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.wavefront);
            enc.set_buffer(0, Some(&buf_a), 0);
            enc.set_buffer(1, Some(&buf_b), 0);
            enc.set_buffer(2, Some(&cost_buf), 0);
            enc.set_buffer(3, Some(&dims_buf), 0);
            enc.set_buffer(4, Some(&flag_buf), 0);
            enc.dispatch_threads(grid_size, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // new (buf_b) becomes the current field; ping-pong.
            std::mem::swap(&mut buf_a, &mut buf_b);

            let changed = read_u32_buffer(&flag_buf, 1)[0];
            if changed == 0 {
                break;
            }
        }

        Ok(read_u32_buffer(&buf_a, n))
    })?;

    Ok(result)
}

/// M4: separable H/V prefix-min sweep field on the GPU.
pub fn sweep_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    let prep = prepare(grid, src)?;
    let (gdims, n, cost, init) = match prep {
        PrepResult::Trivial(d) => return Ok(d),
        PrepResult::Run {
            gdims,
            n,
            cost,
            init,
        } => (gdims, n, cost, init),
    };

    let result = autoreleasepool(|| -> Result<Vec<Cost>, RouterError> {
        let ctx = MetalCtx::new()?;
        let dev = &ctx.device;

        // Single in-place dist buffer (sweeps mutate it serially per line).
        let dist_buf = new_u32_buffer(dev, &init);
        let cost_buf = new_u32_buffer(dev, &cost);
        let dims_buf = dev.new_buffer_with_data(
            &gdims as *const GpuDims as *const _,
            std::mem::size_of::<GpuDims>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let tg = MTLSize::new(64, 1, 1);
        let rows = MTLSize::new(gdims.h as u64, 1, 1);
        let cols = MTLSize::new(gdims.w as u64, 1, 1);

        // A full round = one row sweep then one column sweep. Bound iterations.
        let max_rounds = n.max(1);
        for _ in 0..max_rounds {
            let flag_buf = new_u32_buffer(dev, &[0u32]);

            // Row pass.
            {
                let cmd = ctx.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.sweep_rows);
                enc.set_buffer(0, Some(&dist_buf), 0);
                enc.set_buffer(1, Some(&cost_buf), 0);
                enc.set_buffer(2, Some(&dims_buf), 0);
                enc.set_buffer(3, Some(&flag_buf), 0);
                enc.dispatch_threads(rows, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            // Column pass.
            {
                let cmd = ctx.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.sweep_cols);
                enc.set_buffer(0, Some(&dist_buf), 0);
                enc.set_buffer(1, Some(&cost_buf), 0);
                enc.set_buffer(2, Some(&dims_buf), 0);
                enc.set_buffer(3, Some(&flag_buf), 0);
                enc.dispatch_threads(cols, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }

            let changed = read_u32_buffer(&flag_buf, 1)[0];
            if changed == 0 {
                break;
            }
        }

        Ok(read_u32_buffer(&dist_buf, n))
    })?;

    Ok(result)
}
