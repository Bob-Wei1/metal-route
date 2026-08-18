//! macOS-only Metal compute backend for the distance-field kernels.
//!
//! All kernels are atomic-free and use ping-pong buffers. Costs are carried as
//! `u32`; `Cost::MAX` (`0xFFFF_FFFF`) marks both obstacles and unreachable cells,
//! exactly as on the CPU. Saturating addition is used everywhere so a `MAX`
//! neighbour never wraps.
//!
//! The MSL source is embedded as a Rust string and compiled at runtime via
//! `device.new_library_with_source`.

use std::sync::OnceLock;

use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

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

// Grid dimensions passed as a small constant buffer. `batch` independent
// distance fields are packed consecutively, each with `w*h*layers` cells.
struct Dims {
    uint w;
    uint h;
    uint layers;
    uint batch;
    uint cost_stride;
    uint track_hops;
};

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
    uint plane = dims.w * dims.h;
    uint n = plane * dims.layers;
    uint total = n * dims.batch;
    if (gid >= total) return;

    uint field = gid / n;
    uint field_cell = gid - field * n;
    uint c = cost[(dims.cost_stride == 0u) ? field_cell : gid];
    if (c == COST_MAX) {            // obstacle: stays MAX
        new_dist[gid] = COST_MAX;
        return;
    }

    uint best = old_dist[gid];
    uint local = gid % plane;
    uint x = local % dims.w;
    uint y = local / dims.w;

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
// Each relaxation lowers distance at cur toward distance(prev) + cost(cur).
// Weighted/zero-cost routing also carries the corresponding minimum-hop label;
// distance-only and unit-cost calls skip all full hop-buffer traffic. We never
// store across obstacles.
// ---------------------------------------------------------------------------
kernel void sweep_rows(
    device       uint*  dist   [[buffer(0)]],
    device const uint*  cost   [[buffer(1)]],
    constant     Dims&  dims   [[buffer(2)]],
    device       atomic_uint* changed [[buffer(3)]],
    device       uint*  hops   [[buffer(4)]],
    uint row_gid [[thread_position_in_grid]])
{
    uint n = dims.w * dims.h * dims.layers;
    uint rows_per_field = dims.h * dims.layers;
    if (row_gid >= rows_per_field * dims.batch) return;
    uint field = row_gid / rows_per_field;
    uint row = row_gid % rows_per_field;
    uint base = field * n + row * dims.w;
    bool line_changed = false;

    // left -> right
    for (uint x = 1u; x < dims.w; ++x) {
        uint cur = base + x;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;          // obstacle: skip
        uint prev_idx = base + x - 1u;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand != COST_MAX) {
            if (dims.track_hops == 0u) {
                if (cand < dist[cur]) {
                    dist[cur] = cand;
                    line_changed = true;
                }
            } else {
                uint cand_hops = sat_add(hops[prev_idx], 1u);
                if (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur])) {
                    dist[cur] = cand;
                    hops[cur] = cand_hops;
                    line_changed = true;
                }
            }
        }
    }
    // right -> left
    for (uint xi = dims.w; xi >= 2u; --xi) {
        uint x = xi - 2u;                       // iterate x = w-2 .. 0
        uint cur = base + x;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = base + x + 1u;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand != COST_MAX) {
            if (dims.track_hops == 0u) {
                if (cand < dist[cur]) {
                    dist[cur] = cand;
                    line_changed = true;
                }
            } else {
                uint cand_hops = sat_add(hops[prev_idx], 1u);
                if (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur])) {
                    dist[cur] = cand;
                    hops[cur] = cand_hops;
                    line_changed = true;
                }
            }
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

kernel void sweep_cols(
    device       uint*  dist   [[buffer(0)]],
    device const uint*  cost   [[buffer(1)]],
    constant     Dims&  dims   [[buffer(2)]],
    device       atomic_uint* changed [[buffer(3)]],
    device       uint*  hops   [[buffer(4)]],
    uint col_gid [[thread_position_in_grid]])
{
    uint plane = dims.w * dims.h;
    uint n = plane * dims.layers;
    uint cols_per_field = dims.w * dims.layers;
    if (col_gid >= cols_per_field * dims.batch) return;
    uint field = col_gid / cols_per_field;
    uint layer_col = col_gid % cols_per_field;
    uint layer = layer_col / dims.w;
    uint col = layer_col % dims.w;
    uint base = field * n + layer * plane;
    bool line_changed = false;

    // up -> down
    for (uint y = 1u; y < dims.h; ++y) {
        uint cur = base + y * dims.w + col;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = base + (y - 1u) * dims.w + col;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand != COST_MAX) {
            if (dims.track_hops == 0u) {
                if (cand < dist[cur]) {
                    dist[cur] = cand;
                    line_changed = true;
                }
            } else {
                uint cand_hops = sat_add(hops[prev_idx], 1u);
                if (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur])) {
                    dist[cur] = cand;
                    hops[cur] = cand_hops;
                    line_changed = true;
                }
            }
        }
    }
    // down -> up
    for (uint yi = dims.h; yi >= 2u; --yi) {
        uint y = yi - 2u;                       // iterate y = h-2 .. 0
        uint cur = base + y * dims.w + col;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = base + (y + 1u) * dims.w + col;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, c);
        if (cand != COST_MAX) {
            if (dims.track_hops == 0u) {
                if (cand < dist[cur]) {
                    dist[cur] = cand;
                    line_changed = true;
                }
            } else {
                uint cand_hops = sat_add(hops[prev_idx], 1u);
                if (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur])) {
                    dist[cur] = cand;
                    hops[cur] = cand_hops;
                    line_changed = true;
                }
            }
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Edge-aware M4 for NegotiatedRouter's independent/alone-path phase.
//
// Planar and via edge prices are supplied already rounded into the CPU router's
// fixed-point Cost domain.  A move u -> v costs
// min(edge_price(u,v) * enter_weight(v), COST_MAX - 1), exactly matching
// negotiated.rs::passable_search_cost.  COST_MAX itself remains reserved for
// obstacles/unreachable labels.  Every edge-aware solve carries hop labels.
// ---------------------------------------------------------------------------
inline uint passable_mul(uint edge_price, uint enter_weight) {
    ulong product = ulong(edge_price) * ulong(enter_weight);
    return uint(min(product, ulong(COST_MAX - 1u)));
}

kernel void sweep_rows_edges(
    device       uint*  dist       [[buffer(0)]],
    device const uint*  cost       [[buffer(1)]],
    device const uint*  x_edges    [[buffer(2)]],
    constant     Dims&  dims       [[buffer(3)]],
    device       atomic_uint* changed [[buffer(4)]],
    device       uint*  hops       [[buffer(5)]],
    uint row_gid [[thread_position_in_grid]])
{
    uint n = dims.w * dims.h * dims.layers;
    uint rows_per_field = dims.h * dims.layers;
    if (row_gid >= rows_per_field * dims.batch) return;
    uint field = row_gid / rows_per_field;
    uint row = row_gid % rows_per_field;
    uint base = field * n + row * dims.w;
    bool line_changed = false;

    for (uint x = 1u; x < dims.w; ++x) {
        uint cur = base + x;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur - 1u;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(x_edges[x - 1u], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    for (uint xi = dims.w; xi >= 2u; --xi) {
        uint x = xi - 2u;
        uint cur = base + x;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur + 1u;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(x_edges[x], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

kernel void sweep_cols_edges(
    device       uint*  dist       [[buffer(0)]],
    device const uint*  cost       [[buffer(1)]],
    device const uint*  y_edges    [[buffer(2)]],
    constant     Dims&  dims       [[buffer(3)]],
    device       atomic_uint* changed [[buffer(4)]],
    device       uint*  hops       [[buffer(5)]],
    uint col_gid [[thread_position_in_grid]])
{
    uint plane = dims.w * dims.h;
    uint n = plane * dims.layers;
    uint cols_per_field = dims.w * dims.layers;
    if (col_gid >= cols_per_field * dims.batch) return;
    uint field = col_gid / cols_per_field;
    uint layer_col = col_gid % cols_per_field;
    uint layer = layer_col / dims.w;
    uint col = layer_col % dims.w;
    uint base = field * n + layer * plane;
    bool line_changed = false;

    for (uint y = 1u; y < dims.h; ++y) {
        uint cur = base + y * dims.w + col;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur - dims.w;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(y_edges[y - 1u], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    for (uint yi = dims.h; yi >= 2u; --yi) {
        uint y = yi - 2u;
        uint cur = base + y * dims.w + col;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur + dims.w;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(y_edges[y], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

// One thread owns one (field, x, y) layer column.  `via_allowed[k]` gates the
// adjacent transition k <-> k+1 independently from its (possibly MAX/zero) cost.
kernel void sweep_vias_edges(
    device       uint*  dist       [[buffer(0)]],
    device const uint*  cost       [[buffer(1)]],
    device const uint*  via_edges  [[buffer(2)]],
    device const uint*  via_allowed [[buffer(3)]],
    constant     Dims&  dims       [[buffer(4)]],
    device       atomic_uint* changed [[buffer(5)]],
    device       uint*  hops       [[buffer(6)]],
    uint via_gid [[thread_position_in_grid]])
{
    uint plane = dims.w * dims.h;
    if (via_gid >= plane * dims.batch) return;
    uint field = via_gid / plane;
    uint xy = via_gid % plane;
    uint n = plane * dims.layers;
    uint base = field * n + xy;
    bool line_changed = false;

    for (uint layer = 1u; layer < dims.layers; ++layer) {
        if (via_allowed[layer - 1u] == 0u) continue;
        uint cur = base + layer * plane;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur - plane;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(via_edges[layer - 1u], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    for (uint li = dims.layers; li >= 2u; --li) {
        uint layer = li - 2u;
        if (via_allowed[layer] == 0u) continue;
        uint cur = base + layer * plane;
        uint c = cost[(dims.cost_stride == 0u) ? (cur - field * n) : cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur + plane;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(via_edges[layer], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Experimental ragged/cropped edge-aware batch.
//
// Fields are packed back-to-back at their real per-net window sizes.  Host-built
// line descriptors let one kernel sweep rows and columns without padding every
// field to the full board.  `edge0` maps a cropped line back to the global Hanan
// edge vectors, so weighted planar/via semantics remain identical to M4 above.
// ---------------------------------------------------------------------------
struct RaggedLine {
    uint base;
    uint len;
    uint stride;
    uint edge0;
};

struct RaggedField {
    uint cell_offset;
    uint w;
    uint h;
    uint layers;
    uint x0;
    uint y0;
    uint src;
    uint dst;
};

struct RaggedMeta {
    uint fields;
    uint global_w;
    uint global_h;
    uint reserved;
};

kernel void sweep_ragged_lines_edges(
    device       uint*  dist       [[buffer(0)]],
    device const uint*  cost       [[buffer(1)]],
    device const uint*  edges      [[buffer(2)]],
    device const RaggedLine* lines [[buffer(3)]],
    device       atomic_uint* changed [[buffer(4)]],
    device       uint*  hops       [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    RaggedLine line = lines[gid];
    bool line_changed = false;
    for (uint pos = 1u; pos < line.len; ++pos) {
        uint cur = line.base + pos * line.stride;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur - line.stride;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(edges[line.edge0 + pos - 1u], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    for (uint posi = line.len; posi >= 2u; --posi) {
        uint pos = posi - 2u;
        uint cur = line.base + pos * line.stride;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur + line.stride;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(edges[line.edge0 + pos], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

kernel void sweep_ragged_vias_edges(
    device       uint*  dist       [[buffer(0)]],
    device const uint*  cost       [[buffer(1)]],
    device const uint*  via_edges  [[buffer(2)]],
    device const uint*  via_allowed [[buffer(3)]],
    device const RaggedField* fields [[buffer(4)]],
    device       atomic_uint* changed [[buffer(5)]],
    device       uint*  hops       [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    RaggedField field = fields[gid.y];
    uint plane = field.w * field.h;
    if (gid.x >= plane) return;
    uint base = field.cell_offset + gid.x;
    bool line_changed = false;
    for (uint layer = 1u; layer < field.layers; ++layer) {
        if (via_allowed[layer - 1u] == 0u) continue;
        uint cur = base + layer * plane;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur - plane;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(via_edges[layer - 1u], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    for (uint layeri = field.layers; layeri >= 2u; --layeri) {
        uint layer = layeri - 2u;
        if (via_allowed[layer] == 0u) continue;
        uint cur = base + layer * plane;
        uint c = cost[cur];
        if (c == COST_MAX) continue;
        uint prev_idx = cur + plane;
        uint prev = dist[prev_idx];
        if (prev == COST_MAX) continue;
        uint cand = sat_add(prev, passable_mul(via_edges[layer], c));
        uint cand_hops = sat_add(hops[prev_idx], 1u);
        if (cand != COST_MAX &&
            (cand < dist[cur] || (cand == dist[cur] && cand_hops < hops[cur]))) {
            dist[cur] = cand;
            hops[cur] = cand_hops;
            line_changed = true;
        }
    }
    if (line_changed) {
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

inline bool ragged_pred_matches(
    uint pred,
    uint cur,
    uint edge,
    device const uint* dist,
    device const uint* hops,
    device const uint* cost)
{
    uint dp = dist[pred];
    uint hp = hops[pred];
    uint need_hops = hops[cur];
    if (dp == COST_MAX || hp == COST_MAX || need_hops == 0u || need_hops == COST_MAX) {
        return false;
    }
    uint step = passable_mul(edge, cost[cur]);
    return sat_add(dp, step) == dist[cur] && sat_add(hp, 1u) == need_hops;
}

// Return the canonical lower global CellIdx predecessor. Cropping never changes
// index ordering, so checking lower layer, up, left, right, down, upper is exact.
inline uint ragged_predecessor(
    RaggedField field,
    uint cur,
    device const uint* dist,
    device const uint* hops,
    device const uint* cost,
    device const uint* x_edges,
    device const uint* y_edges,
    device const uint* via_edges,
    device const uint* via_allowed)
{
    uint plane = field.w * field.h;
    uint rel = cur - field.cell_offset;
    uint layer = rel / plane;
    uint xy = rel - layer * plane;
    uint x = xy % field.w;
    uint y = xy / field.w;

    if (layer > 0u && via_allowed[layer - 1u] != 0u) {
        uint pred = cur - plane;
        if (ragged_pred_matches(pred, cur, via_edges[layer - 1u], dist, hops, cost)) return pred;
    }
    if (y > 0u) {
        uint pred = cur - field.w;
        if (ragged_pred_matches(pred, cur, y_edges[field.y0 + y - 1u], dist, hops, cost)) return pred;
    }
    if (x > 0u) {
        uint pred = cur - 1u;
        if (ragged_pred_matches(pred, cur, x_edges[field.x0 + x - 1u], dist, hops, cost)) return pred;
    }
    if (x + 1u < field.w) {
        uint pred = cur + 1u;
        if (ragged_pred_matches(pred, cur, x_edges[field.x0 + x], dist, hops, cost)) return pred;
    }
    if (y + 1u < field.h) {
        uint pred = cur + field.w;
        if (ragged_pred_matches(pred, cur, y_edges[field.y0 + y], dist, hops, cost)) return pred;
    }
    if (layer + 1u < field.layers && via_allowed[layer] != 0u) {
        uint pred = cur + plane;
        if (ragged_pred_matches(pred, cur, via_edges[layer], dist, hops, cost)) return pred;
    }
    return COST_MAX;
}

kernel void ragged_path_lengths(
    device const uint* dist [[buffer(0)]],
    device const uint* hops [[buffer(1)]],
    device const uint* cost [[buffer(2)]],
    device const uint* x_edges [[buffer(3)]],
    device const uint* y_edges [[buffer(4)]],
    device const uint* via_edges [[buffer(5)]],
    device const uint* via_allowed [[buffer(6)]],
    device const RaggedField* fields [[buffer(7)]],
    device uint* lengths [[buffer(8)]],
    device uint* route_costs [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    RaggedField field = fields[gid];
    if (field.src == COST_MAX || field.dst == COST_MAX || dist[field.dst] == COST_MAX) {
        lengths[gid] = 0u;
        route_costs[gid] = COST_MAX;
        return;
    }
    route_costs[gid] = dist[field.dst];
    uint max_len = field.w * field.h * field.layers;
    uint len = 1u;
    uint cur = field.dst;
    while (cur != field.src) {
        if (len >= max_len) {
            lengths[gid] = COST_MAX;
            return;
        }
        cur = ragged_predecessor(
            field, cur, dist, hops, cost, x_edges, y_edges, via_edges, via_allowed);
        if (cur == COST_MAX) {
            lengths[gid] = COST_MAX;
            return;
        }
        ++len;
    }
    lengths[gid] = len;
}

inline uint ragged_global_cell(
    RaggedField field,
    constant RaggedMeta& meta,
    uint packed_cell)
{
    uint plane = field.w * field.h;
    uint rel = packed_cell - field.cell_offset;
    uint layer = rel / plane;
    uint xy = rel - layer * plane;
    uint x = xy % field.w;
    uint y = xy / field.w;
    return layer * meta.global_w * meta.global_h
        + (field.y0 + y) * meta.global_w + field.x0 + x;
}

kernel void ragged_write_paths(
    device const uint* dist [[buffer(0)]],
    device const uint* hops [[buffer(1)]],
    device const uint* cost [[buffer(2)]],
    device const uint* x_edges [[buffer(3)]],
    device const uint* y_edges [[buffer(4)]],
    device const uint* via_edges [[buffer(5)]],
    device const uint* via_allowed [[buffer(6)]],
    device const RaggedField* fields [[buffer(7)]],
    device const uint* lengths [[buffer(8)]],
    device const uint* offsets [[buffer(9)]],
    device uint* paths [[buffer(10)]],
    constant RaggedMeta& meta [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint len = lengths[gid];
    if (len == 0u || len == COST_MAX) return;
    RaggedField field = fields[gid];
    uint cur = field.dst;
    for (uint rev = 0u; rev < len; ++rev) {
        paths[offsets[gid] + len - rev - 1u] = ragged_global_cell(field, meta, cur);
        if (cur == field.src) return;
        cur = ragged_predecessor(
            field, cur, dist, hops, cost, x_edges, y_edges, via_edges, via_allowed);
        if (cur == COST_MAX) return;
    }
}
"#;

/// Dims as laid out for the MSL `Dims` struct (six `u32`s).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuDims {
    w: u32,
    h: u32,
    layers: u32,
    batch: u32,
    /// Zero means every field shares one `n`-cell cost grid; `n` means costs
    /// are packed field-by-field (needed for per-net passable pads).
    cost_stride: u32,
    /// Zero runs the distance-only fast path; one carries minimum hop counts.
    track_hops: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RaggedField {
    pub(crate) cell_offset: u32,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) layers: u32,
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    /// Absolute packed-cell index, or `u32::MAX` when outside the window.
    pub(crate) src: u32,
    /// Absolute packed-cell index, or `u32::MAX` when outside the window.
    pub(crate) dst: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RaggedLine {
    base: u32,
    len: u32,
    stride: u32,
    edge0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RaggedMeta {
    fields: u32,
    global_w: u32,
    global_h: u32,
    reserved: u32,
}

pub(crate) struct RaggedPathBatch {
    pub(crate) paths: Vec<Option<Vec<CellIdx>>>,
    pub(crate) search_costs: Vec<Cost>,
    /// Number of cropped field cells retained on the GPU for this chunk.
    pub(crate) packed_cells: usize,
    /// Host-visible `u32`s: two words per field plus actual routed path cells.
    pub(crate) readback_cells: usize,
}

/// Allocation-free description of every index space and buffer in one cropped
/// batch.  The public wrapper builds this from windows before it reserves the
/// packed cost plane; [`sweep_ragged_paths`] independently rebuilds it from the
/// concrete descriptors before allocating its line/label buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RaggedBatchShape {
    pub(crate) fields: usize,
    pub(crate) packed_cells: usize,
    pub(crate) row_lines: usize,
    pub(crate) col_lines: usize,
    pub(crate) max_plane: usize,
    pub(crate) max_field_cells: usize,
}

/// Conservative cap for one ragged chunk's simultaneous *additional* host and
/// Metal allocations. This is deliberately independent of
/// `MTLDevice::maxBufferLength`: that property limits one buffer, not the several
/// packed planes and host mirrors which coexist during route reconstruction.
const MAX_RAGGED_AGGREGATE_WORKING_SET_BYTES: usize = 2 * 1024 * 1024 * 1024;

// Peak allocation accounting used by `ragged_aggregate_working_set_bytes`:
//
// * 36 bytes / packed cell: host packed-cost, distance-init and hop-init planes;
//   Metal cost, distance and hop planes; plus three conservative reconstruction /
//   readback path planes.
// * 32 bytes / row or column: one 16-byte descriptor on host and in Metal.
// * 128 bytes / field: host + Metal field descriptors, length/cost/offset metadata,
//   nested result bookkeeping, and alignment/headroom.
// * x/y edges need one Metal u32 copy. Via prices and permissions have both the
//   derived host vectors and Metal copies (four u32 words per layer gap total).
const RAGGED_PACKED_CELL_PEAK_BYTES: usize = 9 * std::mem::size_of::<u32>();
const RAGGED_LINE_PEAK_BYTES: usize = 2 * std::mem::size_of::<RaggedLine>();
const RAGGED_FIELD_PEAK_BYTES: usize = 128;
const RAGGED_EDGE_PEAK_BYTES: usize = std::mem::size_of::<u32>();
const RAGGED_VIA_PEAK_BYTES: usize = 4 * std::mem::size_of::<u32>();
const RAGGED_FIXED_OVERHEAD_BYTES: usize = 4096;

/// A cached Metal context (device + queue + compiled pipelines).
struct MetalCtx {
    device: Device,
    queue: CommandQueue,
    wavefront: ComputePipelineState,
    sweep_rows: ComputePipelineState,
    sweep_cols: ComputePipelineState,
    sweep_rows_edges: ComputePipelineState,
    sweep_cols_edges: ComputePipelineState,
    sweep_vias_edges: ComputePipelineState,
    sweep_ragged_lines_edges: ComputePipelineState,
    sweep_ragged_vias_edges: ComputePipelineState,
    ragged_path_lengths: ComputePipelineState,
    ragged_write_paths: ComputePipelineState,
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
            sweep_rows_edges: pipeline("sweep_rows_edges")?,
            sweep_cols_edges: pipeline("sweep_cols_edges")?,
            sweep_vias_edges: pipeline("sweep_vias_edges")?,
            sweep_ragged_lines_edges: pipeline("sweep_ragged_lines_edges")?,
            sweep_ragged_vias_edges: pipeline("sweep_ragged_vias_edges")?,
            ragged_path_lengths: pipeline("ragged_path_lengths")?,
            ragged_write_paths: pipeline("ragged_write_paths")?,
            device,
            queue,
        })
    }
}

/// Compiling MSL and creating pipelines is process setup, not per-board routing
/// work. metal-rs retains these objects and marks them thread-safe, so every
/// server worker can share one device, queue, and set of compiled pipelines.
static METAL_CONTEXT: OnceLock<Result<MetalCtx, String>> = OnceLock::new();

fn with_metal_ctx<T>(
    f: impl FnOnce(&MetalCtx) -> Result<T, RouterError>,
) -> Result<T, RouterError> {
    let cached = METAL_CONTEXT.get_or_init(|| MetalCtx::new().map_err(|error| error.to_string()));
    let ctx = cached
        .as_ref()
        .map_err(|message| RouterError::BackendUnavailable(message.clone()))?;
    f(ctx)
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data) as u64;
    debug_assert!(bytes > 0, "Metal buffer inputs use explicit dummy words");
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        bytes,
        MTLResourceOptions::StorageModeShared,
    )
}

fn new_u32_buffer(device: &Device, data: &[u32]) -> Buffer {
    new_buffer(device, data)
}

fn read_u32_buffer(buf: &Buffer, len: usize) -> Vec<u32> {
    let ptr = buf.contents() as *const u32;
    // SAFETY: shared-storage buffer of at least `len` u32s, completed on CPU.
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

fn reset_u32_buffer(buf: &Buffer) {
    // SAFETY: `buf` is a shared-storage buffer allocated from one `u32` below,
    // and every previous command buffer using it has completed before reset.
    unsafe {
        *(buf.contents() as *mut u32) = 0;
    }
}

fn command_status_result(status: MTLCommandBufferStatus) -> Result<(), RouterError> {
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        Err(RouterError::BackendUnavailable(format!(
            "Metal command buffer did not complete successfully: {status:?}"
        )))
    }
}

/// Commit one encoded command buffer and surface asynchronous Metal failures.
/// `waitUntilCompleted` only blocks; runtime timeout/OOM/device failures are
/// reported through the final status and must become a backend error before any
/// shared-buffer contents are interpreted as a valid routing result.
fn run_command_buffer(cmd: &CommandBufferRef) -> Result<(), RouterError> {
    cmd.commit();
    cmd.wait_until_completed();
    command_status_result(cmd.status())
}

fn validate_buffer_len(device: &Device, elements: usize, label: &str) -> Result<(), RouterError> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| RouterError::BackendUnavailable(format!("{label} buffer is too large")))?;
    let max = device.max_buffer_length() as usize;
    if bytes > max {
        return Err(RouterError::BackendUnavailable(format!(
            "{label} buffer needs {bytes} bytes; Metal device limit is {max}"
        )));
    }
    Ok(())
}

fn validate_edge_array_lengths(
    grid: &Grid,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
    via_allowed_elements: usize,
) -> Result<(), RouterError> {
    let dims = grid.dims;
    let expected_x = (dims.w as usize).saturating_sub(1);
    let expected_y = (dims.h as usize).saturating_sub(1);
    let expected_vias = (dims.layers as usize).saturating_sub(1);
    if x_elements != expected_x
        || y_elements != expected_y
        || via_elements != expected_vias
        || via_allowed_elements != expected_vias
    {
        return Err(RouterError::BackendUnavailable(
            "edge-cost array lengths do not match grid dimensions".into(),
        ));
    }
    Ok(())
}

fn validate_edge_batch_buffers(
    device: &Device,
    total: usize,
    cost_elements: usize,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
    via_allowed_elements: usize,
) -> Result<(), RouterError> {
    validate_buffer_len(device, total, "edge-aware distance")?;
    validate_buffer_len(device, total, "edge-aware hop count")?;
    validate_buffer_len(device, cost_elements, "edge-aware enter weights")?;
    // Empty edge arrays are represented by one-word dummy buffers.
    validate_buffer_len(device, x_elements.max(1), "edge-aware x edges")?;
    validate_buffer_len(device, y_elements.max(1), "edge-aware y edges")?;
    validate_buffer_len(device, via_elements.max(1), "edge-aware via edges")?;
    validate_buffer_len(
        device,
        via_allowed_elements.max(1),
        "edge-aware via permissions",
    )?;
    Ok(())
}

fn validate_ragged_index_shape(shape: RaggedBatchShape) -> Result<(), RouterError> {
    if shape.fields == 0 {
        if shape != RaggedBatchShape::default() {
            return Err(RouterError::BackendUnavailable(
                "empty ragged Metal shape contains work".into(),
            ));
        }
        return Ok(());
    }
    if shape.packed_cells == 0
        || shape.row_lines == 0
        || shape.col_lines == 0
        || shape.max_plane == 0
        || shape.max_field_cells == 0
        || shape.fields > shape.packed_cells
        || shape.row_lines > shape.packed_cells
        || shape.col_lines > shape.packed_cells
        || shape.max_plane > shape.max_field_cells
        || shape.max_field_cells > shape.packed_cells
    {
        return Err(RouterError::BackendUnavailable(
            "invalid ragged Metal batch shape".into(),
        ));
    }
    for (value, label) in [
        (shape.fields, "fields"),
        (shape.packed_cells, "packed cells"),
        (shape.row_lines, "row descriptors"),
        (shape.col_lines, "column descriptors"),
        (shape.max_plane, "maximum plane"),
        (shape.max_field_cells, "maximum field"),
    ] {
        u32::try_from(value).map_err(|_| {
            RouterError::BackendUnavailable(format!(
                "ragged Metal {label} exceed 32-bit shader indexing"
            ))
        })?;
    }
    Ok(())
}

/// Allocation-free upper bound for all transient allocations that can coexist while
/// one ragged chunk is solved and reconstructed. Checked arithmetic is mandatory at
/// this boundary: an overflow is itself an unsafe resource request and must trigger
/// the caller's CPU fallback before any derived vector or Metal buffer is allocated.
fn ragged_aggregate_working_set_bytes(
    shape: RaggedBatchShape,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
) -> Result<usize, RouterError> {
    let too_large =
        || RouterError::BackendUnavailable("ragged aggregate working set is too large".into());
    let scaled = |count: usize, bytes: usize| count.checked_mul(bytes).ok_or_else(&too_large);

    let line_count = shape
        .row_lines
        .checked_add(shape.col_lines)
        .ok_or_else(&too_large)?;
    let planar_edges = x_elements.checked_add(y_elements).ok_or_else(&too_large)?;

    [
        scaled(shape.packed_cells, RAGGED_PACKED_CELL_PEAK_BYTES)?,
        scaled(line_count, RAGGED_LINE_PEAK_BYTES)?,
        scaled(shape.fields, RAGGED_FIELD_PEAK_BYTES)?,
        scaled(planar_edges, RAGGED_EDGE_PEAK_BYTES)?,
        scaled(via_elements, RAGGED_VIA_PEAK_BYTES)?,
    ]
    .into_iter()
    .try_fold(RAGGED_FIXED_OVERHEAD_BYTES, |total, bytes| {
        total.checked_add(bytes).ok_or_else(&too_large)
    })
}

fn validate_ragged_aggregate_working_set(
    shape: RaggedBatchShape,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
) -> Result<(), RouterError> {
    let bytes = ragged_aggregate_working_set_bytes(shape, x_elements, y_elements, via_elements)?;
    if bytes > MAX_RAGGED_AGGREGATE_WORKING_SET_BYTES {
        return Err(RouterError::BackendUnavailable(format!(
            "ragged aggregate working set needs {bytes} bytes; safety cap is \
             {MAX_RAGGED_AGGREGATE_WORKING_SET_BYTES}"
        )));
    }
    Ok(())
}

fn validate_ragged_batch_buffers(
    device: &Device,
    shape: RaggedBatchShape,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
    via_allowed_elements: usize,
) -> Result<(), RouterError> {
    validate_edge_batch_buffers(
        device,
        shape.packed_cells,
        shape.packed_cells,
        x_elements,
        y_elements,
        via_elements,
        via_allowed_elements,
    )?;
    let field_words = shape.fields.checked_mul(8).ok_or_else(|| {
        RouterError::BackendUnavailable("ragged field descriptors are too large".into())
    })?;
    let row_words = shape.row_lines.checked_mul(4).ok_or_else(|| {
        RouterError::BackendUnavailable("ragged row descriptors are too large".into())
    })?;
    let col_words = shape.col_lines.checked_mul(4).ok_or_else(|| {
        RouterError::BackendUnavailable("ragged column descriptors are too large".into())
    })?;
    validate_buffer_len(device, field_words, "ragged fields")?;
    validate_buffer_len(device, row_words, "ragged rows")?;
    validate_buffer_len(device, col_words, "ragged columns")?;
    validate_buffer_len(device, shape.fields, "ragged path lengths")?;
    validate_buffer_len(device, shape.fields, "ragged route costs")?;
    validate_buffer_len(device, shape.fields, "ragged path offsets")?;
    // The compact output cannot exceed one simple path cell per packed field
    // cell. Validate that worst case before any corresponding host allocation.
    validate_buffer_len(device, shape.packed_cells, "ragged compact paths")?;
    Ok(())
}

/// Validate an allocation-free cropped-batch plan against both the shader's
/// 32-bit index domain, the aggregate working-set cap, and every Metal buffer it
/// can create. Callers use this before reserving or packing the cost plane, so an
/// oversized single field is still supported when all resource checks approve it.
pub(crate) fn preflight_ragged_edge_batch(
    grid: &Grid,
    shape: RaggedBatchShape,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
) -> Result<(), RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::BackendUnavailable(
            "ragged Metal preflight received a malformed grid".into(),
        ));
    }
    // Compact cells are converted back to the crate-wide u32 CellIdx domain in
    // the path kernel, so a small crop cannot make an oversized global grid safe.
    checked_batch_cells(grid.dims.len(), 1)?;
    validate_edge_array_lengths(grid, x_elements, y_elements, via_elements, via_elements)?;
    validate_ragged_index_shape(shape)?;
    if shape.fields == 0 {
        return Ok(());
    }
    validate_ragged_aggregate_working_set(shape, x_elements, y_elements, via_elements)?;
    autoreleasepool(|| {
        with_metal_ctx(|ctx| {
            validate_ragged_batch_buffers(
                &ctx.device,
                shape,
                x_elements,
                y_elements,
                via_elements,
                via_elements,
            )
        })
    })
}

pub(crate) fn checked_batch_cells(
    cells_per_field: usize,
    fields: usize,
) -> Result<usize, RouterError> {
    let total = cells_per_field
        .checked_mul(fields)
        .ok_or_else(|| RouterError::BackendUnavailable("Metal batch is too large".into()))?;
    if total > u32::MAX as usize {
        return Err(RouterError::BackendUnavailable(
            "Metal batch exceeds 32-bit shader indexing".into(),
        ));
    }
    Ok(total)
}

/// Validate a packed edge-aware batch before its full host-side cost planes are
/// allocated. Returns the exact packed element count for `Vec::with_capacity`.
pub(crate) fn preflight_packed_edge_batch(
    grid: &Grid,
    fields: usize,
    x_elements: usize,
    y_elements: usize,
    via_elements: usize,
) -> Result<usize, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::MalformedGrid);
    }
    validate_edge_array_lengths(grid, x_elements, y_elements, via_elements, via_elements)?;
    let total = checked_batch_cells(grid.dims.len(), fields)?;
    if fields == 0 || total == 0 {
        return Ok(total);
    }
    autoreleasepool(|| {
        with_metal_ctx(|ctx| {
            validate_edge_batch_buffers(
                &ctx.device,
                total,
                total,
                x_elements,
                y_elements,
                via_elements,
                via_elements,
            )?;
            Ok(total)
        })
    })
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
    if dims.is_empty() || !dims.contains(src) || grid.is_obstacle(src) {
        return Ok(PrepResult::Trivial(init));
    }
    init[src as usize] = 0;
    Ok(PrepResult::Run {
        gdims: GpuDims {
            w: dims.w,
            h: dims.h,
            layers: dims.layers,
            batch: 1,
            cost_stride: 0,
            track_hops: 0,
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

    let result = autoreleasepool(|| {
        with_metal_ctx(|ctx| {
            let dev = &ctx.device;
            validate_buffer_len(dev, n, "wavefront distance")?;
            validate_buffer_len(dev, cost.len(), "wavefront cost")?;

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

            let flag_buf = new_u32_buffer(dev, &[0u32]);
            for _ in 0..max_iters {
                reset_u32_buffer(&flag_buf);
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
                run_command_buffer(cmd)?;

                // new (buf_b) becomes the current field; ping-pong.
                std::mem::swap(&mut buf_a, &mut buf_b);

                let changed = read_u32_buffer(&flag_buf, 1)[0];
                if changed == 0 {
                    break;
                }
            }

            Ok(read_u32_buffer(&buf_a, n))
        })
    })?;

    Ok(result)
}

/// Flat M4 output. Keeping fields contiguous avoids a second full-buffer copy in
/// `MetalRouter`; public nested-vector entry points split only at their boundary.
pub(crate) struct FlatSweep {
    pub(crate) dist: Vec<Cost>,
    pub(crate) hops: Option<Vec<u32>>,
}

/// Solve several independent fields in one Metal submission stream.
///
/// `costs` is either one shared `n`-cell grid or `sources.len()` packed grids.
/// When `track_hops` is false, the shader performs distance-only relaxation and
/// binds only a one-word dummy hop buffer. This is the public field-computation
/// path and the unit-cost router fast path. Weighted/zero-cost routing sets it to
/// true to carry the canonical minimum-hop label alongside distance.
pub(crate) fn sweep_fields_flat(
    grid: &Grid,
    sources: &[CellIdx],
    costs: &[Cost],
    track_hops: bool,
) -> Result<FlatSweep, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::MalformedGrid);
    }
    if sources.is_empty() {
        return Ok(FlatSweep {
            dist: Vec::new(),
            hops: track_hops.then(Vec::new),
        });
    }

    let dims = grid.dims;
    let n = dims.len();
    let total = checked_batch_cells(n, sources.len())?;
    if costs.len() != n && costs.len() != total {
        return Err(RouterError::MalformedGrid);
    }
    if n == 0 {
        return Ok(FlatSweep {
            dist: Vec::new(),
            hops: track_hops.then(Vec::new),
        });
    }

    let batch = u32::try_from(sources.len())
        .map_err(|_| RouterError::BackendUnavailable("Metal batch is too large".into()))?;
    let gdims = GpuDims {
        w: dims.w,
        h: dims.h,
        layers: dims.layers,
        batch,
        cost_stride: if costs.len() == n { 0 } else { n as u32 },
        track_hops: u32::from(track_hops),
    };

    autoreleasepool(|| {
        with_metal_ctx(|ctx| {
            // Validate device limits before allocating/filling any total-sized host
            // vectors. Public callers are chunked too, but this internal guard keeps
            // direct uses from attempting multi-gigabyte allocations first.
            let dev = &ctx.device;
            validate_buffer_len(dev, total, "batched distance")?;
            if track_hops {
                validate_buffer_len(dev, total, "batched hop count")?;
            }
            validate_buffer_len(dev, costs.len(), "batched cost")?;

            let mut init = vec![Cost::MAX; total];
            let mut hop_init = track_hops.then(|| vec![u32::MAX; total]);
            let mut any_runnable = false;
            for (field, &src) in sources.iter().enumerate() {
                let cost_offset = if costs.len() == n { 0 } else { field * n };
                if dims.contains(src) && costs[cost_offset + src as usize] != Cost::MAX {
                    init[field * n + src as usize] = 0;
                    if let Some(hops) = hop_init.as_mut() {
                        hops[field * n + src as usize] = 0;
                    }
                    any_runnable = true;
                }
            }
            if !any_runnable {
                return Ok(FlatSweep {
                    dist: init,
                    hops: hop_init,
                });
            }

            let dist_buf = new_u32_buffer(dev, &init);
            let dummy_hops = [u32::MAX];
            let hop_buf = new_u32_buffer(dev, hop_init.as_deref().unwrap_or(&dummy_hops));
            let cost_buf = new_u32_buffer(dev, costs);
            let dims_buf = dev.new_buffer_with_data(
                &gdims as *const GpuDims as *const _,
                std::mem::size_of::<GpuDims>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let flag_buf = new_u32_buffer(dev, &[0u32]);

            let tg = MTLSize::new(64, 1, 1);
            let rows = MTLSize::new((gdims.h * gdims.layers * gdims.batch) as u64, 1, 1);
            let cols = MTLSize::new((gdims.w * gdims.layers * gdims.batch) as u64, 1, 1);

            // A full round = one row sweep then one column sweep. Both encoders live
            // in one command buffer, so Metal preserves the dependency without a
            // CPU round-trip between passes. A single global flag is sufficient:
            // convergence means no field changed.
            for _ in 0..n.max(1) {
                reset_u32_buffer(&flag_buf);
                let cmd = ctx.queue.new_command_buffer();
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_rows);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&dims_buf), 0);
                    enc.set_buffer(3, Some(&flag_buf), 0);
                    enc.set_buffer(4, Some(&hop_buf), 0);
                    enc.dispatch_threads(rows, tg);
                    enc.end_encoding();
                }
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_cols);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&dims_buf), 0);
                    enc.set_buffer(3, Some(&flag_buf), 0);
                    enc.set_buffer(4, Some(&hop_buf), 0);
                    enc.dispatch_threads(cols, tg);
                    enc.end_encoding();
                }
                run_command_buffer(cmd)?;
                if read_u32_buffer(&flag_buf, 1)[0] == 0 {
                    break;
                }
            }

            Ok(FlatSweep {
                dist: read_u32_buffer(&dist_buf, total),
                hops: track_hops.then(|| read_u32_buffer(&hop_buf, total)),
            })
        })
    })
}

/// Edge-aware batched M4 used by the negotiated router's independent-path seam.
///
/// `costs` contains destination enter weights (one shared plane or one packed plane
/// per field). `x_edges`, `y_edges`, and `via_edges` are already rounded into the
/// caller's fixed-point domain. `via_allowed[k]` independently gates layer step
/// `k <-> k+1`, so a legal zero/MAX-priced via remains distinguishable from a
/// forbidden transition. Both distance and minimum-hop fields are always returned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sweep_fields_flat_edges(
    grid: &Grid,
    sources: &[CellIdx],
    costs: &[Cost],
    x_edges: &[Cost],
    y_edges: &[Cost],
    via_edges: &[Cost],
    via_allowed: &[u32],
) -> Result<FlatSweep, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::MalformedGrid);
    }
    let dims = grid.dims;
    validate_edge_array_lengths(
        grid,
        x_edges.len(),
        y_edges.len(),
        via_edges.len(),
        via_allowed.len(),
    )?;
    if sources.is_empty() {
        return Ok(FlatSweep {
            dist: Vec::new(),
            hops: Some(Vec::new()),
        });
    }

    let n = dims.len();
    let total = checked_batch_cells(n, sources.len())?;
    if costs.len() != n && costs.len() != total {
        return Err(RouterError::MalformedGrid);
    }
    if n == 0 {
        return Ok(FlatSweep {
            dist: Vec::new(),
            hops: Some(Vec::new()),
        });
    }

    let batch = u32::try_from(sources.len())
        .map_err(|_| RouterError::BackendUnavailable("Metal batch is too large".into()))?;
    let gdims = GpuDims {
        w: dims.w,
        h: dims.h,
        layers: dims.layers,
        batch,
        cost_stride: if costs.len() == n { 0 } else { n as u32 },
        track_hops: 1,
    };

    autoreleasepool(|| {
        with_metal_ctx(|ctx| {
            let dev = &ctx.device;
            validate_edge_batch_buffers(
                dev,
                total,
                costs.len(),
                x_edges.len(),
                y_edges.len(),
                via_edges.len(),
                via_allowed.len(),
            )?;

            let mut init = vec![Cost::MAX; total];
            let mut hop_init = vec![u32::MAX; total];
            let mut any_runnable = false;
            for (field, &src) in sources.iter().enumerate() {
                let cost_offset = if costs.len() == n { 0 } else { field * n };
                if dims.contains(src) && costs[cost_offset + src as usize] != Cost::MAX {
                    init[field * n + src as usize] = 0;
                    hop_init[field * n + src as usize] = 0;
                    any_runnable = true;
                }
            }
            if !any_runnable {
                return Ok(FlatSweep {
                    dist: init,
                    hops: Some(hop_init),
                });
            }

            let zero = [0u32];
            let dist_buf = new_u32_buffer(dev, &init);
            let hop_buf = new_u32_buffer(dev, &hop_init);
            let cost_buf = new_u32_buffer(dev, costs);
            let x_buf = new_u32_buffer(dev, if x_edges.is_empty() { &zero } else { x_edges });
            let y_buf = new_u32_buffer(dev, if y_edges.is_empty() { &zero } else { y_edges });
            let via_buf = new_u32_buffer(
                dev,
                if via_edges.is_empty() {
                    &zero
                } else {
                    via_edges
                },
            );
            let via_allowed_buf = new_u32_buffer(
                dev,
                if via_allowed.is_empty() {
                    &zero
                } else {
                    via_allowed
                },
            );
            let dims_buf = dev.new_buffer_with_data(
                &gdims as *const GpuDims as *const _,
                std::mem::size_of::<GpuDims>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let flag_buf = new_u32_buffer(dev, &[0u32]);

            let tg = MTLSize::new(64, 1, 1);
            let rows = MTLSize::new((gdims.h * gdims.layers * gdims.batch) as u64, 1, 1);
            let cols = MTLSize::new((gdims.w * gdims.layers * gdims.batch) as u64, 1, 1);
            let vias = MTLSize::new((gdims.w * gdims.h * gdims.batch) as u64, 1, 1);

            for _ in 0..n.max(1) {
                reset_u32_buffer(&flag_buf);
                let cmd = ctx.queue.new_command_buffer();
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_rows_edges);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&dims_buf), 0);
                    enc.set_buffer(4, Some(&flag_buf), 0);
                    enc.set_buffer(5, Some(&hop_buf), 0);
                    enc.dispatch_threads(rows, tg);
                    enc.end_encoding();
                }
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_cols_edges);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&y_buf), 0);
                    enc.set_buffer(3, Some(&dims_buf), 0);
                    enc.set_buffer(4, Some(&flag_buf), 0);
                    enc.set_buffer(5, Some(&hop_buf), 0);
                    enc.dispatch_threads(cols, tg);
                    enc.end_encoding();
                }
                if gdims.layers > 1 {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_vias_edges);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&via_buf), 0);
                    enc.set_buffer(3, Some(&via_allowed_buf), 0);
                    enc.set_buffer(4, Some(&dims_buf), 0);
                    enc.set_buffer(5, Some(&flag_buf), 0);
                    enc.set_buffer(6, Some(&hop_buf), 0);
                    enc.dispatch_threads(vias, tg);
                    enc.end_encoding();
                }
                run_command_buffer(cmd)?;
                if read_u32_buffer(&flag_buf, 1)[0] == 0 {
                    break;
                }
            }

            Ok(FlatSweep {
                dist: read_u32_buffer(&dist_buf, total),
                hops: Some(read_u32_buffer(&hop_buf, total)),
            })
        })
    })
}

/// Solve exact weighted fields at their true cropped sizes and reconstruct paths
/// on the GPU. Distance/hop planes remain device-side; only two metadata words per
/// field and the actual path cells are copied into Rust vectors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sweep_ragged_paths(
    grid: &Grid,
    fields: &[RaggedField],
    costs: &[Cost],
    x_edges: &[Cost],
    y_edges: &[Cost],
    via_edges: &[Cost],
    via_allowed: &[u32],
) -> Result<RaggedPathBatch, RouterError> {
    if !grid.is_well_formed() {
        return Err(RouterError::BackendUnavailable(
            "ragged Metal batch received a malformed grid".into(),
        ));
    }
    validate_edge_array_lengths(
        grid,
        x_edges.len(),
        y_edges.len(),
        via_edges.len(),
        via_allowed.len(),
    )?;
    if fields.is_empty() {
        if !costs.is_empty() {
            return Err(RouterError::BackendUnavailable(
                "empty ragged Metal batch has packed costs".into(),
            ));
        }
        return Ok(RaggedPathBatch {
            paths: Vec::new(),
            search_costs: Vec::new(),
            packed_cells: 0,
            readback_cells: 0,
        });
    }

    let dims = grid.dims;
    let mut expected_offset = 0usize;
    let mut max_field_cells = 0usize;
    let mut max_plane = 0usize;
    let mut row_lines = 0usize;
    let mut col_lines = 0usize;
    for field in fields {
        if field.w == 0
            || field.h == 0
            || field.layers != dims.layers
            || field.x0.checked_add(field.w).is_none_or(|x| x > dims.w)
            || field.y0.checked_add(field.h).is_none_or(|y| y > dims.h)
            || field.cell_offset as usize != expected_offset
        {
            return Err(RouterError::BackendUnavailable(
                "invalid ragged Metal field descriptor".into(),
            ));
        }
        let plane = (field.w as usize)
            .checked_mul(field.h as usize)
            .ok_or_else(|| RouterError::BackendUnavailable("ragged field is too large".into()))?;
        let cells = plane
            .checked_mul(field.layers as usize)
            .ok_or_else(|| RouterError::BackendUnavailable("ragged field is too large".into()))?;
        let end = expected_offset
            .checked_add(cells)
            .ok_or_else(|| RouterError::BackendUnavailable("ragged batch is too large".into()))?;
        if end > u32::MAX as usize
            || [field.src, field.dst].into_iter().any(|cell| {
                cell != u32::MAX && !((expected_offset as u32)..(end as u32)).contains(&cell)
            })
        {
            return Err(RouterError::BackendUnavailable(
                "ragged Metal endpoint/offset is out of range".into(),
            ));
        }
        max_field_cells = max_field_cells.max(cells);
        max_plane = max_plane.max(plane);
        row_lines = row_lines
            .checked_add(
                (field.h as usize)
                    .checked_mul(field.layers as usize)
                    .ok_or_else(|| {
                        RouterError::BackendUnavailable(
                            "ragged row descriptor count is too large".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                RouterError::BackendUnavailable("ragged row descriptors are too large".into())
            })?;
        col_lines = col_lines
            .checked_add(
                (field.w as usize)
                    .checked_mul(field.layers as usize)
                    .ok_or_else(|| {
                        RouterError::BackendUnavailable(
                            "ragged column descriptor count is too large".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                RouterError::BackendUnavailable("ragged column descriptors are too large".into())
            })?;
        expected_offset = end;
    }
    if costs.len() != expected_offset {
        return Err(RouterError::BackendUnavailable(
            "ragged Metal cost packing does not match descriptors".into(),
        ));
    }
    let total = checked_batch_cells(costs.len(), 1)?;

    let shape = RaggedBatchShape {
        fields: fields.len(),
        packed_cells: total,
        row_lines,
        col_lines,
        max_plane,
        max_field_cells,
    };
    preflight_ragged_edge_batch(grid, shape, x_edges.len(), y_edges.len(), via_edges.len())?;

    let mut rows = Vec::new();
    rows.try_reserve_exact(row_lines).map_err(|_| {
        RouterError::BackendUnavailable("cannot allocate ragged row descriptors".into())
    })?;
    let mut cols = Vec::new();
    cols.try_reserve_exact(col_lines).map_err(|_| {
        RouterError::BackendUnavailable("cannot allocate ragged column descriptors".into())
    })?;
    for field in fields {
        let plane_u32 = field.w.checked_mul(field.h).ok_or_else(|| {
            RouterError::BackendUnavailable("ragged field plane is too large".into())
        })?;
        for layer in 0..field.layers {
            let layer_base = field
                .cell_offset
                .checked_add(layer.checked_mul(plane_u32).ok_or_else(|| {
                    RouterError::BackendUnavailable("ragged field offset is too large".into())
                })?)
                .ok_or_else(|| {
                    RouterError::BackendUnavailable("ragged field offset is too large".into())
                })?;
            for y in 0..field.h {
                rows.push(RaggedLine {
                    base: layer_base
                        .checked_add(y.checked_mul(field.w).ok_or_else(|| {
                            RouterError::BackendUnavailable("ragged row offset is too large".into())
                        })?)
                        .ok_or_else(|| {
                            RouterError::BackendUnavailable("ragged row offset is too large".into())
                        })?,
                    len: field.w,
                    stride: 1,
                    edge0: field.x0,
                });
            }
            for x in 0..field.w {
                cols.push(RaggedLine {
                    base: layer_base.checked_add(x).ok_or_else(|| {
                        RouterError::BackendUnavailable("ragged column offset is too large".into())
                    })?,
                    len: field.h,
                    stride: field.w,
                    edge0: field.y0,
                });
            }
        }
    }
    debug_assert_eq!(rows.len(), row_lines);
    debug_assert_eq!(cols.len(), col_lines);

    let mut init = Vec::new();
    init.try_reserve_exact(total).map_err(|_| {
        RouterError::BackendUnavailable("cannot allocate ragged distance labels".into())
    })?;
    init.resize(total, Cost::MAX);
    let mut hop_init = Vec::new();
    hop_init
        .try_reserve_exact(total)
        .map_err(|_| RouterError::BackendUnavailable("cannot allocate ragged hop labels".into()))?;
    hop_init.resize(total, u32::MAX);
    let mut any_runnable = false;
    for field in fields {
        if field.src != u32::MAX && costs[field.src as usize] != Cost::MAX {
            init[field.src as usize] = 0;
            hop_init[field.src as usize] = 0;
            any_runnable = true;
        }
    }
    if !any_runnable {
        return Ok(RaggedPathBatch {
            paths: vec![None; fields.len()],
            search_costs: vec![Cost::MAX; fields.len()],
            packed_cells: total,
            readback_cells: 0,
        });
    }

    autoreleasepool(|| {
        with_metal_ctx(|ctx| {
            let dev = &ctx.device;
            validate_ragged_batch_buffers(
                dev,
                shape,
                x_edges.len(),
                y_edges.len(),
                via_edges.len(),
                via_allowed.len(),
            )?;

            let zero = [0u32];
            let dist_buf = new_u32_buffer(dev, &init);
            let hop_buf = new_u32_buffer(dev, &hop_init);
            let cost_buf = new_u32_buffer(dev, costs);
            let x_buf = new_u32_buffer(dev, if x_edges.is_empty() { &zero } else { x_edges });
            let y_buf = new_u32_buffer(dev, if y_edges.is_empty() { &zero } else { y_edges });
            let via_buf = new_u32_buffer(
                dev,
                if via_edges.is_empty() {
                    &zero
                } else {
                    via_edges
                },
            );
            let via_allowed_buf = new_u32_buffer(
                dev,
                if via_allowed.is_empty() {
                    &zero
                } else {
                    via_allowed
                },
            );
            let field_buf = new_buffer(dev, fields);
            let row_buf = new_buffer(dev, &rows);
            let col_buf = new_buffer(dev, &cols);
            let flag_buf = new_u32_buffer(dev, &[0u32]);

            let tg = MTLSize::new(64, 1, 1);
            let row_threads = MTLSize::new(rows.len() as u64, 1, 1);
            let col_threads = MTLSize::new(cols.len() as u64, 1, 1);
            let via_threads = MTLSize::new(max_plane as u64, fields.len() as u64, 1);
            for _ in 0..max_field_cells.max(1) {
                reset_u32_buffer(&flag_buf);
                let cmd = ctx.queue.new_command_buffer();
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_ragged_lines_edges);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&row_buf), 0);
                    enc.set_buffer(4, Some(&flag_buf), 0);
                    enc.set_buffer(5, Some(&hop_buf), 0);
                    enc.dispatch_threads(row_threads, tg);
                    enc.end_encoding();
                }
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_ragged_lines_edges);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&y_buf), 0);
                    enc.set_buffer(3, Some(&col_buf), 0);
                    enc.set_buffer(4, Some(&flag_buf), 0);
                    enc.set_buffer(5, Some(&hop_buf), 0);
                    enc.dispatch_threads(col_threads, tg);
                    enc.end_encoding();
                }
                if dims.layers > 1 {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.sweep_ragged_vias_edges);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&cost_buf), 0);
                    enc.set_buffer(2, Some(&via_buf), 0);
                    enc.set_buffer(3, Some(&via_allowed_buf), 0);
                    enc.set_buffer(4, Some(&field_buf), 0);
                    enc.set_buffer(5, Some(&flag_buf), 0);
                    enc.set_buffer(6, Some(&hop_buf), 0);
                    enc.dispatch_threads(via_threads, tg);
                    enc.end_encoding();
                }
                run_command_buffer(cmd)?;
                if read_u32_buffer(&flag_buf, 1)[0] == 0 {
                    break;
                }
            }

            let length_buf = new_u32_buffer(dev, &vec![0u32; fields.len()]);
            let route_cost_buf = new_u32_buffer(dev, &vec![Cost::MAX; fields.len()]);
            let field_threads = MTLSize::new(fields.len() as u64, 1, 1);
            let cmd = ctx.queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.ragged_path_lengths);
                enc.set_buffer(0, Some(&dist_buf), 0);
                enc.set_buffer(1, Some(&hop_buf), 0);
                enc.set_buffer(2, Some(&cost_buf), 0);
                enc.set_buffer(3, Some(&x_buf), 0);
                enc.set_buffer(4, Some(&y_buf), 0);
                enc.set_buffer(5, Some(&via_buf), 0);
                enc.set_buffer(6, Some(&via_allowed_buf), 0);
                enc.set_buffer(7, Some(&field_buf), 0);
                enc.set_buffer(8, Some(&length_buf), 0);
                enc.set_buffer(9, Some(&route_cost_buf), 0);
                enc.dispatch_threads(field_threads, tg);
                enc.end_encoding();
            }
            run_command_buffer(cmd)?;

            let lengths = read_u32_buffer(&length_buf, fields.len());
            let search_costs = read_u32_buffer(&route_cost_buf, fields.len());
            if lengths.contains(&u32::MAX) {
                return Err(RouterError::BackendUnavailable(
                    "ragged Metal path reconstruction failed".into(),
                ));
            }
            let mut offsets = Vec::with_capacity(fields.len());
            let mut path_cells = 0usize;
            for (&len, field) in lengths.iter().zip(fields) {
                let field_cells = (field.w as usize) * (field.h as usize) * (field.layers as usize);
                if len as usize > field_cells {
                    return Err(RouterError::BackendUnavailable(
                        "ragged Metal returned an invalid path length".into(),
                    ));
                }
                offsets.push(u32::try_from(path_cells).map_err(|_| {
                    RouterError::BackendUnavailable("ragged Metal paths are too large".into())
                })?);
                path_cells = path_cells.checked_add(len as usize).ok_or_else(|| {
                    RouterError::BackendUnavailable("ragged Metal paths are too large".into())
                })?;
            }
            if path_cells > total {
                return Err(RouterError::BackendUnavailable(
                    "ragged Metal compact paths exceed packed fields".into(),
                ));
            }

            let flat_paths = if path_cells == 0 {
                Vec::new()
            } else {
                validate_buffer_len(dev, path_cells, "ragged compact paths")?;
                let offset_buf = new_u32_buffer(dev, &offsets);
                let path_buf = new_u32_buffer(dev, &vec![u32::MAX; path_cells]);
                let meta = RaggedMeta {
                    fields: fields.len() as u32,
                    global_w: dims.w,
                    global_h: dims.h,
                    reserved: 0,
                };
                let meta_buf = new_buffer(dev, std::slice::from_ref(&meta));
                let cmd = ctx.queue.new_command_buffer();
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&ctx.ragged_write_paths);
                    enc.set_buffer(0, Some(&dist_buf), 0);
                    enc.set_buffer(1, Some(&hop_buf), 0);
                    enc.set_buffer(2, Some(&cost_buf), 0);
                    enc.set_buffer(3, Some(&x_buf), 0);
                    enc.set_buffer(4, Some(&y_buf), 0);
                    enc.set_buffer(5, Some(&via_buf), 0);
                    enc.set_buffer(6, Some(&via_allowed_buf), 0);
                    enc.set_buffer(7, Some(&field_buf), 0);
                    enc.set_buffer(8, Some(&length_buf), 0);
                    enc.set_buffer(9, Some(&offset_buf), 0);
                    enc.set_buffer(10, Some(&path_buf), 0);
                    enc.set_buffer(11, Some(&meta_buf), 0);
                    enc.dispatch_threads(field_threads, tg);
                    enc.end_encoding();
                }
                run_command_buffer(cmd)?;
                read_u32_buffer(&path_buf, path_cells)
            };

            let mut paths = Vec::with_capacity(fields.len());
            for ((&len, &offset), field) in lengths.iter().zip(&offsets).zip(fields) {
                if len == 0 {
                    paths.push(None);
                    continue;
                }
                let start = offset as usize;
                let end = start + len as usize;
                let path = flat_paths[start..end].to_vec();
                let packed_to_global = |packed: u32| {
                    let rel = packed - field.cell_offset;
                    let plane = field.w * field.h;
                    let layer = rel / plane;
                    let xy = rel % plane;
                    let x = xy % field.w;
                    let y = xy / field.w;
                    dims.idx3(field.x0 + x, field.y0 + y, layer)
                };
                if path.first().copied() != Some(packed_to_global(field.src))
                    || path.last().copied() != Some(packed_to_global(field.dst))
                    || path.iter().any(|&cell| !dims.contains(cell))
                    || search_costs[paths.len()] == Cost::MAX
                {
                    return Err(RouterError::BackendUnavailable(
                        "ragged Metal produced an invalid compact path".into(),
                    ));
                }
                paths.push(Some(path));
            }

            Ok(RaggedPathBatch {
                paths,
                search_costs,
                packed_cells: total,
                readback_cells: fields.len().saturating_mul(2).saturating_add(path_cells),
            })
        })
    })
}

/// M4 single-field compatibility wrapper.
pub fn sweep_field(grid: &Grid, src: CellIdx) -> Result<Vec<Cost>, RouterError> {
    Ok(sweep_fields_flat(grid, &[src], &grid.cost, false)?.dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_aggregate_cap_has_allocation_free_exact_boundary() {
        let fixed =
            RAGGED_FIXED_OVERHEAD_BYTES + 2 * RAGGED_LINE_PEAK_BYTES + RAGGED_FIELD_PEAK_BYTES;
        let safe_cells =
            (MAX_RAGGED_AGGREGATE_WORKING_SET_BYTES - fixed) / RAGGED_PACKED_CELL_PEAK_BYTES;
        let boundary_edge_elements = (MAX_RAGGED_AGGREGATE_WORKING_SET_BYTES
            - fixed
            - safe_cells * RAGGED_PACKED_CELL_PEAK_BYTES)
            / RAGGED_EDGE_PEAK_BYTES;
        let shape = |packed_cells| RaggedBatchShape {
            fields: 1,
            packed_cells,
            row_lines: 1,
            col_lines: 1,
            max_plane: packed_cells,
            max_field_cells: packed_cells,
        };

        let safe = shape(safe_cells);
        validate_ragged_index_shape(safe).unwrap();
        assert_eq!(
            ragged_aggregate_working_set_bytes(safe, boundary_edge_elements, 0, 0).unwrap(),
            MAX_RAGGED_AGGREGATE_WORKING_SET_BYTES
        );
        assert!(validate_ragged_aggregate_working_set(safe, boundary_edge_elements, 0, 0).is_ok());

        let over = shape(safe_cells + 1);
        validate_ragged_index_shape(over).unwrap();
        let error =
            validate_ragged_aggregate_working_set(over, boundary_edge_elements, 0, 0).unwrap_err();
        assert!(matches!(
            error,
            RouterError::BackendUnavailable(message)
                if message.contains("aggregate working set needs")
        ));
    }

    #[test]
    fn ragged_aggregate_cap_preserves_extreme_normal_chunk_without_allocating() {
        // The descriptor-heavy 1x1xN extreme has two line descriptors and one via
        // gap per packed cell. Even this worst normal 16M-cell shape must continue
        // to fit; only the formerly unbounded oversized-single-field path is gated.
        let shape = RaggedBatchShape {
            fields: 1,
            packed_cells: super::super::MAX_BATCH_CELLS,
            row_lines: super::super::MAX_BATCH_CELLS,
            col_lines: super::super::MAX_BATCH_CELLS,
            max_plane: 1,
            max_field_cells: super::super::MAX_BATCH_CELLS,
        };
        validate_ragged_index_shape(shape).unwrap();
        validate_ragged_aggregate_working_set(shape, 0, 0, super::super::MAX_BATCH_CELLS - 1)
            .unwrap();
    }

    #[test]
    fn command_status_requires_successful_completion() {
        assert!(command_status_result(MTLCommandBufferStatus::Completed).is_ok());
        for status in [
            MTLCommandBufferStatus::NotEnqueued,
            MTLCommandBufferStatus::Enqueued,
            MTLCommandBufferStatus::Committed,
            MTLCommandBufferStatus::Scheduled,
            MTLCommandBufferStatus::Error,
        ] {
            assert!(matches!(
                command_status_result(status),
                Err(RouterError::BackendUnavailable(_))
            ));
        }
    }

    #[test]
    fn edge_batch_preflight_checks_every_distinct_buffer_size() {
        autoreleasepool(|| {
            with_metal_ctx(|ctx| {
                let too_many = (ctx.device.max_buffer_length() as usize / 4).saturating_add(1);
                let error_for = |cost, x, y, via, allowed| {
                    validate_edge_batch_buffers(&ctx.device, 1, cost, x, y, via, allowed)
                        .unwrap_err()
                        .to_string()
                };

                assert!(
                    validate_edge_batch_buffers(&ctx.device, too_many, 1, 1, 1, 1, 1)
                        .unwrap_err()
                        .to_string()
                        .contains("distance")
                );
                assert!(error_for(too_many, 1, 1, 1, 1).contains("enter weights"));
                assert!(error_for(1, too_many, 1, 1, 1).contains("x edges"));
                assert!(error_for(1, 1, too_many, 1, 1).contains("y edges"));
                assert!(error_for(1, 1, 1, too_many, 1).contains("via edges"));
                assert!(error_for(1, 1, 1, 1, too_many).contains("via permissions"));
                Ok(())
            })
        })
        .unwrap();
    }

    #[test]
    fn ragged_preflight_checks_descriptor_buffers_without_allocating_them() {
        autoreleasepool(|| {
            with_metal_ctx(|ctx| {
                let max_words = ctx.device.max_buffer_length() as usize / 4;
                let base = RaggedBatchShape {
                    fields: 1,
                    packed_cells: 1,
                    row_lines: 1,
                    col_lines: 1,
                    max_plane: 1,
                    max_field_cells: 1,
                };

                let mut fields = base;
                fields.fields = max_words / 8 + 1;
                assert!(
                    validate_ragged_batch_buffers(&ctx.device, fields, 1, 1, 1, 1)
                        .unwrap_err()
                        .to_string()
                        .contains("fields")
                );

                let mut rows = base;
                rows.row_lines = max_words / 4 + 1;
                assert!(validate_ragged_batch_buffers(&ctx.device, rows, 1, 1, 1, 1)
                    .unwrap_err()
                    .to_string()
                    .contains("rows"));

                let mut cols = base;
                cols.col_lines = max_words / 4 + 1;
                assert!(validate_ragged_batch_buffers(&ctx.device, cols, 1, 1, 1, 1)
                    .unwrap_err()
                    .to_string()
                    .contains("columns"));
                Ok(())
            })
        })
        .unwrap();
    }
}
