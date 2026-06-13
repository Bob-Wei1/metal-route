//! `mr-srj` — tscircuit SimpleRouteJson I/O for metalroute.
//!
//! Two responsibilities:
//!
//! 1. **Input (B3): rasterise.** Parse a [`SimpleRouteJson`] problem (continuous
//!    geometry) and convert it into a cell-space [`RasterizedProblem`]: a
//!    [`mr_core::Grid`] of obstacles, a list of two-point [`mr_core::NetEndpoints`],
//!    and the [`Mapping`] that ties continuous coordinates to grid cells.
//! 2. **Output (B4): de-rasterise.** Turn a routed [`mr_core::BoardRoute`] back
//!    into a tscircuit solution soup ([`PcbTrace`] objects) using the same
//!    [`Mapping`] to recover continuous coordinates from cell centres.
//!
//! ## Coordinate mapping ([`Mapping`])
//!
//! The grid origin is the bounds minimum `(minX, minY)`. Each cell is a square of
//! side `resolution` in continuous units. Grid dimensions are
//! `ceil((maxX-minX)/res) × ceil((maxY-minY)/res)`, clamped to a minimum of 1×1 so
//! degenerate / zero-area bounds still produce a usable single-cell grid. A cell's
//! continuous *centre* is `(originX + (x+0.5)·res, originY + (y+0.5)·res)`.
//!
//! ## Obstacles
//!
//! A `rect` obstacle centred at `(cx, cy)` with `width w` / `height h` covers the
//! continuous box `[cx-w/2, cx+w/2] × [cy-h/2, cy+h/2]`. Every grid cell whose own
//! square area overlaps that box is marked an obstacle (area-overlap, not
//! centre-in-rect), via [`mr_grid::GridBuilder::mark_rect`].
//!
//! ## k-point connection decomposition (plan R8)
//!
//! The router contract is strictly two-point. A connection with `k` points in
//! `pointsToConnect` is decomposed into `k-1` consecutive two-point
//! [`mr_core::NetEndpoints`] in chain order (point[0]→point[1], point[1]→point[2],
//! …). When `k > 2` the resulting nets are named `"<conn.name>#0"`,
//! `"<conn.name>#1"`, …; a plain two-point connection keeps the bare `conn.name`.
//! Connections with fewer than two points produce no nets.

use std::collections::HashMap;

use mr_core::{BoardRoute, CellIdx, Dims, Grid, LayerMap, NetEndpoints};
use mr_grid::GridBuilder;
use serde::{Deserialize, Serialize};

/// A 2-D point in continuous tscircuit coordinates.
///
/// The real harness attaches extra fields (`layer`, `pcb_port_id`); `layer` is
/// captured for completeness and any other unknown fields are ignored by serde.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub layer: Option<String>,
}

/// Axis-aligned problem bounds in continuous coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

/// A single rectangular obstacle (the only obstacle shape tscircuit emits here).
///
/// `kind` mirrors the JSON `type` field (always `"rect"` today) but is kept as a
/// `String` so unknown shapes still round-trip rather than failing to parse.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Obstacle {
    #[serde(rename = "type")]
    pub kind: String,
    pub center: Point,
    pub width: f64,
    pub height: f64,
    /// Layers this obstacle sits on (e.g. `["top"]`). Empty if unspecified.
    #[serde(default)]
    pub layers: Vec<String>,
    /// IDs of the pads/elements this obstacle is electrically connected to
    /// (the harness emits `connectedTo`). Empty if unspecified.
    #[serde(default, rename = "connectedTo")]
    pub connected_to: Vec<String>,
}

/// One named connection: a list of pads/points that must all be electrically
/// joined. Decomposed into chained two-point nets at rasterisation time (R8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Connection {
    pub name: String,
    pub points_to_connect: Vec<Point>,
}

/// A tscircuit SimpleRouteJson routing problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimpleRouteJson {
    pub layer_count: u32,
    /// Minimum trace width in continuous units (the harness emits `minTraceWidth`).
    /// Drives both the emitted wire width and the rasterisation resolution.
    #[serde(default)]
    pub min_trace_width: Option<f64>,
    /// Minimum copper-to-copper clearance in continuous units. Like
    /// [`Self::min_trace_width`], this is parsed from the DSN `(rule (clearance N))`
    /// rule (the harness emits `minClearance`) and feeds the design-rule check.
    #[serde(default)]
    pub min_clearance: Option<f64>,
    #[serde(default)]
    pub obstacles: Vec<Obstacle>,
    #[serde(default)]
    pub connections: Vec<Connection>,
    pub bounds: Bounds,
}

/// Continuous ↔ grid-cell coordinate conversion. See module docs for the rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mapping {
    /// Continuous x of the grid origin = `bounds.min_x`.
    pub origin_x: f64,
    /// Continuous y of the grid origin = `bounds.min_y`.
    pub origin_y: f64,
    /// Side length of one (square) cell in continuous units.
    pub resolution: f64,
    /// Grid dimensions (≥ 1×1).
    pub dims: Dims,
}

impl Mapping {
    /// Build a single-layer mapping for `bounds` at the given `resolution`.
    ///
    /// Dimensions are `ceil(span / resolution)`, floored at 1 on each axis so a
    /// zero- or negative-area bounds still yields a 1×1 grid. Use
    /// [`Mapping::with_layers`] for a multi-layer board; this remains single-layer
    /// so every existing 2D caller is byte-identical.
    pub fn new(bounds: &Bounds, resolution: f64) -> Self {
        Self::with_layers(bounds, resolution, 1)
    }

    /// Build a mapping for `bounds` at `resolution` over `layers` stacked planes.
    /// The planar (x, y) sizing is identical to [`Mapping::new`]; only the layer
    /// axis grows. `layers == 1` is byte-identical to [`Mapping::new`].
    pub fn with_layers(bounds: &Bounds, resolution: f64, layers: u32) -> Self {
        let span_x = (bounds.max_x - bounds.min_x).max(0.0);
        let span_y = (bounds.max_y - bounds.min_y).max(0.0);
        let w = ((span_x / resolution).ceil() as u32).max(1);
        let h = ((span_y / resolution).ceil() as u32).max(1);
        Self {
            origin_x: bounds.min_x,
            origin_y: bounds.min_y,
            resolution,
            dims: Dims::with_layers(w, h, layers),
        }
    }

    /// Continuous centre of cell `cell`. The layer is ignored — every layer shares
    /// the same planar geometry — so a cell and its via-neighbour on another layer
    /// map to the same continuous `(x, y)`.
    pub fn cell_center(&self, cell: CellIdx) -> (f64, f64) {
        let (x, y) = self.dims.xy(cell);
        (
            self.origin_x + (x as f64 + 0.5) * self.resolution,
            self.origin_y + (y as f64 + 0.5) * self.resolution,
        )
    }

    /// Layer of `cell` (0 == top). Always 0 for a single-layer mapping.
    pub fn cell_layer(&self, cell: CellIdx) -> u32 {
        self.dims.layer_of(cell)
    }

    /// Cell on layer 0 containing continuous point `(x, y)`, clamped into the grid.
    /// See [`Mapping::point_to_cell_layer`] to target a specific layer.
    pub fn point_to_cell(&self, point: (f64, f64)) -> CellIdx {
        self.point_to_cell_layer(point, 0)
    }

    /// Cell on `layer` containing continuous point `(x, y)`, clamped into the grid.
    /// The layer is clamped to the last valid layer.
    pub fn point_to_cell_layer(&self, point: (f64, f64), layer: u32) -> CellIdx {
        let (cx, cy) = self.point_to_xy(point);
        let l = layer.min(self.dims.layers.saturating_sub(1));
        self.dims.idx3(cx, cy, l)
    }

    /// Cell `(x, y)` containing a continuous point, clamped into the grid.
    fn point_to_xy(&self, point: (f64, f64)) -> (u32, u32) {
        let fx = ((point.0 - self.origin_x) / self.resolution).floor();
        let fy = ((point.1 - self.origin_y) / self.resolution).floor();
        (clamp_index(fx, self.dims.w), clamp_index(fy, self.dims.h))
    }
}

/// Clamp a (possibly out-of-range / non-finite) floating cell index into
/// `[0, extent-1]`.
fn clamp_index(f: f64, extent: u32) -> u32 {
    if !f.is_finite() || f < 0.0 {
        return 0;
    }
    let max = extent.saturating_sub(1);
    (f as u32).min(max)
}

/// The cell-space form of a [`SimpleRouteJson`] problem: the obstacle grid, the
/// two-point nets to route, and the [`Mapping`] needed to convert results back.
#[derive(Debug, Clone)]
pub struct RasterizedProblem {
    pub grid: Grid,
    pub nets: Vec<NetEndpoints>,
    pub mapping: Mapping,
    /// The board's ordered layer names (index ↔ name). For a single-layer problem
    /// this is `["top"]`. Pass it to [`to_solution_layered`] so emitted vertices
    /// and vias carry the right layer names.
    pub layers: LayerMap,
    /// Exact continuous `(x, y)` of every connection endpoint, keyed by the cell
    /// it rasterised to. Used by [`to_solution`] to snap each trace's first/last
    /// vertex back to the exact port coordinate so connectivity checks
    /// (`distance < 0.01`) pass. If two endpoints collide into one cell only one
    /// survives — the resolution is chosen fine enough that this does not happen.
    pub pin_points: HashMap<CellIdx, (f64, f64)>,
}

/// (B3) Rasterise a continuous tscircuit problem into a cell-space problem.
///
/// See the module docs for the dimension, obstacle-overlap, and k-point
/// decomposition rules.
pub fn rasterize(srj: &SimpleRouteJson, resolution: f64) -> RasterizedProblem {
    // The board's layer axis. `layer_count == 1` yields `["top"]`, so every
    // single-layer construction below collapses onto layer 0 and is byte-identical
    // to the pre-layers path.
    rasterize_with_layers(srj, resolution, LayerMap::standard(srj.layer_count))
}

/// (B3) Rasterise with an explicit [`LayerMap`] — use this when the layer *names*
/// are not the standard `top`/`inner_N`/`bottom` (e.g. a Specctra DSN's `F.Cu` /
/// `B.Cu` stackup), so each [`Point`]/[`Obstacle`]'s named layer resolves to the
/// right grid plane instead of collapsing onto layer 0. The grid is built with
/// `layers.len()` planes; `rasterize` is the standard-naming special case.
pub fn rasterize_with_layers(
    srj: &SimpleRouteJson,
    resolution: f64,
    layers: LayerMap,
) -> RasterizedProblem {
    let layer_count = layers.len();
    let mapping = Mapping::with_layers(&srj.bounds, resolution, layer_count);
    let mut builder = GridBuilder::new(mapping.dims, 1);

    // Collect every connection endpoint as `(x, y, layer)`. These are the pad
    // centres we must connect; the resolved layer is the named `Point.layer`
    // (default "top" / layer 0). The pad each endpoint sits in IS marked as an
    // obstacle in the base grid (correct DRC model: all pads are obstacles); the
    // router later unmasks each net's own pad cells via `passable_pads` so a net
    // may escape its own pads but cannot run through a foreign net's pad.
    let endpoints: Vec<(f64, f64, u32)> = srj
        .connections
        .iter()
        .flat_map(|conn| {
            conn.points_to_connect
                .iter()
                .map(|p| (p.x, p.y, point_layer(p, &layers)))
        })
        .collect();

    for obs in &srj.obstacles {
        // Continuous box covered by the rect.
        let min_x = obs.center.x - obs.width / 2.0;
        let max_x = obs.center.x + obs.width / 2.0;
        let min_y = obs.center.y - obs.height / 2.0;
        let max_y = obs.center.y + obs.height / 2.0;

        // Lower cell holds the box minimum; upper cell is the last cell whose
        // square overlaps the box. A box edge that lands exactly on a cell
        // boundary does not spill into the next cell (half-open cells).
        let (x0, y0) = mapping.point_to_xy((min_x, min_y));
        let x1 = cell_upper(max_x, mapping.origin_x, mapping.resolution, mapping.dims.w);
        let y1 = cell_upper(max_y, mapping.origin_y, mapping.resolution, mapping.dims.h);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        // Place the obstacle on each layer it names; an empty or unknown layer
        // list falls back to ALL layers (so a single-layer "top" fixture blocks
        // layer 0 exactly as before).
        for layer in obstacle_layers(obs, &layers) {
            builder.mark_rect_layer(x0, y0, x1, y1, layer);
        }
    }

    // Record each endpoint's exact continuous coordinate so `to_solution` can
    // snap traces back to the port, keyed by the endpoint's own (layered) cell.
    // Endpoint cells stay obstacles in the base grid; the router unmasks each
    // net's own pad cells per net.
    let mut pin_points: HashMap<CellIdx, (f64, f64)> = HashMap::new();
    for &(px, py, layer) in &endpoints {
        let cell = mapping.point_to_cell_layer((px, py), layer);
        pin_points.insert(cell, (px, py));
    }

    // Via keepout is now enforced in the router's legalization fold (via
    // `ViaModel.keepout`, see `mr_cpu::NegotiatedRouter` / `stamp_owner`), not at
    // rasterise time — committed vias reserve their keepout halo there, so nothing
    // extra is stamped on the base grid here.

    let grid = builder.build();
    let nets = decompose_connections(&srj.connections, &mapping, &srj.obstacles, &layers);

    RasterizedProblem {
        grid,
        nets,
        mapping,
        layers,
        pin_points,
    }
}

/// Resolve a [`Point`]'s grid layer from its optional named `layer`, defaulting to
/// layer 0 ("top") when absent or unknown.
fn point_layer(p: &Point, layers: &LayerMap) -> u32 {
    p.layer
        .as_deref()
        .and_then(|name| layers.index_of(name))
        .unwrap_or(0)
}

/// The grid layers an [`Obstacle`] occupies. Each named layer maps via the
/// [`LayerMap`]; an empty list, or a list of only unknown names, falls back to
/// ALL layers. Returned ascending and deduplicated.
fn obstacle_layers(obs: &Obstacle, layers: &LayerMap) -> Vec<u32> {
    let mut out: Vec<u32> = obs
        .layers
        .iter()
        .filter_map(|name| layers.index_of(name))
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        (0..layers.len()).collect()
    } else {
        out
    }
}

/// All grid cells covered by obstacle rect(s) that contain the continuous point
/// `point`, plus the point's own rasterised cell. This is the set of pad cells a
/// net owning this endpoint is allowed to traverse. The cells are returned in
/// ascending [`CellIdx`] order, deduplicated, for deterministic serialisation.
fn pad_cells_for_point(
    point: (f64, f64),
    layer: u32,
    mapping: &Mapping,
    obstacles: &[Obstacle],
) -> Vec<CellIdx> {
    let (px, py) = point;
    let mut cells: Vec<CellIdx> = Vec::new();
    for obs in obstacles {
        let min_x = obs.center.x - obs.width / 2.0;
        let max_x = obs.center.x + obs.width / 2.0;
        let min_y = obs.center.y - obs.height / 2.0;
        let max_y = obs.center.y + obs.height / 2.0;
        if px < min_x || px > max_x || py < min_y || py > max_y {
            continue;
        }
        // Same cell-range logic as the obstacle marking loop.
        let (x0, y0) = mapping.point_to_xy((min_x, min_y));
        let x1 = cell_upper(max_x, mapping.origin_x, mapping.resolution, mapping.dims.w);
        let y1 = cell_upper(max_y, mapping.origin_y, mapping.resolution, mapping.dims.h);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        // Unmask the pad only on the endpoint's own layer: the net escapes through
        // its pad on the layer it connects, not on every layer the pad spans.
        for y in y0..=y1 {
            for x in x0..=x1 {
                cells.push(mapping.dims.idx3(x, y, layer));
            }
        }
    }
    // Always include the endpoint's own rasterised (layered) cell.
    cells.push(mapping.point_to_cell_layer((px, py), layer));
    cells.sort_unstable();
    cells.dedup();
    cells
}

/// Upper (inclusive) cell index touched by a continuous box that ends at `hi`:
/// `ceil((hi-origin)/res) - 1`, clamped into `[0, extent-1]`.
fn cell_upper(hi: f64, origin: f64, res: f64, extent: u32) -> u32 {
    let edge = ((hi - origin) / res).ceil() - 1.0;
    clamp_index(edge, extent)
}

/// Decompose every connection into chained two-point nets (plan R8).
///
/// Each emitted net's `passable_pads` is the union of the own-pad cells of its
/// src and dst points: the cells of every obstacle rect that contains that
/// endpoint, plus the endpoint's own cell. This lets the router unmask exactly
/// this net's pads while keeping every foreign pad an obstacle.
fn decompose_connections(
    connections: &[Connection],
    mapping: &Mapping,
    obstacles: &[Obstacle],
    layers: &LayerMap,
) -> Vec<NetEndpoints> {
    let mut nets = Vec::new();
    for conn in connections {
        let pts = &conn.points_to_connect;
        if pts.len() < 2 {
            continue;
        }
        let segments = pts.len() - 1;
        for (seg, win) in pts.windows(2).enumerate() {
            let src_layer = point_layer(&win[0], layers);
            let dst_layer = point_layer(&win[1], layers);
            let src = mapping.point_to_cell_layer((win[0].x, win[0].y), src_layer);
            let dst = mapping.point_to_cell_layer((win[1].x, win[1].y), dst_layer);
            let net = if segments == 1 {
                conn.name.clone()
            } else {
                format!("{}#{}", conn.name, seg)
            };
            // Union of the src and dst pad cells (each on its endpoint's layer),
            // sorted + deduped.
            let mut passable_pads =
                pad_cells_for_point((win[0].x, win[0].y), src_layer, mapping, obstacles);
            passable_pads.extend(pad_cells_for_point(
                (win[1].x, win[1].y),
                dst_layer,
                mapping,
                obstacles,
            ));
            passable_pads.sort_unstable();
            passable_pads.dedup();
            nets.push(NetEndpoints {
                net,
                src,
                dst,
                passable_pads,
            });
        }
    }
    nets
}

// ---------------------------------------------------------------------------
// Output soup (B4)
// ---------------------------------------------------------------------------

/// One element of a routed trace: either a wire segment vertex or a via.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "route_type", rename_all = "snake_case")]
pub enum RoutePoint {
    Wire {
        x: f64,
        y: f64,
        width: f64,
        layer: String,
    },
    Via {
        x: f64,
        y: f64,
        from_layer: String,
        to_layer: String,
    },
}

/// A routed trace in tscircuit solution form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcbTrace {
    #[serde(rename = "type")]
    pub kind: String,
    pub route: Vec<RoutePoint>,
}

impl PcbTrace {
    /// Construct a wire-only `pcb_trace` from the given route points.
    pub fn new(route: Vec<RoutePoint>) -> Self {
        Self {
            kind: "pcb_trace".to_string(),
            route,
        }
    }
}

/// (B4) De-rasterise a routed board into a *single-layer* tscircuit solution soup.
///
/// Backward-compatible entry point: every vertex is emitted on `layer` and no vias
/// are produced. Prefer [`to_solution_layered`] for multi-layer boards — this just
/// delegates to it with a one-element [`LayerMap`] named `layer`, so a single-layer
/// path produces byte-identical output to before vias existed.
pub fn to_solution(
    board: &BoardRoute,
    mapping: &Mapping,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    trace_width: f64,
    layer: &str,
) -> Vec<PcbTrace> {
    let layers = LayerMap::from_names(vec![layer.to_string()]);
    to_solution_layered(board, mapping, pin_points, trace_width, &layers)
}

/// (B4) De-rasterise a routed board into a tscircuit solution soup, emitting vias
/// wherever a path changes layers.
///
/// Each [`mr_core::RouteResult`] becomes one [`PcbTrace`]. Walking the path:
///
/// * Consecutive cells with the *same* planar `(x, y)` but a *different* layer form
///   a vertical (via) run. A maximal contiguous vertical run collapses into ONE
///   [`RoutePoint::Via`] from the run's first layer to its last, named via the
///   [`LayerMap`].
/// * Every planar move emits a [`RoutePoint::Wire`] vertex on its cell's layer.
///
/// The first and last *planar* vertex of every trace are snapped to the exact port
/// coordinate via `pin_points` (keyed by the endpoint cell) so the harness'
/// connectivity check (`distance(port, vertex) < 0.01`) matches. Interior vertices
/// stay at cell centres; an endpoint cell missing from `pin_points` falls back to
/// its cell centre.
///
/// A purely single-layer path emits only `Wire` points on layer 0's name and no
/// vias — byte-identical to [`to_solution`]'s historical output.
pub fn to_solution_layered(
    board: &BoardRoute,
    mapping: &Mapping,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    trace_width: f64,
    layers: &LayerMap,
) -> Vec<PcbTrace> {
    board
        .results
        .iter()
        .map(|result| {
            let route = trace_route(result, mapping, pin_points, trace_width, layers);
            PcbTrace::new(route)
        })
        .collect()
}

/// Build the [`RoutePoint`] route for one path, collapsing vertical (via) runs.
fn trace_route(
    result: &mr_core::RouteResult,
    mapping: &Mapping,
    pin_points: &HashMap<CellIdx, (f64, f64)>,
    trace_width: f64,
    layers: &LayerMap,
) -> Vec<RoutePoint> {
    let path = &result.path;
    let last = path.len().saturating_sub(1);
    let mut route: Vec<RoutePoint> = Vec::with_capacity(path.len());
    let mut i = 0;
    while i < path.len() {
        let cell = path[i];
        // Emit a Wire vertex for this cell on its layer. Snap the trace's first
        // and last planar vertex to the exact port coordinate; interior vertices
        // stay at the cell centre.
        let (x, y) = if i == 0 || i == last {
            pin_points
                .get(&cell)
                .copied()
                .unwrap_or_else(|| mapping.cell_center(cell))
        } else {
            mapping.cell_center(cell)
        };
        route.push(RoutePoint::Wire {
            x,
            y,
            width: trace_width,
            layer: layers.name(mapping.dims.layer_of(cell)).to_string(),
        });

        // Detect a maximal vertical run starting at `i`: consecutive cells sharing
        // this cell's planar (x, y) but changing layer. Collapse the whole run into
        // ONE via from the wire we just emitted to the run's final layer, and skip
        // straight to that final cell (its same-(x,y) intermediates carry no wires).
        let (cx, cy) = mapping.dims.xy(cell);
        let mut j = i;
        while j + 1 < path.len() {
            let (nx, ny) = mapping.dims.xy(path[j + 1]);
            if nx == cx && ny == cy && path[j + 1] != path[j] {
                j += 1;
            } else {
                break;
            }
        }
        if j > i {
            let (vx, vy) = mapping.cell_center(cell);
            let from_layer = mapping.dims.layer_of(path[i]);
            let to_layer = mapping.dims.layer_of(path[j]);
            route.push(RoutePoint::Via {
                x: vx,
                y: vy,
                from_layer: layers.name(from_layer).to_string(),
                to_layer: layers.name(to_layer).to_string(),
            });
            // Resume after the run's last cell; that cell's wire is the via's
            // landing and is not re-emitted.
            i = j + 1;
        } else {
            i += 1;
        }
    }
    route
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::RouteResult;

    /// A small but realistic problem: 10×10 board, one rect obstacle, two
    /// connections (one 2-point, one 3-point).
    const SAMPLE: &str = r#"{
        "layerCount": 2,
        "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
        "obstacles": [
            { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
        ],
        "connections": [
            { "name": "VCC", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 1} ] },
            { "name": "GND", "pointsToConnect": [ {"x": 1, "y": 9}, {"x": 5, "y": 9}, {"x": 9, "y": 9} ] }
        ]
    }"#;

    #[test]
    fn deserializes_sample() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(srj.layer_count, 2);
        assert_eq!(srj.obstacles.len(), 1);
        assert_eq!(srj.obstacles[0].kind, "rect");
        assert_eq!(srj.connections.len(), 2);
        assert_eq!(srj.connections[1].points_to_connect.len(), 3);
        assert_eq!(srj.bounds.max_x, 10.0);
    }

    #[test]
    fn rasterize_dims_and_net_count() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        // 10/1 = 10 cells per axis; SAMPLE declares layerCount 2.
        assert_eq!(prob.mapping.dims, Dims::with_layers(10, 10, 2));
        // VCC -> 1 net, GND (3 points) -> 2 nets. Total 3.
        assert_eq!(prob.nets.len(), 3);
    }

    #[test]
    fn rasterize_marks_expected_obstacle_cells() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        // Rect spans [4,6]x[4,6] continuous. At res=1 that's cells x=4..=5,
        // y=4..=5 (cell 4 = [4,5), cell 5 = [5,6); cell 6 = [6,7) does NOT
        // overlap a box that ends exactly at 6).
        assert!(prob.grid.is_obstacle(d.idx(4, 4)));
        assert!(prob.grid.is_obstacle(d.idx(5, 4)));
        assert!(prob.grid.is_obstacle(d.idx(4, 5)));
        assert!(prob.grid.is_obstacle(d.idx(5, 5)));
        // Boundaries / outside stay passable.
        assert!(!prob.grid.is_obstacle(d.idx(6, 5)));
        assert!(!prob.grid.is_obstacle(d.idx(3, 5)));
        assert!(!prob.grid.is_obstacle(d.idx(4, 6)));
        assert!(!prob.grid.is_obstacle(d.idx(0, 0)));
        // The obstacle names no layers, so it blocks BOTH of SAMPLE's 2 layers:
        // a 2x2 block on each layer = 8 cells. The same block is present on
        // layer 1 at the same planar (x, y).
        assert!(prob.grid.is_obstacle(d.idx3(4, 4, 1)));
        assert!(prob.grid.is_obstacle(d.idx3(5, 5, 1)));
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 4 * 2);
    }

    #[test]
    fn two_point_connection_keeps_bare_name() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        let vcc: Vec<_> = prob
            .nets
            .iter()
            .filter(|n| n.net.starts_with("VCC"))
            .collect();
        assert_eq!(vcc.len(), 1);
        assert_eq!(vcc[0].net, "VCC");
        // (1,1) -> cell (1,1); (9,1) -> cell (9,1) at res 1.
        let d = prob.mapping.dims;
        assert_eq!(vcc[0].src, d.idx(1, 1));
        assert_eq!(vcc[0].dst, d.idx(9, 1));
    }

    #[test]
    fn k3_connection_decomposes_into_two_chained_nets() {
        let srj: SimpleRouteJson = serde_json::from_str(SAMPLE).unwrap();
        let prob = rasterize(&srj, 1.0);
        let gnd: Vec<_> = prob
            .nets
            .iter()
            .filter(|n| n.net.starts_with("GND"))
            .collect();
        assert_eq!(gnd.len(), 2);
        assert_eq!(gnd[0].net, "GND#0");
        assert_eq!(gnd[1].net, "GND#1");
        let d = prob.mapping.dims;
        // Chain order: (1,9)->(5,9), then (5,9)->(9,9).
        assert_eq!(gnd[0].src, d.idx(1, 9));
        assert_eq!(gnd[0].dst, d.idx(5, 9));
        assert_eq!(gnd[1].src, d.idx(5, 9));
        assert_eq!(gnd[1].dst, d.idx(9, 9));
    }

    #[test]
    fn to_solution_emits_cell_center_wire_points() {
        // 10x10 grid at res 1, origin (0,0). 3-cell horizontal path on row 0.
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let path = vec![d.idx(0, 0), d.idx(1, 0), d.idx(2, 0)];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "VCC".into(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let pins = HashMap::new();
        let traces = to_solution(&board, &mapping, &pins, 0.2, "top");
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].kind, "pcb_trace");
        assert_eq!(traces[0].route.len(), 3);
        // Cell centres at res 1, origin 0: (0.5,0.5), (1.5,0.5), (2.5,0.5).
        let expected = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)];
        for (pt, &(ex, ey)) in traces[0].route.iter().zip(expected.iter()) {
            match pt {
                RoutePoint::Wire { x, y, width, layer } => {
                    assert_eq!(*x, ex);
                    assert_eq!(*y, ey);
                    assert_eq!(*width, 0.2);
                    assert_eq!(layer, "top");
                }
                _ => panic!("expected wire point"),
            }
        }
    }

    #[test]
    fn solution_serializes_with_expected_tags() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 4.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path: vec![d.idx(0, 0), d.idx(1, 0)],
                cost: 1,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let pins = HashMap::new();
        let traces = to_solution(&board, &mapping, &pins, 0.1, "top");
        let json = serde_json::to_string(&traces).unwrap();
        assert!(json.contains("\"type\":\"pcb_trace\""), "json: {json}");
        assert!(json.contains("\"route_type\":\"wire\""), "json: {json}");
    }

    #[test]
    fn via_route_point_serializes_snake_case() {
        // Guard the via variant's tag/field naming even though to_solution
        // doesn't emit vias yet.
        let v = RoutePoint::Via {
            x: 1.0,
            y: 2.0,
            from_layer: "top".into(),
            to_layer: "bottom".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("\"route_type\":\"via\""), "json: {json}");
        assert!(json.contains("\"from_layer\":\"top\""), "json: {json}");
        assert!(json.contains("\"to_layer\":\"bottom\""), "json: {json}");
    }

    /// Track B: an obstacle declared on "bottom" only blocks bottom-layer cells,
    /// never the top layer at the same (x, y).
    #[test]
    fn rasterize_obstacle_layer_isolated() {
        let blob = r#"{
            "layerCount": 2,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "layers": ["bottom"], "center": {"x": 5, "y": 5},
                  "width": 2, "height": 2 }
            ],
            "connections": []
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        assert_eq!(d.layers, 2);
        // Rect spans cells x=4..=5, y=4..=5 on the bottom layer (index 1).
        for y in 4..=5 {
            for x in 4..=5 {
                assert!(prob.grid.is_obstacle(d.idx3(x, y, 1)), "bottom blocked");
                assert!(!prob.grid.is_obstacle(d.idx3(x, y, 0)), "top free");
            }
        }
        // Exactly the 2x2 block on one layer.
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 4);
    }

    /// Track B: a src Point on "top" resolves to a layer-0 cell; on "bottom" it
    /// resolves to the bottom-layer cell at the same planar (x, y).
    #[test]
    fn rasterize_src_resolves_endpoint_layer() {
        let blob = r#"{
            "layerCount": 2,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [],
            "connections": [
                { "name": "a", "pointsToConnect": [
                    {"x": 1, "y": 1, "layer": "top"},
                    {"x": 9, "y": 9, "layer": "bottom"} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        let net = &prob.nets[0];
        assert_eq!(net.src, d.idx3(1, 1, 0), "top -> layer 0");
        assert_eq!(net.dst, d.idx3(9, 9, 1), "bottom -> layer 1");
    }

    /// Track B: an obstacle with no `layers` (or unknown names) blocks ALL layers,
    /// so a single-layer "top" fixture is unaffected.
    #[test]
    fn rasterize_obstacle_no_layers_blocks_all() {
        let blob = r#"{
            "layerCount": 3,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
            ],
            "connections": []
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        let d = prob.mapping.dims;
        for l in 0..3 {
            assert!(prob.grid.is_obstacle(d.idx3(4, 4, l)), "layer {l} blocked");
        }
        // 2x2 block on each of 3 layers.
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 4 * 3);
    }

    /// Track C: a 2-layer path with exactly one layer transition emits exactly one
    /// Via (with correct from/to names) and Wires on the right layers.
    #[test]
    fn to_solution_layered_emits_one_via() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::with_layers(&bounds, 1.0, 2);
        let d = mapping.dims;
        let layers = LayerMap::standard(2); // ["top","bottom"]
                                            // Move on top to (2,0), via down to bottom, then on bottom to (4,0).
        let path = vec![
            d.idx3(0, 0, 0),
            d.idx3(1, 0, 0),
            d.idx3(2, 0, 0),
            d.idx3(2, 0, 1), // via step
            d.idx3(3, 0, 1),
            d.idx3(4, 0, 1),
        ];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 5,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let pins = HashMap::new();
        let traces = to_solution_layered(&board, &mapping, &pins, 0.2, &layers);
        let route = &traces[0].route;
        // 3 top wires + 1 via + 2 bottom wires = 6.
        assert_eq!(route.len(), 6, "route: {route:?}");
        let vias: Vec<_> = route
            .iter()
            .filter(|p| matches!(p, RoutePoint::Via { .. }))
            .collect();
        assert_eq!(vias.len(), 1, "exactly one via");
        match vias[0] {
            RoutePoint::Via {
                x,
                y,
                from_layer,
                to_layer,
            } => {
                assert_eq!((*x, *y), (2.5, 0.5), "via at the shared cell centre");
                assert_eq!(from_layer, "top");
                assert_eq!(to_layer, "bottom");
            }
            _ => unreachable!(),
        }
        // Wires before the via are on "top"; after are on "bottom".
        let layers_of: Vec<&str> = route
            .iter()
            .filter_map(|p| match p {
                RoutePoint::Wire { layer, .. } => Some(layer.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(layers_of, vec!["top", "top", "top", "bottom", "bottom"]);
    }

    /// Track C: a multi-layer via run (top -> bottom across 3 layers) collapses to
    /// a single Via spanning the full run.
    #[test]
    fn to_solution_layered_collapses_multilayer_via_run() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 4.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::with_layers(&bounds, 1.0, 3);
        let d = mapping.dims;
        let layers = LayerMap::standard(3); // ["top","inner1","bottom"]
        let path = vec![
            d.idx3(0, 0, 0),
            d.idx3(0, 0, 1), // via run start
            d.idx3(0, 0, 2), // via run end (same x,y)
            d.idx3(1, 0, 2),
        ];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 4,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution_layered(&board, &mapping, &HashMap::new(), 0.1, &layers);
        let route = &traces[0].route;
        let vias: Vec<_> = route
            .iter()
            .filter_map(|p| match p {
                RoutePoint::Via {
                    from_layer,
                    to_layer,
                    ..
                } => Some((from_layer.as_str(), to_layer.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(vias, vec![("top", "bottom")], "one via spanning the run");
    }

    /// Track C: a single-layer path through `to_solution` is unchanged — all Wire,
    /// layer "top", no vias (guards byte-identical legacy output).
    #[test]
    fn to_solution_single_layer_unchanged() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let path = vec![d.idx(0, 0), d.idx(1, 0), d.idx(2, 0)];
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution(&board, &mapping, &HashMap::new(), 0.2, "top");
        assert_eq!(traces[0].route.len(), 3);
        assert!(traces[0]
            .route
            .iter()
            .all(|p| matches!(p, RoutePoint::Wire { layer, .. } if layer == "top")));
    }

    #[test]
    fn cell_center_and_point_to_cell_roundtrip() {
        let bounds = Bounds {
            min_x: -5.0,
            max_x: 5.0,
            min_y: 0.0,
            max_y: 4.0,
        };
        let mapping = Mapping::new(&bounds, 2.0);
        // span 10/2 = 5 wide, 4/2 = 2 tall.
        assert_eq!(mapping.dims, Dims::new(5, 2));
        for i in 0..mapping.dims.len() as u32 {
            let (x, y) = mapping.cell_center(i);
            assert_eq!(mapping.point_to_cell((x, y)), i);
        }
    }

    #[test]
    fn degenerate_bounds_yield_min_grid() {
        let bounds = Bounds {
            min_x: 3.0,
            max_x: 3.0,
            min_y: 3.0,
            max_y: 3.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        assert_eq!(mapping.dims, Dims::new(1, 1));
        // Any point clamps to cell 0.
        assert_eq!(mapping.point_to_cell((100.0, -100.0)), 0);
    }

    /// (a) Parse the real-harness shape: `minTraceWidth`, obstacle `layers` /
    /// `connectedTo`, and per-point `layer` / `pcb_port_id` (unknown fields are
    /// ignored, not rejected).
    #[test]
    fn parses_real_harness_fields() {
        let blob = r#"{
            "minTraceWidth": 0.1,
            "layerCount": 1,
            "bounds": { "minX": 0, "maxX": 5, "minY": 0, "maxY": 5 },
            "obstacles": [
                { "type": "rect", "layers": ["top"], "center": {"x": 2, "y": 2},
                  "width": 0.38, "height": 0.38, "connectedTo": ["pcb_smtpad_3"] }
            ],
            "connections": [
                { "name": "source_trace_0", "pointsToConnect": [
                    {"x": 1, "y": 1, "layer": "top", "pcb_port_id": "pcb_port_15"},
                    {"x": 4, "y": 4, "layer": "top", "pcb_port_id": "pcb_port_16"} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        assert_eq!(srj.min_trace_width, Some(0.1));
        assert_eq!(srj.obstacles[0].layers, vec!["top".to_string()]);
        assert_eq!(
            srj.obstacles[0].connected_to,
            vec!["pcb_smtpad_3".to_string()]
        );
        let p = &srj.connections[0].points_to_connect[0];
        assert_eq!(p.x, 1.0);
        assert_eq!(p.layer.as_deref(), Some("top"));
    }

    /// (b) An endpoint that sits at the centre of its own pad obstacle leaves the
    /// pad cells marked as obstacles in the BASE grid (correct DRC model), but the
    /// net's `passable_pads` carries the whole pad (and the src cell) so the router
    /// can unmask it. A decoy pad (no endpoint) is in no net's `passable_pads`.
    #[test]
    fn rasterize_pad_is_obstacle_but_in_own_net_passable_pads() {
        // A pad large enough to span >= 3x3 cells centred on the endpoint at
        // (5,5); at res 0.1 a 0.4mm pad spans several cells.
        let blob = r#"{
            "minTraceWidth": 0.1,
            "layerCount": 1,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 0.4, "height": 0.4,
                  "connectedTo": ["pcb_smtpad_0"] },
                { "type": "rect", "center": {"x": 2, "y": 2}, "width": 0.4, "height": 0.4,
                  "connectedTo": ["pcb_smtpad_decoy"] }
            ],
            "connections": [
                { "name": "n", "pointsToConnect": [ {"x": 5, "y": 5}, {"x": 9, "y": 9} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 0.1);
        let d = prob.mapping.dims;
        let src = prob.nets[0].src;
        // Base grid: the pad cell IS an obstacle now.
        assert!(
            prob.grid.is_obstacle(src),
            "pad cell is an obstacle in the base grid"
        );
        // The own pad's cells (and the src cell) are in this net's passable_pads.
        let pads = &prob.nets[0].passable_pads;
        assert!(pads.contains(&src), "src cell must be in passable_pads");
        let (cx, cy) = prob.mapping.point_to_xy((5.0, 5.0));
        // The own pad spans >= 3x3 cells; every one is in passable_pads.
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (x, y) = ((cx as i32 + dx) as u32, (cy as i32 + dy) as u32);
                let cell = d.idx(x, y);
                assert!(
                    prob.grid.is_obstacle(cell),
                    "own pad cell ({x},{y}) is an obstacle in base grid"
                );
                assert!(
                    pads.contains(&cell),
                    "own pad cell ({x},{y}) must be in passable_pads"
                );
            }
        }
        // The decoy pad at (2,2) contains no endpoint -> in NO net's
        // passable_pads, and remains an obstacle.
        let (dx2, dy2) = prob.mapping.point_to_xy((2.0, 2.0));
        let decoy = d.idx(dx2, dy2);
        assert!(prob.grid.is_obstacle(decoy), "decoy pad stays an obstacle");
        for net in &prob.nets {
            assert!(
                !net.passable_pads.contains(&decoy),
                "decoy pad must not be in any net's passable_pads"
            );
        }
    }

    /// A non-connected (decoy) pad must remain an obstacle even when another
    /// net's endpoint lives elsewhere.
    #[test]
    fn rasterize_keeps_decoy_pad_obstacle() {
        let blob = r#"{
            "layerCount": 1,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 1, "height": 1 }
            ],
            "connections": [
                { "name": "n", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 9} ] }
            ]
        }"#;
        let srj: SimpleRouteJson = serde_json::from_str(blob).unwrap();
        let prob = rasterize(&srj, 1.0);
        // Decoy pad at (5,5) contains no endpoint -> still blocked.
        assert!(prob.grid.is_obstacle(prob.mapping.dims.idx(5, 5)));
        // And it is in no net's passable_pads.
        let decoy = prob.mapping.dims.idx(5, 5);
        for net in &prob.nets {
            assert!(!net.passable_pads.contains(&decoy));
        }
    }

    /// (c) `to_solution` snaps the first and last vertex to the exact port
    /// coordinate carried in `pin_points`, while interior vertices stay at cell
    /// centres.
    #[test]
    fn to_solution_snaps_endpoints_to_exact_ports() {
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let mapping = Mapping::new(&bounds, 1.0);
        let d = mapping.dims;
        let path = vec![d.idx(0, 0), d.idx(1, 0), d.idx(2, 0)];
        // Exact ports offset from the cell centres (0.5,0.5) / (2.5,0.5).
        let mut pins = HashMap::new();
        pins.insert(d.idx(0, 0), (0.12, 0.34));
        pins.insert(d.idx(2, 0), (2.87, 0.65));
        let board = BoardRoute {
            results: vec![RouteResult {
                net: "n".into(),
                path,
                cost: 2,
            }],
            unrouted: vec![],
            congestion: vec![],
        };
        let traces = to_solution(&board, &mapping, &pins, 0.1, "top");
        let r = &traces[0].route;
        match &r[0] {
            RoutePoint::Wire { x, y, .. } => {
                assert_eq!((*x, *y), (0.12, 0.34));
            }
            _ => panic!("wire expected"),
        }
        // Interior vertex stays at the cell centre (1.5, 0.5).
        match &r[1] {
            RoutePoint::Wire { x, y, .. } => {
                assert_eq!((*x, *y), (1.5, 0.5));
            }
            _ => panic!("wire expected"),
        }
        match &r[2] {
            RoutePoint::Wire { x, y, .. } => {
                assert_eq!((*x, *y), (2.87, 0.65));
            }
            _ => panic!("wire expected"),
        }
    }
}
