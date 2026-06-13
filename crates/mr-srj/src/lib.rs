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

use mr_core::{BoardRoute, CellIdx, Dims, Grid, NetEndpoints};
use mr_grid::GridBuilder;
use serde::{Deserialize, Serialize};

/// A 2-D point in continuous tscircuit coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Obstacle {
    #[serde(rename = "type")]
    pub kind: String,
    pub center: Point,
    pub width: f64,
    pub height: f64,
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
    /// Build the mapping for `bounds` at the given `resolution`.
    ///
    /// Dimensions are `ceil(span / resolution)`, floored at 1 on each axis so a
    /// zero- or negative-area bounds still yields a 1×1 grid.
    pub fn new(bounds: &Bounds, resolution: f64) -> Self {
        let span_x = (bounds.max_x - bounds.min_x).max(0.0);
        let span_y = (bounds.max_y - bounds.min_y).max(0.0);
        let w = ((span_x / resolution).ceil() as u32).max(1);
        let h = ((span_y / resolution).ceil() as u32).max(1);
        Self {
            origin_x: bounds.min_x,
            origin_y: bounds.min_y,
            resolution,
            dims: Dims::new(w, h),
        }
    }

    /// Continuous centre of cell `cell`.
    pub fn cell_center(&self, cell: CellIdx) -> (f64, f64) {
        let (x, y) = self.dims.xy(cell);
        (
            self.origin_x + (x as f64 + 0.5) * self.resolution,
            self.origin_y + (y as f64 + 0.5) * self.resolution,
        )
    }

    /// Cell containing continuous point `(x, y)`, clamped into the grid.
    pub fn point_to_cell(&self, point: (f64, f64)) -> CellIdx {
        let (cx, cy) = self.point_to_xy(point);
        self.dims.idx(cx, cy)
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
}

/// (B3) Rasterise a continuous tscircuit problem into a cell-space problem.
///
/// See the module docs for the dimension, obstacle-overlap, and k-point
/// decomposition rules.
pub fn rasterize(srj: &SimpleRouteJson, resolution: f64) -> RasterizedProblem {
    let mapping = Mapping::new(&srj.bounds, resolution);
    let mut builder = GridBuilder::new(mapping.dims, 1);

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
        builder.mark_rect(x0, y0, x1, y1);
    }

    let grid = builder.build();
    let nets = decompose_connections(&srj.connections, &mapping);

    RasterizedProblem {
        grid,
        nets,
        mapping,
    }
}

/// Upper (inclusive) cell index touched by a continuous box that ends at `hi`:
/// `ceil((hi-origin)/res) - 1`, clamped into `[0, extent-1]`.
fn cell_upper(hi: f64, origin: f64, res: f64, extent: u32) -> u32 {
    let edge = ((hi - origin) / res).ceil() - 1.0;
    clamp_index(edge, extent)
}

/// Decompose every connection into chained two-point nets (plan R8).
fn decompose_connections(connections: &[Connection], mapping: &Mapping) -> Vec<NetEndpoints> {
    let mut nets = Vec::new();
    for conn in connections {
        let pts = &conn.points_to_connect;
        if pts.len() < 2 {
            continue;
        }
        let segments = pts.len() - 1;
        for (seg, win) in pts.windows(2).enumerate() {
            let src = mapping.point_to_cell((win[0].x, win[0].y));
            let dst = mapping.point_to_cell((win[1].x, win[1].y));
            let net = if segments == 1 {
                conn.name.clone()
            } else {
                format!("{}#{}", conn.name, seg)
            };
            nets.push(NetEndpoints { net, src, dst });
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

/// (B4) De-rasterise a routed board into a tscircuit solution soup.
///
/// Each [`mr_core::RouteResult`] becomes one [`PcbTrace`] whose route is the
/// sequence of wire vertices at the continuous cell centres of its path. This is
/// single-layer for now: every vertex is on `layer` and no vias are emitted.
pub fn to_solution(
    board: &BoardRoute,
    mapping: &Mapping,
    trace_width: f64,
    layer: &str,
) -> Vec<PcbTrace> {
    board
        .results
        .iter()
        .map(|result| {
            let route = result
                .path
                .iter()
                .map(|&cell| {
                    let (x, y) = mapping.cell_center(cell);
                    RoutePoint::Wire {
                        x,
                        y,
                        width: trace_width,
                        layer: layer.to_string(),
                    }
                })
                .collect();
            PcbTrace::new(route)
        })
        .collect()
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
        // 10/1 = 10 cells per axis.
        assert_eq!(prob.mapping.dims, Dims::new(10, 10));
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
        // Exactly the 2x2 block.
        let count = prob
            .grid
            .cost
            .iter()
            .filter(|&&c| c == mr_core::OBSTACLE)
            .count();
        assert_eq!(count, 4);
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
        let traces = to_solution(&board, &mapping, 0.2, "top");
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].kind, "pcb_trace");
        assert_eq!(traces[0].route.len(), 3);
        // Cell centres at res 1, origin 0: (0.5,0.5), (1.5,0.5), (2.5,0.5).
        let expected = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)];
        for (pt, &(ex, ey)) in traces[0].route.iter().zip(expected.iter()) {
            match pt {
                RoutePoint::Wire {
                    x,
                    y,
                    width,
                    layer,
                } => {
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
        let traces = to_solution(&board, &mapping, 0.1, "top");
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
}
