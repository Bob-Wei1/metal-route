//! Exact continuous board-outline geometry shared by raster routing and DRC.
//!
//! The router consumes only dependency-inverted masks on [`mr_core::Grid`].  This
//! module owns the continuous polygon semantics used to build those masks and to
//! validate emitted trace capsules / via disks, keeping the two decisions identical.

use mr_drc::{point_seg_dist, seg_seg_dist};

use crate::{Bounds, OutlinePoint, SimpleRouteJson};

type Point = (f64, f64);

const GEOMETRY_EPS: f64 = 1e-9;

/// Producer default used when an SRJ carries a physical outline but omits an
/// explicit `minBoardEdgeClearance`.
pub const DEFAULT_MIN_BOARD_EDGE_CLEARANCE_MM: f64 = 0.2;

/// Why an explicitly requested board-edge constraint could not be represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoardOutlineError {
    InvalidBounds,
    InvalidClearance,
    InvalidTraceWidth,
    InvalidViaPadDiameter,
    TooFewVertices,
    NonFiniteVertex,
    DegenerateEdge,
    DegeneratePolygon,
    SelfIntersection,
}

impl std::fmt::Display for BoardOutlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidBounds => "board bounds are not a finite positive rectangle",
            Self::InvalidClearance => "minBoardEdgeClearance is not finite and non-negative",
            Self::InvalidTraceWidth => "routed trace width is not finite and positive",
            Self::InvalidViaPadDiameter => "routed via-pad diameter is not finite and positive",
            Self::TooFewVertices => "board outline has fewer than three distinct vertices",
            Self::NonFiniteVertex => "board outline contains a non-finite vertex",
            Self::DegenerateEdge => "board outline contains a zero-length edge",
            Self::DegeneratePolygon => "board outline has zero area",
            Self::SelfIntersection => "board outline self-intersects",
        };
        f.write_str(message)
    }
}

impl std::error::Error for BoardOutlineError {}

/// A validated physical polygon plus the exact centreline keepouts required for
/// emitted trace capsules and via disks.
#[derive(Clone, Debug, PartialEq)]
pub struct BoardOutlineConstraint {
    vertices: Vec<Point>,
    edge_clearance_mm: f64,
    trace_radius_mm: f64,
    via_radius_mm: f64,
}

impl BoardOutlineConstraint {
    /// Resolve an SRJ's board-edge contract.
    ///
    /// * A physical `outline` activates the producer's 0.2 mm default when the
    ///   explicit clearance is absent.
    /// * An explicit clearance with no outline constrains the declared bounds.
    /// * No outline and no explicit clearance preserves the legacy unconstrained
    ///   route (`Ok(None)`).
    /// * Any malformed active contract returns `Err`; callers must fail closed.
    pub fn from_srj(
        srj: &SimpleRouteJson,
        trace_width_mm: f64,
        via_pad_diameter_mm: f64,
    ) -> Result<Option<Self>, BoardOutlineError> {
        let has_outline = !srj.physical_rules.outline.is_empty();
        let has_clearance = srj.physical_rules.min_board_edge_clearance.is_some();
        if !has_outline && !has_clearance {
            return Ok(None);
        }
        if !trace_width_mm.is_finite() || trace_width_mm <= 0.0 {
            return Err(BoardOutlineError::InvalidTraceWidth);
        }
        if !via_pad_diameter_mm.is_finite() || via_pad_diameter_mm <= 0.0 {
            return Err(BoardOutlineError::InvalidViaPadDiameter);
        }
        let edge_clearance_mm = srj
            .physical_rules
            .min_board_edge_clearance
            .unwrap_or(DEFAULT_MIN_BOARD_EDGE_CLEARANCE_MM);
        if !edge_clearance_mm.is_finite() || edge_clearance_mm < 0.0 {
            return Err(BoardOutlineError::InvalidClearance);
        }

        let vertices = if has_outline {
            normalized_vertices(&srj.physical_rules.outline)?
        } else {
            bounds_vertices(&srj.bounds)?
        };
        validate_simple_polygon(&vertices)?;
        Ok(Some(Self {
            vertices,
            edge_clearance_mm,
            trace_radius_mm: trace_width_mm / 2.0,
            via_radius_mm: via_pad_diameter_mm / 2.0,
        }))
    }

    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    pub fn edge_clearance_mm(&self) -> f64 {
        self.edge_clearance_mm
    }

    pub fn trace_radius_mm(&self) -> f64 {
        self.trace_radius_mm
    }

    pub fn via_radius_mm(&self) -> f64 {
        self.via_radius_mm
    }

    pub fn trace_keepout_mm(&self) -> f64 {
        self.edge_clearance_mm + self.trace_radius_mm
    }

    pub fn via_keepout_mm(&self) -> f64 {
        self.edge_clearance_mm + self.via_radius_mm
    }

    /// Whether a trace-centre point can host its complete copper disk.
    pub fn trace_point_is_legal(&self, point: Point) -> bool {
        self.point_with_keepout_is_legal(point, self.trace_keepout_mm())
    }

    /// Whether a via centre can host its complete annular copper disk.
    pub fn via_point_is_legal(&self, point: Point) -> bool {
        self.point_with_keepout_is_legal(point, self.via_keepout_mm())
    }

    /// Whether the complete capsule around centreline `[a,b]` lies inside the
    /// physical polygon with the declared edge clearance.
    pub fn trace_segment_is_legal(&self, a: Point, b: Point) -> bool {
        self.trace_segment_with_radius_is_legal(a, b, self.trace_radius_mm)
    }

    /// Radius-parameterized form used by DRC for mixed-width external soups.
    pub fn trace_segment_with_radius_is_legal(&self, a: Point, b: Point, radius: f64) -> bool {
        if !radius.is_finite() || radius < 0.0 {
            return false;
        }
        if !finite_point(a) || !finite_point(b) || !point_in_polygon(a, &self.vertices) {
            return false;
        }
        if !point_in_polygon(b, &self.vertices) {
            return false;
        }
        let required = self.edge_clearance_mm + radius;
        polygon_edges(&self.vertices)
            .all(|(p, q)| seg_seg_dist(a, b, p, q) + GEOMETRY_EPS >= required)
    }

    /// Copper-edge gap from a trace capsule to the physical boundary. `None`
    /// means the centreline is not wholly contained by the polygon.
    pub fn trace_edge_gap(&self, a: Point, b: Point) -> Option<f64> {
        self.trace_edge_gap_with_radius(a, b, self.trace_radius_mm)
    }

    /// Radius-parameterized copper-edge gap used by DRC for actual segment widths.
    pub fn trace_edge_gap_with_radius(&self, a: Point, b: Point, radius: f64) -> Option<f64> {
        if !radius.is_finite() || radius < 0.0 {
            return None;
        }
        if !finite_point(a)
            || !finite_point(b)
            || !point_in_polygon(a, &self.vertices)
            || !point_in_polygon(b, &self.vertices)
        {
            return None;
        }
        let distance = polygon_edges(&self.vertices)
            .map(|(p, q)| seg_seg_dist(a, b, p, q))
            .fold(f64::INFINITY, f64::min);
        // A positive-radius centreline that exits and re-enters must cross an edge,
        // making the exact distance zero. Keep the containment branch explicit for
        // defensive zero-width callers too.
        if distance <= GEOMETRY_EPS && segment_crosses_polygon_boundary(a, b, &self.vertices) {
            return None;
        }
        Some(distance - radius)
    }

    /// Copper-edge gap from a via disk to the physical boundary. `None` means the
    /// centre itself lies outside the polygon.
    pub fn via_edge_gap(&self, center: Point) -> Option<f64> {
        self.disk_edge_gap(center, self.via_radius_mm)
    }

    /// Copper-edge gap for an arbitrary disk, used by DRC for actual via pads.
    pub fn disk_edge_gap(&self, center: Point, radius: f64) -> Option<f64> {
        if !radius.is_finite() || radius < 0.0 {
            return None;
        }
        if !finite_point(center) || !point_in_polygon(center, &self.vertices) {
            return None;
        }
        Some(min_point_edge_distance(center, &self.vertices) - radius)
    }

    fn point_with_keepout_is_legal(&self, point: Point, required: f64) -> bool {
        finite_point(point)
            && point_in_polygon(point, &self.vertices)
            && min_point_edge_distance(point, &self.vertices) + GEOMETRY_EPS >= required
    }
}

fn normalized_vertices(points: &[OutlinePoint]) -> Result<Vec<Point>, BoardOutlineError> {
    if points.len() < 3 {
        return Err(BoardOutlineError::TooFewVertices);
    }
    let mut vertices: Vec<Point> = points.iter().map(|p| (p.x, p.y)).collect();
    if vertices.iter().any(|&point| !finite_point(point)) {
        return Err(BoardOutlineError::NonFiniteVertex);
    }
    if vertices.len() > 3 && points_equal(vertices[0], *vertices.last().unwrap()) {
        vertices.pop();
    }
    if vertices.len() < 3 {
        return Err(BoardOutlineError::TooFewVertices);
    }
    Ok(vertices)
}

fn bounds_vertices(bounds: &Bounds) -> Result<Vec<Point>, BoardOutlineError> {
    if !bounds.min_x.is_finite()
        || !bounds.max_x.is_finite()
        || !bounds.min_y.is_finite()
        || !bounds.max_y.is_finite()
        || bounds.min_x >= bounds.max_x
        || bounds.min_y >= bounds.max_y
    {
        return Err(BoardOutlineError::InvalidBounds);
    }
    Ok(vec![
        (bounds.min_x, bounds.min_y),
        (bounds.max_x, bounds.min_y),
        (bounds.max_x, bounds.max_y),
        (bounds.min_x, bounds.max_y),
    ])
}

fn validate_simple_polygon(vertices: &[Point]) -> Result<(), BoardOutlineError> {
    if vertices.len() < 3 {
        return Err(BoardOutlineError::TooFewVertices);
    }
    for (a, b) in polygon_edges(vertices) {
        if points_equal(a, b) {
            return Err(BoardOutlineError::DegenerateEdge);
        }
    }
    let twice_area: f64 = polygon_edges(vertices)
        .map(|(a, b)| a.0 * b.1 - b.0 * a.1)
        .sum();
    if twice_area.abs() <= GEOMETRY_EPS {
        return Err(BoardOutlineError::DegeneratePolygon);
    }
    let n = vertices.len();
    for i in 0..n {
        let a0 = vertices[i];
        let a1 = vertices[(i + 1) % n];
        for j in i + 1..n {
            if j == i || j == (i + 1) % n || (i == 0 && j + 1 == n) {
                continue;
            }
            let b0 = vertices[j];
            let b1 = vertices[(j + 1) % n];
            if segments_intersect(a0, a1, b0, b1) {
                return Err(BoardOutlineError::SelfIntersection);
            }
        }
    }
    Ok(())
}

fn polygon_edges(vertices: &[Point]) -> impl Iterator<Item = (Point, Point)> + '_ {
    vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
}

fn finite_point(point: Point) -> bool {
    point.0.is_finite() && point.1.is_finite()
}

fn points_equal(a: Point, b: Point) -> bool {
    (a.0 - b.0).abs() <= GEOMETRY_EPS && (a.1 - b.1).abs() <= GEOMETRY_EPS
}

fn min_point_edge_distance(point: Point, vertices: &[Point]) -> f64 {
    polygon_edges(vertices)
        .map(|(a, b)| point_seg_dist(point, a, b))
        .fold(f64::INFINITY, f64::min)
}

fn point_in_polygon(point: Point, vertices: &[Point]) -> bool {
    if polygon_edges(vertices).any(|(a, b)| point_seg_dist(point, a, b) <= GEOMETRY_EPS) {
        return true;
    }
    let mut inside = false;
    for (a, b) in polygon_edges(vertices) {
        let crosses_y = (a.1 > point.1) != (b.1 > point.1);
        if crosses_y {
            let x_cross = a.0 + (point.1 - a.1) * (b.0 - a.0) / (b.1 - a.1);
            if x_cross > point.0 {
                inside = !inside;
            }
        }
    }
    inside
}

fn orientation(a: Point, b: Point, c: Point) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn point_on_segment(point: Point, a: Point, b: Point) -> bool {
    orientation(a, b, point).abs() <= GEOMETRY_EPS
        && point.0 >= a.0.min(b.0) - GEOMETRY_EPS
        && point.0 <= a.0.max(b.0) + GEOMETRY_EPS
        && point.1 >= a.1.min(b.1) - GEOMETRY_EPS
        && point.1 <= a.1.max(b.1) + GEOMETRY_EPS
}

fn segments_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool {
    let o1 = orientation(a0, a1, b0);
    let o2 = orientation(a0, a1, b1);
    let o3 = orientation(b0, b1, a0);
    let o4 = orientation(b0, b1, a1);
    if ((o1 > GEOMETRY_EPS && o2 < -GEOMETRY_EPS) || (o1 < -GEOMETRY_EPS && o2 > GEOMETRY_EPS))
        && ((o3 > GEOMETRY_EPS && o4 < -GEOMETRY_EPS) || (o3 < -GEOMETRY_EPS && o4 > GEOMETRY_EPS))
    {
        return true;
    }
    (o1.abs() <= GEOMETRY_EPS && point_on_segment(b0, a0, a1))
        || (o2.abs() <= GEOMETRY_EPS && point_on_segment(b1, a0, a1))
        || (o3.abs() <= GEOMETRY_EPS && point_on_segment(a0, b0, b1))
        || (o4.abs() <= GEOMETRY_EPS && point_on_segment(a1, b0, b1))
}

fn segment_crosses_polygon_boundary(a: Point, b: Point, vertices: &[Point]) -> bool {
    polygon_edges(vertices).any(|(p, q)| segments_intersect(a, b, p, q))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srj(value: serde_json::Value) -> SimpleRouteJson {
        serde_json::from_value(value).unwrap()
    }

    fn concave() -> SimpleRouteJson {
        srj(serde_json::json!({
            "layerCount": 2,
            "minTraceWidth": 0.15,
            "bounds": {"minX": -9.0, "maxX": 9.0, "minY": -7.0, "maxY": 7.0},
            "outline": [
                {"x": -8.0, "y": -6.0}, {"x": -2.0, "y": -6.0},
                {"x": -2.0, "y": 2.0}, {"x": 2.0, "y": 2.0},
                {"x": 2.0, "y": -6.0}, {"x": 8.0, "y": -6.0},
                {"x": 8.0, "y": 6.0}, {"x": -8.0, "y": 6.0}
            ]
        }))
    }

    #[test]
    fn concave_cutout_rejects_crossing_capsule_and_accepts_inboard_detour() {
        let outline = BoardOutlineConstraint::from_srj(&concave(), 0.15, 0.45)
            .unwrap()
            .unwrap();
        assert_eq!(outline.edge_clearance_mm(), 0.2);
        assert!(!outline.trace_segment_is_legal((-4.1066667, -4.7666667), (5.1266667, -4.7666667)));
        assert!(outline.trace_segment_is_legal((-4.0, 3.0), (4.0, 3.0)));
    }

    #[test]
    fn rectangular_capsule_and_via_honor_feature_radius_plus_clearance() {
        let board = srj(serde_json::json!({
            "layerCount": 2,
            "minBoardEdgeClearance": 0.2,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0}
        }));
        let outline = BoardOutlineConstraint::from_srj(&board, 0.2, 0.4)
            .unwrap()
            .unwrap();
        assert!(outline.trace_segment_is_legal((0.3, 1.0), (0.3, 9.0)));
        assert!(!outline.trace_segment_is_legal((0.299, 1.0), (0.299, 9.0)));
        assert!(outline.via_point_is_legal((0.4, 5.0)));
        assert!(!outline.via_point_is_legal((0.399, 5.0)));
    }

    #[test]
    fn inactive_board_contract_preserves_legacy_route() {
        let board = srj(serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0}
        }));
        assert_eq!(
            BoardOutlineConstraint::from_srj(&board, 0.2, 0.4).unwrap(),
            None
        );
    }

    #[test]
    fn malformed_active_outline_fails_closed() {
        let board = srj(serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "outline": [
                {"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0}, {"x": 10.0, "y": 0.0}
            ]
        }));
        assert_eq!(
            BoardOutlineConstraint::from_srj(&board, 0.2, 0.4),
            Err(BoardOutlineError::DegeneratePolygon)
        );
    }

    #[test]
    fn repeated_closing_vertex_is_normalized() {
        let board = srj(serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "outline": [
                {"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0}, {"x": 0.0, "y": 10.0},
                {"x": 0.0, "y": 0.0}
            ]
        }));
        let outline = BoardOutlineConstraint::from_srj(&board, 0.2, 0.4)
            .unwrap()
            .unwrap();
        assert_eq!(outline.vertices().len(), 4);
    }
}
