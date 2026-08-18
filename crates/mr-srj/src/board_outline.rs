//! Exact continuous board-outline geometry shared by raster routing and DRC.
//!
//! The router consumes only dependency-inverted masks on [`mr_core::Grid`].  This
//! module owns the continuous polygon semantics used to build those masks and to
//! validate emitted trace capsules / via disks, keeping the two decisions identical.

use mr_core::{Grid, LayerMap};
use mr_drc::{point_seg_dist, seg_seg_dist};

use crate::{Bounds, OutlinePoint, PcbTrace, RoutePoint, SimpleRouteJson};

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

    /// Rasterise the layer-invariant board mask once.
    ///
    /// The public continuous predicates above deliberately remain the simple,
    /// authoritative reference implementation used by DRC. Routing grids can
    /// contain hundreds of thousands of planar nodes, though, so rescanning every
    /// polygon edge for every node and planar step is prohibitively expensive for
    /// detailed mechanical outlines. This projection keeps the exact predicates
    /// for every final candidate while a deterministic row/column index excludes
    /// only edges whose axis-aligned separation already exceeds the largest
    /// possible keepout. The index is therefore a conservative superset of every
    /// edge that could change an answer.
    pub(crate) fn raster_mask_plane(&self, x_lines: &[f64], y_lines: &[f64]) -> Vec<u8> {
        OutlineRasterIndex::new(self, x_lines, y_lines).raster_mask()
    }
}

#[derive(Clone, Copy, Debug)]
struct IndexedOutlineEdge {
    a: Point,
    b: Point,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl IndexedOutlineEdge {
    fn new(a: Point, b: Point) -> Self {
        Self {
            a,
            b,
            min_x: a.0.min(b.0),
            max_x: a.0.max(b.0),
            min_y: a.1.min(b.1),
            max_y: a.1.max(b.1),
        }
    }
}

/// Deterministic scanline index used only by board-mask projection.
///
/// Edge ids are appended in polygon order, so every exact predicate visits its
/// conservative candidate set in the same order as [`polygon_edges`]. `rows[y]`
/// contains every edge whose y-range lies within `query_radius` of `y_lines[y]`;
/// `columns[x]` is the analogous x-range index. A second axis-separation check at
/// query time turns these scanline buckets into a small spatial candidate set.
struct OutlineRasterIndex<'a> {
    outline: &'a BoardOutlineConstraint,
    x_lines: &'a [f64],
    y_lines: &'a [f64],
    edges: Vec<IndexedOutlineEdge>,
    rows: Vec<Vec<usize>>,
    columns: Vec<Vec<usize>>,
    query_radius: f64,
}

impl<'a> OutlineRasterIndex<'a> {
    fn new(outline: &'a BoardOutlineConstraint, x_lines: &'a [f64], y_lines: &'a [f64]) -> Self {
        let edges: Vec<_> = polygon_edges(&outline.vertices)
            .map(|(a, b)| IndexedOutlineEdge::new(a, b))
            .collect();
        let query_radius = outline
            .trace_keepout_mm()
            .max(outline.via_keepout_mm())
            .max(GEOMETRY_EPS);
        let mut rows = vec![Vec::new(); y_lines.len()];
        let mut columns = vec![Vec::new(); x_lines.len()];

        // Build by ascending edge id so candidate iteration is stable and mirrors
        // the naive polygon scan. The extra epsilon makes range admission robust at
        // the exact clearance boundary; final acceptance still uses the original
        // distance + GEOMETRY_EPS comparison.
        let indexed_radius = query_radius + GEOMETRY_EPS;
        for (edge_id, edge) in edges.iter().enumerate() {
            for (row, &y) in y_lines.iter().enumerate() {
                if axis_interval_distance(y, edge.min_y, edge.max_y) <= indexed_radius {
                    rows[row].push(edge_id);
                }
            }
            for (column, &x) in x_lines.iter().enumerate() {
                if axis_interval_distance(x, edge.min_x, edge.max_x) <= indexed_radius {
                    columns[column].push(edge_id);
                }
            }
        }

        Self {
            outline,
            x_lines,
            y_lines,
            edges,
            rows,
            columns,
            query_radius,
        }
    }

    fn raster_mask(&self) -> Vec<u8> {
        let width = self.x_lines.len();
        let height = self.y_lines.len();
        let mut mask = vec![0u8; width.saturating_mul(height)];
        let mut inside = vec![false; mask.len()];

        for (y, &physical_y) in self.y_lines.iter().enumerate() {
            for (x, &physical_x) in self.x_lines.iter().enumerate() {
                let index = y * width + x;
                let point = (physical_x, physical_y);
                let legality = self.point_legality(point, y);
                inside[index] = legality.inside;
                if !legality.trace {
                    mask[index] |= Grid::BOARD_TRACE_NODE;
                }
                if !legality.via {
                    mask[index] |= Grid::BOARD_VIA_NODE;
                }

                if x > 0 {
                    let left = index - 1;
                    let left_point = (self.x_lines[x - 1], physical_y);
                    if !inside[left]
                        || !legality.inside
                        || !self.horizontal_segment_is_legal(left_point, point, y)
                    {
                        mask[left] |= Grid::BOARD_EDGE_POS_X;
                        mask[index] |= Grid::BOARD_EDGE_NEG_X;
                    }
                }
                if y > 0 {
                    let above = index - width;
                    let above_point = (physical_x, self.y_lines[y - 1]);
                    if !inside[above]
                        || !legality.inside
                        || !self.vertical_segment_is_legal(above_point, point, x)
                    {
                        mask[above] |= Grid::BOARD_EDGE_POS_Y;
                        mask[index] |= Grid::BOARD_EDGE_NEG_Y;
                    }
                }
            }
        }
        mask
    }

    /// Resolve containment, trace-centre clearance, and via-centre clearance in
    /// one candidate scan. The old projection recomputed containment and the
    /// minimum edge distance independently for both feature radii.
    fn point_legality(&self, point: Point, row: usize) -> PointLegality {
        let candidates = &self.rows[row];
        let inside = self.point_in_polygon(point, candidates);
        if !inside {
            return PointLegality {
                inside: false,
                trace: false,
                via: false,
            };
        }

        let trace_required = self.outline.trace_keepout_mm();
        let via_required = self.outline.via_keepout_mm();
        let mut trace = true;
        let mut via = true;
        let axis_limit = self.query_radius + GEOMETRY_EPS;
        for &edge_id in candidates {
            let edge = self.edges[edge_id];
            if axis_interval_distance(point.0, edge.min_x, edge.max_x) > axis_limit {
                continue;
            }
            let distance = point_seg_dist(point, edge.a, edge.b);
            if distance + GEOMETRY_EPS < trace_required {
                trace = false;
            }
            if distance + GEOMETRY_EPS < via_required {
                via = false;
            }
            if !trace && !via {
                break;
            }
        }
        PointLegality { inside, trace, via }
    }

    /// Exact point-in-polygon semantics over a conservative row bucket.
    fn point_in_polygon(&self, point: Point, candidates: &[usize]) -> bool {
        if !finite_point(point) {
            return false;
        }
        if candidates.iter().copied().any(|edge_id| {
            let edge = self.edges[edge_id];
            point_seg_dist(point, edge.a, edge.b) <= GEOMETRY_EPS
        }) {
            return true;
        }

        let mut inside = false;
        for &edge_id in candidates {
            let edge = self.edges[edge_id];
            let crosses_y = (edge.a.1 > point.1) != (edge.b.1 > point.1);
            if crosses_y {
                let x_cross =
                    edge.a.0 + (point.1 - edge.a.1) * (edge.b.0 - edge.a.0) / (edge.b.1 - edge.a.1);
                if x_cross > point.0 {
                    inside = !inside;
                }
            }
        }
        inside
    }

    fn horizontal_segment_is_legal(&self, a: Point, b: Point, row: usize) -> bool {
        self.segment_is_legal(a, b, &self.rows[row])
    }

    fn vertical_segment_is_legal(&self, a: Point, b: Point, column: usize) -> bool {
        self.segment_is_legal(a, b, &self.columns[column])
    }

    /// Check only edges whose AABB is close enough to the segment AABB to
    /// possibly violate clearance, then apply the authoritative exact distance.
    fn segment_is_legal(&self, a: Point, b: Point, candidates: &[usize]) -> bool {
        let required = self.outline.trace_keepout_mm();
        let candidate_limit = required + GEOMETRY_EPS;
        let min_x = a.0.min(b.0);
        let max_x = a.0.max(b.0);
        let min_y = a.1.min(b.1);
        let max_y = a.1.max(b.1);
        candidates.iter().copied().all(|edge_id| {
            let edge = self.edges[edge_id];
            if interval_distance(min_x, max_x, edge.min_x, edge.max_x) > candidate_limit
                || interval_distance(min_y, max_y, edge.min_y, edge.max_y) > candidate_limit
            {
                return true;
            }
            seg_seg_dist(a, b, edge.a, edge.b) + GEOMETRY_EPS >= required
        })
    }
}

#[derive(Clone, Copy)]
struct PointLegality {
    inside: bool,
    trace: bool,
    via: bool,
}

#[inline]
fn axis_interval_distance(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min - value
    } else if value > max {
        value - max
    } else {
        0.0
    }
}

#[inline]
fn interval_distance(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> f64 {
    if a_max < b_min {
        b_min - a_max
    } else if b_max < a_min {
        a_min - b_max
    } else {
        0.0
    }
}

/// Validate the complete emitted solution soup against an SRJ's active board
/// outline contract.
///
/// This deliberately consumes the final [`PcbTrace`] representation rather than
/// a router path: callers must run it after every beautification, legalization,
/// and via-repair pass.  Singleton wire points are checked as copper disks, every
/// physical wire leg is checked as a capsule at its actual emitted width, and
/// every via is checked using `routed_via_pad_diameter_mm`. `effective_layer_count`
/// must be the routed stack size used to emit the soup, including any product
/// layer override, so unknown-name fallback and same-layer legs match DRC.
pub fn solution_respects_board_outline(
    srj: &SimpleRouteJson,
    traces: &[PcbTrace],
    routed_via_pad_diameter_mm: f64,
    effective_layer_count: u32,
) -> Result<bool, BoardOutlineError> {
    // The stored nominal trace radius is not used below: emitted soups may mix
    // widths, so every node and segment supplies its actual radius explicitly.
    let Some(outline) = BoardOutlineConstraint::from_srj(srj, 1.0, routed_via_pad_diameter_mm)?
    else {
        return Ok(true);
    };
    // Match the authoritative emitted-soup DRC: standard stack names resolve to
    // their numeric layers and every unknown alias falls back to top (layer 0).
    let layers = LayerMap::standard(effective_layer_count.max(1));
    let named_layer = |name: &str| layers.index_of(name).unwrap_or(0);
    for trace in traces {
        // Mirrors the emitted-soup leg construction used by DRC: a Via may need
        // a source landing from the preceding Wire and carries its destination
        // landing forward to the next Wire.
        let mut previous_wire: Option<(f64, f64, f64, u32)> = None;
        let mut pending_landing: Option<(f64, f64)> = None;
        for point in &trace.route {
            match point {
                RoutePoint::Wire { x, y, width, layer } => {
                    let center = (*x, *y);
                    let radius = *width / 2.0;
                    if !outline.trace_segment_with_radius_is_legal(center, center, radius) {
                        return Ok(false);
                    }
                    let layer = named_layer(layer);
                    if let Some(landing) = pending_landing.take() {
                        if landing != center
                            && !outline.trace_segment_with_radius_is_legal(landing, center, radius)
                        {
                            return Ok(false);
                        }
                    } else if let Some((px, py, previous_width, previous_layer)) = previous_wire {
                        if previous_layer == layer
                            && (px, py) != center
                            && !outline.trace_segment_with_radius_is_legal(
                                (px, py),
                                center,
                                previous_width.max(*width) / 2.0,
                            )
                        {
                            return Ok(false);
                        }
                    }
                    previous_wire = Some((*x, *y, *width, layer));
                }
                RoutePoint::Via {
                    x, y, from_layer, ..
                } => {
                    let center = (*x, *y);
                    let via_radius = routed_via_pad_diameter_mm / 2.0;
                    if !outline
                        .disk_edge_gap(center, via_radius)
                        .is_some_and(|gap| gap + GEOMETRY_EPS >= outline.edge_clearance_mm())
                    {
                        return Ok(false);
                    }
                    let from_layer = named_layer(from_layer);
                    if let Some((px, py, width, previous_layer)) = previous_wire.take() {
                        if previous_layer == from_layer
                            && (px, py) != center
                            && !outline.trace_segment_with_radius_is_legal(
                                (px, py),
                                center,
                                width / 2.0,
                            )
                        {
                            return Ok(false);
                        }
                    }
                    pending_landing = Some(center);
                }
            }
        }
    }
    Ok(true)
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
    remove_collinear_backtracking_spurs(&mut vertices);
    if vertices.len() < 3 {
        return Err(BoardOutlineError::TooFewVertices);
    }
    Ok(vertices)
}

/// Remove only a zero-area hairpin: `a -> b -> c` must be collinear and the two
/// directed legs must reverse.  This is intentionally much narrower than generic
/// polygon simplification; ordinary collinear boundary points remain byte-for-byte
/// topology, while non-collinear bowties continue to fail validation.
fn remove_collinear_backtracking_spurs(vertices: &mut Vec<Point>) {
    loop {
        let len = vertices.len();
        if len < 3 {
            return;
        }
        let removable = (0..len).find(|&index| {
            let a = vertices[(index + len - 1) % len];
            let b = vertices[index];
            let c = vertices[(index + 1) % len];
            let incoming = (b.0 - a.0, b.1 - a.1);
            let outgoing = (c.0 - b.0, c.1 - b.1);
            orientation(a, b, c) == 0.0 && incoming.0 * outgoing.0 + incoming.1 * outgoing.1 < 0.0
        });
        let Some(index) = removable else {
            return;
        };
        vertices.remove(index);

        // A closed `a -> b -> a` hairpin leaves two adjacent copies of `a`.
        // Their equality is a direct consequence of the proven spur, so folding
        // that duplicate is part of this same narrow normalization.
        if vertices.len() >= 2 {
            let previous = (index + vertices.len() - 1) % vertices.len();
            let next = index % vertices.len();
            if previous != next && points_equal(vertices[previous], vertices[next]) {
                vertices.remove(next);
            }
        }
    }
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
    fn non_collinear_bowtie_is_not_normalized() {
        let points = [
            OutlinePoint { x: 0.0, y: 0.0 },
            OutlinePoint { x: 10.0, y: 10.0 },
            OutlinePoint { x: 0.0, y: 10.0 },
            OutlinePoint { x: 10.0, y: 0.0 },
        ];
        assert_eq!(normalized_vertices(&points).unwrap().len(), points.len());
        let board = srj(serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "outline": points
        }));
        assert!(BoardOutlineConstraint::from_srj(&board, 0.2, 0.4).is_err());
    }

    #[test]
    fn bugreport55_collinear_backtracking_spur_is_normalized() {
        const FIXTURE: &str =
            include_str!("../../../benchmarks/corpus/bug-reports/bugreport55-b7c349.srj.json");
        let board: SimpleRouteJson = serde_json::from_str(FIXTURE).unwrap();
        let original_len = board.physical_rules.outline.len();
        let outline = BoardOutlineConstraint::from_srj(&board, 0.15, 0.45)
            .expect("zero-area spur is accepted")
            .unwrap();
        // The producer also repeats the closing vertex, so normalization drops
        // that conventional duplicate plus the single backtracking turnaround.
        assert_eq!(outline.vertices().len(), original_len - 2);
        assert!(!outline.vertices().contains(&(-6.518, -40.25)));
    }

    #[test]
    fn ordinary_collinear_boundary_vertices_are_preserved() {
        let board = srj(serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "outline": [
                {"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 0.0},
                {"x": 10.0, "y": 0.0}, {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0}
            ]
        }));
        let outline = BoardOutlineConstraint::from_srj(&board, 0.2, 0.4)
            .unwrap()
            .unwrap();
        assert_eq!(outline.vertices().len(), 5);
    }

    #[test]
    fn near_collinear_reversal_is_not_normalized() {
        let points = [
            OutlinePoint { x: 0.0, y: 0.0 },
            OutlinePoint { x: -1.0, y: 1e-12 },
            OutlinePoint { x: 5.0, y: 0.0 },
            OutlinePoint { x: 5.0, y: 5.0 },
            OutlinePoint { x: 0.0, y: 5.0 },
        ];
        assert_eq!(normalized_vertices(&points).unwrap().len(), points.len());
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

    fn wire(x: f64, y: f64, width: f64) -> RoutePoint {
        wire_on_layer(x, y, width, "top")
    }

    fn wire_on_layer(x: f64, y: f64, width: f64, layer: &str) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width,
            layer: layer.to_string(),
        }
    }

    #[test]
    fn final_soup_validator_checks_singleton_wire_segment_and_via_copper() {
        let rectangular = srj(serde_json::json!({
            "layerCount": 2,
            "minBoardEdgeClearance": 0.2,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0}
        }));
        let singleton = PcbTrace::new(vec![wire(0.25, 5.0, 0.2)]);
        assert!(!solution_respects_board_outline(&rectangular, &[singleton], 0.4, 2).unwrap());

        let crossing = PcbTrace::new(vec![
            wire(-4.1066667, -4.7666667, 0.15),
            wire(5.1266667, -4.7666667, 0.15),
        ]);
        assert!(!solution_respects_board_outline(&concave(), &[crossing], 0.45, 2).unwrap());
        let unknown_aliases = PcbTrace::new(vec![
            wire_on_layer(-4.1066667, -4.7666667, 0.15, "signal_a"),
            wire_on_layer(5.1266667, -4.7666667, 0.15, "signal_b"),
        ]);
        assert!(
            !solution_respects_board_outline(&concave(), &[unknown_aliases], 0.45, 2).unwrap(),
            "unknown layer aliases both fall back to top, matching authoritative DRC"
        );
        let known_layer_transition = PcbTrace::new(vec![
            wire_on_layer(-4.1066667, -4.7666667, 0.15, "top"),
            wire_on_layer(5.1266667, -4.7666667, 0.15, "bottom"),
        ]);
        let mut declared_one_layer = concave();
        declared_one_layer.layer_count = 1;
        assert!(
            solution_respects_board_outline(
                &declared_one_layer,
                &[known_layer_transition],
                0.45,
                2,
            )
            .unwrap(),
            "the effective two-layer override keeps top and bottom distinct"
        );

        let via = PcbTrace::new(vec![RoutePoint::Via {
            x: 0.3,
            y: 5.0,
            from_layer: "top".to_string(),
            to_layer: "bottom".to_string(),
        }]);
        assert!(!solution_respects_board_outline(&rectangular, &[via], 0.4, 2).unwrap());

        let legal = PcbTrace::new(vec![
            wire(1.0, 1.0, 0.2),
            wire(9.0, 1.0, 0.2),
            RoutePoint::Via {
                x: 9.0,
                y: 1.0,
                from_layer: "top".to_string(),
                to_layer: "bottom".to_string(),
            },
            wire(9.0, 9.0, 0.2),
        ]);
        assert!(solution_respects_board_outline(&rectangular, &[legal], 0.4, 2).unwrap());
    }
}
