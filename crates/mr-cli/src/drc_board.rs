//! Bridge from a routed [`SimpleRouteJson`] solution (continuous-space
//! [`PcbTrace`]s) to a physical [`mr_drc::DrcBoard`] so the corpus benchmark can
//! run a real geometric DRC over what it actually drew.
//!
//! This mirrors the logic of [`crate::drc::build_drc_board`] but works from the
//! *emitted solution soup* (traces in board units) rather than the cell-space
//! [`mr_core::BoardRoute`]. It is intentionally simple and conservative:
//!
//! * Every routed layer is treated as [`LayerKind::Signal`] (we have no plane
//!   information in the SRJ solution path, so we never *invent* a plane — which
//!   keeps via-through-plane checks honest rather than fabricating shorts).
//! * Net identity is RECONSTRUCTED from geometry, since a [`PcbTrace`] in the
//!   solution soup carries no net field. Traces that share an exact vertex are
//!   the same electrical net (junction-grouped same-net sub-connections meet at a
//!   shared point — the geometric twin of the DSN path's `r.net.split('#')` base-net
//!   collapse). A union-find over shared vertices assigns one synthetic net name
//!   per connected component, so neither a trace's own corners NOR a sibling
//!   sub-net trips a false clearance violation, while genuinely distinct nets stay
//!   mutually foreign.
//! * Obstacles (pads / keepouts) are tagged with the net of whichever trace
//!   *terminates inside* them (a routed net ends on its own pad), so own-pad
//!   contact is immune. An obstacle no trace lands in is foreign copper / a
//!   keepout and keeps net `None` (conflicts with every net — never hides a short).
//!
//! The result is deterministic: vertex keys are quantised to a fixed grid, union
//! roots follow stable trace order, layer indices follow the standard physical
//! stack, and the checker itself sorts its output.

use std::collections::{BTreeMap, BTreeSet};

use mr_core::LayerMap;
use mr_drc::{DrcBoard, DrcRules, LayerKind, Pad, Segment, Via, Violation};
use mr_srj::{PcbTrace, RoutePoint, SimpleRouteJson};

use crate::{VIA_DRILL_MM, VIA_PAD_MM};

/// Resolve a routed layer name against the effective physical stack. This is
/// the same defensive fallback as SRJ point rasterization: absent or unknown
/// names map to top rather than inventing a layer the board does not contain.
fn named_layer(name: &str, layers: &LayerMap) -> u32 {
    layers.index_of(name).unwrap_or(0)
}

#[derive(Clone, Copy)]
enum EndpointSide {
    First,
    Last,
}

fn endpoint_layer(point: &RoutePoint, side: EndpointSide, layers: &LayerMap) -> u32 {
    match point {
        RoutePoint::Wire { layer, .. } => named_layer(layer, layers),
        RoutePoint::Via {
            from_layer,
            to_layer,
            ..
        } => named_layer(
            match side {
                EndpointSide::First => from_layer,
                EndpointSide::Last => to_layer,
            },
            layers,
        ),
    }
}

/// Match the rasterizer's obstacle-layer contract: known names are retained,
/// sorted, and deduplicated; an empty or all-unknown declaration occupies every
/// effective physical layer.
fn obstacle_layers(obstacle: &mr_srj::Obstacle, layers: &LayerMap) -> Vec<u32> {
    let mut occupied: Vec<u32> = obstacle
        .layers
        .iter()
        .filter_map(|name| layers.index_of(name))
        .collect();
    occupied.sort_unstable();
    occupied.dedup();
    if occupied.is_empty() {
        (0..layers.len()).collect()
    } else {
        occupied
    }
}

/// Map every `connectedTo` alias carried by a directly connectivity-labelled
/// obstacle to the declared electrical-net labels that use it. An alias may be
/// ambiguous; callers only promote obstacles whose complete alias union resolves
/// to exactly one label.
fn connectivity_alias_map(obstacles: &[mr_srj::Obstacle]) -> BTreeMap<String, BTreeSet<String>> {
    let mut aliases: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for obstacle in obstacles {
        let Some(net) = mr_srj::obstacle_connectivity_net(obstacle) else {
            continue;
        };
        let label = format!("c{net}");
        for alias in &obstacle.connected_to {
            aliases
                .entry(alias.clone())
                .or_default()
                .insert(label.clone());
        }
    }
    aliases
}

/// Resolve an obstacle's authoritative electrical label. A direct
/// `connectivity_netNNNN` declaration wins. Otherwise, connected aliases promote
/// it only when their known label union contains exactly one connectivity net;
/// ambiguous or entirely unknown aliases retain the conservative legacy fallback.
fn obstacle_connectivity_label(
    obstacle: &mr_srj::Obstacle,
    aliases: &BTreeMap<String, BTreeSet<String>>,
) -> Option<String> {
    if let Some(net) = mr_srj::obstacle_connectivity_net(obstacle) {
        return Some(format!("c{net}"));
    }
    let resolved: BTreeSet<&String> = obstacle
        .connected_to
        .iter()
        .filter_map(|alias| aliases.get(alias))
        .flatten()
        .collect();
    if resolved.len() == 1 {
        resolved.first().map(|label| (*label).clone())
    } else {
        None
    }
}

fn point_xy(point: &RoutePoint) -> (f64, f64) {
    match point {
        RoutePoint::Wire { x, y, .. } | RoutePoint::Via { x, y, .. } => (*x, *y),
    }
}

fn route_point_layers(point: &RoutePoint, layers: &LayerMap) -> std::ops::RangeInclusive<u32> {
    match point {
        RoutePoint::Wire { layer, .. } => {
            let layer = named_layer(layer, layers);
            layer..=layer
        }
        RoutePoint::Via {
            from_layer,
            to_layer,
            ..
        } => {
            let from = named_layer(from_layer, layers);
            let to = named_layer(to_layer, layers);
            from.min(to)..=from.max(to)
        }
    }
}

#[derive(Clone, Debug)]
struct TaggedEndpoint {
    center: (f64, f64),
    net: String,
    /// Physical layer at this terminal side. A first terminal Via owns only its
    /// `from_layer`; a last terminal Via owns only its `to_layer`.
    layer: u32,
}

/// Stable pre-connectivity identity. Keep tagged router groups distinct from
/// untagged union-find roots even if a hand-built tag happens to look like a
/// generated `net#N` display name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum InitialNetIdentity {
    Tagged(String),
    Untagged(usize),
}

/// Build a physical [`DrcBoard`] from a routed SRJ solution.
///
/// * `srj` supplies the static obstacles (pads / keepouts).
/// * `traces` is the emitted solution soup (wires + vias) in board units.
/// * `rules` are the DRC constraints to enforce.
/// * `layers` is the effective routed layer count and defines the standard
///   top-to-bottom physical stack used by rasterization and DRC.
///
/// Pure and deterministic.
/// Reconstruct the per-trace electrical-net label exactly as [`solution_to_drc_board`]
/// (and hence the DRC) sees it, in trace order:
///
/// 1. Start from the router's tag (`PcbTrace::net`, a `g<groupid>`), or a union-find
///    component label over shared vertices for untagged (hand-built) traces.
/// 2. Relabel any trace whose endpoint lands on an SRJ `connectivity_netNNNN` pad
///    on an overlapping physical layer to `c<net>`, collapsing the many router
///    sub-groups that solder to one shared junction pad into a single electrical
///    net — the same immunity the DRC grants. A terminal Via matches only its
///    endpoint-side layer (`from_layer` when first, `to_layer` when last); its
///    barrel remains foreign to intermediate pads.
///
/// Exposed so the clearance legaliser can tag traces with the DRC's own net identity
/// before it runs, making its same-net immunity (and its internal violation gate) agree
/// with the authoritative checker rather than the bare router groups.
pub fn reconstruct_net_labels(
    srj: &SimpleRouteJson,
    traces: &[PcbTrace],
    layers: u32,
) -> Vec<String> {
    let effective_layers = LayerMap::standard(layers);
    let mut uf = UnionFind::new(traces.len());
    let mut first_at: BTreeMap<(i64, i64, u32), usize> = BTreeMap::new();
    for (i, t) in traces.iter().enumerate() {
        for rp in &t.route {
            let (x, y) = point_xy(rp);
            for layer in route_point_layers(rp, &effective_layers) {
                let key = (quantize(x), quantize(y), layer);
                match first_at.get(&key) {
                    Some(&j) => uf.union(i, j),
                    None => {
                        first_at.insert(key, i);
                    }
                }
            }
        }
    }
    let mut net_name: Vec<String> = Vec::with_capacity(traces.len());
    let mut net_identity: Vec<InitialNetIdentity> = Vec::with_capacity(traces.len());
    let mut root_name: BTreeMap<usize, String> = BTreeMap::new();
    for (i, t) in traces.iter().enumerate() {
        let (identity, name) = match &t.net {
            Some(n) => (InitialNetIdentity::Tagged(n.clone()), n.clone()),
            None => {
                let r = uf.find(i);
                let n = root_name.len();
                (
                    InitialNetIdentity::Untagged(r),
                    root_name
                        .entry(r)
                        .or_insert_with(|| format!("net#{n}"))
                        .clone(),
                )
            }
        };
        net_identity.push(identity);
        net_name.push(name);
    }
    let connectivity_aliases = connectivity_alias_map(&srj.obstacles);
    let conn_pads: Vec<(&mr_srj::Obstacle, String, Vec<u32>)> = srj
        .obstacles
        .iter()
        .filter_map(|obstacle| {
            obstacle_connectivity_label(obstacle, &connectivity_aliases)
                .map(|net| (obstacle, net, obstacle_layers(obstacle, &effective_layers)))
        })
        .collect();
    let connectivity_at = |point: &RoutePoint, side: EndpointSide| -> Option<String> {
        let (x, y) = point_xy(point);
        let endpoint_layer = endpoint_layer(point, side, &effective_layers);
        conn_pads
            .iter()
            .filter(|(o, _, occupied_layers)| {
                (x - o.center.x).abs() <= o.width / 2.0 + 1e-6
                    && (y - o.center.y).abs() <= o.height / 2.0 + 1e-6
                    && occupied_layers.contains(&endpoint_layer)
            })
            .min_by(|(a, an, _), (b, bn, _)| {
                (a.width * a.height)
                    .partial_cmp(&(b.width * b.height))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| an.cmp(bn))
            })
            .map(|(_, n, _)| n.clone())
    };
    // A router group / untagged union-find component is already one electrical
    // net. If any member terminates on a declared connectivity pad, promote that
    // identity to every member; relabelling only the touching trace would split
    // same-net copper into false `c...` versus `g...` clearance pairs. Malformed
    // input can tie one group to several declared nets. Pick the lexicographically
    // smallest label deterministically for the whole group; every other pad keeps
    // its distinct declared label and therefore still exposes the short.
    let mut connectivity_by_group: BTreeMap<InitialNetIdentity, BTreeSet<String>> = BTreeMap::new();
    for (i, t) in traces.iter().enumerate() {
        for endpoint_net in [
            (t.route.first(), EndpointSide::First),
            (t.route.last(), EndpointSide::Last),
        ]
        .into_iter()
        .filter_map(|(point, side)| point.and_then(|point| connectivity_at(point, side)))
        {
            connectivity_by_group
                .entry(net_identity[i].clone())
                .or_default()
                .insert(endpoint_net);
        }
    }
    for (name, identity) in net_name.iter_mut().zip(net_identity) {
        if let Some(connectivity) = connectivity_by_group
            .get(&identity)
            .and_then(|labels| labels.first())
        {
            name.clone_from(connectivity);
        }
    }
    net_name
}

pub fn solution_to_drc_board(
    srj: &SimpleRouteJson,
    traces: &[PcbTrace],
    rules: DrcRules,
    layers: u32,
) -> DrcBoard {
    let effective_layers = LayerMap::standard(layers);
    let (via_pad_diameter, via_hole_diameter) = srj
        .uniform_physical_rules()
        .map(|physical| (physical.via_pad_diameter_mm, physical.via_hole_diameter_mm))
        .unwrap_or((VIA_PAD_MM, VIA_DRILL_MM));
    let connectivity_aliases = connectivity_alias_map(&srj.obstacles);
    let mut segments: Vec<Segment> = Vec::new();
    let mut vias: Vec<Via> = Vec::new();
    let mut pads: Vec<Pad> = Vec::new();

    // Reconstruct the per-trace electrical-net identity exactly as the DRC sees it.
    let net_name = reconstruct_net_labels(srj, traces, layers);
    // Immutable trace endpoints tagged with their net, for legacy pad-net
    // resolution below. Interior geometry may move during exact repair and must
    // never acquire ownership of a foreign pad merely by entering it.
    let mut tagged_endpoints: Vec<TaggedEndpoint> = Vec::new();

    // Traces first (stable: outer index, then route order) so layer indices are
    // deterministic and same-net immunity holds within each net component.
    for (i, t) in traces.iter().enumerate() {
        let net = net_name[i].clone();
        // Only route terminals establish ownership for an unlabelled legacy pad.
        // The first/last coordinates are fixed by every geometry repair pass.
        for (point, side) in [
            (t.route.first(), EndpointSide::First),
            (t.route.last(), EndpointSide::Last),
        ] {
            if let Some(point) = point {
                tagged_endpoints.push(TaggedEndpoint {
                    center: point_xy(point),
                    net: net.clone(),
                    layer: endpoint_layer(point, side, &effective_layers),
                });
            }
        }
        // Walk the route, emitting every physical wire leg and one Via per Via
        // point. `to_solution_layered` compresses a vertical run into a Via and
        // intentionally omits the coincident destination-layer landing Wire, so
        // `pending_landing` carries the via center to the next emitted Wire.
        let mut prev: Option<(f64, f64, f64, u32)> = None; // x, y, width, layer
        let mut pending_landing: Option<(f64, f64)> = None;
        for rp in &t.route {
            match rp {
                RoutePoint::Wire { x, y, width, layer } => {
                    let l = named_layer(layer, &effective_layers);
                    if let Some((px, py)) = pending_landing.take() {
                        if (px, py) != (*x, *y) {
                            segments.push(Segment {
                                net: net.clone(),
                                layer: l,
                                a: (px, py),
                                b: (*x, *y),
                                width: *width,
                            });
                        }
                    } else if let Some((px, py, pw, pl)) = prev {
                        if pl == l && (px, py) != (*x, *y) {
                            segments.push(Segment {
                                net: net.clone(),
                                layer: l,
                                a: (px, py),
                                b: (*x, *y),
                                // Use the wider of the two endpoints' widths so a
                                // tapering segment is checked at its fattest.
                                width: pw.max(*width),
                            });
                        }
                    }
                    prev = Some((*x, *y, *width, l));
                }
                RoutePoint::Via {
                    x,
                    y,
                    from_layer,
                    to_layer,
                } => {
                    let from = named_layer(from_layer, &effective_layers);
                    let to = named_layer(to_layer, &effective_layers);
                    // Hand-built/external soups may omit the source landing as well.
                    // Materialize that leg using the last known wire width; normal
                    // metalroute output has a coincident source Wire, so this is a
                    // no-op on canonical traces.
                    if let Some((px, py, pw, pl)) = prev.take() {
                        if pl == from && (px, py) != (*x, *y) {
                            segments.push(Segment {
                                net: net.clone(),
                                layer: from,
                                a: (px, py),
                                b: (*x, *y),
                                width: pw,
                            });
                        }
                    }
                    vias.push(Via {
                        net: net.clone(),
                        center: (*x, *y),
                        pad_diameter: via_pad_diameter,
                        drill_diameter: via_hole_diameter,
                        from_layer: from,
                        to_layer: to,
                        antipad_radius: None,
                    });
                    pending_landing = Some((*x, *y));
                }
            }
        }
    }

    // Obstacles: one Pad per occupied layer. Prefer declared connectivity and its
    // unambiguous aliases; legacy pads fall back to the net of whichever trace
    // terminates inside them. An unresolved obstacle no trace lands in is foreign
    // copper / a keepout and keeps net `None` (never hiding a real short).
    for o in &srj.obstacles {
        // A pad declaring a connectivity net, directly or through aliases that
        // resolve uniquely, is labelled `c<net>` to match the traces relabelled
        // above. An unresolved pad falls back to the trace-containment tag.
        for l in obstacle_layers(o, &effective_layers) {
            pads.push(Pad {
                net: pad_net(o, l, &tagged_endpoints, &connectivity_aliases),
                layer: l,
                center: (o.center.x, o.center.y),
                width: o.width,
                height: o.height,
            });
        }
    }

    let layer_stack: Vec<LayerKind> = (0..effective_layers.len())
        .map(|_| LayerKind::Signal)
        .collect();

    DrcBoard {
        layers: layer_stack,
        segments,
        pads,
        vias,
        rules,
    }
}

/// Check a projected SRJ solution with its feature-pair pad rules when the
/// coherence gate is active; legacy and partial-rule inputs retain the historical
/// single-clearance checker path.
pub(crate) fn check_with_srj_rules(srj: &SimpleRouteJson, board: &DrcBoard) -> Vec<Violation> {
    srj.uniform_physical_rules().map_or_else(
        || board.check(),
        |physical| {
            board.check_with_pad_clearances(
                physical.trace_to_pad_clearance_mm,
                physical.via_to_pad_clearance_mm,
                physical.pad_to_pad_clearance_mm,
                physical.via_hole_to_hole_clearance_mm,
            )
        },
    )
}

/// Quantise a coordinate to a fixed sub-micron grid so two vertices the router
/// emitted at the "same" junction collide despite float round-trip noise.
fn quantize(v: f64) -> i64 {
    // 1e4 units per mm → 0.1 µm resolution, far finer than any real pitch.
    (v * 10_000.0).round() as i64
}

/// The electrical net of obstacle `o`. Prefer the ground-truth `connectivity_netNNNN`
/// it declares, directly or through aliases that resolve uniquely (labelled `c<net>`
/// to match the connectivity-relabelled traces). Otherwise fall back to whichever
/// tagged trace endpoint lands inside the pad rect (inclusive); failing that, `None`
/// (foreign / keepout — conflicts with every net, never hiding a real short).
fn pad_net(
    o: &mr_srj::Obstacle,
    layer: u32,
    tagged: &[TaggedEndpoint],
    connectivity_aliases: &BTreeMap<String, BTreeSet<String>>,
) -> Option<String> {
    if let Some(net) = obstacle_connectivity_label(o, connectivity_aliases) {
        return Some(net);
    }
    let (hw, hh) = (o.width / 2.0, o.height / 2.0);
    let (cx, cy) = (o.center.x, o.center.y);
    tagged
        .iter()
        .find(|endpoint| {
            endpoint.layer == layer
                && (endpoint.center.0 - cx).abs() <= hw
                && (endpoint.center.1 - cy).abs() <= hh
        })
        .map(|endpoint| endpoint.net.clone())
}

/// Minimal union-find for grouping traces into electrical nets by shared vertices.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Attach the higher-index root to the lower so naming is first-seen stable.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi] = lo;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_drc::ViolationClass;
    use mr_srj::PcbTrace;

    fn srj_no_obstacles() -> SimpleRouteJson {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "obstacles": [],
            "connections": [],
        });
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn coherent_srj_projects_pair_specific_pad_clearances_into_drc() {
        let srj: SimpleRouteJson = serde_json::from_value(serde_json::json!({
            "layerCount": 1,
            "minTraceWidth": 0.1,
            "nominalTraceWidth": 0.1,
            "defaultObstacleMargin": 0.05,
            "minTraceToPadEdgeClearance": 0.06,
            "minViaEdgeToPadEdgeClearance": 0.08,
            "minViaHoleDiameter": 0.2,
            "minViaPadDiameter": 0.4,
            "bounds": {"minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0}
        }))
        .unwrap();
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![Pad {
                net: Some("pad".into()),
                layer: 0,
                center: (0.0, 0.0),
                width: 0.2,
                height: 0.2,
            }],
            vias: vec![Via {
                net: "via".into(),
                center: (0.37, 0.0),
                pad_diameter: 0.4,
                drill_diameter: 0.2,
                from_layer: 0,
                to_layer: 0,
                antipad_radius: None,
            }],
            rules: DrcRules {
                clearance: 0.05,
                plane_antipad: 0.0,
                min_annular_ring: 0.0,
            },
        };
        assert!(board.check().is_empty(), "generic 0.05 mm gap is satisfied");
        let findings = check_with_srj_rules(&srj, &board);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].required, 0.08);
    }

    fn wire_on(x: f64, y: f64, layer: &str) -> RoutePoint {
        RoutePoint::Wire {
            x,
            y,
            width: 0.1,
            layer: layer.into(),
        }
    }

    fn wire(x: f64, y: f64) -> RoutePoint {
        wire_on(x, y, "top")
    }

    #[test]
    fn two_nets_crossing_too_close_violates() {
        // Two different traces on the same layer, parallel, copper gap below the
        // 0.2 clearance: centreline gap 0.15, half-widths 0.05 each → 0.05 gap.
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]),
            PcbTrace::new(vec![wire(0.0, 0.15), wire(10.0, 0.15)]),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        let v = board.check();
        let clearance = v
            .iter()
            .filter(|x| x.class == ViolationClass::Clearance)
            .count();
        assert!(
            clearance >= 1,
            "sub-clearance crossing must produce a violation"
        );
    }

    #[test]
    fn clean_solution_has_no_violations() {
        // Same two traces, but well apart (1.0 centreline gap >> clearance).
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]),
            PcbTrace::new(vec![wire(0.0, 1.0), wire(10.0, 1.0)]),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "well-spaced solution must be DRC-clean"
        );
    }

    #[test]
    fn same_trace_corner_is_not_a_violation() {
        // A single trace bending back on itself: same net, so never a clearance hit.
        let srj = srj_no_obstacles();
        let traces = vec![PcbTrace::new(vec![
            wire(0.0, 0.0),
            wire(10.0, 0.0),
            wire(10.0, 0.05),
            wire(0.0, 0.05),
        ])];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "a trace's own corners never conflict"
        );
    }

    #[test]
    fn junction_grouped_subnets_are_one_net() {
        // Two traces meeting at a shared junction vertex (1.0, 0.0) are the same
        // electrical net, so even though they run within clearance they must NOT
        // produce a clearance violation (mirrors the DSN path's `#seg` collapse).
        let srj = srj_no_obstacles();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(1.0, 0.0)]),
            // Shares the (1.0, 0.0) vertex, then runs parallel 0.05 away.
            PcbTrace::new(vec![wire(1.0, 0.0), wire(1.0, 0.05), wire(0.0, 0.05)]),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "sub-nets sharing a junction vertex must be treated as one net"
        );
    }

    #[test]
    fn untagged_trace_junctions_require_physical_layer_overlap() {
        let srj = srj_no_obstacles();
        let traces = vec![
            // These projected vertices share XY but not copper layer.
            PcbTrace::new(vec![wire_on(-1.0, 0.0, "top"), wire_on(0.0, 0.0, "top")]),
            PcbTrace::new(vec![
                wire_on(0.0, 0.0, "bottom"),
                wire_on(1.0, 0.0, "bottom"),
            ]),
            // A via really does occupy both layers and therefore joins the
            // bottom wire that shares its physical vertex.
            PcbTrace::new(vec![RoutePoint::Via {
                x: 2.0,
                y: 0.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            }]),
            PcbTrace::new(vec![
                wire_on(2.0, 0.0, "bottom"),
                wire_on(3.0, 0.0, "bottom"),
            ]),
        ];

        assert_eq!(
            reconstruct_net_labels(&srj, &traces, 2),
            ["net#0", "net#1", "net#2", "net#2"]
        );
    }

    #[test]
    fn pad_is_immune_to_the_net_that_terminates_in_it() {
        // A pad obstacle whose net is the trace ending inside it must not conflict
        // with that trace's copper (own-pad contact is legal).
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": 0.0, "maxX": 10.0, "minY": 0.0, "maxY": 10.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.6, "height": 0.6, "layers": ["top"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        // Trace starts inside the pad rect at (0,0) and runs out.
        let traces = vec![PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)])];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "a pad must be immune to the net that terminates inside it"
        );
    }

    #[test]
    fn shared_connectivity_pad_immunises_sibling_subnets() {
        // Two separately-grouped traces (distinct router `g` labels) both terminate
        // on ONE junction pad declaring `connectivity_net7`. They run within
        // clearance of each other AT the pad. Because the pad and both traces resolve
        // to the same `c<net>` identity via `connectedTo`, this must NOT violate —
        // it is one electrical net meeting at its own pad.
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 10.0, "minY": -1.0, "maxY": 10.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.6, "height": 0.6, "layers": ["top"],
                "connectedTo": ["pcb_smtpad_0", "connectivity_net7", "source_port_1"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        // Both traces END inside the pad at (0,0) but carry DIFFERENT group labels,
        // and run parallel 0.05 apart (well within clearance) leaving the pad.
        let traces = vec![
            PcbTrace::new(vec![wire(10.0, 0.0), wire(0.0, 0.0)]).with_net("g0"),
            PcbTrace::new(vec![wire(10.0, 0.05), wire(0.0, 0.0)]).with_net("g1"),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "sub-nets sharing a connectivity pad must be one electrical net: {:?}",
            board.check()
        );
    }

    #[test]
    fn unique_connectivity_alias_labels_routed_and_unrouted_obstacles() {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 8.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["pcb_smtpad_owner", "connectivity_net7", "source_net_3"]
                },
                {
                    "type": "rect", "center": {"x": 3.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["pcb_smtpad_endpoint", "source_net_3"]
                },
                {
                    "type": "rect", "center": {"x": 6.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["source_net_3"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let trace = PcbTrace::new(vec![wire(3.0, 0.0), wire(4.0, 0.0)]).with_net("g9");

        assert_eq!(
            reconstruct_net_labels(&srj, std::slice::from_ref(&trace), 1),
            ["cconnectivity_net7"],
            "an endpoint pad must inherit the unique connectivity label declared by a sibling alias"
        );

        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &[trace], rules, 1);
        assert_eq!(
            board
                .pads
                .iter()
                .map(|pad| pad.net.as_deref())
                .collect::<Vec<_>>(),
            [
                Some("cconnectivity_net7"),
                Some("cconnectivity_net7"),
                Some("cconnectivity_net7")
            ],
            "unique aliases must label fixed obstacles even without a routed terminal"
        );
        assert!(
            board.check().is_empty(),
            "same-net alias pads and their routed copper must be immune"
        );
    }

    #[test]
    fn ambiguous_connectivity_alias_keeps_obstacle_foreign() {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 8.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["connectivity_net7", "shared_alias"]
                },
                {
                    "type": "rect", "center": {"x": 3.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["connectivity_net8", "shared_alias"]
                },
                {
                    "type": "rect", "center": {"x": 6.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["shared_alias"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &[], rules, 1);

        assert_eq!(
            board
                .pads
                .iter()
                .map(|pad| pad.net.as_deref())
                .collect::<Vec<_>>(),
            [Some("cconnectivity_net7"), Some("cconnectivity_net8"), None],
            "an alias shared by distinct connectivity nets must not confer ownership"
        );
    }

    #[test]
    fn connectivity_relabel_propagates_across_router_group() {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 5.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.4, "height": 0.4, "layers": ["top"],
                "connectedTo": ["connectivity_net7"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(4.0, 0.0)]).with_net("g2"),
            // Same router group, but neither terminal touches the pad.
            PcbTrace::new(vec![wire(1.0, 0.05), wire(4.0, 0.05)]).with_net("g2"),
        ];

        assert_eq!(
            reconstruct_net_labels(&srj, &traces, 1),
            ["cconnectivity_net7", "cconnectivity_net7"]
        );
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            board.check().is_empty(),
            "a connectivity hit must not split one router group into false c/g conflicts"
        );
    }

    #[test]
    fn connectivity_relabel_propagates_across_untagged_component() {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 5.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.4, "height": 0.4, "layers": ["top"],
                "connectedTo": ["connectivity_net7"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(4.0, 0.0)]),
            // Shares one physical top-layer vertex with the first trace, so the
            // untagged union-find fallback makes both traces one component.
            PcbTrace::new(vec![wire(4.0, 0.0), wire(4.0, 0.05), wire(1.0, 0.05)]),
        ];

        assert_eq!(
            reconstruct_net_labels(&srj, &traces, 1),
            ["cconnectivity_net7", "cconnectivity_net7"]
        );
    }

    #[test]
    fn multiple_connectivity_nets_choose_canonical_group_label_and_conflict() {
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 5.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["connectivity_net9"]
                },
                {
                    "type": "rect", "center": {"x": 4.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["top"],
                    "connectedTo": ["connectivity_net2"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let left = PcbTrace::new(vec![wire(0.0, 0.0), wire(2.0, 0.0)]).with_net("g0");
        let right = PcbTrace::new(vec![wire(4.0, 0.0), wire(2.0, 0.0)]).with_net("g0");

        assert_eq!(
            reconstruct_net_labels(&srj, &[left.clone(), right.clone()], 1),
            ["cconnectivity_net2", "cconnectivity_net2"]
        );
        assert_eq!(
            reconstruct_net_labels(&srj, &[right.clone(), left.clone()], 1),
            ["cconnectivity_net2", "cconnectivity_net2"],
            "canonical connectivity identity must not depend on trace order"
        );

        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &[left, right], rules, 1);
        let findings = board.check();
        assert_eq!(
            findings
                .iter()
                .filter(|finding| finding.class == ViolationClass::Clearance)
                .count(),
            1,
            "the noncanonical connectivity pad must remain foreign and expose the short"
        );
        assert_eq!(
            findings[0].nets,
            ("cconnectivity_net2".into(), "cconnectivity_net9".into())
        );
    }

    #[test]
    fn terminal_via_connectivity_relabel_uses_endpoint_side() {
        let v = serde_json::json!({
            "layerCount": 3,
            "bounds": {"minX": -2.0, "maxX": 2.0, "minY": -2.0, "maxY": 2.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.1, "height": 0.1, "layers": ["inner1"],
                    "connectedTo": ["connectivity_net7"]
                },
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.1, "height": 0.1, "layers": ["bottom"],
                    "connectedTo": ["connectivity_net8"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let terminal_via = PcbTrace::new(vec![
            wire_on(-1.0, 0.0, "top"),
            RoutePoint::Via {
                x: 0.0,
                y: 0.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            },
        ])
        .with_net("via-net");

        let labels = reconstruct_net_labels(&srj, std::slice::from_ref(&terminal_via), 3);
        assert_eq!(
            labels,
            ["cconnectivity_net8"],
            "a last terminal Via must relabel from its bottom side, not an intermediate pad"
        );

        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &[terminal_via], rules, 3);
        assert_eq!(
            board
                .pads
                .iter()
                .map(|pad| (pad.layer, pad.net.as_deref()))
                .collect::<Vec<_>>(),
            [
                (1, Some("cconnectivity_net7")),
                (2, Some("cconnectivity_net8"))
            ]
        );
        assert_eq!(
            board
                .check()
                .iter()
                .filter(|violation| violation.class == ViolationClass::Clearance)
                .count(),
            1,
            "the via barrel must conflict with the foreign inner pad while its bottom pad is immune"
        );
    }

    #[test]
    fn connectivity_relabel_matches_obstacle_layer_fallback() {
        let v = serde_json::json!({
            "layerCount": 2,
            "bounds": {"minX": -1.0, "maxX": 8.0, "minY": -1.0, "maxY": 2.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": ["mystery"],
                    "connectedTo": ["connectivity_net1"]
                },
                {
                    "type": "rect", "center": {"x": 3.0, "y": 0.0},
                    "width": 0.4, "height": 0.4,
                    "layers": ["mystery", "top", "top"],
                    "connectedTo": ["connectivity_net2"]
                },
                {
                    "type": "rect", "center": {"x": 6.0, "y": 0.0},
                    "width": 0.4, "height": 0.4, "layers": [],
                    "connectedTo": ["connectivity_net3"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let traces = vec![
            PcbTrace::new(vec![
                wire_on(0.0, 0.0, "bottom"),
                wire_on(0.0, 1.0, "bottom"),
            ])
            .with_net("unknown-only"),
            PcbTrace::new(vec![
                wire_on(3.0, 0.0, "bottom"),
                wire_on(3.0, 1.0, "bottom"),
            ])
            .with_net("mixed-bottom"),
            PcbTrace::new(vec![
                wire_on(3.0, 0.0, "route-alias"),
                wire_on(4.0, 0.0, "route-alias"),
            ])
            .with_net("unknown-route-falls-back-top"),
            PcbTrace::new(vec![
                wire_on(6.0, 0.0, "bottom"),
                wire_on(6.0, 1.0, "bottom"),
            ])
            .with_net("empty"),
        ];

        assert_eq!(
            reconstruct_net_labels(&srj, &traces, 2),
            [
                "cconnectivity_net1",
                "mixed-bottom",
                "cconnectivity_net2",
                "cconnectivity_net3"
            ],
            "empty/all-unknown pads occupy all layers; mixed declarations retain only known layers"
        );
    }

    #[test]
    fn distinct_connectivity_nets_still_conflict() {
        // Two traces on DIFFERENT connectivity nets running sub-clearance must STILL
        // violate — the relabel must not blanket-immunise everything.
        let v = serde_json::json!({
            "layerCount": 1,
            "bounds": {"minX": -1.0, "maxX": 10.0, "minY": -1.0, "maxY": 10.0},
            "obstacles": [
                {"type": "rect", "center": {"x": 0.0, "y": 0.0}, "width": 0.4, "height": 0.4,
                 "layers": ["top"], "connectedTo": ["connectivity_net1"]},
                {"type": "rect", "center": {"x": 0.0, "y": 0.15}, "width": 0.4, "height": 0.4,
                 "layers": ["top"], "connectedTo": ["connectivity_net2"]}
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let traces = vec![
            PcbTrace::new(vec![wire(0.0, 0.0), wire(10.0, 0.0)]).with_net("g0"),
            PcbTrace::new(vec![wire(0.0, 0.15), wire(10.0, 0.15)]).with_net("g1"),
        ];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 1);
        assert!(
            !board.check().is_empty(),
            "different connectivity nets within clearance must still violate"
        );
    }

    #[test]
    fn via_emits_real_geometry() {
        let srj = srj_no_obstacles();
        let traces = vec![PcbTrace::new(vec![
            RoutePoint::Wire {
                x: 1.0,
                y: 1.0,
                width: 0.1,
                layer: "top".into(),
            },
            RoutePoint::Via {
                x: 1.0,
                y: 1.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            },
            RoutePoint::Wire {
                x: 1.0,
                y: 1.0,
                width: 0.1,
                layer: "bottom".into(),
            },
        ])];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };
        let board = solution_to_drc_board(&srj, &traces, rules, 2);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.vias[0].pad_diameter, VIA_PAD_MM);
        assert_eq!(board.vias[0].drill_diameter, VIA_DRILL_MM);
        // 0.45/0.2 via → annular ring 0.125 > 0.05, so no annular violation.
        assert!(board.check().is_empty());
    }

    #[test]
    fn unknown_route_layers_fall_back_to_top_geometry() {
        let srj = srj_no_obstacles();
        let trace = PcbTrace::new(vec![
            wire_on(0.0, 0.0, "route-alias"),
            RoutePoint::Via {
                x: 1.0,
                y: 0.0,
                from_layer: "route-alias".into(),
                to_layer: "bottom".into(),
            },
        ])
        .with_net("n");
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };

        let board = solution_to_drc_board(&srj, &[trace], rules, 2);
        assert_eq!(
            board.layers.len(),
            2,
            "unknown names must not add phantom layers"
        );
        assert_eq!(board.segments[0].layer, 0);
        assert_eq!((board.vias[0].from_layer, board.vias[0].to_layer), (0, 1));
    }

    #[test]
    fn obstacle_layers_match_rasterizer_fallback_and_dedup() {
        let v = serde_json::json!({
            "layerCount": 2,
            "bounds": {"minX": -1.0, "maxX": 8.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.2, "height": 0.2, "layers": []
                },
                {
                    "type": "rect", "center": {"x": 3.0, "y": 0.0},
                    "width": 0.2, "height": 0.2, "layers": ["mystery"]
                },
                {
                    "type": "rect", "center": {"x": 6.0, "y": 0.0},
                    "width": 0.2, "height": 0.2,
                    "layers": ["mystery", "top", "top"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };

        let board = solution_to_drc_board(&srj, &[], rules, 2);
        assert_eq!(
            board.layers.len(),
            2,
            "obstacle aliases must not add phantom layers"
        );
        assert_eq!(
            board
                .pads
                .iter()
                .map(|pad| (pad.center.0, pad.layer))
                .collect::<Vec<_>>(),
            [(0.0, 0), (0.0, 1), (3.0, 0), (3.0, 1), (6.0, 0)]
        );
        assert!(
            board.check().is_empty(),
            "duplicate known names must not emit overlapping self-conflicting pads"
        );
    }

    #[test]
    fn first_through_via_uses_physical_standard_stack_order() {
        let v = serde_json::json!({
            "layerCount": 4,
            "bounds": {"minX": -2.0, "maxX": 2.0, "minY": -2.0, "maxY": 2.0},
            "obstacles": [
                {"type": "rect", "center": {"x": 0.0, "y": 0.0},
                 "width": 0.1, "height": 0.1, "layers": ["inner1"]},
                {"type": "rect", "center": {"x": 0.0, "y": 0.0},
                 "width": 0.1, "height": 0.1, "layers": ["inner2"]}
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        // This is deliberately the first/only trace: first-seen indexing used to
        // assign bottom=1, append inner1/inner2 afterward, and miss both conflicts.
        let traces = vec![PcbTrace::new(vec![
            wire_on(-1.0, 0.0, "top"),
            RoutePoint::Via {
                x: 0.0,
                y: 0.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            },
            wire_on(1.0, 0.0, "bottom"),
        ])
        .with_net("n")];
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };

        let board = solution_to_drc_board(&srj, &traces, rules, 4);
        assert_eq!(board.vias[0].from_layer, 0);
        assert_eq!(board.vias[0].to_layer, 3);
        assert_eq!(
            board.pads.iter().map(|pad| pad.layer).collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(
            board
                .check()
                .iter()
                .filter(|v| v.class == ViolationClass::Clearance)
                .count(),
            2,
            "a through-via must be checked against foreign copper on both inner layers"
        );
    }

    #[test]
    fn legacy_pad_ownership_requires_endpoint_layer_match() {
        let v = serde_json::json!({
            "layerCount": 2,
            "bounds": {"minX": -1.0, "maxX": 3.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [{
                "type": "rect", "center": {"x": 0.0, "y": 0.0},
                "width": 0.4, "height": 0.4, "layers": ["top"]
            }],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let bottom = PcbTrace::new(vec![
            wire_on(0.0, 0.0, "bottom"),
            wire_on(1.0, 0.0, "bottom"),
        ])
        .with_net("bottom-net");
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };

        let mismatch = solution_to_drc_board(&srj, std::slice::from_ref(&bottom), rules, 2);
        assert_eq!(
            mismatch.pads[0].net, None,
            "a bottom endpoint must not claim a top-only legacy pad"
        );

        let top = PcbTrace::new(vec![wire(0.0, 0.0), wire(2.0, 0.0)]).with_net("top-net");
        let matched = solution_to_drc_board(&srj, &[bottom, top], rules, 2);
        assert_eq!(matched.pads[0].net.as_deref(), Some("top-net"));
    }

    #[test]
    fn terminal_via_legacy_pad_ownership_uses_endpoint_side() {
        let v = serde_json::json!({
            "layerCount": 3,
            "bounds": {"minX": -2.0, "maxX": 2.0, "minY": -1.0, "maxY": 1.0},
            "obstacles": [
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.1, "height": 0.1, "layers": ["inner1"]
                },
                {
                    "type": "rect", "center": {"x": 0.0, "y": 0.0},
                    "width": 0.1, "height": 0.1, "layers": ["bottom"]
                }
            ],
            "connections": [],
        });
        let srj: SimpleRouteJson = serde_json::from_value(v).unwrap();
        let trace = PcbTrace::new(vec![
            wire_on(-1.0, 0.0, "top"),
            RoutePoint::Via {
                x: 0.0,
                y: 0.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            },
        ])
        .with_net("n");
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };

        let board = solution_to_drc_board(&srj, &[trace], rules, 3);
        assert_eq!(
            board
                .pads
                .iter()
                .map(|pad| (pad.layer, pad.net.as_deref()))
                .collect::<Vec<_>>(),
            [(1, None), (2, Some("n"))],
            "the last Via terminal owns only its to-layer pad"
        );
        assert_eq!(
            board
                .check()
                .iter()
                .filter(|violation| violation.class == ViolationClass::Clearance)
                .count(),
            1,
            "the via barrel must still conflict with foreign intermediate copper"
        );
    }

    /// `to_solution_layered` intentionally omits the wire vertex coincident with a
    /// via's destination landing. The DRC bridge must still materialize the copper
    /// leg from the via center to the first destination-layer wire.
    #[test]
    fn compressed_via_destination_leg_is_drc_checked() {
        let srj = srj_no_obstacles();
        let routed = PcbTrace::new(vec![
            RoutePoint::Wire {
                x: 0.0,
                y: 0.0,
                width: 0.1,
                layer: "top".into(),
            },
            RoutePoint::Via {
                x: 0.0,
                y: 0.0,
                from_layer: "top".into(),
                to_layer: "bottom".into(),
            },
            RoutePoint::Wire {
                x: 2.0,
                y: 0.0,
                width: 0.1,
                layer: "bottom".into(),
            },
        ])
        .with_net("A");
        let crossing = PcbTrace::new(vec![
            RoutePoint::Wire {
                x: 1.0,
                y: -0.1,
                width: 0.1,
                layer: "bottom".into(),
            },
            RoutePoint::Wire {
                x: 1.0,
                y: 0.1,
                width: 0.1,
                layer: "bottom".into(),
            },
        ])
        .with_net("B");
        let rules = DrcRules {
            clearance: 0.2,
            plane_antipad: 0.25,
            min_annular_ring: 0.05,
        };

        let board = solution_to_drc_board(&srj, &[routed, crossing], rules, 2);
        assert!(
            board.segments.iter().any(|segment| {
                segment.net == "A" && segment.a == (0.0, 0.0) && segment.b == (2.0, 0.0)
            }),
            "destination-layer copper from the compressed via landing was omitted"
        );
        assert_eq!(
            board
                .check()
                .iter()
                .filter(|violation| violation.class == ViolationClass::Clearance)
                .count(),
            1,
            "the omitted destination leg crosses foreign bottom-layer copper"
        );
    }
}
