//! Specctra DSN -> [`SimpleRouteJson`] ingest.
//!
//! A Specctra `.dsn` file is a single S-expression document carrying everything
//! the router needs to build a routing problem: the board outline, the layer
//! stackup, component placement, the footprint library (per-pad geometry), and
//! the netlist. This module parses that file directly into an
//! [`mr_srj::SimpleRouteJson`] so it can be rasterised and routed by the existing
//! pipeline — no separate netlist or PnP export required.
//!
//! # Coordinate units
//!
//! The unit in `(resolution <unit> <factor>)` declares the physical unit used by
//! coordinates in the design. The factor is the precision Freerouting uses for
//! its internal integer grid and for coordinates written to a session; it does
//! **not** divide standard coordinates read from the DSN. A raw design value is
//! therefore converted by `raw * unit_to_mm(unit)`. For the common
//! `(resolution um 10)` header, `148313` is 148.313 mm on input, while the same
//! position is written as `1483130` in a resolution-10 session.
//!
//! The historical bed-of-nails exporter writes session-scaled integer
//! coordinates into its design files. Those files identify themselves with
//! `(host_cad "bed-of-nails")`; for that one producer we divide input geometry by
//! the resolution factor to preserve the established handoff contract.
//!
//! KiCad emits DSN with y pointing *up* but written as negative numbers; we keep
//! coordinates exactly as written (only scaled to mm) since the router only cares
//! about relative geometry and a consistent frame.
//!
//! # Absolute pin positions
//!
//! For each placed component we look up its footprint image, then for each pin
//! compute the absolute board position:
//!
//! ```text
//! abs = place_pos + Rotate(rot) · (mirror_if_back · pin_offset)
//! ```
//!
//! where `rot` is CCW degrees and a `back`-side placement mirrors the x of the
//! pin offset (Specctra/KiCad convention: back-side footprints flip across the y
//! axis) *before* rotation.
//!
//! # What becomes the routing problem
//!
//! * `bounds` = bounding box of the board boundary polygon (mm).
//! * one rect [`Obstacle`] per placed pad, sized from its padstack, tagged with
//!   `connected_to = ["REF-PIN"]` so the rasteriser's own-pad masking works.
//! * one [`Connection`] per net (>= 2 pins), `points_to_connect` = that net's
//!   pin positions in listed order.
//! * `min_trace_width` from a `(rule (width N))` if present, else a default.
//! * `layer_count` from the `(structure (layer ...))` entries.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use mr_core::{LayerMap, ViaModel};
use mr_srj::{Bounds, Connection, Obstacle, Point, SimpleRouteJson};

/// Default trace width in mm when the DSN carries no `(rule (width N))`.
const DEFAULT_TRACE_WIDTH_MM: f64 = 0.15;

/// Default copper-to-copper clearance in mm when the DSN carries no
/// `(rule (clearance N))`.
const DEFAULT_CLEARANCE_MM: f64 = 0.15;

// ---------------------------------------------------------------------------
// S-expression tokenizer + parser
// ---------------------------------------------------------------------------

/// A parsed S-expression node: either an atom (token) or a list of nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Sexpr {
    Atom(String),
    List(Vec<Sexpr>),
}

impl Sexpr {
    /// The atom string, if this node is an atom.
    fn as_atom(&self) -> Option<&str> {
        match self {
            Sexpr::Atom(s) => Some(s.as_str()),
            Sexpr::List(_) => None,
        }
    }

    /// The child nodes, if this node is a list.
    fn as_list(&self) -> Option<&[Sexpr]> {
        match self {
            Sexpr::List(v) => Some(v),
            Sexpr::Atom(_) => None,
        }
    }

    /// The head symbol of a list, e.g. `"resolution"` for `(resolution um 10)`.
    fn head(&self) -> Option<&str> {
        self.as_list()
            .and_then(|v| v.first())
            .and_then(|n| n.as_atom())
    }

    /// All direct child lists whose head symbol equals `name`.
    fn children_named<'a>(&'a self, name: &str) -> impl Iterator<Item = &'a Sexpr> {
        let name = name.to_string();
        self.as_list()
            .into_iter()
            .flatten()
            .filter(move |n| n.head() == Some(name.as_str()))
    }

    /// The first direct child list whose head symbol equals `name`.
    fn child_named(&self, name: &str) -> Option<&Sexpr> {
        self.children_named(name).next()
    }
}

/// Tokenize DSN source into atoms and parentheses, honouring `"`-quoted strings.
///
/// Quoted strings may contain spaces and parentheses; the closing `"` ends the
/// token. We do not interpret escapes (Specctra has none here). Everything else
/// is whitespace-separated.
fn tokenize(src: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' => {
                tokens.push(c.to_string());
                chars.next();
            }
            '"' => {
                chars.next(); // consume the opening quote
                              // Special case the Specctra `(string_quote ")` declaration: a
                              // `"` immediately followed by `)` (or whitespace then `)`) is a
                              // literal lone quote, not the start of a quoted string. Treating
                              // it as a string opener would swallow the rest of the file up to
                              // the next `"`. Emit it as an atom instead.
                if matches!(chars.peek(), Some(')')) {
                    tokens.push("\"".to_string());
                } else {
                    let mut s = String::new();
                    for d in chars.by_ref() {
                        if d == '"' {
                            break;
                        }
                        s.push(d);
                    }
                    tokens.push(s);
                }
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut s = String::new();
                while let Some(&d) = chars.peek() {
                    if d == '(' || d == ')' || d == '"' || d.is_whitespace() {
                        break;
                    }
                    s.push(d);
                    chars.next();
                }
                tokens.push(s);
            }
        }
    }
    tokens
}

/// Parse a full DSN document into a single root [`Sexpr`].
fn parse_sexpr(src: &str) -> Result<Sexpr> {
    let tokens = tokenize(src);
    let mut pos = 0;
    let node = parse_node(&tokens, &mut pos)?;
    if pos != tokens.len() {
        bail!(
            "trailing token '{}' after complete DSN document",
            tokens[pos]
        );
    }
    Ok(node)
}

/// Recursive-descent parse of one node starting at `*pos`.
fn parse_node(tokens: &[String], pos: &mut usize) -> Result<Sexpr> {
    if *pos >= tokens.len() {
        bail!("unexpected end of DSN input");
    }
    let tok = &tokens[*pos];
    if tok == "(" {
        *pos += 1;
        let mut list = Vec::new();
        loop {
            if *pos >= tokens.len() {
                bail!("unterminated '(' in DSN input");
            }
            if tokens[*pos] == ")" {
                *pos += 1;
                break;
            }
            list.push(parse_node(tokens, pos)?);
        }
        Ok(Sexpr::List(list))
    } else if tok == ")" {
        bail!("unexpected ')' in DSN input");
    } else {
        *pos += 1;
        Ok(Sexpr::Atom(tok.clone()))
    }
}

// ---------------------------------------------------------------------------
// Domain model extracted from the parsed tree
// ---------------------------------------------------------------------------

/// One pin of a footprint image: its id and offset (mm) from the footprint
/// origin, plus the padstack name (used to look up pad size).
#[derive(Debug, Clone)]
struct ImagePin {
    id: String,
    off_x: f64,
    off_y: f64,
    padstack: String,
    /// Pin-local pad rotation from `(pin ... (rotate N) ...)`, in degrees.
    rotation: f64,
}

/// A footprint image (library entry): its pins keyed for lookup by pin id.
#[derive(Debug, Clone, Default)]
struct Image {
    pins: Vec<ImagePin>,
}

/// A placed component instance.
#[derive(Debug, Clone)]
struct Placement {
    reference: String,
    image_id: String,
    x: f64,
    y: f64,
    /// True if placed on the back side (mirror x of pin offsets before rotate).
    back: bool,
    /// Rotation in CCW degrees.
    rot: f64,
}

/// A library pin transformed into board coordinates, retaining its composed pad
/// orientation so asymmetric pad geometry can be emitted correctly.
#[derive(Debug, Clone)]
struct PlacedPin {
    x: f64,
    y: f64,
    padstack: String,
    rotation: f64,
}

/// A representative pad geometry for a padstack: a size (mm) plus the copper
/// layers the padstack's shapes touch.
///
/// `layers` is the set of layer NAMES named by the padstack's `(shape (rect
/// LAYER ...))` / `(circle LAYER ...)` entries, in file order. A through-hole pad
/// names every signal layer (or uses a `*` / `signal` wildcard); an SMD pad names
/// exactly one. The wildcard layers `"signal"` / `"*"` are recorded verbatim here
/// and expanded to the real stackup at emit time (see [`pad_layer_names`]).
#[derive(Debug, Clone)]
struct PadSize {
    w: f64,
    h: f64,
    /// Layer names the padstack's shapes are defined on, in file order.
    layers: Vec<String>,
}

/// Parse stats reported alongside the converted problem (for human output).
#[derive(Debug, Clone, PartialEq)]
pub struct ParseStats {
    /// Number of signal/power layers in the stackup.
    pub layers: u32,
    /// Number of placed component instances.
    pub components: usize,
    /// Number of placed pads (one per pin of every placed component) emitted as
    /// obstacles.
    pub pads: usize,
    /// Number of nets with >= 2 pins (i.e. emitted as connections).
    pub nets: usize,
    /// Number of nets skipped because they had < 2 pins.
    pub nets_skipped_small: usize,
    /// Board bounding box width in mm.
    pub board_w_mm: f64,
    /// Board bounding box height in mm.
    pub board_h_mm: f64,
    /// Minimum trace width in mm used for the problem.
    pub min_trace_width_mm: f64,
    /// Minimum copper-to-copper clearance in mm used for the problem.
    pub min_clearance_mm: f64,
    /// Number of distinct via padstacks declared in the DSN structure.
    pub vias_declared: usize,
    /// Whether the resolved [`ViaModel`] is a full through-hole model (every
    /// adjacent step legal). `false` means a restricted blind/buried-only stack.
    pub vias_through_hole: bool,
}

/// The complete outcome of a DSN ingest: the routing problem plus the layer
/// stackup and via model the rasteriser and router need to honour real layer
/// assignments and via spans.
///
/// `srj` carries continuous geometry (obstacles tagged with their real layer
/// names, connection points tagged with their pad layer). `layer_map` is the
/// ordered index ↔ name mapping for the `srj.layer_count` copper layers — layer 0
/// is the first signal layer in DSN file order (typically `F.Cu` / top).
/// `via_model` describes which layer transitions the router may drill and at what
/// cost.
#[derive(Debug, Clone, PartialEq)]
pub struct DsnIngest {
    /// The tscircuit routing problem.
    pub srj: SimpleRouteJson,
    /// Ordered copper layer names (index ↔ name), built from the DSN stackup.
    pub layer_map: LayerMap,
    /// The `(type signal)` layer names in stackup order (poured power planes
    /// excluded). Signal nets should route only on these; vias bridge adjacent
    /// signal layers. Equals all layers when the DSN declares no layer types.
    pub signal_layers: Vec<String>,
    /// The via model resolved from the DSN via padstacks / structure rules.
    pub via_model: ViaModel,
    /// Geometry of the first usable via padstack declared by `(structure (via
    /// ...))`. `None` means the DSN did not provide a resolvable via padstack and
    /// callers should retain their documented fallback geometry.
    pub via_geometry: Option<DsnViaGeometry>,
    /// The DSN `(resolution <unit> <divisor>)` unit (e.g. `"um"`). Needed to write
    /// a coordinate-consistent Specctra session (`.ses`) back out.
    pub resolution_unit: String,
    /// The DSN resolution divisor (e.g. `10`). Raw session units per mm are
    /// `divisor / unit_to_mm(unit)`.
    pub resolution_divisor: f64,
    /// Human-readable parse stats.
    pub stats: ParseStats,
    /// Poured power/ground planes: each binds a net to the copper layer it fills,
    /// parsed from `(structure (plane "NET" (polygon LAYER ...)))`. Used by the DRC
    /// checker to detect signal vias drilling through a foreign plane.
    pub planes: Vec<PlaneDef>,
    /// Map from a pad's `"REF-PIN"` id (as carried in [`mr_srj::Obstacle::connected_to`])
    /// to its net name, derived from the DSN `(network ...)`. Lets a consumer attribute
    /// every placed pad to a net.
    pub pin_nets: HashMap<String, String>,
}

/// A poured plane: the `net` it carries and the copper `layer` (DSN layer name) it
/// fills. Parsed from `(structure (plane "NET" (polygon LAYER ...)))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneDef {
    pub net: String,
    pub layer: String,
}

/// Physical geometry carried by one DSN via padstack.
///
/// Specctra padstacks carry the annular copper geometry directly. KiCad's DSN
/// exporter carries the drill diameter in its conventional padstack name, e.g.
/// `Via[0-1]_600:300_um`; DSN has no separate drill field for that padstack.
#[derive(Debug, Clone, PartialEq)]
pub struct DsnViaGeometry {
    /// Original padstack identifier, reused when writing the session.
    pub padstack_name: String,
    /// Conservative annular-pad diameter in millimetres.
    pub pad_diameter_mm: f64,
    /// Drill diameter decoded from a KiCad-style padstack name, when present.
    pub drill_diameter_mm: Option<f64>,
    /// Physical copper layers on which the padstack has shapes.
    pub layers: Vec<String>,
}

impl DsnIngest {
    /// Raw DSN/SES coordinate units per millimetre, the inverse of `mm_per_raw`.
    /// A mm value is written to the session as `round(mm * units_per_mm)`.
    pub fn units_per_mm(&self) -> f64 {
        // `unit_to_mm` only fails on an unsupported unit, which `dsn_to_ingest`
        // already accepted; fall back to the um/10 convention defensively.
        let unit_mm = unit_to_mm(&self.resolution_unit).unwrap_or(1e-3);
        self.resolution_divisor / unit_mm
    }
}

// ---------------------------------------------------------------------------
// Unit handling
// ---------------------------------------------------------------------------

/// Millimetres per one unit named by the DSN `(resolution <unit> ...)` header.
fn unit_to_mm(unit: &str) -> Result<f64> {
    Ok(match unit {
        "mm" => 1.0,
        "um" => 0.001,
        "inch" => 25.4,
        "mil" => 0.0254,
        other => bail!("unsupported DSN resolution unit '{other}'"),
    })
}

/// Whether this design uses the historical bed-of-nails convention where DSN
/// input coordinates are already multiplied by the resolution factor.
///
/// Keep this producer check deliberately narrow. Standard Specctra/KiCad input
/// coordinates are expressed directly in the declared physical unit.
fn uses_resolution_scaled_input(pcb: &Sexpr) -> bool {
    pcb.child_named("parser")
        .and_then(|parser| parser.child_named("host_cad"))
        .and_then(|host| host.as_list())
        .and_then(|items| items.get(1))
        .and_then(Sexpr::as_atom)
        .is_some_and(|host| host.eq_ignore_ascii_case("bed-of-nails"))
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a Specctra DSN document into a full [`DsnIngest`]: the routing problem,
/// the [`LayerMap`] stackup, the [`ViaModel`], and [`ParseStats`].
///
/// This is the richest entry point — it preserves real per-pad layer assignments
/// and the via span model. Callers that only want the [`SimpleRouteJson`] (or the
/// legacy `(srj, stats)` tuple) can use [`dsn_to_srj`] / [`dsn_to_srj_with_stats`].
///
/// See the module docs for the conversion rules. Returns an error if required
/// sections (resolution, board boundary) are missing or malformed.
pub fn dsn_to_ingest(dsn_text: &str) -> Result<DsnIngest> {
    let root = parse_sexpr(dsn_text).context("failed to parse DSN s-expression")?;

    // The root is `(pcb "name" (parser ...) (resolution ...) (structure ...) ...)`.
    // Section heads may appear at the top level of the pcb list.
    let pcb = if root.head() == Some("pcb") {
        &root
    } else {
        // Tolerate a bare top-level list whose first child list is `pcb`.
        root.children_named("pcb")
            .next()
            .ok_or_else(|| anyhow!("DSN root is not a (pcb ...) form"))?
    };

    // Freerouting's DSN reader treats the declared unit as the physical unit of
    // input coordinates. The resolution factor only sets its internal integer
    // precision and the precision of coordinates written to SES; it does not
    // divide input geometry.
    let res = pcb
        .child_named("resolution")
        .ok_or_else(|| anyhow!("DSN missing (resolution ...) header"))?;
    let res_items = res.as_list().unwrap();
    let unit = res_items
        .get(1)
        .and_then(|n| n.as_atom())
        .ok_or_else(|| anyhow!("malformed (resolution ...): missing unit"))?;
    let divisor: f64 = res_items
        .get(2)
        .and_then(|n| n.as_atom())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("malformed (resolution ...): missing/invalid divisor"))?;
    if !divisor.is_finite() || divisor <= 0.0 {
        bail!("DSN resolution divisor must be positive and finite, got {divisor}");
    }
    let input_factor = if uses_resolution_scaled_input(pcb) {
        divisor
    } else {
        1.0
    };
    let mm_per_raw = unit_to_mm(unit)? / input_factor;
    let to_mm = |raw: f64| raw * mm_per_raw;

    let structure = pcb
        .child_named("structure")
        .ok_or_else(|| anyhow!("DSN missing (structure ...)"))?;

    // Ordered layer NAMES from (layer NAME (type signal|power) ...) entries, in
    // file order. Layer 0 is the first signal layer (typically F.Cu / top); the
    // last is the bottom. This drives both `layer_count` and the LayerMap, which
    // is the single index <-> name authority the rest of the pipeline uses.
    let layer_names = parse_layer_names(structure);
    let layer_map = LayerMap::from_names(layer_names);
    let layer_count = layer_map.len();
    let signal_layers = parse_signal_layer_names(structure);

    // Board bounds from (boundary (path pcb <ap> x1 y1 x2 y2 ...)) or
    // (boundary (rect pcb x1 y1 x2 y2)).
    let bounds = parse_bounds(structure, &to_mm)?;

    // Minimum trace width: structure-level (rule (width N)) wins, else default.
    let min_trace_width_mm =
        parse_rule_width(pcb, structure, &to_mm).unwrap_or(DEFAULT_TRACE_WIDTH_MM);

    // Minimum clearance: structure-level (rule (clearance N)) wins, else default.
    let min_clearance_mm =
        parse_rule_clearance(pcb, structure, &to_mm).unwrap_or(DEFAULT_CLEARANCE_MM);

    // Padstack pad sizes: name -> representative (w,h) in mm.
    let pad_sizes = parse_padstacks(pcb, &to_mm)?;

    // Footprint images: image-id -> pins.
    let images = parse_images(pcb, &to_mm)?;

    // Placed components.
    let placements = parse_placements(pcb, &to_mm)?;

    // Build absolute pin positions and composed pad orientations.
    let mut pin_pos: HashMap<(String, String), PlacedPin> = HashMap::new();
    for pl in &placements {
        let image = match images.get(&pl.image_id) {
            Some(img) => img,
            None => continue, // image with no pins (e.g. mounting holes) -> skip
        };
        let (sin, cos) = pl.rot.to_radians().sin_cos();
        for pin in &image.pins {
            // Mirror x for back side, then rotate CCW, then translate.
            let lx = if pl.back { -pin.off_x } else { pin.off_x };
            let ly = pin.off_y;
            let rx = lx * cos - ly * sin;
            let ry = lx * sin + ly * cos;
            let ax = to_mm_translate(pl.x, rx);
            let ay = to_mm_translate(pl.y, ry);
            // Mirroring reverses a pin-local rotation; placement rotation then
            // applies in board coordinates. For 90/270-degree rectangular pads
            // only parity matters, but retaining the full angle also lets the
            // geometry helper conservatively bound arbitrary rotations.
            let pin_rotation = if pl.back { -pin.rotation } else { pin.rotation };
            pin_pos.insert(
                (pl.reference.clone(), pin.id.clone()),
                PlacedPin {
                    x: ax,
                    y: ay,
                    padstack: pin.padstack.clone(),
                    rotation: pl.rot + pin_rotation,
                },
            );
        }
    }

    // Obstacles: one rect per placed pad we know the position of, tagged with the
    // REAL layer name(s) the pad sits on (through-hole -> all layers, SMD -> one).
    let mut obstacles: Vec<Obstacle> = Vec::with_capacity(pin_pos.len());
    for ((reference, pin_id), pin) in &pin_pos {
        let size = match pad_sizes.get(&pin.padstack) {
            Some(s) => s,
            None => continue, // unknown padstack -> no geometry to mask
        };
        // Skip zero-size pads (no geometry to mask).
        if size.w <= 0.0 || size.h <= 0.0 {
            continue;
        }
        let (width, height) = oriented_pad_extents(size.w, size.h, pin.rotation);
        let layers = pad_layer_names(&size.layers, &layer_map);
        // The obstacle's representative point layer is its first (top-most) layer.
        let point_layer = layers.first().cloned();
        obstacles.push(Obstacle {
            kind: "rect".to_string(),
            center: Point {
                x: pin.x,
                y: pin.y,
                layer: point_layer,
            },
            width,
            height,
            shape: Some("rect".to_string()),
            ccw_rotation_degrees: None,
            layers,
            connected_to: vec![format!("{reference}-{pin_id}")],
        });
    }
    // Deterministic order so output is stable.
    obstacles.sort_by(|a, b| a.connected_to.cmp(&b.connected_to));
    let pads = obstacles.len();

    // Connections: one per net (>= 2 resolvable pins).
    let nets = parse_nets(pcb)?;
    let mut connections: Vec<Connection> = Vec::new();
    let mut nets_skipped_small = 0usize;
    for (net_name, pin_refs) in &nets {
        let mut points = Vec::new();
        for (reference, pin_id) in pin_refs {
            if let Some(pin) = pin_pos.get(&(reference.clone(), pin_id.clone())) {
                // The connection point's layer is its pad's actual (top-most)
                // layer. Through-hole pins resolve to the top layer; SMD pins to
                // whichever single layer they live on.
                let layer = pad_sizes
                    .get(&pin.padstack)
                    .map(|s| pad_layer_names(&s.layers, &layer_map))
                    .and_then(|ls| ls.into_iter().next())
                    .or_else(|| Some(layer_map.name(0).to_string()));
                points.push(Point {
                    x: pin.x,
                    y: pin.y,
                    layer,
                });
            }
        }
        if points.len() < 2 {
            nets_skipped_small += 1;
            continue;
        }
        connections.push(Connection {
            name: net_name.clone(),
            root_connection_name: None,
            rules: mr_srj::ConnectionRules::default(),
            points_to_connect: points,
        });
    }

    // Via model: read the DSN via padstacks (their layer spans) and structure
    // rules. All-through-hole -> ViaModel::through_hole; declared blind/buried
    // spans -> a restricted model permitting exactly those adjacent steps.
    let via_names = declared_via_names(structure);
    let (via_model, vias_declared) = parse_via_model(pcb, &via_names, layer_count, &layer_map);
    let via_geometry = parse_via_geometry(&via_names, &pad_sizes, &layer_map);
    let vias_through_hole = via_model == ViaModel::through_hole(layer_count);

    let stats = ParseStats {
        layers: layer_count,
        components: placements.len(),
        pads,
        nets: connections.len(),
        nets_skipped_small,
        board_w_mm: bounds.max_x - bounds.min_x,
        board_h_mm: bounds.max_y - bounds.min_y,
        min_trace_width_mm,
        min_clearance_mm,
        vias_declared,
        vias_through_hole,
    };

    // Poured planes (net ↔ layer) and a pad-id → net map for the DRC checker.
    let planes = parse_planes(structure);
    let mut pin_nets: HashMap<String, String> = HashMap::new();
    for (net_name, pin_refs) in &nets {
        for (reference, pin_id) in pin_refs {
            pin_nets.insert(format!("{reference}-{pin_id}"), net_name.clone());
        }
    }

    let srj = SimpleRouteJson {
        layer_count,
        min_trace_width: Some(min_trace_width_mm),
        min_clearance: Some(min_clearance_mm),
        physical_rules: mr_srj::SimpleRoutePhysicalRules::default(),
        obstacles,
        connections,
        bounds,
    };

    Ok(DsnIngest {
        srj,
        layer_map,
        signal_layers,
        via_model,
        via_geometry,
        resolution_unit: unit.to_string(),
        resolution_divisor: divisor,
        stats,
        planes,
        pin_nets,
    })
}

/// Translate `base_mm` (already mm) by `offset_mm` (already mm). The placement
/// position is in raw units; this helper keeps the call sites readable. Both are
/// expected pre-converted to mm by the caller.
#[inline]
fn to_mm_translate(base_mm: f64, offset_mm: f64) -> f64 {
    base_mm + offset_mm
}

/// Axis-aligned extents of a rectangular pad after rotation. Exact quarter turns
/// are special-cased so 90/270 degrees swap width and height without floating-point
/// residue. For other angles, the absolute sine/cosine projection gives the
/// conservative axis-aligned bounding box of the rotated rectangle.
fn oriented_pad_extents(width: f64, height: f64, rotation_deg: f64) -> (f64, f64) {
    if !rotation_deg.is_finite() {
        return (width, height);
    }
    let quarter_turns = (rotation_deg / 90.0).round();
    if (rotation_deg - quarter_turns * 90.0).abs() <= 1e-9 {
        if (quarter_turns as i64).rem_euclid(2) == 1 {
            return (height, width);
        }
        return (width, height);
    }

    let (sin, cos) = rotation_deg.to_radians().sin_cos();
    let (sin, cos) = (sin.abs(), cos.abs());
    (width * cos + height * sin, width * sin + height * cos)
}

/// Parse a Specctra DSN document into a [`SimpleRouteJson`] plus [`ParseStats`].
///
/// Back-compat shim over [`dsn_to_ingest`]; drops the [`LayerMap`] / [`ViaModel`].
/// Prefer [`dsn_to_ingest`] when you need real layer/via data.
pub fn dsn_to_srj_with_stats(dsn_text: &str) -> Result<(SimpleRouteJson, ParseStats)> {
    let ingest = dsn_to_ingest(dsn_text)?;
    Ok((ingest.srj, ingest.stats))
}

/// Parse a DSN document into a [`SimpleRouteJson`], discarding everything else.
pub fn dsn_to_srj(dsn_text: &str) -> Result<SimpleRouteJson> {
    Ok(dsn_to_ingest(dsn_text)?.srj)
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------

/// Ordered copper layer names from `(structure (layer NAME (type ...)))` entries,
/// in file order. The DSN names them top-to-bottom (F.Cu first, B.Cu last), which
/// is exactly the index order the [`LayerMap`] / routing grid want. Empty input
/// is left empty for [`LayerMap::from_names`] to promote to a single `"top"`.
/// Poured planes from `(structure (plane "NET" (polygon LAYER ...)))`, in file
/// order. Each binds a net to the copper layer its polygon fills. A plane whose net
/// or layer atom is missing is skipped.
fn parse_planes(structure: &Sexpr) -> Vec<PlaneDef> {
    let mut planes = Vec::new();
    for plane in structure.children_named("plane") {
        let Some(net) = plane
            .as_list()
            .and_then(|l| l.get(1))
            .and_then(|n| n.as_atom())
        else {
            continue;
        };
        let Some(layer) = plane
            .child_named("polygon")
            .and_then(|p| p.as_list())
            .and_then(|l| l.get(1))
            .and_then(|n| n.as_atom())
        else {
            continue;
        };
        planes.push(PlaneDef {
            net: net.to_string(),
            layer: layer.to_string(),
        });
    }
    planes
}

fn parse_layer_names(structure: &Sexpr) -> Vec<String> {
    let mut names = Vec::new();
    for layer in structure.children_named("layer") {
        if let Some(name) = layer
            .as_list()
            .and_then(|l| l.get(1))
            .and_then(|n| n.as_atom())
        {
            names.push(name.to_string());
        }
    }
    names
}

/// The names of the `(type signal)` layers, in stackup (file) order. Power/plane
/// layers (`(type power)`) are excluded — signal traces never route on a poured
/// plane. Falls back to ALL layers when no `(type ...)` is declared (so a DSN
/// without explicit types still routes on every layer, as before).
fn parse_signal_layer_names(structure: &Sexpr) -> Vec<String> {
    let mut signal = Vec::new();
    let mut saw_type = false;
    for layer in structure.children_named("layer") {
        let Some(name) = layer
            .as_list()
            .and_then(|l| l.get(1))
            .and_then(|n| n.as_atom())
        else {
            continue;
        };
        let ty = layer
            .child_named("type")
            .and_then(|t| t.as_list())
            .and_then(|l| l.get(1))
            .and_then(|n| n.as_atom());
        if let Some(ty) = ty {
            saw_type = true;
            if ty == "signal" {
                signal.push(name.to_string());
            }
        }
    }
    if !saw_type || signal.is_empty() {
        parse_layer_names(structure)
    } else {
        signal
    }
}

/// Resolve a padstack's declared shape layers to the REAL stackup layer names a
/// pad obstacle should occupy, in top-to-bottom (grid index) order.
///
/// - A padstack naming layers present in the [`LayerMap`] keeps exactly those,
///   re-sorted into stackup index order (so e.g. `[B.Cu, F.Cu]` becomes
///   `[F.Cu, B.Cu]`).
/// - A through-hole wildcard (`signal`, `*`, `all`, or naming every layer) spans
///   the whole stack -> every layer name.
/// - If nothing resolves (unknown / empty), we fall back to the top layer name,
///   preserving the historical single-`top` behaviour for that pad.
fn pad_layer_names(declared: &[String], layer_map: &LayerMap) -> Vec<String> {
    let count = layer_map.len();
    let all = || {
        (0..count)
            .map(|i| layer_map.name(i).to_string())
            .collect::<Vec<_>>()
    };

    // Through-hole wildcard span.
    let is_wildcard = |s: &str| {
        let s = s.to_ascii_lowercase();
        s == "signal" || s == "*" || s == "all" || s == "@1" || s.is_empty()
    };
    if declared.iter().any(|d| is_wildcard(d)) {
        return all();
    }

    // Collect the indices of declared layers that exist in the stackup.
    let mut indices: Vec<u32> = declared
        .iter()
        .filter_map(|d| layer_map.index_of(d))
        .collect();
    indices.sort_unstable();
    indices.dedup();

    if indices.is_empty() {
        // Nothing recognised: fall back to the top layer name (legacy behaviour).
        return vec![layer_map.name(0).to_string()];
    }
    // A padstack that names every layer is a through-hole pad spanning the stack.
    if indices.len() as u32 == count {
        return all();
    }
    indices
        .into_iter()
        .map(|i| layer_map.name(i).to_string())
        .collect()
}

/// Resolve the board's [`ViaModel`] from the DSN via padstacks and structure
/// rules, returning the model plus the count of distinct via padstacks declared.
///
/// DSN declares the usable vias in `(structure (via "ps1" "ps2" ...))`; each named
/// padstack lives in the library with shapes on the layers it touches. We map each
/// via padstack to its `[lo, hi]` layer span:
///
/// - If every declared via spans the full stack (top..bottom) — the common case,
///   e.g. an 8-layer through-hole bed-of-nails board — we return
///   [`ViaModel::through_hole`]: every adjacent step legal.
/// - If blind/buried spans are declared, we return
///   [`ViaModel::with_allowed_steps`] permitting exactly the adjacent steps those
///   spans cover (a span `[lo, hi]` contributes the steps `lo..hi`).
/// - When no vias are declared, or spans are ambiguous, we default to through-hole.
fn declared_via_names(structure: &Sexpr) -> Vec<String> {
    structure
        .child_named("via")
        .and_then(|v| v.as_list())
        .map(|items| {
            items
                .iter()
                .skip(1)
                .filter_map(|n| n.as_atom())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Decode the drill suffix produced by KiCad, e.g. the `300` in
/// `Via[0-1]_600:300_um`. The dimensions in that name are always micrometres;
/// unlike DSN coordinates they are descriptive text and are not affected by the
/// file's `(resolution ...)` factor.
fn kicad_via_drill_mm(name: &str) -> Option<f64> {
    let (_, drill) = name.rsplit_once(':')?;
    let drill_um = drill.strip_suffix("_um")?.parse::<f64>().ok()?;
    (drill_um.is_finite() && drill_um > 0.0).then_some(drill_um * 0.001)
}

fn parse_via_geometry(
    via_names: &[String],
    pad_sizes: &HashMap<String, PadSize>,
    layer_map: &LayerMap,
) -> Option<DsnViaGeometry> {
    via_names.iter().find_map(|name| {
        let pad = pad_sizes.get(name)?;
        let diameter = pad.w.max(pad.h);
        if !diameter.is_finite() || diameter <= 0.0 {
            return None;
        }
        Some(DsnViaGeometry {
            padstack_name: name.clone(),
            pad_diameter_mm: diameter,
            drill_diameter_mm: kicad_via_drill_mm(name),
            layers: pad_layer_names(&pad.layers, layer_map),
        })
    })
}

fn parse_via_model(
    pcb: &Sexpr,
    via_names: &[String],
    layer_count: u32,
    layer_map: &LayerMap,
) -> (ViaModel, usize) {
    // Padstack name -> layer names it touches, for via padstacks (incl. plated
    // shapes; we reuse the same library shape parsing).
    let library = pcb.child_named("library");
    let via_span = |name: &str| -> Option<(u32, u32)> {
        let lib = library?;
        for ps in lib.children_named("padstack") {
            let items = ps.as_list()?;
            if items.get(1).and_then(|n| n.as_atom()) != Some(name) {
                continue;
            }
            let mut idxs: Vec<u32> = Vec::new();
            let mut wildcard = false;
            for shape in ps.children_named("shape") {
                let inner = shape
                    .child_named("circle")
                    .or_else(|| shape.child_named("rect"));
                if let Some(layer) = inner.and_then(shape_layer) {
                    let l = layer.to_ascii_lowercase();
                    if l == "signal" || l == "*" || l == "all" {
                        wildcard = true;
                    } else if let Some(i) = layer_map.index_of(&layer) {
                        idxs.push(i);
                    }
                }
            }
            if wildcard {
                return Some((0, layer_count.saturating_sub(1)));
            }
            if idxs.is_empty() {
                return None;
            }
            return Some((*idxs.iter().min().unwrap(), *idxs.iter().max().unwrap()));
        }
        None
    };

    let mut spans: Vec<(u32, u32)> = Vec::new();
    for name in via_names {
        if let Some(span) = via_span(name) {
            spans.push(span);
        }
    }
    let declared = via_names.len();

    // No resolvable spans, or every span is full-stack -> through-hole.
    let full = (0u32, layer_count.saturating_sub(1));
    let all_through = !spans.is_empty() && spans.iter().all(|&s| s == full);
    if spans.is_empty() || all_through {
        return (ViaModel::through_hole(layer_count), declared);
    }

    // Restricted stack: permit exactly the adjacent steps the declared spans cover.
    let mut steps: Vec<(u32, u32)> = Vec::new();
    for &(lo, hi) in &spans {
        for s in lo..hi {
            let step = (s, s + 1);
            if !steps.contains(&step) {
                steps.push(step);
            }
        }
    }
    steps.sort_unstable();
    (
        ViaModel::with_allowed_steps(layer_count, ViaModel::DEFAULT_STEP_COST, steps),
        declared,
    )
}

/// Bounding box of the board boundary polygon / rect, in mm.
fn parse_bounds(structure: &Sexpr, to_mm: &impl Fn(f64) -> f64) -> Result<Bounds> {
    let boundary = structure
        .child_named("boundary")
        .ok_or_else(|| anyhow!("DSN structure missing (boundary ...)"))?;

    let mut coords: Vec<(f64, f64)> = Vec::new();

    if let Some(path) = boundary.child_named("path") {
        // (path pcb <aperture> x1 y1 x2 y2 ...). Skip head, layer, aperture.
        let items = path.as_list().unwrap();
        let nums: Vec<f64> = items
            .iter()
            .skip(3)
            .filter_map(|n| n.as_atom())
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        for pair in nums.chunks_exact(2) {
            coords.push((to_mm(pair[0]), to_mm(pair[1])));
        }
    } else if let Some(rect) = boundary.child_named("rect") {
        // (rect pcb x1 y1 x2 y2).
        let items = rect.as_list().unwrap();
        let nums: Vec<f64> = items
            .iter()
            .skip(2)
            .filter_map(|n| n.as_atom())
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if nums.len() >= 4 {
            coords.push((to_mm(nums[0]), to_mm(nums[1])));
            coords.push((to_mm(nums[2]), to_mm(nums[3])));
        }
    }

    if coords.is_empty() {
        bail!("DSN boundary carried no usable coordinates");
    }

    let min_x = coords.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let max_x = coords.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = coords.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let max_y = coords.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);

    Ok(Bounds {
        min_x,
        max_x,
        min_y,
        max_y,
    })
}

/// First valid direct property named `property` across all direct `(rule ...)`
/// blocks at one scope. File order is retained, preserving the historical
/// first-declaration behavior while allowing width and clearance to live in
/// separate blocks.
fn parse_rule_property(node: &Sexpr, property: &str, to_mm: &impl Fn(f64) -> f64) -> Option<f64> {
    node.children_named("rule").find_map(|rule| {
        rule.children_named(property).find_map(|value| {
            let raw = value.as_list()?.get(1)?.as_atom()?.parse::<f64>().ok()?;
            Some(to_mm(raw))
        })
    })
}

/// First `(rule (width N))` found at structure or pcb level, converted to mm.
/// Structure scope wins per property; pcb scope is only a fallback.
fn parse_rule_width(pcb: &Sexpr, structure: &Sexpr, to_mm: &impl Fn(f64) -> f64) -> Option<f64> {
    parse_rule_property(structure, "width", to_mm)
        .or_else(|| parse_rule_property(pcb, "width", to_mm))
}

/// First `(rule (clearance N))` found at structure or pcb level, converted to mm.
///
/// A single `(rule ...)` block can carry both width and clearance, e.g.
/// `(rule (width 150) (clearance 200))`; this reads the `clearance` member.
/// Structure-level rules win over pcb-level, matching [`parse_rule_width`].
fn parse_rule_clearance(
    pcb: &Sexpr,
    structure: &Sexpr,
    to_mm: &impl Fn(f64) -> f64,
) -> Option<f64> {
    parse_rule_property(structure, "clearance", to_mm)
        .or_else(|| parse_rule_property(pcb, "clearance", to_mm))
}

/// The layer atom named by a `(circle LAYER ...)` / `(rect LAYER ...)` shape.
fn shape_layer(shape_inner: &Sexpr) -> Option<String> {
    shape_inner
        .as_list()?
        .get(1)
        .and_then(|n| n.as_atom())
        .map(str::to_string)
}

/// Padstack name -> representative pad size (mm) + the layers it touches.
///
/// Size is the conservative component-wise maximum over every sizeable shape.
/// A padstack may have a larger annulus on one copper layer than another; taking
/// only the first shape would leave real copper routable. Layers are the set of
/// distinct layer names across all shapes, in file order: an SMD padstack names
/// one layer, a through-hole padstack names every signal layer (or a `signal` /
/// `*` wildcard). Wildcards are recorded verbatim and expanded at emit time.
fn parse_padstacks(pcb: &Sexpr, to_mm: &impl Fn(f64) -> f64) -> Result<HashMap<String, PadSize>> {
    let mut sizes = HashMap::new();
    let library = match pcb.child_named("library") {
        Some(l) => l,
        None => return Ok(sizes),
    };
    for ps in library.children_named("padstack") {
        let items = ps.as_list().unwrap();
        let name = match items.get(1).and_then(|n| n.as_atom()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        // Conservatively bound every shape, and collect every shape's layer name.
        let mut size: Option<(f64, f64)> = None;
        let mut layers: Vec<String> = Vec::new();
        for shape in ps.children_named("shape") {
            if let Some(circle) = shape.child_named("circle") {
                // (circle LAYER dia [x y]).
                if let Some(layer) = shape_layer(circle) {
                    if !layers.contains(&layer) {
                        layers.push(layer);
                    }
                }
                let nums: Vec<f64> = circle
                    .as_list()
                    .unwrap()
                    .iter()
                    .skip(2)
                    .filter_map(|n| n.as_atom())
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if let Some(&dia) = nums.first() {
                    let d = to_mm(dia).abs();
                    let (w, h) = size.unwrap_or((0.0, 0.0));
                    size = Some((w.max(d), h.max(d)));
                }
            } else if let Some(rect) = shape.child_named("rect") {
                // (rect LAYER x1 y1 x2 y2).
                if let Some(layer) = shape_layer(rect) {
                    if !layers.contains(&layer) {
                        layers.push(layer);
                    }
                }
                let nums: Vec<f64> = rect
                    .as_list()
                    .unwrap()
                    .iter()
                    .skip(2)
                    .filter_map(|n| n.as_atom())
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if nums.len() >= 4 {
                    let shape_w = to_mm((nums[2] - nums[0]).abs());
                    let shape_h = to_mm((nums[3] - nums[1]).abs());
                    let (w, h) = size.unwrap_or((0.0, 0.0));
                    size = Some((w.max(shape_w), h.max(shape_h)));
                }
            }
        }
        if let Some((w, h)) = size {
            sizes.insert(name, PadSize { w, h, layers });
        }
    }
    Ok(sizes)
}

/// Footprint images: image-id -> pins (id + offset mm + padstack).
fn parse_images(pcb: &Sexpr, to_mm: &impl Fn(f64) -> f64) -> Result<HashMap<String, Image>> {
    let mut images = HashMap::new();
    let library = match pcb.child_named("library") {
        Some(l) => l,
        None => return Ok(images),
    };
    for img in library.children_named("image") {
        let items = img.as_list().unwrap();
        let id = match items.get(1).and_then(|n| n.as_atom()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let mut image = Image::default();
        for pin in img.children_named("pin") {
            // (pin PADSTACK PINID relx rely) OR
            // (pin PADSTACK (rotate N) PINID relx rely). Collect padstack (1st
            // atom child after head), then the trailing PINID relx rely. We
            // gather the atom tokens after the head, skipping any (rotate ...)
            // sub-list, then interpret the last three as id/x/y.
            let pin_items = pin.as_list().unwrap();
            let padstack = match pin_items.get(1).and_then(|n| n.as_atom()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            // Gather atoms after the padstack, ignoring any sub-lists.
            let atoms: Vec<&str> = pin_items
                .iter()
                .skip(2)
                .filter_map(|n| n.as_atom())
                .collect();
            // The last two atoms are relx/rely; the one before them is the id.
            if atoms.len() < 3 {
                continue;
            }
            let n = atoms.len();
            let relx: f64 = match atoms[n - 2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let rely: f64 = match atoms[n - 1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let id = atoms[n - 3].to_string();
            let rotation = pin
                .child_named("rotate")
                .and_then(|rotate| rotate.as_list())
                .and_then(|items| items.get(1))
                .and_then(|item| item.as_atom())
                .and_then(|raw| raw.parse::<f64>().ok())
                .unwrap_or(0.0);
            image.pins.push(ImagePin {
                id,
                off_x: to_mm(relx),
                off_y: to_mm(rely),
                padstack,
                rotation,
            });
        }
        images.insert(id, image);
    }
    Ok(images)
}

/// Placed component instances from `(placement (component "LIBID" (place ...)))`.
fn parse_placements(pcb: &Sexpr, to_mm: &impl Fn(f64) -> f64) -> Result<Vec<Placement>> {
    let mut placements = Vec::new();
    let placement = match pcb.child_named("placement") {
        Some(p) => p,
        None => return Ok(placements),
    };
    for comp in placement.children_named("component") {
        let items = comp.as_list().unwrap();
        let image_id = match items.get(1).and_then(|n| n.as_atom()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        for place in comp.children_named("place") {
            // (place REF x y front|back rot).
            let p = place.as_list().unwrap();
            let reference = match p.get(1).and_then(|n| n.as_atom()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let x: f64 = match p
                .get(2)
                .and_then(|n| n.as_atom())
                .and_then(|s| s.parse().ok())
            {
                Some(v) => v,
                None => continue,
            };
            let y: f64 = match p
                .get(3)
                .and_then(|n| n.as_atom())
                .and_then(|s| s.parse().ok())
            {
                Some(v) => v,
                None => continue,
            };
            let side = p.get(4).and_then(|n| n.as_atom()).unwrap_or("front");
            let back = side.eq_ignore_ascii_case("back");
            let rot: f64 = p
                .get(5)
                .and_then(|n| n.as_atom())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            placements.push(Placement {
                reference,
                image_id: image_id.clone(),
                x: to_mm(x),
                y: to_mm(y),
                back,
                rot,
            });
        }
    }
    Ok(placements)
}

/// A net name paired with its pins, each pin a `(reference, pin-id)`.
type NetPins = (String, Vec<(String, String)>);

/// Nets: `(network (net "NAME" (pins REF-PIN ...)))`. Returns net name ->
/// list of `(ref, pin-id)` split on the LAST `-`. Class lists are ignored.
fn parse_nets(pcb: &Sexpr) -> Result<Vec<NetPins>> {
    let mut nets = Vec::new();
    let network = match pcb.child_named("network") {
        Some(n) => n,
        None => return Ok(nets),
    };
    for net in network.children_named("net") {
        let items = net.as_list().unwrap();
        let name = match items.get(1).and_then(|n| n.as_atom()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let mut refs = Vec::new();
        if let Some(pins) = net.child_named("pins") {
            for tok in pins.as_list().unwrap().iter().skip(1) {
                if let Some(s) = tok.as_atom() {
                    if let Some((reference, pin_id)) = split_ref_pin(s) {
                        refs.push((reference, pin_id));
                    }
                }
            }
        }
        nets.push((name, refs));
    }
    Ok(nets)
}

/// Split a `REF-PIN` token on the LAST `-` into `(ref, pin-id)`. Pin ids may be
/// non-numeric; refs may themselves contain `-`. Returns `None` if there is no
/// `-` (a malformed token).
fn split_ref_pin(tok: &str) -> Option<(String, String)> {
    let idx = tok.rfind('-')?;
    let reference = &tok[..idx];
    let pin = &tok[idx + 1..];
    if reference.is_empty() || pin.is_empty() {
        return None;
    }
    Some((reference.to_string(), pin.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic board: 2 components, one net, with one rotated/back-side
    /// component to exercise the transform.
    ///
    /// DSN coordinates are expressed directly in the declared physical unit;
    /// the resolution factor controls session/internal precision only. Using
    /// `um 10` keeps the coordinates easy to read: raw 1000 = 1 mm.
    const SYNTH: &str = r#"
    (pcb "synth.dsn"
      (parser
        (string_quote ")
        (space_in_quoted_tokens on)
      )
      (resolution um 10)
      (unit mm)
      (structure
        (layer F.Cu (type signal))
        (layer B.Cu (type signal))
        (boundary
          (path pcb 0 0 0 20000 0 20000 20000 0 20000 0 0)
        )
        (rule (width 150))
      )
      (placement
        (component "img_a"
          (place A 5000 5000 front 0)
        )
        (component "img_b"
          (place B 15000 5000 back 90)
        )
      )
      (library
        (image "img_a"
          (pin "ps_rect" 1 -1000 0)
          (pin "ps_rect" 2 1000 0)
        )
        (image "img_b"
          (pin "ps_circ" 1 1000 0)
          (pin "ps_circ" 2 -1000 0)
        )
        (padstack "ps_rect"
          (shape (rect F.Cu -500 -250 500 250))
          (attach off)
        )
        (padstack "ps_circ"
          (shape (circle F.Cu 600 0 0))
          (attach off)
        )
      )
      (network
        (net "N1"
          (pins A-2 B-1)
        )
        (net "ONLYONE"
          (pins A-1)
        )
      )
    )
    "#;

    #[test]
    fn tokenizer_handles_quoted_strings_and_parens() {
        let toks = tokenize(r#"(a "hello world" (b c))"#);
        assert_eq!(toks, vec!["(", "a", "hello world", "(", "b", "c", ")", ")"]);
    }

    #[test]
    fn sexpr_parser_rejects_trailing_and_unmatched_input() {
        for invalid in ["(a) trailing", "(a) (b)", "(a))", "(a", ")"] {
            assert!(
                parse_sexpr(invalid).is_err(),
                "strict full-document parse must reject {invalid:?}"
            );
        }
        assert_eq!(
            parse_sexpr("(a (b c))").unwrap().head(),
            Some("a"),
            "control document remains valid"
        );
    }

    #[test]
    fn resolution_divisor_must_be_positive_and_finite() {
        for divisor in ["0", "-1", "NaN", "inf", "-inf"] {
            let dsn = format!(
                r#"(pcb "bad-resolution"
                    (resolution mm {divisor})
                    (structure
                      (layer F.Cu (type signal))
                      (boundary (rect pcb 0 0 1000 1000))))"#
            );
            assert!(
                dsn_to_ingest(&dsn).is_err(),
                "resolution divisor {divisor} must be rejected"
            );
        }
    }

    #[test]
    fn resolution_factor_does_not_scale_input_but_sets_session_precision() {
        let fixture = |factor: u32| {
            format!(
                r#"(pcb "units"
                    (resolution um {factor})
                    (structure
                      (layer F.Cu (type signal))
                      (boundary (rect pcb 1000 -2000 21000 8000))))"#
            )
        };

        let one = dsn_to_ingest(&fixture(1)).unwrap();
        let ten = dsn_to_ingest(&fixture(10)).unwrap();
        assert_eq!(one.srj.bounds, ten.srj.bounds);
        assert_eq!(ten.srj.bounds.min_x, 1.0);
        assert_eq!(ten.srj.bounds.min_y, -2.0);
        assert_eq!(ten.srj.bounds.max_x, 21.0);
        assert_eq!(ten.srj.bounds.max_y, 8.0);

        // Session coordinates use resolution-sized subunits: one millimetre is
        // 1000 raw units at resolution 1 and 10000 at resolution 10.
        assert_eq!(one.units_per_mm(), 1_000.0);
        assert_eq!(ten.units_per_mm(), 10_000.0);
    }

    #[test]
    fn bed_of_nails_scaled_input_retains_legacy_geometry_and_session_precision() {
        let standard = r#"(pcb "standard"
            (parser (host_cad "KiCad's Pcbnew"))
            (resolution um 10)
            (structure
              (layer F.Cu (type signal))
              (layer B.Cu (type signal))
              (boundary (rect pcb 0 0 1300000 1515750))
              (via "Via[0-1]_450:200_um")
              (rule (width 1500)))
            (library
              (padstack "Via[0-1]_450:200_um"
                (shape (circle F.Cu 4500))
                (shape (circle B.Cu 4500)))))"#;
        let legacy = standard.replace("KiCad's Pcbnew", "BED-OF-NAILS");

        let standard = dsn_to_ingest(standard).unwrap();
        let legacy = dsn_to_ingest(&legacy).unwrap();

        assert_eq!(standard.stats.board_w_mm, 1300.0);
        assert_eq!(standard.stats.board_h_mm, 1515.75);
        assert_eq!(standard.stats.min_trace_width_mm, 1.5);
        assert_eq!(standard.via_geometry.as_ref().unwrap().pad_diameter_mm, 4.5);

        assert_eq!(legacy.stats.board_w_mm, 130.0);
        assert!((legacy.stats.board_h_mm - 151.575).abs() < 1e-12);
        assert_eq!(legacy.stats.min_trace_width_mm, 0.15);
        assert_eq!(legacy.via_geometry.as_ref().unwrap().pad_diameter_mm, 0.45);

        // Both producers write resolution-scaled integer coordinates to SES.
        assert_eq!(standard.units_per_mm(), 10_000.0);
        assert_eq!(legacy.units_per_mm(), 10_000.0);
    }

    #[test]
    fn split_ref_pin_uses_last_dash() {
        assert_eq!(
            split_ref_pin("C1-2"),
            Some(("C1".to_string(), "2".to_string()))
        );
        // Ref containing a dash; split on the LAST dash.
        assert_eq!(
            split_ref_pin("Net-U6-BAT"),
            Some(("Net-U6".to_string(), "BAT".to_string()))
        );
        assert_eq!(split_ref_pin("nodash"), None);
    }

    #[test]
    fn builds_pin_nets_from_network() {
        let ingest = dsn_to_ingest(SYNTH).unwrap();
        // Every pad id resolves to its net.
        assert_eq!(ingest.pin_nets.get("A-2").map(String::as_str), Some("N1"));
        assert_eq!(ingest.pin_nets.get("B-1").map(String::as_str), Some("N1"));
        assert_eq!(
            ingest.pin_nets.get("A-1").map(String::as_str),
            Some("ONLYONE")
        );
        // SYNTH declares no poured planes.
        assert!(ingest.planes.is_empty());
    }

    #[test]
    fn parses_plane_net_layer_binding() {
        // A `(plane "NET" (polygon LAYER ...))` binds a net to the copper layer it
        // fills — the binding the DRC checker needs to flag vias through a plane.
        const DSN: &str = r#"
        (pcb "p.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (layer In1.Cu (type power))
            (layer B.Cu (type signal))
            (boundary (path pcb 0 0 0 1000 1000 1000 1000 0))
            (plane "GND" (polygon In1.Cu 0 0 0 0 1000 1000 1000 1000 0))
          )
        )
        "#;
        let ingest = dsn_to_ingest(DSN).unwrap();
        assert_eq!(
            ingest.planes,
            vec![PlaneDef {
                net: "GND".to_string(),
                layer: "In1.Cu".to_string(),
            }]
        );
    }

    #[test]
    fn parses_synthetic_board() {
        let (srj, stats) = dsn_to_srj_with_stats(SYNTH).unwrap();

        // Two signal layers.
        assert_eq!(srj.layer_count, 2);
        assert_eq!(stats.layers, 2);

        // Board boundary 0..20000 raw um -> 0..20mm.
        assert_eq!(srj.bounds.min_x, 0.0);
        assert_eq!(srj.bounds.max_x, 20.0);
        assert_eq!(srj.bounds.min_y, 0.0);
        assert_eq!(srj.bounds.max_y, 20.0);
        assert_eq!(stats.board_w_mm, 20.0);
        assert_eq!(stats.board_h_mm, 20.0);

        // min trace width 150 raw um -> 0.15mm.
        assert_eq!(srj.min_trace_width, Some(0.15));

        // No (rule (clearance N)) present -> falls back to the default.
        assert_eq!(srj.min_clearance, Some(DEFAULT_CLEARANCE_MM));
        assert_eq!(stats.min_clearance_mm, DEFAULT_CLEARANCE_MM);

        // Two components placed.
        assert_eq!(stats.components, 2);

        // 4 pins total, all with non-zero pad size -> 4 obstacles.
        assert_eq!(srj.obstacles.len(), 4);
        assert_eq!(stats.pads, 4);

        // One net with >= 2 pins (N1); ONLYONE skipped.
        assert_eq!(srj.connections.len(), 1);
        assert_eq!(stats.nets, 1);
        assert_eq!(stats.nets_skipped_small, 1);
        assert_eq!(srj.connections[0].name, "N1");
        assert_eq!(srj.connections[0].points_to_connect.len(), 2);
    }

    #[test]
    fn parses_clearance_rule_alongside_width() {
        // A single (rule ...) block carrying both width and clearance. With
        // With unit `um`, 200 raw -> 0.20 mm; the factor does not scale input.
        let dsn = r#"
        (pcb "clr.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
            (rule (width 150) (clearance 200))
          )
        )
        "#;
        let (srj, stats) = dsn_to_srj_with_stats(dsn).unwrap();
        assert_eq!(srj.min_trace_width, Some(0.15));
        assert_eq!(srj.min_clearance, Some(0.20));
        assert_eq!(stats.min_clearance_mm, 0.20);
    }

    #[test]
    fn clearance_falls_back_to_default_when_absent() {
        // (rule (width N)) only -> clearance uses DEFAULT_CLEARANCE_MM.
        let dsn = r#"
        (pcb "noclr.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
            (rule (width 150))
          )
        )
        "#;
        let (srj, stats) = dsn_to_srj_with_stats(dsn).unwrap();
        assert_eq!(srj.min_clearance, Some(DEFAULT_CLEARANCE_MM));
        assert_eq!(stats.min_clearance_mm, DEFAULT_CLEARANCE_MM);
    }

    #[test]
    fn structure_clearance_overrides_pcb_level() {
        // A pcb-level (rule (clearance N)) and a structure-level one both present:
        // the structure-level rule wins, matching the width precedence.
        let dsn = r#"
        (pcb "ovr.dsn"
          (resolution um 10)
          (rule (clearance 300))
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
            (rule (clearance 200))
          )
        )
        "#;
        let (srj, stats) = dsn_to_srj_with_stats(dsn).unwrap();
        assert_eq!(srj.min_clearance, Some(0.20));
        assert_eq!(stats.min_clearance_mm, 0.20);
    }

    #[test]
    fn aggregates_multiple_rule_blocks_with_per_property_scope_precedence() {
        // Each property is resolved independently. A structure-level declaration
        // wins over pcb-level fallback even when width and clearance live in
        // separate structure rule blocks.
        let dsn = r#"
        (pcb "rules.dsn"
          (resolution um 10)
          (rule (width 100) (clearance 300))
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
            (rule (clearance 200))
            (rule (width 150))
          )
        )
        "#;
        let (srj, stats) = dsn_to_srj_with_stats(dsn).unwrap();
        assert_eq!(srj.min_trace_width, Some(0.15));
        assert_eq!(srj.min_clearance, Some(0.20));
        assert_eq!(stats.min_trace_width_mm, 0.15);
        assert_eq!(stats.min_clearance_mm, 0.20);
    }

    #[test]
    fn asymmetric_pad_geometry_handles_quarter_turns_and_arbitrary_angles() {
        let dsn = r#"
        (pcb "rotated-pads.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000)))
          (placement
            (component "plain"
              (place R0 1000 1000 front 0)
              (place R90 3000 1000 front 90)
              (place R180 5000 1000 front 180)
              (place R270 7000 1000 front 270)
              (place R45 9000 1000 front 45))
            (component "pin-rotated"
              (place P90 3000 4000 front 0)
              (place P270 7000 4000 front 180))
            (component "back-pin-rotated"
              (place B45 5000 7000 back 60)))
          (library
            (image "plain" (pin "asym" 1 0 0))
            (image "pin-rotated" (pin "asym" (rotate 90) 1 0 0))
            (image "back-pin-rotated" (pin "asym" (rotate 15) 1 0 0))
            (padstack "asym" (shape (rect F.Cu -1000 -250 1000 250))))
          (network)
        )
        "#;
        let srj = dsn_to_srj(dsn).unwrap();
        let dims = |id: &str| {
            let obstacle = srj
                .obstacles
                .iter()
                .find(|o| o.connected_to == vec![format!("{id}-1")])
                .unwrap_or_else(|| panic!("missing {id}-1"));
            (obstacle.width, obstacle.height)
        };
        assert_eq!(dims("R0"), (2.0, 0.5));
        assert_eq!(dims("R180"), (2.0, 0.5));
        assert_eq!(dims("R90"), (0.5, 2.0));
        assert_eq!(dims("R270"), (0.5, 2.0));
        assert_eq!(dims("P90"), (0.5, 2.0), "pin-local rotate must apply");
        assert_eq!(
            dims("P270"),
            (0.5, 2.0),
            "pin + placement rotation must compose"
        );

        let expected_45 = 2.5 / 2.0_f64.sqrt();
        for id in ["R45", "B45"] {
            let (width, height) = dims(id);
            assert!(
                (width - expected_45).abs() < 1e-12
                    && (height - expected_45).abs() < 1e-12,
                "{id} 45-degree AABB = ({width}, {height}), expected ({expected_45}, {expected_45})"
            );
        }
    }

    #[test]
    fn multi_shape_padstack_uses_conservative_maximum_extents() {
        let dsn = r#"
        (pcb "multi-shape.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (layer B.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000)))
          (placement (component "part" (place U1 5000 5000 front 0)))
          (library
            (image "part" (pin "stack" 1 0 0))
            (padstack "stack"
              (shape (rect F.Cu -500 -250 500 250))
              (shape (circle B.Cu 3000))))
          (network)
        )
        "#;
        let srj = dsn_to_srj(dsn).unwrap();
        assert_eq!(srj.obstacles.len(), 1);
        let pad = &srj.obstacles[0];
        assert_eq!((pad.width, pad.height), (3.0, 3.0));
        assert_eq!(pad.layers, vec!["F.Cu", "B.Cu"]);
    }

    #[test]
    fn computes_abs_pin_position_front_no_rotation() {
        let (srj, _) = dsn_to_srj_with_stats(SYNTH).unwrap();
        // Component A at (5,5)mm, front, rot 0. Pin A-2 offset (+1000,0) raw um
        // = (+1,0)mm -> abs (6,5)mm.
        let a2 = srj
            .obstacles
            .iter()
            .find(|o| o.connected_to == vec!["A-2".to_string()])
            .expect("A-2 obstacle present");
        assert!((a2.center.x - 6.0).abs() < 1e-9, "x = {}", a2.center.x);
        assert!((a2.center.y - 5.0).abs() < 1e-9, "y = {}", a2.center.y);
        // Rect pad 1000x500 raw um -> 1.0 x 0.5 mm.
        assert!((a2.width - 1.0).abs() < 1e-9);
        assert!((a2.height - 0.5).abs() < 1e-9);
    }

    #[test]
    fn computes_abs_pin_position_back_rotated() {
        let (srj, _) = dsn_to_srj_with_stats(SYNTH).unwrap();
        // Component B at (15,5)mm, back, rot 90 CCW. Pin B-1 offset (+1000,0) raw
        // um = (+1,0)mm. Back mirrors x: (-1,0). Rotate 90 CCW:
        // (x',y') = (x*cos - y*sin, x*sin + y*cos) = (-1*0 - 0*1, -1*1 + 0*0)
        //         = (0, -1). Translate by (15,5) -> (15, 4).
        let b1 = srj
            .obstacles
            .iter()
            .find(|o| o.connected_to == vec!["B-1".to_string()])
            .expect("B-1 obstacle present");
        assert!((b1.center.x - 15.0).abs() < 1e-6, "x = {}", b1.center.x);
        assert!((b1.center.y - 4.0).abs() < 1e-6, "y = {}", b1.center.y);
        // Circle dia 600 raw um -> 0.6mm square.
        assert!((b1.width - 0.6).abs() < 1e-9);
        assert!((b1.height - 0.6).abs() < 1e-9);
    }

    #[test]
    fn tolerates_pin_with_rotate_subtoken() {
        // (pin PADSTACK (rotate 90) id x y) form must still yield the right pin.
        let dsn = r#"
        (pcb "r.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
          )
          (placement
            (component "img" (place X 5000 5000 front 0))
          )
          (library
            (image "img"
              (pin "ps" (rotate 90) 1 0 0)
              (pin "ps" 2 2000 0)
            )
            (padstack "ps" (shape (circle F.Cu 500 0 0)))
          )
          (network (net "n" (pins X-1 X-2)))
        )
        "#;
        let (srj, _) = dsn_to_srj_with_stats(dsn).unwrap();
        // Both pins parsed -> 2 obstacles, one connection with 2 points.
        assert_eq!(srj.obstacles.len(), 2);
        assert_eq!(srj.connections.len(), 1);
        let x1 = srj
            .obstacles
            .iter()
            .find(|o| o.connected_to == vec!["X-1".to_string()])
            .unwrap();
        // Pin 1 at offset (0,0) -> abs (5,5)mm.
        assert!((x1.center.x - 5.0).abs() < 1e-9);
        assert!((x1.center.y - 5.0).abs() < 1e-9);
    }

    #[test]
    fn handles_rect_boundary_form() {
        let dsn = r#"
        (pcb "rb.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (boundary (rect pcb 0 -100000 200000 0))
          )
          (placement)
          (library)
          (network)
        )
        "#;
        let srj = dsn_to_srj(dsn).unwrap();
        // Input coordinates remain whole micrometres despite resolution 10.
        assert!((srj.bounds.max_x - 200.0).abs() < 1e-9);
        assert!((srj.bounds.min_y + 100.0).abs() < 1e-9);
        assert!((srj.bounds.min_x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn missing_resolution_is_an_error() {
        let dsn =
            r#"(pcb "x" (structure (layer F.Cu (type signal)) (boundary (rect pcb 0 0 1 1))))"#;
        assert!(dsn_to_srj(dsn).is_err());
    }

    // -----------------------------------------------------------------------
    // Multi-layer ingest: real layer assignment + via model (Track D)
    // -----------------------------------------------------------------------

    /// A 2-layer board with three padstacks:
    /// - `ps_smd_top`: a rect on F.Cu only (an SMD pad on the top layer).
    /// - `ps_smd_bot`: a rect on B.Cu only (an SMD pad on the bottom layer).
    /// - `ps_th`: circles on BOTH F.Cu and B.Cu (a through-hole pad spanning the
    ///   stack), reused as the via padstack declared in `(structure (via ...))`.
    const MULTILAYER: &str = r#"
    (pcb "ml.dsn"
      (resolution um 10)
      (structure
        (layer F.Cu (type signal))
        (layer B.Cu (type signal))
        (boundary (rect pcb 0 0 20000 20000))
        (via "ps_th")
        (rule (width 150))
      )
      (placement
        (component "img_t" (place T 5000 5000 front 0))
        (component "img_b" (place B 15000 5000 front 0))
        (component "img_h" (place H 10000 10000 front 0))
      )
      (library
        (image "img_t" (pin "ps_smd_top" 1 0 0))
        (image "img_b" (pin "ps_smd_bot" 1 0 0))
        (image "img_h" (pin "ps_th" 1 0 0))
        (padstack "ps_smd_top" (shape (rect F.Cu -500 -250 500 250)))
        (padstack "ps_smd_bot" (shape (rect B.Cu -500 -250 500 250)))
        (padstack "ps_th"
          (shape (circle F.Cu 600 0 0))
          (shape (circle B.Cu 600 0 0))
        )
      )
      (network
        (net "N1" (pins T-1 B-1 H-1))
      )
    )
    "#;

    #[test]
    fn multilayer_layer_map_is_file_ordered() {
        let ingest = dsn_to_ingest(MULTILAYER).unwrap();
        // Two signal layers, named in DSN file order: F.Cu (top, index 0), B.Cu.
        assert_eq!(ingest.srj.layer_count, 2);
        assert_eq!(ingest.layer_map.len(), 2);
        assert_eq!(ingest.layer_map.name(0), "F.Cu");
        assert_eq!(ingest.layer_map.name(1), "B.Cu");
        assert_eq!(ingest.layer_map.index_of("B.Cu"), Some(1));
    }

    #[test]
    fn smd_pad_lands_on_its_real_layer() {
        let ingest = dsn_to_ingest(MULTILAYER).unwrap();
        // The bottom SMD pad B-1 must occupy ONLY B.Cu, and its point.layer is B.Cu.
        let b = ingest
            .srj
            .obstacles
            .iter()
            .find(|o| o.connected_to == vec!["B-1".to_string()])
            .expect("B-1 obstacle present");
        assert_eq!(b.layers, vec!["B.Cu".to_string()]);
        assert_eq!(b.center.layer.as_deref(), Some("B.Cu"));

        // The top SMD pad T-1 occupies only F.Cu.
        let t = ingest
            .srj
            .obstacles
            .iter()
            .find(|o| o.connected_to == vec!["T-1".to_string()])
            .unwrap();
        assert_eq!(t.layers, vec!["F.Cu".to_string()]);
        assert_eq!(t.center.layer.as_deref(), Some("F.Cu"));
    }

    #[test]
    fn through_hole_pad_spans_all_layers() {
        let ingest = dsn_to_ingest(MULTILAYER).unwrap();
        // The through-hole pad H-1 names both layers -> spans both, top-first.
        let h = ingest
            .srj
            .obstacles
            .iter()
            .find(|o| o.connected_to == vec!["H-1".to_string()])
            .expect("H-1 obstacle present");
        assert_eq!(h.layers, vec!["F.Cu".to_string(), "B.Cu".to_string()]);
        // Its representative point sits on the top-most layer.
        assert_eq!(h.center.layer.as_deref(), Some("F.Cu"));
    }

    #[test]
    fn connection_point_layers_follow_their_pad() {
        let ingest = dsn_to_ingest(MULTILAYER).unwrap();
        let n1 = &ingest.srj.connections[0];
        assert_eq!(n1.name, "N1");
        // Points are listed in net order: T-1 (F.Cu), B-1 (B.Cu), H-1 (F.Cu top).
        let layers: Vec<Option<&str>> = n1
            .points_to_connect
            .iter()
            .map(|p| p.layer.as_deref())
            .collect();
        assert_eq!(layers, vec![Some("F.Cu"), Some("B.Cu"), Some("F.Cu")]);
    }

    #[test]
    fn all_through_hole_via_model_matches_through_hole() {
        let ingest = dsn_to_ingest(MULTILAYER).unwrap();
        // The only declared via (ps_th) spans top..bottom -> through-hole model.
        assert_eq!(ingest.via_model, ViaModel::through_hole(2));
        assert_eq!(ingest.stats.vias_declared, 1);
        assert!(ingest.stats.vias_through_hole);
        // Every adjacent step legal (the through-hole semantics).
        assert!(ingest.via_model.is_step_legal(0, 1));
        let geometry = ingest.via_geometry.expect("declared via geometry");
        assert_eq!(geometry.padstack_name, "ps_th");
        assert_eq!(geometry.pad_diameter_mm, 0.6);
        assert_eq!(geometry.drill_diameter_mm, None);
        assert_eq!(geometry.layers, vec!["F.Cu", "B.Cu"]);
    }

    #[test]
    fn kicad_via_name_supplies_drill_and_padstack_supplies_annulus() {
        let dsn = r#"
        (pcb "kicad-via.dsn"
          (resolution um 10)
          (structure
            (layer Top (type signal))
            (layer Bottom (type signal))
            (boundary (rect pcb 0 0 10000 10000))
            (via "Via[0-1]_600:300_um"))
          (library
            (padstack "Via[0-1]_600:300_um"
              (shape (circle Top 600))
              (shape (circle Bottom 600)))))
        "#;
        let ingest = dsn_to_ingest(dsn).unwrap();
        let geometry = ingest.via_geometry.expect("KiCad via geometry");
        assert_eq!(geometry.padstack_name, "Via[0-1]_600:300_um");
        assert_eq!(geometry.pad_diameter_mm, 0.6);
        assert_eq!(geometry.drill_diameter_mm, Some(0.3));
        assert_eq!(geometry.layers, vec!["Top", "Bottom"]);
    }

    #[test]
    fn no_declared_vias_defaults_to_through_hole() {
        // A board with layers but no (structure (via ...)) still yields a usable
        // through-hole model over the declared layer count.
        let dsn = r#"
        (pcb "nv.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (layer In1.Cu (type signal))
            (layer B.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
          )
          (placement)
          (library)
          (network)
        )
        "#;
        let ingest = dsn_to_ingest(dsn).unwrap();
        assert_eq!(ingest.layer_map.len(), 3);
        assert_eq!(ingest.via_model, ViaModel::through_hole(3));
        assert_eq!(ingest.stats.vias_declared, 0);
        assert!(ingest.stats.vias_through_hole);
    }

    #[test]
    fn blind_buried_via_spans_restrict_steps() {
        // A 4-layer board declaring only a buried 2-3 via (In1.Cu..In2.Cu) ->
        // a restricted model: only the (1,2) adjacent step is drillable.
        let dsn = r#"
        (pcb "bb.dsn"
          (resolution um 10)
          (structure
            (layer F.Cu (type signal))
            (layer In1.Cu (type signal))
            (layer In2.Cu (type signal))
            (layer B.Cu (type signal))
            (boundary (rect pcb 0 0 10000 10000))
            (via "ps_buried")
          )
          (placement)
          (library
            (padstack "ps_buried"
              (shape (circle In1.Cu 400 0 0))
              (shape (circle In2.Cu 400 0 0))
            )
          )
          (network)
        )
        "#;
        let ingest = dsn_to_ingest(dsn).unwrap();
        assert!(!ingest.stats.vias_through_hole);
        let v = &ingest.via_model;
        // Only the inner 1-2 step (In1.Cu..In2.Cu) is legal.
        assert!(v.is_step_legal(1, 2));
        assert!(!v.is_step_legal(0, 1), "top step not drilled");
        assert!(!v.is_step_legal(2, 3), "bottom step not drilled");
    }

    #[test]
    fn through_hole_wildcard_padstack_spans_stack() {
        // A padstack using the `signal` wildcard layer should span the full stack.
        let ingest = pad_layer_names(
            &["signal".to_string()],
            &LayerMap::from_names(vec!["F.Cu".into(), "In1.Cu".into(), "B.Cu".into()]),
        );
        assert_eq!(ingest, vec!["F.Cu", "In1.Cu", "B.Cu"]);
    }
}
