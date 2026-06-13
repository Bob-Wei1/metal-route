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
//! The `(resolution <unit> <divisor>)` header declares the unit and divisor used
//! by every raw integer coordinate in the file. A raw value is converted to mm
//! by `raw * unit_to_mm(unit) / divisor`. For the common `(resolution um 10)`
//! header that is `raw / 10 / 1000` mm (raw values are in 0.1 µm).
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
use mr_srj::{Bounds, Connection, Obstacle, Point, SimpleRouteJson};

/// Default trace width in mm when the DSN carries no `(rule (width N))`.
const DEFAULT_TRACE_WIDTH_MM: f64 = 0.15;

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

/// A representative pad size (mm) for a padstack.
#[derive(Debug, Clone, Copy)]
struct PadSize {
    w: f64,
    h: f64,
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

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a Specctra DSN document into a [`SimpleRouteJson`] plus [`ParseStats`].
///
/// See the module docs for the conversion rules. Returns an error if required
/// sections (resolution, board boundary) are missing or malformed.
pub fn dsn_to_srj_with_stats(dsn_text: &str) -> Result<(SimpleRouteJson, ParseStats)> {
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

    // (resolution um 10) -> mm-per-rawunit.
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
    if divisor == 0.0 {
        bail!("DSN resolution divisor is zero");
    }
    let mm_per_raw = unit_to_mm(unit)? / divisor;
    let to_mm = |raw: f64| raw * mm_per_raw;

    let structure = pcb
        .child_named("structure")
        .ok_or_else(|| anyhow!("DSN missing (structure ...)"))?;

    // Layer count: number of (layer NAME (type signal|power) ...) entries.
    let layer_count = structure.children_named("layer").count() as u32;
    let layer_count = layer_count.max(1);

    // Board bounds from (boundary (path pcb <ap> x1 y1 x2 y2 ...)) or
    // (boundary (rect pcb x1 y1 x2 y2)).
    let bounds = parse_bounds(structure, &to_mm)?;

    // Minimum trace width: structure-level (rule (width N)) wins, else default.
    let min_trace_width_mm =
        parse_rule_width(pcb, structure, &to_mm).unwrap_or(DEFAULT_TRACE_WIDTH_MM);

    // Padstack pad sizes: name -> representative (w,h) in mm.
    let pad_sizes = parse_padstacks(pcb, &to_mm)?;

    // Footprint images: image-id -> pins.
    let images = parse_images(pcb, &to_mm)?;

    // Placed components.
    let placements = parse_placements(pcb, &to_mm)?;

    // Build absolute pin positions: (ref, pin-id) -> (x_mm, y_mm, padstack).
    let mut pin_pos: HashMap<(String, String), (f64, f64, String)> = HashMap::new();
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
            pin_pos.insert(
                (pl.reference.clone(), pin.id.clone()),
                (ax, ay, pin.padstack.clone()),
            );
        }
    }

    // Obstacles: one rect per placed pad we know the position of.
    let mut obstacles: Vec<Obstacle> = Vec::with_capacity(pin_pos.len());
    for ((reference, pin_id), (ax, ay, padstack)) in &pin_pos {
        let size = pad_sizes
            .get(padstack)
            .copied()
            .unwrap_or(PadSize { w: 0.0, h: 0.0 });
        // Skip zero-size pads (no geometry to mask).
        if size.w <= 0.0 || size.h <= 0.0 {
            continue;
        }
        obstacles.push(Obstacle {
            kind: "rect".to_string(),
            center: Point {
                x: *ax,
                y: *ay,
                layer: Some("top".to_string()),
            },
            width: size.w,
            height: size.h,
            layers: vec!["top".to_string()],
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
            if let Some((ax, ay, _)) = pin_pos.get(&(reference.clone(), pin_id.clone())) {
                points.push(Point {
                    x: *ax,
                    y: *ay,
                    layer: Some("top".to_string()),
                });
            }
        }
        if points.len() < 2 {
            nets_skipped_small += 1;
            continue;
        }
        connections.push(Connection {
            name: net_name.clone(),
            points_to_connect: points,
        });
    }

    let stats = ParseStats {
        layers: layer_count,
        components: placements.len(),
        pads,
        nets: connections.len(),
        nets_skipped_small,
        board_w_mm: bounds.max_x - bounds.min_x,
        board_h_mm: bounds.max_y - bounds.min_y,
        min_trace_width_mm,
    };

    let srj = SimpleRouteJson {
        layer_count,
        min_trace_width: Some(min_trace_width_mm),
        obstacles,
        connections,
        bounds,
    };

    Ok((srj, stats))
}

/// Translate `base_mm` (already mm) by `offset_mm` (already mm). The placement
/// position is in raw units; this helper keeps the call sites readable. Both are
/// expected pre-converted to mm by the caller.
#[inline]
fn to_mm_translate(base_mm: f64, offset_mm: f64) -> f64 {
    base_mm + offset_mm
}

/// Parse a DSN document into a [`SimpleRouteJson`], discarding stats.
pub fn dsn_to_srj(dsn_text: &str) -> Result<SimpleRouteJson> {
    Ok(dsn_to_srj_with_stats(dsn_text)?.0)
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------

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

/// First `(rule (width N))` found at structure or pcb level, converted to mm.
fn parse_rule_width(pcb: &Sexpr, structure: &Sexpr, to_mm: &impl Fn(f64) -> f64) -> Option<f64> {
    let find = |node: &Sexpr| -> Option<f64> {
        let rule = node.child_named("rule")?;
        let width = rule.child_named("width")?;
        let raw = width.as_list()?.get(1)?.as_atom()?.parse::<f64>().ok()?;
        Some(to_mm(raw))
    };
    find(structure).or_else(|| find(pcb))
}

/// Padstack name -> representative pad size (mm). Uses the first shape found.
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
        // Find the first (shape (circle ...)|(rect ...)) we can size.
        let mut size: Option<PadSize> = None;
        for shape in ps.children_named("shape") {
            if let Some(circle) = shape.child_named("circle") {
                // (circle LAYER dia [x y]).
                let nums: Vec<f64> = circle
                    .as_list()
                    .unwrap()
                    .iter()
                    .skip(2)
                    .filter_map(|n| n.as_atom())
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if let Some(&dia) = nums.first() {
                    let d = to_mm(dia);
                    size = Some(PadSize { w: d, h: d });
                    break;
                }
            } else if let Some(rect) = shape.child_named("rect") {
                // (rect LAYER x1 y1 x2 y2).
                let nums: Vec<f64> = rect
                    .as_list()
                    .unwrap()
                    .iter()
                    .skip(2)
                    .filter_map(|n| n.as_atom())
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if nums.len() >= 4 {
                    let w = to_mm((nums[2] - nums[0]).abs());
                    let h = to_mm((nums[3] - nums[1]).abs());
                    size = Some(PadSize { w, h });
                    break;
                }
            }
        }
        if let Some(s) = size {
            sizes.insert(name, s);
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
            image.pins.push(ImagePin {
                id,
                off_x: to_mm(relx),
                off_y: to_mm(rely),
                padstack,
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

/// Nets: `(network (net "NAME" (pins REF-PIN ...)))`. Returns net name ->
/// list of `(ref, pin-id)` split on the LAST `-`. Class lists are ignored.
fn parse_nets(pcb: &Sexpr) -> Result<Vec<(String, Vec<(String, String)>)>> {
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
    /// Resolution `um 1000` -> mm-per-raw = 0.001/1000 = 1e-6? No: unit_to_mm(um)
    /// = 0.001, divisor 1000 -> 1e-6 mm per raw. We instead use `mm 1000` so raw
    /// values are in micrometres-ish and easy to reason about: mm-per-raw =
    /// 1/1000 = 0.001, i.e. raw is in µm. So raw 1000 = 1mm.
    const SYNTH: &str = r#"
    (pcb "synth.dsn"
      (parser
        (string_quote ")
        (space_in_quoted_tokens on)
      )
      (resolution mm 1000)
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
          (resolution mm 1000)
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
        // um/10 -> 1e-4 mm per raw. 200000 -> 20mm. -100000 -> -10mm.
        assert!((srj.bounds.max_x - 20.0).abs() < 1e-9);
        assert!((srj.bounds.min_y + 10.0).abs() < 1e-9);
        assert!((srj.bounds.min_x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn missing_resolution_is_an_error() {
        let dsn =
            r#"(pcb "x" (structure (layer F.Cu (type signal)) (boundary (rect pcb 0 0 1 1))))"#;
        assert!(dsn_to_srj(dsn).is_err());
    }
}
