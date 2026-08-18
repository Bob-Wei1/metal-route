//! `mr-drc` — a geometric design-rule check for routed boards.
//!
//! This crate is deliberately *pure geometry*: it knows nothing about grids,
//! rasterisation, DSN, or the routers. A caller (today `mr-cli`) builds a
//! [`DrcBoard`] — net-tagged copper features on a physical layer stack, plus the
//! [`DrcRules`] to enforce — and [`DrcBoard::check`] returns every [`Violation`].
//!
//! Scope is intentionally narrow: only the violation classes metalroute can
//! actually produce ([`ViolationClass`]). It is **not** a full KiCad-equivalent
//! DRC (no silkscreen, soldermask, courtyard, zone-fill connectivity, diff-pairs,
//! length matching, footprint checks). The optional `kicad-cli pcb drc` cross-check
//! in the benchmark suite is what validates that we agree with KiCad on the classes
//! we *do* check.
//!
//! All lengths are in the board's continuous units (millimetres for our DSN flow).
//!
//! ## Determinism
//!
//! [`DrcBoard::check`] must return a deterministically ordered `Vec<Violation>`
//! (sort by `class`, then `layer`, then quantised `location`, then `nets`) so the
//! checker can be diffed across runs and against a future GPU implementation via the
//! `mr-oracle` equivalence pattern.
//!
//! ## Performance
//!
//! Clearance checking must be near-linear: bin every feature into a uniform grid of
//! cell size `clearance + 2 * max_feature_extent` and only test feature pairs that
//! share or neighbour a bin. The typed physical-via stream first collapses exact
//! duplicate producer records and retains this bound for the producer's bounded
//! sibling-representation multiplicity; adversarially many distinct records at one
//! site are outside that input contract. A naïve O(n²) full-board sweep is never
//! used.

use serde::{Deserialize, Serialize};

/// Producer-compatible tolerance for recognizing sibling same-net via records as
/// one physical drill location. This is deliberately much wider than the generic
/// floating-point clearance comparison epsilon.
const SAME_VIA_LOCATION_EPS: f64 = 5e-3;

/// The physical role of a copper layer in the stackup, indexed top (0) → bottom.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// A routable signal layer.
    Signal,
    /// A poured power/ground plane carrying `net` across the whole board area.
    /// A via that crosses this layer shorts to it unless it is the same net or the
    /// via carries a sufficient antipad (see [`Via::antipad_radius`]).
    Plane { net: String },
}

/// A straight copper trace segment on one layer, owned by `net`, drawn `width` wide
/// (so its half-width `width/2` is the clearance-relevant inflation radius).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub net: String,
    pub layer: u32,
    pub a: (f64, f64),
    pub b: (f64, f64),
    pub width: f64,
}

/// An axis-aligned rectangular copper pad on one layer.
///
/// `net == None` means the pad's net is unknown; the checker then treats it as
/// conflicting with *every* other net (conservative — never hides a real short).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    pub net: Option<String>,
    pub layer: u32,
    pub center: (f64, f64),
    pub width: f64,
    pub height: f64,
}

/// A via spanning the inclusive physical layer range `[from_layer, to_layer]`.
///
/// `pad_diameter` is the annular copper pad; `drill_diameter` is the hole. On every
/// layer the barrel passes through, the via presents copper of `pad_diameter` for
/// clearance purposes. `antipad_radius` is the clearance hole carved into any plane
/// it crosses: `None` (or too small) means the plane copper is not relieved and the
/// via shorts to a foreign plane (the M1 baseline state); `Some(r)` large enough
/// clears the short (the M2 fix).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Via {
    pub net: String,
    pub center: (f64, f64),
    pub pad_diameter: f64,
    pub drill_diameter: f64,
    pub from_layer: u32,
    pub to_layer: u32,
    pub antipad_radius: Option<f64>,
}

/// The design-rule constraints to enforce, all in board units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrcRules {
    /// Minimum copper-to-copper spacing between features of different nets.
    pub clearance: f64,
    /// Minimum annular clearance a plane must give a foreign via barrel: a via
    /// crossing a foreign plane is clean iff `antipad_radius >= drill/2 + plane_antipad`.
    pub plane_antipad: f64,
    /// Minimum via annular ring `(pad_diameter - drill_diameter) / 2`.
    pub min_annular_ring: f64,
}

/// The full physical board to check: a layer stack plus every copper feature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrcBoard {
    /// Physical copper layers, index 0 = top. Signal vs. plane (with net).
    pub layers: Vec<LayerKind>,
    pub segments: Vec<Segment>,
    pub pads: Vec<Pad>,
    pub vias: Vec<Via>,
    pub rules: DrcRules,
}

/// The class of a reported violation — only the classes metalroute can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationClass {
    /// Two different-net copper features (track/via-pad/pad) are closer than
    /// `clearance` on the same layer.
    Clearance,
    /// A via barrel crosses a plane of a different net without a sufficient antipad.
    ViaThroughPlane,
    /// A via's annular ring is below `min_annular_ring`.
    AnnularRing,
}

/// One reported violation. `nets` holds the two conflicting nets (for
/// [`ViolationClass::ViaThroughPlane`], `nets.0` is the via net and `nets.1` the
/// plane net; for [`ViolationClass::AnnularRing`], `nets.1` is empty). `measured`
/// is the offending value and `required` the rule it broke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    pub class: ViolationClass,
    pub layer: u32,
    pub location: (f64, f64),
    pub nets: (String, String),
    pub measured: f64,
    pub required: f64,
}

/// Violation counts by class, for baselines and human output.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrcSummary {
    pub total: usize,
    pub clearance: usize,
    pub via_through_plane: usize,
    pub annular_ring: usize,
}

impl DrcSummary {
    /// Tally a violation slice by class.
    pub fn of(violations: &[Violation]) -> Self {
        let mut s = DrcSummary {
            total: violations.len(),
            ..Default::default()
        };
        for v in violations {
            match v.class {
                ViolationClass::Clearance => s.clearance += 1,
                ViolationClass::ViaThroughPlane => s.via_through_plane += 1,
                ViolationClass::AnnularRing => s.annular_ring += 1,
            }
        }
        s
    }

    /// True when the board is DRC-clean.
    pub fn is_clean(&self) -> bool {
        self.total == 0
    }
}

/// Return the inclusive portion of a via span that lies on the physical stack.
///
/// Clamp the upper endpoint before constructing an inclusive range: checking the
/// bound inside `lo..=hi` would still walk billions of nonexistent layers for a
/// malformed span ending at `u32::MAX`. An empty stack, or a span wholly above the
/// stack, has no physical layers to stamp.
fn in_stack_via_span(via: &Via, layer_count: u32) -> Option<(u32, u32)> {
    let last_layer = layer_count.checked_sub(1)?;
    let lo = via.from_layer.min(via.to_layer);
    if lo > last_layer {
        return None;
    }
    let hi = via.from_layer.max(via.to_layer).min(last_layer);
    Some((lo, hi))
}

impl DrcBoard {
    /// Run every check and return all violations in deterministic order.
    ///
    /// Stream A implements: clearance (uniform-grid spatial index, per layer,
    /// different-net pairs only), via-through-plane (antipad-aware), annular ring.
    pub fn check(&self) -> Vec<Violation> {
        self.check_with_optional_pad_clearances(None)
    }

    /// Run every check while applying feature-pair-specific edge clearances for
    /// trace↔pad, via-annulus↔trace, via-annulus↔pad, and (when supplied)
    /// pad↔pad pairs. The SRJ trace-to-pad rule also governs via↔trace, matching
    /// the producer checker. An
    /// optional via-hole↔via-hole fabrication rule is net-independent and is
    /// enforced once per physical via pair alongside the generic annular-pad rule.
    /// The stricter rule owns the finding, with drill-rule findings reported in
    /// actual drill-edge units. All other copper pairings continue to use
    /// [`DrcRules::clearance`].
    ///
    /// Invalid pair values fall back to the generic rule. This lets typed board
    /// formats project their pad rules without globally over-reporting legal
    /// trace↔trace or via↔trace spacing.
    pub fn check_with_pad_clearances(
        &self,
        trace_to_pad_clearance: f64,
        via_to_pad_clearance: f64,
        pad_to_pad_clearance: Option<f64>,
        via_hole_to_hole_clearance: Option<f64>,
    ) -> Vec<Violation> {
        let valid_or_generic = |value: f64| {
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                self.rules.clearance
            }
        };
        self.check_with_optional_pad_clearances(Some(PadClearances {
            trace_to_pad: valid_or_generic(trace_to_pad_clearance),
            via_to_pad: valid_or_generic(via_to_pad_clearance),
            pad_to_pad: pad_to_pad_clearance
                .map(valid_or_generic)
                .unwrap_or(self.rules.clearance),
            via_hole_to_hole: via_hole_to_hole_clearance.map(valid_or_generic),
        }))
    }

    fn check_with_optional_pad_clearances(
        &self,
        pad_clearances: Option<PadClearances>,
    ) -> Vec<Violation> {
        let mut out = Vec::new();
        let hole_clearance = pad_clearances.and_then(|rules| rules.via_hole_to_hole);
        let via_sites = hole_clearance
            .map(|_| physical_via_sites(&self.vias))
            .unwrap_or_default();
        self.check_clearance(&mut out, pad_clearances, &via_sites);
        if let Some(clearance) = hole_clearance {
            self.check_canonical_via_pair_rules(&mut out, clearance, &via_sites);
        }
        self.check_via_through_plane(&mut out);
        self.check_annular_ring(&mut out);
        sort_violations(&mut out);
        out
    }

    /// Clearance: bin every copper feature into a per-layer uniform grid and test
    /// only different-net pairs that share or neighbour a bin (near-linear).
    fn check_clearance(
        &self,
        out: &mut Vec<Violation>,
        pad_clearances: Option<PadClearances>,
        via_sites: &[PhysicalViaSite],
    ) {
        #[derive(Clone, Debug)]
        struct IndexedFeature {
            feature: Feature,
            logical_id: usize,
            via_site: Option<usize>,
        }

        // Collect the copper features present on each physical layer. A via pad is
        // present (as a circle) on every layer in its inclusive span.
        let mut by_layer: std::collections::HashMap<u32, Vec<IndexedFeature>> =
            std::collections::HashMap::new();
        let layer_count = self.layers.len() as u32;
        let hole_rule_active = pad_clearances
            .and_then(|rules| rules.via_hole_to_hole)
            .is_some();
        let mut next_logical_id = 0usize;

        for s in &self.segments {
            by_layer.entry(s.layer).or_default().push(IndexedFeature {
                feature: Feature::segment(s),
                logical_id: next_logical_id,
                via_site: None,
            });
            next_logical_id += 1;
        }
        for p in &self.pads {
            by_layer.entry(p.layer).or_default().push(IndexedFeature {
                feature: Feature::pad(p),
                logical_id: next_logical_id,
                via_site: None,
            });
            next_logical_id += 1;
        }
        if hole_rule_active {
            for (site_id, site) in via_sites.iter().enumerate() {
                let logical_id = next_logical_id + site_id;
                for v in &site.representations {
                    if let Some((lo, hi)) = in_stack_via_span(v, layer_count) {
                        for layer in lo..=hi {
                            by_layer.entry(layer).or_default().push(IndexedFeature {
                                feature: Feature::via(v),
                                logical_id,
                                via_site: Some(site_id),
                            });
                        }
                    }
                }
            }
        } else {
            // Preserve the historical no-hole checker byte/count semantics: each
            // raw via record remains an independent copper feature. Physical-site
            // identity activates only with an explicit fabrication-hole rule.
            for v in &self.vias {
                let logical_id = next_logical_id;
                next_logical_id += 1;
                if let Some((lo, hi)) = in_stack_via_span(v, layer_count) {
                    for layer in lo..=hi {
                        by_layer.entry(layer).or_default().push(IndexedFeature {
                            feature: Feature::via(v),
                            logical_id,
                            via_site: None,
                        });
                    }
                }
            }
        }

        // Cell size = clearance + 2·(largest feature extent), so any conflicting
        // pair shares or neighbours a bin. A conflicting pair's centroids are at most
        // `clearance + extent_a + extent_b ≤ clearance + 2·max_extent` apart; sizing
        // the cell to that bound keeps their bin indices within ±1 on each axis, so
        // the 3×3 neighbourhood scan below is exhaustive (a `+ max_extent` cell would
        // let a pair land two bins apart and be missed).
        let max_extent = by_layer
            .values()
            .flat_map(|fs| fs.iter())
            .map(|indexed| indexed.feature.extent())
            .fold(0.0_f64, f64::max);
        let largest_clearance = pad_clearances.map_or(self.rules.clearance, |rules| {
            self.rules
                .clearance
                .max(rules.trace_to_pad)
                .max(rules.via_to_pad)
                .max(rules.pad_to_pad)
        });
        let cell = (largest_clearance + 2.0 * max_extent).max(f64::MIN_POSITIVE);
        let mut aggregated: std::collections::BTreeMap<(u32, usize, usize), Violation> =
            std::collections::BTreeMap::new();
        // Stable iteration order over layers for determinism of `out` before sort.
        let mut layers: Vec<u32> = by_layer.keys().copied().collect();
        layers.sort_unstable();

        for layer in layers {
            let feats = &by_layer[&layer];
            // Bin by the feature centroid cell.
            let mut bins: std::collections::HashMap<(i64, i64), Vec<usize>> =
                std::collections::HashMap::new();
            for (i, f) in feats.iter().enumerate() {
                let (cx, cy) = f.feature.centroid();
                let key = ((cx / cell).floor() as i64, (cy / cell).floor() as i64);
                bins.entry(key).or_default().push(i);
            }

            // To report each unordered pair once, only test feature j against i when
            // (i < j). We scan each feature's own bin and the 3x3 neighbourhood.
            for (&(bx, by), idxs) in &bins {
                for &i in idxs {
                    for ndx in -1..=1 {
                        for ndy in -1..=1 {
                            if let Some(neigh) = bins.get(&(bx + ndx, by + ndy)) {
                                for &j in neigh {
                                    if i >= j {
                                        continue;
                                    }
                                    let a = &feats[i];
                                    let b = &feats[j];
                                    if a.logical_id == b.logical_id {
                                        continue;
                                    }
                                    // With a declared hole rule, all via-site pairs
                                    // are owned by the centralized physical-site
                                    // arbitration below. Raw representation pairs
                                    // must not emit duplicate or contradictory rows.
                                    if hole_rule_active
                                        && a.via_site.is_some()
                                        && b.via_site.is_some()
                                    {
                                        continue;
                                    }
                                    let mut candidate = Vec::with_capacity(1);
                                    self.test_clearance_pair_with_pad_rules(
                                        layer,
                                        &a.feature,
                                        &b.feature,
                                        pad_clearances,
                                        &mut candidate,
                                    );
                                    let Some(candidate) = candidate.pop() else {
                                        continue;
                                    };
                                    let key = (
                                        layer,
                                        a.logical_id.min(b.logical_id),
                                        a.logical_id.max(b.logical_id),
                                    );
                                    match aggregated.entry(key) {
                                        std::collections::btree_map::Entry::Vacant(entry) => {
                                            entry.insert(candidate);
                                        }
                                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                                            let current = entry.get();
                                            let candidate_deficit =
                                                candidate.measured - candidate.required;
                                            let current_deficit =
                                                current.measured - current.required;
                                            if candidate_deficit < current_deficit
                                                || (candidate_deficit == current_deficit
                                                    && violation_cmp(&candidate, current).is_lt())
                                            {
                                                entry.insert(candidate);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out.extend(aggregated.into_values());
    }

    #[cfg(test)]
    fn test_clearance_pair(&self, layer: u32, a: &Feature, b: &Feature, out: &mut Vec<Violation>) {
        self.test_clearance_pair_with_pad_rules(layer, a, b, None, out);
    }

    fn test_clearance_pair_with_pad_rules(
        &self,
        layer: u32,
        a: &Feature,
        b: &Feature,
        pad_clearances: Option<PadClearances>,
        out: &mut Vec<Violation>,
    ) {
        if !nets_conflict(&a.net, &b.net) {
            return;
        }

        // The centroid grid deliberately over-approximates candidate pairs. Reject
        // pairs whose copper AABBs are already separated by at least the violating
        // threshold before running the more expensive exact shape-distance code.
        // This is only enabled for a positive threshold: an overlapping capsule can
        // have a negative exact gap while its (overlapping) AABBs have zero gap.
        let required = required_clearance(self.rules.clearance, pad_clearances, a, b);
        let violation_threshold = required - EPS;
        if aabbs_separated_by_at_least(a, b, violation_threshold) {
            return;
        }

        self.report_clearance_pair(layer, a, b, required, out);
    }

    /// Centralized via-site arbitration for a declared net-independent drill rule.
    /// Raw representations feed one spatial broadphase, but at most one row is
    /// emitted per canonical physical-site pair. Exact duplicate records are
    /// collapsed first; the remaining work is near-linear for the producer's
    /// bounded sibling-representation multiplicity (not for adversarially many
    /// distinct representations at one site). Hole owns an equality tie; a
    /// foreign-net annular-copper row owns only when its *violating actual pair*
    /// has a strictly larger centre requirement.
    fn check_canonical_via_pair_rules(
        &self,
        out: &mut Vec<Violation>,
        hole_clearance: f64,
        sites: &[PhysicalViaSite],
    ) {
        #[derive(Clone, Debug)]
        struct Candidate {
            violation: Violation,
            centre_requirement: f64,
        }

        #[derive(Default)]
        struct PairCandidates {
            hole: Option<Candidate>,
            copper: Option<Candidate>,
        }

        let representations: Vec<(usize, &Via)> = sites
            .iter()
            .enumerate()
            .flat_map(|(site_id, site)| site.representations.iter().map(move |via| (site_id, via)))
            .collect();
        if representations.len() < 2 {
            return;
        }
        let max_drill_radius = representations
            .iter()
            .map(|(_, via)| via.drill_diameter / 2.0)
            .fold(0.0_f64, f64::max);
        let max_pad_radius = representations
            .iter()
            .map(|(_, via)| via.pad_diameter / 2.0)
            .fold(0.0_f64, f64::max);
        let cell = (hole_clearance + 2.0 * max_drill_radius)
            .max(self.rules.clearance + 2.0 * max_pad_radius)
            .max(f64::MIN_POSITIVE);
        let mut bins: std::collections::HashMap<(i64, i64), Vec<usize>> =
            std::collections::HashMap::new();
        for (index, (_, via)) in representations.iter().enumerate() {
            let key = (
                (via.center.0 / cell).floor() as i64,
                (via.center.1 / cell).floor() as i64,
            );
            bins.entry(key).or_default().push(index);
        }

        let mut pairs: std::collections::BTreeMap<(usize, usize), PairCandidates> =
            std::collections::BTreeMap::new();
        let retain_worst = |slot: &mut Option<Candidate>, candidate: Candidate| {
            let replace = slot.as_ref().is_none_or(|current| {
                let candidate_deficit = candidate.violation.measured - candidate.violation.required;
                let current_deficit = current.violation.measured - current.violation.required;
                candidate
                    .centre_requirement
                    .total_cmp(&current.centre_requirement)
                    .is_gt()
                    || (candidate.centre_requirement == current.centre_requirement
                        && (candidate_deficit.total_cmp(&current_deficit).is_lt()
                            || (candidate_deficit == current_deficit
                                && violation_cmp(&candidate.violation, &current.violation)
                                    .is_lt())))
            });
            if replace {
                *slot = Some(candidate);
            }
        };
        let sorted_nets = |a: &Via, b: &Via| {
            if a.net <= b.net {
                (a.net.clone(), b.net.clone())
            } else {
                (b.net.clone(), a.net.clone())
            }
        };
        let layer_count = self.layers.len() as u32;

        for (&(bx, by), indices) in &bins {
            for &i in indices {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let Some(neighbours) = bins.get(&(bx + dx, by + dy)) else {
                            continue;
                        };
                        for &j in neighbours {
                            if i >= j {
                                continue;
                            }
                            let (a_site, a) = representations[i];
                            let (b_site, b) = representations[j];
                            if a_site == b_site {
                                continue;
                            }
                            // First-anchor canonical sites are intentionally not
                            // transitively clustered. Even across two such IDs,
                            // however, the producer treats a same-net raw pair at
                            // <= 0.005 mm as one coincident representation.
                            if same_physical_via_location(a, b) {
                                continue;
                            }
                            let key = (a_site.min(b_site), a_site.max(b_site));
                            let pair = pairs.entry(key).or_default();
                            let distance = dist(a.center, b.center);
                            let hole_measured =
                                distance - a.drill_diameter / 2.0 - b.drill_diameter / 2.0;
                            if hole_measured < hole_clearance - EPS {
                                retain_worst(
                                    &mut pair.hole,
                                    Candidate {
                                        violation: Violation {
                                            class: ViolationClass::Clearance,
                                            layer: a
                                                .from_layer
                                                .min(a.to_layer)
                                                .min(b.from_layer.min(b.to_layer)),
                                            location: (
                                                (a.center.0 + b.center.0) / 2.0,
                                                (a.center.1 + b.center.1) / 2.0,
                                            ),
                                            nets: sorted_nets(a, b),
                                            measured: hole_measured,
                                            required: hole_clearance,
                                        },
                                        centre_requirement: a.drill_diameter / 2.0
                                            + b.drill_diameter / 2.0
                                            + hole_clearance,
                                    },
                                );
                            }

                            if a.net == b.net {
                                continue;
                            }
                            let overlap = match (
                                in_stack_via_span(a, layer_count),
                                in_stack_via_span(b, layer_count),
                            ) {
                                (Some((a_lo, a_hi)), Some((b_lo, b_hi)))
                                    if a_lo <= b_hi && b_lo <= a_hi =>
                                {
                                    Some(a_lo.max(b_lo))
                                }
                                _ => None,
                            };
                            let Some(layer) = overlap else {
                                continue;
                            };
                            let copper_measured =
                                distance - a.pad_diameter / 2.0 - b.pad_diameter / 2.0;
                            if copper_measured < self.rules.clearance - EPS {
                                retain_worst(
                                    &mut pair.copper,
                                    Candidate {
                                        violation: Violation {
                                            class: ViolationClass::Clearance,
                                            layer,
                                            location: (
                                                (a.center.0 + b.center.0) / 2.0,
                                                (a.center.1 + b.center.1) / 2.0,
                                            ),
                                            nets: sorted_nets(a, b),
                                            measured: copper_measured,
                                            required: self.rules.clearance,
                                        },
                                        centre_requirement: a.pad_diameter / 2.0
                                            + b.pad_diameter / 2.0
                                            + self.rules.clearance,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        for pair in pairs.into_values() {
            let chosen = match (pair.hole, pair.copper) {
                (Some(hole), Some(copper))
                    if copper.centre_requirement > hole.centre_requirement + EPS =>
                {
                    copper
                }
                (Some(hole), _) => hole,
                (None, Some(copper)) => copper,
                (None, None) => continue,
            };
            out.push(chosen.violation);
        }
    }

    #[cfg(test)]
    fn test_clearance_pair_exact(
        &self,
        layer: u32,
        a: &Feature,
        b: &Feature,
        out: &mut Vec<Violation>,
    ) {
        if !nets_conflict(&a.net, &b.net) {
            return;
        }
        self.report_clearance_pair(layer, a, b, self.rules.clearance, out);
    }

    fn report_clearance_pair(
        &self,
        layer: u32,
        a: &Feature,
        b: &Feature,
        required: f64,
        out: &mut Vec<Violation>,
    ) {
        let gap = feature_gap(a, b);
        if gap < required - EPS {
            let (ax, ay) = a.centroid();
            let (bx, by) = b.centroid();
            // Canonicalise the net pair so the row is independent of which feature
            // happened to get the lower bin index (input-order independence).
            let na = a.net.clone().unwrap_or_default();
            let nb = b.net.clone().unwrap_or_default();
            let nets = if na <= nb { (na, nb) } else { (nb, na) };
            out.push(Violation {
                class: ViolationClass::Clearance,
                layer,
                location: ((ax + bx) * 0.5, (ay + by) * 0.5),
                nets,
                measured: gap,
                required,
            });
        }
    }

    /// Via-through-plane: a via barrel shorts every foreign-net plane in its
    /// inclusive span unless it carries a sufficient antipad.
    fn check_via_through_plane(&self, out: &mut Vec<Violation>) {
        let layer_count = self.layers.len() as u32;
        let required = |v: &Via| v.drill_diameter / 2.0 + self.rules.plane_antipad;
        for v in &self.vias {
            if let Some((lo, hi)) = in_stack_via_span(v, layer_count) {
                for layer in lo..=hi {
                    let Some(LayerKind::Plane { net }) = self.layers.get(layer as usize) else {
                        continue;
                    };
                    if *net == v.net {
                        continue;
                    }
                    let req = required(v);
                    let antipad = v.antipad_radius.unwrap_or(0.0);
                    let clean = v.antipad_radius.is_some_and(|r| r >= req - EPS);
                    if !clean {
                        out.push(Violation {
                            class: ViolationClass::ViaThroughPlane,
                            layer,
                            location: v.center,
                            nets: (v.net.clone(), net.clone()),
                            measured: antipad,
                            required: req,
                        });
                    }
                }
            }
        }
    }

    /// Annular ring: `(pad_diameter - drill_diameter)/2` must meet `min_annular_ring`.
    fn check_annular_ring(&self, out: &mut Vec<Violation>) {
        for v in &self.vias {
            let measured = (v.pad_diameter - v.drill_diameter) / 2.0;
            if measured < self.rules.min_annular_ring - EPS {
                out.push(Violation {
                    class: ViolationClass::AnnularRing,
                    layer: v.from_layer.min(v.to_layer),
                    location: v.center,
                    nets: (v.net.clone(), String::new()),
                    measured,
                    required: self.rules.min_annular_ring,
                });
            }
        }
    }
}

/// Floating-point slack: a gap counts as a violation only when it is strictly
/// below the rule by more than this, so features placed exactly at the rule pass.
const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
struct PadClearances {
    trace_to_pad: f64,
    via_to_pad: f64,
    pad_to_pad: f64,
    via_hole_to_hole: Option<f64>,
}

fn required_clearance(
    generic: f64,
    pad_clearances: Option<PadClearances>,
    a: &Feature,
    b: &Feature,
) -> f64 {
    let Some(rules) = pad_clearances else {
        return generic;
    };
    match (&a.shape, &b.shape) {
        (Shape::Segment { .. }, Shape::Rect { .. })
        | (Shape::Rect { .. }, Shape::Segment { .. }) => rules.trace_to_pad,
        (Shape::Segment { .. }, Shape::Point { .. })
        | (Shape::Point { .. }, Shape::Segment { .. }) => rules.trace_to_pad,
        (Shape::Point { .. }, Shape::Rect { .. }) | (Shape::Rect { .. }, Shape::Point { .. }) => {
            rules.via_to_pad
        }
        (Shape::Rect { .. }, Shape::Rect { .. }) => rules.pad_to_pad,
        _ => generic,
    }
}

/// Two nets conflict when they are different. A `None` net (unknown pad) is treated
/// as a distinct always-foreign net, so it conflicts with everything — including
/// another `None`.
fn nets_conflict(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x != y,
        _ => true,
    }
}

fn physical_via_representation_cmp(a: &Via, b: &Via) -> std::cmp::Ordering {
    a.net
        .cmp(&b.net)
        .then_with(|| a.center.0.total_cmp(&b.center.0))
        .then_with(|| a.center.1.total_cmp(&b.center.1))
        .then_with(|| {
            a.from_layer
                .min(a.to_layer)
                .cmp(&b.from_layer.min(b.to_layer))
        })
        .then_with(|| {
            a.from_layer
                .max(a.to_layer)
                .cmp(&b.from_layer.max(b.to_layer))
        })
        .then_with(|| a.pad_diameter.total_cmp(&b.pad_diameter))
        .then_with(|| a.drill_diameter.total_cmp(&b.drill_diameter))
        .then_with(|| match (a.antipad_radius, b.antipad_radius) {
            (Some(a), Some(b)) => a.total_cmp(&b),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        })
}

fn same_physical_via_location(a: &Via, b: &Via) -> bool {
    a.net == b.net && dist(a.center, b.center) <= SAME_VIA_LOCATION_EPS
}

#[derive(Clone, Debug)]
struct PhysicalViaSite {
    /// Deterministic canonical record used only for broadphase placement and the
    /// union layer span / maximum fabrication geometry.
    physical: Via,
    /// Original producer records. Exact checks always select the worst actual
    /// representation pair instead of measuring synthetic canonical geometry.
    representations: Vec<Via>,
}

/// Canonicalize sibling same-net via records into physical drill sites in
/// deterministic near-linear time. The first sorted representation remains the
/// site centre; later records within 0.005 mm merge without moving it, matching
/// the producer's pairwise same-location convention rather than transitive
/// clustering.
fn physical_via_sites(vias: &[Via]) -> Vec<PhysicalViaSite> {
    let mut sorted_vias = vias.to_vec();
    sorted_vias.sort_by(physical_via_representation_cmp);
    let mut sites: Vec<PhysicalViaSite> = Vec::with_capacity(sorted_vias.len());
    let mut canonical_bins: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    let mut current_net: Option<String> = None;
    for via in sorted_vias {
        if current_net.as_deref() != Some(via.net.as_str()) {
            canonical_bins.clear();
            current_net = Some(via.net.clone());
        }
        let bin = (
            (via.center.0 / SAME_VIA_LOCATION_EPS).floor() as i64,
            (via.center.1 / SAME_VIA_LOCATION_EPS).floor() as i64,
        );
        let mut canonical_index = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(indices) = canonical_bins.get(&(bin.0 + dx, bin.1 + dy)) {
                    for &index in indices {
                        if same_physical_via_location(&sites[index].physical, &via) {
                            canonical_index = Some(
                                canonical_index.map_or(index, |current: usize| current.min(index)),
                            );
                        }
                    }
                }
            }
        }
        if let Some(index) = canonical_index {
            let existing = &mut sites[index];
            let lo = existing
                .physical
                .from_layer
                .min(existing.physical.to_layer)
                .min(via.from_layer.min(via.to_layer));
            let hi = existing
                .physical
                .from_layer
                .max(existing.physical.to_layer)
                .max(via.from_layer.max(via.to_layer));
            existing.physical.from_layer = lo;
            existing.physical.to_layer = hi;
            existing.physical.pad_diameter = existing.physical.pad_diameter.max(via.pad_diameter);
            existing.physical.drill_diameter =
                existing.physical.drill_diameter.max(via.drill_diameter);
            // Fully identical sibling records are adjacent under the complete
            // deterministic sort above. Collapsing them avoids the common
            // duplicate-soup K² case in typed via-site arbitration without
            // changing the legacy no-hole checker stream.
            if existing.representations.last() != Some(&via) {
                existing.representations.push(via);
            }
        } else {
            let index = sites.len();
            sites.push(PhysicalViaSite {
                physical: via.clone(),
                representations: vec![via],
            });
            canonical_bins.entry(bin).or_default().push(index);
        }
    }
    sites
}

/// A copper feature reduced to its clearance geometry on one layer.
///
/// Every feature is either a capsule (segment inflated by `inflate`), a circle
/// (via pad: a point inflated by its radius), or an axis-aligned rectangle (pad).
/// Modelling capsules and circles as an inflated core lets `feature_gap` compose
/// most cases as `core_gap - inflate_a - inflate_b`.
#[derive(Clone, Debug)]
struct Feature {
    net: Option<String>,
    shape: Shape,
    /// Radius the core shape is inflated by (half-width / pad radius); rects use 0.
    inflate: f64,
}

#[derive(Clone, Debug)]
enum Shape {
    /// A line core from `a` to `b` (inflated → capsule).
    Segment { a: (f64, f64), b: (f64, f64) },
    /// A point core (inflated → circle).
    Point { c: (f64, f64) },
    /// An axis-aligned rectangle, full width/height about its center.
    Rect { c: (f64, f64), w: f64, h: f64 },
}

impl Feature {
    fn segment(s: &Segment) -> Self {
        Feature {
            net: Some(s.net.clone()),
            shape: Shape::Segment { a: s.a, b: s.b },
            inflate: s.width / 2.0,
        }
    }

    fn pad(p: &Pad) -> Self {
        Feature {
            net: p.net.clone(),
            shape: Shape::Rect {
                c: p.center,
                w: p.width,
                h: p.height,
            },
            inflate: 0.0,
        }
    }

    fn via(v: &Via) -> Self {
        Feature {
            net: Some(v.net.clone()),
            shape: Shape::Point { c: v.center },
            inflate: v.pad_diameter / 2.0,
        }
    }

    /// A representative point for binning and reporting.
    fn centroid(&self) -> (f64, f64) {
        match &self.shape {
            Shape::Segment { a, b } => ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5),
            Shape::Point { c } => *c,
            Shape::Rect { c, .. } => *c,
        }
    }

    /// Half the largest planar reach from the centroid, used to size grid cells so
    /// every conflicting pair lands in neighbouring bins.
    fn extent(&self) -> f64 {
        match &self.shape {
            Shape::Segment { a, b } => dist(*a, *b) / 2.0 + self.inflate,
            Shape::Point { .. } => self.inflate,
            Shape::Rect { w, h, .. } => (w.max(*h)) / 2.0,
        }
    }

    /// Conservative copper bounds for broad-phase rejection. Invalid or
    /// non-finite geometry returns `None`, preserving the exact path's behavior.
    fn copper_aabb(&self) -> Option<(f64, f64, f64, f64)> {
        if !self.inflate.is_finite() || self.inflate < 0.0 {
            return None;
        }
        let bounds = match &self.shape {
            Shape::Segment { a, b } => (
                a.0.min(b.0) - self.inflate,
                a.1.min(b.1) - self.inflate,
                a.0.max(b.0) + self.inflate,
                a.1.max(b.1) + self.inflate,
            ),
            Shape::Point { c } => (
                c.0 - self.inflate,
                c.1 - self.inflate,
                c.0 + self.inflate,
                c.1 + self.inflate,
            ),
            Shape::Rect { c, w, h } => {
                if !w.is_finite() || !h.is_finite() || *w < 0.0 || *h < 0.0 {
                    return None;
                }
                (c.0 - w / 2.0, c.1 - h / 2.0, c.0 + w / 2.0, c.1 + h / 2.0)
            }
        };
        if [bounds.0, bounds.1, bounds.2, bounds.3]
            .into_iter()
            .all(f64::is_finite)
        {
            Some(bounds)
        } else {
            None
        }
    }
}

/// True only when one axis alone proves the exact copper gap cannot violate the
/// rule. AABBs enclose their features, so their axis separation is a lower bound
/// on the exact Euclidean gap. Returning false merely falls back to the exact test.
fn aabbs_separated_by_at_least(a: &Feature, b: &Feature, threshold: f64) -> bool {
    if !threshold.is_finite() || threshold <= 0.0 {
        return false;
    }
    let (Some((a_min_x, a_min_y, a_max_x, a_max_y)), Some((b_min_x, b_min_y, b_max_x, b_max_y))) =
        (a.copper_aabb(), b.copper_aabb())
    else {
        return false;
    };
    let x_separation = (a_min_x - b_max_x).max(b_min_x - a_max_x);
    let y_separation = (a_min_y - b_max_y).max(b_min_y - a_max_y);
    x_separation >= threshold || y_separation >= threshold
}

/// Copper-to-copper gap between two features (negative if they overlap). For the
/// capsule/circle cases this is the core-to-core distance minus both inflations;
/// rectangles are handled by the appropriate point/segment-to-rect routine.
fn feature_gap(a: &Feature, b: &Feature) -> f64 {
    match (&a.shape, &b.shape) {
        // Core ⇄ core distances minus inflation (capsules and circles).
        (Shape::Segment { a: p0, b: p1 }, Shape::Segment { a: q0, b: q1 }) => {
            seg_seg_dist(*p0, *p1, *q0, *q1) - a.inflate - b.inflate
        }
        (Shape::Segment { a: p0, b: p1 }, Shape::Point { c }) => {
            point_seg_dist(*c, *p0, *p1) - a.inflate - b.inflate
        }
        (Shape::Point { c }, Shape::Segment { a: p0, b: p1 }) => {
            point_seg_dist(*c, *p0, *p1) - a.inflate - b.inflate
        }
        (Shape::Point { c: c0 }, Shape::Point { c: c1 }) => dist(*c0, *c1) - a.inflate - b.inflate,
        // Rectangle cases: distance from the rect to the other core, minus the
        // other feature's inflation (the rect itself has inflate == 0).
        (Shape::Rect { c, w, h }, Shape::Point { c: p }) => {
            point_rect_gap(*p, *c, *w, *h) - b.inflate
        }
        (Shape::Point { c: p }, Shape::Rect { c, w, h }) => {
            point_rect_gap(*p, *c, *w, *h) - a.inflate
        }
        (Shape::Rect { c, w, h }, Shape::Segment { a: p0, b: p1 }) => {
            seg_rect_gap(*p0, *p1, *c, *w, *h) - b.inflate
        }
        (Shape::Segment { a: p0, b: p1 }, Shape::Rect { c, w, h }) => {
            seg_rect_gap(*p0, *p1, *c, *w, *h) - a.inflate
        }
        (
            Shape::Rect {
                c: c0,
                w: w0,
                h: h0,
            },
            Shape::Rect {
                c: c1,
                w: w1,
                h: h1,
            },
        ) => rect_rect_gap(*c0, *w0, *h0, *c1, *w1, *h1),
    }
}

/// Euclidean point-to-point distance.
pub fn dist(p: (f64, f64), q: (f64, f64)) -> f64 {
    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
}

/// Distance from point `p` to the segment `[a, b]` (0 if `p` lies on it).
pub fn point_seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len2 = abx * abx + aby * aby;
    if len2 <= f64::MIN_POSITIVE {
        return dist(p, a);
    }
    let t = (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len2).clamp(0.0, 1.0);
    let proj = (a.0 + t * abx, a.1 + t * aby);
    dist(p, proj)
}

/// Shortest distance between two segments `[p0,p1]` and `[q0,q1]` (0 if they cross).
///
/// Standard closest-distance-between-two-segments: minimise the squared distance of
/// `s(t) = p0 + t·d1` and `r(u) = q0 + u·d2` over `t,u ∈ [0,1]`, with the degenerate
/// (point) cases handled explicitly.
pub fn seg_seg_dist(p0: (f64, f64), p1: (f64, f64), q0: (f64, f64), q1: (f64, f64)) -> f64 {
    let d1 = (p1.0 - p0.0, p1.1 - p0.1);
    let d2 = (q1.0 - q0.0, q1.1 - q0.1);
    let r = (p0.0 - q0.0, p0.1 - q0.1);
    let a = d1.0 * d1.0 + d1.1 * d1.1; // |d1|^2
    let e = d2.0 * d2.0 + d2.1 * d2.1; // |d2|^2
    let f = d2.0 * r.0 + d2.1 * r.1;

    // Both segments degenerate to points.
    if a <= f64::MIN_POSITIVE && e <= f64::MIN_POSITIVE {
        return dist(p0, q0);
    }
    let (mut s, t);
    if a <= f64::MIN_POSITIVE {
        // First segment is a point.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.0 * r.0 + d1.1 * r.1;
        if e <= f64::MIN_POSITIVE {
            // Second segment is a point.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.0 * d2.0 + d1.1 * d2.1;
            let denom = a * e - b * b;
            s = if denom > f64::MIN_POSITIVE {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut tt = (b * s + f) / e;
            if tt < 0.0 {
                tt = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if tt > 1.0 {
                tt = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            t = tt;
        }
    }
    let cp = (p0.0 + s * d1.0, p0.1 + s * d1.1);
    let cq = (q0.0 + t * d2.0, q0.1 + t * d2.1);
    dist(cp, cq)
}

/// Gap from a point to an axis-aligned rectangle (center `c`, full `w`×`h`); 0 when
/// the point is inside.
pub fn point_rect_gap(p: (f64, f64), c: (f64, f64), w: f64, h: f64) -> f64 {
    let dx = (c.0 - p.0).abs() - w / 2.0;
    let dy = (c.1 - p.1).abs() - h / 2.0;
    let ox = dx.max(0.0);
    let oy = dy.max(0.0);
    if ox == 0.0 && oy == 0.0 {
        // Inside the rectangle.
        0.0
    } else {
        (ox * ox + oy * oy).sqrt()
    }
}

/// Gap from a segment to an axis-aligned rectangle. Sampled against the rectangle's
/// four edges (and inside-test), which is exact for the segment-vs-AABB case.
pub fn seg_rect_gap(p0: (f64, f64), p1: (f64, f64), c: (f64, f64), w: f64, h: f64) -> f64 {
    // If either endpoint is inside, gap is 0.
    if point_rect_gap(p0, c, w, h) == 0.0 || point_rect_gap(p1, c, w, h) == 0.0 {
        return 0.0;
    }
    let (hw, hh) = (w / 2.0, h / 2.0);
    let corners = [
        (c.0 - hw, c.1 - hh),
        (c.0 + hw, c.1 - hh),
        (c.0 + hw, c.1 + hh),
        (c.0 - hw, c.1 + hh),
    ];
    let mut best = f64::INFINITY;
    // Segment-to-each-edge distance.
    for i in 0..4 {
        let e0 = corners[i];
        let e1 = corners[(i + 1) % 4];
        best = best.min(seg_seg_dist(p0, p1, e0, e1));
    }
    best
}

/// Gap between two axis-aligned rectangles (0 if they overlap or touch).
pub fn rect_rect_gap(c0: (f64, f64), w0: f64, h0: f64, c1: (f64, f64), w1: f64, h1: f64) -> f64 {
    let dx = (c0.0 - c1.0).abs() - (w0 + w1) / 2.0;
    let dy = (c0.1 - c1.1).abs() - (h0 + h1) / 2.0;
    let ox = dx.max(0.0);
    let oy = dy.max(0.0);
    (ox * ox + oy * oy).sqrt()
}

/// Class ordering key: explicit so it does not depend on the enum's derive order.
fn class_order(c: ViolationClass) -> u8 {
    match c {
        ViolationClass::Clearance => 0,
        ViolationClass::ViaThroughPlane => 1,
        ViolationClass::AnnularRing => 2,
    }
}

/// Quantise a coordinate to a fixed grid so float jitter can't reorder ties.
fn quantise(x: f64) -> i64 {
    (x / 1e-6).round() as i64
}

/// Total, stable violation ordering: the published class/layer/quantised-location/
/// nets order first, followed by exact geometry and measurement tie-breaks. The
/// latter matter when distinct violations land in the same 1e-6 location bucket;
/// without them stable sort merely preserves HashMap/input discovery order.
fn violation_cmp(a: &Violation, b: &Violation) -> std::cmp::Ordering {
    class_order(a.class)
        .cmp(&class_order(b.class))
        .then_with(|| a.layer.cmp(&b.layer))
        .then_with(|| quantise(a.location.0).cmp(&quantise(b.location.0)))
        .then_with(|| quantise(a.location.1).cmp(&quantise(b.location.1)))
        .then_with(|| a.nets.cmp(&b.nets))
        .then_with(|| a.location.0.total_cmp(&b.location.0))
        .then_with(|| a.location.1.total_cmp(&b.location.1))
        .then_with(|| a.measured.total_cmp(&b.measured))
        .then_with(|| a.required.total_cmp(&b.required))
}

/// Sort an augmented violation stream by the checker's canonical deterministic
/// ordering. Callers that add format-specific findings (for example board-edge
/// checks) use this instead of duplicating the float/tie-break contract.
pub fn sort_violations(violations: &mut [Violation]) {
    violations.sort_by(violation_cmp);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> DrcRules {
        DrcRules {
            clearance: 0.2,
            plane_antipad: 0.1,
            min_annular_ring: 0.05,
        }
    }

    fn seg(net: &str, layer: u32, a: (f64, f64), b: (f64, f64), width: f64) -> Segment {
        Segment {
            net: net.to_string(),
            layer,
            a,
            b,
            width,
        }
    }

    fn classes(vs: &[Violation], class: ViolationClass) -> usize {
        vs.iter().filter(|v| v.class == class).count()
    }

    #[test]
    fn parallel_segments_violate_when_too_close() {
        // Two horizontal traces, width 0.1 (half-width 0.05 each). Centreline gap
        // 0.25 → copper gap 0.25 - 0.05 - 0.05 = 0.15 < clearance 0.2 → violation.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 0.25), (10.0, 0.25), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        let v = board.check();
        assert_eq!(classes(&v, ViolationClass::Clearance), 1);
        let c = &v[0];
        assert!(
            (c.measured - 0.15).abs() < 1e-9,
            "measured = {}",
            c.measured
        );
        assert_eq!(c.required, 0.2);
        assert_eq!(c.layer, 0);
    }

    #[test]
    fn segments_exactly_at_clearance_pass() {
        // Centreline gap 0.30 → copper gap 0.30 - 0.05 - 0.05 = 0.20 == clearance.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 0.30), (10.0, 0.30), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(board.check().is_empty(), "at-clearance must not fire");
    }

    #[test]
    fn segments_above_clearance_pass() {
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 1.0), (10.0, 1.0), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(board.check().is_empty());
    }

    #[test]
    fn same_net_touching_is_clean() {
        // Two overlapping segments on the same net → never a clearance violation.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("NET1", 0, (0.0, 0.0), (10.0, 0.0), 0.2),
                seg("NET1", 0, (5.0, 0.0), (5.0, 5.0), 0.2),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(board.check().is_empty(), "same net never conflicts");
    }

    #[test]
    fn different_layers_dont_interact() {
        // Same planar position, zero gap, but different layers → no clearance.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.2),
                seg("B", 1, (0.0, 0.0), (10.0, 0.0), 0.2),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert!(
            board.check().is_empty(),
            "features on different layers must not conflict"
        );
    }

    #[test]
    fn none_net_pad_conflicts_with_everything() {
        // A net==None pad touching a segment of a real net → violation.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1)],
            pads: vec![Pad {
                net: None,
                layer: 0,
                center: (5.0, 0.1),
                width: 0.2,
                height: 0.2,
            }],
            vias: vec![],
            rules: rules(),
        };
        let v = board.check();
        assert_eq!(classes(&v, ViolationClass::Clearance), 1);
        // None net renders as empty string.
        assert!(v[0].nets.0.is_empty() || v[0].nets.1.is_empty());
    }

    #[test]
    fn via_through_foreign_plane_fires_without_antipad() {
        // Stack: Signal / Plane(GND) / Signal. Via on NET1 spanning 0..=2 crosses
        // the GND plane on layer 1 with no antipad → ViaThroughPlane.
        let board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let v = board.check();
        assert_eq!(classes(&v, ViolationClass::ViaThroughPlane), 1);
        let tp = v
            .iter()
            .find(|x| x.class == ViolationClass::ViaThroughPlane)
            .unwrap();
        assert_eq!(tp.layer, 1);
        assert_eq!(tp.nets, ("NET1".to_string(), "GND".to_string()));
        assert_eq!(tp.measured, 0.0);
        // required = drill/2 + plane_antipad = 0.15 + 0.1 = 0.25.
        assert!((tp.required - 0.25).abs() < 1e-9);
    }

    #[test]
    fn via_through_plane_clean_with_large_antipad() {
        let board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: Some(0.25), // == drill/2 + plane_antipad
            }],
            rules: rules(),
        };
        assert_eq!(
            classes(&board.check(), ViolationClass::ViaThroughPlane),
            0,
            "antipad exactly meeting the rule clears the short"
        );
    }

    #[test]
    fn via_through_same_net_plane_is_clean() {
        let board = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "GND".to_string(), // same net as the plane
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        assert_eq!(
            classes(&board.check(), ViolationClass::ViaThroughPlane),
            0,
            "a via on the plane's own net never shorts it"
        );
    }

    #[test]
    fn endpoint_plane_is_a_short() {
        // Via ends ON a foreign plane (layer 0 is the plane). Endpoints are part of
        // the inclusive span, so this must fire.
        let board = DrcBoard {
            layers: vec![
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 1,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let v = board.check();
        let tp: Vec<_> = v
            .iter()
            .filter(|x| x.class == ViolationClass::ViaThroughPlane)
            .collect();
        assert_eq!(tp.len(), 1);
        assert_eq!(tp[0].layer, 0, "endpoint plane on layer 0 shorts");
    }

    #[test]
    fn oversized_reversed_via_span_stamps_only_in_stack_layers() {
        // The reversed span deliberately reaches u32::MAX. The checker must clamp
        // it before iteration, stamp the via on physical layers 0 and 1, and never
        // manufacture a via feature beside the deliberately out-of-stack L2 pad.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![],
            pads: (0..=2)
                .map(|layer| Pad {
                    net: Some("PAD".to_string()),
                    layer,
                    center: (1.0, 1.0),
                    width: 0.2,
                    height: 0.2,
                })
                .collect(),
            vias: vec![Via {
                net: "VIA".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: u32::MAX,
                to_layer: 0,
                antipad_radius: None,
            }],
            rules: rules(),
        };

        assert_eq!(in_stack_via_span(&board.vias[0], 2), Some((0, 1)));
        assert_eq!(in_stack_via_span(&board.vias[0], 0), None);
        let findings = board.check();
        assert_eq!(findings.len(), 2, "one via-pad finding per real layer");
        assert!(findings
            .iter()
            .all(|v| v.class == ViolationClass::Clearance));
        assert_eq!(
            findings.iter().map(|v| v.layer).collect::<Vec<_>>(),
            [0, 1],
            "the out-of-stack L2 pad must not see a phantom via pad"
        );
    }

    #[test]
    fn reversing_via_span_preserves_exact_drc_output() {
        // Include both a plane short and an annular-ring failure so every
        // span-sensitive stream is covered by the direction-invariance assertion.
        let forward = DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.30,
                drill_diameter: 0.25,
                from_layer: 0,
                to_layer: 2,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let mut reversed = forward.clone();
        let via = &mut reversed.vias[0];
        std::mem::swap(&mut via.from_layer, &mut via.to_layer);

        let expected = forward.check();
        assert_eq!(reversed.check(), expected);
        let annular = expected
            .iter()
            .find(|v| v.class == ViolationClass::AnnularRing)
            .expect("fixture must exercise annular reporting");
        assert_eq!(
            annular.layer, 0,
            "annular reports the canonical lower layer"
        );
    }

    #[test]
    fn annular_ring_below_minimum_fires() {
        // (pad - drill)/2 = (0.30 - 0.25)/2 = 0.025 < min 0.05.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.30,
                drill_diameter: 0.25,
                from_layer: 0,
                to_layer: 1,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        let v = board.check();
        let ar = v
            .iter()
            .find(|x| x.class == ViolationClass::AnnularRing)
            .unwrap();
        assert!((ar.measured - 0.025).abs() < 1e-9);
        assert_eq!(ar.required, 0.05);
        assert_eq!(ar.layer, 0);
        assert_eq!(ar.nets.1, "");
    }

    #[test]
    fn annular_ring_at_minimum_passes() {
        // (0.40 - 0.30)/2 = 0.05 == min.
        let board = DrcBoard {
            layers: vec![LayerKind::Signal, LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![Via {
                net: "NET1".to_string(),
                center: (1.0, 1.0),
                pad_diameter: 0.40,
                drill_diameter: 0.30,
                from_layer: 0,
                to_layer: 1,
                antipad_radius: None,
            }],
            rules: rules(),
        };
        assert_eq!(classes(&board.check(), ViolationClass::AnnularRing), 0);
    }

    /// A small mixed board reused by the determinism tests.
    fn mixed_board(segs: Vec<Segment>, pads: Vec<Pad>, vias: Vec<Via>) -> DrcBoard {
        DrcBoard {
            layers: vec![
                LayerKind::Signal,
                LayerKind::Plane {
                    net: "GND".to_string(),
                },
                LayerKind::Signal,
            ],
            segments: segs,
            pads,
            vias,
            rules: rules(),
        }
    }

    fn sample_features() -> (Vec<Segment>, Vec<Pad>, Vec<Via>) {
        let segs = vec![
            seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
            seg("B", 0, (0.0, 0.25), (10.0, 0.25), 0.1), // clearance vs A
            seg("C", 2, (0.0, 0.0), (5.0, 0.0), 0.1),
        ];
        let pads = vec![Pad {
            net: None,
            layer: 0,
            center: (3.0, 0.1),
            width: 0.1,
            height: 0.1,
        }];
        let vias = vec![
            Via {
                net: "NET1".to_string(),
                center: (7.0, 5.0),
                pad_diameter: 0.30,
                drill_diameter: 0.25, // annular 0.025 < 0.05
                from_layer: 0,
                to_layer: 2, // crosses GND plane on layer 1
                antipad_radius: None,
            },
            Via {
                net: "GND".to_string(),
                center: (8.0, 6.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 2, // same net as plane → no short
                antipad_radius: None,
            },
        ];
        (segs, pads, vias)
    }

    #[test]
    fn determinism_same_board_twice() {
        let (s, p, v) = sample_features();
        let b1 = mixed_board(s.clone(), p.clone(), v.clone());
        let b2 = mixed_board(s, p, v);
        assert_eq!(b1.check(), b2.check());
        // Sanity: it actually produced violations across classes.
        let r = b1.check();
        assert!(classes(&r, ViolationClass::Clearance) >= 1);
        assert!(classes(&r, ViolationClass::ViaThroughPlane) >= 1);
        assert!(classes(&r, ViolationClass::AnnularRing) >= 1);
    }

    #[test]
    fn determinism_shuffled_input_same_output() {
        let (s, p, v) = sample_features();
        let baseline = mixed_board(s.clone(), p.clone(), v.clone()).check();

        // Reverse each input vector — order must not change the sorted result.
        let mut s2 = s;
        s2.reverse();
        let mut v2 = v;
        v2.reverse();
        let shuffled = mixed_board(s2, p, v2).check();

        assert_eq!(
            baseline, shuffled,
            "shuffling feature input order must yield identical sorted violations"
        );
    }

    #[test]
    fn each_pair_reported_once() {
        // Two close different-net segments must produce exactly one clearance row,
        // not two (i<j de-duplication).
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
                seg("B", 0, (0.0, 0.1), (10.0, 0.1), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        assert_eq!(classes(&board.check(), ViolationClass::Clearance), 1);
    }

    #[test]
    fn geometry_helpers_basic() {
        assert!((dist((0.0, 0.0), (3.0, 4.0)) - 5.0).abs() < 1e-12);
        assert!((point_seg_dist((5.0, 3.0), (0.0, 0.0), (10.0, 0.0)) - 3.0).abs() < 1e-12);
        // Parallel segments 2 apart.
        assert!(
            (seg_seg_dist((0.0, 0.0), (10.0, 0.0), (0.0, 2.0), (10.0, 2.0)) - 2.0).abs() < 1e-12
        );
        // Crossing segments → 0.
        assert!(seg_seg_dist((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)) < 1e-12);
        // Point outside a unit rect centred at origin, 1 to the right of the edge.
        assert!((point_rect_gap((1.5, 0.0), (0.0, 0.0), 1.0, 1.0) - 1.0).abs() < 1e-12);
        // Point inside → 0.
        assert!(point_rect_gap((0.0, 0.0), (0.0, 0.0), 1.0, 1.0) < 1e-12);
    }

    #[test]
    fn colliding_sort_keys_are_stable_under_input_permutation() {
        // Every A/B pair has the same published sort fields (class, layer,
        // location, nets), but differing widths give differing measured gaps.
        // The remaining fields must provide a deterministic total tie-break.
        let segs = vec![
            seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
            seg("B", 0, (0.0, 0.0), (10.0, 0.0), 0.1),
            seg("A", 0, (0.0, 0.0), (10.0, 0.0), 0.2),
            seg("B", 0, (0.0, 0.0), (10.0, 0.0), 0.3),
        ];
        let board = |segments| DrcBoard {
            layers: vec![LayerKind::Signal],
            segments,
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };
        let expected = board(segs.clone()).check();
        let mut reversed = segs;
        reversed.reverse();
        assert_eq!(expected, board(reversed).check());
    }

    fn clearance_only_bruteforce(board: &DrcBoard) -> Vec<Violation> {
        let mut by_layer: std::collections::BTreeMap<u32, Vec<Feature>> =
            std::collections::BTreeMap::new();
        for segment in &board.segments {
            by_layer
                .entry(segment.layer)
                .or_default()
                .push(Feature::segment(segment));
        }
        for pad in &board.pads {
            by_layer
                .entry(pad.layer)
                .or_default()
                .push(Feature::pad(pad));
        }
        let layer_count = board.layers.len() as u32;
        for via in &board.vias {
            let lo = via.from_layer.min(via.to_layer);
            let hi = via.from_layer.max(via.to_layer);
            for layer in lo..=hi {
                if layer_count == 0 || layer < layer_count {
                    by_layer.entry(layer).or_default().push(Feature::via(via));
                }
            }
        }

        let mut out = Vec::new();
        for (layer, features) in by_layer {
            for i in 0..features.len() {
                for j in i + 1..features.len() {
                    board.test_clearance_pair_exact(layer, &features[i], &features[j], &mut out);
                }
            }
        }
        out.sort_by(violation_cmp);
        out
    }

    #[test]
    fn spatial_index_matches_naive_all_pairs_reference() {
        // Fixed-seed pseudo-random geometry: deterministic and broad enough to
        // exercise negative bins, long segments, all feature kinds, and layers.
        let mut state = 0x7a5b_9d31_42c6_e817u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as u32) as f64 / u32::MAX as f64
        };
        let coord = |v: f64| v * 30.0 - 15.0;
        let mut segments = Vec::new();
        let mut pads = Vec::new();
        let mut vias = Vec::new();
        for i in 0..48 {
            let net = format!("N{}", i % 7);
            let layer = (i % 3) as u32;
            let (x, y) = (coord(next()), coord(next()));
            match i % 3 {
                0 => segments.push(seg(
                    &net,
                    layer,
                    (x, y),
                    (x + coord(next()) * 0.35, y + coord(next()) * 0.35),
                    0.05 + next() * 0.8,
                )),
                1 => pads.push(Pad {
                    net: Some(net),
                    layer,
                    center: (x, y),
                    width: 0.05 + next() * 1.5,
                    height: 0.05 + next() * 1.5,
                }),
                _ => vias.push(Via {
                    net,
                    center: (x, y),
                    pad_diameter: 0.4 + next() * 0.5,
                    drill_diameter: 0.2,
                    from_layer: layer,
                    to_layer: layer,
                    antipad_radius: None,
                }),
            }
        }
        let board = DrcBoard {
            layers: vec![LayerKind::Signal; 3],
            segments,
            pads,
            vias,
            rules: DrcRules {
                clearance: 0.3,
                plane_antipad: 0.1,
                min_annular_ring: 0.0,
            },
        };
        let indexed: Vec<_> = board
            .check()
            .into_iter()
            .filter(|v| v.class == ViolationClass::Clearance)
            .collect();
        assert_eq!(indexed, clearance_only_bruteforce(&board));
    }

    #[test]
    fn every_copper_feature_pairing_is_checked() {
        fn via(net: &str) -> Via {
            Via {
                net: net.into(),
                center: (0.0, 0.0),
                pad_diameter: 0.6,
                drill_diameter: 0.3,
                from_layer: 0,
                to_layer: 0,
                antipad_radius: None,
            }
        }
        let segment = |net: &str| seg(net, 0, (-1.0, 0.0), (1.0, 0.0), 0.2);
        let pad = |net: &str| Pad {
            net: Some(net.into()),
            layer: 0,
            center: (0.0, 0.0),
            width: 0.5,
            height: 0.5,
        };
        let cases = [
            (vec![segment("A"), segment("B")], vec![], vec![]),
            (vec![segment("A")], vec![pad("B")], vec![]),
            (vec![segment("A")], vec![], vec![via("B")]),
            (vec![], vec![pad("A"), pad("B")], vec![]),
            (vec![], vec![pad("A")], vec![via("B")]),
            (vec![], vec![], vec![via("A"), via("B")]),
        ];
        for (segments, pads, vias) in cases {
            let board = DrcBoard {
                layers: vec![LayerKind::Signal],
                segments,
                pads,
                vias,
                rules: rules(),
            };
            assert_eq!(
                classes(&board.check(), ViolationClass::Clearance),
                1,
                "pairing should produce exactly one clearance violation: {board:?}"
            );
        }
    }

    #[test]
    fn pad_pair_rules_do_not_raise_unrelated_copper_clearance() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.05;
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![
                seg("A", 0, (-1.0, 0.0), (1.0, 0.0), 0.1),
                seg("B", 0, (-1.0, 0.16), (1.0, 0.16), 0.1),
            ],
            pads: vec![],
            vias: vec![],
            rules: pair_rules,
        };
        assert!(board
            .check_with_pad_clearances(0.07, 0.08, None, None)
            .is_empty());
    }

    #[test]
    fn typed_trace_and_via_pad_rules_use_their_own_exact_boundaries() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.05;
        let pad = Pad {
            net: Some("pad".into()),
            layer: 0,
            center: (0.0, 0.0),
            width: 0.2,
            height: 0.2,
        };

        let trace_board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![seg("trace", 0, (0.21, -1.0), (0.21, 1.0), 0.1)],
            pads: vec![pad.clone()],
            vias: vec![],
            rules: pair_rules,
        };
        assert!(trace_board.check().is_empty());
        let trace_findings = trace_board.check_with_pad_clearances(0.07, 0.08, None, None);
        assert_eq!(trace_findings.len(), 1);
        assert_eq!(trace_findings[0].required, 0.07);

        let via_board = |x| DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![pad.clone()],
            vias: vec![Via {
                net: "via".into(),
                center: (x, 0.0),
                pad_diameter: 0.4,
                drill_diameter: 0.2,
                from_layer: 0,
                to_layer: 0,
                antipad_radius: None,
            }],
            rules: pair_rules,
        };
        let just_inside =
            via_board(0.38 - 2.0 * EPS).check_with_pad_clearances(0.07, 0.08, None, None);
        assert_eq!(just_inside.len(), 1);
        assert_eq!(just_inside[0].required, 0.08);
        assert!(
            via_board(0.38)
                .check_with_pad_clearances(0.07, 0.08, None, None)
                .is_empty(),
            "an annulus exactly at the via→pad rule is legal"
        );

        let pad_board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![
                pad,
                Pad {
                    net: Some("other-pad".into()),
                    layer: 0,
                    center: (0.29, 0.0),
                    width: 0.2,
                    height: 0.2,
                },
            ],
            vias: vec![],
            rules: pair_rules,
        };
        assert!(pad_board.check().is_empty());
        let pad_findings = pad_board.check_with_pad_clearances(0.07, 0.08, Some(0.1), None);
        assert_eq!(pad_findings.len(), 1);
        assert_eq!(pad_findings[0].required, 0.1);
    }

    #[test]
    fn typed_trace_rule_governs_via_to_trace_at_the_exact_boundary() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.04;
        pair_rules.min_annular_ring = 0.0;
        let board_at = |x| DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![seg("trace", 0, (x, -1.0), (x, 1.0), 0.1)],
            pads: vec![],
            vias: vec![Via {
                net: "via".into(),
                center: (0.0, 0.0),
                pad_diameter: 0.4,
                drill_diameter: 0.2,
                from_layer: 0,
                to_layer: 0,
                antipad_radius: None,
            }],
            rules: pair_rules,
        };
        assert!(board_at(0.32).check().is_empty());
        assert!(
            board_at(0.32)
                .check_with_pad_clearances(0.07, 0.09, None, None)
                .is_empty(),
            "a via annulus exactly at the SRJ trace rule is legal"
        );
        let findings = board_at(0.32 - 2.0 * EPS).check_with_pad_clearances(0.07, 0.09, None, None);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].required, 0.07);
    }

    #[test]
    fn via_hole_clearance_can_dominate_annular_pad_clearance() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.02;
        pair_rules.min_annular_ring = 0.0;
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![
                Via {
                    net: "a".into(),
                    center: (0.0, 0.0),
                    pad_diameter: 0.2,
                    drill_diameter: 0.18,
                    from_layer: 0,
                    to_layer: 0,
                    antipad_radius: None,
                },
                Via {
                    net: "b".into(),
                    center: (0.25, 0.0),
                    pad_diameter: 0.2,
                    drill_diameter: 0.18,
                    from_layer: 0,
                    to_layer: 0,
                    antipad_radius: None,
                },
            ],
            rules: pair_rules,
        };
        assert!(
            board.check().is_empty(),
            "0.05 annular edge gap satisfies generic 0.02"
        );
        let findings = board.check_with_pad_clearances(0.02, 0.02, None, Some(0.1));
        assert_eq!(findings.len(), 1);
        assert!((findings[0].measured - 0.07).abs() < 1e-12);
        assert!((findings[0].required - 0.1).abs() < 1e-12);

        let mut equal_rules = board.clone();
        equal_rules.vias[1].center.0 = 0.2;
        let equal_findings = equal_rules.check_with_pad_clearances(0.02, 0.02, None, Some(0.04));
        assert_eq!(
            equal_findings.len(),
            1,
            "the equal-strength rule has one owner"
        );
        assert!((equal_findings[0].measured - 0.02).abs() < 1e-12);
        assert!((equal_findings[0].required - 0.04).abs() < 1e-12);

        let mut same_net = board.clone();
        same_net.vias[1].net = "a".into();
        let same_net_findings = same_net.check_with_pad_clearances(0.02, 0.02, None, Some(0.1));
        assert_eq!(same_net_findings.len(), 1);
        assert_eq!(same_net_findings[0].nets, ("a".into(), "a".into()));
        assert!((same_net_findings[0].measured - 0.07).abs() < 1e-12);
        assert!((same_net_findings[0].required - 0.1).abs() < 1e-12);

        same_net.vias[1].center.0 = 0.28;
        assert!(
            same_net
                .check_with_pad_clearances(0.02, 0.02, None, Some(0.1))
                .is_empty(),
            "same-net holes exactly at the fabrication rule are legal"
        );

        let mut duplicate_representation = same_net.clone();
        duplicate_representation.vias[1] = duplicate_representation.vias[0].clone();
        duplicate_representation.layers = vec![LayerKind::Signal; 3];
        duplicate_representation.vias[1].to_layer = 2;
        duplicate_representation.vias[1].pad_diameter = 0.24;
        duplicate_representation.vias[1].drill_diameter = 0.19;
        assert!(
            duplicate_representation
                .check_with_pad_clearances(0.02, 0.02, None, Some(0.1))
                .is_empty(),
            "partial/full sibling records with different geometry are one physical hole"
        );

        duplicate_representation.vias[1].center.0 = SAME_VIA_LOCATION_EPS;
        assert!(
            duplicate_representation
                .check_with_pad_clearances(0.02, 0.02, None, Some(0.1))
                .is_empty(),
            "same-net records exactly 0.005 mm apart are one producer-defined site"
        );
        duplicate_representation.vias[1].center.0 = SAME_VIA_LOCATION_EPS + 2.0 * EPS;
        let outside_coincidence =
            duplicate_representation.check_with_pad_clearances(0.02, 0.02, None, Some(0.1));
        assert_eq!(outside_coincidence.len(), 1);
        assert_eq!(outside_coincidence[0].nets, ("a".into(), "a".into()));

        let mut interleaved = duplicate_representation.clone();
        interleaved.vias = vec![
            duplicate_representation.vias[0].clone(),
            Via {
                center: (0.002, 100.0),
                ..duplicate_representation.vias[0].clone()
            },
            Via {
                center: (0.004, 0.0),
                ..duplicate_representation.vias[1].clone()
            },
        ];
        assert!(
            interleaved
                .check_with_pad_clearances(0.02, 0.02, None, Some(0.1))
                .is_empty(),
            "a far-y via interleaved by x-sort cannot split one coincident site"
        );
    }

    #[test]
    fn disjoint_foreign_via_spans_still_enforce_hole_spacing() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.2;
        pair_rules.min_annular_ring = 0.0;
        let board = DrcBoard {
            layers: vec![LayerKind::Signal; 4],
            segments: vec![],
            pads: vec![],
            vias: vec![
                Via {
                    net: "a".into(),
                    center: (0.0, 0.0),
                    pad_diameter: 1.0,
                    drill_diameter: 0.2,
                    from_layer: 0,
                    to_layer: 1,
                    antipad_radius: None,
                },
                Via {
                    net: "b".into(),
                    center: (0.3, 0.0),
                    pad_diameter: 1.0,
                    drill_diameter: 0.2,
                    from_layer: 2,
                    to_layer: 3,
                    antipad_radius: None,
                },
            ],
            rules: pair_rules,
        };
        assert!(board.check().is_empty(), "annular copper spans do not meet");
        let findings = board.check_with_pad_clearances(0.2, 0.2, None, Some(0.15));
        assert_eq!(findings.len(), 1);
        assert!((findings[0].measured - 0.1).abs() < 1e-12);
        assert!((findings[0].required - 0.15).abs() < 1e-12);
    }

    #[test]
    fn merged_site_geometry_cannot_hide_a_hole_rule_from_actual_layer_copper() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.1;
        pair_rules.min_annular_ring = 0.0;
        let board = DrcBoard {
            layers: vec![LayerKind::Signal; 2],
            segments: vec![],
            pads: vec![],
            vias: vec![
                Via {
                    net: "site".into(),
                    center: (0.0, 0.0),
                    pad_diameter: 2.0,
                    drill_diameter: 0.2,
                    from_layer: 0,
                    to_layer: 0,
                    antipad_radius: None,
                },
                Via {
                    net: "site".into(),
                    center: (0.0, 0.0),
                    pad_diameter: 0.2,
                    drill_diameter: 0.2,
                    from_layer: 1,
                    to_layer: 1,
                    antipad_radius: None,
                },
                Via {
                    net: "foreign".into(),
                    center: (0.35, 0.0),
                    pad_diameter: 0.2,
                    drill_diameter: 0.2,
                    from_layer: 1,
                    to_layer: 1,
                    antipad_radius: None,
                },
            ],
            rules: pair_rules,
        };
        assert!(
            board.check().is_empty(),
            "the only physically overlapping layer carries the small legal pads"
        );
        let findings = board.check_with_pad_clearances(0.1, 0.1, None, Some(0.2));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].nets, ("foreign".into(), "site".into()));
        assert!((findings[0].measured - 0.15).abs() < 1e-12);
        assert!((findings[0].required - 0.2).abs() < 1e-12);
    }

    #[test]
    fn hole_site_reports_the_worst_original_coincident_representation_pair() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.0;
        pair_rules.min_annular_ring = 0.0;
        let via_at = |x| Via {
            net: "same".into(),
            center: (x, 0.0),
            pad_diameter: 0.2,
            drill_diameter: 0.2,
            from_layer: 0,
            to_layer: 0,
            antipad_radius: None,
        };
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![via_at(0.0), via_at(0.005), via_at(0.301)],
            rules: pair_rules,
        };
        let findings = board.check_with_pad_clearances(0.0, 0.0, None, Some(0.1));
        assert_eq!(findings.len(), 1);
        assert!((findings[0].measured - 0.096).abs() < 1e-12);
        assert!((findings[0].location.0 - 0.153).abs() < 1e-12);
        assert_eq!(findings[0].required, 0.1);
        let mut reversed_shifted = board.clone();
        reversed_shifted.vias.reverse();
        assert_eq!(
            reversed_shifted.check_with_pad_clearances(0.0, 0.0, None, Some(0.1)),
            findings,
            "shifted same-site worst-gap selection is input-order independent"
        );

        let chain = DrcBoard {
            vias: vec![via_at(0.0), via_at(0.004), via_at(0.008)],
            ..board
        };
        let chain_findings = chain.check_with_pad_clearances(0.0, 0.0, None, Some(0.1));
        assert_eq!(chain_findings.len(), 1);
        assert!((chain_findings[0].measured + 0.192).abs() < 1e-12);
        assert!((chain_findings[0].location.0 - 0.004).abs() < 1e-12);
        assert_eq!(chain_findings[0].required, 0.1);
        assert_eq!(
            physical_via_sites(&vec![via_at(0.0); 64])[0]
                .representations
                .len(),
            1,
            "identical sibling soup records collapse before typed pair scanning"
        );
    }

    #[test]
    fn via_pair_ownership_uses_the_strictest_violating_actual_rule() {
        let mut pair_rules = rules();
        pair_rules.clearance = 0.09;
        pair_rules.min_annular_ring = 0.0;
        let via = |net: &str, x: f64, pad: f64, drill: f64| Via {
            net: net.into(),
            center: (x, 0.0),
            pad_diameter: pad,
            drill_diameter: drill,
            from_layer: 0,
            to_layer: 0,
            antipad_radius: None,
        };
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![
                via("site", 0.0, 0.21, 0.209),
                via("site", 0.005, 0.21, 0.2),
                via("foreign", 0.304, 0.214, 0.2),
            ],
            rules: pair_rules,
        };

        let findings = board.check_with_pad_clearances(0.09, 0.09, None, Some(0.1));
        assert_eq!(findings.len(), 1);
        assert!((findings[0].required - 0.1).abs() < 1e-12);
        assert!((findings[0].measured - 0.0995).abs() < 1e-12);
        assert!((findings[0].location.0 - 0.152).abs() < 1e-12);
        let original_vias = board.vias.clone();
        for order in [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let mut permuted = board.clone();
            permuted.vias = order
                .into_iter()
                .map(|index| original_vias[index].clone())
                .collect();
            assert_eq!(
                permuted.check_with_pad_clearances(0.09, 0.09, None, Some(0.1)),
                findings,
                "strictest-rule ownership is byte-identical under every input permutation"
            );
        }
    }

    #[test]
    fn canonical_via_pair_arbitration_emits_one_strongest_physical_row() {
        let board_with = |vias: Vec<Via>, clearance: f64| {
            let mut pair_rules = rules();
            pair_rules.clearance = clearance;
            pair_rules.min_annular_ring = 0.0;
            DrcBoard {
                layers: vec![LayerKind::Signal; 2],
                segments: vec![],
                pads: vec![],
                vias,
                rules: pair_rules,
            }
        };
        let via = |net: &str, x: f64, pad: f64, drill: f64, from_layer: u32, to_layer: u32| Via {
            net: net.into(),
            center: (x, 0.0),
            pad_diameter: pad,
            drill_diameter: drill,
            from_layer,
            to_layer,
            antipad_radius: None,
        };

        // The large drill is on a disjoint layer while the overlapping copper rep
        // is smaller. Synthetic max-pad/union-span geometry would choose copper,
        // but the actual stronger physical rule is the net-independent hole row.
        let split_geometry = board_with(
            vec![
                via("site", 0.0, 1.0, 1.0, 0, 0),
                via("site", 0.0, 0.8, 0.2, 1, 1),
                via("foreign", 0.5, 0.2, 0.2, 1, 1),
            ],
            0.1,
        );
        let split = split_geometry.check_with_pad_clearances(0.1, 0.1, None, Some(0.2));
        assert_eq!(split.len(), 1);
        assert!((split[0].measured + 0.1).abs() < 1e-12);
        assert_eq!(split[0].required, 0.2);

        // Duplicate soup records are one physical site. Copper is strictly
        // stronger here, so it owns exactly one row rather than one per duplicate.
        let duplicate = via("a", 0.0, 1.0, 0.2, 0, 0);
        let copper_board = board_with(
            vec![duplicate.clone(), duplicate, via("b", 0.25, 1.0, 0.2, 0, 0)],
            0.1,
        );
        assert_eq!(
            copper_board.check().len(),
            2,
            "without a declared hole rule, raw-via copper rows retain legacy multiplicity"
        );
        let copper = copper_board.check_with_pad_clearances(0.1, 0.1, None, Some(0.1));
        assert_eq!(copper.len(), 1);
        assert!((copper[0].measured + 0.75).abs() < 1e-12);
        assert_eq!(copper[0].required, 0.1);

        let mut reversed = copper_board.clone();
        reversed.vias.reverse();
        assert_eq!(
            reversed.check_with_pad_clearances(0.1, 0.1, None, Some(0.1)),
            copper,
            "canonical arbitration is byte-identical under input permutation"
        );

        // A hole-clean pair still reports its violating annular copper once.
        let hole_clean = board_with(
            vec![via("a", 0.0, 1.0, 0.2, 0, 0), via("b", 0.5, 1.0, 0.2, 0, 0)],
            0.1,
        )
        .check_with_pad_clearances(0.1, 0.1, None, Some(0.1));
        assert_eq!(hole_clean.len(), 1);
        assert!((hole_clean[0].measured + 0.5).abs() < 1e-12);

        // The same logical site is also aggregated against non-via copper.
        let mut site_to_trace = board_with(
            vec![via("a", 0.0, 0.2, 0.1, 0, 0), via("a", 0.0, 0.2, 0.1, 0, 0)],
            0.1,
        );
        site_to_trace
            .segments
            .push(seg("b", 0, (0.15, -0.5), (0.15, 0.5), 0.1));
        let trace_rows = site_to_trace.check_with_pad_clearances(0.1, 0.1, None, Some(0.1));
        assert_eq!(trace_rows.len(), 1);
    }

    #[test]
    fn aabb_rejection_matches_exact_for_all_shape_pairs_at_epsilon_boundary() {
        let make_segment = |net: &str, x: f64| Feature {
            net: Some(net.into()),
            shape: Shape::Segment {
                a: (x, -0.5),
                b: (x, 0.5),
            },
            inflate: 0.05,
        };
        let make_point = |net: &str, x: f64| Feature {
            net: Some(net.into()),
            shape: Shape::Point { c: (x, 0.0) },
            inflate: 0.05,
        };
        let make_rect = |net: &str, x: f64| Feature {
            net: Some(net.into()),
            shape: Shape::Rect {
                c: (x, 0.0),
                w: 0.1,
                h: 1.0,
            },
            inflate: 0.0,
        };
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![],
            rules: rules(),
        };

        for delta in [-2.0 * EPS, -EPS, -0.5 * EPS, 0.0, EPS, 2.0 * EPS] {
            // Every shape has a 0.05 right/left copper reach. Placing the second
            // centroid at 0.1 + clearance + delta makes the exact x gap equal to
            // clearance + delta, exercising both sides of the EPS threshold.
            let x = 0.1 + board.rules.clearance + delta;
            let pairs = [
                (make_segment("A", 0.0), make_segment("B", x)),
                (make_segment("A", 0.0), make_point("B", x)),
                (make_point("A", 0.0), make_point("B", x)),
                (make_rect("A", 0.0), make_point("B", x)),
                (make_rect("A", 0.0), make_segment("B", x)),
                (make_rect("A", 0.0), make_rect("B", x)),
            ];
            for (a, b) in pairs {
                let mut filtered = Vec::new();
                board.test_clearance_pair(0, &a, &b, &mut filtered);
                let mut exact = Vec::new();
                board.test_clearance_pair_exact(0, &a, &b, &mut exact);
                assert_eq!(filtered, exact, "delta={delta}, a={a:?}, b={b:?}");
            }
        }
    }

    #[test]
    fn aabb_rejection_preserves_negative_gap_when_clearance_is_off() {
        let board = DrcBoard {
            layers: vec![LayerKind::Signal],
            segments: vec![],
            pads: vec![],
            vias: vec![],
            rules: DrcRules {
                clearance: 0.0,
                ..rules()
            },
        };
        let a = Feature {
            net: Some("A".into()),
            shape: Shape::Point { c: (0.0, 0.0) },
            inflate: 0.5,
        };
        let b = Feature {
            net: Some("B".into()),
            shape: Shape::Point { c: (0.0, 0.0) },
            inflate: 0.5,
        };
        let mut filtered = Vec::new();
        board.test_clearance_pair(0, &a, &b, &mut filtered);
        let mut exact = Vec::new();
        board.test_clearance_pair_exact(0, &a, &b, &mut exact);
        assert_eq!(filtered, exact);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].measured, -1.0);
    }

    #[test]
    fn geometry_distances_are_symmetric_under_reversal_and_translation() {
        let p = ((-2.0, 1.5), (4.0, -0.5));
        let q = ((-1.0, -3.0), (3.5, 2.0));
        let expected = seg_seg_dist(p.0, p.1, q.0, q.1);
        for (a0, a1) in [(p.0, p.1), (p.1, p.0)] {
            for (b0, b1) in [(q.0, q.1), (q.1, q.0)] {
                assert!((seg_seg_dist(a0, a1, b0, b1) - expected).abs() < 1e-12);
                assert!((seg_seg_dist(b0, b1, a0, a1) - expected).abs() < 1e-12);
            }
        }
        let shift = |v: (f64, f64)| (v.0 + 100.0, v.1 - 70.0);
        assert!(
            (seg_seg_dist(shift(p.0), shift(p.1), shift(q.0), shift(q.1)) - expected).abs() < 1e-12
        );
        assert_eq!(point_seg_dist(p.0, q.0, q.0), dist(p.0, q.0));
    }

    #[test]
    fn drc_summary_tallies_all_classes() {
        let violations = vec![
            Violation {
                class: ViolationClass::Clearance,
                layer: 0,
                location: (0.0, 0.0),
                nets: ("a".into(), "b".into()),
                measured: 0.0,
                required: 1.0,
            },
            Violation {
                class: ViolationClass::ViaThroughPlane,
                layer: 1,
                location: (0.0, 0.0),
                nets: ("a".into(), "gnd".into()),
                measured: 0.0,
                required: 1.0,
            },
            Violation {
                class: ViolationClass::AnnularRing,
                layer: 0,
                location: (0.0, 0.0),
                nets: ("a".into(), String::new()),
                measured: 0.0,
                required: 1.0,
            },
        ];
        let summary = DrcSummary::of(&violations);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.clearance, 1);
        assert_eq!(summary.via_through_plane, 1);
        assert_eq!(summary.annular_ring, 1);
        assert!(!summary.is_clean());
        assert!(DrcSummary::of(&[]).is_clean());
    }
}
