//! `mr-oracle` — the correctness comparator for metalroute.
//!
//! This crate decides whether two [`BoardRoute`]s *agree* under the design's
//! equivalence definition. It is used to grade the Metal port against the CPU
//! oracle, so it must be strict about what counts as equal and explicit about
//! every way two results can diverge.
//!
//! # Equivalence
//!
//! Two [`BoardRoute`]s are equivalent when **all** of the following hold:
//!
//! - equal **total** cost (sum over routed nets), AND
//! - equal **per-net** cost (matched by net *name*, not order), AND
//! - equal per-cell [`BoardRoute::congestion`] vectors, AND
//! - equal **set** of `unrouted` net names.
//!
//! Paths are deliberately **not** required to be bit-identical: equal-cost ties
//! mean two correct routers can pick different cells. The congestion vector is
//! what guards against "same cost, different board" — a result that costs the
//! same but occupies different cells is *not* equivalent.

use std::collections::{BTreeMap, BTreeSet};

use mr_core::BoardRoute;

/// A single way two [`BoardRoute`]s fail to agree.
///
/// [`compare`] returns *all* discrepancies it finds, so a caller can report
/// every divergence at once rather than just the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Discrepancy {
    /// The headline total cost (sum over routed nets) differs.
    TotalCost { a: u64, b: u64 },
    /// A net present in both results has a different cost.
    NetCost { net: String, a: u32, b: u32 },
    /// A net is routed in one result but absent from the other.
    ///
    /// `in_a` is true when the net appears in `a` but not `b`, false for the
    /// reverse. (A net unrouted in one and routed in the other surfaces here
    /// for the routed side and via [`Discrepancy::UnroutedMismatch`] for the
    /// unrouted side.)
    MissingNet { net: String, in_a: bool },
    /// The congestion count for a cell differs.
    Congestion { cell: usize, a: u32, b: u32 },
    /// The congestion vectors have different dimensions. Cell values are still
    /// compared across the longer length, but trailing zeroes do not erase this
    /// structural mismatch.
    CongestionLength { a: usize, b: usize },
    /// The *set* of unrouted net names differs. Each set lists the names that
    /// are unrouted in one side but not the other.
    UnroutedMismatch {
        only_in_a: Vec<String>,
        only_in_b: Vec<String>,
    },
}

/// Compare two [`BoardRoute`]s and return every discrepancy between them.
///
/// An empty vector means the two routes are equivalent under the crate-level
/// equivalence definition. See [`are_equivalent`] for the boolean convenience.
pub fn compare(a: &BoardRoute, b: &BoardRoute) -> Vec<Discrepancy> {
    let mut out = Vec::new();

    // 1. Total cost.
    let (ta, tb) = (a.total_cost(), b.total_cost());
    if ta != tb {
        out.push(Discrepancy::TotalCost { a: ta, b: tb });
    }

    // 2. Per-net cost, matched by NAME and occurrence (order-independent). Input
    // names are normally unique, but the contract does not require that and an
    // equivalence comparator must at minimum remain reflexive for duplicate names.
    fn costs_by_name(route: &BoardRoute) -> BTreeMap<&str, Vec<u32>> {
        let mut by_name: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
        for result in &route.results {
            by_name
                .entry(result.net.as_str())
                .or_default()
                .push(result.cost);
        }
        for costs in by_name.values_mut() {
            costs.sort_unstable();
        }
        by_name
    }

    // Cancel exact cost matches first, leaving the minimum deterministic set of
    // mismatched and one-sided occurrences for diagnostics.
    fn unmatched_costs(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let (mut ia, mut ib) = (0usize, 0usize);
        let (mut only_a, mut only_b) = (Vec::new(), Vec::new());
        while ia < a.len() && ib < b.len() {
            match a[ia].cmp(&b[ib]) {
                std::cmp::Ordering::Less => {
                    only_a.push(a[ia]);
                    ia += 1;
                }
                std::cmp::Ordering::Greater => {
                    only_b.push(b[ib]);
                    ib += 1;
                }
                std::cmp::Ordering::Equal => {
                    ia += 1;
                    ib += 1;
                }
            }
        }
        only_a.extend_from_slice(&a[ia..]);
        only_b.extend_from_slice(&b[ib..]);
        (only_a, only_b)
    }

    let by_name_a = costs_by_name(a);
    let by_name_b = costs_by_name(b);
    let names: BTreeSet<&str> = by_name_a.keys().chain(by_name_b.keys()).copied().collect();
    for net in names {
        let costs_a = by_name_a.get(net).map(Vec::as_slice).unwrap_or(&[]);
        let costs_b = by_name_b.get(net).map(Vec::as_slice).unwrap_or(&[]);
        let (only_a, only_b) = unmatched_costs(costs_a, costs_b);
        let paired = only_a.len().min(only_b.len());
        for i in 0..paired {
            out.push(Discrepancy::NetCost {
                net: net.to_string(),
                a: only_a[i],
                b: only_b[i],
            });
        }
        for _ in paired..only_a.len() {
            out.push(Discrepancy::MissingNet {
                net: net.to_string(),
                in_a: true,
            });
        }
        for _ in paired..only_b.len() {
            out.push(Discrepancy::MissingNet {
                net: net.to_string(),
                in_a: false,
            });
        }
    }

    // 3. Per-cell congestion. Length is part of the vector contract; still report
    // differing present values cell-by-cell against an implicit 0 for diagnostics.
    if a.congestion.len() != b.congestion.len() {
        out.push(Discrepancy::CongestionLength {
            a: a.congestion.len(),
            b: b.congestion.len(),
        });
    }
    let n = a.congestion.len().max(b.congestion.len());
    for cell in 0..n {
        let ca = a.congestion.get(cell).copied().unwrap_or(0);
        let cb = b.congestion.get(cell).copied().unwrap_or(0);
        if ca != cb {
            out.push(Discrepancy::Congestion { cell, a: ca, b: cb });
        }
    }

    // 4. Unrouted set (order-independent).
    let ua: BTreeSet<&str> = a.unrouted.iter().map(String::as_str).collect();
    let ub: BTreeSet<&str> = b.unrouted.iter().map(String::as_str).collect();
    if ua != ub {
        let only_in_a: Vec<String> = ua.difference(&ub).map(|s| s.to_string()).collect();
        let only_in_b: Vec<String> = ub.difference(&ua).map(|s| s.to_string()).collect();
        out.push(Discrepancy::UnroutedMismatch {
            only_in_a,
            only_in_b,
        });
    }

    out
}

/// True when `a` and `b` are equivalent (i.e. [`compare`] finds nothing).
pub fn are_equivalent(a: &BoardRoute, b: &BoardRoute) -> bool {
    compare(a, b).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mr_core::RouteResult;

    fn rr(net: &str, path: &[u32], cost: u32) -> RouteResult {
        RouteResult {
            net: net.into(),
            path: path.to_vec(),
            cost,
        }
    }

    /// A two-net board: nets "a" (cells 0,1) and "b" (cells 1,2) on a 3-cell grid.
    fn base() -> BoardRoute {
        let results = vec![rr("a", &[0, 1], 2), rr("b", &[1, 2], 2)];
        let congestion = vec![1, 2, 1];
        BoardRoute {
            results,
            unrouted: vec![],
            congestion,
            groups: vec![],
        }
    }

    #[test]
    fn identical_routes_are_equivalent() {
        let a = base();
        let b = base();
        assert!(compare(&a, &b).is_empty());
        assert!(are_equivalent(&a, &b));
    }

    #[test]
    fn changed_net_cost_reports_netcost_and_totalcost() {
        let a = base();
        let mut b = base();
        // Bump net "b" cost by 1 (and keep congestion identical so only cost differs).
        b.results[1].cost = 3;

        let d = compare(&a, &b);
        assert!(!are_equivalent(&a, &b));

        assert!(
            d.contains(&Discrepancy::NetCost {
                net: "b".into(),
                a: 2,
                b: 3,
            }),
            "expected NetCost for net b, got {d:?}"
        );
        assert!(
            d.contains(&Discrepancy::TotalCost { a: 4, b: 5 }),
            "expected TotalCost 4 vs 5, got {d:?}"
        );
    }

    #[test]
    fn equal_total_cost_but_different_congestion_fails() {
        // This is the load-bearing case: identical per-net AND total cost, but the
        // boards occupy different cells. Must NOT be equivalent.
        let a = base();
        let mut b = base();
        b.congestion = vec![2, 1, 1]; // same sum, different distribution

        assert_eq!(
            a.total_cost(),
            b.total_cost(),
            "test setup: totals must match"
        );

        let d = compare(&a, &b);
        assert!(!are_equivalent(&a, &b));
        // No cost discrepancies expected.
        assert!(
            !d.iter().any(|x| matches!(
                x,
                Discrepancy::TotalCost { .. } | Discrepancy::NetCost { .. }
            )),
            "no cost discrepancies expected, got {d:?}"
        );
        assert!(
            d.contains(&Discrepancy::Congestion {
                cell: 0,
                a: 1,
                b: 2
            }),
            "expected Congestion at cell 0, got {d:?}"
        );
        assert!(
            d.contains(&Discrepancy::Congestion {
                cell: 1,
                a: 2,
                b: 1
            }),
            "expected Congestion at cell 1, got {d:?}"
        );
    }

    #[test]
    fn different_unrouted_set_reports_mismatch() {
        let mut a = base();
        let mut b = base();
        a.unrouted = vec!["x".into()];
        b.unrouted = vec!["y".into()];

        let d = compare(&a, &b);
        assert!(!are_equivalent(&a, &b));
        assert!(
            d.contains(&Discrepancy::UnroutedMismatch {
                only_in_a: vec!["x".into()],
                only_in_b: vec!["y".into()],
            }),
            "expected UnroutedMismatch, got {d:?}"
        );
    }

    #[test]
    fn unrouted_compared_as_set_not_order() {
        let mut a = base();
        let mut b = base();
        a.unrouted = vec!["x".into(), "y".into()];
        b.unrouted = vec!["y".into(), "x".into()];
        assert!(are_equivalent(&a, &b), "unrouted order must not matter");
    }

    #[test]
    fn nets_compared_by_name_not_order() {
        let a = base();
        // Same nets and congestion, but results listed in reverse order.
        let b = BoardRoute {
            results: vec![rr("b", &[1, 2], 2), rr("a", &[0, 1], 2)],
            unrouted: vec![],
            congestion: vec![1, 2, 1],
            groups: vec![],
        };
        assert!(
            are_equivalent(&a, &b),
            "net order must not matter: {:?}",
            compare(&a, &b)
        );
    }

    #[test]
    fn missing_net_reported_with_side() {
        let a = base();
        // `b` drops net "b".
        let b = BoardRoute {
            results: vec![rr("a", &[0, 1], 2)],
            unrouted: vec![],
            congestion: vec![1, 1, 0],
            groups: vec![],
        };
        let d = compare(&a, &b);
        assert!(!are_equivalent(&a, &b));
        assert!(
            d.contains(&Discrepancy::MissingNet {
                net: "b".into(),
                in_a: true,
            }),
            "expected MissingNet(b, in_a=true), got {d:?}"
        );
    }

    #[test]
    fn duplicate_net_names_preserve_reflexivity_and_multiplicity() {
        let route = BoardRoute {
            results: vec![rr("x", &[0], 1), rr("x", &[1], 2), rr("y", &[2], 3)],
            unrouted: vec![],
            congestion: vec![1, 1, 1],
            groups: vec![],
        };
        assert!(
            are_equivalent(&route, &route),
            "equivalence must be reflexive even when names repeat: {:?}",
            compare(&route, &route)
        );

        let mut permuted = route.clone();
        permuted.results.swap(0, 1);
        assert!(
            are_equivalent(&route, &permuted),
            "duplicate occurrences are compared as a name/cost multiset"
        );

        let mut missing_occurrence = route.clone();
        missing_occurrence.results.remove(0);
        missing_occurrence.congestion = route.congestion.clone();
        let discrepancies = compare(&route, &missing_occurrence);
        assert!(discrepancies.contains(&Discrepancy::MissingNet {
            net: "x".into(),
            in_a: true,
        }));
    }

    #[test]
    fn congestion_vectors_must_have_exactly_equal_lengths() {
        let a = base();
        let mut b = base();
        b.congestion.push(0);
        assert!(!are_equivalent(&a, &b));
        assert!(compare(&a, &b).contains(&Discrepancy::CongestionLength { a: 3, b: 4 }));
    }

    #[test]
    fn equivalence_is_symmetric_and_transitive_across_result_permutations() {
        let a = BoardRoute {
            results: vec![
                rr("dup", &[0, 1], 4),
                rr("other", &[1, 2], 7),
                rr("dup", &[2, 3], 2),
            ],
            unrouted: vec!["u2".into(), "u1".into()],
            congestion: vec![1, 2, 2, 1],
            groups: vec![],
        };
        let mut b = a.clone();
        b.results.rotate_left(1);
        b.unrouted.reverse();
        let mut c = b.clone();
        c.results.reverse();
        for (left, right) in [(&a, &b), (&b, &a), (&b, &c), (&a, &c)] {
            assert!(
                are_equivalent(left, right),
                "expected equivalence, got {:?}",
                compare(left, right)
            );
        }
    }

    #[test]
    fn equivalence_relation_laws_hold_over_small_route_family() {
        let duplicate_names = BoardRoute {
            results: vec![
                rr("x", &[0, 1], 3),
                rr("y", &[1, 2], 5),
                rr("x", &[2, 3], 7),
            ],
            unrouted: vec!["u".into(), "v".into()],
            congestion: vec![1, 2, 2, 1],
            groups: vec![10, 20, 10],
        };
        let mut permuted = duplicate_names.clone();
        permuted.results.rotate_right(1);
        permuted.unrouted.reverse();
        let mut alternate_paths = permuted.clone();
        alternate_paths.results[0].path = vec![3, 2];
        alternate_paths.groups = vec![99, 98, 97];
        let mut changed_cost = duplicate_names.clone();
        changed_cost.results[0].cost += 1;
        let mut changed_congestion = duplicate_names.clone();
        changed_congestion.congestion.swap(0, 1);
        let mut longer_congestion = duplicate_names.clone();
        longer_congestion.congestion.push(0);
        let family = [
            duplicate_names,
            permuted,
            alternate_paths,
            changed_cost,
            changed_congestion,
            longer_congestion,
        ];

        for a in &family {
            assert!(are_equivalent(a, a), "equivalence must be reflexive");
        }
        for a in &family {
            for b in &family {
                assert_eq!(
                    are_equivalent(a, b),
                    are_equivalent(b, a),
                    "equivalence must be symmetric"
                );
            }
        }
        for a in &family {
            for b in &family {
                for c in &family {
                    if are_equivalent(a, b) && are_equivalent(b, c) {
                        assert!(are_equivalent(a, c), "equivalence must be transitive");
                    }
                }
            }
        }
    }

    #[test]
    fn paths_are_intentionally_ignored_when_cost_and_congestion_agree() {
        let a = BoardRoute {
            results: vec![rr("n", &[0, 1, 2], 2)],
            unrouted: vec![],
            congestion: vec![1, 1, 1],
            groups: vec![0],
        };
        let b = BoardRoute {
            results: vec![rr("n", &[2, 1, 0], 2)],
            ..a.clone()
        };
        assert!(are_equivalent(&a, &b));
    }

    #[test]
    fn equal_total_with_per_net_cost_swap_reports_each_net_only() {
        let a = BoardRoute {
            results: vec![rr("a", &[0], 2), rr("b", &[1], 5)],
            unrouted: vec![],
            congestion: vec![1, 1],
            groups: vec![],
        };
        let b = BoardRoute {
            results: vec![rr("a", &[0], 5), rr("b", &[1], 2)],
            ..a.clone()
        };
        let discrepancies = compare(&a, &b);
        assert!(!discrepancies
            .iter()
            .any(|d| matches!(d, Discrepancy::TotalCost { .. })));
        assert_eq!(
            discrepancies
                .iter()
                .filter(|d| matches!(d, Discrepancy::NetCost { .. }))
                .count(),
            2
        );
    }
}
