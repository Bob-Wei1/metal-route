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

use std::collections::BTreeSet;

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

    // 2. Per-net cost, matched by NAME (order-independent).
    let by_name_a: std::collections::BTreeMap<&str, u32> =
        a.results.iter().map(|r| (r.net.as_str(), r.cost)).collect();
    let by_name_b: std::collections::BTreeMap<&str, u32> =
        b.results.iter().map(|r| (r.net.as_str(), r.cost)).collect();

    // Nets in `a`: matched -> compare cost; unmatched -> missing in `b`.
    // Iterate `a` in its own order so reports are stable for the common case.
    for r in &a.results {
        match by_name_b.get(r.net.as_str()) {
            Some(&cost_b) => {
                if r.cost != cost_b {
                    out.push(Discrepancy::NetCost {
                        net: r.net.clone(),
                        a: r.cost,
                        b: cost_b,
                    });
                }
            }
            None => out.push(Discrepancy::MissingNet {
                net: r.net.clone(),
                in_a: true,
            }),
        }
    }
    // Nets in `b` but not `a`.
    for r in &b.results {
        if !by_name_a.contains_key(r.net.as_str()) {
            out.push(Discrepancy::MissingNet {
                net: r.net.clone(),
                in_a: false,
            });
        }
    }

    // 3. Per-cell congestion. Differing lengths still get reported cell-by-cell;
    //    cells present in only one vector are compared against an implicit 0.
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
}
