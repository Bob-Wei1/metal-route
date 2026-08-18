#!/usr/bin/env python3
"""research/score.py — decide whether an experiment beat the champion.

Reads `metalroute bench-corpus` CorpusReport JSON. With --compare it prints the
per-group + overall delta and a final verdict line: KEEP or DISCARD.

Objective (higher = better):
  primary   completion_rate          (relative to the supplied baseline report)
  tiebreak  fully_routed_boards

Keep rule (all must hold):
  - completion_rate > baseline + EPS
  - no sub-corpus group regresses by more than EPS
  - fully_routed_boards >= baseline
  - exact same (corpus, board) cases and per-case nets_total
  - no new board errors
  - total DRC violations do not increase
  - DRC-clean and fully-routed-and-clean board counts do not decrease

The DRC gates are derived from current-schema `per_board[].drc_violations`
because CorpusReport does not carry top-level DRC aggregates. Two legacy reports
that both lack the field remain comparable; mixing legacy and current schemas is
rejected because there is no honest DRC baseline.

Usage:
  score.py <report.json>                         # print the scalar + summary
  score.py --compare <baseline.json> <report.json>
"""
import json
import math
import sys

EPS = 0.002  # ignore sub-noise wiggle (routing is deterministic, so this is small)


class ReportError(ValueError):
    """A benchmark report is malformed or internally inconsistent."""


def load(path):
    with open(path) as f:
        return json.load(f)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def require_ratio(actual, expected, label):
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise ReportError("%s must be a finite number" % label)
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ReportError("%s=%r but derived value is %.17g" % (label, actual, expected))


def require_count(actual, expected, label):
    if type(actual) is not int or actual != expected:
        raise ReportError("%s=%r but per_board derives %d" % (label, actual, expected))


def groups(report, cases=None):
    raw = report.get("groups")
    if not isinstance(raw, list):
        raise ReportError("groups must be a list")

    result = {}
    for i, group in enumerate(raw):
        if not isinstance(group, dict) or not isinstance(group.get("name"), str):
            raise ReportError("groups[%d] is missing a string name" % i)
        name = group["name"]
        if name in result:
            raise ReportError("duplicate corpus group %r" % name)
        result[name] = group
    if cases is None:
        return result

    derived = {}
    for board in cases.values():
        aggregate = derived.setdefault(
            board["corpus"],
            {"boards": 0, "nets_total": 0, "nets_routed": 0, "fully_routed_boards": 0},
        )
        aggregate["boards"] += 1
        aggregate["nets_total"] += board["nets_total"]
        aggregate["nets_routed"] += board["nets_routed"]
        aggregate["fully_routed_boards"] += (
            board.get("error") is None
            and board["nets_total"] > 0
            and board["nets_routed"] == board["nets_total"]
        )

    missing = set(derived) - set(result)
    unexpected = set(result) - set(derived)
    if missing:
        raise ReportError("groups missing per-board corpora: " + ", ".join(sorted(missing)))
    if unexpected:
        raise ReportError(
            "groups contain corpora absent from per_board: " + ", ".join(sorted(unexpected))
        )

    for name, expected in derived.items():
        group = result[name]
        for field in ("boards", "nets_total", "nets_routed", "fully_routed_boards"):
            require_count(group.get(field), expected[field], "group %r %s" % (name, field))
        require_ratio(
            group.get("completion_rate"),
            ratio(expected["nets_routed"], expected["nets_total"]),
            "group %r completion_rate" % name,
        )
    return result


def board_cases(report):
    """Return current workload cases keyed by their stable corpus/board identity."""
    raw = report.get("per_board")
    if not isinstance(raw, list):
        raise ReportError("per_board must be a list")
    require_count(report.get("boards"), len(raw), "boards")

    result = {}
    for i, board in enumerate(raw):
        if not isinstance(board, dict):
            raise ReportError("per_board[%d] must be an object" % i)
        corpus = board.get("corpus")
        name = board.get("board")
        nets_total = board.get("nets_total")
        nets_routed = board.get("nets_routed")
        if not isinstance(corpus, str) or not isinstance(name, str):
            raise ReportError(
                "per_board[%d] is missing string corpus/board identity" % i
            )
        if type(nets_total) is not int or nets_total < 0:
            raise ReportError(
                "%s/%s has invalid nets_total %r" % (corpus, name, nets_total)
            )
        if (
            type(nets_routed) is not int or not 0 <= nets_routed <= nets_total
        ):
            raise ReportError(
                "%s/%s has invalid nets_routed %r for nets_total %d"
                % (corpus, name, nets_routed, nets_total)
            )
        if "error" not in board or not (
            board["error"] is None or isinstance(board["error"], str)
        ):
            raise ReportError(
                "%s/%s has invalid or missing error status %r"
                % (corpus, name, board.get("error"))
            )
        identity = (corpus, name)
        if identity in result:
            raise ReportError("duplicate board identity %s" % format_identity(identity))
        result[identity] = board

    aggregate_nets = sum(board["nets_total"] for board in raw)
    require_count(report.get("nets_total"), aggregate_nets, "nets_total")
    aggregate_routed = sum(board["nets_routed"] for board in raw)
    require_count(report.get("nets_routed"), aggregate_routed, "nets_routed")
    aggregate_full = sum(
        board.get("error") is None
        and board["nets_total"] > 0
        and board["nets_routed"] == board["nets_total"]
        for board in raw
    )
    require_count(
        report.get("fully_routed_boards"), aggregate_full, "fully_routed_boards"
    )
    require_ratio(
        report.get("completion_rate"),
        ratio(aggregate_routed, aggregate_nets),
        "completion_rate",
    )
    return result


def drc_stats(cases):
    """Derive DRC gates, or return None for a wholly legacy report."""
    has_drc = ["drc_violations" in board for board in cases.values()]
    if not any(has_drc):
        return None
    if not all(has_drc):
        raise ReportError("only some per_board entries contain drc_violations")

    violations = {}
    for identity, board in cases.items():
        count = board["drc_violations"]
        if not isinstance(count, int) or count < 0:
            raise ReportError(
                "%s has invalid drc_violations %r"
                % (format_identity(identity), count)
            )
        if "error" not in board:
            raise ReportError(
                "%s is missing error status" % format_identity(identity)
            )
        violations[identity] = count

    clean = sum(
        board.get("error") is None and violations[identity] == 0
        for identity, board in cases.items()
    )
    full_clean = sum(
        board.get("error") is None
        and board["nets_total"] > 0
        and board.get("nets_routed") == board["nets_total"]
        and violations[identity] == 0
        for identity, board in cases.items()
    )
    return {
        "violations": sum(violations.values()),
        "clean": clean,
        "full_clean": full_clean,
    }


def format_identity(identity):
    return "%s/%s" % identity


def format_identities(identities, limit=5):
    identities = sorted(identities)
    shown = ", ".join(format_identity(identity) for identity in identities[:limit])
    if len(identities) > limit:
        shown += ", ... (+%d)" % (len(identities) - limit)
    return shown


def summarize(r):
    return "completion=%.3f (%d/%d)  full=%d/%d  boards=%d" % (
        r["completion_rate"], r["nets_routed"], r["nets_total"],
        r["fully_routed_boards"], r["boards"], r["boards"],
    )


def compare(base, cand):
    print("baseline:  " + summarize(base))
    print("candidate: " + summarize(cand))
    dc = cand["completion_rate"] - base["completion_rate"]
    df = cand["fully_routed_boards"] - base["fully_routed_boards"]
    print("delta: completion %+.4f (%+.2f%%)  fully_routed %+d" % (dc, dc * 100, df))

    reasons = []
    try:
        base_cases = board_cases(base)
    except ReportError as exc:
        reasons.append("invalid baseline report: %s" % exc)
        base_cases = None
    try:
        cand_cases = board_cases(cand)
    except ReportError as exc:
        reasons.append("invalid candidate report: %s" % exc)
        cand_cases = None

    if base_cases is not None and cand_cases is not None:
        base_ids = set(base_cases)
        cand_ids = set(cand_cases)
        missing = base_ids - cand_ids
        unexpected = cand_ids - base_ids
        if missing:
            reasons.append("missing boards: " + format_identities(missing))
        if unexpected:
            reasons.append("unexpected boards: " + format_identities(unexpected))
        if not missing and not unexpected:
            changed_nets = [
                identity
                for identity in base_ids
                if base_cases[identity]["nets_total"]
                != cand_cases[identity]["nets_total"]
            ]
            if changed_nets:
                detail = ", ".join(
                    "%s (%d -> %d)"
                    % (
                        format_identity(identity),
                        base_cases[identity]["nets_total"],
                        cand_cases[identity]["nets_total"],
                    )
                    for identity in sorted(changed_nets)[:5]
                )
                if len(changed_nets) > 5:
                    detail += ", ... (+%d)" % (len(changed_nets) - 5)
                reasons.append("per-board nets_total changed: " + detail)

            new_errors = {
                identity
                for identity in base_ids
                if base_cases[identity].get("error") is None
                and cand_cases[identity].get("error") is not None
            }
            if new_errors:
                reasons.append("new board errors: " + format_identities(new_errors))

    try:
        bg = groups(base, base_cases)
    except ReportError as exc:
        reasons.append("invalid baseline report: %s" % exc)
        bg = {}
    try:
        cg = groups(cand, cand_cases)
    except ReportError as exc:
        reasons.append("invalid candidate report: %s" % exc)
        cg = {}

    if bg or cg:
        missing_groups = set(bg) - set(cg)
        unexpected_groups = set(cg) - set(bg)
        if missing_groups:
            reasons.append("missing corpus groups: " + ", ".join(sorted(missing_groups)))
        if unexpected_groups:
            reasons.append(
                "unexpected corpus groups: " + ", ".join(sorted(unexpected_groups))
            )

    regressed = []
    for name in sorted(set(bg) & set(cg)):
        b = bg[name]["completion_rate"]
        c = cg[name]["completion_rate"]
        flag = ""
        if c < b - EPS:
            regressed.append(name)
            flag = "  <-- REGRESSED"
        print("  group %-14s %.3f -> %.3f (%+.4f)%s" % (name, b, c, c - b, flag))

    base_drc = cand_drc = None
    if base_cases is not None and cand_cases is not None:
        try:
            base_drc = drc_stats(base_cases)
        except ReportError as exc:
            reasons.append("invalid baseline report: %s" % exc)
        try:
            cand_drc = drc_stats(cand_cases)
        except ReportError as exc:
            reasons.append("invalid candidate report: %s" % exc)

        if (base_drc is None) != (cand_drc is None):
            reasons.append("DRC schema differs between baseline and candidate")
        elif base_drc is None:
            print("drc: legacy reports (gate unavailable)")
        else:
            dv = cand_drc["violations"] - base_drc["violations"]
            dclean = cand_drc["clean"] - base_drc["clean"]
            dfull_clean = cand_drc["full_clean"] - base_drc["full_clean"]
            print(
                "drc: violations %d -> %d (%+d)  clean %d -> %d (%+d)  "
                "full+clean %d -> %d (%+d)"
                % (
                    base_drc["violations"], cand_drc["violations"], dv,
                    base_drc["clean"], cand_drc["clean"], dclean,
                    base_drc["full_clean"], cand_drc["full_clean"], dfull_clean,
                )
            )
            if dv > 0:
                reasons.append("DRC violations increased (%+d)" % dv)
            if dclean < 0:
                reasons.append("DRC-clean boards dropped (%+d)" % dclean)
            if dfull_clean < 0:
                reasons.append("fully-routed DRC-clean boards dropped (%+d)" % dfull_clean)

    if dc <= EPS:
        reasons.append("completion not improved (delta %+.4f <= EPS %.3f)" % (dc, EPS))
    if regressed:
        reasons.append("groups regressed: " + ", ".join(regressed))
    if cand["fully_routed_boards"] < base["fully_routed_boards"]:
        reasons.append("fully_routed_boards dropped (%d < %d)"
                       % (cand["fully_routed_boards"], base["fully_routed_boards"]))

    if reasons:
        print("VERDICT: DISCARD — " + "; ".join(reasons))
        return 1
    if base_drc is None:
        print("VERDICT: KEEP — completion %+.4f, fully_routed %+d, no group regressed"
              % (dc, df))
    else:
        print(
            "VERDICT: KEEP — completion %+.4f, fully_routed %+d, "
            "DRC violations %+d, no workload/group/DRC regression"
            % (dc, df, cand_drc["violations"] - base_drc["violations"])
        )
    return 0


def main(argv):
    if len(argv) >= 2 and argv[0] == "--compare":
        return compare(load(argv[1]), load(argv[2]))
    if len(argv) == 1:
        r = load(argv[0])
        print(summarize(r))
        print("objective(completion_rate) = %.6f" % r["completion_rate"])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
