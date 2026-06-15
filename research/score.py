#!/usr/bin/env python3
"""research/score.py — decide whether an experiment beat the champion.

Reads `metalroute bench-corpus` CorpusReport JSON. With --compare it prints the
per-group + overall delta and a final verdict line: KEEP or DISCARD.

Objective (higher = better):
  primary   completion_rate          (baseline 0.738)
  tiebreak  fully_routed_boards

Keep rule (all must hold):
  - completion_rate > baseline + EPS
  - no sub-corpus group regresses by more than EPS
  - fully_routed_boards >= baseline
  - same set of boards scored (a shrunken sweep is a discard, not a win)

Usage:
  score.py <report.json>                         # print the scalar + summary
  score.py --compare <baseline.json> <report.json>
"""
import json
import sys

EPS = 0.002  # ignore sub-noise wiggle (routing is deterministic, so this is small)


def load(path):
    with open(path) as f:
        return json.load(f)


def groups(report):
    return {g["name"]: g for g in report.get("groups", [])}


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

    bg, cg = groups(base), groups(cand)
    regressed = []
    for name in sorted(bg):
        b = bg[name]["completion_rate"]
        c = cg.get(name, {}).get("completion_rate", 0.0)
        flag = ""
        if c < b - EPS:
            regressed.append(name)
            flag = "  <-- REGRESSED"
        print("  group %-14s %.3f -> %.3f (%+.4f)%s" % (name, b, c, c - b, flag))

    reasons = []
    if cand["boards"] < base["boards"]:
        reasons.append("fewer boards scored (%d < %d)" % (cand["boards"], base["boards"]))
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
    print("VERDICT: KEEP — completion %+.4f, fully_routed %+d, no group regressed"
          % (dc, df))
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
