#!/usr/bin/env python3
"""Vendor real tscircuit boards into benchmarks/corpus/ as normalized SRJ.

Source: github.com/tscircuit/tscircuit-autorouter (MIT). Each output file is a
pure SimpleRouteJson object (the `simple_route_json` payload unwrapped from any
bug-report envelope), so `metalroute bench-corpus` consumes them with no
conversion. Invoked by scripts/vendor-corpus.sh, which downloads the source
tarball first and passes its path as argv[1].

Usage: vendor-corpus.py <extracted-source-dir> <repo-root>
"""
import json
import glob
import os
import re
import sys

SRC_ROOT = sys.argv[1]  # .../tscircuit-autorouter-<ref>
REPO = sys.argv[2]
COMMIT = os.environ.get("CORPUS_COMMIT", "main")

SRC = os.path.join(SRC_ROOT, "fixtures")
DST = os.path.join(REPO, "benchmarks", "corpus")


def is_srj(d):
    return isinstance(d, dict) and "bounds" in d and ("connections" in d or "obstacles" in d)


def extract_srj(d):
    """Unwrap a SimpleRouteJson from a raw board or a bug-report envelope."""
    if is_srj(d):
        return d
    srj = d.get("simple_route_json")
    if isinstance(srj, dict) and "bounds" in srj:
        return srj
    return None


def slug(name):
    return re.sub(r"[^a-zA-Z0-9._-]", "-", name)


def vendor(src_files, out_dir, rename=None):
    # Start clean so deleted-upstream boards don't linger.
    if os.path.isdir(out_dir):
        for f in glob.glob(f"{out_dir}/*.srj.json"):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in sorted(src_files):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  SKIP (parse) {os.path.basename(f)}: {e}")
            continue
        srj = extract_srj(d)
        if srj is None:
            print(f"  SKIP (no srj) {os.path.basename(f)}")
            continue
        if len(srj.get("connections", [])) == 0:
            print(f"  SKIP (0 nets) {os.path.basename(f)}")
            continue
        base = rename(f) if rename else os.path.basename(f)
        if base.endswith(".json") and not base.endswith(".srj.json"):
            base = base[:-5] + ".srj.json"
        elif not base.endswith(".srj.json"):
            base += ".srj.json"
        out = os.path.join(out_dir, slug(base))
        json.dump(srj, open(out, "w"), separators=(",", ":"))
        rows.append((slug(base), len(srj.get("connections", [])),
                     len(srj.get("obstacles", [])), srj.get("layerCount", 1)))
    return rows


def br_name(f):
    rel = os.path.relpath(f, f"{SRC}/bug-reports")
    parent = os.path.dirname(rel)
    return parent.split("/")[0] if parent else os.path.basename(f)[:-5]


print("== srj15 ==")
srj15 = vendor(glob.glob(f"{SRC}/datasets/dataset-srj15/*.srj.json"), f"{DST}/srj15")
print(f"  vendored {len(srj15)} boards")

print("== bug-reports ==")
br = vendor(glob.glob(f"{SRC}/bug-reports/**/*.json", recursive=True),
            f"{DST}/bug-reports", rename=br_name)
print(f"  vendored {len(br)} boards")

total = len(srj15) + len(br)
tot_conn = sum(r[1] for r in srj15 + br)
with open(f"{DST}/MANIFEST.md", "w") as m:
    m.write(f"""# Real-board benchmark corpus

Vendored from [tscircuit/tscircuit-autorouter](https://github.com/tscircuit/tscircuit-autorouter)
(MIT, © 2025 tscircuit) at `{COMMIT}`.

Each file is a pure **SimpleRouteJson** object — the `simple_route_json` payload
unwrapped from any bug-report envelope — so `metalroute bench-corpus` consumes
them with no conversion. These are real circuit-derived routing problems (real
pad layouts, multi-layer, real net connectivity), unlike the synthetic
`metalroute bench` generator.

| corpus | boards | connections | what it is |
|--------|-------:|------------:|------------|
| `srj15/` | {len(srj15)} | {sum(r[1] for r in srj15)} | multi-net region-reroute boards |
| `bug-reports/` | {len(br)} | {sum(r[1] for r in br)} | real designs + reported failure cases (arduino-uno, esp32-breakout, LGA15x4, …) |
| **total** | **{total}** | **{tot_conn}** | |

Regenerate with `scripts/vendor-corpus.sh`. Run the benchmark with
`scripts/bench-corpus.sh` (or `metalroute bench-corpus --svg-out <dir>`).
""")
print(f"\nTOTAL vendored: {total} boards, {tot_conn} connections -> {DST}")
