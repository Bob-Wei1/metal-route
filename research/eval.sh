#!/usr/bin/env bash
# research/eval.sh — the autoresearch eval: build the router and route the whole
# real-board corpus, emitting a machine-readable CorpusReport. This is the ONLY
# thing the loop measures. Do not edit it to change the metric (that is cheating).
#
# Usage:
#   research/eval.sh [id]      # id labels the report; defaults to a timestamp
#
# Env:
#   RAYON_NUM_THREADS   pin for determinism if ever needed (routing is already
#                       byte-identical across runs, so unset by default)
#   MAX_CELLS           per-board grid guard (default 12,000,000)
#
# Output: research/results/<id>.json  (a CorpusReport — completion_rate,
#         fully_routed_boards, per-group, per-board). Prints a one-line summary.
#
# Exit non-zero on build failure or a failed sweep → the loop treats that as a
# discard. Never use `cargo run` (zombie risk); run the release binary directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID="${1:-$(date +%Y%m%d-%H%M%S)}"
OUT="$ROOT/research/results/${ID}.json"
MAX_CELLS="${MAX_CELLS:-12000000}"
mkdir -p "$ROOT/research/results"

echo "==> building metalroute (release)"
if ! (cd "$ROOT" && cargo build -p mr-cli --release 2>&1 | tail -3); then
  echo "BUILD FAILED — discard this experiment" >&2
  exit 2
fi

echo "==> routing corpus -> $OUT"
timeout -s KILL 300 "$ROOT/target/release/metalroute" bench-corpus \
  --dir "$ROOT/benchmarks/corpus" \
  --max-cells "$MAX_CELLS" \
  --out "$OUT"

# One-line summary straight from the report.
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("RESULT completion=%.2f%% routed=%d/%d full=%d/%d nets/s=%.0f wall=%.1fs" % (
    d["completion_rate"] * 100, d["nets_routed"], d["nets_total"],
    d["fully_routed_boards"], d["boards"], d["nets_per_sec"],
    d["total_wall_ms"] / 1000.0))
PY
