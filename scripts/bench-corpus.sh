#!/usr/bin/env bash
# Route the vendored real-board corpus (benchmarks/corpus/) and render an SVG
# gallery + JSON report into a timestamped run directory (gitignored).
#
# Usage:
#   scripts/bench-corpus.sh                 # whole corpus, negotiated router
#   scripts/bench-corpus.sh srj15           # one sub-corpus
#   ROUTER=lee LAYERS=2 scripts/bench-corpus.sh
#
# Env:
#   ROUTER=negotiated|lee|ripup   routing backend (default: negotiated)
#   LAYERS=N                      override routed layer count (default: per-board)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="${1:-}"
DIR="$ROOT/benchmarks/corpus${SUB:+/$SUB}"
ROUTER="${ROUTER:-negotiated}"
TS="$(date +%Y%m%d-%H%M%S)"
RUN="$ROOT/benchmarks/runs/${TS}-corpus${SUB:+-$SUB}"

LAYER_ARG=()
if [ -n "${LAYERS:-}" ]; then LAYER_ARG=(--layers "$LAYERS"); fi

echo "==> building metalroute (release)"
(cd "$ROOT" && cargo build -p mr-cli --release >/dev/null 2>&1)

echo "==> routing corpus $DIR ($ROUTER) -> $RUN"
"$ROOT/target/release/metalroute" bench-corpus \
  --dir "$DIR" --router "$ROUTER" "${LAYER_ARG[@]+"${LAYER_ARG[@]}"}" \
  --svg-out "$RUN" --out "$RUN/report.json"

echo "==> open $RUN/index.html"
