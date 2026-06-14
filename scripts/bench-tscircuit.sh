#!/usr/bin/env bash
# Run the real tscircuit autorouting benchmark against mr-server.
#
# Clones github.com/tscircuit/autorouting into .bench/ (gitignored), starts
# mr-server, and runs runBenchmark over the full category suite, scraping the
# completion score per category.
#
# Usage:
#   scripts/bench-tscircuit.sh [sample_count] [category...]
# Examples:
#   scripts/bench-tscircuit.sh                       # 20 samples, all categories
#   scripts/bench-tscircuit.sh 100                   # 100 samples, all categories
#   scripts/bench-tscircuit.sh 50 traces             # 50 samples, just `traces`
#
# Env:
#   MR_SOLVE_LAYERS=N   routing layer budget (default 2). Every problem routes on
#                       max(layerCount, N) layers, so single-layer-declared
#                       categories (traces/keyboards) get the extra layers vias
#                       need to resolve crossings. Set to 1 for the strict
#                       single-layer baseline; bump to 4 for dense keyboards.
#   MR_CLEARANCE=mm     copper clearance budget in mm (trace<->trace and
#                       trace<->pad). Unset => auto (1 trace width, DRC-cleaner
#                       boards). Set MR_CLEARANCE=0 to disable clearance and
#                       reproduce the maximum benchmark (overlap-only) score.
#   VIZ=0               skip the per-run SVG/HTML board gallery (default: render it
#                       into benchmarks/runs/<timestamp>-<N>L/index.html).
#   VIZ_SAMPLES=K       boards/category in the gallery (default 6).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT/.bench/autorouting"
PORT="${MR_PORT:-1234}"
SAMPLES="${1:-20}"
shift || true
CATS=("$@")
if [ ${#CATS[@]} -eq 0 ]; then
  # The full tscircuit suite — every category runBenchmark's "all" expands to.
  CATS=(single-trace distant-single-trace traces keyboards)
fi
SOLVE_LAYERS="${MR_SOLVE_LAYERS:-2}"
# Clearance: pass --clearance only when MR_CLEARANCE is set; otherwise the server
# uses its auto default (1 trace width).
CLR_ARG=()
if [ -n "${MR_CLEARANCE:-}" ]; then CLR_ARG=(--clearance "$MR_CLEARANCE"); fi

# 1. Clone + install harness if missing.
if [ ! -d "$BENCH_DIR/.git" ]; then
  echo "==> cloning tscircuit/autorouting"
  mkdir -p "$ROOT/.bench"
  git clone --depth 1 https://github.com/tscircuit/autorouting "$BENCH_DIR"
fi
if [ ! -d "$BENCH_DIR/node_modules" ]; then
  echo "==> bun install"
  (cd "$BENCH_DIR" && bun install)
fi
cp "$ROOT/scripts/tscircuit-bench-run.ts" "$BENCH_DIR/bench-run.ts"
cp "$ROOT/scripts/tscircuit-bench-viz.ts" "$BENCH_DIR/bench-viz.ts"

# 2. Build + start mr-server.
echo "==> building mr-server (release)"
(cd "$ROOT" && cargo build -p mr-server --release >/dev/null 2>&1)
"$ROOT/target/release/mr-server" --port "$PORT" --solve-layers "$SOLVE_LAYERS" \
  "${CLR_ARG[@]+"${CLR_ARG[@]}"}" >/tmp/mr-server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
  if curl -sS -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
echo "==> mr-server up on :$PORT (pid $SERVER_PID), >= $SOLVE_LAYERS layers, clearance=${MR_CLEARANCE:-auto}"

# 3. Run each category, scrape the RESULT line.
echo
printf '%-26s %12s %10s\n' "category" "completion" "avg"
printf '%-26s %12s %10s\n' "--------" "----------" "---"
for cat in "${CATS[@]}"; do
  line=$(cd "$BENCH_DIR" && SOLVER_URL="http://localhost:$PORT/solve" \
    PROBLEM_TYPE="$cat" SAMPLE_COUNT="$SAMPLES" SAMPLE_SEED=0 \
    bun bench-run.ts 2>/dev/null | grep '^RESULT' || true)
  if [ -z "$line" ]; then
    printf '%-26s %12s %10s\n' "$cat" "ERROR" "-"
  else
    # RESULT <cat> <s>/<n> <pct>% avg=<t>ms
    pct=$(echo "$line" | awk '{print $4}')
    frac=$(echo "$line" | awk '{print $3}')
    avg=$(echo "$line" | awk '{print $5}' | sed 's/avg=//')
    printf '%-26s %12s %10s\n' "$cat" "$pct ($frac)" "$avg"
  fi
done

# 4. Render a visual gallery of the routed boards for this run (unless VIZ=0).
#    Writes one PCB SVG per board + a self-contained index.html into a timestamped
#    run directory (gitignored). VIZ_SAMPLES caps boards/category (default 6).
if [ "${VIZ:-1}" != "0" ]; then
  RUN_DIR="$ROOT/benchmarks/runs/$(date +%Y%m%d-%H%M%S)-${SOLVE_LAYERS}L"
  echo
  echo "==> rendering board gallery -> $RUN_DIR"
  (cd "$BENCH_DIR" && SOLVER_URL="http://localhost:$PORT/solve" \
    VIZ_OUT_DIR="$RUN_DIR" VIZ_CATEGORIES="$(IFS=,; echo "${CATS[*]}")" \
    VIZ_SAMPLES="${VIZ_SAMPLES:-6}" VIZ_SEED="${SAMPLE_SEED:-0}" \
    MR_SOLVE_LAYERS="$SOLVE_LAYERS" \
    bun bench-viz.ts) || echo "   (gallery render failed; see above)"
  echo "==> open $RUN_DIR/index.html"
fi
