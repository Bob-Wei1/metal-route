#!/usr/bin/env bash
# Run the real tscircuit autorouting benchmark against mr-server.
#
# Clones github.com/tscircuit/autorouting into .bench/ (gitignored), starts
# mr-server, and runs runBenchmark over the single-layer "ready" categories,
# scraping the completion score per category.
#
# Usage:
#   scripts/bench-tscircuit.sh [sample_count] [category...]
# Examples:
#   scripts/bench-tscircuit.sh                       # 20 samples, ready categories
#   scripts/bench-tscircuit.sh 100                   # 100 samples, ready categories
#   scripts/bench-tscircuit.sh 50 traces             # 50 samples, just `traces`
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT/.bench/autorouting"
PORT="${MR_PORT:-1234}"
SAMPLES="${1:-20}"
shift || true
CATS=("$@")
if [ ${#CATS[@]} -eq 0 ]; then
  CATS=(single-trace distant-single-trace traces)
fi

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

# 2. Build + start mr-server.
echo "==> building mr-server (release)"
(cd "$ROOT" && cargo build -p mr-server --release >/dev/null 2>&1)
"$ROOT/target/release/mr-server" --port "$PORT" >/tmp/mr-server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
  if curl -sS -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
echo "==> mr-server up on :$PORT (pid $SERVER_PID)"

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
