#!/usr/bin/env bash
# Re-vendor the real-board benchmark corpus from tscircuit/tscircuit-autorouter.
#
# Downloads the source repo tarball, normalizes every usable board into a pure
# SimpleRouteJson under benchmarks/corpus/, and rewrites MANIFEST.md. The result
# is checked in; you only need to re-run this to refresh against upstream.
#
# Usage: scripts/vendor-corpus.sh [git-ref]   (default ref: main)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${1:-main}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> downloading tscircuit/tscircuit-autorouter@$REF"
curl -sSL "https://github.com/tscircuit/tscircuit-autorouter/archive/${REF}.tar.gz" \
  -o "$TMP/repo.tgz"
tar xzf "$TMP/repo.tgz" -C "$TMP"
SRC_DIR="$(find "$TMP" -maxdepth 1 -type d -name 'tscircuit-autorouter-*' | head -1)"

# Record the exact commit when the ref is a branch/tag we can resolve.
COMMIT="$REF"
if command -v gh >/dev/null 2>&1; then
  COMMIT="$(gh api "repos/tscircuit/tscircuit-autorouter/commits/${REF}" --jq '.sha' 2>/dev/null || echo "$REF")"
fi

echo "==> normalizing into benchmarks/corpus/"
CORPUS_COMMIT="\`$COMMIT\`" python3 "$ROOT/scripts/vendor-corpus.py" "$SRC_DIR" "$ROOT"
echo "==> done. Review git diff under benchmarks/corpus/ and commit."
