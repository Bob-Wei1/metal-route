# Autoresearch leaderboard

Append-only log of every experiment (kept and discarded). Champion = `research/baseline.json`.

Metric: corpus net `completion_rate` (tiebreak `fully_routed_boards`) via
`metalroute bench-corpus`. See `research/program.md`.

| id | completion | full | verdict | change |
|----|-----------:|-----:|---------|--------|
| baseline | 73.79% (2337/3167) | 32/112 | — | negotiated router @ corpus `11f5e8e` |
| exp-0001 | 73.95% (2342/3167) | 32/112 | discarded | MAX_ITERS 60→120: +5 nets (+0.16% < EPS), ~2× runtime — not worth it |
