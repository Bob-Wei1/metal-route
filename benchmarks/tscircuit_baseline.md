# tscircuit autorouting benchmark — baseline

Real harness: [`tscircuit/autorouting`](https://github.com/tscircuit/autorouting)
`runBenchmark` driving `mr-server`'s `POST /solve` over HTTP. Reproduce with:

```sh
scripts/bench-tscircuit.sh 100        # 100 samples/category, ready categories
```

A sample counts as **complete** only if the returned solution passes the
harness DRC checks (`@tscircuit/checks`):
`checkEachPcbTraceNonOverlapping` + `checkEachPcbPortConnected` — i.e. every
port connected and no trace overlapping a pad or another trace.

## Baseline (commit before Phase 1), 15 samples/category

| category               | completion | avg    |
|------------------------|-----------:|-------:|
| single-trace           | 0.0% (0/15)| —      |
| distant-single-trace   | 0.0% (0/15)| —      |
| traces                 | 0.0% (0/15)| —      |

**Root cause (not algorithmic):** real SimpleRouteJson lists every net's pads as
obstacles (with `connectedTo`), and each connection's endpoints sit *on* their
own pad. `mr-srj::rasterize` marks those cells as hard `OBSTACLE`s and never
clears the net's own endpoint cells, so `mr-server` returns HTTP 400
`endpoint out of bounds or on an obstacle` (`mr-core::RouterError::InvalidEndpoint`)
for every sample. Fixed in Phase 1.
