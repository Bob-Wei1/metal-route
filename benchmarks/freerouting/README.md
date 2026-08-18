# Freerouting speed comparison

This directory defines the public methodology and JSON schema for comparing
metalroute with [Freerouting 2.3.0](https://github.com/freerouting/freerouting/releases/tag/v2.3.0).
It intentionally contains no Freerouting JAR and no third-party DSN fixtures.

The comparison answers a narrow question: on the same machine and the same DSN,
how long does a fresh process take to load, route, and write a Specctra session?
It does not claim that metalroute and Freerouting are equivalent algorithms.
metalroute is an experimental global/maze router; Freerouting is a detailed
autorouter with fanout and optimization phases.

## Reproduce

Build metalroute in release mode, obtain the official Freerouting 2.3.0
executable JAR independently, and choose DSN files with documented provenance:

```sh
cargo build --locked --release -p mr-cli

python3 scripts/bench-freerouting.py \
  --metalroute target/release/metalroute \
  --freerouting-jar /path/to/freerouting-2.3.0.jar \
  --official-fixture-dir /path/to/freerouting/scripts/benchmark/fixtures/DAC2020_boards
```

Generated JSON, Markdown, logs, DRC reports, and SES files go to the ignored
`benchmarks/runs/<timestamp>-freerouting/` directory by default. The harness
never downloads or copies the JAR or DSNs into this repository. Fixture SHA-256
hashes and binary SHA-256 hashes make independent runs identifiable.

Freerouting publishes its own benchmark DSNs under
[`scripts/benchmark/fixtures`](https://github.com/freerouting/freerouting/tree/v2.3.0/scripts/benchmark/fixtures).
The default smoke profile selects `DAC2020_bm08.dsn`, `DAC2020_bm06.dsn`, and
`DAC2020_bm07.dsn` from an external copy of that directory. The harness records
the exact v2.3.0 upstream paths and rejects fixture or release-JAR hash
mismatches. These are a quick two-layer smoke set, not Freerouting's complete
benchmark corpus. Keep the files outside this repository.

The official v2.3.0 JAR SHA-256 is
`3cf18d608437740bc497db6b8ef5888e2e60a08de0def20691d1bad0c0e0ee24`.
The selected fixture SHA-256 values are:

- `DAC2020_bm08.dsn`: `5d3acaaac47c1851d439150e3b70751b85fe1e8b8afc55278f1487b692b32bc5`
- `DAC2020_bm06.dsn`: `31f38102d90a1bb4b901d4ca8d1877eb41752281ffa9de9f53a3cf69ba5231e2`
- `DAC2020_bm07.dsn`: `39d85afa3133caae9b274350183868ad1fce5a0c64e3d5c6874598a899007c85`

## Measurement contract

- The timed interval is external wall time around a new process: DSN load,
  routing, and SES export. It includes normal process/JVM startup.
- The default statistic is the median of three runs. Engine launch order
  alternates each repetition.
- Router worker pools are capped at one with `RAYON_NUM_THREADS=1` for
  metalroute and `--router.max_threads=1` for Freerouting. This does not assert
  that every auxiliary runtime, GC, or OS thread is disabled.
- Freerouting follows its official v2.3.0 benchmark profile: 500 maximum passes,
  one router worker, an 8 GiB heap ceiling, a 30-minute routing-job timeout, a
  15-minute fanout timeout, and a 10-minute optimizer timeout. Fanout, routing,
  and optimization are enabled. A “pass” is engine-specific and is not treated
  as equal work between the two programs.
- metalroute clears `METALROUTE_EXPERIMENTAL_METAL_ISOLATED` and
  `MR_CELL_BUDGET` before both its ingest probe and timed runs, then sets
  `RAYON_NUM_THREADS=1`. This prevents a caller's experimental backend or grid
  budget from silently changing the profile.
- Both engines receive the byte-identical DSN and must emit a non-empty SES.
- Validation is outside the timed interval. Freerouting 2.3.0 reloads each pair
  as `INPUT.dsn+OUTPUT.ses` and writes its JSON DRC report. This follows
  Freerouting's own
  [`DrcRunner.ps1`](https://github.com/freerouting/freerouting/blob/v2.3.0/scripts/benchmark/lib/DrcRunner.ps1)
  benchmark validation path.

The comparison gate is conservative. Before routing, a zero-net metalroute ingest
probe and a Freerouting baseline DRC probe must both succeed. During routing,
metalroute's two-point task count must equal the task count in Freerouting's
completed autorouter log, and every SES must reload.

Count equality is necessary, not sufficient, for semantic equivalence: the
programs may still interpret a DSN rule or geometry feature differently. For
that reason each result reports post-reload unconnected items and violations
beside wall time. Raw median times remain visible, but a ratio is published only
when the faster engine is no worse in both unconnected-item and violation count.
Freerouting's aggregate quality score is retained as diagnostics but is not
compared with metalroute's unrelated internal cost.

## Report contract

[`report.schema.json`](report.schema.json) is the machine-readable schema. The
important states are:

- `compatible`, `incompatible`, `probe_error`, or `probe_timeout` for input
  probes;
- `matched`, `mismatched`, or `unavailable` for the cross-parser workload gate;
- `route_timeout`, `route_error`, `missing_ses`, `reload_error`, or `ok` for each
  repetition;
- `complete` when both measured run sets and every SES reload complete under the
  input/workload gates. The quality-gated ratio may still be `—`.

The generated Markdown is a compact public table. The JSON remains the source
of truth for individual samples, hashes, commands, compatibility evidence, and
DRC quality.
