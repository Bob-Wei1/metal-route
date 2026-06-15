# Real-board benchmark corpus

Vendored from [tscircuit/tscircuit-autorouter](https://github.com/tscircuit/tscircuit-autorouter)
(MIT, © 2025 tscircuit) at ``11f5e8ea84a4a737e005dd1863d91ddec4457eee``.

Each file is a pure **SimpleRouteJson** object — the `simple_route_json` payload
unwrapped from any bug-report envelope — so `metalroute bench-corpus` consumes
them with no conversion. These are real circuit-derived routing problems (real
pad layouts, multi-layer, real net connectivity), unlike the synthetic
`metalroute bench` generator.

| corpus | boards | connections | what it is |
|--------|-------:|------------:|------------|
| `srj15/` | 55 | 720 | multi-net region-reroute boards |
| `bug-reports/` | 57 | 1225 | real designs + reported failure cases (arduino-uno, esp32-breakout, LGA15x4, …) |
| **total** | **112** | **1945** | |

Regenerate with `scripts/vendor-corpus.sh`. Run the benchmark with
`scripts/bench-corpus.sh` (or `metalroute bench-corpus --svg-out <dir>`).
