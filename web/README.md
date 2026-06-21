# metalroute · router process visualiser

An interactive webapp that **animates what the autorouter does**: pick a board and
watch the negotiated-congestion router route nets, build pressure on contested
("over-used") cells, push nets apart as the present-penalty ramps over iterations,
and finally commit a legalized result.

It is a thin React + Vite frontend over the `mr-server` HTTP API. The server
instruments `NegotiatedRouter::route_traced` to emit a per-iteration
[`RouteTrace`](../crates/mr-core/src/lib.rs); the frontend replays it on a timeline.

## Run it

### One process (built SPA served by mr-server)

```bash
# 1. build the frontend → web/dist
cd web && npm install && npm run build && cd ..

# 2. run the server (serves the SPA + API on :1234)
cargo run -p mr-server --release
# open http://localhost:1234
```

### Dev (hot-reload frontend + separate API)

```bash
# terminal 1 — API
cargo run -p mr-server

# terminal 2 — Vite dev server (proxies /api → :1234)
cd web && npm run dev
# open the URL Vite prints (usually http://localhost:5173)
```

The server flags `--corpus-dir <DIR>` (default `benchmarks/corpus`) and
`--web-dir <DIR>` (default `web/dist`) control where boards are read from and which
built SPA is served.

## API

- `GET /api/boards` — list corpus boards (`id`, `corpus`, `name`, `net_count`).
- `GET /api/boards/{corpus}/{name}` — a board's raw SimpleRouteJson.
- `POST /api/trace` — `{ board_id | simple_route_json, layers?, clearance?, resolution? }`
  → `{ trace, coords, layers, bounds, solution }`. `trace` addresses cells by
  `CellIdx`; `coords.x_lines`/`y_lines` + `trace.dims` map them to mm.

## UI

- **Timeline / transport** — play / pause / step / scrub across negotiation
  iterations, then the final legalized frame; speed control.
- **Display toggles** — over-used cells, ratsnest (unrouted nets), per-layer
  visibility.
- **Net list** — click a net to isolate (highlight) it; unrouted nets are marked.
- **Legalization panel** — on the final frame, shows the candidate group orders the
  router evaluated and which it kept.
- **Canvas** — pan (drag), zoom (wheel), `⟲ fit` to reset the view.
