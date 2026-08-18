//! `mr-server` — the tscircuit autorouting-dataset HTTP solver endpoint.
//!
//! This crate exposes the metalroute Rust router over HTTP so the official
//! tscircuit `autorouting-dataset benchmark --solver-url ...` harness can drive
//! it. The protocol the harness speaks is small:
//!
//! * `POST /solve` with a JSON body `{ "simple_route_json": <SimpleRouteJson>,
//!   "problem_soup": <ignored>, "resolution": <optional f64> }`. The response is
//!   `{ "solution_soup": [ <pcb_trace>, ... ] }`.
//! * `GET /health` — liveness, returns `200 OK`.
//!
//! The actual routing is done by any [`mr_core::Router`] injected as an
//! `Arc<dyn Router + Send + Sync>`, so the CPU routers today and a Metal backend
//! later are drop-in swappable without touching the HTTP layer.
//!
//! ## Resolution policy
//!
//! The continuous problem must be rasterised to a grid before routing. The cell
//! size (`resolution`, in continuous tscircuit units) is chosen as follows, in
//! priority order:
//!
//! 1. An explicit `"resolution"` field in the request body, if present and
//!    finite & positive, wins — the caller knows best.
//! 2. Otherwise it is derived from the problem bounds so the grid stays a
//!    reasonable size regardless of board scale: we target roughly
//!    [`TARGET_CELLS_PER_AXIS`] cells across the longer span, i.e.
//!    `resolution = max_span / TARGET_CELLS_PER_AXIS`.
//! 3. The derived value is floored at [`MIN_RESOLUTION`] so a tiny board can
//!    never produce an absurdly fine (and slow / huge) grid, and is treated as
//!    [`MIN_RESOLUTION`] if the bounds are degenerate (zero span).
//!
//! Targeting ~200 cells/axis keeps the worst-case grid near 200×200 = 40k cells,
//! which Lee/rip-up handle comfortably while still resolving fixture-scale pad
//! pitches.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use mr_core::{GridCoords, LayerMap, RouteTrace, Router, RouterError, ViaModel};
use mr_cpu::NegotiatedRouter;
use mr_srj::{
    rasterize_with_layers, rasterize_with_uniform_physical_rules, to_solution_layered, Bounds,
    PcbTrace, RasterizedProblem, SimpleRouteJson,
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, services::ServeDir};

/// Target number of grid cells along the longer bounds span when deriving a
/// default resolution. See the module-level "Resolution policy" docs.
pub const TARGET_CELLS_PER_AXIS: f64 = 200.0;

/// Floor on the derived cell size, in continuous units, so a small board cannot
/// yield an unreasonably fine grid.
pub const MIN_RESOLUTION: f64 = 0.1;

/// Default trace width emitted in the solution soup (continuous units).
pub const DEFAULT_TRACE_WIDTH: f64 = 0.15;

/// Default layer name for emitted (single-layer) traces.
pub const DEFAULT_LAYER: &str = "top";

/// Default routing layer budget when the caller does not raise it. Single-layer
/// problems (`layerCount=1`) are routed on this many layers so the negotiated
/// router can escape crossings with through-vias — the tscircuit harness checks
/// only connectivity + non-overlap, so multi-layer routes are legal.
pub const DEFAULT_SOLVE_LAYERS: u32 = 2;

/// Builds a routing backend for a given per-problem geometric clearance budget (in
/// continuous units, e.g. mm) and the problem's non-uniform grid [`GridCoords`] (the
/// Hanan line arrays). Both are board-dependent, so they cannot be baked into a
/// single shared router; instead `main.rs` supplies this factory and `lib.rs`
/// stays backend-agnostic (a Metal backend can honour or ignore either input).
/// The CPU [`NegotiatedRouter`] prices A* moves by the
/// geometric step length read off these coords AND keeps the inter-net clearance
/// halo at `clearance_mm` over the same coords, so passing the real Hanan coords
/// (rather than a uniform fallback) is what keeps cost/heuristic/spacing correct on
/// the non-uniform grid.
#[derive(Clone, Debug)]
pub struct RouterConfig {
    /// Foreign trace centreline spacing over the non-uniform grid.
    pub clearance_mm: f64,
    /// Dedicated foreign via centre-to-centre spacing.
    pub via_spacing_mm: f64,
    /// Net-independent drill-centre spacing, including within one routed net.
    pub via_hole_spacing_mm: f64,
    /// Symmetric committed-via↔planar-trace enforcement for coherent typed rules.
    pub committed_via_to_trace_guard: bool,
    /// Through/blind/buried transitions plus via-to-trace keepout.
    pub via_model: ViaModel,
    /// Physical Hanan line coordinates used by every geometric distance.
    pub coords: GridCoords,
}

/// Backward-compatible backend factory used by the original public `app`/`serve`
/// entry points. It receives only the historical generic clearance and grid
/// coordinates.
pub type RouterFactory =
    Arc<dyn Fn(f64, GridCoords) -> Box<dyn Router + Send + Sync> + Send + Sync>;

/// Feature-aware backend factory used by the product path. Unlike
/// [`RouterFactory`], this receives the coherent supported typed-rule projection.
pub type ConfiguredRouterFactory =
    Arc<dyn Fn(RouterConfig) -> Box<dyn Router + Send + Sync> + Send + Sync>;

/// Build the production negotiated backend from a prepared per-board profile.
/// `/api/trace`, the binary's `/solve` factory, and tests all share this function
/// so none can silently omit typed via geometry or spacing.
pub fn configured_negotiated_router(config: RouterConfig) -> NegotiatedRouter {
    NegotiatedRouter::new()
        .with_clearance_mm(config.clearance_mm)
        .with_via_spacing_mm(config.via_spacing_mm)
        .with_via_hole_spacing_mm(config.via_hole_spacing_mm)
        .with_committed_via_to_trace_guard(config.committed_via_to_trace_guard)
        .with_via_model(config.via_model)
        .with_coords(config.coords)
}

/// Shared `/solve` state: the router factory plus the layer + clearance policy.
///
/// A problem is routed on `max(simple_route_json.layerCount, solve_layers)`
/// layers, so the declared stackup is never reduced but single-layer-declared
/// problems still get the extra layers vias need to resolve crossings.
#[derive(Clone)]
struct AppState {
    make_router: ConfiguredRouterFactory,
    solve_layers: u32,
    /// Clearance budget in continuous units. `None` activates a coherent supported
    /// typed-rule projection when available, otherwise the legacy auto policy;
    /// `Some(mm)` is a fixed legacy budget, with `Some(0.0)` disabling clearance.
    clearance_mm: Option<f64>,
    /// Root directory of the board corpus, scanned by the `/api/boards*` routes.
    /// Boards live at `<corpus_dir>/<corpus>/<name>.srj.json`.
    corpus_dir: PathBuf,
}

/// Request body for `POST /solve`.
///
/// `problem_soup` is accepted but ignored for now (the harness may send it). An
/// optional `resolution` overrides the derived cell size.
#[derive(Debug, Deserialize)]
struct SolveRequest {
    simple_route_json: SimpleRouteJson,
    #[serde(default)]
    #[allow(dead_code)]
    problem_soup: Option<serde_json::Value>,
    #[serde(default)]
    resolution: Option<f64>,
}

/// Successful `POST /solve` response.
#[derive(Debug, Serialize)]
struct SolveResponse {
    solution_soup: Vec<PcbTrace>,
}

/// JSON error body returned on a 4xx/5xx.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Choose the rasterisation resolution for a problem per the module policy.
///
/// `override_res` is the optional caller-supplied value; when it is `Some` and
/// finite & strictly positive it is used verbatim. Otherwise the value is
/// derived from `srj`'s bounds and floored at [`MIN_RESOLUTION`].
pub fn choose_resolution(srj: &SimpleRouteJson, override_res: Option<f64>) -> f64 {
    if let Some(r) = override_res {
        if r.is_finite() && r > 0.0 {
            return r;
        }
    }
    let b = &srj.bounds;
    let span_x = (b.max_x - b.min_x).max(0.0);
    let span_y = (b.max_y - b.min_y).max(0.0);
    let max_span = span_x.max(span_y);
    if max_span <= 0.0 {
        return MIN_RESOLUTION;
    }
    let mut res = (max_span / TARGET_CELLS_PER_AXIS).max(MIN_RESOLUTION);
    // Real boards pack 0.1mm traces between ~0.38mm pads pitched ~0.8mm apart;
    // the bounds-derived cell is far too coarse to fit a trace between pads.
    // Cap it at ~2 trace widths so routing has room, while keeping the
    // bounds-based ceiling so a large board never explodes into a huge grid.
    if let Some(w) = srj.min_trace_width {
        if w.is_finite() && w > 0.0 {
            res = res.min((w * 2.0).max(MIN_RESOLUTION));
        }
    }
    res
}

/// The rasterised problem plus the derived geometry/policy values both the `/solve`
/// and `/api/trace` handlers need. Produced by [`prepare`] so the two routes share
/// one identical rasterisation + clearance pipeline.
struct Prepared {
    problem: RasterizedProblem,
    /// Coherent supported per-board geometry handed to both router entry points.
    router: RouterConfig,
    /// Emitted wire width in continuous units.
    trace_width: f64,
}

/// Shared rasterise-and-policy step for `/solve` and `/api/trace`.
///
/// Chooses the resolution, layer budget (`max(layerCount, solve_layers)`), and
/// clearance budget (explicit `clearance_policy` override, else the coherent
/// supported typed-rule projection when available, otherwise the legacy auto
/// policy), then rasterises into a non-uniform Hanan grid and builds the matching
/// [`GridCoords`].
fn prepare(
    srj: &SimpleRouteJson,
    override_res: Option<f64>,
    solve_layers: u32,
    clearance_policy: Option<f64>,
) -> Prepared {
    let resolution = choose_resolution(srj, override_res);
    let effective_layers = srj.layer_count.max(solve_layers);
    // A server/request clearance override retains its established meaning and opts
    // out of typed rules (notably, `0` still disables all clearance). With no
    // override, only a coherent supported uniform projection activates the typed path.
    let physical = clearance_policy
        .is_none()
        .then(|| srj.uniform_physical_rules())
        .flatten();
    let trace_width = physical
        .map(|rules| rules.trace_width_mm)
        .or(srj.min_trace_width)
        .unwrap_or(DEFAULT_TRACE_WIDTH);
    let edge_clearance_mm = clearance_policy.unwrap_or_else(|| {
        physical.map_or_else(
            || srj.min_clearance.unwrap_or(0.0).max(trace_width),
            |rules| rules.obstacle_margin_mm,
        )
    });
    let layers = LayerMap::standard(effective_layers);
    let problem = match physical {
        Some(rules) => rasterize_with_uniform_physical_rules(srj, resolution, layers, rules),
        None => {
            // Convert the legacy clearance to a cell halo radius `ceil(mm / resolution)`,
            // fed to both the compatibility rasteriser and router. `0` => off.
            let clearance_cells = if edge_clearance_mm > 0.0 && resolution > 0.0 {
                (edge_clearance_mm / resolution).ceil() as u32
            } else {
                0
            };
            rasterize_with_layers(
                srj,
                resolution,
                layers,
                clearance_cells,
                edge_clearance_mm,
                0.0,
            )
        }
    };
    let coords = GridCoords::from_lines(
        problem.mapping.x_lines.clone(),
        problem.mapping.y_lines.clone(),
    );
    let (clearance_mm, via_spacing_mm, via_hole_spacing_mm, via_model) = physical.map_or_else(
        || {
            (
                edge_clearance_mm,
                0.0,
                0.0,
                ViaModel::through_hole(effective_layers),
            )
        },
        |rules| {
            let mut via_model = ViaModel::through_hole(effective_layers);
            via_model.keepout_mm = rules.via_pad_diameter_mm / 2.0
                + rules.trace_to_pad_clearance_mm
                + rules.trace_width_mm / 2.0;
            let via_spacing_mm = (rules.via_pad_diameter_mm + rules.obstacle_margin_mm).max(
                rules.via_hole_diameter_mm + rules.via_hole_to_hole_clearance_mm.unwrap_or(0.0),
            );
            let via_hole_spacing_mm = rules
                .via_hole_to_hole_clearance_mm
                .map(|clearance| rules.via_hole_diameter_mm + clearance)
                .unwrap_or(0.0);
            (
                rules.obstacle_margin_mm + rules.trace_width_mm,
                via_spacing_mm,
                via_hole_spacing_mm,
                via_model,
            )
        },
    );
    Prepared {
        problem,
        router: RouterConfig {
            clearance_mm,
            via_spacing_mm,
            via_hole_spacing_mm,
            committed_via_to_trace_guard: physical.is_some(),
            via_model,
            coords,
        },
        trace_width,
    }
}

/// De-rasterise the board identically for `/solve` and `/api/trace`.
fn solution_from_board(prep: &Prepared, board: &mr_core::BoardRoute) -> Vec<PcbTrace> {
    to_solution_layered(
        board,
        &prep.problem.mapping,
        &prep.problem.pin_points,
        prep.trace_width,
        &prep.problem.layers,
    )
}

/// The ordered layer names of an effective stackup (`["top", "bottom", ...]`).
fn layer_names(layers: &LayerMap) -> Vec<String> {
    (0..layers.len())
        .map(|i| layers.name(i).to_string())
        .collect()
}

/// `POST /solve` handler.
async fn solve(
    State(state): State<AppState>,
    body: Result<Json<SolveRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => return bad_request(format!("invalid request body: {rej}")),
    };

    let prep = prepare(
        &req.simple_route_json,
        req.resolution,
        state.solve_layers,
        state.clearance_mm,
    );
    let router = (state.make_router)(prep.router.clone());
    let board = match router.route(&prep.problem.grid, &prep.problem.nets) {
        Ok(b) => b,
        Err(e) => return router_error_response(e),
    };

    let solution_soup = solution_from_board(&prep, &board);
    (StatusCode::OK, Json(SolveResponse { solution_soup })).into_response()
}

/// Map a [`RouterError`] to an HTTP response.
///
/// Endpoint/grid problems are caller faults (`400`); genuine backend failures
/// are `500`. Note: per-net "no path" is NOT an error — the router reports it via
/// `unrouted` and we simply emit fewer traces.
fn router_error_response(e: RouterError) -> Response {
    let status = match e {
        RouterError::InvalidEndpoint { .. } | RouterError::MalformedGrid => StatusCode::BAD_REQUEST,
        RouterError::RipUpExhausted { .. } | RouterError::BackendUnavailable(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    (
        status,
        Json(ErrorResponse {
            error: e.to_string(),
        }),
    )
        .into_response()
}

/// Build a `400 Bad Request` JSON response.
fn bad_request(msg: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg })).into_response()
}

/// `GET /health` handler — liveness.
async fn health() -> StatusCode {
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// Visualiser API: list corpus boards, fetch a board, and produce a route trace.
// ---------------------------------------------------------------------------

/// One corpus board in the `/api/boards` listing.
#[derive(Debug, Serialize)]
struct BoardInfo {
    /// Stable id `"<corpus>/<name>"`, e.g. `"bug-reports/bugreport01-be84eb"`.
    id: String,
    /// Sub-corpus directory name, e.g. `"bug-reports"` or `"srj15"`.
    corpus: String,
    /// Board file stem (without the `.srj.json` suffix).
    name: String,
    /// Number of connections (nets) declared in the board.
    net_count: usize,
}

/// `GET /api/boards` — scan the corpus directory for `*.srj.json` boards.
async fn list_boards(State(state): State<AppState>) -> Response {
    match scan_boards(&state.corpus_dir) {
        Ok(mut boards) => {
            boards.sort_by(|a, b| a.id.cmp(&b.id));
            (StatusCode::OK, Json(boards)).into_response()
        }
        Err(e) => internal_error(format!("cannot scan corpus: {e}")),
    }
}

/// Walk `<corpus_dir>/<sub>/*.srj.json` one level deep and summarise each board.
fn scan_boards(corpus_dir: &Path) -> std::io::Result<Vec<BoardInfo>> {
    let mut out = Vec::new();
    for sub in std::fs::read_dir(corpus_dir)? {
        let sub = sub?;
        if !sub.file_type()?.is_dir() {
            continue;
        }
        let corpus = sub.file_name().to_string_lossy().into_owned();
        for entry in std::fs::read_dir(sub.path())? {
            let entry = entry?;
            let path = entry.path();
            let fname = entry.file_name().to_string_lossy().into_owned();
            let Some(name) = fname.strip_suffix(".srj.json") else {
                continue;
            };
            // Parse to count nets; skip files that don't parse as a board.
            let net_count = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str::<SimpleRouteJson>(&s).ok())
                .map(|srj| srj.connections.len())
                .unwrap_or(0);
            out.push(BoardInfo {
                id: format!("{corpus}/{name}"),
                corpus: corpus.clone(),
                name: name.to_string(),
                net_count,
            });
        }
    }
    Ok(out)
}

/// Resolve a board id (`"<corpus>/<name>"`) to its file and read it verbatim.
/// Rejects ids containing `..` or absolute components to prevent path traversal.
fn board_path(corpus_dir: &Path, id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || id.contains("..") || id.starts_with('/') {
        return Err(format!("invalid board id: {id:?}"));
    }
    Ok(corpus_dir.join(format!("{id}.srj.json")))
}

/// `GET /api/boards/*id` — return a board's raw SimpleRouteJson (so the client
/// renders obstacles/pads/bounds from the original tscircuit fields verbatim).
async fn get_board(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    let path = match board_path(&state.corpus_dir, &id) {
        Ok(p) => p,
        Err(e) => return bad_request(e),
    };
    match std::fs::read_to_string(&path) {
        Ok(body) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response(),
        Err(_) => not_found(format!("board not found: {id}")),
    }
}

/// Request body for `POST /api/trace`. Supply either a `board_id` (loaded from the
/// corpus) or an inline `simple_route_json`. Optional `resolution`, `layers`, and
/// `clearance` override the server defaults for this one request.
#[derive(Debug, Deserialize)]
struct TraceRequest {
    #[serde(default)]
    board_id: Option<String>,
    #[serde(default)]
    simple_route_json: Option<SimpleRouteJson>,
    #[serde(default)]
    resolution: Option<f64>,
    #[serde(default)]
    layers: Option<u32>,
    #[serde(default)]
    clearance: Option<f64>,
}

/// Continuous (mm) positions of the grid lines — the client maps `CellIdx` → (x, y)
/// with these plus `trace.dims`.
#[derive(Debug, Serialize)]
struct CoordsDto {
    x_lines: Vec<f64>,
    y_lines: Vec<f64>,
}

/// Successful `POST /api/trace` response: the replayable trace plus everything the
/// client needs to render it in continuous coordinates.
#[derive(Debug, Serialize)]
struct TraceResponse {
    trace: RouteTrace,
    coords: CoordsDto,
    /// Ordered copper layer names the trace's cells are addressed against.
    layers: Vec<String>,
    /// The board's continuous bounds (for the initial viewport).
    bounds: Bounds,
    /// The final routed result in tscircuit solution-soup form.
    solution: Vec<PcbTrace>,
}

/// `POST /api/trace` — route a board and return a step-by-step [`RouteTrace`].
async fn trace(
    State(state): State<AppState>,
    body: Result<Json<TraceRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => return bad_request(format!("invalid request body: {rej}")),
    };

    // Resolve the problem: inline json wins, else load by board id.
    let srj = match (req.simple_route_json, req.board_id.as_deref()) {
        (Some(s), _) => s,
        (None, Some(id)) => {
            let path = match board_path(&state.corpus_dir, id) {
                Ok(p) => p,
                Err(e) => return bad_request(e),
            };
            match std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str::<SimpleRouteJson>(&s).ok())
            {
                Some(s) => s,
                None => return not_found(format!("board not found or invalid: {id}")),
            }
        }
        (None, None) => {
            return bad_request("trace requires `board_id` or `simple_route_json`".to_string())
        }
    };

    let solve_layers = req.layers.map(|l| l.max(1)).unwrap_or(state.solve_layers);
    // Per-request clearance override, else the server policy.
    let clearance_policy = match req.clearance {
        Some(c) => Some(c),
        None => state.clearance_mm,
    };
    let prep = prepare(&srj, req.resolution, solve_layers, clearance_policy);

    // The trace requires the concrete `NegotiatedRouter` (the generic `make_router`
    // factory only yields the `Router` trait, which has no `route_traced`). Build one
    // mirroring `main.rs`'s factory: clearance budget + the problem's Hanan coords.
    let router = configured_negotiated_router(prep.router.clone());
    let (board, trace) = match router.route_traced(&prep.problem.grid, &prep.problem.nets) {
        Ok(bt) => bt,
        Err(e) => return router_error_response(e),
    };

    let solution = solution_from_board(&prep, &board);
    let resp = TraceResponse {
        trace,
        coords: CoordsDto {
            x_lines: prep.problem.mapping.x_lines.clone(),
            y_lines: prep.problem.mapping.y_lines.clone(),
        },
        layers: layer_names(&prep.problem.layers),
        bounds: srj.bounds,
        solution,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

/// Build a `404 Not Found` JSON response.
fn not_found(msg: String) -> Response {
    (StatusCode::NOT_FOUND, Json(ErrorResponse { error: msg })).into_response()
}

/// Build a `500 Internal Server Error` JSON response.
fn internal_error(msg: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: msg }),
    )
        .into_response()
}

/// Build the axum application.
///
/// Wires the tscircuit solver (`/solve`, `/health`), the visualiser API
/// (`/api/boards`, `/api/boards/*id`, `/api/trace`) backed by `corpus_dir`, and a
/// static-file fallback serving the built SPA from `web_dir` (any non-API path).
/// `make_router` backs `/solve`; the trace route builds its own `NegotiatedRouter`.
/// Routes on `solve_layers` layers (never fewer than a problem's declared
/// `layerCount`) and applies the `clearance_mm` policy (`None` = coherent supported
/// typed rules when available, otherwise legacy auto; `Some(0.0)` = clearance off).
/// CORS is permissive so a separate Vite dev server can call the API.
pub fn app_configured(
    make_router: ConfiguredRouterFactory,
    solve_layers: u32,
    clearance_mm: Option<f64>,
    corpus_dir: PathBuf,
    web_dir: PathBuf,
) -> AxumRouter {
    let state = AppState {
        make_router,
        solve_layers: solve_layers.max(1),
        clearance_mm,
        corpus_dir,
    };
    AxumRouter::new()
        .route("/health", get(health))
        .route("/solve", post(solve))
        .route("/api/boards", get(list_boards))
        .route("/api/boards/*id", get(get_board))
        .route("/api/trace", post(trace))
        .with_state(state)
        .layer(CorsLayer::permissive())
        // Serve the built SPA for any unmatched (non-API) path. A missing `web_dir`
        // simply 404s, so the API still works before the frontend is built.
        .fallback_service(ServeDir::new(web_dir))
}

/// Build the application with the original generic-clearance router-factory API.
/// Typed inputs are still parsed/rasterized coherently, but an old backend sees
/// only the fields its public contract historically exposed. Use
/// [`app_configured`] when a backend must honor typed via geometry and pair rules.
pub fn app(
    make_router: RouterFactory,
    solve_layers: u32,
    clearance_mm: Option<f64>,
    corpus_dir: PathBuf,
    web_dir: PathBuf,
) -> AxumRouter {
    let configured: ConfiguredRouterFactory = Arc::new(move |config| {
        let RouterConfig {
            clearance_mm,
            coords,
            ..
        } = config;
        make_router(clearance_mm, coords)
    });
    app_configured(configured, solve_layers, clearance_mm, corpus_dir, web_dir)
}

/// Bind `addr` and serve until shutdown, building backends via `make_router` and
/// applying the `solve_layers` + `clearance_mm` policy, scanning `corpus_dir` for
/// boards and serving the SPA from `web_dir`.
pub async fn serve_configured(
    addr: std::net::SocketAddr,
    make_router: ConfiguredRouterFactory,
    solve_layers: u32,
    clearance_mm: Option<f64>,
    corpus_dir: PathBuf,
    web_dir: PathBuf,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app_configured(make_router, solve_layers, clearance_mm, corpus_dir, web_dir),
    )
    .await
}

/// Serve with the original generic-clearance router-factory API. See [`app`] for
/// the compatibility behavior and [`serve_configured`] for the typed product path.
pub async fn serve(
    addr: std::net::SocketAddr,
    make_router: RouterFactory,
    solve_layers: u32,
    clearance_mm: Option<f64>,
    corpus_dir: PathBuf,
    web_dir: PathBuf,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app(make_router, solve_layers, clearance_mm, corpus_dir, web_dir),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use tower::ServiceExt; // for `oneshot` (NegotiatedRouter comes from `super::*`)

    const SAMPLE: &str = r#"{
        "simple_route_json": {
            "layerCount": 2,
            "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
            "obstacles": [
                { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
            ],
            "connections": [
                { "name": "VCC", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 1} ] },
                { "name": "GND", "pointsToConnect": [ {"x": 1, "y": 9}, {"x": 9, "y": 9} ] }
            ]
        }
    }"#;

    /// Factory mirroring `main.rs`: builds a `NegotiatedRouter` at the requested
    /// clearance budget (in cells).
    fn test_factory() -> ConfiguredRouterFactory {
        Arc::new(|config| Box::new(configured_negotiated_router(config)))
    }

    /// App wired to the test router at the default solve-layer budget, clearance
    /// off (fast + deterministic for the shape assertions below). The corpus / web
    /// dirs are placeholders the `/solve` + `/health` tests never touch.
    fn test_app() -> AxumRouter {
        app_configured(
            test_factory(),
            DEFAULT_SOLVE_LAYERS,
            Some(0.0),
            PathBuf::from("benchmarks/corpus"),
            PathBuf::from("web/dist"),
        )
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn typed_profile() -> SimpleRouteJson {
        serde_json::from_value(serde_json::json!({
            "layerCount": 2,
            "minTraceWidth": 0.08,
            "nominalTraceWidth": 0.1,
            "defaultObstacleMargin": 0.04,
            "minTraceToPadEdgeClearance": 0.07,
            "minViaEdgeToPadEdgeClearance": 0.09,
            "minViaHoleEdgeToViaHoleEdgeClearance": 0.1,
            "minPadEdgeToPadEdgeClearance": 0.11,
            "minViaHoleDiameter": 0.2,
            "minViaPadDiameter": 0.4,
            "bounds": {"minX": 0.0, "maxX": 4.0, "minY": 0.0, "maxY": 4.0},
            "obstacles": [
                {"type": "rect", "shape": "rect", "center": {"x": 2.0, "y": 2.0},
                 "width": 0.2, "height": 0.2, "layers": ["top"],
                 "connectedTo": ["pad_probe"]},
                {"type": "rect", "shape": "rect", "center": {"x": 0.5, "y": 0.5},
                 "width": 0.2, "height": 0.2, "layers": ["top"],
                 "connectedTo": ["n"]},
                {"type": "rect", "shape": "rect", "center": {"x": 3.5, "y": 3.5},
                 "width": 0.2, "height": 0.2, "layers": ["top"],
                 "connectedTo": ["n"]}
            ],
            "connections": [{
                "name": "n", "nominalTraceWidth": 0.1,
                "pointsToConnect": [
                    {"x": 0.5, "y": 0.5}, {"x": 3.5, "y": 3.5}
                ]
            }]
        }))
        .unwrap()
    }

    #[test]
    fn prepare_projects_coherent_typed_router_geometry_and_preserves_override() {
        let srj = typed_profile();
        let typed = prepare(&srj, Some(0.5), 2, None);
        assert_eq!(typed.trace_width, 0.1);
        assert!((typed.router.clearance_mm - 0.14).abs() < 1e-12);
        assert!((typed.router.via_model.keepout_mm - 0.32).abs() < 1e-12);
        assert!((typed.router.via_spacing_mm - 0.44).abs() < 1e-12);
        assert!((typed.router.via_hole_spacing_mm - 0.3).abs() < 1e-12);
        assert!(typed.router.committed_via_to_trace_guard);
        assert_eq!(typed.router.via_model.layers, 2);
        assert_eq!(typed.problem.grid.dims.layers, 2);
        assert!(
            !typed.problem.grid.via_forbidden.is_empty(),
            "typed via→pad geometry must reach the server raster"
        );

        let overridden = prepare(&srj, Some(0.5), 2, Some(0.0));
        assert_eq!(overridden.router.clearance_mm, 0.0);
        assert_eq!(overridden.router.via_spacing_mm, 0.0);
        assert_eq!(overridden.router.via_hole_spacing_mm, 0.0);
        assert!(!overridden.router.committed_via_to_trace_guard);
        assert_eq!(overridden.router.via_model.keepout_mm, 0.0);
        assert!(
            overridden.problem.grid.via_forbidden.is_empty(),
            "the established explicit zero-clearance policy stays a full opt-out"
        );
    }

    #[tokio::test]
    async fn legacy_app_factory_receives_exact_clearance_and_hanan_coords() {
        let captured = Arc::new(std::sync::Mutex::new(None));
        let captured_by_factory = Arc::clone(&captured);
        let legacy_factory: RouterFactory = Arc::new(move |clearance_mm, coords| {
            *captured_by_factory.lock().unwrap() = Some((clearance_mm, coords.clone()));
            Box::new(
                NegotiatedRouter::new()
                    .with_clearance_mm(clearance_mm)
                    .with_coords(coords),
            )
        });
        let app = app(
            legacy_factory,
            DEFAULT_SOLVE_LAYERS,
            Some(0.37),
            PathBuf::from("benchmarks/corpus"),
            PathBuf::from("web/dist"),
        );
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/solve")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(SAMPLE))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let envelope: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        let srj: SimpleRouteJson =
            serde_json::from_value(envelope["simple_route_json"].clone()).unwrap();
        let expected = prepare(&srj, None, DEFAULT_SOLVE_LAYERS, Some(0.37));
        let (clearance_mm, coords) = captured.lock().unwrap().clone().unwrap();
        assert!((clearance_mm - 0.37).abs() < 1e-12);
        assert_eq!(coords, expected.router.coords);
    }

    #[tokio::test]
    async fn solve_returns_pcb_traces() {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/solve")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(SAMPLE))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        let soup = json
            .get("solution_soup")
            .and_then(|v| v.as_array())
            .expect("solution_soup must be an array");
        assert!(!soup.is_empty(), "expected at least one routed trace");
        for trace in soup {
            assert_eq!(
                trace.get("type").and_then(|v| v.as_str()),
                Some("pcb_trace"),
                "every soup element must be a pcb_trace object: {trace}"
            );
            assert!(trace.get("route").is_some(), "trace must carry a route");
        }
    }

    #[tokio::test]
    #[ignore = "frontier benchmark; run in release"]
    async fn solve_routes_srj29_sample021_through_typed_product_path() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/frontier/srj29/sample021-am62l-lpddr4.srj.json");
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture).unwrap()).unwrap();
        let srj: SimpleRouteJson = serde_json::from_value(raw.clone()).unwrap();
        let body = serde_json::to_vec(&serde_json::json!({
            "simple_route_json": raw
        }))
        .unwrap();
        let app = app_configured(
            test_factory(),
            DEFAULT_SOLVE_LAYERS,
            None,
            PathBuf::from("benchmarks/corpus"),
            PathBuf::from("web/dist"),
        );
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/solve")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let soup = json["solution_soup"].as_array().unwrap();
        assert_eq!(soup.len(), 28, "typed /solve completion regressed");
        assert!(soup
            .iter()
            .flat_map(|trace| trace["route"].as_array().unwrap())
            .filter(|point| point["route_type"] == "wire")
            .all(|point| { (point["width"].as_f64().unwrap() - 0.08128).abs() < 1e-12 }));
        let solution: Vec<PcbTrace> =
            serde_json::from_value(json["solution_soup"].clone()).unwrap();
        let violations = mr_cli::check_srj_solution(&srj, &solution, srj.layer_count);
        assert!(
            violations.is_empty(),
            "typed /solve soup must pass exact supported-rule DRC: {violations:#?}"
        );
    }

    #[tokio::test]
    async fn health_returns_200() {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/health")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn malformed_json_returns_400() {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/solve")
            .header("content-type", "application/json")
            .body(axum::body::Body::from("{ this is not valid json "))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // Must be a JSON error body, not a panic / empty 500.
        let json = body_json(resp).await;
        assert!(json.get("error").is_some(), "error body expected: {json}");
    }

    #[tokio::test]
    async fn valid_json_wrong_shape_returns_400() {
        // Syntactically valid JSON but missing simple_route_json.
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/solve")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{ "foo": 1 }"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn choose_resolution_honors_override() {
        let srj: SimpleRouteJson = serde_json::from_str(
            r#"{ "layerCount": 1, "bounds": { "minX": 0, "maxX": 1000, "minY": 0, "maxY": 1000 } }"#,
        )
        .unwrap();
        assert_eq!(choose_resolution(&srj, Some(2.5)), 2.5);
        // Non-positive / non-finite overrides are ignored.
        assert_ne!(choose_resolution(&srj, Some(0.0)), 0.0);
        assert!(choose_resolution(&srj, Some(f64::NAN)) > 0.0);
    }

    #[test]
    fn choose_resolution_derives_from_span() {
        let srj: SimpleRouteJson = serde_json::from_str(
            r#"{ "layerCount": 1, "bounds": { "minX": 0, "maxX": 1000, "minY": 0, "maxY": 400 } }"#,
        )
        .unwrap();
        // max span 1000 / 200 = 5.0.
        assert_eq!(choose_resolution(&srj, None), 5.0);
    }

    #[test]
    fn choose_resolution_floors_small_and_degenerate() {
        let small: SimpleRouteJson = serde_json::from_str(
            r#"{ "layerCount": 1, "bounds": { "minX": 0, "maxX": 1, "minY": 0, "maxY": 1 } }"#,
        )
        .unwrap();
        // 1/200 = 0.005 < MIN_RESOLUTION, so floored.
        assert_eq!(choose_resolution(&small, None), MIN_RESOLUTION);

        let degenerate: SimpleRouteJson = serde_json::from_str(
            r#"{ "layerCount": 1, "bounds": { "minX": 3, "maxX": 3, "minY": 3, "maxY": 3 } }"#,
        )
        .unwrap();
        assert_eq!(choose_resolution(&degenerate, None), MIN_RESOLUTION);
    }
}
