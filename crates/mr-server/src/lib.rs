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

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use mr_core::{Router, RouterError};
use mr_srj::{rasterize, to_solution, PcbTrace, SimpleRouteJson};
use serde::{Deserialize, Serialize};

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

/// The injected routing backend, shared across requests.
type SharedRouter = Arc<dyn Router + Send + Sync>;

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

/// `POST /solve` handler.
async fn solve(
    State(router): State<SharedRouter>,
    body: Result<Json<SolveRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => return bad_request(format!("invalid request body: {rej}")),
    };

    let resolution = choose_resolution(&req.simple_route_json, req.resolution);
    let problem = rasterize(&req.simple_route_json, resolution);

    let board = match router.route(&problem.grid, &problem.nets) {
        Ok(b) => b,
        Err(e) => return router_error_response(e),
    };

    let trace_width = req
        .simple_route_json
        .min_trace_width
        .unwrap_or(DEFAULT_TRACE_WIDTH);
    let solution_soup = to_solution(
        &board,
        &problem.mapping,
        &problem.pin_points,
        trace_width,
        DEFAULT_LAYER,
    );
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

/// Build the axum application wired to `/solve` and `/health`, backed by the
/// injected `router`.
pub fn app(router: SharedRouter) -> AxumRouter {
    AxumRouter::new()
        .route("/health", get(health))
        .route("/solve", post(solve))
        .with_state(router)
}

/// Bind `addr` and serve the solver until shutdown, using `router` as the
/// backend.
pub async fn serve(addr: std::net::SocketAddr, router: SharedRouter) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(router)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use mr_cpu::NegotiatedRouter;
    use tower::ServiceExt; // for `oneshot`

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

    fn test_router() -> SharedRouter {
        Arc::new(NegotiatedRouter::new())
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn solve_returns_pcb_traces() {
        let app = app(test_router());
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
    async fn health_returns_200() {
        let app = app(test_router());
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
        let app = app(test_router());
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
        let app = app(test_router());
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
