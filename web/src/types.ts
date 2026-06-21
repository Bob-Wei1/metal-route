// TypeScript mirrors of the Rust contract types, so trace data is typed
// end-to-end. Field names match the JSON the server emits (mr-core's RouteTrace
// uses Rust field names = snake_case; mr-srj's Bounds/obstacles use camelCase).

export type CellIdx = number;

export interface Dims {
  w: number;
  h: number;
  layers: number;
}

/** One board in the `/api/boards` listing. */
export interface BoardInfo {
  id: string;
  corpus: string;
  name: string;
  net_count: number;
}

/** Static per-net info (mr-core `TracedNet`). */
export interface TracedNet {
  net: string;
  src: CellIdx;
  dst: CellIdx;
  group: number;
  alone_path: CellIdx[];
}

/** One negotiation-iteration frame (mr-core `IterSnapshot`). */
export interface IterSnapshot {
  iter: number;
  pfac: number;
  paths: CellIdx[][];
  overused_cells: CellIdx[];
  any_overuse: boolean;
}

/** One candidate group-order's score (mr-core `CandidateEval`). */
export interface CandidateEval {
  order: number[];
  routed: number;
  total_cost: number;
}

/** Legalization phase result (mr-core `LegalizationTrace`). */
export interface LegalizationTrace {
  chosen_order: number[];
  candidates: CandidateEval[];
  committed: (CellIdx[] | null)[];
}

/** Replayable trace of a route (mr-core `RouteTrace`). */
export interface RouteTrace {
  dims: Dims;
  nets: TracedNet[];
  n_groups: number;
  iterations: IterSnapshot[];
  legalization: LegalizationTrace | null;
}

/** Continuous (mm) positions of the grid lines (server `CoordsDto`). */
export interface Coords {
  x_lines: number[];
  y_lines: number[];
}

/** Board bounds (mr-srj `Bounds`, camelCase). */
export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/** `POST /api/trace` response. */
export interface TraceResponse {
  trace: RouteTrace;
  coords: Coords;
  layers: string[];
  bounds: Bounds;
  solution: unknown[];
}

// --- Raw SimpleRouteJson (from `/api/boards/{id}`) — the bits we render. ---

export interface Point {
  x: number;
  y: number;
  layer?: string | null;
}

export interface Obstacle {
  type: string;
  center: Point;
  width: number;
  height: number;
  layers?: string[];
  connectedTo?: string[];
}

export interface Connection {
  name: string;
  rootConnectionName?: string | null;
  pointsToConnect: Point[];
}

export interface SimpleRouteJson {
  layerCount: number;
  minTraceWidth?: number | null;
  minClearance?: number | null;
  obstacles: Obstacle[];
  connections: Connection[];
  bounds: Bounds;
}
