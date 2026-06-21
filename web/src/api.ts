import type { BoardInfo, SimpleRouteJson, TraceResponse } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${url} → ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** List every board in the server's corpus. */
export function fetchBoards(): Promise<BoardInfo[]> {
  return getJson<BoardInfo[]>("/api/boards");
}

/** Fetch a board's raw SimpleRouteJson (obstacles/pads/bounds). */
export function fetchBoard(id: string): Promise<SimpleRouteJson> {
  return getJson<SimpleRouteJson>(`/api/boards/${id}`);
}

/** Options that override the server's defaults for a single trace run. */
export interface TraceOptions {
  layers?: number;
  clearance?: number;
  resolution?: number;
}

/** Route a board (by id) and return its step-by-step trace. */
export async function fetchTrace(
  boardId: string,
  opts: TraceOptions = {},
): Promise<TraceResponse> {
  const res = await fetch("/api/trace", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ board_id: boardId, ...opts }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`/api/trace → ${res.status}: ${body}`);
  }
  return res.json() as Promise<TraceResponse>;
}
