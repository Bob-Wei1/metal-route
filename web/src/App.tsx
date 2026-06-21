import { useEffect, useMemo, useState } from "react";
import { fetchBoard, fetchBoards, fetchTrace, type TraceOptions } from "./api";
import { buildFrames } from "./frames";
import { LegalizationPanel } from "./LegalizationPanel";
import { CellMapper } from "./mapper";
import { groupColor } from "./palette";
import { Transport } from "./Transport";
import type { BoardInfo, SimpleRouteJson, TraceResponse } from "./types";
import { Viewer } from "./Viewer";

export default function App() {
  const [boards, setBoards] = useState<BoardInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [board, setBoard] = useState<SimpleRouteJson | null>(null);
  const [resp, setResp] = useState<TraceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(6);

  const [visibleLayers, setVisibleLayers] = useState<boolean[]>([]);
  const [highlight, setHighlight] = useState<number | null>(null);
  const [showRatsnest, setShowRatsnest] = useState(true);
  const [showOveruse, setShowOveruse] = useState(true);

  // Per-run overrides (empty = server default).
  const [layersOpt, setLayersOpt] = useState("");
  const [clearanceOpt, setClearanceOpt] = useState("");
  // Bumped to force a re-route with the current overrides.
  const [rerouteKey, setRerouteKey] = useState(0);

  // Load the board list once.
  useEffect(() => {
    fetchBoards()
      .then((b) => {
        setBoards(b);
        if (b.length > 0) setSelectedId(b[0].id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Load board geometry + trace whenever the selection changes.
  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    const opts: TraceOptions = {};
    if (layersOpt) opts.layers = Number(layersOpt);
    if (clearanceOpt) opts.clearance = Number(clearanceOpt);
    setLoading(true);
    setError(null);
    setPlaying(false);
    Promise.all([fetchBoard(selectedId), fetchTrace(selectedId, opts)])
      .then(([b, r]) => {
        if (cancelled) return;
        setBoard(b);
        setResp(r);
        setIndex(0);
        setHighlight(null);
        setVisibleLayers(r.layers.map(() => true));
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // Re-fetch only on board change or an explicit re-route (key bump below).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, rerouteKey]);

  const mapper = useMemo(
    () => (resp ? new CellMapper(resp.trace.dims, resp.coords) : null),
    [resp],
  );
  const frames = useMemo(() => (resp ? buildFrames(resp.trace) : []), [resp]);
  const frame = frames[Math.min(index, frames.length - 1)];

  const nets = resp?.trace.nets ?? [];
  const routedNow = frame ? frame.routedCount : 0;

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>metalroute</h1>
        <p className="subtitle">router process visualiser</p>

        <label className="field">
          board
          <select
            value={selectedId ?? ""}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {boards.map((b) => (
              <option key={b.id} value={b.id}>
                {b.corpus}/{b.name} · {b.net_count} nets
              </option>
            ))}
          </select>
        </label>

        <div className="reroute">
          <label className="field small">
            layers
            <input
              type="number"
              min={1}
              max={8}
              placeholder="auto"
              value={layersOpt}
              onChange={(e) => setLayersOpt(e.target.value)}
            />
          </label>
          <label className="field small">
            clearance
            <input
              type="number"
              step="0.05"
              min={0}
              placeholder="auto"
              value={clearanceOpt}
              onChange={(e) => setClearanceOpt(e.target.value)}
            />
          </label>
          <button onClick={() => bumpReroute()} disabled={loading}>
            re-route
          </button>
        </div>

        {error && <div className="error">{error}</div>}
        {loading && <div className="muted">routing…</div>}

        {resp && frame && (
          <div className="panel stats">
            <div className="stat">
              <span className="k">routed</span>
              <span className="v">
                {routedNow} / {nets.length}
              </span>
            </div>
            <div className="stat">
              <span className="k">overused</span>
              <span className="v">{frame.overused.length}</span>
            </div>
            <div className="stat">
              <span className="k">iteration</span>
              <span className="v">{frame.iter ?? "—"}</span>
            </div>
            <div className="stat">
              <span className="k">pfac</span>
              <span className="v">{frame.pfac ?? "—"}</span>
            </div>
            <div className="stat">
              <span className="k">grid</span>
              <span className="v">
                {resp.trace.dims.w}×{resp.trace.dims.h}×{resp.trace.dims.layers}
              </span>
            </div>
          </div>
        )}

        {resp && (
          <div className="panel options">
            <h3>display</h3>
            <label className="check">
              <input
                type="checkbox"
                checked={showOveruse}
                onChange={(e) => setShowOveruse(e.target.checked)}
              />
              over-used cells
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={showRatsnest}
                onChange={(e) => setShowRatsnest(e.target.checked)}
              />
              ratsnest (unrouted)
            </label>
            <div className="layers">
              {resp.layers.map((name, i) => (
                <label key={name} className="check">
                  <input
                    type="checkbox"
                    checked={visibleLayers[i] ?? true}
                    onChange={(e) => {
                      const next = [...visibleLayers];
                      next[i] = e.target.checked;
                      setVisibleLayers(next);
                    }}
                  />
                  {name}
                </label>
              ))}
            </div>
          </div>
        )}

        {resp && (
          <div className="panel netlist">
            <h3>nets ({nets.length})</h3>
            <div className="nets">
              {nets.map((n, i) => {
                const routed = frame?.paths[i] != null;
                const active = highlight === i;
                return (
                  <button
                    key={`${n.net}-${i}`}
                    className={`net ${active ? "active" : ""} ${routed ? "" : "unrouted"}`}
                    onClick={() => setHighlight(active ? null : i)}
                    title={n.net}
                  >
                    <span className="swatch" style={{ background: groupColor(n.group) }} />
                    <span className="name">{n.net}</span>
                    {!routed && <span className="tag">·</span>}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {frame?.kind === "final" && resp?.trace.legalization && (
          <LegalizationPanel leg={resp.trace.legalization} />
        )}
      </aside>

      <main className="main">
        {resp && mapper && board && frame ? (
          <>
            <Viewer
              mapper={mapper}
              frame={frame}
              nets={nets}
              obstacles={board.obstacles}
              bounds={resp.bounds}
              layerNames={resp.layers}
              visibleLayers={visibleLayers}
              highlight={highlight}
              showRatsnest={showRatsnest}
              showOveruse={showOveruse}
              fitKey={selectedId ?? ""}
            />
            <Transport
              frames={frames}
              index={index}
              setIndex={setIndex}
              playing={playing}
              setPlaying={setPlaying}
              fps={fps}
              setFps={setFps}
            />
          </>
        ) : (
          <div className="empty">{loading ? "routing…" : "select a board"}</div>
        )}
      </main>
    </div>
  );

  // --- re-route trigger ---
  function bumpReroute() {
    setRerouteKey((k) => k + 1);
  }
}
