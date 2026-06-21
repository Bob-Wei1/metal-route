import type { CellIdx, RouteTrace } from "./types";

/** One renderable frame of the timeline: a negotiation iteration or the final
 * legalized result. `paths[i]` is net `i`'s route this frame (`null` = unrouted). */
export interface Frame {
  kind: "iter" | "final";
  label: string;
  iter: number | null;
  pfac: number | null;
  paths: (CellIdx[] | null)[];
  overused: CellIdx[];
  routedCount: number;
  anyOveruse: boolean;
}

const orEmptyNull = (p: CellIdx[]): CellIdx[] | null => (p.length > 0 ? p : null);

/**
 * Flatten a trace into the ordered list of timeline frames: one per negotiation
 * iteration, then a final "Legalized" frame from the committed routes.
 */
export function buildFrames(trace: RouteTrace): Frame[] {
  const frames: Frame[] = trace.iterations.map((snap) => {
    const paths = snap.paths.map(orEmptyNull);
    return {
      kind: "iter" as const,
      label: `Iteration ${snap.iter} · pfac ${snap.pfac}`,
      iter: snap.iter,
      pfac: snap.pfac,
      paths,
      overused: snap.overused_cells,
      routedCount: paths.filter((p) => p !== null).length,
      anyOveruse: snap.any_overuse,
    };
  });

  const leg = trace.legalization;
  if (leg) {
    const paths = leg.committed.map((c) => (c && c.length > 0 ? c : null));
    frames.push({
      kind: "final",
      label: "Legalized (final)",
      iter: null,
      pfac: null,
      paths,
      overused: [],
      routedCount: paths.filter((p) => p !== null).length,
      anyOveruse: false,
    });
  }
  return frames;
}
