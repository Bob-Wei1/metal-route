import type { LegalizationTrace } from "./types";

/** Shows the candidate group-orders the legalizer evaluated and which it kept. */
export function LegalizationPanel({ leg }: { leg: LegalizationTrace }) {
  const chosen = leg.chosen_order.join(",");
  // Sort candidates best-first (most routed, then lowest cost) for readability.
  const rows = leg.candidates
    .map((c, i) => ({ ...c, i }))
    .sort((a, b) => b.routed - a.routed || a.total_cost - b.total_cost);

  return (
    <div className="panel legalization">
      <h3>Legalization</h3>
      <p className="muted">
        The router commits whole groups in a chosen order; foreign groups become
        obstacles for later ones. It evaluates several orders and keeps the best.
      </p>
      <div className="leg-table">
        <div className="leg-head">
          <span>order</span>
          <span>routed</span>
          <span>cost</span>
        </div>
        {rows.map((c) => {
          const isChosen = c.order.join(",") === chosen;
          return (
            <div key={c.i} className={`leg-row ${isChosen ? "chosen" : ""}`}>
              <span className="order">[{c.order.join(" ")}]</span>
              <span>{c.routed}</span>
              <span>{c.total_cost}</span>
              {isChosen && <span className="badge">kept</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
