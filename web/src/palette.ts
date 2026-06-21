// Per-group colours. A net's colour is chosen by its connection-group id so that
// `#`-sibling sub-nets of one electrical net share a colour and read as one trace.

const HUES = [
  "#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8",
  "#4db6ac", "#fff176", "#f06292", "#9575cd", "#a1887f",
  "#90a4ae", "#7986cb", "#aed581", "#ff8a65", "#4dd0e1",
  "#dce775", "#f48fb1", "#ce93d8", "#80cbc4", "#ffcc80",
];

export function groupColor(group: number): string {
  return HUES[group % HUES.length];
}

/** Hot colour for over-used (contested) cells. */
export const OVERUSE_COLOR = "#ff1744";

/** Ratsnest (unrouted connection) colour. */
export const RATSNEST_COLOR = "#ff5252";

/** Pad / obstacle fill. */
export const PAD_COLOR = "rgba(120, 144, 156, 0.55)";
export const KEEPOUT_COLOR = "rgba(70, 80, 90, 0.45)";

/** Board background + frame. */
export const BG_COLOR = "#0e1116";
export const BOARD_COLOR = "#11151b";
export const BOARD_BORDER = "#2a3340";
