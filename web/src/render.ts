import type { Frame } from "./frames";
import { CellMapper } from "./mapper";
import {
  BOARD_BORDER,
  BOARD_COLOR,
  BG_COLOR,
  groupColor,
  KEEPOUT_COLOR,
  OVERUSE_COLOR,
  PAD_COLOR,
  RATSNEST_COLOR,
} from "./palette";
import type { Bounds, Obstacle, TracedNet } from "./types";

/** World→screen transform. `screenX = wx*scale + offsetX`, `screenY = -wy*scale +
 * offsetY` (the negation puts board +y upward, as a PCB is drawn). */
export interface Camera {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export function sx(cam: Camera, wx: number): number {
  return wx * cam.scale + cam.offsetX;
}
export function sy(cam: Camera, wy: number): number {
  return -wy * cam.scale + cam.offsetY;
}
/** Inverse: screen px → world coords (for zoom-to-cursor). */
export function worldAt(cam: Camera, px: number, py: number): { x: number; y: number } {
  return { x: (px - cam.offsetX) / cam.scale, y: (cam.offsetY - py) / cam.scale };
}

/** Fit `bounds` into a `w×h` viewport (CSS px) with a margin. */
export function fitCamera(bounds: Bounds, w: number, h: number, margin = 28): Camera {
  const bw = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const bh = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const scale = Math.min((w - 2 * margin) / bw, (h - 2 * margin) / bh);
  // Center the board in the viewport.
  const usedW = bw * scale;
  const usedH = bh * scale;
  const offsetX = (w - usedW) / 2 - bounds.minX * scale;
  const offsetY = (h - usedH) / 2 + bounds.maxY * scale;
  return { scale, offsetX, offsetY };
}

export interface DrawParams {
  cam: Camera;
  mapper: CellMapper;
  frame: Frame;
  nets: TracedNet[];
  obstacles: Obstacle[];
  bounds: Bounds;
  layerNames: string[];
  visibleLayers: boolean[];
  highlight: number | null;
  showRatsnest: boolean;
  showOveruse: boolean;
}

/** Draw one frame onto a 2D context already sized to CSS px (dpr handled by caller). */
export function drawFrame(
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  p: DrawParams,
): void {
  const { cam, mapper, frame, nets } = p;

  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, cssW, cssH);

  // Board outline.
  const bx = sx(cam, p.bounds.minX);
  const by = sy(cam, p.bounds.maxY);
  const bw = (p.bounds.maxX - p.bounds.minX) * cam.scale;
  const bh = (p.bounds.maxY - p.bounds.minY) * cam.scale;
  ctx.fillStyle = BOARD_COLOR;
  ctx.fillRect(bx, by, bw, bh);
  ctx.strokeStyle = BOARD_BORDER;
  ctx.lineWidth = 1;
  ctx.strokeRect(bx, by, bw, bh);

  drawObstacles(ctx, p);

  // Endpoints (faint), under traces.
  ctx.fillStyle = "rgba(200, 210, 220, 0.35)";
  for (const net of nets) {
    for (const cell of [net.src, net.dst]) {
      const pos = mapper.pos(cell);
      if (!layerVisible(p, pos.layer)) continue;
      ctx.beginPath();
      ctx.arc(sx(cam, pos.x), sy(cam, pos.y), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Traces / ratsnest, per net.
  for (let i = 0; i < nets.length; i++) {
    const path = frame.paths[i];
    const dim = p.highlight !== null && p.highlight !== i;
    const color = groupColor(nets[i].group);
    if (path === null) {
      if (p.showRatsnest && !dim) drawRatsnest(ctx, p, nets[i]);
      continue;
    }
    drawTrace(ctx, p, path, color, dim, p.highlight === i);
  }

  if (p.showOveruse) drawOveruse(ctx, p);
}

function layerVisible(p: DrawParams, layer: number): boolean {
  return p.visibleLayers[layer] ?? true;
}

function drawObstacles(ctx: CanvasRenderingContext2D, p: DrawParams): void {
  const { cam } = p;
  for (const ob of p.obstacles) {
    // Skip obstacles confined to hidden layers (empty = all layers).
    if (ob.layers && ob.layers.length > 0) {
      const anyVisible = ob.layers.some((name) => {
        const idx = p.layerNames.indexOf(name);
        return idx < 0 || layerVisible(p, idx);
      });
      if (!anyVisible) continue;
    }
    const isPad = (ob.connectedTo?.length ?? 0) > 0;
    const w = ob.width * cam.scale;
    const h = ob.height * cam.scale;
    const x = sx(cam, ob.center.x) - w / 2;
    const y = sy(cam, ob.center.y) - h / 2;
    ctx.fillStyle = isPad ? PAD_COLOR : KEEPOUT_COLOR;
    ctx.fillRect(x, y, w, h);
  }
}

function drawTrace(
  ctx: CanvasRenderingContext2D,
  p: DrawParams,
  path: number[],
  color: string,
  dim: boolean,
  emphasized: boolean,
): void {
  const { cam, mapper } = p;
  ctx.globalAlpha = dim ? 0.12 : 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = emphasized ? 3.2 : 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Draw each same-layer segment; mark a via where the layer changes.
  let prev = mapper.pos(path[0]);
  for (let k = 1; k < path.length; k++) {
    const cur = mapper.pos(path[k]);
    if (cur.layer === prev.layer) {
      if (layerVisible(p, cur.layer)) {
        ctx.beginPath();
        ctx.moveTo(sx(cam, prev.x), sy(cam, prev.y));
        ctx.lineTo(sx(cam, cur.x), sy(cam, cur.y));
        ctx.stroke();
      }
    } else {
      // Via: drawn if either adjacent layer is visible.
      if (layerVisible(p, cur.layer) || layerVisible(p, prev.layer)) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx(cam, cur.x), sy(cam, cur.y), emphasized ? 3.5 : 2.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0e1116";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = emphasized ? 3.2 : 2;
      }
    }
    prev = cur;
  }
  ctx.globalAlpha = 1;
}

function drawRatsnest(ctx: CanvasRenderingContext2D, p: DrawParams, net: TracedNet): void {
  const { cam, mapper } = p;
  const a = mapper.pos(net.src);
  const b = mapper.pos(net.dst);
  ctx.save();
  ctx.strokeStyle = RATSNEST_COLOR;
  ctx.globalAlpha = 0.7;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(cam, a.x), sy(cam, a.y));
  ctx.lineTo(sx(cam, b.x), sy(cam, b.y));
  ctx.stroke();
  ctx.restore();
}

function drawOveruse(ctx: CanvasRenderingContext2D, p: DrawParams): void {
  const { cam, mapper, frame } = p;
  ctx.fillStyle = OVERUSE_COLOR;
  ctx.globalAlpha = 0.85;
  for (const cell of frame.overused) {
    const pos = mapper.pos(cell);
    if (!layerVisible(p, pos.layer)) continue;
    ctx.beginPath();
    ctx.arc(sx(cam, pos.x), sy(cam, pos.y), 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
