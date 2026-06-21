import type { CellIdx, Coords, Dims } from "./types";

/** A cell decoded to its continuous (mm) position and copper layer. */
export interface CellPos {
  x: number;
  y: number;
  layer: number;
}

/**
 * Maps a router `CellIdx` back to continuous coordinates, using the canonical
 * row-major mapping `idx3(x, y, l) = (l*h + y)*w + x` (mr-core `Dims`) and the
 * board's Hanan line arrays (mr-core `GridCoords`).
 */
export class CellMapper {
  private w: number;
  private plane: number;
  private xLines: number[];
  private yLines: number[];

  constructor(dims: Dims, coords: Coords) {
    this.w = dims.w;
    this.plane = dims.w * dims.h;
    this.xLines = coords.x_lines;
    this.yLines = coords.y_lines;
  }

  pos(cell: CellIdx): CellPos {
    const layer = Math.floor(cell / this.plane);
    const r = cell % this.plane;
    const cx = r % this.w;
    const cy = Math.floor(r / this.w);
    return {
      x: this.xLines[cx] ?? cx,
      y: this.yLines[cy] ?? cy,
      layer,
    };
  }

  layerOf(cell: CellIdx): number {
    return Math.floor(cell / this.plane);
  }
}
