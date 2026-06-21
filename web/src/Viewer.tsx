import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Frame } from "./frames";
import { CellMapper } from "./mapper";
import { type Camera, drawFrame, fitCamera, worldAt } from "./render";
import type { Bounds, Obstacle, TracedNet } from "./types";

interface ViewerProps {
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
  /** Bump this to refit the camera (e.g. when a new board loads). */
  fitKey: string;
}

export function Viewer(props: ViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [cam, setCam] = useState<Camera | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  // Track the wrapper size.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // Refit when a new board loads or the viewport first gets a size.
  useEffect(() => {
    if (size.w > 0 && size.h > 0) {
      setCam(fitCamera(props.bounds, size.w, size.h));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.fitKey, size.w, size.h]);

  // Draw whenever inputs change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !cam || size.w === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawFrame(ctx, size.w, size.h, {
      cam,
      mapper: props.mapper,
      frame: props.frame,
      nets: props.nets,
      obstacles: props.obstacles,
      bounds: props.bounds,
      layerNames: props.layerNames,
      visibleLayers: props.visibleLayers,
      highlight: props.highlight,
      showRatsnest: props.showRatsnest,
      showOveruse: props.showOveruse,
    });
  }, [cam, size, props]);

  function onWheel(e: React.WheelEvent) {
    if (!cam) return;
    e.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const before = worldAt(cam, px, py);
    const scale = cam.scale * factor;
    // Keep the world point under the cursor fixed.
    const offsetX = px - before.x * scale;
    const offsetY = py + before.y * scale;
    setCam({ scale, offsetX, offsetY });
  }

  function onPointerDown(e: React.PointerEvent) {
    drag.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }
  function onPointerMove(e: React.PointerEvent) {
    if (!drag.current || !cam) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setCam({ ...cam, offsetX: cam.offsetX + dx, offsetY: cam.offsetY + dy });
  }
  function onPointerUp(e: React.PointerEvent) {
    drag.current = null;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  }

  function resetView() {
    if (size.w > 0) setCam(fitCamera(props.bounds, size.w, size.h));
  }

  return (
    <div className="viewer" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      <button className="reset-view" onClick={resetView} title="Reset view">
        ⟲ fit
      </button>
    </div>
  );
}
