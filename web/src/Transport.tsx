import { useEffect } from "react";
import type { Frame } from "./frames";

interface TransportProps {
  frames: Frame[];
  index: number;
  setIndex: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  fps: number;
  setFps: (f: number) => void;
}

export function Transport(props: TransportProps) {
  const { frames, index, setIndex, playing, setPlaying, fps, setFps } = props;
  const last = frames.length - 1;

  // Advance frames while playing; stop at the end.
  useEffect(() => {
    if (!playing) return;
    if (index >= last) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => setIndex(Math.min(index + 1, last)), 1000 / fps);
    return () => clearTimeout(id);
  }, [playing, index, last, fps, setIndex, setPlaying]);

  const frame = frames[index];
  const step = (d: number) => setIndex(Math.max(0, Math.min(last, index + d)));

  return (
    <div className="transport">
      <div className="transport-buttons">
        <button onClick={() => setIndex(0)} title="First">⏮</button>
        <button onClick={() => step(-1)} title="Step back">◀</button>
        {playing ? (
          <button className="play" onClick={() => setPlaying(false)} title="Pause">⏸ pause</button>
        ) : (
          <button
            className="play"
            onClick={() => {
              if (index >= last) setIndex(0);
              setPlaying(true);
            }}
            title="Play"
          >
            ▶ play
          </button>
        )}
        <button onClick={() => step(1)} title="Step forward">▶</button>
        <button onClick={() => setIndex(last)} title="Last">⏭</button>
      </div>

      <input
        className="scrubber"
        type="range"
        min={0}
        max={last}
        value={index}
        onChange={(e) => {
          setPlaying(false);
          setIndex(Number(e.target.value));
        }}
      />

      <div className="transport-meta">
        <span className={`frame-label ${frame.kind}`}>{frame.label}</span>
        <span className="frame-count">
          {index + 1} / {frames.length}
        </span>
      </div>

      <label className="speed">
        speed
        <input
          type="range"
          min={1}
          max={30}
          value={fps}
          onChange={(e) => setFps(Number(e.target.value))}
        />
        <span>{fps} fps</span>
      </label>
    </div>
  );
}
