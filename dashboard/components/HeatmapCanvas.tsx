"use client";

import { useEffect, useMemo, useRef } from "react";
import { API, Heatmap } from "@/lib/api";

// Draws the person-seconds grid over the people-free background image.
// Square-root colour scale (as in retailflow/heatmap.py) so quieter areas stay visible.
export default function HeatmapCanvas({ sessionId, heat }: { sessionId: string; heat: Heatmap }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rows = heat.grid.length, cols = heat.grid[0]?.length ?? 0;
  const [w, h] = heat.floor ? [cols * 20, rows * 20] : heat.frame_size;
  const raw = Math.max(0, ...heat.grid.flat());
  // Same idea as Heatmap.smoothed() in Python: blur ~2 cells so single feet points
  // become areas. Three box blurs approximate a Gaussian.
  const smooth = useMemo(() => blur(heat.grid, 2, 3), [heat.grid]);
  const max = Math.max(0, ...smooth.flat());

  useEffect(() => {
    const c = ref.current;
    if (!c || !w || !h) return;
    const ctx = c.getContext("2d")!;
    const draw = (bg?: HTMLImageElement) => {
      ctx.clearRect(0, 0, w, h);
      if (bg) ctx.drawImage(bg, 0, 0, w, h);
      // Paint cells on a small canvas, then scale up smoothly (a cheap blur).
      const small = document.createElement("canvas");
      small.width = cols; small.height = rows;
      const sctx = small.getContext("2d")!;
      const img = sctx.createImageData(cols, rows);
      for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) {
        const v = max > 0 ? Math.sqrt(smooth[r][q] / max) : 0;
        const [R, G, B] = ramp(v);
        const i = (r * cols + q) * 4;
        img.data[i] = R; img.data[i + 1] = G; img.data[i + 2] = B;
        img.data[i + 3] = v < 0.03 ? 0 : Math.round(90 + 140 * v);
      }
      sctx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(small, 0, 0, w, h);
    };
    if (heat.has_background && !heat.floor) {
      const bg = new Image();
      bg.crossOrigin = "anonymous";
      bg.onload = () => draw(bg);
      bg.onerror = () => draw();
      bg.src = `${API}/api/sessions/${sessionId}/background.png`;
    } else draw();
  }, [heat, sessionId, w, h, rows, cols, max, smooth]);

  return (
    <div>
      <canvas ref={ref} width={w} height={h} style={{ width: "100%", height: "auto", borderRadius: 6 }}
              role="img" aria-label="Heatmap of where people spent time" />
      <div className="row muted" style={{ marginTop: 6 }}>
        <span>less</span>
        <span style={{ width: 160, height: 10, borderRadius: 3,
          background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((v) => `rgb(${ramp(v).join(",")})`).join(",")})` }} />
        <span>more time · total {heat.grid.flat().reduce((a, b) => a + b, 0).toFixed(0)} person-seconds,
          busiest cell {raw.toFixed(1)} s</span>
      </div>
    </div>
  );
}

// Repeated box blur (radius r, `passes` times) over a 2-D grid.
function blur(grid: number[][], r: number, passes: number): number[][] {
  let g = grid.map((row) => row.slice());
  const R = g.length, C = g[0]?.length ?? 0;
  for (let p = 0; p < passes; p++) {
    const hz = g.map((row) => row.map((_, q) => {
      let s = 0, n = 0;
      for (let k = Math.max(0, q - r); k <= Math.min(C - 1, q + r); k++) { s += row[k]; n++; }
      return s / n;
    }));
    g = hz.map((row, i) => row.map((_, q) => {
      let s = 0, n = 0;
      for (let k = Math.max(0, i - r); k <= Math.min(R - 1, i + r); k++) { s += hz[k][q]; n++; }
      return s / n;
    }));
  }
  return g;
}

// Dark blue -> teal -> yellow -> red.
function ramp(v: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0, [30, 60, 160]], [0.35, [20, 170, 170]], [0.65, [240, 210, 40]], [1, [220, 40, 30]],
  ];
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i - 1], [p2, c2] = stops[i];
    if (v <= p2) {
      const f = (v - p1) / (p2 - p1);
      return [0, 1, 2].map((k) => Math.round(c1[k] + f * (c2[k] - c1[k]))) as [number, number, number];
    }
  }
  return stops[stops.length - 1][1];
}
