"use client";

import { useState } from "react";
import { fmtTime } from "@/lib/api";

export type Series = { key: string; label: string; color: string };

type Props = {
  rows: ({ t: number } & Record<string, number>)[];
  series: Series[];
  height?: number;
  yLabel: string;
  step?: boolean; // draw counts as steps (bucketed), occupancy as lines
};

// Dependency-free SVG line chart with a hover readout. x = video time (s).
export default function LineChart({ rows, series, height = 220, yLabel, step = false }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 640, H = height, L = 40, R = 12, T = 10, B = 26;
  if (rows.length === 0) return <p className="muted">No data yet.</p>;
  const tMax = rows[rows.length - 1].t || 1;
  const yMax = Math.max(1, ...rows.flatMap((r) => series.map((s) => r[s.key] ?? 0)));
  const x = (t: number) => L + (t / tMax) * (W - L - R);
  const y = (v: number) => T + (1 - v / yMax) * (H - T - B);
  const ticks = niceTicks(yMax);
  const xt = niceTicks(tMax, 6);

  const path = (key: string) =>
    rows
      .map((r, i) => {
        const px = x(r.t), py = y(r[key] ?? 0);
        if (i === 0) return `M${px},${py}`;
        return step ? `H${px}V${py}` : `L${px},${py}`;
      })
      .join("");

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const t = (((e.clientX - box.left) / box.width) * W - L) / (W - L - R) * tMax;
    let best = 0;
    rows.forEach((r, i) => { if (Math.abs(r.t - t) < Math.abs(rows[best].t - t)) best = i; });
    setHover(best);
  };

  const h = hover !== null ? rows[hover] : null;
  return (
    <div>
      <div className="legend" style={{ marginBottom: 6 }}>
        {series.map((s) => (
          <span key={s.key}><span className="swatch" style={{ background: s.color }} />{s.label}
            {h && <strong style={{ marginLeft: 4, color: "var(--text)" }}>{h[s.key] ?? 0}</strong>}</span>
        ))}
        {h && <span>at {fmtTime(h.t)}</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={yLabel}
           onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--border)" />
            <text x={L - 6} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        {xt.map((t) => (
          <text key={t} x={x(t)} y={H - 8} textAnchor="middle">{fmtTime(t)}</text>
        ))}
        {series.map((s) => (
          <path key={s.key} d={path(s.key)} fill="none" stroke={s.color} strokeWidth={2} />
        ))}
        {h && <line x1={x(h.t)} x2={x(h.t)} y1={T} y2={H - B} stroke="var(--muted)" strokeDasharray="3 3" />}
      </svg>
      <div className="muted">{yLabel} · x axis: video time (m:ss)</div>
    </div>
  );
}

function niceTicks(max: number, n = 4): number[] {
  const raw = max / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}
