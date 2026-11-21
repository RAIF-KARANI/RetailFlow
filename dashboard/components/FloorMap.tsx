"use client";

import { PersonNow, StoreConfig } from "@/lib/api";

// Top-down floor of a store: cameras, people now (store-wide number) and their last 10 s.
// SVG so it scales to any width. Drawn at S units per metre: browsers apply a minimum font
// size in SVG user units, so text sized in metres (< 1 unit) would come out huge.
const S = 20;
const PEOPLE = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--out)"];
const colour = (id: number) => PEOPLE[id % PEOPLE.length];

export default function FloorMap({ config, people, trails }: {
  config: StoreConfig;
  people: PersonNow[];
  trails: Record<string, [number, number][]>;
}) {
  const [x0, y0, x1, y1] = config.floor_bounds.map((v) => v * S);
  const w = x1 - x0, h = y1 - y0;
  const lines: { k: string; a: number[] }[] = [];
  for (let v = Math.ceil(x0 / (5 * S)) * 5 * S; v <= x1; v += 5 * S) lines.push({ k: `x${v}`, a: [v, y0, v, y1] });
  for (let v = Math.ceil(y0 / (5 * S)) * 5 * S; v <= y1; v += 5 * S) lines.push({ k: `y${v}`, a: [x0, v, x1, v] });
  const r = Math.max(w, h) / 110;           // dot radius, scales with the floor
  const font = r * 2.2;
  return (
    <svg viewBox={`${x0} ${y0} ${w} ${h}`} role="img" aria-label="Store floor with people now"
         style={{ width: "100%", height: "auto", maxHeight: 640, display: "block",
                  background: "var(--bg)", borderRadius: 8 }}>
      {lines.map(({ k, a }) => (
        <line key={k} x1={a[0]} y1={a[1]} x2={a[2]} y2={a[3]} stroke="var(--border)" strokeWidth={1} />
      ))}
      {Object.entries(config.camera_xy).map(([cam, [cx, cy]]) => (
        <g key={cam}>
          <rect x={cx * S - r * 0.7} y={cy * S - r * 0.7} width={r * 1.4} height={r * 1.4} fill="var(--muted)" />
          <text x={cx * S + r} y={cy * S + r * 0.5} fontSize={font * 0.8} fill="var(--muted)">
            {cam.replace("camera_", "")}
          </text>
        </g>
      ))}
      {people.map((p) => {
        const t = trails[String(p.id)] ?? [];
        return t.length > 1 ? (
          <polyline key={`t${p.id}`} points={t.map(([x, y]) => `${x * S},${y * S}`).join(" ")} fill="none"
                    stroke={colour(p.id)} strokeOpacity={0.55} strokeWidth={r / 2.5} strokeLinejoin="round" />
        ) : null;
      })}
      {people.map((p) => (
        <g key={p.id}>
          <circle cx={p.x * S} cy={p.y * S} r={r} fill={colour(p.id)} stroke="var(--panel)" strokeWidth={r / 4}>
            <title>{`person ${p.id} · seen by camera ${p.cameras.join(", ")}`}</title>
          </circle>
          <text x={p.x * S + r * 1.2} y={p.y * S - r * 1.2} fontSize={font} fill="var(--text)" fontWeight={600}>
            {p.id}
          </text>
        </g>
      ))}
    </svg>
  );
}
