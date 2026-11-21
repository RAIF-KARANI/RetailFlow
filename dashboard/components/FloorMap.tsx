"use client";

import { API, FloorHeat, PersonNow, StoreConfig } from "@/lib/api";

// Top-down floor of a store, drawn as a blueprint: deep-blue paper, light 1 m / 5 m grid,
// white camera marks. Two layers: people now (store-wide number, last 10 s) or the heatmap
// (person-seconds per 0.5 m cell). Drawn at S units per metre: browsers apply a minimum font
// size in SVG user units, so text sized in metres (< 1 unit) would come out huge.
const S = 20;
const PAPER = "#0f2f57";
const LINE = "rgba(190, 220, 255, 0.14)";
const LINE_5M = "rgba(190, 220, 255, 0.32)";
const INK = "rgba(235, 244, 255, 0.92)";
const PEOPLE = ["#7cc4ff", "#ffd166", "#c9a7ff", "#6ee7b7", "#ff8f7a"];
const colour = (id: number) => PEOPLE[id % PEOPLE.length];

// cyan -> yellow -> red, transparent where there was nobody
const STOPS: [number, [number, number, number]][] = [[0, [40, 200, 255]], [0.5, [255, 225, 60]], [1, [255, 70, 45]]];
function heatColour(t: number): string {
  const i = t < 0.5 ? 0 : 1;
  const [ta, a] = STOPS[i], [tb, b] = STOPS[i + 1];
  const k = (t - ta) / (tb - ta);
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * k));
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${(0.35 + 0.6 * t).toFixed(2)})`;
}

export function heatScale(heat: FloorHeat): number {
  // 98th percentile, so one spot where someone stood for minutes does not wash out the rest
  const v = heat.cells.map((c) => c[2]).sort((a, b) => a - b);
  return v.length ? Math.max(1, v[Math.min(v.length - 1, Math.floor(0.98 * v.length))]) : 1;
}

export type FloorBase = "blueprint" | "photo" | "grid";

export default function FloorMap({ config, people, trails, heat, mode = "people", base = "grid", zoneNow }: {
  config: StoreConfig;
  people: PersonNow[];
  trails: Record<string, [number, number][]>;
  heat?: FloorHeat;
  mode?: "people" | "heat";
  base?: FloorBase;                       // drawn under everything: floor plan from the cameras, or just the grid
  zoneNow?: Record<string, number>;       // people now per zone, shown in the zone labels
}) {
  const [x0, y0, x1, y1] = config.floor_bounds.map((v) => v * S);
  const w = x1 - x0, h = y1 - y0;
  const lines: { k: string; a: number[]; major: boolean }[] = [];
  for (let v = Math.ceil(x0 / S) * S; v <= x1; v += S)
    lines.push({ k: `x${v}`, a: [v, y0, v, y1], major: Math.round(v / S) % 5 === 0 });
  for (let v = Math.ceil(y0 / S) * S; v <= y1; v += S)
    lines.push({ k: `y${v}`, a: [x0, v, x1, v], major: Math.round(v / S) % 5 === 0 });
  const r = Math.max(w, h) / 110;           // dot radius, scales with the floor
  const font = r * 2.2;
  const top = heat ? heatScale(heat) : 1;
  return (
    <svg viewBox={`${x0} ${y0} ${w} ${h}`} role="img"
         aria-label={mode === "heat" ? "Floor heatmap: where people spent time" : "Store floor with people now"}
         style={{ width: "100%", height: "auto", maxHeight: 640, display: "block", background: PAPER, borderRadius: 8 }}>
      {base !== "grid" && (
        // top-down floor from the cameras (tools/build_floor_plan.py), made over exactly floor_bounds
        <image href={`${API}/api/stores/${encodeURIComponent(config.store_id)}/floor/${base}`}
               x={x0} y={y0} width={w} height={h} preserveAspectRatio="none"
               opacity={base === "photo" ? 0.55 : 0.9} />
      )}
      {lines.map(({ k, a, major }) => (
        <line key={k} x1={a[0]} y1={a[1]} x2={a[2]} y2={a[3]} stroke={major ? LINE_5M : LINE} strokeWidth={major ? 1.4 : 0.7} />
      ))}
      {mode === "heat" && heat && (
        <>
          <defs>
            {/* soften the 0.5 m cells into a continuous glow, like a classic heatmap */}
            <filter id="heat-blur" x="-5%" y="-5%" width="110%" height="110%">
              <feGaussianBlur stdDeviation={heat.cell * S * 0.55} />
            </filter>
          </defs>
          <g filter="url(#heat-blur)">
            {heat.cells.map(([row, col, v]) => (
              <rect key={`${row}-${col}`} x={(heat.x0 + col * heat.cell) * S} y={(heat.y0 + row * heat.cell) * S}
                    width={heat.cell * S * 1.02} height={heat.cell * S * 1.02}
                    fill={heatColour(Math.min(1, Math.sqrt(v / top)))} />
            ))}
          </g>
        </>
      )}
      {(config.zones ?? []).map((z) => {
        const pts = z.polygon.map(([x, y]) => `${x * S},${y * S}`).join(" ");
        const lx = Math.min(...z.polygon.map((q) => q[0]));
        const ly = Math.min(...z.polygon.map((q) => q[1]));
        const n = zoneNow?.[z.name];
        return (
          <g key={z.name}>
            <polygon points={pts} fill="rgba(255, 209, 102, 0.06)" stroke="#ffd166" strokeWidth={r / 4}
                     strokeDasharray={`${r} ${r / 1.5}`} />
            <text x={lx * S + r * 0.8} y={ly * S + font * 1.1} fontSize={font * 0.85} fill="#ffd166" fontWeight={600}>
              {z.name.replace(/_/g, " ")}{n !== undefined ? ` · ${n}` : ""}
            </text>
          </g>
        );
      })}
      {Object.entries(config.camera_xy).map(([cam, [cx, cy]]) => (
        <g key={cam}>
          <rect x={cx * S - r * 0.7} y={cy * S - r * 0.7} width={r * 1.4} height={r * 1.4}
                fill="none" stroke={INK} strokeWidth={r / 4} />
          <text x={cx * S + r} y={cy * S + r * 0.5} fontSize={font * 0.8} fill={INK}>{cam.replace("camera_", "")}</text>
        </g>
      ))}
      {mode === "people" && people.map((p) => {
        const t = trails[String(p.id)] ?? [];
        return t.length > 1 ? (
          <polyline key={`t${p.id}`} points={t.map(([x, y]) => `${x * S},${y * S}`).join(" ")} fill="none"
                    stroke={colour(p.id)} strokeOpacity={0.6} strokeWidth={r / 2.5} strokeLinejoin="round" />
        ) : null;
      })}
      {mode === "people" && people.map((p) => (
        <g key={p.id}>
          <circle cx={p.x * S} cy={p.y * S} r={r} fill={colour(p.id)} stroke={PAPER} strokeWidth={r / 4}>
            <title>{`person ${p.id} · seen by camera ${p.cameras.join(", ")}`}</title>
          </circle>
          <text x={p.x * S + r * 1.2} y={p.y * S - r * 1.2} fontSize={font} fill={INK} fontWeight={600}>{p.id}</text>
        </g>
      ))}
    </svg>
  );
}

export function HeatLegend({ heat }: { heat: FloorHeat }) {
  const top = heatScale(heat);
  const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => `${heatColour(t)} ${t * 100}%`).join(", ");
  return (
    <div className="row" style={{ gap: 10 }}>
      <span className="muted">0</span>
      <span style={{ width: 160, height: 10, borderRadius: 5, background: `linear-gradient(90deg, ${stops})` }} />
      <span className="muted">≥ {top} person-seconds per {heat.cell} m cell · total {Math.round(heat.total_s / 60)} person-minutes</span>
    </div>
  );
}
