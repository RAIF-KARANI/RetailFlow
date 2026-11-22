"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { get, Session } from "@/lib/api";
import { Pt, runner, RunStatus, SceneDict, VideoInfo } from "@/lib/runner";

type Mode = "line" | "in" | "zone";
const ZONE_COLORS = ["var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-1)"];
const r1 = (v: number) => Math.round(v);

// Step 2 of "Add camera": draw the counting line, the IN side and zones on one frame
// (the web version of tools/setup_scene.py), save scenes/<camera>.json, then start it live.
export default function Setup() {
  const { id } = useParams<{ id: string }>();
  const cam = decodeURIComponent(id);
  const router = useRouter();
  const svgRef = useRef<SVGSVGElement>(null);

  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [scene, setScene] = useState<SceneDict | null>(null);
  const [existed, setExisted] = useState(false);
  const [frame, setFrame] = useState(50);
  const [frameInput, setFrameInput] = useState("50");
  const [imgErr, setImgErr] = useState(false);
  const [mode, setMode] = useState<Mode>("line");
  const [lineClicks, setLineClicks] = useState<Pt[]>([]);
  const [draft, setDraft] = useState<Pt[]>([]);
  const [zoneName, setZoneName] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState(false);
  const [minutes, setMinutes] = useState("5");
  const [run, setRun] = useState<RunStatus | null>(null);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    runner.video(cam).then(setInfo, (e) => setErr(String(e.message ?? e)));
    runner.scene(cam).then((r) => { setScene(r.scene); setExisted(r.exists); setSaved(r.exists); },
                           (e) => setErr(String(e.message ?? e)));
  }, [cam]);

  // After Start: wait for the run's live session on the API, then open the camera page.
  useEffect(() => {
    if (!startedAt) return;
    const t = setInterval(async () => {
      try {
        const st = await runner.run(cam);
        setRun(st);
        if (st.state === "failed" || st.state === "finished") { setStartedAt(null); return; }
        const sessions = await get<Session[]>(`/api/cameras/${encodeURIComponent(cam)}/sessions`).catch(() => []);
        const since = new Date(startedAt).getTime();
        if (sessions.some((s) => !s.finished_at && new Date(s.started_at).getTime() >= since)) {
          router.push(`/cameras/${encodeURIComponent(cam)}`);
        }
      } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    }, 2000);
    return () => clearInterval(t);
  }, [startedAt, cam, router]);

  if (!scene) {
    return (
      <div className="grid" style={{ gap: 20 }}>
        <Link href="/" className="muted">← all cameras</Link>
        <h1>Set up {cam}</h1>
        {err ? <p className="error">{err}</p> : <p className="muted">Loading…</p>}
        {err && <Link href="/cameras/new">Upload a video</Link>}
      </div>
    );
  }

  const [W, H] = scene.frame_size;
  const sw = Math.max(2, W / 400);        // stroke width in image pixels
  const fs = Math.max(12, W / 55);        // label size in image pixels
  const ln = scene.line;

  const edit = (s: SceneDict) => { setScene(s); setDirty(true); setSaved(false); setMsg(null); };

  const toImage = (e: React.PointerEvent): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return [r1(((e.clientX - r.left) / r.width) * W), r1(((e.clientY - r.top) / r.height) * H)];
  };

  const onClick = (e: React.PointerEvent) => {
    const p = toImage(e);
    if (mode === "line") {
      const c = [...lineClicks, p];
      if (c.length === 2) {
        edit({ ...scene, line: { ...ln, p1: c[0], p2: c[1] } });
        setLineClicks([]);
        setMode("in");                     // the IN side comes next
      } else setLineClicks(c);
    } else if (mode === "in") {
      edit({ ...scene, line: { ...ln, in_point: p } });
    } else {
      setDraft([...draft, p]);
    }
  };

  const names = scene.zones.map((z) => z.name);
  const zoneNameOk = /^[A-Za-z0-9 _-]{1,40}$/.test(zoneName) && !names.includes(zoneName.trim());
  const addZone = () => {
    if (draft.length < 3 || !zoneNameOk) return;
    edit({ ...scene, zones: [...scene.zones, { name: zoneName.trim(), polygon: draft }] });
    setDraft([]); setZoneName("");
  };

  const save = async () => {
    setErr(null);
    try {
      const r = await runner.saveScene(cam, scene);
      setScene(r.scene); setDirty(false); setSaved(true); setExisted(true);
      setMsg(`Saved ${r.saved}`);
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  };

  const start = async () => {
    setErr(null);
    const m = Number(minutes);
    try {
      const now = new Date(Date.now() - 5000).toISOString();     // small margin for clock rounding
      setRun(await runner.start(cam, m));
      setStartedAt(now);
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  };

  const loadFrame = () => {
    const n = Math.max(0, Math.floor(Number(frameInput) || 0));
    setImgErr(false); setFrame(n); setFrameInput(String(n));
  };

  // IN-side arrow: from the line's middle towards the IN point.
  const mid: Pt = [(ln.p1[0] + ln.p2[0]) / 2, (ln.p1[1] + ln.p2[1]) / 2];
  const starting = !!startedAt;

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div>
        <Link href="/" className="muted">← all cameras</Link>
        <h1>Set up {cam}</h1>
        <p className="muted">
          {info ? `${info.video} · ${info.width}×${info.height} · ${info.fps} fps` : ""}
          {existed ? " · editing the saved scene" : " · new scene"}
        </p>
      </div>

      <div className="grid setup-layout">
        <div className="panel grid" style={{ gap: 10 }}>
          <div className="row">
            <div className="segmented" role="group" aria-label="What to draw">
              <button aria-pressed={mode === "line"} onClick={() => { setMode("line"); setLineClicks([]); }}>1 · Counting line</button>
              <button aria-pressed={mode === "in"} onClick={() => setMode("in")}>2 · IN side</button>
              <button aria-pressed={mode === "zone"} onClick={() => setMode("zone")}>3 · Zones</button>
            </div>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            {mode === "line" && (lineClicks.length === 0
              ? "Click 2 points on the floor, where feet touch the ground, across the path people walk."
              : "Now click the second point of the line.")}
            {mode === "in" && "Click once on the side people walk into (for example inside the shop)."}
            {mode === "zone" && "Click the corners of a zone on the floor, then name it and press Add zone."}
          </p>
          <div className="setup-canvas">
            {!imgErr ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={runner.frameUrl(cam, frame)} alt={`Frame ${frame} of ${cam}`} onError={() => setImgErr(true)}
                   width={W} height={H} draggable={false} />
            ) : (
              <div style={{ aspectRatio: `${W} / ${H}`, display: "grid", placeItems: "center", color: "#ccc" }}>
                No frame {frame} in this video: try a smaller number.
              </div>
            )}
            <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onPointerDown={onClick}
                 style={{ fontSize: fs }} aria-label="Drawing area">
              {scene.zones.map((z, i) => (
                <g key={z.name}>
                  <polygon points={z.polygon.map((p) => p.join(",")).join(" ")} fill={ZONE_COLORS[i % 4]}
                           fillOpacity={0.22} stroke={ZONE_COLORS[i % 4]} strokeWidth={sw} />
                  <text x={z.polygon[0][0]} y={z.polygon[0][1] - fs * 0.4} style={{ fill: "#fff", paintOrder: "stroke",
                        stroke: "#000", strokeWidth: fs / 6 }}>{z.name}</text>
                </g>
              ))}
              {draft.length > 0 && (
                <g>
                  <polyline points={[...draft, draft[0]].map((p) => p.join(",")).join(" ")} fill="none"
                            stroke="#fff" strokeWidth={sw} strokeDasharray={`${sw * 3} ${sw * 2}`} />
                  {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={sw * 2.2} fill="#fff" />)}
                </g>
              )}
              <line x1={ln.p1[0]} y1={ln.p1[1]} x2={ln.p2[0]} y2={ln.p2[1]} stroke="#ffd23f" strokeOpacity={0.3}
                    strokeWidth={Math.max(sw, ln.band_px * 2)} strokeLinecap="round" />
              <line x1={ln.p1[0]} y1={ln.p1[1]} x2={ln.p2[0]} y2={ln.p2[1]} stroke="#ffd23f" strokeWidth={sw * 1.5} />
              <line x1={mid[0]} y1={mid[1]} x2={ln.in_point[0]} y2={ln.in_point[1]} stroke="#4cc28f" strokeWidth={sw}
                    strokeDasharray={`${sw * 3} ${sw * 2}`} />
              <circle cx={ln.in_point[0]} cy={ln.in_point[1]} r={sw * 3} fill="#4cc28f" />
              <text x={ln.in_point[0] + sw * 4} y={ln.in_point[1] + fs * 0.35} style={{ fill: "#4cc28f", paintOrder: "stroke",
                    stroke: "#000", strokeWidth: fs / 6, fontWeight: 700 }}>IN</text>
              {lineClicks.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={sw * 2.5} fill="#ffd23f" />)}
            </svg>
          </div>
          <div className="row">
            <label className="muted" htmlFor="frame">Frame</label>
            <input id="frame" type="number" min={0} value={frameInput} style={{ width: 110 }}
                   onChange={(e) => setFrameInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && loadFrame()} />
            <button onClick={loadFrame}>Show frame</button>
            <span className="muted">Pick a frame where people do not block the floor.</span>
          </div>
        </div>

        <div className="grid" style={{ gap: 16 }}>
          <div className="panel grid" style={{ gap: 10 }}>
            <h2 style={{ margin: 0 }}>Counting line</h2>
            <label className="row muted" htmlFor="band">
              Band
              <input id="band" type="number" min={0} max={200} value={ln.band_px} style={{ width: 80 }}
                     onChange={(e) => edit({ ...scene, line: { ...ln, band_px: Math.max(0, Number(e.target.value) || 0) } })} />
              px
            </label>
            <p className="muted" style={{ margin: 0 }}>The band stops someone standing on the line from being counted again and again.</p>
          </div>

          <div className="panel grid" style={{ gap: 10 }}>
            <h2 style={{ margin: 0 }}>Zones</h2>
            {scene.zones.length === 0 && draft.length === 0 && <p className="muted" style={{ margin: 0 }}>No zones yet (optional).</p>}
            {scene.zones.map((z, i) => (
              <div key={z.name} className="row" style={{ justifyContent: "space-between" }}>
                <span><span className="swatch" style={{ background: ZONE_COLORS[i % 4] }} />{z.name}</span>
                <button className="danger" onClick={() => edit({ ...scene, zones: scene.zones.filter((x) => x.name !== z.name) })}
                        aria-label={`Delete zone ${z.name}`}>Delete</button>
              </div>
            ))}
            {mode === "zone" && (
              <div className="grid" style={{ gap: 8 }}>
                <span className="muted">{draft.length} corner{draft.length === 1 ? "" : "s"} (at least 3)</span>
                <input type="text" placeholder="Zone name, e.g. checkout" value={zoneName}
                       onChange={(e) => setZoneName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addZone()}
                       aria-label="Zone name" />
                {zoneName && !zoneNameOk && <span className="error">Use letters, digits, spaces, _ or -; each name once.</span>}
                <div className="row">
                  <button className="primary" disabled={draft.length < 3 || !zoneNameOk} onClick={addZone}>Add zone</button>
                  <button disabled={draft.length === 0} onClick={() => setDraft(draft.slice(0, -1))}>Undo corner</button>
                  <button disabled={draft.length === 0} onClick={() => setDraft([])}>Clear</button>
                </div>
              </div>
            )}
          </div>

          <div className="panel grid" style={{ gap: 10 }}>
            <h2 style={{ margin: 0 }}>Save and go live</h2>
            <div className="row">
              <button className="primary" onClick={save} disabled={!dirty && saved}>Save scene</button>
              {saved && !dirty && <span className="muted">{msg ?? "Saved"}</span>}
              {dirty && <span className="muted">Unsaved changes</span>}
            </div>
            <label className="row muted" htmlFor="minutes">
              Run for
              <input id="minutes" type="number" min={0} max={1440} value={minutes} style={{ width: 80 }}
                     onChange={(e) => setMinutes(e.target.value)} />
              minutes (0 = until you stop it)
            </label>
            <p className="muted" style={{ margin: 0 }}>The video loops at real speed, like a live CCTV camera.</p>
            <button className="primary" onClick={start} disabled={!saved || dirty || starting}>
              {starting ? "Starting…" : "▶ Start live"}
            </button>
            {starting && (
              <p className="muted" style={{ margin: 0 }}>
                {run?.state === "starting" ? "Starting the camera stream…" : "Loading the model; the camera page opens when the first numbers arrive."}
              </p>
            )}
            {run && run.state === "failed" && (
              <div>
                <p className="error" style={{ margin: 0 }}>{run.error}</p>
                {run.log_tail && <pre className="code" style={{ fontSize: 12 }}>{run.log_tail}</pre>}
              </div>
            )}
          </div>
          {err && <p className="error">{err}</p>}
        </div>
      </div>
    </div>
  );
}
