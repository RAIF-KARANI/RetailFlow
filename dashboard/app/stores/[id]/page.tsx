"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import FloorMap, { HeatLegend } from "@/components/FloorMap";
import { fmtTime, get, Store, StoreRun } from "@/lib/api";

// One store: all its cameras on one floor, one store-wide number per person.
export default function StorePage() {
  const { id } = useParams<{ id: string }>();
  const storeId = decodeURIComponent(id);
  const [store, setStore] = useState<Store | null>(null);
  const [rid, setRid] = useState("");
  const [run, setRun] = useState<StoreRun | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [layer, setLayer] = useState<"people" | "heat">("people");

  useEffect(() => {
    const load = () => get<Store>(`/api/stores/${encodeURIComponent(storeId)}`).then((s) => {
      setStore(s);
      setRid((cur) => cur || s.runs[0]?.run_id || "");
    }, (e) => setErr(String(e)));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [storeId]);

  useEffect(() => {
    if (!rid) return;
    let stop = false;
    let finished = false;
    const load = () => get<StoreRun>(`/api/store-runs/${rid}`).then((r) => {
      if (!stop) { setRun(r); finished = r.finished; setErr(null); }
    }, (e) => !stop && setErr(String(e)));
    load();
    const t = setInterval(() => { if (!finished) load(); }, 2000);
    return () => { stop = true; clearInterval(t); };
  }, [rid]);

  const st = run?.state ?? {};
  const people = st.people ?? [];
  const moves = st.camera_moves;
  const camIdx = moves ? moves.cameras.map((_, i) => i).filter((i) =>
    moves.matrix[i]?.some((v) => v > 0) || moves.matrix.some((row) => row[i] > 0)) : [];
  const maxMove = Math.max(1, ...(moves?.matrix.flat() ?? [0]));

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div>
        <Link href="/" className="muted">← overview</Link>
        <h1>{storeId}</h1>
        <p className="muted">{store?.description}</p>
        <div className="row">
          <label className="muted" htmlFor="run">Run</label>
          <select id="run" value={rid} onChange={(e) => { setRid(e.target.value); setRun(null); }}>
            {store?.runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {new Date(r.started_at).toLocaleString()} · {r.unique_people ?? 0} people{r.finished_at ? "" : " · live"}
              </option>
            ))}
          </select>
          {run && !run.finished && <span className="muted">live · refreshing every 2 s</span>}
        </div>
      </div>
      {err && <p className="error">{err}</p>}
      {store && store.runs.length === 0 && (
        <p className="muted">No runs yet. Start one with <code>python run_store.py --store scenes/stores/{storeId}.json --api http://localhost:8000</code>.</p>
      )}

      {run && (
        <div className="grid kpis">
          <div className="panel kpi"><div className="label">People now</div>
            <div className="value">{st.people_now?.length ?? 0}</div>
            <div className="sub">seen in the last second, any camera</div></div>
          <div className="panel kpi"><div className="label">Visitors so far</div>
            <div className="value">{st.unique_people ?? 0}</div>
            <div className="sub">store-wide people seen ≥ 2 s (over-counts: see below)</div></div>
          <div className="panel kpi"><div className="label">Cameras</div>
            <div className="value">{store?.config.cameras.length ?? 0}</div>
            <div className="sub">{String(run.run_info.analysis_fps_target ?? "")} analyses / s per camera (target)</div></div>
          <div className="panel kpi"><div className="label">Video time</div>
            <div className="value">{fmtTime(st.t ?? 0)}</div>
            <div className="sub">{String(run.run_info.source ?? "")}</div></div>
        </div>
      )}

      {run && store && (
        <div className="panel">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>{layer === "heat" ? "Floor heatmap" : "Floor now"}</h2>
            <div className="segmented" role="group" aria-label="Floor layer">
              <button type="button" aria-pressed={layer === "people"} onClick={() => setLayer("people")}>People now</button>
              <button type="button" aria-pressed={layer === "heat"} onClick={() => setLayer("heat")}>Heatmap</button>
            </div>
          </div>
          <FloorMap config={store.config} people={st.people_now ?? []} trails={st.trails ?? {}}
                    heat={st.heatmap} mode={layer} />
          {layer === "heat" && st.heatmap ? <HeatLegend heat={st.heatmap} /> : null}
          <p className="muted">
            {layer === "heat"
              ? "Where people spent time since the run started: person-seconds per 0.5 m cell, each store-wide person counted once per second (not once per camera that saw them). Thin lines 1 m, thick 5 m; squares are cameras."
              : "Thin lines 1 m, thick 5 m; squares are cameras. Dots: people now with their store-wide number and last 10 s."}
            {" "}The number comes from linking the cameras&apos; tracks by floor position and clothing colour;
            colours are used in memory only and never stored.
          </p>
        </div>
      )}

      {run && (
        <div className="grid two">
          <div className="panel">
            <h2>People</h2>
            <div className="scroll" style={{ maxHeight: 420 }}>
              <table>
                <thead><tr><th className="num">#</th><th className="num">First</th><th className="num">Last</th>
                  <th className="num">Seen</th><th>Cameras, in order</th></tr></thead>
                <tbody>
                  {[...people].reverse().map((p) => (
                    <tr key={p.id}><td className="num">{p.id}</td><td className="num">{fmtTime(p.first_s)}</td>
                      <td className="num">{fmtTime(p.last_s)}</td><td className="num">{p.seen_s} s</td>
                      <td className="muted">{p.cameras.join(" → ")}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted">
              Accuracy on this synthetic store is limited: one person is often split into several numbers
              (doc 11 §7), so visitors are over-counted.
            </p>
          </div>
          <div className="panel">
            <h2>Camera to camera</h2>
            {moves && camIdx.length > 1 ? (
              <div className="scroll">
                <table className="matrix">
                  <thead><tr><th>From \ next</th>{camIdx.map((j) => <th key={j} className="num">{moves.cameras[j]}</th>)}</tr></thead>
                  <tbody>
                    {camIdx.map((i) => (
                      <tr key={i}><th scope="row">{moves.cameras[i]}</th>
                        {camIdx.map((j) => i === j ? <td key={j} className="num muted">–</td> : (
                          <td key={j} className="num" style={{ background:
                            `color-mix(in srgb, var(--accent) ${Math.round(55 * moves.matrix[i][j] / maxMove)}%, transparent)` }}>
                            {moves.matrix[i][j] || ""}</td>))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p className="muted">No moves between cameras yet.</p>}
            <p className="muted">For each person: the order in which cameras first saw them. Views overlap, so this is
              &quot;which camera picked them up next&quot;, not a walking route.</p>
          </div>
        </div>
      )}
    </div>
  );
}
