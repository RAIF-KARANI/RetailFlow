"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import HeatmapCanvas from "@/components/HeatmapCanvas";
import LineChart from "@/components/LineChart";
import { CountRow, fmtTime, get, Heatmap, OccRow, PeopleRow, Session, Summary, Transitions } from "@/lib/api";

const ZONE_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];

export default function CameraPage() {
  const { id } = useParams<{ id: string }>();
  const camera = decodeURIComponent(id);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sid, setSid] = useState<string>("");
  const [bucket, setBucket] = useState(10);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [counts, setCounts] = useState<CountRow[]>([]);
  const [occ, setOcc] = useState<OccRow[]>([]);
  const [people, setPeople] = useState<PeopleRow[]>([]);
  const [heat, setHeat] = useState<Heatmap | null>(null);
  const [trans, setTrans] = useState<Transitions | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Session list (refreshes so a live session appears).
  useEffect(() => {
    const load = () => get<Session[]>(`/api/cameras/${encodeURIComponent(camera)}/sessions`).then((s) => {
      setSessions(s);
      // Default to the newest session (the list itself is grouped by source).
      const newest = s.reduce<Session | undefined>((a, b) => (!a || b.started_at > a.started_at ? b : a), undefined);
      setSid((cur) => cur || newest?.session_id || "");
    }, (e) => setErr(String(e)));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [camera]);

  // Data for the selected session; polled every 5 s while it is still running.
  useEffect(() => {
    if (!sid) return;
    let stop = false;
    const load = async () => {
      try {
        const s = await get<Summary>(`/api/sessions/${sid}/summary`);
        const [c, o, tr, pp] = await Promise.all([
          get<CountRow[]>(`/api/sessions/${sid}/counts?bucket_s=${bucket}`),
          get<OccRow[]>(`/api/sessions/${sid}/occupancy?bucket_s=${Math.max(1, bucket / 2)}`),
          get<Transitions>(`/api/sessions/${sid}/transitions`),
          get<PeopleRow[]>(`/api/sessions/${sid}/people?bucket_s=${Math.max(1, bucket / 2)}`),
        ]);
        if (stop) return;
        setSummary(s); setCounts(c); setOcc(o); setTrans(tr); setPeople(pp); setErr(null);
        if (s.finished) get<Heatmap>(`/api/sessions/${sid}/heatmap`).then((h) => !stop && setHeat(h), () => setHeat(null));
        else setHeat(null);
      } catch (e) { if (!stop) setErr(String(e)); }
    };
    load();
    const t = setInterval(() => { if (!summary?.finished) load(); }, 5000);
    return () => { stop = true; clearInterval(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, bucket]);

  const zones = summary?.zones.map((z) => z.zone) ?? [];
  const removed = summary ? summary.all_in + summary.all_out - summary.cust_in - summary.cust_out : 0;

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div>
        <Link href="/" className="muted">← all cameras</Link>
        <h1>{camera}</h1>
        <div className="row">
          <label className="muted" htmlFor="session">Session</label>
          <select id="session" value={sid} onChange={(e) => { setSid(e.target.value); setHeat(null); }}>
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {s.name} ({s.cust_in} in / {s.cust_out} out){s.finished_at ? "" : " · live"}
              </option>
            ))}
          </select>
          <label className="muted" htmlFor="bucket">Bucket</label>
          <select id="bucket" value={bucket} onChange={(e) => setBucket(Number(e.target.value))}>
            {[5, 10, 30, 60].map((b) => <option key={b} value={b}>{b} s</option>)}
          </select>
          {summary && !summary.finished && <span className="muted">live · refreshing every 5 s</span>}
          {summary && !summary.finished && typeof summary.run_info.preview_url === "string" && (
            // The pipeline's own annotated video (run.py --preview), with a Stop button.
            <a className="watch" href={summary.run_info.preview_url} target="_blank" rel="noopener noreferrer"
               aria-label="Watch live video"
               title="Boxes, counting line, zones and heatmap from the running pipeline">
              &#9654; Watch live video
            </a>
          )}
        </div>
      </div>
      {err && <p className="error">{err}</p>}

      {summary && (
        <div className="grid kpis">
          {summary.finished ? (
            <div className="panel kpi"><div className="label">People in view</div>
              <div className="value">{summary.people_now_mean ?? "–"}</div>
              <div className="sub">on screen on average (session finished)</div></div>
          ) : (
            <div className="panel kpi"><div className="label">People in view now</div>
              <div className="value">{summary.people_now ?? "–"}</div>
              <div className="sub">on screen in the last second</div></div>
          )}
          <div className="panel kpi"><div className="label">Total people</div>
            <div className="value">{summary.people_total ?? "–"}</div>
            <div className="sub">different people seen for {summary.people_min_s} s or more · an estimate:
              one person can count twice if tracking loses them</div></div>
          <div className="panel kpi"><div className="label">Customers in</div>
            <div className="value" style={{ color: "var(--in)" }}>{summary.cust_in}</div>
            <div className="sub">crossed the line towards the IN side</div></div>
          <div className="panel kpi"><div className="label">Customers out</div>
            <div className="value" style={{ color: "var(--out)" }}>{summary.cust_out}</div>
            <div className="sub">crossed towards the OUT side</div></div>
          <div className="panel kpi"><div className="label">Excluded</div>
            <div className="value">{removed}</div>
            <div className="sub">crossings by {summary.staff_tracks} staff and {summary.static_tracks} static tracks</div></div>
          <div className="panel kpi"><div className="label">Video analysed</div>
            <div className="value">{fmtTime(summary.video_seconds)}</div>
            <div className="sub">{String(summary.run_info.model ?? "")} · {String(summary.run_info.imgsz ?? "")} px · {String(summary.run_info.tracker ?? "")}</div></div>
        </div>
      )}

      <div className="grid two">
        <div className="panel">
          <h2>Line crossings per {bucket} s</h2>
          <LineChart rows={counts} step yLabel="people per bucket"
            series={[{ key: "in", label: "In", color: "var(--in)" }, { key: "out", label: "Out", color: "var(--out)" }]} />
        </div>
        <div className="panel">
          <h2>Zone occupancy</h2>
          {zones.length ? (
            <LineChart rows={occ} yLabel="average people in zone"
              series={zones.map((z, i) => ({ key: z, label: z, color: ZONE_COLORS[i % ZONE_COLORS.length] }))} />
          ) : <p className="muted">This camera has no zones.</p>}
        </div>
      </div>

      <div className="grid two">
        <div className="panel">
          <h2>People in view</h2>
          {people.length ? (
            <LineChart rows={people} yLabel="people on screen"
              series={[{ key: "now", label: "In view", color: "var(--series-1)" }]} />
          ) : <p className="muted">No people counts for this session (only runs from this version on have them).</p>}
        </div>
        <div className="panel">
          <h2>Total people so far</h2>
          {people.length ? (
            <LineChart rows={people} step yLabel="people"
              series={[{ key: "total", label: "Total", color: "var(--series-2)" }]} />
          ) : <p className="muted">No people counts for this session.</p>}
        </div>
      </div>

      <div className="grid two">
        <div className="panel">
          <h2>Dwell time by zone</h2>
          <div className="scroll">
            <table>
              <thead><tr><th>Zone</th><th className="num">Visits</th><th className="num">Mean</th>
                <th className="num">Median</th><th className="num">Longest</th><th className="num">In zone now</th></tr></thead>
              <tbody>
                {summary?.zones.map((z) => (
                  <tr key={z.zone}>
                    <td>{z.zone}</td><td className="num">{z.visits}</td>
                    <td className="num">{z.mean_dwell_s ?? "–"} s</td><td className="num">{z.median_dwell_s ?? "–"} s</td>
                    <td className="num">{z.max_dwell_s ?? "–"} s</td><td className="num">{z.occupancy_now}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">A visit counts after 2 s in the zone; leaving for under 2 s does not end it.</p>
        </div>
        <div className="panel">
          <h2>Where people go next</h2>
          {trans && trans.zones.length > 1 ? (() => {
            const max = Math.max(1, ...trans.matrix.flat());
            return (<>
              <div className="scroll">
                <table className="matrix">
                  <thead><tr><th>From \ to</th>{trans.zones.map((z) => <th key={z} className="num">{z}</th>)}</tr></thead>
                  <tbody>
                    {trans.zones.map((a, i) => (
                      <tr key={a}><th scope="row">{a}</th>
                        {trans.zones.map((b, j) => i === j ? <td key={b} className="num muted">–</td> : (
                          <td key={b} className="num"
                              style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(55 * trans.matrix[i][j] / max)}%, transparent)` }}>
                            {trans.matrix[i][j]}
                          </td>))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted">
                {trans.total_steps} moves between zones by {trans.people_moving_between_zones} of {trans.people_with_paths} tracked
                people{trans.steps[0] ? `; most common: ${trans.steps[0].from} → ${trans.steps[0].to} (${trans.steps[0].n})` : ""}.
                A move is lost when tracking breaks between two zones (doc 05).
              </p>
            </>);
          })() : <p className="muted">Needs at least two zones.</p>}
        </div>
      </div>

      <div className="grid two">
        <div className="panel">
          <h2>Heatmap</h2>
          {heat ? <HeatmapCanvas sessionId={sid} heat={heat} />
            : <p className="muted">{summary?.finished === false ? "Shown when the session finishes." : "No heatmap for this session."}</p>}
        </div>
      </div>
    </div>
  );
}
