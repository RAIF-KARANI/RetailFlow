"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Camera, fmtTime, get } from "@/lib/api";

// Overview: every camera with its customer totals over all sessions.
export default function Home() {
  const [cams, setCams] = useState<Camera[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const load = () => get<Camera[]>("/api/cameras").then(setCams, (e) => setErr(String(e)));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div>
        <h1>Cameras</h1>
        <p className="muted">
          Customers only: tracks flagged as staff or as static objects (mannequins) are excluded.
          Each session is one video or stream, on its own video timeline.
        </p>
      </div>
      {err && <p className="error">Cannot reach the API: {err}</p>}
      <div className="panel scroll">
        <table>
          <thead>
            <tr>
              <th>Camera</th>
              <th className="num">Sessions</th>
              <th className="num">Video</th>
              <th className="num">Customers in</th>
              <th className="num">Customers out</th>
              <th>Zones</th>
            </tr>
          </thead>
          <tbody>
            {cams?.map((c) => (
              <tr key={c.camera_id}>
                <td>
                  <Link href={`/cameras/${encodeURIComponent(c.camera_id)}`}>{c.camera_id}</Link>
                  <div className="muted">{c.description}</div>
                </td>
                <td className="num">{c.sessions}</td>
                <td className="num">{fmtTime(c.video_seconds)}</td>
                <td className="num">{c.cust_in}</td>
                <td className="num">{c.cust_out}</td>
                <td className="muted">{c.zones.join(", ") || "none"}</td>
              </tr>
            ))}
            {cams?.length === 0 && (
              <tr><td colSpan={6} className="muted">No cameras yet. Run <code>python run.py ... --api http://localhost:8000</code>.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
