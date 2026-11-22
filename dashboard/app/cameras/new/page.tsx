"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Camera, get } from "@/lib/api";
import { runner, uploadVideo } from "@/lib/runner";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$/;

// Step 1 of "Add camera": name the camera and upload its video. Step 2 is /setup/<camera>.
export default function NewCamera() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [runnerOk, setRunnerOk] = useState<boolean | null>(null);
  const [existing, setExisting] = useState<string[]>([]);
  const [replace, setReplace] = useState(false);

  useEffect(() => {
    runner.health().then(() => setRunnerOk(true), () => setRunnerOk(false));
    get<Camera[]>("/api/cameras").then((c) => setExisting(c.map((x) => x.camera_id)), () => {});
  }, []);

  const onFile = (f: File | null) => {
    setFile(f);
    if (f && !name) {
      // Suggest a camera name from the file name.
      setName(f.name.replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48));
    }
  };

  const nameOk = NAME_RE.test(name);
  const taken = existing.includes(name);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !nameOk) return;
    setErr(null);
    setProgress(0);
    try {
      await uploadVideo(name, file, replace, setProgress);
      router.push(`/setup/${encodeURIComponent(name)}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setProgress(null);
    }
  };

  return (
    <div className="grid" style={{ gap: 20, maxWidth: 640 }}>
      <div>
        <Link href="/" className="muted">← all cameras</Link>
        <h1>Add camera</h1>
        <p className="muted">
          Upload a video. Next you draw the counting line and zones on one frame, then the video runs as a
          live camera here on the dashboard.
        </p>
      </div>

      {runnerOk === false && (
        <div className="panel">
          <p className="error" style={{ marginTop: 0 }}>The runner is not running.</p>
          <p className="muted">It runs the pipeline on this computer (it needs the GPU). In PowerShell, in the project folder:</p>
          <pre className="code">python runner.py</pre>
          <button onClick={() => runner.health().then(() => setRunnerOk(true), () => setRunnerOk(false))}>Check again</button>
        </div>
      )}

      <form className="panel grid" style={{ gap: 14 }} onSubmit={submit}>
        <label className="field">
          <span>Video file</span>
          <input type="file" accept="video/*,.mpg,.mpeg,.mkv,.avi" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
          <span className="muted">MP4, AVI, MOV, MKV or MPG. It is copied to <code>data/uploads/</code> on this computer.</span>
        </label>
        <label className="field">
          <span>Camera name</span>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. shop_entrance"
                 aria-invalid={!!name && !nameOk} />
          {name && !nameOk && <span className="error">Letters, digits, _ or -, up to 48 characters, no spaces.</span>}
          {taken && <span className="muted">A camera with this name is already on the dashboard: the new run is added to it as a new session.</span>}
        </label>
        <label className="row muted">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          Replace a video already uploaded under this name
        </label>
        {progress !== null && (
          <div>
            <progress value={progress} max={1} style={{ width: "100%" }} />
            <div className="muted">{progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : "Checking the video…"}</div>
          </div>
        )}
        {err && <p className="error">{err}</p>}
        <div className="row">
          <button type="submit" className="primary" disabled={!file || !nameOk || progress !== null || runnerOk === false}>
            Upload and set up
          </button>
        </div>
      </form>
    </div>
  );
}
