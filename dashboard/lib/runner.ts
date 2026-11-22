// The host runner (python runner.py): upload a video, set up its scene, start/stop a live run.
// It runs on the host because the pipeline needs the GPU; the browser calls it directly.
export const RUNNER = process.env.NEXT_PUBLIC_RUNNER_URL ?? "http://127.0.0.1:8095";

export type Pt = [number, number];
export type SceneZone = { name: string; polygon: Pt[] };
export type SceneDict = {
  camera_id: string;
  description?: string;
  profile?: string;
  frame_size: [number, number];
  line: { p1: Pt; p2: Pt; in_point: Pt; band_px: number };
  zones: SceneZone[];
  [key: string]: unknown;            // homography, staff, static_filter: kept as they are
};
export type VideoInfo = { camera_id: string; video: string; width: number; height: number;
  fps: number; frames: number | null; seconds: number | null };
export type RunStatus = { camera_id: string; state: "none" | "starting" | "running" | "stopping" | "finished" | "failed";
  error?: string | null; preview_url?: string | null; minutes?: number; log_tail?: string };

async function call<T>(method: string, path: string, body?: BodyInit, json = true): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`${RUNNER}${path}`, {
      method, body, cache: "no-store",
      headers: body && json ? { "Content-Type": "application/json" } : undefined,
    });
  } catch {
    throw new Error("The runner is not running. In PowerShell, in the project folder: python runner.py");
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `${r.status} ${path}`);
  return data as T;
}

const cam = (c: string) => `/api/cameras/${encodeURIComponent(c)}`;

export const runner = {
  health: () => call<{ ok: boolean }>("GET", "/api/health"),
  video: (c: string) => call<VideoInfo>("GET", `${cam(c)}/video`),
  frameUrl: (c: string, n: number) => `${RUNNER}${cam(c)}/frame?n=${n}`,
  scene: (c: string) => call<{ exists: boolean; scene: SceneDict }>("GET", `${cam(c)}/scene`),
  saveScene: (c: string, s: SceneDict) => call<{ saved: string; scene: SceneDict }>("PUT", `${cam(c)}/scene`, JSON.stringify(s)),
  start: (c: string, minutes: number) => call<RunStatus>("POST", `${cam(c)}/start`, JSON.stringify({ minutes })),
  stop: (c: string) => call<RunStatus>("POST", `${cam(c)}/stop`, "{}"),
  run: (c: string) => call<RunStatus>("GET", `${cam(c)}/run`),
};

/** Upload with progress (fetch has no upload progress). The body is the raw file. */
export function uploadVideo(c: string, file: File, replace: boolean, onProgress: (f: number) => void): Promise<VideoInfo> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", `${RUNNER}${cam(c)}/video?filename=${encodeURIComponent(file.name)}${replace ? "&replace=1" : ""}`);
    x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    x.onload = () => {
      let d: { detail?: string } = {};
      try { d = JSON.parse(x.responseText); } catch { /* not JSON */ }
      if (x.status >= 200 && x.status < 300) resolve(d as VideoInfo);
      else reject(new Error(d.detail ?? `upload failed (${x.status})`));
    };
    x.onerror = () => reject(new Error("The runner is not running. In PowerShell, in the project folder: python runner.py"));
    x.send(file);
  });
}
