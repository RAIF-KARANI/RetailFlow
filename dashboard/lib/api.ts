// Typed access to the RetailFlow API. The browser calls the API directly.
export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Camera = {
  camera_id: string;
  description: string;
  sessions: number;
  video_seconds: number;
  cust_in: number;
  cust_out: number;
  zones: string[];
};

export type Totals = {
  cust_in: number;
  cust_out: number;
  all_in: number;
  all_out: number;
  staff_tracks: number;
  static_tracks: number;
};

export type Session = Totals & {
  session_id: string;
  name: string;
  source: string;
  frames: number | null;
  video_fps: number | null;
  finished_at: string | null;
  run_info: Record<string, unknown>;
};

export type ZoneStats = {
  zone: string;
  visits: number;
  mean_dwell_s: number | null;
  median_dwell_s: number | null;
  max_dwell_s: number | null;
  occupancy_now: number;
};

export type Summary = Totals & {
  session_id: string;
  camera_id: string;
  source: string;
  video_seconds: number;
  finished: boolean;
  run_info: Record<string, unknown>;
  zones: ZoneStats[];
};

export type CountRow = { t: number; in: number; out: number };
export type OccRow = { t: number } & Record<string, number>;
export type Heatmap = { cell: number; floor: boolean; grid: number[][]; has_background: boolean; frame_size: [number, number] };

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json() as Promise<T>;
}

/** Seconds -> "m:ss" on the video timeline (never a fake clock time). */
export function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
