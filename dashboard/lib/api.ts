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

export type Transitions = {
  zones: string[];
  matrix: number[][];            // matrix[from][to], people who went from one zone to the other
  steps: { from: string; to: string; n: number }[];
  total_steps: number;
  people_with_paths: number;
  people_moving_between_zones: number;
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

// ---- stores (several cameras, one floor; run_store.py)
export type StoreListItem = { store_id: string; description: string; cameras: number; runs: number; live: boolean };

export type StoreConfig = {
  store_id: string;
  description: string;
  cameras: string[];
  camera_xy: Record<string, [number, number]>;
  floor_bounds: [number, number, number, number];   // x min, y min, x max, y max (metres)
  zones?: { name: string; polygon: [number, number][] }[];
};

export type StoreRunRow = { run_id: string; run_info: Record<string, unknown>; started_at: string;
  finished_at: string | null; t: number | null; unique_people: number | null };

export type Store = { store_id: string; description: string; config: StoreConfig; runs: StoreRunRow[];
  floors?: string[] };

export type PersonNow = { id: number; x: number; y: number; cameras: string[] };
export type StorePerson = { id: number; first_s: number; last_s: number; seen_s: number; cameras: string[] };

export type StoreState = {
  t?: number;
  people_now?: PersonNow[];
  trails?: Record<string, [number, number][]>;
  unique_people?: number;
  people?: StorePerson[];
  camera_people_now?: Record<string, number>;
  camera_moves?: { cameras: string[]; matrix: number[][] };
  heatmap?: FloorHeat;
  zones?: { zone: string; now: number; visits: number; mean_dwell_s: number | null; median_dwell_s: number | null }[];
  zone_moves?: { zones: string[]; matrix: number[][] };
};

// Person-seconds on the floor, counted once per store-wide person per second; sparse cells.
export type FloorHeat = { cell: number; x0: number; y0: number; rows: number; cols: number;
  total_s: number; cells: [number, number, number][] };

export type StoreRun = { run_id: string; store_id: string; run_info: Record<string, unknown>; finished: boolean;
  state: StoreState; summary: Record<string, unknown> | null };
