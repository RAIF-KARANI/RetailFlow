"""RetailFlow API: receives anonymous events from the pipeline and serves the dashboard.

Write side (used by retailflow/sender.py):
  PUT  /api/cameras/{camera_id}            register / update a camera's scene
  DELETE /api/cameras/{camera_id}          remove a camera and all its data
  POST /api/sessions                       start a session (one pipeline run)
  POST /api/sessions/{id}/events           append a batch of events
  POST /api/sessions/{id}/finish           summary + heatmap + background image

Read side (used by the dashboard):
  GET  /api/cameras                        cameras with session counts and customer totals
  GET  /api/cameras/{camera_id}/sessions   sessions of a camera with their totals
  GET  /api/sessions/{id}/summary          IN/OUT, staff/static removed, dwell per zone
  GET  /api/sessions/{id}/counts           IN/OUT per time bucket
  GET  /api/sessions/{id}/occupancy        people per zone per time bucket
  GET  /api/sessions/{id}/heatmap          person-seconds grid
  GET  /api/sessions/{id}/background.png   people-free background for the heatmap

"Customer" analytics always use the customer_events view, which drops every
event of a track later flagged as staff or as a static object.
All times are seconds on the session timeline (video time for files).
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.analytics import counts_series, dwell_stats, occupancy_series, pair_visits, transition_table

DSN = os.environ.get("DATABASE_URL", "postgresql://retailflow:retailflow@localhost:5432/retailflow")
EVENT_TYPES = {"line_cross", "zone_enter", "zone_exit", "staff_flag", "static_flag"}


def connect():
    return psycopg.connect(DSN, row_factory=dict_row)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    with connect() as conn:
        conn.execute((Path(__file__).parent / "schema.sql").read_text())
    yield


app = FastAPI(title="RetailFlow API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ models
class SessionIn(BaseModel):
    camera_id: str
    source: str
    video_fps: float | None = None
    run_info: dict = Field(default_factory=dict)


class EventIn(BaseModel):
    frame: int
    timestamp: float
    type: str
    track_id: int | None = None
    direction: str | None = None
    zone: str | None = None
    x: float | None = None
    y: float | None = None
    value: float | None = None
    staff: bool = False


class HeatmapIn(BaseModel):
    cell: float
    floor: bool = False
    grid: list[list[float]]


class FinishIn(BaseModel):
    summary: dict
    heatmap: HeatmapIn | None = None
    background_png_b64: str | None = None


# ------------------------------------------------------------------ helpers
def _session(conn, sid: uuid.UUID) -> dict:
    row = conn.execute("SELECT s.*, c.scene FROM sessions s JOIN cameras c USING (camera_id) "
                       "WHERE session_id = %s", (sid,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown session")
    return row


def _end_s(conn, s: dict) -> float:
    """Length of the session timeline: frames / fps when finished, else last event time.

    `video_frames` (stream position) is used when present: on a live stream that skipped
    frames, `frames` counts only the analysed ones and would make the video look shorter.
    """
    summ = s["summary"] or {}
    frames = summ.get("video_frames") or summ.get("frames")
    if frames and s["video_fps"]:
        return frames / s["video_fps"]
    r = conn.execute("SELECT max(ts) AS t FROM events WHERE session_id = %s", (s["session_id"],)).fetchone()
    return float(r["t"] or 0.0)


def _totals(conn, sid) -> dict:
    r = conn.execute("""
        SELECT
          (SELECT count(*) FROM customer_events WHERE session_id = %(s)s AND type = 'line_cross' AND direction = 'in')  AS cust_in,
          (SELECT count(*) FROM customer_events WHERE session_id = %(s)s AND type = 'line_cross' AND direction = 'out') AS cust_out,
          (SELECT count(*) FROM events WHERE session_id = %(s)s AND type = 'line_cross' AND direction = 'in')  AS all_in,
          (SELECT count(*) FROM events WHERE session_id = %(s)s AND type = 'line_cross' AND direction = 'out') AS all_out,
          (SELECT count(DISTINCT track_id) FROM events WHERE session_id = %(s)s AND type = 'staff_flag')  AS staff_tracks,
          (SELECT count(DISTINCT track_id) FROM events WHERE session_id = %(s)s AND type = 'static_flag') AS static_tracks
    """, {"s": sid}).fetchone()
    return dict(r)


def _visits(conn, s: dict) -> list[dict]:
    rows = conn.execute("SELECT type, track_id, zone, ts, value FROM customer_events "
                        "WHERE session_id = %s AND type IN ('zone_enter', 'zone_exit') ORDER BY ts, id",
                        (s["session_id"],)).fetchall()
    return pair_visits(rows, _end_s(conn, s))


def _source_name(source: str) -> str:
    """Short label for a session: file stem, or the sequence folder for MOT-style .../SEQ/img1."""
    if "://" in source:
        return source
    p = Path(source.replace("\\", "/").rstrip("/"))
    return p.parent.name if p.name in ("img1", "images", "frames") else p.stem


def _zones(s: dict) -> list[str]:
    return [z["name"] for z in s["scene"].get("zones", [])]


# ------------------------------------------------------------------ write side
@app.get("/api/health")
def health():
    with connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.put("/api/cameras/{camera_id}")
def put_camera(camera_id: str, scene: dict):
    if scene.get("camera_id", camera_id) != camera_id:
        raise HTTPException(400, "camera_id in body does not match the URL")
    with connect() as conn:
        conn.execute("""INSERT INTO cameras (camera_id, description, scene) VALUES (%s, %s, %s)
                        ON CONFLICT (camera_id) DO UPDATE
                        SET description = EXCLUDED.description, scene = EXCLUDED.scene, updated_at = now()""",
                     (camera_id, scene.get("description", ""), Jsonb(scene)))
    return {"camera_id": camera_id}


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str):
    """Remove a camera and all its sessions, events and heatmaps (cascade)."""
    with connect() as conn:
        n = conn.execute("DELETE FROM cameras WHERE camera_id = %s", (camera_id,)).rowcount
    if not n:
        raise HTTPException(404, "unknown camera")
    return {"deleted": camera_id}


@app.post("/api/sessions", status_code=201)
def start_session(body: SessionIn):
    sid = uuid.uuid4()
    with connect() as conn:
        if conn.execute("SELECT 1 FROM cameras WHERE camera_id = %s", (body.camera_id,)).fetchone() is None:
            raise HTTPException(404, "unknown camera: register it with PUT /api/cameras/{id} first")
        conn.execute("INSERT INTO sessions (session_id, camera_id, source, video_fps, run_info) "
                     "VALUES (%s, %s, %s, %s, %s)",
                     (sid, body.camera_id, body.source, body.video_fps, Jsonb(body.run_info)))
    return {"session_id": str(sid)}


@app.post("/api/sessions/{sid}/events")
def add_events(sid: uuid.UUID, events: list[EventIn]):
    bad = {e.type for e in events} - EVENT_TYPES
    if bad:
        raise HTTPException(422, f"unknown event type(s): {sorted(bad)}")
    with connect() as conn:
        _session(conn, sid)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO events (session_id, frame, ts, type, track_id, direction, zone, x, y, value, staff) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(sid, e.frame, e.timestamp, e.type, e.track_id, e.direction, e.zone, e.x, e.y, e.value, e.staff)
                 for e in events])
    return {"inserted": len(events)}


@app.post("/api/sessions/{sid}/finish")
def finish_session(sid: uuid.UUID, body: FinishIn):
    with connect() as conn:
        _session(conn, sid)
        conn.execute("UPDATE sessions SET summary = %s, finished_at = now() WHERE session_id = %s",
                     (Jsonb(body.summary), sid))
        if body.heatmap is not None:
            bg = base64.b64decode(body.background_png_b64) if body.background_png_b64 else None
            conn.execute("""INSERT INTO heatmaps (session_id, cell, floor, grid, background)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE SET cell = EXCLUDED.cell, floor = EXCLUDED.floor,
                                grid = EXCLUDED.grid, background = EXCLUDED.background""",
                         (sid, body.heatmap.cell, body.heatmap.floor, Jsonb(body.heatmap.grid), bg))
    return {"session_id": str(sid)}


# ------------------------------------------------------------------ read side
@app.get("/api/cameras")
def list_cameras():
    with connect() as conn:
        cams = conn.execute("""
            SELECT c.camera_id, c.description, c.scene,
                   count(s.session_id) AS sessions,
                   coalesce(sum(s.video_seconds), 0) AS video_seconds
            FROM cameras c
            LEFT JOIN (SELECT session_id, camera_id,
                              coalesce(summary->>'video_frames', summary->>'frames')::float
                                / nullif(video_fps, 0) AS video_seconds
                       FROM sessions) s USING (camera_id)
            GROUP BY c.camera_id ORDER BY c.camera_id""").fetchall()
        for c in cams:
            t = conn.execute("""
                SELECT count(*) FILTER (WHERE e.direction = 'in')  AS cust_in,
                       count(*) FILTER (WHERE e.direction = 'out') AS cust_out
                FROM customer_events e JOIN sessions s USING (session_id)
                WHERE s.camera_id = %s AND e.type = 'line_cross'""", (c["camera_id"],)).fetchone()
            c.update(t)
            c["zones"] = [z["name"] for z in c.pop("scene").get("zones", [])]
            c["video_seconds"] = round(float(c["video_seconds"]), 1)
    return cams


@app.get("/api/cameras/{camera_id}/sessions")
def camera_sessions(camera_id: str):
    with connect() as conn:
        rows = conn.execute("""SELECT session_id, source, video_fps, run_info, started_at, finished_at,
                                      (summary->>'frames')::int AS frames
                               FROM sessions WHERE camera_id = %s ORDER BY source, started_at""",
                            (camera_id,)).fetchall()
        for r in rows:
            r.update(_totals(conn, r["session_id"]))
            r["session_id"] = str(r["session_id"])
            r["name"] = _source_name(r["source"])
    return rows


@app.get("/api/sessions/{sid}/summary")
def session_summary(sid: uuid.UUID):
    with connect() as conn:
        s = _session(conn, sid)
        end_s = _end_s(conn, s)
        visits = _visits(conn, s)
        return {"session_id": str(sid), "camera_id": s["camera_id"], "source": s["source"],
                "video_seconds": round(end_s, 1), "finished": s["finished_at"] is not None,
                "run_info": s["run_info"], **_totals(conn, sid),
                "zones": dwell_stats(visits, _zones(s)),
                "line": s["scene"].get("line"), "timeline": "video time (seconds from start of source)"}


@app.get("/api/sessions/{sid}/counts")
def session_counts(sid: uuid.UUID, bucket_s: float = Query(10.0, gt=0)):
    with connect() as conn:
        s = _session(conn, sid)
        rows = conn.execute("SELECT ts, direction FROM customer_events "
                            "WHERE session_id = %s AND type = 'line_cross'", (sid,)).fetchall()
        return counts_series([(r["ts"], r["direction"]) for r in rows], bucket_s, _end_s(conn, s))


@app.get("/api/sessions/{sid}/occupancy")
def session_occupancy(sid: uuid.UUID, bucket_s: float = Query(5.0, gt=0)):
    with connect() as conn:
        s = _session(conn, sid)
        return occupancy_series(_visits(conn, s), _zones(s), bucket_s, _end_s(conn, s))


@app.get("/api/sessions/{sid}/transitions")
def session_transitions(sid: uuid.UUID):
    """Zone-to-zone movement of customers (staff and static tracks excluded)."""
    with connect() as conn:
        s = _session(conn, sid)
        return transition_table(_visits(conn, s), _zones(s))


@app.get("/api/sessions/{sid}/heatmap")
def session_heatmap(sid: uuid.UUID):
    with connect() as conn:
        s = _session(conn, sid)
        h = conn.execute("SELECT cell, floor, grid, background IS NOT NULL AS has_background "
                         "FROM heatmaps WHERE session_id = %s", (sid,)).fetchone()
        if h is None:
            raise HTTPException(404, "no heatmap yet (it is stored when the session finishes)")
        w, hgt = s["scene"].get("frame_size", [None, None])
        return {**h, "frame_size": [w, hgt], "unit": "person-seconds per cell"}


@app.get("/api/sessions/{sid}/background.png")
def session_background(sid: uuid.UUID):
    with connect() as conn:
        h = conn.execute("SELECT background FROM heatmaps WHERE session_id = %s", (sid,)).fetchone()
    if h is None or h["background"] is None:
        raise HTTPException(404, "no background image")
    return Response(bytes(h["background"]), media_type="image/png")


# ------------------------------------------------------------------ stores (run_store.py)
class StoreRunIn(BaseModel):
    run_info: dict = Field(default_factory=dict)


class StoreFinishIn(BaseModel):
    state: dict
    summary: dict


def _store_run(conn, rid: uuid.UUID) -> dict:
    row = conn.execute("SELECT * FROM store_runs WHERE run_id = %s", (rid,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown store run")
    return row


@app.put("/api/stores/{store_id}")
def put_store(store_id: str, config: dict):
    if config.get("store_id", store_id) != store_id:
        raise HTTPException(400, "store_id in body does not match the URL")
    with connect() as conn:
        conn.execute("""INSERT INTO stores (store_id, description, config) VALUES (%s, %s, %s)
                        ON CONFLICT (store_id) DO UPDATE
                        SET description = EXCLUDED.description, config = EXCLUDED.config, updated_at = now()""",
                     (store_id, config.get("description", ""), Jsonb(config)))
    return {"store_id": store_id}


@app.post("/api/stores/{store_id}/runs", status_code=201)
def start_store_run(store_id: str, body: StoreRunIn):
    rid = uuid.uuid4()
    with connect() as conn:
        if conn.execute("SELECT 1 FROM stores WHERE store_id = %s", (store_id,)).fetchone() is None:
            raise HTTPException(404, "unknown store: register it with PUT /api/stores/{id} first")
        conn.execute("INSERT INTO store_runs (run_id, store_id, run_info) VALUES (%s, %s, %s)",
                     (rid, store_id, Jsonb(body.run_info)))
    return {"run_id": str(rid)}


@app.put("/api/store-runs/{rid}/state")
def put_store_state(rid: uuid.UUID, state: dict):
    with connect() as conn:
        _store_run(conn, rid)
        conn.execute("UPDATE store_runs SET state = %s, updated_at = now() WHERE run_id = %s", (Jsonb(state), rid))
    return {"run_id": str(rid)}


@app.post("/api/store-runs/{rid}/finish")
def finish_store_run(rid: uuid.UUID, body: StoreFinishIn):
    with connect() as conn:
        _store_run(conn, rid)
        conn.execute("UPDATE store_runs SET state = %s, summary = %s, updated_at = now(), finished_at = now() "
                     "WHERE run_id = %s", (Jsonb(body.state), Jsonb(body.summary), rid))
    return {"run_id": str(rid)}


@app.get("/api/stores")
def list_stores():
    with connect() as conn:
        rows = conn.execute("""
            SELECT s.store_id, s.description, s.config, count(r.run_id) AS runs,
                   bool_or(r.run_id IS NOT NULL AND r.finished_at IS NULL) AS live
            FROM stores s LEFT JOIN store_runs r USING (store_id)
            GROUP BY s.store_id ORDER BY s.store_id""").fetchall()
    for r in rows:
        r["cameras"] = len(r.pop("config").get("cameras", []))
        r["live"] = bool(r["live"])
    return rows


@app.get("/api/stores/{store_id}")
def get_store(store_id: str):
    with connect() as conn:
        s = conn.execute("SELECT * FROM stores WHERE store_id = %s", (store_id,)).fetchone()
        if s is None:
            raise HTTPException(404, "unknown store")
        runs = conn.execute("""SELECT run_id, run_info, started_at, finished_at,
                                      (state->>'t')::float AS t, (state->>'unique_people')::int AS unique_people
                               FROM store_runs WHERE store_id = %s ORDER BY started_at DESC""",
                            (store_id,)).fetchall()
        floors = [r["kind"] for r in conn.execute("SELECT kind FROM store_floors WHERE store_id = %s",
                                                    (store_id,)).fetchall()]
    for r in runs:
        r["run_id"] = str(r["run_id"])
    return {"store_id": s["store_id"], "description": s["description"], "config": s["config"], "runs": runs,
            "floors": floors}


@app.get("/api/store-runs/{rid}")
def get_store_run(rid: uuid.UUID):
    with connect() as conn:
        r = _store_run(conn, rid)
    return {"run_id": str(rid), "store_id": r["store_id"], "run_info": r["run_info"],
            "finished": r["finished_at"] is not None, "state": r["state"] or {}, "summary": r["summary"],
            "timeline": "video time (seconds from start of the store's videos)"}


@app.delete("/api/stores/{store_id}")
def delete_store(store_id: str):
    """Remove a store and all its runs (cascade)."""
    with connect() as conn:
        n = conn.execute("DELETE FROM stores WHERE store_id = %s", (store_id,)).rowcount
    if not n:
        raise HTTPException(404, "unknown store")
    return {"deleted": store_id}


class FloorIn(BaseModel):
    content_type: str
    data_b64: str


@app.put("/api/stores/{store_id}/floor/{kind}")
def put_store_floor(store_id: str, kind: str, body: FloorIn):
    if kind not in ("blueprint", "photo") or body.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(422, "kind must be blueprint or photo; PNG or JPEG")
    with connect() as conn:
        if conn.execute("SELECT 1 FROM stores WHERE store_id = %s", (store_id,)).fetchone() is None:
            raise HTTPException(404, "unknown store")
        conn.execute("""INSERT INTO store_floors (store_id, kind, content_type, data) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (store_id, kind) DO UPDATE SET content_type = EXCLUDED.content_type,
                                                                   data = EXCLUDED.data""",
                     (store_id, kind, body.content_type, base64.b64decode(body.data_b64)))
    return {"store_id": store_id, "kind": kind}


@app.get("/api/stores/{store_id}/floor/{kind}")
def get_store_floor(store_id: str, kind: str):
    with connect() as conn:
        r = conn.execute("SELECT content_type, data FROM store_floors WHERE store_id = %s AND kind = %s",
                         (store_id, kind)).fetchone()
    if r is None:
        raise HTTPException(404, "no floor image")
    return Response(content=bytes(r["data"]), media_type=r["content_type"],
                    headers={"Cache-Control": "max-age=60"})
