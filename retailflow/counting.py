"""Line-crossing counter: the core IN/OUT counting logic.

Geometry
--------
For a line from A to B and a point P, the 2-D cross product

        cross = (B - A) x (P - A) = (Bx-Ax)(Py-Ay) - (By-Ay)(Px-Ax)

is positive on one side, negative on the other and zero on the line. Dividing
by |B - A| turns it into a signed DISTANCE in pixels. We flip the sign so the
scene's `in_point` is always on the positive side, so "+" means IN side.

Counting rule (band / hysteresis)
---------------------------------
Each track remembers the last side it was CLEARLY on: distance > +band (IN
side) or < -band (OUT side). Points inside the band (|d| <= band) change
nothing. A crossing is counted only when the track goes from clearly one side
to clearly the other, so a person jittering on the line cannot generate counts.
With band = 0 this reduces to the classic "sign changed" line counter, which
is the BASELINE.

Extra reliability filters (both off by default = baseline behaviour):
  min_track_age      a track must have been seen for N frames before it can
                     count, which filters out short-lived false tracks.
  once_per_direction each track ID counts at most one IN and one OUT, which
                     filters out back-and-forth double counts.

The crossing must also happen WITHIN the segment (not on its infinite
extension), so people walking around the end of the line are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retailflow.config import LineConfig

IN, OUT = "in", "out"


@dataclass(frozen=True)
class CountingConfig:
    """Run-time counting options (camera geometry lives in LineConfig)."""

    point: str = "feet"            # "feet" (bottom-centre) or "centre" of the box
    use_band: bool = True          # False => band forced to 0 (simple line)
    min_track_age: int = 0         # frames a track must exist before counting
    once_per_direction: bool = False


def anchor_point(x1: float, y1: float, x2: float, y2: float, mode: str) -> tuple[float, float]:
    """The point of a box used for counting: feet (bottom-centre) or centre."""
    if mode == "feet":
        return ((x1 + x2) / 2, y2)
    if mode == "centre":
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    raise ValueError(f"unknown point mode {mode!r}")


def signed_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed perpendicular distance from p to the infinite line through a, b (pixels)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    cross = dx * (p[1] - a[1]) - dy * (p[0] - a[0])
    return cross / (dx * dx + dy * dy) ** 0.5


def crosses_segment(p: tuple[float, float], q: tuple[float, float],
                    a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if the movement p->q crosses the line within the segment a-b.

    We find where p->q meets the infinite line, then check that this meeting
    point projects inside [a, b] (parameter t in [0, 1]).
    """
    dp, dq = signed_distance(p, a, b), signed_distance(q, a, b)
    if dp == dq:  # moving parallel to the line
        return False
    s = dp / (dp - dq)                       # fraction along p->q where it meets the line
    mx, my = p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = ((mx - a[0]) * abx + (my - a[1]) * aby) / (abx * abx + aby * aby)
    return 0.0 <= t <= 1.0


@dataclass
class _TrackState:
    first_frame: int
    side: int = 0                          # +1 IN side, -1 OUT side, 0 unknown yet
    last_clear: tuple[float, float] | None = None  # last point clearly on `side`
    counted: set = field(default_factory=set)


class LineCounter:
    """Stateful counter. Feed it every tracked point in frame order."""

    def __init__(self, line: LineConfig, cfg: CountingConfig = CountingConfig()):
        self.a, self.b = line.p1, line.p2
        # Orient so in_point is on the positive side.
        self.sign = 1.0 if signed_distance(line.in_point, self.a, self.b) > 0 else -1.0
        self.band = line.band_px if cfg.use_band else 0.0
        self.cfg = cfg
        self.tracks: dict[int, _TrackState] = {}
        self.counts = {IN: 0, OUT: 0}

    def distance(self, p: tuple[float, float]) -> float:
        """Signed distance, positive on the IN side."""
        return self.sign * signed_distance(p, self.a, self.b)

    def update(self, track_id: int, frame: int, point: tuple[float, float]) -> str | None:
        """Process one observation. Returns "in", "out" or None."""
        st = self.tracks.get(track_id)
        if st is None:
            st = self.tracks[track_id] = _TrackState(first_frame=frame)

        d = self.distance(point)
        if abs(d) <= self.band:          # inside the band: undecided, keep old side
            return None
        side = 1 if d > 0 else -1
        prev_side, prev_point = st.side, st.last_clear
        st.side, st.last_clear = side, point
        if prev_side == 0 or side == prev_side:
            return None                  # first clear observation, or no change

        # Side flipped from clearly one side to clearly the other.
        if not crosses_segment(prev_point, point, self.a, self.b):
            return None                  # went around the end of the line
        if frame - st.first_frame + 1 < self.cfg.min_track_age:
            return None                  # track too young to trust
        direction = IN if side > 0 else OUT
        if self.cfg.once_per_direction and direction in st.counted:
            return None                  # this ID already counted this way
        st.counted.add(direction)
        self.counts[direction] += 1
        return direction


def count_tracks(tracks: pd.DataFrame, line: LineConfig, cfg: CountingConfig) -> pd.DataFrame:
    """Count crossings for a whole table of tracks (offline / evaluation use).

    `tracks` needs columns frame, track_id, x1, y1, x2, y2. Returns one row per
    crossing: frame, track_id, direction.
    """
    counter = LineCounter(line, cfg)
    events = []
    ordered = tracks.sort_values(["frame", "track_id"])
    for f, tid, x1, y1, x2, y2 in ordered[["frame", "track_id", "x1", "y1", "x2", "y2"]].itertuples(index=False):
        direction = counter.update(int(tid), int(f), anchor_point(x1, y1, x2, y2, cfg.point))
        if direction:
            events.append((int(f), int(tid), direction))
    return pd.DataFrame(events, columns=["frame", "track_id", "direction"])
