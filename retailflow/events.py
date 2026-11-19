"""Analytics events: the anonymous records the pipeline emits.

Privacy by design: an event holds only a camera ID, a time, a track ID (an
integer that is meaningless outside one session), positions and zone names.
No images, no face crops, no appearance features.

Phase 2 writes events to CSV. Phase 5 adds a sender that POSTs the same
records to the FastAPI backend.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Event:
    camera_id: str
    frame: int
    timestamp: float          # seconds on the video timeline (or since stream start)
    type: str                 # line_cross | zone_enter | zone_exit | occupancy | staff_flag | static_flag
    track_id: int | None = None
    direction: str | None = None   # "in" / "out" for line_cross
    zone: str | None = None
    x: float | None = None    # anchor point (pixels) at the moment of the event
    y: float | None = None
    value: float | None = None     # e.g. occupancy count or dwell seconds
    staff: bool = False       # track already marked staff when the event was emitted

# A "staff_flag" event marks a track as staff from then on. Its EARLIER events
# were emitted with staff=False; consumers (analytics, backend) drop every event
# of a flagged track_id, so staff are excluded retroactively.
# A "static_flag" event (sent when a track ends) marks a static object such as
# a mannequin; its events are dropped the same way.


class CsvEventWriter:
    """Append events to a CSV file (header written once)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=[f.name for f in fields(Event)])
        self._w.writeheader()

    def write(self, event: Event) -> None:
        self._w.writerow(asdict(event))

    def close(self) -> None:
        self._f.close()
