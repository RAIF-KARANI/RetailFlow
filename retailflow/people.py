"""People in view now and total people, for one camera.

- people in view now: tracked people on screen, averaged over each second of video.
  Not hurt by ID switches (a new ID for the same person is still one box on screen).
- total people: different track IDs seen for at least MIN_PERSON_S seconds. A track
  shorter than that is usually a fragment of someone already counted (the tracker
  lost them behind a shelf or another person) or a false detection. The threshold was
  chosen on the dev videos (docs/10_RESULTS.md). It is still an over-count when one
  person gets several long IDs, so it is an estimate, not a head count.

Staff and static objects (mannequins) are left out of both, as soon as they are known:
staff when the staff rule flags them, static objects when their track ends, so a total
can go DOWN a little when a still object is recognised.

Only counts are kept: no positions, no images.
"""

from __future__ import annotations

MIN_PERSON_S = 3.0      # a track counts towards the total once it has lasted this long


class PeopleCounter:
    """Feed it every processed frame; it returns one sample per finished second."""

    def __init__(self, fps: float, min_s: float = MIN_PERSON_S):
        self.fps = fps
        self.min_frames = min_s * fps
        self.first: dict[int, int] = {}
        self.counted: set[int] = set()
        self._sec: int | None = None
        self._sum = 0.0
        self._n = 0

    def total(self, excluded: set[int] = frozenset()) -> int:
        return len(self.counted - set(excluded))

    def _sample(self, excluded) -> dict:
        return {"t": float(self._sec), "now": round(self._sum / max(self._n, 1), 2),
                "total": self.total(excluded)}

    def update(self, frame: int, track_ids, excluded: set[int] = frozenset()) -> list[dict]:
        """Add one frame. Returns the sample of the previous second when a new second starts."""
        out = []
        sec = int(frame // self.fps)
        if self._sec is not None and sec != self._sec:
            out.append(self._sample(excluded))
            self._sum, self._n = 0.0, 0
        self._sec = sec
        visible = 0
        for tid in track_ids:
            first = self.first.setdefault(tid, frame)
            if frame - first >= self.min_frames:
                self.counted.add(tid)
            if tid not in excluded:
                visible += 1
        self._sum += visible
        self._n += 1
        return out

    def finish(self, excluded: set[int] = frozenset()) -> list[dict]:
        """The last (possibly partial) second."""
        return [self._sample(excluded)] if self._sec is not None and self._n else []
