"""People in view now and total people (retailflow/people.py)."""

from retailflow.people import PeopleCounter

FPS = 10.0


def test_one_sample_per_second_with_mean_people_in_view():
    pc = PeopleCounter(FPS, min_s=3)
    out = []
    for f in range(20):                                   # 2 s: 2 people, then 1 person
        out += pc.update(f, [1, 2] if f < 10 else [1])
    out += pc.finish()
    assert [s["t"] for s in out] == [0.0, 1.0]
    assert [s["now"] for s in out] == [2.0, 1.0]


def test_total_counts_only_tracks_seen_long_enough():
    pc = PeopleCounter(FPS, min_s=3)
    for f in range(50):                                   # 5 s
        ids = [1]                                         # a shopper, whole time
        if 10 <= f < 25:
            ids.append(2)                                 # a 1.5 s fragment: not counted
        if f >= 15:
            ids.append(3)                                 # 3.5 s: counted
        pc.update(f, ids)
    assert pc.total() == 2


def test_staff_and_static_objects_are_left_out():
    pc = PeopleCounter(FPS, min_s=1)
    for f in range(30):
        out = pc.update(f, [1, 2, 3], excluded={3})       # 3 = staff
    assert out == [] or out[0]["now"] == 2.0
    last = pc.finish(excluded={3})[0]
    assert last["now"] == 2.0 and last["total"] == 2
    assert pc.total(excluded={2, 3}) == 1                 # 2 later found to be a mannequin
