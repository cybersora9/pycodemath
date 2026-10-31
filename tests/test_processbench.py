"""The ProcessBench benchmark (NEXTGEN X0): smoke tests only.

The full run is ``python -m pycodemath.bench.processbench --full`` (downloads the
pinned data once). The gate checks the committed sample, the mapping to
``real_bench``'s record shape, ProcessBench's metric, and that UNDECIDED never
counts as a veto.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pycodemath.bench import processbench as pb
from pycodemath.bench import real_bench as rb


@pytest.fixture(scope="module")
def sample() -> "list[dict]":
    return pb.load()


def test_committed_sample_is_the_measured_one(sample: "list[dict]") -> None:
    assert rb.sha256_of(pb.DATA) == pb.SAMPLE_SHA256
    assert len(sample) == pb.SAMPLE_PER_SPLIT * len(pb.SPLITS) == 100
    assert len({r["id"] for r in sample}) == 100
    for split in pb.SPLITS:
        ids = [int(r["id"].rsplit("-", 1)[1]) for r in sample if r["split"] == split]
        assert len(ids) == pb.SAMPLE_PER_SPLIT and ids == sorted(ids)
    for r in sample:
        assert r["id"].startswith(r["split"] + "-")
        assert -1 <= r["label"] < len(r["steps"])
    notice = (Path(pb.DATA).parent / "PROCESSBENCH_LICENSE").read_text(encoding="utf-8")
    assert pb.COMMIT in notice and "Apache License" in notice


def test_draw_is_deterministic() -> None:
    rows = [{"id": f"x-{i}"} for i in range(200)]
    assert pb.draw(rows, 20, "s") == pb.draw(rows, 20, "s")
    assert pb.draw(rows, 20, "s") != pb.draw(rows, 20, "t")
    ids = [int(r["id"][2:]) for r in pb.draw(rows, 20, "s")]
    assert ids == sorted(ids)


def test_record_maps_the_label_to_real_bench_shape() -> None:
    wrong = pb.record({"id": "math-1", "split": "math", "problem": "p", "steps": ["a", "b", "c"], "label": 1})
    assert wrong["ratings"] == [1, -1] and wrong["first_error"] == 2 and wrong["subject"] == "math"
    right = pb.record({"id": "math-2", "split": "math", "problem": "p", "steps": ["a", "b"], "label": -1})
    assert right["ratings"] == [1, 1] and right["first_error"] is None
    first = pb.record({"id": "math-3", "split": "math", "problem": "p", "steps": ["a"], "label": 0})
    assert first["ratings"] == [-1] and first["first_error"] == 1


def _solution(statuses: "list[str]", label: int, subject: str = "math") -> rb.SolutionResult:
    steps = [rb.StepResult(i + 1, 1, st, [], {}, True, True) for i, st in enumerate(statuses)]
    return rb.SolutionResult("x", subject, 0, None if label == -1 else label + 1, steps, 0.0)


def test_metric_follows_processbench() -> None:
    exact = _solution(["VERIFIED", "REFUTED"], label=1)
    early = _solution(["REFUTED", "REFUTED"], label=1)
    undecided = _solution(["UNDECIDED", "UNDECIDED"], label=0)
    clean = _solution(["VERIFIED", "UNDECIDED"], label=-1)
    alarm = _solution(["REFUTED"], label=-1)
    assert [pb.predicted(r) for r in (exact, early, undecided, clean, alarm)] == [1, 0, -1, -1, 0]
    s = pb.pb_score([exact, early, undecided, clean, alarm])
    assert (s["hit_err"], s["err"], s["hit_ok"], s["ok"]) == (1, 3, 1, 2)
    assert s["f1"] == pytest.approx(2 * (100 / 3) * 50 / (100 / 3 + 50))
    assert pb.pb_score([undecided])["f1"] == 0.0   # UNDECIDED is never a veto


def test_smoke_run_on_committed_records(sample: "list[dict]") -> None:
    by_id = {r["id"]: r for r in sample}
    # sin(7pi/2)/2^6 = 1/2^6 in the labelled step 3 (sin(7pi/2) = -1)
    caught = rb.run_solution(pb.record(by_id["math-467"]))
    assert pb.label(caught) == 3 and pb.predicted(caught) == 3
    again = rb.run_solution(pb.record(by_id["math-467"]))
    assert [s.status for s in again.steps] == [s.status for s in caught.steps]
    table, summary = pb.pb_table([caught])
    assert summary["per_subset"]["math"]["hit_err"] == 1 and "math" in table
