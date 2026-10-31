"""VERIFY on ProcessBench (NEXTGEN X0) — the number the literature reports.

V7 (``real_bench``) measured the verifier on PRM800K. This module runs the SAME
frozen extraction and checking on ProcessBench, so the result sits in the table
every process-reward-model paper prints.

DATA. ProcessBench (Qwen team, arXiv 2412.06559, Apache-2.0 per the dataset card
``huggingface.co/datasets/Qwen/ProcessBench`` at commit ``3bdcd53``): 3400
model-written solutions to GSM8K, MATH, OlympiadBench and Omni-MATH problems,
each split into steps, with an expert label — the 0-based index of the earliest
wrong step, or -1 when every step is right. The four files are downloaded once,
checked against the sha256 below and cached (``PYCODEMATH_BENCH_CACHE``). A draw
of ``SAMPLE_PER_SPLIT`` solutions per subset (``random.Random(f"{SEED}/{split}")``
over file order, nothing filtered by how the verifier does) is committed next to
this module with the licence, for the offline smoke test.

EXTRACTION AND CHECKING are ``real_bench``'s. The first run (X0) used the rules
frozen on PRM800K; a manual review of every veto that disagreed with the label
found 24 extraction errors, and module X0b added rejecting rules for them — found
on this FULL test set, marked "ProcessBench, X0b" in ``real_bench`` and reported
frozen-vs-final in ``dev-journal/BENCH_PROCESSBENCH.md``. ``check_equal`` was never
touched. A record is mapped to ``real_bench``'s shape — steps
before the label rated +1, the label step -1, later steps unlabelled; a correct
solution has every step rated +1 — and run through ``real_bench.run_solution``.

SCORING. ProcessBench's own metric first: the prediction is the 0-based index of
the first REFUTED step, else -1 (UNDECIDED never vetoes). ``acc_err`` = share of
erroneous solutions whose predicted index equals the label, ``acc_ok`` = share of
correct solutions predicted -1, ``F1`` = their harmonic mean, per subset; the
paper's summary is the plain mean of the four F1. Then ``real_bench``'s report
(coverage, exact/early/late/missed, why the error step is out of reach).

    python -m pycodemath.bench.processbench             # committed sample (offline)
    python -m pycodemath.bench.processbench --full      # all 3400, downloads ~8 MB once
    python -m pycodemath.bench.processbench --fetch     # download + checksums only
    python -m pycodemath.bench.processbench --resample  # rebuild the sample file
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pycodemath.bench import real_bench as rb

SEED = 20260930
SAMPLE_PER_SPLIT = 25
BUDGET = rb.BUDGET

COMMIT = "3bdcd5371ed567559a78f559c01c13a6deee7604"
URL = "https://huggingface.co/datasets/Qwen/ProcessBench/resolve/{commit}/{split}.json"

#: split -> (sha256, size in bytes) of the pinned commit; a download that does not
#: hash to this is refused.
SOURCES = {
    "gsm8k": ("fb1c59dfd1e83e1c5a48dbe598db27e4bc0b8bc524a09445f2d4b69178f61349", 574_343),
    "math": ("60cb0d7a69cbec98d43cf50855b9b7e7294599c3c9a24bdd031abd18c3b6b136", 1_938_778),
    "olympiadbench": ("e9d801c9f3c03f949adbfa08e90cb21ad85c0b9224bc5e98326f8215c218aa55", 2_808_955),
    "omnimath": ("e8d7cf4512c82168dae2fe48b1718069a3002e7910af1d180a27ee0d9948ba47", 2_842_872),
}
SPLITS = tuple(SOURCES)

DATA = Path(__file__).parent / "data" / "processbench_sample.jsonl"
#: sha256 of the committed sample — ``--resample`` must reproduce it byte for byte.
SAMPLE_SHA256 = "2ebcf35520667d38aa210d0d10f472756fc0eb0a8c7b6ca186fcf52fce1aa40d"
CACHE = Path(os.environ.get("PYCODEMATH_BENCH_CACHE", Path.home() / ".cache" / "pycodemath-bench"))


# ============================================================================
# Data
# ============================================================================
def fetch(split: str, cache: Path = CACHE) -> Path:
    """The checksummed local copy of one subset, downloaded on first use."""
    sha, _ = SOURCES[split]
    dest = cache / "processbench" / f"{split}.json"
    if dest.exists() and rb.sha256_of(dest) == sha:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(URL.format(commit=COMMIT, split=split)) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = rb.sha256_of(tmp)
    if got != sha:
        tmp.unlink()
        raise RuntimeError(f"{split}.json: sha256 {got} != expected {sha}")
    tmp.replace(dest)
    return dest


def rows(split: str, cache: Path = CACHE) -> "list[dict]":
    """Every row of one subset in file order, tagged with its subset."""
    data = json.loads(fetch(split, cache).read_text(encoding="utf-8"))
    return [dict(row, split=split) for row in data]


def draw(split_rows: "list[dict]", n: int, seed: str) -> "list[dict]":
    """``n`` rows by ``random.Random(seed).sample`` over file order, back in file order."""
    idx = sorted(random.Random(seed).sample(range(len(split_rows)), min(n, len(split_rows))))
    return [split_rows[i] for i in idx]


def sample_rows(cache: Path = CACHE) -> "list[dict]":
    return [r for split in SPLITS for r in draw(rows(split, cache), SAMPLE_PER_SPLIT, f"{SEED}/{split}")]


def dump(data: "list[dict]") -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in data)


def load(path: Path = DATA) -> "list[dict]":
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record(row: dict) -> dict:
    """A ProcessBench row in ``real_bench``'s record shape (1-based ``first_error``)."""
    steps = row["steps"]
    label = row["label"]
    if label == -1:
        ratings, first_error = [1] * len(steps), None
    else:
        ratings, first_error = [1] * label + [-1], label + 1
    return {
        "id": row["id"],
        "subject": row["split"],
        "level": 0,
        "problem": row["problem"],
        "steps": steps,
        "ratings": ratings,
        "first_error": first_error,
    }


# ============================================================================
# Scoring
# ============================================================================
def predicted(r: rb.SolutionResult) -> int:
    """ProcessBench's prediction: 0-based index of the first REFUTED step, else -1."""
    at = r.flagged_at
    return -1 if at is None else at - 1


def label(r: rb.SolutionResult) -> int:
    return -1 if r.first_error is None else r.first_error - 1


def pb_score(results: "list[rb.SolutionResult]") -> dict:
    err = [r for r in results if label(r) != -1]
    ok = [r for r in results if label(r) == -1]
    hit_err = sum(predicted(r) == label(r) for r in err)
    hit_ok = sum(predicted(r) == -1 for r in ok)
    acc_err = 100 * hit_err / len(err) if err else 0.0
    acc_ok = 100 * hit_ok / len(ok) if ok else 0.0
    f1 = 2 * acc_err * acc_ok / (acc_err + acc_ok) if acc_err + acc_ok else 0.0
    return {"n": len(results), "err": len(err), "ok": len(ok), "hit_err": hit_err, "hit_ok": hit_ok,
            "acc_err": acc_err, "acc_ok": acc_ok, "f1": f1}


def pb_table(results: "list[rb.SolutionResult]") -> "tuple[str, dict]":
    per = {s: pb_score([r for r in results if r.subject == s]) for s in SPLITS}
    per = {s: v for s, v in per.items() if v["n"]}
    lines = [
        "PROCESSBENCH METRIC (prediction = first REFUTED step, else -1; UNDECIDED never vetoes)",
        f"  {'subset':<14}{'n':>6}{'acc_err':>22}{'acc_ok':>22}{'F1':>7}",
    ]
    for s, v in per.items():
        lines.append(f"  {s:<14}{v['n']:>6}"
                     f"{v['acc_err']:>8.1f}% ({v['hit_err']:>4}/{v['err']:<4})"
                     f"{v['acc_ok']:>8.1f}% ({v['hit_ok']:>4}/{v['ok']:<4}){v['f1']:>7.1f}")
    avg = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    lines.append(f"  average F1 over {len(per)} subsets: {avg:.1f}")
    return "\n".join(lines), {"per_subset": per, "avg_f1": avg}


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pycodemath.bench.processbench")
    ap.add_argument("--full", action="store_true", help="all 3400 solutions (downloads once)")
    ap.add_argument("--fetch", action="store_true", help="download the four subsets and verify checksums")
    ap.add_argument("--resample", action="store_true", help="rebuild the committed sample from the download")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--budget", type=float, default=BUDGET, help="seconds per claim")
    ap.add_argument("--by", choices=("subset", "generator"), default=None,
                    help="also print real_bench's report per subset / per generating model")
    ap.add_argument("--json", type=Path, default=None, help="write every claim verdict here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every refuted step")
    args = ap.parse_args(argv)

    if args.fetch:
        for split in SPLITS:
            print(f"{split}: {fetch(split, args.cache)} sha256 ok")
        return 0
    if args.resample:
        DATA.write_text(dump(sample_rows(args.cache)), encoding="utf-8", newline="\n")
        print(f"{DATA}: sha256 {rb.sha256_of(DATA)}")
        return 0
    data = [r for s in SPLITS for r in rows(s, args.cache)] if args.full else load()
    generator = {r["id"]: r.get("generator", "?") for r in data}

    results = []
    for row in data:
        r = rb.run_solution(record(row), args.budget)
        results.append(r)
        if args.verbose:
            for st in r.steps:
                if st.status == "REFUTED":
                    c = next(c for c in st.claims if c.status == "REFUTED")
                    print(f"{r.id} step {st.index - 1} (label {label(r)}): {c.left} = {c.right} -> {c.detail}")
    table, _ = pb_table(results)
    print(table)
    print()
    print(rb.report(rb.score(results), "full" if args.full else "sample"))
    if args.by:
        groups: dict[str, list[rb.SolutionResult]] = {}
        for r in results:
            key = r.subject if args.by == "subset" else generator[r.id]
            groups.setdefault(key, []).append(r)
        for key in sorted(groups):
            print()
            print(rb.report(rb.score(groups[key]), f"{args.by} = {key}"))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
