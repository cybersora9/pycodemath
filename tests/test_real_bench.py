"""The real-error benchmark (V7): smoke tests only.

The full run is a separate command (``python -m pycodemath.bench.real_bench``,
seconds on the committed sample; ``--population full`` needs the download). The
gate checks that the rule-based extractor reads a few hand-checked PRM800K steps
the way a person does, that everything is deterministic (extraction, the draw,
the verdicts), that the committed sample is the one the report measured, and that
UNDECIDED never scores as a catch.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pycodemath.bench import real_bench as rb


def _pairs(text: str) -> "list[tuple[str, str]]":
    return [(c.left.replace(" ", ""), c.right.replace(" ", "")) for c in rb.extract(text).claims]


# --- extraction: hand-checked steps ----------------------------------------
# Each step is verbatim PRM800K text (or a minimal variant); the expected claims
# were written by reading the step, not by running the extractor.
@pytest.mark.parametrize(
    ("step", "claims"),
    [
        (r"I know that $200,\!000 = 2^5\cdot 10^4 = 2^9\cdot 5^4$.",
         [("200000", "2^5*10^4"), ("2^5*10^4", "2^9*5^4")]),
        (r"Putting it all together, I get $64\cdot 4-96-16-1=143$.", [("64*4-96-16-1", "143")]),
        (r"So 48/2 = 24 cookies.", [("48/2", "24")]),
        (r"I know that $\log_2(1024) = 10$, since $2^{10} = 1024$.",
         [("log((1024),(2))", "10"), ("2^(10)", "1024")]),
        (r"Therefore, $\sin(-60^\circ) = -\frac{1}{2}$.", [("sin((-60*pi/180))", "-((1)/(2))")]),
        (r"$\binom{5}{2} = {5 \choose 2} = 10$", [("binomial((5),(2))", "binomial((5),(2))"),
                                                  ("binomial((5),(2))", "10")]),
    ],
)
def test_extractor_reads_hand_checked_steps(step: str, claims: "list[tuple[str, str]]") -> None:
    assert _pairs(step) == claims


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        (r"$A = 1$, $E = 5$, and $F = 6$.", "variables"),       # a face named E, not e
        (r"To find $r$, I use the given information that $p(0) = -52$.", "variables"),  # p*0 = 0
        (r"For example, $17\%5=2$.", "modular"),                  # % as the modulo operator
        (r"Simplifying, I get x^2 + 2x - 24 = 0.", "prose-mixed"),  # not "-24 = 0"
        (r"I get $5 = 3 + 4$, which is false.", "stated-false"),
        (r"223 / 3 = 74 with remainder 1", "integer-division"),
        (r"In base 5, $4 + 3 = 12$.", "number-base"),
        (r"So $2\frac{1}{2} = 2.5$.", "mixed-number"),
        (r"So $\log 100 = 2$.", "log-base"),                     # a bare \log has no agreed base
        (r"If $b = 0,$ then the equation becomes $0 = -16,$", "bare-literals"),
        (r"This gives me $\angle ACB = 180 - 54 - 52 = 74^\circ$.", "degrees-one-side"),
    ],
)
def test_extractor_drops_what_it_cannot_read(step: str, reason: str) -> None:
    ex = rb.extract(step)
    assert ex.claims == []
    assert ex.dropped[reason] >= 1


# --- X0b: extraction errors found on ProcessBench -------------------------------
# Verbatim steps (ProcessBench IDs in the comments; dev-journal/BENCH_PROCESSBENCH.md
# §3). Each was a false veto of the frozen rules; each is now read correctly or
# dropped with a named reason, never guessed.
@pytest.mark.parametrize(
    ("step", "claims"),
    [
        # gsm8k-83: "$" is money — two dollar signs are not one LaTeX segment
        ("Her current annual salary is $100,000. Savings = 20% of $100,000 = 0.20 * $100,000 = $20,000.", []),
        # gsm8k-162
        ("The total cost of the pot and pan is $120.00, and the discount is 20%, so the discount amount "
         "is 20/100 * $120 = $24. The discounted price of the pot and pan is $120 - $24 = $96.",
         [("20/100*120", "24"), ("120-24", "96")]),
        # gsm8k-136: "25\% \text{ of } \$2,000.00": the number after "of" is an operand
        ("Calculate 25% of $2,000.00:\n\\[ 25\\% \\text{ of } \\$2,000.00 = 0.25 \\times 2000 = \\$500.00 \\]",
         [("0.25*2000", "500.00")]),
        # gsm8k-193: "\text{Naomi's Points} + 17" leaves "+ 17"
        (r"\[ \text{Brianna's Points} = \text{Naomi's Points} + 17 = 68 + 17 = 85 \]", [("68+17", "85")]),
        # gsm8k-275: "30. 30 - 24" — a period and a space end a sentence
        ("so we need to subtract 24 from 30. 30 - 24 = 6. Now, Patricia has 6 roses left.", [("30-24", "6")]),
        # omnimath-440: four thousands groups
        (r"Notice that: \[7,999,999,999 = 8,000,000,000 - 1.\]", [("7999999999", "8000000000-1")]),
    ],
)
def test_x0b_reads_processbench_steps(step: str, claims: "list[tuple[str, str]]") -> None:
    assert _pairs(step) == claims


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        # olympiadbench-987: "$10\%$ of" before the segment
        (r"- Small cookie price increase: $10\%$ of $\$1.50 = 0.10 \times \$1.50 = \$0.15$", "fragment"),
        # omnimath-719: "× \(" before the segment
        (r"Expected gain = (Expected number of coins after Round 1) × \( \frac{1}{100} = \frac{101}{100} "
         r"\times \frac{1}{100} = \frac{101}{10000} \).", "fragment"),
        # olympiadbench-356: a unit cut out of a difference
        (r"Bottle B: $10 \mathrm{~g} - 0.2(30) = 10 \mathrm{~g} - 6 \mathrm{~g} = 4 \mathrm{~g}$ of acid left.",
         "units"),
        # gsm8k-292
        (r"\[ \text{Alyana's Age} = \text{Chenny's Age} - 4 = 10 - 4 = 6 \]", "text"),
        # olympiadbench-385: "2^4^\circ" is not LaTeX
        (r"\[ \frac{1}{\sin 2^4^\circ} = \frac{1}{2 \sin 2^3^\circ \cos 2^3^\circ} \]", "double-superscript"),
        # math-929: the sentence ends at "3." — a lone digit before the run may still be an operand
        ("so we move on to the next prime number, which is 3. 4641 ÷ 3 = 1547.", "prose-mixed"),
        # olympiadbench-993
        ("Ratio = Area of PQR / Area of STUVWXYZ = 4√3 / 6√3 = 4 / 6 = 2 / 3.", "prose-mixed"),
        # math-372: division with a remainder
        (r"\[2250 \div 360 = 6 \text{ full rotations with } 90 \text{ degrees remaining}\]", "integer-division"),
        # math-883: a rounded input, an exact result
        (r"\[ x^2 = (2.8284)^2 = 8 \]", "approximate-input"),
        # omnimath-305, omnimath-787, olympiadbench-610: a trial the step rejects
        ("- Check: \\( 60 = 5^2 + 6^2 = 25 + 36 = 61 \\)\n- This does not work.", "rejected-trial"),
        (r"\[1 = \frac{1}{3} + \frac{1}{5} + \frac{1}{7}\] This still does not equate to 1 directly.",
         "rejected-trial"),
        ("Substituting the values, we get 0 + (15 - 1) * 71 = 999, which simplifies to 14 * 71 = 994. "
         "Since 994 is less than 999, our first term 'a' should be 5, not 0.", "rejected-trial"),
    ],
)
def test_x0b_drops_what_it_cannot_read(step: str, reason: str) -> None:
    ex = rb.extract(step)
    assert all(check.status != "REFUTED" for check in map(rb.check_claim, ex.claims))
    assert ex.dropped[reason] >= 1


def test_x0b_keeps_real_errors() -> None:
    # math-82: ceil(9.0909) = 9 is wrong by far more than the rounding of 9.0909
    ex = rb.extract(r"\[ \frac{100}{11} \approx 9.0909 \quad \Rightarrow \quad \lceil 9.0909 \rceil = 9 \]")
    assert [rb.check_claim(c).status for c in ex.claims] == ["REFUTED"]
    # math-542: "does not satisfy" rejects the candidate k, not the product 77 * 68 = 5236
    ex = rb.extract(r"First, for \(k = 68\): \[x = 5050 - 77 \times 68 = 5050 - 5256 = -206\] "
                    r"This does not satisfy our condition since \(x\) must be positive.")
    assert "REFUTED" in [rb.check_claim(c).status for c in ex.claims]
    # omnimath-612: 11.5098 * 720 = 8287.056 — a decimal result is not an approximation
    ex = rb.extract(r"\[ 11.5098 \cdot 720 = 8286.656 \]")
    assert [rb.check_claim(c).status for c in ex.claims] == ["REFUTED"]


def test_x0b_continuation_needs_an_expression_right_before() -> None:
    # math-170 step 3: words between two segments of a step — no continuation
    first, second = rb.extract_solution([
        r"we can use the complementary angle identity: $\sin 18^\circ = \cos 72^\circ$.",
        r"Substitute $\cos 72^\circ$ for $\sin 18^\circ$ in the expression: "
        r"$= (\cos 72^\circ)^2 \cdot \cos 72^\circ + (\cos 72^\circ)^2 = \cos^3 72^\circ + \cos^2 72^\circ$.",
    ])
    assert all("sin" not in c.left for c in second.claims)
    # math-170 step 8: the step before ENDED with an expression — it continues it
    _, after = rb.extract_solution([r"So: $= \frac{1}{2} + \frac{1}{4}$.", r"Combine like terms: $= \frac{3}{4}$."])
    assert [(c.left.replace(" ", ""), c.right) for c in after.claims] == [("((1)/(2))+((1)/(4))", "((3)/(4))")]
    # omnimath-539: the step before ended with prose ("\times$ Width.")
    _, after = rb.extract_solution([r"$\frac{1}{3} = \frac{3}{5} \times$ Width.",
                                    r"Width $= \frac{1}{3} \div \frac{3}{5}$."])
    assert after.claims == []


def test_claims_are_checked_by_the_verifier() -> None:
    wrong, true = rb.extract(r"$200,\!000 = 2^5\cdot 10^4 = 2^9\cdot 5^4$").claims
    assert rb.check_claim(wrong).status == "REFUTED"      # 2^5 * 10^4 = 320000
    assert rb.check_claim(true).status == "VERIFIED"
    rounded, = rb.extract("which is 900/1260 = 0.7143.").claims
    assert (rounded.kind, rounded.digits) == ("rounded", 4)
    assert rb.check_claim(rounded).status == "VERIFIED"   # 0.714285... rounds to 0.7143
    off, = rb.extract("so 1/3 = 0.34").claims
    assert rb.check_claim(off).status == "REFUTED"        # neither rounds nor truncates


def test_a_step_opening_with_equals_continues_the_one_before() -> None:
    first, second = rb.extract_solution([r"$a = (1+2)^2$", r"$= 9$"])
    assert first.claims == [] and first.dropped["variables"] == 1
    assert [(c.left, c.right) for c in second.claims] == [("(1+2)^2", "9")]
    # an align line opening with \cdot continues the product, it is not a new side
    ex = rb.extract(r"\[\begin{aligned} 2 \cdot 3 &= (2) \\ &\cdot (3) \end{aligned}\]")
    assert [(c.left.replace(" ", ""), c.right.replace(" ", "")) for c in ex.claims] == [("2*3", "(2)*(3)")]


# --- determinism -------------------------------------------------------------
def test_extraction_is_deterministic(sample: "list[dict]") -> None:
    steps = [s for r in sample[:40] for s in r["steps"]]
    once = [(e.claims, dict(e.dropped)) for e in rb.extract_solution(steps)]
    assert once == [(e.claims, dict(e.dropped)) for e in rb.extract_solution(steps)]


def _row(problem: str, steps: "list[str]", ratings: "list[int]", reason: str) -> dict:
    return {
        "question": {"problem": problem, "pre_generated_steps": steps,
                     "pre_generated_answer": "1", "ground_truth_answer": "1"},
        "label": {"finish_reason": reason,
                  "steps": [{"completions": [{"text": t, "rating": r}]} for t, r in zip(steps, ratings)]},
    }


def test_population_and_draw_are_deterministic() -> None:
    rows = [
        (1, _row("p1", ["a", "b", "c"], [1, -1], "found_error")),
        (2, _row("p2", ["a", "b"], [1, 1], "solution")),
        (3, _row("p3", ["a"], [1], "give_up")),
        (4, _row("p1", ["a", "b", "c"], [1, -1], "found_error")),   # duplicate, same label
        (5, _row("p4", ["x", "y"], [1, 1], "solution")),
        (6, _row("p4", ["x", "y"], [-1], "found_error")),            # duplicate, other label
        (7, _row("p5", ["x", "y"], [1, 0], "solution")),
    ]
    pop, out = rb.population(rows, {"p1": {"subject": "Algebra", "level": 2}}, "t")
    assert [r["id"] for r in pop] == ["t:1", "t:2", "t:7"]
    assert pop[0]["first_error"] == 2 and pop[0]["subject"] == "Algebra"
    assert pop[1]["first_error"] is None
    assert dict(out) == {"give_up": 1, "duplicate": 2, "conflicting-duplicate": 1}
    big = [dict(pop[0], id=f"t:{n}") for n in range(1, 200)]
    assert rb.draw(big, 20, 7) == rb.draw(big, 20, 7)
    assert rb.draw(big, 20, 7) != rb.draw(big, 20, 8)
    assert rb.draw(list(reversed(big)), 20, 7) == rb.draw(big, 20, 7)   # input order irrelevant
    def line(r: dict) -> int:
        return int(r["id"].split(":")[1])

    # the documented draw: random.Random(seed).sample over file order, back in file order
    assert rb.draw(big, 20, 7) == sorted(random.Random(7).sample(sorted(big, key=line), 20), key=line)


# --- the committed sample ------------------------------------------------------
@pytest.fixture(scope="module")
def sample() -> "list[dict]":
    return rb.load()


def test_committed_sample_is_the_measured_one(sample: "list[dict]") -> None:
    assert rb.sha256_of(rb.DATA) == rb.SAMPLE_SHA256
    assert len(sample) == rb.SAMPLE_SIZE == 500
    assert len({r["id"] for r in sample}) == 500
    lines = [int(r["id"].split(":")[1]) for r in sample]
    assert lines == sorted(lines)
    for r in sample:
        assert r["finish_reason"] in ("found_error", "solution")
        first = next((i + 1 for i, x in enumerate(r["ratings"]) if x == -1), None)
        assert r["first_error"] == first
        assert len(r["ratings"]) <= len(r["steps"])
    assert (Path(rb.DATA).parent / "PRM800K_LICENSE").read_text().startswith("MIT License")


def test_smoke_run_on_committed_records(sample: "list[dict]") -> None:
    by_id = {r["id"]: r for r in sample}
    # "$500=10^3=(2\cdot 5)^3$" in step 1, the labelled first error
    caught = rb.run_solution(by_id["phase2_test:626"])
    assert caught.first_error == 1 and caught.flagged_at == 1
    assert rb.outcome(caught) == "exact"
    again = rb.run_solution(by_id["phase2_test:626"])
    assert [s.status for s in again.steps] == [s.status for s in caught.steps]


# --- scoring -------------------------------------------------------------------
def _solution(statuses: "list[str]", first_error: "int | None") -> rb.SolutionResult:
    steps = [rb.StepResult(i + 1, 1, st, [], {}, True, True) for i, st in enumerate(statuses)]
    return rb.SolutionResult("x:1", "Algebra", 1, first_error, steps, 0.0)


def test_undecided_is_never_a_catch() -> None:
    r = _solution(["VERIFIED", "UNDECIDED", "not-covered"], first_error=2)
    assert rb.outcome(r) == "missed"
    assert rb.out_of_reach(r.steps[1]) == "checked-undecided"
    s = rb.score([r])
    assert s["outcomes"] == {"missed": 1} and s["flagged"] == 0
    assert rb.outcome(_solution(["REFUTED", "REFUTED"], first_error=2)) == "early"
    assert rb.outcome(_solution(["VERIFIED", "VERIFIED", "REFUTED"], first_error=2)) == "late"
    assert rb.outcome(_solution(["REFUTED"], first_error=None)) == "false-alarm"
