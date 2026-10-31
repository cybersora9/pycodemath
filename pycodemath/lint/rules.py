"""The rule table of ``pycodemath lint`` — one entry per code, the single source of the docs.

Every rule has a stable code ``PCMnnn`` (family by hundreds, as ruff/flake8 do), a
kebab-case name, a level, and whether it is on by default. ``RULES.md`` in this
package is GENERATED from this table (``python -m pycodemath.lint --generate-docs``)
and ``tests/test_lint.py`` fails when the two drift apart — the docs cannot lie
about a rule the code changed.

Families::

    PCM1xx  precision   bits lost by a float formula (module C2, ``diagnose``)
    PCM2xx  ranges      overflow / lost integers / NaN for a declared dtype (C3)
    PCM3xx  spec        a function whose ``:math:`` / ``@spec`` formula it does not compute (C1)
    PCM4xx  money       binary floats and rounding on amounts (C4)
    PCM5xx  static      ``==`` on floats, ``^`` as a power, banker's ``round``,
                        ``range`` off by one against the spec
    PCM9xx  meta        what was NOT analysed (budget, unsupported construct) — info, off

THE CONTRACT EVERY RULE OBEYS: a finding is emitted only with a witness — a concrete
input that was EXECUTED and whose result differs from the exact (or intended)
value computed by an independent route (``witness.Witness.replay``). A rule that
cannot produce one stays silent. The "level" says how bad the confirmed effect
is, never how sure we are: we are always sure, or we say nothing.

    >>> RULES["PCM502"].name
    'xor-as-power'
    >>> [r.code for r in RULES.values() if r.family == "spec"]
    ['PCM301', 'PCM302']
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Rule", "RULES", "LEVELS", "DOCS_URL", "rule_url", "default_codes", "rules_markdown"]

#: Levels, in SARIF's vocabulary (``note`` is what other tools call "info").
LEVELS: tuple[str, ...] = ("error", "warning", "note")

#: Where ``helpUri`` points: the generated rule reference in the repository.
DOCS_URL = "https://github.com/cybersora9/pycodemath/blob/main/pycodemath/lint/RULES.md"


@dataclass(frozen=True)
class Rule:
    """One lint rule: identity, level, documentation with a bad and a good example."""

    code: str
    name: str
    family: str
    level: str
    default: bool
    summary: str
    why: str
    witness: str
    bad: str
    good: str
    fix: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code, "name": self.name, "family": self.family, "level": self.level,
            "default": self.default, "summary": self.summary, "url": rule_url(self.code),
        }


def rule_url(code: str) -> str:
    """``helpUri`` of a rule: its anchor in ``RULES.md``."""
    return f"{DOCS_URL}#{code.lower()}"


_RULES: tuple[Rule, ...] = (
    Rule(
        "PCM101", "precision-loss", "precision", "warning", True,
        "a float formula loses at least `precision-bits` bits on its input range",
        "The function returns a float expression whose float64 (or float32) evaluation is "
        "far from the exact real value for some input in the declared range — catastrophic "
        "cancellation (`sqrt(x+1) - sqrt(x)`), `log(1 + x)` near 0, `1 - cos(x)`, an "
        "intermediate that underflows. Measured by module C2 against interval ground truth "
        "(bits of error as in Herbie). The range is the docstring's `:range x: [a, b]`, the "
        "`@spec(..., ranges=...)` argument, the config's `ranges`, or by default every value "
        "with `1e-6 <= |x| <= 1e6`.",
        "the worst input found, the value the code computes there, the exact value (20 "
        "digits), and the bits lost; re-executed with the code's own operations and compared "
        "with a SymPy evaluation at 60 digits",
        "def f(x: float) -> float:\n    \"\"\":range x: [1, 1e15]\"\"\"\n"
        "    return math.sqrt(x + 1) - math.sqrt(x)",
        "def f(x: float) -> float:\n    \"\"\":range x: [1, 1e15]\"\"\"\n"
        "    return 1 / (math.sqrt(x + 1) + math.sqrt(x))",
        "a rewrite from C2's `improve`, PROVED equal to the original (with its domain "
        "conditions) and measured more accurate on independent sample points",
    ),
    Rule(
        "PCM201", "overflow", "ranges", "error", True,
        "an operation overflows the declared dtype (wraps around or becomes inf)",
        "Parameters annotated with a NumPy type (`np.int32`, `np.float16`, ...) are analysed "
        "by module C3 in that type, exactly as NumPy 2 array code evaluates it. An integer "
        "result outside the type wraps around silently; a float one becomes `inf`. The range "
        "is the docstring's `:range`; without one, the whole type for integer types and float16 "
        "(float32/float64 need a declared range: on the whole type every `+` overflows at the "
        "largest float, which is true and useless).",
        "the input, the value computed in the dtype (executed by NumPy), and the exact value",
        "def mid(lo: np.int32, hi: np.int32) -> np.int32:\n"
        "    \"\"\":range lo: [0, 2147483647]\n    :range hi: [0, 2147483647]\"\"\"\n"
        "    return (lo + hi) // 2",
        "def mid(lo: np.int32, hi: np.int32) -> np.int32:\n"
        "    \"\"\":range lo: [0, 2147483647]\n    :range hi: [0, 2147483647]\"\"\"\n"
        "    return lo + (hi - lo) // 2",
        "a rewrite from C3's advice, proved equivalent and re-analysed free of the event",
    ),
    Rule(
        "PCM202", "integer-precision-loss", "ranges", "error", True,
        "an integer-valued float is not representable in the dtype (a timestamp + 1 == itself)",
        "A float holding integers (an `:range t: [a, b, int]` input, integer literals and "
        "`+ - *` of those) loses integers above 2**24 in float32 (2**53 in float64): "
        "`t + 1` returns `t`. Module C3.",
        "the input, the float result, and the exact integer",
        "def next_ms(t: np.float32) -> np.float32:\n"
        "    \"\"\":range t: [1.6e12, 1.8e12, int]\"\"\"\n    return t + 1",
        "def next_ms(t: np.float64) -> np.float64:\n"
        "    \"\"\":range t: [1.6e12, 1.8e12, int]\"\"\"\n    return t + 1",
    ),
    Rule(
        "PCM203", "nan-created", "ranges", "warning", False,
        "an operation creates NaN from non-NaN operands (0/0, inf - inf, sqrt of a negative)",
        "Module C3, event `nan_possible` with an executed witness. Off by default: on a "
        "whole-type range almost every division can be fed 0/0.",
        "the input and the NaN observed",
        "def ratio(a: np.float32, b: np.float32) -> np.float32:\n"
        "    \"\"\":range a: [-1, 1]\n    :range b: [-1, 1]\"\"\"\n    return a / b",
        "guard the denominator, or declare the range",
    ),
    Rule(
        "PCM204", "division-by-zero", "ranges", "note", False,
        "a division (or log) can be applied to zero on the declared range",
        "Module C3, event `division_by_zero_possible`. Off by default for the same reason as "
        "PCM203.",
        "the input that makes the divisor zero, and the result",
        "def inv(x: np.float32) -> np.float32:\n    \"\"\":range x: [-1, 1]\"\"\"\n    return 1 / x",
        "declare a range that excludes 0",
    ),
    Rule(
        "PCM205", "underflow-to-zero", "ranges", "note", False,
        "a float result becomes 0 although the exact result is not",
        "Module C3, event `underflow_to_zero`. Off by default: every product of two tiny "
        "floats underflows somewhere on a whole-type range.",
        "the input, the 0 observed, and the exact (nonzero) value",
        "def sq(x: np.float16) -> np.float16:\n    return x * x",
        "declare the range, or work in a wider type",
    ),
    Rule(
        "PCM301", "spec-mismatch", "spec", "error", True,
        "the function does not compute the formula in its `:math:` docstring line or `@spec`",
        "The function is read into a symbolic form by module C1 and compared with the "
        "formula on the declared domain (`:domain n: integer, n>=0` or `@spec(..., "
        "domain=...)`). A counterexample is EXECUTED (in C1's isolated worker) and the "
        "formula evaluated exactly; both must disagree.",
        "the call (`sum_to(n=1)`), what it returned, and the formula's exact value",
        "def sum_to(n: int) -> int:\n    \"\"\":math: n*(n+1)/2\"\"\"\n"
        "    return n * (n - 1) // 2",
        "def sum_to(n: int) -> int:\n    \"\"\":math: n*(n+1)/2\"\"\"\n    return n * (n + 1) // 2",
    ),
    Rule(
        "PCM302", "spec-undecided", "spec", "note", False,
        "a spec could not be checked (unsupported construct, no proof, execution off, budget)",
        "Module C1 answered UNDECIDED and searched for a counterexample without finding one. "
        "Not a problem in the code — a statement about the reach of the checker, so it is "
        "a note, off by default (`--select PCM302` shows them).",
        "none: this is the absence of a verdict, and says why",
        "def f(n: int) -> int:\n    \"\"\":math: n\"\"\"\n    while n > 0: ...",
        "",
    ),
    Rule(
        "PCM401", "float-money", "money", "error", True,
        "an amount computed in binary floats rounds (or truncates) to a different amount",
        "`round(E, 2)` or `int(E * 100)` where `E` involves an amount (a name matching "
        "`money-names`: price, amount, total, cost, vat, tax) and binary floats. Module C4 "
        "runs the expression on floats, exactly as the code does, over a grid of amounts, "
        "and evaluates it exactly with every literal and amount read as the decimal written.",
        "the amounts, the float result and the exact one (`price=0.50: 0.61 vs 0.62`)",
        "def gross(price: float) -> float:\n    return round(price * 1.23, 2)",
        "def gross(price: Decimal) -> Decimal:\n"
        "    return (price * Decimal(\"1.23\")).quantize(Decimal(\"0.01\"), ROUND_HALF_UP)",
    ),
    Rule(
        "PCM402", "double-rounding", "money", "warning", True,
        "rounding twice (`round(round(x, 3), 2)`) gives a different result than rounding once",
        "Rounding an intermediate to more digits and then to fewer is not the same as rounding "
        "once: 0.1349 -> 0.135 -> 0.14, but 0.1349 -> 0.13. Both the float execution and the "
        "exact decimal computation must show the difference.",
        "the value, the double-rounded result, and the singly rounded exact result",
        "total = round(round(x, 3), 2)",
        "total = round(x, 2)",
    ),
    Rule(
        "PCM403", "decimal-from-float", "money", "warning", True,
        "`Decimal(<float literal>)` holds the binary float, not the decimal written",
        "`Decimal(0.1)` is 0.1000000000000000055511151231257827021181583404541015625. "
        "Reported only for literals a binary float cannot hold exactly.",
        "the literal and the exact value the Decimal really holds",
        "rate = Decimal(0.23)",
        "rate = Decimal(\"0.23\")",
        "the same literal as a string; the result is the decimal written",
    ),
    Rule(
        "PCM501", "float-equality", "static", "warning", True,
        "`==` / `!=` on float arithmetic gives the opposite answer from exact arithmetic",
        "A comparison of float expressions (float literals, `/`, names annotated or "
        "assigned as float) with at least one arithmetic operation. Reported only when "
        "values are found for which the float comparison and the exact decimal comparison "
        "disagree; `x == 0.0` or `a == b` alone are never reported.",
        "values for every name (as written in decimal), the float result, the exact result",
        "def is_third(a: float, b: float, c: float) -> bool:\n    return a + b == c",
        "def is_third(a: float, b: float, c: float) -> bool:\n"
        "    return math.isclose(a + b, c)",
        "a suggestion, not an equivalence: `math.isclose` with a tolerance you choose, or "
        "`Decimal`/`Fraction` when the values are exact decimals",
    ),
    Rule(
        "PCM502", "xor-as-power", "static", "error", True,
        "`^` in arithmetic is bitwise XOR in Python, not a power",
        "`x^2` in an arithmetic context (an operand of `+ - * /`, an argument of a math "
        "function, a literal power such as `10^-6`). Not reported in code that also uses "
        "`& | << >> ~` (bit manipulation).",
        "values, the XOR result and the power (`x=3: x ^ 2 = 1, x ** 2 = 9`)",
        "def norm(a: float, b: float) -> float:\n    return math.sqrt(a^2 + b^2)",
        "def norm(a: float, b: float) -> float:\n    return math.sqrt(a**2 + b**2)",
        "`**` — the power the arithmetic context asks for",
    ),
    Rule(
        "PCM503", "bankers-rounding", "static", "warning", True,
        "`round()` rounds half to even where the context (an amount) suggests HALF_UP",
        "Python's `round` is banker's rounding: `round(2.5) == 2`, `round(0.125, 2) == 0.12`. "
        "Invoices and tax law in PL/EU prescribe half up (see module C4). Reported only for "
        "an amount where a reachable tie exists, and only where the float result equals the "
        "exact half-even one (the difference is the mode, not the float).",
        "the amounts, Python's result and the HALF_UP result (computed by C4 `round_money`)",
        "def to_pln(amount: float) -> int:\n    return round(amount)",
        "def to_pln(amount: Decimal) -> Decimal:\n"
        "    return amount.quantize(Decimal(\"1\"), ROUND_HALF_UP)",
    ),
    Rule(
        "PCM504", "range-off-by-one", "static", "error", True,
        "a `range(...)` bound is off by one against the function's `:math:` / `@spec` formula",
        "The spec check (C1) refutes the function, and moving one bound of one `range` call "
        "by one makes C1 PROVE it equal to the formula. The fix is that change.",
        "the call where the function disagrees with the formula, as for PCM301",
        "def sum_to(n: int) -> int:\n    \"\"\":math: n*(n+1)/2\n    :domain n: integer, n>=0\"\"\"\n    t = 0\n"
        "    for i in range(1, n):\n        t += i\n    return t",
        "    for i in range(1, n + 1):",
        "the shifted bound, proved equal to the formula by C1",
    ),
    Rule(
        "PCM901", "skipped-budget", "meta", "note", False,
        "an analysis ran out of its per-function budget — nothing is claimed",
        "The per-function `budget` (seconds) ended before C1/C2/C3 finished. This is not a "
        "finding about the code. Off by default; `--select PCM9` lists what was skipped.",
        "none",
        "", "",
    ),
    Rule(
        "PCM902", "not-analysed", "meta", "note", False,
        "a numeric function uses a construct the analyses do not model — nothing is claimed",
        "Off by default. Lists functions that looked numeric but were not analysed, with "
        "the reason, so that coverage is visible rather than implied.",
        "none",
        "", "",
    ),
)

#: code -> Rule, in documentation order.
RULES: dict[str, Rule] = {r.code: r for r in _RULES}


def default_codes() -> frozenset[str]:
    """Codes that run when nothing is selected."""
    return frozenset(r.code for r in _RULES if r.default)


_FAMILY_TITLES = {
    "precision": "PCM1xx — precision (module C2)",
    "ranges": "PCM2xx — ranges for a declared dtype (module C3)",
    "spec": "PCM3xx — specification (module C1)",
    "money": "PCM4xx — money (module C4)",
    "static": "PCM5xx — static rules, each with an executed witness",
    "meta": "PCM9xx — what was not analysed",
}


def rules_markdown() -> str:
    """The rule reference as Markdown (written to ``RULES.md``)."""
    out = [
        "# pycodemath lint — rules",
        "",
        "<!-- generated by `python -m pycodemath.lint --generate-docs`; edit pycodemath/lint/rules.py -->",
        "",
        "Every finding carries a **witness**: a concrete input that was executed, and the exact "
        "(or intended) value it should have produced, computed by an independent route. A rule "
        "that cannot produce one says nothing. Silence a finding on one line with "
        "`# pcm: ignore[PCM101]` (or `# noqa: PCM101`).",
        "",
        "| code | name | level | default |",
        "|---|---|---|---|",
    ]
    for r in _RULES:
        out.append(f"| [{r.code}](#{r.code.lower()}) | `{r.name}` | {r.level} | {'on' if r.default else 'off'} |")
    family = None
    for r in _RULES:
        if r.family != family:
            family = r.family
            out += ["", f"## {_FAMILY_TITLES[family]}"]
        out += ["", f"### {r.code}", "", f"**`{r.name}`** — {r.summary}.", "",
                f"Level: {r.level}; {'on' if r.default else 'off'} by default.", "", r.why, "",
                f"*Witness:* {r.witness}."]
        if r.bad:
            out += ["", "Reported:", "", "```python", r.bad, "```"]
        if r.good:
            out += ["", "Instead:", "", "```python", r.good, "```"]
        if r.fix:
            out += ["", f"*Fix offered:* {r.fix}."]
    return "\n".join(out) + "\n"
