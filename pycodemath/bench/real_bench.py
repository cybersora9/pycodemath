"""VERIFY on REAL model errors (V7) — PRM800K solutions with a human first-error label.

V6 measured the verifier on injected, clean, parseable slips. This benchmark asks
the question a post would ask: of the mistakes a language model REALLY made, how
many can ``pycodemath.verify`` see at all, and of those, how many does it catch?

DATA. PRM800K (OpenAI, MIT licence, github.com/openai/prm800k at commit
``7ecc794``): model-written solutions to MATH problems, rated step by step by
people (+1 / 0 / -1). Phase 2 labelling stops at the first -1, so the index of
the first -1 is the human "first wrong step". The population is every solution in
``phase2_test.jsonl`` (MATH test problems) that the labeller finished; one random
sample of ``SAMPLE_SIZE`` of them (``random.Random(SEED)``, drawn once from the
whole population, nothing filtered by how the verifier does) is committed next to
this module with the licence notice. ``--population full`` scores the whole
population from the checksummed download; ``--population dev`` is a sample of
``phase2_train`` (MATH TRAIN problems) — the extraction rules were written on it
and frozen (commit ``81f9773``) before the first run on the test sample. Twelve
rules were added AFTER that run, each fixing an extraction error found on test
data (a claim the text does not make: ``$E = 5$`` read as Euler's number, ``17\\%5``
read as a percentage, ...); each is marked "test sample" / "full population" in a
comment, and ``dev-journal/BENCH_REAL.md`` reports the frozen and the final
numbers side by side. None of them was chosen by what the VERIFIER does.

EXTRACTION — rules only, deterministic, no language model anywhere. A step is
prose with mathematics. ``extract`` takes the math segments (``$..$``, ``$$..$$``,
``\\(..\\)``, ``\\[..\\]``, ``align`` environments) and plain-text arithmetic
runs (``48/2 = 24``), splits each at top-level relations and separators, and turns
every pair of neighbours joined by ``=`` into a candidate claim ``left = right``.
LaTeX is translated by fixed rules (``\\frac``, ``\\sqrt``, ``\\cdot``, ``^{..}``,
``\\binom``, ``\\log_b``, ``^\\circ``, ``\\%``, thousands separators, ...). A claim
is CHECKABLE only when both sides parse and are CLOSED (no free variable): an
equation ``2x + 3 = 7`` and an identity ``(x+1)^2 = x^2+2x+1`` look the same, and
reading an equation as an identity would refute every equation a model solves.
Anything the rules cannot read with certainty is dropped with a named reason
(``text`` — words or units, ``modular``, ``number-base``, ``log-base`` — a bare
``\\log`` has no agreed base, ``mixed-number`` — ``2\\frac{1}{2}``, ...), never
guessed. A step is COVERED when it yields at least one checkable claim.

CHECKING. Every checkable claim goes to ``check_equal`` (public API, semantics
untouched) with ``BUDGET`` seconds. A step is REFUTED when any claim is REFUTED,
VERIFIED when all are VERIFIED, else UNDECIDED. A solution is FLAGGED at its first
REFUTED step. Every pre-generated step is checked, also the unlabelled ones after
the human's first error — a deployed checker would not know where to stop.

SCORING (``score``). Coverage first: steps, solutions, and — the number that
bounds everything else — wrong solutions whose labelled first-error step is
covered. A wrong solution is an EXACT catch when the first REFUTED step IS the
labelled step; EARLY when a step people rated fine is refuted first (a false
alarm, even though the solution is wrong); LATE when only later, unlabelled steps
are refuted; else MISSED. UNDECIDED is never a catch. A missed error is sorted by
WHY, mechanically: the error step has no mathematics, no ``=``, only claims with
variables, only claims the rules dropped, or checkable claims that are all TRUE —
the error is outside the arithmetic (reading the problem, choosing the method,
prose). These are "out of reach", reported as such, never as hits.

    python -m pycodemath.bench.real_bench                       # committed sample
    python -m pycodemath.bench.real_bench --fetch               # download + checksums
    python -m pycodemath.bench.real_bench --population full     # whole phase2_test
    python -m pycodemath.bench.real_bench --population dev      # rule-development set
    python -m pycodemath.bench.real_bench --resample            # rebuild the sample file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import sympy as sp

from pycodemath.core.errors import PycodemathError
from pycodemath.frontend.parser import parse
from pycodemath.verify import check_equal

SEED = 20260929
SAMPLE_SIZE = 500
DEV_SIZE = 300
#: Seconds per claim. Closed arithmetic takes milliseconds; the budget only
#: bounds a pathological claim (the committed sample's slowest claim: see report).
BUDGET = 10.0

COMMIT = "7ecc794703b2877f63226f2477a49b34f9b25163"
URL = "https://media.githubusercontent.com/media/openai/prm800k/{commit}/prm800k/{path}"


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    size: int


#: The Git LFS object ids of the pinned commit — a download that does not hash to
#: these is refused.
SOURCES = {
    "phase2_test": Source(
        "data/phase2_test.jsonl",
        "6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad",
        12_240_719,
    ),
    "phase2_train": Source(
        "data/phase2_train.jsonl",
        "1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089",
        456_135_365,
    ),
    "math_test": Source(
        "math_splits/test.jsonl",
        "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        446_564,
    ),
    "math_train": Source(
        "math_splits/train.jsonl",
        "90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076",
        10_896_985,
    ),
}

DATA = Path(__file__).parent / "data" / "prm800k_sample.jsonl"
#: sha256 of the committed sample — ``--resample`` must reproduce it byte for byte.
SAMPLE_SHA256 = "3877e8072e54eb61105f6788be4b654bf9a7e9c031ea8250031619aa3fd187a2"
CACHE = Path(os.environ.get("PYCODEMATH_BENCH_CACHE", Path.home() / ".cache" / "pycodemath-bench"))


# ============================================================================
# Data: download, population, sample
# ============================================================================
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, cache: Path = CACHE) -> Path:
    """The checksummed local copy of ``SOURCES[name]``, downloaded on first use."""
    src = SOURCES[name]
    dest = cache / "prm800k" / src.path
    if dest.exists() and sha256_of(dest) == src.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(URL.format(commit=COMMIT, path=src.path)) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256_of(tmp)
    if got != src.sha256:
        tmp.unlink()
        raise RuntimeError(f"{src.path}: sha256 {got} != expected {src.sha256}")
    tmp.replace(dest)
    return dest


def _jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield n, json.loads(line)


def population(
    rows: Iterable[tuple[int, dict]], meta: "dict[str, dict]", split: str
) -> "tuple[list[dict], Counter[str]]":
    """Every finished, unambiguous phase-2 solution as a record, plus why others left.

    Out: ``give_up`` / ``bad_problem`` (the labeller did not finish), a label step
    whose text is not the pre-generated step (``misaligned``), and repeated
    solutions — the first line is kept, unless the copies disagree on the first
    error (``conflicting-duplicate``: all copies out)."""
    out: Counter[str] = Counter()
    kept: dict[tuple, dict] = {}
    conflicting: set[tuple] = set()
    for line, row in rows:
        q, label = row["question"], row["label"]
        steps = q.get("pre_generated_steps") or []
        reason = label["finish_reason"]
        if reason not in ("found_error", "solution"):
            out[reason] += 1
            continue
        if not steps:
            out["no-steps"] += 1
            continue
        ratings: "list[int | None]" = []
        for i, s in enumerate(label["steps"]):
            first = s["completions"][0]
            if i >= len(steps) or first["text"] != steps[i]:
                break
            ratings.append(first["rating"])
        if len(ratings) != len(label["steps"]):
            out["misaligned"] += 1
            continue
        first_error = next((i + 1 for i, r in enumerate(ratings) if r == -1), None)
        info = meta.get(q["problem"], {})
        record = {
            "id": f"{split}:{line}",
            "subject": info.get("subject", "?"),
            "level": info.get("level", 0),
            "problem": q["problem"],
            "steps": steps,
            "ratings": ratings,
            "first_error": first_error,
            "finish_reason": reason,
            "answer": q.get("pre_generated_answer"),
            "truth": q.get("ground_truth_answer"),
        }
        key = (q["problem"], tuple(steps))
        if key in kept:
            out["duplicate"] += 1
            if kept[key]["first_error"] != first_error:
                conflicting.add(key)
            continue
        kept[key] = record
    for key in conflicting:
        del kept[key]
        out["conflicting-duplicate"] += 1
    return list(kept.values()), out


def draw(records: "list[dict]", n: int, seed: "int | str") -> "list[dict]":
    """``n`` records by ``random.Random(seed).sample`` over the population in file
    order, returned in file order. Nothing about the verifier enters the draw."""
    order = sorted(records, key=lambda r: int(r["id"].rsplit(":", 1)[1]))
    chosen = random.Random(seed).sample(order, min(n, len(order)))
    return sorted(chosen, key=lambda r: int(r["id"].rsplit(":", 1)[1]))


def _meta(name: str, cache: Path) -> "dict[str, dict]":
    return {r["problem"]: r for _, r in _jsonl(fetch(name, cache))}


def build(which: str, cache: Path = CACHE) -> "tuple[list[dict], Counter[str]]":
    """``full`` = the phase2_test population, ``sample`` = its committed draw,
    ``dev`` = a draw from phase2_train (rule development only)."""
    if which == "dev":
        pop, out = population(_jsonl(fetch("phase2_train", cache)), _meta("math_train", cache), "phase2_train")
        return draw(pop, DEV_SIZE, f"{SEED}/dev"), out
    pop, out = population(_jsonl(fetch("phase2_test", cache)), _meta("math_test", cache), "phase2_test")
    return (pop if which == "full" else draw(pop, SAMPLE_SIZE, SEED)), out


def dump(records: "list[dict]") -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)


def load(path: Path = DATA) -> "list[dict]":
    return [r for _, r in _jsonl(path)]


# ============================================================================
# Extraction: prose + LaTeX -> closed claims ``left = right``
# ============================================================================
class Skip(Exception):
    """A part the rules cannot read with certainty; ``args[0]`` names the rule."""


@dataclass(frozen=True)
class Claim:
    left: str       # parser syntax
    right: str
    source: str     # the LaTeX / text the pair came from
    #: ``equal``, or ``rounded``: ``right`` is a decimal literal with ``digits``
    #: places and the claim is that ``left`` rounds (or truncates) to it
    kind: str = "equal"
    digits: int = 0


@dataclass
class Extraction:
    claims: "list[Claim]" = field(default_factory=list)
    dropped: "Counter[str]" = field(default_factory=Counter)  # candidate '=' pairs, by reason
    math: bool = False       # any math segment or plain-text arithmetic run
    relation: bool = False   # any '=' between two parts

    @property
    def covered(self) -> bool:
        return bool(self.claims)


_SEGMENT = re.compile(
    r"\$\$(?P<a>.+?)\$\$"
    r"|\\\[(?P<b>.+?)\\\]"
    r"|\\\((?P<c>.+?)\\\)"
    r"|\\begin\{(?P<env>align\*?|aligned|eqnarray\*?|gather\*?|equation\*?)\}(?P<d>.+?)\\end\{(?P=env)\}"
    r"|\$(?P<e>.+?)\$",
    re.S,
)
#: Plain-text arithmetic: a maximal run of digits and operators (character class,
#: so no backtracking can shorten ``4^4=2^(2x)`` into ``4^4=2``).
_PLAIN = re.compile(r"[0-9.,+\-*/×÷^()= −]+")
_PROSE_MODULAR = re.compile(r"\bmod(ulo)?\b|\bcongruen", re.I)
_PROSE_BASE = re.compile(r"\bbase\b", re.I)

#: Commands that separate parts at depth 0 (a relation other than '=').
_RELATIONS = frozenset(
    "neq ne le leq ge geq lt gt leqslant geqslant approx sim simeq cong propto "
    "Rightarrow rightarrow Longrightarrow longrightarrow implies iff Leftrightarrow "
    "to mapsto quad qquad lor land vee wedge ll gg in notin subset subseteq "
    "text mbox textbf textit textrm mathrm hbox".split()
)
_TEXTUAL = frozenset("text mbox textbf textit textrm mathrm hbox".split())

_SKIPS: "tuple[tuple[re.Pattern[str], str], ...]" = tuple(
    (re.compile(p), reason)
    for p, reason in (
        (r"\\(text|mbox|textbf|textit|textrm|mathrm|hbox|operatorname)(?![A-Za-z])", "text"),
        (r"\\(bmod|pmod|mod|equiv)(?![A-Za-z])", "modular"),
        (r"\\(pm|mp)(?![A-Za-z])|±", "plus-minus"),
        (r"\\(dots|ldots|cdots|vdots|ddots)(?![A-Za-z])|\.\.\.|…", "ellipsis"),
        (r"\\infty(?![A-Za-z])|∞", "infinity"),
        (r"\\(sum|prod|int|iint|oint|lim|limsup|liminf|max|min|sup|inf|det|arg|deg|gcd|lcm)(?![A-Za-z])", "operator"),
        (r"\\(mathbf|vec|langle|rangle|hat|bm|boldsymbol|begin|end)(?![A-Za-z])", "vector-or-environment"),
        (r"\\(overline|overrightarrow|angle|triangle|measuredangle|widehat|parallel|perp|odot|stackrel|overset|underset|underbrace|overbrace|circ)(?![A-Za-z])", "notation"),
        (r"\\(mathbb|cup|cap|setminus|emptyset|varnothing|mathcal)(?![A-Za-z])|\\\{|\\\}", "set"),
        (r"\\(mid|nmid|divides)(?![A-Za-z])", "divides"),
        (r"!!", "double-factorial"),
        (r"'", "notation"),
        (r"\d\s*\\[dtc]?frac", "mixed-number"),
        (r"(\d|\})\s*_", "number-base"),
        (r"√", "notation"),
    )
)

_GREEK = frozenset(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi "
    "rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Sigma Phi Psi Omega".split()
)
_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "ln": "log", "exp": "exp",
}
_INVERSE = {"sin": "asin", "cos": "acos", "tan": "atan", "cot": "acot", "sec": "asec", "csc": "acsc"}
_FRACS = frozenset(("frac", "dfrac", "tfrac", "cfrac"))
_BINOMS = frozenset(("binom", "dbinom", "tbinom"))
_SIMPLE = {
    "cdot": "*", "times": "*", "ast": "*", "div": "/", "pi": " pi ",
    "lfloor": " floor(", "rfloor": ")", "lceil": " ceiling(", "rceil": ")",
    "lvert": "|", "rvert": "|", "vert": "|",
}
_UNICODE = {"−": "-", "–": "-", "×": "*", "·": "*", "⋅": "*", "÷": "/", "²": "^2", "³": "^3", "π": " pi "}


def _command(s: str, i: int) -> "tuple[str, int]":
    """The command name starting at ``s[i] == '\\\\'`` and the index after it."""
    j = i + 1
    if j < len(s) and not s[j].isalpha():
        return s[j], j + 1
    while j < len(s) and s[j].isalpha():
        j += 1
    return s[i + 1:j], j


def _group(s: str, i: int, open_: str = "{", close: str = "}") -> "tuple[str, int]":
    """Content of the balanced group opening at ``s[i]`` and the index after it."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_:
            depth += 1
        elif s[j] == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise Skip("unbalanced")


def _arg(s: str, i: int) -> "tuple[str, int]":
    """One LaTeX argument: a ``{group}``, a command token, or a single character."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise Skip("latex")
    if s[i] == "{":
        return _group(s, i)
    if s[i] == "\\":
        name, j = _command(s, i)
        return s[i:j], j
    return s[i], i + 1


def _function_arg(s: str, i: int) -> "tuple[str, int]":
    """The argument of ``\\sin`` & co.: a parenthesised or braced group, or a
    coefficient followed by ``\\pi``, ``\\frac{..}{..}``, one letter or a Greek
    letter (``\\sin 30°``, ``\\cos 2\\pi``, ``\\tan \\frac{\\pi}{4}``). Anything
    longer is ambiguous (``\\sin x + 1``) and read no further than this."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] in "({":
        return _group(s, i, s[i], ")" if s[i] == "(" else "}")
    atom, end = _bare_atom(s, i)
    if re.fullmatch(r"\d+(\.\d+)?", atom) and re.match(r"\s*(\\(times|cdot|div)(?![A-Za-z])|[*/])", s[end:]):
        # \sin 2 \times 10^\circ: is it sin(2)*10° or sin(20°)? (full population: 1 case)
        raise Skip("function-arg")
    if end < len(s) and s[end] == "^":
        # \log_2 2^6 is log(2^6), \sin x^2 is sin(x^2) (test sample: 1 case)
        _, end = _arg(s, end + 1)
        atom = s[i:end]
    return atom, end


def _bare_atom(s: str, i: int) -> "tuple[str, int]":
    m = re.match(r"\d+(\.\d+)?°?", s[i:])
    j = i + (m.end() if m else 0)
    k = j
    while k < len(s) and s[k] == " ":
        k += 1
    if k < len(s) and s[k] == "\\":
        name, end = _command(s, k)
        if name in _FRACS:
            _, end = _arg(s, end)
            _, end = _arg(s, end)
            return s[i:end], end
        if name == "pi" or name in _GREEK:
            return s[i:end], end
    elif k < len(s) and s[k].isalpha():
        return s[i:k + 1], k + 1
    if j > i:
        return s[i:j], j
    raise Skip("function-arg")


def _convert(s: str) -> str:
    """LaTeX -> parser syntax by fixed rules; ``Skip`` on anything else."""
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            name, j = _command(s, i)
            if name in _FRACS:
                a, j = _arg(s, j)
                b, j = _arg(s, j)
                out.append(f"(({_convert(a)})/({_convert(b)}))")
            elif name == "sqrt":
                if j < len(s) and s[j] == "[":
                    n, j = _group(s, j, "[", "]")
                    a, j = _arg(s, j)
                    out.append(f" root(({_convert(a)}),({_convert(n)}))")
                else:
                    a, j = _arg(s, j)
                    out.append(f" sqrt(({_convert(a)}))")
            elif name in _BINOMS:
                a, j = _arg(s, j)
                b, j = _arg(s, j)
                out.append(f" binomial(({_convert(a)}),({_convert(b)}))")
            elif name == "log":
                if j >= len(s) or s[j] != "_":
                    raise Skip("log-base")
                base, j = _arg(s, j + 1)
                a, j = _function_arg(s, j)
                out.append(f" log(({_convert(a)}),({_convert(base)}))")
            elif name in _FUNCS:
                power = None
                if j < len(s) and s[j] == "^":
                    power, j = _arg(s, j + 1)
                a, j = _function_arg(s, j)
                func = _FUNCS[name]
                if power is not None and power.replace(" ", "") == "-1":
                    # \tan^{-1} is the inverse function, not a reciprocal
                    if func not in _INVERSE:
                        raise Skip("notation")
                    func, power = _INVERSE[func], None
                call = f" {func}(({_convert(a)}))"
                out.append(f"({call})^({_convert(power)})" if power is not None else call)
            elif name in _SIMPLE:
                out.append(_SIMPLE[name])
            elif name in _GREEK:
                out.append(f" {name}_g ")
            elif name in (",", "!", ";", ":", " ", "displaystyle", "left", "right",
                          "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr"):
                pass
            elif name == "%":
                # 20\% is a percentage; 17\%5 is the modulo operator (test sample)
                if re.match(r"\s*[\w({]", s[j:]):
                    raise Skip("modular")
                out.append("/100")
            else:
                raise Skip("latex")
            i = j
        elif c == "{":
            inner, i = _group(s, i)
            out.append(f"({_convert(inner)})")
        elif c == "}":
            raise Skip("unbalanced")
        elif c == "°":
            out.append("*pi/180")
            i += 1
        elif c in "&~":
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _abs_bars(text: str) -> str:
    """``|a|`` -> ``Abs(a)``, bars paired left to right; an odd count is ambiguous."""
    if "|" not in text:
        return text
    parts = text.split("|")
    if len(parts) % 2 == 0:
        raise Skip("abs")
    return "".join(p if k % 2 == 0 else f" Abs({p})" for k, p in enumerate(parts))


def translate(part: str, euler: bool = True) -> str:
    """One side of a claim, LaTeX -> parser syntax (``Skip`` when not certain).

    ``euler``: ``e^..`` is Euler's number — unless the step also uses a bare
    ``e`` ("$1 = e^3$, where $e$ is the edge length"; full population: 1 case)."""
    s = part
    for u, v in _UNICODE.items():
        s = s.replace(u, v)
    if re.search(r"(?<![A-Za-z\\])[EI](?![A-Za-z])", s):
        # a face or a point named E, I: the parser would read Euler's number /
        # the imaginary unit (test sample: $E = 5$ "caught" as e != 5)
        raise Skip("variables")
    s = _degrees(s)
    s = re.sub(r"\{([^{}]*)\\choose([^{}]*)\}", r"\\binom{\1}{\2}", s)
    for pattern, reason in _SKIPS:
        if pattern.search(s):
            raise Skip(reason)
    s = re.sub(r"\\(left|right)\s*\.", " ", s)
    s = re.sub(r"\\(left|right)(?![a-zA-Z])", "", s)
    text = _abs_bars(_convert(s))
    if euler:
        text = re.sub(r"(?<![A-Za-z_])e(?=\s*\^)", "E", text)
    text = text.strip().rstrip(".,;").strip()
    if re.search(r"\d\s+\.?\d", text):
        raise Skip("adjacent-numbers")
    if "," in text.replace("),(", ""):
        raise Skip("list")
    return text


_THOUSANDS = (r"(\d),\\!(\d{3})", r"(\d)\{,\}(\d{3})", r"(\d)\\[,!]\s*(\d{3})(?!\d)", r"(\d),(\d{3})(?!\d)")


def _thousands(s: str) -> str:
    """``200,\\!000``, ``1{,}000``, ``1\\,000``, ``1,000`` -> digits only (before
    splitting, so a thousands comma is never read as a list separator)."""
    for pat in _THOUSANDS:
        prev = None
        while prev != s:
            prev, s = s, re.sub(pat, r"\1\2", s)
    return s


def _split(seg: str) -> "list[tuple[str, str]]":
    """``seg`` -> [(part, relation-after), ...] at depth 0; relation is ``=``, ``rel``
    (any other relation or separator) or ``end``."""
    parts: list[tuple[str, str]] = []
    depth = 0
    start = 0
    i = 0

    def cut(end: int, rel: str, nxt: int) -> None:
        nonlocal start
        parts.append((seg[start:end], rel))
        start = nxt

    while i < len(seg):
        c = seg[i]
        if c == "\\":
            name, j = _command(seg, i)
            if depth == 0 and name == "\\":
                cut(i, "rel", j)
            elif depth == 0 and name in _RELATIONS:
                if name in _TEXTUAL:
                    while j < len(seg) and seg[j] == " ":
                        j += 1
                    if j < len(seg) and seg[j] == "{":
                        _, j = _group(seg, j)
                cut(i, "rel", j)
            elif name in ("{", "}"):
                pass  # escaped brace, a set: rejected by translate
            i = j
            continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        elif depth == 0 and c == "=":
            cut(i, "=", i + 1)
        elif depth == 0 and c in "<>,;:":
            cut(i, "rel", i + 1)
        i += 1
    parts.append((seg[start:], "end"))
    return parts


#: Names the translation emits; any other identifier is a variable or a function
#: of the problem (``p(0)`` parses as ``p*0 = 0`` — closed by accident).
_KNOWN_NAMES = frozenset(
    "sqrt root log exp binomial floor ceiling Abs pi E sin cos tan cot sec csc "
    "asin acos atan acot asec acsc sinh cosh tanh".split()
)


def _closed(text: str) -> bool:
    """True for text that parses to a scalar with no free variable."""
    if set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)) - _KNOWN_NAMES:
        return False
    try:
        e = parse(text)
    except PycodemathError as exc:
        raise Skip("parse") from exc
    if not isinstance(e.sy, sp.Expr):
        raise Skip("parse")
    return not e.free_symbols


def _segments(step: str) -> "tuple[list[str], str]":
    """The math segments of ``step`` (align lines split) and the remaining prose."""
    step = step.replace("\\$", " ")
    segs: list[str] = []
    prose: list[str] = []
    pos = 0
    for m in _SEGMENT.finditer(step):
        prose.append(step[pos:m.start()])
        pos = m.end()
        body = next(g for g in (m.group("a"), m.group("b"), m.group("c"), m.group("d"), m.group("e")) if g is not None)
        body = re.sub(r"\\(begin|end)\{(align\*?|aligned|eqnarray\*?|gather\*?|split)\}", " ", body)
        # a matrix / cases / array body has its own \\ and &: one opaque token
        body = re.sub(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", r" \\begin{env} ", body, flags=re.S)
        body = re.sub(r"\\boxed\s*\{", "{", body)
        segs.append(body)
    prose.append(step[pos:])
    return segs, " ".join(prose)


_LITERAL = re.compile(r"\(?-?\d+(\.\d+)?\)?")
_DECIMAL = re.compile(r"-?\d*\.(\d+)")
_REMAINDER = re.compile(
    r"\bremainders?\b|\bquotient\b|\\(text|mbox)\{\s*R\s*\}", re.I)  # "\text{ R }17": full population
#: A step that calls its own equation false ("$5 = 3 + 4$, which is false") does
#: not claim it (test sample: 1 case, counted as a catch by the frozen rules).
_STATED_FALSE = re.compile(
    r"\b(false|impossible|contradiction|contradicts|absurd|not true|untrue|incorrect|wrong|mistake)\b", re.I)


def _claim(left: str, right: str, source: str) -> Claim:
    """A decimal literal on one side reads as a rounded value (``362/7 = 51.71``):
    the claim becomes "rounds or truncates to" — checked by ``check_claim``."""
    if _DECIMAL.fullmatch(left) and not _LITERAL.fullmatch(right):
        left, right = right, left
    m = _DECIMAL.fullmatch(right)
    if m and not _LITERAL.fullmatch(left):
        return Claim(left, right, source, "rounded", len(m.group(1)))
    return Claim(left, right, source)


def _lines(seg: str) -> "list[str]":
    """``\\\\``-separated lines; a line that opens with an operator continues the
    one before (``&= (a)(b) \\\\ &\\cdot (c)``: full population, 1 case)."""
    lines: list[str] = []
    for line in re.split(r"\\\\", seg):
        head = line.replace("&", " ").lstrip()
        if lines and re.match(r"(\\(cdot|times|div)(?![A-Za-z])|[-+*/])", head):
            lines[-1] += " " + line
        else:
            lines.append(line)
    return lines


def _angle_unit(text: str) -> bool:
    """Does translated ``text`` carry a degree unit outside every function call?
    ``sin((30*pi/180))`` does not (the angle is an argument), ``74*pi/180`` does."""
    depth_in_call: "list[bool]" = []
    for m in re.finditer(r"([A-Za-z]+)?\(|\)|\*pi/180", text):
        tok = m.group(0)
        if tok == ")":
            if depth_in_call:
                depth_in_call.pop()
        elif tok == "*pi/180":
            if not any(depth_in_call):
                return True
        else:
            depth_in_call.append(m.group(1) is not None and m.group(1) != "pi")
    return False


def _degrees(part: str) -> str:
    return re.sub(r"\^\s*(\{\s*\\(circ|degree)\s*\}|\\(circ|degree)(?![A-Za-z]))|\\degree(?![A-Za-z])", "°", part)


class _Extractor:
    """Carries the last part of the previous segment, so a step (or align line)
    that opens with ``= ...`` continues the expression before it."""

    def __init__(self) -> None:
        self.carry: "str | None" = None
        self.remainder = False
        self.stated_false = False
        self.euler = True

    def _pairs(self, parts: "list[tuple[str, str]]", source: str, plain: bool, ex: Extraction) -> None:
        if not plain and parts and not parts[0][0].strip() and parts[0][1] == "=" and self.carry is not None:
            parts = [(self.carry, "=")] + parts[1:]
        for (left, rel), (right, _) in zip(parts, parts[1:]):
            if rel != "=":
                continue
            ex.relation = True
            if not left.strip() or not right.strip():
                ex.dropped["dangling"] += 1
                continue
            try:
                lt = _plain(left) if plain else translate(left, self.euler)
                rt = _plain(right) if plain else translate(right, self.euler)
                if not plain and re.search(r"\\%", left + right) and not (
                        re.search(r"\\%", left) and re.search(r"\\%", right)) and re.search(
                        r"(\\times|\\cdot|\*)\s*100(?!\d)", left + right):
                    # "0.4895 \times 100 = 48.95\%": times 100 converts to percent
                    # (full population, an unlabelled step)
                    raise Skip("percent-conversion")
                if not plain and _angle_unit(lt) != _angle_unit(rt):
                    # "180 - 54 - 52 = 74^\circ": the unit on one side only
                    # (full population: 1 case)
                    raise Skip("degrees-one-side")
                if not lt or not rt:
                    raise Skip("dangling")
                if not (_closed(lt) and _closed(rt)):
                    raise Skip("variables")
                if _LITERAL.fullmatch(lt) and _LITERAL.fullmatch(rt):
                    # "0 = -16, which is impossible", "25 = 37": a stated
                    # contradiction or a tautology, not a computation
                    raise Skip("bare-literals")
                if self.stated_false:
                    raise Skip("stated-false")
                if self.remainder and "/" in lt + rt:
                    # "223 / 3 = 74 with remainder 1": integer division
                    raise Skip("integer-division")
            except Skip as why:
                ex.dropped[why.args[0]] += 1
                continue
            ex.claims.append(_claim(lt, rt, source.strip()))
        last = parts[-1][0] if parts else ""
        if not plain and last.strip():
            self.carry = last

    def step(self, text: str) -> Extraction:
        ex = Extraction()
        self.remainder = bool(_REMAINDER.search(text))
        self.stated_false = bool(_STATED_FALSE.search(text))
        segs, prose = _segments(text)
        self.euler = not any(re.search(r"(?<![A-Za-z\\])e(?![A-Za-z])(?!\s*\^)", seg) for seg in segs)
        modular = bool(_PROSE_MODULAR.search(prose))
        base = bool(_PROSE_BASE.search(prose))
        for seg in segs:
            ex.math = True
            for line in _lines(seg):
                if not line.strip():
                    continue
                parts = _split(_thousands(line.replace("&", " ")))
                if modular or base or re.search(r"\\(bmod|pmod|mod|equiv)\b", line):
                    ex.relation = ex.relation or any(r == "=" for _, r in parts[:-1])
                    n = sum(1 for _, r in parts[:-1] if r == "=")
                    ex.dropped["number-base" if base and not modular else "modular"] += n
                    continue
                self._pairs(parts, line, False, ex)
        for m in _PLAIN.finditer(prose):
            run = m.group(0)
            if "=" not in run or not re.search(r"\d", run):
                continue
            ex.math = ex.relation = True
            start = m.start() + len(run) - len(run.lstrip())
            end = m.end() - (len(run) - len(run.rstrip()))
            if _mixed(prose[:start], prose[start:end], prose[end:]):
                ex.dropped["prose-mixed"] += run.count("=")
                continue
            if modular or base:
                ex.dropped["modular" if modular else "number-base"] += run.count("=")
                continue
            parts = [(p, "=") for p in run.split("=")]
            parts[-1] = (parts[-1][0], "end")
            self._pairs(parts, run, True, ex)
        return ex


#: Words that make a plain-text number part of a larger expression (``3 x 4 = 12``,
#: ``= 24 percent``): the run is then not the whole claim.
_OPERATOR_WORDS = frozenset(
    "x X times plus minus by over of into divided multiplied less more than squared cubed "
    "power percent cents dollars".split()
)


def _mixed(before: str, run: str, after: str) -> bool:
    """Is a plain-text arithmetic run glued to text that belongs to it?

    ``x = y + 7 = 10`` (a variable before), ``6 \\cdot 36 = 216`` (a command
    before), ``3 x 4 = 12`` (an operator word), ``+ 10 = 67`` (starts with a
    binary operator): the run is then not the whole claim."""
    if before[-1:] and (before[-1].isalnum() or before[-1] in "_$\\!%'"):
        return True
    if after[:1] and (after[0].isalnum() or after[0] in "_$\\!%'"):
        return True
    words_before = before.split()
    words_after = after.split()
    if run[:1] in "+-*/^×÷=" and words_before and (words_before[-1][-1:].isalnum() or words_before[-1][-1:] in ")]"):
        # "x^2 + 2x - 24 = 0" leaves the run "- 24 = 0" (test sample: 1 case)
        return True
    for words in (words_before[-1:], words_after[:1]):
        if not words:
            continue
        w = words[0].strip(".,;:")
        if w in _OPERATOR_WORDS or "\\" in w or len(w) == 1 or w[:1] in "%^*/+-" or w[-1:] in "%^*/+-":
            return True
    return False


def _plain(part: str) -> str:
    s = part
    for u, v in _UNICODE.items():
        s = s.replace(u, v)
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    s = s.strip().rstrip(".,").strip()
    if not re.search(r"\d", s):
        raise Skip("dangling")
    if s[:1] in "+*/^×÷" or s[-1:] in "+-*/^×÷":
        raise Skip("prose-mixed")
    if "," in s:
        raise Skip("list")
    if re.search(r"\d\s+\.?\d", s):
        raise Skip("adjacent-numbers")
    if s.count("/") >= 2 and "(" not in s:
        # "15975/25 / 1000/25": a fraction over a fraction written flat
        # (full population, an unlabelled step)
        raise Skip("slash-ambiguity")
    return s


def extract(step: str) -> Extraction:
    """The checkable claims of one step on its own (no continuation)."""
    return _Extractor().step(step)


def extract_solution(steps: Sequence[str]) -> "list[Extraction]":
    """The checkable claims of every step, a leading ``= ...`` continuing the step before."""
    ex = _Extractor()
    return [ex.step(s) for s in steps]


# ============================================================================
# Checking and scoring
# ============================================================================
@dataclass
class ClaimResult:
    left: str
    right: str
    source: str
    status: str      # VERIFIED / REFUTED / UNDECIDED / ERROR
    method: str
    detail: str
    seconds: float


@dataclass
class StepResult:
    index: int                   # 1-based
    rating: "int | None"         # human rating, None past the labelled prefix
    status: str                  # not-covered / VERIFIED / REFUTED / UNDECIDED
    claims: "list[ClaimResult]"
    dropped: "dict[str, int]"
    math: bool
    relation: bool


@dataclass
class SolutionResult:
    id: str
    subject: str
    level: int
    first_error: "int | None"
    steps: "list[StepResult]"
    seconds: float

    @property
    def flagged_at(self) -> "int | None":
        return next((s.index for s in self.steps if s.status == "REFUTED"), None)


def _step_status(claims: "list[ClaimResult]") -> str:
    if not claims:
        return "not-covered"
    statuses = {c.status for c in claims}
    if "REFUTED" in statuses:
        return "REFUTED"
    if statuses == {"VERIFIED"}:
        return "VERIFIED"
    return "UNDECIDED"


def _rounded(claim: Claim, budget: float) -> "tuple[str, str, str]":
    """``left`` rounds half-up OR truncates to the decimal ``right``: two equalities
    of integers, each through ``check_equal``. REFUTED only if both are refuted."""
    scale = f"10^{claim.digits}"
    target = f"{scale}*({claim.right})"
    rounded = check_equal(f"floor({scale}*({claim.left}) + 1/2)", target, budget=budget)
    if rounded.verified:
        return "VERIFIED", "rounded/" + rounded.method, rounded.detail
    cut = check_equal(f"floor({scale}*({claim.left}))", target, budget=budget)
    if cut.verified:
        return "VERIFIED", "truncated/" + cut.method, cut.detail
    if rounded.refuted and cut.refuted:
        return "REFUTED", "rounded/" + rounded.method, (
            f"neither rounds nor truncates to {claim.right}: {rounded.detail}")
    return "UNDECIDED", "rounded/" + rounded.method, rounded.detail


def check_claim(claim: Claim, budget: float = BUDGET) -> ClaimResult:
    t0 = time.perf_counter()
    try:
        if claim.kind == "rounded":
            status, method, detail = _rounded(claim, budget)
        else:
            v = check_equal(claim.left, claim.right, budget=budget)
            status, method, detail = v.status.name, v.method, v.detail
    except PycodemathError as exc:  # counted, reported; never a hit
        status, method, detail = "ERROR", type(exc).__name__, str(exc)
    return ClaimResult(claim.left, claim.right, claim.source, status, method, detail,
                       time.perf_counter() - t0)


def run_solution(record: dict, budget: float = BUDGET) -> SolutionResult:
    t0 = time.perf_counter()
    ratings = record["ratings"]
    steps = []
    for k, ex in enumerate(extract_solution(record["steps"])):
        claims = [check_claim(c, budget) for c in ex.claims]
        steps.append(StepResult(
            k + 1, ratings[k] if k < len(ratings) else None, _step_status(claims),
            claims, dict(ex.dropped), ex.math, ex.relation,
        ))
    return SolutionResult(record["id"], record["subject"], record["level"],
                          record["first_error"], steps, time.perf_counter() - t0)


def outcome(r: SolutionResult) -> str:
    """exact / early / late / missed for a wrong solution; clean / false-alarm for a correct one."""
    at = r.flagged_at
    if r.first_error is None:
        return "clean" if at is None else "false-alarm"
    if at is None:
        return "missed"
    return "exact" if at == r.first_error else ("early" if at < r.first_error else "late")


def out_of_reach(step: StepResult) -> str:
    """Why the labelled error step yields no catch — decided mechanically."""
    if step.status == "REFUTED":
        return "caught"
    if step.status == "UNDECIDED":
        return "checked-undecided"
    if step.status == "VERIFIED":
        return "checked-all-true"
    if not step.math:
        return "no-math"
    if not step.relation:
        return "no-equality"
    if set(step.dropped) == {"variables"}:
        return "variables-only"
    return "dropped-by-rules"


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "— (0/0)"


def score(results: "list[SolutionResult]") -> dict:
    wrong = [r for r in results if r.first_error is not None]
    right = [r for r in results if r.first_error is None]
    all_steps = [s for r in results for s in r.steps]
    labelled = [s for s in all_steps if s.rating is not None]
    ok_steps = [s for s in labelled if s.rating in (0, 1)]
    bad_steps = [s for s in labelled if s.rating == -1]
    claims = [c for s in all_steps for c in s.claims]
    outcomes = Counter(outcome(r) for r in results)
    error_steps = [r.steps[r.first_error - 1] for r in wrong if r.first_error is not None]
    reach = Counter(out_of_reach(s) for s in error_steps)
    refuted_labelled = [s for s in labelled if s.status == "REFUTED"]
    flagged = [r for r in results if r.flagged_at is not None]
    times = sorted(r.seconds for r in results)
    dropped: Counter[str] = Counter()
    for s in all_steps:
        dropped.update(s.dropped)
    return {
        "solutions": len(results),
        "wrong": len(wrong),
        "correct": len(right),
        "steps": len(all_steps),
        "steps_covered": sum(s.status != "not-covered" for s in all_steps),
        "labelled_steps": len(labelled),
        "labelled_covered": sum(s.status != "not-covered" for s in labelled),
        "solutions_covered": sum(any(s.status != "not-covered" for s in r.steps) for r in results),
        "error_steps_covered": sum(s.status != "not-covered" for s in error_steps),
        "claims": len(claims),
        "claims_by_status": dict(Counter(c.status for c in claims)),
        "dropped_by_reason": dict(dropped.most_common()),
        "outcomes": dict(outcomes),
        "error_step_reach": dict(reach),
        "flagged": len(flagged),
        "flagged_wrong": sum(r.first_error is not None for r in flagged),
        "refuted_labelled_steps": len(refuted_labelled),
        "refuted_error_steps": sum(s.rating == -1 for s in refuted_labelled),
        "refuted_ok_steps": sum(s.rating in (0, 1) for s in refuted_labelled),
        "ok_steps": len(ok_steps),
        "ok_steps_covered": sum(s.status != "not-covered" for s in ok_steps),
        "bad_steps_covered": sum(s.status != "not-covered" for s in bad_steps),
        "refuted_unlabelled_steps": sum(s.rating is None and s.status == "REFUTED" for s in all_steps),
        "undecided_steps": sum(s.status == "UNDECIDED" for s in all_steps),
        "median_s": statistics.median(times) if times else 0.0,
        "p90_s": times[int(0.9 * (len(times) - 1))] if times else 0.0,
        "max_s": times[-1] if times else 0.0,
        "total_s": sum(times),
    }


def report(s: dict, title: str) -> str:
    oc = s["outcomes"]
    w = s["wrong"]
    lines = [
        f"== {title}: {s['solutions']} solutions ({w} wrong, {s['correct']} correct by the human label)",
        "",
        "COVERAGE (number 1 — nothing below means more than this)",
        f"  steps with >= 1 checkable claim   {_pct(s['steps_covered'], s['steps'])}",
        f"  labelled steps covered            {_pct(s['labelled_covered'], s['labelled_steps'])}",
        f"  solutions with >= 1 covered step  {_pct(s['solutions_covered'], s['solutions'])}",
        f"  wrong solutions whose FIRST-ERROR step is covered  {_pct(s['error_steps_covered'], w)}",
        "",
        "WRONG SOLUTIONS (first error = the human label)",
        f"  exact  (first REFUTED = labelled step)     {_pct(oc.get('exact', 0), w)}",
        f"  early  (a step rated fine refuted first)   {_pct(oc.get('early', 0), w)}",
        f"  late   (only later, unlabelled steps)      {_pct(oc.get('late', 0), w)}",
        f"  missed (nothing refuted)                   {_pct(oc.get('missed', 0), w)}",
        "  the labelled error step, by reach:",
    ]
    for k in ("caught", "checked-undecided", "checked-all-true", "variables-only",
              "dropped-by-rules", "no-equality", "no-math"):
        lines.append(f"    {k:<18} {_pct(s['error_step_reach'].get(k, 0), w)}")
    lines += [
        "",
        "CORRECT SOLUTIONS",
        f"  flagged (false alarm)  {_pct(oc.get('false-alarm', 0), s['correct'])}",
        "",
        "STEPS",
        f"  REFUTED labelled steps that are the error   {_pct(s['refuted_error_steps'], s['refuted_labelled_steps'])}  (step precision)",
        f"  steps rated fine (0/+1) refuted             {_pct(s['refuted_ok_steps'], s['ok_steps_covered'])} of covered, {s['ok_steps']} rated fine in all",
        f"  REFUTED after the labelled error (no label) {s['refuted_unlabelled_steps']}",
        f"  UNDECIDED steps                             {_pct(s['undecided_steps'], s['steps_covered'])} of covered",
        f"  claims: {s['claims']}  {s['claims_by_status']}",
        f"  dropped '=' pairs by rule: {s['dropped_by_reason']}",
        "",
        f"SOLUTION LEVEL  flagged {s['flagged']}, of them wrong {_pct(s['flagged_wrong'], s['flagged'])}"
        f" (base rate of wrong: {_pct(w, s['solutions'])}); recall {_pct(s['flagged_wrong'], w)}",
        f"TIME  median {s['median_s'] * 1000:.0f} ms, p90 {s['p90_s'] * 1000:.0f} ms, "
        f"max {s['max_s']:.2f} s per solution; total {s['total_s']:.1f} s",
    ]
    return "\n".join(lines)


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pycodemath.bench.real_bench")
    ap.add_argument("--population", choices=("sample", "full", "dev"), default="sample")
    ap.add_argument("--fetch", action="store_true", help="download the sources and verify checksums")
    ap.add_argument("--resample", action="store_true", help="rebuild the committed sample from the download")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--budget", type=float, default=BUDGET, help="seconds per claim")
    ap.add_argument("--by", choices=("subject", "level"), default=None, help="also report per subset")
    ap.add_argument("--json", type=Path, default=None, help="write every claim verdict here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every refuted step")
    args = ap.parse_args(argv)

    if args.fetch:
        for name in SOURCES:
            print(f"{name}: {fetch(name, args.cache)} sha256 ok")
        return 0
    if args.resample:
        records, out = build("sample", args.cache)
        DATA.write_text(dump(records), encoding="utf-8")
        print(f"{DATA}: {len(records)} records, sha256 {sha256_of(DATA)}; excluded: {dict(out)}")
        return 0
    if args.population == "sample":
        records = load()
    else:
        records, out = build(args.population, args.cache)
        print(f"population {args.population}: {len(records)} records; excluded: {dict(out)}")

    results = []
    for rec in records:
        r = run_solution(rec, args.budget)
        results.append(r)
        if args.verbose:
            for st in r.steps:
                if st.status == "REFUTED":
                    c = next(c for c in st.claims if c.status == "REFUTED")
                    print(f"{r.id} step {st.index} (rating {st.rating}, first error {r.first_error}): "
                          f"{c.left} = {c.right} -> {c.detail}")
    print(report(score(results), args.population))
    if args.by:
        groups: dict[object, list[SolutionResult]] = {}
        for r in results:
            groups.setdefault(getattr(r, args.by), []).append(r)
        for key in sorted(groups, key=str):
            print()
            print(report(score(groups[key]), f"{args.by} = {key}"))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
