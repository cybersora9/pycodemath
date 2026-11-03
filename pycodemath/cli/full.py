"""The FULL-series commands (I1) — one grammar for nine modules.

``repl`` merges this table into its own, so every line below is reachable from
the REPL, the one-shot ``python -m pycodemath "<line>"``, and MCP ``math_eval``
(the code pillar also has its own typed MCP tools, see ``mcp_server``). Each
handler returns an ``Outcome``: the text a human reads, the module's own
``to_dict()``, and an X1 certificate that ``recheck`` re-runs from the file alone.

THE GRAMMAR RULES (I1 report, "one rule per conflict"):

1. The first word is the command. A word that names ONE operation everywhere
   (``factor``, ``isprime``, ``convert``, ``stability``, ``ranges``) is the
   command itself; a word that is ambiguous on its own (``round``, ``split``,
   ``loan``) gets its domain as a prefix: ``money round``, ``money split``.
2. Options trail the command as ``key value`` pairs (``budget 5``, ``mode
   half_even``, ``digits 50``), flags as bare words (``deg``, ``normalize``,
   ``int``); a number type is always ``as <dtype>``; variable ranges are always
   ``for x in [a, b], y in [c, d]``; units are always ``name:unit``.
3. An unknown command is a ``ParseError`` with "did you mean" (K1); a known
   command whose line does not fit its syntax answers with its usage — except a
   line that reads as an expression (``mean + 1``, ``power(2, 3)``), which still
   goes to the expression evaluator, so a variable named like a command keeps
   working.
4. Output: the answer first, then indented detail, then ``certificate: <STATUS>
   (<method>) — <detail>`` — the same line ``certify`` prints.
"""

from __future__ import annotations

import contextvars
import json
import re
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import sympy as sp

from ..core.errors import DomainError, ParseError
from ..frontend.parser import parse
from ..verify.certificate import (
    Certificate,
    certify_op,
    code_verdict,
    domain_spec,
    domain_text,
    from_verdict,
    issue,
    range_verdict,
    stability_args,
)
from ..verify.verdict import Verdict

__all__ = ["Outcome", "COMMANDS", "NUMERIC_LOCK", "check_code_line", "convert_line"]

#: C2 and C3 set mpmath's GLOBAL interval precision (``iv.prec``) and restore it;
#: two runs interleaved in threads would compute at each other's precision — a
#: silently wrong bound. Every C2/C3 entry point on every surface takes this lock.
#: (Measured: FastMCP of mcp 1.30 runs sync tools inline on its event loop, so they are
#: serialized already; the lock keeps that true under any other host.)
NUMERIC_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class Outcome:
    """One FULL-series line answered: ``text`` for a human, ``kind`` (the
    certificate kind), ``result`` (the module's ``to_dict()``) and ``certificate``
    (X1; ``None`` where nothing is claimed)."""

    text: str
    kind: str
    result: "dict[str, Any] | None" = None
    certificate: "Certificate | None" = None
    verdict: "Verdict | None" = None

    @property
    def status(self) -> "str | None":
        if self.certificate is None:
            return None
        return str(self.certificate.verdict["status"])


def _cert_line(cert: "Certificate | None") -> str:
    if cert is None:
        return "certificate: none (nothing to re-check)"
    v = cert.verdict
    return f"certificate: {str(v['status']).upper()} ({v['method']}) — {v['detail']}"


def _verdict_line(v: Verdict) -> str:
    return f"{v.status.name} ({v.method}) — {v.detail}"


# --- small readers ---------------------------------------------------------------
def _exact_text(raw: str, what: str) -> str:
    """A number as exact text: ``2^31-1`` -> ``2147483647``, ``1e-3`` -> ``1/1000``."""
    raw = raw.strip()
    try:
        return str(Fraction(Decimal(raw)))
    except (InvalidOperation, ValueError):
        pass
    try:
        value = parse(raw).sy
    except Exception as exc:  # noqa: BLE001 — any unreadable bound is a ParseError
        raise ParseError(f"{what}: {raw!r} is not a number") from exc
    value = sp.nsimplify(value) if isinstance(value, sp.Float) else value
    if not (isinstance(value, sp.Expr) and value.is_number and value.is_real):
        raise ParseError(f"{what}: {raw!r} is not a real number")
    if not value.is_Rational:
        value = sp.Rational(float(value))
    return str(value)


_RANGE = re.compile(
    r"(?P<name>[A-Za-z_]\w*)\s+in\s+\[\s*(?P<lo>[^\],]+?)\s*,\s*(?P<hi>[^\]]+?)\s*\]"
    r"(?P<int>\s+int\b)?"
)


def _ranges(text: str) -> "list[tuple[str, str, str, bool]]":
    """``x in [a, b], y in [c, d] int`` -> ``[(name, lo, hi, is_int)]`` (exact text)."""
    out = []
    pos = 0
    for m in _RANGE.finditer(text):
        gap = text[pos:m.start()].strip().strip(",").strip()
        if gap:
            raise ParseError(f"cannot read {gap!r} in the ranges (expected: x in [a, b], y in [c, d])")
        out.append((m["name"], _exact_text(m["lo"], m["name"]), _exact_text(m["hi"], m["name"]),
                    bool(m["int"])))
        pos = m.end()
    if text[pos:].strip().strip(",").strip() or not out:
        raise ParseError(f"cannot read the ranges {text.strip()!r} (expected: x in [a, b], y in [c, d])")
    return out


def _opts(raw: "str | None", allowed: "dict[str, Callable[[str], Any]]",
          flags: "tuple[str, ...]" = ()) -> dict[str, Any]:
    """Trailing ``key value`` pairs and bare ``flags`` — unknown or repeated is a ParseError."""
    words = (raw or "").split()
    out: dict[str, Any] = {}
    i = 0
    while i < len(words):
        key = words[i].lower()
        if key in flags:
            value: Any = True
            i += 1
        elif key in allowed and i + 1 < len(words):
            value = allowed[key](words[i + 1])
            i += 2
        else:
            known = [f"{k} <value>" for k in allowed] + list(flags)
            raise ParseError(f"unknown option {words[i]!r} — this command takes: {', '.join(known) or 'none'}")
        if key in out:
            raise ParseError(f"option {key!r} given twice")
        out[key] = value
    return out


def _budget(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ParseError(f"invalid number: {raw!r}") from exc
    if not 0 < value < float("inf"):
        raise DomainError(f"the budget must be a positive finite number of seconds, got {raw}")
    return value


def _int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ParseError(f"invalid integer: {raw!r}") from exc


def _word(choices: "tuple[str, ...]") -> Callable[[str], str]:
    def read(raw: str) -> str:
        if raw.lower() not in choices:
            raise ParseError(f"expected one of {', '.join(choices)}, got {raw!r}")
        return raw.lower()
    return read


_DTYPES = ("float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64",
           "uint8", "uint16", "uint32", "uint64")


# --- C1: verify code ----------------------------------------------------------------
_CODE = re.compile(
    r"^(?P<path>\S+?\.py)(?::(?P<func>[A-Za-z_]\w*))?\s*==\s*(?P<formula>.+?)"
    r"(?:\s+where\s+(?P<domain>.+?))?(?P<emit>\s+--emit-test)?(?:\s+budget\s+(?P<budget>\S+))?\s*$",
    re.IGNORECASE,
)
CODE_USAGE = (
    "Usage: verify code <file.py>[:<function>] == <formula> [where <param>: <spec>; ...]"
    " [--emit-test] [budget <s>]   (e.g. verify code sums.py:sum_to == n*(n+1)/2"
    " where n: integer, n>=0)"
)


def check_code_line(text: str, *, source_line: "str | None" = None) -> Outcome:
    """``<file.py>[:<func>] == <formula> [where ...] [--emit-test] [budget s]`` —
    the part of a ``verify code`` line after ``code``. The file is the caller's
    own, so its function IS executed (in the isolated worker) to confirm a
    refutation; source from outside goes through MCP ``math_check_code``."""
    from ..code import frontend

    m = _CODE.match(text.strip())
    if m is None:
        return Outcome(CODE_USAGE, "code_equiv")
    path = Path(m["path"])
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DomainError(f"verify code: cannot read {str(path)!r} ({exc.strerror})") from exc
    domain = domain_spec(m["domain"]) if m["domain"] else {}
    budget = _budget(m["budget"]) if m["budget"] else None
    result = frontend.check_code(source, m["formula"], name=m["func"], domain=domain or None,
                                 budget=budget)
    verdict = code_verdict(result)
    cert = _quiet(lambda: from_verdict(
        "code_equiv", [source, m["formula"], result.function_name, domain_text(result.domain)],
        verdict, source=source_line, budget=budget, execute=True))
    lines = [_verdict_line(verdict)]
    if result.symbolic_form:
        lines.append(f"  reading: {result.function_name}({', '.join(result.params)}) = {result.symbolic_form}")
    lines += [f"  note: {n.message}" for n in result.notes]
    if m["emit"]:
        if result.refuted:
            lines.append(frontend.emit_test(result, module=path.stem))
        else:
            lines.append("  --emit-test: nothing to pin (no counterexample)")
    return Outcome("\n".join(lines), "code_equiv", result.to_dict(), cert, verdict)


def _quiet(build: Callable[[], Certificate]) -> "Certificate | None":
    """A certificate is a by-product: when the claim cannot be recorded, the
    answer still stands (the same rule as ``repl._written_down``)."""
    try:
        return build()
    except DomainError:
        return None


# --- C2: stability / improve ---------------------------------------------------------
_FOR = r"\s+for\s+(?P<ranges>[A-Za-z_]\w*\s+in\s+\[.+\](?:\s+int)?)"


def _float_ranges(text: str) -> dict[str, tuple[float, float]]:
    out = {}
    for name, lo, hi, is_int in _ranges(text):
        if is_int:
            raise ParseError("stability: 'int' marks integer inputs for 'ranges', not here")
        out[name] = (float(Fraction(lo)), float(Fraction(hi)))
    return out


_STAB_OPTS = {"as": _word(("float64", "float32")), "samples": _int, "budget": _budget}


def _cmd_stability(m: re.Match) -> Outcome:
    from ..code import stability as S

    o = _opts(m["opts"], _STAB_OPTS)
    with NUMERIC_LOCK:
        report = S.diagnose(m["expr"], _float_ranges(m["ranges"]), dtype=o.get("as", "float64"),
                            samples=o.get("samples", 256), budget=o.get("budget"))
    lines = [f"{report.expression} [{report.dtype}]: {report.verdict}"]
    if report.worst is not None:
        w = report.worst
        where = ", ".join(f"{k} = {v}" for k, v in w.inputs.items())
        lines.append(f"  worst: at {where} computes {w.computed}, exact {w.exact} ({w.bits:.1f} bits)")
    for c in report.culprits[:3]:
        lines.append(f"  culprit: {c.expression} — {c.cause} ({c.mean_bits:.1f} bits on average); "
                     f"try {c.hint}")
    args = stability_args(report)
    cert = None if args is None else _quiet(lambda: issue("stability", *args, source=m.string.strip()))
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "stability", report.to_dict(), cert)


_IMPROVE_OPTS = {**_STAB_OPTS, "emit": _word(("python", "numpy", "c"))}


def _cmd_improve(m: re.Match) -> Outcome:
    from ..code import stability as S

    o = _opts(m["opts"], _IMPROVE_OPTS)
    with NUMERIC_LOCK:
        imp = S.improve(m["expr"], _float_ranges(m["ranges"]), dtype=o.get("as", "float64"),
                        samples=o.get("samples", 256), budget=o.get("budget"))
    lines = [imp.message]
    cert = None
    best = imp.best
    if best is not None:
        p = best.proof
        cond = f" if {' and '.join(p.conditions)}" if p.conditions else ""
        lines.append(f"  proof: {p.status} ({p.method}){cond} — {p.detail}")
        lang = o.get("emit")
        if lang:
            lines.append({"python": best.to_python, "numpy": best.to_numpy, "c": best.to_c}[lang]())
        if p.method == "check_equal" and not p.conditions:
            cert = _quiet(lambda: issue("identity", imp.expression, best.expression,
                                        source=m.string.strip()))
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "identity" if cert else "stability", imp.to_dict(), cert)


# --- C3: ranges / maxsafe -------------------------------------------------------------
_RANGES_OPTS = {"as": _word(_DTYPES), "budget": _budget}


def _range_spec(text: str) -> dict[str, list[Any]]:
    spec: dict[str, list[Any]] = {}
    for name, lo, hi, is_int in _ranges(text):
        spec[name] = [lo, hi, "int"] if is_int else [lo, hi]
    return check_bounds(spec)


#: Beyond this no float type has a finite value. C3 formats such a bound with
#: ``float()`` inside its own refusal and dies of OverflowError (I1 adversary
#: round, "do poprawki w C3"); the adapter refuses it first, typed.
_FLOAT_LIMIT = Fraction(2) ** 1024


def check_bounds(spec: "dict[str, list[Any]]") -> "dict[str, list[Any]]":
    for name, pair in spec.items():
        for bound in pair[:2]:
            try:
                value = Fraction(str(bound))
            except (ValueError, ZeroDivisionError):
                continue  # C3 reads (and refuses) it with its own message
            if abs(value) >= _FLOAT_LIMIT:
                raise DomainError(f"ranges: a bound of {name!r} is beyond every finite float "
                                  f"(|bound| >= 2^1024)")
    return spec


def _cmd_ranges(m: re.Match) -> Outcome:
    from ..code import ranges as R

    o = _opts(m["opts"], _RANGES_OPTS)
    if "as" not in o:
        raise ParseError("ranges: name the number type — '... as float32' (or int32, float16, bfloat16...)")
    spec = _range_spec(m["ranges"])
    with NUMERIC_LOCK:
        report = R.analyze(m["expr"], spec, o["as"], budget=o.get("budget"))
    verdict = range_verdict(report)
    from ..verify.certificate import _jtext

    cert = _quiet(lambda: from_verdict("range", [m["expr"], _jtext(spec), o["as"]], verdict,
                                       source=m.string.strip(), budget=o.get("budget")))
    lines = [str(report)]
    for a in report.advice:
        lines.append(f"  advice proof: {a.rewrite} — {a.equivalence} ({a.method})")
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "range", report.to_dict(), cert, verdict)


def _cmd_maxsafe(m: re.Match) -> Outcome:
    from ..code import ranges as R

    o = _opts(m["opts"], _RANGES_OPTS)
    if "as" not in o:
        raise ParseError("maxsafe: name the number type — 'maxsafe exp(x) for x as float32'")
    others = _range_spec(m["others"]) if m["others"] else None
    with NUMERIC_LOCK:
        safe = R.max_safe_input(m["expr"], m["var"], o["as"], others, budget=o.get("budget"))
    if safe.safe_up_to is None:
        lines = [f"no {m['var']} from {safe.start} upward is safe in {o['as']} ({safe.method})"]
    else:
        lines = [f"{m['var']} <= {float(Fraction(safe.safe_up_to))!r} is safe in {o['as']}"
                 f" (exact bound {safe.safe_up_to}; {safe.method})"]
    if safe.failure is not None:
        lines.append(f"  first failure: {safe.failure.detail}")
    lines.append(f"  {safe.detail}")
    cert = None
    if safe.safe_up_to is not None:  # the safe interval, certified as a range claim
        from ..verify.certificate import _jtext

        spec = {**(others or {}), m["var"]: [safe.start, safe.safe_up_to]}
        with NUMERIC_LOCK:
            cert = _quiet(lambda: issue("range", m["expr"], _jtext(spec), o["as"],
                                        source=m.string.strip(), budget=o.get("budget")))
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "range", _todict(safe), cert)


def _todict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # type: ignore[no-any-return]
    import dataclasses

    def plain(v: Any) -> Any:
        if hasattr(v, "to_dict"):
            return v.to_dict()
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v if v is None or isinstance(v, (str, int, float, bool)) else str(v)

    return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


# --- C4: money ---------------------------------------------------------------------
def _rate(raw: str) -> str:
    """``7.2%`` -> ``9/125``; ``0.072`` -> ``9/125`` (exact)."""
    raw = raw.strip()
    pct = raw.endswith("%")
    try:
        value = Fraction(Decimal(raw.rstrip("%")))
    except (InvalidOperation, ValueError) as exc:
        raise ParseError(f"money: {raw!r} is not a rate (write 23% or 0.23)") from exc
    return str(value / 100 if pct else value)


def _amounts(raw: str) -> list[str]:
    parts = [p for p in re.split(r"[,\s]+", raw.strip()) if p]
    for p in parts:
        try:
            Decimal(p)
        except InvalidOperation as exc:
            raise ParseError(f"money: {p!r} is not an amount") from exc
    return parts


_MODE = _word(("half_up", "half_even", "half_down", "up", "down", "ceiling", "floor", "05up"))
_MONEY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("round", re.compile(r"^round\s+(?P<amount>\S+)\s+(?P<cur>[A-Za-z]{3}|\d+)(?P<opts>(?:\s+\S+\s+\S+)*)$", re.I)),
    ("split", re.compile(r"^split\s+(?P<amount>\S+)\s+(?P<cur>[A-Za-z]{3}|\d+)\s+(?:into\s+(?P<n>\d+)|by\s+(?P<w>[\d.:/]+))"
                         r"(?P<opts>(?:\s+\S+\s+\S+)*)$", re.I)),
    ("vat", re.compile(r"^vat\s+(?P<lines>[\d.,\s]+?)\s+at\s+(?P<rate>\S+)\s+(?P<cur>[A-Za-z]{3}|\d+)"
                       r"(?P<opts>(?:\s+\S+\s+\S+)*)$", re.I)),
    ("float", re.compile(r"^float\s+(?P<expr>.+?)\s+for\s+(?P<var>[A-Za-z_]\w*)\s+in\s+(?P<lo>[-\d.]+)\.\.(?P<hi>[-\d.]+)"
                         r"(?P<opts>(?:\s+\S+\s+\S+)*)$", re.I)),
    ("interest", re.compile(r"^interest\s+(?P<amount>\S+)\s+(?P<cur>[A-Za-z]{3}|\d+)\s+at\s+(?P<rate>\S+)\s+for\s+"
                            r"(?P<n>\d+)\s+(?P<unit>months|years|periods)(?P<opts>(?:\s+\S+\s+\S+)*)$", re.I)),
    ("loan", re.compile(r"^loan\s+(?P<amount>\S+)\s+(?P<cur>[A-Za-z]{3}|\d+)\s+at\s+(?P<rate>\S+)\s+for\s+"
                        r"(?P<n>\d+)\s+(?P<unit>months|years)"
                        r"(?P<flags>(?:\s+(?:equal_principal|schedule|monthly|yearly|annual|annually))*)"
                        r"(?P<opts>(?:\s+\S+\s+\S+)*)$", re.I)),
    ("npv", re.compile(r"^npv\s+(?P<rate>\S+)\s+(?P<flows>[-\d.,\s]+?)(?:\s+currency\s+(?P<cur>[A-Za-z]{3}))?$", re.I)),
    ("irr", re.compile(r"^irr\s+(?P<flows>[-\d.,\s]+)$", re.I)),
]
MONEY_USAGE = """Usage: money round <amount> <CUR> [mode half_even] [step 0.05]
       money split <amount> <CUR> into <n> | by 1:2:3   [method hamilton|in_order]
       money vat <net1,net2,...> at <rate>% <CUR> [mode half_up]
       money float <float expression> for <x> in <lo>..<hi> [step 0.01] [currency <CUR>] [mode half_up]
       money interest <amount> <CUR> at <rate>% for <n> months|years|periods [mode ...]
           (compounded once per month / year / period)
       money loan <amount> <CUR> at <rate>% for <n> months|years [yearly] [equal_principal] [schedule]
           (paid monthly: "for 25 years" = 300 monthly instalments; "yearly" = one a year)
       money npv <rate>% <c0,c1,...>   |   money irr <c0,c1,...>
   (e.g. money split 100 PLN into 3)"""


def _cmd_money(m: re.Match) -> Outcome:
    rest = m["rest"].strip()
    for op, pattern in _MONEY_PATTERNS:
        mm = pattern.match(rest)
        if mm is not None:
            return _money(op, mm, m.string.strip())
    return Outcome(MONEY_USAGE, "money")


def _money(op: str, m: re.Match, source: str) -> Outcome:
    groups = m.groupdict()
    opts_raw = groups.get("opts")
    if op == "round":
        o = _opts(opts_raw, {"mode": _MODE, "step": str})
        inputs: dict[str, Any] = {"value": m["amount"], "currency": m["cur"].upper(),
                                  "mode": o.get("mode", "half_up").upper(), "step": o.get("step")}
    elif op == "split":
        o = _opts(opts_raw, {"method": _word(("hamilton", "in_order", "largest_remainder"))})
        parts: Any = int(m["n"]) if m["n"] else [p for p in m["w"].split(":") if p]
        inputs = {"total": m["amount"], "parts": parts, "currency": m["cur"].upper(),
                  "method": o.get("method", "hamilton")}
    elif op == "vat":
        o = _opts(opts_raw, {"mode": _MODE})
        inputs = {"lines": _amounts(m["lines"]), "rate": _rate(m["rate"]), "currency": m["cur"].upper(),
                  "mode": o.get("mode", "half_up").upper()}
    elif op == "float":
        o = _opts(opts_raw, {"step": str, "currency": str, "mode": _MODE})
        expr = m["expr"].strip().strip("'\"")
        inputs = {"expr": expr, "grid": {m["var"]: [m["lo"], m["hi"], o.get("step", "0.01")]},
                  "currency": o["currency"].upper() if "currency" in o else None,
                  "mode": o.get("mode", "half_even").upper()}
    elif op == "interest":
        o = _opts(opts_raw, {"mode": _MODE})
        per_year = {"months": 12, "years": 1, "periods": 1}[m["unit"].lower()]
        inputs = {"principal": m["amount"], "rate": _rate(m["rate"]), "periods": int(m["n"]),
                  "currency": m["cur"].upper(), "mode": o.get("mode", "half_up").upper(), "per_year": per_year}
    elif op == "loan":
        o = _opts(opts_raw, {"mode": _MODE})
        flags = (m["flags"] or "").lower().split()
        # 0.5.1: "for 25 years" is the DURATION of the loan, paid monthly (300
        # instalments) — what a borrower means. 0.5.0 read it as 25 annual
        # instalments. Annual instalments are asked for explicitly with "yearly".
        yearly = any(f in ("yearly", "annual", "annually") for f in flags)
        if yearly and "monthly" in flags:
            raise ParseError("money loan: choose one of 'monthly' and 'yearly'")
        n, unit = int(m["n"]), m["unit"].lower()
        if yearly and unit == "months":
            raise ParseError("money loan: 'yearly' instalments need a duration in years "
                             f"(for {n} months is paid monthly)")
        per_year = 1 if yearly else 12
        inputs = {"principal": m["amount"], "rate": _rate(m["rate"]),
                  "periods": n * 12 if unit == "years" and not yearly else n,
                  "currency": m["cur"].upper(), "mode": o.get("mode", "half_up").upper(),
                  "per_year": per_year,
                  "schedule": "equal_principal" if "equal_principal" in flags else "annuity"}
    elif op == "npv":
        inputs = {"rate": _rate(m["rate"]), "cashflows": _amounts(m["flows"]),
                  "currency": m["cur"].upper() if m["cur"] else None}
    else:  # irr
        inputs = {"cashflows": _amounts(m["flows"])}
    value, cert = certify_op("money", "allocate" if op == "split" else op, inputs, source=source)
    lines = _money_text(op, value, groups)
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "money", value, cert)


def _money_text(op: str, v: dict[str, Any], g: dict[str, Any]) -> list[str]:
    cur = v.get("currency", "")
    if op == "round":
        return [f"{v['rounded']} {cur}  ({v['mode']}, error {v['error']})"]
    if op == "split":
        shares = " + ".join(v["shares"])
        return [f"{shares} = {v['total']} {cur}  ({v['method']}; sums exactly: {v['sums_exactly']})"]
    if op == "vat":
        out = [f"VAT per line {v['per_line']} {cur}, on the total {v['on_total']} {cur}"
               f" (difference {v['difference']})"]
        if v.get("always_equal") is False and v.get("witness_lines"):
            out.append(f"  they can differ: lines {', '.join(v['witness_lines'])} give "
                       f"{v['witness_per_line']} vs {v['witness_on_total']}; at most "
                       f"{v['proven_bound']} for this many lines (proved)")
        return out
    if op == "float":
        out = [v["explanation"]]
        if v.get("witness"):
            where = ", ".join(f"{k} = {x}" for k, x in v["witness"].items())
            out.append(f"  witness: at {where} floats give {v['float_value']}, exact {v['exact_value']}"
                       f" ({v['hazards']} of {v['checked']} inputs wrong)")
        return out
    if op == "interest":
        # The unit is the COMPOUNDING period (and the duration counts those periods):
        # say so, so "for 10 years" is not mistaken for monthly capitalisation.
        unit = (g.get("unit") or "periods").lower()
        every = {"months": "monthly", "years": "yearly", "periods": "once per period"}[unit]
        return [f"{v['once']} {cur} rounded once, {v['stepwise']} rounded every period"
                f" (difference {v['difference']}); compounded {every}, {v['periods']} {unit}"]
    if op == "loan":
        flags = (g.get("flags") or "").lower().split()
        often = "yearly" if any(f in ("yearly", "annual", "annually") for f in flags) else "monthly"
        out = [f"{v['periods']} {often} instalments: payment {v['regular_payment']} {cur}, last "
               f"{v['last_payment']}; total paid "
               f"{v['total_paid']}, interest {v['total_interest']} (instalment x count differs by "
               f"{v['naive_difference']})"]
        if "schedule" in (g.get("flags") or "").lower():
            out += [f"  {r['period']:>4}  {r['payment']:>12}  {r['interest']:>12}  {r['principal']:>12}"
                    f"  {r['balance']:>14}" for r in v["rows"]]
        return out
    if op == "npv":
        rounded = f" (rounded {v['rounded']})" if v.get("rounded") else ""
        return [f"NPV = {v['exact']}{rounded}; Excel's NPV() convention gives {v['excel_convention']}"]
    rates = ", ".join(f"{float(Fraction(r)) * 100:.10g}%" for r in v["rates"]) or "none"
    unique = "unique" if v["unique"] else f"{v['count']} rates — the IRR is not unique"
    return [f"IRR: {rates} ({unique}; exact brackets {v['brackets']})"]


# --- K3: calculator ------------------------------------------------------------------
def _calc(op: str, args: list[Any], kwargs: "dict[str, Any] | None" = None, *, source: str) -> Outcome:
    inputs = {"args": args, "kwargs": kwargs or {}}
    value, cert = certify_op("calc", op, inputs, source=source)
    lines = [_calc_value(op, value)]
    if value.get("notes"):
        lines += [f"  note: {n}" for n in value["notes"]]
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "calc", value, cert)


def _calc_value(op: str, v: dict[str, Any]) -> str:
    val = v["value"]
    if op == "factorint" and isinstance(val, dict):
        text = " * ".join(f"{p}^{e}" if e != 1 else str(p) for p, e in val.items()) or "1"
    elif isinstance(val, dict):
        text = ", ".join(f"{k} = {x}" for k, x in val.items())
    elif isinstance(val, list):
        text = "[" + ", ".join(str(x) for x in val) + "]"
    elif isinstance(val, bool):
        text = str(val)
    elif val is None:
        text = "no solution"
    else:
        text = str(val)
    if v.get("approx"):
        text += f" ≈ {v['approx']}"
    return text


_INT_OPS = {
    "factor": ("factorint", 1), "isprime": ("isprime", 1), "nextprime": ("nextprime", 1),
    "prevprime": ("prevprime", 1), "gcd": ("gcd", -1), "lcm": ("lcm", -1), "powmod": ("powmod", 3),
    "modinv": ("modinv", 2), "totient": ("totient", 1), "divisors": ("divisors", 1),
    "sqrtmod": ("sqrt_mod", 2), "fibonacci": ("fibonacci", 1), "lucas": ("lucas", 1),
    "legendre": ("legendre", 2), "jacobi": ("jacobi", 2), "binomial": ("binomial", 2),
    "permutations": ("permutations", 2), "combinations": ("combinations", 2),
    "stirling1": ("stirling1", 2), "stirling2": ("stirling2", 2), "bell": ("bell", 1),
    "catalan": ("catalan", 1), "partitions": ("partitions", 1),
}


_INT_EXAMPLES = {
    "factor": "2^64+1", "isprime": "2^61-1", "nextprime": "2^64", "prevprime": "2^64",
    "gcd": "1071 462", "lcm": "4 6 10", "powmod": "4 13 497", "modinv": "17 3120",
    "totient": "36", "divisors": "360", "sqrtmod": "10 13", "fibonacci": "100", "lucas": "50",
    "legendre": "5 11", "jacobi": "1001 9907", "binomial": "1000 500", "permutations": "10 3",
    "combinations": "10 3", "stirling1": "10 3", "stirling2": "10 3", "bell": "50",
    "catalan": "100", "partitions": "1000",
}


def _split_args(raw: str) -> list[str]:
    """Space- or comma-separated arguments; an expression keeps its own commas
    inside parentheses (``gcd 2^64-1, 3^40-1``)."""
    out, depth, cur = [], 0, ""
    for ch in raw.strip():
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and (ch == "," or ch.isspace()):
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _int_command(word: str) -> Callable[[re.Match], Outcome]:
    op, arity = _INT_OPS[word]

    def handler(m: re.Match) -> Outcome:
        args = _split_args(m["args"])
        if (arity > 0 and len(args) != arity) or (arity < 0 and len(args) < 2):
            want = f"{arity} argument(s)" if arity > 0 else "at least 2 arguments"
            raise ParseError(f"{word}: takes {want}, got {len(args)}")
        return _calc(op, args, source=m.string.strip())
    return handler


def _cmd_mod(m: re.Match) -> Outcome:
    conv = (m["conv"] or "floor").lower()
    return _calc("mod", [m["a"], m["m"]], {"convention": conv}, source=m.string.strip())


def _cmd_crt(m: re.Match) -> Outcome:
    return _calc("crt", [_split_args(m["r"]), _split_args(m["m"])], source=m.string.strip())


def _cmd_special(m: re.Match) -> Outcome:
    dps = int(m["digits"]) if m["digits"] else 15
    return _calc("special", [m["name"], *_split_args(m["args"])], {"dps": dps}, source=m.string.strip())


def _cmd_describe(m: re.Match) -> Outcome:
    word = m["word"].lower()
    op = {"var": "variance", "std": "std", "mean": "mean", "median": "median"}[word]
    kwargs = {"kind": m["kind"].lower()} if m["kind"] else {}
    if word in ("mean", "median") and kwargs:
        raise ParseError(f"{word}: takes no sample|population")
    return _calc(op, [_split_args(m["data"])], kwargs, source=m.string.strip())


def _cmd_quantile(m: re.Match) -> Outcome:
    method: Any = int(m["type"]) if m["type"] and m["type"].isdigit() else (m["type"] or 7)
    return _calc("quantile", [_split_args(m["data"]), m["q"]], {"method": method}, source=m.string.strip())


def _kv(raw: "str | None") -> dict[str, str]:
    out = {}
    for part in (raw or "").split():
        k, sep, v = part.partition("=")
        if not sep or not k.isidentifier():
            raise ParseError(f"expected key=value, got {part!r}")
        out[k] = v
    return out


def _cmd_dist(m: re.Match) -> Outcome:
    kwargs: dict[str, Any] = _kv(m["params"])
    if "digits" in kwargs:
        kwargs["dps"] = int(kwargs.pop("digits"))
    return _calc(m["fn"].lower(), [m["dist"], m["x"]], kwargs, source=m.string.strip())


def _cmd_samplesize(m: re.Match) -> Outcome:
    kw: dict[str, Any] = _kv(m["params"])
    if "p1" not in kw:
        raise ParseError("samplesize: p1=<baseline rate> is required (e.g. samplesize p1=0.10 p2=0.12)")
    p1 = kw.pop("p1")
    if m["relative"]:
        kw["relative"] = True
    return _calc("ab_sample_size", [p1], kw, source=m.string.strip())


def _cmd_power(m: re.Match) -> Outcome:
    kw: dict[str, Any] = _kv(m["params"])
    try:
        args = [kw.pop("p1"), kw.pop("p2"), kw.pop("n")]
    except KeyError as exc:
        raise ParseError("power: p1=, p2= and n= are required (e.g. power p1=0.10 p2=0.12 n=3623)") from exc
    return _calc("ab_power", args, kw, source=m.string.strip())


def _cmd_ci(m: re.Match) -> Outcome:
    op = {"wilson": "wilson_interval", "exact": "clopper_pearson"}[m["how"].lower()]
    kw = _kv(m["params"])
    return _calc(op, [m["s"], m["n"]], kw, source=m.string.strip())


def _cmd_ztest(m: re.Match) -> Outcome:
    return _calc("two_proportion_ztest", [m["s1"], m["n1"], m["s2"], m["n2"]], source=m.string.strip())


def _cmd_welch(m: re.Match) -> Outcome:
    return _calc("welch_ttest", [_split_args(m["a"]), _split_args(m["b"])], source=m.string.strip())


def _cmd_adjust(m: re.Match) -> Outcome:
    return _calc("adjust_pvalues", [_split_args(m["p"])], {"method": m["word"].lower()},
                 source=m.string.strip())


# --- J1: units ------------------------------------------------------------------------
def convert_line(quantity: str, target: str, *, source: "str | None" = None) -> Outcome:
    """``5 km/h`` to ``m/s`` — exact, with the conversion's own certificate."""
    from ..engine import units as U

    conv = U.convert(quantity.strip(), target.strip())
    cert = _quiet(lambda: from_verdict("unit_conversion", [quantity.strip(), target.strip(),
                                                           conv.value_text], conv.as_verdict(),
                                       source=source))
    lines = [str(conv)] + [f"  note: {n}" for n in conv.notes] + [_cert_line(cert)]
    return Outcome("\n".join(lines), "unit_conversion", conv.to_dict(), cert)


def _cmd_convert(m: re.Match) -> Outcome:
    return convert_line(m["q"], m["to"], source=m.string.strip())


def _cmd_units(m: re.Match) -> Outcome:
    from ..engine import units as U

    values = "numbers" if m["numbers"] else "quantities"
    check = U.check_dimensions(m["eq"], units=U.read_units_spec(m["spec"]), values=values)
    verdict = check.as_verdict()
    cert = _quiet(lambda: from_verdict("dimensions", [m["eq"].strip(), m["spec"].strip()], verdict,
                                       source=m.string.strip(), values=values))
    lines = [_verdict_line(verdict)]
    if check.left_dimension or check.right_dimension:
        lines.append(f"  left: {check.left_dimension}; right: {check.right_dimension}")
    lines += [f"  warning: {w.message}" for w in check.warnings]
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "dimensions", check.to_dict(), cert, verdict)


def _cmd_dims(m: re.Match) -> Outcome:
    from ..engine import units as U

    values = "numbers" if m["numbers"] else "quantities"
    rep = U.dimension_of(m["expr"], U.read_units_spec(m["spec"]) if m["spec"] else None, values=values)
    d = rep.to_dict()
    lines = [str(rep)] + [f"  warning: {w['message']}" for w in d.get("warnings", [])]
    lines.append(_cert_line(None))
    return Outcome("\n".join(lines), "dimensions", d, None)


# --- G1: geometry ---------------------------------------------------------------------
def _point(raw: str) -> list[str]:
    parts = [p for p in re.split(r"[,\s()]+", raw.strip()) if p]
    for p in parts:
        _exact_text(p, "coordinate")
    return parts


def _points(raw: str, count: int, dim: int) -> list[list[str]]:
    pts = [_point(p) for p in re.findall(r"\(?\s*[^()\s]+(?:\s*,\s*[^()\s,]+)+\s*\)?", raw)]
    if len(pts) != count or any(len(p) != dim for p in pts):
        raise ParseError(f"expected {count} points of {dim} coordinates (x,y{',z' if dim == 3 else ''})")
    return pts


def _geometry(op: str, inputs: dict[str, Any], source: str, text: Callable[[dict], str]) -> Outcome:
    value, cert = certify_op("geometry", op, inputs, source=source)
    return Outcome("\n".join([text(value), _cert_line(cert)]), "geometry", value, cert)


def _matrix_text(rows: Any) -> str:
    return "\n".join("  [" + ", ".join(f"{float(x): .16g}" if not isinstance(x, str) else x for x in r) + "]"
                     for r in rows)


def _cmd_quat2mat(m: re.Match) -> Outcome:
    o = _opts(m["opts"], {"order": _word(("wxyz", "xyzw")), "algebra": _word(("hamilton", "jpl"))},
              ("normalize",))
    inputs = {"q": _point(m["q"]), "order": o.get("order", "wxyz"), "algebra": o.get("algebra", "hamilton"),
              "normalize": bool(o.get("normalize"))}
    return _geometry("quat2mat", inputs, m.string.strip(), lambda v: _matrix_text(v["value"]))


def _euler_opts(raw: "str | None") -> dict[str, Any]:
    o = _opts(raw, {"seq": str, "frame": _word(("intrinsic", "extrinsic"))}, ("deg",))
    if "seq" not in o or "frame" not in o:
        raise ParseError("name the convention: 'seq zyx frame intrinsic' (24 conventions: any seq, either frame)")
    return o


def _cmd_euler2mat(m: re.Match) -> Outcome:
    o = _euler_opts(m["opts"])
    inputs = {"angles": _point(m["angles"]), "seq": o["seq"], "frame": o["frame"], "degrees": bool(o.get("deg"))}
    return _geometry("euler2mat", inputs, m.string.strip(), lambda v: _matrix_text(v["value"]))


def _cmd_mat2euler(m: re.Match) -> Outcome:
    o = _euler_opts(m["opts"])
    try:
        matrix = json.loads(m["matrix"])
    except json.JSONDecodeError as exc:
        raise ParseError("mat2euler: write the matrix as [[a,b,c],[d,e,f],[g,h,i]]") from exc
    inputs = {"matrix": matrix, "seq": o["seq"], "frame": o["frame"], "degrees": bool(o.get("deg"))}

    def text(v: dict) -> str:
        lock = "  (gimbal lock)" if v.get("gimbal_lock") else ""
        return "angles: " + ", ".join(f"{float(x):.16g}" for x in v["value"]) + lock
    return _geometry("mat2euler", inputs, m.string.strip(), text)


def _cmd_rotcheck(m: re.Match) -> Outcome:
    def text(v: dict) -> str:
        head = f"{v['status'].upper()} ({v['method']})"
        if v["status"] == "refuted":
            causes = "; ".join(v.get("explanations") or []) or "no single convention change explains it"
            where = ", ".join(f"{k} = {w['components']}" for k, w in v["witness"].items())
            return f"{head} — the two sides differ at {where}\n  likely cause: {causes}"
        return head
    return _geometry("rotcheck", {"a": m["a"].strip(), "b": m["b"].strip()}, m.string.strip(), text)


def _predicate(word: str, count: int, dim: int) -> Callable[[re.Match], Outcome]:
    def handler(m: re.Match) -> Outcome:
        inputs = {"points": _points(m["pts"], count, dim)}
        names = {1: "positive", -1: "negative", 0: "zero (degenerate)"}

        def text(v: dict) -> str:
            return f"{word}: {names[v['sign']]} (sign {v['sign']}, decided by {v['stage']})"
        return _geometry(word, inputs, m.string.strip(), text)
    return handler


def _cmd_segment(m: re.Match) -> Outcome:
    def text(v: dict) -> str:
        where = v["point"] or v["segment"]
        return f"{v['kind']}" + (f" at {where}" if where else "")
    return _geometry("segment", {"points": _points(m["pts"], 4, 2)}, m.string.strip(), text)


def _cmd_easing(m: re.Match) -> Outcome:
    def text(v: dict) -> str:
        lo, hi = v["enclosure"]
        return f"y = {v['y']!r} at x = {v['x']!r} (proved inside [{lo!r}, {hi!r}])"
    return _geometry("easing", {"spec": m["spec"].strip(), "x": m["x"]}, m.string.strip(), text)


def _cmd_arclength(m: re.Match) -> Outcome:
    pts = [_point(p) for p in re.findall(r"\(([^)]*)\)", m["pts"])]
    if len(pts) < 2:
        raise ParseError("arclength: give the control points as (x,y) (x,y) ...")

    def text(v: dict) -> str:
        return f"length {v['value']!r} (proved within [{v['lower']!r}, {v['upper']!r}])"
    return _geometry("arclength", {"control": pts}, m.string.strip(), text)


def _cmd_emit(m: re.Match) -> Outcome:
    def text(v: dict) -> str:
        return v["code"].rstrip()
    return _geometry("emit", {"formula": m["formula"], "language": m["lang"].lower()}, m.string.strip(), text)


def _cmd_epsilon(m: re.Match) -> Outcome:
    def text(v: dict) -> str:
        out = [f"{v['status'].upper()} — {v['detail']}"]
        out += [f"  {w['kind']}: input {w['input']} — the epsilon test says {w['epsilon_test']}, "
                f"exactly {w['exact']}" for w in v["witnesses"][:3]]
        return "\n".join(out)
    return _geometry("epsilon", {"test": m["test"], "epsilon": m["eps"]}, m.string.strip(), text)


# =======================================================================================
# WAVE 2 (I2): K2 precision, K5 solution sets and definite integrals, C5 loops and
# recurrences, L1 lint — the same four rules as above. Decisions (I2 report §2):
#
# * ``evalf``, ``identify``, ``iterations``, ``complexity``, ``recurrence``, ``master``,
#   ``lint``, ``reduce``, ``latex`` were free words; K5 does NOT get new words for
#   ``solve`` / ``integrate``: the operation is the same, its SHAPE picks the module
#   (a relation -> inequality, a top-level comma -> system, ``all`` -> every real
#   solution, ``from a to b`` -> definite integral). ``repl._route`` asks
#   ``wave2_route`` before the old patterns, and only these shapes are taken.
# * the C5 file commands read the caller's OWN file, so (as ``verify code``) the
#   function IS executed in the isolated worker; source from outside goes through MCP
#   ``math_iterations`` with ``execute=False`` by default.
# * K2's ``rigorous`` is always printed; an approximation that is not rigorous gets
#   an UNDECIDED certificate, never VERIFIED (review note 5).
# * K5's typed refusals carry their number: "no closed form — converges, value ≈ …"
#   (review note 6); the principal value only with the ``pv`` flag (note 7).
# =======================================================================================

def _status_of(word: object) -> Any:
    from ..verify.verdict import VerdictStatus

    value = getattr(word, "value", word)
    return VerdictStatus(str(value).lower())


def _verdict_from(status: object, method: str, detail: str,
                  counterexample: "dict[str, str] | None" = None) -> Verdict:
    from ..verify.verdict import VerdictStatus

    st = _status_of(status)
    if st is VerdictStatus.REFUTED:
        return Verdict(st, method, counterexample or {}, detail)
    return Verdict(st, method, None, detail)


# --- K2: evalf / identify / N(e, d) / the dual answer --------------------------------
def approx_outcome(expr: str, digits: "int | None", *, source: "str | None" = None) -> Outcome:
    """``evalf <expr> <d>`` and ``N(<expr>, <d>)``: digits that are proved, or a
    stated agreement (``rigorous: no``) — never an unmarked guess."""
    from ..engine.precision import evalf_to
    from ..verify.certificate import approx_verdict

    a = evalf_to(expr, digits)
    verdict = approx_verdict(a)
    cert = _quiet(lambda: from_verdict("approx", [a.exact, a.value], verdict, source=source,
                                       digits=a.digits))
    proof = "rigorous" if a.rigorous else "NOT rigorous: agreement of two precisions, not a proof"
    lines = [f"≈ {a.value}", f"  {a.guaranteed_digits} digits, {a.method} ({proof})",
             _cert_line(cert)]
    return Outcome("\n".join(lines), "approx", a.to_dict(), cert, verdict)


def _cmd_evalf(m: re.Match) -> Outcome:
    return approx_outcome(m["expr"], int(m["digits"]), source=m.string.strip())


def _cmd_identify(m: re.Match) -> Outcome:
    from ..engine.identify import identify

    digits = int(m["digits"]) if m["digits"] else None
    basis = [b for b in (m["basis"] or "").split(",") if b] or None
    res = identify(m["x"], digits=digits, basis=basis)
    verdict = _verdict_from(res.status, res.method, res.detail)
    cert = _quiet(lambda: from_verdict("identify", [m["x"], res.value or ""], verdict,
                                       source=m.string.strip(), digits=digits,
                                       basis=None if basis is None else ",".join(basis)))
    return Outcome("\n".join([str(res), _cert_line(cert)]), "identify", res.to_dict(), cert, verdict)


#: Longest time the dual answer may add to a bare expression (seconds). The exact
#: answer never waits for its decimal: past this, or undecided, the line is omitted.
DUAL_BUDGET = 5.0


def dual_outcome(text: str, value: Any, *, source: str) -> "Outcome | None":
    """The exact answer with its proved decimal beside it (``sqrt(2)/2  ≈ 0.707…``),
    as SymPy Gamma and Wolfram|Alpha show both. ``None`` — the exact text stands
    alone — for a value with free symbols, an integer (its decimal says nothing new),
    a float (already decimal), or when no digit could be certified in time."""
    from ..engine.precision import PrecisionUndecided, approx
    from ..verify.certificate import approx_verdict

    sy = getattr(value, "sy", value)
    if not isinstance(sy, sp.Expr) or sy.free_symbols or not sy.is_number:
        return None
    if sy.has(sp.Float, sp.zoo, sp.nan, sp.oo, -sp.oo) or sy.is_Integer:
        return None
    re_, im_ = sy.as_real_imag() if not sy.is_real else (sy, sp.Integer(0))
    if all(getattr(p, "is_Integer", False) for p in (re_, im_)):
        return None  # 32*I: its decimal adds nothing
    try:
        a = approx(sy, budget=DUAL_BUDGET)
    except (PrecisionUndecided, DomainError, ParseError):
        return None
    except Exception:  # noqa: BLE001 — the decimal is optional; the exact answer stands
        return None
    verdict = approx_verdict(a)
    cert = _quiet(lambda: from_verdict("approx", [a.exact, a.value], verdict, source=source,
                                       digits=a.digits))
    note = "" if a.rigorous else "  (not rigorous: agreement, not a proof)"
    return Outcome(f"{text}  ≈ {a.value}{note}", "approx", a.to_dict(), cert, verdict)


def _cmd_latex(m: re.Match) -> Outcome:
    """LaTeX of the PARSED expression — presentation, so nothing to certify; the
    line shows the reading too, so a misreading is visible (I2 adapter, probe #46)."""
    expr = parse(m["expr"]).sy
    code = sp.latex(expr)
    return Outcome("\n".join([code, f"  reading: {expr}", _cert_line(None)]), "latex",
                   {"kind": "latex", "expr": str(expr), "latex": code}, None)


# --- K5: inequalities, systems, periodic families, definite integrals ----------------
_REL = re.compile(r"(<=|>=|≤|≥|<|>)")


def _top_level_split(text: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [p for p in out if p]


def _free_names(text: str) -> list[str]:
    names: set[str] = set()
    for side in re.split(r"<=|>=|≤|≥|<|>|=", text):
        if side.strip():
            names |= {str(s) for s in parse(side).sy.free_symbols}
    return sorted(names)


def _k5(kind: str, call: Callable[[], Any], inputs: dict[str, Any], source: str,
        budget: "float | None") -> tuple[Any, Verdict, "Certificate | None", "Exception | None"]:
    """Run one K5 operation under the budget; its certificate records ``inputs`` and
    the result's key fields (``certificate.k5_key``) — a typed refusal is UNDECIDED
    with the refusal written down, so ``recheck`` reproduces it too."""
    from ..core.budget import time_budget
    from ..verify.certificate import k5_outcome

    with NUMERIC_LOCK, time_budget(budget if budget is not None else 120.0):
        res, key, verdict, refusal = k5_outcome(kind, call)
    from ..verify.certificate import _jtext

    cert = _quiet(lambda: from_verdict(kind, [_jtext(inputs), _jtext(key)], verdict,
                                       source=source, budget=budget))
    return res, verdict, cert, refusal


def _refusal_text(exc: Exception) -> str:
    approx_value = getattr(exc, "approx", None)
    if approx_value is not None:
        err = getattr(exc, "error", None)
        pm = f" ± {err}" if err else ""
        head = ("no closed form; the integral converges (proved), value ≈ "
                if type(exc).__name__ == "IntegralNoClosedForm" else "UNDECIDED; quadrature gives ≈ ")
        return f"{head}{approx_value}{pm}\n  {exc}"
    return f"UNDECIDED — {exc}"


def inequality_outcome(relation: str, var: "str | None", within: "tuple[str, str] | None",
                       *, source: str, budget: "float | None" = None) -> Outcome:
    from ..engine.solvesets import solve_inequality

    names = [var] if var else _free_names(relation)
    if len(names) != 1:
        raise ParseError(f"solve: an inequality in ONE variable — name it: 'solve {relation} for x'"
                         + (f" (found {', '.join(names)})" if names else ""))
    v = names[0]
    inputs = {"relation": relation, "var": v, "within": list(within) if within else None}
    res, verdict, cert, refusal = _k5(
        "inequality", lambda: solve_inequality(relation, v, within=within), inputs, source, budget)
    if refusal is not None:
        return Outcome("\n".join([_refusal_text(refusal), _cert_line(cert)]), "inequality",
                       {"kind": "inequality", "refused": type(refusal).__name__, "message": str(refusal)},
                       cert, verdict)
    d = res.to_dict()
    lines = [f"{v} ∈ {d['pretty']}"]
    if d.get("approx") and d["approx"] != d["pretty"]:
        lines.append(f"  ≈ {d['approx']}")
    if d.get("domain") not in (None, "Interval(-oo, oo)"):
        lines.append(f"  domain: {d['domain']}" + (f" ({', '.join(d['conditions'])})" if d.get("conditions") else ""))
    lines.append(f"  proof: sign decomposition ({d['method']}) — {len(d['cells'])} cells, "
                 f"{len(d['endpoints'])} endpoints decided exactly")
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "inequality", d, cert, verdict)


def system_outcome(equations: list[str], names: "list[str] | None", *, source: str,
                   budget: "float | None" = None) -> Outcome:
    from ..engine.solvesets import solve_system

    inputs = {"equations": equations, "vars": names}
    res, verdict, cert, refusal = _k5("system", lambda: solve_system(equations, names), inputs,
                                      source, budget)
    if refusal is not None:
        return Outcome("\n".join([_refusal_text(refusal), _cert_line(cert)]), "system",
                       {"kind": "system", "refused": type(refusal).__name__, "message": str(refusal)},
                       cert, verdict)
    d = res.to_dict()
    if d["solutions"]:
        lines = ["; ".join(", ".join(f"{k} = {x}" for k, x in s.items()) for s in d["solutions"])]
    else:
        lines = [f"no solution ({d['status']})" if d["status"] != "finite" else "no solution"]
    if d["status"] not in ("finite", "unique", "inconsistent"):
        lines[0] += f"  [{d['status']}; parameters {', '.join(d['parameters']) or 'none'}]"
    proof = d.get("proof") or {}
    if isinstance(proof, dict) and proof.get("argument"):
        lines.append(f"  proof: {proof['argument']}")
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "system", d, cert, verdict)


def periodic_outcome(equation: str, var: "str | None", *, source: str,
                     budget: "float | None" = None) -> Outcome:
    from ..engine.solvesets import periodic_solutions

    names = [var] if var else _free_names(equation)
    if len(names) != 1:
        raise ParseError(f"solve ... all: one variable — name it: 'solve {equation} for x all'")
    inputs = {"equation": equation, "var": names[0]}
    res, verdict, cert, refusal = _k5("periodic", lambda: periodic_solutions(equation, names[0]),
                                      inputs, source, budget)
    if refusal is not None:
        return Outcome("\n".join([_refusal_text(refusal), _cert_line(cert)]), "periodic",
                       {"kind": "periodic", "refused": type(refusal).__name__, "message": str(refusal)},
                       cert, verdict)
    d = res.to_dict()
    lines = [d["text"], f"  proof: period {d['period']} ({d['periodicity']}); "
                        f"solutions in one period complete (sign decomposition)", _cert_line(cert)]
    return Outcome("\n".join(lines), "periodic", d, cert, verdict)


def definite_outcome(f: str, var: str, a: str, b: str, *, pv: bool, source: str,
                     budget: "float | None" = None, numeric_first: bool = False) -> Outcome:
    from ..engine.definite import definite_integral

    inputs = {"f": f, "var": var, "a": a, "b": b, "principal_value": pv}
    res, verdict, cert, refusal = _k5(
        "definite_integral", lambda: definite_integral(f, var, a, b, principal_value=pv),
        inputs, source, budget)
    if refusal is not None:
        return Outcome("\n".join([_refusal_text(refusal), _cert_line(cert)]), "definite_integral",
                       {"kind": "definite_integral", "refused": type(refusal).__name__,
                        "message": str(refusal), "approx": getattr(refusal, "approx", None),
                        "error": getattr(refusal, "error", None)}, cert, verdict)
    d = res.to_dict()
    if d.get("principal_value") and d.get("value") is not None:
        lines = [f"{d['value']}  ≈ {d['approx']}  (Cauchy principal value — requested; "
                 f"the integral itself {'converges' if d['converges'] else 'diverges'})"]
    elif d["converges"]:
        lines = [f"≈ {d['approx']}  (exact: {d['value']})" if numeric_first
                 else f"{d['value']}  ≈ {d['approx']}"]
    else:
        to = f" to {d['value']}" if d.get("value") else ""
        lines = [f"diverges{to}" + ("" if d.get("value") or pv else
                                    "  (the Cauchy principal value, if wanted: add 'pv')")]
    if d.get("singularities"):
        lines.append(f"  singular points inside: {', '.join(d['singularities'])}")
    lines.append(f"  {d['certificate']}")
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "definite_integral", d, cert, verdict)


_SOLVE_HEAD = re.compile(r"^(?P<word>solve|reduce)\s+(?P<body>.+?)(?:\s+budget\s+(?P<budget>\S+))?\s*$",
                         re.IGNORECASE | re.DOTALL)
_FOR_VARS = re.compile(r"^(?P<eq>.+?)\s+for\s+(?P<vars>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)"
                       r"(?:\s+in\s+\[\s*(?P<lo>[^\],]+?)\s*,\s*(?P<hi>[^\]]+?)\s*\])?(?P<all>\s+all)?\s*$",
                       re.IGNORECASE | re.DOTALL)
_INTEGRATE_DEF = re.compile(
    r"^(?P<word>n?integrate)\s+(?P<expr>.+)\s+d(?P<var>[A-Za-z_]\w*)\s+from\s+(?P<a>\S+)\s+to\s+(?P<b>\S+)"
    r"(?P<pv>\s+pv)?(?:\s+budget\s+(?P<budget>\S+))?\s*$", re.IGNORECASE)


def wave2_route(line: str) -> "Outcome | None":
    """K5's shapes of ``solve`` / ``reduce`` / ``integrate`` / ``nintegrate`` (see the
    rules above) — ``None`` for every other line, which then keeps its old meaning
    (``solve x^2-4 for x`` is still the old root list, ``nintegrate`` with finite
    bounds still the fixed-step rule)."""
    text = line.strip()
    m = _INTEGRATE_DEF.match(text)
    if m is not None:
        infinite = any(re.fullmatch(r"[-+]?oo", m[k]) for k in ("a", "b"))
        if m["word"].lower() == "nintegrate" and not infinite:
            return None
        budget = _budget(m["budget"]) if m["budget"] else None
        return definite_outcome(m["expr"].strip(), m["var"], m["a"], m["b"], pv=bool(m["pv"]),
                                source=text, budget=budget,
                                numeric_first=m["word"].lower() == "nintegrate")
    m = _SOLVE_HEAD.match(text)
    if m is None:
        return None
    word, body = m["word"].lower(), m["body"].strip()
    budget = _budget(m["budget"]) if m["budget"] else None
    fv = _FOR_VARS.match(body)
    eq_text = fv["eq"].strip() if fv else body
    flag_all = bool(fv and fv["all"])
    if fv is None and re.search(r"\s+all$", body, re.I):
        eq_text, flag_all = re.sub(r"\s+all$", "", body, flags=re.I).strip(), True
    names = [n.strip() for n in fv["vars"].split(",")] if fv else None
    within = (_exact_text(fv["lo"], "range"), _exact_text(fv["hi"], "range")) if fv and fv["lo"] else None
    parts = _top_level_split(eq_text)
    if len(parts) > 1 or (names is not None and len(names) > 1):
        if within or flag_all:
            raise ParseError("solve: a system takes no 'in [a, b]' or 'all'")
        return system_outcome(parts, names, source=text, budget=budget)
    if _REL.search(eq_text):
        if flag_all:
            raise ParseError("solve: 'all' is for equations; an inequality's answer is already every solution")
        return inequality_outcome(eq_text, names[0] if names else None, within, source=text, budget=budget)
    if flag_all:
        return periodic_outcome(eq_text, names[0] if names else None, source=text, budget=budget)
    if word == "reduce":
        raise ParseError("reduce: give a relation (x^2 < 4), a system (a, b for x, y) or "
                         "an equation with 'all'")
    if within is not None:
        raise ParseError("solve: 'in [a, b]' bounds an inequality (solve sin(x) > 0 for x in [0, 2*pi])")
    return None


REDUCE_USAGE = ("Usage: reduce <relation> [for x [in [a, b]]] | <eq1>, <eq2> for x, y | <eq> all"
                "   (e.g. reduce x^2 < 4 for x)")


def _cmd_reduce(m: re.Match) -> Outcome:
    out = wave2_route(m.string)
    if out is None:
        raise ParseError(REDUCE_USAGE)
    return out


#: Whether the file commands (``iterations``, ``complexity``, ``lint``) may RUN the
#: functions they read. True in the CLI/REPL — the file is the caller's own; MCP
#: ``math_eval`` sets it False for its call, so an agent naming a path on the server
#: gets reading and proof, never execution (the ``math_check_code`` rule).
EXECUTE_FILES: "contextvars.ContextVar[bool]" = contextvars.ContextVar("pcm_execute_files", default=True)


# --- C5: iterations / complexity / recurrence / master ---------------------------------
_FILE_FUNC = r"(?P<path>\S+?\.py)(?::(?P<func>[A-Za-z_]\w*))?"


def _read_own(path: str, what: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DomainError(f"{what}: cannot read {path!r} ({exc.strerror})") from exc
    except UnicodeDecodeError as exc:
        raise DomainError(f"{what}: {path!r} is not UTF-8 text (byte {exc.start})") from exc


def iterations_outcome(source: str, func: "str | None", claim: "str | None", *, execute: bool,
                       budget: float = 30.0, origin: "str | None" = None) -> Outcome:
    """``count_iterations`` (no claim) or ``check_iterations`` (a claimed count),
    with the ``iterations`` certificate. ``execute=False``: nothing runs — the count
    is read and proved, a refutation stays undecided (MCP default)."""
    from ..code import loops as L
    from ..verify.certificate import _jtext, c5_text_guard, code_execution, iterations_key

    if claim:
        c5_text_guard(claim, "iterations: the claimed count")
    with code_execution(execute):
        if claim:
            res: Any = L.check_iterations(source, claim, name=func, budget=budget)
        else:
            res = L.count_iterations(source, name=func, budget=budget)
    key, verdict = iterations_key(res)
    fname = res.function if hasattr(res, "function") else res.result.function
    cert = _quiet(lambda: from_verdict("iterations", [source, fname, claim or "", _jtext(key)], verdict,
                                       source=origin, budget=budget, execute=execute))
    d = res.to_dict()
    if claim:
        lines = [_verdict_line(verdict), f"  counted: {d['count']} iterations"]
    else:
        lines = [f"{d['function']}: {d['main']} iterations for {d['valid_for']}"
                 + (f" (0 for {d['zero_for']})" if d.get("zero_for") else "")
                 + ("" if d.get("exact") else "  — an UPPER bound (the loop can exit early)")]
        if d.get("theta"):
            lines.append(f"  Θ({d['theta']}), constant {d['theta_constant']} — proved by a limit")
        c = d.get("certificate")
        if c:
            lines.append(f"  counted by running it: {c['points']} points, "
                         f"{'all agree' if c['agree'] else 'DISAGREE'} ({c['domain']})")
        else:
            lines.append("  not executed: the count is read and proved, not counted")
        lines += [f"  note: {n}" for n in d.get("notes", [])]
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "iterations", d, cert, verdict)


def complexity_outcome(source: str, func: "str | None", claim: str, *, execute: bool,
                       budget: float = 30.0, origin: "str | None" = None) -> Outcome:
    from ..code import loops as L
    from ..verify.certificate import _jtext, c5_text_guard, code_execution, complexity_key

    c5_text_guard(claim, "complexity: the claim")
    with code_execution(execute):
        res = L.check_complexity(source, claim, name=func, budget=budget)
    key, verdict = complexity_key(res)
    cert = _quiet(lambda: from_verdict("complexity", [source, res.result.function, claim, _jtext(key)],
                                       verdict, source=origin, budget=budget, execute=execute))
    d = res.to_dict()
    lines = [_verdict_line(verdict), f"  count: {d['count']}; Θ({d['theta']})" if d.get("theta")
             else f"  count: {d['count']}", _cert_line(cert)]
    return Outcome("\n".join(lines), "complexity", d, cert, verdict)


def _cmd_iterations(m: re.Match) -> Outcome:
    source = _read_own(m["path"], "iterations")
    budget = _budget(m["budget"]) if m["budget"] else 30.0
    return iterations_outcome(source, m["func"], m["claim"], execute=EXECUTE_FILES.get(), budget=budget,
                              origin=m.string.strip())


def _cmd_complexity(m: re.Match) -> Outcome:
    source = _read_own(m["path"], "complexity")
    budget = _budget(m["budget"]) if m["budget"] else 30.0
    return complexity_outcome(source, m["func"], m["claim"].strip(), execute=EXECUTE_FILES.get(),
                              budget=budget,
                              origin=m.string.strip())


def recurrence_outcome(text: str, *, source: "str | None" = None) -> Outcome:
    from ..code.recurrence import solve_recurrence
    from ..verify.certificate import _jtext, c5_text_guard, recurrence_key

    c5_text_guard(text, "recurrence")
    _no_conflicting_initial_values(text)
    with NUMERIC_LOCK:
        res = solve_recurrence(text)
    key, verdict = recurrence_key(res)
    cert = _quiet(lambda: from_verdict("recurrence", [text, _jtext(key)], verdict, source=source))
    d = res.to_dict()
    lines = [f"Θ({d['theta']})" if d.get("theta") else "no Θ (undecided)"]
    if d.get("main"):
        lines[0] += f"; exact: T(n) = {d['main']} for {d['valid_for']}"
    lines.append(f"  {d['method']}: {d['reason']}")
    lines += [f"  check: {c}" for c in d.get("checks", [])]
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "recurrence", d, cert, verdict)


def _no_conflicting_initial_values(text: str) -> None:
    """``T(1) = 1, T(1) = 2``: C5 silently kept the last value (I2 adversary round,
    "do poprawki w C5"); two different values for one point are a contradiction."""
    seen: dict[str, str] = {}
    for point, value in re.findall(r"\b[A-Za-z]\(\s*(-?\d+)\s*\)\s*=\s*([^,]+)", text):
        if seen.setdefault(point, value.strip()) != value.strip():
            raise ParseError(f"recurrence: two different initial values at {point}: "
                             f"{seen[point]} and {value.strip()}")


def _cmd_recurrence(m: re.Match) -> Outcome:
    return recurrence_outcome(m["text"].strip(), source=m.string.strip())


def _cmd_master(m: re.Match) -> Outcome:
    from ..code.recurrence import MasterTheoremInapplicable, master_theorem
    from ..verify.certificate import c5_text_guard

    for part in ("a", "b", "f"):
        c5_text_guard(m[part], f"master: {part}")
    try:
        with NUMERIC_LOCK:
            res = master_theorem(m["a"], m["b"], m["f"].strip())
    except MasterTheoremInapplicable as exc:
        alt = getattr(exc, "alternative", None)
        text = (f"the master theorem does not apply: {exc.reason}"  # type: ignore[attr-defined]
                + (f"\n  try: {alt} (recurrence T(n) = {m['a']}*T(n/{m['b']}) + {m['f'].strip()})"
                   if alt else ""))
        return Outcome(text + "\n" + _cert_line(None), "recurrence",
                       {"kind": "recurrence", "applies": False, "reason": exc.reason,  # type: ignore[attr-defined]
                        "alternative": alt}, None)
    d = res.to_dict()
    rec = f"T(n) = {m['a']}*T(n/{m['b']}) + {m['f'].strip()}"
    from ..verify.certificate import issue

    cert = _quiet(lambda: issue("recurrence", rec, _recorded_key(rec), source=m.string.strip()))
    lines = [f"Θ({d['theta']}) — {d['method']}"] + [f"  check: {c}" for c in d["checks"]]
    lines.append(_cert_line(cert))
    return Outcome("\n".join(lines), "recurrence", d, cert)


def _recorded_key(rec: str) -> str:
    from ..code.recurrence import solve_recurrence
    from ..verify.certificate import _jtext, recurrence_key

    with NUMERIC_LOCK:
        key, _ = recurrence_key(solve_recurrence(rec))
    return _jtext(key)


# --- L1: lint ---------------------------------------------------------------------------
def run_lint(argv: "list[str]") -> tuple[int, str]:
    """``pycodemath.lint.__main__.main`` with its output captured: ``(exit code,
    text)``. Exit codes are ruff's (0 clean / 1 findings / 2 usage or error) — for
    ``lint`` only; every other command keeps 0 = answered, 1 = refused."""
    import contextlib
    import io

    from ..lint.__main__ import main

    buf = io.StringIO()
    with NUMERIC_LOCK, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = main(argv)
    return code, buf.getvalue().rstrip("\n")


def lint_outcome(source: str, path: str, *, select: "list[str] | None", execute: bool,
                 budget: "float | None" = None, text: "str | None" = None,
                 origin: "str | None" = None, extra: "dict[str, Any] | None" = None) -> Outcome:
    """One file's findings and their ``lint`` certificate (each finding = a witness,
    re-checked by replay)."""
    from ..verify.certificate import _jtext, lint_findings, lint_verdict

    with NUMERIC_LOCK:
        findings = lint_findings(source, path, select, execute, budget)
    verdict = lint_verdict(findings)
    cert = _quiet(lambda: from_verdict(
        "lint", [source, path, ",".join(select or []), _jtext([f["key"] for f in findings])],
        verdict, source=origin, execute=execute))
    result = {"kind": "lint", "status": "findings" if findings else "clean",
              "findings": [f["finding"] for f in findings], **(extra or {})}
    if text is None:
        text = "\n".join(f"{path}:{f['finding']['location']['row']}:{f['finding']['location']['column']}: "
                         f"{f['finding']['code']} {f['finding']['message']}" for f in findings) or "clean"
    return Outcome("\n".join([text, _cert_line(cert)]), "lint", result, cert, verdict)


def _cmd_lint(m: re.Match) -> Outcome:
    import shlex

    from ..lint.__main__ import build_parser

    argv = shlex.split(m["args"] or "")
    if not EXECUTE_FILES.get() and "--no-execute" not in argv:
        argv.append("--no-execute")
    code, text = run_lint(argv)
    # the summary's wall clock ("in 0.74 s") is the one byte that never reproduces: on
    # this surface the same file gives the same text (README, certificates' source);
    # ``pycodemath-lint`` keeps it
    text = re.sub(r"(?m)^(Found .*?|All checks passed.*?|No .*?) in \d+(?:\.\d+)? s\.$", r"\1.", text)
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):  # run_lint already reported it
            args = build_parser().parse_args(argv)
    except SystemExit:
        args = None
    files = list(getattr(args, "paths", None) or [])
    extra = {"exit_code": code}
    plain_run = args is not None and not (args.list_rules or args.explain or args.output)
    if args is not None and plain_run and len(files) == 1 and Path(files[0]).is_file() \
            and code in (0, 1):
        select = list(args.select) if args.select else None
        return lint_outcome(_read_own(files[0], "lint"), files[0], select=select,
                            execute=not args.no_execute, text=text, origin=m.string.strip(),
                            extra=extra)
    why = ("one file gets a certificate" if plain_run and code in (0, 1) else "nothing to re-check")
    return Outcome("\n".join([text, f"certificate: none ({why})"]), "lint",
                   {"kind": "lint", **extra}, None)


def lint_entry() -> None:
    """Console script ``pycodemath-lint``: L1's ``main`` with output that cannot die
    of the console's code page. L1 prints "—" and "≈"; on cp437/cp1250 a plain
    ``print`` raised ``UnicodeEncodeError`` (I2 review note 1), so characters the
    console cannot show are replaced, never fatal. Exit codes as ruff."""
    import sys

    from ..lint.__main__ import main

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    sys.exit(main())


LINT_USAGE = ("Usage: lint [paths] [--select CODES] [--ignore CODES] [--format text|json|sarif]"
              " [--concise] [--no-execute] [--list-rules] [--explain CODE]"
              "   (e.g. lint tests/data/lint/dev/p401_gross.py --concise --no-cache)")


# --- the table ------------------------------------------------------------------------
_I = re.IGNORECASE
_O = r"(?P<opts>(?:\s+\S+)*?)"
_DATA = r"(?P<data>[-+]?[\d.][-+\d.eE/,\s]*?)"

#: word -> (pattern, handler, usage, one help line). ``repl`` merges this table.
COMMANDS: dict[str, tuple[re.Pattern[str], Callable[[re.Match], Outcome], str]] = {
    # C2
    "stability": (re.compile(rf"^stability\s+(?P<expr>.+?){_FOR}{_O}\s*$", _I), _cmd_stability,
                  "Usage: stability <expr> for x in [a, b][, y in [c, d]] [as float32] [samples N] [budget <s>]"
                  "   (e.g. stability sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15])"),
    "improve": (re.compile(rf"^improve\s+(?P<expr>.+?){_FOR}{_O}\s*$", _I), _cmd_improve,
                "Usage: improve <expr> for x in [a, b] [as float32] [emit python|numpy|c] [budget <s>]"
                "   (e.g. improve sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15])"),
    # C3
    "ranges": (re.compile(rf"^ranges\s+(?P<expr>.+?){_FOR}{_O}\s*$", _I), _cmd_ranges,
               "Usage: ranges <expr> for x in [a, b][ int][, y in [c, d]] as <dtype> [budget <s>]"
               "   (e.g. ranges (lo + hi) // 2 for lo in [0, 2^31-1], hi in [0, 2^31-1] as int32)"),
    "maxsafe": (re.compile(r"^maxsafe\s+(?P<expr>.+?)\s+for\s+(?P<var>[A-Za-z_]\w*)"
                           r"(?:\s*,\s*(?P<others>[A-Za-z_]\w*\s+in\s+\[.+\]))?" + _O + r"\s*$", _I), _cmd_maxsafe,
                "Usage: maxsafe <expr> for <x> as <dtype>   (e.g. maxsafe exp(x) for x as float32)"),
    # C4
    "money": (re.compile(r"^money\s+(?P<rest>.+?)\s*$", _I | re.DOTALL), _cmd_money, MONEY_USAGE),
    # K3 — number theory, combinatorics
    **{word: (re.compile(rf"^{word}\s+(?P<args>.+?)\s*$", _I), _int_command(word),
              f"Usage: {word} " + (" ".join(f"<n{i + 1}>" for i in range(arity)) if arity > 0
                                    else "<n1> <n2> ...") + f"   (e.g. {word} "
              + _INT_EXAMPLES[word] + ")")
       for word, (_, arity) in _INT_OPS.items()},
    "mod": (re.compile(r"^mod\s+(?P<a>\S+)\s+(?P<m>\S+)(?:\s+(?P<conv>floor|euclid|trunc))?\s*$", _I), _cmd_mod,
            "Usage: mod <a> <m> [floor|euclid|trunc]   (e.g. mod -7 3 trunc)"),
    "crt": (re.compile(r"^crt\s+(?P<r>.+?)\s+mod\s+(?P<m>.+?)\s*$", _I), _cmd_crt,
            "Usage: crt <r1,r2,...> mod <m1,m2,...>   (e.g. crt 2,3,2 mod 3,5,7)"),
    "special": (re.compile(r"^special\s+(?P<name>[A-Za-z_]\w*)\s+(?P<args>.+?)(?:\s+digits\s+(?P<digits>\d+))?\s*$", _I),
                _cmd_special, "Usage: special <function> <args> [digits <d>]   (e.g. special zeta 3 digits 50)"),
    # K3 — statistics
    **{word: (re.compile(rf"^(?P<word>{word})\s+(?:(?P<kind>sample|population)\s+)?{_DATA}\s*$", _I), _cmd_describe,
              f"Usage: {word} {'sample|population ' if word in ('var', 'std') else ''}<x1> <x2> ...   "
              f"(e.g. {word} {'sample ' if word in ('var', 'std') else ''}1 2 3 4)")
       for word in ("mean", "median", "var", "std")},
    "quantile": (re.compile(rf"^quantile\s+(?P<q>\S+)(?:\s+type\s+(?P<type>\w+))?\s+{_DATA}\s*$", _I), _cmd_quantile,
                 "Usage: quantile <q> [type 1..9|name] <x1> <x2> ...   (e.g. quantile 0.25 type 7 1 2 3 4)"),
    **{fn: (re.compile(rf"^(?P<fn>{fn})\s+(?P<dist>[A-Za-z_]\w*)\s+(?P<x>\S+)(?P<params>(?:\s+\w+=\S+)*)\s*$", _I),
            _cmd_dist, f"Usage: {fn} <distribution> <x> [param=value ...]   (e.g. {fn} "
            + {"ppf": "t 0.975 df=10", "pmf": "binomial 3 n=10 p=0.5"}.get(fn, "normal 1.96") + ")")
       for fn in ("pdf", "pmf", "cdf", "sf", "ppf")},
    "samplesize": (re.compile(r"^samplesize(?P<params>(?:\s+\w+=\S+)+)(?P<relative>\s+relative)?\s*$", _I),
                   _cmd_samplesize, "Usage: samplesize p1=<rate> p2=<rate>|mde=<d> [power=0.8] [alpha=0.05]"
                   " [method=evanmiller|pooled|arcsine] [relative]   (e.g. samplesize p1=0.10 p2=0.12)"),
    "power": (re.compile(r"^power(?P<params>(?:\s+\w+=\S+)+)\s*$", _I), _cmd_power,
              "Usage: power p1=<rate> p2=<rate> n=<per group> [alpha=0.05]   (e.g. power p1=0.10 p2=0.12 n=3623)"),
    "ci": (re.compile(r"^ci\s+(?P<how>wilson|exact)\s+(?P<s>\d+)\s*/\s*(?P<n>\d+)(?P<params>(?:\s+\w+=\S+)*)\s*$", _I),
           _cmd_ci, "Usage: ci wilson|exact <successes>/<trials> [confidence=0.95]   (e.g. ci wilson 81/263)"),
    "ztest": (re.compile(r"^ztest\s+(?P<s1>\d+)\s*/\s*(?P<n1>\d+)\s+(?P<s2>\d+)\s*/\s*(?P<n2>\d+)\s*$", _I),
              _cmd_ztest, "Usage: ztest <s1>/<n1> <s2>/<n2>   (e.g. ztest 120/1000 150/1000)"),
    "welch": (re.compile(r"^welch\s+\[(?P<a>[^\]]+)\]\s+\[(?P<b>[^\]]+)\]\s*$", _I), _cmd_welch,
              "Usage: welch [x1, x2, ...] [y1, y2, ...]   (e.g. welch [1, 2, 3, 4] [2, 3, 4, 5])"),
    **{word: (re.compile(rf"^(?P<word>{word})\s+(?P<p>[\d.eE,\s-]+?)\s*$", _I), _cmd_adjust,
              f"Usage: {word} <p1> <p2> ...   (e.g. {word} 0.01 0.04 0.03)")
       for word in ("holm", "bonferroni")},
    # J1
    "convert": (re.compile(r"^convert\s+(?P<q>.+?)\s+(?:to|in)\s+(?P<to>\S+)\s*$", _I), _cmd_convert,
                "Usage: convert <quantity> to <unit>   (e.g. convert 5 km/h to m/s)"),
    "units": (re.compile(r"^units\s+(?P<eq>.+?=.+?)\s+where\s+(?P<spec>.+?)(?P<numbers>\s+numbers)?\s*$", _I),
              _cmd_units, "Usage: units <lhs> = <rhs> where name:unit name:unit ... [numbers]"
              "   (e.g. units F = m*v where F:N m:kg v:m/s)"),
    "dims": (re.compile(r"^dims\s+(?P<expr>.+?)(?:\s+where\s+(?P<spec>.+?))?(?P<numbers>\s+numbers)?\s*$", _I),
             _cmd_dims, "Usage: dims <expr> where name:unit ...   (e.g. dims m*v^2/2 where m:kg v:m/s)"),
    # G1
    "quat2mat": (re.compile(rf"^quat2mat\s+(?P<q>[-+\d.eE/]+(?:[\s,]+[-+\d.eE/]+){{3}}){_O}\s*$", _I), _cmd_quat2mat,
                 "Usage: quat2mat <w> <x> <y> <z> [order wxyz|xyzw] [algebra hamilton|jpl] [normalize]"
                 "   (e.g. quat2mat 1 0 0 1 normalize)"),
    "euler2mat": (re.compile(rf"^euler2mat\s+(?P<angles>[-+\d.eE/]+(?:[\s,]+[-+\d.eE/]+){{2}}){_O}\s*$", _I),
                  _cmd_euler2mat, "Usage: euler2mat <a> <b> <c> seq <xyz> frame intrinsic|extrinsic [deg]"
                  "   (e.g. euler2mat 30 45 10 seq zyx frame intrinsic deg)"),
    "mat2euler": (re.compile(rf"^mat2euler\s+(?P<matrix>\[\[.+?\]\]){_O}\s*$", _I), _cmd_mat2euler,
                  "Usage: mat2euler [[...],[...],[...]] seq <xyz> frame intrinsic|extrinsic [deg]"
                  "   (e.g. mat2euler [[0,-1,0],[1,0,0],[0,0,1]] seq zyx frame intrinsic deg)"),
    "rotcheck": (re.compile(r"^rotcheck\s+(?P<a>.+?)\s*==\s*(?P<b>.+?)\s*$", _I), _cmd_rotcheck,
                 "Usage: rotcheck <expr> == <expr>   (e.g. rotcheck q1*q2 == R2@R1)"),
    "orient2d": (re.compile(r"^orient2d\s+(?P<pts>.+?)\s*$", _I), _predicate("orient2d", 3, 2),
                 "Usage: orient2d ax,ay bx,by cx,cy   (e.g. orient2d 0.5,0.5 12,12 24,24)"),
    "orient3d": (re.compile(r"^orient3d\s+(?P<pts>.+?)\s*$", _I), _predicate("orient3d", 4, 3),
                 "Usage: orient3d ax,ay,az bx,by,bz cx,cy,cz dx,dy,dz   (e.g. orient3d 0,0,0 1,0,0 0,1,0 0,0,1)"),
    "incircle": (re.compile(r"^incircle\s+(?P<pts>.+?)\s*$", _I), _predicate("incircle", 4, 2),
                 "Usage: incircle ax,ay bx,by cx,cy dx,dy   (e.g. incircle 0,0 1,0 0,1 1,1)"),
    "segment": (re.compile(r"^segment\s+(?P<pts>.+?)\s*$", _I), _cmd_segment,
                "Usage: segment p1 p2 q1 q2   (e.g. segment 0,0 2,2 0,2 2,0)"),
    "easing": (re.compile(r"^easing\s+(?P<spec>.+?)\s+(?P<x>[-+\d.eE/]+)\s*$", _I), _cmd_easing,
               "Usage: easing <ease|linear|ease-in|...|cubic-bezier(x1,y1,x2,y2)> <x>   (e.g. easing ease 0.5)"),
    "arclength": (re.compile(r"^arclength\s+(?P<pts>(?:\(.+?\)\s*)+)$", _I), _cmd_arclength,
                  "Usage: arclength (x,y) (x,y) ...   (e.g. arclength (0,0) (1,2) (3,3) (4,0))"),
    "emit": (re.compile(r"^emit\s+(?P<formula>[A-Za-z_]\w*)\s+(?P<lang>python|numpy|c|glsl)\s*$", _I), _cmd_emit,
             "Usage: emit <formula> python|numpy|c|glsl   (e.g. emit quat_to_mat3 glsl)"),
    "epsilon": (re.compile(r"^epsilon\s+(?P<test>[A-Za-z_]\w*)\s+(?P<eps>\S+)\s*$", _I), _cmd_epsilon,
                "Usage: epsilon <orient2d|ray_triangle|...> <epsilon>   (e.g. epsilon orient2d 1e-9)"),
    # K2 (I2)
    "evalf": (re.compile(r"^evalf\s+(?P<expr>.+?)\s+(?P<digits>\d+)\s*$", _I), _cmd_evalf,
              "Usage: evalf <expression> <digits>   (e.g. evalf sqrt(2) 50)"),
    "identify": (re.compile(r"^identify\s+(?P<x>\S+)(?:\s+digits\s+(?P<digits>\d+))?"
                            r"(?:\s+basis\s+(?P<basis>\S+))?\s*$", _I), _cmd_identify,
                 "Usage: identify <number> [digits <d>] [basis pi,log(3)]   (e.g. identify zeta(2))"),
    "latex": (re.compile(r"^latex\s+(?P<expr>.+?)\s*$", _I), _cmd_latex,
              "Usage: latex <expression>   (e.g. latex x^2/2)"),
    # K5 (I2) — solve/integrate keep their words, ``wave2_route`` reads the shapes
    "reduce": (re.compile(r"^reduce\s+.+$", _I | re.DOTALL), _cmd_reduce, REDUCE_USAGE),
    # C5 (I2)
    "iterations": (re.compile(rf"^iterations\s+{_FILE_FUNC}(?:\s*==\s*(?P<claim>.+?))?"
                              r"(?:\s+budget\s+(?P<budget>\S+))?\s*$", _I), _cmd_iterations,
                   "Usage: iterations <file.py>[:<function>] [== <claimed count>] [budget <s>]"
                   "   (e.g. iterations tests/data/lint/dev/p504_zero_based_ok.py:sum_below == n)"),
    "complexity": (re.compile(rf"^complexity\s+{_FILE_FUNC}\s+is\s+(?P<claim>.+?)"
                              r"(?:\s+budget\s+(?P<budget>\S+))?\s*$", _I), _cmd_complexity,
                   "Usage: complexity <file.py>[:<function>] is O(...)|Theta(...)|Omega(...) [budget <s>]"
                   "   (e.g. complexity tests/data/lint/dev/p504_zero_based_ok.py:sum_below is O(n log n))"),
    "recurrence": (re.compile(r"^recurrence\s+(?P<text>.+?)\s*$", _I | re.DOTALL), _cmd_recurrence,
                   "Usage: recurrence T(n) = <rhs>[, T(1) = <v>]   (e.g. recurrence T(n) = 2*T(n/2) + n, T(1) = 1)"),
    "master": (re.compile(r"^master\s+(?P<a>\S+)\s+(?P<b>\S+)\s+(?P<f>.+?)\s*$", _I), _cmd_master,
               "Usage: master <a> <b> <f(n)>   — T(n) = a T(n/b) + f(n)   (e.g. master 2 2 n)"),
    # L1 (I2)
    "lint": (re.compile(r"^lint\s+(?P<args>.+?)\s*$", _I), _cmd_lint, LINT_USAGE),
}

#: A bare ``<number> <unit> to <unit>`` line (``5 km/h to m/s``) — Wolfram|Alpha's
#: one-box rule for the commonest unit question. Tried only when the line is not a
#: command and only when both sides read as units; otherwise the strict parser's
#: own refusal stands.
BARE_CONVERSION = re.compile(r"^(?P<q>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*[^\d\s].*?)\s+to\s+(?P<to>\S+)\s*$")


# --- a function call the expression parser does not know (K3's whitelist) -------
#: ``gcd(12, 18)``, ``mod(2^100, 7)``, ``mean([1,2,3,4])``, ``zeta(3)``: names the
#: SymPy-backed parser refuses as "unknown function" but the calculator answers,
#: with a certificate. Consulted ONLY after the parser refused the line, so an
#: expression the parser reads keeps its meaning (``gamma(x)`` stays symbolic).
CALL_ROUTES: dict[str, tuple[str, str]] = {
    **{name: (name, "ints") for name in (
        "gcd", "lcm", "powmod", "modinv", "totient", "divisors", "isprime", "nextprime",
        "prevprime", "fibonacci", "lucas", "legendre", "jacobi", "partitions", "bell", "catalan",
        "stirling1", "stirling2", "permutations", "combinations")},
    "factorint": ("factorint", "ints"),
    "mod": ("mod", "ints"),
    "mean": ("mean", "data"),
    "median": ("median", "data"),
    **{name: ("special", "special") for name in (
        "zeta", "erfinv", "lgamma", "besselj", "bessely", "polylog", "lambertw", "airyai", "airybi")},
}
_CALL = re.compile(r"^(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)$", re.DOTALL)


def call_route(line: str) -> "Outcome | None":
    """The calculator's answer to ``name(args)``, or ``None`` when ``name`` is not
    one of ``CALL_ROUTES``."""
    m = _CALL.match(line.strip())
    if m is None or m["name"] not in CALL_ROUTES:
        return None
    op, shape = CALL_ROUTES[m["name"]]
    args = _split_args(m["args"].replace(", ", ","))
    if shape == "data":
        inner = m["args"].strip()
        if not (inner.startswith("[") and inner.endswith("]")):
            raise ParseError(f"{m['name']}: write the data as a list — {m['name']}([1, 2, 3])")
        return _calc(op, [_split_args(inner[1:-1])], source=line.strip())
    if shape == "special":
        return _calc("special", [m["name"], *args], source=line.strip())
    return _calc(op, args, source=line.strip())
