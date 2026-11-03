"""Finding your way: ``commands``, ``help <topic>``, ``examples <topic>``, ``tour``.

ONE table (``TOPICS``) drives all of them and the capability map of
``intro``. The help text itself is NOT copied here: ``help <topic>`` cuts the
entries out of the REPL's own help (``repl.__doc__``) by their first word, so
there is still exactly one place where a command's syntax is written down. A
test fails if an entry of that help belongs to no topic.

Every example is a real one-shot command; ``tests/test_intro.py`` runs each of
them and requires exit code 0 and no usage message, so they cannot rot.
"""

from __future__ import annotations

import re
import shlex
import sys
import textwrap
from dataclasses import dataclass
from functools import partial
from typing import IO, TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from .intro import Beat
    from .term import Style

WIDTH = 80


@dataclass(frozen=True)
class Topic:
    title: str
    keys: frozenset[str]
    #: (plain-language comment, one-shot command)
    examples: tuple[tuple[str, str], ...]


def _t(title: str, keys: str, *examples: tuple[str, str]) -> Topic:
    return Topic(title, frozenset(keys.split()), tuple(examples))


TOPICS: dict[str, Topic] = {
    "solve": _t(
        "equations, inequalities, systems, matrices, roots, minima",
        "solve reduce root solve_nd min min_nd matrix det inv transpose eig solve_system code_system",
        ("an inequality: the sign of every interval is proved", "solve x^2-4 > 0 for x"),
        ("equations solved exactly", "solve x^2-4 for x"),
        ("a linear system, with a proof that nothing is missing", "solve x + y = 3, x - y = 1 for x, y"),
        ("every solution of a periodic equation", "solve sin(x) = 0 all"),
        ("a matrix system A x = b", "solve_system [[2,1],[1,3]] = [3,5]"),
    ),
    "calculus": _t(
        "derivatives, integrals, limits, series, sums, gradients",
        "diff integrate limit series sum nintegrate grad",
        ("an antiderivative", "integrate x*cos(x) dx"),
        ("a derivative", "diff sin(x)*x dx"),
        ("an improper integral: convergence proved, value cross-checked", "integrate exp(-x^2) dx from -oo to oo"),
        ("a limit", "limit sin(x)/x for x to 0"),
        ("a closed form for a sum", "sum k for k from 1 to n"),
    ),
    "verify": _t(
        "is it true? VERIFIED, REFUTED (counterexample), UNDECIDED",
        "verify verify_steps certify certificate recheck",
        ("a claim that is false for negative x", "verify sqrt(x^2) == x"),
        ("an identity, proved", "verify (x+1)^2 == x^2 + 2*x + 1"),
        ("find the first wrong step of a derivation", "verify steps (x+1)^2 = x^2 + 2*x + 1 = x^2 + 1"),
        ("compute, then check the answer by another route", "certify integrate x*cos(x) dx"),
    ),
    "code": _t(
        "the math in your code: formulas, float error, overflow",
        "verify_code stability improve ranges maxsafe iterations complexity recurrence master lint code",
        ("the largest input exp(x) survives in float32", "maxsafe exp(x) for x as float32"),
        ("how many bits a formula loses in float64, and where", "stability sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15]"),
        ("a rewrite proved equal and measured more accurate", "improve sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15]"),
        ("a midpoint that overflows int16, with the input that triggers it",
         "ranges (a + b) // 2 for a in [0, 32767], b in [0, 32767] as int16"),
        ("the running time of a divide-and-conquer recurrence", "recurrence T(n) = 2*T(n/2) + n, T(1) = 1"),
    ),
    "numbers": _t(
        "exact numbers, proved decimals, primes, modular math",
        "<expression> <exact evalf identify latex factor lucas gcd legendre stirling1 crt special",
        ("factor a 20-digit number, with a certificate", "factor 2^64+1"),
        ("50 digits that are proved, not just printed", "evalf sqrt(2) 50"),
        ("recognise a constant", "identify zeta(2)"),
        ("is it prime? (with a proof)", "isprime 2^61-1"),
        ("the Chinese remainder theorem", "crt 2,3,2 mod 3,5,7"),
    ),
    "money": _t(
        "exact amounts: rounding, splitting, VAT, loans, IRR",
        "money",
        ("split a bill so the cents add up exactly", "money split 100 PLN into 3"),
        ("round half-up, the error stated exactly", "money round 2.675 USD"),
        ("VAT per line vs on the total", "money vat 19.99,5.01 at 23% PLN"),
        ("a loan's monthly payment and total interest", "money loan 10000 PLN at 7% for 12 months"),
        ("every internal rate of return, not just one", "money irr -100,60,60"),
    ),
    "units": _t(
        "exact unit conversion and dimensional checks of formulas",
        "convert units dims",
        ("an exact conversion", "convert 5 km/h to m/s"),
        ("a formula with the wrong dimensions", "units F = m*v where F:N m:kg v:m/s"),
        ("the same formula, fixed", "units F = m*v/t where F:N m:kg v:m/s t:s"),
    ),
    "geometry": _t(
        "rotations, exact orientation tests, easing curves",
        "quat2mat euler2mat mat2euler rotcheck orient2d easing emit",
        ("which side of a line a point is on, decided exactly", "orient2d (0,0) (1,0) (0,1)"),
        ("a quaternion as a rotation matrix", "quat2mat 1 0 0 0"),
        ("Euler angles as a matrix", "euler2mat 90 0 0 seq zyx frame intrinsic deg"),
        ("a CSS easing curve, value proved inside an interval", "easing ease 0.5"),
    ),
    "ode": _t(
        "differential equations: exact, RK4, adaptive, stiff, events",
        "dsolve ode odestiff odestiff_adaptive ode_adaptive odedense odeevents",
        ("solve y' = y exactly", "dsolve y for y(t)"),
        ("the same numerically (RK4)", "ode y for y(t) from 0 to 1 at 1"),
        ("adaptive step size", "ode_adaptive -y for y(t) from 0 to 1 at 1"),
        ("a stiff equation, where explicit methods blow up", "odestiff -1000*(y - cos(t)) for y(t) from 0 to 1 at 0"),
        ("a falling ball: stop when it hits the ground", "odeevents -9.81 for y(t) from 0 to 5 at 10 zero y stop"),
    ),
    "stats": _t(
        "descriptive statistics, distributions, A/B tests",
        "mean quantile cdf samplesize power welch",
        ("an exact mean", "mean 1 2 3 4"),
        ("a quantile of Student's t", "ppf t 0.975 df=10"),
        ("a confidence interval for 45 out of 100", "ci wilson 45/100"),
        ("how many users an A/B test needs", "samplesize p1=0.10 p2=0.12"),
        ("is 45/100 vs 60/100 a real difference?", "ztest 45/100 60/100"),
    ),
    "start": _t(
        "where to begin: intro, commands, help, examples, tour",
        "intro commands examples tour help implicit quit",
    ),
}

#: The capability map of ``intro``: (topic, label, example). The example is the
#: topic's first one, so the test that runs every example covers the map too.
MAP: tuple[tuple[str, str, str], ...] = tuple(
    (topic, label, TOPICS[topic].examples[0][1])
    for topic, label in (
        ("solve", "Solve, with a proof"),
        ("calculus", "Calculus and ODEs"),
        ("verify", "Check a claim or a derivation"),
        ("code", "Overflow and float error in code"),
        ("numbers", "Exact numbers and statistics"),
        ("money", "Money to the cent"),
        ("units", "Units and dimensions"),
        ("geometry", "Geometry and rotations"),
    )
)

HELP_TOPICS = tuple(TOPICS)


# --- the help text, cut by topic ------------------------------------------------------
def _entry_key(text: str) -> str:
    if text.startswith("verify code"):
        return "verify_code"
    if text.startswith("verify steps"):
        return "verify_steps"
    return re.split(r"[\s|;]", text, maxsplit=1)[0]


def help_entries() -> list[tuple[str, str]]:
    """``(key, entry text)`` for every command entry of the REPL help, in order.

    An entry starts on a line indented by exactly two spaces and runs over the
    deeper-indented lines after it; unindented lines (section headers, prose)
    and blank lines end it.
    """
    from . import repl

    doc = repl.__doc__ or ""
    body = doc.split("Commands:", 1)[1].split("\nStrict parsing", 1)[0]
    entries: list[list[str]] = []
    for line in body.splitlines():
        if re.match(r"^  \S", line):
            entries.append([line])
        elif re.match(r"^   +\S", line) and entries:
            entries[-1].append(line)
        else:
            if entries and entries[-1] and entries[-1][-1] != "":
                entries[-1].append("")  # closes it
    out = []
    for e in entries:
        lines = [x for x in e if x]
        out.append((_entry_key(lines[0][2:]), "\n".join(lines)))
    return out


def topic_of(key: str) -> "str | None":
    for name, topic in TOPICS.items():
        if key in topic.keys:
            return name
    return None


def topic_help(name: str) -> str:
    topic = TOPICS[name]
    body = [text for key, text in help_entries() if key in topic.keys]
    lines = [f"help {name} — {topic.title}", "", *body]
    if topic.examples:
        lines += ["", f"examples: pycodemath examples {name}"]
    return "\n".join(lines)


def _first_line(text: str) -> str:
    """The entry's first line if it fits 80 columns, else its syntax alone (cut
    with ``…`` past 80; ``help <topic>`` has the whole entry)."""
    first = text.splitlines()[0].rstrip()
    if len(first) <= WIDTH:
        return first
    syntax = first.split(" — ")[0].rstrip()
    return syntax if len(syntax) <= WIDTH else syntax[: WIDTH - 1] + "…"


def commands_text() -> str:
    """Every command, grouped by topic, one line each."""
    entries = help_entries()
    out = ["All commands by topic. More: pycodemath help <topic> / examples <topic>"]
    for name, topic in TOPICS.items():
        out += ["", f"{name.upper()} — {topic.title}"]
        out += [_first_line(text) for key, text in entries if key in topic.keys]
    return "\n".join(out)


def examples_text(name: str) -> str:
    topic = TOPICS[name]
    out = [f"examples {name} — {topic.title}", "Copy a line and run it:", ""]
    for comment, command in topic.examples:
        out.append(f"  # {comment}")
        out.append(f'  pycodemath "{command}"')
    return "\n".join(out)


def topics_line() -> str:
    return "topics: " + ", ".join(t for t in TOPICS if TOPICS[t].examples or t == "start")


# --- tour ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Demo:
    """A real-life problem worked in steps. Each step is ``(caption, command,
    keep)``: the command runs in the engine at show time; its first line is
    shown, plus every line starting with one of the ``keep`` prefixes (e.g. the
    proof, ``certificate:``)."""

    title: str
    steps: tuple[tuple[str, str, tuple[str, ...]], ...]


_PROOF = ("certificate:",)

TOUR: tuple[Demo, ...] = (
    Demo("Your float formula loses digits for big x. How many, and how to fix it?", (
        ("Measure the error of sqrt(x+1) - sqrt(x) (a bit = a binary digit)",
         "stability sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15]", ("culprit:",)),
        ("Ask for a rewrite that is proved equal and measured better",
         "improve sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15]", _PROOF),
    )),
    Demo("A binary search takes the middle index as (lo + hi) // 2 in int32.", (
        ("Look for inputs that overflow (go past the largest int32)",
         "ranges (lo + hi) // 2 for lo in [0, 2^31-1], hi in [0, 2^31-1] as int32",
         ("confirmed overflow",)),
        ("Check the usual fix over every possible input",
         "ranges lo + (hi - lo) // 2 for lo in [0, 2^31-1], hi in [0, 2^31-1] as int32", _PROOF),
    )),
    Demo("Three friends split a 100 PLN bill. Do the shares add up to the cent?", (
        ("Split exactly; the leftover cent goes to one share", "money split 100 PLN into 3", _PROOF),
    )),
    Demo("Is force = mass * velocity a correct physics formula?", (
        ("Compare the units of both sides", "units F = m*v where F:N m:kg v:m/s", ()),
        ("Divide by time and compare again", "units F = m*v/t where F:N m:kg v:m/s t:s", ()),
    )),
)


def _engine_text(command: str, keep: tuple[str, ...]) -> str:
    """What the engine prints for ``command``: its first line and the kept ones."""
    from . import repl

    lines = repl.handle(command).splitlines()
    shown = lines[:1] + [x for x in lines[1:] if x.strip().startswith(keep)] if keep else lines[:1]
    wrapped: list[str] = []
    for line in shown:
        line = re.sub(r"(\S) {3,}", r"\1  ", line.strip())
        wrapped += textwrap.wrap(line, WIDTH - 4, subsequent_indent="  ") or [""]
    return "\n".join("    " + w for w in wrapped)


def tour_demo_beats(demo: Demo, style: "Style") -> "list[Beat]":
    from .intro import Beat

    beats = [Beat(style.accent(demo.title), pause=0.3)]
    for i, (caption, command, keep) in enumerate(demo.steps, 1):
        beats.append(Beat(f"  {style.accent(str(i))} {caption}", pause=0.1))
        shown = textwrap.wrap(f'$ pycodemath "{command}"', WIDTH - 4, subsequent_indent="    ")
        beats += [Beat("    " + line, pause=0.05) for line in shown]
        beats.append(Beat("", partial(_engine_text, command, keep), pause=0.4))
    return beats


def tour(out: "IO[str] | None" = None, read: "Callable[[str], str] | None" = None,
         static: bool = False) -> int:
    from .intro import demo_beats, play, should_animate
    from .term import Style, color_mode

    out = sys.stdout if out is None else out
    style = Style(color_mode(out))
    animate = should_animate(out, static)
    interactive = animate and sys.stdin.isatty()
    read = input if read is None else read
    demos: "list[Callable[[], list[Beat]]]" = [partial(demo_beats, style, "  ")]
    demos += [partial(tour_demo_beats, d, style) for d in TOUR]
    total = len(demos)
    for n, make in enumerate(demos, 1):
        out.write(f"\n[{n}/{total}]\n")
        play(make(), out, animate, style)
        if n < total and interactive:
            try:
                answer = read("\nEnter = next, q = quit > ")
            except (EOFError, KeyboardInterrupt):
                out.write("\n")
                return 0
            if answer.strip().lower() in ("q", "quit"):
                return 0
    out.write("\nnext: pycodemath commands · pycodemath examples <topic>\n")
    out.flush()
    return 0


# --- dispatch --------------------------------------------------------------------------
WORDS = ("intro", "commands", "examples", "tour", "help")


def handles(args: Sequence[str]) -> bool:
    """Is this command line one of ours? Bare ``help`` stays the REPL's."""
    words = _words(args)
    if not words or words[0].lower() not in WORDS:
        return False
    return not (words[0].lower() == "help" and len(words) == 1)


def _words(args: Sequence[str]) -> list[str]:
    if len(args) == 1:
        try:
            return shlex.split(args[0])
        except ValueError:
            return args[0].split()
    return list(args)


def main(args: Sequence[str], out: "IO[str] | None" = None) -> int:
    """Run one of the discoverability commands; returns the exit code."""
    out = sys.stdout if out is None else out
    words = _words(args)
    cmd, rest = words[0].lower(), words[1:]
    if cmd == "intro":
        from . import intro

        return intro.main(rest, out)
    if cmd == "tour":
        if any(a != "--static" for a in rest):
            return _fail(out, "Usage: tour [--static]")
        return tour(out, static=bool(rest))
    if cmd == "commands":
        if rest:
            return _fail(out, "Usage: commands")
        out.write(commands_text() + "\n")
        return 0
    # help <topic> / examples <topic>
    if len(rest) != 1 or rest[0].lower() not in TOPICS:
        what = "help" if cmd == "help" else "examples"
        got = f" {rest[0]!r} is not one." if rest else ""
        return _fail(out, f"Usage: {what} <topic>.{got} {topics_line()}")
    name = rest[0].lower()
    if cmd == "examples" and not TOPICS[name].examples:
        return _fail(out, f"no examples for {name!r}. {topics_line()}")
    out.write((topic_help(name) if cmd == "help" else examples_text(name)) + "\n")
    return 0


def _fail(out: "IO[str]", message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1
