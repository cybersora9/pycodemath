"""``pycodemath intro`` — the presentation, shown once after install.

What it shows, top to bottom (the static final frame is 30 lines, 80 columns):

1. an 80-column box in krew: the mark (``_logo.FULL``, rasterized from
   ``assets/pycodemath-sygnet.svg`` by ``scripts/gen_logo.py``: 36 columns,
   stroke 12, transparent background) and beside it the name, version,
   tagline, the number of commands and API functions (counted at run time,
   ``counts``) and the byline;
2. ONE live demo — a real problem worked on screen: ``sum_to(n)`` is meant to
   add 0 + 1 + ... + n. The engine reads the loop as a formula, finds the input
   where it is wrong and RUNS it, the loop is fixed, and the engine proves the
   fix. Every number on screen comes from ``check_code`` at run time;
3. the capability map (``discover.MAP``): eight areas, each with a command you
   can paste;
4. where to go next.

Animation happens only on a TTY, and never with ``--static``,
``PYCODEMATH_NO_ANIM=1`` (reduced motion) or when stdout is a pipe: then the
final frame is printed as is, with no escape bytes.

When it shows by itself — once per installed version
----------------------------------------------------
``repl.run()`` (the ``pycodemath`` script and ``python -m pycodemath``) calls
``first_run`` before doing anything else. On the first run after installing or
upgrading — stdout a TTY, and the marker does not name this version — the
intro plays, then the command runs normally. The marker is ``intro-shown``
(it holds the version shown) in the user cache directory
(``PYCODEMATH_CACHE_DIR``, else ``$XDG_CACHE_HOME/pycodemath``,
``~/Library/Caches/pycodemath`` on macOS, ``%LOCALAPPDATA%\\pycodemath`` on
Windows, ``~/.cache/pycodemath`` elsewhere). It is written BEFORE the intro
plays; if it cannot be written the intro is skipped, because it would
otherwise repeat on every run.

Never: when stdout is not a TTY (scripts, pipes, AI agents through a shell),
in the MCP server (it does not go through ``repl.run``), or with
``PYCODEMATH_NO_INTRO=1``. ``pycodemath intro`` replays it any time;
``pycodemath intro --once`` shows it only if this version's intro has not been
shown on this machine yet (agent skills call it), and is silent otherwise.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Mapping, Sequence

from .. import __version__
from . import _logo
from .term import Style, color_mode, logo_rows, logo_width, visible_len

WIDTH = 80

# --- strings ---------------------------------------------------------------------------
S: dict[str, str] = {
    "tagline": "Checks the math in your code.",
    "counts": "{n} commands · {m} API functions",
    "by": "by cybersora",
    "demo": "DEMO",
    "question": "Does sum_to(n) add 0 + 1 + ... + n = n(n+1)/2? (n = 0, 1, 2, ...)",
    "s1": "Read the loop as a formula",
    "s2": "Find where it's wrong, run it",
    "s2_res": "{bad}  {call} returned {got}, not {want}",
    "s3": "Fix the loop: {old} -> {fix}",
    "s4": "Prove the fixed loop correct",
    "s4_res": "{good}  {form} equals n(n+1)/2",
    "map": "WHAT ELSE IT DOES",
    "map_alone": "WHAT IT DOES",
    "live": "Live demo: run  pycodemath intro  in a terminal.",
    "try": "TRY",
    "next": "next: pycodemath commands · pycodemath examples <topic> · pycodemath tour",
    "banner2": "type intro · commands · help <topic> · quit",
}

#: The demo's two programs. The ONLY difference is the loop bound.
DEMO_BAD = (
    "def sum_to(n: int) -> int:\n"
    "    total = 0\n"
    "    for k in range(n):\n"
    "        total += k\n"
    "    return total\n"
)
DEMO_GOOD = DEMO_BAD.replace("range(n)", "range(n + 1)")
DEMO_FORMULA = "n*(n+1)/2"
DEMO_DOMAIN = {"n": "integer, n>=0"}


def _check(source: str, domain: "Mapping[str, str] | None" = None):  # -> CodeCheck
    """One real engine call — looked up at call time, so a test can replace it."""
    from ..code import frontend

    return frontend.check_code(source, DEMO_FORMULA, domain=dict(domain) if domain else None)


def counts() -> tuple[int, int]:
    """``(commands, API functions)``, counted now so the header never goes stale:
    the one-shot/REPL command table and the FULL-series table together, and the
    functions in ``pycodemath.__all__``."""
    import inspect

    import pycodemath

    from . import full, repl

    commands = len(set(repl._COMMANDS) | set(full.COMMANDS))
    functions = sum(1 for name in pycodemath.__all__
                    if inspect.isfunction(getattr(pycodemath, name)))
    return commands, functions


# --- the player: lines that may wait for the engine -------------------------------------
@dataclass
class Beat:
    """One line of output: ``lead`` at once, then ``rest()`` (may call the engine).

    Animated, ``lead`` appears, a spinner turns while ``rest`` runs in a
    thread, then the whole line is rewritten in place; ``pause`` follows.
    Static, the line is printed whole. Either way the final text is identical.
    """

    lead: str
    rest: "Callable[[], str] | None" = None
    pause: float = 0.0


_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def play(beats: Sequence[Beat], out: "IO[str]", animate: bool, style: Style,
         min_wait: float = 0.35) -> None:
    for beat in beats:
        if not animate:
            out.write(beat.lead + (beat.rest() if beat.rest else "") + "\n")
            continue
        if beat.rest is None:
            out.write(beat.lead + "\n")
            out.flush()
            time.sleep(beat.pause)
            continue
        box: dict[str, object] = {}

        def work(rest: Callable[[], str] = beat.rest) -> None:
            try:
                box["text"] = rest()
            except BaseException as exc:  # re-raised on the main thread
                box["error"] = exc

        worker = threading.Thread(target=work, daemon=True)
        started = time.monotonic()
        worker.start()
        i = 0
        while worker.is_alive() or time.monotonic() - started < min_wait:
            out.write("\r" + beat.lead + _SPIN[i % len(_SPIN)])
            out.flush()
            i += 1
            if worker.is_alive():
                worker.join(0.08)
            else:  # the engine was quicker than the shortest spin
                time.sleep(0.08)
        if "error" in box:
            out.write("\n")
            raise box["error"]  # type: ignore[misc]
        line = beat.lead + str(box["text"])
        out.write("\r" + line + ("\x1b[K" if style.color else "  ") + "\n")
        out.flush()
        time.sleep(beat.pause)


# --- the intro ---------------------------------------------------------------------------
def _verdict(style: Style, verdict: str) -> str:
    word = verdict.upper()  # the engine's own word, as every other command prints it
    if verdict == "refuted":
        return style.refuted(word)
    if verdict == "verified":
        return style.verified(word)
    return style.bold(word)


def demo_beats(style: Style, indent: str = "  ") -> list[Beat]:
    """The off-by-one demo as beats. All values are read from ``check_code``."""
    cache: dict[str, object] = {}

    def bad():  # one call serves steps 1 and 2
        if "bad" not in cache:
            cache["bad"] = _check(DEMO_BAD, DEMO_DOMAIN)
        return cache["bad"]

    def step(n: int, caption: str) -> str:
        return f"{indent}{style.accent(str(n))} {caption:<31}"

    def s1() -> str:
        r = bad()
        return f"sum_to(n) = {r.symbolic_form}"  # type: ignore[attr-defined]

    def s2() -> str:
        r = bad()
        if r.counterexample is None:  # type: ignore[attr-defined]
            return _verdict(style, r.verdict)  # type: ignore[attr-defined]
        ce = r.counterexample  # type: ignore[attr-defined]
        return S["s2_res"].format(bad=_verdict(style, r.verdict), call=ce.call,  # type: ignore[attr-defined]
                                  got=ce.function_value, want=ce.formula_value)

    def s4() -> str:
        r = _check(DEMO_GOOD, DEMO_DOMAIN)
        if not r.verified:
            return _verdict(style, r.verdict) + f" {r.reason}"
        return S["s4_res"].format(good=_verdict(style, r.verdict), form=r.symbolic_form)

    return [
        Beat(f"{style.accent(S['demo'])}  {S['question']}", pause=0.25),
        Beat(f"{indent}  for k in {style.refuted('range(n)')}: total += k", pause=0.25),
        Beat(step(1, S["s1"]), s1, pause=0.3),
        Beat(step(2, S["s2"]), s2, pause=0.5),
        Beat(step(3, S["s3"].format(old=style.refuted("range(n)"), fix=style.accent("range(n + 1)"))), pause=0.4),
        Beat(step(4, S["s4"]), s4, pause=0.3),
    ]


def name_line(style: Style) -> str:
    """``pycodemath  v<version>``: bold, py/math kosc, code krew, the version gray."""
    name = (style.paint("py", fg="kosc", bold=True) + style.paint("code", fg="krew", bold=True)
            + style.paint("math", fg="kosc", bold=True))
    return f"{name}  {style.dim('v' + __version__)}"


def text_block(style: Style) -> list[str]:
    n, m = counts()
    return [name_line(style), S["tagline"], "", S["counts"].format(n=n, m=m), style.dim(S["by"])]


def header_lines(style: Style, logo: bool = True) -> list[str]:
    """The rounded krew box: ``│ <logo>  <text> │``, the text centred on the logo rows."""
    text = text_block(style)
    left = logo_rows(_logo.FULL, style) if logo else [""] * len(text)
    lead = 1 + (logo_width(_logo.FULL) + 2 if logo else 0)
    inner = WIDTH - 2
    top = (len(left) - len(text)) // 2
    edge = style.paint("│", fg="krew")
    out = [style.paint("╭" + "─" * inner + "╮", fg="krew")]
    for i, row in enumerate(left):
        right = text[i - top] if 0 <= i - top < len(text) else ""
        cell = " " + (row + "  " if logo else "") + right
        out.append(edge + cell + " " * (inner - lead - visible_len(right)) + edge)
    out.append(style.paint("╰" + "─" * inner + "╯", fg="krew"))
    return out


def map_lines(style: Style, heading: str = "map") -> list[str]:
    from .discover import MAP

    lines = [style.accent(S[heading]) + " " * (34 - len(S[heading])) + style.accent(S["try"])]
    for _topic, label, example in MAP:
        lines.append(f"{label:<34}pycodemath \"{example}\"")
    return lines


def intro_beats(style: Style, logo: bool = True) -> list[Beat]:
    beats = [Beat(line, pause=0.05) for line in header_lines(style, logo)]
    beats.append(Beat(""))
    beats += demo_beats(style)
    beats.append(Beat(""))
    beats += [Beat(line, pause=0.04) for line in map_lines(style)]
    beats.append(Beat(S["next"]))
    return beats


def _isatty(out: "IO[str]") -> bool:
    try:
        return bool(out.isatty())
    except (AttributeError, ValueError):
        return False


def should_animate(out: "IO[str]", static: bool, env: "Mapping[str, str] | None" = None) -> bool:
    env = os.environ if env is None else env
    if static or env.get("PYCODEMATH_NO_ANIM", "") not in ("", "0"):
        return False
    return _isatty(out)


def show(out: "IO[str] | None" = None, *, static: bool = False, logo: bool = True,
         env: "Mapping[str, str] | None" = None) -> None:
    """Render the intro to ``out`` (stdout): animated on a TTY, else the final frame."""
    out = sys.stdout if out is None else out
    style = Style(color_mode(out, env))
    animate = should_animate(out, static, env)
    hide = animate and style.color
    if hide:
        out.write("\x1b[?25l")
    try:
        play(intro_beats(style, logo), out, animate, style)
    finally:
        if hide:
            out.write("\x1b[?25h")
        out.flush()


def repl_banner(out: "IO[str] | None" = None) -> str:
    """The interactive REPL's 3-line banner: the text mark ``∫>_``, version, where to start."""
    out = sys.stdout if out is None else out
    style = Style(color_mode(out))
    mark = (style.paint("∫", fg="krew", bold=True) + style.paint(">", fg="kosc", bold=True)
            + style.paint("_", fg="krew", bold=True))
    return "\n".join([f"{mark}  {name_line(style)} — {S['tagline']}", f"     {S['banner2']}", ""])


def capabilities() -> dict:
    """The MCP ``intro`` payload: plain text, logo, tagline and the map as data."""
    from .discover import MAP, TOPICS, help_entries

    mono = Style("mono")
    entries = help_entries()
    areas = []
    for topic, label, example in MAP:
        keys = TOPICS[topic].keys
        areas.append({
            "topic": topic,
            "label": label,
            "example": example,
            "commands": [text.splitlines()[0].split(" — ")[0].strip()
                         for key, text in entries if key in keys],
        })
    text = "\n".join(header_lines(mono) + [""] + map_lines(mono, "map_alone") + [S["next"], S["live"]])
    return {
        "text": text,
        "logo": "\n".join(row.rstrip() for row in logo_rows(_logo.FULL, mono)),
        "version": __version__,
        "tagline": S["tagline"],
        "areas": areas,
        "next": ["pycodemath commands", "pycodemath examples <topic>", "pycodemath tour"],
    }


# --- once per installed version ---------------------------------------------------------
MARKER = "intro-shown"


def cache_dir(env: "Mapping[str, str] | None" = None) -> Path:
    env = os.environ if env is None else env
    if env.get("PYCODEMATH_CACHE_DIR"):
        return Path(env["PYCODEMATH_CACHE_DIR"])
    if sys.platform == "win32" and env.get("LOCALAPPDATA"):
        return Path(env["LOCALAPPDATA"]) / "pycodemath"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "pycodemath"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "pycodemath"
    return Path.home() / ".cache" / "pycodemath"


def shown_version(env: "Mapping[str, str] | None" = None) -> "str | None":
    """The version whose intro was last shown on this machine, or None."""
    try:
        return (cache_dir(env) / MARKER).read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def seen(env: "Mapping[str, str] | None" = None) -> bool:
    """Has the intro of THIS installed version been shown here already?"""
    return shown_version(env) == __version__


def mark_seen(env: "Mapping[str, str] | None" = None) -> bool:
    """Record this version as shown; False (not an error) if the directory is not writable."""
    marker = cache_dir(env) / MARKER
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(__version__ + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def disabled(env: "Mapping[str, str] | None" = None) -> bool:
    env = os.environ if env is None else env
    return env.get("PYCODEMATH_NO_INTRO", "") not in ("", "0")


def first_run(out: "IO[str] | None" = None, env: "Mapping[str, str] | None" = None) -> bool:
    """Play the intro if this is the first TTY run of this version; True if it played.

    Called by ``repl.run()`` before the command, which then runs normally. Cheap
    when it does nothing: a TTY check, an env lookup, one small file read.
    """
    out = sys.stdout if out is None else out
    if not _isatty(out) or disabled(env) or seen(env):
        return False
    if not mark_seen(env):  # we could not remember it: better never than every run
        return False
    try:
        show(out, env=env)
    except KeyboardInterrupt:  # skips the intro, not the command
        out.write("\n")
    out.write("\n")
    out.flush()
    return True


# --- command line --------------------------------------------------------------------------
USAGE = "Usage: intro [--static] [--no-logo] [--once]"


def main(argv: Sequence[str], out: "IO[str] | None" = None) -> int:
    """``intro`` with its flags; returns the exit code."""
    out = sys.stdout if out is None else out
    static = once = False
    logo = True
    for a in argv:
        if a == "--static":
            static = True
        elif a == "--no-logo":
            logo = False
        elif a == "--once":
            once = True
        else:
            print(f"error: unknown option {a!r}. {USAGE}", file=sys.stderr)
            return 1
    if once and (disabled() or seen()):
        return 0
    try:
        show(out, static=static, logo=logo)
    except KeyboardInterrupt:
        out.write("\n")
        return 130
    if once or _isatty(out):  # seen on a screen (or handed to an agent that asked for once)
        mark_seen()
    return 0
