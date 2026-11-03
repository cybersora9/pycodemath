"""``pycodemath intro`` — the first-run presentation.

What it shows, top to bottom (the static final frame is under 25 lines and 80
columns):

1. the mark, rasterized from ``assets/pycodemath-sygnet.svg`` by
   ``scripts/gen_logo.py`` (``_logo.py``), with the name, version and tagline;
2. ONE live demo — a real problem worked on screen: ``sum_to(n)`` is meant to
   add 0 + 1 + ... + n. The engine reads the loop as a formula, finds the input
   where it is wrong and RUNS it, the loop is fixed, and the engine proves the
   fix. Every number on screen comes from ``check_code`` at run time;
3. the capability map (``discover.MAP``): eight areas, each with a command you
   can paste;
4. where to go next.

Animation happens only on a TTY, and never with ``--static``,
``PYCODEMATH_NO_ANIM=1`` (reduced motion) or when stdout is a pipe: then the
final frame is printed as is, so an agent calling ``pycodemath intro`` through
a non-TTY shell still gets logo + map, with no escape bytes.

``--once`` — "show it the first time in this session"
------------------------------------------------------
The marker is a file ``intro-<hash>`` in the user cache directory
(``PYCODEMATH_CACHE_DIR``, else ``$XDG_CACHE_HOME/pycodemath``,
``~/Library/Caches/pycodemath`` on macOS, ``%LOCALAPPDATA%\\pycodemath`` on
Windows, ``~/.cache/pycodemath`` elsewhere). The hash is of the first
session key found, in this order (``session_key``):

1. ``PYCODEMATH_SESSION`` — set it yourself to force a key;
2. ``CLAUDE_CODE_SESSION_ID`` — one Claude Code session = one intro, however
   many shells the agent spawns;
3. a terminal's own session id: ``TERM_SESSION_ID`` (macOS Terminal),
   ``ITERM_SESSION_ID``, ``WT_SESSION`` (Windows Terminal), ``TMUX`` +
   ``TMUX_PANE``, ``STY`` (screen), ``WEZTERM_PANE``, ``KITTY_WINDOW_ID``;
4. POSIX: the process session id (``os.getsid(0)`` — the login shell or
   terminal that started us; stable across commands in one terminal tab)
   plus the boot time, so a reused PID after a reboot is a new session;
5. otherwise (Windows console without Windows Terminal): the parent PID plus
   the boot time.

Keys 4-5 are PIDs, which the OS recycles, so their markers also expire after
24 hours. Markers older than 7 days are pruned. ``--once`` never animates
differently: it is ``intro`` or nothing, exit code 0 either way.
"""

from __future__ import annotations

import hashlib
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

# --- strings ---------------------------------------------------------------------------
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "tagline": "A proof, or the exact input that breaks your math.",
        "sub": "100+ functions · exact · checked · CLI, Python, MCP",
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
    },
    "pl": {
        "tagline": "Dowód albo wejście, które łamie twoją matematykę.",
        "sub": "100+ funkcji · dokładne wyniki · CLI, Python, MCP",
        "demo": "DEMO",
        "question": "Czy sum_to(n) sumuje 0 + 1 + ... + n = n(n+1)/2? (n = 0, 1, 2, ...)",
        "s1": "Pętla odczytana jako wzór",
        "s2": "Znajdź błąd i uruchom go",
        "s2_res": "{bad}  {call} zwróciło {got}, nie {want}",
        "s3": "Popraw pętlę: {old} -> {fix}",
        "s4": "Udowodnij poprawioną pętlę",
        "s4_res": "{good}  {form} równa się n(n+1)/2",
        "map": "CO JESZCZE UMIE",
        "map_alone": "CO UMIE",
        "live": "Demo na żywo: uruchom  pycodemath intro  w terminalu.",
        "try": "SPRÓBUJ",
        "next": "dalej: pycodemath commands · pycodemath examples <temat> · pycodemath tour",
        "banner2": "wpisz intro · commands · help <temat> · quit",
    },
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
def _verdict(style: Style, s: dict[str, str], verdict: str) -> str:
    word = verdict.upper()  # the engine's own word, as every other command prints it
    if verdict == "refuted":
        return style.refuted(word)
    if verdict == "verified":
        return style.verified(word)
    return style.bold(word)


def demo_beats(style: Style, lang: str = "en", indent: str = "  ") -> list[Beat]:
    """The off-by-one demo as beats. All values are read from ``check_code``."""
    s = STRINGS[lang]
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
            return _verdict(style, s, r.verdict)  # type: ignore[attr-defined]
        ce = r.counterexample  # type: ignore[attr-defined]
        return s["s2_res"].format(bad=_verdict(style, s, r.verdict), call=ce.call,  # type: ignore[attr-defined]
                                  got=ce.function_value, want=ce.formula_value)

    def s4() -> str:
        r = _check(DEMO_GOOD, DEMO_DOMAIN)
        if not r.verified:
            return _verdict(style, s, r.verdict) + f" {r.reason}"
        return s["s4_res"].format(good=_verdict(style, s, r.verdict), form=r.symbolic_form)

    return [
        Beat(f"{style.accent(s['demo'])}  {s['question']}", pause=0.25),
        Beat(f"{indent}  for k in {style.refuted('range(n)')}: total += k", pause=0.25),
        Beat(step(1, s["s1"]), s1, pause=0.3),
        Beat(step(2, s["s2"]), s2, pause=0.5),
        Beat(step(3, s["s3"].format(old=style.refuted("range(n)"), fix=style.accent("range(n + 1)"))), pause=0.4),
        Beat(step(4, s["s4"]), s4, pause=0.3),
    ]


def header_lines(style: Style, lang: str, logo: bool) -> list[str]:
    s = STRINGS[lang]
    text = ["", "", f"{style.bold('pycodemath')} {__version__}", s["tagline"], s["sub"]]
    if not logo:
        return [line for line in text if line]
    width = logo_width(_logo.FULL, style)
    out = []
    for i, row in enumerate(logo_rows(_logo.FULL, style)):
        right = text[i] if i < len(text) else ""
        if right:
            row += " " * (width - visible_len(row)) + "  " + right
        out.append(row if style.color else row.rstrip())
    return out


def map_lines(style: Style, lang: str, heading: str = "map") -> list[str]:
    from .discover import MAP

    s = STRINGS[lang]
    lines = [style.accent(s[heading]) + " " * (34 - len(s[heading])) + style.accent(s["try"])]
    for _topic, label_en, label_pl, example in MAP:
        label = label_pl if lang == "pl" else label_en
        lines.append(f"{label:<34}pycodemath \"{example}\"")
    return lines


def intro_beats(style: Style, lang: str = "en", logo: bool = True) -> list[Beat]:
    beats = [Beat(line, pause=0.05) for line in header_lines(style, lang, logo)]
    beats.append(Beat(""))
    beats += demo_beats(style, lang)
    beats.append(Beat(""))
    beats += [Beat(line, pause=0.04) for line in map_lines(style, lang)]
    beats.append(Beat(STRINGS[lang]["next"]))
    return beats


def should_animate(out: "IO[str]", static: bool, env: "Mapping[str, str] | None" = None) -> bool:
    env = os.environ if env is None else env
    if static or env.get("PYCODEMATH_NO_ANIM", "") not in ("", "0"):
        return False
    try:
        return bool(out.isatty())
    except (AttributeError, ValueError):
        return False


def show(out: "IO[str] | None" = None, *, static: bool = False, logo: bool = True,
         lang: str = "en", env: "Mapping[str, str] | None" = None) -> None:
    """Render the intro to ``out`` (stdout): animated on a TTY, else the final frame."""
    out = sys.stdout if out is None else out
    style = Style(color_mode(out, env))
    animate = should_animate(out, static, env)
    hide = animate and style.color
    if hide:
        out.write("\x1b[?25l")
    try:
        play(intro_beats(style, lang, logo), out, animate, style)
    finally:
        if hide:
            out.write("\x1b[?25h")
        out.flush()


def repl_banner(out: "IO[str] | None" = None, lang: str = "en") -> str:
    """The interactive REPL's 3-line banner: mini mark, version, where to start."""
    out = sys.stdout if out is None else out
    style = Style(color_mode(out))
    s = STRINGS[lang]
    rows = logo_rows(_logo.MINI, style)
    text = [f"{style.bold('Pycodemath')} {__version__} — heavy math + Python",
            s["banner2"], ""]
    return "\n".join((row + "  " + t).rstrip() for row, t in zip(rows, text))


def capabilities(lang: str = "en") -> dict:
    """The MCP ``intro`` payload: plain text, logo, tagline and the map as data."""
    from .discover import MAP, TOPICS, help_entries

    mono = Style("mono")
    s = STRINGS[lang]
    entries = help_entries()
    areas = []
    for topic, label_en, label_pl, example in MAP:
        keys = TOPICS[topic].keys
        areas.append({
            "topic": topic,
            "label": label_pl if lang == "pl" else label_en,
            "example": example,
            "commands": [text.splitlines()[0].split(" — ")[0].strip()
                         for key, text in entries if key in keys],
        })
    text = "\n".join(header_lines(mono, lang, True) + [""] + map_lines(mono, lang, "map_alone")
                     + [s["next"], s["live"]])
    return {
        "text": text,
        "logo": "\n".join(row.rstrip() for row in logo_rows(_logo.FULL, mono)),
        "version": __version__,
        "tagline": s["tagline"],
        "areas": areas,
        "next": ["pycodemath commands", "pycodemath examples <topic>", "pycodemath tour"],
    }


# --- --once ------------------------------------------------------------------------------
_TERMINAL_KEYS = ("TERM_SESSION_ID", "ITERM_SESSION_ID", "WT_SESSION", "STY",
                  "WEZTERM_PANE", "KITTY_WINDOW_ID")
_PID_TTL = 24 * 3600
_PRUNE_AGE = 7 * 24 * 3600


def _boot_time() -> str:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return line.split()[1]
    except OSError:
        pass
    # elsewhere: wall clock minus the monotonic clock, to the minute
    return str(round((time.time() - time.monotonic()) / 60))


def session_key(env: "Mapping[str, str] | None" = None) -> tuple[str, bool]:
    """``(key, is_pid_based)`` — see the module docstring for the order."""
    env = os.environ if env is None else env
    if env.get("PYCODEMATH_SESSION"):
        return "explicit:" + env["PYCODEMATH_SESSION"], False
    if env.get("CLAUDE_CODE_SESSION_ID"):
        return "claude-code:" + env["CLAUDE_CODE_SESSION_ID"], False
    if env.get("TMUX") and env.get("TMUX_PANE"):
        return f"tmux:{env['TMUX']}:{env['TMUX_PANE']}", False
    for name in _TERMINAL_KEYS:
        if env.get(name):
            return f"{name}:{env[name]}", False
    getsid = getattr(os, "getsid", None)
    if getsid is not None:
        try:
            return f"sid:{getsid(0)}:{_boot_time()}", True
        except OSError:
            pass
    return f"ppid:{os.getppid()}:{_boot_time()}", True


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


def _marker(env: "Mapping[str, str] | None" = None) -> tuple[Path, bool]:
    key, pid_based = session_key(env)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir(env) / f"intro-{digest}", pid_based


def seen_this_session(env: "Mapping[str, str] | None" = None) -> bool:
    marker, pid_based = _marker(env)
    try:
        age = time.time() - marker.stat().st_mtime
    except OSError:
        return False
    return not (pid_based and age > _PID_TTL)


def mark_seen(env: "Mapping[str, str] | None" = None) -> None:
    """Write the marker; a cache directory we cannot write is not an error."""
    marker, _ = _marker(env)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for old in marker.parent.glob("intro-*"):
            try:
                if now - old.stat().st_mtime > _PRUNE_AGE:
                    old.unlink()
            except OSError:
                pass
        marker.write_text(session_key(env)[0] + "\n", encoding="utf-8")
    except OSError:
        pass


# --- command line --------------------------------------------------------------------------
USAGE = "Usage: intro [--static] [--no-logo] [--lang en|pl] [--once]"


def main(argv: Sequence[str], out: "IO[str] | None" = None) -> int:
    """``intro`` with its flags; returns the exit code."""
    out = sys.stdout if out is None else out
    static = once = False
    logo = True
    lang = "en"
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--static":
            static = True
        elif a == "--no-logo":
            logo = False
        elif a == "--once":
            once = True
        elif a == "--lang" and args:
            lang = args.pop(0).lower()
        elif a.startswith("--lang="):
            lang = a.split("=", 1)[1].lower()
        else:
            print(f"error: unknown option {a!r}. {USAGE}", file=sys.stderr)
            return 1
    if lang not in STRINGS:
        print(f"error: --lang must be one of {', '.join(STRINGS)}", file=sys.stderr)
        return 1
    if once and seen_this_session():
        return 0
    try:
        show(out, static=static, logo=logo, lang=lang)
    except KeyboardInterrupt:
        out.write("\n")
        return 130
    if once:
        mark_seen()
    return 0
