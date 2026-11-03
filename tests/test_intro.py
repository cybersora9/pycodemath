"""The presentation and the way-finding commands: intro, commands, help <topic>,
examples <topic>, tour, the REPL banner, ``intro --once`` and the MCP ``intro``.

What is pinned here, and why:

* a one-shot command NEVER prints a banner (stdout of ``pycodemath "<cmd>"`` is
  the answer and nothing else — scripts and agents parse it);
* not a TTY / ``NO_COLOR`` -> not one escape byte, no animation;
* the intro's final frame fits 80 columns and ~25 lines, and every number in
  its demo comes from the engine at run time (the engine is replaced and the
  screen must change);
* every listed example really runs (exit code 0, no usage message);
* ``--once`` shows, then stays silent, per session key.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pycodemath import __version__
from pycodemath.cli import _logo, discover, intro, mcp_server, repl
from pycodemath.cli.term import PALETTE, Style, color_mode, nearest_256, visible_len

ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _screen(raw: str) -> list[str]:
    """What a terminal shows after ``raw``: ``\\r`` rewrites the line, ANSI is invisible."""
    lines, cur = [], ""
    for part in re.split(r"(\r|\n)", ANSI.sub("", raw)):
        if part == "\r":
            cur = ""
        elif part == "\n":
            lines.append(cur.rstrip())
            cur = ""
        else:
            cur += part
    if cur:
        lines.append(cur.rstrip())
    return lines


def _words(lines: list[str]) -> list[str]:
    """The text of each line, the logo's block glyphs (which differ between the
    coloured ``▀`` cells and the mono shapes) and spacing left out."""
    return [" ".join(re.sub("[▀▄█]", " ", x).split()) for x in lines]


def _static(lang: str = "en", logo: bool = True) -> str:
    buf = io.StringIO()
    intro.show(buf, static=True, lang=lang, logo=logo, env={})
    return buf.getvalue()


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["pycodemath", *argv])
    code = 0
    try:
        repl.run()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    out, err = capsys.readouterr()
    return code, out, err


# --- intro: the final frame -----------------------------------------------------------
@pytest.mark.parametrize("lang", ["en", "pl"])
@pytest.mark.parametrize("logo", [True, False])
def test_the_final_frame_fits_80_columns_and_25_lines(lang: str, logo: bool) -> None:
    lines = _static(lang, logo).splitlines()
    assert len(lines) <= 25, len(lines)
    for line in lines:
        assert len(line) <= 80, (len(line), line)
    tty = FakeTTY()  # in colour the logo sits on a panel two columns wider
    intro.show(tty, static=True, lang=lang, logo=logo, env={"COLORTERM": "truecolor"})
    for line in tty.getvalue().splitlines():
        assert visible_len(line) <= 80, (visible_len(line), ANSI.sub("", line))


def test_the_final_frame_has_logo_tagline_demo_map_and_next_steps() -> None:
    text = _static()
    mono_logo = Style("mono")
    from pycodemath.cli.term import logo_rows

    for row in logo_rows(_logo.FULL, mono_logo):
        assert row.rstrip() in text
    assert f"pycodemath {__version__}" in text
    assert intro.STRINGS["en"]["tagline"] in text
    assert "REFUTED" in text and "VERIFIED" in text
    assert "sum_to(n=1) returned 0, not 1" in text
    for _topic, label, _pl, example in discover.MAP:
        assert f'{label}' in text and f'pycodemath "{example}"' in text
    assert text.rstrip().endswith(
        "next: pycodemath commands · pycodemath examples <topic> · pycodemath tour")
    assert 6 <= len(discover.MAP) <= 8


def test_not_a_tty_means_no_escape_bytes_and_no_animation() -> None:
    buf = io.StringIO()  # isatty() is False
    intro.show(buf, env={"COLORTERM": "truecolor"})
    assert "\x1b" not in buf.getvalue()
    assert "\r" not in buf.getvalue()
    assert buf.getvalue() == _static()


def test_a_real_pipe_gets_the_static_frame_without_ansi() -> None:
    env = dict(os.environ, COLORTERM="truecolor", PYTHONIOENCODING="utf-8")
    env.pop("NO_COLOR", None)
    p = subprocess.run([sys.executable, "-m", "pycodemath", "intro"], capture_output=True,
                       text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "\x1b" not in p.stdout and "\r" not in p.stdout
    assert "WHAT ELSE IT DOES" in p.stdout and f"pycodemath {__version__}" in p.stdout


def test_no_color_on_a_tty_writes_no_colour() -> None:
    tty = FakeTTY()
    assert color_mode(tty, {"NO_COLOR": "1", "COLORTERM": "truecolor"}) == "mono"
    intro.show(tty, static=True, env={"NO_COLOR": "1", "COLORTERM": "truecolor"})
    assert "\x1b" not in tty.getvalue()
    assert tty.getvalue() == _static()


def test_colour_modes_are_detected() -> None:
    tty = FakeTTY()
    assert color_mode(tty, {"COLORTERM": "truecolor"}) == "truecolor"
    assert color_mode(tty, {"WT_SESSION": "x"}) == "truecolor"
    assert color_mode(tty, {"TERM": "xterm-256color"}) == "256"
    assert color_mode(tty, {"TERM": "dumb"}) == "mono"
    assert color_mode(io.StringIO(), {"COLORTERM": "truecolor"}) == "mono"


def _sgr_colours(text: str) -> set[tuple[str, tuple[int, ...]]]:
    found = set()
    for m in re.finditer(r"\x1b\[([0-9;]*)m", text):
        codes = [int(c) for c in m[1].split(";") if c]
        i = 0
        while i < len(codes):
            if codes[i] in (38, 48) and codes[i + 1] == 2:
                found.add(("rgb", tuple(codes[i + 2:i + 5])))
                i += 5
            elif codes[i] in (38, 48) and codes[i + 1] == 5:
                found.add(("256", (codes[i + 2],)))
                i += 3
            else:
                i += 1
    return found


def test_truecolor_uses_only_the_closed_palette() -> None:
    tty = FakeTTY()
    intro.show(tty, static=True, env={"COLORTERM": "truecolor"})
    used = _sgr_colours(tty.getvalue())
    assert used and used <= {("rgb", rgb) for rgb in PALETTE.values()}
    assert _words(_screen(tty.getvalue())) == _words(_screen(_static()))


def test_256_colour_fallback_maps_to_the_nearest_palette_entries() -> None:
    tty = FakeTTY()
    intro.show(tty, static=True, env={"TERM": "xterm-256color"})
    used = _sgr_colours(tty.getvalue())
    allowed = {("256", (nearest_256(rgb),)) for rgb in PALETTE.values()}
    assert used and used <= allowed
    # the mapping really is nearest: krew is red, kosc is near-white, ink near-black
    assert nearest_256(PALETTE["krew"]) == 161
    assert nearest_256(PALETTE["kosc"]) in (255, 231)
    assert nearest_256(PALETTE["ink"]) in (232, 233, 16)


def test_the_text_on_screen_is_the_same_in_colour_and_mono() -> None:
    tty = FakeTTY()
    intro.show(tty, static=True, env={"COLORTERM": "truecolor"})
    colour = [visible_len(x) for x in tty.getvalue().splitlines()]
    assert max(colour) <= 80
    assert _words(_screen(tty.getvalue())) == _words(_screen(_static()))


# --- intro: animation ---------------------------------------------------------------------
def test_animation_ends_on_the_static_frame_and_finishes_under_six_seconds() -> None:
    intro._check(intro.DEMO_GOOD, intro.DEMO_DOMAIN)  # warm the import, not the timing
    tty = FakeTTY()
    started = time.monotonic()
    intro.show(tty, env={"TERM": "xterm-256color"})
    elapsed = time.monotonic() - started
    assert elapsed < 6.0, elapsed
    raw = tty.getvalue()
    assert "\r" in raw  # it did animate (spinner rewrites)
    assert "\x1b[?25h" in raw  # and gave the cursor back
    assert _words(_screen(raw)) == _words(_screen(_static()))


@pytest.mark.parametrize("env, static", [({"PYCODEMATH_NO_ANIM": "1"}, False), ({}, True)])
def test_reduced_motion_prints_the_final_state_only(env: dict, static: bool) -> None:
    tty = FakeTTY()
    intro.show(tty, static=static, env={"TERM": "xterm-256color", **env})
    assert "\r" not in tty.getvalue() and "?25l" not in tty.getvalue()
    assert _words(_screen(tty.getvalue())) == _words(_screen(_static()))


# --- intro: every demo number comes from the engine ---------------------------------------
def test_the_demo_numbers_come_from_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from pycodemath.code import frontend

    calls = []

    def fake(source, formula, *, name=None, domain=None, budget=None):
        calls.append(source)
        if "range(n + 1)" in source:
            return SimpleNamespace(verdict="verified", verified=True, symbolic_form="FAKEFORM",
                                   reason="", counterexample=None)
        ce = SimpleNamespace(call="sum_to(n=777)", function_value=4242, formula_value=31337)
        return SimpleNamespace(verdict="refuted", verified=False, symbolic_form="n**9",
                               reason="", counterexample=ce)

    monkeypatch.setattr(frontend, "check_code", fake)
    text = _static()
    assert "sum_to(n) = n**9" in text
    assert "sum_to(n=777) returned 4242, not 31337" in text
    assert "FAKEFORM equals n(n+1)/2" in text
    assert "sum_to(n=1)" not in text and "n**2/2" not in text
    assert len(calls) == 2  # steps 1 and 2 share one call; step 4 is the second


def test_the_tour_numbers_come_from_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    real = repl.handle
    monkeypatch.setattr(repl, "handle", lambda line: "ENGINE SAID " + line.split()[0]
                        if not line.startswith("intro") else real(line))
    buf = io.StringIO()
    discover.tour(buf)
    text = buf.getvalue()
    for demo in discover.TOUR:
        for _caption, command, _keep in demo.steps:
            assert f"ENGINE SAID {command.split()[0]}" in text
    assert "33.34" not in text


# --- intro --once ---------------------------------------------------------------------------
def test_once_shows_then_stays_silent_per_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PYCODEMATH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PYCODEMATH_SESSION", "session-a")
    code, out, err = _run(["intro", "--once", "--static"], monkeypatch, capsys)
    assert code == 0 and "WHAT ELSE IT DOES" in out
    code, out, err = _run(["intro", "--once"], monkeypatch, capsys)
    assert (code, out, err) == (0, "", "")
    monkeypatch.setenv("PYCODEMATH_SESSION", "session-b")
    code, out, _ = _run(["intro", "--once"], monkeypatch, capsys)
    assert code == 0 and "WHAT ELSE IT DOES" in out
    code, out, _ = _run(["intro"], monkeypatch, capsys)  # without --once: always
    assert "WHAT ELSE IT DOES" in out


def test_once_markers_keyed_by_pid_expire(tmp_path: Path) -> None:
    env = {"PYCODEMATH_CACHE_DIR": str(tmp_path)}
    key, pid_based = intro.session_key(env)
    assert pid_based and key.startswith(("sid:", "ppid:"))
    assert not intro.seen_this_session(env)
    intro.mark_seen(env)
    assert intro.seen_this_session(env)
    marker = next(tmp_path.glob("intro-*"))
    old = time.time() - 25 * 3600
    os.utime(marker, (old, old))
    assert not intro.seen_this_session(env)


def test_session_key_order() -> None:
    assert intro.session_key({"PYCODEMATH_SESSION": "x", "CLAUDE_CODE_SESSION_ID": "y"}) == ("explicit:x", False)
    assert intro.session_key({"CLAUDE_CODE_SESSION_ID": "y", "WT_SESSION": "w"}) == ("claude-code:y", False)
    assert intro.session_key({"TMUX": "/t", "TMUX_PANE": "%1", "WT_SESSION": "w"}) == ("tmux:/t:%1", False)
    assert intro.session_key({"WT_SESSION": "w"}) == ("WT_SESSION:w", False)
    assert intro.session_key({"TERM_SESSION_ID": "t"}) == ("TERM_SESSION_ID:t", False)


def test_an_unwritable_cache_is_not_an_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    env = {"PYCODEMATH_CACHE_DIR": str(blocker / "sub"), "PYCODEMATH_SESSION": "z"}
    intro.mark_seen(env)  # no exception
    assert not intro.seen_this_session(env)


@pytest.mark.parametrize("argv", [["intro", "--bogus"], ["intro", "--lang", "de"]])
def test_intro_refuses_unknown_flags(argv: list[str], monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(argv, monkeypatch, capsys)
    assert code == 1 and out == "" and "error:" in err


def test_intro_as_one_quoted_argument(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["intro --no-logo --lang pl"], monkeypatch, capsys)
    assert code == 0 and "CO JESZCZE UMIE" in out and "▀" not in out


# --- one-shot commands never print a banner (regression) -------------------------------
@pytest.mark.parametrize("command, expected", [
    ("2+2", "4\n"),
    ("diff x^2 dx", "2*x\n"),
    ("solve x^2-4 for x", "-2, 2\n"),
])
def test_one_shot_prints_the_answer_and_nothing_else(command: str, expected: str,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run([command], monkeypatch, capsys)
    assert (code, out, err) == (0, expected, "")


def test_one_shot_in_a_real_process_has_a_clean_stdout() -> None:
    p = subprocess.run([sys.executable, "-m", "pycodemath", "diff x^2 dx"], capture_output=True,
                       text=True, encoding="utf-8", cwd=ROOT, timeout=120)
    assert (p.returncode, p.stdout) == (0, "2*x\n")


def test_the_interactive_repl_shows_the_three_line_banner() -> None:
    p = subprocess.run([sys.executable, "-m", "pycodemath"], input="2+2\nquit\n", capture_output=True,
                       text=True, encoding="utf-8", cwd=ROOT, timeout=120)
    lines = p.stdout.splitlines()
    assert f"Pycodemath {__version__}" in lines[0]
    assert "intro" in lines[1] and "commands" in lines[1] and "help <topic>" in lines[1]
    assert lines[3].startswith("pcm> 4")
    assert "\x1b" not in p.stdout


def test_repl_banner_is_three_lines_within_80_columns() -> None:
    for out in (io.StringIO(), FakeTTY()):
        banner = intro.repl_banner(out)
        assert len(banner.splitlines()) == 3
        assert all(visible_len(x) <= 80 for x in banner.splitlines())
        assert __version__ in banner


def test_repl_routes_the_new_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    p = subprocess.run([sys.executable, "-m", "pycodemath"],
                       input="help money\nexamples units\nintro --no-logo\nquit\n",
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=120)
    assert "help money —" in p.stdout and "examples units —" in p.stdout
    assert "WHAT ELSE IT DOES" in p.stdout and p.returncode == 0


# --- commands / help / examples ---------------------------------------------------------
def test_every_help_entry_belongs_to_a_topic() -> None:
    entries = discover.help_entries()
    assert len(entries) > 60
    assert [key for key, _ in entries if discover.topic_of(key) is None] == []


def test_the_help_topics_are_the_promised_ones() -> None:
    for name in ("solve", "calculus", "verify", "code", "numbers", "money", "units",
                 "geometry", "ode", "stats"):
        assert name in discover.TOPICS


@pytest.mark.parametrize("name", list(discover.TOPICS))
def test_help_topic_reuses_the_repl_help_verbatim(name: str, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["help", name], monkeypatch, capsys)
    assert code == 0
    body = out.split("\n\n", 1)[1]
    blocks = [b for b in body.split("\nexamples:")[0].strip("\n").split("\n") if b]
    assert blocks
    for line in blocks:
        assert line in (repl.__doc__ or ""), line
    assert len(out.splitlines()[0]) <= 80


def test_bare_help_is_unchanged(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["help"], monkeypatch, capsys)
    assert code == 0 and out == (repl.__doc__ or "") + "\n"


@pytest.mark.parametrize("argv", [["help", "nope"], ["examples"], ["examples", "nope"], ["commands", "x"]])
def test_bad_topic_is_an_error_listing_the_topics(argv: list[str], monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(argv, monkeypatch, capsys)
    assert code == 1 and out == "" and err.startswith("error:")


def test_commands_lists_every_entry_once_per_topic_within_80_columns(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["commands"], monkeypatch, capsys)
    assert code == 0
    for line in out.splitlines():
        assert len(line) <= 80, line
    for name in discover.TOPICS:
        assert f"\n{name.upper()} — " in out
    listed = [x for x in out.splitlines() if x.startswith("  ")]
    assert len(listed) == len(discover.help_entries())


@pytest.mark.parametrize("name", [t for t in discover.TOPICS if discover.TOPICS[t].examples])
def test_examples_list_3_to_5_within_80_columns(name: str, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["examples", name], monkeypatch, capsys)
    assert code == 0
    assert 3 <= len(discover.TOPICS[name].examples) <= 5
    assert out.count('pycodemath "') == len(discover.TOPICS[name].examples)
    for line in out.splitlines():
        assert len(line) <= 80, line


_EXAMPLES = [(t, cmd) for t, topic in discover.TOPICS.items() for _c, cmd in topic.examples]


@pytest.mark.parametrize("topic, command", _EXAMPLES, ids=[c for _t, c in _EXAMPLES])
def test_every_example_runs(topic: str, command: str, monkeypatch: pytest.MonkeyPatch,
                            capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run([command], monkeypatch, capsys)
    assert code == 0, err
    assert out.strip() and not out.startswith("Usage") and not err


def test_the_map_examples_are_listed_examples() -> None:
    listed = {cmd for _t, cmd in _EXAMPLES}
    assert {example for *_x, example in discover.MAP} <= listed


# --- tour ------------------------------------------------------------------------------
def test_tour_without_a_tty_prints_five_demos_in_a_row(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["tour"], monkeypatch, capsys)
    assert code == 0
    for n in range(1, 6):
        assert f"[{n}/5]" in out
    assert "\x1b" not in out and "Enter = next" not in out
    for line in out.splitlines():
        assert len(line) <= 80, line


def test_tour_on_a_tty_waits_for_enter_and_q_quits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    answers = iter(["", "q"])
    prompts = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    tty = FakeTTY()
    monkeypatch.setenv("PYCODEMATH_NO_ANIM", "")
    assert discover.tour(tty, read=read, static=False) == 0
    screen = "\n".join(_screen(tty.getvalue()))
    assert "[1/5]" in screen and "[2/5]" in screen and "[3/5]" not in screen
    assert len(prompts) == 2 and "q = quit" in prompts[0]


# --- MCP ---------------------------------------------------------------------------------
def test_mcp_intro_returns_plain_text_logo_and_map() -> None:
    names = [t.__name__ for t in mcp_server.TOOLS]
    assert names[:8] == ["math_eval", "math_verify", "math_check_code", "math_stability",
                         "math_ranges", "math_recheck", "math_iterations", "math_lint"]
    assert "intro" in names
    r = mcp_server.intro()
    assert "\x1b" not in str(r)
    assert set(r) == {"text", "logo", "version", "tagline", "areas", "next"}
    assert r["version"] == __version__
    assert r["logo"].splitlines()[0].strip() and "█" in r["logo"]
    assert [a["example"] for a in r["areas"]] == [m[3] for m in discover.MAP]
    assert all(a["commands"] for a in r["areas"])
    for line in r["text"].splitlines():
        assert len(line) <= 80


# --- the logo is the SVG's -------------------------------------------------------------------
def test_the_committed_logo_is_what_the_generator_makes_from_the_svg() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import gen_logo
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    for name, (height, threshold) in gen_logo.SIZES.items():
        assert tuple(gen_logo.rasterize(height, threshold)) == getattr(_logo, name.upper())
