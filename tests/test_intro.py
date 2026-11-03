"""The presentation and the way-finding commands: intro, commands, help <topic>,
examples <topic>, tour, the REPL banner, ``intro --once`` and the MCP ``intro``.

What is pinned here, and why:

* a one-shot command NEVER prints a banner (stdout of ``pycodemath "<cmd>"`` is
  the answer and nothing else — scripts and agents parse it);
* not a TTY / ``NO_COLOR`` -> not one escape byte, no animation;
* the intro's final frame fits 80 columns and ~34 lines, and every number in
  its demo and its header comes from the engine / the command tables at run
  time (they are replaced and the screen must change);
* every listed example really runs (exit code 0, no usage message);
* the intro plays by itself once per installed version, on a TTY only, before
  the command's own unchanged output; ``--once`` follows the same marker.
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


def _static(logo: bool = True) -> str:
    buf = io.StringIO()
    intro.show(buf, static=True, logo=logo, env={})
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
@pytest.mark.parametrize("logo", [True, False])
def test_the_final_frame_fits_80_columns_and_34_lines(logo: bool) -> None:
    lines = _static(logo).splitlines()
    assert len(lines) <= 34, len(lines)
    for line in lines:
        assert len(line) <= 80, (len(line), line)
    tty = FakeTTY()
    intro.show(tty, static=True, logo=logo, env={"COLORTERM": "truecolor"})
    for line in tty.getvalue().splitlines():
        assert visible_len(line) <= 80, (visible_len(line), ANSI.sub("", line))


def test_the_header_is_an_80_column_rounded_box_of_12_rows() -> None:
    lines = _static().splitlines()
    box = lines[:12]
    assert box[0] == "╭" + "─" * 78 + "╮" and box[-1] == "╰" + "─" * 78 + "╯"
    assert all(len(x) == 80 and x[0] == "│" and x[-1] == "│" for x in box[1:-1])
    assert lines[12] == ""
    from pycodemath.cli.term import logo_rows

    logo = logo_rows(_logo.FULL, Style("mono"))
    assert len(logo) == 10 and len(_logo.FULL[0]) == 36
    for row, line in zip(logo, box[1:-1]):  # one space, the logo, two spaces, the text
        assert line[1:38] == " " + row and line[38:40] == "  "
    text = [x[40:-1].rstrip() for x in box[1:-1]]
    n, m = intro.counts()
    assert text == ["", "", f"pycodemath  v{__version__}", "Checks the math in your code.", "",
                    f"{n} commands · {m} API functions", "by cybersora", "", "", ""]
    assert "100+" not in _static()


def test_the_box_is_krew_and_the_name_is_py_kosc_code_krew_math_kosc() -> None:
    tty = FakeTTY()
    intro.show(tty, static=True, env={"COLORTERM": "truecolor"})
    raw = tty.getvalue()
    krew, kosc, gray = (";".join(map(str, PALETTE[c])) for c in ("krew", "kosc", "gray"))
    assert raw.startswith(f"\x1b[38;2;{krew}m╭")
    assert (f"\x1b[1;38;2;{kosc}mpy\x1b[0m\x1b[1;38;2;{krew}mcode\x1b[0m"
            f"\x1b[1;38;2;{kosc}mmath\x1b[0m  \x1b[38;2;{gray}mv{__version__}\x1b[0m") in raw
    assert f"\x1b[38;2;{gray}mby cybersora\x1b[0m" in raw
    assert "48;2;" + ";".join(map(str, PALETTE["ink"])) not in raw  # no ink panel any more


def test_the_header_counts_come_from_the_command_tables_and_the_api(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import pycodemath
    from pycodemath.cli import full

    real = intro.counts()
    import inspect

    assert real == (len(set(repl._COMMANDS) | set(full.COMMANDS)),
                    sum(inspect.isfunction(getattr(pycodemath, n)) for n in pycodemath.__all__))
    # the union, not the sum: a name in both tables counts once
    monkeypatch.setattr(repl, "_COMMANDS", {"a": None, "b": None, "shared": None})
    monkeypatch.setattr(full, "COMMANDS", {"shared": None, "c": None})
    monkeypatch.setattr(pycodemath, "__all__", ["parse", "check_code", "Expr", "__version__"])
    assert intro.counts() == (4, 2)  # parse and check_code are functions; Expr, __version__ not
    assert "4 commands · 2 API functions" in _static()


def test_the_final_frame_has_logo_tagline_demo_map_and_next_steps() -> None:
    text = _static()
    mono_logo = Style("mono")
    from pycodemath.cli.term import logo_rows

    for row in logo_rows(_logo.FULL, mono_logo):
        assert row.rstrip() in text
    assert f"pycodemath  v{__version__}" in text
    assert "Checks the math in your code." == intro.S["tagline"] and intro.S["tagline"] in text
    assert "REFUTED" in text and "VERIFIED" in text
    assert "sum_to(n=1) returned 0, not 1" in text
    for _topic, label, example in discover.MAP:
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
    assert "WHAT ELSE IT DOES" in p.stdout and f"pycodemath  v{__version__}" in p.stdout


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
    # the mapping really is nearest: krew is red, kosc is near-white, gray is grey
    assert nearest_256(PALETTE["krew"]) == 161
    assert nearest_256(PALETTE["kosc"]) in (255, 231)
    assert nearest_256(PALETTE["gray"]) in range(244, 247)


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


# --- once per installed version ------------------------------------------------------
def _tty_run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """``repl.run()`` with a TTY stdout (no animation, to keep it fast)."""
    tty = FakeTTY()
    monkeypatch.setattr(sys, "stdout", tty)
    monkeypatch.setattr(sys, "argv", ["pycodemath", *argv])
    monkeypatch.setenv("PYCODEMATH_NO_ANIM", "1")
    monkeypatch.setenv("NO_COLOR", "1")  # compare with the mono frame
    code = 0
    try:
        repl.run()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return code, tty.getvalue()


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PYCODEMATH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PYCODEMATH_NO_INTRO", raising=False)
    return tmp_path


def test_first_tty_run_shows_the_intro_then_the_unchanged_answer(cache: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _tty_run(["2+2"], monkeypatch)
    assert code == 0
    assert out.startswith(_static()) and out == _static() + "\n" + "4\n"
    assert (cache / "intro-shown").read_text().strip() == __version__


def test_second_run_does_not_show_it(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tty_run(["2+2"], monkeypatch)
    code, out = _tty_run(["diff x^2 dx"], monkeypatch)
    assert (code, out) == (0, "2*x\n")


def test_a_new_version_shows_it_again(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (cache / "intro-shown").write_text("0.4.9\n")
    code, out = _tty_run(["2+2"], monkeypatch)
    assert "WHAT ELSE IT DOES" in out and out.endswith("\n4\n")
    assert (cache / "intro-shown").read_text().strip() == __version__
    assert _tty_run(["2+2"], monkeypatch) == (0, "4\n")


def test_the_interactive_repl_on_a_first_run_gets_the_intro_then_its_banner(
        cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    code, out = _tty_run([], monkeypatch)
    assert out.startswith(_static()) and f"pycodemath  v{__version__}" in out.splitlines()[31]


def test_never_off_a_tty(cache: Path, monkeypatch: pytest.MonkeyPatch,
                         capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["2+2"], monkeypatch, capsys) == (0, "4\n", "")
    assert not (cache / "intro-shown").exists()  # so a later TTY run still gets it
    assert intro.first_run(io.StringIO()) is False


def test_the_env_var_disables_it(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCODEMATH_NO_INTRO", "1")
    assert _tty_run(["2+2"], monkeypatch) == (0, "4\n")
    assert not (cache / "intro-shown").exists()


def test_the_auto_intro_animates_on_a_real_tty(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(intro, "show", lambda out, **kw: calls.append(kw))
    assert intro.first_run(FakeTTY(), env={"PYCODEMATH_CACHE_DIR": str(cache)}) is True
    assert calls == [{"env": {"PYCODEMATH_CACHE_DIR": str(cache)}}]  # not static: it may animate


def test_the_intro_command_on_a_first_run_is_shown_once_not_twice(
        cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _tty_run(["intro", "--static"], monkeypatch)
    assert code == 0 and out.count("WHAT ELSE IT DOES") == 1
    assert intro.seen() and _tty_run(["2+2"], monkeypatch) == (0, "4\n")


def test_an_unwritable_cache_means_no_auto_intro_rather_than_one_every_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("PYCODEMATH_CACHE_DIR", str(blocker / "sub"))
    monkeypatch.delenv("PYCODEMATH_NO_INTRO", raising=False)
    assert intro.mark_seen() is False  # no exception
    assert _tty_run(["2+2"], monkeypatch) == (0, "4\n")


def test_once_shows_only_if_this_version_was_not_shown_yet(cache: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(["intro", "--once", "--static"], monkeypatch, capsys)
    assert code == 0 and "WHAT ELSE IT DOES" in out  # off a TTY too: an agent asked for it
    assert _run(["intro", "--once"], monkeypatch, capsys) == (0, "", "")
    (cache / "intro-shown").write_text("0.4.9\n")  # an upgrade
    code, out, _ = _run(["intro", "--once"], monkeypatch, capsys)
    assert code == 0 and "WHAT ELSE IT DOES" in out
    code, out, _ = _run(["intro"], monkeypatch, capsys)  # without --once: always
    assert "WHAT ELSE IT DOES" in out
    monkeypatch.setenv("PYCODEMATH_NO_INTRO", "1")
    (cache / "intro-shown").unlink()
    assert _run(["intro", "--once"], monkeypatch, capsys) == (0, "", "")


def test_the_mcp_server_does_not_go_through_the_first_run_hook() -> None:
    src = (ROOT / "pycodemath" / "cli" / "mcp_server.py").read_text(encoding="utf-8")
    assert "first_run" not in src and "repl.run" not in src


@pytest.mark.parametrize("argv", [["intro", "--bogus"], ["intro", "--lang", "pl"], ["tour", "--lang", "pl"]])
def test_intro_refuses_unknown_flags(argv: list[str], monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(argv, monkeypatch, capsys)
    assert code == 1 and out == "" and "error:" in err


def test_intro_as_one_quoted_argument(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["intro --no-logo --static"], monkeypatch, capsys)
    assert code == 0 and "WHAT ELSE IT DOES" in out and not re.search("[▀▄█]", out)


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
    assert lines[0].startswith("∫>_") and f"pycodemath  v{__version__}" in lines[0]
    assert "intro" in lines[1] and "commands" in lines[1] and "help <topic>" in lines[1]
    assert lines[3].startswith("pcm> 4")
    assert "\x1b" not in p.stdout


def test_repl_banner_is_three_lines_within_80_columns() -> None:
    for out in (io.StringIO(), FakeTTY()):
        banner = intro.repl_banner(out)
        assert ("\x1b" in banner) == isinstance(out, FakeTTY)
        assert len((banner + "\n").splitlines()) == 3 and banner.endswith("\n")  # 2 + a blank
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
    assert [a["example"] for a in r["areas"]] == [m[2] for m in discover.MAP]
    assert all(a["commands"] for a in r["areas"])
    for line in r["text"].splitlines():
        assert len(line) <= 80


# --- the logo is the SVG's -------------------------------------------------------------------
def test_the_committed_logo_is_what_the_generator_makes_from_the_svg() -> None:
    pytest.importorskip("PIL.Image", reason="the generator needs Pillow (dev only)")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import gen_logo
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert (gen_logo.COLS, gen_logo.STROKE) == (36, 12)  # the approved look
    assert tuple(gen_logo.rasterize()) == _logo.FULL


def test_the_logo_is_transparent_half_blocks() -> None:
    from pycodemath.cli.term import logo_rows

    grid = ("r.rk", ".rrr")  # only top, only bottom, same, different
    krew, kosc = (";".join(map(str, PALETTE[c])) for c in ("krew", "kosc"))
    assert logo_rows(grid, Style("mono")) == ["▀▄█▀"]
    assert logo_rows(grid, Style("truecolor")) == [
        f"\x1b[38;2;{krew}m▀\x1b[0m\x1b[38;2;{krew}m▄\x1b[0m\x1b[38;2;{krew}m█\x1b[0m"
        f"\x1b[38;2;{kosc};48;2;{krew}m▀\x1b[0m"]
    assert logo_rows(("..", ".."), Style("truecolor")) == ["  "]  # empty cells: no colour at all
    colour = logo_rows(_logo.FULL, Style("truecolor"))
    assert [ANSI.sub("", x) for x in colour] == logo_rows(_logo.FULL, Style("mono"))
    assert _logo.FULL[0].strip(".") and _logo.FULL[-1].strip(".")  # trimmed
