"""``pycodemath lint`` (module L1): CLI, configuration, selection, cache, formats, docs, pytest plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pycodemath.lint import spec
from pycodemath.lint.__main__ import main
from pycodemath.lint.config import Config, ConfigError, load_config
from pycodemath.lint.engine import lint_paths, lint_source
from pycodemath.lint.rules import RULES, rules_markdown

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
PRE = "import math\nimport numpy as np\nfrom decimal import Decimal, ROUND_HALF_UP\n\n"

BUGGY = textwrap.dedent('''\
    def gross(price: float) -> float:
        return round(price * 1.23, 2)


    EPS = 10^-6


    def sum_to(n: int) -> int:
        """:math: n*(n+1)/2
        :domain n: integer, n>=0
        """
        total = 0
        for i in range(1, n):
            total += i
        return total
''')


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- rules / docs


def test_rule_codes_are_unique_and_well_formed() -> None:
    assert len(RULES) == 17
    for code, r in RULES.items():
        assert code == r.code and code.startswith("PCM") and len(code) == 6
        assert r.level in ("error", "warning", "note")
        assert r.summary and r.why and r.witness
        assert r.name == r.name.lower() and " " not in r.name


def test_rules_markdown_is_in_sync() -> None:
    """RULES.md is generated from rules.py (``--generate-docs``); they must not drift."""
    on_disk = (ROOT / "pycodemath" / "lint" / "RULES.md").read_text(encoding="utf-8")
    assert on_disk == rules_markdown()


@pytest.mark.parametrize("code", [c for c, r in RULES.items() if r.bad and r.family != "meta" and c != "PCM302"])
def test_documented_examples_behave_as_documented(code: str) -> None:
    """Each rule's "Reported" example triggers it; its "Instead" example does not."""
    r = RULES[code]
    cfg = Config(select=(code,), cache=False)
    assert code in {f.code for f in lint_source(PRE + r.bad, "doc.py", cfg)}
    if r.good and r.good.lstrip().startswith(("def", "rate", "total", "VAT")) and code != "PCM504":
        assert code not in {f.code for f in lint_source(PRE + r.good, "doc.py", cfg)}


# --------------------------------------------------------------------------- spec decorator


def test_spec_decorator_has_no_runtime_effect() -> None:
    def tri(n: int) -> int:
        return n * (n + 1) // 2

    same = spec("n*(n+1)/2", domain={"n": "integer, n>=0"})(tri)
    assert same is tri and tri(4) == 10
    assert tri.__pcm_spec__ == {"formula": "n*(n+1)/2", "domain": {"n": "integer, n>=0"}, "ranges": {}}
    with pytest.raises(TypeError):
        spec("")


def test_importing_spec_loads_nothing_of_the_linter() -> None:
    """``@spec`` in production code must not pull in the engine (or C1-C4)."""
    code = ("import sys; from pycodemath.lint import spec; "
            "print(sorted(m for m in sys.modules if m.startswith(('pycodemath.lint.', 'pycodemath.code.')) or m == 'pycodemath.code'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_sphinx_math_roles_and_foreign_formulas_are_not_specs() -> None:
    from pycodemath.lint.engine import spec_functions

    src = textwrap.dedent('''\
        def a(p, v):
            """Power.

            :math:`\\frac{d^2 P}{dV dV_d}`
            """
            return p * v


        def b(n: int) -> int:
            """:math: k*(k+1)/2"""
            return n


        def c(n: int) -> int:
            """:math: Sum(i, (i, 1, n))"""
            return n * (n + 1) // 2
    ''')
    assert [t.name for t in spec_functions(src)] == ["c"]


def test_spec_decorator_and_docstring_are_both_read() -> None:
    src = textwrap.dedent('''\
        from pycodemath.lint import spec


        @spec("n**2", domain={"n": "integer, n>=0"})
        def sq(n: int) -> int:
            return n * (n + 1)


        def sq2(n: int) -> int:
            """:math: n**2
            :domain n: integer, n>=0
            """
            return n * n
    ''')
    got = [(f.code, f.function) for f in lint_source(src, "s.py")]
    assert got == [("PCM301", "sq")]


# --------------------------------------------------------------------------- configuration


def test_select_ignore_and_prefixes() -> None:
    assert Config().enabled_codes() == frozenset(c for c, r in RULES.items() if r.default)
    assert Config(select=("PCM",)).enabled_codes() == frozenset(RULES)
    assert Config(select=("ALL",)).enabled_codes() == frozenset(RULES)
    c = Config.from_mapping({"select": ["PCM4", "PCM502"], "ignore": ["PCM403"], "extend-select": ["PCM901"]})
    assert sorted(c.enabled_codes()) == ["PCM401", "PCM402", "PCM502", "PCM901"]


def test_per_file_ignores() -> None:
    c = Config.from_mapping({"per-file-ignores": {"tests/*": ["PCM5"], "legacy.py": ["PCM"]}})
    assert "PCM502" not in c.enabled_codes("tests/test_x.py")
    assert "PCM401" in c.enabled_codes("tests/test_x.py")
    assert c.enabled_codes("legacy.py") == frozenset()
    assert "PCM502" in c.enabled_codes("src/a.py")


@pytest.mark.parametrize("bad", [
    {"select": ["E501"]}, {"select": ["PCM999"]}, {"unknown-key": 1}, {"precision-bits": 0},
    {"default-range": [0, 1]}, {"default-range": [5, 1]}, {"budget": -1}, {"execute": "yes"},
    {"money-names": "("}, {"ranges": {"x": [1]}}, {"per-file-ignores": ["x"]},
])
def test_bad_configuration_is_refused(bad: dict) -> None:
    with pytest.raises(ConfigError):
        Config.from_mapping(bad)


def test_pyproject_is_found_and_read(tmp_path: Path) -> None:
    _write(tmp_path, "pyproject.toml", '[tool.pycodemath.lint]\nignore = ["PCM503"]\nprecision-bits = 30\n')
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    c = load_config(start=sub)
    assert c.ignore == ("PCM503",) and c.precision_bits == 30.0 and Path(c.root) == tmp_path.resolve()
    _write(tmp_path, "pyproject.toml", "[tool.pycodemath.lint\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "pyproject.toml")


def test_config_ranges_by_name_feed_the_precision_rule() -> None:
    src = "import math\n\ndef gap(x: float) -> float:\n    return math.sqrt(x + 1) - math.sqrt(x)\n"
    assert lint_source(src, "g.py") == []  # default |x| <= 1e6: 19 bits < 26
    cfg = Config(ranges=(("x", (1.0, 1e15)),))
    assert [f.code for f in lint_source(src, "g.py", cfg)] == ["PCM101"]


# --------------------------------------------------------------------------- suppression


def test_file_level_ignore() -> None:
    src = "# pcm: file-ignore[PCM502]\nEPS = 10^-6\nK = 2^10 - 1\n"
    assert lint_source(src, "f.py") == []
    src2 = "# pcm: file-ignore[PCM4]\nEPS = 10^-6\n"
    assert [f.code for f in lint_source(src2, "f.py")] == ["PCM502"]


def test_ignore_on_def_line_covers_function_level_findings() -> None:
    src = BUGGY.replace("def sum_to(n: int) -> int:", "def sum_to(n: int) -> int:  # pcm: ignore[PCM504]")
    codes = [f.code for f in lint_source(src, "b.py")]
    assert "PCM504" not in codes and "PCM502" in codes


# --------------------------------------------------------------------------- CLI


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = _write(tmp_path, "clean.py", "def f(x: float) -> float:\n    return x * 2\n")
    buggy = _write(tmp_path, "buggy.py", BUGGY)
    assert main([str(clean), "--no-cache"]) == 0
    assert "All checks passed" in capsys.readouterr().out
    assert main([str(buggy), "--no-cache"]) == 1
    out = capsys.readouterr().out
    assert "buggy.py:2:12: PCM401" in out and "witness:" in out and "PCM504" in out
    assert main([str(buggy), "--no-cache", "--exit-zero"]) == 0
    capsys.readouterr()
    assert main([str(tmp_path / "missing.py")]) == 2
    assert main(["--select", "E501", str(clean)]) == 2
    assert main(["--bogus-flag"]) == 2
    assert main([str(buggy), "--no-cache", "--select", "PCM502"]) == 1
    out = capsys.readouterr().out
    assert "PCM502" in out and "PCM401" not in out


def test_cli_formats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    buggy = _write(tmp_path, "buggy.py", BUGGY)
    main([str(buggy), "--no-cache", "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert {d["code"] for d in data} == {"PCM401", "PCM503", "PCM502", "PCM504"}
    d = next(d for d in data if d["code"] == "PCM504")
    assert d["witness"]["program"] == "sum_to(n=1)" and d["fix"]["replacement"] == "n + 1"
    assert d["fix"]["applicability"] == "safe" and d["url"].endswith("#pcm504")
    main([str(buggy), "--no-cache", "--format", "sarif"])
    log = json.loads(capsys.readouterr().out)
    assert log["version"] == "2.1.0" and len(log["runs"][0]["results"]) == 4
    main([str(buggy), "--no-cache", "--concise"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 5 and lines[-1].startswith("Found 4 findings")
    main([str(buggy), "--no-cache", "--statistics"])
    out = capsys.readouterr().out
    assert "1\tPCM401\tfloat-money" in out


def test_cli_output_file_and_explain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    buggy = _write(tmp_path, "buggy.py", BUGGY)
    target = tmp_path / "out.sarif"
    assert main([str(buggy), "--no-cache", "--format", "sarif", "-o", str(target)]) == 1
    assert json.loads(target.read_text())["runs"][0]["results"]
    assert main(["--explain", "pcm502"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### PCM502") and "xor-as-power" in out
    assert main(["--explain", "PCM000"]) == 2
    assert main(["--list-rules"]) == 0
    assert capsys.readouterr().out.count("PCM") >= 17


def test_color_only_on_a_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pycodemath.lint.report import format_text, use_color

    class Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert use_color(Tty())  # type: ignore[arg-type]
    monkeypatch.setenv("NO_COLOR", "1")
    assert not use_color(Tty())  # type: ignore[arg-type]
    fs = lint_source("EPS = 10^-6\n", "e.py")
    assert "\033[" in format_text(fs, color=True) and "\033[" not in format_text(fs, color=False)


def test_python_m_entry_point(tmp_path: Path) -> None:
    buggy = _write(tmp_path, "buggy.py", "EPS = 10^-6\n")
    out = subprocess.run([sys.executable, "-m", "pycodemath.lint", str(buggy), "--no-cache"],
                         capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "NO_COLOR": "1"})
    assert out.returncode == 1, out.stderr
    assert "buggy.py:1:7: PCM502" in out.stdout


def test_unparseable_file_is_never_a_silent_pass(tmp_path: Path) -> None:
    """A file that cannot be parsed was NOT checked: exit 2, not "All checks passed" (exit 0)."""
    broken = _write(tmp_path, "broken.py", "def f(:\n")
    env = {**os.environ, "NO_COLOR": "1"}
    out = subprocess.run([sys.executable, "-m", "pycodemath.lint", str(broken), "--no-cache"],
                         capture_output=True, text=True, cwd=tmp_path, env=env)
    assert out.returncode == 2, (out.returncode, out.stdout, out.stderr)
    assert "All checks passed" not in out.stdout and "not analysed" in out.stdout
    assert "syntax error" in out.stderr
    zero = subprocess.run([sys.executable, "-m", "pycodemath.lint", str(broken), "--no-cache", "--exit-zero"],
                          capture_output=True, text=True, cwd=tmp_path, env=env)
    assert zero.returncode == 0


# --------------------------------------------------------------------------- directories, exclusion, cache


def test_directory_walk_and_excludes(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/a.py", "EPS = 10^-6\n")
    _write(tmp_path, ".venv/lib/b.py", "EPS = 10^-6\n")
    _write(tmp_path, "build/c.py", "EPS = 10^-6\n")
    _write(tmp_path, "vendor/d.py", "EPS = 10^-6\n")
    _write(tmp_path, "pkg/broken.py", "def (:\n")
    cfg = Config(root=str(tmp_path), exclude=("vendor",), cache=False)
    res = lint_paths([tmp_path], cfg)
    assert sorted(r.path for r in res.files) == ["pkg/a.py", "pkg/broken.py"]
    assert [f.path for f in res.findings] == ["pkg/a.py"]
    assert res.errors and res.errors[0][0] == "pkg/broken.py" and "syntax error" in res.errors[0][1]


def test_cache_by_file_hash(tmp_path: Path) -> None:
    p = _write(tmp_path, "m.py", BUGGY)
    cfg = Config(root=str(tmp_path))
    first = lint_paths([p], cfg)
    second = lint_paths([p], cfg)
    assert not first.files[0].cached and second.files[0].cached
    assert [f.to_dict() for f in first.findings] == [f.to_dict() for f in second.findings]
    # selection is applied AFTER the cache: no fresh analysis needed to narrow it
    narrowed = lint_paths([p], Config(root=str(tmp_path), select=("PCM502",)))
    assert narrowed.files[0].cached and [f.code for f in narrowed.findings] == ["PCM502"]
    p.write_text(BUGGY.replace("10^-6", "10**-6"), encoding="utf-8")
    third = lint_paths([p], cfg)
    assert not third.files[0].cached and "PCM502" not in {f.code for f in third.findings}
    assert (tmp_path / ".pcm_lint_cache" / ".gitignore").read_text().endswith("*\n")


# --------------------------------------------------------------------------- meta rules


def test_no_execute_turns_refutations_into_notes() -> None:
    src = "def tri(n: int) -> int:\n    \"\"\":math: n*(n+1)/2\"\"\"\n    return n * (n - 1) // 2\n"
    assert [f.code for f in lint_source(src, "t.py")] == ["PCM301"]
    cfg = Config(execute=False, select=("PCM3",))
    got = lint_source(src, "t.py", cfg)
    assert [f.code for f in got] == ["PCM302"] and got[0].witness is None


def test_unsupported_numeric_function_is_a_note_not_a_finding() -> None:
    src = "import math\n\ndef f(x: float) -> float:\n    return math.gamma(x) - 1.0\n"
    assert lint_source(src, "u.py") == []
    notes = lint_source(src, "u.py", Config(select=("PCM9",)))
    assert [f.code for f in notes] == ["PCM902"] and notes[0].witness is None


# --------------------------------------------------------------------------- pytest plugin


def test_pytest_plugin_turns_specs_into_tests(pytester: pytest.Pytester,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    # the child writes in the console code page on Windows (cp1250), pytester decodes UTF-8
    monkeypatch.setenv("PYTHONUTF8", "1")
    pytester.makepyfile(mathlib=textwrap.dedent('''\
        def good(n: int) -> int:
            """:math: n*(n+1)/2
            :domain n: integer, n>=0
            """
            return n * (n + 1) // 2


        def bad(n: int) -> int:
            """:math: n*(n+1)/2
            :domain n: integer, n>=0
            """
            total = 0
            for i in range(1, n):
                total += i
            return total


        def unknown(n: int) -> int:
            """:math: n"""
            while n > 100:
                n -= 1
            return n
    '''))
    result = pytester.runpytest_subprocess("-p", "pycodemath.lint.pytest_plugin", "--pcm", "-rs")
    result.assert_outcomes(passed=1, failed=1, skipped=1)
    result.stdout.fnmatch_lines(["*bad: spec `n*(n+1)/2` refuted (PCM504 range-off-by-one)*",
                                 "*witness: bad(n=1) returns 0; the formula gives 1*"])
    plain = pytester.runpytest_subprocess("-p", "pycodemath.lint.pytest_plugin")
    plain.assert_outcomes()  # without --pcm nothing is collected from mathlib.py


# --------------------------------------------------------------------------- pre-commit / GitHub Action


def test_pre_commit_hook_definition() -> None:
    text = (ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")
    yaml = pytest.importorskip("yaml")
    hooks = yaml.safe_load(text)
    assert hooks[0]["id"] == "pycodemath-lint"
    assert hooks[0]["entry"] == "python -m pycodemath.lint" and hooks[0]["language"] == "python"
    assert hooks[0]["types"] == ["python"]


def test_github_action_example_uploads_sarif() -> None:
    text = (ROOT / "examples" / "github-action" / "pcm-lint.yml").read_text(encoding="utf-8")
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(text)
    steps = wf["jobs"]["pcm-lint"]["steps"]
    assert any("--format sarif" in s.get("run", "") for s in steps)
    assert any(s.get("uses", "").startswith("github/codeql-action/upload-sarif") for s in steps)
    assert wf["permissions"]["security-events"] == "write"
