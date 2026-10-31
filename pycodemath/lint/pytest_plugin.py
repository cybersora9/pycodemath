"""``pytest --pcm``: every ``:math:`` / ``@spec`` function in the collected tree becomes a test.

Enable it with ``-p pycodemath.lint.pytest_plugin`` (or ``pytest_plugins =
["pycodemath.lint.pytest_plugin"]`` in a ``conftest.py``; once the ``pytest11``
entry point is installed, ``--pcm`` alone). With ``--pcm``, pytest asks this plugin
about EVERY ``.py`` file under the paths it collects — source files too, not only
``test_*.py`` — and each top-level function with a spec becomes an item
``path::pcm[name]``:

* the spec is checked by C1 (``engine.check_spec``) — the same check as the lint
  rules PCM301/PCM504;
* REFUTED fails the test, and the failure message is the witness: the call, what
  it returned, what the formula gives (and the proved one-off fix for PCM504);
* UNDECIDED is a SKIP with C1's reason — never a pass, never a failure;
* VERIFIED passes.

``--pcm-no-execute`` keeps C1 from running the functions (a refutation that needs
execution then becomes a skip). ``--pcm-budget S`` sets the seconds per function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

__all__ = ["PcmSpecFailure"]


class PcmSpecFailure(AssertionError):
    """A spec refuted by an executed counterexample."""


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("pycodemath", "pycodemath: check :math: / @spec specifications")
    group.addoption("--pcm", action="store_true", default=False,
                    help="check every function with a :math: docstring line or @spec as a test")
    group.addoption("--pcm-no-execute", action="store_true", default=False,
                    help="do not execute functions (refutations that need execution become skips)")
    group.addoption("--pcm-budget", type=float, default=10.0, help="seconds per spec (default 10)")


def pytest_collect_file(file_path: Path, parent: Any) -> "PcmFile | None":
    config = parent.config
    if not config.getoption("--pcm") or file_path.suffix != ".py":
        return None
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if ":math:" not in text and "spec(" not in text:
        return None
    return PcmFile.from_parent(parent, path=file_path)


class PcmFile(pytest.File):
    def collect(self) -> Iterator["PcmItem"]:
        from .engine import spec_functions

        text = self.path.read_text(encoding="utf-8")
        for target in spec_functions(text, str(self.path)):
            yield PcmItem.from_parent(self, name=f"pcm[{target.name}]", target=target, source=text)


class PcmItem(pytest.Item):
    def __init__(self, *, target: Any, source: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.target = target
        self.source = source
        self.add_marker("pcm")

    def runtest(self) -> None:
        from .config import Config
        from .engine import check_spec

        opt = self.config.getoption
        cfg = Config(budget=float(opt("--pcm-budget")), execute=not opt("--pcm-no-execute"), cache=False)
        findings = check_spec(self.source, str(self.path), self.target.name, cfg)
        bad = [f for f in findings if f.code in ("PCM301", "PCM504")]
        if bad:
            f = bad[0]
            lines = [f"{self.target.name}: spec `{self.target.formula}` refuted ({f.code} {f.name})",
                     f"  witness: {f.witness.describe() if f.witness else '-'}"]
            if f.fix is not None:
                lines.append(f"  fix (line {f.fix.line}): {f.fix.replacement} — {f.fix.description}")
            raise PcmSpecFailure("\n".join(lines))
        notes = [f for f in findings if f.code in ("PCM302", "PCM901")]
        if notes:
            pytest.skip(notes[0].message)

    def repr_failure(self, excinfo: Any, style: Any = None) -> str:
        if isinstance(excinfo.value, PcmSpecFailure):
            return str(excinfo.value)
        return str(super().repr_failure(excinfo))

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, self.target.line - 1, f"pcm spec {self.target.name}: {self.target.formula}"


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "pcm: a :math: / @spec specification checked by pycodemath")
