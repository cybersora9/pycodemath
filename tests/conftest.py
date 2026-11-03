"""Shared test setup.

The intro plays by itself on the first TTY run of each installed version and
records that in the user cache directory (``pycodemath.cli.intro``). Tests
must neither read nor write the real one, so every test gets its own empty
cache directory. (Not ``PYCODEMATH_NO_INTRO``: that would also hide a
regression where the intro leaks into a non-TTY run.)
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _private_intro_cache(tmp_path_factory: pytest.TempPathFactory,
                         monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path_factory.mktemp("pycodemath-cache")
    monkeypatch.setenv("PYCODEMATH_CACHE_DIR", str(path))
    return path
