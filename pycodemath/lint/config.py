"""Configuration of ``pycodemath lint``: ``[tool.pycodemath.lint]`` in the user's ``pyproject.toml``.

Keys (all optional; kebab-case as in ruff)::

    [tool.pycodemath.lint]
    select = ["PCM"]                  # codes or prefixes; default: the rules marked "on"
    extend-select = ["PCM203"]        # added to select
    ignore = ["PCM503"]
    per-file-ignores = { "tests/*" = ["PCM501"], "legacy/*.py" = ["PCM"] }
    exclude = ["build", "vendor/*"]   # added to the defaults (.git, .venv, build, ...)
    precision-bits = 26               # PCM101 threshold for float64 (float32: precision-bits * 24 / 53)
    default-range = [1e-6, 1e6]       # |x| range used when a variable has no :range
    ranges = { x = [0, 1] }           # range by variable NAME, before the default
    budget = 5.0                      # seconds per function per analysis
    execute = true                    # PCM3xx: run the function in C1's isolated worker
    cache = true                      # cache results by file hash in .pcm_lint_cache/
    money-names = "price|amount|total|cost|vat|tax"

``--select/--ignore/--extend-select/--config/--no-cache/--no-execute`` on the command
line override the file. Selection by prefix works as in ruff: ``PCM5`` is every
PCM5xx rule, ``PCM`` or ``ALL`` is everything (the ``note`` rules included).

    >>> c = Config.from_mapping({"select": ["PCM5"], "ignore": ["PCM503"]})
    >>> sorted(c.enabled_codes())
    ['PCM501', 'PCM502', 'PCM504']
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 and older are not supported by the package
    tomllib = None  # type: ignore[assignment]

from ..core.errors import DomainError
from .rules import RULES, default_codes

__all__ = ["Config", "ConfigError", "load_config", "find_pyproject", "DEFAULT_EXCLUDE", "DEFAULT_MONEY_NAMES"]

DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".git", ".hg", ".svn", ".venv", "venv", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "__pycache__", "build", "dist", "node_modules", ".pcm_lint_cache", "site-packages",
    ".eggs", "*.egg-info",
)

DEFAULT_MONEY_NAMES = "price|amount|total|cost|vat|tax"


class ConfigError(DomainError):
    """The configuration (file or command line) is invalid — exit code 2."""


_KEYS = {"select", "extend-select", "ignore", "per-file-ignores", "exclude", "precision-bits",
         "default-range", "ranges", "budget", "execute", "cache", "cache-dir", "money-names"}


def _codes(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key} must be a list of rule codes, not {value!r}")
    out = []
    for v in value:
        v = v.strip().upper()
        if v != "ALL" and not re.fullmatch(r"PCM\d{0,3}", v):
            raise ConfigError(f"{key}: {v!r} is not a rule code or prefix (PCM101, PCM5, PCM, ALL)")
        if re.fullmatch(r"PCM\d{3}", v) and v not in RULES:
            raise ConfigError(f"{key}: unknown rule {v}")
        out.append(v)
    return tuple(out)


def _matches(code: str, selector: str) -> bool:
    return selector == "ALL" or code.startswith(selector)


def _range(value: Any, key: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value[:2]):
        raise ConfigError(f"{key} must be [low, high], not {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if not lo < hi:
        raise ConfigError(f"{key}: low must be below high, got {value!r}")
    return lo, hi


@dataclass(frozen=True)
class Config:
    """The effective configuration. ``root`` is the directory paths are relative to
    (the one holding ``pyproject.toml``, or the working directory)."""

    select: "tuple[str, ...] | None" = None
    extend_select: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    per_file_ignores: tuple[tuple[str, tuple[str, ...]], ...] = ()
    exclude: tuple[str, ...] = ()
    precision_bits: float = 26.0
    default_range: tuple[float, float] = (1e-6, 1e6)
    ranges: tuple[tuple[str, tuple[float, float]], ...] = ()
    budget: float = 5.0
    execute: bool = True
    cache: bool = True
    cache_dir: str = ".pcm_lint_cache"
    money_names: str = DEFAULT_MONEY_NAMES
    root: str = "."
    source: str = "defaults"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, root: str = ".", source: str = "mapping") -> "Config":
        unknown = set(data) - _KEYS
        if unknown:
            raise ConfigError(f"[tool.pycodemath.lint]: unknown key(s) {sorted(unknown)}; known: {sorted(_KEYS)}")
        kw: dict[str, Any] = {"root": root, "source": source}
        if "select" in data:
            kw["select"] = _codes(data["select"], "select")
        if "extend-select" in data:
            kw["extend_select"] = _codes(data["extend-select"], "extend-select")
        if "ignore" in data:
            kw["ignore"] = _codes(data["ignore"], "ignore")
        if "per-file-ignores" in data:
            pfi = data["per-file-ignores"]
            if not isinstance(pfi, Mapping):
                raise ConfigError("per-file-ignores must be a table of glob = [codes]")
            kw["per_file_ignores"] = tuple((str(g), _codes(c, f"per-file-ignores[{g!r}]")) for g, c in pfi.items())
        if "exclude" in data:
            ex = data["exclude"]
            if not isinstance(ex, (list, tuple)) or not all(isinstance(e, str) for e in ex):
                raise ConfigError("exclude must be a list of globs")
            kw["exclude"] = tuple(ex)
        if "precision-bits" in data:
            pb = data["precision-bits"]
            if not isinstance(pb, (int, float)) or isinstance(pb, bool) or not 1 <= pb <= 64:
                raise ConfigError(f"precision-bits must be a number from 1 to 64, not {pb!r}")
            kw["precision_bits"] = float(pb)
        if "default-range" in data:
            lo, hi = _range(data["default-range"], "default-range")
            if lo <= 0:
                raise ConfigError("default-range is a range of MAGNITUDES |x|: both ends must be positive")
            kw["default_range"] = (lo, hi)
        if "ranges" in data:
            rg = data["ranges"]
            if not isinstance(rg, Mapping):
                raise ConfigError("ranges must be a table of name = [low, high]")
            kw["ranges"] = tuple((str(k), _range(v, f"ranges.{k}")) for k, v in rg.items())
        if "budget" in data:
            b = data["budget"]
            if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0 < b <= 600:
                raise ConfigError(f"budget must be seconds in (0, 600], not {b!r}")
            kw["budget"] = float(b)
        for key, attr in (("execute", "execute"), ("cache", "cache")):
            if key in data:
                if not isinstance(data[key], bool):
                    raise ConfigError(f"{key} must be true or false")
                kw[attr] = data[key]
        if "cache-dir" in data:
            kw["cache_dir"] = str(data["cache-dir"])
        if "money-names" in data:
            mn = data["money-names"]
            try:
                re.compile(mn)
            except (re.error, TypeError) as exc:
                raise ConfigError(f"money-names must be a regular expression: {exc}") from None
            kw["money_names"] = mn
        return cls(**kw)

    def with_cli(self, *, select: "Sequence[str] | None" = None, extend_select: "Sequence[str] | None" = None,
                 ignore: "Sequence[str] | None" = None, execute: "bool | None" = None,
                 cache: "bool | None" = None) -> "Config":
        c = self
        if select is not None:
            c = replace(c, select=_codes(list(select), "--select"))
        if extend_select:
            c = replace(c, extend_select=c.extend_select + _codes(list(extend_select), "--extend-select"))
        if ignore:
            c = replace(c, ignore=c.ignore + _codes(list(ignore), "--ignore"))
        if execute is not None:
            c = replace(c, execute=execute)
        if cache is not None:
            c = replace(c, cache=cache)
        return c

    def enabled_codes(self, path: "str | None" = None) -> frozenset[str]:
        """Codes active for ``path`` (relative to ``root``), after every selection rule."""
        if self.select is None:
            on = set(default_codes())
        else:
            on = {c for c in RULES if any(_matches(c, s) for s in self.select)}
        on |= {c for c in RULES if any(_matches(c, s) for s in self.extend_select)}
        on -= {c for c in RULES if any(_matches(c, s) for s in self.ignore)}
        if path is not None:
            rel = PurePosixPath(Path(path).as_posix())
            for glob, codes in self.per_file_ignores:
                if fnmatch.fnmatch(str(rel), glob) or rel.match(glob):
                    on -= {c for c in RULES if any(_matches(c, s) for s in codes)}
        return frozenset(on)

    def range_for(self, name: str) -> "tuple[float, float] | None":
        for k, v in self.ranges:
            if k == name:
                return v
        return None

    def is_excluded(self, rel: str) -> bool:
        parts = PurePosixPath(rel).parts
        for pat in DEFAULT_EXCLUDE + self.exclude:
            if any(fnmatch.fnmatch(p, pat) for p in parts) or fnmatch.fnmatch(rel, pat):
                return True
        return False

    def fingerprint(self) -> str:
        """Hash of everything that changes analysis results (for the cache key)."""
        data = {"precision_bits": self.precision_bits, "default_range": self.default_range,
                "ranges": self.ranges, "budget": self.budget, "execute": self.execute,
                "money_names": self.money_names}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def find_pyproject(start: "str | Path") -> "Path | None":
    """The nearest ``pyproject.toml`` with a ``[tool.pycodemath.lint]`` table, walking up."""
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for d in (p, *p.parents):
        f = d / "pyproject.toml"
        if f.is_file():
            try:
                data = tomllib.loads(f.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if "lint" in data.get("tool", {}).get("pycodemath", {}):
                return f
    return None


def load_config(path: "str | Path | None" = None, *, start: "str | Path" = ".") -> Config:
    """Read ``[tool.pycodemath.lint]`` from ``path`` or the nearest ``pyproject.toml``.

    No table found: the defaults, rooted at ``start``.
    """
    if path is None:
        found = find_pyproject(start)
        if found is None:
            root = Path(start).resolve()
            return Config(root=str(root if root.is_dir() else root.parent))
        path = found
    f = Path(path)
    try:
        data = tomllib.loads(f.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {f}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{f} is not valid TOML: {exc}") from None
    table = data.get("tool", {}).get("pycodemath", {}).get("lint", {})
    if not isinstance(table, Mapping):
        raise ConfigError(f"{f}: [tool.pycodemath.lint] must be a table")
    return Config.from_mapping(table, root=str(f.resolve().parent), source=str(f))


del field
