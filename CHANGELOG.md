# Changelog

All notable changes to Pycodemath are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Known limitations
- **`time_budget` / `TimeBudgetError` is best-effort, not a hard guarantee.**
  The interrupt mechanism (`ctypes.PyThreadState_SetAsyncExc`, the only
  cross-platform option without a per-call subprocess) can occasionally fail
  to deliver on heavily loaded or virtualized systems, letting a hung call
  run past its budget instead of refusing on time. Measured directly:
  Python 3.13 under WSL2. Normal calls (the fast path, which is nearly all
  of them) are unaffected, and no wrong answer is ever produced — only a
  late refusal in the rare pathological-hang case. A subprocess-based hard
  backstop is planned but not yet built.

## [0.3.0] - 2026-08-19

Every failure and every success now carries evidence an agent can branch on,
instead of a bare value or a string to pattern-match.

### Added
- **Typed exception hierarchy** under `PycodemathError`: `ParseError`,
  `DomainError`, `DivergenceError`, `StagnationError`, `NonConvergenceError`.
  A caller can catch the specific outcome instead of string-matching the
  English message; `except PycodemathError` still catches everything.
- **Structured solver results**: `root_find`, `root_find_nd`, `minimize` and
  `minimize_nd` take a keyword-only `full_result=True` and return a
  `SolveResult` (value / iterations / residual / converged / status).
  `integrate_num` likewise returns a `QuadratureResult` (adds an
  `error_estimate`) and gained an adaptive `tol` that refines panels until the
  estimate fits inside it, instead of always running a fixed 100-panel grid.
  Default calls are unchanged, bit-identical to 0.2.0.
- **`converged` is now corroborated**, not just "the tolerance test fired": a
  vanishing gradient at a saddle or a maximum, or `|f| < tol` reached because
  `f` decayed rather than because `x` is a root, no longer reports false
  convergence.
- **A symbolic refusal taxonomy**: `NoClosedFormError` (the engine searched
  and found nothing) vs. `UnsupportedFormError` (no method for this shape at
  all) — so a caller knows whether to stop or re-ask numerically instead of
  parsing English.
- **A time budget for symbolic calls**: `pycodemath.time_budget(seconds)` (a
  context manager) and a `budget <s>` trailing REPL/MCP option bound how long
  a symbolic call may run and raise `TimeBudgetError` instead of hanging a
  tool call forever with no result.
- **REPL/MCP grammar reaches solver knobs the Python API already had**:
  trailing `tol <t>` / `max_iter <k>` options on `min` / `min_nd` /
  `nintegrate`, and `budget <s>` on the symbolic commands.
- **The MCP surface carries the structured results**: `math_eval` returns
  `structuredContent` (`text` / `solve` / `quadrature` / `error` fields) that
  a client validates against the declared schema, not just prose text. A
  raising call now always returns rather than throwing past the tool
  boundary.
- `py.typed` + packaging fixes: an external `mypy` now actually sees the
  package's types (previously silently fell back to `Any`), and every symbol
  a public function returns (e.g. `SolveResult`) is importable from the
  top-level package.

### Fixed
- **Security**: matrix literals (`matrix`/`det`/`inv`/`eig`/`solve_system`/
  `M`/`V`) parsed user text with bare `sp.sympify()`, so attribute access
  evaluated (`().__class__.__bases__` reaches `object`). Routed through the
  same whitelist parser scalars already use.
- Identifiers with a trailing digit (`q1`, `x2`, `v0`, `R1` — a common
  physics/engineering subscript convention) crashed the parser with a leaked
  `NameError` instead of a clean `ParseError`.
- `Abs` / `sign` / `Min` / `Max` did not differentiate on a plain
  (assumption-free) symbol — silently wrong output from `diff`, and a raw
  SymPy crash instead of a `PycodemathError` from anything that
  differentiates internally (`min`, `solve_nd`, the stiff-ODE Jacobian).
- A non-finite `tol` (e.g. `tol inf`) certified an unmoved starting guess as
  converged; tolerances are now validated finite.
- `mcp` dependency pinned to `>=1.0,<2.0` — the unbounded constraint let a
  fresh install pick up `mcp` 2.0.0, which removed an API the MCP test
  helper and integration path both relied on.

## [0.2.0] - 2026-07-10

First public release (sdist + wheel; MIT license, English README).

### Added
- **Full ODE suite** (`engine/ode.py`): symbolic `dsolve`; fixed-step RK4
  (scalar + systems); adaptive Dormand–Prince 5(4) with FSAL; dense output
  (callable `sol(t)`, 4th-order interpolant); event detection with direction
  filter and terminal events; stiff solvers (BDF2 + symbolic-Jacobian Newton,
  scalar + systems); **variable-step BDF2** with local error control; backward
  integration (`t1 < t0`) in every solver. Singularities and divergence raise
  a readable `PycodemathError` — never NaN/garbage.
- **ODE codegen**: standalone RK4, adaptive DOPRI5 and dense-output
  integrators (`emit_ode*` / `generate_ode*`) — emitted solvers are
  bit-for-bit identical to the engine.
- **Second-order optimization**: `minimize` / `minimize_nd`
  `method="newton" | "bfgs"` with Armijo backtracking line search
  (default `"gd"` unchanged).
- **Symbolic extensions**: `limit` (one-sided directions, `oo` endpoints),
  `series` (Taylor/Laurent, plain `Expr` result with the O-term removed) and
  `summation` (finite and symbolic bounds incl. `oo`; divergent or
  closed-form-less sums refuse with a readable error) + REPL commands
  `limit` / `series` / `sum`.
- **Hardened parser**: `parse()` resolves only a curated whitelist of
  mathematical functions/constants and an explicit empty-builtins global
  namespace — `__import__`, `eval`, `open` etc. are unresolvable at the
  parser level; string literals and attribute access are rejected outright.
  Unknown names still become plain symbols, so ordinary math is unaffected.
- Distribution metadata: `pyproject.toml` (0.2.0, MIT), `LICENSE`, English
  `README.md`, and a `mypy` typecheck step in CI (lenient baseline — the
  package checks clean).

### Fixed
- Type annotations tightened across the package so `mypy` passes with zero
  errors (no behavioral changes).
- `PycodemathError` contract enforced on all input boundaries (matrix
  literals, REPL numbers), emitted RK4 refuses divergence with a clean ASCII
  `ValueError`, reserved names rejected in codegen, adaptive integrator lands
  exactly on `t1`.
- Convergent sums without a closed form no longer report "divergent"
  (divergence check ran before the closed-form check on unevaluated `Sum`).

## [0.1.0] - 2026-07-08

Initial working core.

### Added
- IR (`Expr` / `Matrix`) with structural hash/eq contract and compiled-function
  LRU cache; parser with implicit multiplication (`2x`, `sin(x)x`).
- Symbolic engine: simplify, expand, `diff`, `integrate`, `solve`.
- Linear algebra: det/inv/transpose/eig, linear-system solving.
- Numerics: Newton root finding, Simpson integration, gradient descent (1D
  and multivariate: gradient/Jacobian/Hessian, `root_find_nd`, `minimize_nd`).
- Code generation with symbolic simplification + CSE (measured ~1.7× faster
  than naive expansion).
- REPL on a regex command table, one-shot CLI (`python -m pycodemath "…"`),
  MCP server (`math_eval` tool, extra `[mcp]`), GitHub Actions CI
  (Ubuntu + Windows, Python 3.11/3.12).
