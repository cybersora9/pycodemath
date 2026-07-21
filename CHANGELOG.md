# Changelog

All notable changes to Pycodemath are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
