# Changelog

All notable changes to Pycodemath are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.4.0] - (date set on release)

Pycodemath now checks math, not just computes it: a verifier that answers
VERIFIED, REFUTED (with a counterexample you can re-check yourself) or
UNDECIDED — and never promotes "agrees numerically" to "proved".

### Added
- **`pycodemath.verify.check_equal(a, b)`** returns a `Verdict`
  (`VERIFIED` / `REFUTED` / `UNDECIDED`, the method used, a counterexample
  and a one-line reason). Symbolic proof first; then deterministic
  high-precision sampling that avoids singularities. A counterexample is
  re-checked at higher precision before it is reported, so rounding noise is
  never called a refutation. Domain traps are caught: `sqrt(x^2)` vs `x`
  is REFUTED at a negative point, not "simplified" into equality.
- **`certify(...)`** — engine results checked by an independent route:
  integrals by differentiating back, roots by substitution plus a
  completeness check, derivatives by finite differences, limits by
  two-sided numeric sequences, ODE solutions by substitution, numeric
  integrals by a second, independent quadrature (`certify_integrate`,
  `certify_diff`, `certify_solve`, `certify_limit`, `certify_dsolve`,
  `certify_nintegrate`). Existing result types are unchanged.
- **`check_steps(...)`** — checks a derivation step by step: per-step
  verdicts, the index of the first wrong step and a counterexample. Equation
  chains are compared as solution sets, with a warning when a step adds roots
  (squaring both sides) or loses them (dividing by an expression that can be
  zero).
- **REPL and MCP surface**: `verify <a> == <b>`, `verify steps ...` and
  `certify <command>` in the REPL; a `math_verify` MCP tool next to
  `math_eval`, returning `structuredContent` with the same rules (failure is
  an `error` field, never an exception).
- **`isolated(fn, *args, budget=...)`** — a hard time limit. The in-process
  `time_budget` cannot stop one long C call that holds the GIL
  (`7**(10**7)` under a 2 s budget ran 7.9–8.7 s). `isolated` runs the call
  in a warm worker process and kills it at budget + 0.5 s (same case: 2.5 s),
  raising the new `IsolationError`. Opt-in (~0.4 ms per call); the fast path
  of `time_budget` is unchanged.
- **Benchmarks, with their denominators** (`python -m
  pycodemath.bench.verify_bench`, `python -m pycodemath.bench.real_bench`):
  - Synthetic: 210 generated derivations, half with an injected typical
    model error — 105/105 errors caught, 0 false positives. These are
    synthetic, and the verifier was fixed against misses found on this set.
  - Real model errors (PRM800K, MIT-licensed sample): the verifier can
    extract a checkable claim from 9.5% of steps; it pinpoints the labelled
    first wrong step in 24 of 378 flawed solutions (6.3%) and raised 0 false
    alarms on 122 correct ones. Low recall, zero false alarms: when it says
    a step is wrong, it shows why.
- `hypothesis` joins the `dev` extra (property-based tests of the verifier).

### Changed
- **A call to an unknown function is now a `ParseError`** ("unknown
  function"), not a product of symbols. Previously `zeta(2)` parsed as
  `2*zeta`, which let `verify zeta(2) == pi^2/6` "refute" a true identity.
- A bare function name (`sin`, `log`, `(sqrt)`) and stray brackets (`)`,
  `x)`) raise `ParseError` instead of a raw `TypeError` / `IndexError`.
- `Expr.to_source()` round-trip contract is now stated and tested as
  mathematical equivalence: `parse(e.to_source())` is always equivalent to
  `e` (or the parser refuses), never a different expression; structural
  `==` is not promised, because SymPy re-distributes constants on parse.

### Fixed
- `Expr.evalf()` and compiled evaluation no longer leak a bare `TypeError`;
  nested `zoo` becomes a `DomainError`.
- `time_budget`: an interrupt could land after the guarded block had
  exited and escape as an internal `_Deadline`; a nested budget poisoned the
  outer one. Both fixed (Windows reproduced it most).
- A huge literal such as `2^1e10` no longer hangs before the budget is armed.
- `certify_solve` accepts real roots of a cubic in the casus irreducibilis
  (an imaginary part at 1-bit precision is treated as zero only when the
  real part has full precision); quartics take candidates from `nroots`
  instead of spending the whole budget on Ferrari's formula; decimal
  exponents (`5^7.5` vs `sqrt(5^15)`) no longer yield a false REFUTED.
- The MCP server imports on Python 3.11 again.
- **Python 3.13: the time-budget interrupt no longer escapes a `with
  time_budget(...)` block.** CPython 3.13 (only — gh-139622, open) lets an
  asynchronous exception skip every handler of the frame it is raised in, so
  a block around code in the caller's own frame let the raw internal
  interrupt out and kept re-injecting into whatever ran next (measured: 14 of
  18 test runs on Linux 3.13, 5 of 6 on Windows 3.13, 0 on 3.12). On 3.13 the
  watchdog now holds back while the interrupted frame sits inside a handler
  and waits for a frame where nothing would be skipped; engine calls are
  interrupted as before. CI now runs 3.11–3.14.

### Known limitations
- `check_equal("cosh(t1 - log(0))", ...)` can raise a bare `TypeError`,
  and `trigsimp` a bare `RecursionError`; `sqrt(2)^10^20` can still hang in
  the parser.
- `solve` can miss periodic solutions; `QuadratureResult.error_estimate`
  can under-estimate.
- `isolated` is not yet wired into the MCP server or the REPL.
- `time_budget` on its own stays best-effort (see 0.3.0 below); `isolated`
  is the hard limit.

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
