# Changelog

All notable changes to Pycodemath are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-release history (0.1.0, 0.2.0) was reconstructed from the git log.

## [Unreleased]

### Added
- **First run and way-finding.** `pycodemath intro` (also `python -m
  pycodemath intro` and `intro` in the REPL): the logo, rasterized from
  `assets/pycodemath-sygnet.svg` into half-block text art by
  `scripts/gen_logo.py`, a tagline, one live demo (an off-by-one loop read as a
  formula, REFUTED with the executed call, fixed, VERIFIED; every number comes
  from `check_code` at run time) and a capability map of eight areas, each with
  a runnable command. Animated only on a TTY, under 6 s; `--static`,
  `PYCODEMATH_NO_ANIM=1` or a pipe print the final frame (under 25 lines, 80
  columns). Truecolor, a 256-colour fallback mapped to the nearest palette
  entries, monochrome under `NO_COLOR` or off a TTY. Flags: `--static`,
  `--no-logo`, `--lang en|pl`, `--once`.
- `pycodemath intro --once`: shows the intro once per session (session key:
  `PYCODEMATH_SESSION`, `CLAUDE_CODE_SESSION_ID`, the terminal's own id, else
  the process session + boot time), then prints nothing and exits 0.
- `pycodemath commands` (every command by topic), `pycodemath help <topic>`
  (the REPL help, cut by topic: solve, calculus, verify, code, numbers, money,
  units, geometry, ode, stats, start), `pycodemath examples <topic>` (3-5
  runnable examples each; a test runs every one) and `pycodemath tour` (five
  short live demos; Enter = next, q = quit; sequential off a TTY).
- MCP tool `intro`: the logo as plain text, the tagline and the capability map
  as text and data (no ANSI, no engine call). The existing tools are unchanged.
- `assets/intro.svg` / `assets/intro.pl.svg`, recorded from the real command by
  `scripts/record_intro_svg.py`, in a new "First run" section of both READMEs.

### Changed
- The interactive REPL greets with a 3-line banner (mini logo, version, where
  to start). One-shot commands still print no banner — pinned by a test.

## [0.5.0] - 2026-10-30

### Added
- **Check the math in your code** — nine modules of the FULL series, one product
 : every feature below is reachable from the Python API, the CLI/REPL, MCP,
  and documented in both READMEs; every answer on the CLI and MCP ends with a
  certificate that `recheck` re-runs from the file alone.
  - **Code against its formula.** `check_code(func_or_source, formula,
    domain=...)` proves a Python/NumPy function equal to a formula (symbolic,
    branch by branch, or an exact quasi-polynomial certificate for integer code
    with `//`/`%`) or refutes it with a call that was really executed in an
    isolated worker; `to_symbolic` reads a function into SymPy; `emit_test`
    writes the pytest that pins a counterexample. CLI: `verify code
    file.py:f == <formula> [where n: integer, n>=0] [--emit-test]`.
  - **Floating-point accuracy.** `diagnose(expr, ranges)` measures bits lost
    against the exact value, the worst input and the culprit operation;
    `improve(...)` returns rewrites that are proved equal (or a Taylor branch
    proved within the unit roundoff) and measured more accurate, with
    Python/NumPy/C code. FPCore input. CLI: `stability`, `improve`.
  - **Value ranges.** `pycodemath.code.analyze(expr, ranges, dtype)` finds
    overflow, underflow, subnormals, NaN, division by zero and lost integer
    precision in int8…uint64, float16, bfloat16, float32, float64 — each with an
    input executed in NumPy, or a proof of safety — and proposes rewrites proved
    equivalent; `max_safe_input` gives the exact largest safe input. CLI:
    `ranges ... as <dtype>`, `maxsafe`. Optional extra `ranges` (`ml_dtypes`,
    `z3-solver`).
  - **Money.** Exact amounts at the ISO 4217 scale: `allocate` (shares
    proved to sum to the total, each within one unit of its quota),
    `round_money` (8 modes, cash steps), `vat_per_line_vs_total` (with the
    proved worst case), `float_money_hazard` (which input loses a cent in
    floats), `compound_interest`, `amortization_schedule`, `npv`, `irr` (every
    rate, not one). CLI: `money round|split|vat|float|interest|loan|npv|irr`.
  - **Calculator.** `pycodemath.calc`: number theory with primality proofs
    (deterministic Miller–Rabin to 3.3e24, Lucas–Lehmer/Pocklington beyond),
    combinatorics, 13 special functions at any precision;
    `pycodemath.stats`: exact descriptive statistics, 7 distributions, an A/B
    toolkit. CLI: `factor`, `isprime`, `gcd`, `binomial`, `special`, `mean`,
    `var`, `cdf`, `samplesize`, `ci`, `ztest`, `holm`, … (43 commands).
  - **Units.** `convert` (exact, certified), `check_dimensions` (refuted
    with a witness), `dimension_of`, `units_from_names`. CLI: `convert 5 km/h to
    m/s` (the bare `5 km/h to m/s` works too), `units F = m*v where F:N m:kg
    v:m/s`, `dims`.
  - **Geometry.** Rotations across quaternions, matrices, 24 Euler
    conventions and axis-angle with independent checks;
    `compare_rotation_expressions` proves two rotation formulas equal or gives
    integer quaternions where they differ and the likely convention bug; robust
    `orient2d`/`orient3d`/`incircle`; `epsilon_witness`; CSS easing and Bézier
    arc length with proved enclosures; `emit` to Python/NumPy/C/GLSL. CLI:
    `quat2mat`, `euler2mat`, `mat2euler`, `rotcheck`, `orient2d`, `orient3d`,
    `incircle`, `segment`, `easing`, `arclength`, `emit`, `epsilon`.
- **Wave 2 of the FULL series, one product with the first nine.**
  - **Numbers you can trust.** `approx(expr, digits)` returns the correctly
    rounded decimal, decided by interval arithmetic (own rigorous enclosures for
    `zeta(n)`, `erf`, `LambertW`, `CRootOf`; an exactly-zero part proved by its
    minimal polynomial); where no enclosure exists the digits are an agreement of
    two precisions and carry `rigorous=False`. `evalf_to`, `dual` (the exact
    answer with its proved decimal), `identify` (PSLQ over 150 families, VERIFIED
    only with >= 50 digits and a coincidence margin >= 20). CLI: `evalf <expr>
    <d>`, `N(<expr>, <d>)`, `<expr> digits <d>`, `identify <x>`, and every exact
    number answers with its decimal (`sqrt(2)/2  ≈ 0.707106781186548`).
  - **Solution sets and definite integrals.** `solve_inequality` (proved by
    sign decomposition), `solve_system` (substitution + a completeness count, or
    the combination that reads `0 = 1`), `periodic_solutions` (period proved, one
    period proved complete), `definite_integral` (convergence or divergence by
    the limit comparison test at every singular end, value cross-checked by an
    independent quadrature; principal value only on request). CLI: `solve x^2-4 >
    0 for x [in [a, b]]`, `solve e1, e2 for x, y`, `solve <eq> all`, `reduce`,
    `integrate <f> dx from <a> to <b> [pv]`; `nintegrate` takes infinite bounds.
  - **Loops and recurrences.** `count_iterations` (the exact count, where it
    holds, Θ proved by a limit, then counted by running the function),
    `check_iterations` (off-by-one refuted with the executed n),
    `check_complexity`, `solve_recurrence`, `master_theorem`, `akra_bazzi`,
    `recurrence_from_code`. CLI: `iterations file.py:f [== <count>]`,
    `complexity file.py:f is O(...)`, `recurrence T(n) = ...`, `master a b f`.
  - **Lint the math in your repo.** `python -m pycodemath.lint` /
    `pycodemath-lint` / `pycodemath lint`: 17 rules (precision loss, overflow,
    float money, double rounding, `Decimal(0.1)`, `==` on floats, `^` as a power,
    banker's rounding, a `range` off by one against `:math:`), every finding with
    an executed witness and a proved fix where one exists; SARIF, pre-commit hook,
    `pytest --pcm` (entry point `pytest11`, inert without the flag), ruff's exit
    codes 0/1/2 (for `lint` only). API: `lint_source`, `lint_paths`, `spec`,
    `LintConfig`.
  - **MCP: `math_iterations` and `math_lint`** — source code in, the shared
    six-field answer out; by default nothing is executed (`execute=true` runs the
    function in the isolated worker, and is recorded in the certificate).
    Approximations, inequalities and recurrences are lines of `math_eval`.
  - **New certificate kinds:** `approx`, `identify`, `inequality`, `system`,
    `periodic`, `definite_integral`, `iterations`, `complexity`, `recurrence`,
    `lint` — each re-checked by running the module again and comparing; a
    non-rigorous approximation is UNDECIDED, never VERIFIED; a `lint` finding's
    witness is replayed independently.
- **Re-exports.** `pycodemath.code` exports every public name of the code-checking and geometry modules;
  `pycodemath` exports the most used ones (`check_code`, `diagnose`, `improve`,
  `allocate`, `convert`, `check_dimensions`, `issue`, `recheck`, …) and the
  modules `code`, `stability`, `ranges`, `money`, `geometry`, `units`, `calc`,
  `stats` — loaded lazily, so `import pycodemath` is not slower.
- **MCP: four new tools.** `math_check_code` (by default the source is only read
  and proved, never run; `execute=true` runs it in the isolated worker),
  `math_stability`, `math_ranges` and `math_recheck` (re-run a certificate) —
  one response shape: `text`, `kind`, `status`, `result`, `certificate`,
  `error`. The calculator, units, money and geometry are lines of `math_eval`.
- **New certificate kinds**: `code_equiv`, `stability`, `range`, `money`,
  `calc`, `geometry`, `unit_conversion`, `dimensions` — each re-checked by
  re-running the operation and comparing the recorded result, plus a route
  independent of the module where one is cheap (re-multiplied factors, the
  Bezout identity, shares re-added in fractions, the worst float input
  re-evaluated by the emitted NumPy code against SymPy at 40 digits, an integer
  overflow re-executed against Python ints).
- **Re-checkable certificates.** `pycodemath.verify.issue(kind, ...)` runs a
  check and writes it down as one JSON object (`schema: 1`): the claim as typed
  and as parsed, the verdict with its exact counterexample, the budget, the
  versions of pycodemath/SymPy/mpmath/Python and a sha256 digest of all of it.
  `kind` is open-ended (`identity`, `derivative`, `integral`, `roots`, `limit`,
  `ode`, `quadrature`, `steps`; `register_kind` adds more). `from_verdict`
  exports a check already run; `recheck` re-runs a certificate from the record
  alone and reports a different verdict, a digest broken by a hand edit, or a
  claim that parses differently today (version drift is a note, not a failure).
- CLI: `pycodemath certificate <verify/certify line>` prints the certificate;
  `pycodemath recheck file.json [budget <s>]` reproduces it — exit code 1 with
  every mismatch named when it does not.
- MCP: `math_verify` responses carry a sixth field, `certificate` (the JSON
  object, `null` when nothing was checked).

### Changed
- MCP `math_eval` responses carry two more fields, `result` and `certificate`
  (`null` for every command that existed before).
- `factor 2^64+1`, `isprime 2^61-1` and `5 km/h to m/s` are answered
  instead of refused as unknown commands; so are `N(pi, 50)`, `evalf sqrt(2) 50`,
  `latex x^2/2`, `solve x^2-4 > 0 for x`, `reduce x^2 < 4 for x`, `solve x^2+y^2-1,
  x-y for x,y` and `integrate exp(-x^2) dx from -oo to oo`.
- A bare exact number prints its proved decimal beside it (`pi  ≈
  3.14159265358979`); integers, floats and expressions with symbols are unchanged.
- `math_recheck` asks for `execute=true` before re-running an `iterations`,
  `complexity` or `lint` certificate whose code was executed (as for `code_equiv`).
- **Strict parsing is the default ("never guesses").** A multi-letter name
  off the function/constant whitelist is now ONE variable everywhere — Python
  API, verifier, MCP, CLI: `diff speed*time dtime` gives `speed` (it was `0`,
  exit code 0), and `verify mass*c^2 == m*a*s*s*c^2` is no longer VERIFIED.
  **Breaking:** `parse("xy")` is `Symbol("xy")`, not `x*y`. The old splitting
  is opt-in: `implicit on|off` in the REPL, `python -m pycodemath --implicit
  "<command>"` one-shot, `parse_mode(strict=False)` in Python. The MCP tools
  always parse strictly. `2x`, `2sin(x)cos(x)`, `n(n+1)`, `x y` are unchanged.

### Fixed
- `pycodemath-lint` on a file it cannot parse no longer ends with "All checks
  passed" and exit 0: the file was not checked, so the exit code is 2 (as the
  documented "2 = error") and the summary says "not analysed". `--exit-zero`
  still returns 0.
- An unknown command word is a `ParseError` with a "did you mean …" hint
  instead of a product of letters: `factor 2^64+1`, `isprime 2^61-1`,
  `latex x^2/2`, `evalf sqrt(2) 50`, `integrat x dx` (→ `integrate`).
- An unknown word next to another term is a `ParseError`, in both parsing
  modes: `5 km/h to m/s` was `5*k*m**2*o*t/(h*s)`.
- A command whose variable shows up in only one of the two readings of a
  multi-letter name (`diff xy dx` strict, `diff speed*time dtime` with
  implicit on) is a `ParseError` naming the setting, not a silent `0`.
- `sqrt(2)^10^20` (and `cbrt(3)^(10^20)`, `(2^(1/3))^(10^20)`,
  `root(5,4)^(10^30)`) is refused by the cost guard before evaluation; it hung
  the CLI and the MCP server for over 60 s.
- Raw Python exceptions that reached the user are typed: `solve x^2-4 > 0`
  (`'Or' object is not iterable` → `UnsupportedFormError`), `N(pi, 50)` /
  `f(x, y)` (a single letter called with a list → `ParseError` "unknown
  function", no `SymPyDeprecationWarning` on stderr),
  `check_equal("cosh(t1 - log(0))", "1")` (`Invalid NaN comparison` →
  UNDECIDED), `2^20000` (`Exceeds the limit (4300 digits)` → printed; the
  digit limit is raised to the largest result the cost guard admits).
- SymPy deprecation warnings are silenced on the CLI and MCP stderr.
- A certificate records the parsing mode (`strict` in its options): one issued
  after `implicit on` re-checked as "the claim parses differently".
- A certificate's SymPy tree is refused on reading when it is too costly to
  evaluate (`Pow(10, Pow(10, 999))` built instantly and hung once checked),
  over 100 000 characters, 20 000 nodes or 100 levels deep.
- `diagnose` and the ranges analysis set mpmath's global interval precision; their entry points on the
  CLI and MCP are serialized so two runs never compute at each other's
  precision.

## [0.4.0] - 2026-09-30

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
instead of a bare value or a string to pattern-match (v0.3 module series,
Modules 1-11).

### Added
- **Typed exception hierarchy** under `PycodemathError`: `ParseError`,
  `DomainError`, `DivergenceError`, `StagnationError`, `NonConvergenceError`
  (plus a reserved `BudgetExhaustedError`). A caller can catch the specific
  mathematical outcome instead of string-matching the message; `except
  PycodemathError` still catches everything.
- **Structured solver results**: `root_find`, `root_find_nd`, `minimize` and
  `minimize_nd` take a keyword-only `full_result=True` and return a
  `SolveResult` (value / iterations / residual / converged / status).
  `integrate_num` likewise returns a `QuadratureResult` (adds an
  `error_estimate`) and gained an adaptive `tol` that refines panels until
  the estimate fits inside it. Default calls are unchanged, bit-identical.
- **`converged` is now corroborated**, not just "the tolerance test fired":
  a vanishing gradient at a saddle or a maximum, or `|f| < tol` reached
  because `f` decayed rather than because `x` is a root, no longer reports
  false convergence.
- **A symbolic refusal taxonomy**: `NoClosedFormError` (the engine searched
  and found nothing) vs. `UnsupportedFormError` (no method for this shape at
  all) — so a caller knows whether to stop or re-ask numerically.
- **A time budget for symbolic calls**: `pycodemath.time_budget(seconds)`
  and a `budget <s>` trailing REPL/MCP option bound how long a symbolic call
  may run and raise `TimeBudgetError` instead of hanging forever.
- **REPL/MCP grammar reaches solver knobs the Python API already had**:
  trailing `tol <t>` / `max_iter <k>` on `min` / `min_nd` / `nintegrate`, and
  `budget <s>` on the symbolic commands.
- **The MCP surface carries the structured results**: `math_eval` returns
  `structuredContent` (`text` / `solve` / `quadrature` / `error`) validated
  against the declared schema, not just prose text.
- `py.typed` + packaging fixes: an external `mypy` now sees the package's
  types, and every symbol a public function returns is importable from the
  top-level package.
- **Symbolic extensions**: `limit` (one-sided directions, `oo` endpoints),
  `series` (Taylor/Laurent, plain `Expr` result with the O-term removed) and
  `summation` (finite and symbolic bounds incl. `oo`; divergent or
  closed-form-less sums refuse with a readable error) + REPL commands
  `limit` / `series` / `sum`.
- `CHANGELOG.md` (this file) and a `mypy` typecheck step in CI
  (extra `[dev]`, lenient baseline — package checks clean).

### Changed
- **Hardened parser**: `parse()` resolves only a curated whitelist of
  mathematical functions/constants and an explicit empty-builtins global
  namespace — `__import__`, `eval`, `open` etc. are unresolvable at the
  parser level; string literals and attribute access are rejected outright.
  Unknown names still become plain symbols, so ordinary math is unaffected.
- Type annotations tightened across the package so `mypy` passes with zero
  errors (no behavioral changes).

### Removed
- **Code-protection subsystem** (`codegen/protect.py`, extra `[protect]`) and
  the **offline licensing subsystem** (`pycodemath.license`, extra `[license]`,
  `cli/license_cli.py`, the product-command license gate). The AES-on-formulas
  "protection" was a gimmick — the emitted loader decrypts and `exec`s the body
  in-process, so encrypting `sin(x)+x^2` protected nothing — and licensing is
  being handled separately. Also dropped: the REPL `protect` / `protect_key` /
  `license` commands and the MCP license file-I/O gate. This is a breaking
  removal of public API (hence the minor bump). The code generator, the whole
  math engine and MCP `math_eval` are unchanged.

### Fixed
- **Security**: matrix literals parsed user text with bare `sp.sympify()`,
  so attribute access evaluated (`().__class__.__bases__` reaches `object`).
  Routed through the same whitelist parser scalars already use.
- Identifiers with a trailing digit (`q1`, `x2`, `v0`, `R1`) crashed the
  parser with a leaked `NameError` instead of a clean `ParseError`.
- `Abs` / `sign` / `Min` / `Max` did not differentiate on a plain symbol —
  silently wrong `diff` output, and a raw SymPy crash instead of a
  `PycodemathError` from anything that differentiates internally (`min`,
  `solve_nd`, the stiff-ODE Jacobian).
- A non-finite `tol` (e.g. `tol inf`) certified an unmoved starting guess as
  converged; tolerances are now validated finite.
- `mcp` dependency pinned to `>=1.0,<2.0` — the unbounded constraint let a
  fresh install pick up `mcp` 2.0.0, which removed an API the MCP test
  helper and integration path both relied on.
- Audit follow-up (5×P1, 8×P2): `PycodemathError` contract enforced on all
  input boundaries (matrix literals, REPL numbers, base64 keys), emitted RK4
  refuses divergence with a clean ASCII `ValueError`, reserved names rejected
  in codegen, adaptive integrator lands exactly on `t1`.
- Convergent sums without a closed form no longer report "divergent"
  (divergence check ran before the closed-form check on unevaluated `Sum`).

## [0.2.0] - 2026-07-10

First packaged release (sdist + wheel; MIT license, English README).

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
  bit-for-bit identical to the engine, and work with `protect=True`.
- **Second-order optimization**: `minimize` / `minimize_nd`
  `method="newton" | "bfgs"` with Armijo backtracking line search
  (default `"gd"` unchanged).
- Distribution metadata: `pyproject.toml` (0.2.0, MIT), `LICENSE`, English
  `README.md` (Polish original kept as `README.pl.md`).

## [0.1.0] - 2026-07-08

Initial working core (never published).

### Added
- IR (`Expr` / `Matrix`) with structural hash/eq contract and compiled-function
  LRU cache; parser with implicit multiplication (`2x`, `sin(x)x`).
- Symbolic engine: simplify, expand, `diff`, `integrate`, `solve`.
- Linear algebra: det/inv/transpose/eig, linear-system solving.
- Numerics: Newton root finding, Simpson integration, gradient descent (1D
  and multivariate: gradient/Jacobian/Hessian, `root_find_nd`, `minimize_nd`).
- Code generation with symbolic simplification + CSE (measured ~1.7× faster
  than naive expansion); AES-GCM protected artifacts with a standalone loader
  (extra `[protect]`).
- REPL on a regex command table, one-shot CLI (`python -m pycodemath "…"`),
  MCP server (`math_eval` tool, extra `[mcp]`), GitHub Actions CI
  (Ubuntu + Windows, Python 3.11/3.12).
