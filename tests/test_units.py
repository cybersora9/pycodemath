"""J1: jednostki i wymiary — strażniki, zestaw oceny, tabela, runda adwersarza.

Strażniki (widziane CZERWONE przed implementacją, wpis w dev-journal/full/J1.md):
``test_dimension_mismatch_is_refuted_with_units`` i
``test_celsius_doubling_is_refused_or_explicit``.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from pycodemath.core.errors import DomainError, ParseError, PycodemathError
from pycodemath.engine import _units_table as table
from pycodemath.engine import units as U
from pycodemath.engine.units import (
    AffineTemperatureError,
    Dimension,
    DimensionError,
    Quantity,
    UnitParseError,
    check_dimensions,
    constant,
    convert,
    dimension_of,
    parse_unit,
    units_from_names,
)

_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Strażniki
# =============================================================================


def test_dimension_mismatch_is_refuted_with_units() -> None:
    r = check_dimensions("F", "m*v", {"F": "N", "m": "kg", "v": "m/s"})
    assert r.status.value == "refuted"
    # Obie strony z wymiarem, i to wymiarem fizycznym, nie „lewa != prawa".
    assert "kg·m/s²" in r.detail and "kg·m/s" in r.detail
    # Zero alarmów bez świadka: konkretne wejście + zmiana jednostki.
    assert r.counterexample and "rescale" in r.counterexample
    # A dobry wzór przechodzi — kontrola fałszywego alarmu na tym samym przykładzie.
    ok = check_dimensions("F", "m*a", {"F": "N", "m": "kg", "a": "m/s^2"})
    assert ok.status.value == "verified"


def test_celsius_doubling_is_refused_or_explicit() -> None:
    t = Quantity.parse("20 degC")
    with pytest.raises(AffineTemperatureError) as exc:
        _ = t * 2
    assert isinstance(exc.value, DomainError)
    assert "delta_degC" in str(exc.value) or "K" in str(exc.value)
    # Różnica temperatur jest jawna i dozwolona.
    d = Quantity.parse("30 degC") - Quantity.parse("20 degC")
    assert str(d.value) == "10" and d.unit.text == "delta_degC"
    # Absolutna konwersja z przesunięciem — dokładnie.
    c = convert("100 degC", "degF")
    assert c.value_text == "212"


# =============================================================================
# Zestaw oceny (dane: pycodemath/code/data/J1_units_eval.jsonl)
# =============================================================================


def _runner():
    path = _ROOT / "tests" / "evalsets" / "j1_run_eval.py"
    spec = importlib.util.spec_from_file_location("j1_run_eval", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RUN = _runner()
_CASES = _RUN.load("all")


def test_eval_set_has_the_promised_shape() -> None:
    assert len(_CASES) >= 60
    holdout = [c for c in _CASES if c["split"] == "holdout"]
    assert len(holdout) * 3 >= len(_CASES) - 2  # jedna trzecia wstrzymana
    kinds = {c["kind"] for c in _CASES}
    assert kinds == {"convert", "check", "names"}


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_eval_case(case: dict) -> None:
    ok, observed, false_refuted = _RUN.evaluate(case)
    assert not false_refuted, observed
    assert ok, observed


# =============================================================================
# Wymiar, jednostka, parser
# =============================================================================


def test_dimension_algebra_and_text() -> None:
    force = parse_unit("N").dimension
    assert force == Dimension.of(mass=1, length=1, time=-2)
    assert force.text == "kg·m/s²" and force.name == "force"
    assert parse_unit("V").dimension.text == "kg·m²/(s³·A)"
    assert (force / force).dimensionless
    assert (parse_unit("m").dimension ** Fraction(1, 2)).text == "m^(1/2)"
    with pytest.raises(DomainError):
        Dimension.of(colour=1)


@pytest.mark.parametrize(
    ("text", "factor"),
    [
        ("km/h", sp.Rational(5, 18)),
        ("m²", 1),
        ("s⁻¹", 1),
        ("kg·m/s²", 1),
        ("J/kg/K", 1),
        ("µs", sp.Rational(1, 10**6)),
        ("μs", sp.Rational(1, 10**6)),
        ("5/9 K", sp.Rational(5, 9)),
        ("m^(1/2)", 1),
        ("KiB", 8 * 1024),
        ("kB", 8 * 1000),
        ("Å", sp.Rational(1, 10**10)),
        ("°", sp.pi / 180),
        ("ft*lbf/s", sp.Rational(381, 1250) * sp.Rational("0.45359237") * sp.Rational("9.80665")),
    ],
)
def test_parse_unit_exact_factor(text: str, factor: object) -> None:
    assert sp.simplify(parse_unit(text).factor - factor) == 0


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("KB", "1000 B"),
        ("kb", "ambiguous"),
        ("dB", "logarithmic"),
        ("month", "not a fixed"),
        ("J/kg K", "ambiguous"),
        ("meterz", "did you mean"),
        ("m^1000", "out of range"),
        ("m/0", "finite positive"),
        ("(" * 3000 + "m" + ")" * 3000, "too long"),
        ("(" * 1900 + "m" + ")" * 1900, "nested too deeply"),
        ("m ^ 2", "unexpected"),
    ],
)
def test_parse_unit_refusals(text: str, fragment: str) -> None:
    with pytest.raises(UnitParseError) as exc:
        parse_unit(text)
    assert isinstance(exc.value, ParseError)
    assert fragment in str(exc.value)


def test_offset_unit_inside_a_compound_is_an_explicit_difference() -> None:
    u = parse_unit("J/(kg*degC)")
    assert u.offset is None and u.is_delta
    assert any("temperature difference" in n for n in u.notes)
    c = convert("4.2 J/(kg*degC)", "J/(kg*K)")
    assert c.value == sp.Rational(21, 5) and c.notes


# =============================================================================
# convert i certyfikat
# =============================================================================


def test_convert_km_h_certificate() -> None:
    c = convert("5 km/h", "m/s")
    assert c.value == sp.Rational(25, 18)
    assert str(c) == "25/18 m/s ≈ 1.38888888888889 m/s"
    cert = c.certificate
    assert cert["dimension_ratio"] == "1" and cert["round_trip"] is True
    assert cert["cross_check"] == {"by": "sympy.physics.units", "agrees": True, "factor": "5/18"}
    d = c.to_dict()
    assert json.loads(json.dumps(d))["value"] == "25/18" and d["status"] == "verified"


@pytest.mark.parametrize("name", sorted(U._SYMPY_REF))
def test_every_cross_checked_unit_agrees_with_sympy(name: str) -> None:
    # Niezależna implementacja (tabela SymPy) liczy ten sam współczynnik.
    si = {"length": "m", "mass": "kg", "time": "s", "current": "A", "temperature": "K",
          "amount": "mol", "luminous_intensity": "cd", "angle": "rad", "information": "bit"}
    u = parse_unit(name)
    target = "*".join(f"{si[b]}^{e}" for b, e in u.dimension.exponents) or "1"
    c = convert(f"1 {name}", target)
    assert c.certificate["cross_check"]["agrees"] is True, c.certificate


def test_cross_check_is_honest_when_unavailable() -> None:
    c = convert("1 psi", "Pa")
    assert c.certificate["cross_check"]["agrees"] is None
    assert "no exact counterpart" in c.certificate["cross_check"]["reason"]


def test_convert_refusals_are_typed() -> None:
    with pytest.raises(DimensionError) as exc:
        convert("1 rad/s", "Hz")
    assert "2*pi" in str(exc.value)
    with pytest.raises(DimensionError):
        convert("1 m", "s")
    with pytest.raises(AffineTemperatureError):
        convert("-273.16 degC", "K")
    with pytest.raises(AffineTemperatureError):
        convert("-1 K", "degC")
    with pytest.raises(AffineTemperatureError):
        convert("20 degC", "delta_degC")
    with pytest.raises(DomainError):
        convert("nan m", "m")
    with pytest.raises(UnitParseError):
        convert(["1 m"], "m")  # type: ignore[arg-type]
    assert convert("-273.15 degC", "K").value == 0


def test_convert_notes_name_the_interpretation() -> None:
    assert any("10^9 B" in n for n in convert("1 GB", "GiB").notes)
    assert any("Julian" in n for n in convert("1 yr", "d").notes)


# =============================================================================
# Quantity
# =============================================================================


def test_quantity_parse_forms() -> None:
    assert Quantity.parse("3/4 in").value == sp.Rational(3, 4)
    assert Quantity.parse("1.5e3 m").value == 1500
    assert Quantity.parse("-40 °C").si_value() == sp.Rational(4663, 20)
    assert Quantity.parse("km/h").value == 1
    assert Quantity.parse("1/s").unit.dimension == Dimension.of(time=-1)
    with pytest.raises(UnitParseError):
        Quantity.parse("")


def test_quantity_arithmetic() -> None:
    q = Quantity.parse("2 kg") * Quantity.parse("3 m/s^2")
    assert q.to("N").value == 6
    assert (Quantity.parse("1 km") + Quantity.parse("1 m")).value == sp.Rational(1001, 1000)
    assert (Quantity.parse("20 degC") + Quantity.parse("10 delta_degC")).value == 30
    assert (Quantity.parse("10 delta_degC") + Quantity.parse("20 degC")).unit.text == "degC"
    assert (Quantity.parse("20 degC") - Quantity.parse("68 degF")).value == 0
    assert (2 * Quantity.parse("3 m")).value == 6
    assert (Quantity.parse("6 m") / 2).value == 3
    assert (1 / Quantity.parse("2 s")).unit.dimension == Dimension.of(time=-1)
    assert (Quantity.parse("2 m") ** 2).to("m^2").value == 4
    for bad in (lambda: Quantity.parse("20 degC") + Quantity.parse("20 degC"),
                lambda: Quantity.parse("20 degC") ** 2,
                lambda: -Quantity.parse("20 degC"),
                lambda: Quantity.parse("10 delta_degC") - Quantity.parse("20 degC")):
        with pytest.raises(AffineTemperatureError):
            bad()
    with pytest.raises(DimensionError):
        Quantity.parse("1 m") + Quantity.parse("1 s")
    with pytest.raises(DimensionError):
        Quantity.parse("1 m") + 1
    with pytest.raises(DomainError):
        Quantity.parse("1 m") * float("nan")
    with pytest.raises(DomainError):
        Quantity.parse("1 m") * True


# =============================================================================
# dimension_of / check_dimensions
# =============================================================================


def test_dimension_of_report() -> None:
    r = dimension_of("m*v**2/2", {"m": "kg", "v": "m/s"})
    assert r.dimension is not None and r.dimension.name == "energy"
    d = r.to_dict()
    assert json.loads(json.dumps(d))["dimension"] == "kg·m²/s²"


def test_dimension_error_is_located_with_both_sides() -> None:
    with pytest.raises(DimensionError) as exc:
        dimension_of("v*t + a*t", {"v": "m/s", "a": "m/s^2", "t": "s"})
    e = exc.value
    assert isinstance(e, DomainError)
    assert e.location == "a*t + t*v"
    assert {e.left, e.right} == {"a*t", "t*v"}
    assert {e.left_dimension, e.right_dimension} == {"m/s", "m"}
    assert e.witness and e.witness["rescale"].startswith("time")


def _check_witness(lhs_text: str, rhs_text: str, units: dict, cx: dict) -> None:
    """Niezależne sprawdzenie świadka: przeliczamy obie strony w punkcie i w punkcie
    po podwojeniu jednostki bazowej — dokładnie, w wymiernych."""
    base_sym = cx["rescale"].split("-> 2 ")[1]
    base = {v: k for k, v in U._SYMBOL_OF_BASE.items()}[base_sym]
    left, right = U._to_sympy(cx["left"]), U._to_sympy(cx["right"])  # E is a symbol here
    point = {sp.Symbol(k): sp.Rational(v) for k, v in cx.items()
             if k not in ("rescale", "left", "right", "left_scales_by", "right_scales_by")}
    scaled = {s: v / 2 ** parse_unit(units[s.name]).dimension.get(base) for s, v in point.items()}
    for side, key in ((left, "left_scales_by"), (right, "right_scales_by")):
        ratio = sp.simplify(side.subs(scaled) / side.subs(point))
        assert ratio == sp.Rational(cx[key])
    assert cx["left_scales_by"] != cx["right_scales_by"]


@pytest.mark.parametrize(
    ("lhs", "rhs", "units"),
    [
        ("F", "m*v", {"F": "N", "m": "kg", "v": "m/s"}),
        ("E", "m*c", {"E": "J", "m": "kg", "c": "m/s"}),
        ("P", "W*t", {"P": "W", "W": "J", "t": "s"}),
        ("T", "2*pi*sqrt(L*g)", {"T": "s", "L": "m", "g": "m/s^2"}),
    ],
)
def test_refutation_witness_is_checkable_independently(lhs: str, rhs: str, units: dict) -> None:
    r = check_dimensions(lhs, rhs, units)
    assert r.refuted and r.counterexample is not None
    _check_witness(lhs, rhs, units, r.counterexample)


def test_check_dimensions_equation_text_and_to_dict() -> None:
    r = check_dimensions("F = m*a", units={"F": "N", "m": "kg", "a": "m/s^2"})
    assert r.verified and r.detail == "both sides are kg·m/s² (force)"
    assert json.loads(json.dumps(r.to_dict()))["status"] == "verified"
    with pytest.raises(UnitParseError):
        check_dimensions("F + m", units={})
    with pytest.raises(DomainError):
        check_dimensions(3, units={})  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        check_dimensions("x", "x", {"x": "m"}, values="number")


def test_check_dimensions_undecided_is_never_refuted() -> None:
    r = check_dimensions("F", "m*a*k", {"F": "N", "m": "kg", "a": "m/s^2"})
    assert r.status.value == "undecided" and r.unknown == ("k",)
    assert check_dimensions("y", "x**n", {"y": "m", "x": "m"}).status.value == "undecided"


def test_angle_is_soft_but_reported() -> None:
    r = check_dimensions("v", "omega*r", {"v": "m/s", "omega": "rad/s", "r": "m"})
    assert r.verified and "up to an angle" in r.detail
    assert any(w.kind == "angle-only" for w in r.warnings)


def test_transcendental_arguments_must_be_dimensionless() -> None:
    for f in ("sin", "exp", "log", "tanh"):
        r = check_dimensions("y", f"{f}(t)", {"y": "1", "t": "s"})
        assert r.refuted and r.location == f"{f}(t)", f


# =============================================================================
# values="numbers" i nazwy
# =============================================================================


def test_units_from_names_conventions() -> None:
    nu = units_from_names(["t_ms", "timeout_s", "widthPx", "delayMs", "T_K", "count",
                           "size_MB", "size_mb", "x_in", "h_n"])
    assert nu.units == {"t_ms": "ms", "timeout_s": "s", "widthPx": "px", "delayMs": "ms",
                        "T_K": "K", "size_MB": "MB"}
    assert set(nu.unmatched) == {"count", "size_mb", "x_in", "h_n"}
    assert units_from_names("v_kmh*t_hr").units == {"t_hr": "h", "v_kmh": "km/h"}
    assert units_from_names(["a_m"], exclude=["m"]).units == {}
    assert units_from_names(["dx", "zoom"], extra={"zoom": "1"}).units == {"zoom": "1"}
    with pytest.raises(UnitParseError):
        units_from_names(["x"], extra={"x": "furlong"})
    assert json.loads(json.dumps(nu.to_dict()))["kind"] == "units_from_names"


def test_numbers_mode_refutes_mixed_units_with_exact_witness() -> None:
    nu = units_from_names(["t_ms", "t_s", "total_s"])
    r = check_dimensions("total_s", "t_ms + t_s", nu, values="numbers")
    assert r.refuted and r.location == "t_ms + t_s"
    cx = r.counterexample
    assert cx is not None
    v = sp.Rational(cx["left_value"])
    assert sp.Rational(cx["left_in_right_units"]) == v / 1000
    # Ta sama suma w trybie „quantities" jest poprawna: oba składniki to czas.
    assert check_dimensions("total_s", "t_ms + t_s", nu).verified


def test_numbers_mode_accepts_explicit_conversions_and_warns_on_coefficients() -> None:
    nu = units_from_names("d_m - v_kmh*t_hr*1000 + timeout_ms - timeout_s + x_m + y_km")
    assert check_dimensions("d_m", "v_kmh*t_hr*1000", nu, values="numbers").verified
    assert check_dimensions("timeout_ms", "timeout_s*1000", nu, values="numbers").verified
    r = dimension_of("2*x_m + y_km", nu, values="numbers")
    assert [w.kind for w in r.warnings] == ["scale-with-coefficient"]


def test_degrees_into_trig_is_a_warning_not_a_refutation() -> None:
    nu = units_from_names(["angle_deg"])
    r = dimension_of("sin(angle_deg)", nu, values="numbers")
    assert [w.kind for w in r.warnings] == ["degrees-to-trig"]
    assert dimension_of("sin(angle_deg*pi/180)", nu, values="numbers").warnings == ()
    assert dimension_of("math.sin(math.radians(angle_deg))", nu, values="numbers").warnings == ()
    assert dimension_of("sin(angle_deg)", nu).warnings == ()  # quantities: sin(30°) is fine


def test_affine_temperature_in_expressions() -> None:
    units = {"T1": "degC", "T2": "degC", "dT": "delta_degC"}
    assert dimension_of("T1 - T2", units).warnings == ()
    assert dimension_of("T1 + dT", units).warnings == ()
    assert [w.kind for w in dimension_of("2*T1", units).warnings] == ["affine-temperature"]
    assert [w.kind for w in dimension_of("T1 + T2", units).warnings] == ["affine-temperature"]


# =============================================================================
# Runda adwersarza — regresje (pełny bilans: dev-journal/full/J1.md)
# =============================================================================


def test_zero_has_every_dimension() -> None:
    assert check_dimensions("F", "0", {"F": "N"}).verified
    assert check_dimensions("0", "F", {"F": "N"}).verified
    r = dimension_of("max(x, 0)", {"x": "m"})
    assert r.dimension == Dimension.of(length=1)


def test_log_differences_cancel_but_lone_logs_do_not() -> None:
    assert check_dimensions("y", "log(x) - log(x0)", {"y": "1", "x": "m", "x0": "m"}).verified
    assert check_dimensions("y", "2*log(x) - log(A)", {"y": "1", "x": "m", "A": "m^2"}).verified
    r = check_dimensions("y", "log(x) - log(t)", {"y": "1", "x": "m", "t": "s"})
    assert r.refuted
    with pytest.raises(DimensionError):
        dimension_of("x*log(x)", {"x": "m"})


def test_complex_valued_witness_still_counts() -> None:
    r = check_dimensions("y", "sqrt(-x) + t", {"y": "s", "x": "m^2", "t": "s"})
    assert r.refuted and r.counterexample


def test_physics_names_are_symbols_not_sympy_constants() -> None:
    r = check_dimensions("E", "I*N*S", {"E": "J", "I": "A", "N": "V", "S": "s"})
    assert r.verified
    assert check_dimensions("E = m*c^2", units={"E": "J", "m": "kg", "c": "m/s"}).verified


@pytest.mark.parametrize(
    "text",
    ["__import__('os').system('true')", "x.real", "lambda: 1", "x if y else z",
     "2**2**2**2**2", "1e309*x", "+".join(["x"] * 1500), "[x]", "f(x)", "'m'"],
)
def test_expression_reader_is_closed(text: str) -> None:
    with pytest.raises(UnitParseError):
        dimension_of(text, {"x": "m"})


def test_sympy_input_and_float_exponent() -> None:
    x = sp.Symbol("x")
    assert dimension_of(x ** 0.5, {"x": "m^2"}).text == "m"
    from pycodemath import E

    assert dimension_of(E("x^2"), {"x": "m"}).text == "m²"


def test_bad_units_argument() -> None:
    with pytest.raises(DomainError):
        dimension_of("x", {"x": 3})  # type: ignore[dict-item]
    with pytest.raises(UnitParseError):
        dimension_of("x", {"x": "meterz"})


def test_every_failure_is_a_pycodemath_error() -> None:
    bad = [lambda: convert("1 m", "s"), lambda: parse_unit("KB"),
           lambda: Quantity.parse("20 degC") * 2, lambda: constant("planck"),
           lambda: dimension_of("x+", {})]
    for fn in bad:
        with pytest.raises(PycodemathError):
            fn()


# =============================================================================
# Tabela i stałe
# =============================================================================


def test_every_table_name_resolves_to_an_exact_factor() -> None:
    names = [n for s in table.UNITS for n in (s.name, *s.aliases)]
    assert len(names) == len(set(names))
    for n in names:
        u = parse_unit(n)
        assert not u.factor.atoms(sp.Float), n
        assert u.factor.is_positive, n
    offsets = {s.name for s in table.UNITS if s.offset is not None}
    assert offsets == {"degC", "degF"}
    assert len(table.SI_PREFIXES) == 26 and len(table.BINARY_PREFIXES) == 8


@pytest.mark.parametrize(
    ("name", "value", "exact"),
    [
        ("c", "299792458", True), ("h_planck", "6.62607015e-34", True),
        ("e_charge", "1.602176634e-19", True), ("k_B", "1.380649e-23", True),
        ("N_A", "6.02214076e23", True), ("R_gas", "8.31446261815324", True),
        ("F_faraday", "96485.33212331001", True),
        ("G", "6.67430e-11", False), ("m_e", "9.1093837139e-31", False),
        ("m_p", "1.67262192595e-27", False), ("alpha_fs", "7.2973525643e-3", False),
        ("mu_0", "1.25663706127e-6", False), ("epsilon_0", "8.8541878188e-12", False),
    ],
)
def test_constants(name: str, value: str, exact: bool) -> None:
    # Wartości CODATA 2022 porównane 30.10.2026 ze scipy.constants 1.17.1.
    c = constant(name)
    assert c.exact is exact
    if exact and name not in ("R_gas", "F_faraday"):
        assert c.quantity.value == sp.Rational(value)
    else:
        assert abs(sp.N(c.quantity.value, 30) / sp.Rational(value) - 1) < 1e-14
    assert (c.uncertainty is None) is exact
    assert json.loads(json.dumps(c.to_dict()))["name"] == name


def test_derived_constants_are_exact_products() -> None:
    assert constant("R").quantity.value == constant("N_A").quantity.value * constant("k_B").quantity.value
    assert sp.simplify(constant("hbar").quantity.value * 2 * sp.pi - constant("h").quantity.value) == 0
    assert constant("R").quantity.unit.dimension == parse_unit("J/(mol*K)").dimension


def test_adapters_for_x1_certificates_and_the_cli() -> None:
    from pycodemath.verify.verdict import Verdict

    r = check_dimensions("F", "m*v", U.read_units_spec("F:N m:kg, v:m/s"))
    v = r.as_verdict()
    assert isinstance(v, Verdict) and v.refuted and v.counterexample == r.counterexample
    ok = check_dimensions("F = m*a", units=U.read_units_spec("F:N m:kg a:m/s^2")).as_verdict()
    assert ok.verified and ok.method == "dimensional-analysis"
    cv = convert("5 km/h", "m/s").as_verdict()
    assert cv.verified and "25/18" in cv.detail and cv.counterexample is None
    for bad in ("F", "F:N F:kg", "F:meterz", "1x:m"):
        with pytest.raises(UnitParseError):
            U.read_units_spec(bad)


def test_module_doctests() -> None:
    result = doctest.testmod(U, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0 and result.attempted >= 10
