"""X1 — a certificate that can be re-checked: JSON round trip, tamper evidence.

A certificate is the RECORD of one check: the claim as written and as parsed,
the verdict, the counterexample, the budget and the versions it ran under, and a
sha256 of all of it. ``recheck`` re-runs the check from the record alone and
reports every way the two disagree — a different verdict, a digest that no
longer matches (a hand edit), a claim that now parses differently.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pycodemath.core.errors import DomainError, PycodemathError
from pycodemath.verify import (
    SCHEMA_VERSION,
    Certificate,
    CertificateMismatch,
    check_equal,
    issue,
    recheck,
)
from pycodemath.verify.certificate import KINDS, content_digest, from_verdict


def _edited(cert: Certificate, edit, *, redigest: bool = False) -> str:
    """The certificate's JSON after a hand edit; ``redigest`` forges the digest."""
    data = json.loads(cert.to_json())
    edit(data)
    if redigest:
        data["digest"] = content_digest(data)
    return json.dumps(data)


def test_certificate_roundtrip_and_tamper():
    cert = issue("identity", "sqrt(x^2)", "x", source="verify sqrt(x^2) == x")
    data = json.loads(cert.to_json())

    # The record: every field the plan names, exact values, versioned schema.
    assert data["schema"] == SCHEMA_VERSION == 1
    assert data["kind"] == "identity"
    assert data["source"] == "verify sqrt(x^2) == x"
    assert data["claim"]["args"] == {"left": "sqrt(x^2)", "right": "x"}
    assert data["claim"]["parsed"] == {"left": "sqrt(x**2)", "right": "x"}
    assert data["verdict"]["status"] == "refuted"
    assert data["verdict"]["counterexample"] == {"x": "-1"}
    assert data["budget"] > 0
    assert set(data["versions"]) == {"pycodemath", "sympy", "mpmath", "python"}
    assert data["digest"].startswith("sha256:") and len(data["digest"]) == 71

    # Round trip: text -> object -> text is the identity, and it re-checks clean.
    again = Certificate.from_json(cert.to_json())
    assert again == cert
    assert again.to_json() == cert.to_json()
    result = recheck(again)
    assert result.ok, result.problems
    assert result.verdict["counterexample"] == {"x": "-1"}

    # A hand edit without a new digest: caught by the digest alone.
    flipped = _edited(cert, lambda d: d["verdict"].update(
        status="verified", counterexample=None))
    result = recheck(flipped)
    assert not result.ok
    assert not result.digest_ok
    assert not result.reproduced  # and the re-run disagrees with the edit too
    assert any("digest" in p for p in result.problems)

    # A forged digest: the digest passes, the re-run still catches the lie.
    forged = _edited(cert, lambda d: d["verdict"].update(
        status="verified", counterexample=None), redigest=True)
    result = recheck(forged)
    assert result.digest_ok and not result.reproduced and not result.ok
    assert any("refuted" in p and "verified" in p for p in result.problems)

    # A forged counterexample: same status, a different point — still a mismatch.
    moved = _edited(cert, lambda d: d["verdict"].update(
        counterexample={"x": "-2"}), redigest=True)
    assert not recheck(moved).ok

    # The claim edited under a valid digest: the parsed form no longer matches.
    reclaimed = _edited(cert, lambda d: d["claim"]["args"].update(right="abs(x)"),
                        redigest=True)
    result = recheck(reclaimed)
    assert not result.claim_ok and not result.ok

    # The parsed form edited (or: the parser moved under the claim) — caught too.
    drifted = _edited(cert, lambda d: d["claim"]["parsed"].update(right="x + 1"),
                      redigest=True)
    result = recheck(drifted)
    assert result.digest_ok and result.reproduced and not result.claim_ok

    # Not a certificate at all: a typed refusal, never a raw exception.
    for bad in ("{", "[]", json.dumps({**data, "schema": 99}),
                json.dumps({**data, "kind": "telepathy"})):
        with pytest.raises(DomainError):
            Certificate.from_json(bad)


def test_every_kind_round_trips_and_rechecks():
    cases = [
        ("identity", ("(x+1)^2", "x^2+2*x+1"), {}, "verified"),
        ("derivative", ("x^3", "x", "3*x^2"), {}, "verified"),
        ("derivative", ("x^3", "x", "3*x"), {}, "refuted"),
        ("integral", ("2*x", "x", "x^2 + 5"), {}, "verified"),
        ("roots", ("x^2-4", "x", ["2", "-2"]), {}, "verified"),
        ("roots", ("x^2-4", "x", ["2"]), {}, "refuted"),
        ("limit", ("sin(x)/x", "x", "0", "1"), {}, "verified"),
        ("ode", ("-2*t*y", "y", "t", ["C1*exp(-t^2)"]), {}, "verified"),
        ("steps", ("2x + 3 = 7 -> 2x = 4 -> x = 3",), {}, "refuted"),
    ]
    assert {k for k, *_ in cases} <= set(KINDS)
    for kind, args, opts, status in cases:
        cert = issue(kind, *args, **opts)
        assert cert.verdict["status"] == status, (kind, args, cert.verdict)
        if status == "refuted":
            assert cert.verdict["counterexample"] is not None
        back = Certificate.from_json(cert.to_json())
        assert back == cert
        result = recheck(back)
        assert result.ok, (kind, result.problems)


def test_export_from_an_existing_verdict_does_not_rerun_and_rechecks():
    verdict = check_equal("log(x^2)", "2*log(x)")
    cert = from_verdict("identity", ("log(x^2)", "2*log(x)"), verdict)
    assert cert.verdict["status"] == verdict.status.value
    assert cert.verdict["counterexample"] == verdict.counterexample
    assert recheck(cert).ok


def test_unknown_kind_and_bad_arguments_are_typed_refusals():
    with pytest.raises(DomainError):
        issue("telepathy", "x")
    with pytest.raises(DomainError):
        issue("identity", "x")  # one side missing
    with pytest.raises(PycodemathError):
        issue("identity", "x +* 2", "x")


def test_version_drift_is_a_note_not_a_failure():
    cert = issue("identity", "x+x", "2*x")
    drifted = _edited(cert, lambda d: d["versions"].update(sympy="0.0.1"),
                      redigest=True)
    result = recheck(drifted)
    assert result.ok
    assert any("sympy" in n for n in result.notes)


def test_repl_certificate_and_recheck_commands(tmp_path):
    from pycodemath.cli import repl

    text = repl.handle("certificate verify sqrt(x^2) == x")
    data = json.loads(text)
    assert data["kind"] == "identity" and data["verdict"]["status"] == "refuted"
    assert data["source"] == "verify sqrt(x^2) == x"

    path = tmp_path / "c.json"
    path.write_text(text, encoding="utf-8")
    out = repl.handle(f"recheck {path}")
    assert out.startswith("REPRODUCED")

    cert = repl.handle("certificate certify integrate x*cos(x) dx")
    data = json.loads(cert)
    assert data["kind"] == "integral" and data["verdict"]["status"] == "verified"
    assert data["claim"]["args"]["expr"] == "x*cos(x)"

    path.write_text(_edited(Certificate.from_json(text), lambda d: d["verdict"].update(
        status="verified", counterexample=None)), encoding="utf-8")
    with pytest.raises(CertificateMismatch) as exc:
        repl.handle(f"recheck {path}")
    assert "digest" in str(exc.value)


def test_cli_recheck_exit_codes(tmp_path):
    cert = issue("identity", "sqrt(x^2)", "x")
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(cert.to_json(), encoding="utf-8")
    bad.write_text(_edited(cert, lambda d: d["verdict"].update(
        counterexample={"x": "-5"})), encoding="utf-8")

    def run(path):
        return subprocess.run([sys.executable, "-m", "pycodemath", "recheck", str(path)],
                              capture_output=True, text=True, timeout=120)

    ok = run(good)
    assert ok.returncode == 0 and ok.stdout.startswith("REPRODUCED"), ok
    tampered = run(bad)
    assert tampered.returncode == 1 and "digest" in tampered.stderr, tampered


def test_mcp_verify_carries_the_certificate():
    from pycodemath.cli.mcp_server import math_verify

    out = math_verify("sqrt(x^2) == x")
    assert out["certificate"] is not None
    cert = Certificate.from_dict(out["certificate"])
    assert cert.verdict["counterexample"] == {"x": "-1"}
    assert recheck(cert).ok
    refused = math_verify("x +* 2 == x")
    assert refused["certificate"] is None


def test_engine_output_the_parser_cannot_read_is_kept_as_an_exact_tree():
    from pycodemath.cli import repl

    # ``erf`` is printed by the engine but unknown to the parser (measured): the
    # result is recorded as the exact SymPy tree, and the re-check reads it back.
    for line, kind in [("certify integrate exp(-x^2) dx", "integral"),
                       ("certify solve x^5-x-1 for x", "roots")]:
        cert = repl.handle_checked(line).certificate
        assert cert is not None and cert.kind == kind
        assert "sympy" in json.dumps(cert.claim["args"])
        assert recheck(Certificate.from_json(cert.to_json())).ok, line


@pytest.mark.parametrize("tree", [
    "__import__('os').system('true')",
    "Symbol('x').__class__",
    "open('certificate.json')",
    "Integer(10**10)",
    "Pow(Integer(2), Integer(" + "9" * 1200 + "))",
    "(lambda: Symbol('x'))()",
])
def test_a_hostile_sympy_tree_is_refused_never_evaluated(tree):
    cert = issue("identity", "x", "x")
    data = json.loads(cert.to_json())
    data["claim"]["args"]["left"] = {"sympy": tree}
    data["digest"] = content_digest(data)
    with pytest.raises(DomainError, match="unreadable SymPy tree"):
        recheck(json.dumps(data))


def test_every_certify_command_carries_a_certificate_that_reproduces():
    from pycodemath.cli import repl

    for line in [
        "verify steps 2x + 3 = 7 -> 2x = 4 -> x = 2",
        "certify diff sin(x)*x dx",
        "certify solve x^2-4 for x complex",
        "certify limit 1/x for x to 0 dir -",
        "certify dsolve -2*t*y for y(t)",
        "certify nintegrate exp(-x^2) dx from 0 to 1",
    ]:
        cert = repl.handle_checked(line).certificate
        assert cert is not None, line
        assert cert.source == line
        assert recheck(cert.to_dict()).ok, line
