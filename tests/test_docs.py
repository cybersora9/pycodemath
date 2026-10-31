"""Moduł 7: dokumentacja jest URUCHAMIANA, nie zakładana.

Zasada domu: „przykład w README, który nigdy nie został wykonany, to bug z długim
lontem". Ten plik jest zapalnikiem — wyłuskuje KAŻDY przykład z obu README i
porównuje zapisane wyjście z prawdziwym.

POMIAR, KTÓRY TEN PLIK ISTNIEJE, ŻEBY POWTÓRZYĆ (moduł 7, przed zmianą): z 14
przykładów one-shot w README.md **3 się nie zgadzały** — wszystkie zapisywały
POLSKIE wyjście, które kod dawno emituje po angielsku (`(41 kroków
adaptacyjnych)` przeciw `(41 adaptive steps)`), a przykład `code` w README.pl.md
pokazywał kod SPRZED CSE, którego generator nie produkuje. Nikt tego nie
zauważył, bo nic tego nie uruchamiało.

--- JAK OBA README TRZYMAJĄ SIĘ RAZEM ------------------------------------------

README.md (angielskie, kanoniczne — to ono jedzie do metadanych pakietu) i
README.pl.md (polskie) mają OPOWIADAĆ TĘ SAMĄ HISTORIĘ. Mechanizmem synchronizacji
jest tabela ``SURFACES`` niżej: jeden wiersz = jedna powierzchnia v0.3 + marker,
po którym widać ją w KAŻDYM z dwóch plików. Dopisanie powierzchni to jeden wiersz,
a test pada, dopóki oba dokumenty jej nie opiszą. Tłumaczenia zdanie-w-zdanie NIE
pilnujemy i to jest świadome: pilnowalibyśmy formy zamiast treści, a polski tekst
przestałby być tekstem, stając się kalką.

--- KONWENCJE PRZYKŁADÓW -------------------------------------------------------

Dwie formy, obie wykonywalne:

* ``$ python -m pycodemath "<komenda>"`` + wyjście — droga one-shot (``repl.handle``);
* blok ``pycon`` w stylu doctest (``>>>`` / ``...``) — API Pythona.

PORÓWNANIE JEST TOLERANCYJNE NA LICZBY i to nie jest miękkość: proza musi zgadzać
się CO DO BAJTU (tak wyszły trzy stare kłamstwa), ale liczby zmiennoprzecinkowe
porównujemy względnie (``_FLOAT_RTOL``). Powód jest ten sam, co w całym zestawie —
„najpierw zmierz prawdziwą liczbę, potem asertuj ją z marginesem": README pinuje
pełną precyzję wyników z NumPy/lambdify, a te potrafią różnić się ostatnim ulp
między Ubuntu a Windows (macierz CI). Ostatni ulp nie jest twierdzeniem
dokumentacji; rząd wielkości i cała reszta tekstu — jest.
"""

from __future__ import annotations

import io
import math
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pycodemath.cli import repl

_ROOT = Path(__file__).resolve().parent.parent
README_EN = _ROOT / "README.md"
README_PL = _ROOT / "README.pl.md"
READMES = (README_EN, README_PL)

#: Względna tolerancja na liczby w przykładach (patrz docstring modułu).
_FLOAT_RTOL = 1e-9

#: Powierzchnie v0.3, których OBA README muszą dotykać — mechanizm synchronizacji.
#: (nazwa, marker w README.md, marker w README.pl.md)
SURFACES: tuple[tuple[str, str, str], ...] = (
    ("exception hierarchy", "PycodemathError", "PycodemathError"),
    ("input refusal vs run outcome", "DomainError", "DomainError"),
    ("full_result", "full_result=True", "full_result=True"),
    ("SolveResult", "SolveResult", "SolveResult"),
    ("QuadratureResult", "QuadratureResult", "QuadratureResult"),
    ("adaptive tolerance", "tol=", "tol="),
    ("estimate is not a bound", "never a bound", "nigdy granic"),
    ("branch on converged", "converged", "converged"),
    ("MCP response shape", "structuredContent", "structuredContent"),
    ("py.typed / no cast", "py.typed", "py.typed"),
    # moduł 8: converged odpowiada na PYTANIE, KTÓRE ZADANO — nie na „odpalił się
    # test wyjścia". Marker jest nazwą nowego statusu, bo to on jest widoczną
    # częścią zmiany; sekcja wokół niego tłumaczy obie strony rozjazdu.
    ("converged answers the question asked", "not_a_minimum", "not_a_minimum"),
    # moduł 9: wywołanie symboliczne NIE MOŻE wisieć w nieskończoność. Marker jest
    # nazwą nowej klasy, bo to ona jest tym, co agent czyta z payloadu; sekcja
    # wokół niej tłumaczy, czym ta odmowa RÓŻNI SIĘ od „no closed form" i od
    # BudgetExhaustedError, oraz czym broniona jest wartość domyślna.
    ("symbolic calls are bounded in time", "TimeBudgetError", "TimeBudgetError"),
    ("the caller sets the budget", "time_budget", "time_budget"),
    # moduł 10: odmowa symboliczna mówi, CO ZROBIĆ DALEJ, i mówi to danymi.
    # Dwa markery, bo zmiana ma dwie połowy, których nie wolno rozdzielić: KLASA
    # (jakiego rodzaju jest ta odmowa — silnik szukał czy nie ma metody) i TRASA
    # (czy jest gdzie pójść dalej). Sama klasa bez trasy nie odróżnia całki, która
    # ma numeryczny odpowiednik, od sumy nieskończonej, która go nie ma.
    ("symbolic refusal has a kind", "NoClosedFormError", "NoClosedFormError"),
    ("symbolic refusal has a route", "route", "route"),
    # moduł 11: gramatyka REPL-a/MCP dociera do knobów, które API Pythona miało
    # od zawsze. Marker jest opcją NAJPÓŹNIEJ dodaną do gramatyki (`budget`, na
    # sześciu komendach symbolicznych) — bo to ona zamyka serię odraczania.
    ("trailing options reach the grammar", "`budget <s>`", "`budget <s>`"),
    # VERIFY V4: werdykty (VERIFIED/REFUTED/UNDECIDED) wychodzą na powierzchnię —
    # REPL (`verify`, `certify`) i drugie narzędzie MCP. Marker jest nazwą
    # narzędzia, bo to ją agent widzi na liście narzędzi.
    ("verify / certify / math_verify", "math_verify", "math_verify"),
    # moduł B: twarda granica czasu przez proces roboczy. Marker to nazwa funkcji,
    # bo to ją wołający wpisuje; sekcja wokół mówi, ile kosztuje i czemu opt-in.
    ("hard time bound in a worker process", "isolated(", "isolated("),
    # I1: seria FULL jako jeden produkt — sekcja na samej górze obu README i jej
    # podsekcje. Marker to nazwa, którą wpisuje człowiek albo agent: funkcja C1,
    # narzędzia MCP filaru kodu, polecenie kalkulatora, J1 (wg propozycji J1),
    # pieniądze, geometria; K1 — ustawienie parsera.
    ("check the math in your code (C1)", "check_code(", "check_code("),
    ("float accuracy (C2)", "math_stability", "math_stability"),
    ("value ranges (C3)", "math_ranges", "math_ranges"),
    ("money (C4)", "money split", "money split"),
    ("calculator (K3)", "factor 2^64+1", "factor 2^64+1"),
    ("units (J1)", "check_dimensions", "check_dimensions"),
    ("geometry (G1)", "rotcheck", "rotcheck"),
    ("strict parser (K1)", "implicit on", "implicit on"),
    ("certificate re-check over MCP (X1)", "math_recheck", "math_recheck"),
    # I2: fala 2 — marker to polecenie albo nazwa, którą wpisuje człowiek/agent
    ("proved digits and identify (K2)", "identify 1.6449340668482264", "identify 1.6449340668482264"),
    ("solution sets, definite integrals (K5)", "integrate 1/x^2 dx from -1 to 1",
     "integrate 1/x^2 dx from -1 to 1"),
    ("loop counts and recurrences (C5)", "math_iterations", "math_iterations"),
    ("the math linter (L1)", "pycodemath-lint", "pycodemath-lint"),
)


# --- wyłuskiwanie i porównywanie ------------------------------------------


_FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+")


def _tolerant_equal(recorded: str, actual: str) -> bool:
    """Proza co do bajtu, liczby zmiennoprzecinkowe względnie (patrz docstring)."""
    if recorded == actual:
        return True
    r_nums = _FLOAT.findall(recorded)
    a_nums = _FLOAT.findall(actual)
    if len(r_nums) != len(a_nums):
        return False
    # szkielet tekstowy z wyciętymi liczbami musi być IDENTYCZNY
    if _FLOAT.sub("<f>", recorded) != _FLOAT.sub("<f>", actual):
        return False
    return all(
        math.isclose(float(r), float(a), rel_tol=_FLOAT_RTOL, abs_tol=1e-300)
        for r, a in zip(r_nums, a_nums)
    )


def _console_examples(md: str) -> list[tuple[str, str]]:
    """``$ python -m pycodemath "<cmd>"`` + zapisane wyjście, do końca bloku."""
    return [
        (cmd, out.rstrip("\n"))
        for cmd, out in re.findall(
            r'^\$ python -m pycodemath "(.+?)"\n((?:(?!\$ |```)[^\n]*\n)*)',
            md,
            re.MULTILINE,
        )
    ]


def _pycon_examples(md: str) -> list[tuple[str, str]]:
    """Bloki ``pycon``: (źródło jednej instrukcji, zapisane wyjście)."""
    out: list[tuple[str, str]] = []
    for block in re.findall(r"```pycon\n(.*?)```", md, re.DOTALL):
        lines = block.split("\n")
        i = 0
        while i < len(lines):
            if not lines[i].startswith(">>> "):
                i += 1
                continue
            src = [lines[i][4:]]
            i += 1
            while i < len(lines) and lines[i].startswith("... "):
                src.append(lines[i][4:])
                i += 1
            expected: list[str] = []
            while i < len(lines) and lines[i] and not lines[i].startswith(">>> "):
                # ``<BLANKLINE>`` jak w doctest: pusta linia KOŃCZY zapisane wyjście,
                # więc wielolinijkowy wydruk (źródło z generatora) potrzebuje znacznika
                expected.append("" if lines[i] == "<BLANKLINE>" else lines[i])
                i += 1
            out.append(("\n".join(src), "\n".join(expected).rstrip("\n")))
    return out


def _run_pycon(source: str, namespace: dict) -> str:
    """Wykonaj jedną instrukcję jak REPL: wypisz repr wartości albo nazwę+komunikat wyjątku."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            try:
                value = eval(compile(source, "<readme>", "eval"), namespace)
            except SyntaxError:
                exec(compile(source, "<readme>", "exec"), namespace)
                value = None
    except Exception as exc:  # noqa: BLE001 — README pokazuje też odmowy
        printed = buf.getvalue()
        return (printed + f"{type(exc).__name__}: {exc}").rstrip("\n")
    printed = buf.getvalue()
    if value is not None:
        printed += repr(value) + "\n"
    return printed.rstrip("\n")


# --- testy ----------------------------------------------------------------


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
def test_every_one_shot_example_in_the_readme_is_what_the_code_really_prints(
    readme: Path,
) -> None:
    md = readme.read_text(encoding="utf-8")
    examples = _console_examples(md)
    assert examples, f"{readme.name}: nie znaleziono ani jednego przykładu one-shot"
    for cmd, recorded in examples:
        actual = repl.handle(cmd).rstrip("\n")
        assert _tolerant_equal(recorded, actual), (
            f"{readme.name}: `{cmd}`\n  zapisane: {recorded!r}\n  prawdziwe: {actual!r}"
        )


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
def test_every_python_api_example_in_the_readme_is_what_the_code_really_returns(
    readme: Path,
) -> None:
    md = readme.read_text(encoding="utf-8")
    examples = _pycon_examples(md)
    assert examples, f"{readme.name}: nie znaleziono ani jednego bloku pycon"
    namespace: dict = {}
    for source, recorded in examples:
        actual = _run_pycon(source, namespace)
        assert _tolerant_equal(recorded, actual), (
            f"{readme.name}: >>> {source}\n"
            f"  zapisane: {recorded!r}\n  prawdziwe: {actual!r}"
        )


@pytest.mark.parametrize("surface,marker_en,marker_pl", SURFACES)
def test_both_readmes_document_the_same_v03_surfaces(
    surface: str, marker_en: str, marker_pl: str
) -> None:
    """Mechanizm synchronizacji obu README — jeden wiersz tabeli na powierzchnię."""
    en = README_EN.read_text(encoding="utf-8")
    pl = README_PL.read_text(encoding="utf-8")
    assert marker_en in en, f"README.md nie opisuje powierzchni: {surface}"
    assert marker_pl in pl, f"README.pl.md nie opisuje powierzchni: {surface}"


def _stated_test_count(md: str) -> int:
    found = re.findall(r"\*\*(\d+) (?:tests|testów)\*\*", md)
    assert len(found) == 1, f"oczekiwano dokładnie jednej deklaracji liczby testów, jest {found}"
    return int(found[0])


def test_both_readmes_state_the_same_test_count() -> None:
    """Zawsze sprawdzane — dwa dokumenty nie mogą chwalić się różną liczbą."""
    counts = {p.name: _stated_test_count(p.read_text(encoding="utf-8")) for p in READMES}
    assert len(set(counts.values())) == 1, counts


def test_the_test_count_the_readmes_state_is_the_real_one(
    request: pytest.FixtureRequest,
) -> None:
    """A liczba jest PRAWDZIWA — pinowane wobec tego, co pytest naprawdę zebrał.

    README.md ogłaszało **230 testów** długo po tym, jak zestaw urósł do 479:
    liczba w dokumentacji starzeje się cicho, bo nic jej nie czyta. Teraz czyta.
    Sprawdzalne tylko przy pełnym przebiegu — przy ``pytest <plik>`` albo ``-k``
    sesja zbiera podzbiór i nie ma się do czego porównać.
    """
    opt = request.config.option
    partial = (
        list(getattr(opt, "file_or_dir", []) or []) not in ([], ["."], ["tests"])
        or bool(getattr(opt, "keyword", ""))
        or bool(getattr(opt, "markexpr", ""))
    )
    if partial:
        pytest.skip("przebieg częściowy — liczba zebranych testów nie jest całością")
    stated = _stated_test_count(README_EN.read_text(encoding="utf-8"))
    assert stated == request.session.testscollected
