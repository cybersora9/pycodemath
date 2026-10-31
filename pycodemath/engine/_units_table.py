"""The unit and constant table behind ``engine.units`` (module J1).

Every factor here is EXACT: a decimal literal is read as the rational number it
spells (``0.3048`` is ``381/1250``, never the nearest binary double), and a
factor that involves pi (an angle, the parsec) stays a SymPy expression in pi.
Nothing is rounded on the way in, so ``convert`` can hand back ``25/18 m/s`` for
``5 km/h`` and mean it.

A unit is defined by a short definition string in terms of units defined
before it (``"12 in"``, ``"lbf/in^2"``), read by the same parser that reads the
caller's units. Writing the CHAIN rather than a pre-multiplied SI number is
deliberate: it is the form the sources state (NIST SP 811 defines the psi as a
pound-force per square inch, not as 6894.757...), and it is the form a reviewer
can check against them line by line. Two competitors measured with a rounded
psi on 30.10.2026: numbat ``6.894757 kPa`` and astropy ``6894.75738822 Pa``;
the exact value is ``6894.757293168361...``.

SOURCES (every row names one):

* ``SI2019``  — BIPM, The International System of Units, 9th ed. (2019),
  incl. Table 8 (non-SI units accepted for use with the SI).
* ``SP811``   — NIST Special Publication 811 (2008), Appendix B (conversion
  factors; the international yard and pound of 1959).
* ``IEC``     — IEC 80000-13:2008 (byte = 8 bit, binary prefixes Ki..Yi).
* ``IAU``     — IAU 2012 Resolution B2 (astronomical unit), IAU 2015 B2
  (parsec), IAU Julian year (light-year).
* ``ISO4``    — ISO 80000-4 (conventional mmHg, standard gravity, mechanical
  horsepower via ft·lbf/s), ISO 31-4 (International Table BTU).
* ``CODATA2022`` — CODATA 2022 recommended values (NIST, 2024); for the
  constants SI 2019 fixes, the value is exact by definition.

PSEUDO-DIMENSIONS (the decision the prompt asks to write down):

* ``angle`` is its own base, but SOFT. SI 2019 makes the radian "1" (m/m), and
  following it blindly loses the classic error ``rad/s`` vs ``Hz`` (a factor of
  2*pi). Following numbat (``dimension Angle``) catches it, but turns every
  textbook formula that silently drops the radian (``v = omega*r``) into a
  mismatch. So: angle is tracked, a conversion across it is refused, but an
  equation or sum that differs ONLY in angle is a warning, never REFUTED.
* ``information`` is its own HARD base (as in numbat's ``DigitalInformation``):
  adding bytes to seconds, or to a plain count, is a real error.
* ``pixel`` is its own HARD base (numbat's ``unit pixel`` example): a pixel is a
  count on a raster, not a length. CSS defines ``1px = 1/96 in``, but code that
  adds a pixel offset to a metre is wrong whatever CSS says.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BASES",
    "SOFT_BASES",
    "UnitSpec",
    "UNITS",
    "SI_PREFIXES",
    "BINARY_PREFIXES",
    "REFUSED_NAMES",
    "ConstantSpec",
    "CONSTANTS",
    "SUFFIX_UNITS",
]

#: Base dimensions, in display order. The seven SI bases, then the three
#: pseudo-bases described in the module docstring.
BASES: tuple[str, ...] = (
    "length", "mass", "time", "current", "temperature", "amount",
    "luminous_intensity", "angle", "information", "pixel",
)

#: Bases whose mismatch alone is a warning, not a refutation.
SOFT_BASES: frozenset[str] = frozenset({"angle"})


@dataclass(frozen=True)
class UnitSpec:
    """One named unit.

    ``definition`` is either ``"base:<dimension>"`` for a base unit or a unit
    expression in earlier units. ``prefixes`` is ``""``, ``"si"`` or
    ``"si+binary"``. ``offset`` (absolute temperature scales only) is the SI
    value of zero on this scale: ``T_K = factor * v + offset``.
    """

    name: str
    definition: str
    source: str
    aliases: tuple[str, ...] = ()
    prefixes: str = ""
    offset: "str | None" = None
    note: str = ""


#: SI prefixes (SI2019 + the 2022 CGPM additions ronna/quetta/ronto/quecto).
SI_PREFIXES: dict[str, str] = {
    "Q": "10**30", "R": "10**27", "Y": "10**24", "Z": "10**21", "E": "10**18",
    "P": "10**15", "T": "10**12", "G": "10**9", "M": "10**6", "k": "10**3",
    "h": "10**2", "da": "10", "d": "10**-1", "c": "10**-2", "m": "10**-3",
    "u": "10**-6", "µ": "10**-6", "μ": "10**-6", "n": "10**-9", "p": "10**-12",
    "f": "10**-15", "a": "10**-18", "z": "10**-21", "y": "10**-24",
    "r": "10**-27", "q": "10**-30",
}

#: IEC binary prefixes — only ever applied to bit and byte.
BINARY_PREFIXES: dict[str, str] = {
    "Ki": "2**10", "Mi": "2**20", "Gi": "2**30", "Ti": "2**40",
    "Pi": "2**50", "Ei": "2**60", "Zi": "2**70", "Yi": "2**80",
}

_S = "si"
_SB = "si+binary"

UNITS: tuple[UnitSpec, ...] = (
    # --- SI base units ----------------------------------------------------
    UnitSpec("m", "base:length", "SI2019", ("meter", "metre", "meters", "metres"), _S),
    UnitSpec("kg", "base:mass", "SI2019", ("kilogram", "kilograms")),
    UnitSpec("g", "1/1000 kg", "SI2019", ("gram", "grams"), _S),
    UnitSpec("s", "base:time", "SI2019", ("second", "seconds", "sec", "secs"), _S),
    UnitSpec("A", "base:current", "SI2019", ("ampere", "amperes", "amp", "amps"), _S),
    UnitSpec("K", "base:temperature", "SI2019", ("kelvin",), _S),
    UnitSpec("mol", "base:amount", "SI2019", ("mole", "moles"), _S),
    UnitSpec("cd", "base:luminous_intensity", "SI2019", ("candela",), _S),
    # --- pseudo-bases (see module docstring) --------------------------------
    UnitSpec("rad", "base:angle", "SI2019", ("radian", "radians"), _S),
    UnitSpec("bit", "base:information", "IEC", ("bits",), _SB),
    UnitSpec("px", "base:pixel", "own: a raster count, not a length", ("pixel", "pixels")),
    # --- SI derived units with special names (SI2019 Table 4) ---------------
    UnitSpec("Hz", "1/s", "SI2019", ("hertz",), _S),
    UnitSpec("N", "kg*m/s^2", "SI2019", ("newton", "newtons"), _S),
    UnitSpec("Pa", "N/m^2", "SI2019", ("pascal", "pascals"), _S),
    UnitSpec("J", "N*m", "SI2019", ("joule", "joules"), _S),
    UnitSpec("W", "J/s", "SI2019", ("watt", "watts"), _S),
    UnitSpec("C", "A*s", "SI2019", ("coulomb", "coulombs"), _S),
    UnitSpec("V", "W/A", "SI2019", ("volt", "volts"), _S),
    UnitSpec("F", "C/V", "SI2019", ("farad", "farads"), _S),
    UnitSpec("ohm", "V/A", "SI2019", ("Ω", "ohms"), _S),
    UnitSpec("S", "1/ohm", "SI2019", ("siemens",), _S),
    UnitSpec("Wb", "V*s", "SI2019", ("weber",), _S),
    UnitSpec("T", "Wb/m^2", "SI2019", ("tesla",), _S),
    UnitSpec("H", "Wb/A", "SI2019", ("henry",), _S),
    UnitSpec("sr", "rad^2", "SI2019", ("steradian",), _S),
    UnitSpec("lm", "cd*sr", "SI2019", ("lumen",), _S),
    UnitSpec("lx", "lm/m^2", "SI2019", ("lux",), _S),
    UnitSpec("Bq", "1/s", "SI2019", ("becquerel",), _S),
    UnitSpec("Gy", "J/kg", "SI2019", ("gray",), _S),
    UnitSpec("Sv", "J/kg", "SI2019", ("sievert",), _S),
    UnitSpec("kat", "mol/s", "SI2019", ("katal",), _S),
    # --- non-SI units accepted for use with the SI (SI2019 Table 8) ---------
    UnitSpec("min", "60 s", "SI2019", ("minute", "minutes", "mins")),
    UnitSpec("h", "60 min", "SI2019", ("hour", "hours", "hr", "hrs")),
    UnitSpec("d", "24 h", "SI2019", ("day", "days")),
    UnitSpec("wk", "7 d", "SP811", ("week", "weeks")),
    UnitSpec("yr", "365.25 d", "IAU", ("year", "years", "a_j", "julian_year"),
             note="yr is the Julian year (365.25 d), the astronomical convention; "
                  "a calendar year is 365 or 366 days"),
    UnitSpec("au", "149597870700 m", "IAU", ("AU", "astronomical_unit")),
    UnitSpec("deg", "pi/180 rad", "SI2019", ("°", "degree", "degrees")),
    UnitSpec("arcmin", "1/60 deg", "SI2019", ("arcminute", "arcminutes", "′")),
    UnitSpec("arcsec", "1/60 arcmin", "SI2019", ("arcsecond", "arcseconds", "″"), _S),
    UnitSpec("ha", "10000 m^2", "SI2019", ("hectare", "hectares")),
    UnitSpec("L", "1/1000 m^3", "SI2019", ("l", "liter", "litre", "liters", "litres"), _S),
    UnitSpec("t", "1000 kg", "SI2019", ("tonne", "tonnes")),
    UnitSpec("eV", "1.602176634e-19 J", "SI2019", ("electronvolt", "electronvolts"), _S),
    # --- angles and rotation --------------------------------------------------
    UnitSpec("turn", "2*pi rad", "SP811", ("rev", "revolution", "revolutions", "turns")),
    UnitSpec("gon", "pi/200 rad", "SP811", ("grad", "gradian")),
    UnitSpec("rpm", "rev/min", "SP811"),
    # --- information (IEC 80000-13) -------------------------------------------
    UnitSpec("B", "8 bit", "IEC", ("byte", "bytes", "octet", "octets"), _SB),
    UnitSpec("bps", "bit/s", "IEC", (), _S),
    # --- US customary / imperial, international 1959 definitions (SP811) -----
    UnitSpec("in", "0.0254 m", "SP811", ("inch", "inches")),
    UnitSpec("ft", "12 in", "SP811", ("foot", "feet")),
    UnitSpec("yd", "3 ft", "SP811", ("yard", "yards")),
    UnitSpec("mi", "5280 ft", "SP811", ("mile", "miles")),
    UnitSpec("nmi", "1852 m", "SP811", ("nautical_mile", "nautical_miles")),
    UnitSpec("lb", "0.45359237 kg", "SP811", ("lbs", "pound", "pounds")),
    UnitSpec("oz", "1/16 lb", "SP811", ("ounce", "ounces")),
    UnitSpec("st", "14 lb", "SP811", ("stone",)),
    UnitSpec("gal", "231 in^3", "SP811", ("gallon", "gallons"),
             note="gal is the US liquid gallon (231 in^3); the imperial gallon is 4.54609 L"),
    UnitSpec("mph", "mi/h", "SP811"),
    UnitSpec("kmh", "km/h", "SP811", ("kph",)),
    UnitSpec("kn", "nmi/h", "SP811", ("knot", "knots")),
    UnitSpec("g_n", "9.80665 m/s^2", "ISO4", ("gee",)),
    UnitSpec("lbf", "lb*g_n", "SP811", ("pound_force",)),
    UnitSpec("kgf", "kg*g_n", "SP811", ("kilogram_force",)),
    UnitSpec("psi", "lbf/in^2", "SP811"),
    UnitSpec("atm", "101325 Pa", "SP811", ("atmosphere",)),
    UnitSpec("bar", "100000 Pa", "SI2019", (), _S),
    UnitSpec("Torr", "1/760 atm", "SP811", ("torr",)),
    UnitSpec("mmHg", "13595.1 kg/m^3 * mm * g_n", "ISO4",
             note="mmHg is the conventional millimetre of mercury (133.322387415 Pa)"),
    UnitSpec("cal", "4.184 J", "SP811", ("calorie", "calories"), _S,
             note="cal is the thermochemical calorie (4.184 J); kcal = 4184 J"),
    UnitSpec("Cal", "1000 cal", "SP811", ("food_calorie",),
             note="Cal (capital C) is the food calorie, 1 kcal"),
    UnitSpec("BTU", "1055.05585262 J", "ISO4", ("Btu", "btu")),
    UnitSpec("Wh", "W*h", "SP811", (), _S),
    UnitSpec("hp", "550 ft*lbf/s", "ISO4", ("horsepower",),
             note="hp is the mechanical horsepower (550 ft·lbf/s)"),
    UnitSpec("angstrom", "1/10000000000 m", "SP811", ("\u00c5", "\u212b")),
    UnitSpec("ly", "299792458 m/s * yr", "IAU", ("lightyear", "lightyears", "light_year")),
    UnitSpec("pc", "648000/pi au", "IAU", ("parsec", "parsecs"), _S),
    # --- temperature scales ---------------------------------------------------
    UnitSpec("degC", "K", "SI2019", ("°C", "℃", "celsius"), offset="273.15",
             note="°C is an absolute temperature: 0 °C = 273.15 K"),
    UnitSpec("degF", "5/9 K", "SP811", ("°F", "℉", "fahrenheit"), offset="45967/180",
             note="°F is an absolute temperature: 32 °F = 273.15 K"),
    UnitSpec("degR", "5/9 K", "SP811", ("°R", "rankine")),
    UnitSpec("delta_degC", "K", "SI2019", ("Δ°C", "ΔdegC", "delta_°C")),
    UnitSpec("delta_degF", "5/9 K", "SP811", ("Δ°F", "ΔdegF", "delta_°F")),
    # --- dimensionless ratios ---------------------------------------------------
    UnitSpec("percent", "1/100", "SI2019", ("%", "pct")),
    UnitSpec("ppm", "1/1000000", "SI2019"),
)

#: Names that look like units but are refused, each with the reason. A unit
#: table that silently picked one reading of an ambiguous name would be the
#: kind of quiet wrong answer this package exists to prevent.
REFUSED_NAMES: dict[str, str] = {
    "KB": "'KB' is used for both 1000 B and 1024 B — write kB (1000 B, SI) or KiB (1024 B, IEC)",
    "Kb": "'Kb' is ambiguous — write kbit (1000 bit) or Kibit (1024 bit)",
    "kb": "'kb' is ambiguous (kilobit or kilobyte?) — write kbit or kB",
    "Mb": "'Mb' is ambiguous (megabit or megabyte?) — write Mbit or MB",
    "Gb": "'Gb' is ambiguous (gigabit or gigabyte?) — write Gbit or GB",
    "b": "'b' is ambiguous (bit or barn?) — write bit",
    "dB": "the decibel is a logarithmic ratio, not a multiplicative unit; it is not supported",
    "month": "a month is not a fixed duration — write days (d) or Julian years (yr)",
    "months": "a month is not a fixed duration — write days (d) or Julian years (yr)",
    "ton": "'ton' is ambiguous (metric tonne, short ton, long ton) — write t for the tonne",
    "tons": "'ton' is ambiguous (metric tonne, short ton, long ton) — write t for the tonne",
    "Gal": "'Gal' is the galileo (cm/s^2), easily confused with the gallon — write cm/s^2 or gal",
    "pt": "'pt' is ambiguous (pint or typographic point?) — not supported",
    "c": "'c' is the speed of light (a constant), not a unit — use the constant c",
}


@dataclass(frozen=True)
class ConstantSpec:
    """A physical constant: value in the SI unit given, exact or with its standard
    uncertainty (absolute, same unit)."""

    name: str
    value: str
    unit: str
    exact: bool
    uncertainty: "str | None"
    source: str
    description: str


#: The seven defining constants SI 2019 fixes are exact; every other entry is a
#: CODATA 2022 measured value with its uncertainty (cross-checked 30.10.2026
#: against scipy.constants 1.17.1, which ships CODATA 2022). Derived constants
#: that are exact products of exact ones (hbar, R, F, sigma) are exact too.
CONSTANTS: tuple[ConstantSpec, ...] = (
    ConstantSpec("c", "299792458", "m/s", True, None, "SI2019", "speed of light in vacuum"),
    ConstantSpec("h_planck", "6.62607015e-34", "J*s", True, None, "SI2019", "Planck constant"),
    ConstantSpec("hbar", "6.62607015e-34/(2*pi)", "J*s", True, None, "SI2019",
                 "reduced Planck constant h/(2*pi)"),
    ConstantSpec("e_charge", "1.602176634e-19", "C", True, None, "SI2019", "elementary charge"),
    ConstantSpec("k_B", "1.380649e-23", "J/K", True, None, "SI2019", "Boltzmann constant"),
    ConstantSpec("N_A", "6.02214076e23", "1/mol", True, None, "SI2019", "Avogadro constant"),
    ConstantSpec("delta_nu_Cs", "9192631770", "Hz", True, None, "SI2019",
                 "caesium-133 hyperfine transition frequency"),
    ConstantSpec("K_cd", "683", "lm/W", True, None, "SI2019", "luminous efficacy of 540 THz radiation"),
    ConstantSpec("R_gas", "6.02214076e23*1.380649e-23", "J/(mol*K)", True, None, "SI2019",
                 "molar gas constant N_A*k_B"),
    ConstantSpec("F_faraday", "6.02214076e23*1.602176634e-19", "C/mol", True, None, "SI2019",
                 "Faraday constant N_A*e"),
    ConstantSpec("sigma_SB", "2*pi**5*(1.380649e-23)**4/(15*(6.62607015e-34)**3*299792458**2)",
                 "W/(m^2*K^4)", True, None, "SI2019", "Stefan-Boltzmann constant"),
    ConstantSpec("g_n", "9.80665", "m/s^2", True, None, "ISO4", "standard acceleration of gravity"),
    ConstantSpec("G", "6.67430e-11", "m^3/(kg*s^2)", False, "0.00015e-11", "CODATA2022",
                 "Newtonian constant of gravitation"),
    ConstantSpec("m_e", "9.1093837139e-31", "kg", False, "0.0000000028e-31", "CODATA2022",
                 "electron mass"),
    ConstantSpec("m_p", "1.67262192595e-27", "kg", False, "0.00000000052e-27", "CODATA2022",
                 "proton mass"),
    ConstantSpec("m_u", "1.66053906892e-27", "kg", False, "0.00000000052e-27", "CODATA2022",
                 "atomic mass constant (dalton)"),
    ConstantSpec("alpha_fs", "7.2973525643e-3", "1", False, "0.0000000011e-3", "CODATA2022",
                 "fine-structure constant"),
    ConstantSpec("mu_0", "1.25663706127e-6", "N/A^2", False, "0.00000000020e-6", "CODATA2022",
                 "vacuum magnetic permeability (measured since SI 2019, no longer 4*pi*1e-7)"),
    ConstantSpec("epsilon_0", "8.8541878188e-12", "F/m", False, "0.0000000014e-12", "CODATA2022",
                 "vacuum electric permittivity"),
)

#: Default suffix convention for unit-carrying variable names (``units_from_names``).
#: Each suffix maps to a unit expression. Deliberately ABSENT, because each is a
#: common non-unit suffix and would turn a name into a false accusation:
#: ``_n`` (count), ``_a`` / ``_b`` (labels), ``_l`` / ``_r`` (left/right),
#: ``_in`` (input), ``_h`` (height), ``_g`` (green / group), ``_t`` (type / time
#: index), ``_c`` (count / constant), ``_v`` (vector), ``_x`` / ``_y``, ``_kb``
#: (ambiguous), ``_p`` (probability / pressure), ``_d`` (diameter / delta).
SUFFIX_UNITS: dict[str, str] = {
    # length
    "m": "m", "meters": "m", "metres": "m", "km": "km", "cm": "cm", "mm": "mm",
    "um": "um", "ft": "ft", "feet": "ft", "inch": "in", "inches": "in", "mi": "mi",
    "miles": "mi", "nmi": "nmi",
    # time
    "s": "s", "sec": "s", "secs": "s", "seconds": "s", "ms": "ms", "msec": "ms",
    "millis": "ms", "us": "us", "usec": "us", "micros": "us", "ns": "ns", "nanos": "ns",
    "min": "min", "mins": "min", "minutes": "min", "hr": "h", "hrs": "h", "hours": "h",
    "days": "d",
    # mass
    "kg": "kg", "mg": "mg", "lb": "lb", "lbs": "lb", "grams": "g",
    # speed and frequency
    "kmh": "km/h", "kph": "km/h", "mps": "m/s", "mph": "mph", "knots": "kn",
    "hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz", "rpm": "rpm",
    # angle
    "deg": "deg", "degs": "deg", "degrees": "deg", "rad": "rad", "rads": "rad",
    "radians": "rad",
    # raster
    "px": "px", "pixels": "px",
    # information
    "bytes": "B", "bits": "bit", "kib": "KiB", "mib": "MiB", "gib": "GiB",
    "kbytes": "kB", "mbytes": "MB",
    # temperature
    "kelvin": "K", "degc": "degC", "celsius": "degC", "degf": "degF", "fahrenheit": "degF",
    # mechanics / electricity / energy
    "pa": "Pa", "kpa": "kPa", "mpa": "MPa", "hpa": "hPa", "psi": "psi", "bar": "bar",
    "newtons": "N", "kn": "kN", "joules": "J", "kj": "kJ", "kwh": "kWh", "kcal": "kcal",
    "ev": "eV", "kw": "kW", "mw": "mW", "watts": "W", "volts": "V", "mv": "mV",
    "ohm": "ohm", "ohms": "ohm", "amps": "A", "ma": "mA",
    # volume / area
    "ml": "mL", "liters": "L", "litres": "L", "m2": "m^2", "m3": "m^3",
}
