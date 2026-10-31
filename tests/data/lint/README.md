# Evaluation set of `pycodemath lint` (module L1)

Every `.py` file here is one case. A line that must carry a finding ends with
`# expect PCMnnn` (several codes: `# expect PCM401 PCM503`); every other line must
carry none. A file may start with `# select: PCM203` to enable rules that are off
by default. `# source:` names where a case comes from.

* `dev/` — looked at while developing the rules.
* `held/` — written at the same time as `dev/`, before the rules were tuned, and run
  only for the final measurement (`dev-journal/full/L1.md`, section 6).

Licences: the cases are written for this repository (MIT). Where a case is modelled
on a published example, the `# source:` line cites it; formulas and facts are not
copyrightable, and no text is copied: Python tutorial "Floating Point Arithmetic:
Issues and Limitations" (PSF), Python docs of `round()` (PSF), FPBench / Herbie
benchmark formulas (MIT, FPBench Project Contributors), J. Bloch, "Extra, Extra -
Read All About It: Nearly All Binary Searches and Mergesorts are Broken" (Google
Research blog, 2006).

`sarif-schema-2.1.0.json` is the OASIS SARIF 2.1.0 JSON schema (errata01), from
https://github.com/oasis-tcs/sarif-spec (sarif-2.1/schema/), distributed under the
OASIS IPR policy that lets the standard be copied and implemented; used only to
validate our output in a test.
