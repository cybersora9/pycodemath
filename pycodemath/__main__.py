"""Entry point: ``python -m pycodemath`` — REPL or one-shot from arguments.

Forcing UTF-8 for the Windows console is done by ``run()`` (shared with the
``pycodemath`` console script).
"""

from .cli.repl import run

if __name__ == "__main__":
    run()
