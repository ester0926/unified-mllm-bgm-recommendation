"""Shared evaluation entrypoint for scripts that import ``evaluate``.

The original experiment scripts import evaluation helpers as ``from evaluate
import ...``. In this release layout the implementation lives in
``scripts/diagnostics/evaluate.py``, so this module preserves the original
import style while keeping one canonical implementation.
"""

from scripts.diagnostics.evaluate import *  # noqa: F401,F403
