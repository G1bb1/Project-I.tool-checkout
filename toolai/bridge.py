"""Method A and the lighting (retake) check, taken from YOUR toolcheck package so the
hybrid uses exactly the same numbers as your trials.

toolcheck should have the Nov 17 retake patch (features.py with gap_mask and
lighting_change). Whatever can't be imported from toolcheck falls back to a copy
of the Nov 17 code in _toolcheck_copy.py, and SOURCE says which was used (it is
written into every session/replay so the logbook shows it).
"""
SOURCE = {}
try:
    from toolcheck.methods import method_a                        # noqa: F401
    SOURCE["method_a"] = "toolcheck"
except ImportError:
    from ._toolcheck_copy import method_a                          # noqa: F401
    SOURCE["method_a"] = "built-in copy (toolcheck not importable)"
try:
    from toolcheck.features import gap_mask, lighting_change      # noqa: F401
    SOURCE["lighting"] = "toolcheck"
except ImportError:
    from ._toolcheck_copy import gap_mask, lighting_change         # noqa: F401
    SOURCE["lighting"] = "built-in copy of the Nov 17 patch (toolcheck has no retake patch)"
