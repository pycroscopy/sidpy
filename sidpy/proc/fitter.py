"""
Compatibility wrapper for the legacy ``sidpy.proc.fitter`` import path.

The old implementation has been retired in favor of :class:`sidpy.proc.fitter_refactor.SidpyFitter`.
"""

from .fitter_refactor import SidpyFitter


class SidFitter(SidpyFitter):
    """Backward-compatible alias of :class:`SidpyFitter`."""

    pass


__all__ = ["SidFitter", "SidpyFitter"]
