#!/usr/bin/env python
# -*- coding: utf-8 -*-


# =============================================================================
# DOCS
# =============================================================================

"""Utilities shared across the neuromsi package.

Two things live here: the small data structures used by the result classes
(:class:`Bunch` and :class:`AccessorABC`) and :mod:`ddtype_tools`, which
backs the ``astype``/``dtypes`` methods of NDResult.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from . import ddtype_tools
from .bunch import Bunch


# =============================================================================
# ACCESSOR ABC
# =============================================================================


class AccessorABC:
    """Base class of the accessors attached to a result object.

    Accessors hang off their host object through a property, e.g.
    ``ndresult.plot`` and ``ndresult.stats``.

    Parameters
    ----------
    result : NDResult
        The object the accessor gives access to.

    """

    def __init__(self, result):
        self._result = result

    def __repr__(self):
        """Return a string representation of the accessor."""
        cls_name = type(self).__name__
        host_name = type(self._result).__name__
        return f"<{cls_name} of {host_name}>"


# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "AccessorABC",
    "Bunch",
    "ddtype_tools",
]
