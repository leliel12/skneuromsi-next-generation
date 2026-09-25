#!/usr/bin/env python
# -*- coding: utf-8 -*-


# =============================================================================
# DOCS
# =============================================================================

"""A dict that also exposes its keys as attributes."""


# =============================================================================
# BUNCH
# =============================================================================


class Bunch(dict):
    """A dictionary whose keys are also reachable as attributes.

    Parameters
    ----------
    name : str
        The name of the bunch, only used for representation.
    mapping : dict, optional
        The initial content of the bunch.
    **kwargs
        Extra key-value pairs added to the bunch.

    Examples
    --------
    >>> params = Bunch("run_parameters", {"time_res": 0.01})
    >>> params.time_res
    0.01

    """

    def __init__(self, name, mapping=None, **kwargs):
        super().__init__(mapping or {}, **kwargs)
        self._name = str(name)

    def __getattr__(self, key):
        """Return the value of ``key``, or raise AttributeError."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        """Store non-private attributes as items of the bunch."""
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delattr__(self, key):
        """Delete the item ``key`` from the bunch."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def to_dict(self):
        """Return a plain dict copy of the bunch."""
        return dict(self)

    def __repr__(self):
        """Return a string representation of the bunch."""
        cls_name = type(self).__name__
        return f"{cls_name}('{self._name}', {dict.__repr__(self)})"
