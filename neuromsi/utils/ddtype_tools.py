#!/usr/bin/env python
# -*- coding: utf-8 -*-


# =============================================================================
# DOCS
# =============================================================================

"""Tools to inspect and convert the data types of (nested) objects."""


# =============================================================================
# IMPORTS
# =============================================================================

import sys
from collections.abc import Mapping
from numbers import Number
from typing import NamedTuple

import numpy as np

from .bunch import Bunch


# =============================================================================
# CONSTANTS
# =============================================================================

#: Types that carry a single value, so a single dtype describes them.
SINGLE_DTYPE_TYPES = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
    np.bool_,
    np.number,
)


def _numpy_dtype_names():
    """Return the names of the numpy scalar types, e.g. 'float64'."""
    try:
        return frozenset(
            t.__name__ for t in np.sctypeDict.values() if isinstance(t, type)
        )
    except Exception:  # pragma: no cover - depends on the numpy version
        return frozenset()


#: Names of the types in SINGLE_DTYPE_TYPES, plus the numpy containers and
#: scalars, all of which are described by a single dtype.
SINGLE_DTYPE_NAMES = frozenset(
    t.__name__ for t in SINGLE_DTYPE_TYPES
) | frozenset({"ndarray"}) | _numpy_dtype_names()

#: Units used to render a memory size in a human readable way.
_SIZE_UNITS = ("B", "kB", "MB", "GB", "TB")


# =============================================================================
# OBJECT INFO
# =============================================================================


class SizeInfo(NamedTuple):
    """Memory size of an object.

    Attributes
    ----------
    nbytes : int
        The memory size of the object, in bytes.
    hsize : str
        The memory size of the object in a human readable format.

    """

    nbytes: int
    hsize: str


class ObjInfo(NamedTuple):
    """Description of a single object.

    Attributes
    ----------
    type : str
        The name of the type of the object.
    dtype : object
        The data type of the object, when it has a meaningful one.
    size : int
        The number of elements of the object.
    memory : SizeInfo
        The memory size of the object.

    """

    type: str
    dtype: object
    size: int
    memory: SizeInfo


# =============================================================================
# HELPERS
# =============================================================================


def single_dtype_class(obj_type):
    """Whether a type is described by a single data type.

    Parameters
    ----------
    obj_type : type or str
        The type, or the name of the type, to check.

    Returns
    -------
    bool
        True if a single data type describes the type.

    """
    if isinstance(obj_type, str):
        return obj_type in SINGLE_DTYPE_NAMES
    if not isinstance(obj_type, type):
        return False
    return issubclass(obj_type, SINGLE_DTYPE_TYPES) or issubclass(
        obj_type, (np.ndarray, np.generic)
    )


def _dtype_of(obj):
    """Return the data type of an object, or the name of its type."""
    if isinstance(obj, (str, bytes)) or obj is None:
        return type(obj).__name__
    dtype = getattr(obj, "dtype", None)
    return type(obj).__name__ if dtype is None else dtype


def _size_of(obj):
    """Return the number of elements of an object."""
    return int(getattr(obj, "size", 1))


def _nbytes_of(obj):
    """Return the memory size of an object, in bytes."""
    nbytes = getattr(obj, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if hasattr(obj, "__sizeof__"):
        return int(obj.__sizeof__())
    return int(sys.getsizeof(obj))


def _hsize(nbytes):
    """Return a memory size in bytes as a human readable string."""
    for unit in _SIZE_UNITS:
        if abs(nbytes) < 1024 or unit == _SIZE_UNITS[-1]:
            fmt = "{:.0f}" if unit == "B" else "{:.1f}"
            return f"{fmt.format(nbytes)} {unit}"
        nbytes /= 1024


# =============================================================================
# CAST
# =============================================================================


def deep_astype(obj, dtype):
    """Cast an object to a data type, recursively.

    Containers are traversed and rebuilt, so the returned object has the same
    structure than the input one. Types that cannot be cast (e.g. str) are
    returned untouched.

    Parameters
    ----------
    obj : object
        The object to cast.
    dtype : data type
        The data type to cast the object to.

    Returns
    -------
    object
        The object cast to the data type.

    """
    if dtype is None:
        return obj

    if isinstance(obj, Bunch):
        return Bunch(
            obj._name, {k: deep_astype(v, dtype) for k, v in obj.items()}
        )

    if isinstance(obj, Mapping):
        return {k: deep_astype(v, dtype) for k, v in obj.items()}

    if isinstance(obj, list):
        return [deep_astype(v, dtype) for v in obj]

    if isinstance(obj, tuple):
        return tuple(deep_astype(v, dtype) for v in obj)

    if isinstance(obj, (set, frozenset)):
        return type(obj)(deep_astype(v, dtype) for v in obj)

    if hasattr(obj, "astype"):  # numpy arrays/scalars, xarray objects
        return obj.astype(dtype)

    if isinstance(obj, Number):
        return np.asarray(obj).astype(dtype).item()

    return obj


# =============================================================================
# DESCRIBE
# =============================================================================


def _describe(obj, *, name, max_deep):
    """Describe an object and, up to 'max_deep', its items."""
    info = {
        name: ObjInfo(
            type=type(obj).__name__,
            dtype=_dtype_of(obj),
            size=_size_of(obj),
            memory=SizeInfo(
                nbytes=_nbytes_of(obj), hsize=_hsize(_nbytes_of(obj))
            ),
        )
    }

    if max_deep > 1 and isinstance(obj, Mapping):
        for key, value in obj.items():
            info.update(
                _describe(value, name=str(key), max_deep=max_deep - 1)
            )

    return info


def deep_dtypes(obj, *, root="root", max_deep=2, memory_usage=False):
    """Describe the data types of an object.

    Parameters
    ----------
    obj : object
        The object to describe.
    root : str, optional
        The name given to the object itself.
    max_deep : int, optional
        The maximum depth of the description. Only meaningful for mappings.
    memory_usage : bool, optional
        Kept for signature compatibility, the memory size is always computed.

    Returns
    -------
    dict
        A dict mapping 'root' to a tuple with the name of the object and a
        mapping from attribute names to their ObjInfo.

    """
    info = _describe(obj, name=root, max_deep=max_deep)
    return {root: (root, info)}
