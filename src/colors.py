"""Utilities for manipulating colors.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/colors.bnf


"""
from typing import TypeAlias, Literal, TypedDict
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplcolors


COLOR_SPEC_T: TypeAlias = \
    dict[Literal["rgb"], tuple[float, float, float]] | \
    dict[Literal["rgba"], tuple[float, float, float, float]] | \
    dict[Literal["tol"], int] | \
    dict[Literal["tol-light"], int] | \
    dict[Literal["ibm"], int] | \
    dict[Literal["wong"], int] | \
    tuple[float, float, float] | \
    tuple[float, float, float, float]
"""A COLOR_SPEC_T is anything that parseSpec can turn into an rgb or rgba triple.

It may be one of the following things:

1. ``{"rgb": (0.1, 0.2, 0.3)}``
    giving an rgb triple.
2. ``{"rgba": (0.1, 0.2, 0.3, 0.5)}``
    giving an rgb triple with an alpha value.
3. ``{"tol": 1}``
    giving a numbered color from the tol palette.
    Valid numbers are 0 to 7.
4. ``{"tol-light": 1}``
    giving a numbered color from the tolLight palette.
    Valid numbers are 0 to 6.
5. ``{"wong": 1}``
    giving a numbered color from the wong palette.
    Valid numbers are 0 to 7.
6. ``{"ibm": 1}``
    giving a numbered color from the ibm palette.
    Valid numbers are 0 to 4.
7. ``(0.1, 0.2, 0.3)``
    giving an rgb triple.
8. ``(0.1, 0.2, 0.3)``
    giving an rgb triple with an alpha value.
"""


# pylint: disable=invalid-name
class DNA_COLOR_SPEC_T(TypedDict):
    """A type that assigns a color to each of the four bases.

    It is a dictionary mapping the bases onto colorSpecs, like this::

        {"A": {"wong": 3}, "C": {"wong": 5},
         "G": {"wong": 4}, "T": {"wong": 6}}
    """

    A: COLOR_SPEC_T
    C: COLOR_SPEC_T
    G: COLOR_SPEC_T
    T: COLOR_SPEC_T
# pylint: enable=invalid-name


RGB_T: TypeAlias = \
    tuple[float, float, float] | \
    tuple[float, float, float, float]
"""An rgb or rgba triple."""


def _toFractions(colorList: tuple[tuple[int, int, int], ...]) -> tuple[RGB_T, ...]:
    ret = ()
    for c in colorList:
        r, g, b = c
        ret = ret + ((r / 256, g / 256, b / 256),)
    return ret


wongRgb = ((0, 0, 0),
           (230, 159, 0),
           (86, 180, 233),
           (0, 158, 115),
           (240, 228, 66),
           (0, 114, 178),
           (213, 94, 0),
           (204, 121, 167))

ibmRgb = ((100, 143, 255),
          (120, 94, 240),
          (220, 38, 127),
          (254, 97, 0),
          (255, 176, 0))

tolRgb = ((51, 34, 136),
          (136, 204, 238),
          (68, 170, 153),
          (17, 119, 51),
          (153, 153, 51),
          (221, 204, 119),
          (204, 102, 119),
          (136, 34, 85),
          (170, 68, 153),
          (221, 221, 221))

tolLightRgb = ((153, 221, 255),
               (68, 187, 153),
               (187, 204, 51),
               (238, 221, 136),
               (238, 136, 102),
               (255, 170, 187),
               (221, 221, 221))

tol: tuple[RGB_T, ...] = _toFractions(tolRgb)
"""The muted color scheme by Paul Tol.

.. image:: ../../doc/presentations/tol.png

"""

ibm: tuple[RGB_T, ...] = _toFractions(ibmRgb)
"""The IBM design library's color blind palette.

.. image:: ../../doc/presentations/ibm.png

"""

wong: tuple[RGB_T, ...] = _toFractions(wongRgb)
"""Bang Wong's palette. Used for DNA.

.. image:: ../../doc/presentations/wong.png

"""

tolLight: tuple[RGB_T, ...] = _toFractions(tolLightRgb)
"""The light color scheme by Paul Tol, but with light blue and olive deleted.

.. image:: ../../doc/presentations/tolLight.png

"""

dnaWong: DNA_COLOR_SPEC_T = {"A": {"wong": 3}, "C": {"wong": 5},
                             "G": {"wong": 4}, "T": {"wong": 6}}
"""The default color map for DNA bases.

A is green, C is blue, G is yellow, and T is red.
"""

defaultProfile: COLOR_SPEC_T = {"tol": 0}
"""The default color for profile plots. It's tol[0]."""


_oldPisaCmap = mpl.colormaps["RdBu_r"].resampled(256)
_newPisaColors = _oldPisaCmap(np.linspace(0, 1, 256))
_pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
_green = np.array([24 / 256, 248 / 256, 148 / 256, 1])
_newPisaColors[:5] = _green
_newPisaColors[-5:] = _pink

pisaClip = mplcolors.ListedColormap(_newPisaColors)
"""The color map used for PISA plots, including green and pink clipping colors.

.. image:: ../../doc/presentations/pisaClip.png

"""

pisaNoClip = mplcolors.ListedColormap(_oldPisaCmap(np.linspace(0, 1, 256)))
"""The color map for PISA plots but without the clipping warning colors.

.. image:: ../../doc/presentations/pisaNoClip.png

"""


def parseSpec(  # pylint: disable=too-many-return-statements
        colorSpec: COLOR_SPEC_T) -> RGB_T:
    """Given a color-spec (See the BNF), convert it into an rgb or rgba tuple.

    :param colorSpec: The color specification.

    If colorSpec is a 3-tuple, it is interpreted as an rgb color. If it is a
    4-tuple, it is interpreted as rgba. If it is a dictionary containing
    ``{"rgb": (0.1, 0.2, 0.3)}`` then it is interpreted as an rgb color, and
    it's the same story if the dictionary has structure ``{"rgba": (0.1, 0.2,
    0.3, 0.8)}``. If it is a dictionary with a key naming a palette (one of
    "tol", "tol-light", "wong", or "ibm") and an integer value, then the value
    is the ith color of the corresponding palette.
    """
    match colorSpec:
        case(r, g, b):
            return (r, g, b)
        case(r, b, g, a):
            return (r, g, b, a)
        case {"rgb": (r, g, b)}:
            return (r, g, b)
        case {"rgba": (r, g, b, a)}:
            return (r, g, b, a)
        case {"tol": num}:
            return tol[num]
        case {"tol-light": num}:
            return tolLight[num]
        case {"ibm": num}:
            return ibm[num]
        case {"wong": num}:
            return wong[num]
        case _:
            assert False, f"Invalid color spec: {colorSpec}"

# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa