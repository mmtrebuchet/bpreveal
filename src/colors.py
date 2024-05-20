"""Utilities for manipulating colors.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/colors.bnf


"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplcolors
import matplotlib.font_manager
import matplotlib.pyplot as plt
from bpreveal import logUtils
from bpreveal.internal.constants import RGB_T, DNA_COLOR_SPEC_T, COLOR_SPEC_T


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

:type: tuple[:py:class:`RGB_T<bpreveal.internal.constants.RGB_T>`, ...]

:meta hide-value:
"""

ibm: tuple[RGB_T, ...] = _toFractions(ibmRgb)
"""The IBM design library's color blind palette.

.. image:: ../../doc/presentations/ibm.png

:type: tuple[:py:class:`RGB_T<bpreveal.internal.constants.RGB_T>`, ...]

:meta hide-value:
"""

wong: tuple[RGB_T, ...] = _toFractions(wongRgb)
"""Bang Wong's palette. Used for DNA.

.. image:: ../../doc/presentations/wong.png

:type: tuple[:py:class:`RGB_T<bpreveal.internal.constants.RGB_T>`, ...]

:meta hide-value:
"""

tolLight: tuple[RGB_T, ...] = _toFractions(tolLightRgb)
"""The light color scheme by Paul Tol, but with light blue and olive deleted.

.. image:: ../../doc/presentations/tolLight.png

:type: tuple[:py:class:`RGB_T<bpreveal.internal.constants.RGB_T>`, ...]

:meta hide-value:
"""

dnaWong: DNA_COLOR_SPEC_T = {"A": {"wong": 3}, "C": {"wong": 5},
                             "G": {"wong": 4}, "T": {"wong": 6}}
"""The default color map for DNA bases.

A is green, C is blue, G is yellow, and T is red.
"""

defaultProfile: COLOR_SPEC_T = {"tol": 0}
"""The default color for profile plots. It's tol[0].

:type: :py:class:`COLOR_SPEC_T<bpreveal.internal.constants.COLOR_SPEC_T>`

:meta hide-value:
"""


_oldPisaCmap = mpl.colormaps["RdBu_r"].resampled(256)  # pylint: disable=unsubscriptable-object
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
        colorSpec: COLOR_SPEC_T | str) -> RGB_T:
    """Given a color-spec (See the BNF), convert it into an rgb or rgba tuple.

    :param colorSpec: The color specification.
    :type colorSpec: :py:data:`COLOR_SPEC_T<bpreveal.internal.constants.COLOR_SPEC_T>` | str
    :return: An rgb triple (or rgba quadruple).

    Based on the shape of colorSpec, this function interprets it differently.
    If colorSpec is a...

    3-tuple
        it is interpreted as an rgb color,

    4-tuple
        it is interpreted as rgba,

    ``{"rgb": (0.1, 0.2, 0.3)}``
        it is interpreted as an rgb color,

    ``{"rgba": (0.1, 0.2, 0.3, 0.8)}``
        it is interpreted as an rgba color,

    ``{"<palette-name>": 3}``
        where ``<palette-name>`` is one of "tol", "tol-light",
        "wong", or "ibm"), then the value is the color ``i`` in
        the corresponding palette.

    ``"b"``
        or any other string, then it is interpreted as a matplotlib
        color string and is passed to ``matplotlib.colors.to_rgb``.

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
        case str():
            return mplcolors.to_rgb(colorSpec)
        case _:
            assert False, f"Invalid color spec: {colorSpec}"


def loadFonts():
    """Configures the matplotlib default fonts to be in the Libertinus family.

    This places Libertinus fonts at the top of the order for serif and sans-serif
    fontfamily, but does not overwrite the monospace family.
    """
    try:
        cwd = pathlib.Path(__file__).parent.parent.parent.resolve()
        fontdir = str(cwd / "doc" / "fonts" / "Libertinus-7.040" / "static/") + "/"
        fontFiles = matplotlib.font_manager.findSystemFonts(fontpaths=[fontdir])
        for ff in fontFiles:  # pylint: disable=not-an-iterable
            try:
                matplotlib.font_manager.fontManager.addfont(ff)
            except:  # pylint: disable=bare-except  # noqa
                # Whatever happened, it's just going to alter fonts a bit.
                pass
        oldFaces = plt.rcParams["font.serif"]
        plt.rcParams["font.serif"] = ["Libertinus Serif"] + oldFaces
        oldFaces = plt.rcParams["font.sans-serif"]
        plt.rcParams["font.sans-serif"] = ["Libertinus Sans"] + oldFaces
        plt.rcParams["font.family"] = "serif"
        logUtils.debug("Configured matplotlib default fonts.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # If it didn't work, that's okay. User will still get good enough fonts.
        logUtils.warning(f"Failed to set font family because {e}")


# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
