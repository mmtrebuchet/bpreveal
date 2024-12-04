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
from bpreveal.internal.constants import RGB_T, DNA_COLOR_SPEC_T, COLOR_SPEC_T, \
    setDefaultFontFamily


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


def getGraphCmap(minValue: float, colorSpan: float,
                 baseCmap: mplcolors.Colormap) -> mplcolors.Colormap:
    """Remove the central portion of a color map for PISA graphs.

    :param minValue: The value (in logit space) below which no lines will
        be drawn in the PISA graph. This is the region of the returned color
        map that will be white.
    :param colorSpan: The total range of the color map. Lines with higher PISA
        scores than colorSpan will be clipped to the extreme values of the color
        map.
    :param baseCmap: The colormap to modify. This will be either ``pisaClip`` or
        ``pisaNoClip``, or you may provide your own Colormap object.
    :return: A new color map that can be used to color lines in a PISA
        graph.
    """
    sampledCmap = baseCmap.resampled(256)
    colorList = sampledCmap(np.linspace(0, 1, 256))
    # The range for a color map must be from 0 to 1
    # but the minValue and color span parameters will be centered
    # at zero, not 1/2.
    # If the resampled color map is:
    #         a b c d e f g h i j k l m o p q r s t u v w x y z
    #         |                 |     |     |                 |
    #     -colorSpan       -minValue  0  +minValue         +colorSpan
    #         0                      0.5                      1
    # where the top line of labels is in the zero-centered space we think in
    # then the bottom line is the points on the 0 to 1 cmap object.

    lowerBound = (colorSpan - minValue) / (2 * colorSpan)
    upperBound = (colorSpan + minValue) / (2 * colorSpan)
    white = np.array([1, 1, 1, 1])
    sliceStart = int(lowerBound * 256)
    sliceEnd = int(upperBound * 256)
    colorList[sliceStart:sliceEnd] = white
    return mplcolors.ListedColormap(colorList)


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
            raise ValueError(f"Invalid color spec: {colorSpec}")


def loadFonts(serif: bool = True) -> None:
    """Configure the matplotlib default fonts to be in the Libertinus family.

    :param serif: Should a serif font be used? The default is True, and this will use the
        Libertinus font family. If False, then instead the Fira Sans font family will be
        used.

    This places the given font (Libertinus or Fira Sans) at the top of the
    order for serif and sans-serif fontfamily, but does not overwrite the monospace family.
    It also sets the pdf backend to save text boxes rather than individual letters so that
    text can be easily edited later.
    """
    try:
        cwd = pathlib.Path(__file__).parent.parent.parent.resolve()
        fontdirLibertinus = str(cwd / "doc" / "fonts" / "Libertinus-7.040" / "static/") + "/"
        fontdirSans = str(cwd / "doc" / "fonts" / "Fira_Sans/") + "/"
        fontFiles = matplotlib.font_manager.findSystemFonts(
            fontpaths=[fontdirLibertinus, fontdirSans])
        for ff in fontFiles:  # pylint: disable=not-an-iterable
            try:
                matplotlib.font_manager.fontManager.addfont(ff)
            except:  # pylint: disable=bare-except  # noqa
                # Whatever happened, it's just going to alter fonts a bit.
                pass

        oldFaces = plt.rcParams["font.serif"]
        plt.rcParams["font.serif"] = ["Libertinus Serif"] + oldFaces
        oldFaces = plt.rcParams["font.sans-serif"]
        plt.rcParams["font.sans-serif"] = ["Fira Sans"] + oldFaces
        if serif:
            setDefaultFontFamily("serif")
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "custom"
            plt.rcParams["mathtext.rm"] = "Libertinus Serif"
            plt.rcParams["mathtext.it"] = "Libertinus Serif:italic"
            plt.rcParams["mathtext.bf"] = "Libertinus Serif:bold"
            plt.rcParams["mathtext.cal"] = "Libertinus Serif:italic"
        else:
            setDefaultFontFamily("sans-serif")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["mathtext.fontset"] = "custom"
            plt.rcParams["mathtext.rm"] = "Fira Sans"
            plt.rcParams["mathtext.it"] = "Fira Sans:italic"
            plt.rcParams["mathtext.bf"] = "Fira Sans:bold"
            plt.rcParams["mathtext.cal"] = "Fira Sans:italic"
        # Set the pdf output to use text boxes, rather than drawing each character
        # as a shape on its own.
        plt.rcParams["pdf.fonttype"] = 42
        logUtils.debug("Configured matplotlib default fonts.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # If it didn't work, that's okay. User will still get good enough fonts.
        logUtils.warning(f"Failed to set font family because {e}")


# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
