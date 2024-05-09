"""Utilities for manipulating colors."""
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


class DNA_COLOR_SPEC_T(TypedDict):
    """A type that assigns a color to each of the four bases."""
    A: COLOR_SPEC_T
    C: COLOR_SPEC_T
    G: COLOR_SPEC_T
    T: COLOR_SPEC_T


RGB_T: TypeAlias = \
    tuple[float, float, float] | \
    tuple[float, float, float, float]


def _toFractions(colorList: tuple[tuple[int, int, int], ...]) -> tuple[RGB_T, ...]:
    ret = ()
    for c in colorList:
        r, g, b = c
        ret = ret + ((r / 256, g / 256, b / 256),)
    return ret


class ColorMaps:
    """The color maps used in BPReveal. These are colorblind-friendly."""

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
    """The muted color scheme by Paul Tol."""
    tol: tuple[RGB_T, ...] = _toFractions(tolRgb)

    """The IBM design library's color blind palette."""
    ibm: tuple[RGB_T, ...] = _toFractions(ibmRgb)

    """Bang Wong's palette. Used for DNA."""
    wong: tuple[RGB_T, ...] = _toFractions(wongRgb)

    """The light color scheme by Paul Tol, but with light blue and olive deleted."""
    tolLight: tuple[RGB_T, ...] = _toFractions(tolLightRgb)

    """The default color map for DNA bases.

    A is green, C is blue, G is yellow, and T is red.
    """
    dnaWong: DNA_COLOR_SPEC_T = {"A": {"wong": 3}, "C": {"wong": 5},
                              "G": {"wong": 4}, "T": {"wong": 6}}

    """The default color for profile plots."""
    defaultProfile: COLOR_SPEC_T = {"tol": 0}

    _oldPisaCmap = mpl.colormaps["RdBu_r"].resampled(256)
    _newPisaColors = _oldPisaCmap(np.linspace(0, 1, 256))
    _pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    _green = np.array([24 / 256, 248 / 256, 148 / 256, 1])
    _newPisaColors[:5] = _green
    _newPisaColors[-5:] = _pink

    """The color map used for PISA plots, including green and pink clipping colors."""
    pisaClip = mplcolors.ListedColormap(_newPisaColors)

    """The color map for PISA plots but without the clipping warning colors."""
    pisaNoClip = mplcolors.ListedColormap(_oldPisaCmap(np.linspace(0, 1, 256)))

    @classmethod
    def parseSpec(cls,  # pylint: disable=too-many-return-statements
                  colorSpec: COLOR_SPEC_T) -> RGB_T:
        """Given a color-spec (See the BNF), convert it into an rgb or rgba tuple.

        :param colorSpec: The color specification.

        If colorSpec is a 3-tuple, it is interpreted as an rgb color.
        If it is a 4-tuple, it is interpreted as rgba.
        If it is a dictionary containing ``{"rgb": (0.1, 0.2, 0.3)}``
        then it is interpreted as an rgb color, and it's the same story if
        the dictionary has structure ``{"rgba": (0.1, 0.2, 0.3, 0.8)}``.
        If it is a dictionary with a key naming a palette (one of "tol", "tol-light",
        "wong", or "ibm"), then the value is the ith color of the corresponding palette.
        """
        match colorSpec:
            case (r, g, b):
                return (r, g, b)
            case (r, b, g, a):
                return (r, g, b, a)
            case {"rgb": (r, g, b)}:
                return (r, g, b)
            case {"rgba": (r, g, b, a)}:
                return (r, g, b, a)
            case {"tol": num}:
                return cls.tol[num]
            case {"tol-light": num}:
                return cls.tolLight[num]
            case {"ibm": num}:
                return cls.ibm[num]
            case {"wong": num}:
                return cls.wong[num]
            case _:
                assert False, f"Invalid color spec: {colorSpec}"

    def __init__(self):
        raise NotImplementedError("ColorMaps should not be instantiated.")

# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
