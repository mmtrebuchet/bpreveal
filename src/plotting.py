"""Utilities for making plots with your data.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/plotting.bnf

"""
import math
from typing import TypeAlias, Literal
import h5py
import pysam
import pyBigWig
import pybedtools
import numpy as np
# You must install bpreveal with conda develop in order to import bpreveal tools.
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.axes import Axes as AXES_T
from matplotlib.transforms import Bbox, Affine2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import matplotlib.colors as mplcolors
import matplotlib.ticker as mplticker
from bpreveal import logUtils
from bpreveal.internal.constants import IMPORTANCE_AR_T, IMPORTANCE_T, PRED_AR_T, ONEHOT_AR_T
from bpreveal import utils
from bpreveal import motifUtils
from bpreveal import schema


def _toFractions(colorList):
    return tuple(tuple(y / 256 for y in rgb) for rgb in colorList)


FONT_FAMILY = "serif"

COLOR_SPEC: TypeAlias = \
    dict[Literal["rgb"], tuple[float, float, float]] | \
    dict[Literal["rgba"], tuple[float, float, float, float]] | \
    dict[Literal["tol"], int] | \
    dict[Literal["tol-light"], int] | \
    dict[Literal["ibm"], int] | \
    dict[Literal["wong"], int]

DNA_COLOR_SPEC: TypeAlias = dict[str, COLOR_SPEC]


class COLOR_MAPS:  # pylint: disable=invalid-name
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
              (17, 119, 51),
              (68, 170, 153),
              (136, 204, 238),
              (221, 204, 119),
              (204, 102, 119),
              (170, 68, 153),
              (136, 34, 85))

    tolLightRgb = ((153, 221, 255),
                   (68, 187, 153),
                   (187, 204, 51),
                   (238, 221, 136),
                   (238, 136, 102),
                   (255, 170, 187),
                   (221, 221, 221))

    tol = _toFractions(tolRgb)
    ibm = _toFractions(ibmRgb)
    wong = _toFractions(wongRgb)
    tolLight = _toFractions(tolLightRgb)

    dnaTol: DNA_COLOR_SPEC = {"A": {"wong": 3}, "C": {"wong": 5},
                              "G": {"wong": 4}, "T": {"wong": 6}}

    defaultProfile = {"tol": 0}

    _oldPisaCmap = mpl.colormaps["RdBu_r"].resampled(256)
    _newPisaColors = _oldPisaCmap(np.linspace(0, 1, 256))
    _pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    _green = np.array([24 / 256, 248 / 256, 148 / 256, 1])
    _newPisaColors[:5] = _green
    _newPisaColors[-5:] = _pink

    pisaClip = mplcolors.ListedColormap(_newPisaColors)
    pisaNoClip = mplcolors.ListedColormap(_oldPisaCmap(np.linspace(0, 1, 256)))

    @classmethod
    def parseSpec(cls, colorSpec: dict | tuple):  # pylint: disable=too-many-return-statements
        """Given a color-spec (See the BNF), convert it into an rgb or rgba triple.

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


def plotPisaGraph(config: dict, fig: matplotlib.figure.Figure, validate: bool = True):
    r"""Make a graph-style PISA plot.

    :param config: The JSON (or any dictionary, really) configuration for this PISA plot.
    :param fig: The matplotlib figure that will be drawn on.
    :param validate: Should the configuration be checked? If you're passing in numpy
        arrays, the json validator is prone to exploding.
    :return: A dictionary containing the created Axes objects, the assigned colors for
        annotations, and the genomic coordinates is the plot.

    The structure of the returned dict is:

    .. highlight:: none

    ::

        {"axes": {"pisa": matplotlib.axes,
                  "importance": matplotlib.axes,
                  "predictions": matplotlib.axes,
                  "annotations": matplotlib.axes}
            "name-colors": Same structure as the config dict, but with missing entries added.
            "genome-start": int,
            "genome-end": int,
            "config": config dict with data loaded.}

    .. highlight:: python

    Example::

        fig = plt.figure()
        pisaSection = {
            "h5-name": WORKING_DIRECTORY + "/shap/pisa_nanog_positive.h5",
            "receptive-field": RECEPTIVE_FIELD
        }

        coordinatesSection = {
            "genome-fasta": GENOME_FASTA,
            "midpoint-offset": 1150,
            "input-slice-width": 200,
            "output-slice-width": 300,
            "genome-window-start": windowStart,
            "genome-window-chrom": windowChrom
        }

        predictionSection = {
            "bigwig-name": WORKING_DIRECTORY + "/pred/nanog_residual_positive.bw",
            "show-sequence": False,
            "color": {"tol": 0}
        }

        importanceSection = {
            "bigwig-name": WORKING_DIRECTORY + "/shap/nanog_profile.bw",
            "show-sequence": True,
            "color": bprplots.COLOR_MAPS.dnaTol
        }

        annotationSection = {
            "bed-name": WORKING_DIRECTORY + "/scan/nanog_profile.bed",
            "custom": []
        }

        figureSectionGraph = {
            "left": 0.1,
            "bottom": 0.05,
            "width": 0.9,
            "height": 0.4,
            "annotation-height": 0.5,
            "tick-font-size" : 6,
            "label-font-size" : 8
        }

        graphConfig = {
            "pisa": pisaSection,
            "coordinates": coordinatesSection,
            "importance": importanceSection,
            "predictions": predictionSection,
            "annotations": annotationSection,
            "figure": figureSectionGraph,
            "min-value": 0.1,
            "max-value": 0.5,
            "use-annotation-colors": True
        }

        bpreveal.plotting.plotPisaGraph(plotUpperGraph, fig);

    """
    if validate:
        schema.pisaGraph.validate(config)
    cfg = _buildConfig(config)
    del config  # Don't accidentally use the old one.
    logUtils.debug("Starting to draw PISA graph.")

    axGraph, axSeq, axProfile, axAnnot = _getPisaGraphAxes(fig,
                                                           cfg["figure"]["left"],
                                                           cfg["figure"]["bottom"],
                                                           cfg["figure"]["width"],
                                                           cfg["figure"]["height"])
    # Pisa image plotting
    coords = cfg["coordinates"]
    sliceStart = coords["midpoint-offset"] - coords["input-slice-width"] // 2
    sliceEnd = sliceStart + coords["input-slice-width"]
    genomeStart = coords["genome-window-start"] + sliceStart
    genomeEnd = coords["genome-window-start"] + sliceEnd
    shearMat = cfg["pisa"]["values"][sliceStart:sliceEnd, sliceStart:sliceEnd]
    colorBlocks = []
    for annot in cfg["annotations"]["custom"]:
        colorBlocks.append((annot["start"] - genomeStart,
                            annot["end"] - genomeStart,
                            annot["color"]))
    logUtils.debug("Axes set. Drawing graph.")
    _addPisaGraph(similarityMat=shearMat, minValue=cfg["min-value"],
                  maxValue=cfg["max-value"], colorBlocks=colorBlocks,
                  useAnnotationColor=cfg["use-annotation-colors"],
                  ax=axGraph)
    # Now set up the sequence/importance axis.
    logUtils.debug("Graph complete. Finishing plot.")
    _addHorizontalProfilePlot(cfg["importance"]["values"][sliceStart:sliceEnd],
                              cfg["importance"]["color"][sliceStart:sliceEnd],
                              coords["sequence"][sliceStart:sliceEnd],
                              genomeStart,
                              genomeEnd,
                              axSeq, axGraph,
                              cfg["figure"]["tick-font-size"],
                              cfg["importance"]["show-sequence"],
                              True, False)

    _addAnnotations(axAnnot, cfg["annotations"]["custom"], 0.13, genomeStart, genomeEnd,
                    coords["input-slice-width"],
                    cfg["figure"]["label-font-size"], False)
    # Now, add the profiles.
    _addHorizontalProfilePlot(cfg["predictions"]["values"][sliceStart:sliceEnd],
                              cfg["predictions"]["color"][sliceStart:sliceEnd],
                              coords["sequence"][sliceStart:sliceEnd],
                              genomeStart,
                              genomeEnd,
                              axProfile, None,
                              cfg["figure"]["tick-font-size"],
                              cfg["predictions"]["show-sequence"],
                              False, False)

    logUtils.debug("PISA graph plot complete.")
    return {"axes": {"graph": axGraph, "importance": axSeq, "predictions": axProfile,
                     "annotations": axAnnot},
            "name-colors": cfg["annotations"]["name-colors"],
            "genome-start": genomeStart,
            "genome-end": genomeEnd,
            "config": cfg}


def plotPisa(config: dict, fig: matplotlib.figure.Figure, validate: bool = True):
    r"""Given the actual vectors to show, make a pretty pisa plot.

    :param config: The JSON (or any dictionary, really) configuration for this PISA plot.
    :param fig: The matplotlib figure that will be drawn on.
    :param validate: Should the configuration be checked? If you're passing in numpy
        arrays, the json validator is prone to exploding.
    :return: A dict containing the generated axes, along with the assigned name colors and
        the coordinates that were plotted.

    The returned dict will have the following structure:

    .. highlight:: none

    ::

        {"axes": {"pisa": matplotlib.axes,
                  "importance": matplotlib.axes,
                  "predictions": matplotlib.axes,
                  "annotations": matplotlib.axes}
            "name-colors": Same structure as the config dict, but with missing entries added.
            "genome-start": int,
            "genome-end": int,
            "config": config dict with data loaded}

    .. highlight:: python

    Example::

        fig = plt.figure()
        pisaSection = {
            "h5-name": WORKING_DIRECTORY + "/shap/pisa_nanog_positive.h5",
            "receptive-field": RECEPTIVE_FIELD
        }

        coordinatesSection = {
            "genome-fasta": GENOME_FASTA,
            "midpoint-offset": 1150,
            "input-slice-width": 200,
            "output-slice-width": 300,
            "genome-window-start": windowStart,
            "genome-window-chrom": windowChrom
        }

        predictionSection = {
            "bigwig-name": WORKING_DIRECTORY + "/pred/nanog_residual_positive.bw",
            "show-sequence": False,
            "color": {"tol": 0}
        }

        importanceSection = {
            "bigwig-name": WORKING_DIRECTORY + "/shap/nanog_profile.bw",
            "show-sequence": True,
            "color": bprplots.COLOR_MAPS.dnaTol
        }

        annotationSection = {
            "bed-name": WORKING_DIRECTORY + "/scan/nanog_profile.bed",
            "custom": []
        }

        figureSectionPlot = {
            "left": 0.1,
            "bottom": 0.55,
            "width": 0.9,
            "height": 0.4,
            "annotation-height": 0.5,
            "tick-font-size" : 6,
            "label-font-size" : 8,
            "grid-mode": "on",
            "diagonal-mode": "edge"
        }

        plotConfig = {
            "pisa": pisaSection,
            "coordinates": coordinatesSection,
            "importance": importanceSection,
            "predictions": predictionSection,
            "annotations": annotationSection,
            "figure": figureSectionPlot,
            "color-span" : 0.5,
            "miniature": False
        }

        bpreveal.plotting.plotPisa(plotUpper, fig)

    """
    if validate:
        schema.pisaPlot.validate(config)
    cfg = _buildConfig(config)
    del config  # Don't accidentally edit the old one.
    logUtils.debug("Starting to draw PISA graph.")

    axPisa, axSeq, axProfile, axCbar, axAnnot, axLegend = _getPisaAxes(
        fig, cfg["figure"]["left"], cfg["figure"]["bottom"],
        cfg["figure"]["width"], cfg["figure"]["height"],
        cfg["miniature"])
    # Pisa image plotting
    coords = cfg["coordinates"]
    sliceStartX = coords["midpoint-offset"] - coords["input-slice-width"] // 2
    sliceEndX = sliceStartX + coords["input-slice-width"]
    sliceStartY = coords["midpoint-offset"] - coords["output-slice-width"] // 2
    sliceEndY = sliceStartY + coords["output-slice-width"]
    genomeStartX = coords["genome-window-start"] + sliceStartX
    genomeEndX = coords["genome-window-start"] + sliceEndX
    shearMat = np.copy(cfg["pisa"]["values"][sliceStartY:sliceEndY, sliceStartX:sliceEndX])
    colorBlocks = []
    for annot in cfg["annotations"]["custom"]:
        colorBlocks.append((annot["start"] - coords["genome-window-start"],
                            annot["end"] - coords["genome-window-start"],
                            annot["color"]))
    logUtils.debug("Axes set. Drawing graph.")
    pisaCax = _addPisaPlot(shearMat, cfg["color-span"], axPisa,
                           cfg["figure"]["diagonal-mode"], cfg["figure"]["grid-mode"],
                           cfg["figure"]["tick-font-size"], cfg["figure"]["label-font-size"],
                           genomeStartX, cfg["miniature"])
    # Now set up the sequence/importance axis.
    logUtils.debug("Graph complete. Finishing plot.")
    _addHorizontalProfilePlot(cfg["importance"]["values"][sliceStartX:sliceEndX],
                              cfg["importance"]["color"][sliceStartX:sliceEndX],
                              coords["sequence"][sliceStartX:sliceEndX],
                              genomeStartX,
                              genomeEndX,
                              axSeq, axPisa,
                              cfg["figure"]["tick-font-size"],
                              cfg["importance"]["show-sequence"],
                              True, cfg["miniature"])

    usedNames = _addAnnotations(axAnnot, cfg["annotations"]["custom"], 0.13, genomeStartX,
                                genomeEndX, coords["input-slice-width"],
                                cfg["figure"]["label-font-size"], False)
    # Now, add the profiles.
    _addVerticalProfilePlot(cfg["predictions"]["values"][sliceStartY:sliceEndY],
                            axProfile,
                            cfg["predictions"]["color"][sliceStartY:sliceEndY],
                            coords["sequence"][sliceStartY:sliceEndY],
                            cfg["figure"]["tick-font-size"],
                            cfg["predictions"]["show-sequence"],
                            cfg["miniature"])
    if cfg["miniature"]:
        _addLegend(usedNames, axLegend, cfg["figure"]["label-font-size"])
    _addCbar(pisaCax, axCbar, cfg["figure"]["label-font-size"], cfg["miniature"])
    return {"axes": {"pisa": axPisa, "importance": axSeq, "predictions": axProfile,
                     "annotations": axAnnot, "colorbar": axCbar,
                     "legend": axLegend},
            "name-colors": cfg["annotations"]["name-colors"],
            "genome-start": genomeStartX,
            "genome-end": genomeEndX,
            "config": cfg}


def plotModiscoPattern(pattern: motifUtils.Pattern,  # pylint: disable=too-many-statements
                       fig: matplotlib.figure.Figure, sortKey=None):
    """Create a plot showing a pattern's seqlets and their match scores.

    :param sortKey: Either None (do not sort) or an array of shape (numSeqlets,)
        giving the order in which the seqlets should be displayed. See example below
        for common use cases.
    :type sortKey: None or ndarray
    :param pattern: The pattern to plot. This pattern must have already had its seqlets loaded.
    :param fig: The matplotlib figure upon which the plots should be drawn.

    Example::

        # Background ACGT frequency
        bgProbs = [0.29, 0.21, 0.21, 0.29]
        patZld = motifUtils.Pattern("pos_patterns", "pattern_1", "Zld")
        with h5py.File("modisco_results.h5", "r") as fp:
            patZld.loadCwm(fp, 0.3, 0.3, bgProbs)
            patZld.loadSeqlets(fp)
        fig = plt.figure()
        # Sort the seqlets by their contribution match.
        sortKey = [x.contribMatch for x in patZld.seqlets]
        plotModiscoPattern(patZld, fig, sortKey=sortKey)

    """
    if sortKey is None:
        sortKey = np.arange(len(pattern.seqlets))
    sortOrder = np.argsort(sortKey)
    HIST_HEIGHT = 0.1  # pylint: disable=invalid-name
    PAD = 0.01  # pylint: disable=invalid-name
    axHmap = fig.add_axes((0.1, 0.1 + HIST_HEIGHT + PAD,
                           0.2 - PAD, 0.8 - HIST_HEIGHT))
    hmapAr = np.zeros((len(pattern.seqlets),
                       len(pattern.seqlets[0].sequence),
                       4), dtype=np.uint8)
    for outIdx, seqletIdx in enumerate(sortOrder):
        hmapAr[outIdx] = utils.oneHotEncode(pattern.seqlets[seqletIdx].sequence)
    plotSequenceHeatmap(hmapAr, axHmap)
    axHmap.set_xticks([])
    axLogo = fig.add_axes((0.1, 0.1, 0.2 - PAD, HIST_HEIGHT))
    cwm = pattern.cwm
    plotLogo(cwm, cwm.shape[0], axLogo, colors=COLOR_MAPS.dnaTol)
    axLogo.set_xlim(0, cwm.shape[0])
    cwmPos = np.zeros_like(cwm)
    cwmPos[cwm > 0] = cwm[cwm > 0]
    cwmNeg = np.zeros_like(cwm)
    cwmNeg[cwm < 0] = cwm[cwm < 0]
    sumPos = np.sum(cwmPos, axis=1)
    sumNeg = np.sum(cwmNeg, axis=1)

    axLogo.set_ylim(min(sumNeg), max(sumPos))
    axLogo.set_yticks([])
    axLogo.set_xticks([pattern.cwmTrimLeftPoint, pattern.cwmTrimRightPoint])

    yvals = np.arange(len(pattern.seqlets), 0, -1)

    def plotStat(stat, axPos, name, rightTicks):
        stat = np.array(stat)
        axCurStat = fig.add_axes((0.3 + axPos * 0.2, 0.1 + HIST_HEIGHT + PAD,
                                  0.2 - PAD, 0.8 - HIST_HEIGHT))
        statSort = stat[sortOrder]
        colorFix = COLOR_MAPS.parseSpec(COLOR_MAPS.defaultProfile)
        axCurStat.plot(statSort, yvals, ".", color=colorFix, alpha=0.5)
        statFilt = np.sort(statSort)
        colorFix = COLOR_MAPS.parseSpec({"tol": 2})
        axCurStat.plot(statFilt, yvals, "-", color=colorFix)
        axCurStat.set_ylim(0, len(pattern.seqlets))
        axCurStat.set_yticks([])
        axCurStat.set_title(name, fontdict={"fontsize": 10})
        axCurStat.set_xticks([])
        axCurStat.set_xlim(min(stat), max(stat))
        tickPoses = np.linspace(0, len(pattern.seqlets), 11, endpoint=True)
        tickLabels = np.arange(0, 110, 10)[::-1]
        axCurStat.tick_params(axis="y", labelleft=False, labelright=rightTicks,
                            left=False, right=rightTicks)
        axCurStat.set_yticks(tickPoses, tickLabels)
        axCurStat.grid()
        if rightTicks:
            axCurStat.yaxis.set_label_position("right")
            axCurStat.set_ylabel("Percentile")
        axCurHist = fig.add_axes((0.3 + axPos * 0.2, 0.1, 0.2 - PAD, HIST_HEIGHT))
        hist = np.histogram(stat, bins=50)
        binMiddles = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2
        axCurHist.plot(binMiddles, hist[0])
        axCurHist.set_yticks([])
        axCurHist.set_xlim(min(stat), max(stat))
        return axCurStat
    plotStat([x.seqMatch for x in pattern.seqlets], 0, "seqMatch", False)
    plotStat([x.contribMatch for x in pattern.seqlets], 1, "contribMatch", False)
    plotStat([x.contribMagnitude for x in pattern.seqlets], 2, "contribMag", True)


def plotSequenceHeatmap(hmap: ONEHOT_AR_T, ax: AXES_T, upsamplingFactor: int = 10):
    """Show a sequence heatmap from an array of one-hot encoded sequences.

    :param hmap: An array of sequences of shape (numSequences, length, 4)
    :param ax: A matplotlib Axes object upon which the heatmap will be drawn.
    :param upsamplingFactor: How much should the x-axis be sharpened?
        If upsamplingFactor * hmap.shape[1] >> ax.width_in_pixels then you
        may get aliasing artifacts.
        If upsamplingFactor * hmap.shape[1] << ax.width_in_pixels then you
        will get blurry borders.

    """
    displayAr = np.zeros((hmap.shape[0], hmap.shape[1] * upsamplingFactor, 3),
                         dtype=np.float32)
    for base, baseName in enumerate("ACGT"):
        hmapBase = np.array(hmap[:, :, base], dtype=np.float32)
        ar = hmapBase
        for colorIdx in range(3):
            color = COLOR_MAPS.parseSpec(COLOR_MAPS.dnaTol[baseName])[colorIdx]
            for col in range(hmap.shape[1]):
                for colOffset in range(upsamplingFactor):
                    writeCol = col * upsamplingFactor + colOffset
                    displayAr[:, writeCol, colorIdx] += ar[:, col] * color / 256
    ax.imshow(displayAr, aspect="auto", interpolation="antialiased",
              interpolation_stage="data")


def plotLogo(values: PRED_AR_T, width: float, ax: AXES_T,
             colors: dict | list[dict],
             spaceBetweenLetters: float = 0) -> None:
    """Plot an array of sequence data (like a pwm).

    :param values: An (N,4) array of sequence data. This could be, for example,
        a pwm or a one-hot encoded sequence.
    :param width: The width of the total logo, useful for aligning axis labels.
    :param ax: A matplotlib axes object on which the logo will be drawn.
    :param colors: The colors to use for shading the sequence. See below for details.
    :param spaceBetweenLetters: How much should the letters be squished? This is
        given as a fraction of the total letter width. For example, to have
        a gap of 2 pixels between letters that are 10 pixels wide, set
        ``spaceBetweenLetters=0.2``.

    Colors, if provided, can have several meanings:
        1. Give a color for each base type by RGB value.
            In this case, colors will be a dict of tuples:
            ``{"A": (.8, .3, .2), "C": (.5, .3, .9), "G": (1., .4, .0), "T": (1., .7, 0.)}``
            This will make each instance of a particular base have the same color.
        2. Give a color for each base by color-spec.
            This would be something like:
            ``{"A": {"wong": 3}, "C": {"wong": 5}, "G": {"wong": 4}, "T": {"wong": 6}}``
            You can get the default BPReveal color map at :py:class:`~COLOR_MAPS`.
        3. Give a list of colors for each base. This will be a list of length
            ``values.shape[0]`` and each entry should be a dictionary in either format
            1 or 2 above. This gives each base its own color palette, useful for shading
            bases by some profile.
    """

    def _drawLetter(text: str, left: float, right: float, bottom: float, top: float,
                    color, ax, flip=False) -> None:

        height = top - bottom
        width = right - left
        bbox = Bbox.from_bounds(left, bottom, width, height)
        fontProperties = FontProperties(family="sans", weight="bold")
        tmpPath = TextPath((0, 0), text, size=1, prop=fontProperties)
        if flip:
            flipTransformation = Affine2D().scale(sx=1, sy=-1)
            # pylint: disable=redefined-variable-type
            tmpPath = flipTransformation.transform_path(tmpPath)
            # pylint: enable=redefined-variable-type
        tmpBbox = tmpPath.get_extents()
        hstretch = bbox.width / tmpBbox.width
        vstretch = bbox.height / tmpBbox.height
        transformation = Affine2D().\
            translate(tx=-tmpBbox.xmin, ty=-tmpBbox.ymin).\
            scale(sx=hstretch, sy=vstretch).\
            translate(tx=bbox.xmin, ty=bbox.ymin)
        charPath = transformation.transform_path(tmpPath)
        patch = PathPatch(charPath, facecolor=color, lw=0.3)
        ax.add_patch(patch)

    def getColor(pos, base):
        match colors:
            case list():
                # Colors is indexed by position.
                return COLOR_MAPS.parseSpec(colors[pos][base])
            case dict():
                return COLOR_MAPS.parseSpec(colors[base])
            case _:
                assert False, "Invalid color spec."

    for predIdx in range(values.shape[0]):
        a, c, g, t = values[predIdx]
        lettersToDraw = [("A", a), ("C", c), ("G", g), ("T", t)]
        posLetters = [x for x in lettersToDraw if x[1] > 0]
        negLetters = [x for x in lettersToDraw if x[1] < 0]
        posLetters.sort(key=lambda x: x[1])
        negLetters.sort(key=lambda x: -x[1])

        # Draw the negative letters.
        top = 0
        for nl in negLetters:
            # Note that top < base because nl[1] < 0
            base = top + nl[1]
            left = predIdx * width / values.shape[0] + spaceBetweenLetters / 2
            right = left + width / values.shape[0] - spaceBetweenLetters
            _drawLetter(nl[0], left=left, right=right, bottom=base, top=top,
                       color=getColor(predIdx, nl[0]), ax=ax, flip=True)
            top = base

        # Draw the positive letters.
        base = 0
        for pl in posLetters:
            top = base + pl[1]
            left = predIdx * width / values.shape[0] + spaceBetweenLetters / 2
            right = left + width / values.shape[0] - spaceBetweenLetters
            _drawLetter(pl[0], left=left, right=right, bottom=base, top=top,
                       color=getColor(predIdx, pl[0]), ax=ax)
            base = top


def getCoordinateTicks(start: int, end: int, numTicks: int,
                       zeroOrigin: bool) -> tuple[list[float], list[str]]:
    """Given a start and end coordinate, return x-ticks that should be used for plotting.

    :param start: The genomic coordinate where your ticks start, inclusive.
    :param end: The genomic coordinate where your ticks end, *inclusive*.
    :param numTicks: The approximate number of ticks you want.
    :param zeroOrigin: The actual x coordinate of the ticks should start
        at zero, even though the labels start at ``start``. Otherwise,
        the ticks will be positioned at coordinate ``start`` to ``end``,
        and so your axes limits should actually correspond to genomic coordinates.
    :return: Two lists. The first is the x-coordinate of the ticks, and the second
        is the string labels that should be used at each tick.

    Given a start and end coordinate, return a list of ticks and tick labels that
    1. include exactly the start and stop coordinates
    2. Contain approximately numTicks positions and labels.
    3. Try to fall on easy multiples 1, 2, and 5 times powers of ten.
    4. Are formatted to reduce redundant label noise by omitting repeated initial digits.
    """
    reverse = False
    if start > end:
        start, end = end, start
        reverse = True
    Δ = abs(end - start)
    tickWidth = 1
    multiplier = 1
    scales = [1, 2, 5]
    scaleIdx = 0
    while tickWidth < Δ / numTicks:
        tickWidth = multiplier * scales[scaleIdx]
        scaleIdx += 1
        if scaleIdx == len(scales):
            multiplier *= 10
            scaleIdx = 0
    multiLoc = mplticker.MultipleLocator(tickWidth)
    multiLoc.view_limits(start, end)
    innerTickPoses = multiLoc.tick_values(start, end)
    while innerTickPoses[0] < start + tickWidth / 2:
        innerTickPoses = innerTickPoses[1:]
    while len(innerTickPoses) and innerTickPoses[-1] > end - tickWidth / 2:
        innerTickPoses = innerTickPoses[:-1]
    tickPoses = [float(x) for x in [start] + list(innerTickPoses) + [end]]
    tickLabelStrs = [f"{int(x):,}" for x in tickPoses]
    if zeroOrigin:
        tickPoses = [x - start for x in tickPoses]
    tickLabels = _massageTickLabels(tickLabelStrs)
    if reverse:
        tickPoses = tickPoses[::-1]
        tickLabels = tickLabels[::-1]
    return tickPoses, tickLabels


def _massageTickLabels(labelList):  # pylint: disable=too-many-branches
    allThousands = True
    for lbl in labelList[1:-1]:
        if lbl[-4:] != ",000":
            allThousands = False
    labelsThousands = []
    if allThousands:
        for lbl in labelList:
            if lbl[-4:] == ",000":
                labelsThousands.append(lbl[:-4] + "k")
            else:
                labelsThousands.append(lbl)
    else:
        labelsThousands = labelList[:]

    for pos in range(len(labelsThousands) - 2, 0, -1):
        prevLabel = labelsThousands[pos - 1]
        curLabel = list(labelsThousands[pos])
        apostrophePos = 0
        for lpos in range(min(len(curLabel), len(prevLabel))):
            if curLabel[lpos] == prevLabel[lpos]:
                if curLabel[lpos] == ",":
                    apostrophePos = lpos
            else:
                break
        if apostrophePos > 1:
            newLabel = "´" + "".join(curLabel[apostrophePos + 1:])
        else:
            newLabel = "".join(curLabel)
        labelsThousands[pos] = newLabel
    return labelsThousands


def _buildConfig(oldConfig):
    """Read in a config and add any missing data.

    :param oldConfig: The original configuration dictionary. All entries from this
        original dict are copied, so you can mutate the returned dict without
        messing with the original data.
    This loads in profile and pisa data from files and expands the color specs.
    """
    newConfig = {
        "pisa": {},
        "coordinates": {
            "midpoint-offset": oldConfig["coordinates"]["midpoint-offset"],
            "input-slice-width": oldConfig["coordinates"]["input-slice-width"],
            "output-slice-width": oldConfig["coordinates"]["output-slice-width"],
            "genome-window-start": oldConfig["coordinates"]["genome-window-start"],
            "genome-window-chrom": oldConfig["coordinates"]["genome-window-chrom"]},
        "importance": {"show-sequence": oldConfig["importance"]["show-sequence"]},
        "predictions": {"show-sequence": oldConfig["predictions"]["show-sequence"]},
        "annotations": {},
        "figure": oldConfig["figure"]}

    if "min-value" in oldConfig:
        # We have a graph-style config.
        newConfig["min-value"] = oldConfig["min-value"]
        newConfig["max-value"] = oldConfig["max-value"]
        newConfig["use-annotation-colors"] = oldConfig["use-annotation-colors"]
    else:
        # We have a plot-style config.
        newConfig["color-span"] = oldConfig["color-span"]
        newConfig["miniature"] = oldConfig["miniature"]

    # First, the pisa data.
    if "h5-name" in oldConfig["pisa"]:
        # We need to load from file.
        newConfig["pisa"]["values"] = _loadPisa(oldConfig["pisa"]["h5-name"],
                                                oldConfig["pisa"]["receptive-field"])
    else:
        newConfig["pisa"]["values"] = np.array(oldConfig["pisa"]["values"])

    # Do we need to load bigwig data?
    if "bigwig-name" in oldConfig["importance"]:
        vals = _loadFromBigwig(oldConfig["importance"]["bigwig-name"],
                              oldConfig["coordinates"]["genome-window-start"],
                              oldConfig["coordinates"]["genome-window-chrom"],
                              newConfig["pisa"]["values"].shape[1])
        newConfig["importance"]["values"] = vals
    else:
        newConfig["importance"]["values"] = np.array(oldConfig["importance"]["values"])

    # Convert the color spec into a list for each base.
    newConfig["importance"]["color"] = _normalizeProfileColor(
        oldConfig["importance"]["color"],
        len(newConfig["importance"]["values"]))
    if "bigwig-name" in oldConfig["predictions"]:
        vals = _loadFromBigwig(oldConfig["predictions"]["bigwig-name"],
                              oldConfig["coordinates"]["genome-window-start"],
                              oldConfig["coordinates"]["genome-window-chrom"],
                              newConfig["pisa"]["values"].shape[0])
        newConfig["predictions"]["values"] = vals
    else:
        newConfig["predictions"]["values"] = np.array(oldConfig["predictions"]["values"])
    # The profile will be strictly positive.
    newConfig["predictions"]["values"] = np.abs(newConfig["predictions"]["values"])
    newConfig["predictions"]["color"] = _normalizeProfileColor(
        oldConfig["predictions"]["color"],
        len(newConfig["predictions"]["values"]))

    if "bed-name" in oldConfig["annotations"]:
        nameColors = oldConfig["annotations"].get("name-colors", {})
        oldCustom = oldConfig["annotations"].get("custom", [])
        newCustom = _loadPisaAnnotations(oldConfig["annotations"]["bed-name"],
                                         nameColors,
                                         oldConfig["coordinates"]["genome-window-start"],
                                         oldConfig["coordinates"]["genome-window-chrom"],
                                         newConfig["pisa"]["values"].shape[1])
        newConfig["annotations"]["name-colors"] = nameColors
        newConfig["annotations"]["custom"] = oldCustom + newCustom

    if "genome-fasta" in oldConfig["coordinates"]:
        newConfig["coordinates"]["sequence"] = _loadSequence(
            oldConfig["coordinates"]["genome-fasta"],
            oldConfig["coordinates"]["genome-window-start"],
            oldConfig["coordinates"]["genome-window-chrom"],
            newConfig["pisa"]["values"].shape[1])
    return newConfig


def _loadPisa(fname: str, receptiveField: int) -> IMPORTANCE_AR_T:
    with h5py.File(fname, "r") as fp:
        pisaShap = np.array(fp["shap"])
    pisaVals = np.sum(pisaShap, axis=2)
    numRegions = pisaVals.shape[0]
    shearMat = np.zeros((numRegions, pisaVals.shape[1] + numRegions),
                        dtype=IMPORTANCE_T)
    for i in range(0, numRegions):
        offset = i
        shearMat[i, offset:offset + pisaVals.shape[1]] = pisaVals[i]
    shearMat = shearMat[:, receptiveField // 2:-receptiveField // 2]
    return shearMat


def _loadFromBigwig(bwFname: str, start: int, chrom: str, length: int):
    impFp = pyBigWig.open(bwFname)
    impScores = np.nan_to_num(impFp.values(chrom, start, start + length))
    impFp.close()
    return impScores


def _normalizeProfileColor(colorSpec, numItems):
    match colorSpec:
        case {"A": aColor, "C": cColor, "G": gColor, "T": tColor}:
            a = aColor
            c = cColor
            g = gColor
            t = tColor
        case list():
            return colorSpec
        case _:
            a = c = g = t = colorSpec
    colorDict = {"A": a, "C": c, "G": g, "T": t}
    return [colorDict] * numItems


def _loadSequence(genomeFastaFname: str, genomeWindowStart: int,
                  genomeWindowChrom: str, length: int) -> str | None:
    with pysam.FastaFile(genomeFastaFname) as genome:
        seq = genome.fetch(genomeWindowChrom, genomeWindowStart, genomeWindowStart + length)
    return seq.upper()


def _loadPisaAnnotations(bedFname, nameColors, start, chrom, length):
    annotations = []
    bedFp = pybedtools.BedTool(bedFname)
    for line in bedFp:
        if line.chrom == chrom and line.end > start\
                and line.start < start + length:
            if line.name not in nameColors:
                nameColors[line.name] = \
                    {"tol-light": len(nameColors) % len(COLOR_MAPS.tolLight)}
            if line.start < start:
                line.start = start
            if line.end > start + length:
                line.end = start + length
            annotations.append({
                "start": line.start,
                "end": line.end,
                "name": line.name,
                "color": nameColors[line.name]
            })
    return annotations


def _getPisaAxes(fig, left, bottom, width, height, mini) -> tuple[AXES_T, AXES_T,
        AXES_T, AXES_T, AXES_T, AXES_T | None]:
    xweightPisa = 40
    xweightProfile = 6
    xweightCbar = 3 if mini else 1
    widthScale = 1
    totalWeight = xweightPisa + xweightProfile + xweightCbar
    pisaWidth = width * xweightPisa / totalWeight * widthScale
    profileWidth = width * xweightProfile / totalWeight * widthScale
    cbarWidth = width * xweightCbar / totalWeight * widthScale
    pisaHeight = height * 7 / 8
    seqHeight = height / 8

    axPisa = fig.add_axes([left, bottom + seqHeight, pisaWidth, pisaHeight])
    axSeq = fig.add_axes([left, bottom, pisaWidth, seqHeight])
    axProfile = fig.add_axes([left + pisaWidth + profileWidth * 0.02,
                              bottom + seqHeight, profileWidth * 0.9, pisaHeight])
    axCbar = fig.add_axes([left + pisaWidth + profileWidth,
                           bottom + seqHeight + pisaHeight / (8 if mini else 4),
                           cbarWidth, pisaHeight / (3 if mini else 2)])
    axLegend = None
    if mini:
        axLegend = fig.add_axes([left + pisaWidth + profileWidth,
                            bottom + seqHeight + pisaHeight * (1 / 3 + 1 / 7),
                            cbarWidth * 3, pisaHeight * (1 - 1 / 3 - 1 / 7)])
        axLegend.set_axis_off()
    axAnnot = fig.add_axes([left,
                            bottom + seqHeight + 2 * pisaHeight / 3,
                            pisaWidth,
                            pisaHeight / 3])
    axAnnot.set_axis_off()

    axSeq.set_frame_on(False)
    axSeq.set_yticks([])
    axAnnot.set_ylim(-1, 0)
    axAnnot.set_axis_off()
    axProfile.set_yticks([])
    axProfile.set_frame_on(False)
    axProfile.yaxis.set_visible(False)
    return axPisa, axSeq, axProfile, axCbar, axAnnot, axLegend


def _addVerticalProfilePlot(profile, axProfile, colors, sequence, fontsize,
                            fontSizeAxLabel, mini):
    plotProfile = list(profile)
    for pos, val in enumerate(plotProfile):
        y = len(plotProfile) - pos
        axProfile.fill_betweenx([y, y + 1], val, step="post",
                                color=COLOR_MAPS.parseSpec(colors[pos][sequence[pos]]))
    axProfile.set_ylim(0, len(profile))
    axProfile.set_xlim(0, np.max(profile))
    if mini:
        axProfile.set_xticks([])
        axProfile.xaxis.set_visible(False)
    else:
        profileXticks = axProfile.get_xticks()
        if max(profileXticks) > np.max(profile) * 1.01:
            profileXticks = profileXticks[:-1]
        axProfile.set_xticks(profileXticks, profileXticks, fontsize=fontsize)
        axProfile.set_xlabel("Profile", fontsize=fontSizeAxLabel, fontfamily=FONT_FAMILY)


def _addAnnotations(axAnnot, annotations, boxHeight, genomeStartX,
                    genomeEndX, cutLengthX, fontsize, mini):
    offset = -boxHeight * 1.3
    lastR = 0
    usedNames = {}
    for annot in sorted(annotations, key=lambda x: x["start"]):
        aleft = annot["start"]
        aright = annot["end"]
        if aright < genomeStartX or aleft > genomeEndX:
            continue
        # No directly abutting annotations - at least 1%
        if aleft > lastR + cutLengthX / 100:
            offset = -boxHeight * 1.3
        lastR = max(lastR, aright)
        if offset < -1:
            # We're off the page - reset offset and deal with the overlap.
            offset = -boxHeight * 1.3
        axAnnot.fill([aleft, aleft, aright, aright],
                     [offset, boxHeight + offset, boxHeight + offset, offset],
                     label=annot["name"], color=COLOR_MAPS.parseSpec(annot["color"]))
        if not mini:
            axAnnot.text((aleft + aright) / 2, offset + boxHeight / 2, annot["name"],
                     fontstyle="italic", fontsize=fontsize, fontfamily=FONT_FAMILY,
                     ha="center", va="center")
        usedNames[annot["name"]] = annot["color"]
        offset -= boxHeight * 1.5
    axAnnot.set_xlim(genomeStartX, genomeEndX)
    return usedNames


def _addPisaPlot(shearMat, colorSpan, axPisa: AXES_T, diagMode, gridMode, fontsize,
                 fontSizeAxLabel, genomeWindowStart, mini):

    xlen = shearMat.shape[1]
    axStartY = (xlen - shearMat.shape[0]) // 2
    axStopY = axStartY + shearMat.shape[0]
    cmap = COLOR_MAPS.pisaClip

    plotMat = np.array(shearMat)
    plotMat *= math.log10(math.e) * 10
    colorSpan *= math.log10(math.e) * 10
    extent = (0, xlen, axStopY, axStartY)
    pisaCax = axPisa.imshow(plotMat, vmin=-colorSpan, vmax=colorSpan, extent=extent,
                            cmap=cmap, aspect="auto", interpolation="nearest")
    match diagMode:
        case "off":
            pass
        case "on":
            axPisa.plot([0, xlen], [0, xlen], "k--", lw=0.5)
        case "edge":
            axPisa.plot([0, xlen * 0.02], [0, xlen * 0.02], "k-", lw=2.0)
            axPisa.plot([xlen * 0.98, xlen], [xlen * 0.98, xlen], "k-", lw=2.0)
    if not mini:
        axPisa.set_ylabel("Output base coordinate", fontsize=fontSizeAxLabel,
                      fontfamily=FONT_FAMILY, labelpad=-5)
    numYTicks = 4 if mini else 10
    ticksY, tickLabelsY = getCoordinateTicks(genomeWindowStart,
                      genomeWindowStart + shearMat.shape[0], numYTicks, True)
    ticksY = [x + axStartY for x in ticksY]
    axPisa.set_yticks(ticksY, tickLabelsY, fontsize=fontsize, fontfamily=FONT_FAMILY)
    match gridMode:
        case "on":
            axPisa.grid()
        case "off":
            pass
    return pisaCax


def _addCbar(pisaCax, axCbar: AXES_T, fontsize, mini):
    cbar = plt.colorbar(mappable=pisaCax, cax=axCbar)
    bottom, top = axCbar.get_ylim()
    axCbar.set_yticks(cbar.get_ticks(), [f"{x:0.1f}" for x in cbar.get_ticks()],
                      fontsize=fontsize, fontfamily=FONT_FAMILY)
    axCbar.set_ylim(bottom, top)
    if mini:
        axCbar.set_xlabel("PISA\neffect\n(dBr)", fontsize=fontsize, fontfamily=FONT_FAMILY)
    else:
        axCbar.set_xlabel("PISA effect\n(dBr)", fontsize=fontsize, fontfamily=FONT_FAMILY)


def _addLegend(usedNames, axLegend, fontsize):
    offset = 1
    for name, color in usedNames.items():
        axLegend.fill([0, 0, 1, 1],
                      [offset, offset + 1, offset + 1, offset],
                      color=color)
        axLegend.text(0.5, offset + 0.5, name, fontstyle="italic",
                      fontsize=fontsize, fontfamily=FONT_FAMILY,
                      ha="center", va="center")
        offset += 2
    axLegend.set_xlim(0, 1)
    axLegend.set_ylim(0, max(5, offset - 1))


def _getPisaGraphAxes(fig, left, bottom, width, height):
    #  |  .   ..  . |  ↑
    #  | .:  .::  : |  | profileHeight
    #  --------------  ↓
    #                   ↕ profileSpace
    #  --------------  ↑
    #  |   |        |  |  graphHeight
    #  |    \       |  |
    #  |     \      |  |
    #  |   nanog    |  |  ↕ annotHeight
    #  |      |     |  |   ↕ annotOffset
    #  --------------  ↓
    #                    ↕ seqSpace
    #  ___C_aTCa_____  ↕ seqHeight
    # 105          131
    graphλ = 8
    profileλ = 1
    seqλ = 1
    offsetFracAnnot = 0.01  # Relative to the height of the graph, not bbox.
    heightFracAnnot = 0.2  # ditto.
    spacingλ = 0.05

    # Now we get the heights.
    totalλ = graphλ + profileλ + seqλ + 2 * spacingλ
    δ = height / totalλ  # This gives us the units for positioning.
    seqBase = 0
    seqHeight = δ * seqλ
    space = δ * spacingλ
    graphBase = seqBase + seqHeight + space
    graphHeight = δ * graphλ
    profileBase = graphBase + graphHeight + space
    profileHeight = δ * profileλ
    annotBase = graphBase + graphHeight * offsetFracAnnot
    annotHeight = graphHeight * heightFracAnnot

    axSeq = fig.add_axes([left, bottom + seqBase, width, seqHeight])
    axGraph = fig.add_axes([left, bottom + graphBase, width, graphHeight])
    axProfile = fig.add_axes([left, bottom + profileBase,
                              width, profileHeight])

    axAnnot = fig.add_axes([left, bottom + annotBase, width, annotHeight])
    axAnnot.set_axis_off()
    axGraph.set_yticks([])
    axProfile.set_xticks([])
    axSeq.set_yticks([])
    axAnnot.set_ylim(0, -1)
    return axGraph, axSeq, axProfile, axAnnot


def _addHorizontalProfilePlot(values, colors, seq, genomeStartX, genomeEndX,
                              axSeq, axGraph, fontsize, showSequence, labelAxis, mini):
    numXTicks = 4 if mini else 10
    ticksX, tickLabelsX = getCoordinateTicks(genomeStartX, genomeEndX, numXTicks, True)

    axSeq.set_xlim(0, values.shape[0])
    if showSequence:
        # We have a short enough window to draw individual letters.
        seqOhe = utils.oneHotEncode(seq) * 1.0
        for i in range(len(seq)):
            seqOhe[i, :] *= values[i]
        # Draw the letters.
        plotLogo(seqOhe, len(seq), axSeq, colors=colors)
        axSeq.set_ylim(np.min(seqOhe), np.max(seqOhe))
    else:
        # Window span too big - just show a profile.
        for pos, score in enumerate(values):
            axSeq.bar([pos], [score],
                      linewidth=1, facecolor=COLOR_MAPS.parseSpec(colors[pos][seq[pos]]),
                      edgecolor=COLOR_MAPS.parseSpec(colors[pos][seq[pos]]))
        axSeq.plot([0, values.shape[0]], [0, 0], "k--", lw=0.5)
    if labelAxis:
        axSeq.set_xticks(ticksX, tickLabelsX, fontsize=fontsize, fontfamily=FONT_FAMILY)

        axSeq.xaxis.set_tick_params(labelbottom=True, which="major")
        axSeq.set_ylabel("Contrib.\nscore", fontsize=fontsize,
                        fontfamily=FONT_FAMILY, rotation=0, loc="bottom", labelpad=40)
        axSeq.set_xlabel("Input base coordinate", fontsize=fontsize, fontfamily=FONT_FAMILY)
        axGraph.set_xticks(ticksX)
        axGraph.tick_params(axis="x", which="major", length=0, labelbottom=False)


def _addPisaGraph(similarityMat: IMPORTANCE_AR_T, minValue: float, maxValue: float,
                  colorBlocks: list[tuple[int, int, tuple[float, float, float]]],
                  useAnnotationColor: bool,
                  ax: AXES_T):
    """Draw a graph representation of a PISA matrix.

    :param similarityMat: The PISA array, already sheared. It should be square.
    :param minValue: PISA values less than this will not be plotted at all.
    :param maxValue: Values higher than this will be clipped.
    :param ax: The axes to draw on. The xlim and ylim will be clobbered by this function.
    """
    cmap = COLOR_MAPS.pisaClip

    def addLine(xLower, xUpper, value):
        if abs(value) < minValue:
            return False
        if value < 0:
            value += minValue
        elif value > 0:
            value -= minValue
        if value > maxValue:
            value = maxValue
        if value < -maxValue:
            value = -maxValue
        xmax = 1
        turnPoint = 0.5
        verts = [
            (xLower + 0.5, 0.),
            (xLower + 0.5, xmax * turnPoint),
            (xUpper + 0.5, xmax * turnPoint),
            (xUpper + 0.5, xmax)]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4]
        path = Path(verts, codes)
        normValue = (value + maxValue) / (2 * maxValue)
        normα = abs(value / maxValue)

        color = cmap(normValue, alpha=normα)
        if useAnnotationColor:
            for colorBlock in colorBlocks:
                start, end, colorSpec = colorBlock
                r, g, b = COLOR_MAPS.parseSpec(colorSpec)[:3]
                if start <= xLower < end:
                    color = (r, g, b, normα)
                    break

        curPatch = patches.PathPatch(path, facecolor="none", lw=1,
                                     edgecolor=color)
        return (abs(value), curPatch)
    patchList = []

    for row in logUtils.wrapTqdm(range(similarityMat.shape[0]), "DEBUG"):
        for col in range(similarityMat.shape[1]):
            if curPatch := addLine(col, row, similarityMat[row, col]):
                patchList.append(curPatch)
    psSorted = sorted(patchList, key=lambda x: x[0])
    for p in logUtils.wrapTqdm(psSorted, "DEBUG"):
        ax.add_patch(p[1])
    ax.set_xlim(0, np.max(similarityMat.shape))
    ax.set_ylim(0, 1)

# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
