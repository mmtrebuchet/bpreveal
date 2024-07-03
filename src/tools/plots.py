"""Old plotting tools. Deprecated.

    .. warning::
        This module is deprecated and will be removed in 6.0.0.
"""
import math
import numpy as np
import h5py
import pysam
import pyBigWig
import pybedtools

# You must install bpreveal with conda develop in order to import bpreveal tools.
from bpreveal.internal.constants import IMPORTANCE_AR_T, IMPORTANCE_T, PRED_AR_T, ONEHOT_AR_T
from bpreveal import utils
from bpreveal import motifUtils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.axes import Axes as AXES_T
import matplotlib.colors as mplcolors
from bpreveal import logUtils
logUtils.warning("The tools/plots.py module is deprecated and will be retired in "
        "BPReveal 6.0.0. The new plotting module in main BPReveal maintained and "
        "much better-organized. This module will not be maintained and will "
        "likely break as new features are added.")
_seqCmap = {"A": (0, 158, 115), "C": (0, 114, 178), "G": (240, 228, 66), "T": (213, 94, 0)}

cmapIbm = [[100, 143, 255],
           [120, 94, 240],
           [220, 38, 127],
           [254, 97, 0],
           [255, 176, 0]]

cmapTolRgb = [[51, 34, 136],
              [17, 119, 51],
              [68, 170, 153],
              [136, 204, 238],
              [221, 204, 119],
              [204, 102, 119],
              [170, 68, 153],
              [136, 34, 85]]

cmapTol = [[y / 256 for y in x] for x in cmapTolRgb]

cmapTolLightRgb = [
    [153, 221, 255],
    [68, 187, 153],
    [187, 204, 51],
    [238, 221, 136],
    [238, 136, 102],
    [255, 170, 187],
    [221, 221, 221]]

cmapTolLight = [[y / 256 for y in x] for x in cmapTolLightRgb]

PROFILE_COLOR = cmapTol[0]


def plotPisaGraphWithFiles(pisaDats: str | IMPORTANCE_AR_T, cutMiddle: int, cutLength: int,
                           receptiveField: int, genomeWindowStart: int,
                           genomeWindowChrom: str, genomeFastaFname: str, importanceBwFname: str,
                           motifScanBedFname: str, profileBwFname: str,
                           nameColors: dict[str, tuple[float, float, float]],
                           useAnnotationColor: bool,
                           fig: matplotlib.figure.Figure, bbox: tuple[float, float, float, float],
                           minValue: float, maxValue: float, fontsize: int = 5):
    """Given the names of files, make a pisa graph plot.

    :param pisaDats: Either a string naming an hdf5 file or an array from loadPisa.
    :param cutMiddle: The midpoint of the pisa plot, relative to the start of the profile.
    :param cutLength: How wide is the window you want plotted?
        If 99 or less, a sequence will be plotted.
    :param receptiveField: What is the model's receptive field?
    :param genomeWindowStart: Where in the genome does pisaDats start?
        This is used to generate the x axis.
    :param genomeWindowChrom: What chromosome is the sequence on?
    :param genomeFastaFname: Name of the fasta file containing the genome.
    :param importanceBwFname: The bigwig of importance scores from interpretFlat
    :param motifScanBedFname: The bed file containing mapped motifs.
    :param profileBwFname: The bigwig file containing predicted profile.
    :param nameColors: A dict containing the color to be used for each motif name.
        If a motif is encountered that is not in this dict, then it is added with a color
        taken from Paul Tol's bright palette. This dict is mutated when names are added,
        so you can use one dict for many PISA plots in order to have a consistent
        color scheme.
    :param useAnnotationColor: If True, then lines starting from a region within the annotations
        will be colored according to that annotation's color. If False, then all lines
        will use the RdBu_r color scheme that is used for pisa plots.
    :param fig: The matplotlib figure to put this plot on.
    :param bbox: The bounding box to use for drawing the figure. Lets you put multiple
        pisa plots on a single matplotlib Figure.
    :param minValue: PISA values less than this will not be plotted.
    :param maxValue: PISA values above this will be clipped.
    :param fontsize: How big should the font be?
    :return: Same as plotPisaGraph

    This is just a wrapper around plotPisaGraph that loads data in from named files.
    """
    impScores = _loadPisaImportance(cutMiddle, cutLength, importanceBwFname,
                                   genomeWindowStart, genomeWindowChrom)

    seq = _loadPisaSequence(cutMiddle, cutLength, genomeWindowStart, genomeWindowChrom,
                           genomeFastaFname)
    annotations = _loadPisaAnnotations(cutMiddle, cutLength, genomeWindowStart,
                                      genomeWindowChrom, motifScanBedFname, nameColors)
    profile = _loadPisaProfile(cutMiddle, cutLength, genomeWindowStart,
                              genomeWindowChrom, profileBwFname)
    profile = np.abs(profile)
    return plotPisaGraph(pisaDats=pisaDats, cutMiddle=cutMiddle, cutLength=cutLength,
                         receptiveField=receptiveField, genomeWindowStart=genomeWindowStart,
                         seq=seq, impScores=impScores, annotations=annotations,
                         useAnnotationColor=useAnnotationColor,
                         profile=profile, fig=fig, bbox=bbox,
                         minValue=minValue, maxValue=maxValue, fontsize=fontsize)


def plotPisaGraph(pisaDats: str | IMPORTANCE_AR_T, cutMiddle: int, cutLength: int,
                  receptiveField: int, genomeWindowStart: int, seq: str | None,
                  impScores: IMPORTANCE_AR_T,
                  annotations: tuple[tuple[tuple[int, int], str, tuple[float, float, float]]],
                  useAnnotationColor: bool,
                  profile: PRED_AR_T | None, fig: matplotlib.figure.Figure,
                  bbox: tuple[float, float, float, float], minValue: float,
                  maxValue: float, fontsize: int = 5) -> tuple[AXES_T, AXES_T, AXES_T, AXES_T]:

    """Take PISA data and generate a graph showing connections.

    :param pisaDats: Either a string naming an hdf5 file or a (pre-sheared) PISA array.
    :param cutMiddle: Where, relative to the beginning of the PISA array, do you want the
        graph centered?
    :param cutLength: How wide do you want the graph?
    :param receptiveField: The receptive field of your model. Ignored if pisaDats is
        already an array.
    :param genomeWindowStart: Where does the pisa data start, in genomic coordinates?
    :param seq: The DNA sequence, with length ``cutLength`` and centered at
        ``genomeWindowStart + cutMiddle`` This will be used if ``cutLength < 100``.
    :param impScores: The importance scores to plot on the bottom of the graph. If None,
        then no scores will be plotted, but sequence will be shown if ``cutLength < 100``
        Should have shape (cutLength,)
    :param annotations: The motifs to draw, a list of tuples with structure
        ((start, end), name, (red, green, blue))
    :param useAnnotationColor: If True, then lines starting from a region within the annotations
        will be colored according to that annotation's color. If False, then all lines
        will use the RdBu_r color scheme that is used for pisa plots.
    :param profile: The profile data to plot on top of the graph.
        Should have shape (cutLength,).
    :param fig: The matplotlib Figure to add the plot to.
    :param bbox: (left, bottom, width, height), relative to ``fig``.
        Note that axis labels and such will extend outside of this bounding box.
    :param minValue: PISA values less than this will not be included on the graph plot.
    :param maxValue: PISA values above this will be clipped.
    :param fontsize: What font size should be used for all text?
    :return: (axGraph, axSeq, axProfile, axAnnot)
    """
    logUtils.debug("Starting to draw PISA graph.")
    genomeStart = cutMiddle - cutLength // 2 + genomeWindowStart
    genomeEnd = cutMiddle + cutLength // 2 + genomeWindowStart

    axGraph, axSeq, axProfile, axAnnot = _getPisaGraphAxes(fig, bbox)
    # Pisa image plotting
    match pisaDats:
        case str():
            shearMat = _loadPisa(pisaDats, receptiveField)
        case _:
            # We got an array.
            shearMat = np.copy(pisaDats)
    sliceStart = cutMiddle - cutLength // 2
    sliceEnd = sliceStart + cutLength
    shearMat = shearMat[sliceStart:sliceEnd, sliceStart:sliceEnd]
    colorBlocks = []
    for annot in annotations:
        (start, end), _, (r, g, b) = annot
        colorBlocks.append(((start - genomeStart, end - genomeStart), (r, g, b)))
    logUtils.debug("Axes set. Drawing graph.")
    _addPisaGraph(similarityMat=shearMat, minValue=minValue, maxValue=maxValue,
                  colorBlocks=colorBlocks, useAnnotationColor=useAnnotationColor, ax=axGraph)
    # Now set up the sequence/importance axis.
    logUtils.debug("Graph complete. Finishing plot.")
    _addGraphSequencePlot(impScores, seq, cutLength, genomeStart, genomeEnd,
                        axSeq, axGraph, fontsize, False)

    _addAnnotations(axAnnot, annotations, 0.13, genomeStart, genomeEnd,
                    cutLength, fontsize, False)
    # Now, add the profiles.
    if profile is not None:
        _addHorizontalProfilePlot(profile, cutLength, axProfile, fontsize)
    else:
        axProfile.set_axis_off()
    logUtils.debug("PISA graph plot complete.")
    return (axGraph, axSeq, axProfile, axAnnot)


def plotPisaWithFiles(pisaDats: str | IMPORTANCE_AR_T, cutMiddle: int, cutLengthX: int,
                      cutLengthY: int, receptiveField: int, genomeWindowStart: int,
                      genomeWindowChrom: str, genomeFastaFname: str, importanceBwFname: str,
                      motifScanBedFname: str, profileDats: str,
                      nameColors: dict[str, tuple[float, float, float]],
                      fig: matplotlib.figure.Figure, bbox: tuple[float, float, float, float],
                      colorSpan: float = 1.0, boxHeight: float = 0.1, fontsize: int = 5,
                      mini: bool = False):
    """Given the names of files, make a pisa plot.

    :param pisaDats: Either a string naming an hdf5 file or an array from loadPisa.
    :param cutMiddle: The midpoint of the pisa plot, relative to the start of the profile.
    :param cutLengthX: How wide should the X axis be? If 99 or less, a sequence will be plotted.
    :param cutLengthY: How tall should the plot be?
    :param receptiveField: What is the model's receptive field?
    :param genomeWindowStart: Where in the genome does pisaDats start?
        This is used to generate the x axis.
    :param genomeWindowChrom: What chromosome is the sequence on?
    :param genomeFastaFname: Name of the fasta file containing the genome.
    :param importanceBwFname: The bigwig of importance scores from interpretFlat
    :param motifScanBedFname: The bed file containing mapped motifs.
    :param profileDats: The bigwig file containing predicted profile.
    :param nameColors: A dict containing the color to be used for each motif name.
        If a motif is encountered that is not in this dict, then it is added with a color
        taken from the IBM palette.
    :param fig: The matplotlib figure to put this plot on.
    :param bbox: The bounding box to use for drawing the figure. Lets you put multiple
        pisa plots on a single matplotlib Figure.
    :param colorSpan: What are the maximum and minimum values in the color map.
    :param boxHeight: How tall should the boxes containing motif names be?
    :param fontsize: How big should the font be?
    :param mini: Would you like a smaller plot, suitable for one-column printing?
        Default: False.
    :return: Same as plotPisa
    """
    impScores = _loadPisaImportance(cutMiddle, cutLengthX, importanceBwFname,
                                   genomeWindowStart, genomeWindowChrom)

    seq = _loadPisaSequence(cutMiddle, cutLengthX, genomeWindowStart, genomeWindowChrom,
                           genomeFastaFname)
    annotations = _loadPisaAnnotations(cutMiddle, cutLengthX, genomeWindowStart,
                                      genomeWindowChrom, motifScanBedFname, nameColors)
    profile = _loadPisaProfile(cutMiddle, cutLengthY, genomeWindowStart,
                              genomeWindowChrom, profileDats)
    profile = np.abs(profile)
    return plotPisa(pisaDats=pisaDats, cutMiddle=cutMiddle, cutLengthX=cutLengthX,
                    cutLengthY=cutLengthY, receptiveField=receptiveField,
                    genomeWindowStart=genomeWindowStart, seq=seq, impScores=impScores,
                    annotations=annotations, profile=profile, nameColors=nameColors,
                    fig=fig, bbox=bbox, colorSpan=colorSpan, boxHeight=boxHeight,
                    fontsize=fontsize, mini=mini)


def plotPisa(pisaDats: str | IMPORTANCE_AR_T, cutMiddle: int, cutLengthX: int,
             cutLengthY: int, receptiveField: int, genomeWindowStart: int,
             seq: str | None, impScores: IMPORTANCE_AR_T,
             annotations: tuple[tuple[tuple[int, int], str, tuple[float, float, float]]],
             profile: PRED_AR_T,
             nameColors: dict[str, tuple[float, float, float]],
             fig: matplotlib.figure.Figure, bbox: tuple[float, float, float, float],
             colorSpan: float = 1.0, boxHeight: float = 0.1, fontsize: int = 5,
             showGrid: bool = True, showDiag: bool = True,
             mini: bool = False):
    """Given the actual vectors to show, make a pretty pisa plot.

    :param pisaDats: Either a string naming an hdf5 file, or an array from loadPisa.
    :param cutMiddle: Where should the midpoint of the plot be, relative to the pisaDats array?
    :param cutLengthX: How wide should the plot be?
    :param cutLengthY: How tall should the plot be?
    :param receptiveField: What is the model's receptive field?
    :param genomeWindowStart: Where in the genome does this sequence start?
    :param seq: The sequence of the region.
    :param impScores: The importance scores.
    :param annotations: A list of annotations, containing ((start, stop), name, color).
    :param profile: A vector containing profile information.
    :param nameColors: A dict mapping motif name to color. This is ignored.
    :param fig: The matplotlib Figure onto which the plot should be drawn.
    :param bbox: The bounding box on the figure that will be used.
    :param colorSpan: The limit of the color scale.
    :param boxHeight: How tall should the motif name boxes be?
    :param fontsize: How large should the font be?
    :param showGrid: Should the grid be plotted? Default: True
    :param showDiag: Should a dotted line be plotted along the diagonal?
        Default: True
    :param mini: Should a small-scale plot be made? This shrinks down
        the border elements for small printing and display.
    :return: (axPisa, axSeq, axProfile, nameColors, axCbar)
    """
    fontSizeAxLabel = fontsize * 1.5
    genomeStartX = cutMiddle - cutLengthX // 2 + genomeWindowStart
    genomeEndX = cutMiddle + cutLengthX // 2 + genomeWindowStart

    axPisa, axSeq, axProfile, axCbar, axAnnot, axLegend = _getPisaAxes(fig, bbox, mini)
    # Pisa image plotting

    pisaCax = _addPisaPlot(pisaDats, receptiveField, cutMiddle, cutLengthX, cutLengthY,
                 colorSpan, axPisa, showDiag, showGrid, fontsize,
                 fontSizeAxLabel, genomeWindowStart, mini)
    # Now set up the sequence/importance axis.
    _addSequencePlot(impScores, seq, cutLengthX, genomeStartX, genomeEndX,
                     axSeq, axPisa, fontsize, fontSizeAxLabel, mini)

    # Now it's time to add annotations from the motif scanning step.
    usedNames = _addAnnotations(axAnnot, annotations, boxHeight, genomeStartX,
                    genomeEndX, cutLengthX, fontsize, mini)

    # Now, add the profiles.
    if profile is not None:
        _addProfilePlot(profile, cutLengthY, axProfile, fontsize, fontSizeAxLabel, mini)
    else:
        axProfile.set_axis_off()
    if mini:
        _addLegend(nameColors, axLegend, fontsize, usedNames)
    _addCbar(pisaCax, axCbar, fontsize, mini)

    return (axPisa, axSeq, axProfile, nameColors, axCbar)


def plotModiscoPattern(pattern: motifUtils.Pattern, fig: matplotlib.figure.Figure, sortKey=None):
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
    plotLogo(cwm, cwm.shape[0], axLogo)
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
        colorFix = cmapTol[0]
        axCurStat.plot(statSort, yvals, ".", color=colorFix, alpha=0.5)
        statFilt = np.sort(statSort)
        colorFix = cmapTol[2]
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
            color = _seqCmap[baseName][colorIdx]
            for col in range(hmap.shape[1]):
                for colOffset in range(upsamplingFactor):
                    writeCol = col * upsamplingFactor + colOffset
                    displayAr[:, writeCol, colorIdx] += ar[:, col] * color / 256
    ax.imshow(displayAr, aspect='auto', interpolation='antialiased', interpolation_stage='data')


def plotLogo(values: PRED_AR_T, width: float, ax, colors="seq",
             spaceBetweenLetters: float = 0) -> None:
    """A convenience function to plot an array of sequence data (like a pwm)
    on a matplotlib axes object.

    Arguments:
    values is an (N,4) array of sequence data. This could be, for example,
    a pwm or a one-hot encoded sequence.

    width is the width of the total logo, useful for aligning axis labels.

    ax is a matplotlib axes object on which the logo will be drawn.

    Colors, if provided, can have several meanings:
        1. Give an explicit rgba color for each base.
            colors should be an array of shape (N, 4, 4), where the first dimension is the
            sequence position, the second is the base (A, C, G, T, in that order), and the
            third gives an rgba color to use for that base at that position.
        2. Give a color for each base type. In this case, colors will be a dict of tuples:
            {"A": (220, 38, 127), "C": (120, 94, 240), "G": (254, 97, 0), "T": (255, 176, 0)}
            This will make each instance of a particular base have the same color.
        3. Give a matplotlib colormap and a min and max value. Each base will be colored
            based on its magnitude. For example, to highlight bases with large negative values,
            you might specify ('Blues_r', -2, 0) to draw all bases with negative scores as blue,
            and bases with less negative colors will be drawn lighter. Bases with scores outside
            the limits you provide will be clipped to the limit values.
        4. The string 'seq' means A will be drawn green, C will be blue, G will be orange,
            and T will be red. These colors are drawn from a colorblind-aware palette.
    """
    from matplotlib.transforms import Bbox, Affine2D
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch

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

    def getColor(pos, base, value):
        match colors:
            case np.ndarray():
                # Colors is an array. Just index it.
                getBase = {"A": 0, "C": 1, "G": 2, "T": 3}
                return colors[pos, getBase[base]]
            case "seq":
                rgb = _seqCmap[base]
            case (name, vmin, vmax):
                cmap = mplcolors.Colormap(name)
                if value < vmin:
                    value = vmin
                if value > vmax:
                    value = vmax
                Δv = vmax - vmin
                return cmap((vmin + value) * Δv)
            case dict():
                rgb = colors[base]
            case _:
                raise ValueError()
        return np.array(rgb) / 256.

    for predIdx in range(values.shape[0]):
        A, C, G, T = values[predIdx]
        lettersToDraw = [("A", A), ("C", C), ("G", G), ("T", T)]
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
                       color=getColor(predIdx, nl[0], nl[1]), ax=ax, flip=True)
            top = base

        # Draw the positive letters.
        base = 0
        for pl in posLetters:
            top = base + pl[1]
            left = predIdx * width / values.shape[0] + spaceBetweenLetters / 2
            right = left + width / values.shape[0] - spaceBetweenLetters
            _drawLetter(pl[0], left=left, right=right, bottom=base, top=top,
                       color=getColor(predIdx, pl[0], pl[1]), ax=ax)
            base = top


def getCoordinateTicks(start: int, end: int, numTicks: int,
                       zeroOrigin: bool) -> tuple[list[float], list[str]]:
    """Given a start and end coordinate, return x-ticks that should be used for plotting.
    Given a start and end coordinate, return a list of ticks and tick labels that
    1. include exactly the start and stop coordinates
    2. Contain approximately numTicks positions and labels.
    3. Try to fall on easy multiples 1, 2, and 5 times powers of ten.
    4. Are formatted to reduce redundant label noise by omitting repeated initial digits.
    """
    import matplotlib.ticker as mt
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
    multiLoc = mt.MultipleLocator(tickWidth)
    multiLoc.view_limits(start, end)
    innerTickPoses = multiLoc.tick_values(start, end)
    while innerTickPoses[0] < start + tickWidth / 2:
        innerTickPoses = innerTickPoses[1:]
    while len(innerTickPoses) and innerTickPoses[-1] > end - tickWidth / 2:
        innerTickPoses = innerTickPoses[:-1]
    tickPoses = [float(x) for x in [start] + list(innerTickPoses) + [end]]
    tickLabelStrs = ["{0:,}".format(int(x)) for x in tickPoses]
    if zeroOrigin:
        tickPoses = [x - start for x in tickPoses]
    tickLabels = _massageTickLabels(tickLabelStrs)
    if reverse:
        tickPoses = tickPoses[::-1]
        tickLabels = tickLabels[::-1]
    return tickPoses, tickLabels


def _massageTickLabels(labelList):
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


def _loadPisa(fname, receptiveField) -> IMPORTANCE_AR_T:
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


def _loadPisaImportance(cutMiddle, cutLengthX, importanceBwFname,
                       genomeWindowStart, genomeWindowChrom) -> IMPORTANCE_AR_T:
    cutStartX = cutMiddle - cutLengthX // 2
    if importanceBwFname is not None:
        impFp = pyBigWig.open(importanceBwFname)
        impScores = np.nan_to_num(impFp.values(genomeWindowChrom,
                                               genomeWindowStart + cutStartX,
                                               genomeWindowStart + cutStartX + cutLengthX))
        impFp.close()
    else:
        impScores = np.ones((cutLengthX,), dtype=IMPORTANCE_T)
    return impScores


def _loadPisaSequence(cutMiddle, cutLengthX, genomeWindowStart, genomeWindowChrom,
                     genomeFastaFname) -> str | None:
    genomeStartX = cutMiddle - cutLengthX // 2 + genomeWindowStart
    genomeEndX = cutMiddle + cutLengthX // 2 + genomeWindowStart
    if genomeFastaFname is not None:
        with pysam.FastaFile(genomeFastaFname) as genome:
            seq = genome.fetch(genomeWindowChrom, genomeStartX, genomeEndX)
    else:
        seq = None
    return seq


def _loadPisaAnnotations(cutMiddle, cutLengthX, genomeWindowStart, genomeWindowChrom,
                        motifScanBedFname, nameColors):
    genomeStartX = cutMiddle - cutLengthX // 2 + genomeWindowStart
    genomeEndX = cutMiddle + cutLengthX // 2 + genomeWindowStart
    readHead = 0
    annotations = []
    if motifScanBedFname is not None:
        bedFp = pybedtools.BedTool(motifScanBedFname)
        for line in bedFp:
            if line.chrom == genomeWindowChrom and line.end > genomeStartX\
                    and line.start < genomeEndX:
                if line.name not in nameColors:
                    nameColors[line.name] = \
                        np.array(cmapTolLight[len(nameColors) % len(cmapTolLight)])
                    readHead += 1
                if line.start < genomeStartX:
                    line.start = genomeStartX
                if line.end > genomeEndX:
                    line.end = genomeEndX
                annotations.append(((line.start, line.end), line.name, nameColors[line.name]))
    return tuple(annotations)


def _loadPisaProfile(cutMiddle, cutLengthY, genomeWindowStart,
                    genomeWindowChrom, profileDats):
    cutStartY = cutMiddle - cutLengthY // 2
    if profileDats is not None:
        profileFp = pyBigWig.open(profileDats)
        profile = np.nan_to_num(
            profileFp.values(genomeWindowChrom,
                             genomeWindowStart + cutStartY,
                             genomeWindowStart + cutStartY + cutLengthY))
        profileFp.close()
        return profile
    else:
        return np.ones((cutLengthY))


def _getPisaAxes(fig, bbox, mini) -> tuple[AXES_T, AXES_T,
        AXES_T, AXES_T, AXES_T, AXES_T | None]:
    l, b, w, h = bbox
    xweightPisa = 40
    xweightProfile = 6
    xweightCbar = 3 if mini else 1
    widthScale = 1
    pisaWidth = w * xweightPisa / (xweightPisa + xweightProfile + xweightCbar) * widthScale
    profileWidth = w * xweightProfile / (xweightPisa + xweightProfile + xweightCbar) * widthScale
    cbarWidth = w * xweightCbar / (xweightPisa + xweightProfile + xweightCbar) * widthScale
    pisaHeight = h * 7 / 8
    seqHeight = h / 8

    axPisa = fig.add_axes([l, b + seqHeight, pisaWidth, pisaHeight])
    axSeq = fig.add_axes([l, b, pisaWidth, seqHeight])
    axProfile = fig.add_axes([l + pisaWidth + profileWidth * 0.02,
                              b + seqHeight, profileWidth * 0.9, pisaHeight])
    axCbar = fig.add_axes([l + pisaWidth + profileWidth,
                           b + seqHeight + pisaHeight / (8 if mini else 4),
                           cbarWidth, pisaHeight / (3 if mini else 2)])
    axLegend = None
    if mini:
        axLegend = fig.add_axes([l + pisaWidth + profileWidth,
                            b + seqHeight + pisaHeight * (1 / 3 + 1 / 7),
                            cbarWidth * 3, pisaHeight * (1 - 1 / 3 - 1 / 7)])
        axLegend.set_axis_off()
    axAnnot = fig.add_axes([l, b + seqHeight + 2 * pisaHeight / 3, pisaWidth, pisaHeight / 3])
    axAnnot.set_axis_off()

    axSeq.set_frame_on(False)
    axSeq.set_yticks([])
    axAnnot.set_ylim(-1, 0)
    axAnnot.set_axis_off()
    axProfile.set_yticks([])
    axProfile.set_frame_on(False)
    axProfile.yaxis.set_visible(False)
    return axPisa, axSeq, axProfile, axCbar, axAnnot, axLegend


def _addProfilePlot(profile, cutLengthY, axProfile, fontsize, fontSizeAxLabel, mini):
    plotProfile = list(profile)
    plotProfile.append(plotProfile[-1])
    axProfile.fill_betweenx(range(cutLengthY, -1, -1), plotProfile,
                            step="post", color=PROFILE_COLOR)
    axProfile.set_ylim(0, cutLengthY)
    axProfile.set_xlim(0, np.max(profile))
    if mini:
        axProfile.set_xticks([])
        axProfile.xaxis.set_visible(False)
    else:
        profileXticks = axProfile.get_xticks()
        if max(profileXticks) > np.max(profile) * 1.01:
            profileXticks = profileXticks[:-1]
        axProfile.set_xticks(profileXticks, profileXticks, fontsize=fontsize)
        axProfile.set_xlabel("Profile", fontsize=fontSizeAxLabel, fontfamily="serif")


def _addAnnotations(axAnnot, annotations, boxHeight, genomeStartX,
                    genomeEndX, cutLengthX, fontsize, mini):
    offset = -boxHeight * 1.3
    lastR = 0
    usedNames = []
    for annot in sorted(annotations, key=lambda x: x[0][0]):
        aleft, aright = annot[0]
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
                     label=annot[1], color=annot[2])
        if not mini:
            axAnnot.text((aleft + aright) / 2, offset + boxHeight / 2, annot[1],
                     fontstyle="italic", fontsize=fontsize, fontfamily="serif",
                     ha="center", va="center")
        offset -= boxHeight * 1.5
        usedNames.append(annot[1])
    axAnnot.set_xlim(genomeStartX, genomeEndX)
    return usedNames


def _addSequencePlot(impScores, seq, cutLengthX, genomeStartX, genomeEndX,
                     axSeq, axPisa, fontsize, fontSizeAxLabel, mini):
    numXTicks = 4 if mini else 10
    ticksX, tickLabelsX = getCoordinateTicks(genomeStartX, genomeEndX, numXTicks, True)

    axSeq.set_xlim(0, impScores.shape[0])
    if seq is not None and cutLengthX < 100:
        seqOhe = utils.oneHotEncode(seq) * 1.0
        for i in range(len(seq)):
            seqOhe[i, :] *= impScores[i]
        # Draw the letters.
        plotLogo(seqOhe, len(seq), axSeq, colors="seq")
        axSeq.set_ylim(np.min(seqOhe), np.max(seqOhe))
    else:
        axSeq.bar(range(len(impScores)), impScores, linewidth=1, edgecolor=PROFILE_COLOR)
        axSeq.plot([0, impScores.shape[0]], [0, 0], "k--", lw=0.5)

    axSeq.set_xticks(ticksX, tickLabelsX, fontsize=fontsize, fontfamily='serif')

    axSeq.xaxis.set_tick_params(labelbottom=True, which="major")
    if not mini:
        axSeq.set_ylabel("Contrib.\nscore", fontsize=fontSizeAxLabel,
                     fontfamily="serif", rotation=0, loc="bottom", labelpad=40)
        axSeq.set_xlabel("Input base coordinate", fontsize=fontSizeAxLabel, fontfamily="serif")
    if seq is None and np.sum(impScores) == len(impScores):
        # We got neither profile data nor sequence information.
        axSeq.set_axis_off()
        if not mini:
            axPisa.set_ylabel("Input base coordinate", fontsize=fontSizeAxLabel, fontfamily="serif")
        axPisa.set_xticks(ticksX, tickLabelsX, fontsize=fontsize)
    else:
        # First of all, turn off the pisa axis x ticks and labels.
        # (But don't set_xticks([]), because the grid is based on the ticks.)
        axPisa.set_xticks(ticksX)
        axPisa.tick_params(axis="x", which="major", length=0, labelbottom=False)


def _addPisaPlot(pisaDats, receptiveField, cutMiddle, cutLengthX, cutLengthY,
                 colorSpan, axPisa: AXES_T, showDiag, showGrid, fontsize,
                 fontSizeAxLabel, genomeWindowStart, mini):

    match pisaDats:
        case str():
            shearMat = _loadPisa(pisaDats, receptiveField)
        case _:
            # We got an array.
            shearMat = np.copy(pisaDats)

    axStartY = (cutLengthX - cutLengthY) // 2
    axStopY = axStartY + cutLengthY
    oldCmap = mpl.colormaps["RdBu_r"].resampled(256)
    newColors = oldCmap(np.linspace(0, 1, 256))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    green = np.array([24 / 256, 248 / 256, 148 / 256, 1])
    colorLimit = 8 if mini else 5
    newColors[:colorLimit] = green
    newColors[-colorLimit:] = pink
    cmap = mplcolors.ListedColormap(newColors)

    cutStartX = cutMiddle - cutLengthX // 2
    cutStartY = cutMiddle - cutLengthY // 2
    plotMat = shearMat[cutStartY:cutStartY + cutLengthY,
                       cutStartX:cutStartX + cutLengthX]
    plotMat *= math.log10(math.e) * 10
    colorSpan *= math.log10(math.e) * 10
    extent = (0, cutLengthX, axStopY, axStartY)
    pisaCax = axPisa.imshow(plotMat, vmin=-colorSpan, vmax=colorSpan, extent=extent,
                            cmap=cmap, aspect="auto", interpolation="nearest")
    if showDiag:
        axPisa.plot([0, cutLengthX], [0, cutLengthX], "k--", lw=0.5)
    if not mini:
        axPisa.set_ylabel("Output base coordinate", fontsize=fontSizeAxLabel,
                      fontfamily="serif", labelpad=-5)
    numYTicks = 4 if mini else 10
    ticksY, tickLabelsY = getCoordinateTicks(genomeWindowStart + cutStartY,
                      genomeWindowStart + cutStartY + cutLengthY, numYTicks, True)
    ticksY = [x + axStartY for x in ticksY]
    axPisa.set_yticks(ticksY, tickLabelsY, fontsize=fontsize, fontfamily='serif')
    if showGrid:
        axPisa.grid()
    return pisaCax


def _addCbar(pisaCax, axCbar: AXES_T, fontsize, mini):
    cbar = plt.colorbar(mappable=pisaCax, cax=axCbar)
    bottom, top = axCbar.get_ylim()
    axCbar.set_yticks(cbar.get_ticks(), ["{0:0.1f}".format(x)
                      for x in cbar.get_ticks()], fontsize=fontsize, fontfamily='serif')
    axCbar.set_ylim(bottom, top)
    if mini:
        axCbar.set_xlabel("PISA\neffect\n(dBr)", fontsize=fontsize, fontfamily="serif")
    else:
        axCbar.set_xlabel("PISA effect\n(dBr)", fontsize=fontsize, fontfamily="serif")


def _addLegend(nameColors, axLegend, fontsize, usedNames):
    offset = 1
    for name, color in nameColors.items():
        if name in usedNames:
            axLegend.fill([0, 0, 1, 1],
                          [offset, offset + 1, offset + 1, offset],
                          color=color)
            axLegend.text(0.5, offset + 0.5, name, fontstyle="italic",
                          fontsize=fontsize, fontfamily="serif",
                          ha="center", va="center")
            offset += 2
    axLegend.set_xlim(0, 1)
    axLegend.set_ylim(0, max(5, offset - 1))


def _getPisaGraphAxes(fig, bbox):
    l, b, w, h = bbox
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
    #
    graphλ = 8
    profileλ = 1
    seqλ = 1
    offsetFracAnnot = 0.01  # Relative to the height of the graph, not bbox.
    heightFracAnnot = 0.2  # ditto.
    spacingλ = 0.05

    # Now we get the heights.
    totalλ = graphλ + profileλ + seqλ + 2 * spacingλ
    δ = h / totalλ  # This gives us the units for positioning.
    seqBase = 0
    seqHeight = δ * seqλ
    space = δ * spacingλ
    graphBase = seqBase + seqHeight + space
    graphHeight = δ * graphλ
    profileBase = graphBase + graphHeight + space
    profileHeight = δ * profileλ
    annotBase = graphBase + graphHeight * offsetFracAnnot
    annotHeight = graphHeight * heightFracAnnot

    axSeq = fig.add_axes([l, b + seqBase, w, seqHeight])
    axGraph = fig.add_axes([l, b + graphBase, w, graphHeight])
    axProfile = fig.add_axes([l, b + profileBase,
                              w, profileHeight])

    axAnnot = fig.add_axes([l, b + annotBase, w, annotHeight])
    axAnnot.set_axis_off()
    axGraph.set_yticks([])
    axProfile.set_xticks([])
    axSeq.set_yticks([])
    axAnnot.set_ylim(0, -1)
    return axGraph, axSeq, axProfile, axAnnot


def _addGraphSequencePlot(impScores, seq, cutLengthX, genomeStartX, genomeEndX,
                     axSeq, axGraph, fontsize, mini):
    numXTicks = 4 if mini else 10
    ticksX, tickLabelsX = getCoordinateTicks(genomeStartX, genomeEndX, numXTicks, True)
    if seq is None and impScores is None:
        # We actually didn't want this plot.
        axGraph.set_xticks(ticksX, tickLabelsX, fontsize=fontsize, fontfamily="serif")
        axGraph.set_ylabel("Input base coordinate", fontsize=fontsize, fontfamily="serif")
        axSeq.set_axis_off()
        return

    axSeq.set_xlim(0, impScores.shape[0])
    if seq is not None and cutLengthX < 100:
        # We have a short enough window to draw individual letters.
        seqOhe = utils.oneHotEncode(seq) * 1.0
        for i in range(len(seq)):
            seqOhe[i, :] *= impScores[i]
        # Draw the letters.
        plotLogo(seqOhe, len(seq), axSeq, colors="seq")
        axSeq.set_ylim(np.min(seqOhe), np.max(seqOhe))
    else:
        # Window span too big - just show a profile.
        axSeq.bar(range(len(impScores)), impScores, linewidth=1, edgecolor=PROFILE_COLOR)
        axSeq.plot([0, impScores.shape[0]], [0, 0], "k--", lw=0.5)

    axSeq.set_xticks(ticksX, tickLabelsX, fontsize=fontsize, fontfamily='serif')

    axSeq.xaxis.set_tick_params(labelbottom=True, which="major")
    axSeq.set_ylabel("Contrib.\nscore", fontsize=fontsize,
                    fontfamily="serif", rotation=0, loc="bottom", labelpad=40)
    axSeq.set_xlabel("Input base coordinate", fontsize=fontsize, fontfamily="serif")
    axGraph.set_xticks(ticksX)
    axGraph.tick_params(axis="x", which="major", length=0, labelbottom=False)


def _addHorizontalProfilePlot(profile, cutLength, axProfile, fontsize):
    plotProfile = list(profile)
    plotProfile.append(plotProfile[-1])
    axProfile.fill_between(range(0, cutLength + 1),
                           plotProfile, step="pre", color=PROFILE_COLOR)
    axProfile.set_ylim(0, np.max(profile))
    axProfile.set_xlim(0, cutLength)
    profileYticks = axProfile.get_yticks()
    axProfile.set_ylabel("Profile", fontsize=fontsize, fontfamily="serif")
    if max(profileYticks) > np.max(profile) * 1.01:
        profileYticks = profileYticks[:-1]
    axProfile.set_yticks(profileYticks, profileYticks, fontsize=fontsize,
                         fontfamily="serif")


def _addPisaGraph(similarityMat: IMPORTANCE_AR_T, minValue: float, maxValue: float,
                  colorBlocks: list[tuple[tuple[int, int], tuple[float, float, float]]],
                  useAnnotationColor: bool,
                  ax: AXES_T):
    """Draw a graph representation of a PISA matrix.

    :param similarityMat: The PISA array, already sheared. It should be square.
    :param minValue: PISA values less than this will not be plotted at all.
    :param maxValue: Values higher than this will be clipped.
    :param ax: The axes to draw on. The xlim and ylim will be clobbered by this function.
    """
    oldCmap = mpl.colormaps["RdBu_r"].resampled(256)
    newColors = oldCmap(np.linspace(0, 1, 256))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    green = np.array([24 / 256, 248 / 256, 148 / 256, 1])
    newColors[:5] = green
    newColors[-5:] = pink
    cmap = mplcolors.ListedColormap(newColors)

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
                (start, end), (r, g, b) = colorBlock
                if start <= xLower < end:
                    color = (r, g, b, normα)
                    break

        curPatch = patches.PathPatch(path, facecolor='none', lw=1,
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
