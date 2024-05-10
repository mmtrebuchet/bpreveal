"""A bunch of helper functions for making plots."""
import math
from typing import Literal
import pysam
import pyBigWig
import pybedtools
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.transforms import Bbox, Affine2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import matplotlib.ticker as mplticker
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.axes import Axes as AXES_T
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mplcolors


from bpreveal import logUtils
from bpreveal.internal.constants import IMPORTANCE_AR_T, PRED_AR_T, FONT_FAMILY, \
    FONT_SIZE_TICKS, FONT_SIZE_LABELS
from bpreveal import utils
from bpreveal.colors import COLOR_SPEC_T, DNA_COLOR_SPEC_T, parseSpec
import bpreveal.colors as bprcolors


def plotLogo(values: PRED_AR_T, width: float, ax: AXES_T,
             colors: DNA_COLOR_SPEC_T | list[DNA_COLOR_SPEC_T],
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
           You can get the default BPReveal color map at :py:class:`~ColorMaps`.
        3. Give a list of colors for each base.
           This will be a list of length ``values.shape[0]`` and each entry
           should be a dictionary in either format
           1 or 2 above. This gives each base its own color palette, useful
           for shading bases by some profile.
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
                return parseSpec(colors[pos][base])
            case dict():
                return parseSpec(colors[base])
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
    tickLabels = massageTickLabels(tickLabelStrs)
    if reverse:
        tickPoses = tickPoses[::-1]
        tickLabels = tickLabels[::-1]
    return tickPoses, tickLabels


def replaceThousands(labelList: list[str]) -> list[str]:
    """If every label ends with ``,000``, replace the last four letters with ``k``."""
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
    return labelsThousands


def massageTickLabels(labelList: list[str]) -> list[str]:
    """Remove identical leading digits from labels."""
    labelsThousands = replaceThousands(labelList)
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


def normalizeProfileSection(oldConfig: dict, newConfig: dict, group: str):
    """Take a raw config dict and populate any defaults and load up bigwig data."""
    # Do we need to load bigwig data?
    newConfig[group]["show-sequence"] = oldConfig[group].get("show-sequence", False)
    if "bigwig-name" in oldConfig[group]:
        vals = loadFromBigwig(oldConfig[group]["bigwig-name"],
                              oldConfig["coordinates"]["genome-window-start"],
                              oldConfig["coordinates"]["genome-window-chrom"],
                              newConfig["pisa"]["values"].shape[1])
        newConfig[group]["values"] = vals
    else:
        newConfig[group]["values"] = np.array(oldConfig[group]["values"])

    # Convert the color spec into a list for each base.
    if newConfig[group]["show-sequence"]:
        backupColor = bprcolors.dnaWong
    else:
        backupColor = bprcolors.defaultProfile

    newConfig[group]["color"] = normalizeProfileColor(
        oldConfig[group].get("color", backupColor),
        len(newConfig[group]["values"]))


def buildConfig(oldConfig: dict) -> dict:
    """Read in a config and add any missing data.

    :param oldConfig: The original configuration dictionary. All entries from this
        original dict are copied, so you can mutate the returned dict without
        messing with the original data.

    This loads in profile and pisa data from files and expands the color specs.
    """
    oldCoords = oldConfig["coordinates"]
    newCoords = {
        "midpoint-offset": oldCoords["midpoint-offset"],
        "input-slice-width": oldCoords["input-slice-width"],
        "output-slice-width": oldCoords["output-slice-width"],
        "genome-window-start": oldCoords["genome-window-start"],
        "genome-window-chrom": oldCoords["genome-window-chrom"]}
    oldFig = oldConfig["figure"]
    newFig = oldFig.copy()  # Bring over (left, bottom, width, height)
    newFig["annotation-height"] = oldFig.get("annotation-height", 0.13)
    newFig["tick-font-size"] = oldFig.get("tick-font-size", FONT_SIZE_TICKS)
    newFig["label-font-size"] = oldFig.get("label-font-size", FONT_SIZE_LABELS)
    newFig["line-width"] = oldFig.get("line-width", 1)

    newConfig = {
        "pisa": {},
        "coordinates": newCoords,
        "predictions": {},
        "importance": {},
        "annotations": {},
        "figure": newFig}

    if "min-value" in oldConfig:
        # We have a graph-style config.
        newConfig["min-value"] = oldConfig["min-value"]
        newConfig["use-annotation-colors"] = oldConfig.get("use-annotation-colors", False)
    else:
        # We have a plot-style config.
        newConfig["miniature"] = oldConfig.get("miniature", False)
        newFig["grid-mode"] = oldFig.get("grid-mode", "on")
        newFig["diagonal-mode"] = oldFig.get("diagonal-mode", "edge")

    # First, the pisa data.
    if "h5-name" in oldConfig["pisa"]:
        # We need to load from file.
        newConfig["pisa"]["values"] = utils.loadPisa(oldConfig["pisa"]["h5-name"])
    else:
        newConfig["pisa"]["values"] = np.array(oldConfig["pisa"]["values"])

    normalizeProfileSection(oldConfig, newConfig, "importance")
    normalizeProfileSection(oldConfig, newConfig, "predictions")

    newConfig["annotations"]["custom"] = oldConfig["annotations"].get("custom", [])
    if "bed-name" in oldConfig["annotations"]:
        nameColors = oldConfig["annotations"].get("name-colors", {})
        newCustom = loadPisaAnnotations(oldConfig["annotations"]["bed-name"],
                                        nameColors,
                                        oldCoords["genome-window-start"],
                                        oldCoords["genome-window-chrom"],
                                        newConfig["pisa"]["values"].shape[1])
        newConfig["annotations"]["name-colors"] = nameColors  # nameColors was mutated!
        newConfig["annotations"]["custom"] = newConfig["annotations"]["custom"] + newCustom

    if "genome-fasta" in oldCoords:
        newCoords["sequence"] = loadSequence(
            oldCoords["genome-fasta"],
            oldCoords["genome-window-start"],
            oldCoords["genome-window-chrom"],
            newConfig["pisa"]["values"].shape[1])
    else:
        newCoords["sequence"] = oldCoords["sequence"]
    return newConfig


def loadFromBigwig(bwFname: str, start: int, chrom: str, length: int) -> PRED_AR_T:
    """Read in the given region from a bigwig file."""
    impFp = pyBigWig.open(bwFname)
    impScores = np.nan_to_num(impFp.values(chrom, start, start + length))
    impFp.close()
    return impScores


def normalizeProfileColor(colorSpec: DNA_COLOR_SPEC_T | COLOR_SPEC_T |  # noqa
                                     list[DNA_COLOR_SPEC_T | COLOR_SPEC_T],  # noqa
                          numItems: int) -> list[DNA_COLOR_SPEC_T]:
    """Take the config color spec and expand it to a list of DNA_COLOR_SPEC_T."""
    match colorSpec:
        case {"A": aColor, "C": cColor, "G": gColor, "T": tColor}:
            a = aColor
            c = cColor
            g = gColor
            t = tColor
        case list():
            ret = []
            for cv in colorSpec:
                if "A" in cv:
                    ret.append(cv)
                else:
                    ret.append({"A": cv, "C": cv, "G": cv, "T": cv})
            return ret
        case _:
            color: COLOR_SPEC_T = colorSpec  # type: ignore
            a = c = g = t = color
    colorDict: DNA_COLOR_SPEC_T = {"A": a, "C": c, "G": g, "T": t}
    return [colorDict] * numItems


def loadSequence(genomeFastaFname: str, genomeWindowStart: int,
                 genomeWindowChrom: str, length: int) -> str:
    """Read a sequence in from a fasta. Uppercases the resulting string."""
    with pysam.FastaFile(genomeFastaFname) as genome:
        seq = genome.fetch(genomeWindowChrom, genomeWindowStart, genomeWindowStart + length)
    return seq.upper()


def loadPisaAnnotations(bedFname: str, nameColors: dict[str, COLOR_SPEC_T],
                        start: int, chrom: str, length: int) -> list[dict]:
    """Load in a bed full of annotations and prepare boxes for ones in this region.

    :param bedFname: The name of the bed file.
    :param nameColors: A dict mapping names onto colorSpecs. Used to determine
        the colors that will be drawn for the annotations. If a name already exists
        in nameColors, use it. If a new name is found in the bed, add it to nameColors,
        drawing from the tolLight palette.
    :param start: Genomic start coordinate.
    :param chrom: Chromosome that the region is on.
    :param length: The length of the region being plotted.
    :return: A list of dicts, structured like
        ``{"start": 1234, "end": 1244, "name": "Abf1", "color": {"tol": 0}``.
    """
    annotations = []
    bedFp = pybedtools.BedTool(bedFname)
    for line in bedFp:
        if line.chrom == chrom and line.end > start\
                and line.start < start + length:
            if line.name not in nameColors:
                nameColors[line.name] = \
                    {"tol-light": len(nameColors) % len(bprcolors.tolLight)}
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


def addVerticalProfilePlot(profile: PRED_AR_T, axProfile: AXES_T,
                           colors: list[DNA_COLOR_SPEC_T], sequence: str,
                           fontsize: int, fontSizeAxLabel: int, mini: bool):
    """Plot a profile on a vertical axes.

    :param profile: The values that should be plotted. This will be an ndarray.
    :param axProfile: The axes to draw the profile on.
    :param colors: A DNA_COLOR_SPEC_T for each base.
    :param sequence: The underlying DNA sequence. Used to determine the colors
        to use.
    :param fontsize: How big do you want the tick labels, in points?
    :param fontSizeAxLabel: How big do you want the word "Profile" on your axes?
    :param mini: If True, then all ticks are removed and the axis is not labeled.
    """
    plotProfile = list(profile)
    for pos, val in enumerate(plotProfile):
        y = len(plotProfile) - pos
        axProfile.fill_betweenx([y, y + 1], val, step="post",
                                color=parseSpec(colors[pos][sequence[pos]]))
    axProfile.set_ylim(0, len(profile))
    axProfile.set_xlim(0, float(np.max(profile)))
    if mini:
        axProfile.set_xticks([])
        axProfile.xaxis.set_visible(False)
    else:
        profileXticks = axProfile.get_xticks()
        if max(profileXticks) > np.max(profile) * 1.01:
            profileXticks = profileXticks[:-1]
        axProfile.set_xticks(profileXticks, profileXticks, fontsize=fontsize,
                             fontfamily=FONT_FAMILY)
        axProfile.set_xlabel("Profile", fontsize=fontSizeAxLabel, fontfamily=FONT_FAMILY)


def addAnnotations(axAnnot: AXES_T, annotations: list[dict], boxHeight: float,
                   genomeStartX: int, genomeEndX: int, fontsize: int,
                   mini: bool) -> dict[str, COLOR_SPEC_T]:
    """Apply the given annotations to the drawing area given by axAnnot.

    :param axAnnot: The matplotlib axes to draw upon. This will usually overlap with
        some other component of the figure.
    :param annotations: A list of dicts of the form given by :py:func:`~loadPisaAnnotations`.
    :param boxHeight: How tall, as a fraction of the height of axAnnot, should the boxes be?
    :param genomeStartX: Where does the x-axis of the annotation axis start,
        in genomic coordinates?
    :param genomeEndX: Where does the annotation axis end, in genomic coordinates?
    :param fontsize: How big do you want the text in the boxes?
    :param mini: If True, then don't write the names of the annotations in the boxes.
    :return: A dict of the names that were actually plotted, mapping each name to its
        colorSpec.
    """
    offset = -boxHeight * 1.3
    lastR = 0
    usedNames = {}
    for annot in sorted(annotations, key=lambda x: x["start"]):
        aleft = annot["start"]
        aright = annot["end"]
        if aright < genomeStartX or aleft > genomeEndX:
            continue
        # No directly abutting annotations - at least 1 base.
        if aleft > lastR + 1:
            offset = -boxHeight * 1.3
        lastR = max(lastR, aright)
        if offset < -1:
            # We're off the page - reset offset and deal with the overlap.
            offset = -boxHeight * 1.3
        axAnnot.fill([aleft, aleft, aright, aright],
                     [offset, boxHeight + offset, boxHeight + offset, offset],
                     label=annot["name"], color=parseSpec(annot["color"]))
        if not mini:
            axAnnot.text((aleft + aright) / 2, offset + boxHeight / 2, annot["name"],
                     fontstyle="italic", fontsize=fontsize, fontfamily=FONT_FAMILY,
                     ha="center", va="center")
        usedNames[annot["name"]] = annot["color"]
        offset -= boxHeight * 1.5
    axAnnot.set_xlim(genomeStartX, genomeEndX)
    return usedNames


def addPisaPlot(shearMat: IMPORTANCE_AR_T, colorSpan: float, axPisa: AXES_T,
                diagMode: Literal["on"] | Literal["off"] | Literal["edge"],
                gridMode: Literal["on"] | Literal["off"], fontsize: int,
                fontSizeAxLabel: int, genomeWindowStart: int, mini: bool) -> ScalarMappable:
    """Plot the pisa data on an axes.

    :param shearMat: The PISA data to actually plot, already sheared and cropped.
    :param colorSpan: The span of the color bar. Note that the colorSpan parameter
        is specified in logit space (which is where PISA data are calculated) but the
        colorbar is shown in dB, which is a more intuitive unit. The color bar will
        therefore NOT stop where your colorSpan does.
    :param axPisa: The axes to draw on.
    :param diagMode: How should the diagonal be drawn? "on" and "off" are self-explanatory
        and "edge" means that there will be thick ticks drawn on the borders to
        indicate where the diagonal is, but the middle of the plot will not have
        a diagonal line.
    :param gridMode: Should a grid be drawn? Options are "on" or "off".
    :param fontsize: How big should the text be on the ticks, in points?
    :param fontSizeAxLabel: How big should the font size be for labels, in points?
    :param genomeWindowStart: Where does the x-axis of shearMat start, in genomic
        coordinates?
    :param mini: If True, then draw a plot that works better as a half-page visual.
        The axes are simplified and the annotation text is moved to a separate legend.
    :return: A mappable that can be used to generate the color bar.
    """
    xlen = shearMat.shape[1]
    axStartY = (xlen - shearMat.shape[0]) // 2
    axStopY = axStartY + shearMat.shape[0]
    cmap = bprcolors.pisaClip

    plotMat = np.array(shearMat)
    plotMat *= math.log10(math.e) * 10
    colorSpan *= math.log10(math.e) * 10
    extent = (0, xlen, axStopY, axStartY)
    axPisa.imshow(plotMat, vmin=-colorSpan, vmax=colorSpan, extent=extent,
                  cmap=cmap, aspect="auto", interpolation="nearest")

    match diagMode:
        case "off":
            pass
        case "on":
            axPisa.plot([0, xlen], [0, xlen], "k--", lw=0.5)
        case "edge":
            if xlen > shearMat.shape[0]:
                # We have a wide plot, so clip in appropriately.
                xStart = (xlen - shearMat.shape[0]) // 2
                xEnd = xlen - xStart
            else:
                xStart = 0
                xEnd = xlen
            axPisa.plot([xStart, xStart + xlen * 0.02], [xStart, xStart + xlen * 0.02],
                        "k-", lw=2.0)
            axPisa.plot([xEnd - xlen * 0.02, xEnd], [xEnd - xlen * 0.02, xEnd], "k-", lw=2.0)
    if not mini:
        axPisa.set_ylabel("Output base coordinate", fontsize=fontSizeAxLabel,
                      fontfamily=FONT_FAMILY, labelpad=-5)
    numYTicks = 4 if mini else 10
    ticksY, tickLabelsY = getCoordinateTicks(genomeWindowStart,
                      genomeWindowStart + shearMat.shape[0], numYTicks, True)
    ticksY = [x + axStartY for x in ticksY]
    axPisa.set_yticks(ticksY, tickLabelsY, fontsize=fontsize, fontfamily=FONT_FAMILY)
    axPisa.set_ylim(axStopY, axStartY)
    match gridMode:
        case "on":
            axPisa.grid()
        case "off":
            pass
    norm = mplcolors.Normalize(vmin=-colorSpan, vmax=colorSpan)
    smap = ScalarMappable(norm=norm, cmap=cmap)
    return smap


def addCbar(pisaCax: ScalarMappable, axCbar: AXES_T, fontSize: int,
            fontSizeAxLabel: int, mini: bool):
    """Add a color bar to the given axes.

    :param pisaCax: The mappable generated by the PISA plotting/graphing function.
    :param axCbar: The axes on which the color bar will be drawn.
    :param fontSize: How big should the tick labels be, in points?
    :param fontSizeAxLabel: How big should the label at the bottom be?
    :param mini: If True, squish the label a bit for printing in a smaller space.
    """
    cbar = plt.colorbar(mappable=pisaCax, cax=axCbar)
    bottom, top = axCbar.get_ylim()
    axCbar.set_yticks(cbar.get_ticks(), [f"{x:0.1f}" for x in cbar.get_ticks()],
                      fontsize=fontSize, fontfamily=FONT_FAMILY)
    axCbar.set_ylim(bottom, top)
    if mini:
        axCbar.set_xlabel("PISA\neffect\n(dBr)", fontsize=fontSizeAxLabel, fontfamily=FONT_FAMILY)
    else:
        axCbar.set_xlabel("PISA effect\n(dBr)", fontsize=fontSizeAxLabel, fontfamily=FONT_FAMILY)


def addLegend(usedNames: dict[str, COLOR_SPEC_T], axLegend: AXES_T, fontsize: int):
    """Add a legend to map the annotations to colors.

    :param usedNames: The names that are present in this view. Comes from
        :py:func:`~addAnnotations`.
    :param axLegend: The axes to draw the legend on.
    :param fontsize: How big do you want the text, in points?
    """
    offset = 1
    for name, color in usedNames.items():
        axLegend.fill([0, 0, 1, 1],
                      [offset, offset + 1, offset + 1, offset],
                      color=parseSpec(color))
        axLegend.text(0.5, offset + 0.5, name, fontstyle="italic",
                      fontsize=fontsize, fontfamily=FONT_FAMILY,
                      ha="center", va="center")
        offset += 2
    axLegend.set_xlim(0, 1)
    axLegend.set_ylim(0, max(5, offset - 1))


def getPisaAxes(fig: matplotlib.figure.Figure, left: float, bottom: float,
                width: float, height: float, mini: bool) -> tuple[AXES_T, AXES_T,
        AXES_T, AXES_T, AXES_T, AXES_T | None]:
    """Generates the various axes that will be needed for a PISA plot.

    :param fig: The figure to draw the axes on.
    :param left: The left edge, as a fraction of the figure width, for the plots.
    :param bottom: The bottom, as a fraction of figure height, of the plots.
    :param width: The width, as a fraction of the figure width, for the plots.
    :param height: The height, as a fraction of the figure height, for the plots.
    :param mini: Should the axes be arranged for smaller display? If so, returns an
        additional axes object for the legend.
    :return: A tuple of axes, in order Pisa, importance, predictions,
        cbar, annotations, legend. Legend will be None if mini is False.
    """
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

    axPisa = fig.add_axes((left, bottom + seqHeight, pisaWidth, pisaHeight))
    axSeq = fig.add_axes((left, bottom, pisaWidth, seqHeight))
    axProfile = fig.add_axes((left + pisaWidth + profileWidth * 0.02,
                              bottom + seqHeight, profileWidth * 0.9, pisaHeight))
    axCbar = fig.add_axes((left + pisaWidth + profileWidth,
                           bottom + seqHeight + pisaHeight / (8 if mini else 4),
                           cbarWidth, pisaHeight / (3 if mini else 2)))
    axLegend = None
    if mini:
        axLegend = fig.add_axes((left + pisaWidth + profileWidth,
                            bottom + seqHeight + pisaHeight * (1 / 3 + 1 / 7),
                            cbarWidth * 3, pisaHeight * (1 - 1 / 3 - 1 / 7)))
        axLegend.set_axis_off()
    axAnnot = fig.add_axes((left,
                            bottom + seqHeight + 2 * pisaHeight / 3,
                            pisaWidth,
                            pisaHeight / 3))
    axAnnot.set_axis_off()

    axSeq.set_frame_on(False)
    axSeq.set_yticks([])
    axAnnot.set_ylim(-1, 0)
    axAnnot.set_axis_off()
    axProfile.set_yticks([])
    axProfile.set_frame_on(False)
    axProfile.yaxis.set_visible(False)
    return axPisa, axSeq, axProfile, axCbar, axAnnot, axLegend


def getPisaGraphAxes(fig: matplotlib.figure.Figure, left: float, bottom: float, width: float,
                     height: float) -> tuple[AXES_T, AXES_T, AXES_T, AXES_T, AXES_T]:
    """Get axes appropriate for drawing a PISA graph.

    :param fig: The figure to draw the axes on.
    :param left: The left edge, as a fraction of the figure width, for the plots.
    :param bottom: The bottom, as a fraction of figure height, of the plots.
    :param width: The width, as a fraction of the figure width, for the plots.
    :param height: The height, as a fraction of the figure height, for the plots.
    :return: A tuple of axes, in order Graph, importance, predictions,
        annotations, colorbar.
    """
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

    # Get the actual width of the graph so that it matches pisa plots.
    # These numbers should match _getPisaAxes.
    graphXWeight = 40
    profileXWeight = 6
    cbarXWeight = 1
    totalXWeight = graphXWeight + profileXWeight + cbarXWeight
    graphWidth = width * graphXWeight / totalXWeight
    cbarWidth = width * cbarXWeight / totalXWeight
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

    axImportance = fig.add_axes((left, bottom + seqBase, graphWidth, seqHeight))
    axGraph = fig.add_axes((left, bottom + graphBase, graphWidth, graphHeight))
    axPredictions = fig.add_axes((left, bottom + profileBase,
                              graphWidth, profileHeight))

    axCbar = fig.add_axes((left + width - cbarWidth,
                           bottom + graphBase + graphHeight / 4,
                           cbarWidth, graphHeight / 2))

    axAnnot = fig.add_axes((left, bottom + annotBase, graphWidth, annotHeight))
    axAnnot.set_axis_off()
    axGraph.set_yticks([])
    axPredictions.set_xticks([])
    axPredictions.set_yticks([])
    axPredictions.set_frame_on(False)
    axImportance.set_frame_on(False)
    axImportance.set_yticks([])
    axAnnot.set_ylim(0, -1)

    return axGraph, axImportance, axPredictions, axAnnot, axCbar


def addHorizontalProfilePlot(values: PRED_AR_T, colors: list[DNA_COLOR_SPEC_T], seq: str,
                             genomeStartX: int, genomeEndX: int, axSeq: AXES_T,
                             axGraph: AXES_T | None, fontSizeTicks: int, fontSizeAxLabel: int,
                             showSequence: bool, labelAxis: bool, mini: bool):
    """Draw a profile on a horizontal axes.

    :param values: The values to plot.
    :param colors: A list of DNA_COLOR_SPEC_T, one for each base.
    :param seq: The sequence of the region. Used to determine the color for each base,
        and of course to set the sequence if ``showSequence`` is ``True``.
    :param genomeStartX: Where, in genomic coordinates, does the x-axis start?
    :param genomeEndX: Where, in genomic coordinates, does the x-axis end?
    :param axSeq: The axes where the plot will be drawn.
    :param axGraph: This is the axes from the PISA graph or plot. If ``labelAxis`` is set
        then the x-ticks on axGraph will be turned off. If axGraph is None, then
        it won't be changed (obviously).
    :param fontSizeTicks: How big should the tick text be, in points?
    :param fontSizeAxLabel: How big should the labels be, in points?
    :param showSequence: Should the DNA sequence be drawn, or just a bar plot?
    :param labelAxis: If True, then put ticks and tick labels on the x-axis, and also
        remove any labels from axGraph, if axGraph is not None.
    :param mini: If True, use fewer x-ticks.
    """
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
        axSeq.set_ylim(float(np.min(seqOhe)), float(np.max(seqOhe)))
    else:
        # Window span too big - just show a profile.
        for pos, score in enumerate(values):
            axSeq.bar([pos], [score],
                      linewidth=1, facecolor=parseSpec(colors[pos][seq[pos]]),
                      edgecolor=parseSpec(colors[pos][seq[pos]]))
        axSeq.plot([0, values.shape[0]], [0, 0], "k--", lw=0.5)
    if labelAxis:
        axSeq.set_xticks(ticksX, tickLabelsX, fontsize=fontSizeTicks, fontfamily=FONT_FAMILY)

        axSeq.xaxis.set_tick_params(labelbottom=True, which="major")
        axSeq.set_ylabel("Contrib.\nscore", fontsize=fontSizeAxLabel,
                        fontfamily=FONT_FAMILY, rotation=0, loc="bottom", labelpad=40)
        axSeq.set_xlabel("Input base coordinate", fontsize=fontSizeAxLabel,
                         fontfamily=FONT_FAMILY)
        if axGraph is not None:
            axGraph.set_xticks(ticksX)
            axGraph.tick_params(axis="x", which="major", length=0, labelbottom=False)


def addPisaGraph(similarityMat: IMPORTANCE_AR_T, minValue: float, colorSpan: float,
                 colorBlocks: list[tuple[int, int, tuple[float, float, float]]],
                 lineWidth: float, ax: AXES_T) -> ScalarMappable:
    """Draw a graph representation of a PISA matrix.

    :param similarityMat: The PISA array, already sheared. It should be square.
    :param minValue: PISA values less than this will not be plotted at all.
    :param colorSpan: Values higher than this will be clipped.
    :param colorBlocks: Regions of the plot that override the color of the lines.
        These are tuples of (start, end, (r, g, b)).
        If the origin of a line overlaps with a ColorBlock, then its color is set
        to the rgb color in the block.
    :param lineWidth: The thickness of the drawn lines. For large figures,
        thicker lines avoid Moiré patterns.
    :param ax: The axes to draw on. The xlim and ylim will be clobbered by this function.
    """
    cmap = bprcolors.pisaClip

    plotMat = np.array(similarityMat)
    # convert into dB
    plotMat *= math.log10(math.e) * 10
    colorSpan *= math.log10(math.e) * 10
    minValue *= math.log10(math.e) * 10

    def addLine(xLower, xUpper, value):
        if abs(value) < minValue:
            return False
        if value < 0:
            value += minValue
        elif value > 0:
            value -= minValue
        if value > colorSpan:
            value = colorSpan
        if value < -colorSpan:
            value = -colorSpan
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
        normValue = (value + colorSpan) / (2 * colorSpan)
        normα = abs(value / colorSpan)

        color = cmap(normValue, alpha=normα)
        for colorBlock in colorBlocks:
            start, end, colorSpec = colorBlock
            r, g, b = parseSpec(colorSpec)[:3]
            if start <= xLower < end:
                color = (r, g, b, normα)
                break

        curPatch = patches.PathPatch(path, facecolor="none", lw=lineWidth,
                                     edgecolor=color)
        return (abs(value), curPatch)
    patchList = []

    for row in logUtils.wrapTqdm(range(plotMat.shape[0]), "DEBUG"):
        for col in range(plotMat.shape[1]):
            if curPatch := addLine(col, row, plotMat[row, col]):
                patchList.append(curPatch)
    psSorted = sorted(patchList, key=lambda x: x[0])
    for p in logUtils.wrapTqdm(psSorted, "DEBUG"):
        ax.add_patch(p[1])
    ax.set_xlim(0, np.max(plotMat.shape))
    ax.set_ylim(0, 1)

    # Last quick thing to do - generate a color map.
    norm = mplcolors.Normalize(vmin=-colorSpan, vmax=colorSpan)
    smap = ScalarMappable(norm=norm, cmap=cmap)
    return smap
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
