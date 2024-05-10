"""Utilities for making plots with your data.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/plotting.bnf

Parameter notes
---------------

``pisa-section``
''''''''''''

You can specify either a file of PISA data or a custom array.
If you give a file, you need to be very careful about coordinates!
See the documentation for loadPisa for the coordinate convention used
to make PISA plots.

``h5-name``
    The name of a PISA hdf5 file, generated by
    :py:mod:`interpretPisa<bpreveal.interpretPisa>`. Cannot be used if
    ``values`` is present.

``values``
    Instead of giving a PISA file, give an array of values to plot. These
    should already be sheared and in the form generated by
    :py:func:`loadPisa<bpreveal.plotting.loadPisa>`. It will be an array of
    shape (num-regions, num-regions). Cannot be used if ``h5-name`` is given.
    Note that the cropping (in ``coordinates``) is applied *to* this array, so
    it should represent the data for the whole window and then the plotting
    functions will crop in appropriately.

``coordinates-section``
'''''''''''''''''''

``genome-fasta``
    The name of a fasta-format file containing the genome of the organism. The
    relevant sequence information will be extracted based on
    ``genome-window-start`` and ``genome-window-chrom``. Cannot be used if
    ``sequence`` is provided.

``sequence``
    Instead of looking up the sequence in a fasta, you can provide it manually.
    This should be a string of ``A``, ``C``, ``G``, and ``T``, and it must have
    the same length as the shape of the values array (or, equivalently, the
    number of regions in your PISA hdf5 file).
    Default: If no sequence is provided, then you cannot use ``show-sequence``
    in the importance or prediction sections.

``midpoint-offset``
    For making a plot, how far from the left edge of the PISA data do you want
    the middle of the plot to be? For example, if your PISA data start at
    coordinate 131,500 and you want coordinate 132,000 to be in the middle of
    the plot, you'd set ``"midpoint-offset": 500``.

``input-slice-width``, ``output-slice-width``
    How wide of a region do you want to be plotted? ``input-slice-width``
    determines the width of the x-axis. Use a smaller value to see the effect
    of fewer and fewer bases. ``output-slice-width`` determines the height of
    the y-axis. Use smaller values to zoom in on local effects.

``genome-window-start``, ``genome-window-chrom``
    These are used to set the values for the tick labels on the plots,
    and also to extract the sequence from a fasta file if you provided
    ``genome-fasta``.

``profile-section``
'''''''''''''''

The profile section gives the data that should be plotted in the importance score
or prediction tracks. The format for both tracks is identical.

``bigwig-name``
    The name of the bigwig file to read from. Data will be extracted based on
    ``genome-window-start`` and ``genome-window-chrom``. Cannot be used with
    ``values``.

``values``
    An array of shape (num-regions) containing the profile values to use at
    each base, starting at ``genome-window-start``.

``show-sequence``
    If ``true``, then draw the actual letters of the DNA sequence to represent the
    profile. If ``false``, then draw a bar plot like in IGV.
    Default: ``false``.

``color``
    Color, for a profile, can be one of several things. It can be a single
    ``color-spec``, or a dictionary with one ``color-spec`` for each DNA base.
    Alternatively, it can be a list of ``color-spec`` (or dictionary with one
    ``color-spec`` for each base) with one entry for each position in the profile.
    This way, you can color each bar (or each letter in the logo) a different
    color. See the :py:mod:`colors<bpreveal.colors>` documentation
    for a description of ``color-spec``.
    Default: If ``show-sequence`` is ``true``, then the default Wong color map
    for bases. If ``show-sequence`` is ``false``, then Tol color 0.

``annotation-section``
''''''''''''''''''

Annotations are used to label specific areas of the plot, typically with
things like motifs and genes.

``bed-name``
    If you scanned for motifs, then you can give a bed file here.
    Hits in that bed file will be drawn on the plot.
    Default: Do not read in a bed file.

``name-colors``
    In order to provide a consistent color to each named motif, you can
    specify which names get which colors. It is a simple dict mapping
    names (e.g., "Abf1") to colorSpecs. Whenever an entry from the bed
    file is drawn, the code will check to see if that entry's name appears
    in this dict. If it does, then that color will be used to draw its
    annotation box.
    Default: Assign colors as names are encountered.

``custom``
    A list of annotations that you provide that do not come from the bed file.
    An annotation is a simple dict mapping ``start`` and ``end`` to (genomic)
    coordinates, along with a ``name`` giving the text to draw and a
    ``color`` giving a colorSpec to use. See
    :py:mod:`colors<bpreveal.colors>` for colorSpec documentation.
    Default: No custom annotations.

``figure-section``
''''''''''''''

``bottom``, ``left``, ``width``, ``height``
    Where on the given figure should the plot be placed? Note that axis labels
    and other text can overflow this bounding box. For figure prep, you'll
    likely have to tweak these parameters.

``annotation-height``
    How tall, as a fraction of the total height allocated for annotations,
    should the drawn boxes be? Note that some extra space will be placed
    between the boxes for clarity. For large figures, you'll probably want to
    make this smaller.
    Default: 0.13

``tick-font-size``, ``label-font-size``
    For coordinate ticks and text labels, what font size should be used?
    Defaults: ``tick-font-size`` defaults to 6 and ``label-font-size`` defaults to 8.

``color-span``
    This sets the color limit for PISA values. If a clipping color map
    is used, then any values above this will show as clipped colors.

``grid-mode``, ``diagonal-mode``
    For PISA plots only (not PISA graphs), these parameters determine whether
    or not the grid lines and diagonal line should be drawn. ``grid-mode`` can
    be either ``"on"`` or ``"off"``, while ``diagonal-mode`` can be one of
    ``"on"``, ``"off"``, or ``"edge"``. ``edge`` means that the diagonal lines
    should only be drawn at the border of the PISA plot, like inward-facing
    axis ticks.
    Defaults: ``grid-mode`` defaults to ``"on"`` and
    ``diagonal-mode`` defaults to ``"edge"``.

``line-width``
    For PISA graphs only (not plots). This sets the width of the lines that
    connect the cause to effect. For large images, increasing this reduces
    Moiré interference.
    Default: 1.0

Specific parameters
'''''''''''''''''''

``min-value``
    Only applicable to PISA graphs, not plots.
    If (the absolute value of) a PISA entry between two bases
    is less than min-value, no line will be drawn at all.
    I recommend keeping this at about the 95th percentile
    of your PISA data, as otherwise an absolutely enormous number of
    splines will be drawn.

``use-annotation-colors``
    Only applicable to PISA graphs.
    If ``true``, then any splines that originate from a base that
    overlaps with an annotation will be drawn in the color of that
    annotation instead of the normal color map. This is useful
    for showing the effects of different motifs, for example.
    If ``false``, then all splines will use the default color map
    based on the PISA value between the bases they connect.
    Default: ``false``.

``miniature``
    Only applicable to PISA plots, not graphs. If ``true``, then the format of
    the graph is changed to make it better-suited to one-column figures. The
    text for annotations is moved to a legend, and the axis labels are
    simplified.
    Default: ``false``.

Module contents
---------------

"""
import numpy as np

import matplotlib.figure
from matplotlib.axes import Axes as AXES_T

from bpreveal import logUtils
from bpreveal.internal.constants import PRED_AR_T, ONEHOT_AR_T, \
    FONT_FAMILY, FONT_SIZE_TICKS, FONT_SIZE_LABELS
import bpreveal.internal.plotUtils as pu
from bpreveal import utils
from bpreveal import motifUtils
from bpreveal import schema
import bpreveal.colors as bprcolors
from bpreveal.colors import parseSpec


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

        fig = plt.figure(figsize=(7,4))
        pisaSection = {"h5-name": f"{WORK_DIR}/shap/pisa_nanog_positive.h5"}

        coordinatesSection = {
            "genome-fasta": GENOME_FASTA,
            "midpoint-offset": 1150,
            "input-slice-width": 200,
            "output-slice-width": 500,
            "genome-window-start": windowStart,
            "genome-window-chrom": windowChrom
        }

        predictionSection = {"bigwig-name": f"{WORK_DIR}/pred/nanog_residual_positive.bw"}

        importanceSection = {
            "bigwig-name": f"{WORK_DIR}/shap/nanog_profile.bw",
            "show-sequence": True
        }

        annotationSection = {"bed-name": f"{WORK_DIR}/scan/nanog_profile.bed"}

        figureSectionGraph = {
            "left": 0.12,
            "bottom": 0.13,
            "width": 0.8,
            "height": 0.85,
            "color-span": 0.5,
        }

        graphConfig = {
            "pisa": pisaSection,
            "coordinates": coordinatesSection,
            "importance": importanceSection,
            "predictions": predictionSection,
            "annotations": annotationSection,
            "figure": figureSectionGraph,
            "min-value": 0.1,
            "use-annotation-colors": False
        }

        bpreveal.plotting.plotPisaGraph(graphConfig, fig);

    This code produces a graph that looks like this:

    .. image:: ../../doc/presentations/pisaGraph.png
        :width: 800
        :alt: Representative PISA graph.

    """
    if validate:
        schema.pisaGraph.validate(config)
    cfg = pu.buildConfig(config)
    del config  # Don't accidentally use the old one.
    logUtils.debug("Starting to draw PISA graph.")

    axGraph, axSeq, axProfile, axAnnot, axCbar = pu.getPisaGraphAxes(fig,
                                                                   cfg["figure"]["left"],
                                                                   cfg["figure"]["bottom"],
                                                                   cfg["figure"]["width"],
                                                                   cfg["figure"]["height"])
    coords = cfg["coordinates"]
    sliceStart = coords["midpoint-offset"] - coords["input-slice-width"] // 2
    sliceEnd = sliceStart + coords["input-slice-width"]
    genomeStart = coords["genome-window-start"] + sliceStart
    genomeEnd = coords["genome-window-start"] + sliceEnd

    shearMat = cfg["pisa"]["values"][sliceStart:sliceEnd, sliceStart:sliceEnd]
    colorBlocks = []
    if cfg["use-annotation-colors"]:
        for annot in cfg["annotations"]["custom"]:
            colorBlocks.append((annot["start"] - genomeStart,
                                annot["end"] - genomeStart,
                                annot["color"]))
    logUtils.debug("Axes set. Drawing graph.")
    pisaCax = pu.addPisaGraph(similarityMat=shearMat, minValue=cfg["min-value"],
                  colorSpan=cfg["figure"]["color-span"], colorBlocks=colorBlocks,
                  lineWidth=cfg["figure"]["line-width"],
                  ax=axGraph)

    # Now set up the sequence/importance axis.
    logUtils.debug("Graph complete. Finishing plot.")
    pu.addHorizontalProfilePlot(cfg["importance"]["values"][sliceStart:sliceEnd],
                              cfg["importance"]["color"][sliceStart:sliceEnd],
                              coords["sequence"][sliceStart:sliceEnd],
                              genomeStart,
                              genomeEnd,
                              axSeq, axGraph,
                              cfg["figure"]["tick-font-size"],
                              cfg["figure"]["label-font-size"],
                              cfg["importance"]["show-sequence"],
                              True, False)

    pu.addAnnotations(axAnnot, cfg["annotations"]["custom"],
                    cfg["figure"]["annotation-height"], genomeStart, genomeEnd,
                    cfg["figure"]["label-font-size"], False)

    # Now, add the profiles.
    pu.addHorizontalProfilePlot(cfg["predictions"]["values"][sliceStart:sliceEnd],
                              cfg["predictions"]["color"][sliceStart:sliceEnd],
                              coords["sequence"][sliceStart:sliceEnd],
                              genomeStart,
                              genomeEnd,
                              axProfile, None,
                              cfg["figure"]["tick-font-size"],
                              cfg["figure"]["label-font-size"],
                              cfg["predictions"]["show-sequence"],
                              False, False)

    pu.addCbar(pisaCax, axCbar, cfg["figure"]["tick-font-size"],
               cfg["figure"]["label-font-size"], False)
    logUtils.debug("PISA graph complete.")
    return {"axes": {"graph": axGraph, "importance": axSeq, "predictions": axProfile,
                     "annotations": axAnnot},
            "name-colors": cfg["annotations"].get("name-colors", {}),
            "genome-start": genomeStart,
            "genome-end": genomeEnd,
            "config": cfg}


def plotPisa(config: dict, fig: matplotlib.figure.Figure, validate: bool = True):
    r"""Given the actual vectors to show, make a pretty PISA plot.

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

        fig = plt.figure(figsize=(7,4))

        pisaSection = {"h5-name": f"{WORK_DIR}/shap/pisa_nanog_positive.h5"}

        coordinatesSection = {
            "genome-fasta": GENOME_FASTA,
            "midpoint-offset": 1150,
            "input-slice-width": 200,
            "output-slice-width": 500,
            "genome-window-start": windowStart,
            "genome-window-chrom": windowChrom
        }

        predictionSection = {"bigwig-name": f"{WORK_DIR}/pred/nanog_residual_positive.bw"}

        importanceSection = {
            "bigwig-name": f"{WORK_DIR}/shap/nanog_profile.bw",
            "show-sequence": True
        }

        annotationSection = {"bed-name": f"{WORK_DIR}/scan/nanog_profile.bed"}

        figureSectionPlot = {
            "left": 0.12,
            "bottom": 0.13,
            "width": 0.8,
            "height": 0.85,
            "color-span" : 0.5,
        }

        plotConfig = {
            "pisa": pisaSection,
            "coordinates": coordinatesSection,
            "importance": importanceSection,
            "predictions": predictionSection,
            "annotations": annotationSection,
            "figure": figureSectionPlot
        }

        bpreveal.plotting.plotPisa(plotConfig, fig)

    This code produces a plot that looks like this:

    .. image:: ../../doc/presentations/pisaPlot.png
        :width: 800
        :alt: Representative PISA plot.

    """
    if validate:
        schema.pisaPlot.validate(config)
    cfg = pu.buildConfig(config)
    del config  # Don't accidentally edit the old one.
    logUtils.debug("Starting to draw PISA graph.")

    axPisa, axSeq, axProfile, axCbar, axAnnot, axLegend = pu.getPisaAxes(
        fig, cfg["figure"]["left"], cfg["figure"]["bottom"],
        cfg["figure"]["width"], cfg["figure"]["height"],
        cfg["miniature"])
    # PISA image plotting
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
    pisaCax = pu.addPisaPlot(shearMat, cfg["figure"]["color-span"], axPisa,
                           cfg["figure"]["diagonal-mode"], cfg["figure"]["grid-mode"],
                           cfg["figure"]["tick-font-size"], cfg["figure"]["label-font-size"],
                           genomeStartX, cfg["miniature"])
    # Now set up the sequence/importance axis.
    logUtils.debug("Graph complete. Finishing plot.")
    pu.addHorizontalProfilePlot(cfg["importance"]["values"][sliceStartX:sliceEndX],
                              cfg["importance"]["color"][sliceStartX:sliceEndX],
                              coords["sequence"][sliceStartX:sliceEndX],
                              genomeStartX,
                              genomeEndX,
                              axSeq, axPisa,
                              cfg["figure"]["tick-font-size"],
                              cfg["figure"]["label-font-size"],
                              cfg["importance"]["show-sequence"],
                              True, cfg["miniature"])

    usedNames = pu.addAnnotations(axAnnot, cfg["annotations"]["custom"],
                                cfg["figure"]["annotation-height"], genomeStartX,
                                genomeEndX, cfg["figure"]["label-font-size"], False)
    # Now, add the profiles.
    pu.addVerticalProfilePlot(cfg["predictions"]["values"][sliceStartY:sliceEndY],
                            axProfile,
                            cfg["predictions"]["color"][sliceStartY:sliceEndY],
                            coords["sequence"][sliceStartY:sliceEndY],
                            cfg["figure"]["tick-font-size"],
                            cfg["predictions"]["show-sequence"],
                            cfg["miniature"])
    if axLegend is not None:
        pu.addLegend(usedNames, axLegend, cfg["figure"]["label-font-size"])
    pu.addCbar(pisaCax, axCbar, cfg["figure"]["tick-font-size"],
               cfg["figure"]["label-font-size"], cfg["miniature"])
    return {"axes": {"pisa": axPisa, "importance": axSeq, "predictions": axProfile,
                     "annotations": axAnnot, "colorbar": axCbar,
                     "legend": axLegend},
            "name-colors": cfg["annotations"].get("name-colors", {}),
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
        bgProbs = [0.291, 0.208, 0.208, 0.291]
        patSox2 = motifUtils.Pattern("pos_patterns", "pattern_0", "Sox2")
        with h5py.File(f"{WORK_DIR}/modisco/sox2_profile/modisco.h5", "r") as fp:
            patSox2.loadCwm(fp, 0.3, 0.3, bgProbs)
            patSox2.loadSeqlets(fp)
        fig = plt.figure(figsize=(5, 5))
        # Sort the seqlets by their contribution match.
        sortKey = [x.contribMatch for x in patSox2.seqlets]
        plotModiscoPattern(patSox2, fig, sortKey=sortKey)

    This produces the following plot:

    .. image:: ../../doc/presentations/modiscoPattern.png
        :width: 800
        :alt: Representative modisco pattern plot.

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
    hmapYticks = axHmap.get_yticks()[1:-1]
    axHmap.set_yticks(hmapYticks, [str(x) for x in hmapYticks], fontsize=FONT_SIZE_TICKS,
                      fontfamily=FONT_FAMILY)
    axLogo = fig.add_axes((0.1, 0.1, 0.2 - PAD, HIST_HEIGHT))
    cwm = pattern.cwm
    plotLogo(cwm, cwm.shape[0], axLogo, colors=bprcolors.dnaWong)
    axLogo.set_xlim(0, cwm.shape[0])
    cwmPos = np.zeros_like(cwm)
    cwmPos[cwm > 0] = cwm[cwm > 0]
    cwmNeg = np.zeros_like(cwm)
    cwmNeg[cwm < 0] = cwm[cwm < 0]
    sumPos = np.sum(cwmPos, axis=1)
    sumNeg = np.sum(cwmNeg, axis=1)

    axLogo.set_ylim(min(sumNeg), max(sumPos))
    axLogo.set_yticks([])
    axLogo.set_xticks([pattern.cwmTrimLeftPoint, pattern.cwmTrimRightPoint],
                      [str(pattern.cwmTrimLeftPoint), str(pattern.cwmTrimRightPoint)],
                      fontsize=FONT_SIZE_TICKS,
                      fontfamily=FONT_FAMILY)

    yvals = np.arange(len(pattern.seqlets), 0, -1)

    def plotStat(stat, axPos, name, rightTicks):
        stat = np.array(stat)
        axCurStat = fig.add_axes((0.3 + axPos * 0.2, 0.1 + HIST_HEIGHT + PAD,
                                  0.2 - PAD, 0.8 - HIST_HEIGHT))
        statSort = stat[sortOrder]
        colorFix = parseSpec(bprcolors.defaultProfile)
        axCurStat.plot(statSort, yvals, ".", color=colorFix, alpha=0.5)
        statFilt = np.sort(statSort)
        colorFix = parseSpec({"tol": 2})
        axCurStat.plot(statFilt, yvals, "-", color=colorFix)
        axCurStat.set_ylim(0, len(pattern.seqlets))
        axCurStat.set_yticks([])
        axCurStat.set_title(name, fontdict={"fontsize": FONT_SIZE_LABELS,
                                            "fontfamily": FONT_FAMILY})
        axCurStat.set_xticks([])
        axCurStat.set_xlim(min(stat), max(stat))
        tickPoses = np.linspace(0, len(pattern.seqlets), 11, endpoint=True)
        tickLabels = np.arange(0, 110, 10)[::-1]
        axCurStat.tick_params(axis="y", labelleft=False, labelright=rightTicks,
                            left=False, right=rightTicks)
        axCurStat.set_yticks(tickPoses, tickLabels, fontsize=FONT_SIZE_TICKS,
                             fontfamily=FONT_FAMILY)
        axCurStat.grid()
        if rightTicks:
            axCurStat.yaxis.set_label_position("right")
            axCurStat.set_ylabel("Percentile", fontsize=FONT_SIZE_LABELS,
                                 fontfamily=FONT_FAMILY)
        axCurHist = fig.add_axes((0.3 + axPos * 0.2, 0.1, 0.2 - PAD, HIST_HEIGHT))
        hist = np.histogram(stat, bins=50)
        binMiddles = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2
        axCurHist.plot(binMiddles, hist[0])
        axCurHist.set_yticks([])
        axCurHist.set_xlim(min(stat), max(stat))
        xticks = axCurHist.get_xticks()
        xticks = xticks[1:-1]
        axCurHist.set_xticks(xticks, [str(x) for x in xticks], fontsize=FONT_SIZE_TICKS,
                             fontfamily=FONT_FAMILY)
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
            color = parseSpec(bprcolors.dnaWong[baseName])[colorIdx]
            for col in range(hmap.shape[1]):
                for colOffset in range(upsamplingFactor):
                    writeCol = col * upsamplingFactor + colOffset
                    displayAr[:, writeCol, colorIdx] += ar[:, col] * color
    ax.imshow(displayAr, aspect="auto", interpolation="antialiased",
              interpolation_stage="data")


def plotLogo(values: PRED_AR_T, width: float, ax: AXES_T,
             colors: bprcolors.DNA_COLOR_SPEC_T | list[bprcolors.DNA_COLOR_SPEC_T],
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
            You can get the default BPReveal color map at :py:mod:`colors<bpreveal.colors>`.
        3. Give a list of colors for each base.
            This will be a list of length ``values.shape[0]`` and each entry
            should be a dictionary in either format
            1 or 2 above. This gives each base its own color palette, useful
            for shading bases by some profile.
    """
    # This is just a wrapper around a function that's in the internal library
    # to avoid circular imports.
    pu.plotLogo(values, width, ax, colors, spaceBetweenLetters)


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
    return pu.getCoordinateTicks(start, end, numTicks, zeroOrigin)


# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
