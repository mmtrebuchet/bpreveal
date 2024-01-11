import logging
import numpy as np
import h5py
import pysam
import pyBigWig
import pybedtools
import math
# You must install bpreveal with conda develop in order to import bpreveal tools.
from bpreveal.utils import PRED_AR_T
import bpreveal.utils as utils
import bpreveal.motifUtils as motifUtils
import matplotlib.pyplot as plt
_seqCmap = {"A": (0, 158, 115), "C": (0, 114, 178), "G": (240, 228, 66), "T": (213, 94, 0)}


cmapIbm = [[100, 143, 255],
           [120, 94, 240],
           [220, 38, 127],
           [254, 97, 0],
           [255, 176, 0]]


def plotLogo(values: PRED_AR_T, width: float, ax, colors='seq') -> None:
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
    import matplotlib.colors as mcolor
    from matplotlib.transforms import Bbox, Affine2D
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch

    def _drawLetter(text: str, left: float, right: float, bottom: float, top: float,
                    color, ax, flip=False) -> None:

        height = top - bottom
        width = right - left
        bbox = Bbox.from_bounds(left, bottom, width, height)
        fontProperties = FontProperties(family='sans', weight='bold')
        tmpPath = TextPath((0, 0), text, size=1, prop=fontProperties)
        if flip:
            flipTransformation = Affine2D().scale(sx=1, sy=-1)
            tmpPath = flipTransformation.transform_path(tmpPath)
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
            case 'seq':
                rgb = _seqCmap[base]
            case (name, vmin, vmax):
                cmap = mcolor.Colormap(name)
                if value < vmin:
                    value = vmin
                if value > vmax:
                    value = vmax
                Δv = vmax - vmin
                return cmap((vmin + value) * Δv)
            case dict():
                rgb = colors[base]
            case _:
                assert False
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
            left = predIdx * width / values.shape[0]
            right = left + width / values.shape[0]
            _drawLetter(nl[0], left=left, right=right, bottom=base, top=top,
                       color=getColor(predIdx, nl[0], nl[1]), ax=ax, flip=True)
            top = base

        # Draw the positive letters.
        base = 0
        for pl in posLetters:
            top = base + pl[1]
            left = predIdx * width / values.shape[0]
            right = left + width / values.shape[0]
            _drawLetter(pl[0], left=left, right=right, bottom=base, top=top,
                       color=getColor(predIdx, pl[0], pl[1]), ax=ax)
            base = top


def loadPisa(fname, offset, buffer):
    with h5py.File(fname, 'r') as fp:
        dats = np.sum(fp["shap"], axis=2)
    skewMat = np.zeros((dats.shape[0], dats.shape[1] + dats.shape[0]))
    for i in range(0, dats.shape[0]):
        if i + offset < 0 or i + offset >= 1000:
            continue
        skewMat[i + offset, i:i + dats.shape[1]] = dats[i]
    skewMat = skewMat[:, buffer:buffer + 1000]
    return skewMat


def plotPisa(pisaDats, cutMiddle, cutLengthX, cutLengthY, receptiveField,
             genomeWindowStart, genomeWindowChrom, genomeFastaFname, importanceBwFname,
             motifScanBedFname, profileDats, nameColors,
             fig, bbox, colorSpan=1.0, bottomSequence=True, boxHeight=0.1, fontsize=5):
    """pisaDats can either be the name of an hdf5 file or an array from loadPisa.
    profileDats can be the name of a bigwig or an array.
    """
    fontSizeAxLabel = fontsize*1.5
    fontSizeMajorTick = fontsize*1.25
    match pisaDats:
        case str():
            with h5py.File(pisaDats, "r") as fp:
                pisaShap = np.array(fp["shap"])
            pisaVals = np.sum(pisaShap, axis=2)
            numRegions = pisaVals.shape[0]
            shearMat = np.zeros((numRegions, pisaVals.shape[1] + numRegions))
            for i in range(0, numRegions):
                offset = i
                shearMat[i, offset:offset + pisaVals.shape[1]] = pisaVals[i]
            shearMat = shearMat[:, receptiveField // 2:-receptiveField // 2]
        case _:
            # We got an array.
            shearMat = pisaDats
    cutStartX = cutMiddle - cutLengthX // 2
    cutStartY = cutMiddle - cutLengthY // 2
    plotMat = shearMat[cutStartY:cutStartY + cutLengthY,
                       cutStartX:cutStartX + cutLengthX]
    plotMat *= math.log10(math.e)*10
    colorSpan *= math.log10(math.e)*10
    axStartY = (cutLengthX - cutLengthY) // 2
    axStopY = axStartY + cutLengthY
    extent = [0, cutLengthX, axStopY, axStartY]
    l, b, w, h = bbox
    xweightPisa = 40
    xweightProfile = 6
    xweightCbar = 1
    pisaWidth = w * xweightPisa / (xweightPisa + xweightProfile + xweightCbar) * (1 - 2*l)
    profileWidth = w * xweightProfile / (xweightPisa + xweightProfile + xweightCbar) * (1 - 2*l)
    cbarWidth = w * xweightCbar / (xweightPisa + xweightProfile + xweightCbar) * (1 - 2*l)
    pisaHeight = h * 7/8
    seqHeight = h/8
    axPisa = fig.add_axes([l, b + seqHeight, pisaWidth, pisaHeight])
    axSeq = fig.add_axes([l, b, pisaWidth, seqHeight])
    axProfile = fig.add_axes([l + pisaWidth + profileWidth * 0.01,
                              b + seqHeight, profileWidth * 0.9, pisaHeight])
    axCbar = fig.add_axes([l + pisaWidth + profileWidth,
                           b + seqHeight + pisaHeight / 4,
                           cbarWidth, pisaHeight/2])
    axAnnot = fig.add_axes([l, b + seqHeight + 2 * pisaHeight / 3, pisaWidth, pisaHeight / 3])
    axAnnot.set_axis_off()


    pisaCax = axPisa.imshow(plotMat, vmin=-colorSpan, vmax=colorSpan, extent=extent,
                            cmap='RdBu_r', aspect='auto', interpolation='nearest')
    axPisa.plot([0, cutLengthX], [0, cutLengthX], 'k--', lw=0.5)
    axPisa.set_ylabel("Output base coordinate", fontsize=fontSizeAxLabel,
                      fontfamily='serif', labelpad=-10)
    # And let's get the sequence for that:
    lowerBound = genomeWindowStart + cutStartX
    upperBound = lowerBound + cutLengthX
    minorTicksX = axPisa.get_xticks()
    minorTickLabelsX = ['+' + str(int(x)) for x in minorTicksX][1:-1]
    minorTicksY = axPisa.get_yticks()
    minorTickLabelsY = ['+' + str(int(x - extent[3])) for x in minorTicksY][1:-1]

    axPisa.set_yticks([axStartY, axStopY],
                      [str(genomeWindowStart + cutStartY),
                       str(genomeWindowStart + cutStartY + cutLengthY)],
                      fontsize=fontSizeMajorTick)
    axPisa.set_yticks(minorTicksY[1:-1], minorTickLabelsY, fontsize=fontsize, minor=True)
    axSeq.set_frame_on(False)
    axSeq.set_yticks([])
    with pysam.FastaFile(genomeFastaFname) as genome:
        seq = genome.fetch(genomeWindowChrom, lowerBound, upperBound)
    seqOhe = utils.oneHotEncode(seq) * 1.0

    if importanceBwFname is not None:
        impFp = pyBigWig.open(importanceBwFname)
        impScores = np.nan_to_num(impFp.values(genomeWindowChrom,
                                               genomeWindowStart + cutStartX,
                                               genomeWindowStart + cutStartX + cutLengthX))
        impFp.close()
        for i in range(len(seq)):
            seqOhe[i, :] *= impScores[i]
        if bottomSequence and upperBound - lowerBound < 100:
            plotLogo(seqOhe, len(seq), axSeq, colors='seq')
            axSeq.set_xlim(0, len(seq))
            axSeq.set_ylim(np.min(seqOhe), np.max(seqOhe))
            #axSeq.xaxis.set_tick_params(labelbottom=False)
            axPisa.xaxis.set_tick_params(labelbottom=False)
        else:
            axSeq.plot(impScores)
            axSeq.plot([0, impScores.shape[0]], [0, 0], 'k--', lw=0.5)
            axSeq.set_xlim(0, impScores.shape[0])
        axSeq.set_xticks([0, cutLengthX], [str(lowerBound), str(upperBound)],
                          fontsize=fontSizeMajorTick)
        axPisa.set_xticks([])
        axSeq.set_xticks(minorTicksX[1:-1], minorTickLabelsX, fontsize=fontsize, minor=True)
        axSeq.xaxis.set_tick_params(labelbottom=True, which='minor')
        axPisa.set_xticks(minorTicksX[1:-1], '' * len(minorTickLabelsX), minor=True)
        axPisa.tick_params(axis='x', which='minor', length=0)
        axSeq.set_ylabel("Contrib.\nscore", fontsize=fontSizeAxLabel,
                         fontfamily='serif', rotation=0, loc='bottom', labelpad = 40)
        axSeq.set_xlabel("Input base coordinate", fontsize=fontSizeAxLabel, fontfamily='serif')

    else:
        axSeq.set_axis_off()
        axPisa.set_ylabel("Input base coordinate", fontsize=fontSizeAxLabel, fontfamily='serif')
        axPisa.set_xticks(minorTicksX[1:-1], minorTickLabelsX, fontsize=fontsize, minor=True)
        axPisa.set_xticks([0, cutLengthX], [str(lowerBound), str(upperBound)],
                          fontsize=fontSizeMajorTick)

    axPisa.grid(visible=True, which='minor')

    # Now it's time to add annotations from the motif scanning step.
    if motifScanBedFname is not None:
        readHead = 0
        bedFp = pybedtools.BedTool(motifScanBedFname)
        annotations = []
        for line in bedFp:
            if line.chrom == genomeWindowChrom and line.end > lowerBound\
                    and line.start < upperBound:
                if line.name not in nameColors:
                    nameColors[line.name] = np.array(cmapIbm[readHead % len(cmapIbm)]) / 256
                    readHead += 1
                annotations.append(((line.start, line.end), line.name, nameColors[line.name]))
        offset = -boxHeight * 1.3
        lastR = 0
        for annot in sorted(annotations, key=lambda x: x[0][0]):
            aleft, aright = annot[0]
            if aleft > lastR:
                offset = -boxHeight * 1.3
            lastR = aright
            if offset < -1:
                continue
            axAnnot.fill([aleft, aleft, aright, aright],
                         [offset, boxHeight + offset, boxHeight + offset, offset],
                         label=annot[1], color=annot[2])
            axAnnot.text((aleft + aright) / 2, offset + boxHeight / 2, annot[1],
                         fontstyle='italic', fontsize=fontsize, fontfamily='serif',
                         ha='center', va='center')
            offset -= boxHeight * 1.5
        axAnnot.set_xlim(lowerBound, upperBound)
        axAnnot.set_ylim(-1, 0)
        axAnnot.set_axis_off()

    # Now, add the profiles.
    if profileDats is not None:
        match profileDats:
            case str():
                profileFp = pyBigWig.open(profileDats)
                profile = np.nan_to_num(
                    profileFp.values(genomeWindowChrom,
                                     genomeWindowStart + cutStartY,
                                     genomeWindowStart + cutStartY + cutLengthY))
                profileFp.close()
            case _:
                profile = np.zeros((cutLengthY,), dtype=np.float64)
                if profileDats.shape[0] <= cutLengthY:
                    offset = (cutLengthY - profileDats.shape[0]) // 2
                    profile[offset:offset + profileDats.shape[0]] = profileDats
                else:
                    offset = (profileDats.shape[0] - cutLengthY) // 2
                    profile = profileDats[offset:offset + cutLengthY]

        axProfile.fill_betweenx(range(cutLengthY, 0, -1), profile, step='mid')
        axProfile.set_ylim(0, cutLengthY)
        axProfile.set_xlim(0, np.max(profile))
        axProfile.set_yticks([])
        profileXticks = axProfile.get_xticks()
        if max(profileXticks) > np.max(profile) * 1.01:
            profileXticks = profileXticks[:-1]
        axProfile.set_xticks(profileXticks, profileXticks, fontsize=fontsize)
        axProfile.set_frame_on(False)
        axProfile.yaxis.set_visible(False)
        axProfile.set_xlabel("Profile", fontsize=fontSizeAxLabel, fontfamily='serif')
    else:
        axProfile.set_axis_off()
    #axCbar.set_axis_off()
    cbar = plt.colorbar(mappable = pisaCax, cax=axCbar)
    bottom, top = axCbar.get_ylim()
    axCbar.set_yticks(cbar.get_ticks(), cbar.get_ticks(), fontsize=fontsize)
    axCbar.set_ylim(bottom, top)
    axCbar.set_xlabel("PISA effect\n(dB fold-change)", fontsize=fontsize, fontfamily='serif')

    return (axPisa, axSeq, axProfile, nameColors, axCbar)



def plotModiscoPattern(pattern: motifUtils.Pattern, fig, sortKey=None):
    if sortKey is None:
        sortKey = np.arange(pattern.numSeqlets)
    sortOrder = np.argsort(sortKey)
    HIST_HEIGHT = 0.1
    PAD = 0.01
    axHmap = fig.add_axes([0.1, 0.1 + HIST_HEIGHT + PAD,
                           0.2 - PAD, 0.8 - HIST_HEIGHT])
    hmapAr = np.zeros((pattern.numSeqlets,
                       len(pattern.seqletSequences[0]),
                       3), dtype=np.uint8)
    for outIdx, seqletIdx in enumerate(sortOrder):
        for charIdx, char in enumerate(pattern.seqletSequences[seqletIdx]):
            hmapAr[outIdx, charIdx, :] = _seqCmap[char]
    axHmap.imshow(hmapAr, aspect='auto', interpolation='nearest')
    axHmap.set_xticks([])
    axLogo = fig.add_axes([0.1, 0.1, 0.2 - PAD, HIST_HEIGHT])
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

    yvals = np.arange(pattern.numSeqlets, 0, -1)

    def plotStat(stat, axPos, name, rightTicks):
        axCurStat = fig.add_axes([0.3 + axPos * 0.2, 0.1 + HIST_HEIGHT + PAD,
                                  0.2 - PAD, 0.8 - HIST_HEIGHT])
        statSort = stat[sortOrder]
        colorFix = [c / 255.0 for c in cmapIbm[0]]
        axCurStat.plot(statSort, yvals, '.', color=colorFix, alpha=0.5)
        statFilt = np.sort(statSort)
        colorFix = [c / 255.0 for c in cmapIbm[2]]
        axCurStat.plot(statFilt, yvals, '-', color=colorFix)
        axCurStat.set_ylim(0, pattern.numSeqlets)
        axCurStat.set_yticks([])
        axCurStat.set_title(name, fontdict={'fontsize': 10})
        axCurStat.set_xticks([])
        axCurStat.set_xlim(min(stat), max(stat))
        tickPoses = np.linspace(0, pattern.numSeqlets, 11, endpoint=True)
        tickLabels = np.arange(0, 110, 10)[::-1]
        axCurStat.tick_params(axis='y', labelleft=False, labelright=rightTicks,
                            left=False, right=rightTicks)
        axCurStat.set_yticks(tickPoses, tickLabels)
        axCurStat.grid()
        if rightTicks:
            axCurStat.yaxis.set_label_position("right")
            axCurStat.set_ylabel("Percentile")
        axCurHist = fig.add_axes([0.3 + axPos * 0.2, 0.1, 0.2 - PAD, HIST_HEIGHT])
        hist = np.histogram(stat, bins=50)
        binMiddles = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2
        axCurHist.plot(binMiddles, hist[0])
        axCurHist.set_yticks([])
        axCurHist.set_xlim(min(stat), max(stat))
        return axCurStat
    plotStat(pattern.seqletSeqMatches, 0, "seqMatch", False)
    plotStat(pattern.seqletContribMatches, 1, "contribMatch", False)
    plotStat(pattern.seqletContribMagnitudes, 2, "contribMag", True)
