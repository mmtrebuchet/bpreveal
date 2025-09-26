#!/usr/bin/env python3
"""Select regions from a bed file to match the GC distribution of a reference bed.

For training bias models, the ChromBPNet method requires that the bias regions
match the peaks regions in GC content. This little script arranges for that.

You feed it two bed files. One represents your training regions and one
represents possible bias regions. This program first calculates the distribution
of GC content in the peaks.
Then, it selects a subset of regions in the bias bed file such that the selected
regions mirror the GC content of the training bed.
"""
import argparse
import matplotlib.pyplot as plt
import pybedtools
import pysam
import numpy as np
import matplotlib
from bpreveal import logUtils




def getGc(region: pybedtools.Interval, genome: pysam.FastaFile) -> int:
    """How mane Gs and Cs are there in the given Interval?

    :param region: A PyBedTool Interval object.
    :param genome: An (opened) pysam genome, used to fetch the sequence.
    :return: The number of Gs and Cs in the region, or -1 if the region
        contains any Ns.
    """
    seq = genome.fetch(region.chrom, region.start, region.end)
    if "N" in seq.upper():
        return -1
    numGc = 0
    for c in seq:
        if c in "cgCG":
            numGc += 1
    return numGc


def getDistributionFromBed(bedFname: str, genome: pysam.FastaFile) -> dict[int, int]:
    """Get a histogram of GC content in all of the regions in a bed file.

    :param bedFname: The name of a bed file on disk.
    :param genome: An (opened) pysam FastaFile, for extracting the sequence.
    :return: A dictionary where the keys are the number of Cs and Gs in a region
        and the values are the number of regions with that GC content.
    """
    logUtils.debug(f"Analyzing bed file {bedFname}")
    gcCounts = {}
    bt = pybedtools.BedTool(bedFname)
    numBases = 0
    totalGc = 0
    numRegions = 0
    for r in bt:
        gcContent = getGc(r, genome)
        gcCounts[gcContent] = 1 + gcCounts.get(gcContent, 0)
        if gcContent >= 0:
            numRegions += 1
            numBases += r.end - r.start
            totalGc += gcContent
    logUtils.info(f"File {bedFname} has {numRegions} regions "
                  f"with an average of {totalGc / numBases} GC content.")
    return gcCounts


def plotHist(bedFname: str, gcCounts: dict[int, int]) -> None:
    """Plot the GC distribution for a bed file.

    :param bedFname: Just a label.
    :param gcCounts: A dictionary as returned by getDistributionFromBed.
    """
    logUtils.debug(f"Plotting file {bedFname}")
    totalRegions = sum(gcCounts.values())
    xvals = range(min(gcCounts.keys()), max(gcCounts.keys()) + 1)
    yvals = np.array([gcCounts.get(x, 0) / totalRegions for x in xvals])
    plt.plot(xvals, yvals / np.max(yvals), label=bedFname)


def getCorrectionFactors(peakCounts: dict[int, int], biasCounts: dict[int, int],
                         strictness: float) -> np.ndarray:
    """Determine sampling rates that will transform the bias GC content to match peaks.

    The returned array gives you the sampling weight you should apply to bias regions
    to transform their GC distribution to match the peak GC distribution.

    The proper use of this array is best explained with some code::

        cf = getCorrectionFactors, peakDist, biasDist)
        for region in biasBed:
            emitProbability = cf[getGc(region, genome)]
            if random() < emitProbability:
                outputBed.append(region)

    As you can see, when you have a bias region with GC content ``i``,
    then the probability that it will be included in the selected regions
    is ``correctionFactors[i]``. Note that you should make sure that
    ``i != -1``!

    :param peakCounts: A dict as returned by getDistributionFromBed
    :param biasCounts: A dict as returned by getDistributionFromBed
    :param strictness: A float from 0 to 1. Default is 0.9. Higher strictness
        better match the input distribution at the price of fewer regions making
        it through.
    :return: An array of floats.
    """
    logUtils.debug("Calculating correction array.")
    numPeakRegions = sum(peakCounts.values())
    numBiasRegions = sum(biasCounts.values())
    # pylint: disable=nested-min-max
    xvals = range(0, max(max(peakCounts.keys()), max(biasCounts.keys())) + 1)
    # pylint: enable=nested-min-max
    peakDist = [peakCounts.get(x, 0) / numPeakRegions for x in xvals]
    biasDist = [biasCounts.get(x, 0) / numBiasRegions for x in xvals]
    correctionFactors = []
    for gcNum in xvals:
        if biasDist[gcNum] == 0:
            correctionFactors.append(0)
        elif peakDist[gcNum] == 0:
            correctionFactors.append(0)
        else:
            correctionFactors.append(peakDist[gcNum] / biasDist[gcNum])
    cors = np.array(correctionFactors)
    topQuantile = np.quantile(cors, [strictness])[0]
    logUtils.info(f"Strictness cutoff: {topQuantile}")
    cors[cors > topQuantile] = topQuantile
    cors /= np.max(cors)
    return cors


def applyCorrection(inBed: pybedtools.BedTool, correctionFactors: np.ndarray,
                    genome: pysam.FastaFile,
                    rng: np.random.Generator | None = None) -> pybedtools.BedTool:
    """Subsample the input bed according to the correction factors.

    :param inBed: The bias input bed.
    :param correctionFactors: The factors given by getCorrectionFactors.
    :param genome: The (opened) pysam FastaFile.
    :param rng: A numpy Generator that will be used to get random samples.
    :return: A new BedTool that contains a subset of the regions in inBed.
    """
    if rng is None:
        logUtils.warning(
            "Using default RNG for subsampling. Results will not be reproducible!")
        rng = np.random.default_rng()
    ret = []
    numInRegions = 0
    numOutRegions = 0
    for r in inBed:
        numInRegions += 1
        gcContent = getGc(r, genome)
        if gcContent == -1:
            continue
        acceptFrac = correctionFactors[gcContent]
        if rng.random() < acceptFrac:
            ret.append(r)
            numOutRegions += 1
    logUtils.info(
        f"From {numInRegions}, selected {numOutRegions} to match GC distribution.")
    return pybedtools.BedTool(ret)


def getParser() -> argparse.ArgumentParser:
    """Build an arg parser (but don't parse args)."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--peaks", help="A bed file of peaks that will be used for training.")
    ap.add_argument(
        "--bias", help="A bed file of peaks used for the bias model.")
    ap.add_argument("--genome", help="A fasta file for extracting sequences.")
    ap.add_argument(
        "--output", help="The name of the bed file that will be written.")
    ap.add_argument("--strictness", help="How strict should the algorithm be? "
                    "0.5 is lax, 0.9 is the default, and 0.98 is very strict. "
                    "At higher strictness, the sampled set will be smaller.",
                    default=0.9, type=float)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="Show some pretty plots.")
    return ap


def runMain() -> None:
    """Run the program."""
    ap = getParser()
    args = ap.parse_args()
    if args.verbose:
        logUtils.setVerbosity("INFO")
    with pysam.FastaFile(args.genome) as gn:
        p = getDistributionFromBed(args.peaks, gn)
        b = getDistributionFromBed(args.bias, gn)
        if args.plot:
            plotHist(args.peaks, p)
            plotHist(args.bias, b)
    cv = getCorrectionFactors(p, b, args.strictness)
    correctedBias = []
    for i, c in enumerate(cv):
        pb = b.get(i, 0) * c
        correctedBias.append(pb)
    correctedBiasAr = np.array(correctedBias)
    correctedBiasAr /= np.max(correctedBiasAr)
    if args.plot:
        matplotlib.use("TkAgg")
        plt.plot(cv, label="correction factors")
        plt.plot(correctedBiasAr, label="corrected bias dist.")
        plt.legend()
        plt.show()
    rng = np.random.default_rng(seed=1234)
    with pysam.FastaFile(args.genome) as gn:
        outBed = applyCorrection(pybedtools.BedTool(args.bias), cv, gn, rng)
        if args.output is not None:
            outBed.saveas(args.output)


def main() -> None:
    """Entry point for script."""
    runMain()


if __name__ == "__main__":
    main()
