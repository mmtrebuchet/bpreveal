#!/usr/bin/env python3
r"""Calculates useful metrics of model performance.

This little helper program reads in two bigwig files and a bed of regions. For
each of the regions, it calculates several metrics, and then displays the
quintiles of the values of those metrics over the regions. For each region,
four metrics are calculated based on the profiles: ``mnll``, ``jsd``,
``pearsonr``, and ``spearmanr``. Then, the program displays the quintiles of
the observed metrics values. The 50th percentile indicates the median value,
and is likely to be the most useful for typical work.

Additionally, this tool calculates the total counts over each region and
calculates, for all regions in the reference against all regions in the
prediction, the Pearson and Spearman correlation of the counts values.

This program uses command-line arguments rather than a configuration json,
since it doesn't do very much. The arguments are given by running
``metrics --help``.

The metrics are:

mnll
    The multinomial log-likelihood of seeing the observed data
    given the predicted probability distribution.
    This is the profile loss function.

jsd
    The Jensen-Shannon distance.
    This technique comes from information theory, and determines how
    similar two probability distributions are.
    This is a *distance* metric, so lower values indicate a better
    match.

pearsonr
    The Pearson correlation coefficient of the profiles.

spearmanr
    The Spearman correlation coefficient of the profiles.

Counts pearson
    The Pearson correlation coefficient of total counts
    in every region.

Counts spearman
    The Spearman correlation coefficient of the total
    counts in every region.

Output specification
--------------------

Normal
^^^^^^
If you don't specify ``--json-output``, then you get a table of metrics. The
first row gives the percentile cutoffs for the various metrics. The following
rows give the value of each metric at each percentile value. For example, if
the 75% column for jsd is 0.72, that means that 75% of your regions have a jsd
value under 0.72.

The two last rows give the statistics for the counts performance. Since counts
metrics are not evaluated region-by-region, there are no quantile statistics
for them.


json
^^^^

.. highlight:: none

If you request a JSON output with ``--json-output``, then you will get
a json with the following format::

    <metrics-json> ::= {
        "reference" : <string>,
        "predicted" : <string>,
        "regions" : <string>,
        "mnll" : <metrics-quantile-section>,
        "jsd" : <metrics-quantile-section>,
        "pearsonr" : <metrics-quantile-section>,
        "spearmanr" : <metrics-quantile-section>,
        "counts-pearson" : <number>,
        "counts-spearman" : <number>
        }

    <metrics-quantile-section> ::= {
        "quantile-cutoffs" : [<list-of-numbers>],
        "quantiles" : [<list-of-numbers>]
        }

Output notes
^^^^^^^^^^^^

quantile-cutoffs
    A list of the quantile thresholds used for the metrics. These will be
    ``[0.0, 0.25, 0.5, 0.75, 1.0]``

quantiles
    A list of numbers giving the value of the given metric at the given
    quantile. For example, the quantiles[1] will give the 25th percentile of
    the value of that metric.

API
---

"""
import argparse
import json
import typing
from multiprocessing import Process
import sys
from numpy._typing import NDArray
import pyBigWig
import numpy as np
import scipy.stats
import scipy.spatial.distance
from bpreveal import logUtils
from bpreveal.internal.crashQueue import CrashQueue


class Region:
    """A simple container from a line from a bed file."""

    def __init__(self, line: str):
        lsp = line.split()
        self.chrom = lsp[0]
        self.start = int(lsp[1])
        self.stop = int(lsp[2])


class MetricsCalculator:
    """Calculates metrics as it receives queries from inQueue, puts results in outQueue.

    :param referenceBwFname: The file name of the reference bigwig.
    :param predictedBwFname: The file name of the predicted bigwig.
    :param applyAbs: Should the values in the bigwig be made positive?
    :param inQueue: The queue that will provide queries.
    :param outQueue: The queue where results will be put.
    :param tid: The thread ID of this process.
    """

    def __init__(self, referenceBwFname: str, predictedBwFname: str, applyAbs: bool,
                 inQueue: CrashQueue, outQueue: CrashQueue, tid: int):
        self.referenceBw = pyBigWig.open(referenceBwFname, "r")
        self.predictedBw = pyBigWig.open(predictedBwFname, "r")
        self.applyAbs = applyAbs
        self.inQueue = inQueue
        self.outQueue = outQueue
        self.tid = tid

    def runRegions(self, regionReference: Region,
                   regionPredicted: Region, regionID: typing.Any) -> None:
        """Run the calculation on a single region.

        :param regionReference: A region in the reference bigwig.
        :param regionPredicted: A region in the predicted bigwig.
        :param regionID: A tag passed into the output queue.

        Given a region, loads up profiles from the reference and predicted bigwigs
        and calculates the various metrics. Puts its results into the output queue.
        """
        referenceData = np.nan_to_num(self.referenceBw.values(regionReference.chrom,
                                                              regionReference.start,
                                                              regionReference.stop))
        if self.applyAbs:
            referenceData = np.abs(referenceData)
        predictedData = np.nan_to_num(self.predictedBw.values(regionPredicted.chrom,
                                                              regionPredicted.start,
                                                              regionPredicted.stop))
        if self.applyAbs:
            predictedData = np.abs(predictedData)
        if np.sum(referenceData) > 0 and np.sum(predictedData) > 0:
            referencePmf = referenceData / np.sum(referenceData)
            predictedPmf = predictedData / np.sum(predictedData)
            mnll = scipy.stats.multinomial(np.sum(referenceData), predictedPmf)
            mnllVal = mnll.logpmf(referenceData)

            jsd = scipy.spatial.distance.jensenshannon(referencePmf, predictedPmf)

            pearsonr = scipy.stats.pearsonr(referenceData, predictedData)[0]

            spearmanr = scipy.stats.spearmanr(referenceData, predictedData)[0]
            referenceCounts = np.sum(referenceData)
            predictedCounts = np.sum(predictedData)
        else:
            # We had a zero in our inputs. Poison the results.
            mnllVal = jsd = pearsonr = spearmanr = np.nan
            referenceCounts = predictedCounts = np.nan

        ret = (regionID, mnllVal, jsd, pearsonr, spearmanr, referenceCounts, predictedCounts)
        self.outQueue.put(ret)

    def run(self) -> None:
        """Watch the input queue and run queries until you get the stop signal."""
        while True:
            match self.inQueue.get():
                case None:
                    self.finish()
                    return
                case (regionRef, regionPred, regionID):
                    self.runRegions(regionRef, regionPred, regionID)

    def finish(self) -> None:
        """Wrap up shop, close the bigwigs."""
        self.referenceBw.close()
        self.predictedBw.close()


def calculatorThread(referenceBwFname: str, predictedBwFname: str, applyAbs: bool,
                     inQueue: CrashQueue, outQueue: CrashQueue, tid: int) -> None:
    """Just spawns a MetricsCalculator and runs it.

    :param referenceBwFname: The file name of the reference bigwig.
    :param predictedBwFname: The file name of the predicted bigwig.
    :param applyAbs: Should the values in the bigwig be made positive?
    :param inQueue: The queue that will provide queries.
    :param outQueue: The queue where results will be put.
    :param tid: The thread ID of this process.
    """
    calc = MetricsCalculator(referenceBwFname, predictedBwFname, applyAbs, inQueue,
                             outQueue, tid)
    calc.run()


def regionGenThread(regionsFname: str, regionQueue: CrashQueue,
                    numThreads: int, numberQueue: CrashQueue) -> None:
    """A thread to generate regions and stuff them in the regionQueue.

    :param regionsFname: The bed file to read in.
    :param regionQueue: The queue that the calculator threads will be getting
        their queries from.
    :param numThreads: How many calculator threads will be running?
    :param numberQueue: A queue that will hear the number of regions
        in the regions file.

    numberQueue is needed so that the parent thread can know how many results
    to expect. This thread counts the number of regions in regionsFname
    and then puts that number in numberQueue *before* it starts putting
    regions into regionQueue.
    """
    logUtils.debug("Initializing generator.")
    with open(regionsFname, "r") as fp:
        regions = [Region(x) for x in fp]
    logUtils.info(f"Number of regions: {len(regions)}")
    numberQueue.put(len(regions))
    for i, r in enumerate(regions):
        regionQueue.put((r, r, i))
    for _ in range(numThreads):
        regionQueue.put(None)
    logUtils.debug("Generator done.")


def percentileStats(name: str, vector: np.ndarray, jsonDict: dict,
                    header: bool = False, write: bool = True,
                    outputFp: typing.TextIO = sys.stdout) -> None:
    """Given a vector of statistics, calculate percentile values.

    :param name: The name of the statistic being calculated. Used for output.
    :param vector: The data that you want processed.
    :param jsonDict: A dict where you want quantile information stored.
    :param header: Should a header be written? If so, prints a row with
        the quantile cutoff values.
    :param write: Should the results be written at all? If not, they will
        still be added to the jsonDict.
    :param outputFp: The (opened) file object where output should be written.

    Doesn't return anything, but does put information in jsonDict.
    """
    quantileCutoffs = np.linspace(0, 1, 5)
    if vector.shape[0] < 5:
        vector = np.zeros((5,))
    quantiles = np.quantile(vector, quantileCutoffs)
    jsonDict[name] = {"quantile-cutoffs": list(quantileCutoffs),
                      "quantiles": list(quantiles)}
    if header:
        cutoffStr = "".join([f"\t{x * 100:14f}%" for x in quantileCutoffs])
        outputFp.write(f"metric    {cutoffStr}\tregions\n")
    if write:
        quantileStr = "".join([f"\t{x:15f}" for x in quantiles])

        outputFp.write(f"{name:10s}{quantileStr}\t{vector.shape[0]:d}\n")


def receiveThread(numRegions: int, outputQueue: CrashQueue,
                  skipZeroes: bool, jsonOutput: bool, jsonDict: dict,
                  outputFile: str | None) -> None:
    """Listen to the output from the calculator threads and process it.

    :param numRegions: How many total regions will be calculated?
        This is calculated inside regionGenThread and passed back through
        numberQueue.
    :param outputQueue: The queue that the calculator threads are putting
        their results in.
    :param skipZeroes: Should regions where the metrics are undefined be
        filtered out? If not, your results will be contaminated with NaN,
        but this can also be a good indication that something is wrong.
    :param jsonOutput: Should a json file be written? If so, the normal
        tabular output will not be printed.
    :param jsonDict: Any additional information you'd like included in
        your json file, like the names of the files that were processed.
    :param outputFile: The name of the file where the output should be saved.
        If this is ``None``, then write to stdout.
    """
    mnlls = np.zeros((numRegions,))
    jsds = np.zeros((numRegions,))
    pearsonrs = np.zeros((numRegions,))
    spearmanrs = np.zeros((numRegions,))
    referenceCounts = np.zeros((numRegions,))
    predictedCounts = np.zeros((numRegions,))
    pbar = logUtils.wrapTqdm(range(numRegions), logUtils.INFO)
    for _ in pbar:
        ret = outputQueue.get()
        (regionID, mnllVal, jsd, pearsonr, spearmanr, referenceCount, predictedCount) = ret
        mnlls[regionID] = mnllVal
        jsds[regionID] = jsd
        pearsonrs[regionID] = pearsonr
        spearmanrs[regionID] = spearmanr
        referenceCounts[regionID] = referenceCount
        predictedCounts[regionID] = predictedCount
    if skipZeroes:
        def norm(vector: NDArray) -> NDArray:
            return vector[np.isfinite(vector)]
        mnlls = norm(mnlls)
        jsds = norm(jsds)
        pearsonrs = norm(pearsonrs)
        spearmanrs = norm(spearmanrs)
        countsSelection = np.logical_and(np.isfinite(referenceCounts),
                                         np.isfinite(predictedCounts))
        referenceCounts = referenceCounts[countsSelection]
        predictedCounts = predictedCounts[countsSelection]

    # Calculate the percentiles for each of the profile metrics.
    if outputFile is None:
        outputFp = sys.stdout
    else:
        outputFp = open(outputFile, "w")  # pylint: disable=consider-using-with
    w = not jsonOutput
    percentileStats("mnll", mnlls, jsonDict, header=w, write=w, outputFp=outputFp)
    percentileStats("jsd", jsds, jsonDict, header=False, write=w, outputFp=outputFp)
    percentileStats("pearsonr", pearsonrs, jsonDict, header=False, write=w, outputFp=outputFp)
    percentileStats("spearmanr", spearmanrs, jsonDict, header=False, write=w, outputFp=outputFp)
    if referenceCounts.shape[0] > 2:
        countsPearson = scipy.stats.pearsonr(referenceCounts, predictedCounts)
        countsSpearman = scipy.stats.spearmanr(referenceCounts, predictedCounts)
    else:
        countsPearson = countsSpearman = [np.nan, np.nan]
    if not jsonOutput:
        outputFp.write(f"Counts pearson \t{countsPearson[0]:10f}\n")
        outputFp.write(f"Counts spearman\t{countsSpearman[0]:10f}\n")
    else:
        jsonDict["counts-pearson"] = countsPearson[0]
        jsonDict["counts-spearman"] = countsSpearman[0]

    if jsonOutput:
        outputFp.write(json.dumps(jsonDict, indent=4))
    outputFp.close()


def runMetrics(reference: str, predicted: str, regions: str, threads: int, applyAbs: bool,
               skipZeroes: bool, jsonOutput: bool, outputFile: str | None) -> None:
    """Run the calculation.

    :param reference: The name of the bigwig file with reference data.
    :param predicted: The name of the bigwig file with predictions.
    :param regions: The name of the bed file with regions to analyze.
    :param threads: How many parallel workers should be used?
    :param applyAbs: Should all values in the bigwigs be made positive?
    :param skipZeroes: If one of the metrics from one region is NaN,
        should it be ignored (skipZeros = True) or should all the results
        be contaminated with NaN (skipZeros = False)?
    :param jsonOutput: Should json output be written instead of a table?
    :param outputFile: If not None, gives the name of a file that the output should
        be saved to.

    Doesn't return anything, but will print to stdout.
    """
    regionQueue = CrashQueue()
    resultQueue = CrashQueue()
    numberQueue = CrashQueue()
    if not jsonOutput:
        print(f"reference {reference} predicted {predicted} regions {regions}")  # noqa: T201
    regionThread = Process(target=regionGenThread,
                           args=(regions, regionQueue, threads, numberQueue),
                           daemon=True)
    processorThreads = []
    for i in range(threads):
        logUtils.debug(f"Creating thread {i}")
        newThread = Process(target=calculatorThread,
                            args=(reference, predicted, applyAbs, regionQueue, resultQueue, i),
                            daemon=True)
        processorThreads.append(newThread)
        newThread.start()
    regionThread.start()
    # Since the number of regions is not known before starting up the regions thread, I have to
    # start the regions thread and then get a message telling me how many regions there are.
    # Clunky, but avoids re-reading the bed file.
    numRegions = numberQueue.get()

    writerThread = Process(target=receiveThread,
        args=(numRegions, resultQueue, skipZeroes, jsonOutput,
              {"reference": reference, "predicted": predicted, "regions": regions},
              outputFile),
        daemon=True)

    writerThread.start()
    writerThread.join()


def getParser() -> argparse.ArgumentParser:
    """Generate the argument parser."""
    parser = argparse.ArgumentParser(description="Take two bigwig-format files and calculate "
                                                 "an assortment of metrics on their contents.")
    parser.add_argument("--reference",
            help="The name of the reference (i.e., experimental) bigwig.",
            required=True)
    parser.add_argument("--predicted",
            help="The name of the bigwig file generated by a model.",
            required=True)
    parser.add_argument("--regions",
            help="The name of a bed file containing the regions to use to calculate the metrics.",
            required=True)
    parser.add_argument("--verbose",
            help="Display progress as the file is being written. "
                 "Cannot be used with --json-output.",
            action="store_true")
    parser.add_argument("--threads",
            help="Number of parallel threads to use. Default 1.",
            default=1,
            type=int)
    parser.add_argument("--skip-zeroes",
            help="When a region has zero counts, the default behavior is to poison the results"
                 "with NaN, to indicate that a problem has occurred. If this flag is set,"
                 "then regions with zero counts in either bigwig will be silently skipped.",
            action="store_true",
            dest="skipZeroes")
    parser.add_argument("--json-output",
            help="Instead of producing a human-readable output, "
                 "generate a machine-readable json file. Cannot be used with --verbose.",
            action="store_true",
            dest="jsonOutput")
    parser.add_argument("--output-file",
            help="Instead of writing to stdout, write to this file.",
            dest="outputFile")
    parser.add_argument("--apply-abs",
            help="Use the absolute value of the entries in the bigwig. Useful if one bigwig "
                 "contains negative values.",
            action="store_true",
            dest="applyAbs")
    return parser


def main() -> None:
    """Run the whole thing."""
    args = getParser().parse_args()
    if args.verbose:
        logUtils.setVerbosity("INFO")
        assert not args.jsonOutput, "--json-output and --verbose cannot be specified together."
    else:
        logUtils.setVerbosity("WARNING")

    runMetrics(args.reference, args.predicted, args.regions, args.threads,
               args.applyAbs, args.skipZeroes, args.jsonOutput, args.outputFile)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
