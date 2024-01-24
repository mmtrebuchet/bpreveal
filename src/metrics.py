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
import pyBigWig
import numpy as np
import argparse
import logging
import scipy.stats
import scipy.spatial.distance
from bpreveal import utils
import json
from bpreveal.utils import QUEUE_TIMEOUT
from multiprocessing import Process, Queue


class Region:
    def __init__(self, line):
        lsp = line.split()
        self.chrom = lsp[0]
        self.start = int(lsp[1])
        self.stop = int(lsp[2])


class MetricsCalculator:
    def __init__(self, referenceBwFname, predictedBwFname, applyAbs, inQueue, outQueue, tid):
        logging.debug("Starting thread {0:d} for metrics calculation.".format(tid))
        self.referenceBw = pyBigWig.open(referenceBwFname, "r")
        self.predictedBw = pyBigWig.open(predictedBwFname, "r")
        self.applyAbs = applyAbs
        self.inQueue = inQueue
        self.outQueue = outQueue
        self.tid = tid

    def runRegions(self, regionReference, regionPredicted, regionID):

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
        self.outQueue.put(ret, timeout=QUEUE_TIMEOUT)

    def run(self):
        while (True):
            match self.inQueue.get(timeout=QUEUE_TIMEOUT):
                case None:
                    logging.debug("Finishing calculator thread {0:d}.".format(self.tid))
                    self.finish()
                    return
                case (regionRef, regionPred, regionID):
                    self.runRegions(regionRef, regionPred, regionID)

    def finish(self):
        self.referenceBw.close()
        self.predictedBw.close()


def calculatorThread(referenceBwFname, predictedBwFname, applyAbs, inQueue, outQueue, tid):
    calc = MetricsCalculator(referenceBwFname, predictedBwFname, applyAbs, inQueue, outQueue, tid)
    calc.run()


def regionGenThread(regionsFname, regionQueue, numThreads, numberQueue):
    logging.debug("Initializing generator.")
    regions = [Region(x) for x in open(regionsFname, 'r')]
    logging.info("Number of regions: {0:d}".format(len(regions)))
    numberQueue.put(len(regions), timeout=QUEUE_TIMEOUT)
    for i, r in enumerate(regions):
        regionQueue.put((r, r, i), timeout=QUEUE_TIMEOUT)
    for i in range(numThreads):
        regionQueue.put(None, timeout=QUEUE_TIMEOUT)
    logging.debug("Generator done.")


def percentileStats(name, vector, jsonDict, header=False, write=True):
    quantileCutoffs = np.linspace(0, 1, 5)
    if vector.shape[0] < 5:
        vector = np.zeros((5,))
    quantiles = np.quantile(vector, quantileCutoffs)
    jsonDict[name] = {"quantile-cutoffs": list(quantileCutoffs),
                      "quantiles": list(quantiles)}
    if header:
        print("{0:10s}".format("metric")
              + "".join(["\t{0:14f}%".format(x * 100) for x in quantileCutoffs])
              + "\t{0:s}".format("regions"))
    if write:
        print("{0:10s}".format(name)
              + "".join(["\t{0:15f}".format(x) for x in quantiles])
              + "\t{0:d}".format(vector.shape[0]))


def receiveThread(numRegions, outputQueue, skipZeroes, jsonOutput, jsonDict):
    mnlls = np.zeros((numRegions,))
    jsds = np.zeros((numRegions,))
    pearsonrs = np.zeros((numRegions,))
    spearmanrs = np.zeros((numRegions,))
    referenceCounts = np.zeros((numRegions,))
    predictedCounts = np.zeros((numRegions,))
    pbar = utils.wrapTqdm(range(numRegions), logging.INFO)
    for _ in pbar:  # type: ignore
        ret = outputQueue.get(timeout=QUEUE_TIMEOUT)
        (regionID, mnllVal, jsd, pearsonr, spearmanr, referenceCount, predictedCount) = ret
        mnlls[regionID] = mnllVal
        jsds[regionID] = jsd
        pearsonrs[regionID] = pearsonr
        spearmanrs[regionID] = spearmanr
        referenceCounts[regionID] = referenceCount
        predictedCounts[regionID] = predictedCount
    if skipZeroes:
        def norm(vector):
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
    w = not jsonOutput
    percentileStats("mnll", mnlls, jsonDict, header=w, write=w)
    percentileStats("jsd", jsds, jsonDict, header=False, write=w)
    percentileStats("pearsonr", pearsonrs, jsonDict, header=False, write=w)
    percentileStats("spearmanr", spearmanrs, jsonDict, header=False, write=w)

    countsPearson = scipy.stats.pearsonr(referenceCounts, predictedCounts)
    countsSpearman = scipy.stats.spearmanr(referenceCounts, predictedCounts)
    if not jsonOutput:
        print("Counts pearson \t{0:10f}".format(countsPearson[0]))
        print("Counts spearman\t{0:10f}".format(countsSpearman[0]))
    else:
        jsonDict["counts-pearson"] = countsPearson[0]
        jsonDict["counts-spearman"] = countsSpearman[0]

    if jsonOutput:
        print(json.dumps(jsonDict, indent=4))


def runMetrics(reference, predicted, regions, threads, applyAbs, skipZeroes, jsonOutput):
    regionQueue = Queue()
    resultQueue = Queue()
    numberQueue = Queue()
    if not jsonOutput:
        print("reference {0:s} predicted {1:s} regions {2:s}".format(reference, predicted, regions))
    regionThread = Process(target=regionGenThread,
                           args=(regions, regionQueue, threads, numberQueue),
                           daemon=True)
    processorThreads = []
    for i in range(threads):
        logging.debug("Creating thread {0:d}".format(i))
        newThread = Process(target=calculatorThread,
                            args=(reference, predicted, applyAbs, regionQueue, resultQueue, i),
                            daemon=True)
        processorThreads.append(newThread)
        newThread.start()
    regionThread.start()
    # Since the number of regions is not known before starting up the regions thread, I have to
    # start the regions thread and then get a message telling me how many regions there are.
    # Clunky, but avoids re-reading the bed file.
    numRegions = numberQueue.get(timeout=QUEUE_TIMEOUT)

    writerThread = Process(target=receiveThread,
        args=(numRegions, resultQueue, skipZeroes, jsonOutput,
        {"reference": reference, "predicted": predicted, "regions": regions}),
        daemon=True)

    writerThread.start()
    writerThread.join()


def getParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Take two bigwig-format files and calculate "
                                                 "an assortment of metrics on their contents.")
    parser.add_argument("--reference",
            help='The name of the reference (i.e., experimental) bigwig.',
            required=True)
    parser.add_argument("--predicted",
            help='The name of the bigwig file generated by a model.',
            required=True)
    parser.add_argument("--regions",
            help="The name of a bed file containing the regions to use to calculate the metrics.",
            required=True)
    parser.add_argument("--verbose",
            help="Display progress as the file is being written. "
                 "Cannot be used with --json-output.",
            action='store_true')
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
    parser.add_argument("--apply-abs",
            help="Use the absolute value of the entries in the bigwig. Useful if one bigwig "
                 "contains negative values.",
            action='store_true',
            dest='applyAbs')
    return parser


def main():
    args = getParser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        assert not args.jsonOutput, "--json-output and --verbose cannot be specified together."
    else:
        logging.basicConfig(level=logging.WARNING)

    runMetrics(args.reference, args.predicted, args.regions, args.threads,
               args.applyAbs, args.skipZeroes, args.jsonOutput)


if __name__ == "__main__":
    main()
