#!/usr/bin/env python3
"""Generates test, train, and validation splits and optionally performs some filtering.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/prepareBed.bnf


Parameter Notes
---------------

bigwig-names
    A list of the data bigwigs that correspond to this head.
    For example, these might be the positive and negative strands of a ChIP-nexus
    sample.

resize-mode
    specifies where in the regions in the bed file the output regions should
    be centered.
    Note that this program assumes your bed files are in bed3 format, that
    is, (chrom, start, stop).
    If you have additional columns with information like peak offset, those
    data will be ignored.

max-quantile, min-quantile
    The max and min quantile values, if provided, will be used to
    threshold which regions are included in the output.
    First, all of the counts in the given regions are computed (which takes
    a while!), and then the given quantile is computed.
    All regions exceeding that value are not included in the output files.

max-counts, min-counts
    Similarly, if max and min counts are given, all regions having more
    (or fewer) reads than the given number will be excluded.

output-prefix
    Specifies the base name for the output bed files. You can use either
    output-prefix OR list all three output-train, output-val, and output-test.
    If you specify output-prefix, then five bed files will be made, called
    ``output-prefix_train.bed``, ``output-prefix_val.bed``,
    ``output-prefix_test.bed``, ``output-prefix_all.bed``, and
    ``output-prefix_reject.bed``

output-train, output-val, output-test
    If you give these file names, then the training, validation and test splits
    will be written to these three files, respectively.

regions
    Needed for splits by chromosome or by regex. This is a bed file of every
    possible region that you might want to train on.

train-chroms, val-chroms, test-chroms
    Split up your input regions by chromosome.

train-regex, val-regex, test-regex
    If you use a regex, then the name field of each bed line will be matched
    against each of the three regexes.
    The line will be added to the split where it matches.
    If a bed line matches more than one regex, that will raise an error.
    If a line matches no regexes, it is added to the rejects.

train-regions, val-regions, test-regions
    You may provide a specific bed file for each of the splits. In this case,
    the regions in each of these files are used to construct each respective
    split.

remove-overlaps
    flag can be set to ``true`` if you'd
    like to exclude overlapping regions.
    This is done by resizing all regions down to
    ``overlap-max-distance``, and then, if multiple regions have an
    overlap, one is deleted at random.
    If ``remove-overlaps`` is ``false``, then
    it is an error to set ``overlap-max-distance``.

num-threads
    How many threads should be used for loading counts information?
    I recommend setting this to as many threads as your machine has.

Additional information
----------------------

Counts windowing
^^^^^^^^^^^^^^^^

I should mention that the maximum and minimum counts are not compared across the
same window.
When comparing a region against the maximum counts value, all counts within a
window of size ``input-length + 2*jitter`` are added up.
This way, if you have a crazy-huge spike just outside your region, that region
will be rejected if the jittering could include it in the training data.
Conversely, for minimum counts, the counts within a window of length
``output-length - 2*jitter`` will be considered.
This way, no matter what jitter value is selected, there will be at least the
given number of counts in the region.

Most columns ignored
^^^^^^^^^^^^^^^^^^^^

prepareBed takes a very lenient approach to validating your bed files.
It will not check that the score column in your file is numeric, nor will
it check to see if you have flipped some columns in your input file.


History
-------

The old ``bigwigs`` format was deprecated in BPReveal 4.0.0 and will be
removed in BPReveal 5.0.0

The ``remove-overlaps`` field became mandatory in BPReveal 3.0.0.

API
---

"""
# This file contains several helper functions for dealing with bed files.

import random
import re
import jsonschema
import pysam
import numpy as np
import pybedtools
from bpreveal import logUtils
from bpreveal.bedUtils import resize, sequenceChecker, lineToInterval, ParallelCounter
from bpreveal.internal import interpreter
random.seed(735014)


def loadRegionsByChrom(trainChroms: list[str], valChroms: list[str],
                       testChroms: list[str], regionFnames: list[str]) -> \
        tuple[list[pybedtools.Interval], list[pybedtools.Interval],
              list[pybedtools.Interval], list[pybedtools.Interval]]:
    """Load splits based on chromosomes.

    Given chromosomes for the training, validation, and test splits, and a list of
    bed files, generate the regions in each split.

    :param trainChroms: A list of chromosome names for the training split.
    :param valChroms: A list of chromosomes for the validation split.
    :param testChroms: A list of chromosome names for the test split.
    :param regionFnames: A list of bed file names to read in.
    :return: The training, test, validation, and reject regions as lists.
    """
    trainRegions = []
    testRegions = []
    valRegions = []
    rejectRegions = []
    for bedFile in regionFnames:
        with open(bedFile) as fp:
            for line in fp:
                if r := lineToInterval(line):
                    if r.chrom in trainChroms:
                        trainRegions.append(r)
                    elif r.chrom in valChroms:
                        valRegions.append(r)
                    elif r.chrom in testChroms:
                        testRegions.append(r)
                    else:
                        logUtils.debug(f"        Rejected region {line.strip()} because "
                                       "it's not in any of the chromosome sets.")
                        rejectRegions.append(r)
    return trainRegions, testRegions, valRegions, rejectRegions


def loadRegionsByBed(trainRegionFnames: list[str], valRegionFnames: list[str],
                     testRegionFnames: list[str]) -> \
        tuple[list[pybedtools.Interval], list[pybedtools.Interval],
              list[pybedtools.Interval], list[pybedtools.Interval]]:
    """Given bed file names, load them into lists of intervals.

    :param trainRegionFnames: A list of bed files containing regions to train on.
    :param valRegionFnames: A list of bed files containing regions to use for validation.
    :param testRegionFnames: A list of bed files for the test set of regions.
    :return: Four lists of regions: Training, test, validation, and rejects.
        (Rejects will always be empty with this function.)
    """
    trainRegions = []
    testRegions = []
    valRegions = []
    for trainBedFile in trainRegionFnames:
        with open(trainBedFile) as fp:
            for line in fp:
                if r := lineToInterval(line):
                    trainRegions.append(r)
    for valBedFile in valRegionFnames:
        with open(valBedFile) as fp:
            for line in fp:
                if r := lineToInterval(line):
                    valRegions.append(r)
    for testBedFile in testRegionFnames:
        with open(testBedFile) as fp:
            for line in fp:
                if r := lineToInterval(line):
                    testRegions.append(r)
    return trainRegions, testRegions, valRegions, []


def loadRegionsByRegex(trainString: str, testString: str, valString: str,
                       regionFnames: list[str]) -> \
        tuple[list[pybedtools.Interval], list[pybedtools.Interval],
              list[pybedtools.Interval], list[pybedtools.Interval]]:
    """Go over the bed files and assign splits based on regexes matched against the name column.

    :param trainString: The regex that matches samples in the training split
    :param testString: The regex that matches samples in the test split
    :param valString: The regex that matches samples in the validation split.
    :param regionFnames: A list of bed files that will be read in.
    :return: Four lists of Intervals, corresponding to the training, test, validation,
        and rejected regions.
    """
    trainRegions = []
    testRegions = []
    valRegions = []
    rejectRegions = []
    trainRegex = re.compile(trainString)
    valRegex = re.compile(valString)
    testRegex = re.compile(testString)
    for bedFile in regionFnames:
        with open(bedFile) as fp:
            for line in fp:
                if r := lineToInterval(line):
                    foundTrain = False
                    foundVal = False
                    foundTest = False
                    if trainRegex.search(r.name) is not None:
                        foundTrain = True
                        trainRegions.append(r)
                    if valRegex.search(r.name) is not None:
                        assert not foundTrain, f"Region {line.strip()} matches multiple regexes."
                        foundVal = True
                        valRegions.append(r)
                    if testRegex.search(r.name) is not None:
                        assert not (foundTrain or foundVal), f"Region {line.strip()} matches "\
                                                             "multiple regexes."
                        foundTest = True
                        testRegions.append(r)
                    if not (foundTrain or foundVal or foundTest):
                        logUtils.debug(f"        Rejected region {line.strip()} because it "
                                       "didn't match any of your split regexes.")
                        rejectRegions.append(r)
    return trainRegions, testRegions, valRegions, rejectRegions


def loadRegions(config: dict) -> tuple[pybedtools.BedTool, pybedtools.BedTool,
                                       pybedtools.BedTool, pybedtools.BedTool]:
    """Given a configuration (see the specification), return four PyBedTools BedTool objects.

    :param config: A JSON object satisfying the prepareBed specification.
    :return: Four BedTools:

        1. The first will consist of the training regions,
        2. the second will be the validation regions,
        3. then the test regions,
        4. finally any regions that were rejected on loading.
    """
    trainRegions = []
    testRegions = []
    valRegions = []
    rejectRegions = []
    match config["splits"]:
        case {"train-chroms": trainChroms, "val-chroms": valChroms,
              "test-chroms": testChroms, "regions": regionFnames}:
            trainRegions, testRegions, valRegions, rejectRegions = \
                loadRegionsByChrom(trainChroms, valChroms, testChroms, regionFnames)
        case {"train-regions": trainRegionFnames, "val-regions": valRegionFnames,
              "test-regions": testRegionFnames}:
            trainRegions, testRegions, valRegions, rejectRegions = \
                loadRegionsByBed(trainRegionFnames, valRegionFnames,
                                 testRegionFnames)
        case {"train-regex": trainString, "val-regex": valString,
              "test-regex": testString, "regions": regionFnames}:
            trainRegions, testRegions, valRegions, rejectRegions = \
                loadRegionsByRegex(trainString, testString, valString, regionFnames)
        case _:
            raise ValueError(f"Config invalid: {config['splits']}")

    logUtils.info(f"Training regions: {len(trainRegions)}")
    logUtils.info(f"Validation regions: {len(valRegions)}")
    logUtils.info(f"Testing regions: {len(testRegions)}")
    logUtils.info(f"Rejected on loading: {len(rejectRegions)}")

    return (pybedtools.BedTool(trainRegions),
            pybedtools.BedTool(valRegions),
            pybedtools.BedTool(testRegions),
            pybedtools.BedTool(rejectRegions))


def removeOverlaps(config: dict, regions: pybedtools.BedTool,
                   genome: pysam.FastaFile) -> tuple[pybedtools.BedTool, pybedtools.BedTool]:
    """Remove overlaps among the given regions.

    :param config: Straight from the JSON.
    :param regions: A BedTool (or list of Intervals)
    :param genome: A FastaFile (not string) giving the genome.

    Takes in the list of regions, resizes each to the minimum size, and if there are overlaps,
    randomly chooses one of the overlapping regions.
    """
    # Resize the regions down to the minimum size.
    resizedRegions = regions.each(resize,
                                  config["resize-mode"],
                                  config["overlap-max-distance"], genome).saveas()
    # The algorithm here requires that the regions be sorted.
    sortedRegions = resizedRegions.sort()
    piles = []
    curPile = [sortedRegions[0]]
    for r in sortedRegions:
        if curPile[0].chrom == r.chrom and curPile[0].end > r.start:
            # We have an overlap.
            curPile.append(r)
        else:
            # No overlap, commit and reset the pile.
            piles.append(curPile)
            curPile = [r]
    if len(curPile) > 0:
        piles.append(curPile)
    ret = []
    rejects = []
    for pile in piles:
        selectedIdx = random.randrange(0, len(pile))
        for i, elem in enumerate(pile):
            if i == selectedIdx:
                ret.append(elem)
            else:
                printStr = str(elem).strip()
                logUtils.debug(f"        Rejected region {printStr} because it overlaps.")
                rejects.append(elem)
    return (pybedtools.BedTool(ret), pybedtools.BedTool(rejects))


def filterByMaxCounts(config: dict, bigRegionsList: list[pybedtools.Interval],
                      bigwigLists: list[list[str]], validRegions: np.ndarray,
                      numThreads: int) -> pybedtools.BedTool:
    """Filters the regions in bigRegionList based on the max-quantile or max-counts in the config.

    :param config: Straight from the configuration JSON.
    :param bigRegionsList: A list of intervals that have already been inflated to account for jitter
    :param bigwigLists: The bigwigs that should be scanned, grouped by head.
    :param validRegions: A vector booleans, for each region in bigRegionList, if region i is
        rejected, then validRegions[i] will be 0 when this function exits.
    :param numThreads: How many parallel workers should be used?
    :return: A BedTool containing only valid regions.
    """
    pbar = logUtils.wrapTqdm(len(bigRegionsList) * len(config["heads"]) * 2, logUtils.INFO)
    for i, headSpec in enumerate(config["heads"]):
        if "max-quantile" in headSpec:
            if headSpec["max-quantile"] == 1:
                # Don't reject any regions. Since validRegions starts with all ones
                # (i.e., all regions are valid), we just jump to the next head to see
                # if we need to look at max counts there.
                continue
        # Get the counts over every region.
        counter = ParallelCounter(bigwigLists[i], numThreads)
        bigCounts = np.zeros((len(bigRegionsList),))
        for j, r in enumerate(bigRegionsList):
            counter.addQuery((r.chrom, r.start, r.end), j)
            pbar.update()
        for _ in range(len(bigRegionsList)):
            val, idx = counter.getResult()
            bigCounts[idx] = val
            pbar.update()
        counter.done()
        if "max-counts" in headSpec:
            maxCounts = headSpec["max-counts"]
        else:
            maxCounts = np.quantile(bigCounts, [headSpec["max-quantile"]])[0]
        logUtils.debug(f"    Max counts: {maxCounts}, file {headSpec['bigwig-names']}")
        numReject = 0
        for regionIdx in range(len(bigRegionsList)):
            if bigCounts[regionIdx] > maxCounts:
                numReject += 1
                validRegions[regionIdx] = 0
        fracReject = numReject * 100 / len(bigRegionsList)
        logUtils.debug(f"    Rejected {fracReject}% of regions for having too many counts")
    pbar.close()
    # We've now validated that the regions don't have too many counts when you inflate them.
    # We also need to check that the regions won't have too few counts in the output.
    logUtils.info(f"    Validated inflated regions. Surviving: {int(np.sum(validRegions))}")
    bigRegionsBed = pybedtools.BedTool(bigRegionsList)
    return bigRegionsBed


def filterByMinCounts(config: dict, smallRegionsList: list[pybedtools.Interval],
                      bigRegionsList: list[pybedtools.Interval],
                      bigwigLists: list[list[str]], validRegions: np.ndarray,
                      numThreads: int) -> None:
    """Filters the regions in smallRegionList based on the min-quantile or min-counts in the config.

    :param config: Straight from the configuration JSON.
    :param bigRegionsList: A list of intervals that have already been inflated to account for jitter
    :param smallRegionsList: A list of intervals that have already been inflated to
        account for jitter
    :param bigwigLists: The bigwigs that should be scanned, grouped by head.
    :param validRegions: A vector booleans, for each region k in smallRegionList,
        corresponding to region i in bigRegionsList, if region k is rejected,
        then validRegions[i] will be 0 when this function exits.
    :param numThreads: How many parallel workers should be used?
    """
    pbar = logUtils.wrapTqdm(len(smallRegionsList) * len(config["heads"]) * 2, logUtils.INFO)
    for i, headSpec in enumerate(config["heads"]):
        # Since this is a slow step, check to see if min counts is zero. If so, no need to filter.
        if "min-quantile" in headSpec:
            if headSpec["min-quantile"] == 0:
                continue
        smallCounts = np.zeros((len(smallRegionsList),))
        counter = ParallelCounter(bigwigLists[i], numThreads)
        for j, r in enumerate(smallRegionsList):
            counter.addQuery((r.chrom, r.start, r.end), j)
            pbar.update()
        for _ in range(len(smallRegionsList)):
            val, idx = counter.getResult()
            smallCounts[idx] = val
            pbar.update()
        counter.done()
        if "min-counts" in headSpec:
            minCounts = headSpec["min-counts"]
        else:
            minCounts = np.quantile(smallCounts, [headSpec["min-quantile"]])[0]
        logUtils.debug(f"    Min counts: {minCounts}, file {headSpec['bigwig-names']}")
        numReject = 0
        for regionIdx in range(len(bigRegionsList)):
            # within len(bigRegions) in case a region was lost during the resize - we want that to
            # crash because resizing down should never invalidate a region due to sequence problems.
            if smallCounts[regionIdx] < minCounts:
                numReject += 1
                validRegions[regionIdx] = 0
        fracReject = numReject * 100 / len(bigRegionsList)
        logUtils.debug(f"    Rejected {fracReject}% of small regions.")
    pbar.close()


def validateRegions(config: dict, regions: pybedtools.BedTool,
                    genome: pysam.FastaFile, bigwigLists: list[list[str]],
                    numThreads: int) -> tuple[pybedtools.BedTool, pybedtools.BedTool]:
    """The workhorse of this program.

    :param config: Straight from the JSON.
    :param regions: A BedTool or list.
    :param genome: A FastaFile (not the name as a str.)
    :param bigwigLists: The names of the data files to use.
    :param numThreads: How many parallel workers should be used?
    :return: Two BedTools, one for regions that passed the filters and another for
        those that failed.

    Given a config (see the spec), a BedTool of regions, an open pysam FastaFile, and a list of
    bigwigs to check, filter down the regions so that they satisfy the configuration.
    Returns two BedTools: The first contains the regions that passed the filters, and the second
    contains the rejected regions.
    """
    # First, I want to eliminate any regions that are duplicates. To do this, I'll resize all of
    # the regions to the minimum size, then sort them and remove overlaps.
    if config["remove-overlaps"]:
        noOverlapRegions, initialRejects = removeOverlaps(config, regions, genome)
        noOverlapRegions = noOverlapRegions.saveas()
        initialRejects = initialRejects.saveas()
        initialRegions = noOverlapRegions
        logUtils.info(f"    Removed overlaps, {noOverlapRegions.count()} regions remain.")
    else:
        initialRegions = regions
        if "overlap-max-distance" in config:
            logUtils.warning("    You have set remove-overlaps to false, but you still provided an"
                             " overlap-max-distance parameter. This parameter is meaningless.")
        logUtils.debug("    Skipping region overlap removal.")
    # Second, resize the regions to their biggest size.
    unfilteredBigRegions = initialRegions.each(resize,
                                               config["resize-mode"],
                                               config["input-length"] + config["max-jitter"] * 2,
                                               genome).saveas()
    logUtils.info(f"    Resized sequences. {unfilteredBigRegions.count()} remain.")
    bigRegionsList = list(unfilteredBigRegions.filter(sequenceChecker, genome).saveas())
    logUtils.info(f"    Filtered for weird nucleotides. {len(bigRegionsList)} remain.")
    # Now, we have the possible regions. Get their counts values.
    validRegions = np.ones((len(bigRegionsList),))
    # Note: The bigwigLists correspond to the heads in here.
    # So go over every region and measure its counts (unless max-quantile == 1)
    # and reject regions that are over-full on reads.
    bigRegionsBed = filterByMaxCounts(config, bigRegionsList, bigwigLists, validRegions, numThreads)
    smallRegionsList = list(bigRegionsBed.each(resize,
                                               "center",
                                               config["output-length"] - config["max-jitter"] * 2,
                                               genome).saveas())

    filterByMinCounts(config, smallRegionsList, bigRegionsList, bigwigLists,
                      validRegions, numThreads)
    logUtils.info(f"    Validated small regions. Surviving regions: {int(np.sum(validRegions))}")
    # Now we resize to the final output size.
    smallRegionsBed = pybedtools.BedTool(smallRegionsList)
    outRegionsBed = smallRegionsBed.each(resize,
                                         "center",
                                         config["output-length"],
                                         genome).saveas()

    # Since we kept the array of valid regions separately,
    # we now have to create the result by combing over that array
    # and picking out the remaining valid regions.

    filteredRegions = []
    rejectedRegions = []
    for i, r in enumerate(outRegionsBed):
        if validRegions[i] == 1:
            filteredRegions.append(r)
        else:
            rejectedRegions.append(r)
    logUtils.info(f"    Total surviving regions: {len(filteredRegions)}")
    if config["remove-overlaps"]:
        rejects = initialRejects.cat(pybedtools.BedTool(rejectedRegions),  # type: ignore
                                     postmerge=False)
    else:
        rejects = pybedtools.BedTool(rejectedRegions)
    return (pybedtools.BedTool(filteredRegions), rejects)


def rewriteOldBigwigsFormat(config: dict) -> None:
    """If the config has a bigwigs section, rewrite it to the new style."""
    logUtils.error("You are using a deprecated JSON format for prepareBed.py")
    logUtils.error("This will result in an error in BPReveal 6.0")
    logUtils.error("Instead of providing individual bigwig names with a \"bigwigs\"")
    logUtils.error("section in your json, use the new \"heads\" section.")
    logUtils.error("Example update: If you currently have")
    logUtils.error('"bigwigs": [{"file-name": "protein1_pos.bw",')
    logUtils.error('             "max-counts": 10000,')
    logUtils.error('             "min-counts": 10},')
    logUtils.error('            {"file-name": "protein1_neg.bw",')
    logUtils.error('             "max-counts": 10000,')
    logUtils.error('             "min-counts": 10},')
    logUtils.error('            {"file-name": "protein2_pos.bw",')
    logUtils.error('             "max-counts": 3000,')
    logUtils.error('             "min-counts": 20},')
    logUtils.error('            {"file-name": "protein2_neg.bw",')
    logUtils.error('             "max-counts": 3000,')
    logUtils.error('              "min-counts" : 20} ]')
    logUtils.error("you should update this to reflect the head structure of your model:")
    logUtils.error('"heads": [{"bigwig-names": ["protein1_pos.bw", "protein1_neg.bw"],')
    logUtils.error('           "max-counts": 20000,')
    logUtils.error('           "min-counts": 20},')
    logUtils.error('          {"bigwig-names": ["protein2_pos.bw", "protein2_neg.bw"],')
    logUtils.error('           "max-counts": 6000,')
    logUtils.error('           "min-counts": 40}]')
    logUtils.error("Note how the max-counts and min-counts values double, since the bigwigs")
    logUtils.error("in each head will be added together to determine the total counts in")
    logUtils.error("a region. (You don't need to change quantiles, though.)")

    headsConfig = []
    for bwConf in config["bigwigs"]:
        bwNames = [bwConf["file-name"]]
        bwConf["bigwig-names"] = bwNames
        del bwConf["file-name"]
        headsConfig.append(bwConf)
    logUtils.error("Your heads config has been automatically converted to the new format,")
    logUtils.error("with each bigwig being considered as its own head:")
    logUtils.error(str(headsConfig))
    config["heads"] = headsConfig


def prepareBeds(config: dict) -> None:
    """The main function of this script.

    :param config: A JSON object matching the prepareBed specification.
    """
    logUtils.info("Starting bed file generation.")
    # FUTURE: In BPReveal 6.0, raise an error inside this if block.
    # In BPReveal 7.0, remove it entirely.
    if "bigwigs" in config:
        rewriteOldBigwigsFormat(config)
    if "num-threads" not in config:
        numThreads = 1
        logUtils.warning("You have not specified a number of threads in your prepareBed config. "
                         "Defaulting to one thread. "
                         "You may see a performance gain if you set num-threads around 20.")
    else:
        numThreads = config["num-threads"]
        logUtils.debug(f"Using {numThreads} threads")

    genome = pysam.FastaFile(config["genome"])
    (trainRegions, valRegions, testRegions, rejectRegions) = loadRegions(config)
    logUtils.debug("Regions loaded.")
    if "output-prefix" in config:
        outputTrainFname = config["output-prefix"] + "_train.bed"
        outputValFname = config["output-prefix"] + "_val.bed"
        outputTestFname = config["output-prefix"] + "_test.bed"
        outputAllFname = config["output-prefix"] + "_all.bed"
        outputRejectFname = config["output-prefix"] + "_reject.bed"
    else:
        outputTrainFname = config["output-train"]
        outputValFname = config["output-val"]
        outputTestFname = config["output-test"]
        outputAllFname = config["output-all"]
        if "output-reject" in config:
            outputRejectFname = config["output-reject"]
        else:
            outputRejectFname = False

    bigwigLists = []
    for head in config["heads"]:
        bigwigLists.append(head["bigwig-names"])
    logUtils.info("Training regions validation beginning.")
    validTrain, rejectTrain = validateRegions(config, trainRegions, genome, bigwigLists, numThreads)
    logUtils.info("Validation regions validation beginning.")
    validVal, rejectVal = validateRegions(config, valRegions, genome, bigwigLists, numThreads)
    logUtils.info("Test regions validation beginning.")
    validTest, rejectTest = validateRegions(config, testRegions, genome, bigwigLists, numThreads)

    logUtils.info("Saving region lists to bed files.")
    validTrain.saveas(outputTrainFname)
    validVal.saveas(outputValFname)
    validTest.saveas(outputTestFname)
    validAll = validTrain.cat(validVal, postmerge=False).cat(validTest, postmerge=False).sort()
    validAll.saveas(outputAllFname)
    if outputRejectFname:
        allRejects = rejectRegions\
            .cat(rejectTrain, postmerge=False)\
            .cat(rejectVal, postmerge=False)\
            .cat(rejectTest, postmerge=False)\
            .sort()
        allRejects.saveas(outputRejectFname)
    logUtils.info("Bed preparation complete. Exiting.")


if __name__ == "__main__":
    import sys
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    logUtils.setVerbosity(configJson["verbosity"])
    import bpreveal.schema
    try:
        bpreveal.schema.prepareBed_old.validate(configJson)
        logUtils.error("Json validated against the old prepareBed format."
                       "This will be an error in BPReveal 6.0")
    except jsonschema.ValidationError:
        bpreveal.schema.prepareBed.validate(configJson)
    prepareBeds(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
