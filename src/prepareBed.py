#!/usr/bin/env python3
# This file contains several helper functions for dealing with bed files.

import json
import pyBigWig
import pysam
import tqdm
import logging
import utils
import numpy as np
import pybedtools
import random
import re


def resize(interval, mode, width, genome):
    start = interval.start
    end = interval.end
    match mode:
        case "none":
            if (end - start != width):
                assert False, \
                       "An input region is not the expected width: {0:s}".format(str(interval))
        case "center":
            center = (end + start) // 2
            start = center - width // 2
            end = start + width
        case "start":
            start = start - width // 2
            end = start + width
        case _:
            assert False, "Unsupported resize mode: {0:s}".format(mode)
    if (start <= 0 or end >= genome.get_reference_length(interval.chrom)):
        return False  # We're off the edge of the chromosome.
    return pybedtools.Interval(interval.chrom, start, end, name=interval.name,
                               score=interval.score, strand=interval.strand)


def getCounts(interval, bigwigs):
    total = 0
    for bw in bigwigs:
        vals = np.nan_to_num(bw.values(interval.chrom, interval.start, interval.end))
        total += np.sum(vals)
    return total


def sequenceChecker(interval, genome):
    seq = genome.fetch(interval.chrom, interval.start, interval.end)
    if (len(seq.upper().lstrip('ACGT')) != 0):
        # There were letters that aren't regular bases. (probably Ns)
        return False
    return True


def stripCountsBelow(bed, cutoff, bigwigs):
    return bed.filter(lambda interval: getCounts(interval, bigwigs) >= cutoff).saveas()


def stripCountsAbove(bed, cutoff, bigwigs):
    return bed.filter(lambda interval: getCounts(interval, bigwigs) <= cutoff).saveas()


def lineToInterval(line):
    initInterval = pybedtools.cbedtools.create_interval_from_list(line.split())
    return initInterval


def loadRegions(config):
    trainRegions = []
    testRegions = []
    valRegions = []
    rejectRegions = []
    numRejected = 0
    match config["splits"]:
        case {"train-chroms": trainChroms, "val-chroms": valChroms,
              "test-chroms": testChroms, "regions": regionFnames}:
            for bedFile in regionFnames:
                for line in open(bedFile):
                    r = lineToInterval(line)
                    if (r.chrom in trainChroms):
                        trainRegions.append(r)
                    elif (r.chrom in valChroms):
                        valRegions.append(r)
                    elif (r.chrom in testChroms):
                        testRegions.append(r)
                    else:
                        numRejected += 1
                        logging.debug("Rejected region {0:s} because it's not in any "
                                      "of the chromosome sets.".format(line.strip()))
                        rejectRegions.append(r)
        case {"train-regions": trainRegionFnames, "val-regions": valRegionFnames,
              "test-regions": testRegionFnames}:
            for trainBedFile in trainRegionFnames:
                for line in open(trainBedFile):
                    trainRegions.append(lineToInterval(line))
            for valBedFile in valRegionFnames:
                for line in open(valBedFile):
                    valRegions.append(lineToInterval(line))
            for testBedFile in testRegionFnames:
                for line in open(testBedFile):
                    testRegions.append(lineToInterval(line))
        case {"train-regex": trainString, "val-regex": valString,
              "test-regex": testString, "regions": regionFnames}:
            trainRegex = re.compile(trainString)
            valRegex = re.compile(valString)
            testRegex = re.compile(testString)
            for bedFile in regionFnames:
                for line in open(bedFile):
                    r = lineToInterval(line)
                    foundTrain = False
                    foundVal = False
                    foundTest = False
                    if (trainRegex.search(r.name) is not None):
                        foundTrain = True
                        trainRegions.append(r)
                    if (valRegex.search(r.name) is not None):
                        assert not foundTrain, "Region {0:s} matches multiple "\
                                               "regexes.".format(line)
                        foundVal = True
                        valRegions.append(r)
                    if (testRegex.search(r.name) is not None):
                        assert not (foundTrain or foundVal), "Region {0:s} matches "\
                                                             "multiple regexes.".format(line)
                        foundTest = True
                        testRegions.append(r)
                    if (not (foundTrain or foundVal or foundTest)):
                        numRejected += 1
                        logging.debug("Rejected region {0:s} because it didn't match "
                                      "any of your split regexes.".format(line.strip()))
                        rejectRegions.append(r)
        case _:
            assert False, "Config invalid: {0:s}".format(str(config["splits"]))
    logging.info("Training regions: {0:d}".format(len(trainRegions)))
    logging.info("Validation regions: {0:d}".format(len(valRegions)))
    logging.info("Testing regions: {0:d}".format(len(testRegions)))
    logging.info("Rejected on loading: {0:d}".format(len(rejectRegions)))
    return (pybedtools.BedTool(trainRegions),
            pybedtools.BedTool(valRegions),
            pybedtools.BedTool(testRegions),
            pybedtools.BedTool(rejectRegions))


def removeOverlaps(config, regions, genome):
    """Takes in the list of regions, resizes each to the minimum size, and if there are overlaps,
        randomly chooses one of the overlapping regions."""
    # Resize the regions down to the minimum size.
    resizedRegions = regions.each(resize,
                                  config['resize-mode'],
                                  config["overlap-max-distance"], genome).saveas()
    # The algorithm here requires that the regions be sorted.
    sortedRegions = resizedRegions.sort()
    piles = []
    curPile = [sortedRegions[0]]
    for r in sortedRegions:
        if (curPile[0].chrom == r.chrom and curPile[0].end > r.start):
            # We have an overlap.
            curPile.append(r)
        else:
            # No overlap, commit and reset the pile.
            piles.append(curPile)
            curPile = [r]
    if (len(curPile)):
        piles.append(curPile)
    ret = []
    rejects = []
    for pile in piles:
        selectedIdx = random.randrange(0, len(pile))
        for i, elem in enumerate(pile):
            if (i == selectedIdx):
                ret.append(elem)
            else:
                logging.debug("Rejected region {0:s} because it overlaps.".format(str(elem)))
                rejects.append(elem)
    return (pybedtools.BedTool(ret), pybedtools.BedTool(rejects))


def validateRegions(config, regions, genome, bigwigLists):

    # First, I want to eliminate any regions that are duplicates. To do this, I'll resize all of
    # the regions to the minimum size, then sort them and remove overlaps.
    if (config["remove-overlaps"]):
        noOverlapRegions, initialRejects = removeOverlaps(config, regions, genome)
        noOverlapRegions = noOverlapRegions.saveas()
        initialRejects = initialRejects.saveas()
        initialRegions = noOverlapRegions
        logging.info("Removed overlaps, {0:d} regions remain.".format(noOverlapRegions.count()))
    else:
        initialRegions = regions
        if ("overlap-max-distance" in config):
            logging.warning("You have set remove-overlaps to false, but you still provided an "
                            "overlap-max-distance parameter. This parameter is meaningless.")
        logging.debug("Skipping region overlap removal.")
    # Second, resize the regions to their biggest size.
    unfilteredBigRegions = initialRegions.each(resize,
                                               config["resize-mode"],
                                               config["input-length"] + config["max-jitter"] * 2,
                                               genome).saveas()
    logging.info("Resized sequences. {0:d} remain.".format(unfilteredBigRegions.count()))
    bigRegionsList = list(unfilteredBigRegions.filter(sequenceChecker, genome).saveas())
    logging.info("Filtered for weird nucleotides. {0:d} remain.".format(len(bigRegionsList)))
    # Now, we have the possible regions. Get their counts values.
    validRegions = np.ones((len(bigRegionsList),))
    pbar = tqdm.tqdm(total=len(bigRegionsList) * len(bigwigLists))
    # Note: The bigwigLists correspond to the heads in here.
    # So go over every region and measure its counts (unless max-quantile == 1)
    # and reject regions that are over-full on reads.
    for i, headSpec in enumerate(config["heads"]):
        if ("max-quantile" in headSpec):
            if (headSpec["max-quantile"] == 1):
                # Don't reject any regions. Since validRegions starts with all ones
                # (i.e., all regions are valid), we just jump to the next head to see
                # if we need to look at max counts there.
                continue
        # Get the counts over every region.
        bigCounts = np.zeros((len(bigRegionsList),))
        for j, r in enumerate(bigRegionsList):
            bigCounts[j] = getCounts(r, bigwigLists[i])
            pbar.update()
        if ("max-counts" in headSpec):
            maxCounts = headSpec["max-counts"]
        else:
            maxCounts = np.quantile(bigCounts, [headSpec["max-quantile"]])[0]
        logging.debug("Max counts: {0:s}, file {1:s}".format(str(maxCounts),
                                                             str(headSpec["bigwig-names"])))
        numReject = 0
        for regionIdx in range(len(bigRegionsList)):
            if (bigCounts[regionIdx] > maxCounts):
                numReject += 1
                validRegions[regionIdx] = 0
        logging.debug("Rejected {0:f}% of regions for having too many"
            "counts.".format(numReject * 100. / len(bigRegionsList)))
    pbar.close()

    # We've now validated that the regions don't have too many counts when you inflate them.
    # We also need to check that the regions won't have too few counts in the output.
    logging.info("Validated inflated regions. Surviving: {0:d}".format(int(np.sum(validRegions))))
    pbar = tqdm.tqdm(total=len(bigRegionsList) * len(bigwigLists))
    bigRegionsBed = pybedtools.BedTool(bigRegionsList)
    smallRegionsList = list(bigRegionsBed.each(resize,
                                               'center',
                                               config["output-length"] - config["max-jitter"] * 2,
                                               genome).saveas())
    for i, headSpec in enumerate(config["heads"]):
        # Since this is a slow step, check to see if min counts is zero. If so, no need to filter.
        if ("min-quantile" in headSpec):
            if (headSpec["min-quantile"] == 0):
                continue
        smallCounts = np.zeros((len(smallRegionsList),))
        for j, r in enumerate(smallRegionsList):
            smallCounts[j] = getCounts(r, bigwigLists[i])
            pbar.update()
        if ("min-counts" in headSpec):
            minCounts = headSpec["min-counts"]
        else:
            minCounts = np.quantile(smallCounts, [headSpec["min-quantile"]])[0]
        logging.debug("Min counts: {0:s}, file {1:s}".format(str(minCounts),
                                                             str(headSpec["bigwig-names"])))
        numReject = 0
        for regionIdx in range(len(bigRegionsList)):
            # within len(bigRegions) in case a region was lost during the resize - we want that to
            # crash because resizing down should never invalidate a region due to sequence problems.
            if (smallCounts[regionIdx] < minCounts):
                numReject += 1
                validRegions[regionIdx] = 0
        logging.debug("Rejected {0:f}% of small regions."
                      .format(numReject * 100. / len(bigRegionsList)))
    pbar.close()
    logging.info("Validated small regions. Surviving regions: {0:d}"
                 .format(int(np.sum(validRegions))))
    # Now we resize to the final output size.
    smallRegionsBed = pybedtools.BedTool(smallRegionsList)
    outRegionsBed = smallRegionsBed.each(resize,
                                         'center',
                                         config["output-length"],
                                         genome).saveas()

    # Since we kept the array of valid regions separately,
    # we now have to create the result by combing over that array
    # and picking out the remaining valid regions.

    filteredRegions = []
    rejectedRegions = []
    for i, r in enumerate(outRegionsBed):
        if (validRegions[i] == 1):
            filteredRegions.append(r)
        else:
            rejectedRegions.append(r)
    logging.info("Total surviving regions: {0:d}".format(len(filteredRegions)))
    if (config["remove-overlaps"]):
        rejects = initialRejects.cat(pybedtools.BedTool(rejectedRegions), postmerge=False)
    else:
        rejects = pybedtools.BedTool(rejectedRegions)
    return (pybedtools.BedTool(filteredRegions), rejects)


def prepareBeds(config):
    logging.info("Starting bed file generation.")
    # FUTURE: In BPReveal 5.0, raise an error inside this if block.
    # In BPReveal 7.0, remove it entirely.
    if "bigwigs" in config:
        logging.warning("You are using a deprecated JSON format for prepareBed.py")
        logging.warning("This will result in an error in BPReveal 5.0")
        logging.warning("Instead of providing individual bigwig names with a \"bigwigs\"")
        logging.warning("section in your json, use the new \"heads\" section.")
        logging.warning("Example update: If you currently have")
        logging.warning('"bigwigs": [{"file-name": "protein1_pos.bw",')
        logging.warning('             "max-counts": 10000,')
        logging.warning('             "min-counts": 10},')
        logging.warning('            {"file-name": "protein1_neg.bw",')
        logging.warning('             "max-counts": 10000,')
        logging.warning('             "min-counts": 10},')
        logging.warning('            {"file-name": "protein2_pos.bw",')
        logging.warning('             "max-counts": 3000,')
        logging.warning('             "min-counts": 20},')
        logging.warning('            {"file-name": "protein2_neg.bw",')
        logging.warning('             "max-counts": 3000,')
        logging.warning('              "min-counts" : 20} ]')
        logging.warning("you should update this to reflect the head structure of your model:")
        logging.warning('"heads": [{"bigwig-names": ["protein1_pos.bw", "protein1_neg.bw"],')
        logging.warning('           "max-counts": 20000,')
        logging.warning('           "min-counts": 20},')
        logging.warning('          {"bigwig-names": ["protein2_pos.bw", "protein2_neg.bw"],')
        logging.warning('           "max-counts": 6000,')
        logging.warning('           "min-counts": 40}]')
        logging.warning("Note how the max-counts and min-counts values double, since the bigwigs")
        logging.warning("in each head will be added together to determine the total counts in")
        logging.warning("a region. (You don't need to change quantiles, though.)")

        headsConfig = []
        for bwConf in config["bigwigs"]:
            bwNames = [bwConf["file-name"]]
            bwConf["bigwig-names"] = bwNames
            del bwConf["file-name"]
            headsConfig.append(bwConf)
        logging.warning("Your heads config has been automatically converted to the new format,")
        logging.warning("with each bigwig being considered as its own head:")
        logging.warning(str(headsConfig))
        config["heads"] = headsConfig

    genome = pysam.FastaFile(config["genome"])
    (trainRegions, valRegions, testRegions, rejectRegions) = loadRegions(config)
    logging.debug("Regions loaded.")
    if ("output-prefix" in config):
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
        if ("output-reject" in config):
            outputRejectFname = config["output-reject"]
        else:
            outputRejectFname = False

    bigwigLists = []
    for head in config["heads"]:
        bigwigLists.append([pyBigWig.open(bwName) for bwName in head["bigwig-names"]])
    logging.info("Validating training regions.")
    validTrain, rejectTrain = validateRegions(config, trainRegions, genome, bigwigLists)
    logging.info("Validating validation regions.")
    validVal, rejectVal = validateRegions(config, valRegions, genome, bigwigLists)
    logging.info("Validating testing regions.")
    validTest, rejectTest = validateRegions(config, testRegions, genome, bigwigLists)

    for bwList in bigwigLists:
        for f in bwList:
            f.close()

    validTrain.saveas(outputTrainFname)
    validVal.saveas(outputValFname)
    validTest.saveas(outputTestFname)
    validAll = validTrain.cat(validVal, postmerge=False).cat(validTest, postmerge=False).sort()
    validAll.saveas(outputAllFname)
    if (outputRejectFname):
        allRejects = rejectRegions\
            .cat(rejectTrain, postmerge=False)\
            .cat(rejectVal, postmerge=False)\
            .cat(rejectTest, postmerge=False)\
            .sort()
        allRejects.saveas(outputRejectFname)
    logging.info("Regions saved.")


if (__name__ == "__main__"):
    import sys
    config = json.load(open(sys.argv[1]))
    utils.setVerbosity(config["verbosity"])
    prepareBeds(config)
