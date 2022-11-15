#!/usr/bin/env python3
#This file contains several helper functions for dealing with bed files. 


import json
import pyBigWig
import pysam
import tqdm
import logging
import utils
import numpy as np
import pybedtools

def resize(interval, mode, width, genome):
    start = interval.start
    end = interval.end
    match mode:
        case "none":
            if(end - start != width):
                assert False, "An input region is not the expected width: {0:s}".format(str(self))
        case "center" : 
            center = (end + start) // 2
            start = center - width//2
            end = start + width
        case "start":
            start = start - width//2
            end = start + width
        case _ :
            assert False, "Unsupported resize mode: {0:s}".format(mode)
    if(start <=0 or end>=genome.get_reference_length(interval.chrom)):
        return False #We're off the edge of the chromosome.
    return pybedtools.Interval(interval.chrom, start, end)

def getCounts(interval, bigwig):
    vals = np.nan_to_num(bigwig.values(interval.chrom, interval.start, interval.end))
    return np.sum(vals)

def sequenceChecker(interval, genome):
    seq = genome.fetch(interval.chrom, interval.start, interval.end)
    if(len(seq.upper().lstrip('ACGT')) != 0):
        #There were letters that aren't regular bases. (probably Ns)
        return False
    return True



def generateTilingRegions(genome, width, chromEdgeBoundary, spaceBetween, allowChroms):
    chromRegions = []
    numRegions = 0
    #To use window_maker from pybedtools, I first need to create a bed containing the chromosomes where I want regions made. 
    for chrom in genome.references:
        if(chrom not in allowChroms):
            continue
        
        startPos = chromEdgeBoundary
        chromSize = genome.get_reference_length(chrom)
        stopPos = chromSize - chromEdgeBoundary
        chromRegions.append(pybedtools.Interval(chrom, startPos, stopPos))
    
    windows = pybedtools.window_maker(b=pybedtools.BedTool(chromRegions), w=width, s=spaceBetween + width)
    return windows


def stripCountsBelow(bed, cutoff, bigwig):
    return bed.filter(lambda interval : getCounts(interval, bigwig) >= cutoff).saveas()
def stripCountsAbove(bed, cutoff, bigwig):
    return bed.filter(lambda interval : getCounts(interval, bigwig) <= cutoff).saveas()


def lineToInterval(line):
    return pybedtools.cbedtools.create_interval_from_list(line.split())

def loadRegions(config):
    trainRegions = [] 
    testRegions = []
    valRegions = []
    numRejected = 0
    match config["splits"]:
        case {"train-chroms" : trainChroms, "val-chroms" : valChroms, "test-chroms" : testChroms, "regions" : regionFnames}:
            for bedFile in regionFnames:
                for line in open(bedFile):
                    r = lineToInterval(line)
                    if(r.chrom in trainChroms):
                        trainRegions.append(r)
                    elif(r.chrom in valChroms):
                        valRegions.append(r)
                    elif(r.chrom in testChroms):
                        testRegions.append(r)
                    else:
                        numRejected += 1
        case {"train-regions" : trainRegionFnames, "val-regions" : valRegionFnames, "test-regions" : testRegionFnames}:
            for trainBedFile in trainRegionFnames:
                for line in open(trainBedFile):
                    trainRegions.append(lineToInterval(line))
            for valBedFile in valRegionFnames:
                for line in open(valBedFile):
                    valRegions.append(lineToInterval(line))
            for testBedFile in testRegionFnames:
                for line in open(testBedFile):
                    testRegions.append(lineToInterval(line))
        case {"train-regex" : trainString, "val-regex" : valString, "test-regex" : testString, "regions" : regionFnames}:
            trainRegex = re.compile(trainString)
            valRegex = re.compile(valString)
            testRegex = re.compile(testString)
            for bedFile in regionFnames:
                for line in open(bedFile):
                    r = lineToInterval(line)
                    foundTrain = False
                    foundVal = False
                    foundTest = False
                    if(trainRegex.search(r.name) is not None):
                        foundTrain = True
                        trainRegions.append(r)
                    if(valRegex.search(r.name) is not None):
                        assert (not foundTrain), "Region {0:s} matches multiple regexes.".format(line)
                        foundVal = True
                        valRegions.append(r)
                    if(testRegex.search(r.name) is not None):
                        assert (not (foundTrain or foundVal)), "Region {0:s} matches multiple regexes.".format(line)
                        foundTest = True
                        testRegions.append(r)
                    if(not (foundTrain or foundVal or foundTest)):
                        numRejected += 1
        case _:
            assert False, "Config invalid: {0:s}".format(str(config["splits"]))
    logging.info("Rejected on loading: {0:d}".format(numRejected))
    return (pybedtools.BedTool(trainRegions), pybedtools.BedTool(valRegions), pybedtools.BedTool(testRegions))



def validateRegions(config, regions, genome, bigwigs):
    #First, resize the regions to their biggest size.
    bigRegions = regions.each(resize, config["resize-mode"], config["input-length"] + config["max-jitter"]*2, genome).saveas()
    logging.info("Resized sequences. {0:d} remain.".format(bigRegions.count()))
    bigRegions = bigRegions.filter(sequenceChecker, genome).saveas()
    logging.info("Filtered sequences. {0:d} remain.".format(bigRegions.count()))
    #Now, we have the possible regions. Get their counts values.
    validRegions = np.ones((len(bigRegions),))
    countsStats = dict()
    pbar = tqdm.tqdm(total=bigRegions.count() * len(bigwigs)) 
    for i, bwSpec in enumerate(config["bigwigs"]):
        if("max-quantile" in bwSpec):
            if(bwSpec["max-quantile"] == 1):
                countsStats[bwSpec["file-name"]] = [-1,-2]
                continue
        bigCounts = np.zeros((bigRegions.count(),))
        for j, r in enumerate(bigRegions):
            bigCounts[j] = getCounts(r, bigwigs[i])
            pbar.update()
        if("max-counts" in bwSpec):
            maxCounts = bwSpec["max-counts"]
        else:
            maxCounts = np.quantile(bigCounts, [bwSpec["max-quantile"]]);
        logging.debug("Max counts: {0:s}, file {1:s}".format(str(maxCounts), bwSpec["file-name"]))
        countsStats[bwSpec["file-name"]] = [maxCounts[0], -2]
        numReject = 0
        for regionIdx in range(len(bigRegions)):
            if(bigCounts[regionIdx] > maxCounts[0]):
                numReject += 1
                validRegions[regionIdx] = 0
        logging.debug("Rejected {0:f}% of big regions.".format(numReject*100./len(bigRegions)))
    pbar.close()
    #We've now validated that the regions don't have too many counts when you inflate them. We also need to check that the regions won't 
    #have too few counts in the output. 
    logging.info("Validated inflated regions. Surviving regions: {0:d}".format(int(np.sum(validRegions))))
    pbar = tqdm.tqdm(total=bigRegions.count() * len(bigwigs)) 
    smallRegions = bigRegions.each(resize, 'center', config["output-length"] - config["max-jitter"] * 2, genome).saveas()
    for i, bwSpec in enumerate(config["bigwigs"]):
        #Since this is a slow step, check to see if the min counts is zero. If so, no need to filter.
        if("min-quantile" in bwSpec):
            if(bwSpec["min-quantile"] == 0):
                countsStats[bwSpec["file-name"]][1] = -1
                continue
        smallCounts = np.zeros((bigRegions.count(),))
        for j, r in enumerate(smallRegions):
            smallCounts[j] = getCounts(r, bigwigs[i])
            pbar.update()
        if("min-counts" in bwSpec):
            minCounts = bwSpec["min-counts"]
        else:
            minCounts = np.quantile(smallCounts, [bwSpec["min-quantile"]]);
        logging.debug("Min counts: {0:s}, file {1:s}".format(str(minCounts), bwSpec["file-name"]))
        countsStats[bwSpec["file-name"]][1] = minCounts[0]
        numReject = 0
        for regionIdx in range(len(bigRegions)): #within len(bigRegions) in case a region was lost during the resize - we want that to crash
                                                 #because resizing down should never invalidate a region due to sequence problems. 
            if(smallCounts[regionIdx] < minCounts[0]):
                numReject += 1
                validRegions[regionIdx] = 0
        logging.debug("Rejected {0:f}% of small regions.".format(numReject*100./len(bigRegions)))
    pbar.close()
    logging.info("Validated small regions. Surviving regions: {0:d}".format(int(np.sum(validRegions))))
    #Now we resize to the final output size.
    outRegions = smallRegions.each(resize, 'center', config["output-length"], genome).saveas()
    #Since we kept the array of valid regions separately, we now have to create the result manually. 
    logging.info("Counts statistics. Name, max-counts, min-counts.")
    for k in countsStats.keys():
        logging.info("{0:50s}\t{1:f}\t{2:f}".format(k, float(countsStats[k][0]), float(countsStats[k][1])))
        
    filteredRegions = []
    for i, r in enumerate(outRegions):
        if(validRegions[i] == 1):
            filteredRegions.append(r)
    return pybedtools.BedTool(filteredRegions)



def prepareBeds(config):
    logging.info("Starting bed file generation.")
    bigwigs = [pyBigWig.open(f["file-name"]) for f in config["bigwigs"]]
    genome = pysam.FastaFile(config["genome"])
    (trainRegions, valRegions,testRegions) = loadRegions(config)
    logging.info("Regions loaded.")
    if("output-prefix" in config):
        outputTrainFname = config["output-prefix"] + "_train.bed"
        outputValFname = config["output-prefix"] + "_val.bed"
        outputTestFname = config["output-prefix"] + "_test.bed"
        outputAllFname = config["output-prefix"] + "_all.bed"
    else:
        outputTrainFname = config["output-train"]
        outputValFname = config["output-val"]
        outputTestFname = config["output-test"]
        outputAllFname = config["output-all"]
    logging.info("Validating training regions.")
    validTrain = validateRegions(config, trainRegions, genome, bigwigs) 
    logging.info("Validating validation regions.")
    validVal = validateRegions(config, valRegions, genome, bigwigs) 
    logging.info("Validating testing regions.")
    validTest = validateRegions(config, testRegions, genome, bigwigs) 
    validTrain.saveas( outputTrainFname)
    validVal.saveas( outputValFname)
    validTest.saveas( outputTestFname)
    validAll = validTrain.cat(validVal, postmerge=False).cat(validTest, postmerge=False)
    validAll.saveas(outputAllFname)
    logging.info("Regions saved.")
    for f in bigwigs:
        f.close()


if(__name__ == "__main__"):
    import sys
    config = json.load(open(sys.argv[1]))
    utils.setVerbosity(config["verbosity"])
    prepareBeds(config)

