#This file contains several helper functions for dealing with bed files. 


import json
import pyBigWig
import pysam
import tqdm
import logging
import utils
import numpy as np

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
    counts = np.sum(vals)
    return counts

def sequenceChecker(interval, genome):
    seq = genome.fetch(interval.chrom, interval.start, interval.end)
    if(len(seq.upper().lstrip('ACGT')) != 0):
        #There were letters that aren't regular bases. (probably Ns)
        return False
    return True
        return ret



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

def getCountsQuantiles(bed, quantiles):
    counts = [getCounts(r, bigwig) for r in bed]
    return np.quantile(counts, quantiles)


def stripCountsBelow(bed, cutoff, bigwig):
    return bed.filter(lambda interval : getCounts(interval, bigwig) >= cutoff)
def stripCountsAbove(bed, cutoff, bigwig):
    return bed.filter(lambda interval : getCounts(interval, bigwig) <= cutoff)


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
    bigRegions = regions.each(resize, config["resize-mode"], config["input-length"] + config["max-jitter"]*2, genome)
    bigRegions = bigRegions.filter(sequenceChecker, genome)
    #Now, we have the possible regions. Get their counts values.
    validRegions = np.ones((len(bigRegions),))
    for i, bwSpec in enumerate(config["bigwigs"]):
        bigCounts = getCounts(bigRegions, bigwigs[i])
        if("max-counts" in bwSpec):
            maxCounts = bwSpec["max-counts"]
        else:
            maxCounts = np.quantile(bigCounts, bwSpec["max-quantile"]);
        for regionIdx in range(len(bigRegions)):
            if(bigCounts[i] > maxCounts):
                validRegions[i] = 0
    #We've now validated that the regions don't have too many counts when you inflate them. We also need to check that the regions won't 
    #have too few counts in the output. 
    smallRegions = bigRegions.each(resize, 'center', config["output-length"] - config["max-jitter"] * 2, genome)
    for i, bwSpec in enumerate(config["bigwigs"]):
        smallCounts = getCounts(smallRegions, bigwigs[i])
        if("min-counts" in bwSpec):
            minCounts = bwSpec["min-counts"]
        else:
            minCounts = np.quantile(bigCounts, bwSpec["min-quantile"]);
        for regionIdx in range(len(bigRegions)): #within len(bigRegions) in case a region was lost during the resize - we want that to crash
                                                 #because resizing down should never invalidate a region due to sequence problems. 
            if(smallCounts[i] < minCounts):
                validRegions[i] = 0
    #Now we resize to the final output size.
    outRegions = smallRegions.each(resize, 'center', config["output-length"], genome)
    #Since we kept the array of valid regions separately, we now have to create the result manually. 
    filteredRegions = []
    for i, r in enumerate(outRegions):
        if(validRegions[i] == 1):
            filteredRegions.append(r)
    return pybedtools.BedTool(filteredRegions)



def prepareBeds(config):
    logging.info("Starting bed file generation.")
    bigwigs = [pyBigWig.open(f) for f in config["bigwigs"]]
    genome = pysam.FastaFile(config["genome"])
    (trainRegions, valRegions,testRegions) = loadRegions(config)
    logging.info("Regions loaded.")
    logging.info("Validating regions.")
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

    validTrain = validateRegions(config, trainRegions, genome, bigwigs) 
    validVal = validateRegions(config, valRegions, genome, bigwigs) 
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
    config = json.load(open(sys.argv[1]))
    utils.setVerbosity(config["verbosity"])
    prepareBeds(config)


