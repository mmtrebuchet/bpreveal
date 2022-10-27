#!/usr/bin/env python3
import pysam
import tqdm
import pyBigWig
import numpy as np
import json
import logging
import utils
import sys

class Region:
    def __init__(self, line):
        lsp = line.split()
        self.chrom = lsp[0]
        self.start = int(lsp[1])
        self.stop = int(lsp[2])
        if(len(lsp) >= 6):
            self.name = lsp[3]
            self.score = lsp[4]
            self.strand = lsp[5]
        if(len(lsp) >= 10):
            self.signal = lsp[6]
            self.pValue = lsp[7]
            self.qValue = lsp[8]
            self.peak = int(lsp[9])
        self.valid = True
    def resize(self, mode, width, jitter, genome, padding):
        match mode:
            case "none":
                if(self.stop - self.start != width):
                    logging.warning("An input region is not the expected width: {0:s}".format(str(self)))
                    self.valid = False
            case "center" : 
                center = (self.stop + self.start) // 2
                self.start = center - width//2
                self.stop = self.start + width
            case "start":
                self.start = self.start - width//2
                self.stop = self.start + width
            case "peak":
                self.start = self.start + self.peak - width//2
                self.stop = self.start + width
            case _ :
                assert False, "Unsupported resize mode: {0:s}".format(mode)
        minPosition = self.start - jitter - padding
        maxPosition = self.stop + jitter + padding
        if(minPosition <= 0):
            self.valid = False
        if(maxPosition >= genome.get_reference_length(self.chrom)):
            self.valid = False

    def checkCounts(self, countsWindowType, maximumCounts, maxJitter, padding, bigwigs):
        match countsWindowType:
            case "output":
                offset = 0
            case "output+jitter":
                offset = maxJitter
            case "input":
                offset = padding
            case "input+jitter":
                offset = padding + maxJitter
        fetchStart = self.start - offset
        fetchStop = self.stop + offset
        slicePosition = offset + maxJitter
        countsByBigwig = []
        for i, bw in enumerate(bigwigs):
            vals = np.nan_to_num(bw.values(self.chrom, fetchStart, fetchStop))
            windowCounts = np.sum(vals)
            sliceCounts = np.sum(vals[slicePosition:-slicePosition])
            countsByBigwig.append(windowCounts)
            if(maximumCounts is not None):
                #Don't return regions that overflow your maximum counts parameter. 
                if(windowCounts > maximumCounts[i]):
                    self.valid = False
                    return None
            if(sliceCounts == 0):
                #One of the bigwigs has zero reads in this region. That's bad for the multinomial, so skip it.
                self.valid = False
                return None
        return countsByBigwig

    def checkSequence(self, genome, maxJitter, padding):
        seq = genome.fetch(self.chrom, self.start - padding - maxJitter, self.stop + padding + maxJitter)
        if(len(seq.upper().lstrip('ACGT')) != 0):
            #There were letters that aren't regular bases. (probably Ns)
            self.valid = False
    def __str__(self):
        ret = "{0:s}\t{1:d}\t{2:d}".format(self.chrom, self.start, self.stop)
        if(hasattr(self, 'name')):
            ret += '\t{0:s}\t{1:s}\t{2:s}'.format(self.name, self.score, self.strand)
        if(hasattr(self, 'signal')):
            ret += '\t{0:s}\t{1:s}\t{2:s}\t{3:d}'.format(self.signal, self.pValue, self.qValue, self.peak)
        return ret
            
        



def loadRegions(config):
    trainRegions = [] 
    testRegions = []
    valRegions = []
    numRejected = 0
    match config["splits"]:
        case {"train-chroms" : trainChroms, "val-chroms" : valChroms, "test-chroms" : testChroms, "regions" : regionFnames}:
            for bedFile in regionFnames:
                for line in open(bedFile):
                    r = Region(line)
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
                    trainRegions.append(Region(line))
            for valBedFile in valRegionFnames:
                for line in open(valBedFile):
                    valRegions.append(Region(line))
            for testBedFile in testRegionFnames:
                for line in open(testBedFile):
                    testRegions.append(Region(line))
        case {"train-regex" : trainString, "val-regex" : valString, "test-regex" : testString, "regions" : regionFnames}:
            trainRegex = re.compile(trainString)
            valRegex = re.compile(valString)
            testRegex = re.compile(testString)
            for bedFile in regionFnames:
                for line in open(bedFile):
                    r = Region(line)
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
    return (trainRegions, valRegions, testRegions)


def prepareBeds(config):
    logging.info("Starting bed file generation.")
    bigwigs = [pyBigWig.open(f) for f in config["bigwigs"]]
    genome = pysam.FastaFile(config["genome"])
    (trainRegions, valRegions,testRegions) = loadRegions(config)
    logging.info("Regions loaded.")
    totalLength = len(trainRegions) + len(valRegions) + len(testRegions)
    logging.info("Validating regions.")
    if(config["verbosity"] in ["INFO", "DEBUG"]):
        pbar = tqdm.tqdm(total=totalLength)
    else:
        pbar = None
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
    countsDict = [dict() for x in bigwigs] #For each bigwig, generate a dictionary of counts. 
    validTrain = validateRegions(config, trainRegions, genome, bigwigs, pbar, countsDict)
    validVal = validateRegions(config, valRegions, genome, bigwigs, pbar, countsDict)
    validTest = validateRegions(config, testRegions, genome, bigwigs, pbar, countsDict)
    logging.info("Regions validated. Writing to disk.")
    writeRegions(validTrain, outputTrainFname)
    writeRegions(validVal, outputValFname)
    writeRegions(validTest, outputTestFname)
    writeRegions(validTrain + validVal + validTest, outputAllFname)
    logging.info("Regions saved.")
    for f in bigwigs:
        f.close()
    if("write-counts-to" in config):
        logging.info("Writing statistics.")
        with open(config["write-counts-to"], "w") as fp:
            for i, bwName in enumerate(config["bigwigs"]):
                #Get the mean counts, while we're at it.
                totalCounts = 0
                totalRegions = 0
                for countsValue in countsDict[i].keys():
                    totalCounts += countsValue * countsDict[i][countsValue]
                    totalRegions += countsDict[i][countsValue]

                fp.write("{0:s}\t{1:f}\n".format(bwName, totalCounts / totalRegions))
                fp.write("counts\ttimes\n")
                countsVals = sorted(countsDict[i].keys())
                for countsKey in countsVals:
                    fp.write("{0:d}\t{1:d}\n".format(int(countsKey), countsDict[i][countsKey]))



def validateRegions(config, regions, genome, bigwigs, pbar, countsDict):
    #First, resize the regions. 
    padding = (config["input-width"] - config["output-width"])//2
    for r in regions:
        if(pbar is not None):
            pbar.update()
        r.resize(config["resize-mode"], config["output-width"], config["max-jitter"], genome, padding)
        if(not r.valid):
            continue
        r.checkSequence(genome, config["max-jitter"], padding)
        if(not r.valid):
            continue
        if("maximum-counts" in config):
            maxCounts = config['maximum-counts']
        else:
            maxCounts = None
        if("counts-window-type" in config):
            windowType = config["counts-window-type"]
        else:
            windowType = 'output'
        c = r.checkCounts(windowType, maxCounts, config["max-jitter"], padding, bigwigs)
        if(c is not None):
            for i, cv in enumerate(c):
                if(not (cv in countsDict[i])):
                    countsDict[i][cv] = 0
                countsDict[i][cv] += 1
    #Okay, so they're all now the right size. 
    #Okay, all the checks are done. 
    validRegions = [r for r in regions if r.valid]
    return validRegions

def writeRegions(validRegions, outFname):
    validRegions.sort(key=lambda r: (r.chrom, r.start))
    prevChrom = ""
    prevStop = 0
    with open(outFname, "w") as fp:
        for r in validRegions:
            if(r.chrom == prevChrom):
                if(r.start < prevStop):
                    continue
                prevStop = r.stop
            else:
                prevChrom = r.chrom
                prevStop = 0
            fp.write(str(r) + '\n')
    

if(__name__ == "__main__"):
    config = json.load(open(sys.argv[1]))
    utils.setVerbosity(config["verbosity"])
    prepareBeds(config)


