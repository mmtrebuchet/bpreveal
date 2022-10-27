#!/usr/bin/env python3
import json
import pyBigWig
import pysam
import sys
import tqdm
import logging
import argparse
import utils
import numpy as np

def generateTilingRegions(genome, inputWidth, outputWidth, jitter, edgeBoundary, minSpacing, allowChroms):
    regions = dict()
    numSpaces = 0
    logging.debug("Preparing progress bar")
    for chrom in genome.references:
        if(chrom not in allowChroms):
            continue
        
        padding = (inputWidth - outputWidth) //2
        startPos = edgeBoundary + jitter + padding
        chromSize = genome.get_reference_length(chrom)
        stopPos = chromSize - (edgeBoundary + minSpacing + jitter + padding + outputWidth)
        numSpaces += (stopPos - startPos) / (inputWidth + jitter + minSpacing)
    logging.info("Generating candidate regions.")
    pbar = tqdm.tqdm(total=numSpaces)
    for chrom in genome.references:
        if(chrom not in allowChroms):
            continue
        
        padding = (inputWidth - outputWidth) //2
        startPos = edgeBoundary + jitter + padding
        chromSize = genome.get_reference_length(chrom)
        stopPos = chromSize - (edgeBoundary + minSpacing + jitter + padding + outputWidth)
        while(startPos < stopPos):
            pbar.update()
            seqStart = startPos - padding - jitter
            seqStop = seqStart + inputWidth + 2*jitter
            seq = genome.fetch(chrom, seqStart, seqStop)
            if(len(seq.upper().lstrip("ACGT")) == 0):
                if(chrom not in regions):
                    regions[chrom] = []
                regions[chrom].append(seqStart)
            startPos += inputWidth + jitter + minSpacing
    pbar.close()
    logging.info("Candidate regions generated.")
    return regions

def generateCountsSums(regions, outputWidth, jitter, headBigwigs):
    numRegions = sum([len(x) for x in regions.values()])
    countsSumsHigh = np.zeros((numRegions,len(headBigwigs)))
    countsSumsLow = np.zeros((numRegions,len(headBigwigs)))
    curIdx = 0
    logging.info("Generating counts sums")
    pbar = tqdm.tqdm(total=numRegions)
    for chrom in sorted(regions.keys()):
        for pos in regions[chrom]:
            for i, head in enumerate(headBigwigs):
                maxCount = None
                minCount = None
                for bwFile in head:
                    region = np.nan_to_num(bwFile.values(chrom, pos-jitter, pos+outputWidth+jitter))
                    totalCountsLow = np.abs(np.sum(region[jitter*2:-jitter*2]))
                    totalCountsHigh = np.abs(np.sum(region))
                    if(maxCount is None):
                        maxCount = totalCountsHigh
                    else:
                        maxCount = max(totalCountsHigh, maxCount)
                    if(minCount is None):
                        minCount = totalCountsLow
                    else:
                        minCount = min(totalCountsLow, minCount)
                countsSumsHigh[curIdx, i] = maxCount
                countsSumsLow[curIdx, i] = minCount
            curIdx += 1
            pbar.update()
    return countsSumsHigh, countsSumsLow


def loadRegions(bedFnames):
    regions = dict()
    logging.info("Loading peaks from bed files")
    for bedFname in bedFnames:
        with open(bedFname, "r") as bedFile:
            for line in bedFile:
                lsp = line.split()
                chrom = lsp[0]
                start = int(lsp[1])
                if(chrom not in regions):
                    regions[chrom] = []
                regions[chrom].append(start)
    logging.info("Sorting previous regions")
    sortRegions = dict()
    for chrom in regions.keys():
        sortRegions[chrom] = sorted(list(set(regions[chrom])))
    return regions

def removeOverlaps(peaksRegions, backgroundRegions, inputWidth, outputWidth, maxJitter):
    leftPadding = (inputWidth - outputWidth) //2 + maxJitter
    rightPadding = leftPadding + outputWidth
    numRegions = sum([len(x) for x in backgroundRegions.values()])
    logging.info("Removing overlaps.")
    pbar = tqdm.tqdm(total=numRegions)
    ret = dict()
    for chrom in backgroundRegions.keys():
        if(chrom not in peaksRegions):
            ret[chrom] = backgroundRegions[chrom]
            continue
        ret[chrom] = []
        chromPeaks = peaksRegions[chrom]
        peaksIdx = 0
        for regionStart in backgroundRegions[chrom]:
            pbar.update()
            seqStart = regionStart - leftPadding
            seqStop = regionStart + rightPadding
            while(peaksIdx < len(chromPeaks) and \
                    chromPeaks[peaksIdx] + rightPadding < seqStart):
                peaksIdx += 1
            #We're up to a peak that might overlap. 
            if(peaksIdx < len(chromPeaks)):
                #There is a remaining peak.
                peakStart = chromPeaks[peaksIdx] - leftPadding
                peakStop = chromPeaks[peaksIdx] + rightPadding
                #We know that this peak does not stop before the background region starts, but does it start before the background stops?
                if(peakStart <= seqStop):
                    continue
            #We passed both tests, this is a valid region. 
            ret[chrom].append(regionStart)
    return ret

        
def printQuantiles(countsSumsHigh, countsSumsLow, bigwigNames):
    print(countsSumsHigh.shape)
    for i, bwName in enumerate(bigwigNames):
        print(bwName)
        countsHigh = countsSumsHigh[:,i]
        countsLow = countsSumsLow[:,i]
        quantileCutoffs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        header = "".join(["{0:10.2f} ".format(qc) for qc in quantileCutoffs])
        quantilesHigh = np.quantile(countsHigh, quantileCutoffs)
        quantilesLow = np.quantile(countsLow, quantileCutoffs)
        datsHigh = "".join(["{0:10.2f} ".format(q) for q in quantilesHigh])
        datsLow = "".join(["{0:10.2f} ".format(q) for q in quantilesLow])
        print("{0:6s}".format("Pad")  + header)
        print("{0:6s}".format("High") + datsHigh)
        print("{0:6s}".format("Low")  + datsLow)


def boundsValid(countsHigh, countsLow, bounds):
    for i in range(len(countsHigh)):
        cvh = countsHigh[i]
        cvl = countsLow[i]
        if(cvl < bounds[i][0] or cvh > bounds[i][1]):
            return False
    return True

def filterQuantiles(regions, countsSumsHigh, countsSumsLow, quantileBounds):
    countsBounds = []
    for i, quantileBound in enumerate(quantileBounds):
        quantileHigh = np.quantile(countsSumsHigh[:,i], quantileBound[1])
        quantileLow = np.quantile(countsSumsLow[:,i], quantileBound[0])
        countsBounds.append((quantileLow, quantileHigh))
    logging.info("Quantiles are {0:s}".format(str(countsBounds)))
    curIdx = 0
    ret = dict()
    for chrom in sorted(regions.keys()):
        ret[chrom] = []
        for pos in regions[chrom]:
            if(boundsValid(countsSumsHigh[curIdx], countsSumsLow[curIdx], countsBounds)):
                ret[chrom].append(pos)
            curIdx += 1
    return ret

def writeBed(regions, outputWidth, bedFname):
    logging.info("Writing {0:d} regions.".format(sum([len(regions[x]) for x in regions.keys()])))
    with open(bedFname, "w") as fp:
        for chrom in sorted(regions.keys()):
            for pos in regions[chrom]:
                fp.write("{0:s}\t{1:d}\t{2:d}\n".format(chrom, pos, pos+outputWidth))
    logging.info("Output saved.")



def generateBackground(config):
    #First, get a list of candidate regions.
    genome = pysam.FastaFile(config["genome"])

    tilingRegions = generateTilingRegions(genome, config["input-width"], config["output-width"], \
            config["max-jitter"], config["edge-boundary"], config["min-spacing"], \
            config["allow-chroms"]) 

    bigwigs = [[pyBigWig.open(x, "r") for x in y] for y in config["head-bigwigs"]]
    
    
    dataRegions = loadRegions(config["bed-files"])
    nonOverlapRegions = removeOverlaps(dataRegions, tilingRegions, config["input-width"], config["output-width"], config["max-jitter"] )

    countsSumsHigh, countsSumsLow = generateCountsSums(nonOverlapRegions, config["output-width"], config["max-jitter"], bigwigs)
    printQuantiles(countsSumsHigh, countsSumsLow, config["head-bigwigs"])
    if('quantile-bounds' in config):
        filteredRegions = filterQuantiles (nonOverlapRegions, countsSumsHigh, countsSumsLow, config["quantile-bounds"])
        writeBed(filteredRegions, config["output-width"], config["output-bed"])
    else:
        logging.warn("No quantile bounds was specified in your input json. No files have been written.")





def main():
    parser = argparse.ArgumentParser(description="Read in a json file and generate background regions for training the bias model.")
    parser.add_argument("json", help="The JSON-format config file.")
    args = parser.parse_args()
    config = json.load(open(args.json))
    utils.setVerbosity(config["verbosity"])
    generateBackground(config)

if(__name__ == "__main__"):
    main()
