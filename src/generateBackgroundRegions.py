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
    print("Preparing progress bar")
    for chrom in genome.references:
        if(chrom not in allowChroms):
            continue
        
        padding = (inputWidth - outputWidth) //2
        startPos = edgeBoundary + jitter + padding
        chromSize = genome.get_reference_length(chrom)
        stopPos = chromSize - (edgeBoundary + minSpacing + jitter + padding + outputWidth)
        numSpaces += (stopPos - startPos) / (inputWidth + jitter + minSpacing)
    print("Generating candidate regions.")
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
    print("Candidate regions generated.")
    return regions

def generateCountsSums(regions, outputWidth, jitter, headBigwigs):
    numRegions = sum([len(x) for x in regions.values()])
    pbar = tqdm.tqdm(total=numRegions)
    countsSums = np.zeros((numRegions,len(headBigwigs)))
    curIdx = 0
    print("Generating counts sums")
    for chrom in sorted(regions.keys()):
        for pos in regions[chrom]:
            for i, head in enumerate(headBigwigs):
                for bwFile in head:
                    totalCounts = np.abs(np.sum(np.nan_to_num(bwFile.values(chrom, pos+jitter, pos + outputWidth - jitter))))
                    countsSums[curIdx, i] += totalCounts
            curIdx += 1
            pbar.update()
    return countsSums


def loadRegions(bedFnames):
    regions = dict()
    print("Loading peaks from bed files")
    for bedFname in bedFnames:
        with open(bedFname, "r") as bedFile:
            for line in bedFile:
                lsp = line.split()
                chrom = lsp[0]
                start = int(lsp[1])
                if(chrom not in regions):
                    regions[chrom] = []
                regions[chrom].append(start)
    print("Sorting previous regions")
    sortRegions = dict()
    for chrom in regions.keys():
        sortRegions[chrom] = sorted(list(set(regions[chrom])))
    return regions

def removeOverlaps(peaksRegions, backgroundRegions, inputWidth, outputWidth, maxJitter):
    leftPadding = (inputWidth - outputWidth) //2 + maxJitter
    rightPadding = leftPadding + outputWidth
    numRegions = sum([len(x) for x in backgroundRegions.values()])
    print("Removing overlaps.")
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

        
def printQuantiles(countsSums, bigwigNames):
    print(countsSums.shape)
    for i, bwName in enumerate(bigwigNames):
        print(bwName)
        counts = countsSums[:,i]
        quantileCutoffs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        header = "".join(["{0:10.2f} ".format(qc) for qc in quantileCutoffs])
        quantiles = np.quantile(counts, quantileCutoffs)
        dats = "".join(["{0:10.2f} ".format(q) for q in quantiles])
        print(header)
        print(dats)


def boundsValid(counts, bounds):
    for i in range(len(counts)):
        cv = counts[i]
        if(cv < bounds[i][0] or cv > bounds[i][1]):
            return False
    return True

def filterQuantiles(regions, countsSums, quantileBounds):
    countsBounds = []
    for i, quantileBound in enumerate(quantileBounds):
        quantiles = np.quantile(countsSums[:,i], quantileBound)
        countsBounds.append(quantiles)
    curIdx = 0
    ret = dict()
    for chrom in sorted(regions.keys()):
        ret[chrom] = []
        for pos in regions[chrom]:
            if(boundsValid(countsSums[curIdx], countsBounds)):
                ret[chrom].append(pos)
            curIdx += 1
    return ret

def writeBed(regions, outputWidth, bedFname):
    with open(bedFname, "w") as fp:
        for chrom in sorted(regions.keys()):
            for pos in regions[chrom]:
                fp.write("{0:s}\t{1:d}\t{2:d}\n".format(chrom, pos, pos+outputWidth))



def generateBackground(config, showQuantiles):
    #First, get a list of candidate regions.
    genome = pysam.FastaFile(config["genome"])

    tilingRegions = generateTilingRegions(genome, config["input-width"], config["output-width"], \
            config["max-jitter"], config["edge-boundary"], config["min-spacing"], \
            config["allow-chroms"]) 

    bigwigs = [[pyBigWig.open(x, "r") for x in y] for y in config["head-bigwigs"]]
    
    
    dataRegions = loadRegions(config["bed-files"])
    nonOverlapRegions = removeOverlaps(dataRegions, tilingRegions, config["input-width"], config["output-width"], config["max-jitter"] )

    countsSums = generateCountsSums(nonOverlapRegions, config["output-width"], config["max-jitter"], bigwigs)
    if(showQuantiles):
        printQuantiles(countsSums, config["head-bigwigs"])
    else:
        filteredRegions = filterQuantiles (nonOverlapRegions, countsSums, config["quantile-bounds"])
        writeBed(filteredRegions, config["output-width"], config["output-bed"])





def main():
    parser = argparse.ArgumentParser(description="Read in a json file and generate background regions for training the bias model.")
    parser.add_argument("json", help="The JSON-format config file.")
    parser.add_argument("--show-quantiles", action='store_true', help="Instead of writing the regions that fall in the specified quantiles," +\
            "show the counts quantiles in background regions.", dest='showQuantiles')
    args = parser.parse_args()
    config = json.load(open(args.json))
    utils.setVerbosity(config["verbosity"])
    generateBackground(config, args.showQuantiles)

if(__name__ == "__main__"):
    main()
