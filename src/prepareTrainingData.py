#!/usr/bin/env python3

import numpy as np
import h5py
import pyBigWig
import json
import pysam
import logging
import pybedtools
import utils

def getSequences(bed, genome, outputLength, inputLength, jitter):
    seqs = np.zeros((bed.count(), inputLength + 2*jitter, 4), dtype=np.int8)
    padding = ((inputLength + 2 * jitter) - outputLength)//2
    for i, region in enumerate(bed):
        chrom = region.chrom
        start = region.start - padding
        stop = region.stop + padding
        seq = genome.fetch(chrom, start, stop)
        seqs[i] = utils.oneHotEncode(seq)
    return seqs

def getHead(bed, bigwigFnames, outputLength, jitter):
    headVals = np.zeros((bed.count(), outputLength + 2*jitter, len(bigwigFnames)))
    for i,bwFname in enumerate(bigwigFnames):
        with pyBigWig.open(bwFname, "r") as fp:
            for j,region in enumerate(bed):
                chrom = region.chrom
                start = region.start - jitter
                stop = region.stop + jitter
                bwVals = np.nan_to_num(fp.values(chrom, start, stop))
                headVals[j, :,i] = bwVals
    return headVals

def writeH5(config):
    regions = pybedtools.BedTool(config["regions"])
    outputLength = config["output-length"]
    inputLength = config["input-length"]
    jitter = config["max-jitter"]
    genome = pysam.FastaFile(config["genome"])
    outFile = h5py.File(config["output-h5"], "w")
    seqs = getSequences(regions, genome, outputLength, inputLength, jitter)
    outFile.create_dataset("sequence", data=seqs, dtype='i1')
    logging.debug("Sequence dataset created.")
    for i, head in enumerate(config["heads"]):
        headVals = getHead(regions, head["bigwig-files"], outputLength, jitter)
        outFile.create_dataset("head_{0:d}".format(i), data=headVals, dtype='f4')
        logging.debug("Added data for head {0:d}".format(i))
    outFile.close()


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    utils.setVerbosity(config["verbosity"])
    writeH5(config)

        
