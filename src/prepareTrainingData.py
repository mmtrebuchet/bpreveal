#!/usr/bin/env python3

import numpy as np
import h5py
import pyBigWig
import json
import pysam
import logging
import pybedtools
import bpreveal.utils as utils
from typing import Literal
from bpreveal.utils import ONEHOT_T, ONEHOT_AR_T, PRED_T, PRED_AR_T, H5_CHUNK_SIZE


def revcompSeq(oneHotSeq: ONEHOT_AR_T) -> ONEHOT_AR_T:
    # Since the order of the one-hot encoding is ACGT, if we flip the array
    # up-down, we complement the sequence, and if we flip it left-right, we
    # reverse it. So reverse complement of the one hot sequence is just:
    return np.flip(oneHotSeq)


def getSequences(bed, genome, outputLength, inputLength, jitter, revcomp):
    numSequences = bed.count()
    if (not revcomp):
        seqs = np.zeros((numSequences, inputLength + 2 * jitter, 4), dtype=ONEHOT_T)
    else:
        seqs = np.zeros((numSequences * 2, inputLength + 2 * jitter, 4), dtype=ONEHOT_T)
    padding = ((inputLength + 2 * jitter) - outputLength) // 2
    for i, region in enumerate(bed):
        chrom = region.chrom
        start = region.start - padding
        stop = region.stop + padding
        seq = genome.fetch(chrom, start, stop)
        if (not revcomp):
            seqs[i] = utils.oneHotEncode(seq)
        else:
            sret = utils.oneHotEncode(seq)
            seqs[i * 2] = sret
            seqs[i * 2 + 1] = revcompSeq(sret)
    return seqs


def getHead(bed, bigwigFnames: list[str], outputLength: int, jitter: int,
            revcomp: Literal[False] | list[int]) -> PRED_AR_T:
    # Note that revcomp should be either False or the task-order array (which is truthy).
    numSequences = bed.count()
    if (not revcomp):
        headVals = np.zeros((numSequences, outputLength + 2 * jitter, len(bigwigFnames)),
                            dtype=PRED_T)
    else:
        headVals = np.zeros((numSequences * 2, outputLength + 2 * jitter, len(bigwigFnames)),
                            dtype=PRED_T)

    for i, bwFname in enumerate(bigwigFnames):
        with pyBigWig.open(bwFname, "r") as fp:
            for j, region in enumerate(bed):
                chrom = region.chrom
                start = region.start - jitter
                stop = region.stop + jitter
                bwVals = np.nan_to_num(fp.values(chrom, start, stop))
                if (not revcomp):
                    headVals[j        , :, i         ] = bwVals
                else:
                    headVals[j * 2    , :, i         ] = bwVals
                    headVals[j * 2 + 1, :, revcomp[i]] = np.flip(bwVals)
    return headVals


def writeH5(config):
    regions = pybedtools.BedTool(config["regions"])
    outputLength = config["output-length"]
    inputLength = config["input-length"]
    jitter = config["max-jitter"]
    genome = pysam.FastaFile(config["genome"])
    logging.debug("Opening ouptut file.")
    outFile = h5py.File(config["output-h5"], "w")
    logging.debug("Loading sequence information.")
    seqs = getSequences(regions, genome, outputLength,
                        inputLength, jitter, config["reverse-complement"])

    outFile.create_dataset("sequence", data=seqs, dtype=ONEHOT_T,
                           chunks=(H5_CHUNK_SIZE, seqs.shape[1], 4), compression='gzip')
    logging.debug("Sequence dataset created.")
    for i, head in enumerate(config["heads"]):
        if (config["reverse-complement"]):
            revcomp = head["revcomp-task-order"]
            if (revcomp == "auto"):
                # The user has left reverse-complementing up to us.
                match len(head["bigwig-files"]):
                    case 1:
                        revcomp = [0]
                    case 2:
                        revcomp = [1, 0]
                    case _:
                        assert False, "Cannot automatically determine revcomp "\
                                      "order with more than two tasks."
        else:
            revcomp = False
        headVals = getHead(regions, head["bigwig-files"], outputLength, jitter, revcomp)
        outFile.create_dataset("head_{0:d}".format(i), data=headVals, dtype=PRED_T,
                               chunks=(H5_CHUNK_SIZE, headVals.shape[1], headVals.shape[2]),
                               compression='gzip')
        logging.debug("Added data for head {0:d}".format(i))
    outFile.close()
    logging.info("File created; closing.")


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    utils.setVerbosity(config["verbosity"])
    writeH5(config)
