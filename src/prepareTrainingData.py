#!/usr/bin/env python3
"""Create the data files that will be used to train the model.

This program reads in a genome file, a list of regions in bed format, and a set
of bigwig files containing profiles that the model will use to train. It
generates an hdf5-format file that is used during training. If you want to
train on a custom genome, or you don't have a meaningful genome for your
experiment, you can still provide sequences and profiles by creating an hdf5
file in the same format as this tool generates.


BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/prepareTrainingData.bnf


Parameter Notes
---------------

genome
    The name of the fasta-format file for your organism.
regions
    is the name of the bed file of regions you will train on. These regions
    must be ``output-width`` in length.
reverse-complement
    A boolean that sets whether the data files will include reverse-complement
    augmentation. If this is set to ``true`` then you must include
    ``revcomp-task-order`` in every head section.
revcomp-task-order
    A list specifying which tasks in the forward sample should map to the tasks
    in the reverse sample.
    Alternatively, this may be the string ``"auto"``.
    If ``reverse-complement`` is false, it is an error to specify
    ``revcomp-task-order``.

Output specification
--------------------

It will generate a file that is organized as follows:

head_0, head_1, head_2, ...
    There will be a ``head_n`` entry for each head in your model. It will have
    shape ``(num-regions x (output-length + 2*jitter) x num-tasks)``.
sequence
    The one-hot encoded sequence for each corresponding region. It will have
    shape ``(num-regions x (input-width + 2*jitter) x 4)``.

Additional information
----------------------

Revcomp tasks
^^^^^^^^^^^^^

The ``revcomp-task-order`` parameter can be a bit tricky to understand.
Generally, ask yourself "If we had sequenced the other strand of this
chromosome, which profile would look like which?" If the data from one task,
say, the positive task, would appear on the other task in this hypothetical
universe, then you should flip the tasks.

For example, if the two tasks represent reads on the plus and minus
strand, then when you create a reverse-complemented training example,
the minus strand becomes the plus strand, and vice versa.
So you'd set this parameter to ``[1,0]`` to indicate that the data
for the two tasks should be swapped (in addition to reversed 5' to 3',
of course).

If you only have one task in a head, you should set this to
``[0]``, to indicate that there is no swapping.
If you have multiple tasks, say, a task for the left end of a read,
one for the right end, and one for the middle, then the left and right
should be swapped and the middle left alone.
In this case, you'd set ``revcomp-task-order`` to ``[1,0,2]``.
If this parameter is set to ``"auto"``, then it will choose
``[1,0]`` if there are two strands, ``[0]`` if there is only
one strand, and it will issue an error if there are more strands than
that.

``auto`` is appropriate for data like ChIP-nexus.

History
-------

``reverse-complement`` became mandatory in BPReveal 2.0.0

API
---
"""
import numpy as np
import h5py
import pyBigWig
import json
import pysam
import logging
import pybedtools
from bpreveal import utils
from typing import Literal
from bpreveal.utils import ONEHOT_T, ONEHOT_AR_T, PRED_T, PRED_AR_T, H5_CHUNK_SIZE


def revcompSeq(oneHotSeq: ONEHOT_AR_T) -> ONEHOT_AR_T:
    """Reverse-complement the given sequence."""
    # Since the order of the one-hot encoding is ACGT, if we flip the array
    # up-down, we complement the sequence, and if we flip it left-right, we
    # reverse it. So reverse complement of the one hot sequence is just:
    return np.flip(oneHotSeq)


def getSequences(bed, genome, outputLength, inputLength, jitter, revcomp):
    numSequences = bed.count()
    if not revcomp:
        seqs = np.zeros((numSequences, inputLength + 2 * jitter, 4), dtype=ONEHOT_T)
    else:
        seqs = np.zeros((numSequences * 2, inputLength + 2 * jitter, 4), dtype=ONEHOT_T)
    padding = ((inputLength + 2 * jitter) - outputLength) // 2
    for i, region in enumerate(bed):
        chrom = region.chrom
        start = region.start - padding
        stop = region.stop + padding
        seq = genome.fetch(chrom, start, stop)
        if not revcomp:
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
    if not revcomp:
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
                if not revcomp:
                    headVals[j        , :, i         ] = bwVals  # noqa
                else:
                    headVals[j * 2    , :, i         ] = bwVals  # noqa
                    headVals[j * 2 + 1, :, revcomp[i]] = np.flip(bwVals)
    return headVals


def writeH5(config):
    """Main method, load the config and then generate training data hdf5 files."""
    regions = pybedtools.BedTool(config["regions"])
    outputLength = config["output-length"]
    inputLength = config["input-length"]
    jitter = config["max-jitter"]
    genome = pysam.FastaFile(config["genome"])
    logging.debug("Opening output file.")
    outFile = h5py.File(config["output-h5"], "w")
    logging.debug("Loading sequence information.")
    seqs = getSequences(regions, genome, outputLength,
                        inputLength, jitter, config["reverse-complement"])

    outFile.create_dataset("sequence", data=seqs, dtype=ONEHOT_T,
                           chunks=(H5_CHUNK_SIZE, seqs.shape[1], 4), compression='gzip')
    logging.debug("Sequence dataset created.")
    for i, head in enumerate(config["heads"]):
        if config["reverse-complement"]:
            revcomp = head["revcomp-task-order"]
            if revcomp == "auto":
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
        configJson = json.load(configFp)

    import bpreveal.schema
    bpreveal.schema.prepareTrainingData.validate(configJson)
    utils.setVerbosity(configJson["verbosity"])
    writeH5(configJson)
