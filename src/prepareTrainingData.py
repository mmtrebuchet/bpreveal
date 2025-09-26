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
    must be ``output-length`` in length.
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
    shape ``(num-regions x (input-length + 2*jitter) x NUM_BASES)``.
metadata
    A group containing the configuration used when the program was run.

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
import sys
from typing import Literal
import numpy as np
import h5py
import pyBigWig
import pysam
import pybedtools
import bpreveal.schema
from bpreveal import logUtils
from bpreveal import utils
from bpreveal.internal.constants import ONEHOT_T, ONEHOT_AR_T, PRED_AR_T, \
    H5_CHUNK_SIZE, PRED_T, NUM_BASES
import bpreveal.internal.files
from bpreveal.internal import interpreter


def revcompSeq(oneHotSeq: ONEHOT_AR_T) -> ONEHOT_AR_T:
    """Reverse-complement the given sequence.

    :param oneHotSeq: The input sequence.
    :return: The sequence, but reverse-complemented.
    """
    # Since the order of the one-hot encoding is ACGT, if we flip the array
    # up-down, we complement the sequence, and if we flip it left-right, we
    # reverse it. So reverse complement of the one hot sequence is just:
    return np.flip(oneHotSeq)


def getSequences(bed: pybedtools.BedTool, genome: pysam.FastaFile, outputLength: int,
                 inputLength: int, jitter: int, revcomp: bool) -> ONEHOT_AR_T:
    """Extract sequences from the fasta.

    :param bed: A BedTool containing the regions to get sequence data for.
        These regions should be outputLength wide.
    :param genome: The (open) FastaFile containing your genome sequence
    :param outputLength: The output length of your model
    :param inputLength: The input length of your model
    :param jitter: The maximum jitter that will be applied during training
    :param revcomp: Should the returned sequence include reverse-complement data?
        If so, then each region will produce two sequences: one forward and
        one reverse-complemented.
    :return: An array of one-hot-encoded sequences of shape
        ```(numSequences x inputLength + 2*jitter, NUM_BASES)```.
    """
    numSequences = bed.count()
    if not revcomp:
        seqs = np.zeros((numSequences, inputLength + 2 * jitter, NUM_BASES), dtype=ONEHOT_T)
    else:
        seqs = np.zeros((numSequences * 2, inputLength + 2 * jitter, NUM_BASES), dtype=ONEHOT_T)
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


def getHead(bed: pybedtools.BedTool, bigwigFnames: list[str], outputLength: int,
            jitter: int, revcomp: Literal[False] | list[int]) -> PRED_AR_T:
    """Get all the data for a particular head.

    :param bed: A BedTool containing the regions that will be used to train.
        Each interval in this BedTool should have length outputLength.
    :param bigwigFnames: The names of the bigwig files that will be loaded.
    :param outputLength: The output length of your model
    :param jitter: The jitter that will be applied during training.
    :param revcomp: If no reverse-complement is desired, this is just ``False``.
        Otherwise, see the section on ``revcomp-task-order`` for what this list means.
    :return: An array of data that can be put in the training hdf5. It has shape
        (numSequences * outputLength + 2 * jitter, numTasks). numSequences will be
        the length of your bed file if ``revcomp == False``, or twice the length of your
        bed file if you include revcomp augmentation.
    :rtype: PRED_AR_T
    """
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


def prepareTrainingData(config: dict) -> None:
    """Main method, load the config and then generate training data hdf5 files.

    :param config: The configuration json.
    :raise ValueError: If you try to use auto revcomp with more than two tasks.
    """
    regions = pybedtools.BedTool(config["regions"])
    outputLength = config["output-length"]
    inputLength = config["input-length"]
    jitter = config["max-jitter"]
    genome = pysam.FastaFile(config["genome"])
    logUtils.debug("Opening output file.")
    outFile = h5py.File(config["output-h5"], "w")
    bpreveal.internal.files.addH5Metadata(outFile, config=str(config))
    logUtils.debug("Loading sequence information.")
    seqs = getSequences(regions, genome, outputLength,
                        inputLength, jitter, config["reverse-complement"])

    outFile.create_dataset("sequence", data=seqs, dtype=ONEHOT_T,
                           chunks=(H5_CHUNK_SIZE, seqs.shape[1], NUM_BASES), compression="gzip")
    logUtils.debug("Sequence dataset created.")
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
                        raise ValueError("Cannot automatically determine revcomp "
                                         "order with more than two tasks.")
        else:
            revcomp = False  # pylint: disable=redefined-variable-type
        headVals = getHead(regions, head["bigwig-files"], outputLength, jitter, revcomp)
        outFile.create_dataset(f"head_{i}", data=headVals, dtype=PRED_T,
                               chunks=(H5_CHUNK_SIZE, headVals.shape[1], headVals.shape[2]),
                               compression="gzip")
        logUtils.debug(f"Added data for head {i}")
    outFile.close()
    logUtils.info("File created; closing.")


def main() -> None:
    """A zero-argument wrapper around the main function."""
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    bpreveal.schema.prepareTrainingData.validate(configJson)
    logUtils.setVerbosity(configJson["verbosity"])
    prepareTrainingData(configJson)


if __name__ == "__main__":
    main()
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
