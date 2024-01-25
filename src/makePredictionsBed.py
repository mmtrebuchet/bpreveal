#!/usr/bin/env python3
"""A script to make predictions using a bed file and a genome.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/makePredictionsBed.bnf


Parameter Notes
---------------

heads
    Gives the number of output heads for your model.
    You don't need to tell this program how many tasks there are for each
    head, since it just blindly sticks whatever the model outputs into the
    hdf5 file.

output-h5
    The name of the output file that will contain the predictions.

genome
    The name of the fasta-format file containing the appropriate genome.

batch-size
    How many samples should be run simultaneously? I recommend 64 or so.

model-file
    The name of the Keras model file on disk.

input-length, output-length
    The input and output lengths of your model.

bed-file
    A file containing the regions for which you'd like predictions. Each region in
    this bed file must be ``output-length`` long.

Output Specification
--------------------
This program will produce an hdf5-format file containing the predicted values.
It is organized as follows:

chrom_names
    A list of strings that give you the meaning
    of each index in the ``coords_chrom`` dataset.
    This is particularly handy when you want to make a bigwig file, since
    you can extract a header from this data.

chrom_sizes
    The size of each chromosome in the same order
    as ``chrom_names``.
    Mostly used to create bigwig headers.

coords_chrom
    A list of integers, one for each region
    predicted, that gives the chromosome index (see ``chrom_names``)
    for that region.

coords_start
    The start base of each predicted region.

coords_stop
    The end point of each predicted region.

head_0, head_1, head_2, ...
    You get a subgroup for each output head of the model. The subgroups are named
    ``head_N``, where N is 0, 1, 2, etc.
    Each head contains:

    logcounts
        A vector of shape (numRegions,) that gives
        the logcounts value for each region.

    logits
        The array of logit values for each track for
        each region.
        The shape is (numRegions x outputWidth x numTasks).
        Don't forget that you must calculate the softmax on the whole
        set of logits, not on each task's logits independently.
        (Use :py:func:`bpreveal.utils.logitsToProfile` to do this.)

API
---

"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
import pybedtools
from bpreveal import utils
from bpreveal.utils import ONEHOT_T, PRED_T, wrapTqdm
if __name__ == "__main__":
    utils.setMemoryGrowth()
import numpy as np
import pysam
import h5py
import logging

# Generate a simple sequence model taking one-hot encoded input and
# producing a logits profile and a log(counts) scalar.


def main(config):
    """Run the predictions.

    :param config: A JSON object satisfying the makePredictionsBed specification.
    """
    utils.setVerbosity(config["verbosity"])
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    batchSize = config["settings"]["batch-size"]
    genomeFname = config["settings"]["genome"]
    numHeads = config["settings"]["heads"]
    logging.debug("Opening output hdf5 file.")
    outFile = h5py.File(config["settings"]["output-h5"], "w")
    regions = pybedtools.BedTool(config["bed-file"])
    seqs = np.zeros((len(regions), inputLength, 4), dtype=ONEHOT_T)
    genome = pysam.FastaFile(genomeFname)
    padding = (inputLength - outputLength) // 2

    logging.info("Loading regions")
    for i, region in wrapTqdm(list(enumerate(regions))):  # type: ignore
        curSeq = genome.fetch(region.chrom, region.start - padding, region.stop + padding)
        seqs[i] = utils.oneHotEncode(curSeq)
    logging.info("Input prepared. Loading model.")
    model = utils.loadModel(config["settings"]["architecture"]["model-file"])
    logging.info("Model loaded. Predicting.")
    preds = model.predict(seqs, batch_size=batchSize, verbose=True,
                          workers=10, use_multiprocessing=True)
    logging.info("Predictions complete. Writing hdf5.")
    writePreds(regions, preds, outFile, numHeads, genome)


def addCoordsInfo(regions, outFile, genome):
    """Initialize an hdf5 with coordinate information.

    Creates the chrom_names, chrom_sizes, coords_chrom, coords_start,
    and coords_stop datasets.

    :param regions: A BedTool of regions that will be written.
    :param outFile: The opened hdf5 file.
    :param genome: An opened pysam.FastaFile with your genome.

    """
    stringDtype = h5py.string_dtype(encoding='utf-8')
    outFile.create_dataset('chrom_names', (genome.nreferences,), dtype=stringDtype)
    outFile.create_dataset('chrom_sizes', (genome.nreferences,), dtype='u8')
    chromNameToIndex = dict()
    chromDtype = np.uint8
    if genome.nreferences > 127:
        # We could store up to 255 in a uint8, but people might
        # .astype(int8) and that would be a problem. So we sacrifice a
        # bit if there are 128 to 255 chromosomes.
        chromDtype = np.uint16
    assert len(genome.references) < 65535, "The genome has more than 2^16 chromosomes, "\
                                           "and cannot be saved using the current hdf5 "\
                                           "format. Increase the width of the coords_chrom "\
                                           "dataset to fix. Alternatively, consider predicting "\
                                           "from a fasta file, which lets you use arbitrary "\
                                           "names for each sequence."
    chromPosDtype = np.uint32
    for i, chromName in enumerate(genome.references):
        outFile['chrom_names'][i] = chromName
        chromNameToIndex[chromName] = i
        refLen = genome.get_reference_length(chromName)
        if refLen > (2 ** 31 - 1):
            logging.debug("The genome contains a chromosome that is over four billion bases long. "
                          "Using an 8-byte integer for chromosome positions.")
            chromPosDtype = np.uint64
        outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

    # Build a table of chromosome numbers. For space savings, only store the
    # index into the chrom_names table.
    chromDset = [chromNameToIndex[r.chrom] for r in regions]
    startDset = [r.start for r in regions]
    stopDset = [r.stop for r in regions]
    logging.debug("Datasets created. Populating regions.")
    outFile.create_dataset('coords_chrom', dtype=chromDtype, data=chromDset)
    outFile.create_dataset('coords_start', dtype=chromPosDtype, data=startDset)
    outFile.create_dataset('coords_stop',  dtype=chromPosDtype, data=stopDset)  # noqa


def writePreds(regions, preds, outFile, numHeads, genome):
    """Write the predictions to an HDF5.

    :param regions: The BedTool taken from the config's bed file.
    :param preds: The output of the model's predict function, no transformations.
    :param outFile: The (opened) hdf5 file to write to
    :param numHeads: How many heads does this model have?
    :param genome: The pysam FastaFile containing your genome.

    """
    logging.info("Writing predictions")
    addCoordsInfo(regions, outFile, genome)
    logging.debug("Writing predictions.")
    for headID in wrapTqdm(range(numHeads)):  # type: ignore
        headGroup = outFile.create_group("head_{0:d}".format(headID))
        headGroup.create_dataset("logcounts", data=preds[numHeads + headID], dtype=PRED_T)
        headGroup.create_dataset("logits", data=preds[headID], dtype=PRED_T)
    outFile.close()
    logging.info("File saved.")


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r", encoding='utf-8') as configFp:
        configJson = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.makePredictionsBed.validate(configJson)
    main(configJson)
