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

num-threads
    (Optional) The number of parallel batchers to run. For very large prediction tasks,
    increasing this can give a performance boost. If omitted, use one predictor.
    Note that each predictor loads an entire model on the GPU, so you can quickly run
    out of memory if you use more than a couple.

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

descriptions
    This is just to mirror the structure of the makePredictionsFasta output.
    Its contents are all empty strings.

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
import json
from bpreveal import utils
from bpreveal.logUtils import wrapTqdm
from bpreveal import logUtils
from bpreveal.internal import predictUtils


def main(config):
    """Run the predictions.

    :param config: A JSON object satisfying the makePredictionsBed specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    modelFname = config["settings"]["architecture"]["model-file"]
    batchSize = config["settings"]["batch-size"]
    bedFname = config["bed-file"]
    genomeFname = config["settings"]["genome"]
    numHeads = config["settings"]["heads"]
    logUtils.debug("Opening output hdf5 file.")
    outFname = config["settings"]["output-h5"]
    padding = (inputLength - outputLength) // 2

    logUtils.info("Loading regions")
    bedReader = predictUtils.BedReader(bedFname, genomeFname, padding)
    logUtils.debug("Creating predictor.")
    if "num-threads" in config:
        batcher = utils.ThreadedBatchPredictor(
            modelFname, batchSize, numThreads=config["num-threads"])
    else:
        batcher = utils.BatchPredictor(modelFname, batchSize)
    logUtils.debug("Creating writer.")
    writer = predictUtils.H5Writer(outFname, numHeads, bedReader.numPredictions,
                                   bedFname, genomeFname)

    logUtils.debug("Entering prediction loop")
    with batcher:
        pbar = wrapTqdm(bedReader.numPredictions)
        for _ in range(bedReader.numPredictions):
            batcher.submitString(bedReader.curSequence, "")
            while batcher.outputReady():
                ret = batcher.getOutput()
                pbar.update()
                writer.addEntry(ret)
        logUtils.debug("Done with main loop. Finishing stragglers.")
        while not batcher.empty():
            ret = batcher.getOutput()
            pbar.update()
            writer.addEntry(ret)
    logUtils.debug("Closing saver.")
    writer.close()
    logUtils.info("Predictions complete.")


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        configJson = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.makePredictionsBed.validate(configJson)
    main(configJson)
