#!/usr/bin/env python3
"""A script to make predictions using a BPReveal model.

This program streams input from disk and writes output as it calculates, so
it can run with very little memory even for extremely large prediction tasks.


BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/makePredictions.bnf


Parameter Notes
---------------

heads
    Gives the number of output heads for your model.
    You don't need to tell this program how many tasks there are for each
    head, since it just blindly sticks whatever the model outputs into the
    hdf5 file.

output-h5
    The name of the output file that will contain the predictions.

batch-size
    How many samples should be run simultaneously? I recommend 64 or so.

model-file
    The name of the Keras model file on disk.

input-length, output-length
    The input and output lengths of your model.

fasta-file
    A file containing the sequences for which you'd like predictions. Each sequence in
    this bed file must be ``input-length`` long. If you specify ``fasta-file``,
    you cannot also specify ``bed-file`` and ``genome``
    (except, optionally, in the ``coordinates`` section.)

bed-file, genome
    If you do not give ``fasta-file``, you can instead give a
    ``bed-file`` and ``genome`` fasta. Each region in the bed file should be
    ``output-length`` long, and the program will automatically inflate the regions
    to the ``input-length`` of your model.

num-threads
    (Optional) How many parallel predictors should be run? Unless you're really taxed
    for performance, leave this at 1.

coordinates
    (Optional, only valid with ``fasta-file``.)
    The ``bed-file`` and ``genome`` entries may be specified to add coordinate information
    when you predict from ``fasta-file``. If provided, then the output hdf5 will contain
    ``chrom_names``, ``chrom_sizes``, ``coords_chrom``, ``coords_start``, and
    ``coords_end`` datasets, in addition to the descriptions dataset.
    Only the *coordinate* information is taken from the bed file, and only chromosome
    size information is loaded from the genome file.
    The actual sequences to predict will be drawn from ``fasta-file``.
    This way, you can make predictions from a fasta but then easily convert it to a
    bigwig.

Output Specification
--------------------
This program will produce an hdf5-format file containing the predicted values.
It is organized as follows:

descriptions
    A list of strings of length (numRegions,).
    If you give a fasta file, these will correspond to
    the description lines (i.e., the lines starting with ``>``).
    If you gave a bed file as input, each one will be an empty string.

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

chrom_names
    A list of strings that give you the meaning
    of each index in the ``coords_chrom`` dataset.
    This is particularly handy when you want to make a bigwig file, since
    you can extract a header from this data.
    Only populated if a bed file and genome were provided.

chrom_sizes
    The size of each chromosome in the same order
    as ``chrom_names``.
    Mostly used to create bigwig headers.
    Only populated if a bed file and genome were provided.

coords_chrom
    A list of integers, one for each region
    predicted, that gives the chromosome index (see ``chrom_names``)
    for that region.
    Only populated if a bed file and genome were provided.

coords_start
    The start base of each predicted region.
    Only populated if a bed file and genome were provided.

coords_stop
    The end point of each predicted region.
    Only populated if a bed file and genome were provided.

metadata
    A group containing the configuration that was used when the program was run.

API
---

"""


import json
import pybedtools  # pylint: disable=unused-import # noqa
import bpreveal.schema
from bpreveal import utils
import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
from bpreveal import logUtils
from bpreveal.logUtils import wrapTqdm
from bpreveal.internal import predictUtils
import bpreveal.internal.files
from bpreveal.internal import interpreter


def getReader(config: dict) -> predictUtils.BedReader | predictUtils.FastaReader:
    """Loads the reader appropriate for the configuration."""
    if "bed-file" in config:
        # We're reading from a bed.
        inputLength = config["settings"]["architecture"]["input-length"]
        outputLength = config["settings"]["architecture"]["output-length"]
        padding = (inputLength - outputLength) // 2
        bedFname = config["bed-file"]
        genomeFname = config["genome"]
        reader = predictUtils.BedReader(bedFname, genomeFname, padding)
    elif "fasta-file" in config:
        fastaFname = config["fasta-file"]
        reader = predictUtils.FastaReader(fastaFname)
    else:
        raise ValueError("Could not find an input source in your config.")
    return reader


def getWriter(config: dict, numPredictions: int) -> predictUtils.H5Writer:
    """Creates a writer appropriate for the configuration."""
    outFname = config["settings"]["output-h5"]
    numHeads = config["settings"]["heads"]
    if "bed-file" in config:
        bedFname = config["bed-file"]
        genomeFname = config["genome"]
        writer = predictUtils.H5Writer(fname=outFname, numHeads=numHeads,
                                       numPredictions=numPredictions, bedFname=bedFname,
                                       genomeFname=genomeFname, config=str(config))
        logUtils.debug("Initialized writer from a bed reader.")
    elif "fasta-file" in config:
        bedFname = config["coordinates"]["bed-file"]
        genomeFname = config["coordinates"]["genome"]
        if "coordinates" in config:
            writer = predictUtils.H5Writer(fname=outFname, numHeads=numHeads,
                                           numPredictions=numPredictions, bedFname=bedFname,
                                           genomeFname=genomeFname, config=str(config))
            logUtils.debug("Initialized writer from a fasta reader with coordinates.")
        else:
            writer = predictUtils.H5Writer(outFname, numHeads, numPredictions)
            logUtils.debug("Initialized writer from a fasta reader without coordinates.")
    else:
        raise ValueError("Could not construct a writer.")
    return writer


def main(config: dict) -> None:
    """Run the predictions.

    :param config: is taken straight from the json specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    if "genome" in config["settings"]:
        config["genome"] = config["settings"]["genome"]
        del config["settings"]["genome"]
        logUtils.error("You are using an old-format bed prediction json.")
        logUtils.error('The genome argument should be moved from "settings"')
        logUtils.error("to the root of the JSON object.")
        logUtils.error("This will be an error in BPReveal 6.0.0")
        logUtils.error("Here is a corrected version:")
        logUtils.error(json.dumps(config, indent=4))
    batchSize = config["settings"]["batch-size"]
    modelFname = config["settings"]["architecture"]["model-file"]

    # Before we can build the output dataset in the hdf5 file, we need to
    # know how many regions we will be asked to predict.
    reader = getReader(config)
    writer = getWriter(config, reader.numPredictions)
    if "num-threads" in config:
        batcher = utils.ThreadedBatchPredictor(modelFname, batchSize,
                                               numThreads=config["num-threads"])
    else:
        batcher = utils.BatchPredictor(modelFname, batchSize)
    logUtils.info("Entering prediction loop.")
    # Now we just iterate over the reader and submit to our batcher.
    with batcher:
        pbar = wrapTqdm(reader.numPredictions, smoothing=0.1)
        for _ in range(reader.numPredictions):
            batcher.submitString(reader.curSequence, reader.curLabel)
            reader.pop()
            while batcher.outputReady():
                # We've just run a batch. Write it out.
                ret = batcher.getOutput()
                pbar.update()
                writer.addEntry(ret)
        # Done with the main loop, clean up the batcher.
        logUtils.debug("Done submitting queries. Draining batcher.")
        while not batcher.empty():
            # We've just run a batch. Write it out.
            ret = batcher.getOutput()
            pbar.update()
            writer.addEntry(ret)
    writer.close()


if __name__ == "__main__":
    import sys
    if sys.argv[0].split("/")[-1] in {"makePredictionsBed", "makePredictionsFasta",
                       "makePredictionsBed.py", "makePredictionsFasta.py"}:
        logUtils.warning(
            "DEPRECATION: You are calling a program named " + sys.argv[0] + ". "
            "It is now just called makePredictions and automatically detects if you're "
            "using a bed or fasta file. Instructions for updating: Call the program "
            "makePredictions. These old program names will be removed in BPReveal 6.0.0.")
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    bpreveal.schema.makePredictions.validate(configJson)
    main(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
