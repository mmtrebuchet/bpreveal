#!/usr/bin/env python3
"""A script to generate PISA scores.

PISA is described :doc:`here<pisa>`

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/interpretPisa.bnf


Parameter Notes
---------------

model-file
    The name of the saved Keras model to interpret.

head-id
    The head number that you'd like to interpret.

task-id
    The task within that head that you'd like to interpret. Note that while
    interpretFlat can combine multiple tasks, PISA can only consider one task
    at a time.

output-h5
    The name of the hdf5 file where you'd like the output saved.

output-length, input-length
    The output and input length of your model.

genome, bed-file
    If you provide a bed file, this file represents the *individual bases* that
    should be shapped. There is no restriction on the number of regions, nor on
    their length. If you are interested in the effects of a particular motif,
    then you'd put the region surrounding that motif in the bed file, making it
    as large as you want to see the interactions you're interested in.

fasta-file
    You can also supply the sequences directly with a fasta file. Since PISA
    calculates shap scores for a single base, this tool always calculates the
    PISA scores for the *leftmost* base in the output window. Suppose the input
    length is 3090 bp, and the output is 1000 bp. In this case, the receptive
    field is 2091 bp, and there are 1045 bp of overhang on each end of the
    input sequence. So, for each input sequence, this program will assign shap
    scores to the 1046th base (one-indexed) of the input sequence.


num-shuffles
    This is the number of background samples that should be used for
    calculating shap values. I recommend 20.

kmer-size
    (Optional) If provided, this changes how the shuffles work. By default (or
    if you specify ``kmer-size = 1``) all of the bases in the input are jumbled
    randomly. However, if you specify ``kmer-size=2``, then the distribution of
    dimers will be preserved in the shuffled sequences. If you specify
    ``kmer-size=3``, then trimers will be preserved, and so on.

num-threads
    (Optional) If provided, use multiple batcher threads in parallel, on the same
    GPU. Shap is relatively inefficient on the GPU, and by using two or three threads,
    you can get better throughput. If you run into memory issues, use one thread.

correct-receptive-field
    (Optional) If set to ``true``, then the output array will have the correct width,
    which is input-length - output-length + 1. By default, use the (incorrect) value
    of input-length - output-length, for compatibility with old scripts. In version
    5.0.0, the default switched from ``False`` to ``True``.


Output Specification
--------------------

It produces an hdf5 format which is organized as follows:

input_predictions
    A ``(numSamples,)`` array of the logit value
    of the target base when that sequence is run through the network.

shuffle_predictions
    A ``(numSamples, numShuffles)`` array of
    the logits of the target base in the shuffled reference sequences.

sequence
    A one-hot encoded array representing the sequence
    under each PISA value.
    The shape is ``(num regions * receptive-field * NUM_BASES)``.
    Note that this is receptive field, not input length, since each base
    being shapped will only be affected by bases in its receptive field,
    and there's no reason to store the noise.

shap
    A table of the shap scores.
    The shape is the same as the sequence table, and each position in the
    shap table represents the corresponding base in the sequence table.
    These values are contribution scores to the difference-from-reference
    of the logit at this base.

metadata
    A group containing the configuration used to perform the calculation.

If you specified a bed file with input regions, it will also have these datasets:

chrom_names
    A list of strings giving the name of each chromosome. This is used to
    figure out which chromosome each number in `coords_chrom` corresponds to.

chrom_sizes
    A list of integers giving the size of each chromosome. This is mostly here
    as a handy reference when you want to make a bigwig file.

coords_base
    The center point for each of the regions in the table of PISA values.

coords_chrom
    The chromosome on which each PISA vector is found. This is a list of
    integers. The width of the integer data type may vary from run to run, and
    is calculated based on the number of chromosomes in the genome file.

If, however, you gave a fasta file of input sequences, it will instead have the
following dataset:

descriptions
    A string taken from the fasta file. These are the comment (``>``) lines,
    and are stored without the leading ``>``. This will have shape
    ``(numSamples,)``

HISTORY
-------
Before BPReveal 4.0.0, this was two programs: ``interpretPisaBed`` and
``interpretPisaFasta``. They shared almost all of the same code, so they were
merged into interpretPisa. The old names are still present in the bin/
directory, where they symlink to the same python file in src.
In BPReveal 5.0.0, support for calling the input fasta ``sequence-fasta`` was made an error.
In BPReveal 6.0.0, the old symlinks will be removed and all reference to sequence-fasta
be removed.


API
---

"""
from collections.abc import Callable
import bpreveal.schema
from bpreveal import logUtils
from bpreveal.internal import interpretUtils
from bpreveal.internal import interpreter


def pisaMetric(headID: int, taskID: int) -> Callable:
    """A metric to extract the leftmost output logit of a model.

    :param headID: The head number that you want counts for.
    :param taskID: The task within this head that you want shap values for.
    :return: A function (that takes a model as its argument) that can
        be passed into the depths of the interpretation system.
    """
    #             Extract the output from the given task.
    #                Use the leftmost output logit.     |
    #             Use all samples in this batch.  |     |
    #           Use the given output head.     |  |     |
    #                                    ↓     ↓  ↓     ↓
    return lambda model: model.outputs[headID][:, 0, taskID]


def main(config: dict) -> None:
    """Run the calculation.

    :param config: A JSON object matching the interpretPisa specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    receptiveField = config["input-length"] - config["output-length"]
    if "correct-receptive-field" in config:
        if config["correct-receptive-field"]:
            receptiveField += 1
    else:
        logUtils.info(
            "You have not specified correct-receptive-field in your configuration. "
            "As of BPReveal 5.0.0, the default shape of the output has increased by one "
            "to fix an off-by-one error in the receptive field calculation."
            "Instructions for updating: "
            "Add 'correct-receptive-field': false to your config to keep using the "
            "(incorrect) receptive field calculation. Or add "
            "'correct-receptive-field': true to your config silence this message "
            "and keep using the new (correct) behavior. "
            "In BPReveal 6.0.0, the receptive field will automatically be corrected "
            "without any warning and this parameter will have no effect.")
        receptiveField += 1

    kmerSize = 1
    if "kmer-size" in config:
        kmerSize = config["kmer-size"]
    else:
        logUtils.info("Did not find a kmer-size property in configuration file. "
                      "Using default kmer-size of 1.")
    numThreads = 1
    if "num-threads" in config:
        numThreads = config["num-threads"]
    else:
        logUtils.info("Did not find a num-threads property in configuration file. "
                      "Using default of 1. Using more batchers may give a "
                      "performance boost.")

    if "fasta-file" in config or "sequence-fasta" in config:
        # We're doing a fasta run.
        if "sequence-fasta" in config:
            logUtils.error("DEPRECATION: You are referring to the fasta file in your "
                           "PISA JSON as sequence-fasta. This is deprecated, please "
                           "change the parameter name to fasta-file. This will be an "
                           "error in BPReveal 6.0.0.")
            config["fasta-file"] = config["sequence-fasta"]
        generator = interpretUtils.FastaGenerator(config["fasta-file"])
        logUtils.info(f"Initialized generator for {config["fasta-file"]}")
        genome = None
    else:
        generator = interpretUtils.PisaBedGenerator(bedFname=config["bed-file"],
                                                    genomeFname=config["genome"],
                                                    inputLength=config["input-length"],
                                                    outputLength=config["output-length"])
        genome = config["genome"]
        logUtils.info(f"Initialized generator for {config["bed-file"]}")

    writer = interpretUtils.PisaH5Saver(outputFname=config["output-h5"],
                                        numSamples=generator.numRegions,
                                        numShuffles=config["num-shuffles"],
                                        receptiveField=receptiveField,
                                        genome=genome,
                                        useTqdm=logUtils.getVerbosity() <= logUtils.INFO,
                                        config=str(config))
    logUtils.info(f"Initialized saver with output file {config["output-h5"]}")
    # If you want to use a custom metric, you could add that here.
    # I can't think of a good reason to use anything other than the PISA
    # metric for PISA calculations, though.
    metric = pisaMetric(config["head-id"], config["task-id"])
    logUtils.info(f"Constructed metric for head {config["head-id"]} and task {config["task-id"]}.")
    batcher = interpretUtils.InterpRunner(modelFname=config["model-file"],
                                          metrics=[metric],
                                          batchSize=10,
                                          generator=generator,
                                          savers=[writer],
                                          numShuffles=config["num-shuffles"],
                                          kmerSize=kmerSize,
                                          numThreads=numThreads,
                                          backend="shap",
                                          useHypotheticalContribs=False)
    batcher.run()
    logUtils.info("PISA interpretation complete, exiting.")


if __name__ == "__main__":
    import sys
    if sys.argv[0] in {"interpretPisaBed", "interpretPisaFasta"}:
        logUtils.error("DEPRECATION: You are calling the program " + sys.argv[0] + ". "
                       "It is now just called interpretPisa and automatically detects if you're "
                       "using a bed or fasta file. Instructions for updating: Call the program "
                       "interpretPisa. These old program names will be removed in BPReveal 6.0.0.")
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    bpreveal.schema.interpretPisa.validate(configJson)
    main(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
