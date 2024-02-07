#!/usr/bin/env python3
"""A script to generate PISA scores.

PISA is described in detail in the overview document.

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
    pisa scores for the *leftmost* base in the output window. Suppose the input
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


Output Specification
--------------------

It produces an hdf5 format which is organized as follows:

head-id
    An integer representing which head of the model was
    used to generate the data.

task-id
    An integer giving the task number within the specified head.

input_predictions
    A ``(numSamples,)`` array of the logit value
    of the target base when that sequence is run through the network.

shuffle_predictions
    A ``(numSamples, numShuffles)`` array of
    the logits of the target base in the shuffled reference sequences.

sequence
    A one-hot encoded array representing the sequence
    under each PISA value.
    The shape is ``(num regions * receptive-field * 4)``.
    Note that this is receptive field, not input width, since each base
    being shapped will only be affected by bases in its receptive field,
    and there's no reason to store the noise.

shap
    A table of the shap scores.
    The shape is the same as the sequence table, and each position in the
    shap table represents the corresponding base in the sequence table.
    These values are contribution scores to the difference-from-reference
    of the logit at this base.

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
In BPReveal 6.0.0, the old symlinks will be removed.


API
---

"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
from bpreveal import interpretUtils
from bpreveal import logUtils


def main(config):
    """Run the calculation.

    :param config: A JSON object matching the interpretPisa specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    receptiveField = config["input-length"] - config["output-length"]
    kmerSize = 1
    if "kmer-size" in config:
        kmerSize = config["kmer-size"]
    else:
        logUtils.info("Did not find a kmer-size property in configuration file. "
                      "Using default kmer-size of 1.")
    if "fasta-file" in config or "sequence-fasta" in config:
        # We're doing a fasta run.
        if "sequence-fasta" in config:
            logUtils.warning("DEPRECATION: You are referring to the fasta file in your "
                             "pisa JSON as sequence-fasta. This is deprecated, please "
                             "change the parameter name to fasta-file. This will be an "
                             "error in BPReveal 6.0.0.")
            config["fasta-file"] = config["sequence-fasta"]
        generator = interpretUtils.FastaGenerator(config["fasta-file"])
        genome = None
    else:
        generator = interpretUtils.PisaBedGenerator(config["bed-file"], config["genome"],
                                                    config["input-length"],
                                                    config["output-length"])
        genome = config["genome"]

    writer = interpretUtils.PisaH5Saver(config["output-h5"], generator.numRegions,
                                        config["num-shuffles"],
                                        receptiveField, genome=genome, useTqdm=True)
    # For benchmarking, I've added a feature where you can dump a
    # python profiling session to disk. You should probably
    # never use this feature unless you're tuning shap performance or something.
    # Long story short, all of the code's time is spent inside the shap library.
    profileFname = None
    if "DEBUG_profile-output" in config:
        profileFname = config["DEBUG_profile-output"]

    batcher = interpretUtils.PisaRunner(config["model-file"],
                                        config["head-id"], config["task-id"],
                                        10, generator, writer,
                                        config["num-shuffles"], receptiveField,
                                        kmerSize, profileFname)
    batcher.run()


if __name__ == "__main__":
    import sys
    if sys.argv[0] in ["interpretPisaBed", "interpretPisaFasta"]:
        logUtils.warning("DEPRECATION: You are calling a program named " + sys.argv[0] + ". "
            "It is now just called interpretPisa and automatically detects if you're "
            "using a bed or fasta file. Instructions for updating: Call the program "
            "interpretPisa. These old program names will be removed in BPReveal 5.0.0.")
    with open(sys.argv[1], "r", encoding='utf-8') as configFp:
        configJson = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.interpretPisa.validate(configJson)
    main(configJson)
