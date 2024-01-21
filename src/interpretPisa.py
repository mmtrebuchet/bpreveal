#!/usr/bin/env python3
"""A script to generate PISA scores."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import bpreveal.utils as utils
import json
import bpreveal.interpretUtils as interpretUtils
import logging


def main(config):
    """Run the calculation.

    :param config: A JSON object matching the interpretPisa specification.
    """
    utils.setVerbosity(config["verbosity"])
    receptiveField = config["input-length"] - config["output-length"]
    kmerSize = 1
    if "kmer-size" in config:
        kmerSize = config["kmer-size"]
    else:
        logging.info("Did not find a kmer-size property in configuration file. "
                     "Using default kmer-size of 1.")
    if "fasta-file" in config or "sequence-fasta" in config:
        # We're doing a fasta run.
        if "sequence-fasta" in config:
            logging.warning("DEPRECATION: You are referring to the fasta file in your pisa JSON as "
                            "sequence-fasta. This is deprecated, please change the parameter "
                            "name to fasta-file. This will be an error in BPReveal 5.0.0.")
            config["fasta-file"] = config["sequence-fasta"]
        generator = interpretUtils.FastaGenerator(config["fasta-file"])
        genome = None
    else:
        generator = interpretUtils.PisaBedGenerator(config["bed-file"], config["genome"],
                                    config["input-length"], config["output-length"])
        genome = config["genome"]

    writer = interpretUtils.PisaH5Saver(config["output-h5"], generator.numRegions,
                                        config["num-shuffles"],
                                        receptiveField, genome=genome, useTqdm=True)
    # For benchmarking, I've added a feature where you can dump a
    # python profiling session to disk. You should probably
    # never use this feature unless you're tuning shap performance or something.
    # Long story short, all of the code's time is spent inside the shap library.
    profileFname = None
    if ("DEBUG_profile-output" in config):
        profileFname = config["DEBUG_profile-output"]

    batcher = interpretUtils.PisaRunner(config["model-file"], config["head-id"], config["task-id"],
                              10, generator, writer, config["num-shuffles"],
                              receptiveField, kmerSize, profileFname)
    batcher.run()


if (__name__ == "__main__"):
    import sys
    if sys.argv[0] in ["interpretPisaBed", "interpretPisaFasta"]:
        logging.warning("DEPRECATION: You are calling a program named " + sys.argv[0] + ". "
            "It is now just called interpretPisa and automatically detects if you're "
            "using a bed or fasta file. Instructions for updating: Call the program "
            "interpretPisa. These old program names will be removed in BPReveal 5.0.0.")
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.interpretPisa.validate(config)
    main(config)
