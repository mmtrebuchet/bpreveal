#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import utils
import json
import pisa


def main(config):
    utils.setVerbosity(config["verbosity"])
    receptiveField = config["input-length"] - config["output-length"]
    generator = pisa.BedGenerator(config["bed-file"], config["genome"],
                                  config["input-length"], config["output-length"])
    writer = pisa.H5Saver(config["output-h5"], generator.numRegions, config["num-shuffles"],
                          receptiveField, genome=config["genome"], useTqdm=True)
    # For benchmarking, I've added a feature where you can dump a
    # python profiling session to disk. You should probably
    # never use this feature unless you're tuning shap performance or something.
    # Long story short, all of the code's time is spent inside the shap library.
    profileFname = None
    if ("DEBUG_profile-output" in config):
        profileFname = config["DEBUG_profile-output"]

    batcher = pisa.PisaRunner(config["model-file"], config["head-id"], config["task-id"],
                              10, generator, writer, config["num-shuffles"],
                              receptiveField, profileFname)
    batcher.run()


if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
