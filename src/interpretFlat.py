#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import utils
import json
import interpretUtils


def main(config):
    utils.setVerbosity(config["verbosity"])
    genomeFname = None
    if "bed-file" in config:
        print(config)
        genomeFname = config["genome"]
        generator = interpretUtils.FlatBedGenerator(config["bed-file"], genomeFname,
                                      config["input-length"], config["output-length"])
    else:
        generator = interpretUtils.FastaGenerator(config["fasta-file"])

    profileWriter = interpretUtils.FlatH5Saver(config["profile-h5"], generator.numRegions,
                                               config["input-length"], genome=genomeFname,
                                               useTqdm=True)
    countsWriter = interpretUtils.FlatH5Saver(config["counts-h5"], generator.numRegions,
                                               config["input-length"], genome=genomeFname,
                                               useTqdm=False)
    # For benchmarking, I've added a feature where you can dump a
    # python profiling session to disk. You should probably
    # never use this feature unless you're tuning shap performance or something.
    # Long story short, all of the code's time is spent inside the shap library.
    profileFname = None
    if ("DEBUG_profile-output" in config):
        profileFname = config["DEBUG_profile-output"]

    batcher = interpretUtils.FlatRunner(config["model-file"], config["head-id"], config["heads"],
                                        config["profile-task-ids"], 10, generator, profileWriter,
                                        countsWriter, config["num-shuffles"], profileFname)
    batcher.run()


if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
