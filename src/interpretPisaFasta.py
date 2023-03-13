#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import utils
import json
import pisa
import logging

def main(config):
    utils.setVerbosity(config["verbosity"])
    receptiveField = config["input-length"] - config["output-length"]
    generator = pisa.FastaGenerator(config["sequence-fasta"])
    writer = pisa.H5Saver(config["output-h5"], generator.numRegions, config["num-shuffles"],
            receptiveField, genome=None, useTqdm=True)

    batcher = pisa.PisaRunner(config["model-file"], config["head-id"], config["task-id"], 10, 
            generator, writer, config["num-shuffles"], receptiveField, profileFname=None)
    batcher.run()



if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
