#!/usr/bin/env python3
import logging
import cwmUtils
import utils
import json


def cwmQuantileMain(config):
    utils.setVerbosity(config["verbosity"])
    logging.info("Starting seqlet analysis")
    # First, make the pattern objects.
    csvFname = None
    if "seqlets-csv" in config:
        csvFname = config["seqlets-csv"]
    scanPatternDict = cwmUtils.analyzeSeqlets(config["modisco-h5"],
                                              config["modisco-contrib-h5"],
                                              config["patterns"],
                                              config["ic-quantile"],
                                              config["contrib-quantile"],
                                              config["L1-quantile"],
                                              config["trim-threshold"],
                                              config["trim-padding"],
                                              config["background-probs"],
                                              csvFname
                                              )
    logging.info("Analysis complete.")
    if "quantile-json" in config:
        logging.info("Saving pattern json.")
        with open(config["quantile-json"], "w") as fp:
            json.dump(scanPatternDict, fp, indent=4)


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    cwmQuantileMain(config)
