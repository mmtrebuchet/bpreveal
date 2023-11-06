#!/usr/bin/env python3
import logging
import bpreveal.motifUtils as motifUtils
import bpreveal.utils as utils
import json


def motifSeqletCutoffsMain(config):
    utils.setVerbosity(config["verbosity"])
    logging.info("Starting seqlet analysis")
    # First, make the pattern objects.
    tsvFname = None
    if "seqlets-tsv" in config:
        tsvFname = config["seqlets-tsv"]
    scanPatternDict = motifUtils.seqletCutoffs(config["modisco-h5"],
                                               config["modisco-contrib-h5"],
                                               config["patterns"],
                                               config["seq-match-quantile"],
                                               config["contrib-match-quantile"],
                                               config["contrib-magnitude-quantile"],
                                               config["trim-threshold"],
                                               config["trim-padding"],
                                               config["background-probs"],
                                               tsvFname
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
    motifSeqletCutoffsMain(config)
