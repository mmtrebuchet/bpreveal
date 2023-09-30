#!/usr/bin/env python3
import logging
import cwmUtils
import utils
import json


def cwmScanMain(config):
    utils.setVerbosity(config["verbosity"])
    if "quantile-settings" in config:
        logging.info("Modisco seqlet analysis requested. Starting.")
        quantileConfig = config["quantile-settings"]
        # First, make the pattern objects.
        csvFname = None
        if "seqlets-csv" in quantileConfig:
            csvFname = quantileConfig["seqlets-csv"]
        scanPatternDict = cwmUtils.analyzeSeqlets(quantileConfig["modisco-h5"],
                                              quantileConfig["modisco-contrib-h5"],
                                              quantileConfig["patterns"],
                                              quantileConfig["ic-quantile"],
                                              quantileConfig["contrib-quantile"],
                                              quantileConfig["L1-quantile"],
                                              quantileConfig["trim-threshold"],
                                              quantileConfig["trim-padding"],
                                              quantileConfig["background-probs"],
                                              csvFname
                                              )
        logging.info("Analysis complete.")
        if "quantile-json" in config:
            # You specified both quantile-settings and quantile-json.
            # In this case, save out the results of the seqlet analysis.
            logging.info("Saving pattern json.")
            with open(config["quantile-json"], "w") as fp:
                json.dump(scanPatternDict, fp, indent=4)
    else:
        # We didn't have quantile-settings, so we'd better have quantile-json.
        # (In this case, we're reading quantile-json)
        logging.info("Loading scanner parameters.")
        with open(config["quantile-json"], "r") as fp:
            scanPatternDict = json.load(fp)

    cwmUtils.scanPatterns(config["contrib-h5"],
                          scanPatternDict,
                          config["hits-csv"],
                          config["hits-bed"],
                          config["window-size"],
                          config["num-threads"])

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    cwmScanMain(config)
