#!/usr/bin/env python3
import logging
import bpreveal.motifUtils as motifUtils
import bpreveal.utils as utils
import json


def main(config):
    """Run the scan.

    :param config: A JSON object matching the motifScan specification.
    """
    utils.setVerbosity(config["verbosity"])
    if "seqlet-cutoff-settings" in config:
        assert "seqlet-cutoff-json" not in config, "You cannot name a seqlet-cutoff-json to " \
            "read in the config file if you also specify seqlet-cutoff-settings."

        logging.info("Modisco seqlet analysis requested. Starting.")
        cutoffConfig = config["seqlet-cutoff-settings"]
        # First, make the pattern objects.
        tsvFname = None
        if "seqlets-tsv" in cutoffConfig:
            tsvFname = cutoffConfig["seqlets-tsv"]
        scanPatternDict = motifUtils.seqletCutoffs(cutoffConfig["modisco-h5"],
                                                   cutoffConfig["modisco-contrib-h5"],
                                                   cutoffConfig["patterns"],
                                                   cutoffConfig["seq-match-quantile"],
                                                   cutoffConfig["contrib-match-quantile"],
                                                   cutoffConfig["contrib-magnitude-quantile"],
                                                   cutoffConfig["trim-threshold"],
                                                   cutoffConfig["trim-padding"],
                                                   cutoffConfig["background-probs"],
                                                   tsvFname
                                                   )
        logging.info("Analysis complete.")
        if "quantile-json" in cutoffConfig:
            # You specified a quantile-json inside the cutoffs config.
            # Even though it isn't necessary since we just pass scanPatternDict
            # to the scanner directly, go ahead and save out the quantile json file.
            # In this case, save out the results of the seqlet analysis.
            logging.info("Saving pattern json.")
            with open(cutoffConfig["quantile-json"], "w") as fp:
                json.dump(scanPatternDict, fp, indent=4)
    else:
        # We didn't have quantile-settings, so we'd better have quantile-json.
        # (In this case, we're reading quantile-json)
        logging.debug("Loading scanner parameters.")
        with open(config["seqlet-cutoff-json"], "r") as fp:
            scanPatternDict = json.load(fp)

    scanConfig = config["scan-settings"]
    logging.info("Beginning motif scan.")
    motifUtils.scanPatterns(scanConfig["scan-contrib-h5"],
                            scanPatternDict,
                            scanConfig["hits-tsv"],
                            scanConfig["num-threads"])


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.motifScan.validate(config)
    main(config)
