#!/usr/bin/env python3

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
import bpreveal.utils as utils
utils.setMemoryGrowth()
import pybedtools
import numpy as np
import pysam
import h5py
import logging
from bpreveal.utils import ONEHOT_T, PRED_T, wrapTqdm

# Generate a simple sequence model taking one-hot encoded input and
# producing a logits profile and a log(counts) scalar.


def main(config):
    utils.setVerbosity(config["verbosity"])
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    batchSize = config["settings"]["batch-size"]
    genomeFname = config["settings"]["genome"]
    numHeads = config["settings"]["heads"]
    logging.debug("Opening output hdf5 file.")
    outFile = h5py.File(config["settings"]["output-h5"], "w")
    regions = pybedtools.BedTool(config["bed-file"])
    seqs = np.zeros((len(regions), inputLength, 4), dtype=ONEHOT_T)
    genome = pysam.FastaFile(genomeFname)
    padding = (inputLength - outputLength) // 2

    logging.info("Loading regions")
    for i, region in wrapTqdm(list(enumerate(regions))):
        curSeq = genome.fetch(region.chrom, region.start - padding, region.stop + padding)
        seqs[i] = utils.oneHotEncode(curSeq)
    logging.info("Input prepared. Loading model.")
    model = utils.loadModel(config["settings"]["architecture"]["model-file"])
    logging.info("Model loaded. Predicting.")
    preds = model.predict(seqs, batch_size=batchSize, verbose=True,
                          workers=10, use_multiprocessing=True)
    logging.info("Predictions complete. Writing hdf5.")
    writePreds(regions, preds, outFile, numHeads, genome)


def writePreds(regions, preds, outFile, numHeads, genome):
    """Regions is the BedTool taken from the config's bed file.
    preds is the output of the model's predict function, no transformations.
    outputTrackList is straight from the json file.
    numheads is the number of output heads.
    chromSizes is a dict mapping chromosome names to size. """
    logging.info("Writing predictions")
    stringDtype = h5py.string_dtype(encoding='utf-8')
    outFile.create_dataset('chrom_names', (genome.nreferences,), dtype=stringDtype)
    outFile.create_dataset('chrom_sizes', (genome.nreferences,), dtype='u8')
    chromNameToIndex = dict()
    chromDtype = np.uint8
    if genome.nreferences > 127:
        # We could store up to 255 in a uint8, but people might
        # .astype(int8) and that would be a problem. So we sacrifice a
        # bit if there are 128 to 255 chromosomes.
        chromDtype = np.uint16
    assert len(genome.references) < 65535, "The genome has more than 2^16 chromosomes, "\
                                           "and cannot be saved using the current hdf5 "\
                                           "format. Increase the width of the coords_chrom "\
                                           "dataset to fix. Alternatively, consider predicting "\
                                           "from a fasta file, which lets you use arbitrary "\
                                           "names for each sequence."
    chromPosDtype = np.uint32
    for i, chromName in enumerate(genome.references):
        outFile['chrom_names'][i] = chromName
        chromNameToIndex[chromName] = i
        refLen = genome.get_reference_length(chromName)
        if refLen > (2 ** 31 - 1):
            logging.debug("The genome contains a chromosome that is over four billion bases long. "
                          "Using an 8-byte integer for chromosome positions.")
            chromPosDtype = np.uint64
        outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

    # Build a table of chromosome numbers. For space savings, only store the
    # index into the chrom_names table.
    chromDset = [chromNameToIndex[r.chrom] for r in regions]
    startDset = [r.start for r in regions]
    stopDset = [r.stop for r in regions]
    logging.debug("Datasets created. Populating regions.")
    outFile.create_dataset('coords_chrom', dtype=chromDtype, data=chromDset)
    outFile.create_dataset('coords_start', dtype=chromPosDtype, data=startDset)
    outFile.create_dataset('coords_stop',  dtype=chromPosDtype, data=stopDset)

    logging.debug("Writing predictions.")
    for headId in wrapTqdm(range(numHeads)):
        headGroup = outFile.create_group("head_{0:d}".format(headId))
        headGroup.create_dataset("logcounts", data=preds[numHeads + headId], dtype=PRED_T)
        headGroup.create_dataset("logits", data=preds[headId], dtype=PRED_T)
    outFile.close()
    logging.info("File saved.")


if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    import jsonschema
    import bpreveal.schema
    jsonschema.validate(schema=bpreveal.schema.makePredictionsBed,
                        instance=config)
    main(config)
