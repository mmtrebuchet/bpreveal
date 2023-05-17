#!/usr/bin/env python3

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
import utils
import pybedtools
import numpy as np
import pysam
from keras.models import load_model
import h5py
import tqdm
import losses
import logging
# Generate a simple sequence model taking one-hot encoded input and
# producing a logits profile and a log(counts) scalar.


def main(config):
    utils.setVerbosity(config["verbosity"])
    utils.setMemoryGrowth()
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    batchSize = config["settings"]["batch-size"]
    genomeFname = config["settings"]["genome"]
    numHeads = config["settings"]["heads"]
    logging.debug("Opening output hdf5 file.")
    outFile = h5py.File(config["settings"]["output-h5"], "w")
    regions = pybedtools.BedTool(config["bed-file"])
    seqs = np.zeros((len(regions), inputLength, 4))
    genome = pysam.FastaFile(genomeFname)
    padding = (inputLength - outputLength) // 2

    logging.info("Loading regions")
    for i, region in enumerate(regions):
        curSeq = genome.fetch(region.chrom, region.start - padding, region.stop + padding)
        seqs[i] = utils.oneHotEncode(curSeq)
    logging.info("Input prepared. Loading model.")
    model = load_model(config["settings"]["architecture"]["model-file"],
                       custom_objects={'multinomialNll': losses.multinomialNll})
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
    outFile.create_dataset('chrom_sizes', (genome.nreferences,), dtype='u4')
    chromNameToIndex = dict()
    assert len(genome.references) < 65535, "The genome has more than 2^16 chromosomes, "\
                                           "and cannot be saved using the current hdf5 "\
                                           "format. Increase the width of the coords_chrom "\
                                           "dataset to fix. Alternatively, consider predicting "\
                                           "from a fasta file, which lets you use arbitrary "\
                                           "names for each sequence."

    for i, chromName in enumerate(genome.references):
        outFile['chrom_names'][i] = chromName
        chromNameToIndex[chromName] = i
        refLen = genome.get_reference_length(chromName)
        assert refLen < (2 ** 32 - 1), "The genome contains a chromosome that is over four "\
                                       "billion bases long. This will overflow the coords_start "\
                                       "and coords_stop fields in the output file. Widen the "\
                                       "data type of these fields to fix."
        outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

    # Build a table of chromosome numbers. For space savings, only store the
    # index into the chrom_names table.
    chromDset = [chromNameToIndex[r.chrom] for r in regions]
    startDset = [r.start for r in regions]
    stopDset = [r.stop for r in regions]
    logging.debug("Datasets created. Populating regions.")
    outFile.create_dataset('coords_chrom', dtype='u2', data=chromDset)
    outFile.create_dataset('coords_start', dtype='u4', data=startDset)
    outFile.create_dataset('coords_stop',  dtype='u4', data=stopDset)
    logging.debug("Writing predictions.")
    for headId in tqdm.tqdm(range(numHeads)):
        headGroup = outFile.create_group("head_{0:d}".format(headId))
        headGroup.create_dataset("logcounts", data=preds[numHeads + headId])
        headGroup.create_dataset("logits", data=preds[headId])
    outFile.close()
    logging.info("File saved.")


if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
