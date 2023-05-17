#!/usr/bin/env python3

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
import utils
import numpy as np
from keras.models import load_model
import h5py
import tqdm
import losses
import logging


def loadFasta(fastaFname):
    curSeq = ""
    curTitle = ""
    sequences = []
    titles = []
    with open(fastaFname, "r") as fp:
        for line in fp:
            line = line.strip()  # Get rid of newlines.
            if (len(line) == 0):
                continue  # There is a blank line. Ignore it.
            elif (line[0] == '>'):
                if (len(curSeq)):
                    sequences.append(curSeq)
                    titles.append(curTitle)
                    curSeq = ""
                curTitle = line[1:]
            else:
                curSeq = curSeq + line
        else:
            if (len(curSeq)):
                # Add the last sequence in the fasta.
                sequences.append(curSeq)
                titles.append(curTitle)
    return sequences, titles


def main(config):
    utils.setVerbosity(config["verbosity"])
    utils.setMemoryGrowth()
    inputLength = config["settings"]["architecture"]["input-length"]
    fastaFname = config["fasta-file"]
    batchSize = config["settings"]["batch-size"]
    numHeads = config["settings"]["heads"]
    logging.debug("Opening output hdf5 file.")
    outFile = h5py.File(config["settings"]["output-h5"], "w")

    logging.info("Loading regions")
    sequences, descriptions = loadFasta(fastaFname)
    seqs = np.zeros((len(sequences), inputLength, 4))
    for i, seq in enumerate(sequences):
        seqs[i] = utils.oneHotEncode(seq)
    logging.info("Input prepared. Loading model.")
    model = load_model(config["settings"]["architecture"]["model-file"],
                       custom_objects={'multinomialNll': losses.multinomialNll})
    logging.info("Model loaded. Predicting.")
    preds = model.predict(seqs, batch_size=batchSize,
                          verbose=(config["verbosity"] in ["INFO", "DEBUG"]),
                          workers=10, use_multiprocessing=True)
    logging.info("Predictions complete. Writing hdf5.")
    writePreds(descriptions, preds, outFile, numHeads)


def writePreds(descriptions, preds, outFile, numHeads):
    """descriptions is a list of strings, the '>' line of each entry in the input fasta.
    preds is the output of the model's predict function, no transformations.
    outputTrackList is straight from the json file.
    numheads is the number of output heads."""
    logging.info("Writing predictions")
    stringDtype = h5py.string_dtype(encoding='utf-8')
    outFile.create_dataset("descriptions", dtype=stringDtype, data=descriptions)
    logging.info("Saved fasta description lines.")
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
