#!/usr/bin/env python3

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
import bpreveal.utils as utils
import numpy as np
import h5py
import tqdm
import logging
from bpreveal.utils import ONEHOT_T, PRED_T


class FastaReader:
    curSequence = ""
    curLabel = ""
    numPredictions = 0
    _nextLabel = ""

    def __init__(self, fastaFname):
        # First, scan over the file and count how many sequences are in it.
        logging.info("Counting number of samples.")
        with open(fastaFname, "r") as fp:
            for line in fp:
                line = line.strip()  # Get rid of newlines.
                if len(line) == 0:
                    continue  # There is a blank line. Ignore it.
                elif line[0] == '>':
                    self.numPredictions += 1
            fp.close()
        logging.info("Found {0:d} entries in input fasta".format(self.numPredictions))
        # Note that we close the file after that with block, so this re-opens it
        # at position zero.
        self._fp = open(fastaFname, "r")
        # We know a fasta starts with >, so read in the first label.
        self._nextLabel = self._fp.readline().strip()[1:]

    def pop(self):
        # We know we're at the start of the sequence section of a fasta.
        self.curLabel = self._nextLabel
        self.curSequence = ""
        inSequence = True
        curLine = self._fp.readline()
        while inSequence and len(curLine) > 1:
            if curLine[0] != '>':
                self.curSequence = self.curSequence + curLine.strip()
            else:
                self._nextLabel = curLine[1:].strip()
                inSequence = False
                break
            curLine = self._fp.readline()


class H5Writer:
    def __init__(self, fname, numHeads, numPredictions):
        self._fp = h5py.File(fname, 'w')
        self.numHeads = numHeads
        self.numPredictions = numPredictions
        self.writeHead = 0
        self.batchWriteHead = 0
        self.writeChunkSize = 100
        # We don't know the output length yet, since the model hasn't run any batches.
        # We'll construct the datasets on the fly once we get our first output.

    def buildDatasets(self, sampleOutputs):
        # Since descriptions will not consume an inordinate amout of memory, and string
        # handling is messy with h5py, just store all the descriptions in a list and
        # write them out at the end.
        self._descriptionList = []
        self.headBuffers = []
        # h5 files are very slow if you write many times to non-chunked datasets.
        # So I create chunked datasets, and then create internal buffers to store
        # up to 100 entries before actually committing to the hdf5 file.
        # This optimization means that the program is now GPU-limited and not
        # h5py-limited, which is how things should be.
        for headId in range(self.numHeads):
            headGroup = self._fp.create_group("head_{0:d}".format(headId))
            # These are the storage buffers for incoming data.
            headBuffer = [np.empty((self.writeChunkSize, ), dtype=PRED_T),  # counts
                          np.empty((self.writeChunkSize, ) + sampleOutputs[headId].shape,
                                   dtype=PRED_T)]  # profile
            self.headBuffers.append(headBuffer)
            headGroup.create_dataset("logcounts", (self.numPredictions,),
                                     dtype=PRED_T,
                                     chunks=(self.writeChunkSize,))
            headGroup.create_dataset("logits",
                                     ((self.numPredictions,) + sampleOutputs[headId].shape),
                                     dtype=PRED_T,
                                     chunks=((self.writeChunkSize,) + sampleOutputs[headId].shape))
        logging.debug("Initialized datasets.")

    def addEntry(self, batcherOut):
        # Give this exactly the output from the batcher, and it will queue the data
        # to be written to the hdf5 on the next commit.

        logitsLogcounts, label = batcherOut
        if self.writeHead == 0 and self.batchWriteHead == 0:
            # We haven't constructed our datasets yet. Do so now, because
            # now we know the output size of the model.
            self.buildDatasets(logitsLogcounts)

        self._descriptionList.append(label)

        for headId in range(self.numHeads):
            logits = logitsLogcounts[headId]
            logcounts = logitsLogcounts[headId + self.numHeads]
            self.headBuffers[headId][0][self.batchWriteHead] = logcounts
            self.headBuffers[headId][1][self.batchWriteHead] = logits
        self.batchWriteHead += 1
        # Have we filled our storage buffers? If so, write them out.
        if self.batchWriteHead == self.writeChunkSize:
            self.commit()

    def commit(self):
        # Actually write the data out to the backing hdf5 file.
        start = self.writeHead
        stop = start + self.batchWriteHead
        for headId in range(self.numHeads):
            headGroup = self._fp["head_{0:d}".format(headId)]
            headBuffer = self.headBuffers[headId]
            headGroup["logits"][start:stop] = headBuffer[1][:self.batchWriteHead]
            headGroup["logcounts"][start:stop] = headBuffer[0][:self.batchWriteHead]
        self.writeHead += self.batchWriteHead
        self.batchWriteHead = 0

    def close(self):
        # You MUST call close on this object, as otherwise the last bit of data won't
        # get written to disk.
        if self.batchWriteHead != 0:
            self.commit()
        stringDType = h5py.string_dtype(encoding='utf-8')
        self._fp.create_dataset("descriptions", dtype=stringDType, data=self._descriptionList)
        logging.info("Closing h5.")
        self._fp.close()


def main(config):
    utils.setVerbosity(config["verbosity"])
    utils.setMemoryGrowth()
    fastaFname = config["fasta-file"]
    batchSize = config["settings"]["batch-size"]
    modelFname = config["settings"]["architecture"]["model-file"]
    numHeads = config["settings"]["heads"]
    logging.debug("Opening output hdf5 file.")
    outFname = config["settings"]["output-h5"]

    # Before we can build the output dataset in the hdf5 file, we need to
    # know how many fasta regions we will be asked to predict.
    fastaReader = FastaReader(fastaFname)
    batcher = utils.BatchPredictor(modelFname, batchSize)
    writer = H5Writer(outFname, numHeads, fastaReader.numPredictions)
    logging.info("Entering prediction loop.")
    # Now we just iterate over the fasta file and submit to our batcher.
    if config["verbosity"] in ["INFO", "DEBUG"]:
        pbar = tqdm.tqdm(range(fastaReader.numPredictions))
    else:
        pbar = range(fastaReader.numPredictions)
    for _ in pbar:
        fastaReader.pop()
        batcher.submitString(fastaReader.curSequence, fastaReader.curLabel)
        while batcher.outputReady():
            # We've just run a batch. Write it out.
            writer.addEntry(batcher.getOutput())
    # Done with the main loop, clean up the batcher.
    logging.debug("Done with main loop.")
    batcher.runBatch()
    while batcher.outputReady():
        # We've just run a batch. Write it out.
        writer.addEntry(batcher.getOutput())
    writer.close()


if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
