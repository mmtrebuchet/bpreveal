#!/usr/bin/env python3
"""A script to make predictions using a list of sequences.

This program streams input from disk and writes output as it calculates, so
it can run with very little memory even for extremely large prediction tasks.


BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/makePredictionsFasta.bnf


Parameter Notes
---------------

heads
    Gives the number of output heads for your model.
    You don't need to tell this program how many tasks there are for each
    head, since it just blindly sticks whatever the model outputs into the
    hdf5 file.

output-h5
    The name of the output file that will contain the predictions.

batch-size
    How many samples should be run simultaneously? I recommend 64 or so.

model-file
    The name of the Keras model file on disk.

input-length, output-length
    The input and output lengths of your model.

fasta-file
    A file containing the sequences for which you'd like predictions. Each sequence in
    this bed file must be ``input-length`` long.

num-threads
    (Optional) How many parallel predictors should be run? Unless you're really taxed
    for performance, leave this at 1.

bed-file, genome
    These are optional. If provided, then the output hdf5 will contain ``chrom_names``,
    ``chrom_sizes``, ``coords_chrom``, ``coords_start``, and ``coords_end`` datasets,
    in addition to the descriptions dataset. This way, you can make predictions from a
    fasta but then easily convert it to a bigwig.

Output Specification
--------------------
This program will produce an hdf5-format file containing the predicted values.
It is organized as follows:

descriptions
    A list of strings of length (numRegions,).
    Each string corresponds to one description line (i.e., a line starting
    with ``>``).

head_0, head_1, head_2, ...
    You get a subgroup for each output head of the model. The subgroups are named
    ``head_N``, where N is 0, 1, 2, etc.
    Each head contains:

    logcounts
        A vector of shape (numRegions,) that gives
        the logcounts value for each region.

    logits
        The array of logit values for each track for
        each region.
        The shape is (numRegions x outputWidth x numTasks).
        Don't forget that you must calculate the softmax on the whole
        set of logits, not on each task's logits independently.
        (Use :py:func:`bpreveal.utils.logitsToProfile` to do this.)

chrom_names, chrom_sizes, coords_chrom, coords_start, coords_stop
    If you provided ``bed-file`` and ``genome`` entries in your json,
    these datasets will be populated. They mirror their meaning in the
    output from :py:mod:`makePredictionsBed<bpreveal.makePredictionsBed>`.

API
---

"""


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
from bpreveal import utils
import numpy as np
import h5py
from bpreveal import logging
from bpreveal.logging import wrapTqdm
from bpreveal.utils import PRED_T


class FastaReader:
    """Streams a fasta file from disk lazily.

    :param fastaFname: The name of the fasta file to load.
    """

    curSequence = ""
    """The current sequence in this file. Updated by :py:meth:`~pop`."""
    curLabel = ""
    """The current description line in this file. Updated by :py:meth:`~pop`."""
    numPredictions = 0
    _nextLabel = ""

    def __init__(self, fastaFname):
        """Scan the file to count the total lines, then load the first sequence."""
        # First, scan over the file and count how many sequences are in it.
        logging.info("Counting number of samples.")
        with open(fastaFname, "r") as fp:
            for line in wrapTqdm(fp):
                line = line.strip()  # Get rid of newlines.
                if len(line) == 0:
                    continue  # There is a blank line. Ignore it.
                if line[0] == '>':
                    self.numPredictions += 1
            fp.close()
        logging.info("Found {0:d} entries in input fasta".format(self.numPredictions))
        # Note that we close the file after that with block, so this re-opens it
        # at position zero.
        self._fp = open(fastaFname, "r")
        # We know a fasta starts with >, so read in the first label.
        self._nextLabel = self._fp.readline().strip()[1:]

    def pop(self):
        """Pop the current sequence off the queue. Updates curSequence and curLabel."""
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
    """Batches up predictions and saves them in chunks.

    :param fname: The name of the hdf5 file to save.
    :param numHeads: The total number of heads for this model.
    :param numPredictions: How many total predictions will be made?

    """

    def __init__(self, fname, numHeads, numPredictions,
                 bedFname: str | None = None, genomeFname: str | None = None):
        """Load everything that can be loaded before the subprocess launches."""
        self._fp = h5py.File(fname, 'w')
        self.numHeads = numHeads
        self.numPredictions = numPredictions
        self.writeHead = 0
        self.batchWriteHead = 0
        self.writeChunkSize = 100
        if bedFname is not None:
            logging.info("Adding coordinate information.")
            import pybedtools
            import pysam
            from bpreveal import makePredictionsBed
            assert genomeFname is not None, "Must supply a genome to get coordinate information."
            regions = pybedtools.BedTool(bedFname)
            with pysam.FastaFile(genomeFname) as genome:
                makePredictionsBed.addCoordsInfo(regions, self._fp, genome)
        # We don't know the output length yet, since the model hasn't run any batches.
        # We'll construct the datasets on the fly once we get our first output.

    def buildDatasets(self, sampleOutputs):
        """Actually construct the output hdf5 file.

        You must give this function the first prediction from the model so that
        it can size its datasets appropriately.

        :param sampleOutputs: An output from the Batcher. This is not written to the file,
            it's just used to get the right size for the datasets.
        """
        # Since descriptions will not consume an inordinate amount of memory, and string
        # handling is messy with h5py, just store all the descriptions in a list and
        # write them out at the end.
        self._descriptionList = []
        self.headBuffers = []
        # h5 files are very slow if you write many times to non-chunked datasets.
        # So I create chunked datasets, and then create internal buffers to store
        # up to 100 entries before actually committing to the hdf5 file.
        # This optimization means that the program is now GPU-limited and not
        # h5py-limited, which is how things should be.
        for headID in range(self.numHeads):
            headGroup = self._fp.create_group("head_{0:d}".format(headID))
            # These are the storage buffers for incoming data.
            headBuffer = [np.empty((self.writeChunkSize, ), dtype=PRED_T),  # counts
                          np.empty((self.writeChunkSize, ) + sampleOutputs[headID].shape,
                                   dtype=PRED_T)]  # profile
            self.headBuffers.append(headBuffer)
            headGroup.create_dataset("logcounts", (self.numPredictions,),
                                     dtype=PRED_T,
                                     chunks=(self.writeChunkSize,))
            headGroup.create_dataset("logits",
                                     ((self.numPredictions,) + sampleOutputs[headID].shape),
                                     dtype=PRED_T,
                                     chunks=(self.writeChunkSize,) + sampleOutputs[headID].shape)
        logging.debug("Initialized datasets.")

    def addEntry(self, batcherOut):
        """Add a single output from the Batcher."""
        # Give this exactly the output from the batcher, and it will queue the data
        # to be written to the hdf5 on the next commit.

        logitsLogcounts, label = batcherOut
        if self.writeHead == 0 and self.batchWriteHead == 0:
            # We haven't constructed our datasets yet. Do so now, because
            # now we know the output size of the model.
            self.buildDatasets(logitsLogcounts)

        self._descriptionList.append(label)

        for headID in range(self.numHeads):
            logits = logitsLogcounts[headID]
            logcounts = logitsLogcounts[headID + self.numHeads]
            self.headBuffers[headID][0][self.batchWriteHead] = logcounts
            self.headBuffers[headID][1][self.batchWriteHead] = logits
        self.batchWriteHead += 1
        # Have we filled our storage buffers? If so, write them out.
        if self.batchWriteHead == self.writeChunkSize:
            self.commit()

    def commit(self):
        """Actually write the data out to the backing hdf5 file."""
        start = self.writeHead
        stop = start + self.batchWriteHead
        for headID in range(self.numHeads):
            headGroup = self._fp["head_{0:d}".format(headID)]
            headBuffer = self.headBuffers[headID]
            headGroup["logits"][start:stop] = headBuffer[1][:self.batchWriteHead]
            headGroup["logcounts"][start:stop] = headBuffer[0][:self.batchWriteHead]
        self.writeHead += self.batchWriteHead
        self.batchWriteHead = 0

    def close(self):
        """Close the output hdf5.

        You MUST call close on this object, as otherwise the last bit of data won't
        get written to disk.
        """
        if self.batchWriteHead != 0:
            self.commit()
        stringDType = h5py.string_dtype(encoding='utf-8')
        self._fp.create_dataset("descriptions", dtype=stringDType, data=self._descriptionList)
        logging.info("Closing h5.")
        self._fp.close()


def main(config):
    """Run the predictions.

    :param config: is taken straight from the json specification.
    """
    logging.setVerbosity(config["verbosity"])
    fastaFname = config["fasta-file"]
    batchSize = config["settings"]["batch-size"]
    modelFname = config["settings"]["architecture"]["model-file"]
    numHeads = config["settings"]["heads"]
    logging.debug("Opening output hdf5 file.")
    outFname = config["settings"]["output-h5"]

    # Before we can build the output dataset in the hdf5 file, we need to
    # know how many fasta regions we will be asked to predict.
    fastaReader = FastaReader(fastaFname)
    if "num-threads" in config:
        batcher = utils.ThreadedBatchPredictor(modelFname, batchSize,
                                               numThreads=config["num-threads"])
    else:
        batcher = utils.BatchPredictor(modelFname, batchSize)
    if "coordinates" in config:
        writer = H5Writer(outFname, numHeads, fastaReader.numPredictions,
                          config["coordinates"]["bed-file"],
                          config["coordinates"]["genome"])
    else:
        writer = H5Writer(outFname, numHeads, fastaReader.numPredictions)
    logging.info("Entering prediction loop.")
    # Now we just iterate over the fasta file and submit to our batcher.
    with batcher:
        pbar = wrapTqdm(fastaReader.numPredictions)
        for _ in range(fastaReader.numPredictions):
            fastaReader.pop()
            batcher.submitString(fastaReader.curSequence, fastaReader.curLabel)
            while batcher.outputReady():
                # We've just run a batch. Write it out.
                ret = batcher.getOutput()
                pbar.update()
                writer.addEntry(ret)
        # Done with the main loop, clean up the batcher.
        logging.debug("Done with main loop.")
        while not batcher.empty():
            # We've just run a batch. Write it out.
            ret = batcher.getOutput()
            pbar.update()
            writer.addEntry(ret)
        writer.close()


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        configJson = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.makePredictionsFasta.validate(configJson)
    main(configJson)
