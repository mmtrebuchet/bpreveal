from tensorflow import keras
import numpy as np
import math
import logging
import time
import h5py
from bpreveal.utils import ONEHOT_T, MODEL_ONEHOT_T


class H5BatchGenerator(keras.utils.Sequence):

    def __init__(self, headList: dict, dataH5: h5py.File, inputLength: int,
                 outputLength: int, maxJitter: int, batchSize: int):
        logging.info("Initial load of dataset for hdf5-based generator.")
        self.headList = headList
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.maxJitter = maxJitter
        self.batchSize = batchSize
        # The shape of the sequence dataset is
        # (numRegions x (inputLength + jitter*2 x 4))
        self.fullSequences = np.array(dataH5["sequence"], dtype=MODEL_ONEHOT_T)
        self.numRegions = self.fullSequences.shape[0]
        # The shape of the profile is
        # (num-heads) x (numRegions x (outputLength + jitter*2) x numTasks)
        # Similar to the prediction script outputs, the heads are all separate,
        # and are named "head_N", where N is 0,1,2, etc.
        self.fullData = []
        for i, h in enumerate(headList):
            self.fullData.append(np.array(dataH5["head_{0:d}".format(i)]))
        self.loadData()
        self.addMeanCounts()
        logging.info("Batch generator initialized.")

    def addMeanCounts(self):
        """For all heads, calculate the average number of reads over all regions.
        In the BPNet paper, it was shown that λ½ = ĉ/2, where ĉ is the average
        number of counts in each region, and if that value of λ is used in as the
        counts loss weight, then the profile and counts losses will be given equal weight.

        For each head in self.headList, adds a new field INTERNAL_mean-counts that contains
        the average counts over the output windows.
        For a target counts loss weight fraction f, you can calculate an initial λ value
        for the counts loss based on:
        λ = f * ĉ
        """
        for i, head in enumerate(self.headList):
            sumCounts = np.sum(self.fullData[i][:, self.maxJitter:-self.maxJitter, :])
            head["INTERNAL_mean-counts"] = sumCounts / self.numRegions
            logging.debug("For head {0:s}, mean counts is {1:f}"
                          .format(head["head-name"], sumCounts / self.numRegions))

    def __len__(self) -> int:
        return math.ceil(self.numRegions / self.batchSize)

    def __getitem__(self, idx):
        return self.batchSequences[idx], (self.batchVals[idx] + self.batchCounts[idx])

    def loadData(self) -> None:
        self.batchSequences = []
        self.batchVals = []
        self.batchCounts = []
        regionsRemaining = self.numRegions
        for i in range(len(self)):
            # Build an empty sequence array. Note we have to special-case the last round
            # in case the number of regions is not divisible by batch size.
            if (regionsRemaining > self.batchSize):
                curBatchSize = self.batchSize
            else:
                curBatchSize = regionsRemaining
            regionsRemaining -= self.batchSize
            self.batchSequences.append(np.empty((curBatchSize, self.inputLength, 4),
                                                dtype=MODEL_ONEHOT_T))
            newBatchVals = []
            newBatchCounts = []
            for head in self.headList:
                newBatchVals.append(
                    np.empty((curBatchSize,
                              self.outputLength,
                              head["num-tasks"]),
                             dtype=np.float32))
                newBatchCounts.append(np.empty((curBatchSize, )))
            self.batchVals.append(newBatchVals)
            self.batchCounts.append(newBatchCounts)
        self.regionIndexes = np.arange(0, self.numRegions)
        self.rng = np.random.default_rng(seed=1234)
        self.refreshData()

    def refreshData(self) -> None:
        # Go over all the data and load it into the data structures
        # allocated in loadData.
        # First, randomize which regions go into which batches.
        logging.debug("Refreshing batch data.")
        startTime = time.perf_counter()
        self.rng.shuffle(self.regionIndexes)
        for i in range(self.numRegions):
            tmpSequence = self.fullSequences[i]
            # fullData is (num-heads)
            #            x (  num-regions
            #               x output-width+jitter*2
            #               x numTasks)
            # so this slice takes the ith region of each of the head datasets.
            tmpData = [x[i, :, :] for x in self.fullData]
            if (self.maxJitter > 0):
                jitterOffset = self.rng.integers(0, self.maxJitter * 2 + 1)
                tmpSequence = tmpSequence[jitterOffset:jitterOffset + self.inputLength, :]
                for j in range(len(tmpData)):
                    tmpData[j] = tmpData[j][jitterOffset:jitterOffset + self.outputLength, :]
                # Note that this generator does *not* revcomp the data,
                # in case the input are stranded like chip-nexus.
            # We've collected and trimmed the data, now to fill in the
            # batch arrays.
            regionIdx = self.regionIndexes[i]
            batchIdx = regionIdx // self.batchSize
            batchRegionIdx = regionIdx % self.batchSize
            batchSeqs = self.batchSequences[batchIdx]
            batchVals = self.batchVals[batchIdx]
            batchCounts = self.batchCounts[batchIdx]
            batchSeqs[batchRegionIdx, :] = tmpSequence
            for headIdx, head in enumerate(self.headList):
                batchVals[headIdx][batchRegionIdx, :] = tmpData[headIdx]
                batchCounts[headIdx][batchRegionIdx] = \
                    np.log(np.sum(tmpData[headIdx]))
        stopTime = time.perf_counter()
        Δt = stopTime - startTime
        logging.debug("Loaded new batch in {0:5f} seconds.".format(Δt))

    def on_epoch_end(self):
        self.refreshData()
