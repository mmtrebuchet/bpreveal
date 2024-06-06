"""A class to load up hdf5 files.

These files will have been generated by
:py:mod:`prepareTrainingData<bpreveal.prepareTrainingData>`.
"""
import math
import time
import h5py
import numpy as np
from numpy._typing import NDArray
import tf_keras as keras
from bpreveal import logUtils
from bpreveal.internal.constants import MODEL_ONEHOT_T, NUM_BASES, ONEHOT_T, PRED_T
from bpreveal.internal.libslide import slide, slideChar


class H5BatchGenerator(keras.utils.Sequence):
    """Loads up training data and presents it to the model.

    :param headList: The list of heads straight from the configuration JSON.
        This gets mutated when mean counts are added, see :meth:`~addMeanCounts`.
    :param dataH5: The (opened) hdf5 file generated by
        :py:mod:`prepareTrainingData<bpreveal.prepareTrainingData>`.
    :param inputLength: The input length of your model.
    :param outputLength: The output length of your model.
    :param maxJitter: How much random offset can the generator apply when it creates
        a batch?
    :param batchSize: How many samples will the model be trained on in each batch?
    """

    def __init__(self, headList: dict, dataH5: h5py.File, inputLength: int,
                 outputLength: int, maxJitter: int, batchSize: int):
        """Create the generator and do the initial data load."""
        logUtils.info("Initial load of dataset for hdf5-based generator.")
        self.headList = headList
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.maxJitter = maxJitter
        self.batchSize = batchSize
        # The shape of the sequence dataset is
        # (numRegions x (inputLength + jitter*2 x NUM_BASES))
        self.fullSequences = np.array(dataH5["sequence"], dtype=ONEHOT_T)
        self.numRegions = self.fullSequences.shape[0]
        # The shape of the profile is
        # (num-heads) x (numRegions x (outputLength + jitter*2) x numTasks)
        # Similar to the prediction script outputs, the heads are all separate,
        # and are named "head_N", where N is 0,1,2, etc.
        self.fullData = []
        for i, curHead in enumerate(headList):
            logUtils.debug(f"Loading data for {curHead['head-name']}")
            self.fullData.append(np.array(dataH5[f"head_{i}"],
                                          dtype=PRED_T))
        self.loadData()
        self.addMeanCounts()
        logUtils.info("Batch generator initialized.")

    def addMeanCounts(self) -> None:
        """For all heads, calculate the average number of reads over all regions.

        In the BPNet paper, it was shown that λ½ = ĉ/2, where ĉ is the average
        number of counts in each region, and if that value of λ is used in as the
        counts loss weight, then the profile and counts losses will be given equal weight.

        For each head in self.headList, adds a new field INTERNAL_mean-counts that
        contains the average counts over the output windows.
        For a target counts loss weight fraction f, you can calculate an initial λ value
        for the counts loss based on:
        λ = f * ĉ
        """
        for i, head in enumerate(self.headList):
            sumCounts = np.sum(self.fullData[i][:, self.maxJitter:-self.maxJitter, :])
            mean = sumCounts / self.numRegions
            head["INTERNAL_mean-counts"] = mean
            logUtils.debug(f"For head {head['head-name']}, mean counts is {mean}")

    def __len__(self) -> int:
        """How many *batches* of data are there in this Generator?"""
        return math.ceil(self.numRegions / self.batchSize)

    def __getitem__(self, idx: int) -> tuple[NDArray, list[NDArray]]:
        """Get the next *batch* of data."""
        batchStart = idx * self.batchSize
        batchEnd = min((idx + 1) * self.batchSize, self.numRegions)
        vals = []
        counts = []
        for i in range(len(self.headList)):
            vals.append(self._allBatchValues[i][batchStart:batchEnd])
            counts.append(self._allBatchCounts[i][batchStart:batchEnd])
        return self._allBatchSequences[batchStart:batchEnd], (vals + counts)

    def loadData(self) -> None:
        """Read in the hdf5 file and suck all the data into memory.

        Called only once.
        """
        self._allBatchSequences = np.empty((self.numRegions, self.inputLength, NUM_BASES),
                                           dtype=MODEL_ONEHOT_T)
        self._allBatchValues = []
        self._allBatchCounts = []
        for head in self.headList:
            self._allBatchValues.append(
                np.empty((self.numRegions, self.outputLength, head["num-tasks"]),
                         dtype=PRED_T))
            self._allBatchCounts.append(
                np.empty((self.numRegions,), dtype=PRED_T))
        self.regionIndexes = np.arange(0, self.numRegions)
        self.rng = np.random.default_rng(seed=1234)
        self.refreshData()

    def refreshData(self) -> None:
        """Go over all the data and load it into the data structures from loadData.

        Called once every epoch.
        """
        # First, randomize which regions go into which batches.
        logUtils.debug("Refreshing batch data.")
        startTime = time.perf_counter()
        self.rng.shuffle(self.regionIndexes)
        sliceCols = self.rng.integers(0,
                                      self.maxJitter * 2, size=(self.numRegions,),
                                      dtype=np.int32)
        self._shiftSequence(self.regionIndexes, sliceCols)
        self._shiftData(self.regionIndexes, sliceCols)
        stopTime = time.perf_counter()
        Δt = stopTime - startTime
        logUtils.debug(f"Loaded new batch in {Δt:5f} seconds.")

    def _shiftSequence(self, regionIndexes: NDArray, sliceCols: NDArray) -> None:
        # This is a good target for optimization - it takes multiple seconds!
        slideChar(self.fullSequences, self._allBatchSequences,
              regionIndexes, sliceCols)

    def _shiftData(self, regionIndexes: NDArray, sliceCols: NDArray) -> None:
        # This is a big target for optimization - it takes seconds to load a batch.
        for headIdx, _ in enumerate(self.headList):
            slide(self.fullData[headIdx], self._allBatchValues[headIdx],
                  regionIndexes, sliceCols)
            valSums = np.sum(self._allBatchValues[headIdx], axis=(1, 2))
            self._allBatchCounts[headIdx] = np.log(valSums)

    def on_epoch_end(self) -> None:  # pylint: disable=invalid-name
        """When the epoch is done, re-jitter the data by calling refreshData."""
        self.refreshData()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
