"""Utility functions for adding noise.

.. warning::
    This tool is deprecated and will be removed in BPReveal 6.0.
    It turns out that it's not very useful.

This program is needed for a reason that I don't quite understand - gmstar is
not picklable if I put it inside addNoise.py.
"""
import json
import random
import h5py
import numpy as np
from bpreveal.internal.constants import ONEHOT_T, PRED_T, H5_CHUNK_SIZE
from bpreveal import logUtils
random.seed(1234)
SEQ_AR = None
HEAD_DATS = []


def loadFile(h5Fname: str, numHeads: int) -> None:
    """Initializer for a multiprocessing pool, loads up a global hdf5 file.

    :param h5Fname: The name of the input hdf5 File that the Pool workers will read from.
    :param numHeads: The number of heads of your model.
    """
    global SEQ_AR, HEAD_DATS  # pylint: disable=global-variable-not-assigned
    h5fp = h5py.File(h5Fname, "r")
    SEQ_AR = np.array(h5fp["sequence"])
    for i in range(numHeads):
        HEAD_DATS.append(np.array(h5fp[f"head_{i}"]))


def gmstar(getMutArgs: tuple[str, list[int], int]):
    """A wrapper around getMutated that is to be used with a Pool of workers.

    :param getMutArgs: A 3-tuple containing:
        the configuration json as a string,
        a list giving the number of tasks for each head, and
        an integer that will be used to seed the rng.
    :return: See :py:func`~getMutated`.
    """
    return getMutated(json.loads(getMutArgs[0]), getMutArgs[1], getMutArgs[2])


def applyAddSub(maxReads: int, minReads: int, fracMut: float, headData: np.ndarray,
                maxChange: int, add: bool, rng):
    """Either add or subtract random noise from headData

    :param maxReads: If a locus has more reads than this, don't add noise.
    :param minReads: If a locus has fewer reads than this, don't add noise.
    :param fracMut: What fraction of bases should be noised?
    :param headData: An array of data from a single head, of shape
        ((output-length + jitter * 2) x num-tasks)
    :param maxChange: Add or subtract a random number of bases up to this maximium.
    :param add: Do you want to add noise or subtract? True to add, False to subtract.
    """
    # We want to randomly add.
    assert add or minReads >= 1, "Cannot subtract reads with minimum-reads 0"
    outIdxPool = []
    if maxReads is None:
        maxReads = int(np.sum(headData))
    outIdxPoolTuple = np.nonzero(np.logical_and(headData <= maxReads, headData >= minReads))
    outIdxPool = np.empty((outIdxPoolTuple[0].shape[0], 2), dtype=np.int64)
    outIdxPool[:, 0] = outIdxPoolTuple[0]
    outIdxPool[:, 1] = outIdxPoolTuple[1]
    numSamples = int(fracMut * headData.shape[0] * headData.shape[1])
    numSamples = min(numSamples, len(outIdxPool))
    outIdxes = rng.choice(outIdxPool, numSamples)
    numsToAdd = rng.choice(np.arange(1, maxChange + 1, dtype=np.int64), numSamples)
    if not add:
        numsToAdd = -numsToAdd
    headData[outIdxes[:, 0], outIdxes[:, 1]] = numsToAdd
    if not add:
        headData[headData < 0] = 0


def applyShift(maxDistance: int, shiftIndependently: bool, fracMut: float,
               headData: np.ndarray, rng):
    """Shift the data randomly but keep it near its source.

    :param rng: A numpy random number generator.
    :param maxDistance: What is the furthest that a read is allowed to drift?
    :param shiftIndependently: Should each read be moved on its own, or should all of
        the reads at one position move together?
    :param fracMut: What fraction of bases should be subject to the shuffling?
    :param headData: An array containing the data for a specific head.
    """
    readIdxes = np.nonzero(headData)
    if shiftIndependently:
        readProbs = headData[readIdxes]
    else:
        readProbs = np.ones(readIdxes[0].shape)
    readProbs /= np.sum(readProbs)
    # Now we have our possible things to shift.
    numSamples = int(fracMut * headData.shape[0] * headData.shape[1])
    numSamples = min(len(readIdxes), numSamples)
    shiftInputs = rng.choice(np.array(readIdxes).T, size=numSamples, replace=False,
                             p=readProbs)
    runShiftAlgo(shiftInputs, maxDistance, headData, shiftIndependently)


def runShiftAlgo(shiftInputs: np.ndarray, maxDistance: int,
                 headData: np.ndarray, shiftIndependently: bool):
    """Actually perform the shifting.

    :param shiftInputs: The read indexes that should have their data shifted.
    :param maxDistance: What's the furthest that a read can be shifted?
    :param headData: The array of data for a head.
    :param shiftIndependently: Should each read be shifted independently?
    """
    for inIdx in shiftInputs:
        offset = random.randint(1, maxDistance)
        if random.random() < 0.5:
            offset = -offset
        outIdx = inIdx[0] + offset
        # If we run off the end, flip the offset.
        if outIdx >= headData.shape[0]:
            outIdx -= 2 * offset
        if outIdx < 0:
            outIdx += 2 * offset
        if shiftIndependently:
            headData[inIdx[0], inIdx[1]] -= 1
            headData[outIdx, inIdx[1]] += 1
        else:
            tmp = headData[inIdx[0], inIdx[1]]
            headData[inIdx[0], inIdx[1]] = headData[outIdx, inIdx[1]]
            headData[outIdx, inIdx[1]] = tmp


def mutateProfile(mutationTypes: list[dict], headData: np.ndarray, rng) -> np.ndarray:
    """Given a single head's data (that may contain multiple tasks), perturb it.

    :param mutationTypes: Straight from the config json, these are the profile-mutation-types.
    :param headData: An array containing the data for the given head, from the input H5.
    :return: A new array with the same shape as headData that has been perturbed.
    """
    mutFracs = []
    for mt in mutationTypes:
        mutFracs.append(mt["output-distribution-fraction"])
    totalMutFrac = sum(mutFracs)
    # Add in a fraction to represent the identity mutation.
    mutFracs.append(1 - totalMutFrac)
    mutIdxToPerform = random.choices(
        range(len(mutFracs)), weights=mutFracs, k=1)
    mutIdxToPerform = mutIdxToPerform[0]
    if mutIdxToPerform == len(mutationTypes):
        # We don't actually want to make a mutation.
        return headData
    mutToPerform = mutationTypes[mutIdxToPerform]
    match mutToPerform:
        case {"type": "add", "maximum-reads": maxReads, "minimum-reads": minReads,
              "max-change": maxChange, "fraction-mutated": fracMut,
              "output-distribution-fraction": _}:
            applyAddSub(maxReads, minReads, fracMut, headData, maxChange, True, rng)
        case {"type": "subtract", "maximum-reads": maxReads, "minimum-reads": minReads,
              "max-change": maxChange, "fraction-mutated": fracMut,
              "output-distribution-fraction": _}:
            applyAddSub(maxReads, minReads, fracMut,
                        headData, maxChange, False, rng)
        case {"type": "shift", "shift-max-distance": maxDistance,
              "shift-reads-independently": shiftIndependently,
              "fraction-mutated": fracMut,
              "output-distribution-fraction": _}:
            applyShift(maxDistance, shiftIndependently, fracMut, headData, rng)
    return headData


def getMutated(config: dict, tasksPerHead: list[int], seed: int)\
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Performs the requested mutations.

    :param config: The json configuration.
    :param tasksPerHead: A list giving, for each head, how many tasks it has.
    :param seed: An integer that will be used to seed the RNG.
    :return: A tuple. The first element is a dataset containing one-hot-encoded sequences.
        The second element is a list containing profile data for each of the heads, in order.
    """
    random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    global SEQ_AR, HEAD_DATS  # pylint: disable=global-variable-not-assigned
    numInputRegions = SEQ_AR.shape[0]
    inputIdx = random.randrange(0, numInputRegions - 1)
    inputSeq = SEQ_AR[inputIdx]
    headDats = []
    numSeqMuts = int(config["sequence-fraction-mutated"] * inputSeq.shape[0])
    for i in range(len(tasksPerHead)):
        headDats.append(HEAD_DATS[i][inputIdx])
    if random.random() < config["sequence-distribution-fraction"]:
        # We want to mutate the sequence.
        mutateIndexes = random.sample(
            list(range(0, inputSeq.shape[0])), k=numSeqMuts)
        mutateIndexes = sorted(mutateIndexes)
        for mutLoc in mutateIndexes:
            inputBase = inputSeq[mutLoc, :]
            mutationTargets = np.random.random(size=(4,))
            # Zero out the base that is already present.
            mutationTargets *= (1 - inputBase)
            newBaseOneHotPos = np.argmax(mutationTargets)
            inputSeq[mutLoc] = np.array([0, 0, 0, 0])
            inputSeq[mutLoc, newBaseOneHotPos] = 1
    # We've now (possibly) mutated the input sequence.

    for i in range(len(tasksPerHead)):
        headDats[i] = mutateProfile(
            config["profile-mutation-types"], headDats[i], rng)
    return inputSeq, headDats


def writeOutput(outFname: str, sequences: np.ndarray, heads: list[np.ndarray]) -> None:
    """Writes the given datasets to an hdf5 file that can be used to train BPReveal models.

    :param outFname: The name of the hdf5 file to write
    :param sequences: An array of shape (num-regions x (input-length + 2*jitter) x 4), containing
        one-hot encoded sequences.
    :param heads: A list of profile data for each head. Each element of this list has shape
        (num-regions x (output-length + 2 * jitter) * num-tasks).
    """
    logUtils.info("Writing output files.")
    outFp = h5py.File(outFname, "w")
    outFp.create_dataset("sequence", data=sequences, dtype=ONEHOT_T,
                         chunks=(H5_CHUNK_SIZE, sequences.shape[1], 4),
                         compression="gzip")
    for i, head in enumerate(heads):
        logUtils.debug(f"Saving head {i}")
        outFp.create_dataset(f"head_{i}", data=head, dtype=PRED_T,
                             chunks=(H5_CHUNK_SIZE,
                                     head.shape[1], head.shape[2]),
                             compression="gzip")
    outFp.close()
    logUtils.info("Output file saved.")
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
