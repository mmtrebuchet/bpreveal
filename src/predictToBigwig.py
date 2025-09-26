#!/usr/bin/env python3
"""A script to take the predictions hdf5 file and turn it into a bigwig."""
import argparse
import multiprocessing
from bpreveal.internal.constants import PRED_T
import tqdm
import h5py
import pyBigWig
import numpy as np
import numpy.typing as npt
from bpreveal import logUtils
from bpreveal import utils


class Region:
    """Represents a single region.

    It knows where it is in the genome and also what index it occupies in the hdf5.

    :param chromIdx: The chromosome index, corresponding to ``chrom_names`` in the hdf5.
    :param start: The genomic start coordinate, inclusive.
    :param end: The genomic end coordinate, exclusive.
    :param h5Idx: The index in the hdf5 file where this region is found.
    """

    def __init__(self, chromIdx: int, start: int, end: int, h5Idx: int):
        self.chromIdx = chromIdx
        self.start = start
        self.end = end
        self.h5Idx = h5Idx

    def getValues(self, h5fp: h5py.File, mode: str, head: int, taskID: int) -> np.ndarray:
        """Get the values in the hdf5 at this region.

        :param h5fp: An opened hdf5 file containing predictions.
        :param mode: One of ``profile``, ``logits``, ``mnlogits``, ``logcounts``,
            or ``counts``.
        :param head: Which head index do you want the data from?
        :param taskID: Which task do you want the data for?
        :raise ValueError: if you asked for an invalid mode.
        :return: A vector containing the requested data.
        """
        headLogits = h5fp[f"head_{head}"]["logits"]
        headLogcounts = h5fp[f"head_{head}"]["logcounts"]

        match mode:
            case "profile":
                logits = headLogits[self.h5Idx]
                # Logits will have shape (output-length x numTasks)
                logCounts = headLogcounts[self.h5Idx]
                headProfile = utils.logitsToProfile(logits, logCounts)
                taskProfile = headProfile[:, taskID]
                profile = taskProfile
            case "logits":
                profile = headLogits[self.h5Idx, :, taskID]
            case "mnlogits":
                profile = headLogits[self.h5Idx, :, taskID]
                profile -= np.mean(profile)
            case "logcounts":
                profile = np.zeros((self.end - self.start)) + headLogcounts[self.h5Idx]
            case "counts":
                profile = np.zeros((self.end - self.start)) \
                    + np.exp(headLogcounts[self.h5Idx])
            case _:
                raise ValueError(f"{mode} is not a valid mode.")
        return profile


def getChromInserts(arg: tuple[list[Region], str, int, int, str]) -> \
        list[tuple[np.ndarray, int]]:
    """Packs all the arguments into one so it's easier to use with pool.map().

    :param arg: In order, regionList, h5Fname, headID, taskID, mode.
    :return: The inserts from vectorToListOfInserts.
    """
    regionList, h5Fname, headID, taskID, mode = arg
    with h5py.File(h5Fname, "r") as h5fp:
        vec = getChromVector(regionList, h5fp, headID, taskID, mode)
        inserts = vectorToListOfInserts(vec)
    logUtils.info(f"Finished region {str(regionList[0].chromIdx)}")
    return inserts


def getChromVector(regionList: list[Region], h5fp: h5py.File, headID: int,
                   taskID: int, mode: str) -> npt.NDArray:
    """Map the values at each Region onto a vector representing the chromosome.

    regionList should only contain regions from one chromosome, as regionList[0].chromIdx
    is used to determine the size of the returned vector.

    :param regionList: The regions you want data for.
    :param h5fp: The (open) hdf5 file of predictions.
    :param headID: The head you want data for.
    :param taskID: The task within that head that you want data for.
    :param mode: One of ``profile``, ``logits``, ``mnlogits``, ``logcounts``,
        or ``counts``.
    :return: An array as long as the chromosome, with zeros everywhere that
        regionList did not cover, and the values of the data wherever the regions
        do exist. For overlapping regions, the predictions are averaged.
    """
    chromSize = h5fp["chrom_sizes"][regionList[0].chromIdx]
    regionCounts = np.zeros((chromSize,), dtype=np.uint16)
    regionValues = np.zeros((chromSize,), dtype=PRED_T)
    for r in regionList:
        regionCounts[r.start:r.end] += 1
        regionValues[r.start:r.end] += r.getValues(h5fp, mode, headID, taskID)
    regionCounts[regionCounts == 0] = 1
    return regionValues / regionCounts


def vectorToListOfInserts(dataVector: npt.NDArray) -> list[tuple[np.ndarray, int]]:
    """Convert a chromosome vector to a list of regions that actually have data.

    Given a vector of data from getChromVector, remove all the zeros and give you
    a list of regions with actual data. For example::

        vectorToListOfInserts([0,0,0,1,2,3,0,0,0,5,6,7])
        [([1, 2, 3], 3), ([5, 6, 7], 9)]

    :param dataVector: An array representing the data along an entire chromosome.
    :return: A list of tuples. The first element of each tuple is an array of data,
        and the second is the genomic coordinate where that dataset starts.
    """
    rets = []
    regionStart = 0
    poses = np.nonzero(dataVector)[0]
    lastPoint = -1
    for pos in poses:
        if pos == lastPoint + 1:
            # We're in a block.
            pass
        else:
            # We just ended a block.
            if lastPoint > 0:
                rets.append((dataVector[regionStart:lastPoint + 1], regionStart))
            regionStart = pos

        lastPoint = pos
    rets.append((dataVector[regionStart:poses[-1] + 1], regionStart))
    return rets


def buildRegionList(inH5: h5py.File) -> dict[int, list[Region]]:
    """Builds a list of Region objects for each chromosome in the hdf5.

    :param inH5: The (open) h5py file containing predictions.
    :return: A dict mapping chromosome ID to a list of Regions on that chromosome.
    """
    numRegions = inH5["coords_chrom"].shape[0]

    # Sort the regions.
    logUtils.debug("Loading coordinate data")
    coordsChrom = list(inH5["coords_chrom"])
    coordsStart = np.array(inH5["coords_start"])
    coordsStop = np.array(inH5["coords_stop"])
    logUtils.debug("Region data loaded. Sorting.")
    regionList = []
    regionRange = range(numRegions)
    for regionNumber in regionRange:
        regionList.append(Region(coordsChrom[regionNumber],
                                 coordsStart[regionNumber],
                                 coordsStop[regionNumber],
                                 regionNumber))
    logUtils.debug("Generated list of regions to sort.")
    regionOrder = sorted(range(numRegions),
                         key=lambda x: (regionList[x].chromIdx, regionList[x].start))
    logUtils.info("Region order calculated.")

    regionsByChrom = {}
    for idx in regionOrder:
        r = regionList[idx]
        if r.chromIdx not in regionsByChrom:
            regionsByChrom[r.chromIdx] = []
        regionsByChrom[r.chromIdx].append(r)
    return regionsByChrom


def writeBigWig(inH5Fname: str, outFname: str, headID: int, taskID: int, mode: str,
                verbose: bool, negate: bool, numThreads: int) -> None:
    """Load in the h5 files and write the predictions to a bigwig file.

    :param inH5Fname: The name of an hdf5 file on disk containing predictions.
    :param outFname: The name of the bigwig file to write.
    :param headID: The head you want predictions from.
    :param taskID: The task within that head that you want predictions for.
    :param mode: One of ``profile``, ``logits``, ``mnlogits``, ``logcounts``,
        or ``counts``.
    :param verbose: Should the program emit logging information?
    :param negate: Should the predictions be negated in the output bigwig?
        Useful for chip-nexus.
    :param numThreads: How many threads should be used?
    """
    inH5 = h5py.File(inH5Fname, "r")
    logUtils.info(f"Starting to write {outFname}, head {headID} task {taskID}")
    bwHeader = []
    # pylint: disable=no-member
    for i, name in enumerate(inH5["chrom_names"].asstr()):
        bwHeader.append((str(name), int(inH5["chrom_sizes"][i])))
    # pylint: enable=no-member
    outBw = pyBigWig.open(outFname, "w")

    outBw.addHeader(bwHeader)
    logUtils.debug(str(bwHeader))
    logUtils.info("Added header.")
    regionsByChrom = buildRegionList(inH5)
    chromList = sorted(regionsByChrom.keys())
    # In order to use multiprocessing, I need to unstaple the dict into a list.
    # The order of the list is sorted(regionsByChrom.keys())
    chromRegionLists = []
    for chromIdx in chromList:
        chromRegionLists.append(
            (regionsByChrom[chromIdx], inH5Fname, headID, taskID, mode))
    logUtils.info("Extracted list of regions to process.")
    logUtils.info("Beginning to extract profile data.")

    with multiprocessing.Pool(numThreads) as p:
        # Get the insert list for each chromosome in a subprocess.
        chromInsertLists = p.map(getChromInserts, chromRegionLists)
    logUtils.info("Insert lists calculated. Writing bigwig.")
    pbar = None
    if verbose:
        totalInserts = 0
        for il in chromInsertLists:
            totalInserts += len(il)
        pbar = tqdm.tqdm(total=totalInserts)
    for i, chromIdx in enumerate(chromList):
        inserts = chromInsertLists[i]
        chromName = inH5["chrom_names"][chromIdx].decode("utf-8")
        for ins in inserts:
            if pbar is not None:
                pbar.update()
            values, start = ins
            if negate:
                values *= -1
            insertValues = [float(x) for x in values]
            outBw.addEntries(chromName,
                             start,
                             values=insertValues,
                             span=1, step=1)
    if pbar is not None:
        pbar.close()
    logUtils.info("Bigwig written. Closing.")
    outBw.close()
    logUtils.info("Done.")


def getParser() -> argparse.ArgumentParser:
    """Generate the argument parser.

    :return: An ArgumentParser, ready to call parse_args()
    """
    parser = argparse.ArgumentParser(
        description="Take an hdf5-format file generated by "
                    "the predict script and render it to a bigwig.")
    parser.add_argument("--h5",
                        help="The name of the hdf5-format file to be read in.")
    parser.add_argument("--bw",
                        help="The name of the bigwig file that should be written.")
    parser.add_argument("--head-id",
                        help="Which head number do you want data for?",
                        dest="headID", type=int)
    parser.add_argument("--task-id",
                        help="Which task in that head do you want?",
                        dest="taskID", type=int)
    parser.add_argument("--mode",
                        help="What do you want written? Options are 'profile', meaning "
                        "you want (softmax(logits) * exp(logcounts)), or 'logits', "
                        "meaning you just want logits, or 'mnlogits', meaning you want "
                        "the logits, but mean-normalized (for easier display), or "
                        "'logcounts', meaning you want the log counts for every region, "
                        "or 'counts', meaning you want exp(logcounts). "
                        "You will usually want 'profile'")
    parser.add_argument("--verbose",
                        help="Display progress as the file is being written.",
                        action="store_true")
    parser.add_argument("--negate",
                        help="Negate all of the values written to the bigwig. "
                        "Used for negative-strand predictions.", action="store_true")
    parser.add_argument("--threads",
                        help="Number of threads to use for calculating profile "
                        "tracks (max: num chromosomes)", type=int,
                        default=24, dest="numThreads")
    return parser


def main() -> None:
    """Run the program."""
    args = getParser().parse_args()
    logUtils.setBooleanVerbosity(args.verbose)
    writeBigWig(args.h5, args.bw, args.headID, args.taskID, args.mode, args.verbose,
                args.negate, args.numThreads)


if __name__ == "__main__":
    main()
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
