#!/usr/bin/env python3
"""A little script that takes an hdf5 generated by interpretFlat and renders a bigwig file."""
import argparse
import h5py
import pyBigWig
import numpy as np
from bpreveal import logUtils
from bpreveal.internal.constants import H5_CHUNK_SIZE, IMPORTANCE_AR_T, ONEHOT_AR_T


class BatchedH5Reader:
    """Reads in an hdf5 in chunks.

    :param batchSize: The number of elements that should be read in at once
    :param h5fp: The (open) hdf5 file object to read from.
    """
    curChunkStart: int
    curChunkEnd: int

    def __init__(self, h5fp: h5py.File, batchSize: int):
        self.batchSize = batchSize
        self.h5fp = h5fp
        self._maxIndex = self.h5fp["hyp_scores"].shape[0]
        self._curChunkStart = -100000
        self._curChunkEnd = -100000
        self._loadChunk(0)

    def _loadChunk(self, index: int) -> None:
        """Makes sure that the given chunk has been read."""
        if self._curChunkStart <= index < self._curChunkEnd:
            return
        self._curChunkStart = index
        self._curChunkEnd = min(index + self.batchSize, self._maxIndex)
        self.curScore = np.array(self.h5fp["hyp_scores"]
                                 [self._curChunkStart:self._curChunkEnd, :, :])
        self.curSeqs = np.array(self.h5fp["input_seqs"]
                                [self._curChunkStart:self._curChunkEnd, :, :])

    def readScore(self, idx: int) -> IMPORTANCE_AR_T:
        """Read in score information from the hdf5.

        :param idx: The index of the score that we want to read.
        :return: An array of shape (input-length, NUM_BASES) containing hypothetical scores.
        """
        self._loadChunk(idx)
        return self.curScore[idx - self._curChunkStart, :, :]

    def readSeq(self, idx: int) -> ONEHOT_AR_T:
        """Read in sequence information from the hdf5.

        :param idx: The index of the sequence to read.
        :return: One-hot encoded sequences, shape (input-length, NUM_BASES)
        """
        self._loadChunk(idx)
        return self.curSeqs[idx - self._curChunkStart, :, :]


def writeBigWig(inH5: h5py.File, outFname: str) -> None:  # pylint: disable=too-many-statements
    """Write the data in the h5 file to a bigwig on disk.

    :param inH5: The (open) hdf5 file to use
    :param outFname: The name of the bigwig to save
    """
    bwHeader = []
    h5Reader = BatchedH5Reader(inH5, H5_CHUNK_SIZE)
    chromIdxToName = {}
    for i, name in enumerate(inH5["chrom_names"].asstr()):
        bwHeader.append((str(name), int(inH5["chrom_sizes"][i])))
        chromIdxToName[i] = name
    outBw = pyBigWig.open(outFname, "w")
    outBw.addHeader(sorted(bwHeader))
    logUtils.debug("Bigwig header" + str((sorted(bwHeader))))
    numRegions = inH5["coords_chrom"].shape[0]
    if isinstance(inH5["coords_chrom"][0], bytes):
        logUtils.warning("You are using an old-style hdf5 file for importance scores. "
            "Support for these files will be removed in BPReveal 5.0. "
            "Instructions for updating: Re-calculate importance scores.")
        coordsChrom = np.array(inH5["coords_chrom"].asstr())
    else:
        coordsChromIdxes = np.array(inH5["coords_chrom"])
        coordsChrom = np.array([chromIdxToName[x] for x in coordsChromIdxes])
    # Sort the regions.
    coordsStart = np.array(inH5["coords_start"])
    coordsEnd   = np.array(inH5["coords_end"])  # noqa
    regionOrder = sorted(range(numRegions), key=lambda x: (coordsChrom[x], coordsStart[x]))
    startWritingAt = 0
    regionID = regionOrder[0]
    regionChrom = coordsChrom[regionID]
    curChrom = regionChrom
    regionStart = coordsStart[regionID]
    regionStop = coordsEnd[regionID]
    logUtils.info("Files opened; writing regions")
    regionRange = logUtils.wrapTqdm(range(numRegions))
    nextRegion = None
    nextStop = 0
    for regionNumber in regionRange:
        # Extract the appropriate region from the sorted list.

        if regionChrom != curChrom:
            curChrom = regionChrom
            startWritingAt = 0

        if startWritingAt < regionStart:
            # The next region starts beyond the end
            # of the previous one. Some bases will not be filled in.
            startWritingAt = regionStart
        # By default, write the whole region.
        stopWritingAt = regionStop
        # As long as we aren't on the last region, check for overlaps.

        if regionNumber < numRegions - 1:
            nextRegion = regionOrder[regionNumber + 1]
            nextChrom = coordsChrom[nextRegion]
            nextStart = coordsStart[nextRegion]
            nextStop = coordsEnd[nextRegion]
            if nextChrom == regionChrom and nextStart < stopWritingAt:
                # The next region overlaps. So stop writing before then.
                overlapSize = regionStop - nextStart
                stopWritingAt = stopWritingAt - overlapSize // 2
        dataSliceStart = startWritingAt - regionStart
        dataSliceStop = stopWritingAt - regionStart

        # Okay, now it's time to actually do the thing to the data!

        importances = h5Reader.readScore(regionID)
        seq = h5Reader.readSeq(regionID)
        projected = np.array(importances) * np.array(seq)
        # Add up all the bases to get a vector of projected importances.
        profile = np.sum(projected, axis=1)
        vals = [float(x) for x in profile[dataSliceStart:dataSliceStop]]
        outBw.addEntries(regionChrom,
                         int(startWritingAt),
                         values=vals,
                         span=1, step=1)

        # Update the region. By pulling the first setting of the region variables out of the loop,
        # I avoid double-dipping to get those data from the H5.
        startWritingAt = stopWritingAt
        if nextRegion is not None:
            regionID = nextRegion
            regionChrom = nextChrom  # type: ignore
            regionStart = nextStart  # type: ignore
            regionStop = nextStop  # type: ignore

    logUtils.info("Regions written; closing bigwig.")
    outBw.close()
    logUtils.info("Done saving shap scores.")


def getParser() -> argparse.ArgumentParser:
    """Generate the parser

    :return: An ArgumentParser, ready to call parse_args()
    """
    parser = argparse.ArgumentParser(
        description="Take an hdf5-format file generated by the flat shap "
                    "script and render it to a bigwig.")
    parser.add_argument("--h5", help="The name of the hdf5-format file to be read in.")
    parser.add_argument("--bw", help="The name of the bigwig file that should be written.")
    parser.add_argument("--verbose", help="Print progress messages.", action="store_true")
    return parser


def main() -> None:
    """Run the program."""
    args = getParser().parse_args()
    logUtils.setBooleanVerbosity(args.verbose)
    inH5 = h5py.File(args.h5, "r")
    writeBigWig(inH5, args.bw)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
