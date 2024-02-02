#!/usr/bin/env python3
import h5py
import pyBigWig
import numpy as np
import argparse
from bpreveal import logUtils
import tqdm
from bpreveal.utils import H5_CHUNK_SIZE


class BatchedH5Reader:
    curChunkStart: int
    curChunkEnd: int

    def __init__(self, h5fp, batchSize):
        self.batchSize = batchSize
        self.h5fp = h5fp
        self.maxIndex = self.h5fp["hyp_scores"].shape[0]
        self.curChunkStart = -100000
        self.curChunkEnd = -100000
        self.loadChunk(0)

    def loadChunk(self, index: int):
        if self.curChunkStart <= index < self.curChunkEnd:
            return
        self.curChunkStart = index
        self.curChunkEnd = min(index + self.batchSize, self.maxIndex)
        self.curScore = np.array(self.h5fp["hyp_scores"]
                                 [self.curChunkStart:self.curChunkEnd, :, :])
        self.curSeqs = np.array(self.h5fp["input_seqs"]
                                [self.curChunkStart:self.curChunkEnd, :, :])

    def readScore(self, idx: int):
        self.loadChunk(idx)
        return self.curScore[idx - self.curChunkStart, :, :]

    def readSeq(self, idx: int):
        self.loadChunk(idx)
        return self.curSeqs[idx - self.curChunkStart, :, :]


def writeBigWig(inH5, outFname, verbose):
    bwHeader = []
    h5Reader = BatchedH5Reader(inH5, H5_CHUNK_SIZE)
    chromIdxToName = dict()
    for i, name in enumerate(inH5["chrom_names"].asstr()):
        bwHeader.append(("{0:s}".format(name), int(inH5['chrom_sizes'][i])))
        chromIdxToName[i] = name
    outBw = pyBigWig.open(outFname, 'w')
    outBw.addHeader(sorted(bwHeader))
    logUtils.debug("Bigwig header" + str((sorted(bwHeader))))
    numRegions = inH5['coords_chrom'].shape[0]
    if type(inH5['coords_chrom'][0]) is bytes:
        logUtils.warning("You are using an old-style hdf5 file for importance scores. "
            "Support for these files will be removed in BPReveal 5.0. "
            "Instructions for updating: Re-calculate importance scores.")
        coordsChrom = np.array(inH5['coords_chrom'].asstr())
    else:
        coordsChromIdxes = np.array(inH5['coords_chrom'])
        coordsChrom = np.array([chromIdxToName[x] for x in coordsChromIdxes])
    # Sort the regions.
    coordsStart = np.array(inH5['coords_start'])
    coordsEnd   = np.array(inH5['coords_end'])  # noqa
    regionOrder = sorted(range(numRegions), key=lambda x: (coordsChrom[x], coordsStart[x]))
    startWritingAt = 0
    regionID = regionOrder[0]
    regionChrom = coordsChrom[regionID]
    curChrom = regionChrom
    regionStart = coordsStart[regionID]
    regionStop = coordsEnd[regionID]
    logUtils.info("Files opened; writing regions")
    regionRange = range(numRegions)
    nextRegion = None
    if verbose:
        regionRange = tqdm.tqdm(regionRange)
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
    parser = argparse.ArgumentParser(
        description="Take an hdf5-format file generated by the flat shap "
                    "script and render it to a bigwig.")
    parser.add_argument("--h5", help='The name of the hdf5-format file to be read in.')
    parser.add_argument("--bw", help='The name of the bigwig file that should be written.')
    parser.add_argument("--verbose", help="Print progress messages.", action='store_true')
    return parser


def main():

    args = getParser().parse_args()
    if args.verbose:
        logUtils.setVerbosity("INFO")
    else:
        logUtils.setVerbosity("WARNING")
    inH5 = h5py.File(args.h5, 'r')
    writeBigWig(inH5, args.bw, args.verbose)


if __name__ == "__main__":
    main()
