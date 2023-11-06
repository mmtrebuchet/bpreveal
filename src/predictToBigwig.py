#!/usr/bin/env python3
import h5py
import pyBigWig
import numpy as np
import argparse
import logging
import tqdm
import multiprocessing
import bpreveal.utils as utils


class Region:
    def __init__(self, chromIdx, start, end, h5Idx):
        self.chromIdx = chromIdx
        self.start = start
        self.end = end
        self.h5Idx = h5Idx

    def getValues(self, h5fp, mode, head, taskId):
        headLogits = h5fp["head_{0:d}".format(head)]["logits"]
        headLogcounts = h5fp["head_{0:d}".format(head)]["logcounts"]

        match mode:
            case 'profile':
                logits = headLogits[self.h5Idx]
                # Logits will have shape (output-width x numTasks)
                logCounts = headLogcounts[self.h5Idx]
                headProfile = utils.logitsToProfile(logits, logCounts)
                taskProfile = headProfile[:, taskId]
                profile = taskProfile
            case 'logits':
                profile = headLogits[self.h5Idx, :, taskId]
            case 'mnlogits':
                profile = headLogits[self.h5Idx, :, taskId]
                profile -= np.mean(profile)
            case 'logcounts':
                profile = np.zeros((self.end - self.start)) + headLogcounts[self.h5Idx]
            case 'counts':
                profile = np.zeros((self.end - self.start)) + np.exp(headLogcounts[self.h5Idx])
        return profile


def getChromInserts(arg):
    """Packs all the arguments into one so it's easier to use with pool.map()."""
    regionList, h5Fname, headId, taskId, mode = arg
    with h5py.File(h5Fname, "r") as h5fp:
        vec = getChromVector(regionList, h5fp, headId, taskId, mode)
        inserts = vectorToListOfInserts(vec)
    logging.info("Finished region {0:s}".format(str(regionList[0].chromIdx)))
    return inserts


def getChromVector(regionList, h5fp, headId, taskId, mode):
    chromSize = h5fp["chrom_sizes"][regionList[0].chromIdx]
    regionCounts = np.zeros((chromSize,), dtype=np.uint16)
    regionValues = np.zeros((chromSize,), dtype=np.float32)
    for r in regionList:
        regionCounts[r.start:r.end] += 1
        regionValues[r.start:r.end] += r.getValues(h5fp, mode, headId, taskId)
    regionCounts[regionCounts == 0] = 1
    return regionValues / regionCounts


def vectorToListOfInserts(dataVector):
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
            rets.append((dataVector[regionStart:lastPoint + 1], regionStart))
            regionStart = pos

        lastPoint = pos
    rets.append((dataVector[regionStart:poses[-1] + 1], regionStart))
    return rets


def buildRegionList(inH5):
    numRegions = inH5['coords_chrom'].shape[0]

    # Sort the regions.
    logging.debug("Loading coordinate data")
    coordsChrom = list(inH5['coords_chrom'])
    coordsStart = np.array(inH5['coords_start'])
    coordsStop = np.array(inH5['coords_stop'])
    logging.debug("Region data loaded. Sorting.")
    regionList = []
    regionRange = range(numRegions)
    for regionNumber in regionRange:
        regionList.append(Region(coordsChrom[regionNumber],
                                 coordsStart[regionNumber],
                                 coordsStop[regionNumber],
                                 regionNumber))
    logging.debug("Generated list of regions to sort.")
    regionOrder = sorted(range(numRegions),
                         key=lambda x: (regionList[x].chromIdx, regionList[x].start))
    logging.info("Region order calculated.")

    regionsByChrom = dict()
    for idx in regionOrder:
        r = regionList[idx]
        if r.chromIdx not in regionsByChrom:
            regionsByChrom[r.chromIdx] = []
        regionsByChrom[r.chromIdx].append(r)
    return regionsByChrom


def writeBigWig(inH5Fname, outFname, headId, taskId, mode, verbose, negate, numThreads):
    inH5 = h5py.File(inH5Fname, "r")
    logging.info("Starting to write {0:s}, head {1:d} task {2:d}".format(outFname, headId, taskId))
    bwHeader = []
    for i, name in enumerate(inH5["chrom_names"].asstr()):
        bwHeader.append(("{0:s}".format(name), int(inH5['chrom_sizes'][i])))
    outBw = pyBigWig.open(outFname, 'w')

    outBw.addHeader(bwHeader)
    logging.debug(bwHeader)
    logging.info("Added header.")
    regionsByChrom = buildRegionList(inH5)
    chromList = sorted(regionsByChrom.keys())
    # In order to use multiprocessing, I need to unstaple the dict into a list.
    # The order of the list is sorted(regionsByChrom.keys())
    chromRegionLists = []
    for chromIdx in chromList:
        chromRegionLists.append((regionsByChrom[chromIdx], inH5Fname, headId, taskId, mode))
    logging.info("Extracted list of regions to process.")
    logging.info("Beginning to extract profile data.")

    with multiprocessing.Pool(numThreads) as p:
        # Get the insert list for each chromosome in a subprocess.
        chromInsertLists = p.map(getChromInserts, chromRegionLists)
    logging.info("Insert lists calculated. Writing bigwig.")
    pbar = None
    if verbose:
        totalInserts = 0
        for il in chromInsertLists:
            totalInserts += len(il)
        pbar = tqdm.tqdm(total=totalInserts)
    for i, chromIdx in enumerate(chromList):
        inserts = chromInsertLists[i]
        chromName = inH5["chrom_names"][chromIdx].decode('utf-8')
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
    logging.info("Bigwig written. Closing.")
    outBw.close()
    logging.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Take an hdf5-format file generated by "
                                                 "the predict script and render it to a bigwig.")
    parser.add_argument("--h5", help='The name of the hdf5-format file to be read in.')
    parser.add_argument("--bw", help='The name of the bigwig file that should be written.')
    parser.add_argument("--head-id", help="Which head number do you want data for?",
                        dest='headId', type=int)
    parser.add_argument("--task-id", help='Which task in that head do you want?',
                        dest='taskId', type=int)
    parser.add_argument("--mode",
                        help='What do you want written? Options are "profile", meaning you '
                        'want (softmax(logits) * exp(logcounts)), or "logits", meaning you '
                        'just want logits, or "mnlogits", meaning you want the logits, but '
                        'mean-normalized (for easier display), or "logcounts", meaning you '
                        'want the log counts for every region, or "counts", meaning you want '
                        'exp(logcounts). You will usually want "profile"')
    parser.add_argument("--verbose", help="Display progress as the file is being written.",
                        action='store_true')
    parser.add_argument("--negate", help="Negate all of the values written to the bigwig. "
                        "Used for negative-strand predictions.", action='store_true')
    parser.add_argument("--threads", help="Number of threads to use for calculating profile "
                        "tracks (max: num chromosomes)", type=int, default=24, dest='numThreads')
    args = parser.parse_args()
    if (args.verbose):
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    writeBigWig(args.h5, args.bw, args.headId, args.taskId, args.mode, args.verbose,
                args.negate, args.numThreads)


if (__name__ == "__main__"):
    main()
