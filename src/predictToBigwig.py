#!/usr/bin/env python3
import h5py
import pyBigWig
import scipy.special
import numpy as np
import argparse
import logging
import tqdm
import utils
def writeBigWig(inH5Fname, outFname, headId, taskId, mode, verbose, negate):
    inH5 = h5py.File(inH5Fname, "r")
    logging.info("Starting to write {0:s}, head {1:d} task {2:d}".format(outFname, headId, taskId))
    bwHeader = []
    nameToNum = dict()
    for i, name in enumerate(inH5["chrom_names"].asstr()):
        bwHeader.append(("{0:s}".format(name), int(inH5['chrom_sizes'][i])))
    outBw = pyBigWig.open(outFname, 'w')

    outBw.addHeader(bwHeader)
    logging.debug(bwHeader)
    logging.info("Added header.")
    numRegions = inH5['coords_chrom'].shape[0]

    #Sort the regions. 
    logging.debug("Loading coordinate data")
    coordsChrom = list(inH5['coords_chrom'])
    coordsStart = np.array(inH5['coords_start'])
    coordsStop = np.array(inH5['coords_stop'])
    logging.debug("Region data loaded. Sorting.")
    regionList = []
    if(verbose):
        regionRange = tqdm.tqdm(range(numRegions))
    else:
        regionRange = range(numRegions)
    for regionNumber in regionRange:
        regionList.append((coordsChrom[regionNumber], coordsStart[regionNumber]))
    logging.debug("Generated list of regions to sort.")
    regionOrder = sorted(range(numRegions), key = lambda x: regionList[x])
    logging.info("Region order calculated.")
    curChrom = 0
    startWritingAt = 0
    regionID = regionOrder[0]
    regionChrom = coordsChrom[regionID]
    regionStart = coordsStart[regionID]
    regionStop = coordsStop[regionID]
    logging.info("Loading head data.")
    head = inH5['head_{0:d}'.format(headId)]
    headLogits = np.array(head['logits'])
    headLogcounts = np.array(head['logcounts'])

    logging.info("Starting to write data.")
    if(verbose):
        regionRange = tqdm.tqdm(range(numRegions))
    else:
        regionRange = range(numRegions)
    for regionNumber in regionRange:
        #Extract the appropriate region from the sorted list. 

        if(regionChrom != curChrom):
            curChrom = regionChrom
            startWritingAt = 0

        if(startWritingAt < regionStart):
            #The next region starts beyond the end 
            #of the previous one. Some bases will not be filled in.
            startWritingAt = regionStart
        #By default, write the whole region. 
        stopWritingAt = regionStop
        #As long as we aren't on the last region, check for overlaps.
        if(regionNumber < numRegions-1):
            nextRegion = regionOrder[regionNumber+1]
            nextChrom = coordsChrom[nextRegion]
            nextStart = coordsStart[nextRegion]
            nextStop = coordsStop[nextRegion]
            if(nextChrom == regionChrom and nextStart < stopWritingAt):
                #The next region overlaps. So stop writing before then. 
                overlapSize = regionStop - nextStart
                stopWritingAt = stopWritingAt - overlapSize //2
        dataSliceStart = startWritingAt - regionStart
        dataSliceStop = stopWritingAt - regionStart
        
        #Okay, now it's time to actually do the thing to the data! 
        match mode:
            case 'profile':
                logits = headLogits[regionID]
                #Logits will have shape (output-width x numTasks)
                logCounts = headLogcounts[regionID]
                headProfile = utils.logitsToProfile(logits, logCounts)
                taskProfile = headProfile[:,taskId]
                profile = taskProfile
            case 'logits':
                profile = headLogits[regionID, :, taskId]
            case 'mnlogits':
                profile = headLogits[regionID, :, taskId]
                profile -= np.mean(profile)
            case 'logcounts':
                profile = np.zeros((regionStop - regionStart)) + headLogcounts[regionID]
            case 'counts':
                profile = np.zeros((regionStop - regionStart)) + np.exp(headLogcounts[regionID])
        

        if(negate):
            profile *= -1
        vals = [float(x) for x in profile[dataSliceStart:dataSliceStop]]
        try:
            if(startWritingAt == 0):
                raise RuntimeError()
            outBw.addEntries(bwHeader[regionChrom][0], 
                    int(startWritingAt),
                    values = vals,
                    span=1, step=1)
        except RuntimeError as e:
            print(e)
            print(startWritingAt)
            print(bwHeader[regionChrom][0])
            print(regionID)
            print(nextRegion)
            print(regionStart)
            print(nextStart)
            print(regionStop)
            print(nextStop)
            print(regionChrom)
            print(nextChrom)
            raise

        #Update the region. By pulling the first setting of the region variables out of the loop,
        #I avoid double-dipping to get those data from the H5. 
        startWritingAt = stopWritingAt
        regionID = nextRegion
        regionChrom = nextChrom
        regionStart = nextStart
        regionStop = nextStop
    logging.debug("Closing bigwig.")
    outBw.close()
    logging.info("Bigwig closed.")



def main():
    parser = argparse.ArgumentParser(description='Take an hdf5-format file generated by the predict script and render it to a bigwig.')
    parser.add_argument("--h5", help='The name of the hdf5-format file to be read in.')
    parser.add_argument("--bw", help='The name of the bigwig file that should be written.')
    parser.add_argument("--head-id", help="Which head number do you want data for?", dest='headId', type=int)
    parser.add_argument("--task-id", help='Which task in that head do you want?', dest='taskId', type=int)
    parser.add_argument("--mode", help='What do you want written? Options are "profile", meaning you want (softmax(logits) * exp(logcounts)), or "logits", meaning you just want logits, or "mnlogits", meaning you want the logits, but mean-normalized (for easier display), or "logcounts", meaning you want the log counts for every region, or "counts", meaning you want exp(logcounts). You will usually want "profile"')
    parser.add_argument("--verbose", help="Display progress as the file is being written.", action='store_true')
    parser.add_argument("--negate", help="Negate all of the values written to the bigwig. Used for negative-strand predictions.", action='store_true')
    args = parser.parse_args()
    if(args.verbose):
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    writeBigWig(args.h5, args.bw, args.headId, args.taskId, args.mode, args.verbose, args.negate)

if(__name__ == "__main__"):
    main()


