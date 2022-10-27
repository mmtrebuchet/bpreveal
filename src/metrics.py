#!/usr/bin/env python3
import pyBigWig
import numpy as np
import argparse
import logging
import scipy.stats
import scipy.spatial.distance
import sys
import tqdm
from multiprocessing import Process, Queue


class Region:
    def __init__(self, line):
        lsp = line.split()
        self.chrom = lsp[0]
        self.start = int(lsp[1])
        self.stop = int(lsp[2])

class MetricsCalculator:
    def __init__(self, referenceBwFname, predictedBwFname, applyAbs, inQueue, outQueue, tid):
        logging.debug("Starting thread {0:d} for metrics calculation.".format(tid))
        self.referenceBw = pyBigWig.open(referenceBwFname, "r")
        self.predictedBw = pyBigWig.open(predictedBwFname, "r")
        self.applyAbs = applyAbs
        self.inQueue = inQueue
        self.outQueue = outQueue
        self.tid = tid

    def runRegions(self, regionReference, regionPredicted, regionID):
        
        referenceData = np.nan_to_num(self.referenceBw.values(regionReference.chrom, regionReference.start, regionReference.stop))
        if(self.applyAbs):
            referenceData = np.abs(referenceData)
        predictedData = np.nan_to_num(self.predictedBw.values(regionPredicted.chrom, regionPredicted.start, regionPredicted.stop))
        if(self.applyAbs):
            predictedData = np.abs(predictedData)
        if(np.sum(referenceData) > 0 and np.sum(predictedData) > 0): 
            referencePmf = referenceData / np.sum(referenceData)
            predictedPmf = predictedData / np.sum(predictedData)
            mnll = scipy.stats.multinomial(np.sum(referenceData), predictedPmf)
            mnllVal = mnll.logpmf(referenceData)
    
            jsd = scipy.spatial.distance.jensenshannon(referencePmf, predictedPmf)

            pearsonr = scipy.stats.pearsonr(referenceData, predictedData)[0]
        
            spearmanr = scipy.stats.spearmanr(referenceData, predictedData)[0]
        else:
            mnllVal = jsd = pearsonr = spearmanr = np.nan

        referenceCounts = np.sum(referenceData)
        predictedCounts = np.sum(predictedData)
        ret = (regionID, mnllVal, jsd, pearsonr, spearmanr, referenceCounts, predictedCounts)
        self.outQueue.put(ret)

    def run(self):
        while(True):
            match self.inQueue.get():
                case None:
                    logging.debug("Finishing calculator thread {0:d}.".format(self.tid))
                    self.finish()
                    return
                case (regionRef, regionPred, regionID):
                    self.runRegions(regionRef, regionPred, regionID)
    def finish(self):
        self.referenceBw.close()
        self.predictedBw.close()

def calculatorThread(referenceBwFname, predictedBwFname, applyAbs, inQueue, outQueue, tid):
    calc = MetricsCalculator(referenceBwFname, predictedBwFname, applyAbs, inQueue, outQueue, tid)
    calc.run()

def regionGenThread(regionsFname, regionQueue, numThreads, numberQueue):
    logging.debug("Initializing generator.")
    regions = [Region(x) for x in open(regionsFname, 'r')]
    logging.info("Number of regions: {0:d}".format(len(regions)))
    numberQueue.put(len(regions))
    for i, r in enumerate(regions): 
        regionQueue.put((r,r,i))
    for i in range(numThreads):
        regionQueue.put(None)
    logging.debug("Generator done.")
    



def percentileStats(name, vector, header=False):
    vector = vector[np.logical_not(np.isnan(vector))]
    quantileCutoffs = np.linspace(0, 1, 5)
    quantiles = np.quantile(vector, quantileCutoffs)
    if(header):
        print("{0:10s}".format("metric")
                + "".join(["\t{0:14f}%".format(x*100) for x in quantileCutoffs])
                + "\t{0:s}".format("regions"))
    print("{0:10s}".format(name) 
        + "".join(["\t{0:15f}".format(x) for x in quantiles])
        + "\t{0:d}".format(vector.shape[0]))

def receiveThread(numRegions, outputQueue):
    mnlls = np.zeros((numRegions,))
    jsds = np.zeros((numRegions,))
    pearsonrs = np.zeros((numRegions,))
    spearmanrs = np.zeros((numRegions,))
    referenceCounts = np.zeros((numRegions,))
    predictedCounts = np.zeros((numRegions,))

    for i in tqdm.tqdm(range(numRegions)):
        ret = outputQueue.get()
        (regionID, mnllVal, jsd, pearsonr, spearmanr, referenceCount, predictedCount) = ret
        mnlls[regionID] = mnllVal
        jsds[regionID] = jsd
        pearsonrs[regionID] = pearsonr
        spearmanrs[regionID] = spearmanr
        referenceCounts[regionID] = referenceCount
        predictedCounts[regionID] = predictedCount

    #Calculate the percentiles for each of the profile metrics. 
    percentileStats("mnll", mnlls, header=True)
    percentileStats("jsd", jsds)
    percentileStats("pearsonr", pearsonrs)
    percentileStats("spearmanr", spearmanrs)

    countsPearson = scipy.stats.pearsonr(referenceCounts, predictedCounts)
    countsSpearman = scipy.stats.spearmanr(referenceCounts, predictedCounts)
    print("Counts pearson \t{0:10f}".format(countsPearson[0]))
    print("Counts spearman\t{0:10f}".format(countsSpearman[0]))


def runMetrics(reference, predicted, regions, threads, applyAbs):
    regionQueue = Queue()
    resultQueue = Queue()
    numberQueue=Queue()
    print("reference {0:s} predicted {1:s} regions {2:s}".format(reference, predicted, regions))
    regionThread = Process(target=regionGenThread, args=(regions, regionQueue, threads, numberQueue))
    processorThreads = []
    for i in range(threads):
        logging.debug("Creating thread {0:d}".format(i))
        newThread = Process(target=calculatorThread, args=(reference, predicted, applyAbs, regionQueue, resultQueue, i))
        processorThreads.append(newThread)
        newThread.start()
    regionThread.start()
    #Since the number of regions is not known before starting up the regions thread, I have to 
    #start the regions thread and then get a message telling me how many regions there are. 
    #Clunky, but avoids re-reading the bed file.
    numRegions = numberQueue.get()
    
    writerThread = Process(target=receiveThread, args=(numRegions, resultQueue))

    writerThread.start()
    writerThread.join()
    


def main():
    parser = argparse.ArgumentParser(description='Take two bigwig-format files and calculate an assortment of metrics on their contents.')
    parser.add_argument("--reference", help='The name of the reference (i.e., experimental) bigwig.')
    parser.add_argument("--predicted", help='The name of the bigwig file generated by a model.')
    parser.add_argument("--regions", help="The name of a bed file containing the regions to use to calculate the metrics.")
    parser.add_argument("--verbose", help="Display progress as the file is being written.", action='store_true')
    parser.add_argument("--threads", help="Number of parallel threads to use. Default 1.", default=1, type=int)
    parser.add_argument("--apply-abs", help="Use the absolute value of the entries in the bigwig. Useful if one bigwig contains negative values.", action='store_true', dest='applyAbs')
    args = parser.parse_args()
    if(args.verbose):
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    runMetrics(args.reference, args.predicted, args.regions, args.threads, args.applyAbs)


if(__name__ == "__main__"):
    main()


