import random
import pysam
import pybedtools
import pyBigWig
from tensorflow import keras
import numpy as np
import h5py
import math
import logging
import time
import tqdm

"""
class Region:
    def __init__(self, bedLine, bigwigFileList, genome, inputLength, outputLength, maxJitter):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.chrom = bedLine.chrom
        self.outputStart = bedLine.start
        self.outputStop = bedLine.stop
        inputPadding = (inputLength - outputLength) // 2
        self.inputStart =  self.outputStart - inputPadding
        self.inputStop = self.outputStop + inputPadding
        #We need to add some extra padding to account for jitter. 
        self.maxJitter = maxJitter

        self.sequence = oneHotEncode(genome.fetch(self.chrom, self.inputStart - maxJitter, self.inputStop + maxJitter))
        bigwigProfiles = []
        for bigwigFile in bigwigFileList:
            bigwigProfiles.append(np.nan_to_num(bigwigFile.values(self.chrom, 
                                                             self.outputStart - maxJitter,
                                                             self.outputStop + maxJitter)))
        self.values = np.array(bigwigProfiles)
    
    def get(self):
        jitter = random.randrange(0, 2*self.maxJitter)
        #No transpose. Sequence is by position, then by base.
        seq = self.sequence[jitter : jitter + self.inputLength, :]
        #Here's the transpose. Values are by position, then by task. 
        vals = self.values[:,jitter:jitter+self.outputLength].T
        #If jittering is allowed, also randomly reverse complement the data.
        if(self.maxJitter > 0):
            if(random.randint(0,1)):
                vals = np.flip(vals, axis=0)
                seq = np.flip(seq, axis=0)
        return (seq, vals, np.sum(vals, axis=0))
        

class OneBedRegions:
    def __init__(self, individualBed, bigwigFileList, genome, inputLength, outputLength):
        self.regions = []
        bt = pybedtools.BedTool(individualBed["bed-file"])
        if(logging.getLogger().isEnabledFor(logging.INFO)):
            bt = tqdm.tqdm(bt)

        for line in bt:
            self.regions.append(Region(line, bigwigFileList, 
                genome, inputLength, outputLength, individualBed["max-jitter"]))
        self.numSamples = int(len(self.regions) * individualBed["absolute-sampling-weight"])

    def get(self):
        regionIds = random.sample(range(len(self.regions)), self.numSamples)
        return [self.regions[i].get() for i in regionIds]

class BedBatchGenerator(keras.utils.Sequence):
    "This class loads in all of the data when initialized, and then generates samples for training and evaluating the model."

    def __init__(self, individualBedList, headList, genomeFname, inputLength, outputLength, batchSize):
        "Load in all of the relevant data, and load enough that we can crop the data for jittering it.
        individualBedList is straight from the JSON, 
        headList is straight from the JSON,
        genomeFname is a string naming the fasta-format file with the appropriate genome,
        inputLength is the width of input sequences for the network, 
        outputLength is the size of the network's output, 
        maxJitter is the maximum tolerated shift of the frame for each region. 
        Unless you're doing something really weird, maxJitter should be 0 for predicting. 
        "
        logging.info("Loading data for batch generator.")
        
        bigwigList = []
        self.headIndexes = []
        curBwIdx = 0
        for head in headList:
            bigwigList.extend(head["bigwig-files"])
            self.headIndexes.append(range(curBwIdx, curBwIdx + len(head["bigwig-files"])))
            curBwIdx += len(head["bigwig-files"])
        self.bedsWithRegions = []
        genome = pysam.FastaFile(genomeFname)
        self.bigwigList = bigwigList
        bigwigFiles = []
        self.length = 0
        self.batchSize = batchSize
        for individualBw in self.bigwigList:
            bigwigFiles.append(pyBigWig.open(individualBw, "r"))
        startTime = time.perf_counter() 
        for individualBed in individualBedList:
            logging.info("Loading bed file {0:s}".format(str(individualBed)))
            newBed = OneBedRegions(individualBed, bigwigFiles, genome, inputLength, outputLength)
            logging.debug("oneBedRegions created.")
            self.bedsWithRegions.append(newBed)
            self.length += newBed.numSamples
        stopTime = time.perf_counter()
        logging.debug("Loaded regions in {0:f} seconds".format(stopTime - startTime))

        for bwf in bigwigFiles:
            bwf.close()
        
        self.sequences = np.empty((self.length, inputLength, 4))
        self.values = np.empty((self.length, outputLength, len(bigwigList)))
        self.counts = np.empty((self.length, len(bigwigList)))
        logging.debug("Storage allocated.")
        self.loadData()
        logging.info("Batch generator initalized.")

    def __len__(self):
        return math.ceil(self.length / self.batchSize)

    def __getitem__(self, idx):
        seqs = self.sequences[idx*self.batchSize : (idx + 1) * self.batchSize]
        vals = self.values[idx*self.batchSize : (idx+1) * self.batchSize]
        counts = self.counts[idx*self.batchSize : (idx+1) * self.batchSize]
        retVals = []
        retCounts = []
        for h in self.headIndexes:
            retVals.append(vals[:,:,h])
            curHeadCounts = counts[:,h]
            retCounts.append(np.log(np.sum(curHeadCounts, axis=1)))
        return (seqs, (retVals + retCounts))


    def refreshData(self):
        startTime = time.perf_counter()
        i = 0
        for bwr in self.bedsWithRegions:
            for region in bwr.get():
                self.sequences[i] = region[0]
                self.values[i] = region[1]
                self.counts[i] = region[2]
                i += 1
        assert i == self.length
        stopTime = time.perf_counter()
        Δt = stopTime - startTime
        logging.debug("Loaded new batch in {0:5f} seconds.".format(Δt))

    def on_epoch_end(self):
        self.refreshData()
"""
class H5BatchGenerator(keras.utils.Sequence):

    #def __init__(self, individualBedList, headList, genomeFname, inputLength, outputLength, batchSize):
    def __init__(self, headList, dataH5, inputLength, outputLength, maxJitter, batchSize):
        logging.info("Initial load of dataset for hdf5-based generator.")
        self.headList = headList
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.maxJitter = maxJitter
        self.batchSize = batchSize
        #The shape of the sequence dataset is 
        #(numRegions x (inputLength + jitter*2 x 4))
        self.fullSequences = np.array(dataH5["sequence"])
        print(self.fullSequences.shape)
        self.numRegions = self.fullSequences.shape[0]
        #The shape of the profile is 
        #(num-heads) x (numRegions x (outputLength + jitter*2) x numTasks)
        #Similar to the prediction script outputs, the heads are all separate, and are named "head_N", where
        #N is 0,1,2, etc. 
        self.fullData = []
        for i, h in enumerate(headList):
            self.fullData.append(np.array(dataH5["head_{0:d}".format(i)]))
        print(np.sum(self.fullData[0]))
        self.loadData()

    def __len__(self):
        return math.ceil(self.numRegions / self.batchSize)

    def __getitem__(self,idx):
        return self.batchSequences[idx], (self.batchVals[idx] + self.batchCounts[idx])

    def loadData(self):
        self.batchSequences = []
        self.batchVals = []
        self.batchCounts = []
        regionsRemaining = self.numRegions
        for i in range(len(self)):
            #Build an empty sequence array. Note we have to special-case the last round 
            #in case the number of regions is not divisible by batch size. 
            if(regionsRemaining > self.batchSize):
                curBatchSize = self.batchSize
            else:
                curBatchSize = regionsRemaining
            regionsRemaining -= self.batchSize
            self.batchSequences.append(np.empty((curBatchSize, self.inputLength, 4)))
            newBatchVals = []
            newBatchCounts = []
            for head in self.headList:
                newBatchVals.append(np.empty((curBatchSize, self.outputLength, head["num-tasks"])))
                newBatchCounts.append(np.empty((curBatchSize,)))
            self.batchVals.append(newBatchVals)
            self.batchCounts.append(newBatchCounts)
        self.regionIndexes = np.arange(0, self.numRegions)
        self.rng = np.random.default_rng(seed=1234)
        self.refreshData()

    def refreshData(self):
        #Go over all the data and load it into the data structures allocated in loadData.
        #First, randomize which regions go into which batches. 
        logging.debug("Refreshing batch data.")
        startTime = time.perf_counter()
        #self.rng.shuffle(self.regionIndexes)
        for i in range(self.numRegions):
            tmpSequence = self.fullSequences[i]
            #fullData is (num-heads) x (num-regions x output-width+jitter*2 x numTasks)
            #so this slice takes the ith region of each of the head datasets. 
            tmpData = [x[i,:,:] for x in self.fullData]
            if(self.maxJitter > 0):
                jitterOffset = self.rng.integers(0, self.maxJitter * 2+1)
                tmpSequence = tmpSequence[jitterOffset: jitterOffset + self.inputLength, :]
                for j in range(len(tmpData)):
                    tmpData[j] = tmpData[j][jitterOffset:jitterOffset+self.outputLength,:]
                #Note that this generator does *not* revcomp the data, in case the input are stranded like chip-nexus.
            #We've collected and trimmed the data, now to fill in the batch arrays. 
            regionIdx = self.regionIndexes[i]
            batchIdx = regionIdx // self.batchSize
            batchRegionIdx = regionIdx % self.batchSize
            #print(i, regionIdx, batchIdx, batchRegionIdx)
            batchSeqs = self.batchSequences[batchIdx]
            batchVals = self.batchVals[batchIdx]
            batchCounts = self.batchCounts[batchIdx]
            batchSeqs[batchRegionIdx,:] = tmpSequence
            for headIdx, head in enumerate(self.headList):
                batchVals[headIdx][batchRegionIdx,:] = tmpData[headIdx]
                batchCounts[headIdx][batchRegionIdx] = np.log(np.sum(tmpData[headIdx]))
        stopTime = time.perf_counter()
        Δt = stopTime - startTime
        logging.debug("Loaded new batch in {0:5f} seconds.".format(Δt))

                
    def on_epoch_end(self):
        self.refreshData()

                



    
