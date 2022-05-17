import random
import pysam
import pybedtools
import pyBigWig
from tensorflow import keras
import numpy as np
import math
import logging

def oneHotEncode(sequence):
    ret = np.empty((len(sequence),4), dtype='int8')
    ordSeq = np.fromstring(sequence, np.int8)
    ret[:,0] = (ordSeq == ord("A")) + (ordSeq == ord('a'))
    ret[:,1] = (ordSeq == ord("C")) + (ordSeq == ord('c'))
    ret[:,2] = (ordSeq == ord("G")) + (ordSeq == ord('g'))
    ret[:,3] = (ordSeq == ord("T")) + (ordSeq == ord('t'))
    assert (np.sum(ret) == len(sequence)), (sorted(sequence), sorted(ret.sum(axis=1)))
    return ret

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
        return (seq, vals, np.log(np.sum(vals, axis=0)))
        

class OneBedRegions:
    def __init__(self, individualBed, bigwigFileList, genome, inputLength, outputLength):
        self.regions = []
        for line in pybedtools.BedTool(individualBed["bed-file"]):
            self.regions.append(Region(line, bigwigFileList, 
                genome, inputLength, outputLength, individualBed["max-jitter"]))
        self.numSamples = int(len(self.regions) * individualBed["absolute-sampling-weight"])

    def get(self):
        regionIds = random.sample(range(len(self.regions)), self.numSamples)
        return [self.regions[i].get() for i in regionIds]

class BatchGenerator(keras.utils.Sequence):
    """This class loads in all of the data when initialized, and then generates samples for training and evaluating the model."""

    def __init__(self, individualBedList, headList, genomeFname, inputLength, outputLength, batchSize):
        """Load in all of the relevant data, and load enough that we can crop the data for jittering it.
        individualBedList is straight from the JSON, 
        headList is straight from the JSON,
        genomeFname is a string naming the fasta-format file with the appropriate genome,
        inputLength is the width of input sequences for the network, 
        outputLength is the size of the network's output, 
        maxJitter is the maximum tolerated shift of the frame for each region. 
        Unless you're doing something really weird, maxJitter should be 0 for predicting. 
        """
        logging.info("Loading data for batch generator.")
        
        bigwigList = []
        self.headIndexes = []
        curBwIdx = 0
        for head in headList:
            bigwigList.extend(head["data"])
            self.headIndexes.append(range(curBwIdx, curBwIdx + len(head["data"])))

        self.bedsWithRegions = []
        genome = pysam.FastaFile(genomeFname)
        self.bigwigList = bigwigList
        bigwigFiles = []
        self.length = 0
        self.batchSize = batchSize
        for individualBw in self.bigwigList:
            bigwigFiles.append(pyBigWig.open(individualBw, "r"))
        
        for individualBed in individualBedList:
            newBed = OneBedRegions(individualBed, bigwigFiles, genome, inputLength, outputLength)
            self.bedsWithRegions.append(newBed)
            self.length += newBed.numSamples

        for bwf in bigwigFiles:
            bwf.close()
        
        self.sequences = np.empty((self.length, inputLength, 4))
        self.values = np.empty((self.length, outputLength, len(bigwigList)))
        self.logcounts = np.empty((self.length, len(bigwigList)))
        self.loadData()
        logging.info("Batch generator initalized.")

    def __len__(self):
        return math.ceil(self.length / self.batchSize)

    def __getitem__(self, idx):
        seqs = self.sequences[idx*self.batchSize : (idx + 1) * self.batchSize]
        vals = self.values[idx*self.batchSize : (idx+1) * self.batchSize]
        logcounts = self.logcounts[idx*self.batchSize : (idx+1) * self.batchSize]
        retVals = []
        retCounts = []
        for h in self.headIndexes:
            retVals.append(vals[:,:,h])
            retCounts.append(logcounts[:,h])
        return (seqs, (retVals + retCounts))


    def loadData(self):
        i = 0
        for bwr in self.bedsWithRegions:
            for region in bwr.get():
                self.sequences[i] = region[0]
                self.values[i] = region[1]
                self.logcounts[i] = region[2]
                i += 1
        assert i == self.length

    def on_epoch_end(self):
        self.loadData()


