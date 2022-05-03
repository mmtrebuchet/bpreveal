#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import json
import tensorflow as tf
from tensorflow import keras
import generators
from utils import loadChromSizes
import pybedtools
import numpy as np
import pyBigWig
import pysam
from keras.models import load_model
import tqdm
import losses
#Generate a simple sequence model taking one-hot encoded input and producing a logits profile and a log(counts) scalar. 



def main(configJsonFname):
    with open(configJsonFname, "r") as configFp:
        config = json.load(configFp)
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    batchSize = config["settings"]["batch-size"]
    genomeFname = config["settings"]["genome"]
    numHeads = config["settings"]["architecture"]["heads"] 

    regions = pybedtools.BedTool(config["bed-file"])
    seqs = np.zeros((len(regions), inputLength, 4))
    genome = pysam.FastaFile(genomeFname)
    padding = (inputLength - outputLength) // 2
    chromSizes = loadChromSizes(config["settings"]["chrom-sizes"])


    for i, region in enumerate(regions):
        curSeq = genome.fetch(region.chrom, region.start - padding, region.stop + padding)
        seqs[i] = generators.oneHotEncode(curSeq)
    
    model = load_model(config["settings"]["architecture"]["model-file"], custom_objects = {'multinomialNll' : losses.multinomialNll})
    print(model.summary())
    print(model.outputs)
    preds = model.predict(seqs, batch_size=batchSize, verbose=True)
    
    writePreds(regions, preds, config["settings"]["tracks"], numHeads, chromSizes)

def writePreds(regions, preds, outputTrackList, numHeads, chromSizes):
    """Regions is the BedTool taken from the config's bed file. 
    preds is the output of the model's predict function, no transformations.
    outputTrackList is straight from the json file. 
    numheads is the number of output heads. 
    chromSizes is a dict mapping chromosome names to size. """
    #get the sparse scale array and a list of write blocks. 
    scales, blocks = getScaling(regions, chromSizes)

    for task in outputTrackList:
        headProfile = preds[task["head-id"]]
        #Since the heads are organized as (profile1, profile2, ..., head1, head2...)
        #I add numheads to get to the counts heads. 
        headCounts = preds[task["head-id"] + numHeads]
        #A head may have multiple tasks in it, so slice out the appropriate data. 
        taskProfile = headProfile[:,:,task["task-id"]]
        taskCounts = headCounts[:,task["task-id"]]
        writeBigWig(regions, taskProfile, taskCounts, task, chromSizes, scales, blocks)

def writeBigWig(regions, profileValues, countsValues, task, chromSizes, scales, writeBlocks):
    """Open up the logits and counts bigwigs and write the given values.
    regions is a BedTool, with each entry in profileValues or countsValues corresponding to a region. 
    profileValues is a numRegions x outputWidth array of predicted logits,
    countsValues is a (numRegions,) array of predicted log counts. 
    task is taken from the json, and contains information about the names of the files to be written.
    chromSizes is the usual dictionary mapping chromosome name to size.
    scales and writeBlocks are the sparse array of region overlap counts and the dense regions of that array.
    """
    header = []
    for chromName in chromSizes.keys():
        header.append((chromName, chromSizes[chromName]))
    countsBigWig = pyBigWig.open(task["counts-bw"], 'w')
    countsBigWig.addHeader(header)
    profileBigWig = pyBigWig.open(task["logits-bw"], 'w')
    profileBigWig.addHeader(header)
    print("Writing {0:s}".format(task["logits-bw"]))
    writeScaledBw(regions, profileValues, chromSizes, profileBigWig, scales, writeBlocks)
    print("Writing {0:s}".format(task["counts-bw"]))
    writeScaledBw(regions, countsValues, chromSizes, countsBigWig, scales, writeBlocks)
    profileBigWig.close()
    countsBigWig.close()
    
def getScaling(regions, chromSizes):
    """
    Given a list of regions, build up a sparse array of pileup of those regions. 
    chromSizes is the usual dictionary mapping chromosome name to its size.
    This function returns two things: First, it returns a sparse array indicating, for each base in the genome,
    how many regions overlap that base. 
    This array tells you how to normalize the data you want to write to account for multiple regions predicting the same base.
    The second value is a list of blocks of data that need to be written. 
    The list is formatted as [ [[start_chrom, start_pos], [stop_chrom, stop_pos]], ...]
    Importantly the write blocks are INCLUSIVE, unlike python slices. So if you want to slice 
    based on a block, you'd slice as data[start:stop+1]. 
    """
    from scipy.sparse import lil_array
    chromNameToNum = dict()
    chromNumToName = dict()
    i = 0
    for chromName in chromSizes.keys():
        chromNameToNum[chromName] = i
        chromNumToName[i] = chromName
        i += 1
    

    scales = lil_array((len(chromSizes.keys()), max(chromSizes.values())), dtype=np.int16)
    prevChrom = None
    chromIdx = None
    print("Computing scaling")
    for i in tqdm.tqdm(range(len(regions))):
        r = regions[i]
        if(r.chrom != prevChrom):
            chromIdx = chromNameToNum[r.chrom]
            prevChrom = r.chrom
        scales[[chromIdx], r.start:r.stop] = scales[[chromIdx],r.start:r.stop].toarray() +  1
    #print(scales.nonzero())
    nonzeroPoses = np.array(scales.nonzero()).T
    #Note that writeBlocks are INCLUSIVE, unlike python slices.
    writeBlocks = []
    curStart = nonzeroPoses[0]
    curStop = nonzeroPoses[0]
    print("Generating write blocks.")
    for pos in tqdm.tqdm(nonzeroPoses[1:]):
        if(pos[0] == curStop[0] and pos[1] == curStop[1] + 1):
            #We're extending the current gap.
            curStop = pos
        else:
            writeBlocks.append((curStart, curStop))
            curStart = pos
            curStop = pos
    return scales, writeBlocks

def writeScaledBw(regions, data, chromSizes, outBw, scales, writeBlocks):
    """Regions is a BedTool
    data is the value for each region. In the case of profile, data will be an array, while
    in the case of counts, data will be a scalar. 
    chromSizes is the usual dictionary.
    outBw is a pyBigWig file opened in writing mode.
    scales and writeBlocks are the values returned by getScaling()
    """
    #Build up sparse arrays of occupancy for the genome. This is basically a poor man's bigwig. 
    from scipy.sparse import lil_array
    chromNameToNum = dict()
    chromNumToName = dict()
    i = 0
    for chromName in chromSizes.keys():
        chromNameToNum[chromName] = i
        chromNumToName[i] = chromName
        i += 1
    

    vals = lil_array((len(chromSizes.keys()), max(chromSizes.values())), dtype=np.float32)
    prevChrom = None
    chromIdx = None
    print("Generating sparse data array.")
    for i in tqdm.tqdm(range(len(regions))):
        r = regions[i]
        d = data[i]
        if(r.chrom != prevChrom):
            chromIdx = chromNameToNum[r.chrom]
            prevChrom = r.chrom
        vals[[chromIdx], r.start:r.stop] = vals[[chromIdx], r.start:r.stop].toarray() + d
    #print(scales.nonzero())
    #Note that writeBlocks are INCLUSIVE, unlike python slices.
    print("Writing to bigwig")
    for wb in tqdm.tqdm(writeBlocks):
        chrom = wb[0][0]
        start = wb[0][1]
        stop = wb[1][1]+1
        blockVals = vals[[chrom], start:stop].toarray()
        blockScales = scales[[chrom], start:stop].toarray()
        curVals = blockVals / blockScales
        
        outBw.addEntries(chromNumToName[wb[0][0]], 
                wb[0][1], 
                values=[float(x) for x in curVals[0]],
                span=1,
                step=1)



if (__name__ == "__main__"):
    import sys
    main(sys.argv[1])


