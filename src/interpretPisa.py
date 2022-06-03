#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import utils
utils.setMemoryGrowth()
import json
import pysam
import numpy as np

import shap
from keras.models import load_model
import losses
import generators
import h5py
#Randomly chosen by /dev/urandom
RANDOM_SEED=355687
def shuffleGenerator(numShuffles):
    def generateShuffles(model_inputs):
        rng = np.random.RandomState(RANDOM_SEED)
        #print(model_inputs[0].shape)
        shuffles = [rng.permutation(model_inputs[0]) for x in range(numShuffles)]
        shuffles = np.array(shuffles)
        #print(shuffles.shape)
        return [shuffles]
    return generateShuffles




def main(jsonFname):
    with open(jsonFname, "r") as configFp:
        config = json.load(configFp)
    utils.setVerbosity(config["verbosity"])
    model = load_model(config["model-file"], 
                       custom_objects = {'multinomialNll' : losses.multinomialNll})
    genome = pysam.FastaFile(config["genome"])

    #Build up a list of all the regions that need to be shapped. 
    shapTargets = []
    with open(config["bed-file"]) as fp:
        for line in fp:
            lsp = line.split()
            chrom = lsp[0]
            start = int(lsp[1])
            stop = int(lsp[2])
            for pos in range(start, stop):
                shapTargets.append((chrom, pos))
    #Now build up the array of one-hot encoded sequences.
    oneHotSequences = np.zeros((len(shapTargets), config["input-length"], 4), dtype='float64')
    padding = (config["input-length"] - config["output-length"]) //2
    for i, target in enumerate(shapTargets):
        #I need to generate a one-hot encoded sequence that has the current base on its left-most side. 
        startPos = target[1] - padding
        stopPos = startPos + config["input-length"]
        seq = genome.fetch(target[0], startPos, stopPos)
        oneHot = generators.oneHotEncode(seq)
        oneHotSequences[i] = oneHot
    
    shuffler = shuffleGenerator(config["num-shuffles"])
    shuffles = shuffler(oneHotSequences[:1])
    sar = np.array(shuffles)
    #print(sar.shape)
    #print(np.sum(np.array(shuffles), axis=1))
    #                                      Keep the first dimension so it seems like a batch size of one.v    
    #                                       Leftmost base in output v                                    |
    #                            All of the samples in this batch v |     Sum samples in batch. v        |
    #Oh boy, this slice.                      |--Current head---| V V  |current task----|       V        V
    outputTarget = tf.reduce_sum(model.outputs[config["head-id"]][:,0, config["task-id"]], axis=0, keepdims=True)
    #print(model.outputs[0])
    profileExplainer = shap.TFDeepExplainer( (model.input, outputTarget), 
                                    shuffles)
    profileShapScores = profileExplainer.shap_values([oneHotSequences])

    outputFile = h5py.File(config["output-h5"], 'w')

    outputFile.attrs["head-id"] = config["head-id"]
    outputFile.attrs["task-id"] = config["task-id"]
    
    stringDtype = h5py.string_dtype(encoding='utf-8')
    outputFile.create_dataset('chrom_names', (genome.nreferences,), dtype=stringDtype)
    outputFile.create_dataset('chrom_sizes', (genome.nreferences,), dtype='u4')
    chromNameToIdx = dict()
    for i, chromName in enumerate(genome.references):
        outputFile['chrom_names'][i] = chromName
        chromNameToIdx[chromName] = i
        outputFile['chrom_sizes'][i] = genome.get_reference_length(chromName)
    outputFile.create_dataset('coords_chrom', (len(shapTargets),), dtype='u1')
    outputFile.create_dataset('coords_base', (len(shapTargets),), dtype='u4')
    for i, pos in enumerate(shapTargets):
        outputFile['coords_chrom'][i] = chromNameToIdx[pos[0]]
        outputFile['coords_base'][i] = pos[1]
    outputFile.create_dataset('sequence', data=oneHotSequences[:,:-config['output-length'],:])
    outputFile.create_dataset('shap', data=profileShapScores[:,:-config['output-length'],:])
    outputFile.close()



if (__name__ == "__main__"):
    import sys
    main(sys.argv[1])
