#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import utils
import json
import pysam
import numpy as np
import tqdm
import shap
from keras.models import load_model
import losses
import h5py
import logging
#Randomly chosen by /dev/urandom
RANDOM_SEED=355687
def shuffleGenerator(numShuffles):
    def generateShuffles(model_inputs):
        rng = np.random.RandomState(RANDOM_SEED)
        shuffles = [rng.permutation(model_inputs[0]) for x in range(numShuffles)]
        shuffles = np.array(shuffles)
        return [shuffles]
    return generateShuffles



def buildReferences(model, oneHotSequences, shuffler, headId, taskId):
    #Run through the model and predict each of the input sequences, and also predict the shuffles. 
    #Returns two arrays, the first is the activation of the leftmost neuron of the headId/taskId task
    #and the second array is the average activation of that neuron on the shuffled sequences. 
    referenceActivations = np.zeros((oneHotSequences.shape[0],))
    sampleActivations = np.zeros((oneHotSequences.shape[0],))
    #Now run predictions on all of the sequences and their shuffles. 
    #outputTarget = tf.reduce_sum(model.outputs[config["head-id"]][:,0, config["task-id"]], axis=0, keepdims=True)
    logging.info("Predicting sequences and their shuffles...")
    for i in tqdm.tqdm(range(oneHotSequences.shape[0])):
        curSeq = oneHotSequences[i:i+1]
        curPred = model.predict(curSeq)
        curTarget = np.mean(curPred[headId][:,0, taskId])
        curShuffles = shuffler(curSeq)
        shufPred = model.predict(curShuffles[0])
        shufTarget = np.mean(shufPred[headId][:,0,taskId])
        referenceActivations[i] = shufTarget
        sampleActivations[i] = curTarget
    return sampleActivations, referenceActivations



def main(config):
    utils.setVerbosity(config["verbosity"])
    utils.setMemoryGrowth()
    model = load_model(config["model-file"], 
                       custom_objects = {'multinomialNll' : losses.multinomialNll})
    outputFile = h5py.File(config["output-h5"], 'w')
    genome = pysam.FastaFile(config["genome"])

    #Build up a list of all the regions that need to be shapped. 
    shapTargets = []
    logging.debug("Reading in bed file.")
    with open(config["bed-file"]) as fp:
        for line in fp:
            lsp = line.split()
            chrom = lsp[0]
            start = int(lsp[1])
            stop = int(lsp[2])
            for pos in range(start, stop):
                shapTargets.append((chrom, pos))
    #Now build up the array of one-hot encoded sequences.
    logging.debug("Regions loaded. Populating sequences.")
    oneHotSequences = np.zeros((len(shapTargets), config["input-length"], 4), dtype='float64')
    padding = (config["input-length"] - config["output-length"]) //2
    for i, target in tqdm.tqdm(enumerate(shapTargets)):
        #I need to generate a one-hot encoded sequence that has the current base on its left-most side. 
        startPos = target[1] - padding
        stopPos = startPos + config["input-length"]
        seq = genome.fetch(target[0], startPos, stopPos)
        oneHot = utils.oneHotEncode(seq)
        oneHotSequences[i] = oneHot
    logging.debug("Sequences loaded.")
    shuffler = shuffleGenerator(config["num-shuffles"])
    shuffles = shuffler(oneHotSequences[:1])
    
    #Calculate the reference effects. 
    logging.debug("Predicting reference and shuffles")
    refs = buildReferences(model, oneHotSequences, shuffler, config["head-id"], config["task-id"])
    outputFile.create_dataset('input_predictions', data=refs[0])
    outputFile.create_dataset('shuffle_predictions', data=refs[1])
    logging.debug("Predictions complete. Beginning shap.")
    
    #                                      Keep the first dimension so it seems like a batch size of one.v    
    #                                       Leftmost base in output v                                    |
    #                            All of the samples in this batch v |     Sum samples in batch. v        |
    #Oh boy, this slice.                      |--Current head---| V V  |current task----|       V        V
    outputTarget = tf.reduce_sum(model.outputs[config["head-id"]][:,0, config["task-id"]], axis=0, keepdims=True)
    profileExplainer = shap.TFDeepExplainer( (model.input, outputTarget), 
                                    shuffles)
    profileShapScores = profileExplainer.shap_values([oneHotSequences])


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
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
