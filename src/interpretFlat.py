#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
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
        shuffles = [rng.permutation(model_inputs[0]) for x in range(numShuffles)]
        return np.array(shuffles)
    return generateShuffles

def combineMultAndDiffref(mult, orig_inp, bg_data):
    #This is copied from Zahoor's code. 
    projected_hypothetical_contribs = \
            np.zeros_like(bg_data[0]).astype('float')
    assert (len(orig_inp[0].shape) == 2)
    for i in range(4): #We're going to go over all the base possibilities. 
        hypothetical_input = np.zeros_like(orig_inp[0]).astype('float')
        hypothetical_input[:,i] = 1.0
        hypothetical_diffref = hypothetical_input[None,:,:] - bg_data[0]
        hypothetical_contribs = hypothetical_diffref * mult[0]
        projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs, axis=-1)
    #There are no bias importances, so the np.zeros_like(orig_inp[1]) is not needed. 
    return [np.mean(projected_hypothetical_contribs, axis=0)]


def weightedMeannormLogits(model, headId, taskIds):
    profileOutput = model.outputs[headId]
    stackedLogits = tf.stack([profileOutput[:,:,x] for x in taskIds], axis=2)
    inputShape = stackedLogits.shape
    numBatches = inputShape[0]
    numSamples = inputShape[1] * inputShape[2]
    logits = tf.reshape(stackedLogits, [-1, numSamples])

    #logits = tf.unstack(stackedLogits, axis=2)
    meannormedLogits = logits - tf.reduce_mean(logits, axis=1)[:,None]

    stopgradMeannormedLogits = tf.stop_gradient(meannormedLogits)
    softmaxOut = tf.nn.softmax(stopgradMeannormedLogits, axis=1)
    weightedSum = tf.reduce_sum(softmaxOut * meannormedLogits, axis=1)
    return weightedSum


def writeHdf5(shapTargets, oneHotSequences, shapScores, outputFname, genome):
    outputFile = h5py.File(outputFname, "w")
    stringDtype = h5py.string_dtype(encoding='utf-8')
    outputFile.create_dataset('chrom_names', (genome.nreferences,), dtype=stringDtype)
    outputFile.create_dataset('chrom_sizes', (genome.nreferences,), dtype='u4')
    for i, chromName in enumerate(genome.references):
        outputFile['chrom_names'][i] = chromName
        outputFile['chrom_sizes'][i] = genome.get_reference_length(chromName)
    
    

    outputFile.create_dataset("coords_chrom", data=[x[0] for x in shapTargets])
    outputFile.create_dataset("coords_start", data=[x[1] for x in shapTargets])
    outputFile.create_dataset("coords_end", data=[x[2] for x in shapTargets])

    outputFile.create_dataset('hyp_scores', data=shapScores)
    outputFile.create_dataset('input_seqs', data=oneHotSequences)
    outputFile.close()


def main(jsonFname):
    with open(jsonFname, "r") as configFp:
        config = json.load(configFp)
    utils.setVerbosity(config["verbosity"])
    model = load_model(config["model-file"], 
                       custom_objects = {'multinomialNll' : losses.multinomialNll})
    genome = pysam.FastaFile(config["genome"])

    #Build up a list of all the regions that need to be shapped. 
    shapTargets = []
    padding = (config["input-length"] - config["output-length"]) //2
    with open(config["bed-file"]) as fp:
        for line in fp:
            lsp = line.split()
            chrom = lsp[0]
            start = int(lsp[1]) - padding
            stop = int(lsp[2]) + padding
            shapTargets.append((chrom, start, stop))
    

    #Now build up the array of one-hot encoded sequences.
    oneHotSequences = np.zeros((len(shapTargets), config["input-length"], 4), dtype='float64')
    for i, target in enumerate(shapTargets):
        #I need to generate a one-hot encoded sequence that has the current base on its left-most side. 
        startPos = target[1]
        stopPos = target[2]
        seq = genome.fetch(target[0], startPos, stopPos)
        oneHot = generators.oneHotEncode(seq)
        oneHotSequences[i] = oneHot

    
    shuffler = shuffleGenerator(config["num-shuffles"])
    
    
    profileMetric = weightedMeannormLogits(model, config["head-id"], config["profile-task-ids"])
    profileExplainer = shap.TFDeepExplainer( (model.input, profileMetric), 
                                    shuffler,
                                    combine_mult_and_diffref = combineMultAndDiffref)
    profileShapScores = profileExplainer.shap_values([oneHotSequences])
    writeHdf5(shapTargets, oneHotSequences, profileShapScores, config["profile-h5"], genome)


    countsMetric = model.outputs[config["heads"] + config["head-id"]][:,0]
    print(countsMetric)
    print(countsMetric.shape)
    countsExplainer = shap.TFDeepExplainer( (model.input, countsMetric), 
                                    shuffler,
                                    combine_mult_and_diffref = combineMultAndDiffref)
    countsShapScores = countsExplainer.shap_values([oneHotSequences])
    writeHdf5(shapTargets, oneHotSequences, countsShapScores, config["counts-h5"], genome)



if (__name__ == "__main__"):
    import sys
    main(sys.argv[1])







