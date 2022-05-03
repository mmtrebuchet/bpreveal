#!/usr/bin/env python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import generators
import losses
import layers
from callbacks import getCallbacks
import models

def buildSimpleRegressionModel(architectureSpecification, inputShape, headName, inputLayer):
    """The simple type of regression takes a list of activation types and adds all of them together. 
    The input is a single output layer (NOT a profile and counts, but EITHER profile or counts)
    for a single head.
    Based on the architecture specification, create a bunch of layers that perform different activations on the input,
    then sum all those activations up. 
    In general, the format is
    input -> linear -> activation -> linear -> output. 
    though, for obvious reasons, if you're only using a linear activation, it's just 
    input -> linear -> output. 

    """
    activationLayers = []
    for layerType in architectureSpecification["types"]:
        match layerType:
            case "linear":
                activationLayers.append(layers.LinearRegression(name='regress_linear_{0:s}'.format(headName))(inputLayer))
            case 'sigmoid':
                inputLinear = layers.LinearRegression(name='sigmoid_in_linear_{0:s}'.format(headName))(inputLayer)
                sigmoided = keras.layers.Activation(activation=keras.activations.sigmoid, name='sigmoid_activation_{0:s}'.format(headName))(inputLinear)
                outputLinear = layers.LinearRegression(name='sigmoid_out_linear_{0:s}'.format(headName))(sigmoided)
                activationLayers.append(outputLinear)
            case _:
                raise ValueError("The simple layer type you gave ({0:s}) is not supported".format(layerType))
    if(len(activationLayers) > 1):
        sumLayer = keras.layers.Add(name='regress_sum_{0:s}'.format(headName))(activationLayers)
    else:
        sumLayer = activationLayers[0]
    return sumLayer


def regressionHead(biasProfile, biasCounts, individualHead, architectureSpecification):
    """Based on the architecture requested for the bias model, build a single output head
    that takes one output of the bias model and transforms it according to the architecture. 
    This (very simple) model will then be regressed to match the experimental data, in order to 
    remove bias."""

    numOutputs = len(individualHead["data"])
    match architectureSpecification["name"]:
        case 'simple':
            profileRegression = buildSimpleRegressionModel(architectureSpecification, 
                    (architectureSpecification['output-length'], numOutputs), 
                    individualHead["head-name"] + "_profile",
                    biasProfile)
            countsRegression = buildSimpleRegressionModel(architectureSpecification, 
                    (numOutputs,),
                    individualHead["head-name"] + "_counts",
                    biasCounts)
        case _:
            raise ValueError("Currently, only simple regression is supported.")


    return (profileRegression, countsRegression)



def regressionModel(biasModel, architectureSpecification, headList):
    profileOutputs = []
    countsOutputs = []
    numHeads = len(headList)
    for i, individualHead in enumerate(headList):
        profileHead, countsHead = regressionHead(
                biasModel.outputs[i], 
                biasModel.outputs[i+numHeads],
                individualHead,
                architectureSpecification)
        profileOutputs.append(profileHead)
        countsOutputs.append(countsHead)
    m = keras.Model(inputs=biasModel.input, outputs = profileOutputs + countsOutputs, name="regression_model")
    return m

def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop, outputPrefix):
    callbacks = getCallbacks(earlyStop, outputPrefix)
    history = model.fit(trainBatchGen, epochs=epochs, validation_data=valBatchGen, callbacks=callbacks)
    return history


def main(configJsonFname):
    with open(configJsonFname, "r") as configFp:
        config = json.load(configFp)
    inputLength = config["settings"]["bias-input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    genomeFname = config["settings"]["genome"]
    numHeads = len(config["heads"]) 
    biasModel = load_model(config["settings"]["bias-model-file"], 
            custom_objects = {'multinomialNll' : losses.multinomialNll})

    biasModel.trainable=False #We're in the regression phase, no training the bias model!

    model = regressionModel(biasModel,
            config["settings"]["architecture"],
            config["heads"])

    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = ['mse'] * numHeads
    profileWeights = []
    countsWeights = []
    print(profileLosses)
    print(countsLosses)
    for head in config['heads']:
        profileWeights.append(head["profile-loss-weight"])
        countsWeights.append(head["counts-loss-weight"])
    print(profileWeights)
    print(countsWeights)
    

    model.compile(optimizer=keras.optimizers.Adam(learning_rate = config["settings"]["learning-rate"]),
            #run_eagerly=True,
            loss=profileLosses + countsLosses,
            loss_weights = profileWeights + countsWeights) #+ is list concatenation, not addition!
    print(model.summary())
    print(model.outputs)
    #variables = [(v.name, v.shape, v.trainable) for v in model.variables]
    #for v in variables:
    #    print(v)
    #print(profileLosses, countsLosses)
    trainBeds = [x for x in config["regions"] if x["split"] == "train"]
    valBeds = [x for x in config["regions"] if x["split"] == "val"]
    
    trainGenerator = generators.BatchGenerator(trainBeds, config["heads"], genomeFname, 
            inputLength, outputLength, config["settings"]["batch-size"])
    valGenerator = generators.BatchGenerator(valBeds, config["heads"], genomeFname, 
            inputLength, outputLength, config["settings"]["batch-size"])
    history = trainModel(model, inputLength, outputLength, trainGenerator, valGenerator, config["settings"]["epochs"], 
                         config["settings"]["early-stopping-patience"], 
                         config["settings"]["output-prefix"])
    model.save(config["settings"]["output-prefix"] + ".model")
    with open("{0:s}.history.json".format(config["settings"]["output-prefix"]), "w") as fp:
        json.dump(history.history, fp, ensure_ascii = False, indent = 4)



if (__name__ == "__main__"):
    import sys
    main(sys.argv[1])


