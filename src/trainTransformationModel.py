#!/usr/bin/env python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import utils
import tensorflow as tf
utils.setMemoryGrowth()

from tensorflow import keras
from keras.models import load_model
import generators
import losses
import layers
from callbacks import getCallbacks
import models
import logging

def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop, outputPrefix):
    callbacks = getCallbacks(earlyStop, outputPrefix)
    history = model.fit(trainBatchGen, epochs=epochs, validation_data=valBatchGen, callbacks=callbacks)
    return history


def main(configJsonFname):
    with open(configJsonFname, "r") as configFp:
        config = json.load(configFp)
    inputLength = config["settings"]["sequence-input-length"]
    outputLength = config["settings"]["output-length"]
    genomeFname = config["settings"]["genome"]
    numHeads = len(config["heads"]) 
    soloModel = load_model(config["settings"]["solo-model-file"], 
            custom_objects = {'multinomialNll' : losses.multinomialNll})

    soloModel.trainable=False #We're in the regression phase, no training the bias model!

    model = models.transformationModel(soloModel,
            config["settings"]["profile-architecture"],
            config["settings"]["counts-architecture"],
            config["heads"])

    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = ['mse'] * numHeads
    profileWeights = []
    countsWeights = []
    for head in config['heads']:
        profileWeights.append(head["profile-loss-weight"])
        countsWeights.append(head["counts-loss-weight"])
    

    model.compile(optimizer=keras.optimizers.Adam(learning_rate = config["settings"]["learning-rate"]),
            #run_eagerly=True,
            loss=profileLosses + countsLosses,
            loss_weights = profileWeights + countsWeights) #+ is list concatenation, not addition!
    logging.info(model.summary())
    logging.info(model.outputs)
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


