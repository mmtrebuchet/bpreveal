#!/usr/bin/env python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import utils
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape
import generators
import losses
import logging
from callbacks import getCallbacks
import models



def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop, outputPrefix, plateauPatience):
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience)
    history = model.fit(trainBatchGen, epochs=epochs, validation_data=valBatchGen, callbacks=callbacks)
    #Turn the learning rate data into python floats, since they come as numpy floats and those are not serializable.
    history.history['lr'] = [float(x) for x in history.history["lr"]]
    return history


def main(config):
    utils.setVerbosity(config["verbosity"])
    logging.info("Initializing")
    utils.setMemoryGrowth()
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    numHeads = len(config["heads"]) 

    model = models.soloModel(inputLength, outputLength,
            config["settings"]["architecture"]["filters"],
            config["settings"]["architecture"]["layers"],
            config["settings"]["architecture"]["input-filter-width"],
            config["settings"]["architecture"]["output-filter-width"],
            config["heads"], "solo")
    logging.info("Model built.")
    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = ['mse'] * numHeads
    profileWeights = []
    countsWeights = []
    logging.info(profileLosses)
    logging.info(countsLosses)
    for head in config['heads']:
        profileWeights.append(head["profile-loss-weight"])
        countsWeights.append(head["counts-loss-weight"])
    
    logging.info(profileWeights)
    logging.info(countsWeights)
    

    model.compile(optimizer=keras.optimizers.Adam(learning_rate = config["settings"]["learning-rate"]),
            loss=profileLosses + countsLosses,
            loss_weights = profileWeights + countsWeights) #+ is list concatenation, not addition!
    logging.info("Model compiled.")
    logging.debug(model.summary())
    
    trainH5 = h5py.File(config["train-data"], "r")
    valH5 = h5py.File(config["val-data"], "r")
    
    trainGenerator = generators.H5BatchGenerator(config["heads"], trainH5, 
            inputLength, outputLength, config["settings"]["max-jitter"], config["settings"]["batch-size"])
    valGenerator = generators.H5BatchGenerator(config["heads"], valH5, 
            inputLength, outputLength, config["settings"]["max-jitter"], config["settings"]["batch-size"])
    logging.info("Generators initialized.")
    history = trainModel(model, inputLength, outputLength, trainGenerator, valGenerator, config["settings"]["epochs"], 
                         config["settings"]["early-stopping-patience"], 
                         config["settings"]["output-prefix"],
                         config["settings"]["learning-rate-plateau-patience"])
    logging.info("Model trained. Saving.")
    model.save(config["settings"]["output-prefix"] + ".model")
    with open("{0:s}.history.json".format(config["settings"]["output-prefix"]), "w") as fp:
        json.dump(history.history, fp, ensure_ascii = False, indent = 4)



if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)


