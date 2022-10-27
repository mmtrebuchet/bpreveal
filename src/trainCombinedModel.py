#!/usr/bin/env python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape
from keras.models import load_model
import generators
import losses
from callbacks import getCallbacks
import models



def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop, outputPrefix, plateauPatience):
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience)
    history = model.fit(trainBatchGen, epochs=epochs, validation_data=valBatchGen, callbacks=callbacks)
    #Turn the learning rates into native python float values so they can be saved to json.
    history.history['lr'] = [float(x) for x in history.history['lr']]
    return history


def main(config):
    utils.setVerbosity(config["verbosity"])
    utils.setMemoryGrowth()
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    genomeFname = config["settings"]["genome"]
    numHeads = len(config["heads"]) 
    regressionModel = load_model(config["settings"]["transformation-model"]["transformation-model-file"], 
            custom_objects = {'multinomialNll' : losses.multinomialNll})
    regressionModel.trainable = False
     
    combinedModel, residualModel, transformationModel = models.combinedModel(inputLength, outputLength,
            config["settings"]["architecture"]["filters"],
            config["settings"]["architecture"]["layers"],
            config["settings"]["architecture"]["input-filter-width"],
            config["settings"]["architecture"]["output-filter-width"],
            config["heads"], regressionModel)

    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = ['mse'] * numHeads
    profileWeights = []
    countsWeights = []
    for head in config['heads']:
        profileWeights.append(head["profile-loss-weight"])
        countsWeights.append(head["counts-loss-weight"])

    residualModel.compile(optimizer=keras.optimizers.Adam(learning_rate = config["settings"]["learning-rate"]),
            jit_compile = True,
            loss=profileLosses + countsLosses,
            loss_weights = profileWeights + countsWeights) #+ is list concatenation, not addition!
    combinedModel.compile(optimizer=keras.optimizers.Adam(learning_rate = config["settings"]["learning-rate"]),
            jit_compile=True,
            loss=profileLosses + countsLosses,
            loss_weights = profileWeights + countsWeights) #+ is list concatenation, not addition!
    trainBeds = [x for x in config["regions"] if x["split"] == "train"]
    valBeds = [x for x in config["regions"] if x["split"] == "val"]
    
    trainGenerator = generators.BatchGenerator(trainBeds, config["heads"], genomeFname, 
            inputLength, outputLength, config["settings"]["batch-size"])
    valGenerator = generators.BatchGenerator(valBeds, config["heads"], genomeFname, 
            inputLength, outputLength, config["settings"]["batch-size"])
    history = trainModel(combinedModel, inputLength, outputLength, trainGenerator, valGenerator, config["settings"]["epochs"], 
                         config["settings"]["early-stopping-patience"], 
                         config["settings"]["output-prefix"],
                         config["settings"]["learning-rate-plateau-patience"])
    combinedModel.save(config["settings"]["output-prefix"] + "_combined"+ ".model")
    residualModel.save(config["settings"]["output-prefix"] + "_residual"+ ".model")
    with open("{0:s}.history.json".format(config["settings"]["output-prefix"]), "w") as fp:
        json.dump(history.history, fp, ensure_ascii = False, indent = 4)



if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)


