#!/usr/bin/env python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape
import generators
import losses
from callbacks import getCallbacks
import models

#Generate a simple sequence model taking one-hot encoded input and producing a logits profile and a log(counts) scalar. 


def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop, outputPrefix):
    callbacks = getCallbacks(earlyStop, outputPrefix)
    history = model.fit(trainBatchGen, epochs=epochs, validation_data=valBatchGen, callbacks=callbacks)
    return history


def main(configJsonFname):
    with open(configJsonFname, "r") as configFp:
        config = json.load(configFp)
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    genomeFname = config["settings"]["genome"]
    numHeads = len(config["heads"]) 

    model = models.biasModel(inputLength, outputLength,
            config["settings"]["architecture"]["filters"],
            config["settings"]["architecture"]["layers"],
            config["settings"]["architecture"]["input-filter-width"],
            config["settings"]["architecture"]["output-filter-width"],
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
    #profileLosses = losses.multinomialLoss(config["heads"])
    print(profileWeights)
    print(countsWeights)
    
    #countsLosses = losses.countsLoss(config["heads"])

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


