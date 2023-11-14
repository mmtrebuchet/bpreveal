#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import json
import bpreveal.utils as utils
import h5py
from tensorflow import keras
from keras.models import load_model
import bpreveal.generators as generators
import bpreveal.losses as losses
from bpreveal.callbacks import getCallbacks
import bpreveal.models as models
import logging
import tensorflow as tf


def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop,
               outputPrefix, plateauPatience, heads, tensorboardDir=None):
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience, heads)
    if (tensorboardDir is not None):
        from callbacks import tensorboardCallback
        callbacks.append(tensorboardCallback(tensorboardDir))
    history = model.fit(trainBatchGen, epochs=epochs,
                        validation_data=valBatchGen, callbacks=callbacks)
    # Turn the learning rates into python floats for json serialization.
    history.history['lr'] = [float(x) for x in history.history['lr']]
    lossCallback = callbacks[3]
    history.history["counts-loss-weight"] = lossCallback.weightHistory
    return history


def main(config):
    utils.setVerbosity(config["verbosity"])
    utils.setMemoryGrowth()
    if ("sequence-input-length" in config):
        assert False, "Sequence-input-length has been renamed "\
                      "input-length in transformation config files."
    inputLength = config["settings"]["input-length"]
    outputLength = config["settings"]["output-length"]
    numHeads = len(config["heads"])
    soloModel = load_model(config["settings"]["solo-model-file"],
            custom_objects={'multinomialNll': losses.multinomialNll})

    soloModel.trainable = False  # We're in the regression phase, no training the bias model!

    model = models.transformationModel(soloModel,
        config["settings"]["profile-architecture"],
        config["settings"]["counts-architecture"],
        config["heads"])

    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = []
    profileWeights = []
    countsWeights = []
    for head in config['heads']:
        profileWeights.append(head["profile-loss-weight"])
        countsWeight = tf.Variable(head["counts-loss-weight"], dtype=tf.float32)
        head["INTERNAL_counts-loss-weight-variable"] = countsWeight
        countsWeights.append(1)
        countsLosses.append(losses.weightedMse(countsWeight))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["settings"]["learning-rate"]),
        loss=profileLosses + countsLosses,
        loss_weights=profileWeights + countsWeights)  # + is list concatenation, not addition!
    model.summary(print_fn=logging.debug)
    trainH5 = h5py.File(config["train-data"], "r")
    valH5 = h5py.File(config["val-data"], "r")

    trainGenerator = generators.H5BatchGenerator(
        config["heads"], trainH5,
        inputLength, outputLength,
        config["settings"]["max-jitter"], config["settings"]["batch-size"])
    valGenerator = generators.H5BatchGenerator(
        config["heads"], valH5,
        inputLength, outputLength,
        config["settings"]["max-jitter"], config["settings"]["batch-size"])
    logging.info("Generators initialized. Training.")
    tensorboardDir = None
    if ("tensorboard-log-dir" in config):
        tensorboardDir = config["tensorboard-log-dir"]

    history = trainModel(model, inputLength, outputLength, trainGenerator,
                         valGenerator, config["settings"]["epochs"],
                         config["settings"]["early-stopping-patience"],
                         config["settings"]["output-prefix"],
                         config["settings"]["learning-rate-plateau-patience"],
                         config["heads"],
                         tensorboardDir)

    model.save(config["settings"]["output-prefix"] + ".model")
    with open("{0:s}.history.json".format(config["settings"]["output-prefix"]), "w") as fp:
        json.dump(history.history, fp, ensure_ascii=False, indent=4)


if (__name__ == "__main__"):
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
