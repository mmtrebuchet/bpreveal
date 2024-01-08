#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import json
import bpreveal.utils as utils
utils.setMemoryGrowth()
import h5py
from tensorflow import keras
from tensorflow.keras.backend import int_shape
import bpreveal.generators as generators
import bpreveal.losses as losses
import logging
from bpreveal.callbacks import getCallbacks
import bpreveal.models as models
import tensorflow as tf


def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs,
               earlyStop, outputPrefix, plateauPatience, heads, tensorboardDir=None):
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience, heads)
    if (tensorboardDir is not None):
        from bpreveal.callbacks import tensorboardCallback
        callbacks.append(tensorboardCallback(tensorboardDir))
    history = model.fit(trainBatchGen, epochs=epochs,
                        validation_data=valBatchGen, callbacks=callbacks)
    # Turn the learning rate data into python floats, since they come as
    # numpy floats and those are not serializable.
    history.history['lr'] = [float(x) for x in history.history["lr"]]
    # Add the counts loss weight history to the history json.
    lossCallback = callbacks[3]
    history.history["counts-loss-weight"] = lossCallback.λHistory
    return history


def main(config):
    utils.setVerbosity(config["verbosity"])
    logging.debug("Initializing")
    #utils.setMemoryGrowth()
    import jsonschema
    import bpreveal.schema
    jsonschema.validate(schema=bpreveal.schema.trainSoloModel, instance=config)
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]
    numHeads = len(config["heads"])

    model = models.soloModel(
        inputLength, outputLength,
        config["settings"]["architecture"]["filters"],
        config["settings"]["architecture"]["layers"],
        config["settings"]["architecture"]["input-filter-width"],
        config["settings"]["architecture"]["output-filter-width"],
        config["heads"], "solo")
    logging.debug("Model built.")
    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = []
    profileWeights = []
    countsWeights = []
    for head in config['heads']:
        profileWeights.append(head["profile-loss-weight"])
        λInit = head["counts-loss-weight"] if "counts-loss-weight" in head else 1
        λ = tf.Variable(λInit, dtype=tf.float32)
        head["INTERNAL_λ-variable"] = λ
        # The actual loss_weights parameter will be one - weighting
        # will be done inside the loss function proper.
        countsWeights.append(1)
        countsLosses.append(losses.weightedMse(λ))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["settings"]["learning-rate"]),
        loss=profileLosses + countsLosses,
        loss_weights=profileWeights + countsWeights)  # + is list concatenation, not addition!
    logging.info("Model compiled.")
    model.summary(print_fn=logging.debug)

    trainH5 = h5py.File(config["train-data"], "r")
    valH5 = h5py.File(config["val-data"], "r")

    trainGenerator = generators.H5BatchGenerator(
        config["heads"], trainH5, inputLength, outputLength,
        config["settings"]["max-jitter"], config["settings"]["batch-size"])
    valGenerator = generators.H5BatchGenerator(
        config["heads"], valH5, inputLength, outputLength,
        config["settings"]["max-jitter"], config["settings"]["batch-size"])
    logging.info("Generators initialized.")
    tensorboardDir = None
    if ("tensorboard-log-dir" in config):
        tensorboardDir = config["tensorflow-log-dir"]
    history = trainModel(model, inputLength, outputLength, trainGenerator, valGenerator,
                         config["settings"]["epochs"],
                         config["settings"]["early-stopping-patience"],
                         config["settings"]["output-prefix"],
                         config["settings"]["learning-rate-plateau-patience"],
                         config["heads"],
                         tensorboardDir)
    logging.debug("Model trained. Saving.")
    model.save(config["settings"]["output-prefix"] + ".model")
    with open("{0:s}.history.json".format(config["settings"]["output-prefix"]), "w") as fp:
        json.dump(history.history, fp, ensure_ascii=False, indent=4)


if (__name__ == "__main__"):
    import sys

    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    main(config)
