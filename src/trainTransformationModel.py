#!/usr/bin/env python3
"""Trains up a simple regression model to match a bias model to an experiment.

The transformation input file is a JSON file that names a solo model and gives
the experimental data that it should be fit to.
Note that it may occasionally be appropriate to chain several transformation
models together.
Currently, the easiest way to do this is to feed the first transformation model
in as the solo model for the second transformation.
A better way to do it would be to write your own custom transformation Model.


BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/trainTransformationModel.bnf


Parameter Notes
---------------
Most of the parameters for the transformation model are the same as for a solo
model, and they are described at
:py:mod:`trainSoloModel<bpreveal.trainSoloModel>`.

solo-model-file
    The name of the file (or directory, since
    that's how keras likes to save models) that contains the solo model.

passthrough
    This transformation does nothing to the solo model,
    it doesn't regress anything.

simple
    This transformation applies the specified functions to
    the output of the solo model, and adjusts the parameters to best fit the
    experimental data.
    A linear model applies :math:`y=m x+b` to the solo predictions (which,
    remember, are in log-space),
    a sigmoid applies :math:`y = m_1 *sigmoid(m_2x+b_2) + b_1`,
    and a relu applies :math:`y = m_1 * relu(m_2x+b_2) + b_1`.
    In other words, there's a linear model both before and after the sigmoid
    or relu activation.
    Generally, you need to use these more complex functions when the solo
    model is not a great fit for the experimental bias.

HISTORY
-------

Before BPReveal 3.0.0, there was a ``cropdown`` transformation option.
It turned out to be mathematically inappropriate, and so it was removed.

Also in BPReveal 3.0.0, a parameter named ``sequence-input-length`` was renamed to
just ``input-length``.

API
---
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import json
from bpreveal import utils
if __name__ == "__main__":
    utils.setMemoryGrowth()
import h5py
from tensorflow import keras
from bpreveal import generators
from bpreveal import losses
from bpreveal import models
from bpreveal.callbacks import getCallbacks
import logging
import tensorflow as tf


def trainModel(model, inputLength, outputLength, trainBatchGen, valBatchGen, epochs, earlyStop,
               outputPrefix, plateauPatience, heads, tensorboardDir=None):
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience, heads)
    if tensorboardDir is not None:
        from callbacks import tensorboardCallback
        callbacks.append(tensorboardCallback(tensorboardDir))
    history = model.fit(trainBatchGen, epochs=epochs,
                        validation_data=valBatchGen, callbacks=callbacks)
    # Turn the learning rates into python floats for json serialization.
    history.history['lr'] = [float(x) for x in history.history['lr']]
    lossCallback = callbacks[3]
    history.history["counts-loss-weight"] = lossCallback.λHistory
    return history


def main(config):
    utils.setVerbosity(config["verbosity"])
    inputLength = config["settings"]["input-length"]
    outputLength = config["settings"]["output-length"]
    numHeads = len(config["heads"])
    soloModel = utils.loadModel(config["settings"]["solo-model-file"])

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
        λInit = head["counts-loss-weight"] if "counts-loss-weight" in head else 1
        λ = tf.Variable(λInit, dtype=tf.float32)
        head["INTERNAL_λ-variable"] = λ
        countsWeights.append(1)
        countsLosses.append(losses.weightedMse(λ))

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
    if "tensorboard-log-dir" in config:
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


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.trainTransformationModel.validate(config)
    main(config)
