"""A simple set of functions that train with a curses display."""
import json
from typing import Any
import h5py
from bpreveal.internal import disableTensorflowLogging  # pylint: disable=unused-import # noqa
import tensorflow as tf
from tensorflow import keras
import numpy as np
from bpreveal.callbacks import getCallbacks
from keras.callbacks import History  # type: ignore
from bpreveal import logUtils
from bpreveal import generators
from bpreveal import losses


def buildLosses(heads: dict) -> tuple[list, list]:
    r"""Given the output head specification (from the configuration JSON), build losses.

    :param heads: The heads section from a configuration file.
    :return: A tuple. The first element contains the losses, and the second contains the
        loss weights.

    This method injects the λ parameter used for the adaptive loss algorithm into the
    heads data structure. After this function is done, each head in heads will contain
    #ifdef MAN_PAGE
    a member called ``INTERNAL_λ-variable``. This is an actual TensorFlow
    #else
    a member called :math:`\tt INTERNAL\_\lambda{}\text{-}variable`. This is an actual TensorFlow
    #endif
    variable, and it is hooked in to the counts loss so that the adaptive loss callback
    (which also gets a copy of ``heads``) can adjust it during training.

    Returns a 2-tuple, structured so:

    1. A list of all the profile losses followed by all the counts losses:
        ``profLoss1, profLoss2, ... profLossN, countsLoss1, countsLoss2, ... countsLossN``
    2. A list of all the loss weights.
        Since the counts loss weight is included in the actual loss function,
        the weights that are given to the Keras training routine are all ones
        for the counts. The profile loss weights are taken straight from your
        json.
        ``profileWeight1, profileWeight2, ... profileWeightN, 1, 1, ... 1``
    """
    logUtils.info("Building loss functions.")
    numHeads = len(heads)
    profileLosses = [losses.multinomialNll] * numHeads
    countsLosses = []
    profileWeights = []
    countsWeights = []
    for head in heads:
        profileWeights.append(head["profile-loss-weight"])
        # For adaptive loss weights, make the counts loss a keras variable so I
        # can update it during training.
        λInit = head["counts-loss-weight"] if "counts-loss-weight" in head else 1
        λ = tf.Variable(λInit, dtype=tf.float32)
        # Store the (keras) variable with the loss weight in the head dictionary.
        # We'll need it in the callbacks, and this is a reasonable place to store it.
        head["INTERNAL_λ-variable"] = λ
        # The actual loss_weights parameter will be one - weighting
        # will be done inside the loss function proper.
        countsWeights.append(1.0)
        countsLosses.append(losses.weightedMse(λ))
        logUtils.debug(f"Initialized head {head['head-name']} with λinit = {λInit}")
    allLosses = profileLosses + countsLosses  # + is concatenation, not addition!
    allWeights = profileWeights + countsWeights  # + is still concatenation.
    return (allLosses, allWeights)


def _makeJsonSerializable(cfg: Any) -> Any:  # pylint: disable=too-many-return-statements
    """Takes history dictionary that may contain ndarrays and tf.Variables and makes it json-safe.

    :param cfg: The object to serialize
    :return: An object that JSON can handle.
    """
    match cfg:
        case dict():
            ret = {}
            for k, v in cfg.items():
                ret[str(k)] = _makeJsonSerializable(v)
            return ret
        case int():
            return cfg
        case float():
            return cfg
        case list():
            return [_makeJsonSerializable(x) for x in cfg]
        case tuple():
            return [_makeJsonSerializable(x) for x in cfg]
        case str():
            return cfg
    if isinstance(cfg, tf.Variable):
        return {_makeJsonSerializable(cfg.name): _makeJsonSerializable(cfg.numpy())}
    if isinstance(cfg, np.ndarray):
        return [_makeJsonSerializable(x) for x in cfg]
    try:
        r = float(cfg)
        return r
    except (TypeError, ValueError):
        pass
    logUtils.warning(f"Unknown type for config dict: {cfg} has type {type(cfg)}.")
    return str(cfg)


def trainWithGenerators(model: keras.Model, config: dict, inputLength: int,
                        outputLength: int) -> History:
    """Load up the generators from your config file and train the model!

    :param model: A compiled Keras model.
    :param config: The configuration JSON, AFTER you have injected ``config["heads"]``
        with :py:func:`buildLosses<bpreveal.training.buildLosses>`.
    :param inputLength: The input length of your model.
    :param outputLength: The output length of your model.
    :return: The history dictionary from the training.
    """
    model.summary(print_fn=logUtils.debug)

    logUtils.info("Loading data into generators.")
    trainH5 = h5py.File(config["train-data"], "r")
    valH5 = h5py.File(config["val-data"], "r")

    trainGenerator = generators.H5BatchGenerator(
        config["heads"], trainH5, inputLength, outputLength,
        config["settings"]["max-jitter"], config["settings"]["batch-size"])
    valGenerator = generators.H5BatchGenerator(
        config["heads"], valH5, inputLength, outputLength,
        config["settings"]["max-jitter"], config["settings"]["batch-size"])
    logUtils.info("Generators initialized. Training.")
    history = trainModel(
        model, trainGenerator,
        valGenerator, config["settings"]["epochs"],
        config["settings"]["early-stopping-patience"],
        config["settings"]["output-prefix"],
        config["settings"]["learning-rate-plateau-patience"],
        config["heads"])
    logUtils.info("Saving history.")
    historyName = f"{config['settings']['output-prefix']}.history.json"
    hist = history.history
    hist["config"] = config
    with open(historyName, "w") as fp:
        json.dump(_makeJsonSerializable(hist), fp, ensure_ascii=False, indent=4)
    return history


def trainModel(model: keras.Model, trainBatchGen: generators.H5BatchGenerator,
               valBatchGen: generators.H5BatchGenerator, epochs: int,
               earlyStop: int, outputPrefix: str,
               plateauPatience: int, heads: list[dict]) -> History:
    """Constructs callbacks and actually runs the training loop.

    :param model: The compiled model to train.
    :param trainBatchGen: The batch generator for training samples.
    :param valBatchGen: The batch generator for validation samples.
    :param epochs: The maximum number of epochs to train for.
    :param earlyStop: How many epochs without validation loss improvement before we stop
        training?
    :param outputPrefix: Where would you like your model saved?
        (This is used by the checkpoint callback.)
    :param plateauPatience: How many epochs without validation loss improvement before
        we reduce the learning rate?
    :param heads: The ``heads`` from the configuration JSON, AFTER you have injected losses
        with :py:func:`buildLosses<bpreveal.training.buildLosses>`.
    :return: The history dictionary generated by the Keras training function.
    """
    logUtils.info("Generating callbacks.")
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience, heads,
                             trainBatchGen, valBatchGen)
    logUtils.info("Beginning training loop.")
    history = model.fit(trainBatchGen, epochs=epochs,
                        validation_data=valBatchGen, callbacks=callbacks,
                        verbose=0)  # type: ignore
    logUtils.info("Training complete! Hooray!")
    # Add the counts loss weight history to the history json.
    lossCallback = callbacks[4]
    history.history["counts-loss-weight"] = lossCallback.λHistory
    return history
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
