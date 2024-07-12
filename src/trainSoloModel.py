#!/usr/bin/env python3
"""Trains up a sequence-to-profile model in the BPNet style.

This program generates a neural network that maps genomic sequence to
experimental readout.
In cases where you don't need to regress out a bias track, this is the only
training program you'll use.

The required input files are two hdf5-format files that contain sequences and
profiles.
One of these is used for training, one for validation during training.
See :py:mod:`prepareTrainingData<bpreveal.prepareTrainingData>` for
the specification for this file.

The program will produce two outputs, one being a Keras model.
This model will be used later for prediction and interpretation.
The other output records the progress of the model during training, and it is
read in by :py:mod:`makeLossPlots<bpreveal.makeLossPlots>`.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/trainSoloModel.bnf


Parameter Notes
---------------
profile-loss-weight
    Simply a scalar that the profile loss is multiplied by. This comes in handy
    when you're training on two datasets with disparate coverage. Since the
    MNLL loss is proportional to the number of reads, a track with higher
    coverage will dominate the loss. Instead of calculating this *a priori*, I
    find it easiest to start training the model, look at the loss values, and
    then pick multipliers that will make them about even.

counts-loss-weight
    Similar to ``profile-loss-weight``, but do keep in mind that you need to
    set it even when you're training a single output head, since the mean
    squared error value of the counts prediction tends to be minuscule compared
    to the MNLL loss of the profile. Again, instead of calculating it
    *a priori*, I start training with an initial guess and then refine the
    value later, or you can use the adaptive loss weight algorithm to set it
    automatically. In the original BPNet, this parameter was called λ, and this
    is the name used inside the codebase.

counts-loss-frac-target
    (Optional) Turns on the adaptive counts loss adjustment algorithm. After
    each epoch, the ``counts-loss-weight`` parameter is adjusted so that the
    fraction of the loss due to counts (for this head) matches the specified
    number. See :doc:`countsLossReweighting`.

output-prefix
    The file name where you want your model saved. For example, if you are
    saving models in a directory called ``models``, and you want the model to
    be called ``solo``, then you'd write ``"output-prefix" : "models/solo"``.
    In this case, you'll find the files ``models/solo.keras``, which is the
    Keras model, and ``models/solo.history.json``, containing the training
    history.

    .. note:
        Before BPReveal 5.0, the created file was given a ``.model`` extension
        but this changed to ``.keras`` as of BPReveal 5.0.

early-stopping-patience
    Controls how long the network should wait for an improvement in the loss
    before quitting. I recommend a bit more than double the
    ``learning-rate-plateau-patience``, on the order of 11.

batch-size
    Determines how many regions the network will look at simultaneously during
    training. It doesn't really matter, but if you make it too big your data
    won't fit on the GPU and if you make it too small your network will take an
    eternity to train. I like 64 or so.

learning-rate determines
    Sets how aggressive the optimizer will be as the network trains. 0.004 is a
    good bet. (Note that the LR will decrease during training because of the
    plateau patience.)

learning-rate-plateau-patience
    How many epochs must pass without improvement before the optimizer
    decreases the learning rate? I recommend 5 or so.

architecture-name
    (UNUSED) A future-proofing argument that will determine what type of
    network you want. Currently, only the basic BPNet-style architecture is
    supported. See :doc:`modelArchitectures` for details.

input-length
    The length of the input sequence that will be fed into the network. You can
    use the :py:mod:`lengthCalc<bpreveal.lengthCalc>` script to calculate this
    based on a desired profile length and architecture.

output-length
    The length of the predicted profile. This is usually on the order of 1000.

model-name
    A string that is stored along with the model. BPReveal does not use it
    internally.

model-args
    (UNUSED) A future-proofing argument. If there is a new feature added to a
    particular architecture, the ``model-args`` string will be passed to the
    architecture and the architecture may do with that string as it pleases.
    Currently, this serves no purpose.

filters
    The number of convolutional filters at each layer. The more filters you
    add, the more patterns the model will try to learn. Typically this is
    between 32 and 128, smaller for simpler tasks.

input-filter-width
    The size of the very first motif-scanning layer in BPNet. Lately, there's
    been a trend of making this small, on the order of 7.

output-filter-width
    The width of the very last convolution, the one that actually results in
    the predicted profile. This layer is placed at the very bottom of the
    dilated layers. I use a width of 75, but many people use smaller output
    widths, on the order of 25.

max-jitter
    The maximum allowed shifting of the regions. This random shifting is
    applied during training, and helps to create some variety in the counts
    values to prevent over-fitting. Note that you must use the same jitter you
    used when you created your training data file - if you want to try a
    different jitter, you need to re-generate your data hdf5 files.

Additional information
----------------------

Window padding
^^^^^^^^^^^^^^

By and large, you can refer to the BPNet paper for details on how this program
works, the only difference is in the input padding.
In the original BPNet, for an output window of 1 kb, the input sequence was
1 kb long, and if a neuron needed to know about bases outside that window, it
got all zeros.
For image processing, this makes sense, because you can't un-crop an image.
However, for DNA in a genome, you can just expand out the windows and get as
much DNA sequence as you like.
Therefore, BPReveal models require an input length that is larger than output
length, so that the model can use DNA sequence information that is outside of
its output window.


API
---

"""
import bpreveal.schema
import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
from bpreveal import utils
if __name__ == "__main__":
    utils.setMemoryGrowth()
from tensorflow import keras
from bpreveal import logUtils
from bpreveal import models
import bpreveal.training
from bpreveal.internal import interpreter
# pylint: disable=duplicate-code


def main(config: dict) -> None:
    """Build and train a model."""
    logUtils.setVerbosity(config["verbosity"])
    logUtils.debug("Initializing")
    inputLength = config["settings"]["architecture"]["input-length"]
    outputLength = config["settings"]["architecture"]["output-length"]

    model = models.soloModel(
        inputLength, outputLength,
        config["settings"]["architecture"]["filters"],
        config["settings"]["architecture"]["layers"],
        config["settings"]["architecture"]["input-filter-width"],
        config["settings"]["architecture"]["output-filter-width"],
        config["heads"], "solo")
    logUtils.debug("Model built.")
    losses, lossWeights = bpreveal.training.buildLosses(config["heads"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["settings"]["learning-rate"]),
        loss=losses, loss_weights=lossWeights, metrics=losses)
    bpreveal.training.trainWithGenerators(model, config, inputLength, outputLength)
    model.save(config["settings"]["output-prefix"] + ".keras")
    logUtils.info("Training job completed successfully.")


if __name__ == "__main__":
    import sys
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    bpreveal.schema.trainSoloModel.validate(configJson)
    main(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
