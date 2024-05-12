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

History
-------

Before BPReveal 3.0.0, there was a ``cropdown`` transformation option.
It turned out to be mathematically inappropriate, and so it was removed.

Also in BPReveal 3.0.0, a parameter named ``sequence-input-length`` was renamed to
just ``input-length``.

API
---
"""
import json
import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
from bpreveal import utils
if __name__ == "__main__":
    utils.setMemoryGrowth()
import tf_keras as keras  # pylint: disable=wrong-import-order
from bpreveal import logUtils
from bpreveal import models
import bpreveal.training
# pylint: disable=duplicate-code


def main(config):
    """Build and train the transformation model."""
    logUtils.setVerbosity(config["verbosity"])
    logUtils.debug("Initializing")
    soloModel = utils.loadModel(config["settings"]["solo-model-file"])

    # We're in the regression phase, no training the bias model!
    soloModel.trainable = False

    model = models.transformationModel(soloModel,
        config["settings"]["profile-architecture"],
        config["settings"]["counts-architecture"],
        config["heads"])

    losses, lossWeights = bpreveal.training.buildLosses(config["heads"])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config["settings"]["learning-rate"]),
        loss=losses, loss_weights=lossWeights)
    bpreveal.training.trainWithGenerators(model, config,
                                          config["settings"]["input-length"],
                                          config["settings"]["output-length"])
    model.save(config["settings"]["output-prefix"] + ".model")
    logUtils.info("Training job completed successfully.")


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "r") as configFp:
        configJson = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.trainTransformationModel.validate(configJson)
    main(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
