"""Defines the Multinomial NLL loss and the adaptive counts loss."""
from collections.abc import Callable
import tensorflow as tf
import tensorflow_probability as tfp
from keras import ops  # type: ignore
from bpreveal import logUtils
from keras.saving import register_keras_serializable  # type: ignore


@register_keras_serializable(package="bpreveal", name="multinomialNll")
def multinomialNll(trueCounts: tf.Tensor, logits: tf.Tensor) -> float:
    """The heart of what makes BPNet great - the loss function for profiles.

    :param trueCounts: The experimentally-observed counts.
        Shape ``(batch-size x output-length x num-tasks)``
    :param logits: The logits that the model is currently emitting.
        Shape ``(batch-size x output-length x num-tasks)``
    :return: A scalar representing the profile loss of this batch.
    """
    logUtils.debug("Creating multinomial NLL.")
    inputShape = ops.shape(trueCounts)
    numBatches = inputShape[0]
    numSamples = inputShape[1] * inputShape[2]  # output length * num_tasks

    flatCounts = ops.reshape(trueCounts, [numBatches, numSamples])
    flatLogits = ops.reshape(logits, [numBatches, numSamples])
    totalCounts = ops.sum(flatCounts, axis=1)
    distribution = tfp.distributions.Multinomial(total_count=totalCounts,
            logits=flatLogits)
    logprobs = distribution.log_prob(flatCounts)
    batchSize = ops.shape(trueCounts)[0]
    sumProbs = ops.sum(logprobs)
    curLoss = -sumProbs / ops.cast(batchSize, dtype=tf.float32)
    return curLoss


def weightedMse(weightTensor: tf.Variable) -> Callable:
    """Loss for the adaptive counts loss weight.

    Given a weight tensor (a tensorflow Variable of shape (1,))
    return a loss function that calculates mean square error and multiplies
    the error by the weight. This is used to implement the automatic
    counts weight algorithm.

    :param weightTensor: The tensor that will be adjusted by the dynamic
        counts loss weight algorithm.

    :return: A loss function.
    """
    logUtils.debug("Creating weighted mse.")

    @register_keras_serializable(package="bpreveal", name="reweightableMse")
    def reweightableMse(yTrue: tf.Tensor, yPred: tf.Tensor) -> float:
        squaredDiff = ops.square(yTrue - yPred)
        mse = ops.mean(squaredDiff, axis=-1)
        scaledMse = mse * weightTensor
        return scaledMse
    return reweightableMse


dummyMse = weightedMse(tf.constant(float("nan")))
"""Used for loading models. If you're going to train, you have to create a
weightedMse, since this one has no loss weight to it. But if you're just
predicting, you can pass this function to custom_objects when you load a
BPReveal model
Always returns nan, so that if you do accidentally use it, all of your
values will be poisoned.
"""
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
