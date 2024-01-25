"""Defines the Multinomial NLL loss and the adaptive counts loss."""
import tensorflow as tf
import tensorflow_probability as tfp
import logging
from keras import backend


def multinomialNll(trueCounts, logits):
    """The heart of what makes BPNet great - the loss function for profiles.

    :param trueCounts: The experimentally-observed counts.
        Shape ``(batch-size x output-length x num-tasks)``
    :param logits: The logits that the model is currently emitting.
        Shape ``(batch-size x output-length x num-tasks)``
    :return: A scalar representing the profile loss of this batch.
    """
    logging.debug("Creating multinomial NLL.")
    inputShape = tf.shape(trueCounts)
    numBatches = inputShape[0]
    numSamples = inputShape[1] * inputShape[2]  # output length * num_tasks

    flatCounts = tf.reshape(trueCounts, [numBatches, numSamples])
    flatLogits = tf.reshape(logits, [numBatches, numSamples])
    totalCounts = tf.reduce_sum(flatCounts, axis=1)
    distribution = tfp.distributions.Multinomial(total_count=totalCounts,
            logits=flatLogits)
    logprobs = distribution.log_prob(flatCounts)
    batchSize = tf.shape(trueCounts)[0]
    sumProbs = tf.reduce_sum(logprobs)
    curLoss = -sumProbs / tf.cast(batchSize, dtype=tf.float32)
    return curLoss


def weightedMse(weightTensor):
    """Loss for the adaptive counts loss weight.

    Given a weight tensor (a tensorflow Variable of shape (1,))
    return a loss function that calculates mean square error and multiplies
    the error by the weight. This is used to implement the automatic
    counts weight algorithm.

    :param weightTensor: The tensor that will be adjusted by the dynamic
        counts loss weight algorithm.

    :return: A loss function.
    """
    logging.debug("Creating weighted mse.")

    def reweightableMse(yTrue, yPred):
        yPred = tf.convert_to_tensor(yPred)
        yTrue = tf.cast(yTrue, yPred.dtype)
        mse = backend.mean(tf.math.squared_difference(yPred, yTrue), axis=-1)
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
