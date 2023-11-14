import tensorflow as tf
import tensorflow_probability as tfp
import logging
from keras import backend


def multinomialNll(trueCounts, logits):
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
    """Given a weight tensor (a tensorflow Variable of shape (1,))
    return a loss function that calculates mean square error and multiplies
    the error by the weight. This is used to implement the automatic
    counts weight algorithm.
    Returns a loss function."""
    logging.debug("Creating weighted mse.")

    def mse(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mse = backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)
        scaledMse = mse * weightTensor
        return scaledMse
    return mse
