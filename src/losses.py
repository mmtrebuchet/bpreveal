import tensorflow as tf
import tensorflow_probability as tfp
import logging


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
