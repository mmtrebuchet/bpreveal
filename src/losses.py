import tensorflow as tf
import tensorflow_probability as tfp

"""
def countsLoss(headList):
    lossFunctions = []
    for individualHead in headList:
        lossFunctions.append(_individualCounts(individualHead))
    return lossFunctions

def _individualCounts(individualHead):
    def mse(trueLogcounts, predLogcounts):
        #Both of the input tensors will be of shape (batchSize ×  numTasks)
        Δ = trueLogcounts - predLogcounts
        ΔSquared = Δ * Δ
        mseVal = tf.reduce_mean(ΔSquared) * individualHead["counts-loss-weight"]
        return mseVal
    return mse
"""
def multinomialNll(trueCounts, logits):
    inputShape = tf.shape(trueCounts)
    numBatches = inputShape[0] 
    numSamples = inputShape[1] * inputShape[2] #output length * num_tasks 

    flatCounts = tf.reshape(trueCounts, [numBatches, numSamples])
    flatLogits = tf.reshape(logits, [numBatches, numSamples])
    totalCounts = tf.reduce_sum(flatCounts, axis=1)
    distribution = tfp.distributions.Multinomial(total_count = totalCounts, 
            logits = flatLogits)
    logprobs = distribution.log_prob(flatCounts)
    batchSize = tf.shape(totalCounts)[0]
    sumProbs = tf.reduce_sum(logprobs)
    curLoss = -sumProbs / tf.cast(batchSize, dtype=tf.float32)
    #tf.print("loss ", totalLoss)
    return curLoss

"""
def multinomialLoss(headList):
    " ""Creates a multinomial loss that can combine multiple tasks into a single multinomial distribution. 
    headList is taken directly from a <head-list> in the configuration JSON. 
    Returns a list of functions that can be used as a loss with the network.
    " ""
    #The network's profile output is (batch × numTasks × outputLength). 
    #where numTasks is the number of tasks for the *current* head, not all the heads together. 
    #The trueCounts input must be the same shape. 
    lossFunctions = []
    for individualHead in headList:
        lossFunctions.append(_individualMultinomial(individualHead))
    return lossFunctions

def _individualMultinomial(individualHead):

    def multinomialNll(trueCounts, logits):
        #Slice out the counts and logits that comprise this multinomial distribution.
        batchLogits = []
        batchCounts = []
        #tf.print("logits ", logits)
        #tf.print("counts ", trueCounts)
        for i in range(len(individualHead["data"])):
            batchLogits.append(logits[:,:,i])
            batchCounts.append(trueCounts[:,:,i])
        #These tensors are all of shape (batch × outputLength)
        #Concatenate them into long tensors of shape (batch × (outputLength × numTasksInLoss))
        concatLogits = tf.concat(batchLogits, axis=1)
        concatCounts = tf.concat(batchCounts, axis=1)
        #For the multinomial, we need to know the total counts for each example in the batch.
        totalCounts = tf.reduce_sum(concatCounts, axis=1)
        #totalCounts should have shape (batch)
        #Now evaluate the multinomial.
        distribution = tfp.distributions.Multinomial(total_count = totalCounts, 
                logits = concatLogits)
        logprobs = distribution.log_prob(concatCounts)
        batchSize = tf.shape(totalCounts)[0]
        sumProbs = tf.reduce_sum(logprobs)
        curLoss = -sumProbs / tf.cast(batchSize, dtype=tf.float32)
        totalLoss = curLoss * individualHead["profile-loss-weight"]
        #tf.print("loss ", totalLoss)
        return totalLoss
    #Return the function itself.
    return multinomialNll
"""
