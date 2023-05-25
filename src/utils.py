import logging
import numpy as np
import scipy



def setMemoryGrowth():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.info("GPU memory growth enabled.")
    except Exception:
        logging.warning("Not using GPU")
        pass


def loadChromSizes(fname):
    # Read in a chrom sizes file and return a dictionary mapping chromosome name → size
    ret = dict()
    with open(fname, 'r') as fp:
        for line in fp:
            if (len(line) > 2):
                chrom, size = line.split()
                ret[chrom] = int(size)
    return ret


def setVerbosity(userLevel):
    levelMap = {"CRITICAL": logging.CRITICAL,
                "ERROR": logging.ERROR,
                "WARNING": logging.WARNING,
                "INFO": logging.INFO,
                "DEBUG": logging.DEBUG}
    logging.basicConfig(level=levelMap[userLevel])


def oneHotEncode(sequence):
    ret = np.empty((len(sequence), 4), dtype='int8')
    ordSeq = np.fromstring(sequence, np.int8)
    ret[:, 0] = (ordSeq == ord("A")) + (ordSeq == ord('a'))
    ret[:, 1] = (ordSeq == ord("C")) + (ordSeq == ord('c'))
    ret[:, 2] = (ordSeq == ord("G")) + (ordSeq == ord('g'))
    ret[:, 3] = (ordSeq == ord("T")) + (ordSeq == ord('t'))
    assert (np.sum(ret) == len(sequence)), (sorted(sequence), sorted(ret.sum(axis=1)))
    return ret


def logitsToProfile(logitsAcrossSingleRegion, logCountsAcrossSingleRegion):
    """
    Purpose: Given a single task and region sequence prediction (position x channels),
        convert output logits/logcounts to human-readable representation of profile prediction.
    """
    # Logits will have shape (output-width x numTasks)
    assert len(logitsAcrossSingleRegion.shape) == 2
    assert len(logCountsAcrossSingleRegion.shape) == 1  # Logits will be a scalar value

    profileProb = scipy.special.softmax(logitsAcrossSingleRegion)
    profile = profileProb * np.exp(logCountsAcrossSingleRegion)
    return profile


class BatchPredictor:
    """This is a utility class for when you need to make lots of predictions,
    and you may be generating sequences dynamically. Here's how it works. 
    You first create a predictor by calling BatchPredictor(modelName, batchSize).
    If you're not sure, a batch size of 64 is probably good. 

    Now, you submit any sequences you want predicted, using the submit methods.
    Under the hood, every batchSize-th time you submit a sequence, this class will
    round up those sequences into a batch and run them through the model. 

    Once you've submitted all the sequences you want, you call runBatch() 
    to make sure there aren't any lingering sequences in the input. 

    Now that you've run all of your sequences, you can get your results with
    the getOutput() method.

    Note that the getOutput() method returns *one* result at a time, and 
    you have to call getOutput() once for every time you called one of the 
    submit methods. """
    def __init__(self, modelFname, batchSize):
        """Starts up the BatchPredictor. This will load your model,
        and get ready to make predictions."""
        logging.debug("Creating batch predictor.")
        from keras.models import load_model
        import losses
        from collections import deque

        self._model = load_model(modelFname,
                custom_objects={"multinomialNll" : losses.multinomialNLL})
        logging.debug("Model loaded.")
        self._batchSize = batchSize
        # Since I'll be putting things in and taking them out often,
        # I'm going to use a queue data structure, where those operations
        # are efficient.
        self._inQueue = deque()
        self._outQueue = deque()
        self._inWaiting = 0
        self._outWaiting = 0

    def submitOHE(self, sequence, label):
        """Sequence is an (input-length x 4) ndarray containing the
        one-hot encoded sequence to predict.
        label is any object, and it will be returned with the prediction."""
        self._inQueue.appendLeft((sequence, label))
        self._inWaiting += 1
        if(self._inWaiting > self._batchSize):
            self.runBatch()

    def submitString(self, sequence, label):
        """Submits a given sequence for prediction.
        sequence is a string of length input-length, and
        label is any object. Label will be returned to you with the
        prediction. """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def runBatch(self):
        """This actually runs the batch. Normally, this will be called
        by the submit functions, but you should also call it manually
        once you've submitted all of the sequences you're interested in,
        otherwise you could have sequences left in the input queue."""
        logging.debug("Starting batch run.")
        if(self._inWaiting == 0):
            # There are no samples to process right now, so return
            # (successfully) immediately.
            return
        numSamples = self._inWaiting
        labels = []
        firstElem = self._inQueue.pop()
        labels.append(firstElem[1])
        self._inWaiting -= 1
        # I need to determine the input length, and I'll do that by looking at
        # the first sequence.
        inputLength = firstElem[0].shape[0]
        modelInputs = np.zeros((numSamples, inputLength, 4))
        modelInputs[0] = firstElem[0]
        # With that ugliness out of the way, now I just populate the rest of
        # the prediction table.
        while self._inWaiting:
            nextElem = self._inQueue.pop()
            modelInputs[i+1] = nextElem[0]
            labels.append(nextElem[1])
            self._inWaiting -= 1
        logging.debug("Running batch through model.")
        preds = self._model(modelInputs[:numSamples,:,:])
        # I now need to parse out the shape of the prediction toa
        # generate the correct outputs.
        numHeads = len(preds)//2  # Two predictions (logits & logcounts)
                                  # for each head.
        for i in range(numSamples):
            curHeads = []
            for j in range(numHeads):
                curHeads.append(preds[j][i], preds[j+numHeads][i])
            self._outQueue.appendLeft((curHeads, labels[i]))
            self._outWaiting += 1

    def outputReady(self):
        """Is there any output ready for you? Returns True if you can safely call
        getOutput(), and False otherwise."""
        return self._outWaiting > 0

    def getOutput(self):
        """Returns one of the predictions made by the model.
        This implementation guarantees that predictions will be returned in
        the same order as they were submitted, but you should not rely on that
        in the future. Instead, you should use a label when you submit your
        sequences and use that to determine order.
        The output will be a two-tuple.
        The first element will be a list of length numHeads,
        each element being a two-tuple.
        The first element of *that* tuple will be a (outputLength * numTasks)
        ndarray of the logits. The kth tuple in the list corresponds to head k.
        The second element of the inner tuple is a scalar logcounts value.
        You can pass the logits and logcounts values to utils.logitsToProfile
        to get your profile.
        Going back to the outer tuple, the second element will be the label you
        passed in with the original sequence.
        Graphically:
        ( [ ( <head-1-logits>, <head-1-logcounts> ),
            ( <head-2-logits>, <head-2-logcounts> ),
            ...
          ],
          label)
        Note that this is a VERY different arrangement from the output of calling the
        model directly, but should be more convenient.
        """
        assert self._outWaiting > 0, "You cannot get output when there is none ready. "\
            "Did you forget to call runBatch()?"
        ret = self._outQueue.pop()
        self._outWaiting -= 1
        return ret




