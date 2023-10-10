import logging
import numpy as np
import scipy


def setMemoryGrowth():
    """Turn on the tensorflow option to grow memory usage as needed, instead
       of allocating the whole GPU. All of the main programs in BPReveal
       do this, so that you can use your GPU for other stuff as you work
       with models."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.debug("GPU memory growth enabled.")
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
    logging.basicConfig(level=levelMap[userLevel],
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.debug("Logger configured.")


def oneHotEncode(sequence):
    """Converts the string sequence into a one-hot encoded numpy array.
       The returned array will have shape (len(sequence), 4).
       The columns are, in order, A, C, G, and T.
       This function will error out if your sequence contains any
       characters other than ACGTacgt, so N nucleotides are rejected."""

    ret = np.empty((len(sequence), 4), dtype='int8')
    ordSeq = np.fromstring(sequence, np.int8)
    ret[:, 0] = (ordSeq == ord("A")) + (ordSeq == ord('a'))
    ret[:, 1] = (ordSeq == ord("C")) + (ordSeq == ord('c'))
    ret[:, 2] = (ordSeq == ord("G")) + (ordSeq == ord('g'))
    ret[:, 3] = (ordSeq == ord("T")) + (ordSeq == ord('t'))
    assert (np.sum(ret) == len(sequence)), \
        "Sequence contains unrecognized nucleotides. Maybe your sequence contains 'N'?"
    return ret


def oneHotDecode(oneHotSequence):
    """Given an array representing a one-hot encoded sequence, convert it back
    to a string. The input shall have shape (sequenceLength, 4), and the output
    will be a Python string. """
    # Convert to an int8 array, since if we get floating point
    # values, the chr() call will fail.
    oneHotArray = oneHotSequence.astype(np.uint8)
    #ret = np.zeros((oneHotArray.shape[0], ), dtype=np.uint8)

    ret = oneHotArray[:, 0] * ord('A') + \
          oneHotArray[:, 1] * ord('C') + \
          oneHotArray[:, 2] * ord('G') + \
          oneHotArray[:, 3] * ord('T')
    return ret.tobytes().decode('ascii')


def logitsToProfile(logitsAcrossSingleRegion, logCountsAcrossSingleRegion):
    """
    Purpose: Given a single task and region sequence prediction (position x channels),
        convert output logits/logcounts to human-readable representation of profile prediction.
    """
    # Logits will have shape (output-width x numTasks)
    assert len(logitsAcrossSingleRegion.shape) == 2
    # If the logcounts passed in is a float, this will break.
    # assert len(logCountsAcrossSingleRegion.shape) == 1  # Logits will be a scalar value

    profileProb = scipy.special.softmax(logitsAcrossSingleRegion)
    profile = profileProb * np.exp(logCountsAcrossSingleRegion)
    return profile


class BatchPredictor:
    """This is a utility class for when you need to make lots of predictions,
    and you may be generating sequences dynamically. Here's how it works.
    You first create a predictor by calling BatchPredictor(modelName, batchSize).
    If you're not sure, a batch size of 64 is probably good.

    Now, you submit any sequences you want predicted, using the submit methods.

    Once you've submitted all of your sequences, you can get your results with
    the getOutput() method.

    Note that the getOutput() method returns *one* result at a time, and
    you have to call getOutput() once for every time you called one of the
    submit methods. """
    def __init__(self, modelFname, batchSize):
        """Starts up the BatchPredictor. This will load your model,
        and get ready to make predictions.
        modelFname is the name of the BPReveal model that you want
        to make predictions from. It's the same name you give for
        the model in any of the other BPReveal tools.
        batchSize is the number of samples that should be run simultaneously through the model."""
        logging.debug("Creating batch predictor.")
        from keras.models import load_model
        import losses
        from collections import deque

        self._model = load_model(modelFname,
                custom_objects={"multinomialNll": losses.multinomialNll})
        logging.debug("Model loaded.")
        self._batchSize = batchSize
        # Since I'll be putting things in and taking them out often,
        # I'm going to use a queue data structure, where those operations
        # are efficient.
        self._inQueue = deque()
        self._outQueue = deque()
        self._inWaiting = 0
        self._outWaiting = 0

    def clear(self):
        """If you've left your predictor in some weird state, you can reset it
        by calling clear(). This empties all the queues."""
        self._inQueue.clear()
        self._outQueue.clear()
        self._inWaiting = 0
        self._outWaiting = 0

    def submitOHE(self, sequence, label):
        """Sequence is an (input-length x 4) ndarray containing the
        one-hot encoded sequence to predict.
        label is any object, and it will be returned with the prediction."""
        self._inQueue.appendleft((sequence, label))
        self._inWaiting += 1
        if self._inWaiting > self._batchSize * 64:
            # We have a ton of sequences to run, so go ahead
            # and run a batch real quick.
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
        by the submit functions, and it will also be called if you ask
        for output and the output queue is empty (assuming there are
        sequences waiting in the input queue.)"""
        if self._inWaiting == 0:
            # There are no samples to process right now, so return
            # (successfully) immediately.
            logging.info("runBatch was called even though there was nothing to do.")
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
        writeHead = 1
        while self._inWaiting:
            nextElem = self._inQueue.pop()
            modelInputs[writeHead] = nextElem[0]
            labels.append(nextElem[1])
            writeHead += 1
            self._inWaiting -= 1
        preds = self._model.predict(modelInputs[:numSamples, :, :],
                                    verbose=0,
                                    batch_size=self._batchSize)
        # I now need to parse out the shape of the prediction toa
        # generate the correct outputs.
        numHeads = len(preds) // 2  # Two predictions (logits & logcounts)
                                    # for each head.
        # The output from the prediction is an awkward shape for
        # decomposing the batch.
        # Each head produces a logits tensor of
        # (batch-size x output-length x num-tasks)
        # and a logcounts tensor of (batch-size,)
        # but I want to return something for each batch.
        # So I'll mimic a batch size of one.
        # Note that I'm collapsing the batch dimension out,
        # so you don't have to always have a [0] index to
        # indicate the first element of the batch.

        for i in range(numSamples):
            curHeads = []
            # The logits come first.
            for j in range(numHeads):
                curHeads.append(preds[j][i])
            # and then the logcounts. For ease of processing,
            # I'm converting the logcounts to a float, rather than
            # a scalar value inside a numpy array.
            for j in range(numHeads):
                curHeads.append(float(preds[j + numHeads][i]))
            self._outQueue.appendleft((curHeads, labels[i]))
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
        The first element will be a list of length numHeads*2, representing the
        output from the model. Since the output of the model will always have
        a dimension representing the batch size, and this function only returns
        the result of running a single sequence, the dimension representing
        the batch size is removed. In other words, running the model on a
        single example would give a logits output of shape
        (1 x output-length x num-tasks).
        But this function will remove that, so you will get an array of shape
        (output-length x numTasks)
        As with calling the model directly, the first numHeads elements are the
        logits arrays, and then come the logcounts for each head.
        You can pass the logits and logcounts values to utils.logitsToProfile
        to get your profile.
        Going back to the outer tuple, the second element will be the label you
        passed in with the original sequence.
        Graphically:
        ( [<head-1-logits>, <head-2-logits>, ...
           <head-1-logcounts>, <head-2-logcounts>, ...
          ],
          label)
        """
        if not self._outWaiting:
            if self._inWaiting:
                # There are inputs that have not been processed. Run the batch.
                self.runBatch()
            else:
                assert False, "There are no outputs ready, and the input queue is empty."
        ret = self._outQueue.pop()
        self._outWaiting -= 1
        return ret
