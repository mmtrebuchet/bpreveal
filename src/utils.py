import logging
import multiprocessing
import numpy as np
import scipy
import numpy.typing as npt
import typing
import tqdm

ONEHOT_T = np.uint8
ONEHOT_AR_T = npt.NDArray[ONEHOT_T]
PRED_T = np.float32
PRED_AR_T = npt.NDArray[PRED_T]

# Store importance scores with 16 bits of precision. Since importance scores
# (particularly PISA values) take up a lot of space, I use a small floating point type
# and compression to mitigate the amount of data.
IMPORTANCE_T = np.float16

# Inside the models, we use floating point numbers to represent one-hot sequences.
# For reasons I don't understand, setting this to uint8 DESTROYS pisa values.
MODEL_ONEHOT_T = np.float32

# The type used to represent cwms and pwms, and also the type used by the jaccard code.
# If you change this, be sure to change libJaccard.c and libJaccard.pyf (and run make)
# so that the jaccard library uses the correct data type.
MOTIF_FLOAT_T = np.float32


# When saving large hdf5 files, store the data in compressed chunks.
# This constant sets the number of entries in each chunk that gets compressed.
# For good performance, whenever you read a compressed hdf5 file, it really helps
# if you read out whole chunks at a time and buffer them. See, for example,
# shapToBigwig.py for an example of a chunked reader.
H5_CHUNK_SIZE = 128

# In parallel code, if something goes wrong, a queue could stay stuck forever.
# Python's queues have a nifty timeout parameter so that they'll crash if they wait
# too long. If a queue has been blocking for longer than this timeout, have the
# program crash.
QUEUE_TIMEOUT = 60  # (seconds)


# This gets set to True if you use any of the tensorflow-importing functions in this
# file. If you import tensorflow in a parent process, child processes will not be able
# to use tensorflow, because tensorflow is dumb like that. Tools like the easy® functions
# and the threaded batcher check to see if Tensorflow has been loaded in the parent process
# before they spawn children.
GLOBAL_TENSORFLOW_LOADED = False


def loadModel(modelFname: str):
    """A simple wrapper to load up a BPReveal model.
    modelFname is the name of the model that Keras saved earlier, typically
    a directory.
    Returns a Keras Model object.
    The returned model does NOT support additional training, since it uses a
    dummy loss."""
    from keras.models import load_model
    from bpreveal.losses import multinomialNll, dummyMse
    model = load_model(modelFname,
                       custom_objects={"multinomialNll": multinomialNll,
                                       "reweightableMse": dummyMse})
    global GLOBAL_TENSORFLOW_LOADED
    GLOBAL_TENSORFLOW_LOADED = True
    return model


def setMemoryGrowth() -> None:
    """Turn on the tensorflow option to grow memory usage as needed, instead
       of allocating the whole GPU. All of the main programs in BPReveal
       do this, so that you can use your GPU for other stuff as you work
       with models."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.debug("GPU memory growth enabled.")
    except Exception as inst:
        logging.warning(str(inst))
        logging.warning("Not using GPU")
        pass
    global GLOBAL_TENSORFLOW_LOADED
    GLOBAL_TENSORFLOW_LOADED = True


def limitMemoryUsage(fraction: float, offset: float) -> None:
    # Limit tensorflow to use only the given fraction of memory.
    assert 0.0 < fraction < 1.0, "Must give a memory fraction between 0 and 1."
    import os
    import subprocess as sp
    import re
    free = total = 0.0
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Do we have a MIG GPU? If so, I need to get its memory available from its name.
        if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 10:
            if os.environ["CUDA_VISIBLE_DEVICES"][:3] == "MIG":
                # Yep, it's a MIG. Grab its properties from nvidia-smi.
                logging.debug("Found a MIG card, attempting to guess memory "
                              "available based on name.")
                cmd = ["nvidia-smi", "-L"]
                ret = sp.run(cmd, capture_output=True)
                lines = ret.stdout.decode('utf-8').split('\n')
                matchRe = re.compile(r".*MIG.* ([0-9]+)g\.([0-9]+)gb.*{0:s}.*".format(
                                     os.environ["CUDA_VISIBLE_DEVICES"]))
                if (smiOut := re.match(matchRe, lines[1])):
                    total = free = float(smiOut[2]) * 1024  # Convert to MiB
                    logging.debug("Found {0:f} GB of memory.".format(total))
                else:
                    assert False, "Could not parse nvidia-smi line: " + lines[1]
    if total == 0.0:
        # We didn't find memory in CUDA_VISIBLE_DEVICES.
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv"]
        ret = sp.run(cmd, capture_output=True)
        line = ret.stdout.decode('utf-8').split('\n')[1]
        logging.debug("Memory usage limited based on {0:s}".format(line))
        lsp = line.split(' ')
        total = float(lsp[0])
        free = float(lsp[2])
    assert total * fraction < free, "Attempting to request more memory than is free!"

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    useMem = int(total * fraction - offset)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=useMem)])
    logging.debug("Configured gpu with {0:d} MiB of memory.".format(useMem))
    global GLOBAL_TENSORFLOW_LOADED
    GLOBAL_TENSORFLOW_LOADED = True


def loadChromSizes(fname: str) -> dict[str, int]:
    # Read in a chrom sizes file and return a dictionary mapping chromosome name → size
    ret = dict()
    with open(fname, 'r') as fp:
        for line in fp:
            if (len(line) > 2):
                chrom, size = line.split()
                ret[chrom] = int(size)
    return ret


def setVerbosity(userLevel: str) -> None:
    levelMap = {"CRITICAL": logging.CRITICAL,
                "ERROR": logging.ERROR,
                "WARNING": logging.WARNING,
                "INFO": logging.INFO,
                "DEBUG": logging.DEBUG}
    logging.basicConfig(level=levelMap[userLevel],
        format='%(levelname)s : %(asctime)s : %(filename)s:%(lineno)d : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.debug("Logger configured.")


def wrapTqdm(iterable, logLevel: str | int = logging.INFO, **tqdmKwargs) -> tqdm.tqdm:
    """Sometimes, you want to display a tqdm progress bar only if the logging level is
    high. Call this with something you want to iterate over OR an integer giving the
    total number of things that will be processed
    (correspoinding to
    pbar = tqdm.tqdm(total=10000)
    while condition:
        pbar.update()
    )
    If iterable is an integer, then this will return a tqdm that you need to
    call update() on, otherwise it'll return something you can use as a loop iterable.

    logLevel may either be a level from the logging module (like logging.INFO) or a
    string naming the log level (like "info")
    """
    if type(logLevel) is str:
        # We were given a string, so convert that to a logging level.
        levelMap = {"CRITICAL": logging.CRITICAL,
                    "ERROR": logging.ERROR,
                    "WARNING": logging.WARNING,
                    "INFO": logging.INFO,
                    "DEBUG": logging.DEBUG}
        logLevel = levelMap[logLevel.upper()]

    class dummyPbar:
        """This serves as a tqdm-like object that doesn't print anything."""

        def update(self):
            pass

        def close(self):
            pass

    if type(iterable) is int:
        if logging.root.isEnabledFor(logLevel):  # type: ignore
            return tqdm.tqdm(total=iterable, **tqdmKwargs)
        else:
            return dummyPbar  # type: ignore
    else:
        if logging.root.isEnabledFor(logLevel):  # type: ignore
            return tqdm.tqdm(iterable, **tqdmKwargs)
        else:
            return iterable


def oneHotEncode(sequence: str, allowN: bool = False) -> ONEHOT_AR_T:
    """Converts the string sequence into a one-hot encoded numpy array.
       The returned array will have shape (len(sequence), 4).
       The columns are, in order, A, C, G, and T.
       This function will error out if your sequence contains any
       characters other than ACGTacgt, so N nucleotides are rejected.
       If allowN is set to True, then any letters that are not in
       "ACGTacgt" will be encoded to [0, 0, 0, 0].
       if allowN is False (the default) then any characters other than
       "ACGTacgt" will raise an AssertionError.
       The mapping is as follows:
       Aa → [1, 0, 0, 0]
       Cc → [0, 1, 0, 0]
       Gg → [0, 0, 1, 0]
       Tt → [0, 0, 0, 1]
       Other → [0, 0, 0, 0]
    """
    if allowN:
        initFunc = np.zeros
    else:
        initFunc = np.empty
    ret = initFunc((len(sequence), 4), dtype=ONEHOT_T)
    ordSeq = np.fromstring(sequence, np.int8)  # type:ignore
    ret[:, 0] = (ordSeq == ord("A")) + (ordSeq == ord('a'))
    ret[:, 1] = (ordSeq == ord("C")) + (ordSeq == ord('c'))
    ret[:, 2] = (ordSeq == ord("G")) + (ordSeq == ord('g'))
    ret[:, 3] = (ordSeq == ord("T")) + (ordSeq == ord('t'))
    if not allowN:
        assert (np.sum(ret) == len(sequence)), \
            "Sequence contains unrecognized nucleotides. Maybe your sequence contains 'N'?"
    return ret


def oneHotDecode(oneHotSequence: np.ndarray) -> str:
    """Given an array representing a one-hot encoded sequence, convert it back
    to a string. The input shall have shape (sequenceLength, 4), and the output
    will be a Python string.
    The decoding is performed based on the following mapping:
    [1, 0, 0, 0] → A
    [0, 1, 0, 0] → C
    [0, 0, 1, 0] → G
    [0, 0, 0, 1] → T
    [0, 0, 0, 0] → N

    """
    # Convert to an int8 array, since if we get floating point
    # values, the chr() call will fail.
    oneHotArray = oneHotSequence.astype(ONEHOT_T)

    ret = \
        oneHotArray[:, 0] * ord('A') + \
        oneHotArray[:, 1] * ord('C') + \
        oneHotArray[:, 2] * ord('G') + \
        oneHotArray[:, 3] * ord('T')
    # Anything that was not encoded is N.
    ret[ret == 0] = ord('N')
    return ret.tobytes().decode('ascii')


def easyPredict(sequences: typing.Iterable[str] | str, modelFname: str) -> PRED_AR_T:
    """
    The easy (and very slow!) way to make predictions with your model.
    Spawns a separate process to make a single batch of predictions,
    then shuts it down. Why make it complicated? Because it frees the
    GPU after it's done so other programs and stuff can use it.
    Arguments:
        sequences is a list of strings that the model will make
        predictions with. If you pass in a single string containing
        a sequence to predict on, that's okay, too. sequences should be as
        long as the receptive field of your model.
        modelFname is the name of the Keras model that should be used.
    Returns an array of profiles, unless you passed in a string for sequences,
    in which case it returns a single array containing the predicted profile.
    The shape of the array will be (numSequences x outputLength) if you passed
    in an iterable of strings (like a list of strings), and it will be
    (outputLength,) if you passed in a single string.
    """
    singleReturn = False
    assert not GLOBAL_TENSORFLOW_LOADED, "Cannot use easy functions after loading tensorflow."

    if type(sequences) is str:
        sequences = [sequences]
        singleReturn = True
    else:
        # In case we got some weird iterable, turn it into a list.
        sequences = list(sequences)

    def predictThread(seqs, model, outQueue):
        setMemoryGrowth()
        remainingToRead = len(seqs)
        bp = BatchPredictor(model, 64)
        preds = []
        for s in seqs:
            bp.submitString(s, 1)
            while bp.outputReady():
                preds.append(bp.getOutput())
                remainingToRead -= 1
        for _ in range(remainingToRead):
            preds.append(bp.getOutput())
        for p in preds:
            logits, logcounts = p[0]
            outQueue.put(logitsToProfile(logits, logcounts))
    import multiprocessing as mp
    resultQueue = mp.Queue()
    # daemon=True makes it so that the child process dies if the parent dies
    # which is useful if something goes wrong.
    proc = mp.Process(target=predictThread,
                      args=(sequences, modelFname, resultQueue),
                      daemon=True)
    proc.start()
    ret = []
    for _ in range(len(sequences)):
        ret.append(resultQueue.get())
    proc.join()
    if singleReturn:
        return ret[0]
    else:
        return np.array(ret, dtype=PRED_T)


def easyInterpretFlat(sequences: typing.Iterable[str] | str, modelFname: str,
                      heads: int, headID: int, taskIDs: list[int],
                      numShuffles: int = 20, kmerSize: int = 1,
                      keepHypotheticals: bool = False) -> dict[str, npt.NDArray]:
    """Spins up an entire interpret pipeline just to interpret your sequences.
    You should only use this for quick one-off things since it is EXTREMELY
    slow.
    sequences is a list (or technically any Iterable) of strings, and the
        returned importance scores will be in an order that corresponds
        to your sequences.
        You can also provide just one string, in which case the return type
        will change: The first (length-one) dimension will be stripped.
    modelFname is the name of the BPReveal model on disk.
    heads is the TOTAL number of heads that the model has.
    headID is the index of the head of the model that you want interpreted.
    taskIDs is the list of tasks that should be included in the profile score
        calculation. For most cases, you'd want a list of all the tasks,
        like [0,1].
    numShuffles is the number of shuffled sequences that are used to calculate
        shap values.
    kmerSize is the length of kmers for which the distribution should be preserved
        during the shuffle. If 1, shuffle each base independently. If 2, preserve
        the distribution of dimers, etc.
    keepHypotheticals controls whether the output contains hypothetical
        contribution scores or just the actual ones.

    Returns a dict.
    If you passed in an iterable of strings (like a list), then the output's first
    dimension will be the number of sequences and it will depend on keepHypotheticals:
        If keepHypotheticals is True, then it will be structured so:
            {"profile": array of shape (numSequences x inputLength x 4),
             "counts": array of shape (numSequences x inputLength x 4),
             "sequence": array of shape (numSequences x inputLength x 4)}
            This dict has the same meaning as shap scores stored in an
            interpretFlat hdf5.

        If keepHypotheticals is False (the default), then the
            shap scores will be condensed down to the normal scores that we plot
            in a genome browser.:
            {"profile": array of shape (numSequences x inputLength),
             "counts": array of shape (numSequences x inputLength)}
    However, if sequences was a string instead of an iterable, then the numSequences
    dimension will be suppressed:
        For keepHypotheticals == True, you get
            {"profile": array of shape (inputLength x 4),
             "counts": array of shape (inputLength x 4),
             "sequence": array of shape (inputLength x 4)}
        and if keepHypotheticals is False, you get:
            {"profile": array of shape (inputLength,),
             "counts": array of shape (inputLength,)}

    """
    from bpreveal.interpretUtils import ListGenerator, FlatListSaver, FlatRunner
    assert not GLOBAL_TENSORFLOW_LOADED, "Cannot use easy functions after loading tensorflow."
    logging.debug("Starting interpretation of sequences.")
    singleReturn = False
    if type(sequences) is str:
        sequences = [sequences]
        singleReturn = True
    generator = ListGenerator(sequences)
    profileSaver = FlatListSaver(generator.numSamples, generator.inputLength)
    countsSaver = FlatListSaver(generator.numSamples, generator.inputLength)
    batcher = FlatRunner(modelFname, headID, heads, taskIDs, 1, generator,
                         profileSaver, countsSaver, numShuffles, kmerSize)
    batcher.run()
    logging.debug("Interpretation complete. Organizing outputs.")
    if keepHypotheticals:
        if singleReturn:
            return {"profile": profileSaver.shap[0],
                    "counts": countsSaver.shap[0],
                    "sequence": profileSaver.seq[0]}
        else:
            return {"profile": profileSaver.shap,
                    "counts": countsSaver.shap,
                    "sequence": profileSaver.seq}
    # Collapse down the hypothetical importances.
    profileOneHot = profileSaver.shap * profileSaver.seq
    countsOneHot = countsSaver.shap * countsSaver.seq
    profile = np.sum(profileOneHot, axis=2)
    counts = np.sum(countsOneHot, axis=2)
    if singleReturn:
        return {"profile": profile[0], "counts": counts[0]}
    else:
        return {"profile": profile, "counts": counts}


def logitsToProfile(logitsAcrossSingleRegion: npt.NDArray,
                    logCountsAcrossSingleRegion: float) -> npt.NDArray[np.float32]:
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
    return profile.astype(np.float32)


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

    def __init__(self, modelFname: str, batchSize: int, start: bool = True) -> None:
        """Starts up the BatchPredictor. This will load your model,and get
        ready to make predictions.
        modelFname is the name of the BPReveal model that you want to make
        predictions from. It's the same name you give for the model in any of
        the other BPReveal tools.

        batchSize is the number of samples that should be run simultaneously
        through the model.
        start is unused, but present here to give BatchPredictor the same API
        as ThreadedBatchPredictor.
        """
        logging.debug("Creating batch predictor.")
        from collections import deque
        if not GLOBAL_TENSORFLOW_LOADED:
            # We haven't loaded Tensorflow yet.
            setMemoryGrowth()
        self._model = loadModel(modelFname)  # type: ignore
        logging.debug("Model loaded.")
        self._batchSize = batchSize
        # Since I'll be putting things in and taking them out often,
        # I'm going to use a queue data structure, where those operations
        # are efficient.
        self._inQueue = deque()
        self._outQueue = deque()
        self._inWaiting = 0
        self._outWaiting = 0
        del start  # We don't refer to start.

    def __enter__(self):
        """Using this batcher with a context manager does nothing."""
        pass

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        """If this batcher was used in a context manager, exiting does nothing, but raises
        any exceptions that happened."""
        if exceptionType is not None:
            raise Exception(exceptionType + "\n" + exceptionValue + '\n' + exceptionTraceback)

    def clear(self) -> None:
        """If you've left your predictor in some weird state, you can reset it
        by calling clear(). This empties all the queues."""
        self._inQueue.clear()
        self._outQueue.clear()
        self._inWaiting = 0
        self._outWaiting = 0

    def submitOHE(self, sequence: ONEHOT_AR_T, label: typing.Any) -> None:
        """Sequence is an (input-length x 4) ndarray containing the
        one-hot encoded sequence to predict.
        label is any object, and it will be returned with the prediction."""
        self._inQueue.appendleft((sequence, label))
        self._inWaiting += 1
        if self._inWaiting > self._batchSize * 64:
            # We have a ton of sequences to run, so go ahead
            # and run a batch real quick.
            self.runBatch()

    def submitString(self, sequence: str, label: typing.Any) -> None:
        """Submits a given sequence for prediction.
        sequence is a string of length input-length, and
        label is any object. Label will be returned to you with the
        prediction. """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def runBatch(self, maxSamples: int | None = None) -> None:
        """This actually runs the batch. Normally, this will be called
        by the submit functions, and it will also be called if you ask
        for output and the output queue is empty (assuming there are
        sequences waiting in the input queue.)
        maxSamples, if provided is the maximum number of samples to
        run in this batch. It should probably be a multiple of the
        batch size."""
        if self._inWaiting == 0:
            # There are no samples to process right now, so return
            # (successfully) immediately.
            logging.info("runBatch was called even though there was nothing to do.")
            return
        if maxSamples is None:
            numSamples = self._inWaiting
        else:
            numSamples = min(self._inWaiting, maxSamples)
        labels = []
        firstElem = self._inQueue.pop()
        labels.append(firstElem[1])
        self._inWaiting -= 1
        # I need to determine the input length, and I'll do that by looking at
        # the first sequence.
        inputLength = firstElem[0].shape[0]
        modelInputs = np.zeros((numSamples, inputLength, 4), dtype=ONEHOT_T)
        modelInputs[0] = firstElem[0]
        # With that ugliness out of the way, now I just populate the rest of
        # the prediction table.
        writeHead = 1
        for i in range(numSamples - 1):
            nextElem = self._inQueue.pop()
            modelInputs[writeHead] = nextElem[0]
            labels.append(nextElem[1])
            writeHead += 1
            self._inWaiting -= 1
        preds = self._model.predict(modelInputs[:numSamples, :, :],
                                    verbose=0,  # type: ignore
                                    batch_size=self._batchSize)
        # I now need to parse out the shape of the prediction toa
        # generate the correct outputs.
        numHeads = len(preds) // 2  # Two predictions (logits & logcounts) for each head.
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

    def outputReady(self) -> bool:
        """Is there any output ready for you? Returns True if a prediction has been
        made and is waiting for you in the output queue, and False otherwise."""
        return self._outWaiting > 0

    def empty(self) -> bool:
        """Can you get output from this Batcher? Returns False if calling outputReady()
        would give you data. This will be the case if there's data in the input queue
        but the prediction hasn't been run."""
        return self._outWaiting == 0 and self._inWaiting == 0

    def getOutput(self) -> tuple[list, typing.Any]:
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


class ThreadedBatchPredictor:
    """Mirrors the API of the BachPredictor class, but runs predictions in a
    separate thread. This can give you a performance boost, and also lets you
    shut down the predictor thread when you don't need it.
    Supports the with statement to only turn on the batcher when you're using it,
    or you can leave it running in the background.

    Usage examples:

    predictor = utils.ThreadedBatchPredictor(modelFname, 64, start=True)
    # Use as you would a normal batchPredictor
    # When not needed any more:
    predictor.stop()

    Alternatively, you can use this as a context manager.
    predictor = utils.ThreadedBatchPredictor(modelFname, 64, start=False)

    with predictor:
        # use as a normal BatchPredictor.
    # On leaving the context, the predictor is shut down.

    The batcher guarantees that the order in which you get results is the same as
    the order you submitted them in, but this could change in the future!
    """

    def __init__(self, modelFname: str, batchSize: int, start: bool = False,
                 numThreads: int = 1) -> None:
        """Starts up the BatchPredictor. This will load your model,
        and get ready to make predictions.
        modelFname is the name of the BPReveal model that you want
        to make predictions from. It's the same name you give for
        the model in any of the other BPReveal tools.
        batchSize is the number of samples that should be run simultaneously through the model."""
        logging.debug("Creating threaded batch predictor.")
        self._batchSize = batchSize
        self._modelFname = modelFname
        self._batchSize = batchSize
        # Since I'll be putting things in and taking them out often,
        # I'm going to use a queue data structure, where those operations
        # are efficient.
        self._batchers = None
        self._numThreads = numThreads
        self.running = False
        if start:
            self.start()

    def __enter__(self):
        """Used in a context manager, this is the first thing that gets called
        inside a with statement."""
        self.start()

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        """When leaving a context manager's with statement, shut down the batcher."""
        self.stop()
        if exceptionType is not None:
            return False
        del exceptionValue  # Disable unused warning
        del exceptionTraceback  # Disable unused warning

    def start(self):
        """Spin up the batcher thread. If you submit sequences without starting the batcher,
        this method will be called automatically (with a warning).
        """
        if not self.running:
            logging.debug("Starting threaded batcher.")
            assert self._batchers is None, "Attempting to start a new batcher when an old "\
                "one is still alive." + str(self._batchers)
            self._inQueues = []
            self._outQueues = []
            self._batchers = []

            for _ in range(self._numThreads):
                nextInQueue = multiprocessing.Queue(10000)
                nextOutQueue = multiprocessing.Queue(10000)
                self._inQueues.append(nextInQueue)
                self._outQueues.append(nextOutQueue)
                nextBatcher = multiprocessing.Process(target=_batcherThread,
                    args=(self._modelFname, self._batchSize, nextInQueue, nextOutQueue),
                    daemon=True)
                nextBatcher.start()
                self._batchers.append(nextBatcher)
            self._inFlight = 0
            self._inQueueIdx = 0
            self.running = True
            from collections import deque
            self._outQueueOrder = deque()
        else:
            logging.warning("Attempted to start a batcher that was already running.")

    def __del__(self):
        """General cleanup - kill off the child process when this object goes out of scope."""
        logging.debug("Destructor called.")
        if self.running:
            self.stop()

    def stop(self):
        """Shut down the processor thread."""
        if self.running:
            logging.debug("Shutting down threaded batcher.")
            if self._batchers is None:
                assert False, "Attempting to shut down a running ThreadedBatchPredictor" \
                    "When its _batchers is None"
            for i in range(self._numThreads):
                self._inQueues[i].put("shutdown")
                self._inQueues[i].close()
                self._batchers[i].join(1)  # Wait one second.
                if self._batchers[i].exitcode is None:
                    # The process failed to die. Kill it more forcefully.
                    self._batchers[i].terminate()
                self._batchers[i].join(1)  # Wait one second.
                self._batchers[i].close()
                self._outQueues[i].close()
            del self._inQueues
            del self._batchers
            del self._outQueues
            # Explicitly set None so that start won't panic.
            self._batchers = None
            self.running = False
        else:
            logging.warning("Attempting to stop a batcher that is already stopped.")

    def clear(self):
        """Resets the batcher, emptying any queues and reloading the model."""
        if self.running:
            self.stop()
        self.start()

    def submitOHE(self, sequence: ONEHOT_AR_T, label: typing.Any) -> None:
        """Sequence is an (input-length x 4) ndarray containing the
        one-hot encoded sequence to predict.
        label is any object, and it will be returned with the prediction."""
        if not self.running:
            logging.warning("Submitted a query when the batcher is stopped. Starting.")
            self.start()
        q = self._inQueues[self._inQueueIdx]
        query = (sequence, label)
        q.put(query, True, QUEUE_TIMEOUT)
        self._outQueueOrder.appendleft(self._inQueueIdx)
        self._inQueueIdx = (self._inQueueIdx + 1) % self._numThreads
        self._inFlight += 1

    def submitString(self, sequence: str, label: typing.Any) -> None:
        """Submits a given sequence for prediction.
        sequence is a string of length input-length, and
        label is any object. Label will be returned to you with the
        prediction. """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def outputReady(self) -> bool:
        """Is there any output ready for you? Returns True if the batcher is sitting
        on results, and False otherwise."""
        if self._inFlight:
            outIdx = self._outQueueOrder[-1]
            return not self._outQueues[outIdx].empty()
        return False

    def empty(self) -> bool:
        """Is it possible to getOutput()? This will be true if predictions haven't
        been made yet, but they would be made if you asked for output."""
        return self._inFlight == 0

    def getOutput(self) -> tuple[list, typing.Any]:
        """Same semantics as BatchPredictor.getOutput."""
        nextQueueIdx = self._outQueueOrder.pop()
        if self._outQueues[nextQueueIdx].empty():
            if self._inFlight:
                # There are inputs that have not been processed. Run the batch.
                self._inQueues[nextQueueIdx].put("finishBatch")
            else:
                assert False, "There are no outputs ready, and the input queue is empty."
        ret = self._outQueues[nextQueueIdx].get(True, QUEUE_TIMEOUT)
        self._inFlight -= 1
        return ret


def _batcherThread(modelFname, batchSize, inQueue, outQueue):
    assert not GLOBAL_TENSORFLOW_LOADED, "Cannot use the threaded predictor " \
        "after loading tensorflow."
    logging.debug("Starting subthread")
    import queue
    setMemoryGrowth()
    # Instead of reinventing the wheel, the thread that actually runs the batches
    # just creates a BatchPredictor.
    batcher = BatchPredictor(modelFname, batchSize)
    predsInFlight = 0
    numWaits = 0
    while True:
        # No timeout because this batcher could be waiting for a very long time to get
        # inputs.
        try:
            inVal = inQueue.get(True, 0.1)
        except queue.Empty:
            numWaits += 1
            # There was no input. Are we sitting on predictions that we could go ahead and make?
            if batcher._inWaiting > batcher._batchSize / numWaits:
                # division by numWeights so that if you wait a long time, it will even
                # run a partial batch.
                # Nope, go ahead and give the batcher a spin while we wait.
                batcher.runBatch(maxSamples=batchSize)
                while not outQueue.full() and batcher.outputReady():
                    outQueue.put(batcher.getOutput(), True, QUEUE_TIMEOUT)
                    predsInFlight -= 1
            continue
        numWaits = 0
        match inVal:
            case (sequence, label):
                if type(sequence) is str:
                    batcher.submitString(sequence, label)
                else:
                    batcher.submitOHE(sequence, label)
                predsInFlight += 1
                # If there's an answer and the out queue can handle it, go ahead
                # and send it.
                while not outQueue.full() and batcher.outputReady():
                    outQueue.put(batcher.getOutput(), True, QUEUE_TIMEOUT)
                    predsInFlight -= 1
            case "finishBatch":
                while predsInFlight:
                    outPred = batcher.getOutput()
                    outQueue.put(outPred, True, QUEUE_TIMEOUT)
                    predsInFlight -= 1
            case "shutdown":
                # End the thread.
                logging.debug("Shutdown signal received.")
                return
