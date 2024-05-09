"""Lots of helpful utilities for working with models."""
from collections import deque
import multiprocessing
import os
import queue
import re
import subprocess as sp
import typing
from collections.abc import Iterable
import h5py
import scipy
import pyBigWig
import pysam
import numpy as np
from bpreveal import logUtils
# Public import so that old code that expects these functions to be here can still
# find them.
from bpreveal.logUtils import setVerbosity, wrapTqdm  # pylint: disable=unused-import  # noqa
from bpreveal.internal.constants import ONEHOT_AR_T, PRED_AR_T, ONEHOT_T, \
    QUEUE_TIMEOUT, LOGCOUNT_T, LOGIT_AR_T, IMPORTANCE_AR_T, IMPORTANCE_T
from bpreveal.internal import constants


def loadModel(modelFname: str):
    """Load up a BPReveal model.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.

    :param modelFname: The name of the model that Keras saved earlier, typically
        a directory.
    :return: A Keras Model object.

    The returned model does NOT support additional training, since it uses a
    dummy loss.

    **Example:**

    .. code-block:: python

        from bpreveal.utils import loadModel
        m = loadModel("path/to/model")
        preds = m.predict(myOneHotSequences)

    """
    # pylint: disable=import-outside-toplevel
    import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
    from keras.models import load_model
    from bpreveal.losses import multinomialNll, dummyMse
    # pylint: enable=import-outside-toplevel
    model = load_model(filepath=modelFname,
                       custom_objects={"multinomialNll": multinomialNll,
                                       "reweightableMse": dummyMse})
    constants.setTensorflowLoaded()
    return model


def setMemoryGrowth() -> None:
    """Turn on the tensorflow option to grow memory usage as needed.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.

    All of the main programs in BPReveal do this, so that you can
    use your GPU for other stuff as you work with models.
    """
    # pylint: disable=import-outside-toplevel, unused-import
    import bpreveal.internal.disableTensorflowLogging  # noqa
    import tensorflow as tf
    # pylint: enable=import-outside-toplevel, unused-import
    gpus = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        logUtils.debug("GPU memory growth enabled.")
    except Exception as inst:  # pylint: disable=broad-exception-caught
        logUtils.warning("Not using GPU")
        logUtils.debug("Because: " + str(inst))
    constants.setTensorflowLoaded()


def loadPisa(fname: str) -> IMPORTANCE_AR_T:
    """Loads up a PISA file, shears it, and crops it to a standard array.

    :param fname: The name of the hdf5-format file on disk, containing your PISA data.
    :return: An array of shape (num-samples, num-samples) containing the sheared PISA data.

    This is probably best demonstrated with an image or two. Here's how PISA data are
    stored in the hdf5 file:

    .. image:: ../../doc/presentations/pisaRaw.png
        :width: 400
        :alt: Unsheared pisa data straight from an hdf5.

    This function first shears the PISA data into a more normal form:

    .. image:: ../../doc/presentations/pisaShear.png
        :width: 400
        :alt: The PISA data has been sheared, where each row is "indented" one pixel more
              than the one above it.

    (In this figure, I've colored pixels where we didn't have any starting data
    dark blue so that they stand out.) There is a lot of wasted space in this
    image. So we crop it by deleting ``receptiveField // 2`` pixels from each
    side:

    .. image:: ../../doc/presentations/pisaLoad.png
        :width: 400
        :alt: The PISA matrix has been cropped on the left and right by half of the
              receptive field.

    This is the output of this function. (except that I have added in the dark
    blue patches where there was no data before shearing - the actual return
    from this function just contains zeros in those regions.) Now, in preparing
    your regions to run PISA, you need to be pretty careful so that the
    coordinate you think you are explaining is actually the one that the PISA
    starts with! Here's a representation of where each base in the sheared
    image comes from, relative to the actual model input:

    .. image:: ../../doc/presentations/pisaDiagram.png
        :width: 400
        :alt: An illustration of where the model input is relative to the PISA output.

    In this figure, the input to the model is shown in black, the model's output is
    shown in red, and output being explained is shown as a blue dot. The green line
    shows the receptive field of the model centered around the output base (This is
    where we have data in the PISA plot).
    I've put some helpful marks on the x-axis that line up with the topmost PISA row.
    If you supply a bed file to the PISA interpretation script, then it will provide
    PISA values where each base in the window is an *output* from the model.
    In other words, the bed file for these data would have started at position '3752.
    This keeps life easy, and it also means that when you use this function, the
    array that gets loaded corresponds exactly to the bed region you used.

    If, however, you use a fasta-format input, things get hairy. The
    fasta-format input must contain enough bases to fill the entire model's
    input (i.e., the black lines in this figure), and so it will include bases
    to the *left* of the output being explained. The number of padding bases
    will be ``receptiveField // 2``. In this case, my receptive field is 2057,
    and so there are 1028 extra bases on the left of the blue output being
    explained.

    Each entry in a fasta file could in principle be a completely different sequence.
    However, to make a comprehensible PISA plot, the sequences will typically
    all be drawn from the same region but offset by one each time.
    In other words, the lines in the fasta file would be the black lines
    in this figure.

    This line diagram represents the uncropped data. The matrix returned
    from this function would start at position '3752 and end at 3752 + numEntries.

    """
    with h5py.File(fname, "r") as fp:
        pisaShap = np.array(fp["shap"])
    pisaVals = np.sum(pisaShap, axis=2)
    numRegions = pisaVals.shape[0]
    receptiveField = pisaVals.shape[1]
    shearMat = np.zeros((numRegions, pisaVals.shape[1] + numRegions),
                        dtype=IMPORTANCE_T)
    for i in range(0, numRegions):
        offset = i
        shearMat[i, offset:offset + pisaVals.shape[1]] = pisaVals[i]
    shearMat = shearMat[:, receptiveField // 2:-receptiveField // 2]
    return shearMat


def limitMemoryUsage(fraction: float, offset: float) -> float:
    """Limit tensorflow to use only the given fraction of memory.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.

    This will allocate ``total-memory * fraction - offset``
    Why use this? Well, for running multiple processes on the same GPU, you don't
    want to have them both allocating all the memory. So if you had two processes,
    you'd do something like::

        def child1():
            utils.limitMemoryUsage(0.5, 1024)
            # Load model, do stuff.

        def child2():
            utils.limitMemoryUsage(0.5, 1024)
            # Load model, do stuff.

        p1 = multiprocessing.Process(target=child1); p1.start()
        p2 = multiprocessing.Process(target=child2); p2.start()

    And now each process will use (1024 MB less than) half the total GPU memory.

    :param fraction: How much of the memory on the GPU can I have?
    :param offset: How much memory (in MB) should be reserved when
        I carve out my fraction?
    :return: The memory (in MB) reserved.
    """
    assert 0.0 < fraction < 1.0, "Must give a memory fraction between 0 and 1."
    free = total = 0.0
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Do we have a MIG GPU? If so, I need to get its memory available from its name.
        if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 10:
            if os.environ["CUDA_VISIBLE_DEVICES"][:3] == "MIG":
                # Yep, it's a MIG. Grab its properties from nvidia-smi.
                logUtils.debug("Found a MIG card, attempting to guess memory "
                              "available based on name.")
                cmd = ["nvidia-smi", "-L"]
                ret = sp.run(cmd, capture_output=True, check=True)
                lines = ret.stdout.decode("utf-8").split("\n")
                devices = os.environ["CUDA_VISIBLE_DEVICES"]
                matchRe = re.compile(fr".*MIG.* ([0-9]+)g\.([0-9]+)gb.*{devices}.*")
                if (smiOut := re.match(matchRe, lines[1])):
                    total = free = float(smiOut[2]) * 1024  # Convert to MiB
                    logUtils.debug(f"Found {total} GB of memory.")
                else:
                    assert False, "Could not parse nvidia-smi line: " + lines[1]
    if total == 0.0:
        # We didn't find memory in CUDA_VISIBLE_DEVICES.
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv"]
        try:
            ret = sp.run(cmd, capture_output=True, check=True)
        except sp.CalledProcessError as e:
            logUtils.error("Problem parsing nvidia-smi output.")
            logUtils.error(e.stdout.decode("utf-8"))
            logUtils.error(e.stderr.decode("utf-8"))
            logUtils.error(e.returncode)
            raise
        line = ret.stdout.decode("utf-8").split("\n")[1]
        logUtils.debug(f"Memory usage limited based on {line}")
        lsp = line.split(" ")
        total = float(lsp[0])
        free = float(lsp[2])
    assert total * fraction < free, "Attempting to request more memory than is free!"

    # pylint: disable=import-outside-toplevel, unused-import
    import bpreveal.internal.disableTensorflowLogging  # noqa
    import tensorflow as tf
    # pylint: enable=import-outside-toplevel, unused-import
    gpus = tf.config.list_physical_devices("GPU")
    useMem = int(total * fraction - offset)
    tf.config.set_logical_device_configuration(
        device=gpus[0],
        logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=useMem)])
    logUtils.debug(f"Configured gpu with {useMem} MiB of memory.")
    constants.setTensorflowLoaded()
    return useMem


def loadChromSizes(chromSizesFname: str | None = None,
                   genomeFname: str | None = None,
                   bwHeader: dict[str, int] | None = None,
                   bw: pyBigWig.pyBigWig | None = None,
                   fasta: pysam.FastaFile | None = None) -> dict[str, int]:
    """Read in a chrom sizes file and return a dictionary mapping chromosome name → size.

    Exactly one of the supplied parameters may be None.

    :param chromSizesFname: The name of a chrom.sizes file on disk.
    :param genomeFname: The name of a genome fasta file on disk.
    :param bwHeader: A dictionary loaded from a bigwig.
        (Using this makes this function a no-op.)
    :param bw: An opened bigwig file.
    :param fasta: An opened genome fasta.

    :return: {"chr1": 1234567, "chr2": 43212567, ...}

    **Example:**

    .. code-block:: python

        from bpreveal.utils import loadChromSizes, blankChromosomeArrays, writeBigwig
        import pysam
        genome = pysam.FastaFile("path/to/genome.fa")
        chromSizeDict = loadChromSizes(fasta=genome)
        chromArs = blankChromosomeArrays(chromSizes=chromSizeDict, numTracks=1)
        myRegionDats = ...  # Some function that returns tuples of (chrom, start, end, data)
        for rChrom, rStart, rEnd, rValues in myRegionDats:
            chromArs[rChrom][rStart:rEnd] = rValues
        writeBigwig(bwFname="path/to/output.bw", chromArs)

    """
    if chromSizesFname is not None:
        ret = {}
        with open(chromSizesFname, "r") as fp:
            for line in fp:
                if len(line) > 2:
                    chrom, size = line.split()
                    ret[chrom] = int(size)
        return ret
    if genomeFname is not None:
        with pysam.FastaFile(genomeFname) as genome:
            chromNames = genome.references
            ret = {}
            for chromName in chromNames:
                ret[chromName] = genome.get_reference_length(chromName)
        return ret
    if bwHeader is not None:
        return bwHeader
    if bw is not None:
        return bw.chroms()
    if fasta is not None:
        chromNames = fasta.references
        ret = {}
        for chromName in chromNames:
            ret[chromName] = fasta.get_reference_length(chromName)
        return ret
    assert False, "You can't ask for chrom sizes without some argument!"


def blankChromosomeArrays(genomeFname: str | None = None,
                          chromSizesFname: str | None = None,
                          bwHeader: dict[str, int] | None = None,
                          chromSizes: dict[str, int] | None = None,
                          bw: pyBigWig.pyBigWig | None = None,
                          fasta: pysam.FastaFile | None = None,
                          dtype: type = np.float32,
                          numTracks: int = 1) -> dict[str, np.ndarray]:
    """Get a set of blank numpy arrays that you can use to save genome-wide data.

    Exactly one of ``chromSizesFname``, ``genomeFname``, ``bwHeader``,
    ``chromSizes``, ``bw``, or ``fasta`` may be None.

    :param chromSizesFname: The name of a chrom.sizes file on disk.
    :param genomeFname: The name of a genome fasta file on disk.
    :param bwHeader: A dictionary loaded from a bigwig.
    :param chromSizes: A dictionary mapping chromosome name to length.
    :param bw: An opened bigwig file.
    :param fasta: An opened genome fasta.
    :param dtype: The type of the arrays that will be returned.
    :param numTracks: How many tracks of data do you have?

    :return: A dict mapping chromosome name to a numpy array.

    The returned dict will have an element for every chromosome in the input.
    The shape of each element of the dictionary will be (chromosome-length, numTracks).

    See :py:func:`loadChromSizes<bpreveal.utils.loadChromSizes>` for an example.
    """
    if chromSizes is None:
        chromSizes = loadChromSizes(genomeFname=genomeFname,
                                    chromSizesFname=chromSizesFname,
                                    bwHeader=bwHeader, bw=bw, fasta=fasta)
    ret = {}
    for chromName in chromSizes.keys():
        newAr = np.zeros((chromSizes[chromName], numTracks), dtype=dtype)
        ret[chromName] = newAr
    return ret


def writeBigwig(bwFname: str, chromDict: dict[str, np.ndarray] | None = None,
                regionList: list[tuple[str, int, int]] | None = None,
                regionData: typing.Any = None,
                chromSizes: dict[str, int] | None = None):
    """Write a bigwig file given some region data.

    You must specify either:

    * ``chromDict``, in which case ``regionList``, ``chromSizes``
        and ``regionData`` must be ``None``, or
    * ``regionList``, ``chromSizes``, and ``regionData``, in which
        case ``chromDict`` must be ``None``.

    :param bwFname: The name of the bigwig file to write.
    :param chromDict: A dict mapping chromosome names to the data for that
        chromosome. The data should have shape (chromosome-length,).
    :param regionList: A list of (chrom, start, end) tuples giving the
        locations where the data should be saved.
    :param regionData: An iterable with the same length as ``regionList``.
        The ith element of ``regionData`` will be
        written to the ith location in ``regionList``.
    :param chromSizes: A dict mapping chromosome name → chromosome size.

    See :py:func:`loadChromSizes<bpreveal.utils.loadChromSizes>` for an example.
    """
    if chromDict is None:
        logUtils.debug("Got regionList, regionData, chromSizes. "
                       "Building chromosome dict.")
        assert chromSizes is not None \
               and regionList is not None \
               and regionData is not None, \
            "Must provide chromSizes, regionList, and regionData if chromDict is None."
        chromDict = blankChromosomeArrays(bwHeader=chromSizes)
        for i, r in enumerate(regionList):
            chrom, start, end = r
            chromDict[chrom][start:end] = regionData[i]
    else:
        chromSizes = {}
        for c in chromDict.keys():
            chromSizes[c] = len(chromDict[c])
    # Now we just write the chrom dict.
    outBw = pyBigWig.open(bwFname, "w")
    logUtils.debug("Starting to write data to bigwig.")
    header = [(x, chromSizes[x]) for x in sorted(list(chromSizes.keys()))]
    outBw.addHeader(header)

    for chromName in sorted(list(chromDict.keys())):
        vals = [float(x) for x in chromDict[chromName]]
        outBw.addEntries(chromName, 0, values=vals,
                         span=1, step=1)
    logUtils.debug("Data written. Closing bigwig.")
    outBw.close()
    logUtils.debug("Bigwig closed.")


def oneHotEncode(sequence: str, allowN: bool = False) -> ONEHOT_AR_T:
    """Convert the string sequence into a one-hot encoded numpy array.

    :param sequence: A DNA sequence to encode.
        May contain uppercase and lowercase letters.
    :param allowN: If False (the default), raise an AssertionError if
        the sequence contains letters other than ``ACGTacgt``.
        If True, any other characters will be encoded as ``[0, 0, 0, 0]``.
    :return: An array with shape (len(sequence), 4).
    :rtype: ONEHOT_AR_T


    The columns are, in order, A, C, G, and T.
    The mapping is as follows::

        A or a → [1, 0, 0, 0]
        C or c → [0, 1, 0, 0]
        G or g → [0, 0, 1, 0]
        T or t → [0, 0, 0, 1]
        Other  → [0, 0, 0, 0]

    **Example:**

    .. code-block:: python

        from bpreveal.utils import oneHotEncode, oneHotDecode
        seq = "ACGTTT"
        x = oneHotEncode(seq)
        print(x)
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 1 0]
        #  [0 0 0 1]
        #  [0 0 0 1]
        #  [0 0 0 1]]
        y = oneHotDecode(x)
        print(y)
        # ACGTTT

    """
    if allowN:
        initFunc = np.zeros
    else:
        # We're going to overwrite every position, so don't bother with
        # initializing the array.
        initFunc = np.empty
    ret = initFunc((len(sequence), 4), dtype=ONEHOT_T)
    ordSeq = np.fromstring(sequence, np.int8)  # type:ignore
    ret[:, 0] = (ordSeq == ord("A")) + (ordSeq == ord("a"))
    ret[:, 1] = (ordSeq == ord("C")) + (ordSeq == ord("c"))
    ret[:, 2] = (ordSeq == ord("G")) + (ordSeq == ord("g"))
    ret[:, 3] = (ordSeq == ord("T")) + (ordSeq == ord("t"))
    if not allowN:
        assert (np.sum(ret) == len(sequence)), \
            "Sequence contains unrecognized nucleotides. "\
            "Maybe your sequence contains 'N'?"
    return ret


def oneHotDecode(oneHotSequence: np.ndarray) -> str:
    """Take a one-hot encoded sequence and turn it back into a string.

    Given an array representing a one-hot encoded sequence, convert it back
    to a string. The input shall have shape (sequenceLength, 4), and the output
    will be a Python string.
    The decoding is performed based on the following mapping::

        [1, 0, 0, 0] → A
        [0, 1, 0, 0] → C
        [0, 0, 1, 0] → G
        [0, 0, 0, 1] → T
        [0, 0, 0, 0] → N

    See :py:func:`oneHotEncode<bpreveal.utils.oneHotEncode>` for an example.
    """
    # Convert to an int8 array, since if we get floating point
    # values, the chr() call will fail.
    oneHotArray = oneHotSequence.astype(ONEHOT_T)

    ret = \
        oneHotArray[:, 0] * ord("A") + \
        oneHotArray[:, 1] * ord("C") + \
        oneHotArray[:, 2] * ord("G") + \
        oneHotArray[:, 3] * ord("T")
    # Anything that was not encoded is N.
    ret[ret == 0] = ord("N")
    return ret.tobytes().decode("ascii")


def logitsToProfile(logitsAcrossSingleRegion: LOGIT_AR_T,
                    logCountsAcrossSingleRegion: LOGCOUNT_T) -> PRED_AR_T:
    """Take logits and logcounts and turn it into a profile.

    :param logitsAcrossSingleRegion: An array of shape (output-length * num-tasks)
    :type logitsAcrossSingleRegion: LOGIT_AR_T
    :param logCountsAcrossSingleRegion: A single floating-point number
    :type logCountsAcrossSingleRegion: LOGCOUNT_T
    :return: An array of shape (output-length * num-tasks), giving the profile
        predictions.
    :rtype: PRED_AR_T

    **Example:**

    .. code-block:: python

        from bpreveal.utils import loadModel, oneHotEncode, logitsToProfile
        import pysam
        import numpy as np
        genome = pysam.FastaFile("/scratch/genomes/sacCer3.fa")
        seq = genome.fetch("chrII", 429454, 432546)
        oneHotSeq = oneHotEncode(seq)
        print(oneHotSeq.shape)
        model = loadModel("/scratch/mnase.model")
        preds = model.predict(np.array([oneHotSeq]))
        print(preds[0].shape)
        # (1, 1000, 2)
        # because there was one input sequence, the output-length is 1000 and
        # there are two tasks in this head.
        print(preds[1].shape)
        # (1, 1)
        # because there is one input sequence and there's just one logcounts value
        # for each region.
        # Note that if the model had two heads, preds[1] would be the logits from the
        # second head and preds[2] and preds[3] would be the logcounts from head 1 and
        # head 2, respectively.
        profiles = logitsToProfile(preds[0][0], preds[1][0])
        print(profiles.shape)
        # (1000, 2)
        # Because we have an output length of 1000 and two tasks.
        # These are now the predicted coverage, in read-space.

    """
    # Logits will have shape (output-length x numTasks)
    assert len(logitsAcrossSingleRegion.shape) == 2
    # If the logcounts passed in is a float, this will break.
    # assert len(logCountsAcrossSingleRegion.shape) == 1  # Logits will be a scalar value

    profileProb = scipy.special.softmax(logitsAcrossSingleRegion)
    profile = profileProb * np.exp(logCountsAcrossSingleRegion)
    return profile.astype(np.float32)


# Easy functions


def easyPredict(sequences: Iterable[str] | str, modelFname: str) -> \
        list[list[PRED_AR_T]] | list[PRED_AR_T]:
    """Make predictions with your model.

    :param sequences: The DNA sequence(s) that you want to predict on.
    :param modelFname: The name of the Keras model to use.
    :return: An array of profiles or a single profile, depending on ``sequences``
    :rtype: list[list[PRED_AR_T]] or list[PRED_AR_T]

    Spawns a separate process to make a single batch of predictions,
    then shuts it down. Why make it complicated? Because it frees the
    GPU after it's done so other programs and stuff can use it.
    If ``sequences`` is a single string containing a sequence to predict
    on, that's okay, it will be treated as a length-one list of sequences
    to predict. ``sequences`` should be as long as the input length of
    your model.

    If you passed in an iterable of strings (like a list of strings),
    the shape of the returned profiles will be
    (numSequences x numHeads x outputLength x numTasks).
    Since different heads can have different numbers of tasks, the returned object
    will be a list (one entry per sequence) of lists (one entry per head)
    of arrays of shape (outputLength x numTasks).
    If, instead, you passed in a single string as ``sequences``,
    it will be (numHeads x outputLength x numTasks). As before, this will be a list
    (one entry per head) of arrays of shape (outputLength x numTasks)

    **Example:**

    .. code-block:: python

        from bpreveal.utils import easyPredict
        import pysam
        genome = pysam.FastaFile("/scratch/genomes/sacCer3.fa")
        seq = genome.fetch("chrII", 429454, 432546)
        profile = easyPredict([seq], "/scratch/mnase.model")
        print(len(profile))
        # 1
        # because we ran one sequence.
        print(len(profile[0]))
        # 1
        # because there is one head in this model.
        print(profile[0][0].shape)
        # (1000, 2)
        # Because we have an output length of 1000 and two tasks.
        # These are now the predicted coverage, in read-space.
        singleProfile = easyPredict(seq, "/scratch/mnase.model")
        print(singleProfile[0].shape)
        # (1000, 2)
        # Note how I only had to index singleProfile once, (to get the first head)
        # since I passed in a single string as the sequence.
    """
    singleReturn = False
    assert not constants.getTensorflowLoaded(), \
        "Cannot use easy functions after loading tensorflow."

    if isinstance(sequences, str):
        sequences = [sequences]
        singleReturn = True
    else:
        # In case we got some weird iterable, turn it into a list.
        sequences = list(sequences)

    predictor = ThreadedBatchPredictor(modelFname, 64, start=False)
    ret = []
    remainingToRead = 0
    with predictor:
        for s in sequences:
            predictor.submitString(s, 1)
            remainingToRead += 1
            while predictor.outputReady():
                outputs = predictor.getOutput()[0]
                numHeads = len(outputs) // 2
                headProfiles = []
                for h in range(numHeads):
                    logits = outputs[h]
                    logcounts = outputs[h + numHeads]
                    headProfiles.append(logitsToProfile(logits, logcounts))  # type: ignore
                ret.append(headProfiles)
                remainingToRead -= 1
        for _ in range(remainingToRead):
            outputs = predictor.getOutput()[0]
            numHeads = len(outputs) // 2
            headProfiles = []
            for h in range(numHeads):
                logits = outputs[h]
                logcounts = outputs[h + numHeads]
                headProfiles.append(logitsToProfile(logits, logcounts))  # type: ignore
            ret.append(headProfiles)
    if singleReturn:
        return ret[0]
    return ret


def easyInterpretFlat(sequences: Iterable[str] | str, modelFname: str,
                      heads: int, headID: int, taskIDs: list[int],
                      numShuffles: int = 20, kmerSize: int = 1,
                      keepHypotheticals: bool = False) \
        -> dict[str, IMPORTANCE_AR_T | ONEHOT_AR_T]:
    """Spin up an entire interpret pipeline just to interpret your sequences.

    You should only use this for quick one-off things since it is EXTREMELY
    slow.

    :param sequences: is a list (or technically any Iterable) of strings, and the
        returned importance scores will be in an order that corresponds
        to your sequences.
        You can also provide just one string, in which case the return type
        will change: The first (length-one) dimension will be stripped.
    :param modelFname: The name of the BPReveal model on disk.
    :param heads: The TOTAL number of heads that the model has.
    :param headID: The index of the head of the model that you want interpreted.
    :param taskIDs: The list of tasks that should be included in the profile score
        calculation. For most cases, you'd want a list of all the tasks,
        like ``[0,1]``.
    :param numShuffles: The number of shuffled sequences that are used to calculate
        shap values.
    :param kmerSize: The length of kmers for which the distribution should be preserved
        during the shuffle. If 1, shuffle each base independently. If 2, preserve
        the distribution of dimers, etc.
    :param keepHypotheticals: Controls whether the output contains hypothetical
        contribution scores or just the actual ones.

    :return: A dict containing the importance scores.
    :rtype: dict[str, IMPORTANCE_AR_T | ONEHOT_AR_T]

    If you passed in an iterable of strings (like a list), then the output's first
    dimension will be the number of sequences and it will depend on keepHypotheticals:

    * If keepHypotheticals is True, then it will be structured so::

            {"profile": array of shape (numSequences x inputLength x 4),
             "counts": array of shape (numSequences x inputLength x 4),
             "sequence": array of shape (numSequences x inputLength x 4)}

      This dict has the same meaning as shap scores stored in an
      interpretFlat hdf5.

    * If keepHypotheticals is False (the default), then the
      shap scores will be condensed down to the normal scores that we plot
      in a genome browser::

          {"profile": array of shape (numSequences x inputLength),
           "counts": array of shape (numSequences x inputLength)}

    However, if sequences was a string instead of an iterable, then the numSequences
    dimension will be suppressed:

    * For keepHypotheticals == True, you get::

          {"profile": array of shape (inputLength x 4),
           "counts": array of shape (inputLength x 4),
           "sequence": array of shape (inputLength x 4)}

    * and if keepHypotheticals is False, you get::

          {"profile": array of shape (inputLength,),
           "counts": array of shape (inputLength,)}
    """
    # pylint: disable=import-outside-toplevel
    from bpreveal.interpretUtils import ListGenerator, FlatListSaver, FlatRunner
    # pylint: enable=import-outside-toplevel
    assert not constants.getTensorflowLoaded(), \
        "Cannot use easy functions after loading tensorflow."
    logUtils.debug("Starting interpretation of sequences.")
    singleReturn = False
    if isinstance(sequences, str):
        sequences = [sequences]
        singleReturn = True
    generator = ListGenerator(sequences)
    profileSaver = FlatListSaver(generator.numSamples, generator.inputLength)
    countsSaver = FlatListSaver(generator.numSamples, generator.inputLength)
    batcher = FlatRunner(modelFname=modelFname, headID=headID, numHeads=heads,
                         taskIDs=taskIDs, batchSize=1, generator=generator,
                         profileSaver=profileSaver, countsSaver=countsSaver,
                         numShuffles=numShuffles, kmerSize=kmerSize)
    batcher.run()
    logUtils.debug("Interpretation complete. Organizing outputs.")
    if keepHypotheticals:
        if singleReturn:
            return {"profile": profileSaver.shap[0],
                    "counts": countsSaver.shap[0],
                    "sequence": profileSaver.seq[0]}
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
    return {"profile": profile, "counts": counts}


# The batchers.


class BatchPredictor:
    """A utility class for when you need to make lots of predictions.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.

    It's doubly-useful if you are generating sequences dynamically. Here's how
    it works. You first create a predictor by calling BatchPredictor(modelName,
    batchSize). If you're not sure, a batch size of 64 is probably good.

    Now, you submit any sequences you want predicted, using the submit methods.

    Once you've submitted all of your sequences, you can get your results with
    the getOutput() method.

    Note that the getOutput() method returns *one* result at a time, and
    you have to call getOutput() once for every time you called one of the
    submit methods.

    :param modelFname: The name of the BPReveal model that you want to make
        predictions from. It's the same name you give for the model in any of
        the other BPReveal tools.
    :param batchSize: is the number of samples that should be run simultaneously
        through the model.
    :param start: Ignored, but present here to give BatchPredictor the same API
        as ThreadedBatchPredictor.
    """

    def __init__(self, modelFname: str, batchSize: int, start: bool = True) -> None:
        """Start up the BatchPredictor.

        This will load your model, and get ready to make predictions.
        """
        logUtils.debug("Creating batch predictor.")
        if not constants.getTensorflowLoaded():
            # We haven't loaded Tensorflow yet.
            setMemoryGrowth()
        self._model = loadModel(modelFname)  # type: ignore
        logUtils.debug("Model loaded.")
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
        """Do nothing; context managers not supported."""

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        """Quit the context manager.

        If this batcher was used in a context manager, exiting does nothing, but raises
        any exceptions that happened.
        """
        if exceptionType is not None:
            return False
        del exceptionValue
        del exceptionTraceback
        return True

    def clear(self) -> None:
        """Reset the predictor.

        If you've left your predictor in some weird state, you can reset it
        by calling clear(). This empties all the queues.
        """
        self._inQueue.clear()
        self._outQueue.clear()
        self._inWaiting = 0
        self._outWaiting = 0

    def submitOHE(self, sequence: ONEHOT_AR_T, label: typing.Any) -> None:
        """Submit a one-hot-encoded sequence.

        :param sequence: An (input-length x 4) ndarray containing the
            one-hot encoded sequence to predict.
        :param label: Any object; it will be returned with the prediction.
        """
        self._inQueue.appendleft((sequence, label))
        self._inWaiting += 1
        if self._inWaiting >= self._batchSize:
            # We have a ton of sequences to run, so go ahead
            # and run a batch real quick.
            self.runBatch()

    def submitString(self, sequence: str, label: typing.Any) -> None:
        """Submit a given sequence for prediction.

        :param sequence: A string of length input-length
        :param label: Any object. Label will be returned to you with the
            prediction.
        """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def runBatch(self, maxSamples: int | None = None) -> None:
        """Actually run the batch.

        Normally, this will be called
        by the submit functions, and it will also be called if you ask
        for output and the output queue is empty (assuming there are
        sequences waiting in the input queue.)

        :param maxSamples: (Optional) The maximum number of samples to
            run in this batch. It should probably be a multiple of the
            batch size.
        """
        if self._inWaiting == 0:
            # There are no samples to process right now, so return
            # (successfully) immediately.
            logUtils.info("runBatch was called even though there was nothing to do.")
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
        # I now need to parse out the shape of the prediction to
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
        """Is there any output ready for you?

        :return: True if the batcher is sitting on results, and False otherwise.
        """
        return self._outWaiting > 0

    def empty(self) -> bool:
        """Is it possible to getOutput()?

        :return: True if predictions haven't been made yet, but
            they would be made if you asked for output.
        """
        return self._outWaiting == 0 and self._inWaiting == 0

    def getOutput(self) -> tuple[list, typing.Any]:
        """Return one of the predictions made by the model.

        This implementation guarantees that predictions will be returned in
        the same order as they were submitted, but you should not rely on that
        in the future. Instead, you should use a label when you submit your
        sequences and use that to determine order.

        :return: A two-tuple.
        :rtype: tuple[list[LOGIT_AR_T, LOGIT_T], typing.Any]

        * The first element will be a list of length numHeads*2, representing the
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

        * The second element will be the label you
          passed in with the original sequence.

        Graphically::

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
    """Mirrors the API of the BachPredictor class, but predicts in a separate thread.

    This can give you a performance boost, and also lets you
    shut down the predictor thread when you don't need it.
    Supports the with statement to only turn on the batcher when you're using it,
    or you can leave it running in the background.

    Usage examples::

        predictor = utils.ThreadedBatchPredictor(modelFname, 64, start=True)
        # Use as you would a normal batchPredictor
        # When not needed any more:
        predictor.stop()

    Alternatively, you can use this as a context manager::

        predictor = utils.ThreadedBatchPredictor(modelFname, 64, start=False)

        with predictor:
            # use as a normal BatchPredictor.
        # On leaving the context, the predictor is shut down.

    The batcher guarantees that the order in which you get results is the same as
    the order you submitted them in, but this could change in the future!

    :param modelFname: The name of the model to use to make predictions.
    :param batchSize: The number of samples to calculate at once.
    :param start: Should the predictor start right away?
    :param numThreads: How many predictors should be spawned?

    """

    def __init__(self, modelFname: str, batchSize: int, start: bool = False,
                 numThreads: int = 1) -> None:
        """Build the batch predictor."""
        logUtils.debug("Creating threaded batch predictor.")
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
        """Start up a context manager.

        Used in a context manager, this is the first thing that gets called
        inside a with statement.
        """
        self.start()

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        """When leaving a context manager's with statement, shut down the batcher."""
        self.stop()
        if exceptionType is not None:
            return False
        del exceptionValue  # Disable unused warning
        del exceptionTraceback  # Disable unused warning
        return True

    def start(self):
        """Spin up the batcher thread.

        If you submit sequences without starting the batcher,
        this method will be called automatically (with a warning).
        """
        if not self.running:
            logUtils.debug("Starting threaded batcher.")
            assert self._batchers is None, "Attempting to start a new batcher when an "\
                "old one is still alive." + str(self._batchers)
            self._inQueues = []
            self._outQueues = []
            self._batchers = []

            for _ in range(self._numThreads):
                nextInQueue = multiprocessing.Queue(10000)
                nextOutQueue = multiprocessing.Queue(10000)
                self._inQueues.append(nextInQueue)
                self._outQueues.append(nextOutQueue)
                nextBatcher = multiprocessing.Process(
                    target=_batcherThread,
                    args=(self._modelFname, self._batchSize, nextInQueue, nextOutQueue),
                    daemon=True)
                nextBatcher.start()
                self._batchers.append(nextBatcher)
            self._inFlight = 0
            self._inQueueIdx = 0
            self.running = True
            self._outQueueOrder = deque()
        else:
            logUtils.warning("Attempted to start a batcher that was already running.")

    def __del__(self):
        """General cleanup - kill the child process when this object leaves scope."""
        if logUtils is not None:
            logUtils.debug("Destructor called.")
        if self.running:
            self.stop()

    def stop(self):
        """Shut down the processor thread."""
        if self.running:
            if logUtils is not None:
                logUtils.debug("Shutting down threaded batcher.")
            if self._batchers is None:
                assert False, "Attempting to shut down a running ThreadedBatchPredictor" \
                    "When its _batchers is None"
            for i in range(self._numThreads):
                self._inQueues[i].put("shutdown")
                self._inQueues[i].close()
                self._batchers[i].join(QUEUE_TIMEOUT)  # Wait one second.
                if self._batchers[i].exitcode is None:
                    # The process failed to die. Kill it more forcefully.
                    self._batchers[i].terminate()
                self._batchers[i].join(QUEUE_TIMEOUT)  # Wait one second.
                self._batchers[i].close()
                self._outQueues[i].close()
            del self._inQueues
            del self._batchers
            del self._outQueues
            # Explicitly set None so that start won't panic.
            self._batchers = None
            self.running = False
        else:
            if logUtils is not None:  # pylint: disable=else-if-used
                logUtils.warning("Attempting to stop a batcher that is already stopped.")

    def clear(self):
        """Reset the batcher, emptying any queues and reloading the model."""
        if self.running:
            self.stop()
        self.start()

    def submitOHE(self, sequence: ONEHOT_AR_T, label: typing.Any) -> None:
        """Submit a one-hot-encoded sequence.

        :param sequence: An (input-length x 4) ndarray containing the
            one-hot encoded sequence to predict.
        :param label: Any object; it will be returned with the prediction.
        """
        if not self.running:
            logUtils.warning("Submitted a query when the batcher is stopped. Starting.")
            self.start()
        q = self._inQueues[self._inQueueIdx]
        query = (sequence, label)
        q.put(query, True, QUEUE_TIMEOUT)
        self._outQueueOrder.appendleft(self._inQueueIdx)
        self._inQueueIdx = (self._inQueueIdx + 1) % self._numThreads
        self._inFlight += 1

    def submitString(self, sequence: str, label: typing.Any) -> None:
        """Submit a given sequence for prediction.

        :param sequence: A string of length input-length
        :param label: Any object. Label will be returned to you with the
            prediction.
        """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def outputReady(self) -> bool:
        """Is there any output ready for you?

        :return: True if the batcher is sitting on results, and False otherwise.
        """
        if self._inFlight:
            outIdx = self._outQueueOrder[-1]
            return not self._outQueues[outIdx].empty()
        return False

    def empty(self) -> bool:
        """Is it possible to getOutput()?

        :return: True if predictions haven't been made yet, but
            they would be made if you asked for output.
        """
        return self._inFlight == 0

    def getOutput(self) -> tuple[list, typing.Any]:
        """Get a single output.

        :return: The model's predictions.
        :rtype: tuple[list[LOGIT_AR_T, LOGCOUNT_T], typing.Any]

        Same semantics as
        :py:meth:`BatchPredictor.getOutput<bpreveal.utils.BatchPredictor.getOutput>`.
        """
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
    """Run batches from the ThreadedBatchPredictor in this separate thread.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.
    """
    assert not constants.getTensorflowLoaded(), "Cannot use the threaded predictor " \
        "after loading tensorflow."
    # pylint: disable=import-outside-toplevel, unused-import
    import bpreveal.internal.disableTensorflowLogging  # noqa
    # pylint: enable=import-outside-toplevel, unused-import
    logUtils.debug("Starting subthread")
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
            # There was no input. Are we sitting on predictions that we could go ahead
            # and make?
            # pylint: disable=protected-access
            if batcher._inWaiting > batcher._batchSize / numWaits:
                # pylint: enable=protected-access
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
            case(sequence, label):
                if isinstance(sequence, str):
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
                logUtils.debug("Shutdown signal received.")
                return
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
