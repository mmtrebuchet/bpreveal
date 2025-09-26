"""Lots of helpful utilities for working with models."""
from collections import deque
import multiprocessing
import multiprocessing.synchronize
import subprocess as sp
import typing
import queue
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
from bpreveal.internal.constants import NUM_BASES, ONEHOT_AR_T, PRED_AR_T, ONEHOT_T, \
    LOGCOUNT_T, LOGIT_AR_T, IMPORTANCE_AR_T, IMPORTANCE_T, PRED_T
from bpreveal.internal import constants
from bpreveal.internal.crashQueue import CrashQueue


def loadModel(modelFname: str):  # noqa: ANN201
    """Load up a BPReveal model.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.

    :param modelFname: The name of the model that Keras saved earlier, either a directory
        ending in ``.model`` for models trained before BPReveal 5.0.0, or a file ending in
        ``.keras`` for models trained with BPReveal 5.0.0 or later.
    :return: A Keras ``Model`` object.
    :rtype: keras.Model

    For pre-5.0.0 models, the returned model does NOT support additional training,
    since it uses a dummy loss. New-style models remember their losses and so you can
    continue to train them if you like.

    **Example:**

    .. code-block:: python

        from bpreveal.utils import loadModel
        m = loadModel("path/to/model")
        preds = m.predict(myOneHotSequences)

    """
    ret = None
    renamedModel = False
    # pylint: disable=import-outside-toplevel
    import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
    from bpreveal.losses import multinomialNll, dummyMse
    from bpreveal.layers import CountsLogSumExp
    # pylint: enable=import-outside-toplevel
    constants.setTensorflowLoaded()
    if modelFname.endswith("model"):
        try:
            logUtils.info(
                "You are attempting to a load a model with a '.model' extension. As of "
                "BPReveal 5.0.0, models are given the extension '.keras' because keras "
                "3.0 requires it. I will attempt to load the old-style model, but be "
                "prepared for some janky behavior and possible bugs.")
            import tf_keras
            ret = tf_keras.models.load_model(
                filepath=modelFname,
                custom_objects={"multinomialNll": multinomialNll,
                                "reweightableMse": dummyMse})
            ret.useOldKeras = True
            logUtils.debug(f"Loaded old-style model {modelFname}.")
        except OSError:
            logUtils.error(
                f"You specified a model named {modelFname} but I couldn't find it. "
                "Attempting to load a model with a '.keras' extension. If this "
                "works, you should update your configuration file to use the new "
                "extension.")
            modelFname = modelFname[:-5] + "keras"
            renamedModel = True
    if ret is None:
        # pylint: disable=import-outside-toplevel
        from keras.models import load_model  # type: ignore
        # pylint: enable=import-outside-toplevel
        ret = load_model(
            filepath=modelFname,
            custom_objects={"multinomialNll": multinomialNll,
                            "reweightableMse": dummyMse,
                            "CountsLogSumExp": CountsLogSumExp})
        ret.useOldKeras = False
        logUtils.debug(f"Loaded new-style model {modelFname}.")
        if renamedModel:
            logUtils.error("I was able to load your model by renaming it from "
                           "'.model' to '.keras'. You should update your config "
                           "file to use the new extension.")

    return ret


def setMemoryGrowth() -> None:
    """Turn on the tensorflow option to grow memory usage as needed.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.

    All of the main programs in BPReveal do this, so that you can
    use your GPU for other stuff as you work with models.

    If a GPU is not found, this function will emit a warning and do nothing else.
    """
    # pylint: disable=unused-import
    import bpreveal.internal.disableTensorflowLogging  # noqa
    import tensorflow as tf
    # pylint: enable=unused-import
    gpus = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        logUtils.debug("GPU memory growth enabled.")
    except Exception as inst:  # pylint: disable=broad-exception-caught
        logUtils.warning("Not using GPU")
        logUtils.debug("Because: " + str(inst))
    constants.setTensorflowLoaded()


def loadPisa(fname: str) -> IMPORTANCE_AR_T:
    """Load up a PISA file, shear it, and crop it to a standard array.

    :param fname: The name of the hdf5-format file on disk, containing your PISA data.
    :return: An array of shape (num-samples, num-samples) containing the sheared PISA data.

    This is probably best demonstrated with an image or two. Here's how PISA data are
    stored in the hdf5 file:

    .. image:: ../../doc/presentations/pisaRaw.png
        :width: 400
        :alt: Unsheared PISA data straight from an hdf5.

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
    logUtils.debug(f"Loading PISA data from {fname}")
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
    logUtils.debug(f"The loaded PISA data have shape {pisaVals.shape} "
                   "and the sheared matrix has shape {shearMat.shape}.")
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

    If this function can't find a GPU (because nvidia-smi gives an error or cannot be found)
    it will emit an error message but won't crash.

    :param fraction: How much of the memory on the GPU can I have?
    :param offset: How much memory (in MB) should be reserved when
        I carve out my fraction?
    :return: The memory (in MB) reserved.
    """
    assert 0.0 < fraction < 1.0, "Must give a memory fraction between 0 and 1."
    free = total = 0.0
    cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv"]
    try:
        ret = sp.run(cmd, capture_output=True, check=True)
    except sp.CalledProcessError as e:
        logUtils.warning("Problem running nvidia-smi. Did you remember to allocate a GPU?")
        logUtils.warning(e.stdout.decode("utf-8"))
        logUtils.warning(e.stderr.decode("utf-8"))
        logUtils.warning(str(e.returncode))
        logUtils.warning("I couldn't find nvidia-smi, so I won't limit memory. If you meant "
                         "to allocate a GPU, check your configuration because I don't "
                         "think one is available! If you are intentionally running on "
                         "CPU, you can ignore this warning.")
        return 0
        # raise
    except FileNotFoundError as e:
        logUtils.warning("Problem running nvidia-smi. Did you remember to allocate a GPU?")
        logUtils.warning(str(e))
        logUtils.warning("I couldn't find nvidia-smi, so I won't limit memory. If you meant "
                         "to allocate a GPU, check your configuration because I don't "
                         "think one is available! If you are intentionally running on "
                         "CPU, you can ignore this warning.")
        return 0

    line = ret.stdout.decode("utf-8").split("\n")[1]
    logUtils.debug(f"Memory usage limited based on {line}")
    lsp = line.split(" ")
    total = float(lsp[0])
    free = float(lsp[2])
    assert total * fraction < free, f"Attempting to request more memory ({total * fraction}) "\
        f"than is free ({free})!"

    # pylint: disable=unused-import
    import bpreveal.internal.disableTensorflowLogging  # noqa
    import tensorflow as tf
    # pylint: enable=unused-import
    gpus = tf.config.list_physical_devices("GPU")
    logUtils.debug(f"Available devices: {gpus}")
    useMem = int(total * fraction - offset)
    tf.config.set_logical_device_configuration(
        device=gpus[0],
        logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=useMem)])
    logUtils.debug(f"Configured gpu with {useMem} MiB of memory.")
    constants.setTensorflowLoaded()
    return useMem


def loadChromSizes(*, chromSizesFname: str | None = None,
                   genomeFname: str | None = None,
                   bwHeader: dict[str, int] | None = None,
                   bw: pyBigWig.pyBigWig | None = None,
                   fasta: pysam.FastaFile | None = None) -> dict[str, int]:
    """Read in a chrom sizes file and return a dictionary mapping chromosome name → size.

    Exactly one of the parameters may be specified, all others must be ``None``.

    :param chromSizesFname: The name of a chrom.sizes file on disk.
    :param genomeFname: The name of a genome fasta file on disk.
    :param bwHeader: A dictionary loaded from a bigwig.
        (Using this makes this function an identity function.)
    :param bw: An opened bigwig file.
    :param fasta: An opened genome fasta.
    :raise ValueError: If you called the function incorrectly.
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
    raise ValueError("You can't ask for chrom sizes without some argument!")


def blankChromosomeArrays(*, genomeFname: str | None = None,
                          chromSizesFname: str | None = None,
                          bwHeader: dict[str, int] | None = None,
                          chromSizes: dict[str, int] | None = None,
                          bw: pyBigWig.pyBigWig | None = None,
                          fasta: pysam.FastaFile | None = None,
                          dtype: type = PRED_T,
                          numTracks: int = 1) -> dict[str, np.ndarray]:
    """Get a set of blank numpy arrays that you can use to save genome-wide data.

    Exactly one of ``chromSizesFname``, ``genomeFname``, ``bwHeader``,
    ``chromSizes``, ``bw``, or ``fasta`` may be specified, all other
    parameters must be ``None``.

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
    The shape of each element of the dictionary will be ``(chromosome-length, numTracks)``.

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
                chromSizes: dict[str, int] | None = None) -> None:
    """Write a bigwig file given some region data.

    You must specify either:

    * ``chromDict``,
        in which case ``regionList``, ``chromSizes``
        and ``regionData`` must be ``None``, or
    * ``regionList``, ``chromSizes``, and ``regionData``,
        in which case ``chromDict`` must be ``None``.

    :param bwFname: The name of the bigwig file to write.
    :param chromDict: A dict mapping chromosome names to the data for that
        chromosome. The data should have shape ``(chromosome-length,)``.
    :param regionList: A list of ``(chrom, start, end)`` tuples giving the
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
    logUtils.debug(f"Starting to write data to bigwig {bwFname}.")
    header = [(x, chromSizes[x]) for x in sorted(list(chromSizes.keys()))]
    outBw.addHeader(header)

    for chromName in sorted(list(chromDict.keys())):
        if isinstance(chromDict[chromName], np.ndarray):
            flatItems = chromDict[chromName].reshape((chromDict[chromName].shape[0],))
            vals = flatItems.tolist()
        else:
            vals = [float(x) for x in chromDict[chromName]]
        logUtils.debug(f"Adding data for chromosome {chromName}.")
        outBw.addEntries(chromName, 0, values=vals,
                         span=1, step=1)
    logUtils.debug(f"Data written. Closing bigwig {bwFname}.")
    outBw.close()
    logUtils.debug(f"Bigwig {bwFname} closed.")


def oneHotEncode(sequence: str, allowN: bool = False, alphabet: str = "ACGT") -> ONEHOT_AR_T:
    """Convert the string sequence into a one-hot encoded numpy array.

    :param sequence: A DNA sequence to encode.
        May contain uppercase and lowercase letters.
    :param allowN: If ``False`` (the default), raise an ``AssertionError`` if
        the sequence contains letters other than ``ACGTacgt``.
        If ``True``, any other characters will be encoded as ``[0, 0, 0, 0]``.
    :param alphabet: The order of the bases in the output array.
    :return: An array with shape ``(len(sequence), NUM_BASES)``.
    :rtype: ``ONEHOT_AR_T``


    The columns are, in order, A, C, G, and T.
    The mapping is as follows::

        A or a → [1, 0, 0, 0]
        C or c → [0, 1, 0, 0]
        G or g → [0, 0, 1, 0]
        T or t → [0, 0, 0, 1]
        Other  → [0, 0, 0, 0]

    A convenient property of this mapping is that calculating a reverse-complement
    sequence is trivial::

        seq = "AAGAGGCT"
        ohe = oneHotEncode(seq)
        revcompOhe = np.flip(ohe)
        revcompSeq = oneHotDecode(revcompOhe)
        # revcompSeq is now "AGCCTCTT"

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
    assert len(alphabet) == NUM_BASES, f"Your alphabet {alphabet} has the wrong length. "\
        f"Should be {NUM_BASES}, but got {len(alphabet)}"
    if allowN:
        initFunc = np.zeros
    else:
        # We're going to overwrite every position, so don't bother with
        # initializing the array.
        initFunc = np.empty
    ret = initFunc((len(sequence), len(alphabet)), dtype=ONEHOT_T)
    ordSeq = np.fromstring(sequence, np.int8)  # type:ignore
    for i, base in enumerate(alphabet):
        ret[:, i] = (ordSeq == ord(base)) + (ordSeq == ord(base.lower()))
    if not allowN:
        assert (np.sum(ret) == len(sequence)), \
            "Sequence contains unrecognized nucleotides. "\
            "Maybe your sequence contains 'N'?"
    return ret


def oneHotDecode(oneHotSequence: np.ndarray, alphabet: str = "ACGT") -> str:
    """Take a one-hot encoded sequence and turn it back into a string.

    :param oneHotSequence: An array of shape ``(n, NUM_BASES)``. It may have any type
        that can be converted into a ``uint8``.
    :param alphabet: The order in which the bases are encoded.
    :return: Your sequence as a string.

    Given an array representing a one-hot encoded sequence, convert it back
    to a string. The input shall have shape ``(sequenceLength, NUM_BASES)``,
    and the output will be a Python string.
    The decoding is performed based on the following mapping::

        [1, 0, 0, 0] → A
        [0, 1, 0, 0] → C
        [0, 0, 1, 0] → G
        [0, 0, 0, 1] → T
        [0, 0, 0, 0] → N

    See :py:func:`oneHotEncode<bpreveal.utils.oneHotEncode>` for an example.
    """
    assert len(alphabet) == NUM_BASES, f"Your alphabet {alphabet} has the wrong length. "\
        f"Should be {NUM_BASES}, but got {len(alphabet)}"
    # Convert to an int8 array, since if we get floating point
    # values, the chr() call will fail.
    oneHotArray = oneHotSequence.astype(ONEHOT_T)

    ret = np.zeros_like(oneHotArray[:, 0])
    for i, base in enumerate(alphabet):
        ret += oneHotArray[:, i] * ord(base.upper())
    # Anything that was not encoded is N.
    ret[ret == 0] = ord("N")
    return ret.tobytes().decode("ascii")


def logitsToProfile(logitsAcrossSingleRegion: LOGIT_AR_T,
                    logCountsAcrossSingleRegion: LOGCOUNT_T) -> PRED_AR_T:
    """Take logits and logcounts and turn it into a profile.

    :param logitsAcrossSingleRegion: An array of shape ``(output-length * num-tasks)``
    :type logitsAcrossSingleRegion: ``LOGIT_AR_T``
    :param logCountsAcrossSingleRegion: A single floating-point number
    :type logCountsAcrossSingleRegion: ``LOGCOUNT_T``
    :return: An array of shape ``(output-length * num-tasks)``, giving the profile
        predictions.
    :rtype: ``PRED_AR_T``

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
        # > (1, 1000, 2)
        # because there was one input sequence, the output-length is 1000 and
        # there are two tasks in this head.
        print(preds[1].shape)
        # > (1, 1)
        # because there is one input sequence and there's just one logcounts value
        # for each region.
        # Note that if the model had two heads, preds[1] would be the logits from the
        # second head and preds[2] and preds[3] would be the logcounts from head 1 and
        # head 2, respectively.
        profiles = logitsToProfile(preds[0][0], preds[1][0])
        print(profiles.shape)
        # > (1000, 2)
        # Because we have an output length of 1000 and two tasks.
        # These are now the predicted coverage, in read-space.

    """
    # Logits will have shape (output-length x numTasks)
    assert len(logitsAcrossSingleRegion.shape) == 2
    # If the logcounts passed in is a float, this will break.

    profileProb = scipy.special.softmax(logitsAcrossSingleRegion)
    profile = profileProb * np.exp(logCountsAcrossSingleRegion)
    return profile.astype(np.float32)


# Easy functions


def easyPredict(sequences: Iterable[str] | str, modelFname: str, quiet: bool = False) -> \
        list[list[PRED_AR_T]] | list[PRED_AR_T]:
    """Make predictions with your model.

    :param sequences: The DNA sequence(s) that you want to predict on.
    :param modelFname: The name of the Keras model to use.
    :param quiet: If True, all stderr spew from tensorflow is deleted. Set to True
        for interactive use, but set to False if you're getting errors, since they'll
        be deleted otherwise and make debugging a nightmare.
    :return: An array of profiles or a single profile, depending on ``sequences``
    :rtype: ``list[list[PRED_AR_T]]`` or ``list[PRED_AR_T]``

    Spawns a separate process to make a single batch of predictions,
    then shuts it down. Why make it complicated? Because it frees the
    GPU after it's done so other programs and stuff can use it.
    If ``sequences`` is a single string containing a sequence to predict
    on, that's okay, it will be treated as a length-one list of sequences
    to predict. The ``sequences`` string should be at least as long as
    the input length of your model.

    If you passed in an iterable of strings (like a list of strings),
    the shape of the returned profiles will be
    ``(numSequences x numHeads x outputLength x numTasks)``.
    Since different heads can have different numbers of tasks, the returned object
    will be a list (one entry per sequence) of lists (one entry per head)
    of arrays of shape ``(outputLength x numTasks)``.
    If, instead, you passed in a single string as ``sequences``,
    it will be ``(numHeads x outputLength x numTasks)``. As before, this will be a list
    (one entry per head) of arrays of shape ``(outputLength x numTasks)``

    As a bonus feature, if you pass in a sequence that is longer than your model's
    input length, this function will make tiling predictions over as much of the
    sequence as possible. For example, if my model has an input length of 3 kb
    and an output of 1 kb, then if I provide an input sequence that is 4 kb long,
    I will get a 2 kb output prediction.

    **Example:**

    .. code-block:: python

        from bpreveal.utils import easyPredict
        import pysam
        genome = pysam.FastaFile("/scratch/genomes/sacCer3.fa")
        seq = genome.fetch("chrII", 429454, 432546)
        profile = easyPredict([seq], "/scratch/mnase.model")
        print(len(profile))
        # > 1
        # because we ran one sequence.
        print(len(profile[0]))
        # > 1
        # because there is one head in this model.
        print(profile[0][0].shape)
        # > (1000, 2)
        # Because we have an output length of 1000 and two tasks.
        # These are now the predicted coverage, in read-space.
        singleProfile = easyPredict(seq, "/scratch/mnase.model")
        print(singleProfile[0].shape)
        # > (1000, 2)
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
    logUtils.debug(f"Running {len(sequences)} predictions using model {modelFname}")
    predictor = ThreadedBatchPredictor(modelFname, 64, start=False, produceProfiles=True,
                                       quiet=quiet)
    ret = []
    remainingToRead = 0
    with predictor:
        for s in sequences:
            predictor.submitString(s, 1)
            remainingToRead += 1
            while predictor.outputReady():
                outputs = predictor.getOutputProfile()[0]
                ret.append(outputs)
                remainingToRead -= 1
        for _ in range(remainingToRead):
            outputs = predictor.getOutputProfile()[0]
            ret.append(outputs)
    if singleReturn:
        return ret[0]
    return ret


def easyInterpretFlat(sequences: Iterable[str] | str, modelFname: str,
                      heads: int, headID: int, taskIDs: list[int],
                      numShuffles: int = 20, kmerSize: int = 1,
                      keepHypotheticals: bool = False) \
        -> dict[str, IMPORTANCE_AR_T | ONEHOT_AR_T]:
    """Spin up an entire interpret pipeline just to interpret your sequences.

    You should only use this for quick one-off things since it takes a long time
    to spin up and shut down the interpretation machinery.

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
    :rtype: ``dict[str, IMPORTANCE_AR_T | ONEHOT_AR_T]``

    If you passed in an iterable of strings (like a list), then the output's first
    dimension will be the number of sequences and it will depend on ``keepHypotheticals``:

    * If ``keepHypotheticals == True``, then it will be structured so::

            {"profile": array of shape (numSequences x inputLength x NUM_BASES),
             "counts": array of shape (numSequences x inputLength x NUM_BASES),
             "sequence": array of shape (numSequences x inputLength x NUM_BASES)}

      This dict has the same meaning as shap scores stored in an
      :py:mod:`interpretFlat<bpreveal.interpretFlat>` hdf5.

    * If ``keepHypotheticals == False`` (the default), then the
      shap scores will be condensed down to the normal scores that we plot
      in a genome browser::

          {"profile": array of shape (numSequences x inputLength),
           "counts": array of shape (numSequences x inputLength)}

    However, if ``sequences`` was a string instead of an iterable, then the ``numSequences``
    dimension will be suppressed:

    * For ``keepHypotheticals == True``, you get::

          {"profile": array of shape (inputLength x NUM_BASES),
           "counts": array of shape (inputLength x NUM_BASES),
           "sequence": array of shape (inputLength x NUM_BASES)}

    * and if ``keepHypotheticals == False``, you get::

          {"profile": array of shape (inputLength,),
           "counts": array of shape (inputLength,)}
    """
    # pylint: disable=import-outside-toplevel
    from bpreveal import interpretFlat
    from bpreveal.internal.interpretUtils import ListGenerator, FlatListSaver, InterpRunner
    # pylint: enable=import-outside-toplevel
    assert not constants.getTensorflowLoaded(), \
        "Cannot use easy functions after loading tensorflow."
    logUtils.debug("Starting interpretation of sequences.")
    singleReturn = False
    if isinstance(sequences, str):
        sequences = [sequences]
        singleReturn = True
    else:
        sequences = list(sequences)
    logUtils.debug(f"Running {len(sequences)} shaps using model {modelFname}")
    generator = ListGenerator(sequences)
    profileSaver = FlatListSaver(generator.numSamples, generator.inputLength)
    countsSaver = FlatListSaver(generator.numSamples, generator.inputLength)
    profileMetric = interpretFlat.profileMetric(headID, taskIDs)
    countsMetric = interpretFlat.countsMetric(heads, headID)
    batcher = InterpRunner(modelFname=modelFname, metrics=[profileMetric, countsMetric],
                           batchSize=1, generator=generator, savers=[profileSaver, countsSaver],
                           numShuffles=numShuffles, kmerSize=kmerSize, numThreads=2,
                           backend="shap", useHypotheticalContribs=True, shuffler=None)
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
    it works. You first create a predictor by calling
    ``BatchPredictor(modelName, batchSize)``.
    If you're not sure, a batch size of 64 is probably good.

    Now, you submit any sequences you want predicted, using the submit methods.

    Once you've submitted some or all of your sequences, you can get your results with
    the ``getOutput()`` method.

    Note that the ``getOutput()` method returns *one* result at a time, and
    you have to call ``getOutput()`` once for every time you called one of the
    submit methods.

    The typical use case for a batcher streams its input, so you'd normally check
    to see if there's output waiting after adding every input::

        for query, label in queryGenerator:
            batcher.submitString(query, label)
            while batcher.outputReady():
                # Any time the batcher has results, process them
                # immediately.
                preds, outLabel = batcher.getOutput()
                processPredictions(preds, outLabel)
        while not batcher.empty():
            # We've finished adding our queries, now drain out
            # any last results.
            preds, outLabel = batcher.getOutput()
            processPredictions(preds, outLabel)

    Using the batcher in this way (checking to see if ``outputReady()`` after every
    query submission) has the benefit of using very little memory. Instead of
    building a huge array of queries and then predicting them in one go, the
    batcher analyzes them as you come up with them. This means you can analyze
    far more sequences than you can store in memory. (But if you have huge
    numbers of sequences, consider using a :py:class:`~ThreadedBatchPredictor`
    since it can do the calculations in parallel and in a separate thread.)

    For small numbers of sequences, you can also submit all of them and then get
    all of the results later:

        for i in range(numQueries):
            batcher.submitString(queries[i], None)
        for _ in range(numQueries):
            preds, _ = batcher.getOutput()
            processPredictions(preds)

    In this example, I'm not using the labels, so I just pass in None
    as the label for each sequence and ignore the labels from ``getOutput()``

    You should not, however, demand an output after every submission, since this
    will use a batch size of one and be painfully slow::

        for query, label in queryGenerator:
            batcher.submitString(query, label)
            preds, outLabel = batcher.getOutput()  # WRONG: Runs a whole batch for each query
            processPredictions(preds, outLabel)

    :param modelFname: The name of the BPReveal model on disk that you want to
        make predictions from. It's the same name you give for the model in any
        of the other BPReveal tools.
    :param batchSize: is the number of samples that should be run simultaneously
        through the model.
    :param start: Ignored, but present here to give ``BatchPredictor`` the same API
        as ``ThreadedBatchPredictor``. Creating a ``BatchPredictor`` loads up the model
        and sets memory growth right then and there.
    :param numThreads: Ignored, only present for compatibility with the API for
        ``ThreadedBatchPredictor``. A (non-threaded)``BachPredictor`` runs its calculations
        in the main thread and will block when it's actually doing calculations.
    :param produceProfiles: Ignored, only for compatibility with ThreadedBatchPredictor.
        With a non-threaded BatchPredictor, you can choose to use getOutput() or
        getOutputProfile() after the prediction has been made.
    :param quiet: If True, then all output to stderr will be suppressed in tensorflow-related
        code. Useful for interactive use.
    """

    def __init__(self, modelFname: str, batchSize: int, start: bool = True,
                 numThreads: int = 0, produceProfiles: bool = False,
                 quiet: bool = False) -> None:
        """Start up the BatchPredictor.

        This will load your model, and get ready to make predictions.
        """
        logUtils.debug(f"Creating batch predictor for model {modelFname}.")
        import bpreveal.internal.disableTensorflowLogging as disableTfLogging
        if quiet:
            self.suppress = disableTfLogging.SuppressStderr
        else:
            self.suppress = disableTfLogging.LeaveStderrAlone
        with self.suppress():
            setMemoryGrowth()
            self._model = loadModel(modelFname)  # type: ignore
        if isinstance(self._model.input, list):
            # Workaround for a bug in Keras 3.10.0 where inputs to loaded
            # models are lists instead of tensors.
            self._inputLength = self._model.input[0].shape[1]
        else:
            self._inputLength = self._model.input.shape[1]
        self._outputLength = self._model.output[0].shape[1]
        self._tasksPerHead = []
        outputs = self._model.output
        for i in range(len(outputs) // 2):
            # The // 2 is because we only care about profile outputs here.
            self._tasksPerHead.append(outputs[i].shape[2])
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
        del numThreads
        del produceProfiles

    def __enter__(self):
        """Do nothing; context manager is a no-op for a non-threaded ``BatchPredictor``."""

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):  # noqa: ANN001
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
        by calling ``clear()``. This empties all the queues.
        """
        logUtils.info(f"Clearing batch predictor, purging {self._inWaiting} inputs "
                      f"and {self._outWaiting} outputs.")
        self._inQueue.clear()
        self._outQueue.clear()
        self._inWaiting = 0
        self._outWaiting = 0

    def submitOHE(self, sequence: ONEHOT_AR_T, label: typing.Any) -> None:
        """Submit a one-hot-encoded sequence.

        :param sequence: An ``(input-length x NUM_BASES)`` ndarray containing the
            one-hot encoded sequence to predict.
        :param label: Any object; it will be returned with the prediction.
        """
        # pylint: disable=import-outside-toplevel
        import pybedtools
        from bpreveal import bedUtils
        # pylint: enable=import-outside-toplevel
        if sequence.shape[0] > self._inputLength:
            # We need to tile.
            logUtils.logFirstN(
                logUtils.INFO,
                f"Found an input with length {sequence.shape[0]} but "
                f"the model has input length {self._inputLength}. Automatic "
                "tiling of input window enabled. This may incur a performance cost.",
                1)
            outputTiles = list(
                bedUtils.tileSegments(self._inputLength, self._outputLength,
                                      [pybedtools.Interval("chrN", 0, sequence.shape[0])],
                                      0))
            for ot in outputTiles:
                center = (ot.end + ot.start) // 2
                start = center - self._inputLength // 2
                end = start + self._inputLength
                query = {"sequence": sequence[start:end],
                         "label": label,
                         "numTiles": len(outputTiles),
                         "tileStart": ot.start - (self._inputLength - self._outputLength) // 2}
                #  Note that tileStart is relative to the OUTPUT, not the input.
                self._inQueue.appendleft(query)
                self._inWaiting += 1
        else:
            self._inQueue.appendleft(
                {"sequence": sequence,
                 "label": label,
                 "numTiles": 1,
                 "tileStart": 0})
            self._inWaiting += 1

        if self._inWaiting >= self._batchSize * 16:
            # We have a ton of sequences to run, so go ahead
            # and run a batch real quick.
            self.runBatch()

    def submitString(self, sequence: str, label: typing.Any) -> None:
        """Submit a given sequence for prediction.

        :param sequence: A string of length ``input-length``
        :param label: Any object. Label will be returned to you with the
            prediction.
        """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def runBatch(self, maxSamples: int | None = None) -> None:
        """Actually run the batch.

        Normally, this will be called by the submit functions, and it will also
        be called if you ask for output and the output queue is empty (assuming
        there are sequences waiting in the input queue.) In other words, you
        don't need to call this function.

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
        counts = []
        tileStarts = []
        modelInputs = np.zeros((numSamples, self._inputLength, NUM_BASES), dtype=ONEHOT_T)
        writeHead = 0
        for _ in range(numSamples):
            nextElem = self._inQueue.pop()
            modelInputs[writeHead] = nextElem["sequence"]
            labels.append(nextElem["label"])
            counts.append(nextElem["numTiles"])
            tileStarts.append(nextElem["tileStart"])
            writeHead += 1
            self._inWaiting -= 1
        with self.suppress():
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
            self._outQueue.appendleft(
                {"preds": curHeads, "label": labels[i], "numTiles":
                 counts[i], "tileStart": tileStarts[i]})
            self._outWaiting += 1

    def outputReady(self) -> bool:
        """Is there any output ready for you?

        If output is ready, then calling ``getOutput()`` will give a result immediately.

        :return: True if the batcher is sitting on results, and False otherwise.
        """
        return self._outWaiting > 0

    def empty(self) -> bool:
        """Is the batcher totally idle?

        If the batcher is not empty, then you can safely call ``getOutput()``, though
        it may block if it needs to run a calculation.

        :return: True if there are no predictions at all in the queue.
        """
        return self._outWaiting == 0 and self._inWaiting == 0

    def getOutput(self) -> tuple[list, typing.Any]:
        """Return one of the predictions made by the model.

        This implementation guarantees that predictions will be returned in
        the same order as they were submitted.

        :return: A two-tuple.
        :raise queue.Empty: if you try to get an output when the batcher is empty.
        :rtype: ``tuple[list[LOGIT_AR_T, LOGIT_T], typing.Any]``

        * The first element will be a list of length ``numHeads * 2``, representing the
          output from the model. Since the output of the model will always have
          a dimension representing the batch size, and this function only returns
          the result of running a single sequence, the dimension representing
          the batch size is removed. In other words, running the model on a
          single example would give a logits output of shape
          ``(1 x output-length x num-tasks)``.
          But this function will remove that, so you will get an array of shape
          ``(output-length x numTasks)``
          As with calling the model directly, the first numHeads elements are the
          logits arrays, and then come the logcounts for each head.
          You can pass the logits and logcounts values to
          :py:func:`utils.logitsToProfile<bpreveal.utils.logitsToProfile>`
          to get your profile.

        * The second element will be the label you passed in with the original
          sequence.

        Graphically::

            ( [<head-1-logits>, <head-2-logits>, ...
               <head-1-logcounts>, <head-2-logcounts>, ...
              ],
              label)

        If the batcher doesn't have any output ready but does have some work in the input
        queue, then calling this function will block until the calculation is complete.
        If there is output ready, then this function will not block.

        This function will error out if an input sequence was longer than the model's input
        length, since there's no logical way to combine the logits and logcounts in that case.
        If you want to use longer sequences, you should get outputs with getOutputProfile().

        """
        if not self._outWaiting:
            if self._inWaiting:
                # There are inputs that have not been processed. Run the batch.
                self.runBatch()
            else:
                raise queue.Empty("There are no outputs ready, and the input queue is empty.")
        ret = self._outQueue.pop()
        assert ret["numTiles"] == 1, "Attempted to getOutput but the query input size was not " \
            "equal to the model's input size. Use getOutputProfile() instead."
        self._outWaiting -= 1
        return (ret["preds"], ret["label"])

    def getOutputProfile(self) -> tuple[list, typing.Any]:
        """Return one of the predictions made by the model, in profile space.

        Whereas getOutput returns the logits and logcounts directly from the model,
        this function converts those results into a profile. This is necessary if you
        provide an input sequence that is longer than the model's input length.

        This implementation guarantees that predictions will be returned in
        the same order as they were submitted.

        :return: A two-tuple.
        :raise queue.Empty: if you try to get an output when the batcher is empty.
        :rtype: ``tuple[list[PRED_AR_T], typing.Any]``

        * The first element will be a list of length ``numHeads``, representing the
          output from the model. Since the output of the model will always have
          a dimension representing the batch size, and this function only returns
          the result of running a single sequence, the dimension representing
          the batch size is removed. In other words, running the model on a
          single example would give a profile of shape
          ``(1 x output-length x num-tasks)``.
          But this function will remove that, so you will get an array of shape
          ``(output-length x numTasks)``

        * The second element will be the label you passed in with the original
          sequence.

        Graphically::

            ( [<head-1-profile>, <head-2-profile>, ...],
              label)

        If the batcher doesn't have any output ready but does have some work in the input
        queue, then calling this function will block until the calculation is complete.
        If there is output ready, then this function will not block.
        """
        if not self._outWaiting:
            if self._inWaiting:
                # There are inputs that have not been processed. Run the batch.
                self.runBatch()
            else:
                raise queue.Empty("There are no outputs ready, and the input queue is empty.")
        rets = []
        ret = self._outQueue.pop()
        self._outWaiting -= 1
        rets.append(ret)
        for _ in range(ret["numTiles"] - 1):
            rets.append(self._outQueue.pop())
            self._outWaiting -= 1
        outputWidth = rets[-1]["tileStart"] + self._outputLength
        headProfiles = [np.zeros((outputWidth, tasks)) for tasks in self._tasksPerHead]
        headNumPreds = [np.zeros((outputWidth, tasks)) for tasks in self._tasksPerHead]
        for ret in rets:
            preds = ret["preds"]
            outputStart = ret["tileStart"]
            outputStop = outputStart + self._outputLength
            for h in range(len(self._tasksPerHead)):
                logits = preds[h]
                logcounts = preds[h + len(self._tasksPerHead)]
                headProfiles[h][outputStart:outputStop] += logitsToProfile(logits, logcounts)
                headNumPreds[h][outputStart:outputStop] += 1
        for h in range(len(self._tasksPerHead)):
            assert np.min(headNumPreds[h]) == 1, \
                "Missed placing an output somewhere! Please report this bug."
            headProfiles[h] /= headNumPreds[h]

        return (headProfiles, rets[0]["label"])


class ThreadedBatchPredictor:
    """Mirrors the API of :py:class:`~BatchPredictor`, but predicts in a separate thread.

    This can give you a performance boost, and also lets you shut down the
    predictor thread when you don't need it (thus freeing the GPU for other
    things). Supports the ``with`` statement to only turn on the batcher when
    you're using it, or you can leave it running in the background.

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
        # But you can spin it up if you need it again:
        with predictor:
            # use the predictor some more.

    The batcher guarantees that the order in which you get results is the same as
    the order you submitted them in, even though the internal calculations may
    happen out-of-order.

    :param modelFname: The name of the model to use to make predictions.
    :param batchSize: The number of samples to calculate at once.
    :param start: Should the predictor start right away? This should be False
        if you're going to use this ThreadedBatchPredictor inside a context manager
        (i.e., a ``with`` statement).
    :param numThreads: How many predictors should be spawned?
        I recommend 2 or 3.
    :param produceProfiles: If True, then you can call getOutputProfile()
        If False (the default), then you can only call getProfile().
        You have to specify this before starting the batcher because the profile
        production is done in parallel and so this class needs to know what sort
        of output you will want so it can have it ready for you when you need it.
    :param quiet: If True, redirect all stdout from tensorflow to the trash.

    """

    def __init__(self, modelFname: str, batchSize: int, start: bool = False,
                 numThreads: int = 1, produceProfiles: bool = False,
                 quiet: bool = False) -> None:
        """Build the batch predictor."""
        logUtils.debug(f"Creating threaded batch predictor for model {modelFname}.")
        self._batchSize = batchSize
        self._modelFname = modelFname
        self._batchSize = batchSize
        self._produceProfiles = produceProfiles
        # Since I'll be putting things in and taking them out often,
        # I'm going to use a queue data structure, where those operations
        # are efficient.
        self._batchers = None
        self._numThreads = numThreads
        self.running = False
        self._quiet = quiet
        self._contextDepth = 0
        if start:
            self.start()

    def __enter__(self):
        """Start up a context manager.

        Used in a context manager, this is the first thing that gets called
        inside a with statement.

        This context manager is reusable, meaning that it can be nested like this::

            with predictor:
                # blah, blah, blah
                with predictor:
                    # make predictions

        In this case, the inner call will simply keep batcher alive, and it will
        only shut down when the outermost context exits.
        """
        self._contextDepth += 1
        if self._contextDepth == 1:
            self.start()

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):  # noqa: ANN001
        """When leaving a context manager's with statement, shut down the batcher."""
        self._contextDepth -= 1
        if self._contextDepth == 0:
            # We're out of the last context manager now.
            self.stop()
        if exceptionType is not None:
            return False
        del exceptionValue  # Disable unused warning
        del exceptionTraceback  # Disable unused warning
        return True

    def start(self) -> None:
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
                nextInQueue = CrashQueue(maxsize=10000)
                nextOutQueue = CrashQueue(maxsize=10000)
                self._inQueues.append(nextInQueue)
                self._outQueues.append(nextOutQueue)
                nextBatcher = multiprocessing.Process(
                    target=_batcherThread,
                    args=(self._modelFname, self._batchSize, nextInQueue,
                          nextOutQueue, self._produceProfiles, self._quiet),
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
        if self.running:
            self.stop()

    def stop(self) -> None:
        """Shut down the processor thread.

        :raise ValueError: If you try to stop a predictor that hasn't started any batchers.
        """
        if self.running:
            if logUtils is not None:  # logUtils may be None if stop() is called from __del__.
                logUtils.debug("Shutting down threaded batcher.")
            if self._batchers is None:
                raise ValueError("Attempting to shut down a running ThreadedBatchPredictor"
                                 "When its _batchers is None.")
            for i in range(self._numThreads):
                self._inQueues[i].put("shutdown")
                self._inQueues[i].close()
                self._batchers[i].join(5)  # Wait 5 seconds.
                if self._batchers[i].exitcode is None:
                    # The process failed to die. Kill it more forcefully.
                    self._batchers[i].terminate()
                self._batchers[i].join(5)  # Wait 5 seconds.
                self._batchers[i].close()
                self._outQueues[i].close()
            del self._inQueues
            del self._batchers
            del self._outQueues
            # Explicitly set None so that start won't panic.
            self._batchers = None
            self.running = False
        elif logUtils is not None:
            logUtils.warning("Attempting to stop a batcher that is already stopped.")

    def clear(self) -> None:
        """Reset the batcher, emptying any queues and reloading the model.

        This also starts the batcher.
        """
        logUtils.info(f"Clearing threaded batcher. Canceling {self._inFlight} predictions.")
        if self.running:
            self.stop()
        self.start()

    def submitOHE(self, sequence: ONEHOT_AR_T, label: typing.Any) -> None:
        """Submit a one-hot-encoded sequence.

        :param sequence: An ``(input-length x NUM_BASES)`` ndarray containing the
            one-hot encoded sequence to predict.
        :param label: Any (picklable) object; it will be returned with the prediction.
        """
        if not self.running:
            logUtils.warning("Submitted a query when the batcher is stopped. Starting.")
            self.start()
        q = self._inQueues[self._inQueueIdx]
        query = (sequence, label)
        q.put(query)
        self._outQueueOrder.appendleft(self._inQueueIdx)
        # Assign work in a round-robin fashion.
        self._inQueueIdx = (self._inQueueIdx + 1) % self._numThreads
        self._inFlight += 1

    def submitString(self, sequence: str, label: typing.Any) -> None:
        """Submit a given sequence for prediction.

        :param sequence: A string of length ``input-length``
        :param label: Any (picklable) object. Label will be returned to you with the
            prediction.
        """
        seqOhe = oneHotEncode(sequence)
        self.submitOHE(seqOhe, label)

    def outputReady(self) -> bool:
        """Is there any output ready for you?

        If output is ready, then calling ``getOutput()`` will give a result immediately.

        :return: ``True`` if the batcher is sitting on results, and ``False`` otherwise.
        """
        if self._inFlight:
            outIdx = self._outQueueOrder[-1]
            return not self._outQueues[outIdx].empty()
        return False

    def empty(self) -> bool:
        """Is the batcher totally idle?

        If the batcher is not empty, then you can call ``getOutput()``, though
        it may block if it needs to run a calculation.

        :return: ``True`` if there are no predictions at all in the queue.
        """
        return self._inFlight == 0

    def getOutput(self) -> tuple[list, typing.Any]:
        """Get a single output.

        :return: The model's predictions.
        :raise queue.Empty: if you try to get an output when the batcher is empty.
        :rtype: ``tuple[list[LOGIT_AR_T, LOGCOUNT_T], typing.Any]``

        Same semantics and blocking behavior as
        :py:meth:`BatchPredictor.getOutput<bpreveal.utils.BatchPredictor.getOutput>`.
        """
        assert not self._produceProfiles, "Cannot getOutput() on a batcher that is set " \
            "to produce profiles. Use getOutputProfile() instead."
        nextQueueIdx = self._outQueueOrder.pop()
        if self._outQueues[nextQueueIdx].empty():
            if self._inFlight:
                # There are inputs that have not been processed. Run the batch.
                self._inQueues[nextQueueIdx].put("finishBatch")
            else:
                raise queue.Empty("The batcher is empty; cannot getOutput().")
        ret = self._outQueues[nextQueueIdx].get()
        self._inFlight -= 1
        return ret

    def getOutputProfile(self) -> tuple[list, typing.Any]:
        """Get a single output, but in profile space instead of logits.

        :return: The model's predictions.
        :raise queue.Empty: if you try to get an output when the batcher is empty.
        :rtype: ``tuple[list[PRED_AR_T], typing.Any]``

        Same semantics and blocking behavior as
        :py:meth:`BatchPredictor.getOutputProfile<bpreveal.utils.BatchPredictor.getOutputProfile>`.
        """
        assert self._produceProfiles, "Cannot getOutputProfile unless the batcher was " \
            "configured with produceProfile=True."
        nextQueueIdx = self._outQueueOrder.pop()
        if self._outQueues[nextQueueIdx].empty():
            if self._inFlight:
                # There are inputs that have not been processed. Run the batch.
                self._inQueues[nextQueueIdx].put("finishBatch")
            else:
                raise queue.Empty("The batcher is empty; cannot getOutputProfile().")
        ret = self._outQueues[nextQueueIdx].get()
        self._inFlight -= 1
        return ret


def _batcherThread(modelFname: str, batchSize: int, inQueue: CrashQueue,
                   outQueue: CrashQueue, produceProfiles: bool, quiet: bool) -> None:
    """Run batches from the ``ThreadedBatchPredictor`` in this separate thread.

    If produceProfiles is True, then this will emit results from getOutputProfile().
    Otherwise, it will emit results from getOutput().
    Since this thread will put outputs into the queue before the user has asked,
    we need to know a priori whether we should produce profiles.

    .. note::
        Sets :py:data:`bpreveal.internal.constants.GLOBAL_TENSORFLOW_LOADED`.
    """
    assert not constants.getTensorflowLoaded(), "Cannot use the threaded predictor " \
        "after loading tensorflow."
    logUtils.debug("Starting subthread")
    # Instead of reinventing the wheel, the thread that actually runs the batches
    # just creates a BatchPredictor.
    batcher = BatchPredictor(modelFname, batchSize, quiet=quiet)
    predsInFlight = 0
    numWaits = 0
    getOutput = batcher.getOutput
    if produceProfiles:
        getOutput = batcher.getOutputProfile
    while True:
        # No timeout because this batcher could be waiting for a very long time to get
        # inputs.
        try:
            inVal = inQueue.get(timeout=0.1)
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
                    outQueue.put(getOutput())
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
                    outQueue.put(getOutput())
                    predsInFlight -= 1
            case "finishBatch":
                while predsInFlight:
                    outQueue.put(getOutput())
                    predsInFlight -= 1
            case "shutdown":
                # End the thread.
                logUtils.debug("Shutdown signal received.")
                return
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
