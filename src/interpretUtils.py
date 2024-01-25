"""A big ol' module that contains high-efficiency tools for calculating shap scores."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '4'
from bpreveal import utils
import pysam
import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Iterator, Iterable
import tqdm
import h5py
import logging
import pybedtools
import multiprocessing
from bpreveal import ushuffle
from bpreveal.utils import ONEHOT_T, ONEHOT_AR_T, IMPORTANCE_T, \
    H5_CHUNK_SIZE, MODEL_ONEHOT_T
import ctypes


class Query:
    """This is what is passed to the batcher.

    It has three things.

    :param sequence: The ``(input-length, 4)`` one-hot encoded sequence of the current base.
    :param passData: As with Result objects, is either a tuple of ``(chromName, position)``
        (for when you have a bed file) or a string with a fasta description line
        (for when you're starting with a fasta). If you're using the
        :py:class:`~ListGenerator`, then it can be anything.
    :param index: Indicates which output slot this data should be put in.
        Since there's no guarantee that the results will arrive in order, we have to
        track which query was which.
    """

    sequence: ONEHOT_AR_T
    passData: Any
    index: int

    def __init__(self, oneHotSequence: ONEHOT_AR_T, passData: Any, index: int):
        """Just store the given data."""
        self.sequence = oneHotSequence
        self.passData = passData
        self.index = index

    def __str__(self):
        """Make a string with information about this Query."""
        fmtStr = "seqSize {0:s} passData {1:s} index {2:d}"
        return fmtStr.format(str(self.sequence.shape), str(self.passData), self.index)


class Result:
    """The base class for results.

    Subclassed by :py:class:`~PisaResult` and :py:class:`flatResult`.
    """


class PisaResult(Result):
    """This is the output from shapping a single base.

    It contains a few things.

    :param inputPrediction: A scalar floating point value, of the predicted logit
        from the input sequence at the base that was being shapped.

    :param shufflePredictions: A ``(numShuffles,)`` numpy array of the logits
        returned by running predictions on the reference sequence, again evaluated
        at the position of the base that was being shapped.

    :param sequence: is a ``(receptive-field, 4)`` numpy array of the
        one-hot encoded input sequence.

    :param shap: is a ``(receptive-field, 4)`` numpy array of shap scores.

    :param passData: is data that is not touched by the batcher, but added by
        the generator and necessary for creating the output file.
        If the generator is reading from a bed file, then it is a tuple of (chromName, position)
        and that data should be used to populate the coords_chrom and coords_base fields.
        If the generator was using a fasta file, it is the title line from the original fasta,
        with its leading ``>`` removed.

    :param index: Indicates which address the data should be stored at in the output hdf5.
        Since there's no order guarantee when you're receiving data, we have to keep track
        of the order in the original input.
    """

    def __init__(self, inputPrediction: npt.NDArray[np.float32],
                 shufflePredictions: npt.NDArray[np.float32],
                 sequence: ONEHOT_AR_T,
                 shap: npt.NDArray[IMPORTANCE_T],
                 passData: Any,
                 index: int):
        """Just store all the provided data."""
        self.inputPrediction = inputPrediction
        self.shufflePredictions = shufflePredictions
        self.sequence = sequence
        self.shap = shap
        self.passData = passData
        self.index = index


class FlatResult(Result):
    """A Result object that is given to savers for flat interpretation analysis.

    :param sequence: A one-hot encoded array of the sequence that was explained, of shape
        ``(input-length, 4)``
    :param shap: An array of shape ``(input-length, 4)``, containing the shap scores.

    :param passData: is a (picklable) object that is passed through from the generator.
        For bed-based interpretations, it will be a three-tuple of (chrom, start, end)
        (The start and end positions correspond to the INPUT to the model, so they are inflated
        with respect to the bed file.)
        For fasta-based interpretations it will be a string.

    :param index: Gives the position in the output hdf5 where the scores should be saved.
    """

    def __init__(self, sequence: ONEHOT_AR_T,
                 shap: npt.NDArray[IMPORTANCE_T],
                 passData: Any, index: int):
        """Just store all the provided data."""
        self.sequence = sequence
        self.shap = shap
        self.passData = passData
        self.index = index

    def __str__(self) -> str:
        """Make a string with useful information."""
        minVal = np.min(self.shap)
        maxVal = np.max(self.shap)
        formatStr = "index: {0:d}, sequence: {1:s}, shap: {2:s}, "\
            "passData: {3:s}, min {4:f} max {5:f}"
        return formatStr.format(self.index, str(self.sequence[:5]), str(self.shap[:5]),
                                str(self.passData), minVal, maxVal)


class Generator:
    """This is the base class for generating pisa samples."""

    def __init__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Query]:
        raise NotImplementedError()

    def __next__(self):
        """Get the next step for the iterator.

        If your generator's __iter__ returns self, you must implement __next__.
        """
        raise NotImplementedError()

    def construct(self):
        """Set up the generator (in the child thread).

        When the generator is about to start, this method is called once before
        requesting the iterator.
        """
        raise NotImplementedError()

    def done(self):
        """When the batcher is done, this is called to free any allocated resources."""
        raise NotImplementedError()


class Saver:
    """The saver runs in the main thread, so it's safe to open files and stuff in __init__.

    Descendants of this class shall receive Results from the batcher and save them to an
    appropriate structure, usually on disk.
    The user creates a Saver object in the main thread, and then the saver gets
    construct()ed inside a separate thread created by the runners. Therefore, you
    should only create members that are serializable in __init__.
    """

    def __init__(self):
        raise NotImplementedError()

    def construct(self):
        """Do any setup necessary in the child thread.

        This function should actually open the output files, since it will
        be called from inside the actual saver thread.
        """
        raise NotImplementedError()

    def add(self, result: Result):
        """Add the given Result to wherever you're saving them out.

        Note that this will not be called with None at the end of the run,
        that is special-cased in the code that runs your saver.
        """
        del result
        raise NotImplementedError()

    def parentFinish(self):
        """This is called in the parent thread when the saver is done.

        Usually, there's nothing to do, but the parent thread might need
        to close shared memory, so this function is guaranteed to be called
        by the Runners.
        """
        logging.debug("Called parentFinish on the base Saver class.")

    def done(self):
        """Called when the batcher is complete (indicated by putting a None in its output queue).

        Now is the time to close all of your files.
        """
        logging.debug("Called done on the base Saver class.")


class FlatRunner:
    """Runs shap scores.

    I try to avoid class-based wrappers around simple things, but this is not simple.
    This class creates threads to read in data, and then creates two threads that run interpretation
    samples in batches. Finally, it takes the results from the interpretation thread and saves
    those out to an hdf5-format file.

    :param modelFname: is the name of the model on disk
    :param headID: The head to shap.
    :param taskIDs: The tasks to shap, obviously.
        Typically, you'd want to interpret all of the tasks in a head, so for a two-task
        head, taskIDs would be ``[0,1]``.
    :param batchSize: is the *shap* batch size, which should be your usual batch size divided
        by the number of shuffles. (since the original sequence and shuffles are run together)
    :param generator: is a Generator object that will be passed to _generatorThread
    :param saver: is a Saver object that will be used to save out the data.
    :param numShuffles: is the number of reference sequences that should be generated for each
        shap evaluation. (I recommend 20 or so)
    :param kmerSize: Should the shuffle preserve k-mer distribution? If kmerSize == 1, then no.
        If kmerSize > 1, preserve the distribution of kmers of the given size.
    :param profileFname: is an optional parameter. If provided, profiling data (i.e., performance
        of this code) will be written to the given file name. Note that this has
        nothing to do with genomic profiles, it's just benchmarking data for this code.
    """

    def __init__(self, modelFname: str, headID: int, numHeads: int, taskIDs: list[int],
                 batchSize: int, generator: Generator, profileSaver: Saver,
                 countsSaver: Saver, numShuffles: int, kmerSize: int,
                 profileFname: Optional[str] = None):
        logging.info("Initializing interpretation runner.")
        self._profileSaver = profileSaver
        self._countsSaver = countsSaver
        self._profileInQueue = multiprocessing.Queue(1000)
        self._countsInQueue = multiprocessing.Queue(1000)
        self._profileOutQueue = multiprocessing.Queue(1000)
        self._countsOutQueue = multiprocessing.Queue(1000)

        def ap(txt) -> Optional[str]:
            """Add some text to the profileFname"""
            if profileFname is None:
                return None
            return profileFname + txt

        self._genThread = multiprocessing.Process(target=_generatorThread,
            args=([self._profileInQueue, self._countsInQueue], generator,
                  ap("_gen.dat")),
            daemon=True)

        self._profileBatchThread = multiprocessing.Process(target=_flatBatcherThread,
            args=(modelFname, batchSize, self._profileInQueue, self._profileOutQueue,
                  headID, numHeads, taskIDs, numShuffles, "profile", kmerSize,
                  ap("_profile.dat")),
            daemon=True)
        self._countsBatchThread = multiprocessing.Process(target=_flatBatcherThread,
            args=(modelFname, batchSize, self._countsInQueue, self._countsOutQueue,
                  headID, numHeads, taskIDs, numShuffles, "counts", kmerSize,
                  ap("_counts.dat")),
            daemon=True)

        self._profileSaverThread = multiprocessing.Process(target=_saverThread,
            args=(self._profileOutQueue, profileSaver, ap("_profileWrite.dat")),
            daemon=True)
        self._countsSaverThread = multiprocessing.Process(target=_saverThread,
            args=(self._countsOutQueue, countsSaver, ap("_countsWrite.dat")),
            daemon=True)
        if profileFname is not None:
            import pstats
            for end in ["_gen", "_profile", "_counts", "_profileWrite", "_countsWrite"]:
                with open(ap(end + ".txt"), "w") as fp:  # type: ignore
                    s = pstats.Stats(ap(end + ".dat"), stream=fp)
                    s.sort_stats('cumulative')
                    s.print_stats()

    def run(self):
        """Start up the threads and waits for them to finish."""
        logging.info("Beginning flat run.")
        self._genThread.start()
        logging.debug("Started generator.")
        self._profileBatchThread.start()
        self._countsBatchThread.start()
        logging.debug("Started batchers. Starting savers.")
        self._profileSaverThread.start()
        self._countsSaverThread.start()
        logging.info("All processes started. Beginning main loop.")
        self._genThread.join()
        logging.debug("Generator joined.")
        self._profileBatchThread.join()
        self._countsBatchThread.join()
        logging.debug("Batchers joined.")
        self._profileSaverThread.join()
        self._countsSaverThread.join()
        self._profileSaver.parentFinish()
        self._countsSaver.parentFinish()
        logging.info("Savers complete. Done.")


class PisaRunner:
    """Tool to run pisa batches.

    I try to avoid class-based wrappers around simple things, but this is not simple.
    This class creates threads to read in data, and then creates a thread that runs pisa
    samples in batches. Finally, it takes the results from the pisa thread and saves
    those out to an hdf5-format file.

    :param modelFname: is the name of the model on disk
    :param headID: The head to shap.
    :param taskID: The task to shap.
    :param batchSize: is the *shap* batch size, which should be your usual batch size divided
        by the number of shuffles. (since the original sequence and shuffles are run together)
    :param generator: is a Generator object that will be passed to _generatorThread
    :param saver: is a Saver object that will be used to save out the data.
    :param numShuffles: is the number of reference sequences that should be generated for each
        shap evaluation. (I recommend 20 or so)
    :param receptiveField: is the receptive field of the model. To save on writing a lot of
        zeroes, the result objects only contain bases that are in the receptive field
        of the base being shapped.
    :param kmerSize: Should the shuffle preserve k-mer distribution? If kmerSize == 1, then no.
        If kmerSize > 1, preserve the distribution of kmers of the given size.
    :param profileFname: is an optional parameter. If provided, profiling data (i.e., performance
        of this code) will be written to the given file name. Note that this has
        nothing to do with genomic profiles, it's just benchmarking data for this code.
    """

    def __init__(self, modelFname: str, headID: int, taskID: int, batchSize: int,
                 generator: Generator, saver: Saver, numShuffles: int,
                 receptiveField: int, kmerSize: int, profileFname: Optional[str] = None):
        logging.info("Initializing pisa runner.")
        self._inQueue = multiprocessing.Queue(1000)
        self._outQueue = multiprocessing.Queue(1000)

        def ap(txt):
            """Add some text to the profileFname"""
            if profileFname is None:
                return None
            return profileFname + txt

        self._genThread = multiprocessing.Process(target=_generatorThread,
            args=([self._inQueue], generator, ap("_generate.dat")),
            daemon=True)
        self._batchThread = multiprocessing.Process(target=_pisaBatcherThread,
                                                    args=(modelFname, batchSize,
                                                          self._inQueue, self._outQueue, headID,
                                                          taskID, numShuffles, receptiveField,
                                                          kmerSize,
                                                          ap("batcher.dat")),
                                                    daemon=True)
        self._saver = saver

    def run(self):
        """Start up the threads and waits for them to finish."""
        logging.info("Beginning pisa run.")
        self._genThread.start()
        logging.debug("Started generator.")
        self._batchThread.start()
        logging.debug("Started batcher.")
        _saverThread(self._outQueue, self._saver)
        logging.info("Saver complete. Finishing.")
        self._genThread.join()
        logging.debug("Generator joined.")
        self._batchThread.join()
        logging.debug("Batcher joined.")
        self._saver.parentFinish()
        logging.info("Done.")


class FlatListSaver(Saver):
    """A simple Saver that holds the results in memory so you can use them immediately.

    Since the Saver is created in its own thread, just storing the results
    in this object doesn't work - they get removed when the writer
    process completes.
    So we need to create some shared memory. This Saver takes care of that
    for sequences and shap scores, but discards passData, since we don't know
    a priori how large passData objects will be. In a typical use case, you'd
    use this in situations where you already know which sequence is which,
    so saving passData doesn't really make sense anyway.

    :param numSamples: The total number of samples that this saver will get.
        It has to know this during construction so it can allocate
        enough memory.
    :param inputLength: The input length of the model.
    """

    inputLength: int
    numSamples: int
    shap: npt.NDArray[IMPORTANCE_T]
    seq: ONEHOT_AR_T

    def __init__(self, numSamples: int, inputLength: int):
        self.inputLength = inputLength
        self.numSamples = numSamples
        # Note that these arrays are shared with the child.
        # Also note that the internal shared arrays are float32 or int8,
        # not float16 like normal importance scores (this is because
        # there's a float in ctypes, but not a float16.)
        self._outShapArray = multiprocessing.Array(ctypes.c_float, numSamples * inputLength * 4)
        self._outSeqArray = multiprocessing.Array(ctypes.c_int8, numSamples * inputLength * 4)
        logging.debug("Created shared arrays for the list saver.")

    def construct(self):
        """Set up the data sets.

        This is run in the child process.
        """
        # Do this in the child so the parent doesn't accidentally mess with
        # an empty array.
        self._results = []
        logging.debug("Constructed child thread flat list saver.")

    def parentFinish(self):
        """Extract the data from the child process.

        This must be called to load the shap data from the child,
        since it's currently packed away inside a linear array.
        I could just expose _outShapArray, but this reorganizes it in a much
        more intuitive way.
        """
        self.shap = np.zeros((self.numSamples, self.inputLength, 4), dtype=np.float32)
        self.seq = np.zeros((self.numSamples, self.inputLength, 4), dtype=ONEHOT_T)
        for idx in range(self.numSamples):
            for outOffset in range(self.inputLength):
                for k in range(4):
                    readHead = idx * self.inputLength * 4 + outOffset * 4 + k
                    self.shap[idx, outOffset, k] = self._outShapArray[readHead]
                    self.seq[idx, outOffset, k] = self._outSeqArray[readHead]
        logging.debug("Finished list saver in parent thread. Your data are ready!")

    def done(self):
        """Copy over the data from the child thread to the parent.

        This method is called from the child process.
        """
        for r in self._results:
            idx = r.index
            svals = r.shap
            seqvals = r.sequence
            for outOffset in range(self.inputLength):
                for k in range(4):
                    writeHead = idx * self.inputLength * 4 + outOffset * 4 + k
                    self._outShapArray[writeHead] = svals[outOffset, k]
                    oneHotBase = ctypes.c_int8(int(seqvals[outOffset, k]))
                    self._outSeqArray[writeHead] = oneHotBase
        logging.debug("Finished list saver in child thread.")

    def add(self, result: FlatResult):
        """Add the result to the internal list.

        This is called from the child process.
        """
        self._results.append(result)


class FlatH5Saver(Saver):
    """Saves the shap scores to the output file.

    :param outputFname: is the name of the hdf5-format file that the shap scores will
        be deposited in.
    :param numSamples: is the number of regions (i.e., bases) that pisa will be run on.
        This is needed because we store reference predictions.
    :param inputLength: The input length of the model.
    :param genome: (Optional) Gives the name of a fasta-format file that contains
        the genome of the organism. If provided, then chromosome name and size information
        will be included in the output, and, additionally, two other datasets will be
        created: coords_chrom, and coords_base.
    :param useTqdm: Should a progress bar be displayed?
    """

    chunkShape: tuple[int, int, int]
    _outputFname: str
    numSamples: int
    genomeFname: Optional[str]
    inputLength: int
    _useTqdm: bool = False
    _outFile: h5py.File
    _chunksToWrite: dict[int, dict]
    pbar: Optional[tqdm.tqdm] = None

    def __init__(self, outputFname: str, numSamples: int, inputLength: int,
                 genome: Optional[str] = None, useTqdm: bool = False):
        self.chunkShape = (min(H5_CHUNK_SIZE, numSamples), inputLength, 4)
        self._outputFname = outputFname
        self.numSamples = numSamples
        self.genomeFname = genome
        self.inputLength = inputLength
        self._useTqdm = useTqdm

    def construct(self):
        """Set up the data sets for writing.

        This is called inside the child thread.
        """
        logging.info("Initializing saver.")
        self._outFile = h5py.File(self._outputFname, "w")
        self._chunksToWrite = dict()

        self._outFile.create_dataset("hyp_scores",
                                     (self.numSamples, self.inputLength, 4),
                                     dtype=IMPORTANCE_T, chunks=self.chunkShape,
                                     compression='gzip')
        self._outFile.create_dataset('input_seqs',
                                     (self.numSamples, self.inputLength, 4),
                                     dtype=ONEHOT_T, chunks=self.chunkShape,
                                     compression='gzip')
        if self.genomeFname is not None:
            self._loadGenome()
        else:
            self._outFile.create_dataset("descriptions", (self.numSamples,),
                                         dtype=h5py.string_dtype('utf-8'))
        logging.debug("Saver initialized, hdf5 file created.")

    def _loadGenome(self):
        """Does a few things.

        1. It creates chrom_names and chrom_sizes datasets in the output hdf5.
          These two datasets just contain the (string) names and (unsigned) sizes for each one.
        2. It populates these datasets.
        3. It creates two datasets: coords_chrom and coords_base, which store, for every
           shap value calculated, what chromosome it was on, and where on that chromosome
           it was. These fields are populated during data receipt, because we don't have
           an ordering guarantee when we get data from the batcher.
        """
        logging.info("Loading genome information.")
        with pysam.FastaFile(self.genomeFname) as genome:  # type: ignore
            self._outFile.create_dataset("chrom_names",
                                         (genome.nreferences,),
                                         dtype=h5py.string_dtype(encoding='utf-8'))
            self.chromNameToIdx = dict()
            posDtype = 'u4'
            for chromName in genome.references:
                # Are any chromosomes longer than 2^31 bp long? If so (even though it's unlikely),
                # I must increase the storage size for positions.
                if genome.get_reference_length(chromName) > 2**31:
                    posDtype = 'u8'

            self._outFile.create_dataset("chrom_sizes", (genome.nreferences, ), dtype=posDtype)

            for i, chromName in enumerate(genome.references):
                self._outFile['chrom_names'][i] = chromName
                self.chromNameToIdx[chromName] = i
                self._outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

            if genome.nreferences <= 255:  # type: ignore
                # If there are less than 255 chromosomes,
                # I only need 8 bits to store the chromosome ID.
                chromDtype = 'u1'
            elif genome.nreferences <= 65535:  # type: ignore
                # If there are less than 2^16 chromosomes, I can use 16 bits.
                chromDtype = 'u2'
            else:
                # If you have over four billion chromosomes, you deserve to have code
                # that will break. So I just assume that 32 bits will be enough.
                chromDtype = 'u4'
            self._outFile.create_dataset("coords_chrom", (self.numSamples, ), dtype=chromDtype)

            self._outFile.create_dataset("coords_start", (self.numSamples, ), dtype=posDtype)
            self._outFile.create_dataset("coords_end", (self.numSamples, ), dtype=posDtype)
        logging.debug("Genome data loaded.")

    def done(self):
        """Close up shop.

        Called in the child process.
        """
        logging.debug("Saver closing.")
        if self.pbar is not None:
            self.pbar.close()
        self._outFile.close()

    def add(self, result: FlatResult):
        """Add the given result to the output file.

        :param result: The output from the batcher.
        """
        if self.pbar is None and self._useTqdm:
            # Initialize the progress bar.
            # I would do it earlier, but starting it before the batchers
            # have warmed up skews the stats.
            self.pbar = tqdm.tqdm(total=self.numSamples)
        if self.pbar is not None:
            self.pbar.update()
        index = result.index
        # Which chunk does this result go in?
        chunkIdx = index // self.chunkShape[0]
        # And where in the chunk does it go?
        chunkOffset = index % self.chunkShape[0]
        if chunkIdx not in self._chunksToWrite:
            # Allocate a new chunk.
            # Note that we can't just statically allocate one chunk buffer because we don't
            # guarantee the order that the results will arrive in.
            numToAdd = self.chunkShape[0]
            if chunkIdx == self.numSamples // self.chunkShape[0]:
                numToAdd = self.numSamples % self.chunkShape[0]
            curChunkShape = (numToAdd, self.chunkShape[1], self.chunkShape[2])
            self._chunksToWrite[chunkIdx] = {
                "hyp_scores": np.empty(curChunkShape, dtype=IMPORTANCE_T),
                "input_seqs": np.empty(curChunkShape, dtype=ONEHOT_T),
                "numToAdd": numToAdd,
                "writeStart": chunkIdx * self.chunkShape[0],
                "writeEnd": chunkIdx * self.chunkShape[0] + numToAdd}
        curChunk = self._chunksToWrite[chunkIdx]
        curChunk["numToAdd"] -= 1
        curChunk["hyp_scores"][chunkOffset] = result.shap
        curChunk["input_seqs"][chunkOffset] = result.sequence
        if curChunk["numToAdd"] == 0:
            # We added the last missing entry to this chunk. Write it to the file.
            start = curChunk["writeStart"]
            end = curChunk["writeEnd"]
            self._outFile["input_seqs"][start:end, :, :] = curChunk["input_seqs"]
            self._outFile["hyp_scores"][start:end, :, :] = curChunk["hyp_scores"]
            # Free the memory held by this chunk.
            del self._chunksToWrite[chunkIdx]

        # Okay, now we either add the description line, or add a genomic coordinate.
        # These are not chunked, since the data aren't compressed.
        if self.genomeFname is not None:
            self._outFile["coords_chrom"][index] = self.chromNameToIdx[result.passData[0]]
            self._outFile["coords_start"][index] = result.passData[1]
            self._outFile["coords_end"][index] = result.passData[2]
        else:
            self._outFile["descriptions"][index] = result.passData


class PisaH5Saver(Saver):
    """Saves the shap scores to the output file.

    :param outputFname: is the name of the hdf5-format file that the shap scores
        will be deposited in.
    :param numSamples: is the number of regions (i.e., bases) that pisa will be run on.
    :param numShuffles: is the number of shuffles that are used to generate the reference.
        This is needed because we store reference predictions.
    :param receptiveField: How wide is the model's receptive field?
    :param genome: an optional parameter, gives the name of a fasta-format file that contains
        the genome of the organism. If provided, then chromosome name and size information
        will be included in the output, and, additionally, two other datasets will be
        created: coords_chrom, and coords_base.
    :param useTqdm: Should a progress bar be displayed?
    """

    def __init__(self, outputFname: str, numSamples: int, numShuffles: int, receptiveField: int,
                 genome: Optional[str] = None, useTqdm: bool = False):
        logging.info("Initializing saver.")
        self._outputFname = outputFname
        self.numSamples = numSamples
        self.numShuffles = numShuffles
        self.genomeFname = genome
        self._useTqdm = useTqdm
        self.receptiveField = receptiveField
        self.chunkShape = (min(H5_CHUNK_SIZE, numSamples), receptiveField, 4)

    def construct(self):
        """Run in the child thread."""
        self._outFile = h5py.File(self._outputFname, "w")
        self._outFile.create_dataset("input_predictions",
                                     (self.numSamples,), dtype='f4')
        self._outFile.create_dataset("shuffle_predictions",
                                     (self.numSamples, self.numShuffles),
                                     dtype='f4')
        # self._outFile.create_dataset("shap",
        #                             (self.numSamples, self.receptiveField, 4),
        #                             dtype='f4')
        # TODO for later: Adding compression to the sequence here absolutely tanks performance
        # self._outFile.create_dataset('sequence',
        #                             (self.numSamples, self.receptiveField, 4),
        #                             dtype='u1')
        self._chunksToWrite = dict()

        self._outFile.create_dataset("shap",
                                     (self.numSamples, self.receptiveField, 4),
                                     dtype=IMPORTANCE_T, chunks=self.chunkShape,
                                     compression='gzip')
        self._outFile.create_dataset('sequence',
                                     (self.numSamples, self.receptiveField, 4),
                                     dtype=ONEHOT_T, chunks=self.chunkShape,
                                     compression='gzip')
        self.pbar = None
        if self._useTqdm:
            self.pbar = tqdm.tqdm(total=self.numSamples)
        if self.genomeFname is not None:
            self._loadGenome()
        else:
            self._outFile.create_dataset("descriptions", (self.numSamples,),
                                         dtype=h5py.string_dtype('utf-8'))
        logging.debug("Saver initialized, hdf5 file created.")

    def _loadGenome(self):
        """Does a few things.

        1. It creates chrom_names and chrom_sizes datasets in the output hdf5.
           These two datasets just contain the (string) names and (unsigned) sizes for each one.
        2. It populates these datasets.
        3. It creates two datasets: coords_chrom and coords_base, which store, for every
           shap value calculated, what chromosome it was on, and where on that chromosome
           it was. These fields are populated during data receipt, because we don't have
           an ordering guarantee when we get data from the batcher.
        """
        logging.info("Loading genome information.")
        with pysam.FastaFile(self.genomeFname) as genome:  # type: ignore
            self._outFile.create_dataset("chrom_names",
                                         (genome.nreferences,),
                                         dtype=h5py.string_dtype(encoding='utf-8'))
            self.chromNameToIdx = dict()
            posDtype = 'u4'
            for chromName in genome.references:
                # Are any chromosomes longer than 2^31 bp long? If so (even though it's unlikely),
                # I must increase the storage size for positions.
                if genome.get_reference_length(chromName) > 2**31:
                    posDtype = 'u8'

            self._outFile.create_dataset("chrom_sizes", (genome.nreferences, ), dtype=posDtype)

            for i, chromName in enumerate(genome.references):
                self._outFile['chrom_names'][i] = chromName
                self.chromNameToIdx[chromName] = i
                self._outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

            if genome.nreferences <= 255:  # type: ignore
                # If there are less than 255 chromosomes,
                # I only need 8 bits to store the chromosome ID.
                chromDtype = 'u1'
            elif genome.nreferences <= 65535:  # type: ignore
                # If there are less than 2^16 chromosomes, I can use 16 bits.
                chromDtype = 'u2'
            else:
                # If you have over four billion chromosomes, you deserve to have code
                # that will break. So I just assume that 32 bits will be enough.
                chromDtype = 'u4'
            self._outFile.create_dataset("coords_chrom", (self.numSamples, ), dtype=chromDtype)
            self._outFile.create_dataset("coords_base", (self.numSamples, ), dtype='u4')
        logging.debug("Genome data loaded.")

    def done(self):
        """Called from the child process."""
        logging.debug("Saver closing.")
        if self.pbar is not None:
            self.pbar.close()
        self._outFile.close()

    def add(self, result: PisaResult):
        """Add the given result to the output file.

        :param result: The output from the batcher.
        """
        if self.pbar is not None:
            self.pbar.update()
        index = result.index
        # Which chunk does this result go in?
        chunkIdx = index // self.chunkShape[0]
        # And where in the chunk does it go?
        chunkOffset = index % self.chunkShape[0]
        if chunkIdx not in self._chunksToWrite:
            # Allocate a new chunk.
            # Note that we can't just statically allocate one chunk buffer because we don't
            # guarantee the order that the results will arrive in.
            numToAdd = self.chunkShape[0]
            if chunkIdx == self.numSamples // self.chunkShape[0]:
                numToAdd = self.numSamples % self.chunkShape[0]
            curChunkShape = (numToAdd, self.chunkShape[1], self.chunkShape[2])
            self._chunksToWrite[chunkIdx] = {
                "shap": np.empty(curChunkShape, dtype=IMPORTANCE_T),
                "sequence": np.empty(curChunkShape, dtype=ONEHOT_T),
                "numToAdd": numToAdd,
                "writeStart": chunkIdx * self.chunkShape[0],
                "writeEnd": chunkIdx * self.chunkShape[0] + numToAdd}

        curChunk = self._chunksToWrite[chunkIdx]
        curChunk["numToAdd"] -= 1
        curChunk["shap"][chunkOffset] = result.shap
        curChunk["sequence"][chunkOffset] = result.sequence
        if curChunk["numToAdd"] == 0:
            # We added the last missing entry to this chunk. Write it to the file.
            start = curChunk["writeStart"]
            end = curChunk["writeEnd"]
            self._outFile["sequence"][start:end, :, :] = curChunk["sequence"]
            self._outFile["shap"][start:end, :, :] = curChunk["shap"]
            # Free the memory held by this chunk.
            del self._chunksToWrite[chunkIdx]
        self._outFile["input_predictions"][index] = result.inputPrediction
        self._outFile["shuffle_predictions"][index, :] = result.shufflePredictions
        # self._outFile["sequence"][index, :, :] = result.sequence
        # self._outFile["shap"][index, :, :] = result.shap
        # Okay, now we either add the description line, or add a genomic coordinate.
        if self.genomeFname is not None:
            self._outFile["coords_chrom"][index] = self.chromNameToIdx[result.passData[0]]
            self._outFile["coords_base"][index] = result.passData[1]
        else:
            self._outFile["descriptions"][index] = result.passData


class ListGenerator(Generator):
    """A very simple Generator that is initialized with an iterable of strings.

    (A list of strings is an iterable, but this works with generator functions
    and other things too!)

    :param sequences: Any iterable that yields strings, like a list of strings.
        Note that this function immediately converts whatever you pass in
        into a list, so very large iterables will consume a lot of memory.
    :param passDataList: (optional) Will be passed through the batcher to
        the saver.
    """

    def __init__(self, sequences: Iterable[str],
                 passDataList: Optional[list] = None):
        self._sequences = list(sequences)
        self.numSamples = len(self._sequences)
        self.inputLength = len(self._sequences[0])
        self._passDataList = passDataList

    def construct(self):
        """Set up stuff in the child thread.

        Note that this doesn't load data - because the child thread is
        forked from the parent, it already contains the lists of data.
        """
        self._indexes = list(range(len(self._sequences)))
        if self._passDataList is None:
            self._passData = [""] * len(self._sequences)
        else:
            self._passData = self._passDataList

    def done(self):
        """Called in the child thread, does nothing."""

    def __iter__(self):
        """Returns self, because Generators are iterable."""
        return self

    def __next__(self) -> Query:
        """Get the next query, or raise StopIteration."""
        if len(self._sequences) == 0:
            raise StopIteration()
        # Eat the first sequence, index, and passData, then return a query.
        oneHotSequence = utils.oneHotEncode(self._sequences[0])
        self._sequences = self._sequences[1:]
        idx = self._indexes[0]
        self._indexes = self._indexes[1:]
        passData = self._passData[0]
        self._passData = self._passData[1:]
        q = Query(oneHotSequence, passData, idx)
        return q


class FastaGenerator(Generator):
    """Reads a fasta file from disk and generates Queries from it.

    :param fastaFname: The name of the fasta-format file containing
        query sequences.
    """

    def __init__(self, fastaFname: str):
        logging.info("Creating fasta generator.")
        self.fastaFname = fastaFname
        self.nowStop = False
        numRegions = 0

        with open(fastaFname, "r") as fp:
            for line in fp:
                if len(line) > 0 and line[0] == ">":
                    numRegions += 1
        # Iterate through the input file real quick and find out how many regions I'll have.
        self.numRegions = numRegions
        self.index = 0
        logging.info("Fasta generator initialized with {0:d} regions.".format(self.numRegions))

    def construct(self):
        """Open the file and start reading."""
        logging.info("Constructing fasta generator in its thread.")
        self.fastaFile = open(self.fastaFname, "r")
        self.nextSequenceID = self.fastaFile.readline()[1:].strip()
        # [1:] to get rid of the '>'.
        logging.debug("Initial sequence to read: {0:s}".format(self.nextSequenceID))

    def done(self):
        """Close the Fasta file."""
        logging.info("Closing fasta generator.")
        self.fastaFile.close()

    def __iter__(self):
        """Return self, because generators are Iterable."""
        logging.debug("Creating fasta iterator.")
        return self

    def __next__(self) -> Query:
        """Get the next Query."""
        if self.nowStop:
            raise StopIteration()
        sequence = ""
        prevSequenceID = self.nextSequenceID
        while True:
            nextLine = self.fastaFile.readline()
            if len(nextLine) == 0:
                self.nowStop = True
                break
            if nextLine[0] == ">":
                self.nextSequenceID = nextLine[1:].strip()
                break
            sequence += nextLine.strip()
        oneHotSequence = utils.oneHotEncode(sequence)
        q = Query(oneHotSequence, prevSequenceID, self.index)
        self.index += 1
        return q


class FlatBedGenerator(Generator):
    """Reads in lines from a bed file and fetches the genomic sequence around them.

    Note that the regions should have width outputLength, and they will be automatically
    padded to the appropriate input length.

    :param bedFname: The bed file to read.
    :param genomeFname: The genome fasta that sequences will be drawn from.
    :param inputLength: The input length of your model.
    :param outputLength: The output length of your model.
    """

    def __init__(self, bedFname: str, genomeFname: str, inputLength: int, outputLength: int):
        logging.info("Creating bed generator.")
        self.bedFname = bedFname
        self.genomeFname = genomeFname
        self.inputLength = inputLength
        self.outputLength = outputLength
        numRegions = 0
        # Note that this opens a file during construction (a no-no for my threading model)
        # but it just reads it and closes it right back up. When the initializer finishes,
        # there are no pointers to file handles in this object.
        fp = pybedtools.BedTool(self.bedFname)
        for _ in fp:
            numRegions += 1
        self.numRegions = numRegions
        logging.info("Bed generator initialized with {0:d} regions".format(self.numRegions))

    def construct(self):
        """Open the bed file and fasta genome."""
        # We create a list of all the regions now, but we'll look up the sequences
        # and stuff on the fly.
        logging.info("Constructing bed generator it its thread.")
        self.shapTargets = []
        self.genome = pysam.FastaFile(self.genomeFname)
        self.readHead = 0
        fp = pybedtools.BedTool(self.bedFname)
        for line in fp:
            self.shapTargets.append(line)

    def __iter__(self):
        logging.debug("Creating iterator for bed generator.")
        return self

    def __next__(self) -> Query:
        """Get the next sequence and make a Query with it."""
        if self.readHead >= len(self.shapTargets):
            raise StopIteration()
        r = self.shapTargets[self.readHead]
        padding = (self.inputLength - self.outputLength) // 2
        startPos = r.start - padding
        endPos = startPos + self.inputLength
        seq = self.genome.fetch(r.chrom, startPos, endPos)
        oneHot = utils.oneHotEncode(seq)
        ret = Query(oneHot, (r.chrom, startPos, endPos), self.readHead)
        self.readHead += 1
        return ret

    def done(self):
        """Close the fasta file."""
        logging.debug("Closing bed generator, read {0:d} entries".format(self.readHead))
        self.genome.close()


class PisaBedGenerator(Generator):
    """Reads in lines from a bed file and fetches the genomic sequence at every base.

    This is very different than the :py:class:`~FlatBedGenerator`, which generates
    one sequence query per bed file entry. This class generates a query for every
    base that the bed file contains.

    :param bedFname: The bed file to read.
    :param genomeFname: The genome fasta that sequences will be drawn from.
    :param inputLength: The input length of your model.
    :param outputLength: The output length of your model.
    """

    def __init__(self, bedFname: str, genomeFname: str, inputLength: int, outputLength: int):
        logging.info("Creating bed generator.")
        self.bedFname = bedFname
        self.genomeFname = genomeFname
        self.inputLength = inputLength
        self.outputLength = outputLength
        numRegions = 0
        # Note that this opens a file during construction (a no-no for my threading model)
        # but it just reads it and closes it right back up. When the initializer finishes,
        # there are no pointers to file handles in this object.
        fp = pybedtools.BedTool(self.bedFname)
        for line in fp:
            numRegions += line.end - line.start
        self.numRegions = numRegions
        logging.info("Bed generator initialized with {0:d} regions".format(self.numRegions))

    def construct(self):
        """Run in the child thread, opens up the files and reads the bed."""
        # We create a list of all the regions now, but we'll look up the sequences
        # and stuff on the fly.
        logging.debug("Constructing bed generator it its thread.")
        self.shapTargets = []
        self.genome = pysam.FastaFile(self.genomeFname)
        self.readHead = 0
        fp = pybedtools.BedTool(self.bedFname)
        for line in fp:
            for pos in range(line.start, line.end):
                self.shapTargets.append((line.chrom, pos))

    def __iter__(self):
        logging.debug("Creating iterator for bed generator.")
        return self

    def __next__(self) -> Query:
        """Get the next sequence and make a Query with it."""
        if self.readHead >= len(self.shapTargets):
            raise StopIteration()
        curChrom, curStart = self.shapTargets[self.readHead]
        padding = (self.inputLength - self.outputLength) // 2
        startPos = curStart - padding
        stopPos = startPos + self.inputLength
        seq = self.genome.fetch(curChrom, startPos, stopPos)
        oneHot = utils.oneHotEncode(seq)
        ret = Query(oneHot, (curChrom, curStart), self.readHead)
        self.readHead += 1
        return ret

    def done(self):
        """Close the fasta."""
        logging.debug("Closing bed generator, read {0:d} entries".format(self.readHead))
        self.genome.close()


def _flatBatcherThread(modelName: str, batchSize: int, inQueue: multiprocessing.Queue,
                       outQueue: multiprocessing.Queue, headID: int, numHeads: int,
                       taskIDs: list[int], numShuffles: int, mode: str, kmerSize: int,
                       profileFname: Optional[str] = None):
    """The thread that spins up the batcher."""
    logging.debug("Starting flat batcher thread.")
    import cProfile
    profiler = cProfile.Profile()
    if profileFname is not None:
        profiler.enable()
    b = _FlatBatcher(modelName, batchSize, outQueue, headID,
                     numHeads, taskIDs, numShuffles, mode, kmerSize)
    logging.debug("Batcher created.")
    while True:
        query = inQueue.get(timeout=utils.QUEUE_TIMEOUT)
        if query is None:
            break
        b.addSample(query)
    logging.debug("Last query received. Finishing batcher thread.")
    b.finishBatch()
    outQueue.put(None, timeout=utils.QUEUE_TIMEOUT)
    outQueue.close()
    if profileFname is not None:
        profiler.create_stats()
        profiler.dump_stats(profileFname)
    logging.debug("Batcher thread finished.")


def _pisaBatcherThread(modelName: str, batchSize: int, inQueue: multiprocessing.Queue,
                       outQueue: multiprocessing.Queue, headID: int, taskID: int,
                       numShuffles: int, receptiveField: int, kmerSize: int,
                       profileFname: Optional[str] = None):
    """The thread that spins up the batcher."""
    logging.debug("Starting batcher thread.")
    import cProfile
    profiler = cProfile.Profile()
    if profileFname is not None:
        profiler.enable()
    b = _PisaBatcher(modelName, batchSize, outQueue, headID, taskID, numShuffles,
                     receptiveField, kmerSize)
    logging.debug("Batcher created.")
    while True:
        query = inQueue.get(timeout=utils.QUEUE_TIMEOUT)
        if query is None:
            break
        b.addSample(query)
    logging.debug("Last query received. Finishing batcher thread.")
    b.finishBatch()
    outQueue.put(None, timeout=utils.QUEUE_TIMEOUT)
    outQueue.close()
    if profileFname is not None:
        profiler.create_stats()
        profiler.dump_stats(profileFname)
    logging.debug("Batcher thread finished.")


def _generatorThread(inQueues: list[multiprocessing.Queue], generator: Generator,
                     profileFname: Optional[str] = None):
    """The thread that spins up the generator and emits queries."""
    import cProfile
    profiler = cProfile.Profile()
    if profileFname is not None:
        profiler.enable()
    logging.debug("Starting generator thread.")
    generator.construct()
    for elem in generator:
        for inQueue in inQueues:
            inQueue.put(elem, timeout=utils.QUEUE_TIMEOUT)
    for inQueue in inQueues:
        inQueue.put(None, timeout=utils.QUEUE_TIMEOUT)
        inQueue.close()
    logging.debug("Done with generator, None added to queue.")
    generator.done()
    logging.debug("Generator thread finished.")
    if profileFname is not None:
        profiler.create_stats()
        profiler.dump_stats(profileFname)


def _saverThread(outQueue: multiprocessing.Queue, saver: Saver,
                 profileFname: Optional[str] = None):
    """The thread that spins up the saver."""
    logging.debug("Saver thread started.")
    import cProfile
    profiler = cProfile.Profile()
    if profileFname is not None:
        profiler.enable()
    saver.construct()
    while True:
        rv = outQueue.get(timeout=utils.QUEUE_TIMEOUT)
        if rv is None:
            break
        saver.add(rv)
    saver.done()
    logging.debug("Saver thread finished.")
    if profileFname is not None:
        profiler.create_stats()
        profiler.dump_stats(profileFname)


class _PisaBatcher:
    """The workhorse of this stack.

    Accepts queries until its internal storage is full,
    then predicts them all at once, and runs shap.

    :param modelFname: The name of the keras model to interpret.
    :param batchSize: How many sequences should be interpreted at once?
    :param outQueue: The batcher will put its :py:class:`~Result` objects here.
    :param headID: The head that is being interpreted.
    :param taskID: The task within that head that is being interpreted.
    :param numShuffles: How many shuffled samples should be run?
    :param receptiveField: What is the receptive field of the model?
    :param kmerSize: What length of kmer should have its distribution preserved in the shuffles?

    """

    def __init__(self, modelFname: str, batchSize: int, outQueue: multiprocessing.Queue,
                 headID: int, taskID: int, numShuffles: int, receptiveField: int,
                 kmerSize: int):
        logging.info("Initializing batcher.")
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        from bpreveal import shap
        utils.setMemoryGrowth()
        self.model = utils.loadModel(modelFname)
        self.batchSize = batchSize
        self.outQueue = outQueue
        self.headID = headID
        self.taskID = taskID
        self.curBatch = []
        self.numShuffles = numShuffles
        self.receptiveField = receptiveField
        self.kmerSize = kmerSize
        logging.debug("Batcher prepared, creating explainer.")
        # This slice....
        # Oh, this slice is a doozy!
        #                      Keep the first dimension, so it looks like a batch of size one--,
        #                                          Sum the samples in this batch-,             |
        #                                       The current task ----,           |             |
        #                          The leftmost predicted base--,    |           |             |
        #                           All samples in the batch-,  |    |           |             |
        #                                            Current |  |    |           |             |
        #                                             head   V  V    V           |             |
        outTarget = tf.reduce_sum(self.model.outputs[headID][:, 0, taskID], axis=0, keepdims=True)
        self.profileExplainer = shap.TFDeepExplainer(
            (self.model.input, outTarget),
            self.generateShuffles)
        logging.info("Batcher initialized, Explainer initialized. Ready for Queries to explain.")

    def generateShuffles(self, modelInputs):
        """Callback for shap."""
        if self.kmerSize == 1:
            rng = np.random.default_rng(seed=355687)
            shuffles = [rng.permutation(modelInputs[0], axis=0) for _ in range(self.numShuffles)]
            shuffles = np.array(shuffles)
            return [shuffles]
        else:
            shuffles = ushuffle.shuffleOHE(modelInputs[0], self.kmerSize, self.numShuffles,
                                           seed=355687)
            return [shuffles]

    def addSample(self, query: Query):
        """Add a query to the batch.

        Runs the batch if it has enough work accumulated.
        """
        self.curBatch.append(query)
        if len(self.curBatch) >= self.batchSize:
            self.finishBatch()

    def finishBatch(self):
        """If there's any work waiting to be done, do it."""
        if len(self.curBatch) > 0:
            self.runPrediction()
            self.curBatch = []

    def runPrediction(self):
        """Actually run the batch."""
        # Now for the meat of all this boilerplate. Take the query sequences and
        # run them through the model, then run the shuffles, then run shap.
        # Finally, put all the results in the output queue.
        # This needs more optimization, but for this initial pass, I'm not actually batching -
        # instead, I'm running all the samples individually. But I can change this later,
        # and that's what counts.
        # First, build up an array of sequences to test.
        numQueries = len(self.curBatch)
        inputLength = self.curBatch[0].sequence.shape[0]
        oneHotBuf = np.empty((numQueries * (self.numShuffles + 1), inputLength, 4),
                             dtype=MODEL_ONEHOT_T)
        # To predict on as large a batch as possible, I put the actual sequences and all the
        # references for the current batch into this array. The first <nsamples> rows are the real
        # sequence, the next <numShuffles> rows are the shuffles of the first sequence, then
        # the next is the shuffles of the second sequence, like this:
        # REAL_SEQUENCE_1_REAL_SEQUENCE_1_REAL_SEQUENCE_1
        # REAL_SEQUENCE_2_REAL_SEQUENCE_2_REAL_SEQUENCE_2
        # SEQUENCE_1_FIRST_SHUFFLE_SEQUENCE_1_FIRST_SHUFF
        # SEQUENCE_1_SECOND_SHUFFLE_SEQUENCE_1_SECOND_SHU
        # SEQUENCE_1_THIRD_SHUFFLE_SEQUENCE_1_THIRD_SHUFF
        # SEQUENCE_2_FIRST_SHUFFLE_SEQUENCE_2_FIRST_SHUFF
        # SEQUENCE_2_SECOND_SHUFFLE_SEQUENCE_2_SECOND_SHU
        # SEQUENCE_2_THIRD_SHUFFLE_SEQUENCE_2_THIRD_SHUFF

        # Points to the index into oneHotBuf where the next data should be added.
        shuffleInsertHead = numQueries
        # These are the (real) sequences that will be passed to the explainer.
        # Note that it's a list of arrays, and each array has shape (1,inputLength,4)

        for i, q in enumerate(self.curBatch):
            oneHotBuf[i, :, :] = q.sequence
            shuffles = self.generateShuffles([q.sequence])[0]
            oneHotBuf[shuffleInsertHead:shuffleInsertHead + self.numShuffles, :, :] = shuffles
            shuffleInsertHead += self.numShuffles
        # Okay, now the data structures are set up.
        fullPred = self.model.predict(oneHotBuf)
        outBasePreds = fullPred[self.headID][:, 0, self.taskID]
        # (We'll deconvolve that in a minute...)
        shapScores = self.profileExplainer.shap_values([oneHotBuf[:numQueries, :, :]])
        # And now we need to run over that batch again to write the output.
        shuffleReadHead = numQueries
        for i, q in enumerate(self.curBatch):
            querySequence = oneHotBuf[i, :self.receptiveField, :]
            queryPred = outBasePreds[i]
            queryShufPreds = outBasePreds[shuffleReadHead:shuffleReadHead + self.numShuffles]
            shuffleReadHead += self.numShuffles
            queryShapScores = shapScores[i, 0:self.receptiveField, :]  # type: ignore
            ret = PisaResult(queryPred, queryShufPreds, querySequence,  # type: ignore
                             queryShapScores, q.passData, q.index)
            self.outQueue.put(ret, timeout=utils.QUEUE_TIMEOUT)


class _FlatBatcher:
    """The workhorse of this stack.

    Accepts queries until its internal storage is full,
    then predicts them all at once, and runs shap.

    :param modelFname: The name of the keras model to interpret.
    :param batchSize: How many sequences should be interpreted at once?
    :param outQueue: The batcher will put its :py:class:`~Result` objects here.
    :param headID: The head that is being interpreted.
    :param numHeads: How many heads does this model have, in total?
    :param taskIDs: Within the given head, what tasks should be considered?
    :param numShuffles: How many shuffled samples should be run?
    :param mode: What type of importance score is being generated? Options are
        ``profile`` or ``counts``.
    :param receptiveField: What is the receptive field of the model?
    :param kmerSize: What length of kmer should have its distribution preserved in the shuffles?
    """

    def __init__(self, modelFname: str, batchSize: int, outQueue: multiprocessing.Queue,
                 headID: int, numHeads: int, taskIDs: list[int], numShuffles: int, mode: str,
                 kmerSize: int):
        logging.info("Initializing batcher.")
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        from bpreveal import shap
        utils.limitMemoryUsage(0.4, 1024)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = utils.loadModel(modelFname)
        self.batchSize = batchSize
        self.outQueue = outQueue
        self.kmerSize = kmerSize
        self.headID = headID
        self.taskIDs = taskIDs
        self.curBatch = []
        self.numShuffles = numShuffles
        logging.debug("Batcher prepared, creating explainer.")
        match mode:
            case "profile":
                # Calculate the weighted meannormed logits that are used for the
                # profile explanation.
                profileOutput = self.model.outputs[headID]
                stackedLogits = tf.stack([profileOutput[:, :, x] for x in taskIDs], axis=2)
                inputShape = stackedLogits.shape
                numSamples = inputShape[1] * inputShape[2]
                logits = tf.reshape(stackedLogits, [-1, numSamples])
                meannormedLogits = logits - tf.reduce_mean(logits, axis=1)[:, None]
                stopgradMeannormedLogits = tf.stop_gradient(meannormedLogits)
                softmaxOut = tf.nn.softmax(stopgradMeannormedLogits, axis=1)
                weightedSum = tf.reduce_sum(softmaxOut * meannormedLogits, axis=1)
                self.explainer = shap.TFDeepExplainer(
                    (self.model.input, weightedSum),
                    self.generateShuffles,
                    combine_mult_and_diffref=combineMultAndDiffref)
            case "counts":
                # Now for counts - much easier!
                countsMetric = self.model.outputs[numHeads + headID][:, 0]
                self.explainer = shap.TFDeepExplainer((self.model.input, countsMetric),
                    self.generateShuffles,
                    combine_mult_and_diffref=combineMultAndDiffref)

        logging.info("Batcher initialized, Explainer initialized. Ready for Queries to explain.")

    def generateShuffles(self, modelInputs):
        """Callback for shap."""
        if self.kmerSize == 1:
            rng = np.random.default_rng(seed=355687)
            shuffles = [rng.permutation(modelInputs[0], axis=0) for _ in range(self.numShuffles)]
            shuffles = np.array(shuffles)
            return [shuffles]
        else:
            shuffles = ushuffle.shuffleOHE(modelInputs[0], self.kmerSize, self.numShuffles,
                                           seed=355687)
            return [shuffles]

    def addSample(self, query: Query):
        """Append the sample to the work queue."""
        self.curBatch.append(query)
        if len(self.curBatch) >= self.batchSize:
            self.finishBatch()

    def finishBatch(self):
        """If there's any work to do, finish it."""
        if len(self.curBatch) > 0:
            self.runPrediction()
            self.curBatch = []

    def runPrediction(self):
        """Actually run the calculation."""
        # Now for the meat of all this boilerplate. Take the query sequences and
        # run them through the model, then run the shuffles, then run shap.
        # Finally, put all the results in the output queue.
        # This needs more optimization, but for this initial pass, I'm not actually batching -
        # instead, I'm running all the samples individually. But I can change this later,
        # and that's what counts.
        # First, build up an array of sequences to test.
        numQueries = len(self.curBatch)
        inputLength = self.curBatch[0].sequence.shape[0]
        oneHotBuf = np.empty((numQueries, inputLength, 4), dtype=MODEL_ONEHOT_T)
        # To predict on as large a batch as possible, I put all of the sequences
        # to explain in this array, like this:
        # REAL_SEQUENCE_1_REAL_SEQUENCE_1_REAL_SEQUENCE_1
        # REAL_SEQUENCE_2_REAL_SEQUENCE_2_REAL_SEQUENCE_2

        for i, q in enumerate(self.curBatch):
            oneHotBuf[i, :, :] = q.sequence
        # Okay, now the data structures are set up.
        # (We'll deconvolve that in a minute...)
        scores = self.explainer.shap_values([oneHotBuf])
        # And now we need to run over that batch again to write the output.
        for i, q in enumerate(self.curBatch):
            querySequence = oneHotBuf[i, :, :]
            queryScores = scores[i, :, :]  # type: ignore
            ret = FlatResult(querySequence, queryScores, q.passData, q.index)  # type: ignore
            self.outQueue.put(ret, timeout=utils.QUEUE_TIMEOUT)


def combineMultAndDiffref(mult, orig_inp, bg_data):
    """Combine the shap multipliers and difference from reference to generate hypothetical scores.

    This is injected deep into shap and generates the hypothetical importance scores.
    """
    # This is copied from Zahoor's code.
    projectedHypotheticalContribs = \
        np.zeros_like(bg_data[0]).astype('float')
    assert len(orig_inp[0].shape) == 2
    for i in range(4):  # We're going to go over all the base possibilities.
        hypotheticalInput = np.zeros_like(orig_inp[0]).astype('float')
        hypotheticalInput[:, i] = 1.0
        hypotheticalDiffref = hypotheticalInput[None, :, :] - bg_data[0]
        hypotheticalContribs = hypotheticalDiffref * mult[0]
        projectedHypotheticalContribs[:, :, i] = np.sum(hypotheticalContribs, axis=-1)
    # There are no bias importances, so the np.zeros_like(orig_inp[1]) is not needed.
    return [np.mean(projectedHypotheticalContribs, axis=0)]
