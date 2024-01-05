#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '4'
import utils
import pysam
import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Iterator
import tqdm
import h5py
import logging
import pybedtools
import multiprocessing
import abc

from utils import ONEHOT_T, ONEHOT_AR_T, IMPORTANCE_T, H5_CHUNK_SIZE, MODEL_ONEHOT_T
# Use a low-precision floating point format for importance scores.


class Query:
    """This is what is passed to the batcher. It has two things.
    sequence is the (INPUT_LENGTH, 4) one-hot encoded sequence of the current base.
    passData, as with Result objects, is either a tuple of (chromName, position)
    (for when you have a bed file) or a string with a fasta description line
    (for when you're starting with a fasta).
    index, an integer, indicates which output slot this data should be put in.
    Since there's no guarantee that the results will arrive in order, we have to
    track which query was which.
    """
    sequence: ONEHOT_AR_T
    passData: Any
    index: int

    def __init__(self, oneHotSequence: ONEHOT_AR_T, passData: Any, index: int):
        self.sequence = oneHotSequence
        self.passData = passData
        self.index = index


class Result:
    pass


class PisaResult(Result):
    """This is the output from shapping a single base. It contains a few things.
    inputPrediction is a scalar floating point value, of the predicted logit
    from the input sequence at the base that was being shapped.

    shufflePredictions is a (numShuffles,) numpy array of the logits
    returned by running predictions on the reference sequence, again evaluated
    at the position of the base that was being shapped.

    sequence is a (RECEPTIVE_FIELD, 4) numpy array of the one-hot encoded input sequence.

    shap is a (RECEPTIVE_FIELD, 4) numpy array of shap scores.

    passData is data that is not touched by the batcher, but added by the generator
    and necessary for creating the output file.
    If the generator is reading from a bed file, then it is a tuple of (chromName, position)
    and that data should be used to populate the coords_chrom and coords_base fields.
    If the generator was using a fasta file, it is the title line from the original fasta,
    with its leading '>' removed.

    index, an int, indicates which address the data should be stored at in the output hdf5.
    Since there's no order guarantee when you're receiving data, we have to keep track
    of the order in the original input.
    """

    def __init__(self, inputPrediction: npt.NDArray[np.float32],
                 shufflePredictions: npt.NDArray[np.float32],
                 sequence: ONEHOT_AR_T,
                 shap: npt.NDArray[IMPORTANCE_T],
                 passData: Any,
                 index: int):
        self.inputPrediction = inputPrediction
        self.shufflePredictions = shufflePredictions
        self.sequence = sequence
        self.shap = shap
        self.passData = passData
        self.index = index


class FlatResult(Result):
    """A Result object that is given to savers for flat interpretation analysis.
    sequence is a one-hot encoded array of the sequence that was explained, of shape
    (input-length, 4)
    shap is an array of shape (input-length, 4), containing the shap scores.

    passData is a (picklable) object that is passed through from the generator.
        For bed-based interpretations, it will be a three-tuple of (chrom, start, end)
        (The start and end positions correspond to the INPUT to the model, so they are inflated
        with respect to the bed file.)
        For fasta-based interpretations it will be a string.

    index, an int, gives the position in the output hdf5 where the scores should be saved.
    """
    def __init__(self, sequence: ONEHOT_AR_T,
                 shap: npt.NDArray[IMPORTANCE_T],
                 passData: Any, index: int):
        self.sequence = sequence
        self.shap = shap
        self.passData = passData
        self.index = index


class Generator:
    """This is the base class for generating pisa samples."""
    def __init__(self):
        raise NotImplemented()

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Query]:
        raise NotImplemented()

    def __next__(self):
        raise NotImplemented()

    def construct(self):
        # When the generator is about to start, this method is called once before
        # requesting the iterator.
        raise NotImplemented()

    def done(self):
        # When the batcher is done, this is called to free any allocated resources.
        raise NotImplemented()


class Saver:
    """The saver runs in the main thread, so it's safe to open files and stuff in __init__.
    Descendants of this class shall receive Results from the batcher and save them to an
    appropriate structure, usually on disk.
    The user creates a Saver object in the main thread, and then the saver gets
    construct()ed inside a separate thread created by the runners. Therefore, you
    should only create members that are serializable in __init__."""
    def __init__(self):
        raise NotImplemented()

    def construct(self):
        """This function should actually open the output files, since it will
        be called from inside the actual saver thread."""
        raise NotImplemented()

    def add(self, result: Result):
        """Add the given Result to wherever you're saving them out.
        Note that this will not be called with None at the end of the run,
        that is special-cased in the code that runs your saver."""
        raise NotImplemented()

    def done(self):
        """Called when the batcher is complete (indicated by putting a None in its output queue).
        Now is the time to close all of your files."""
        raise NotImplemented()


class FlatRunner:
    """I try to avoid class-based wrappers around simple things, but this is not simple.
    This class creates threads to read in data, and then creates a thread that runs interpretation
    samples in batches. Finally, it takes the results from the interpretation thread and saves
    those out to an hdf5-format file."""

    def __init__(self, modelFname: str, headId: int, numHeads: int, taskIDs: list[int],
                 batchSize: int, generator: Generator, profileSaver: Saver,
                 countsSaver: Saver, numShuffles: int, profileFname: Optional[str] = None):
        """modelFname is the name of the model on disk
        headId and taskIDs are the head and tasks to shap, obviously.
            Typically, you'd want to interpret all of the tasks in a head, so for a two-task
            head, taskIDs would be [0,1].
        batchSize is the *shap* batch size, which should be your usual batch size divided
            by the number of shuffles. (since the original sequence and shuffles are run together)
        generator is a Generator object that will be passed to _generatorThread
        saver is a Saver object that will be used to save out the data.
        numShuffles is the number of reference sequences that should be generated for each
            shap evaluation. (I recommend 20 or so)
        profileFname is an optional parameter. If provided, profiling data (i.e., performance
            of this code) will be written to the given file name. Note that this has
            nothing to do with genomic profiles, it's just benchmarking data for this code.
        """
        logging.info("Initializing interpretation runner.")
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
                  ap("_gen.dat")))

        self._profileBatchThread = multiprocessing.Process(target=_flatBatcherThread,
            args=(modelFname, batchSize, self._profileInQueue, self._profileOutQueue,
                  headId, numHeads, taskIDs, numShuffles, "profile",
                  ap("_profile.dat")))
        self._countsBatchThread = multiprocessing.Process(target=_flatBatcherThread,
            args=(modelFname, batchSize, self._countsInQueue, self._countsOutQueue,
                  headId, numHeads, taskIDs, numShuffles, "counts",
                  ap("_counts.dat")))

        self._profileSaverThread = multiprocessing.Process(target=_saverThread,
            args=(self._profileOutQueue, profileSaver, ap("_profileWrite.dat")))
        self._countsSaverThread = multiprocessing.Process(target=_saverThread,
            args=(self._countsOutQueue, countsSaver, ap("_countsWrite.dat")))
        if profileFname is not None:
            import pstats
            for end in ["_gen", "_profile", "_counts", "_profileWrite", "_countsWrite"]:
                with open(ap(end + ".txt"), "w") as fp:  # type: ignore
                    s = pstats.Stats(ap(end + ".dat"), stream=fp)
                    s.sort_stats('cumulative')
                    s.print_stats()

    def run(self):
        """Starts up the threads and waits for them to finish."""
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
        logging.info("Savers complete. Done.")


class PisaRunner:
    """I try to avoid class-based wrappers around simple things, but this is not simple.
    This class creates threads to read in data, and then creates a thread that runs pisa
    samples in batches. Finally, it takes the results from the pisa thread and saves
    those out to an hdf5-format file."""

    def __init__(self, modelFname: str, headId: int, taskId: int, batchSize: int,
                 generator: Generator, saver: Saver, numShuffles: int,
                 receptiveField: int, profileFname: Optional[str] = None):
        """modelFname is the name of the model on disk
        headId and taskID are the head and task to shap, obviously,
        batchSize is the *shap* batch size, which should be your usual batch size divided
            by the number of shuffles. (since the original sequence and shuffles are run together)
        generator is a Generator object that will be passed to _generatorThread
        saver is a Saver object that will be used to save out the data.
        numShuffles is the number of reference sequences that should be generated for each
            shap evaluation. (I recommend 20 or so)
        receptiveField is the receptive field of the model. To save on writing a lot of
            zeroes, the result objects only contain bases that are in the receptive field
            of the base being shapped.
        profileFname is an optional parameter. If provided, profiling data (i.e., performance
            of this code) will be written to the given file name. Note that this has
            nothing to do with genomic profiles, it's just benchmarking data for this code.
        """
        logging.info("Initializing pisa runner.")
        self._inQueue = multiprocessing.Queue(1000)
        self._outQueue = multiprocessing.Queue(1000)

        def ap(txt):
            """Add some text to the profileFname"""
            if profileFname is None:
                return None
            return profileFname + txt

        self._genThread = multiprocessing.Process(target=_generatorThread,
            args=([self._inQueue], generator, ap("_generate.dat")))
        self._batchThread = multiprocessing.Process(target=_pisaBatcherThread,
                                                    args=(modelFname, batchSize,
                                                          self._inQueue, self._outQueue, headId,
                                                          taskId, numShuffles, receptiveField,
                                                          ap("batcher.dat")))
        self._saver = saver

    def run(self):
        """Starts up the threads and waits for them to finish."""
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
        logging.info("Done.")


class FlatH5Saver(Saver):
    """Saves the shap scores to the output file. """
    CHUNK_SHAPE: tuple[int, int, int]
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
        """
        OutputFname is the name of the hdf5-format file that the shap scores will be deposited in.
        numSamples is the number of regions (i.e., bases) that pisa will be run on.
        This is needed because we store reference predictions.
        genome, an optional parameter, gives the name of a fasta-format file that contains
            the genome of the organism. If provided, then chromosome name and size information
            will be included in the output, and, additionally, two other datasets will be
            created: coords_chrom, and coords_base. """
        self.CHUNK_SHAPE = (min(H5_CHUNK_SIZE, numSamples), inputLength, 4)
        self._outputFname = outputFname
        self.numSamples = numSamples
        self.genomeFname = genome
        self.inputLength = inputLength
        self._useTqdm = useTqdm

    def construct(self):
        logging.info("Initializing saver.")
        self._outFile = h5py.File(self._outputFname, "w")
        self._chunksToWrite = dict()

        self._outFile.create_dataset("hyp_scores",
                                     (self.numSamples, self.inputLength, 4),
                                     dtype=IMPORTANCE_T, chunks=self.CHUNK_SHAPE,
                                     compression='gzip')
        self._outFile.create_dataset('input_seqs',
                                     (self.numSamples, self.inputLength, 4),
                                     dtype=ONEHOT_T, chunks=self.CHUNK_SHAPE,
                                     compression='gzip')
        if (self.genomeFname is not None):
            self._loadGenome()
        else:
            self._outFile.create_dataset("descriptions", (self.numSamples,),
                                         dtype=h5py.string_dtype('utf-8'))
        logging.debug("Saver initialized, hdf5 file created.")

    def _loadGenome(self):
        """Does a few things.
        First, it creates chrom_names and chrom_sizes datasets in the output hdf5.
            These two datasets just contain the (string) names and (unsigned) sizes for each one.
        Second, it populates these datasets.
        Third, it creates two datasets: coords_chrom and coords_base, which store, for every
            shap value calculated, what chromosome it was on, and where on that chromosome
            it was. These fields are populated during data receipt, because we don't have
            an ordering guarantee when we get data from the batcher.
        """
        import pysam
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
                if (genome.get_reference_length(chromName) > 2**31):
                    posDtype = 'u8'

            self._outFile.create_dataset("chrom_sizes", (genome.nreferences, ), dtype=posDtype)

            for i, chromName in enumerate(genome.references):
                self._outFile['chrom_names'][i] = chromName
                self.chromNameToIdx[chromName] = i
                self._outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

            if (genome.nreferences <= 255):  # type: ignore
                # If there are less than 255 chromosomes,
                # I only need 8 bits to store the chromosome ID.
                chromDtype = 'u1'
            elif (genome.nreferences <= 65535):  # type: ignore
                # If there are less than 2^16 chromosomes, I can use 16 bits.
                chromDtype = 'u2'
            else:
                # If you have over four billion chromosomes, you deserve to have code
                # that will break. So I just assume that 32 bits will be enough.
                chromDtype = 'u4'
            self._outFile.create_dataset("coords_chrom", (self.numSamples, ), dtype=chromDtype)

            #self._outFile.create_dataset("coords_chrom", (self.numSamples, ),
            #                             dtype=h5py.string_dtype(encoding='utf-8'))
            self._outFile.create_dataset("coords_start", (self.numSamples, ), dtype=posDtype)
            self._outFile.create_dataset("coords_end", (self.numSamples, ), dtype=posDtype)
        logging.debug("Genome data loaded.")

    def done(self):
        logging.debug("Saver closing.")
        if self.pbar is not None:
            self.pbar.close()
        self._outFile.close()

    def add(self, result: FlatResult):
        if self.pbar is None and self._useTqdm:
            # Initialize the progress bar.
            # I would do it earlier, but starting it before the batchers
            # have warmed up skews the stats.
            self.pbar = tqdm.tqdm(total=self.numSamples)
        if self.pbar is not None:
            self.pbar.update()
        index = result.index
        # Which chunk does this result go in?
        chunkIdx = index // self.CHUNK_SHAPE[0]
        # And where in the chunk does it go?
        chunkOffset = index % self.CHUNK_SHAPE[0]
        if chunkIdx not in self._chunksToWrite:
            # Allocate a new chunk.
            # Note that we can't just statically allocate one chunk buffer because we don't
            # guarantee the order that the results will arrive in.
            numToAdd = self.CHUNK_SHAPE[0]
            if chunkIdx == self.numSamples // self.CHUNK_SHAPE[0]:
                numToAdd = self.numSamples % self.CHUNK_SHAPE[0]
            curChunkShape = (numToAdd, self.CHUNK_SHAPE[1], self.CHUNK_SHAPE[2])
            self._chunksToWrite[chunkIdx] = {
                "hyp_scores": np.empty(curChunkShape, dtype=IMPORTANCE_T),
                "input_seqs": np.empty(curChunkShape, dtype=ONEHOT_T),
                "numToAdd": numToAdd,
                "writeStart": chunkIdx * self.CHUNK_SHAPE[0],
                "writeEnd": chunkIdx * self.CHUNK_SHAPE[0] + numToAdd}
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
        if (self.genomeFname is not None):
            self._outFile["coords_chrom"][index] = self.chromNameToIdx[result.passData[0]]
            self._outFile["coords_start"][index] = result.passData[1]
            self._outFile["coords_end"][index] = result.passData[2]
        else:
            self._outFile["descriptions"][index] = result.passData


class PisaH5Saver(Saver):
    """Saves the shap scores to the output file. """
    def __init__(self, outputFname: str, numSamples: int, numShuffles: int, receptiveField: int,
                 genome: Optional[str] = None, useTqdm: bool = False):
        """
        OutputFname is the name of the hdf5-format file that the shap scores will be deposited in.
        numSamples is the number of regions (i.e., bases) that pisa will be run on.
        numShuffles is the number of shuffles that are used to generate the reference.
        This is needed because we store reference predictions.
        genome, an optional parameter, gives the name of a fasta-format file that contains
            the genome of the organism. If provided, then chromosome name and size information
            will be included in the output, and, additionally, two other datasets will be
            created: coords_chrom, and coords_base. """
        logging.info("Initializing saver.")
        self._outputFname = outputFname
        self.numSamples = numSamples
        self.numShuffles = numShuffles
        self.genomeFname = genome
        self._useTqdm = useTqdm
        self.receptiveField = receptiveField
        self.CHUNK_SHAPE = (min(H5_CHUNK_SIZE, numSamples), receptiveField, 4)

    def construct(self):
        self._outFile = h5py.File(self._outputFname, "w")
        self._outFile.create_dataset("input_predictions",
                                     (self.numSamples,), dtype='f4')
        self._outFile.create_dataset("shuffle_predictions",
                                     (self.numSamples, self.numShuffles),
                                     dtype='f4')
        # self._outFile.create_dataset("shap",
        #                             (self.numSamples, self.receptiveField, 4),
        #                             dtype='f4')
        # TODO for later: Adding compression to the sequence here absolutely tanks performan
        # self._outFile.create_dataset('sequence',
        #                             (self.numSamples, self.receptiveField, 4),
        #                             dtype='u1')
        self._chunksToWrite = dict()

        self._outFile.create_dataset("shap",
                                     (self.numSamples, self.receptiveField, 4),
                                     dtype=IMPORTANCE_T, chunks=self.CHUNK_SHAPE,
                                     compression='gzip')
        self._outFile.create_dataset('sequence',
                                     (self.numSamples, self.receptiveField, 4),
                                     dtype=ONEHOT_T, chunks=self.CHUNK_SHAPE,
                                     compression='gzip')
        self.pbar = None
        if (self._useTqdm):
            self.pbar = tqdm.tqdm(total=self.numSamples)
        if (self.genomeFname is not None):
            self._loadGenome()
        else:
            self._outFile.create_dataset("descriptions", (self.numSamples,),
                                         dtype=h5py.string_dtype('utf-8'))
        logging.debug("Saver initialized, hdf5 file created.")

    def _loadGenome(self):
        """Does a few things.
        First, it creates chrom_names and chrom_sizes datasets in the output hdf5.
            These two datasets just contain the (string) names and (unsigned) sizes for each one.
        Second, it populates these datasets.
        Third, it creates two datasets: coords_chrom and coords_base, which store, for every
            shap value calculated, what chromosome it was on, and where on that chromosome
            it was. These fields are populated during data receipt, because we don't have
            an ordering guarantee when we get data from the batcher.
        """
        import pysam
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
                if (genome.get_reference_length(chromName) > 2**31):
                    posDtype = 'u8'

            self._outFile.create_dataset("chrom_sizes", (genome.nreferences, ), dtype=posDtype)

            for i, chromName in enumerate(genome.references):
                self._outFile['chrom_names'][i] = chromName
                self.chromNameToIdx[chromName] = i
                self._outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)

            if (genome.nreferences <= 255):  # type: ignore
                # If there are less than 255 chromosomes,
                # I only need 8 bits to store the chromosome ID.
                chromDtype = 'u1'
            elif (genome.nreferences <= 65535):  # type: ignore
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
        logging.debug("Saver closing.")
        if (self.pbar is not None):
            self.pbar.close()
        self._outFile.close()

    def add(self, result: PisaResult):
        if (self.pbar is not None):
            self.pbar.update()
        index = result.index
        # Which chunk does this result go in?
        chunkIdx = index // self.CHUNK_SHAPE[0]
        # And where in the chunk does it go?
        chunkOffset = index % self.CHUNK_SHAPE[0]
        if chunkIdx not in self._chunksToWrite:
            # Allocate a new chunk.
            # Note that we can't just statically allocate one chunk buffer because we don't
            # guarantee the order that the results will arrive in.
            numToAdd = self.CHUNK_SHAPE[0]
            if chunkIdx == self.numSamples // self.CHUNK_SHAPE[0]:
                numToAdd = self.numSamples % self.CHUNK_SHAPE[0]
            curChunkShape = (numToAdd, self.CHUNK_SHAPE[1], self.CHUNK_SHAPE[2])
            self._chunksToWrite[chunkIdx] = {
                "shap": np.empty(curChunkShape, dtype=IMPORTANCE_T),
                "sequence": np.empty(curChunkShape, dtype=ONEHOT_T),
                "numToAdd": numToAdd,
                "writeStart": chunkIdx * self.CHUNK_SHAPE[0],
                "writeEnd": chunkIdx * self.CHUNK_SHAPE[0] + numToAdd}

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
        if (self.genomeFname is not None):
            self._outFile["coords_chrom"][index] = self.chromNameToIdx[result.passData[0]]
            self._outFile["coords_base"][index] = result.passData[1]
        else:
            self._outFile["descriptions"][index] = result.passData


class FastaGenerator(Generator):
    def __init__(self, fastaFname: str):
        logging.info("Creating fasta generator.")
        self.fastaFname = fastaFname
        self.nowStop = False
        numRegions = 0

        with open(fastaFname, "r") as fp:
            for line in fp:
                if (len(line) > 0 and line[0] == ">"):
                    numRegions += 1
        # Iterate through the input file real quick and find out how many regions I'll have.
        self.numRegions = numRegions
        self.index = 0
        logging.info("Fasta generator initialized with {0:d} regions.".format(self.numRegions))

    def construct(self):
        logging.info("Constructing fasta generator in its thread.")
        self.fastaFile = open(self.fastaFname, "r")
        self.nextSequenceID = self.fastaFile.readline()[1:].strip()
        # [1:] to get rid of the '>'.
        logging.debug("Initial sequence to read: {0:s}".format(self.nextSequenceID))

    def done(self):
        logging.info("Closing fasta generator.")
        self.fastaFile.close()

    def __iter__(self):
        logging.debug("Creating fasta iterator.")
        return self

    def __next__(self) -> ONEHOT_AR_T:
        if (self.nowStop):
            raise StopIteration()
        sequence = ""
        prevSequenceID = self.nextSequenceID
        while (True):
            nextLine = self.fastaFile.readline()
            if (len(nextLine) == 0):
                self.nowStop = True
                break
            if (nextLine[0] == ">"):
                self.nextSequnceID = nextLine[1:].strip()
                break
            sequence += nextLine.strip()
        oneHotSequence = utils.oneHotEncode(sequence)
        q = Query(oneHotSequence, prevSequenceID, self.index)
        self.index += 1
        return q


class FlatBedGenerator(Generator):
    """Reads in lines from a bed file and fetches the genomic sequence around them.
    Note that the regions should have width outputLength, and they will be automatically
    padded to the appropriate input length."""
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

    def __next__(self) -> ONEHOT_AR_T:
        # Get the sequence and make a Query with it.
        if (self.readHead >= len(self.shapTargets)):
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
        logging.debug("Closing bed generator, read {0:d} entries".format(self.readHead))
        self.genome.close()


class PisaBedGenerator(Generator):

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

    def __next__(self) -> ONEHOT_AR_T:
        # Get the sequence and make a Query with it.
        if (self.readHead >= len(self.shapTargets)):
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
        logging.debug("Closing bed generator, read {0:d} entries".format(self.readHead))
        self.genome.close()


def _flatBatcherThread(modelName: str, batchSize: int, inQueue: multiprocessing.Queue,
                       outQueue: multiprocessing.Queue, headId: int, numHeads: int,
                       taskIDs: list[int], numShuffles: int, mode: str,
                       profileFname: Optional[str] = None):
    logging.debug("Starting flat batcher thread.")
    import cProfile
    profiler = cProfile.Profile()
    if (profileFname is not None):
        profiler.enable()
    b = _FlatBatcher(modelName, batchSize, outQueue, headId,
                     numHeads, taskIDs, numShuffles, mode)
    logging.debug("Batcher created.")
    while (True):
        query = inQueue.get()
        if (query is None):
            break
        b.addSample(query)
    logging.debug("Last query received. Finishing batcher thread.")
    b.finishBatch()
    outQueue.put(None)
    outQueue.close()
    if (profileFname is not None):
        profiler.create_stats()
        profiler.dump_stats(profileFname)
    logging.debug("Batcher thread finished.")


def _pisaBatcherThread(modelName: str, batchSize: int, inQueue: multiprocessing.Queue,
                       outQueue: multiprocessing.Queue, headId: int, taskId: int,
                       numShuffles: int, receptiveField: int,
                       profileFname: Optional[str] = None):
    logging.debug("Starting batcher thread.")
    import cProfile
    profiler = cProfile.Profile()
    if (profileFname is not None):
        profiler.enable()
    b = _PisaBatcher(modelName, batchSize, outQueue, headId, taskId, numShuffles, receptiveField)
    logging.debug("Batcher created.")
    while (True):
        query = inQueue.get()
        if (query is None):
            break
        b.addSample(query)
    logging.debug("Last query received. Finishing batcher thread.")
    b.finishBatch()
    outQueue.put(None)
    outQueue.close()
    if (profileFname is not None):
        profiler.create_stats()
        profiler.dump_stats(profileFname)
    logging.debug("Batcher thread finished.")


def _generatorThread(inQueues: list[multiprocessing.Queue], generator: Generator,
                     profileFname: Optional[str] = None):
    import cProfile
    profiler = cProfile.Profile()
    if (profileFname is not None):
        profiler.enable()
    logging.debug("Starting generator thread.")
    generator.construct()
    for elem in generator:
        for inQueue in inQueues:
            inQueue.put(elem)
    for inQueue in inQueues:
        inQueue.put(None)
        inQueue.close()
    logging.debug("Done with generator, None added to queue.")
    generator.done()
    logging.debug("Generator thread finished.")
    if (profileFname is not None):
        profiler.create_stats()
        profiler.dump_stats(profileFname)


def _saverThread(outQueue: multiprocessing.Queue, saver: Saver,
                 profileFname: Optional[str] = None):
    logging.debug("Saver thread started.")
    import cProfile
    profiler = cProfile.Profile()
    if (profileFname is not None):
        profiler.enable()
    saver.construct()
    while (True):
        rv = outQueue.get()
        if (rv is None):
            break
        saver.add(rv)
    saver.done()
    logging.debug("Saver thread finished.")
    if (profileFname is not None):
        profiler.create_stats()
        profiler.dump_stats(profileFname)


class _PisaBatcher:
    """The workhorse of this stack, it accepts queries until its internal storage is full,
    then predicts them all at once, and runs shap."""
    def __init__(self, modelFname: str, batchSize: int, outQueue: multiprocessing.Queue,
                 headId: int, taskId: int, numShuffles: int, receptiveField: int):
        logging.info("Initializing batcher.")
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        import shap
        from keras.models import load_model
        import losses
        import utils
        utils.setMemoryGrowth()
        self.model = load_model(modelFname,
                custom_objects={'multinomialNll': losses.multinomialNll,
                                'reweightableMse': losses.dummyMse})
        self.batchSize = batchSize
        self.outQueue = outQueue
        self.headId = headId
        self.taskId = taskId
        self.curBatch = []
        self.numShuffles = numShuffles
        self.receptiveField = receptiveField
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
        outTarget = tf.reduce_sum(self.model.outputs[headId][:, 0, taskId], axis=0, keepdims=True)
        self.profileExplainer = shap.TFDeepExplainer(
            (self.model.input, outTarget),
            self.generateShuffles)
        logging.info("Batcher initialized, Explainer initialized. Ready for Queries to explain.")

    def generateShuffles(self, model_inputs):
        rng = np.random.default_rng(seed=355687)
        shuffles = [rng.permutation(model_inputs[0], axis=0) for x in range(self.numShuffles)]
        shuffles = np.array(shuffles)
        return [shuffles]

    def addSample(self, query: Query):
        self.curBatch.append(query)
        if (len(self.curBatch) >= self.batchSize):
            self.finishBatch()

    def finishBatch(self):
        if (len(self.curBatch)):
            self.runPrediction()
            self.curBatch = []

    def runPrediction(self):
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
        outBasePreds = fullPred[self.headId][:, 0, self.taskId]
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
            ret = PisaResult(queryPred, queryShufPreds, querySequence,
                         queryShapScores, q.passData, q.index)
            self.outQueue.put(ret)


class _FlatBatcher:
    """The workhorse of this stack, it accepts queries until its internal storage is full,
    then predicts them all at once, and runs shap."""
    def __init__(self, modelFname: str, batchSize: int, outQueue: multiprocessing.Queue,
                 headId: int, numHeads: int, taskIDs: list[int], numShuffles: int, mode: str):
        logging.info("Initializing batcher.")
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        import shap
        from keras.models import load_model
        import losses
        import utils
        utils.limitMemoryUsage(0.4, 1024)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = load_model(modelFname,
                custom_objects={'multinomialNll': losses.multinomialNll,
                                'reweightableMse': losses.dummyMse})
        self.batchSize = batchSize
        self.outQueue = outQueue
        self.headId = headId
        self.taskIDs = taskIDs
        self.curBatch = []
        self.numShuffles = numShuffles
        logging.debug("Batcher prepared, creating explainer.")
        match mode:
            case "profile":
                # Calculate the weighted meannormed logits that are used for the
                # profile explanation.
                profileOutput = self.model.outputs[headId]
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
                countsMetric = self.model.outputs[numHeads + headId][:, 0]
                self.explainer = shap.TFDeepExplainer((self.model.input, countsMetric),
                    self.generateShuffles,
                    combine_mult_and_diffref=combineMultAndDiffref)

        logging.info("Batcher initialized, Explainer initialized. Ready for Queries to explain.")

    def generateShuffles(self, model_inputs):
        rng = np.random.default_rng(seed=355687)
        shuffles = [rng.permutation(model_inputs[0], axis=0) for x in range(self.numShuffles)]
        shuffles = np.array(shuffles)
        return [shuffles]

    def addSample(self, query: Query):
        self.curBatch.append(query)
        if (len(self.curBatch) >= self.batchSize):
            self.finishBatch()

    def finishBatch(self):
        if (len(self.curBatch)):
            self.runPrediction()
            self.curBatch = []

    def runPrediction(self):
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
            ret = FlatResult(querySequence, queryScores, q.passData, q.index)
            self.outQueue.put(ret)


def combineMultAndDiffref(mult, orig_inp, bg_data):
    # This is copied from Zahoor's code.
    projected_hypothetical_contribs = \
        np.zeros_like(bg_data[0]).astype('float')
    assert (len(orig_inp[0].shape) == 2)
    for i in range(4):  # We're going to go over all the base possibilities.
        hypothetical_input = np.zeros_like(orig_inp[0]).astype('float')
        hypothetical_input[:, i] = 1.0
        hypothetical_diffref = hypothetical_input[None, :, :] - bg_data[0]
        hypothetical_contribs = hypothetical_diffref * mult[0]
        projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
    # There are no bias importances, so the np.zeros_like(orig_inp[1]) is not needed.
    return [np.mean(projected_hypothetical_contribs, axis=0)]
