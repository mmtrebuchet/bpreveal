#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import utils
import json
import pysam
import numpy as np
import tqdm
import h5py
import logging
import multiprocessing

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
    def __init__(self, oneHotSequence, passData, index):
        self.sequence = oneHotSequence
        self.passData = passData
        self.index = index

class Result:
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
    If the generator was using a fasta file, it is the title line from the original fasta, with its 
    leading '>' removed. 

    index, an int, indicates which address the data should be stored at in the output hdf5. 
    Since there's no order guarantee when you're receiving data, we have to keep track 
    of the order in the original input. 
    """

    def __init__(self, inputPrediction, shufflePredictions, sequence, shap, passData, index):
        self.inputPrediction = inputPrediction
        self.shufflePredictions = shufflePredictions
        self.sequence = sequence
        self.shap = shap
        self.passData = passData
        self.index = index


class PisaRunner:
    """I try to avoid class-based wrappers around simple things, but this is not simple. 
    This class creates threads to read in data, and then creates a thread that runs pisa samples in batches. 
    Finally, it takes the results from the pisa thread and saves those out to an hdf5-format file."""

    def __init__(self, modelFname, headId, taskId, batchSize, generator, saver, numShuffles, receptiveField, profileFname=None):
        """modelFname is the name of the model on disk
        headId and taskID are the head and task to shap, obviously,
        batchSize is the *shap* batch size, which should be your usual batch size divided by the number of shuffles.
        (since the original sequence and shuffles are run together)
        generator is a Generator object that will be passed to _generatorThread
        saver is a Saver object that will be used to save out the data.
        numShuffles is the number of reference sequences that should be generated for each shap evaluation. 
        (I recommend 20 or so)
        receptiveField is the receptive field of the model. To save on writing a lot of zeroes, the result
        objects only contain bases that are in the receptive field of the base being shapped. 
        profileFname is an optional parameter. If provided, profiling data (i.e., performance of this code)
        will be written to the given file name. Note that this has nothing to do with genomic profiles, it's 
        just benchmarking data for this code.
        """
        logging.info("Initializing pisa runner.")
        self._inQueue = multiprocessing.Queue(1000)
        self._outQueue = multiprocessing.Queue(1000)
        self._genThread = multiprocessing.Process(target=_generatorThread, args=(self._inQueue, generator))
        self._batchThread = multiprocessing.Process(target = _batcherThread, 
                                                    args=(modelFname, batchSize, 
                                                          self._inQueue, self._outQueue, headId, 
                                                          taskId, numShuffles, receptiveField, profileFname))
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

class Saver:
    """The saver runs in the main thread, so it's safe to open files and stuff in __init__.
    Descendants of this class shall receive Results from the batcher and save them to an 
    appropriate structure, usually on disk."""
    def __init__(self):
        raise NotImplemented()
    def add(self, result):
        """Add the given Result to wherever you're saving them out.
        Note that this will not be called with None at the end of the run, 
        that is special-cased in the code that runs your saver."""
        raise NotImplemented()
    def done(self):
        """Called when the batcher is complete (indicated by putting a None in its output queue). 
        Now is the time to close all of your files."""
        raise NotImplemented()

                                                                                    
class H5Saver(Saver):
    """Saves the shap scores to the output file. """
    def __init__(self, outputFname, numSamples, numShuffles, receptiveField, genome=None, useTqdm=False):
        """OutputFname is the name of the hdf5-format file that the shap scores will be deposited in.
        numSamples is the number of regions (i.e., bases) that pisa will be run on. 
        numShuffles is the number of shuffles that are used to generate the reference. 
        This is needed because we store reference predictions. 
        genome, an optional parameter, gives the name of a fasta-format file that contains the genome 
        of the organism. If provided, then chromosome name and size information will be 
        included in the output, and, additionally, 
        two other datasets will be created: coords_chrom, and coords_base. """
        logging.info("Initializing saver.")
        self._outFile = h5py.File(outputFname, "w")
        self.numSamples = numSamples
        self.numShuffles = numShuffles
        self.genomeFname = genome
        self._outFile.create_dataset("input_predictions", (numSamples,), dtype='f4')
        self._outFile.create_dataset("shuffle_predictions", (numSamples, numShuffles), dtype='f4')
        self._outFile.create_dataset("shap", (numSamples, receptiveField, 4), dtype='f4')
        #TODO for later: Adding compression to the sequence here absolutely tanks performan
        self._outFile.create_dataset('sequence', (numSamples, receptiveField, 4), dtype='u1')
        self.pbar = None
        if(useTqdm):
            self.pbar = tqdm.tqdm(total=numSamples)
        if(self.genomeFname is not None):
            self._loadGenome()
        else:
            self._outFile.create_dataset("descriptions", (numSamples,), dtype=h5py.string_dtype('utf-8'))
        logging.debug("Saver initialized, hdf5 file created.")

    def _loadGenome(self):
        """Does a few things. First, it creates chrom_names and chrom_sizes datasets in the output hdf5. 
        These two datasets just contain the (string) names and (unsigned) sizes for each one.
        Second, it populates these datasets. 
        Third, it creates two datasets: coords_chrom and coords_base, which store, for every shap value calculated, 
        what chromosome it was on, and where on that chromosome it was. These fields are populated during data receipt,
        because we don't have an ordering guarantee when we get data from the batcher. 
        """
        import pysam
        logging.debug("Loading genome information.")
        with pysam.FastaFile(self.genomeFname) as genome:
            self._outFile.create_dataset("chrom_names", (genome.nreferences,), dtype=h5py.string_dtype(encoding='utf-8'))
            self.chromNameToIdx = dict()
            posDtype = 'u4'
            for chromName in genome.references:
                #Are any chromosomes longer than 2^31 bp long? If so (even though it's unlikely), 
                #I must increase the storage size for positions. 
                if(genome.get_reference_length(chromName) > 2**31):
                    posDtype = 'u8'
    
            self._outFile.create_dataset("chrom_sizes", (genome.nreferences,), dtype=posDtype)
            
            for i, chromName in enumerate(genome.references):
                self._outFile['chrom_names'][i] = chromName
                self.chromNameToIdx[chromName] = i
                self._outFile['chrom_sizes'][i] = genome.get_reference_length(chromName)
    
            if(genome.nreferences <= 255):
                #If there are less than 255 chromosomes, I only need 8 bits to store the chromosome ID. 
                chromDtype = 'u1'
            elif(genome.nreferences <= 65535):
                #If there are less than 2^16 chromosomes, I can use 16 bits.
                chromDtype='u2'
            else:
                #If you have over four billion chromosomes, you deserve to have code that will break. 
                #So I just assume that 32 bits will be enough. 
                chromDtype = 'u4'
            self._outFile.create_dataset("coords_chrom", (self.numSamples,), dtype=chromDtype)
            self._outFile.create_dataset("coords_base", (self.numSamples,), dtype='u4')
        logging.debug("Genome data loaded.")
    
    def done(self):
        logging.debug("Saver closing.")
        if(self.pbar is not None):
            self.pbar.close()
        self._outFile.close()

    def add(self, result):
        if(self.pbar is not None):
            self.pbar.update()
        index = result.index
        self._outFile["input_predictions"][index] = result.inputPrediction
        self._outFile["shuffle_predictions"][index, :] = result.shufflePredictions
        self._outFile["sequence"][index,:,:] = result.sequence
        self._outFile["shap"][index,:,:] = result.shap
        #Okay, now we either add the description line, or add a genomic coordinate. 
        if(self.genomeFname is not None):
            self._outFile["coords_chrom"][index] = self.chromNameToIdx[result.passData[0]]
            self._outFile["coords_base"][index] = result.passData[1]
        else:
            self._outFile["descriptions"][index] = result.passData

class Generator:
    """This is the base class for generating pisa samples."""
    def __init__(self):
        raise NotImplemented()
    def __iter__(self):
        raise NotImplemented()
    def __next__(self):
        raise NotImplemented()
    def construct(self):
        #When the generator is about to start, this method is called once before requesting the iterator.
        raise NotImplemented()
    def done(self):
        #When the batcher is done, this is called to free any allocated resources.
        raise NotImplemented()


class FastaGenerator(Generator):
    def __init__(self, fastaFname):
        logging.info("Creating fasta generator.")
        self.fastaFname = fastaFname
        self.nowStop = False
        numRegions = 0

        with open(fastaFname, "r") as fp:
            for line in fp:
                if(len(line) > 0 and line[0] == ">"):
                    numRegions += 1
        #Iterate through the input file real quick and find out how many regions I'll have. 
        self.numRegions = numRegions
        self.index = 0
        logging.debug("Fasta generator initialized.")
    
    def construct(self):

        logging.info("Constructing fasta generator in its thread.")
        self.fastaFile = open(self.fastaFname, "r")
        self.nextSequenceID = self.fastaFile.readline()[1:].strip() #[1:] to get rid of the '>'. 
        logging.debug("Initial sequence to read: {0:s}".format(self.nextSequenceID))
    def done(self):
        logging.info("Closing fasta generator.")
        self.fastaFile.close()
    def __iter__(self):
        logging.debug("Creating fasta iterator.")
        return self

    def __next__(self):
        if(self.nowStop):
            raise StopIteration()
        sequence = ""
        prevSequenceID = self.nextSequenceID
        while(True):
            nextLine = self.fastaFile.readline()
            if(len(nextLine) == 0):
                self.nowStop = True
                break
            if(nextLine[0] == ">"):
                self.nextSequnceId = nextLine[1:].strip()
                break
            sequence += nextLine.strip()
        oneHotSequence = utils.oneHotEncode(sequence)
        q = Query(oneHotSequence, prevSequenceID, self.index)
        self.index += 1
        return q


class BedGenerator(Generator):

    def __init__(self, bedFname, genomeFname, inputLength, outputLength):
        logging.debug("Creating bed generator.")
        self.bedFname = bedFname
        self.genomeFname = genomeFname
        self.inputLength = inputLength
        self.outputLength = outputLength
        numRegions = 0
        #Note that this opens a file during construction (a no-no for my threading model)
        #but it just reads it and closes it right back up. When the initializer finishes, 
        #there are no pointers to file handles in this object.
        with open(self.bedFname, "r") as fp:
            for line in fp:
                lsp = line.split()
                start = int(lsp[1])
                stop = int(lsp[2])
                numRegions += (stop - start)
        self.numRegions = numRegions
        logging.info("Bed generator initialized with {0:d} regions".format(self.numRegions))

    def construct(self):
        #We create a list of all the regions now, but we'll look up the sequences and stuff on the fly. 
        logging.info("Constructing bed generator it its thread.")
        self.shapTargets = []
        self.genome = pysam.FastaFile(self.genomeFname)
        self.readHead = 0
        with open(self.bedFname) as fp:
            for line in fp:
                lsp = line.split()
                chrom = lsp[0]
                start = int(lsp[1])
                stop = int(lsp[2])
                for pos in range(start, stop):
                    self.shapTargets.append((chrom,pos))

    def __iter__(self):
        logging.debug("Creating iterator for bed generator.")
        return self

    def __next__(self):
        #Get the sequence and make a Query with it. 
        if(self.readHead >= len(self.shapTargets)):
            raise StopIteration()
        curChrom, curStart = self.shapTargets[self.readHead]
        padding = (self.inputLength -  self.outputLength) //2
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




def _batcherThread(modelName, batchSize, inQueue, outQueue, headId, taskId, numShuffles, receptiveField, profileFname=None):
    logging.debug("Starting batcher thread.")
    if(profileFname is not None):
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    b = _Batcher(modelName, batchSize, outQueue, headId, taskId, numShuffles, receptiveField)
    logging.debug("Batcher created.")
    while(True):
        query = inQueue.get()
        if(query is None):
            break
        b.addSample(query)
    logging.debug("Last query received. Finishing batcher thread.")
    b.finishBatch()
    outQueue.put(None)
    outQueue.close()
    if(profileFname is not None):
        profiler.create_stats()
        profiler.print_stats()
        profiler.dump_stats(profileFname)
    logging.debug("Batcher thread finished.")

def _generatorThread(inQueue, generator):
    logging.debug("Starting generator thread.")
    generator.construct()
    for elem in generator:
        inQueue.put(elem)
    inQueue.put(None)
    logging.debug("Done with generator, None added to queue.")
    inQueue.close()
    generator.done()
    logging.debug("Generator thread finished.")

def _saverThread(outQueue, saver):
    logging.debug("Saver 'thread' started.")
    while(True):
        rv = outQueue.get()
        if(rv is None):
            break
        saver.add(rv)
    saver.done()
    logging.debug("Saver 'thread' finished.")

class _Batcher:
    """The workhorse of this stack, it accepts queries until its internal storage is full, 
    then predicts them all at once, and runs shap."""
    def __init__(self, modelFname, batchSize, outQueue, headId, taskId, numShuffles, receptiveField):
        logging.info("Initializing batcher.")
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        import shap
        from keras.models import load_model
        import losses
        import utils
        import numpy as np
        utils.setMemoryGrowth()
        self.model = load_model(modelFname, 
                custom_objects = {'multinomialNll' : losses.multinomialNll})
        self.batchSize = batchSize
        self.outQueue = outQueue
        self.headId = headId
        self.taskId = taskId
        self.curBatch = []
        self.numShuffles = numShuffles
        self.receptiveField = receptiveField
        logging.debug("Batcher prepared, creating explainer.")
        #This slice....

        #                                                                 
        #                                                                 
        #                        Keep the first dimension, so it looks like a batch of size one--,              
        #                                            Sum the samples in this batch-,             |  
        #                                         The current task ----,           |             |         
        #                            The leftmost predicted base--,    |           |             |                 
        #                              All samples in the batch-, |    |           |             |
        #                                               Current | |    |           |             |
        #                                                head   V V    V           |             |
        outputTarget = tf.reduce_sum(self.model.outputs[headId][:,0, taskId], axis=0, keepdims=True)
        self.profileExplainer = shap.TFDeepExplainer((self.model.input, outputTarget), self.generateShuffles)
        logging.info("Batcher initialized, Explainer initialized. Ready for Queries to explain.")

    def generateShuffles(self, model_inputs):
        rng = np.random.default_rng(seed=355687)
        shuffles = [rng.permutation(model_inputs[0], axis=0) for x in range(self.numShuffles)]
        shuffles = np.array(shuffles)
        return [shuffles]

    def addSample(self, query):
        self.curBatch.append(query)
        if(len(self.curBatch) >= self.batchSize):
            self.finishBatch()

    def finishBatch(self):
        if(len(self.curBatch)):
            self.runPrediction()
            self.curBatch = []

    def runPrediction(self):
        #Now for the meat of all this boilerplate. Take the query sequences and 
        #run them through the model, then run the shuffles, then run shap. 
        #Finally, put all the results in the output queue.
        #This needs more optimization, but for this initial pass, I'm not actually batching -
        #instead, I'm running all the samples individually. But I can change this later,
        #and that's what counts. 
        #First, build up an array of sequences to test. 
        numQueries = len(self.curBatch)
        inputLength = self.curBatch[0].sequence.shape[0]
        oneHotBuf = np.empty((numQueries*(self.numShuffles+1), inputLength, 4),dtype=np.int8)
        #To predict on as large a batch as possible, I put the actual sequences and all the references for the current
        #batch into this array. The first <nsamples> rows are the real sequence, the 
        #next <numShuffles> rows are the shuffles of the first sequence
        #sequence, then the next is the shuffles of the second sequence, like this:
        #REAL_SEQUENCE_1_REAL_SEQUENCE_1_REAL_SEQUENCE_1
        #REAL_SEQUENCE_2_REAL_SEQUENCE_2_REAL_SEQUENCE_2
        #SEQUENCE_1_FIRST_SHUFFLE_SEQUENCE_1_FIRST_SHUFF
        #SEQUENCE_1_SECOND_SHUFFLE_SEQUENCE_1_SECOND_SHU
        #SEQUENCE_1_THIRD_SHUFFLE_SEQUENCE_1_THIRD_SHUFF
        #SEQUENCE_2_FIRST_SHUFFLE_SEQUENCE_2_FIRST_SHUFF
        #SEQUENCE_2_SECOND_SHUFFLE_SEQUENCE_2_SECOND_SHU
        #SEQUENCE_2_THIRD_SHUFFLE_SEQUENCE_2_THIRD_SHUFF

        #Points to the index into oneHotBuf where the next data should be added. 
        shuffleInsertHead = numQueries
        #These are the (real) sequences that will be passed to the explainer. 
        #Note that it's a list of arrays, and each array has shape (1,inputLength,4)
        
        for i,q in enumerate(self.curBatch):
            oneHotBuf[i,:,:] = q.sequence
            shuffles = self.generateShuffles([q.sequence])[0]
            oneHotBuf[shuffleInsertHead:shuffleInsertHead+self.numShuffles,:,:] = shuffles
            shuffleInsertHead += self.numShuffles
        #Okay, now the data structures are set up.
        fullPred = self.model.predict(oneHotBuf)
        outBasePreds = fullPred[self.headId][:,0,self.taskId]
        #(We'll deconvolve that in a minute...)
        shapScores = self.profileExplainer.shap_values([oneHotBuf[:numQueries,:,:]])
        #And now we need to run over that batch again to write the output. 
        shuffleReadHead = numQueries
        for i,q in enumerate(self.curBatch):
            querySequence = oneHotBuf[i,:self.receptiveField,:]
            queryPred = outBasePreds[i]
            queryShufPreds = outBasePreds[shuffleReadHead:shuffleReadHead+self.numShuffles]
            shuffleReadHead += self.numShuffles
            queryShapScores = shapScores[i,0:self.receptiveField,:]
            ret = Result(queryPred, queryShufPreds, querySequence, queryShapScores, q.passData, q.index)
            self.outQueue.put(ret)






