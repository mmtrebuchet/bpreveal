import logging
try:
    import jaccard
except ModuleNotFoundError:
    logging.error("Could not find the Jaccard module. You may need to run `make`"
                  " in the src/ directory.")
    raise

import h5py
import numpy as np
import utils
import csv
import scipy.signal


def slidingDotproduct(importanceScores, cwm):
    # The funny slice here is because scipy.signal.correlate doesn't collapse the
    # length-one dimension that is a result of both sequences having the same
    # second dimension.
    return scipy.signal.correlate(importanceScores, cwm, mode='valid')[:, 0]


def ppmToPwm(ppm, backgroundProbs):
    """Given a position probability matrix, which gives the probability of each
    base at each position, convert that into a position weight matrix, which gives
    the information (in bits) contained at each position.

    ppm is an ndarray of shape (motifLength, 4), representing the motif.
    backgroundProbs is an array of shape (4,), giving the background distribution of
    bases in the genome. For a genome with 60% AT and 40% GC, this would be
    [0.3, 0.2, 0.2, 0.3]

    returns the pwm, which is an array of shape (motifLength, 4)
    """
    return np.log2(ppm / backgroundProbs)


def ppmToPssm(ppm, backgroundProbs):
    """Given a position probability matrix, convert that to a pssm array,
    which is a measure of the information contained by each base.
    This method adds a small (1%) pseudocount at each position.

    ppm is an ndarray of shape (motifLength, 4), representing the motif.
    backgroundProbs is an array of shape (4,), with the same meaning as in
    ppmToPwm.

    Returns the pssm, an array of shape (motifLength, 4)"""

    # Add some pseudo counts and re-normalize
    ppm = ppm + 0.01
    ppm = ppm / np.sum(ppm, axis=1, keepdims=True)

    return np.log(ppm / backgroundProbs)


def cwmTrimPoints(cwm, trimThreshold, padding):
    """Given a cwm and a threshold, give the slice coordinates that should be used
    to remove bases with low contribution. For example:

               C
               C A  A
               C A CA
              AC ATCA
        nnnnTnACGATCAGnnnnn
        0123456789012345678
    where the number of letters indicates the importance, should be trimmed to get rid
    of the noisy (n) bases, and also maybe the flanking T and G.
    We calculate the maximum contribution (in this case, the stack of 5 Cs)
    and trim off all bases on the outside that are below trimThreshold*maxContrib
    If trimThreshold were 0.25, then any bases with less than 1.25 contribution would be
    trimmed, like so:
         C
         C A  A
         C A CA
        AC ATCA
        ACGATCA
        6789012

    If padding is set to zero, then this function will return (6,13), the indices of the passing
    bases. (Note that it goes to 13, not 12, because Python slices go *up to* the second index.
    We add padding to each side. If padding were 3, then the returned indices would be
         C
         C A  A
         C A CA
        AC ATCA
     nTnACGATCAgnn
     3456789012345
     and the returned indexes would be (3,16)

    cwm is a ndarray of shape (cwmlength, 4)
    trimThreshold is a floating point number. The lower this is,
        the more flanking bases will be kept.

    returns the start and stop indices of the trimmed motif, so:
    start, stop = cwmTrimPoints(cwm, threshold)
    newCwm = cwm[start:stop,:]
    (the : in the second axis is because the cwm will have shape (length, 4)
    """

    cwmSums = np.sum(np.abs(cwm), axis=1)
    cutoffValue = np.max(cwmSums) * trimThreshold

    passingBases = np.where(cwmSums > cutoffValue)
    startBase = np.min(passingBases) - padding
    if startBase < 0:
        startBase = 0

    stopBase = np.max(passingBases) + padding + 1
    if stopBase > cwm.shape[0] + 1:
        stopBase = cwm.shape[0] + 1

    return (startBase, stopBase)


class Pattern:
    """A pattern is a simple data storage class, representing a Modisco pattern and
    information about its seqlets and quantiles and stuff."""
    # The name of the metacluster, like "neg_patterns" or "pos_patterns"
    metaclusterName = None
    # The name of the pattern (i.e., motif), like "pattern_0"
    patternName = None
    # A (length, 4) array of the contribution weight matrix.
    cwm = None
    # A (length, 4) array of the probability of each base at each position in the pattern.
    ppm = None
    # The position weight matrix for this motif, the usual motif representation in logos.
    pwm = None
    # The information content at each base in the motif.
    pssm = None
    # When trimming the motif using cwmTrimPoints, where should you start and stop?
    cwmTrimLeftPoint = None
    cwmTrimRightPoint = None
    # For quick reference, I store the trimmed cwm and pssm.
    cwmTrim = None
    pssmTrim = None
    # (if you want the trimmed pwm, it's pwm[cwmTrimLeftPoint:cwmTrimRightPoint].)

    # How many seqlets are in this pattern?
    numSeqlets = None

    # The following arrays are of shape (numSeqlets,):
    # Relative to the cwm window (given by seqletIndexes), where does this seqlet start?
    seqletStarts = None
    # Ditto, but where does it end?
    seqletEnds = None
    # This is the index from the modisco output file, which is currently meaningless.
    # TODO When tfmodisco-lite carries through the seqlet indexes correctly, modify this.
    seqletIndexes = None
    # For each seqlet, is it reverse-complemented?
    seqletRevcomps = None

    # A list of strings giving the sequence of each seqlet.
    seqletSequences = None

    # The following arrays are of shape (numSeqlets, seqletLength, 4):
    # The one-hot encoded sequence.
    seqletOneHots = None
    # The contribution scores for each base.
    seqletContribs = None

    # The following arrays are of shape (numSeqlets,):
    # The information content match for each seqlet to the pattern's pssm.
    seqletICs = None
    # The continuous Jaccard similarity between each seqlet and the pattern's cwm
    seqletJaccards = None
    # The sum of the absolute value of all contribution scores for each seqlet.
    seqletL1s = None
    # For each seqlet, what quantile of IC match does it fall in? Numbers near zero indicate
    # a poor match, high numbers indicate high sequence similarity.
    seqletICQuantiles = None
    # Ditto, but for the match between importance scores and the cwm
    seqletJaccardQuantiles = None
    # Ditto, but just based on the total contribution of each seqlet.
    seqletL1Quantiles = None

    # When you give the quantile bounds to getCutoffs, these get stored:
    # What is the minimal information content (i.e., pssm) match score for a hit?
    cutoffIC = None
    # What is the minimal Jaccard similarity between a seqlet and the cwm for a hit?
    cutoffJaccard = None
    # What is the minimum total contribution a seqlet must have to be a hit?
    cutoffL1 = None

    # If you load in genomic coordinates by providing the contribution hdf5, these
    # members will get populated. Each is a list of length numSeqlets.
    seqletChroms = None
    # After trimming, where does the motif start?
    seqletGenomicStarts = None
    # After trimming, where does the motif end?
    seqletGenomicEnds = None

    def __init__(self, metaclusterName, patternName):
        self.metaclusterName = metaclusterName
        self.patternName = patternName

    def loadCwm(self, modiscoFp, trimThreshold, padding, backgroundProbs):
        """Given an opened hdf5 file object, load up the contribution scores for this
        pattern.
        trimThreshold and padding are used to trim the motifs, see cwmTrimPoints
        for documentation on those parameters.

        backgroundProbs gives the average frequency of each base across the genome as
        an array of shape (4,). See ppmToPwm and ppmToPssm for details on this parameter.

        After running this function, this Pattern object will contain a few arrays:

        pwm, which contains the positiong weight matrix for the underlying seqlets
        pssm, the information content of the motif
        ppm, the frequency of each base at each position.
        cwm, the contribution scores at each position.

        """
        h5Pattern = modiscoFp[self.metaclusterName][self.patternName]
        self.cwm = np.array(h5Pattern["contrib_scores"])
        self.ppm = np.array(h5Pattern["sequence"])
        self.pwm = ppmToPwm(self.ppm, backgroundProbs)
        self.pssm = ppmToPssm(self.ppm, backgroundProbs)
        self.cwmTrimLeftPoint, self.cwmTrimRightPoint = cwmTrimPoints(self.cwm,
                                                            trimThreshold,
                                                            padding)
        self.cwmTrim = self.cwm[self.cwmTrimLeftPoint:self.cwmTrimRightPoint]
        self.pssmTrim = self.pssm[self.cwmTrimLeftPoint:self.cwmTrimRightPoint]

    def loadSeqlets(self, modiscoFp):
        """This function loads up all the seqlet data from the modisco file and calculates
        quantile values for information content match, contribution jaccard match,
        and contribution L1 match.
        """
        seqletsGroup = modiscoFp[self.metaclusterName][self.patternName]["seqlets"]
        self.seqletContribs = np.array(seqletsGroup["contrib_scores"])
        self.seqletStarts = np.array(seqletsGroup["start"])
        self.seqletEnds = np.array(seqletsGroup["end"])
        self.seqletIndexes = np.array(seqletsGroup["example_idx"])
        self.seqletRevcomps = np.array(seqletsGroup["is_revcomp"])
        self.seqletOneHots = np.array(seqletsGroup["sequence"])
        self.seqletSequences = [utils.oneHotDecode(x) for x in self.seqletOneHots]
        self.numSeqlets = int(seqletsGroup["n_seqlets"][0])
        # Now it's time to calculate stuff!
        # First, scan the seqlets and get their similarities to the called motif.
        seqletICScan = [
            float(slidingDotproduct(x[self.cwmTrimLeftPoint:self.cwmTrimRightPoint], self.cwmTrim))
            for x in self.seqletOneHots]
        self.seqletICs = np.array(seqletICScan)
        self.seqletJaccards = np.zeros((self.numSeqlets,))
        self.seqletL1s = np.zeros((self.numSeqlets,))
        for i in range(self.numSeqlets):
            trimmedSeqlet = self.seqletContribs[i, self.cwmTrimLeftPoint:self.cwmTrimRightPoint]
            jaccardScores, L1 = jaccard.slidingJaccard(trimmedSeqlet, self.cwmTrim)
            self.seqletJaccards[i] = float(jaccardScores)
            self.seqletL1s[i] = float(L1)

        # Now we're ready to get quantile information.
        def makeQuantiles(ar):
            arSort = np.argsort(ar)
            ret = np.zeros(ar.shape, dtype=np.float32)
            ret[arSort] = np.linspace(0, 1, num=self.numSeqlets, endpoint=True)
            return ret
        self.seqletICQuantiles = makeQuantiles(self.seqletICs)
        self.seqletJaccardQuantiles = makeQuantiles(self.seqletJaccards)
        self.seqletL1Quantiles = makeQuantiles(self.seqletL1s)

    def getCutoffs(self, quantileIC, quantileJaccard, quantileL1):
        """Given the quantile values you want to use as cutoffs, actually
        look at the seqlet data and pick the right parameters. If you want to choose
        your own quantile cutoffs, set the cutoffIC, cutoffJaccard, and cutoffL1 members
        of this object."""
        self.cutoffIC = np.quantile(self.seqletICs, quantileIC)
        self.cutoffJaccard = np.quantile(self.seqletJaccards, quantileJaccard)
        self.cutoffL1 = np.quantile(self.seqletL1s, quantileL1)

    def seqletInfoIterator(self):
        """An iterator (meaning you can use it as the range in a for loop)
        that goes over every seqlet and returns a dictionary of information about it.
        This is useful when you want to write those seqlets to a file.
        Returns a dictionary of
        {   "chrom" : <string>,
            "start" : <integer>,
            "end" : <integer>,
            "short-name" : <string>,
            "L1-score" : <float>,
            "strand" : <character>,

            "metacluster-name" : <string>,
            "pattern-name" : <string>
            "sequence" : <string>,
            "index" : <integer>,
            "ic-match" : <float>,
            "contrib-match" : <float>,
            "L1-quantile" : <float>,
            "ic-quantile" : <float>,
            "contrib-quantile" : <float>
        }
        """
        shortName = (self.metaclusterName.split("_")[0]) + "_" + (self.patternName.split("_")[1])
        for i in range(self.numSeqlets):
            ret = dict()
            # TODO Once we have the abilitiy to map back the seqlet positions, make this meaningful!
            ret["chrom"] = self.seqletChroms[i]
            # TODO ditto
            ret["start"] = self.seqletGenomicStarts[i]
            # TODO ditto
            ret["end"] = self.seqletGenomicEnds[i]
            ret["short-name"] = shortName
            ret["L1-score"] = self.seqletL1s[i]
            ret["strand"] = '-' if self.seqletRevcomps[i] else '+'
            ret["metacluster-name"] = self.metaclusterName
            ret["pattern-name"] = self.patternName
            ret["sequence"] = self.seqletSequences[i]
            ret["index"] = self.seqletIndexes[i]
            ret["ic-match"] = self.seqletICs[i]
            ret["contrib-match"] = self.seqletJaccards[i]
            ret["L1-quantile"] = self.seqletL1Quantiles[i]
            ret["ic-quantile"] = self.seqletICQuantiles[i]
            ret["contrib-quantile"] = self.seqletJaccardQuantiles[i]
            yield ret

    def loadSeqletCoordinates(self, contribH5fp):
        """This method is incomplete because we haven't fixed a bug in tfmodisco-lite that allows
        us to recover the genomic coordinates of the seqlets identified by modisco.
        When it is done, it will use the seqlet indexes from the modisco output to look
        up the genomic coordinates of each seqlet and create three new fields in this object:
        seqletChroms, seqletGenomicStarts, seqletGenomicEnds.

        """
        self.seqletChroms = ["chr1" for _ in range(self.numSeqlets)]
        self.seqletGenomicStarts = self.seqletStarts
        self.seqletGenomicEnds = self.seqletEnds

    def getScanningInfo(self):
        """Get a dictionary (that can be converted to json) containing
        all the information needed to map motifs by cwm scanning."""
        return {"metacluster-name": self.metaclusterName,
                "pattern-name": self.patternName,
                "cwm": self.cwmTrim.tolist(),
                "pssm": self.pssmTrim.tolist(),
                "ic-cutoff": self.cutoffIC,
                "jaccard-cutoff": self.cutoffJaccard,
                "L1-cutoff": self.cutoffL1}


def analyzeSeqlets(modiscoH5Fname, contribH5Fname, patternSpec,
                   quantileIC, quantileJaccard, quantileL1,
                   trimThreshold, trimPadding, backgroundProbs,
                   outputSeqletsFname=None):
    """Given a modisco hdf5 file, go over the seqlets and establish the quantile boundaries.
    If you give hard cutoffs for information content and L1 norm match, this function need not
    be called.

    modiscoH5Name is a string giving the name of the modisco output hdf5.

    contribH5Name is the name of the hdf5 file that was generated by interpretFlat.py.
        This file contains genomic coordinates of the seqlets, and is used to determine the
        location of each seqlet identified by modisco.
        FIXME TODO FIXME:
        Since modisco doesn't preserve seqlet indexes right now, this parameter is ignored.
        All seqlets will be reported as being on chromosome chr1 and will have a meaningless
        position.

    patternSpec is either a list of dicts, or the string "all". See makePatternObjects for how
        this parameter is interpreted.

    quantileIC is the information content shared between the PSSM (which is based on nucleotide
        frequency, NOT contribution scores. A lower value means allow sequences which are a worse
        match to the PSSM.
    quantileJaccard gives the similarity required between the contribution scores of each seqlet
        and the cwm of the pattern (i.e., motif). Lower means let through worse matches to the
        contribution scores.
    quantileL1 gives the cutoff in terms of total importance for a seqlet for it to be considered.
        A low value means that seqlets that have low total contribution (sum(abs(contrib scores)))
        can still be considered hits.

    trimThreshold and trimBuffer give the cutoffs for trimming off uninformative flanks of the cwms.
        See cwmTrimPoints for details of these parameters.

    backgroundProbs is an array of shape (4,) of floats that gives the background distribution for
        each base in the genome. See ppmToPwm and pwmToPssm for details on this argument.

    outputSeqletsFname, if provided, gives a name for a file where the all of the seqlets in the
        Modisco output should be saved as a csv file.

    Returns a list of dicts that will be needed by the cwm scanning utility.
        It is structured as follows:
        [{  "metacluster-name" : <string>,
            "pattern-name" : <string>,
            "cwm" : < array of floats of shape (length, 4) >,
            "pssm" : < array of floats of shape (length, 4) >,
            "ic-cutoff" : <float>,
            "jaccard-cutoff" : <float>,
            "L1-cutoff" : <float> },
        ...
        ]

    This dictionary can be saved as a json for use with the cwm-scanning tool, or passed directly to
    the cwm scanning Python functions.

    """
    patterns = makePatternObjects(patternSpec, modiscoH5Fname)
    logging.info("Initialized patterns, beginning to load data.")
    with h5py.File(modiscoH5Fname, "r") as modiscoFp:
        for pattern in patterns:
            pattern.loadCwm(modiscoFp, trimThreshold, trimPadding, backgroundProbs)
            pattern.loadSeqlets(modiscoFp)
            pattern.getCutoffs(quantileIC, quantileJaccard, quantileL1)
    logging.info("Loaded and analyzed seqlet data.")
    if outputSeqletsFname is not None:
        # We should load up the genomic coordinates of the seqlets.
        logging.info("Writing csv of seqlet information.")
        with h5py.File(contribH5Fname, "r") as contribH5fp:
            for pattern in patterns:
                pattern.loadSeqletCoordinates(contribH5fp)
        # Now, write the output.
        with open(outputSeqletsFname, "w", newline='') as outFp:
            # Write the header.
            fieldNames = ["chrom", "start", "end", "short-name", "L1-score", "strand",
                         "metacluster-name", "pattern-name", "sequence", "index", "ic-match",
                         "contrib-match", "L1-quantile", "ic-quantile", "contrib-quantile"]
            writer = csv.DictWriter(outFp, fieldnames=fieldNames)
            writer.writeheader()
            for pattern in patterns:
                for seqletDict in pattern.seqletInfoIterator():
                    writer.writerow(seqletDict)
    # Now, we've saved all the metadata and it's time to return the actual information that
    # will be needed for mapping.
    ret = []
    for pattern in patterns:
        ret.append(pattern.getScanningInfo())
    logging.info("Done with analyzing seqlets.")
    return ret


def makePatternObjects(patternSpec, modiscoH5Fname):
    """Get a list of patterns to scan. If patternSpec is a list,
    it may have one of two forms:
    [ {"metacluster-name" : <string>,
       "pattern-name" : <string>},
      ...
    ]
    where each entry gives one metacluster with ONE pattern,
    or it may give multiple patterns as a list:
    [ {"metacluster-name" : <string>,
       "pattern-names" : [<string>, ...]},
      ...
    ]
    (Note the 's' in pattern-names, as opposed to the singular pattern-name above.)

    Alternatively, patternSpec may be the string "all"
    in which case all patterns in the modisco hdf5 file will be scanned."""
    patterns = []
    if patternSpec == "all":
        logging.debug("Loading full pattern information since patternSpec was all")
        with h5py.File(modiscoH5Fname, "r") as modiscoFp:
            for metaclusterName in modiscoFp.keys():
                for patternName in modiscoFp[metaclusterName].keys():
                    patterns.append(Pattern(metaclusterName, patternName))
    else:
        for metaclusterSpec in patternSpec:
            logging.debug("Initializing patterns {0:s}".format(str(metaclusterSpec)))
            if "pattern-names" in metaclusterSpec:
                for patternName in metaclusterSpec["pattern-names"]:
                    patterns.append(Pattern(metaclusterSpec["metacluster-name"],
                                            patternName))
            else:
                patterns.append(Pattern(metaclusterSpec["metacluster-name"],
                                        metaclusterSpec["pattern-name"]))
    logging.info("Initialized {0:d} patterns.".format(len(patterns)))
    return patterns


def runCwmScan(patterns, contribPath, outputHitsFname):
    pass
