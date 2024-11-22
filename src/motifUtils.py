"""Provides a set of classes and functions to perform CWM-scanning.

While there is very little shared code, the algorithm is taken from the
original BPNet repository, which is released under an MIT-style license.
You can find a copy it ``etc/bpnet_license.txt``.
"""

# In brief, a summary of these classes and functions are below:

# def arrayQuantileMap: maps quantiles of a value from a standard distribution
# def slidingDotproduct: scanning mechanism for scoring hits
# def ppmToPwm: converts PPM to PWM (log2)
# def ppmToPssm: converts PPM to PSSM (log)
# def cwmTrimPoints: determines TF-MoDISco trimmed pattern boundaries
# class Pattern: holds TF-MoDISco pattern metadata and associated functions
# def seqletCutoffs: defines quantile thresholds from seqlets of a pattern
# def makePatternObjects: create a consolidated set of desired patterns to scan
# class MiniPattern: compressed reimplementation of class Pattern
# class Hit: holds mapped coordinate and quality information of each hit
# class PatternScanner: constructs scanner
# def scannerThread: thread involved with scanning a contribution window
# def writerThread: thread involved with writing hits
# def scanPatterns: implements other utilities to scan TF-MoDISco patterns

import csv
import multiprocessing
from typing import Literal
from collections.abc import Iterable
import queue
import h5py
import numpy as np
import numpy.typing as npt
import scipy.signal
from bpreveal import utils
from bpreveal.internal.constants import IMPORTANCE_AR_T, IMPORTANCE_T, \
    MOTIF_FLOAT_AR_T, ONEHOT_AR_T, ONEHOT_T, MOTIF_FLOAT_T, \
    GENOME_NUCLEOTIDE_FREQUENCY
from bpreveal.logUtils import wrapTqdm
from bpreveal import logUtils
from bpreveal.internal.crashQueue import CrashQueue
try:
    from bpreveal import jaccard
except ModuleNotFoundError:
    logUtils.error("Could not find the Jaccard module. You may need to run `make`"
                   " in the src/ directory.")
    raise


def arrayQuantileMap(standard: npt.NDArray, samples: npt.NDArray,
                     standardSorted: bool = False) -> MOTIF_FLOAT_AR_T:
    """Get each sample's quantile in standard array.

    :param standard: The reference array.
    :param samples: The values that will be placed among the standard.
    :param standardSorted: Is the standard array already sorted?
    :return: For each sample in `samples`, what quantile of `standard` would it fall in?

    For each sample in samples (samples is an array of shape (N,)), in
    which quantile of the array of standards (of shape (M,)) would
    that sample fall?

    For example, if standard were [1,2,3,5, 100],
    then a sample of 1 would be in the 0 quantile (since it'd fall at the
    beginning of the array of standards) while a sample of 10 would be somewhere
    the 0.75 quantile (corresponding to 5) and the 1.0 quantile (corresponding to
    the 100 standard).
    Its precise position is determined by interpolating the quantiles in the
    standards. So in the above example, 10 would be closer to 0.75 than 1.0, because
    10 is closer to 5 than it is to 100.
    Returns a single array of the quantile positions for the samples (of shape (N,)).
    For a sample array of [2, 4, 6, 99], this would be something like
    [0.25, 0.625, 0.75001, 0.9999].
    Values in samples that fall outside the bounds of the standard are clipped
    to have a quantile of exactly 0 or exactly 1.0.

    standardSorted, if True, means that the standard array is already sorted.
    It doesn't change the behavior of this function, but it can speed it up since
    this function will otherwise have to sort the array of standards.
    """
    if not standardSorted:
        standard = np.sort(standard)
    standardQuantiles = np.linspace(0, 1, num=standard.shape[0], endpoint=True, dtype=MOTIF_FLOAT_T)
    minStandard = standard[0]
    maxStandard = standard[-1]
    if np.any(samples < minStandard) or np.any(samples > maxStandard):
        # If we have values outside the bounds of standard, we need to copy samples
        # in order to do clipping.
        newSamples = np.copy(samples)
        newSamples[samples < minStandard] = minStandard
        newSamples[samples > maxStandard] = maxStandard
    else:
        newSamples = samples

    return np.interp(x=samples, xp=standard, fp=standardQuantiles).astype(MOTIF_FLOAT_T)


def slidingDotproduct(seqletValues: npt.NDArray, pssm: npt.NDArray) -> MOTIF_FLOAT_AR_T:
    """Run the sliding dotproduct algorithm used in the original BPNet.

    Simply put, it's just a convolution.

    :param seqletValues: is a ndarray of shape (numSequences NUM_BASES),
    :param pssm: the pssm, an array of shape (motifLength, NUM_BASES)
    :return: Returns the scores of comparing the seqlets to their PSSM,
        an array of shape (motifLength, NUM_BASES)

    """
    # The funny slice here is because scipy.signal.correlate doesn't collapse the
    # length-one dimension that is a result of both sequences having the same
    # second dimension.
    return scipy.signal.correlate(seqletValues, pssm, mode="valid")[:, 0]


def ppmToPwm(ppm: npt.NDArray, backgroundProbs: npt.NDArray) -> MOTIF_FLOAT_AR_T:
    """Turn a position probability matrix into a position weight matrix.

    Given a position probability matrix, which gives the probability of each
    base at each position, convert that into a position weight matrix, which gives
    the information (in bits) contained at each position.

    ppm is an ndarray of shape (motifLength, NUM_BASES), representing the motif.
    backgroundProbs is an array of shape (NUM_BASES,), giving the background distribution of
    bases in the genome. For a genome with 60% AT and 40% GC, this would be
    [0.3, 0.2, 0.2, 0.3]

    returns the pwm, which is an array of shape (motifLength, NUM_BASES)

    :param ppm: is a ndarray of shape (pwmlength, NUM_BASES)
    :param backgroundProbs: is an ndarray of shape (NUM_BASES) representing
        probabilities to normalize information context over occurring
        frequencies
    :return: Returns the pwm, an array of shape (motifLength, NUM_BASES)

    """
    # Add a minute amount of pseudocounts just so that numpy doesn't whine about
    # overflow in the log.
    return np.log2((ppm / backgroundProbs) + 1e-30, dtype=MOTIF_FLOAT_T)


def ppmToPssm(ppm: npt.NDArray, backgroundProbs: npt.NDArray) -> MOTIF_FLOAT_AR_T:
    """Turn a position probability matrix into an information content matrix.

    Given a position probability matrix, convert that to a pssm array,
    which is a measure of the information contained by each base.
    This method adds a small (1%) pseudocount at each position.

    ppm is an ndarray of shape (motifLength, NUM_BASES), representing the motif.
    backgroundProbs is an array of shape (NUM_BASES,), with the same meaning as in
    ppmToPwm.

    :param ppm: is a ndarray of shape (pwmlength, NUM_BASES)
    :param backgroundProbs: is an ndarray of shape (NUM_BASES) representing
        probabilities to normalize information context over occurring
        frequencies
    :return: Returns the pssm, an array of shape (motifLength, NUM_BASES)
    """
    # Add some pseudo counts and re-normalize
    ppm = ppm + 0.01
    ppm = ppm / np.sum(ppm, axis=1, keepdims=True)

    return np.log(ppm / backgroundProbs, dtype=MOTIF_FLOAT_T)


def cwmTrimPoints(cwm: npt.NDArray,
                  trimThreshold: float, padding: int) -> tuple[int, int]:
    """Find where the motif actually is inside a CWM.

    :param cwm: is a ndarray of shape (cwmlength, NUM_BASES)
    :param trimThreshold: is a floating point number. The lower this is,
        the more flanking bases will be kept.
    :param padding: The number of bases that should be added back on each
        side of the trimmed motif.
    :return: the start and stop indices of the trimmed motif

    Given a cwm and a threshold, give the slice coordinates that should be used
    to remove bases with low contribution. For example::

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
    trimmed, like so::

         C
         C A  A
         C A CA
        AC ATCA
        ACGATCA
        6789012

    If padding is set to zero, then this function will return (6,13), the indices of the passing
    bases. (Note that it goes to 13, not 12, because Python slices go *up to* the second index.
    We add padding to each side. If padding were 3, then the returned indices would be::

            C
            C A  A
            C A CA
            AC ATCA
        nTnACGATCAgnn
        3456789012345

    and the returned indexes would be (3,16)

    The returned start and stop coordinates are a tuple, so::

        start, stop = cwmTrimPoints(cwm, threshold)
        newCwm = cwm[start:stop,:]

    (the : in the second axis is because the cwm will have shape (length, NUM_BASES)
    """
    cwmSums = np.sum(np.abs(cwm), axis=1)
    cutoffValue = np.max(cwmSums) * trimThreshold

    passingBases = np.where(cwmSums > cutoffValue)
    startBase = int(np.min(passingBases) - padding)
    if startBase < 0:
        startBase = 0

    stopBase = int(np.max(passingBases) + padding + 1)
    if stopBase > cwm.shape[0] + 1:
        stopBase = int(cwm.shape[0] + 1)

    return (startBase, stopBase)


class Seqlet:
    """Represents a single seqlet inside a pattern.

    :param start: Relative to the modisco window, where does this seqlet start?
    :param end: Relative to the modisco window, where does this seqlet end?
    :param index: What is the example_idx for this seqlet? This is the row number
        in the importance score hdf5.
    :param revcomp: Is this seqlet reverse-complemented?
    :param oneHot: The one-hot encoded sequence of this seqlet.
    :param contribs: The contribution scores from this seqlet.
    """

    start: int
    """Relative to the modisco window, where does this seqlet start?"""
    end: int
    """Relative to the modisco window, where does this seqlet end?"""

    index: int
    """This is example_idx in the modisco output file.

    It is used to index into the contribution hdf5 to get coordinate information.
    """
    revcomp: bool
    """Is this seqlet reverse-complemented?"""

    sequence: str
    """The sequence of this seqlet, as a string"""

    # The following arrays are of shape (numSeqlets, seqletLength, NUM_BASES):
    oneHot: npt.NDArray[ONEHOT_T]
    """The one-hot encoded sequence.

    Shape (seqletLength, NUM_BASES)
    """
    contribs: IMPORTANCE_AR_T
    """The contribution scores for each base.

    Shape (seqletLength, NUM_BASES)
    """

    seqMatch: MOTIF_FLOAT_T
    """The information content match for this seqlet to its pattern's pssm.
    """

    contribMatch: MOTIF_FLOAT_T
    """The continuous Jaccard similarity between this seqlet and its pattern's cwm"""

    contribMagnitude: MOTIF_FLOAT_T
    """The sum of the absolute value of all contribution scores for this seqlet"""

    chrom: str = "UNDEFINED"
    """ What chromosome is this seqlet on?

    Populated when you load in coordinates from the contribution hdf5.
    """

    genomicStart: int = -10000
    """After trimming, where does the motif start?
    Populated when you load in coordinates from the contribution hdf5.
    """
    genomicEnd: int = -10000
    """After trimming, where does the motif end?
    Populated when you load in coordinates from the contribution hdf5.
    """
    def __init__(self, start: int, end: int, index: int, revcomp: bool,
                 oneHot: ONEHOT_AR_T, contribs: IMPORTANCE_AR_T):
        self.start = start
        self.end = end
        self.index = index
        self.revcomp = revcomp
        self.oneHot = oneHot
        self.contribs = contribs
        self.sequence = utils.oneHotDecode(oneHot)

    def calcMatches(self, trimLeft: int, trimRight: int, cwm: MOTIF_FLOAT_AR_T,
                    pssm: MOTIF_FLOAT_AR_T) -> None:
        """See how this seqlet measures up against its pattern.

        :param trimLeft: How many bases should be trimmed from the left of the seqlet
            before measuring similarity?
        :param trimRight: How many bases should be trimmed from the right before measuring
            similarity?
        :param cwm: The pattern's cwm, used to calculate contribution similarity.
        :param pssm: The pattern's pssm, used to calculate sequence similarity.
        """
        trimmedSeqlet = self.contribs[trimLeft:trimRight]
        contribMatch, contribMag = jaccard.slidingJaccard(trimmedSeqlet, cwm)
        self.contribMatch = contribMatch[0]
        self.contribMagnitude = contribMag[0]
        seqMatch = slidingDotproduct(self.oneHot[trimLeft:trimRight], pssm)
        self.seqMatch = seqMatch[0]

    def loadCoordinates(self, contribFp: h5py.File, modiscoWindow: int) -> None:
        """Use this seqlet's index to figure out its original genomic coordinates.

        :param contribFp: The hdf5 file generated by interpretFlat.
        :param modiscoWindow: The size of the scanning window used by modisco.
        """
        contribChrom = contribFp["coords_chrom"][self.index]
        contribChromName = contribFp["chrom_names"].asstr()[contribChrom]
        contribStart = contribFp["coords_start"][self.index]
        contribEnd = contribFp["coords_end"][self.index]
        contribSeq = self.oneHot
        if self.revcomp:
            contribSeq = np.flip(contribSeq)
        contribMiddle = (contribStart + contribEnd) // 2
        windowStart = contribMiddle - modiscoWindow // 2
        genomeStart = windowStart + min(self.start, self.end)
        genomeEnd = windowStart + max(self.start, self.end)
        self.chrom = contribChromName
        self.genomicStart = genomeStart
        self.genomicEnd = genomeEnd


class Pattern:
    """A pattern is a simple data storage class.

    This class represents a TF-MoDISco pattern that contains metadata stored in
    the modisco.h5 file, re-represented in a helpful way to perform CWM-scanning
    and compute seqlet-derived quantile distribution information.
    """

    metaclusterName: str
    """The name of the metacluster, like "neg_patterns" or "pos_patterns" """
    patternName: str
    """The name of the pattern (i.e., motif), like "pattern_0" """
    shortName: str
    """The human-readable name of this pattern."""
    cwm: MOTIF_FLOAT_AR_T
    """A (length, NUM_BASES) array of the contribution weight matrix."""
    ppm: MOTIF_FLOAT_AR_T
    """A (length, NUM_BASES) array of the probability of each base at each position."""
    pwm: MOTIF_FLOAT_AR_T
    """The position weight matrix for this motif, the usual motif representation in logos.

    (if you want the trimmed pwm, it's pwm[cwmTrimLeftPoint:cwmTrimRightPoint].)
    """
    pssm: MOTIF_FLOAT_AR_T
    """The information content at each base in the motif."""
    cwmTrimLeftPoint: int
    """When trimming the motif using cwmTrimPoints, where should you start and stop?"""
    cwmTrimRightPoint: int
    """When trimming the motif using cwmTrimPoints, where should you start and stop?"""
    cwmTrim: MOTIF_FLOAT_AR_T
    """For quick reference, I store the trimmed cwm and pssm."""
    pssmTrim: MOTIF_FLOAT_AR_T
    """For quick reference, I store the trimmed cwm and pssm.

    (if you want the trimmed pwm, it's pwm[cwmTrimLeftPoint:cwmTrimRightPoint].)
    """

    numSeqlets: int
    """How many seqlets are in this pattern?"""
    seqlets: list[Seqlet]
    """All of the seqlets that comprise this pattern."""

    quantileSeqMatch: float | None
    """The quantile cutoff for sequence similarity. None means the cutoff is not used."""
    quantileContribMatch: float | None
    """The quantile cutoff for contribution similarity. None means the cutoff is not used."""
    quantileContribMagnitude: float | None
    """The quantile cutoff for contribution magnitude. None means the cutoff is not used."""

    # When you give the quantile bounds to getCutoffs, these get stored:
    cutoffSeqMatch: float | None
    """What is the minimal information content (i.e., pssm) match score for a hit?

    Stored when you give quantile bounds to getCutoffs
    """
    cutoffContribMatch: float | None
    """What is the minimal Jaccard similarity between a seqlet and the cwm for a hit?

    Stored when you give quantile bounds to getCutoffs
    """
    cutoffContribMagnitude: float | None
    """What is the minimum total contribution a seqlet must have to be a hit?

    Stored when you give quantile bounds to getCutoffs
    """

    def __init__(self, metaclusterName: str, patternName: str,
                 shortName: str | None = None) -> None:
        self.metaclusterName = metaclusterName
        self.patternName = patternName
        if shortName is None:
            shortMName = self.metaclusterName.split("_")[0]
            shortPName = self.patternName.split("_")[1]
            self.shortName = shortMName + "_" + shortPName
        else:
            self.shortName = shortName

    def setQuantiles(self, quantileSeqMatch: float | None,
                     quantileContribMatch: float | None,
                     quantileContribMagnitude: float | None) -> None:
        """Set up the quantile values that will be used to calculate cutoffs.

        :param quantileSeqMatch: is a float designating the minimum PSSM match quantile
            threshold that a mapped hit must meet based on the distribution of seqlet
            PSSM match scores. If value is None instead of float, skip this threshold.
        :param quantileContribMatch: is a float designating the minimum CWM match quantile
            threshold that a mapped hit must meet based on the distribution of seqlet
            CWM match scores. If value is None instead of float, skip this threshold.
        :param quantileContribMagnitude: is a float designating the minimum contribution
            quantile threshold that a mapped hit must meet based on the distribution
            of seqlet contribution scores. If value is None instead of float, skip this
            threshold.
        """
        self.quantileSeqMatch = quantileSeqMatch
        self.quantileContribMatch = quantileContribMatch
        self.quantileContribMagnitude = quantileContribMagnitude

    def loadCwm(self, modiscoFp: h5py.File, trimThreshold: float,
                padding: int, backgroundProbs: MOTIF_FLOAT_AR_T) -> None:
        """Given an opened hdf5 file object, load up the contribution scores for this pattern.

        :param modiscoFp: is a filepath referencing the modisco.h5 file
            generated by tfmodiscolite
        :param trimThreshold: is a floating point number. The lower this is,
            the more flanking bases will be kept.
        :param padding: is an integer. After trimming, pad each end of motif
            by this many flanking bases to reconstruct flank specificity.
        :param backgroundProbs: is an ndarray of shape (NUM_BASES) representing
            probabilities to normalize information context over occurring
            frequencies

        trimThreshold and padding are used to trim the motifs, see cwmTrimPoints
        for documentation on those parameters.

        backgroundProbs gives the average frequency of each base across the genome as
        an array of shape (NUM_BASES,). See ppmToPwm and ppmToPssm for details on this parameter.

        After running this function, this Pattern object will contain a few arrays:

        pwm, which contains the position weight matrix for the underlying seqlets
        pssm, the information content of the motif
        ppm, the frequency of each base at each position.
        cwm, the contribution scores at each position.

        """
        h5Pattern = modiscoFp[self.metaclusterName][self.patternName]
        self.cwm = np.array(h5Pattern["contrib_scores"], dtype=MOTIF_FLOAT_T)
        self.ppm = np.array(h5Pattern["sequence"], dtype=MOTIF_FLOAT_T)
        self.pwm = ppmToPwm(self.ppm, backgroundProbs)
        self.pssm = ppmToPssm(self.ppm, backgroundProbs)
        self.cwmTrimLeftPoint, self.cwmTrimRightPoint = cwmTrimPoints(self.cwm,
                                                                      trimThreshold,
                                                                      padding)
        self.cwmTrim = self.cwm[self.cwmTrimLeftPoint:self.cwmTrimRightPoint]
        self.pssmTrim = self.pssm[self.cwmTrimLeftPoint:self.cwmTrimRightPoint]

    def _callJaccard(self, seqlet: IMPORTANCE_AR_T,
                     cwm: MOTIF_FLOAT_AR_T) ->\
            tuple[MOTIF_FLOAT_AR_T, MOTIF_FLOAT_AR_T]:
        return jaccard.slidingJaccard(seqlet, cwm)

    def loadSeqlets(self, modiscoFp: h5py.File) -> None:
        """Load seqlets from the modisco hdf5.

        :param modiscoFp: is a filepath referencing the modisco.h5 file
            generated by tfmodiscolite

        This function loads up all the seqlet data from the modisco file and calculates
        quantile values for information content match, contribution jaccard match,
        and contribution L1 match.
        """
        seqletsGroup = modiscoFp[self.metaclusterName][self.patternName]["seqlets"]
        numSeqlets = int(seqletsGroup["n_seqlets"][0])
        self.seqlets = []
        for i in range(numSeqlets):
            seqlet = Seqlet(start=seqletsGroup["start"][i],
                            end=seqletsGroup["end"][i],
                            index=seqletsGroup["example_idx"][i],
                            revcomp=seqletsGroup["is_revcomp"][i],
                            oneHot=np.array(seqletsGroup["sequence"][i]),
                            contribs=np.array(seqletsGroup["contrib_scores"][i]))
            seqlet.calcMatches(self.cwmTrimLeftPoint, self.cwmTrimRightPoint,
                               self.cwmTrim, self.pssmTrim)
            self.seqlets.append(seqlet)

    def getCutoffs(self) -> None:
        """Calculate cutoff values given target quantiles.

        Given the quantile values you want to use as cutoffs, actually
        look at the seqlet data and pick the right parameters. If you want to choose
        your own quantile cutoffs, set the cutoffSeqMatch, cutoffContribMatch,
        and cutoffContribMagnitude members of this object.
        """
        if self.quantileSeqMatch is not None:
            seqMatches = np.array([x.seqMatch for x in self.seqlets])
            self.cutoffSeqMatch =\
                float(np.quantile(seqMatches, self.quantileSeqMatch))
        else:
            self.cutoffSeqMatch = None
        if self.quantileContribMatch is not None:
            contribMatches = np.array([x.contribMatch for x in self.seqlets])
            self.cutoffContribMatch =\
                float(np.quantile(contribMatches, self.quantileContribMatch))
        else:
            self.cutoffContribMatch = None
        if self.quantileContribMagnitude is not None:
            contribMagnitudes = np.array([x.contribMagnitude for x in self.seqlets])
            self.cutoffContribMagnitude =\
                float(np.quantile(contribMagnitudes, self.quantileContribMagnitude))
        else:
            self.cutoffContribMagnitude = None

    def seqletInfoIterator(self) -> Iterable[dict]:
        """Make an iterator to go over the seqlets.

        An iterator (meaning you can use it as the range in a for loop)
        that goes over every seqlet and returns a dictionary of information about it.
        This is useful when you want to write those seqlets to a file.
        Returns a dictionary of::

            {   "chrom" : <string>,
                "start" : <integer>,
                "end" : <integer>,
                "short-name" : <string>,
                "contrib-magnitude" : <float>,
                "strand" : <character>,

                "metacluster-name" : <string>,
                "pattern-name" : <string>
                "sequence" : <string>,
                "index" : <integer>,
                "seq-match" : <float>,
                "contrib-match" : <float>
            }
        """
        for s in self.seqlets:
            ret = {
                "chrom": s.chrom,
                "start": s.genomicStart,
                "end": s.genomicEnd,
                "short_name": self.shortName,
                "contrib_magnitude": s.contribMagnitude,
                "strand": "-" if s.revcomp else "+",
                "metacluster_name": self.metaclusterName,
                "pattern_name": self.patternName,
                "sequence": s.sequence,
                "region_index": s.index,
                "seq_match": s.seqMatch,
                "contrib_match": s.contribMatch
            }
            yield ret

    def loadSeqletCoordinates(self, contribH5fp: h5py.File, modiscoWindow: int) -> None:
        """Read in seqlet coordinates saved in the contribution hdf5.

        If modiscoWindow is 0, store invalid data. In BPReveal 6.0.0, a modiscoWindow
        value of 0 will trigger an assertion failure. For now, it issues an error log.

        There is a reported bug in tfmodisco-lite that resets the indexes of the seqlets.
        CHECK your outputs and make sure that the reported coordinates make sense!

        This method creates three new fields in this Pattern object:

        seqletChroms
            A list of strings containing the chromosome name for each seqlet.
            Each chromosome will be ``UNDEFINED`` if modiscoWindow is 0.
        seqletGenomicStarts
            The coordinates in the genome where the seqlet starts, or -10000
            if modiscoWindow is 0.
        seqletGenomicEnds
            The coordinates in the genome where the seqlet ends, or -10000
            if modiscoWindow is 0.
        """
        if modiscoWindow == 0:
            logUtils.error("modiscoWindow not provided. Cannot load coordinates! "
                           "This will be a fatal error in BPReveal 6.0.0.")
            return

        logUtils.debug("Loading coordinate information from contribution score file.")
        for s in self.seqlets:
            s.loadCoordinates(contribH5fp, modiscoWindow)

    def getScanningInfo(self) -> dict:
        """Get what you need to know to scan CWMs.

        metacluster-name
            Will be assigned as `pos_patterns` or `neg_patterns` to represent
            whether the sequence feature increases or decreases SHAP value
        pattern-name
            Identifier corresponding to the pattern returned by tf-modiscolite
        short-name
            Combined identifier of metacluster-name and pattern-name
        cwm
            The CWM, an array of shape (motifLength, NUM_BASES)
        pssm
            The PSSM, an array of shape (motifLength, NUM_BASES)
        seq-match-cutoff
            A float designating the minimum PSSM match quantile
            threshold that a mapped hit must meet based on the distribution of seqlet
            PSSM match scores. If value is None instead of float, skip this threshold.
        contrib-match-cutoff
            A float designating the minimum CWM match quantile
            threshold that a mapped hit must meet based on the distribution of seqlet
            CWM match scores. If value is None instead of float, skip this threshold.
        contrib-magnitude-cutoff
            A float designating the minimum contribution
            quantile threshold that a mapped hit must meet based on the distribution
            of seqlet contribution scores. If value is None instead of float, skip this
            threshold.

        Get a dictionary (that can be converted to json) containing
        all the information needed to map motifs by cwm scanning.
        """
        return {"metacluster-name": self.metaclusterName,
                "pattern-name": self.patternName,
                "short-name": self.shortName,
                "cwm": self.cwmTrim.tolist(),
                "pssm": self.pssmTrim.tolist(),
                "seq-match-cutoff": self.cutoffSeqMatch,
                "contrib-match-cutoff": self.cutoffContribMatch,
                "contrib-magnitude-cutoff": self.cutoffContribMagnitude}

    @property
    def seqletSeqMatches(self) -> list[MOTIF_FLOAT_T]:
        """Returns the contribution match for each seqlet.

        This is only present for backwards compatibility and returns a warning.
        """
        logUtils.warning(
            "You are calling seqletSeqMatches on a Pattern object. "
            "These have been moved into the Seqlet class. "
            "Instructions for updating: change myPattern.seqletSeqMatches to "
            "[x.seqMatch for x in pattern.seqlets]")
        return [x.seqMatch for x in self.seqlets]

    @property
    def seqletContribMatches(self) -> list[MOTIF_FLOAT_T]:
        """Returns the contribution match for each seqlet.

        This is only present for backwards compatibility and returns a warning.
        """
        logUtils.warning(
            "You are calling seqletContribMatches on a Pattern object. "
            "These have been moved into the Seqlet class. "
            "Instructions for updating: change myPattern.seqletContribMatches to "
            "[x.contribMatch for x in pattern.seqlets]")
        return [x.contribMatch for x in self.seqlets]

    @property
    def seqletContribMagnitudes(self) -> list[MOTIF_FLOAT_T]:
        """Returns the contribution magnitude for each seqlet.

        This is only present for backwards compatibility and returns a warning.
        """
        logUtils.warning(
            "You are calling seqletContribMagnitudes on a Pattern object. "
            "These have been moved into the Seqlet class. "
            "Instructions for updating: change myPattern.seqletContribMagnitudes to "
            "[x.contribMagnitude for x in pattern.seqlets]")
        return [x.contribMagnitude for x in self.seqlets]


def seqletCutoffs(modiscoH5Fname: str, contribH5Fname: str,
                  patternSpec: list[dict] | Literal["all"], quantileSeqMatch: float,
                  quantileContribMatch: float, quantileContribMagnitude: float,
                  trimThreshold: float, trimPadding: int,
                  backgroundProbs: MOTIF_FLOAT_AR_T,
                  modiscoWindow: int,
                  outputSeqletsFname: str | None = None) -> list[dict]:
    """Given a modisco hdf5 file, go over the seqlets and establish the quantile boundaries.

    If you give hard cutoffs for information content and L1 norm match, this function need not
    be called.

    :param modiscoH5Fname: Gives the name of the modisco output hdf5.

    :param contribH5Fname: The name of the hdf5 file that was generated by interpretFlat.py.
        This file contains genomic coordinates of the seqlets, and is used to determine the
        location of each seqlet identified by modisco. If outputSeqletsFname is None, this
        parameter is ignored.

    :param patternSpec: Either a list of dicts, or the string ``all``. See makePatternObjects
        for how this parameter is interpreted.

    :param quantileSeqMatch: The information content shared between the PSSM (which is based on
        nucleotide frequency, NOT contribution scores. A lower value means allow sequences
        which are a worse match to the PSSM.
    :param quantileContribMatch: The similarity required between the contribution scores of
        each seqlet and the cwm of the pattern (i.e., motif). Lower means let through worse
        matches to the contribution scores.
    :param quantileContribMagnitude: gives the cutoff in terms of total importance for a seqlet for
        it to be considered. A low value means that seqlets that have low total contribution
        (sum(abs(contrib scores))) can still be considered hits.

    :param trimThreshold: Gives how aggressive the flank-trimming will be.
    :param trimPadding: Gives the padding to be added to the flanks.
        See :func:`bpreveal.motifUtils.cwmTrimPoints` for details of these parameters.

    :param backgroundProbs: An array of shape (NUM_BASES,) of floats that gives the background
        distribution for each base in the genome. See :func:`~ppmToPwm` and
        :func:`ppmToPssm` for details on this argument.

    :param modiscoWindow: The size of the window that was used by Modisco during scanning.
        (That's the ``-w`` argument to ``modisco motifs``.) Until BPReveal 6.0.0, passing
        in a modiscoWindow value of 0 will disable seqlet coordinate extraction and raise
        a warning. After 6.0.0, this will cause an assertion error.
        if outputSeqletsFname is None, this parameter is ignored.

    :param outputSeqletsFname: (Optional) Gives a name for a file where the all of the
        seqlets in the Modisco output should be saved as a tsv file.

    :return: A list of dicts that will be needed by the cwm scanning utility.

    The returned list is structured as follows::
        [{"metacluster-name": <string>,
        "pattern-name": <string>,
        "cwm": <array of floats of shape (length, NUM_BASES)>,
        "pssm": <array of floats of shape (length, NUM_BASES)>,
        "seq-match-cutoff": <float-or-null>,
        "contrib-match-cutoff": <float-or-null>,
        "contrib-magnitude-cutoff": <float-or-null> },
        ...
        ]

    This dictionary can be saved as a json for use with the motif scanning tool, or passed
    directly to the motif scanning Python functions.
    """
    if isinstance(backgroundProbs, str):
        backgroundProbsVec = np.array(GENOME_NUCLEOTIDE_FREQUENCY[backgroundProbs])
        logUtils.debug(f"Loaded background {backgroundProbsVec} for genome {backgroundProbs}")
    else:
        backgroundProbsVec = np.array(backgroundProbs)
    patterns = makePatternObjects(patternSpec, modiscoH5Fname,
                                  quantileSeqMatch, quantileContribMatch,
                                  quantileContribMagnitude)
    logUtils.info("Initialized patterns, beginning to load data.")
    with h5py.File(modiscoH5Fname, "r") as modiscoFp:
        for pattern in patterns:
            pattern.loadCwm(modiscoFp, trimThreshold, trimPadding, backgroundProbsVec)
            pattern.loadSeqlets(modiscoFp)
            pattern.getCutoffs()
    logUtils.info("Loaded and analyzed seqlet data.")
    if outputSeqletsFname is not None:
        # We should load up the genomic coordinates of the seqlets.
        logUtils.info("Writing tsv of seqlet information.")
        with h5py.File(contribH5Fname, "r") as contribH5fp:
            for pattern in patterns:
                pattern.loadSeqletCoordinates(contribH5fp, modiscoWindow)
        # Now, write the output.
        with open(outputSeqletsFname, "w", newline="") as outFp:
            # Write the header.
            fieldNames = ["chrom", "start", "end", "short_name", "contrib_magnitude", "strand",
                          "metacluster_name", "pattern_name", "sequence", "region_index",
                          "seq_match", "contrib_match"]
            writer = csv.DictWriter(outFp, fieldnames=fieldNames,
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for pattern in patterns:
                for seqletDict in pattern.seqletInfoIterator():
                    writer.writerow(seqletDict)
    # Now, we've saved all the metadata and it's time to return the actual information that
    # will be needed for mapping.
    ret = []
    for pattern in patterns:
        ret.append(pattern.getScanningInfo())
    logUtils.info("Done with analyzing seqlets.")

    return ret


def makePatternObjects(patternSpec: list[dict] | str, modiscoH5Fname: str,
                       quantileSeqMatch: float | None, quantileContribMatch: float | None,
                       quantileContribMagnitude: float | None) -> list[Pattern]:
    """Get a list of patterns to scan.

    :param patternSpec: The pattern specs from the config json.
    :type patternSpec: list[dict] | "all"
    :param modiscoH5Fname: The name of the hdf5-format file generated by MoDISco.

    :param quantileSeqMatch: The default sequence match quantile if a pattern spec doesn't
        specify its own.
    :param quantileContribMatch: The default contribution match quantile cutoff if a
        pattern spec doesn't specify its own.
    :param quantileContribMagnitude: The default contribution magnitude cutoff to use
        if a pattern spec doesn't specify its own.

    If patternSpec is a list, it may have one of two forms::

        [ {"metacluster-name" : <string>,
        "pattern-name" : <string>,
        «"short-name" : <string> »
        «"seq-match-quantile": <number-or-null>»
        «"contrib-match-quantile": <number-or-null>»
        «"contrib-magnitude-quantile": <number-or-null>»
        },
        ...
        ]

    where each entry gives one metacluster with ONE pattern,
    or it may give multiple patterns as a list::

        [ {"metacluster-name" : <string>,
        "pattern-names" : [<string>, ...],
        «"short-names" : [<string>,...]»
        },
        ...
        ]

    (Note the 's' in pattern-names, as opposed to the singular pattern-name above.)
    short-name, or short-names, is an optional parameter. If given, then instead of the
    patterns being named "pos_2", they will have the name you give, which could be something
    much more human-readable, like "Sox2".

    If patternSpec is a list with one entry naming only a single pattern (the first
    form above), then the optional quantile cutoffs values, if provided, override the
    global cutoff value for scanning that pattern only.

    Alternatively, patternSpec may be the string "all"
    in which case all patterns in the modisco hdf5 file will be scanned.
    (And the short names will be pos_0, pos_1, ...)
    """
    patterns = []
    if patternSpec == "all":
        logUtils.debug("Loading full pattern information since patternSpec was all")
        with h5py.File(modiscoH5Fname, "r") as modiscoFp:
            for metaclusterName, metacluster in modiscoFp.items():
                for patternName in metacluster:
                    pat = Pattern(metaclusterName, patternName)
                    pat.setQuantiles(quantileSeqMatch=quantileSeqMatch,
                                     quantileContribMatch=quantileContribMatch,
                                     quantileContribMagnitude=quantileContribMagnitude)
                    patterns.append(pat)

    else:
        patternSpecList: list[dict] = patternSpec  # type: ignore
        for metaclusterSpec in patternSpecList:
            curSeqMatchQuantile = metaclusterSpec.get("seq-match-quantile",
                                                      quantileSeqMatch)
            curContribMatchQuantile = metaclusterSpec.get("contrib-match-quantile",
                                                          quantileContribMatch)
            curContribMagQuantile = metaclusterSpec.get("contrib-magnitude-quantile",
                                                        quantileContribMagnitude)
            logUtils.debug(f"Initializing patterns {metaclusterSpec}")
            if "pattern-names" in metaclusterSpec:
                for i, patternName in enumerate(metaclusterSpec["pattern-names"]):
                    if "short-names" in metaclusterSpec:
                        shortName = metaclusterSpec["short-names"][i]
                    else:
                        shortName = None
                    pat = Pattern(metaclusterSpec["metacluster-name"],
                                  patternName, shortName=shortName)
                    pat.setQuantiles(quantileSeqMatch=curSeqMatchQuantile,
                                     quantileContribMatch=curContribMatchQuantile,
                                     quantileContribMagnitude=curContribMagQuantile)
                    patterns.append(pat)
            else:
                if "short-name" in metaclusterSpec:
                    shortName = metaclusterSpec["short-name"]
                else:
                    shortName = None
                pat = Pattern(metaclusterSpec["metacluster-name"],
                              metaclusterSpec["pattern-name"],
                              shortName)
                pat.setQuantiles(quantileSeqMatch=curSeqMatchQuantile,
                                 quantileContribMatch=curContribMatchQuantile,
                                 quantileContribMagnitude=curContribMagQuantile)
                patterns.append(pat)
    logUtils.info(f"Initialized {len(patterns)} patterns.")
    return patterns


class MiniPattern:
    """A smaller pattern object for the scanners to use.

    metacluster-name
        Will be assigned as `pos_patterns` or `neg_patterns` to represent
        whether the sequence feature increases or decreases SHAP value
    pattern-name
        Identifier corresponding to the pattern returned by tf-modiscolite
    short-name
        Combined identifier of metacluster-name and pattern-name
    cwm
        The CWM, an array of shape (motifLength, NUM_BASES)
    rcwm
        The reverse complement CWM, an array of shape (motifLength, NUM_BASES)
    pssm
        The PSSM, an array of shape (motifLength, NUM_BASES)
    rpssm
        The reverse complement PSSM, an array of shape (motifLength, NUM_BASES)
    seqMatchCutoff
        A float designating the minimum PSSM match quantile
        threshold that a mapped hit must meet based on the distribution of seqlet
        PSSM match scores. If value is None instead of float, skip this threshold.
    contribMatchCutoff
        A float designating the minimum CWM match quantile
        threshold that a mapped hit must meet based on the distribution of seqlet
        CWM match scores. If value is None instead of float, skip this threshold.
    contribMagnitudeCutoff
        A float designating the minimum contribution
        quantile threshold that a mapped hit must meet based on the distribution
        of seqlet contribution scores. If value is None instead of float, skip this
        threshold.

    During the CWM scanning step, the full Pattern class is unnecessary, since it
    loads up a lot of seqlet data. This lightweight class is designed to be used inside
    the scanning threads. It represents one pattern, and can quickly scan sequences and
    contribution score tracks against its pattern.
    """

    def __init__(self, config: dict) -> None:
        """The config here is from the pattern json generated by the quantile script."""
        self.metaclusterName = config["metacluster-name"]
        self.patternName = config["pattern-name"]
        self.shortName = config["short-name"]
        self.cwm = np.array(config["cwm"], dtype=MOTIF_FLOAT_T)
        self.rcwm = np.flip(self.cwm)  # (revcomp)
        self.pssm = np.array(config["pssm"], dtype=MOTIF_FLOAT_T)
        self.rpssm = np.flip(self.pssm)  # (revcomp)
        self.seqMatchCutoff = config["seq-match-cutoff"]
        self.contribMatchCutoff = config["contrib-match-cutoff"]
        self.contribMagnitudeCutoff = config["contrib-magnitude-cutoff"]

    def _callJaccard(self, scores: IMPORTANCE_AR_T,
                     cwm: MOTIF_FLOAT_AR_T) -> \
            tuple[MOTIF_FLOAT_AR_T, MOTIF_FLOAT_AR_T]:
        # This is just a separate function so the profiler can see the call to Jaccard.
        return jaccard.slidingJaccard(scores, cwm)

    def _scanOneWay(self, sequence: ONEHOT_AR_T,
                    scores: IMPORTANCE_AR_T, cwm: MOTIF_FLOAT_AR_T,
                    pssm: MOTIF_FLOAT_AR_T, strand: Literal["+", "-"]
                    ) -> list[tuple[int, Literal["+", "-"], float, float, float]]:
        """Don't do revcomp - let scan take care of that.

        You should never call this method. Use scan instead!
        """
        contribMatchScores, contribMagnitudes = self._callJaccard(scores, cwm)
        seqMatchScores = slidingDotproduct(sequence, pssm)
        if self.contribMatchCutoff is not None:
            contribMatchPass = contribMatchScores > self.contribMatchCutoff
        else:
            contribMatchPass = np.full(contribMatchScores.shape, True, dtype="bool")

        if self.contribMagnitudeCutoff is not None:
            contribMagnitudePass = contribMagnitudes > self.contribMagnitudeCutoff
            contribPass = np.logical_and(contribMatchPass, contribMagnitudePass)
        else:
            contribPass = contribMatchPass

        if self.seqMatchCutoff is not None:
            seqMatchPass = seqMatchScores > self.seqMatchCutoff
            allPass = np.logical_and(contribPass, seqMatchPass)
        else:
            allPass = contribPass

        # Great! Now where did we get hits?
        passLocations = allPass.nonzero()[0]
        # Now build up a list of things to return.
        ret = []
        for passLoc in passLocations:
            contribMatchScore = contribMatchScores[passLoc]
            contribMagnitude = contribMagnitudes[passLoc]
            seqMatchScore = seqMatchScores[passLoc]
            ret.append((passLoc, strand, contribMagnitude, contribMatchScore, seqMatchScore))
        return ret

    def _scanWithoutCutoffsOneWay(self, sequence: ONEHOT_AR_T, scores: IMPORTANCE_AR_T,
                                  cwm: MOTIF_FLOAT_AR_T,
                                  pssm: MOTIF_FLOAT_AR_T)\
            -> tuple[MOTIF_FLOAT_AR_T,
                     MOTIF_FLOAT_AR_T,
                     MOTIF_FLOAT_AR_T]:
        contribMatchScores, contribMagnitudes = self._callJaccard(scores, cwm)
        seqMatchScores = slidingDotproduct(sequence, pssm)
        return (contribMatchScores, contribMagnitudes, seqMatchScores)

    def scanWithoutCutoffs(self, sequence: ONEHOT_AR_T, scores: IMPORTANCE_AR_T
                           ) -> list[MOTIF_FLOAT_AR_T]:
        """See how this pattern reacts to an importance profile.

        Instead of getting hits and putting them in the output queue, just return
        the arrays generated during scanning.
        This is useful to see how the scanner viewed your importance profile.

        :param sequence: The one-hot encoded sequence.
        :param scores: The importance scores.
        :return: A list of vectors from scanning this pattern against the sequence.

        The returned vectors will be, in order,

        0. Positive strand contribution match scores
        1. Positive strand contribution magnitudes
        2. Positive strand sequence match scores
        3. Negative strand contribution match scores
        4. Negative strand contribution magnitudes
        5. Negative strand sequence match scores
        """
        rets = []
        rets.extend(self._scanWithoutCutoffsOneWay(sequence, scores, self.cwm, self.pssm))
        rets.extend(self._scanWithoutCutoffsOneWay(sequence, scores, self.rcwm, self.rpssm))
        return rets

    def scan(self, sequence: ONEHOT_AR_T, scores: IMPORTANCE_AR_T
             ) -> list[tuple[int, Literal["+", "-"], float, float, float]]:
        """Given a sequence and a contribution track, identify places where this pattern matches.

        sequence is a (length, NUM_BASES) one-hot encoded
        DNA fragment, and scores is the actual (not hypothetical) contribution score
        data from that region. Scores is (length, NUM_BASES) in shape, but all of the
        bases that are not present should have zero contribution score.
        Returns a list of hits. A hit is a tuple with five elements.
        First, an integer giving the offset in the sequence where the hit starts.
        Second, the strand on which the hit was found.
        Third, the contribution magnitude of the pattern at that position.
        Fourth, the contribution match score of the importance scores at that position.
        Finally, the sequence match score against the pattern at that position.

        """
        hitsPos = self._scanOneWay(sequence, scores, self.cwm, self.pssm, "+")
        hitsNeg = self._scanOneWay(sequence, scores, self.rcwm, self.rpssm, "-")
        return hitsPos + hitsNeg


class Hit:
    """A small struct used to bundle up found hits for insertion in the pipe to the writer thread.

    chrom
        Chromosome belonging to hit coordinate
    start
        0-based start site of hit coordinate
    end
        0-based end site of hit coordinate
    shortName
        Combined identifier of metacluster-name and pattern-name
    patternName
        Identifier corresponding to the pattern returned by tf-modiscolite
    metaclusterName
        Will be assigned as `pos_patterns` or `neg_patterns` to represent
        whether the sequence feature increases or decreases SHAP value
    strand
        Strand alignment of the hit coordinate
    sequence
        Sequence of the hit coordinate
    index
        Region index that hit coordinate belongs to. Should following indexing of given
        contribution .h5 by which the hit was mapped across.
    contribMagnitude
        L1 magnitude of contribution (sum(abs(x))) across hit
    contribMatchScore
        Match score to the hit's motif CWM based on Jaccard similarity
    seqMatchScore
        Match score to the hit's motif PSSM given the log likelihood of a match

    """

    def __init__(self, chrom: str, start: int, end: int, shortName: str,
                 metaclusterName: str, patternName: str, strand: Literal["+", "-"],
                 sequence: str, index: int, contribMagnitude: float,
                 contribMatchScore: float, seqMatchScore: float) -> None:
        self.chrom = chrom
        self.start = start
        self.end = end
        self.shortName = shortName
        self.patternName = patternName
        self.metaclusterName = metaclusterName
        self.strand = strand
        self.sequence = sequence
        self.index = index
        self.contribMagnitude = contribMagnitude
        self.contribMatchScore = contribMatchScore
        self.seqMatchScore = seqMatchScore

    def toDict(self) -> dict:
        """Converts the class to a dictionary that's easier to use with tsv writers.

        chrom
            Chromosome belonging to hit coordinate
        start
            0-based start site of hit coordinate
        end
            0-based end site of hit coordinate
        short_name
            Combined identifier of metacluster-name and pattern-name
        contrib_magnitude
            L1 magnitude of contribution (sum(abs(x))) across hit
        strand
            Strand alignment of the hit coordinate
        sequence
            Sequence of the hit coordinate
        region_index
            Region index that hit coordinate belongs to. Should following indexing of given
            contribution .h5 by which the hit was mapped across.
        metacluster_name
            Will be assigned as `pos_patterns` or `neg_patterns` to represent
            whether the sequence feature increases or decreases SHAP value
        pattern_name
            Identifier corresponding to the pattern returned by tf-modiscolite
        seq_match
            Match score to the hit's motif PSSM given the log likelihood of a match
        contrib_match
            Match score to the hit's motif CWM based on Jaccard similarity

        """
        ret = {
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "short_name": self.shortName,
            "contrib_magnitude": self.contribMagnitude,
            "strand": self.strand,
            "sequence": self.sequence,
            "region_index": self.index,
            "metacluster_name": self.metaclusterName,
            "pattern_name": self.patternName,
            "seq_match": self.seqMatchScore,
            "contrib_match": self.contribMatchScore
        }
        return ret


class RegionScanner:
    """Used to scan one particular region at a time, without cutoffs."""

    def __init__(self, contribFname: str, patternConfig: dict):
        self.miniPatterns = [MiniPattern(x) for x in patternConfig]
        self.contribFp = h5py.File(contribFname, "r")
        self.chromIdxToName = {}
        for i, name in enumerate(self.contribFp["chrom_names"].asstr()):
            self.chromIdxToName[i] = name

    def scanIndex(self, idx: int) -> dict[tuple[str, str], list[MOTIF_FLOAT_AR_T]]:
        """Scan a single locus.

        Given an index into the hdf5 contribution file, extract the sequence
        and contribution scores there and run a scan.
        idx is an integer ranging from 0 to the number of entries in the hdf5 file
        (minus one, of course, since zero-based indexing).
        Returns a dict, where the keys are (patternName, patternShortName) tuples,
        and the values are (in order) contribMatch, contribMagnitude, seqMatch,
        reverseContribMatch, reverseContribMagnitude, reverseSeqMatch
        """
        # Get all the data for this index.
        oneHotSequence = np.array(self.contribFp["input_seqs"][idx])
        hypScores = np.array(self.contribFp["hyp_scores"][idx], dtype=MOTIF_FLOAT_T, order="C")
        contribScores = np.array(hypScores * oneHotSequence, dtype=IMPORTANCE_T, order="C")
        # Now we perform the scanning.
        ret = {}
        for pattern in self.miniPatterns:

            patRet = pattern.scanWithoutCutoffs(oneHotSequence, contribScores)
            ret[pattern.patternName, pattern.shortName] = patRet
        return ret

    def findRegions(self, chrom: str, start: int, end: int) -> list[tuple[int, int, int, int]]:
        """I don't know the index, but I know the chromosome and position I want scanned.

        Given a region, find where in the importance hdf5 that region
        would be found, and then scan it.
        """
        ret = []
        for i, chromIdx in enumerate(self.contribFp["coords_chrom"]):
            chromName = self.chromIdxToName[chromIdx]
            if chromName == chrom:
                startPos = self.contribFp["coords_start"][i]
                endPos = self.contribFp["coords_end"][i]
                if start <= startPos and endPos <= end:
                    ret.append((i, startPos, endPos, (startPos + endPos) // 2))
        return ret


class PatternScanner:
    """Used inside a scanner thread to do the actual scanning.

    Provides functionality to scan a CWM/PSSM/other
    motif representation across a given set of contribution score windows
    to assign scores.
    """

    def __init__(self, hitQueue: CrashQueue, contribFname: str,
                 patternConfig: list[dict]) -> None:
        """A tool to run the actual scans.

        hitQueue is a Multiprocessing Queue where found hits should be put().

        (A Hit put in this queue should be an instance of the Hit class defined
        above.)
        The contribFname is the name of the contribution hdf5 file that will
        be scanned. Even though this thread will only scan part of the file,
        each thread needs to open a copy of it to extract data, so each thread
        gets the name of the whole file.

        patternConfig is the dictionary/json generated by the quantile script.
        It contains the cwm and pssm of each pattern, as well as the cutoffs for
        hits.
        """
        self.hitQueue = hitQueue
        self.miniPatterns = [MiniPattern(x) for x in patternConfig]
        self.contribFp = h5py.File(contribFname, "r")
        self.firstHit = True
        self.firstIndex = True
        self.chromIdxToName = {}
        for i, name in enumerate(self.contribFp["chrom_names"].asstr()):
            self.chromIdxToName[i] = name

    def scanIndex(self, idx: int) -> None:
        """Scan a single locus.

        Given an index into the hdf5 contribution file, extract the sequence
        and contribution scores there and run a scan.
        idx is an integer ranging from 0 to the number of entries in the hdf5 file
        (minus one, of course, since zero-based indexing).
        This function will look for hits in this region and then stuff any hits it finds
        into hitQueue for saving by the writer thread.
        """
        # Get all the data for this index.
        chromIdx = self.contribFp["coords_chrom"][idx]
        if isinstance(chromIdx, bytes):
            logUtils.logFirstN(logUtils.ERROR,
                               "Detected an importance score file from before version 4.0. "
                               "This will be an error in BPReveal 7.0. "
                               "Instructions for updating: Re-calculate importance scores.",
                               1)
            chrom = chromIdx.decode("utf-8")
        else:
            chrom = self.chromIdxToName[chromIdx]
        regionStart = self.contribFp["coords_start"][idx]
        oneHotSequence = np.array(self.contribFp["input_seqs"][idx])
        hypScores = np.array(self.contribFp["hyp_scores"][idx], dtype=MOTIF_FLOAT_T, order="C")
        contribScores = np.array(hypScores * oneHotSequence, dtype=IMPORTANCE_T, order="C")
        # Now we perform the scanning.
        for pattern in self.miniPatterns:
            hits = pattern.scan(oneHotSequence, contribScores)
            if len(hits) > 0:
                # Hey, we found something!
                for hit in hits:
                    start = hit[0]
                    end = hit[0] + pattern.cwm.shape[0]
                    strand = hit[1]
                    hitOneHot = oneHotSequence[start:end, :]
                    if strand == "-":
                        hitOneHot = np.flip(hitOneHot)
                    hitSequence = utils.oneHotDecode(hitOneHot)
                    madeHit = Hit(chrom, regionStart + start,
                                  regionStart + end,
                                  pattern.shortName,
                                  pattern.metaclusterName,
                                  pattern.patternName,
                                  strand,
                                  hitSequence, idx,
                                  hit[2],
                                  hit[3], hit[4])
                    self.hitQueue.put(madeHit)

    def done(self) -> None:
        """When scanning is finished, close the contribution .h5 file."""
        self.contribFp.close()
        # Send a -1 as a signal that a thread has finished up.
        self.hitQueue.put(-1)


def scannerThread(queryQueue: CrashQueue, hitQueue: CrashQueue,
                  contribFname: str, patternConfig: list[dict]) -> None:
    """The thread for one scanner.

    Each scanner is looking for every pattern,
    and gets regions to scan from queryQueue. Every time it finds a hit in one of the
    queries it pulled, it stuffs those Hit objects into hitQueue.

    :param queryQueue: The queue that this thread will read from to get its queries.
    :param hitQueue: The queue this thread will put its results into.
    :param contribFname: is a string naming the hdf5-format file generated by
        interpretFlat.py.
    :param patternConfig:  a dictionary/json generated by the quantile script
        that contains the necessary information to scan for the patterns.

    """
    scanner = PatternScanner(hitQueue, contribFname, patternConfig)
    while True:
        curQuery = queryQueue.get()
        if curQuery == -1:
            # End of the line, finish up!
            scanner.done()
            break
        scanner.scanIndex(curQuery)


def writerThread(hitQueue: CrashQueue, scannerThreads: int, tsvFname: str) -> None:
    """A thread that runs concurrently with however many scanners you're using.

    It reads from hitQueue and saves out any hits to a csv file, a bed file, or both.

    :param hitQueue: The queue that the scanners will write to.
    :param scannerThreads: gives the number of scanners running concurrently.
        Why does the writer need to know this? Because each scanner sends a special flag
        down the queue when it's done, and the writer needs to know how many of these
        special values to expect before it knows that all the scanners have finished.
    :param tsvFname: The name of the tsv file that should be written.
    """
    threadsRemaining = scannerThreads
    logUtils.debug("Starting writer.")
    with open(tsvFname, "w", newline="") as outTsv:
        # Must match Hit.toDict()
        fieldNames = ["chrom", "start", "end", "short_name", "contrib_magnitude", "strand",
                      "metacluster_name", "pattern_name", "sequence", "region_index",
                      "seq_match", "contrib_match"]
        writer = csv.DictWriter(outTsv, fieldnames=fieldNames,
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        numWaits = 0
        while True:
            try:
                ret = hitQueue.get()
            except queue.Empty:
                logUtils.warning("Exceeded timeout waiting to see a hit. Either your motif is very"
                                 " rare, or there is a bug in the code. If you see this message"
                                 " multiple times, that's a bug.")
                numWaits += 1
                if numWaits > 10:
                    logUtils.error("Over ten timeouts have occurred. Aborting.")
                    raise
                continue  # Go back to the top of the loop - we don't have a ret to process.
            if ret == -1:
                threadsRemaining -= 1
                if threadsRemaining <= 0:
                    break
                continue
            writer.writerow(ret.toDict())


def scanPatterns(contribH5Fname: str, patternConfig: list[dict],
                 tsvFname: str, numThreads: int) -> None:
    """ContribH5Fname is the name of a contribution score file generated by interpretFlat.py.

    :param contribH5Fname: a string naming the hdf5-format file generated by
        interpretFlat.py.
    :param patternConfig: a dictionary/json generated by the quantile script
        that contains the necessary information to scan for the patterns.
    :param tsvFname:  the name of the output file containing the hits. Columns 1-6 of this file can
        be extracted with cut to get a bed file.
    :param numThreads: the *total* number of threads to use for the scanning. This must be at least
        three, since this function builds a pipeline with one thread generating queries,
        one saving results, and all the rest furiously scanning the query sequences for matches.
        I suggest running rampant with as many threads as they'll let you use, the scanning is very
        computationally expensive!

    """
    assert numThreads >= 3, "Scanning requires at least three threads. " \
                            "(but works great with fifty!)"
    # A queue size of 1024 is reasonable.
    queryQueue = multiprocessing.Queue(1024)
    hitQueue = multiprocessing.Queue(1024)
    logUtils.debug("Queues built. Starting threads.")
    scannerProcesses = []
    for _ in range(numThreads - 2):
        scanProc = multiprocessing.Process(target=scannerThread,
                                           args=[queryQueue, hitQueue, contribH5Fname,
                                                 patternConfig], daemon=True)
        scannerProcesses.append(scanProc)
    writeProc = multiprocessing.Process(target=writerThread,
                                        args=[hitQueue, numThreads - 2, tsvFname],
                                        daemon=True)
    logUtils.info("Starting threads.")
    for t in scannerProcesses:
        t.start()

    writeProc.start()
    with h5py.File(contribH5Fname, "r") as fp:
        # How many queries do I need to stuff down the query queue?
        numRegions = np.array(fp["coords_start"]).shape[0]
    logUtils.debug("Starting to send queries to the processes.")
    for i in wrapTqdm(range(numRegions)):
        # The queries are generated by the main thread.
        queryQueue.put(i)
    logUtils.debug("Done adding queries to the processes. Waiting for scanners to finish.")
    for _ in range(numThreads - 2):
        queryQueue.put(-1)

    for t in scannerProcesses:
        t.join()
    writeProc.join()
    logUtils.info("Done scanning.")
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
