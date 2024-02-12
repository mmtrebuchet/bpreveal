"""TODO MELANIE Document
"""
import csv
import multiprocessing
import queue
from typing import Optional, Literal
import h5py
import numpy as np
import numpy.typing as npt
import scipy.signal
from bpreveal import utils
from bpreveal.utils import ONEHOT_AR_T, ONEHOT_T, MOTIF_FLOAT_T, QUEUE_TIMEOUT
from bpreveal.logUtils import wrapTqdm
from bpreveal import logUtils
try:
    from bpreveal import jaccard
except ModuleNotFoundError:
    logUtils.error("Could not find the Jaccard module. You may need to run `make`"
                  " in the src/ directory.")
    raise


def arrayQuantileMap(standard: npt.NDArray, samples: npt.NDArray,
                     standardSorted: Optional[bool] = False) -> npt.NDArray[MOTIF_FLOAT_T]:
    """Get each sample's quantile in standard array.

    :param standard: The reference array.
    :param sample: The values that will be placed among the standard.
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

    return np.interp(samples, standard, standardQuantiles).astype(MOTIF_FLOAT_T)


def slidingDotproduct(seqletValues: npt.NDArray, pssm: npt.NDArray) -> npt.NDArray[MOTIF_FLOAT_T]:
    """Run the sliding dotproduct algorithm used in the original BPNet.

    TODO MELANIE Document params and return

    It's just a convolution.
    """
    # The funny slice here is because scipy.signal.correlate doesn't collapse the
    # length-one dimension that is a result of both sequences having the same
    # second dimension.
    return scipy.signal.correlate(seqletValues, pssm, mode="valid")[:, 0]


def ppmToPwm(ppm: npt.NDArray, backgroundProbs: npt.NDArray) -> npt.NDArray[MOTIF_FLOAT_T]:
    """Turn a position probability matrix into a position weight matrix.

    Given a position probability matrix, which gives the probability of each
    base at each position, convert that into a position weight matrix, which gives
    the information (in bits) contained at each position.

    ppm is an ndarray of shape (motifLength, 4), representing the motif.
    backgroundProbs is an array of shape (4,), giving the background distribution of
    bases in the genome. For a genome with 60% AT and 40% GC, this would be
    [0.3, 0.2, 0.2, 0.3]

    returns the pwm, which is an array of shape (motifLength, 4)

    TODO MELANIE Convert to RST.
    """
    # Add a minute amount of pseudocounts just so that numpy doesn't whine about
    # overflow in the log.
    return np.log2((ppm / backgroundProbs) + 1e-30, dtype=MOTIF_FLOAT_T)


def ppmToPssm(ppm: npt.NDArray, backgroundProbs: npt.NDArray) -> npt.NDArray[MOTIF_FLOAT_T]:
    """Turn a position probability matrix into an information content matrix.

    Given a position probability matrix, convert that to a pssm array,
    which is a measure of the information contained by each base.
    This method adds a small (1%) pseudocount at each position.

    ppm is an ndarray of shape (motifLength, 4), representing the motif.
    backgroundProbs is an array of shape (4,), with the same meaning as in
    ppmToPwm.

    Returns the pssm, an array of shape (motifLength, 4)

    TODO MELANIE Convert to RST.
    """
    # Add some pseudo counts and re-normalize
    ppm = ppm + 0.01
    ppm = ppm / np.sum(ppm, axis=1, keepdims=True)

    return np.log(ppm / backgroundProbs, dtype=MOTIF_FLOAT_T)


def cwmTrimPoints(cwm: npt.NDArray,
                  trimThreshold: float, padding: int) -> tuple[int, int]:
    """Find where the motif actually is inside a CWM.

    :param cwm: is a ndarray of shape (cwmlength, 4)
    :param trimThreshold: is a floating point number. The lower this is,
        the more flanking bases will be kept.
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

    (the : in the second axis is because the cwm will have shape (length, 4)
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


class Pattern:
    """A pattern is a simple data storage class.

    TODO MELANIE Document constructor here in RST form.

    It represents a Modisco pattern and
    information about its seqlets and quantiles and stuff.
    """

    metaclusterName: str
    """The name of the metacluster, like "neg_patterns" or "pos_patterns" """
    patternName: str
    """The name of the pattern (i.e., motif), like "pattern_0" """
    shortName: str
    """The human-readable name of this pattern."""
    cwm: npt.NDArray[MOTIF_FLOAT_T]
    """A (length, 4) array of the contribution weight matrix."""
    ppm: npt.NDArray[MOTIF_FLOAT_T]
    """A (length, 4) array of the probability of each base at each position in the pattern."""
    pwm: npt.NDArray[MOTIF_FLOAT_T]
    """The position weight matrix for this motif, the usual motif representation in logos.

    (if you want the trimmed pwm, it's pwm[cwmTrimLeftPoint:cwmTrimRightPoint].)
    """
    pssm: npt.NDArray[MOTIF_FLOAT_T]
    """The information content at each base in the motif."""
    cwmTrimLeftPoint: int
    """When trimming the motif using cwmTrimPoints, where should you start and stop?"""
    cwmTrimRightPoint: int
    """When trimming the motif using cwmTrimPoints, where should you start and stop?"""
    cwmTrim: npt.NDArray[MOTIF_FLOAT_T]
    """For quick reference, I store the trimmed cwm and pssm."""
    pssmTrim: npt.NDArray[MOTIF_FLOAT_T]
    """For quick reference, I store the trimmed cwm and pssm.

    (if you want the trimmed pwm, it's pwm[cwmTrimLeftPoint:cwmTrimRightPoint].)
    """

    numSeqlets: int
    """How many seqlets are in this pattern?"""

    seqletStarts: npt.NDArray[np.int32]
    """Relative to the cwm window (given by seqletIndexes), where does this seqlet start?

    Shape (numSeqlets,)
    """
    seqletEnds: npt.NDArray[np.int32]
    """Relative to the cwm window (given by seqletIndexes), where does this seqlet end?

    Shape (numSeqlets,)
    """
    seqletIndexes: npt.NDArray[np.int32]
    """This is the index from the modisco output file, which is currently meaningless.

    TODO When tfmodisco-lite carries through the seqlet indexes correctly, modify this.
    Shape (numSeqlets,)
    """
    seqletRevcomps: npt.NDArray[np.bool_]
    """For each seqlet, is it reverse-complemented?

    Shape (numSeqlets,)
    """

    seqletSequences: list[str]
    """A list of strings giving the sequence of each seqlet.

    Shape (numSeqlets,)
    """

    # The following arrays are of shape (numSeqlets, seqletLength, 4):
    seqletOneHots: npt.NDArray[ONEHOT_T]
    """The one-hot encoded sequence.

    Shape (numSeqlets, seqletLength, 4)
    """
    seqletContribs: npt.NDArray[MOTIF_FLOAT_T]
    """The contribution scores for each base.

    Shape (numSeqlets, seqletLength, 4)
    """

    seqletSeqMatches: npt.NDArray[MOTIF_FLOAT_T]
    """The information content match for each seqlet to the pattern's pssm.

    Shape (numSeqlets, seqletLength, 4)
    """

    seqletContribMatches: npt.NDArray[MOTIF_FLOAT_T]
    """The continuous Jaccard similarity between each seqlet and the pattern's cwm

    Shape (numSeqlets,)
    """

    seqletContribMagnitudes: npt.NDArray[MOTIF_FLOAT_T]
    """The sum of the absolute value of all contribution scores for each seqlet.

    Shape (numSeqlets,)
    """

    # When you give the quantile bounds to getCutoffs, these get stored:
    cutoffSeqMatch: float
    """What is the minimal information content (i.e., pssm) match score for a hit?

    Stored when you give quantile bounds to getCutoffs
    """
    cutoffContribMatch: float
    """What is the minimal Jaccard similarity between a seqlet and the cwm for a hit?

    Stored when you give quantile bounds to getCutoffs
    """
    cutoffContribMagnitude: float
    """What is the minimum total contribution a seqlet must have to be a hit?

    Stored when you give quantile bounds to getCutoffs
    """

    seqletChroms: list[str]
    """ What chromosome is each seqlet on?

    Populated when you load in coordinates from the contribution hdf5.
    Shape (numSeqlets,)
    """

    seqletGenomicStarts: npt.NDArray[np.int32]
    """After trimming, where does the motif start?
    Populated when you load in coordinates from the contribution hdf5.
    Shape (numSeqlets,)
    """
    seqletGenomicEnds: npt.NDArray[np.int32]
    """After trimming, where does the motif end?
    Populated when you load in coordinates from the contribution hdf5.
    Shape (numSeqlets,)
    """

    def __init__(self, metaclusterName: str, patternName: str,
                 shortName: Optional[str] = None) -> None:
        self.metaclusterName = metaclusterName
        self.patternName = patternName
        if shortName is None:
            shortMName = self.metaclusterName.split("_")[0]
            shortPName = self.patternName.split("_")[1]
            self.shortName = shortMName + "_" + shortPName
        else:
            self.shortName = shortName

    def loadCwm(self, modiscoFp: h5py.File, trimThreshold: float,
                padding: int, backgroundProbs: npt.NDArray[MOTIF_FLOAT_T]) -> None:
        """Given an opened hdf5 file object, load up the contribution scores for this pattern.

        TODO MELANIE Convert to RST.

        trimThreshold and padding are used to trim the motifs, see cwmTrimPoints
        for documentation on those parameters.

        backgroundProbs gives the average frequency of each base across the genome as
        an array of shape (4,). See ppmToPwm and ppmToPssm for details on this parameter.

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

    def _callJaccard(self, seqlet: npt.NDArray[MOTIF_FLOAT_T],
                     cwm: npt.NDArray[MOTIF_FLOAT_T]) -> npt.NDArray[MOTIF_FLOAT_T]:
        return jaccard.slidingJaccard(seqlet, cwm)

    def loadSeqlets(self, modiscoFp: h5py.File) -> None:
        """Load seqlets from the modisco hdf5.

        TODO MELANIE Convert to RST.

        This function loads up all the seqlet data from the modisco file and calculates
        quantile values for information content match, contribution jaccard match,
        and contribution L1 match.
        """
        seqletsGroup = modiscoFp[self.metaclusterName][self.patternName]["seqlets"]
        self.seqletContribs = np.array(seqletsGroup["contrib_scores"], dtype=MOTIF_FLOAT_T)
        self.seqletStarts = np.array(seqletsGroup["start"])
        self.seqletEnds = np.array(seqletsGroup["end"])
        self.seqletIndexes = np.array(seqletsGroup["example_idx"])
        self.seqletRevcomps = np.array(seqletsGroup["is_revcomp"])
        self.seqletOneHots = np.array(seqletsGroup["sequence"])
        self.seqletSequences = [utils.oneHotDecode(x) for x in self.seqletOneHots]
        self.numSeqlets = int(seqletsGroup["n_seqlets"][0])
        # Now it's time to calculate stuff!
        # First, scan the seqlets and get their similarities to the called motif.
        self.seqletSeqMatches = np.zeros((self.numSeqlets,), dtype=MOTIF_FLOAT_T)
        self.seqletContribMatches = np.zeros((self.numSeqlets,), dtype=MOTIF_FLOAT_T)
        self.seqletContribMagnitudes = np.zeros((self.numSeqlets,), dtype=MOTIF_FLOAT_T)
        for i in range(self.numSeqlets):
            trimmedSeqlet = self.seqletContribs[i, self.cwmTrimLeftPoint:self.cwmTrimRightPoint]
            contribMatch, contribMagnitude = self._callJaccard(trimmedSeqlet, self.cwmTrim)
            self.seqletContribMatches[i] = float(contribMatch)
            self.seqletContribMagnitudes[i] = float(contribMagnitude)
            seqMatchScore = slidingDotproduct(self.seqletOneHots[i]
                                              [self.cwmTrimLeftPoint:self.cwmTrimRightPoint],
                                              self.pssmTrim)
            self.seqletSeqMatches[i] = float(seqMatchScore)

    def getCutoffs(self, quantileSeqMatch: float,
                   quantileContribMatch: float,
                   quantileContribMagnitude: float) -> None:
        """Calculate cutoff values given target quantiles.

        TODO MELANIE Convert to RST.

        Given the quantile values you want to use as cutoffs, actually
        look at the seqlet data and pick the right parameters. If you want to choose
        your own quantile cutoffs, set the cutoffSeqMatch, cutoffContribMatch,
        and cutoffContribMagnitude members of this object.
        """
        if quantileSeqMatch is not None:
            self.cutoffSeqMatch =\
                np.quantile(self.seqletSeqMatches, quantileSeqMatch)  # type: ignore
        else:
            self.cutoffSeqMatch = None
        if quantileContribMatch is not None:
            self.cutoffContribMatch =\
                np.quantile(self.seqletContribMatches, quantileContribMatch)  # type: ignore
        else:
            self.cutoffContribMatch = None
        if quantileContribMagnitude is not None:
            self.cutoffContribMagnitude =\
                np.quantile(self.seqletContribMagnitudes, quantileContribMagnitude)  # type: ignore
        else:
            self.cutoffContribMagnitude = None

    def seqletInfoIterator(self):
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
        for i in range(self.numSeqlets):
            ret = {}
            # TODO Once we have the ability to map back the seqlet positions, make this meaningful!
            ret["chrom"] = self.seqletChroms[i]
            # TODO ditto
            ret["start"] = self.seqletGenomicStarts[i]
            # TODO ditto
            ret["end"] = self.seqletGenomicEnds[i]
            ret["short_name"] = self.shortName
            ret["contrib_magnitude"] = self.seqletContribMagnitudes[i]
            ret["strand"] = "-" if self.seqletRevcomps[i] else "+"
            ret["metacluster_name"] = self.metaclusterName
            ret["pattern_name"] = self.patternName
            ret["sequence"] = self.seqletSequences[i]
            ret["region_index"] = self.seqletIndexes[i]
            ret["seq_match"] = self.seqletSeqMatches[i]
            ret["contrib_match"] = self.seqletContribMatches[i]
            yield ret

    def loadSeqletCoordinates(self, contribH5fp: h5py.File) -> None:
        """Break stuff.

        This method is incomplete because we haven't fixed a bug in tfmodisco-lite that allows
        us to recover the genomic coordinates of the seqlets identified by modisco.
        When it is done, it will use the seqlet indexes from the modisco output to look
        up the genomic coordinates of each seqlet and create three new fields in this object:
        seqletChroms, seqletGenomicStarts, seqletGenomicEnds.

        """
        del contribH5fp
        self.seqletChroms = ["UNDEFINED" for _ in range(self.numSeqlets)]
        self.seqletGenomicStarts = np.array([-10000 for _ in range(self.numSeqlets)])
        self.seqletGenomicEnds = np.array([-10000 for _ in range(self.numSeqlets)])

    def getScanningInfo(self) -> dict:
        """Get what you need to know to scan CWMs.

        TODO MELANIE Document dictionary shape.

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


def seqletCutoffs(modiscoH5Fname: str, contribH5Fname: str,
                  patternSpec: list[dict] | Literal["all"], quantileSeqMatch: float,
                  quantileContribMatch: float, quantileContribMagnitude: float,
                  trimThreshold: float, trimPadding: int,
                  backgroundProbs: npt.NDArray[MOTIF_FLOAT_T],
                  outputSeqletsFname: Optional[str] = None) -> list[dict]:
    """Given a modisco hdf5 file, go over the seqlets and establish the quantile boundaries.

    If you give hard cutoffs for information content and L1 norm match, this function need not
    be called.

    :param modiscoH5Name: Gives the name of the modisco output hdf5.

    :param contribH5Name: The name of the hdf5 file that was generated by interpretFlat.py.
        This file contains genomic coordinates of the seqlets, and is used to determine the
        location of each seqlet identified by modisco.
        FIXME TODO FIXME:
        Since modisco doesn't preserve seqlet indexes right now, this parameter is ignored.
        All seqlets will be reported as being on chromosome chr1 and will have a meaningless
        position.

    :param patternSpec: Either a list of dicts, or the string "all". See makePatternObjects for how
        this parameter is interpreted.

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
    :param trimBuffer: Gives the padding to be added to the flanks.
        See :func:`bpreveal.motifUtils.cwmTrimPoints` for details of these parameters.

    :param backgroundProbs: An array of shape (4,) of floats that gives the background
        distribution for each base in the genome. See :func:`ppmToPwm` and
        :func:`pwmToPssm` for details on this argument.

    :param outputSeqletsFname: (Optional) Gives a name for a file where the all of the
        seqlets in the Modisco output should be saved as a tsv file.

    :return: A list of dicts that will be needed by the cwm scanning utility.

    The returned list is structured as follows::
        [{"metacluster-name": <string>,
        "pattern-name": <string>,
        "cwm": <array of floats of shape (length, 4)>,
        "pssm": <array of floats of shape (length, 4)>,
        "seq-match-cutoff": <float-or-null>,
        "contrib-match-cutoff": <float-or-null>,
        "contrib-magnitude-cutoff": <float-or-null> },
        ...
        ]

    This dictionary can be saved as a json for use with the motif scanning tool, or passed
    directly to the motif scanning Python functions.

    """
    patterns = makePatternObjects(patternSpec, modiscoH5Fname)
    logUtils.info("Initialized patterns, beginning to load data.")
    with h5py.File(modiscoH5Fname, "r") as modiscoFp:
        for pattern in patterns:
            pattern.loadCwm(modiscoFp, trimThreshold, trimPadding, backgroundProbs)
            pattern.loadSeqlets(modiscoFp)
            pattern.getCutoffs(quantileSeqMatch, quantileContribMatch, quantileContribMagnitude)
    logUtils.info("Loaded and analyzed seqlet data.")
    if outputSeqletsFname is not None:
        # We should load up the genomic coordinates of the seqlets.
        logUtils.info("Writing tsv of seqlet information.")
        with h5py.File(contribH5Fname, "r") as contribH5fp:
            for pattern in patterns:
                pattern.loadSeqletCoordinates(contribH5fp)
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


def makePatternObjects(patternSpec: list[dict] | str, modiscoH5Fname: str) -> list[Pattern]:
    """Get a list of patterns to scan.

    :param patternspec: The pattern specs from the config json.
    :type patternspec: list[dict] | "all"
    :param modiscoH5Fname: The name of the hdf5-format file generated by MoDISco.

    If patternSpec is a list, it may have one of two forms::

        [ {"metacluster-name" : <string>,
        "pattern-name" : <string>,
        ??"short-name" : <string> ??
        },
        ...
        ]

    where each entry gives one metacluster with ONE pattern,
    or it may give multiple patterns as a list::

        [ {"metacluster-name" : <string>,
        "pattern-names" : [<string>, ...],
        ??"short-names" : [<string>,...]??
        },
        ...
        ]

    (Note the 's' in pattern-names, as opposed to the singular pattern-name above.)
    short-name, or short-names, is an optional parameter. If given, then instead of the
    patterns being named "pos_2", they will have the name you give, which could be something
    much more human-readable, like "Sox2".

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
                    patterns.append(Pattern(metaclusterName, patternName))
    else:
        patternSpecList: list[dict] = patternSpec  # type: ignore
        for metaclusterSpec in patternSpecList:
            logUtils.debug("Initializing patterns {0:s}".format(str(metaclusterSpec)))
            if "pattern-names" in metaclusterSpec:
                for i, patternName in enumerate(metaclusterSpec["pattern-names"]):
                    if "short-names" in metaclusterSpec:
                        shortName = metaclusterSpec["short-names"][i]
                    else:
                        shortName = None
                    patterns.append(Pattern(metaclusterSpec["metacluster-name"],
                                            patternName, shortName=shortName))
            else:
                if "short-name" in metaclusterSpec:
                    shortName = metaclusterSpec["short-name"]
                else:
                    shortName = None
                patterns.append(Pattern(metaclusterSpec["metacluster-name"],
                                        metaclusterSpec["pattern-name"],
                                        shortName))
    logUtils.info("Initialized {0:d} patterns.".format(len(patterns)))
    return patterns


class MiniPattern:
    """A smaller pattern object for the scanners to use.

    TODO MELANIE Document constructor parameters.

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

    def _callJaccard(self, scores: npt.NDArray[MOTIF_FLOAT_T],
                     cwm: npt.NDArray[MOTIF_FLOAT_T]) -> npt.NDArray[MOTIF_FLOAT_T]:
        # This is just a separate function so the profiler can see the call to Jaccard.
        return jaccard.slidingJaccard(scores, cwm)

    def _scanOneWay(self, sequence: ONEHOT_AR_T,
                    scores: npt.NDArray[MOTIF_FLOAT_T], cwm: npt.NDArray[MOTIF_FLOAT_T],
                    pssm: npt.NDArray[MOTIF_FLOAT_T], strand: Literal["+", "-"]
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

    def scan(self, sequence: ONEHOT_AR_T, scores: npt.NDArray[MOTIF_FLOAT_T]
             ) -> list[tuple[int, Literal["+", "-"], float, float, float]]:
        """Given a sequence and a contribution track, identify places where this pattern matches.

        sequence is a (length, 4) one-hot encoded
        DNA fragment, and scores is the actual (not hypothetical) contribution score
        data from that region. Scores is (length, 4) in shape, but all of the
        bases that are not present should have zero contribution score.
        Returns a list of hits. A hit is a tuple with five elements.
        First, an integer giving the offset in the sequence where the hit starts.
        Second, the strand on which the hit was found.
        Third, the contribution magnitude of the pattern at that position.
        Fourth, the contribution match score of the importance scores at that position.
        Finally, the sequence match score against the pattern at that position.

        TODO MELANIE Convert to RST.
        """
        hitsPos = self._scanOneWay(sequence, scores, self.cwm, self.pssm, "+")
        hitsNeg = self._scanOneWay(sequence, scores, self.rcwm, self.rpssm, "-")
        return hitsPos + hitsNeg


class Hit:
    """A small struct used to bundle up found hits for insertion in the pipe to the writer thread.

    TODO MELANIE Document initializer.
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

        TODO MELANIE Document returned dictionary.
        """
        ret = {}
        ret["chrom"] = self.chrom
        ret["start"] = self.start
        ret["end"] = self.end
        ret["short_name"] = self.shortName
        ret["contrib_magnitude"] = self.contribMagnitude
        ret["strand"] = self.strand
        ret["sequence"] = self.sequence
        ret["region_index"] = self.index
        ret["metacluster_name"] = self.metaclusterName
        ret["pattern_name"] = self.patternName
        ret["seq_match"] = self.seqMatchScore
        ret["contrib_match"] = self.contribMatchScore
        return ret


class PatternScanner:
    """TODO MELANIE Document initializer."""

    def __init__(self, hitQueue: multiprocessing.Queue, contribFname: str,
                 patternConfig: dict) -> None:
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
        logUtils.logFirstN(logUtils.DEBUG,
            "Got first index for thread {0:d}".format(
                multiprocessing.current_process().pid),
            1)

        # Get all the data for this index.
        chromIdx = self.contribFp["coords_chrom"][idx]
        if isinstance(chromIdx, bytes):
            logUtils.logFirstN(logUtils.WARNING,
                               "Detected an importance score file from before version 4.0. "
                               "This will be an error in BPReveal 5.0. "
                               "Instructions for updating: Re-calculate importance scores.",
                               1)
            chrom = chromIdx.decode("utf-8")
        else:
            chrom = self.chromIdxToName[chromIdx]
            logUtils.logFirstN(logUtils.DEBUG,
                               "First index, found chrom {0:s}".format(str(chrom)), 1)
        regionStart = self.contribFp["coords_start"][idx]
        oneHotSequence = np.array(self.contribFp["input_seqs"][idx])
        hypScores = np.array(self.contribFp["hyp_scores"][idx], dtype=MOTIF_FLOAT_T, order="C")
        contribScores = np.array(hypScores * oneHotSequence, dtype=MOTIF_FLOAT_T, order="C")
        # Now we perform the scanning.
        for pattern in self.miniPatterns:
            logUtils.logFirstN(logUtils.DEBUG,
                               "Scanning pattern {0:s}".format(pattern.shortName), 1)

            hits = pattern.scan(oneHotSequence, contribScores)
            logUtils.logFirstN(logUtils.DEBUG,
                               "Completed scanning {0:s}".format(pattern.shortName), 1)
            if len(hits) > 0:
                # Hey, we found something!
                for hit in hits:
                    # ret.append((passLoc, strand, L1Score, jaccardScore, icScore))
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
                    self.hitQueue.put(madeHit, timeout=QUEUE_TIMEOUT)
                    logUtils.logFirstN(logUtils.DEBUG,
                                       "First hit from thread {0:d}".format(
                                           multiprocessing.current_process().pid),
                                       1)
        logUtils.logFirstN(logUtils.DEBUG, "First pattern complete.", 1)

    def done(self) -> None:
        """TODO MELANIE Document"""
        self.contribFp.close()
        # Send a -1 as a signal that a thread has finished up.
        self.hitQueue.put(-1, timeout=QUEUE_TIMEOUT)


def scannerThread(queryQueue: multiprocessing.Queue, hitQueue: multiprocessing.Queue,
                  contribFname: str, patternConfig: dict) -> None:
    """This is the thread for one scanner.

    Each scanner is looking for every pattern,
    and gets regions to scan from queryQueue. Every time it finds a hit in one of the
    queries it pulled, it stuffs those Hit objects into hitQueue.
    contribFname is a string naming the hdf5-format file generated by interpretFlat.py.
    patternConfig is the dictionary/json generated by the quantile script. It contains
    a cwm, pssm, and cutoff values for each pattern you want to scan for.
    TODO MELANIE Convert to RST
    """
    scanner = PatternScanner(hitQueue, contribFname, patternConfig)
    logUtils.debug("Started scanner {0:d}".format(multiprocessing.current_process().pid))
    while True:
        curQuery = queryQueue.get(timeout=QUEUE_TIMEOUT)
        if curQuery == -1:
            # End of the line, finish up!
            logUtils.debug("Done with scanner {0:d}".format(multiprocessing.current_process().pid))
            scanner.done()
            break
        scanner.scanIndex(curQuery)


def writerThread(hitQueue: multiprocessing.Queue, scannerThreads: int, tsvFname: str) -> None:
    """This thread runs concurrently with however many scanners you're using.

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
                ret = hitQueue.get(timeout=QUEUE_TIMEOUT)
                logUtils.logFirstN(logUtils.DEBUG,
                                   "Writer got first hit, {0:s}".format(str(ret)),
                                   1)
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
                logUtils.debug("Writer got thread done message, "
                              "{0:d} remain.".format(threadsRemaining))
                if threadsRemaining <= 0:
                    break
                continue
            writer.writerow(ret.toDict())


def scanPatterns(contribH5Fname: str, patternConfig: dict, tsvFname: str, numThreads: int) -> None:
    """ContribH5Fname is the name of a contribution score file generated by interpretFlat.py

    patternConfig is a dictionary/json generated by the quantile script that contains the necessary
    information to scan for the patterns.
    tsvFname is the name of the output file containing the hits. Columns 1-6 of this file can
    be extracted with cut to get a bed file.
    numThreads is the *total* number of threads to use for the scanning. This must be at least
    three, since this function builds a pipeline with one thread generating queries,
    one saving results, and all the rest furiously scanning the query sequences for matches.
    I suggest running rampant with as many threads as they'll let you use, the scanning is very
    computationally expensive!
    TODO MELANIE Document
    """
    assert numThreads >= 3, "Scanning requires at least three threads. " \
                            "(but works great with fifty!)"
    # A queue size of 1024 is reasonable.
    queryQueue = multiprocessing.Queue(1024)
    hitQueue = multiprocessing.Queue(1024)
    logUtils.debug("Queues built. Starting threads.")
    scannerProcesses = []
    for i in range(numThreads - 2):
        scanProc = multiprocessing.Process(target=scannerThread,
                                           args=[queryQueue, hitQueue, contribH5Fname,
                                                 patternConfig], daemon=True)
        scannerProcesses.append(scanProc)
    writeProc = multiprocessing.Process(target=writerThread,
                                        args=[hitQueue, numThreads - 2, tsvFname],
                                        daemon=True)
    logUtils.info("Starting threads.")
    [x.start() for x in scannerProcesses]  # pylint: disable=expression-not-assigned

    writeProc.start()
    with h5py.File(contribH5Fname, "r") as fp:
        # How many queries do I need to stuff down the query queue?
        numRegions = np.array(fp["coords_start"]).shape[0]
    logUtils.debug("Starting to send queries to the processes.")

    for i in wrapTqdm(range(numRegions)):
        # The queries are generated by the main thread.
        queryQueue.put(i, timeout=QUEUE_TIMEOUT)
    logUtils.debug("Done adding queries to the processes. Waiting for scanners to finish.")
    for i in range(numThreads - 2):
        queryQueue.put(-1, timeout=QUEUE_TIMEOUT)

    [x.join() for x in scannerProcesses]  # pylint: disable=expression-not-assigned
    writeProc.join()
    logUtils.info("Done scanning.")
