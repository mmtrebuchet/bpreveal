import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import random
from typing import List, Tuple, TypeAlias, Callable, Optional
import numpy.typing as npt
import bpreveal.utils as utils
from bpreveal.utils import PRED_AR_T
import numpy as np
import matplotlib.pyplot as plt
# A few variables that you can use to get the accented letters Ǎ, Č, Ǧ, and Ť
# in case they aren't easily typeable on your keyboard:
IN_A = "Ǎ"
IN_C = "Č"
IN_G = "Ǧ"
IN_T = "Ť"
IN_L = "ǍČǦŤ"
IN_D = {"A": "Ǎ", "C": "Č", "G": "Ǧ", "T": "Ť"}

# Use these to map corruptors to integers.
CORRUPTOR_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3,
                    "Ǎ": 4, "Č": 5, "Ǧ": 6, "Ť": 7,
                    "d": 8}
# Given an integer, which corruptor does it represent?
# This is the inverse of CORRUPTOR_TO_IDX.
IDX_TO_CORRUPTOR = "ACGTǍČǦŤd"

Corruptor: TypeAlias = tuple[int, str]
Profile: TypeAlias = List[Tuple[PRED_AR_T, float]]

_seqCmap = {"A": (0, 158, 115), "C": (0, 114, 178), "G": (240, 228, 66), "T": (213, 94, 0)}
for k in "ACGT":
    _seqCmap[k] = tuple((x / 255 for x in _seqCmap[k]))
corruptorColors = {"A": _seqCmap["A"], "C": _seqCmap["C"], "G": _seqCmap["G"], "T": _seqCmap["T"],
                   "d": "black",
                   "Ǎ": _seqCmap["A"], "Č": _seqCmap["C"], "Ǧ": _seqCmap["G"], "Ť": _seqCmap["T"]}


def corruptorsToArray(corruptorList: List[Corruptor]) -> List[Tuple[int, int]]:
    """Given a list of corruptor tuples, like [(1354, 'C'), (1514, 'Ť'), (1693, 'd')],
    convert that to an integer list that can be easily saved. This list will have the format
    [ [1354, 1], [1514, 7], [1693, 8] ]
    where the second number indicates the type of corruptor, as given by CORRUPTOR_TO_IDX.
    Returns a list of lists, not a numpy array!
    If you want an array, use `np.array(corruptorsToArray(myCorruptors))`"""
    ret = []
    for corruptor in corruptorList:
        ret.append([corruptor[0], CORRUPTOR_TO_IDX[corruptor[1]]])
    return ret


def arrayToCorruptors(corruptorArray: List[Tuple[int, int]]) -> List[Corruptor]:
    """The inverse of corruptorsToArray, takes an array of numerical corruptors in the form
    [ [1354, 1], [1514, 7], [1693, 8] ]
    and generates the canonical list with letters, as used in the rest of the code.
    This function will work on a list or a numpy array of shape (N x 2).
    Returns a list of tuples, one for each position that was corrupted, like:
    [(1354, 'C'), (1514, 'Ť'), (1693, 'd')]
    """
    ret = []
    for corruptorPair in corruptorArray:
        pos = int(corruptorPair[0])
        idx = int(corruptorPair[1])
        ret.append((pos, IDX_TO_CORRUPTOR[idx]))
    return ret


def stringToCorruptorList(corruptorStr: str) -> List[Corruptor]:
    """Takes a string representing a list of corruptors and generates
    the actual list as a python object. For example, if the string is
        "[(1354, 'C'), (1514, 'Ť'), (1693, 'd')]"
    this function will parse it to the list of tuples
        [(1354, 'C'), (1514, 'Ť'), (1693, 'd')]

    (The inverse of this function is simply `str` on a corruptor list.)
    """
    import ast
    ret = []
    tree = ast.parse(corruptorStr)
    elems = tree.body[0].value.elts  # type: ignore
    for e in elems:
        pos = e.elts[0].value  # type: ignore
        cor = e.elts[1].value  # type: ignore
        ret.append((pos, cor))
    return ret


class Organism:
    """This represents the set of corruptors that are to be applied to the
    input sequence."""
    profile: Profile
    score: float

    def __init__(self, corruptors: List[Corruptor]) -> None:
        """Construct an organism with the given corruptors.
        """

        self.corruptors = sorted(corruptors)
        assert validCorruptorList(self.corruptors), "Invalid corruptors in constructor."

    def getSequence(self, initialSequence: str, inputLength: int) -> str:
        """Applies this organism's corruptors to initialSequence, a string.
        and then returns the first inputLength bases of the result.
        Note that initialSequence will need to be longer than the inputLength
        to your model, since deletion corruptors can shorten the sequence.
        The length of initialSequence must be at least
        inputLength + maxDeletions,
        and unless you're filtering somehow, the maximum number of deletions
        is the number of corruptors for this organism.
        """

        seq = []
        readHead = 0  # The position in the input sequence where the
        # next base should be taken from.

        writeHead = 0  # Our current position in the output sequence.
        # Note that writeHead will be close, but not exactly the same,
        # as readHead. If we encounter a deletion, readHead will advance,
        # but writeHead will stay the same. If we encounter
        # an insertion, writeHead will advance twice, but readHead only once.

        for c in self.corruptors:
            if readHead < c[0]:
                seq.append(initialSequence[readHead:c[0]])
                # We added a bunch of bases to the output.
                writeHead += c[0] - readHead
                readHead = c[0]
            if c[1] in "ACGT":  # We have a SNP.
                seq.append(c[1])
                readHead += 1
                # Since the behavior of insertion depends on if I just had a SNP,
                # store the location of the last SNP.
                writeHead = c[0]
            elif c[1] in "ǍČǦŤ":  # Insertion.
                # Put in the initial base unless we just had a SNP.
                if writeHead < c[0]:
                    seq.append(initialSequence[readHead])
                    readHead += 1
                # And now put in the insertion.
                seq.append({"Ǎ": "A", "Č": "C", "Ǧ": "G", "Ť": "T"}[c[1]])
                # No increase in readHead, but set writeHead so I don't copy
                # over a base if I have two subsequent insertions.
                writeHead = c[0]
            elif c[1] == "d":
                # Nothing to do here. A deleted base can never have
                # an insertion or SNP, so no need to check.
                readHead += 1
        # Done applying corruptors.
        fullSequence = "".join(seq) + initialSequence[readHead:]
        return fullSequence[:inputLength]

    def setScore(self, scoreFn: Callable[[Profile, List[Corruptor]], float]) -> None:
        assert self.profile is not None, \
            "Attempting to score organism with no profile."
        self.score = scoreFn(self.profile, self.corruptors)

    def __eq__(self, other: 'Organism') -> bool:
        """Returns True if this organism has the same corruptors
        as the other. Does *not* check that the profiles or score
        are identical!
        """
        return self.cmp(other) == 0

    def __hash__(self) -> int:
        """Returns an integer representing a hash of this organism's
        corruptors. This means that you can use organisms as a key
        in a dictionary, for example.
        Note that profile and score are not integrated in this hash!"""
        return str(self.corruptors).__hash__()

    def cmp(self, other: 'Organism') -> int:
        """A general comparator between two organisms based on their
        corruptors.
        Returns 1 if this organism is after (alphabetically) the other
        one, returns -1 if this organism comes first, and returns 0
        if the two organisms have the same corruptors.
        Note that this does not compare profile or score!"""

        for i in range(min(len(self.corruptors),
                           len(other.corruptors))):
            mine = self.corruptors[i]
            them = other.corruptors[i]
            if mine[0] < them[0]:
                # My corruptor comes first.
                return -1
            elif mine[0] > them[0]:
                # My corruptor comes second.
                return 1
            else:
                if mine[1] < them[1]:
                    # My letter is earlier.
                    return -1
                elif mine[1] > them[1]:
                    # My letter is later.
                    return 1
        if len(self.corruptors) > len(other.corruptors):
            return 1
        elif len(self.corruptors) < len(other.corruptors):
            return -1
        # We have identical corruptors, we are the same organism.
        return 0

    def mutated(self, allowedCorruptors: List[Corruptor],
                checkCorruptors: Callable[[List[Corruptor]], bool]) -> 'Organism':
        """This is something you may want to override in a subclass.
        This returns a NEW organism that has one changed corruptor.
        It chooses one of its current corruptors at random, and then changes
        it to a random selection from allowedCorruptors. (allowedCorruptors
        has the same structure as in a Population).
        It makes sure that the new corruptor doesn't occur on the same base as
        an existing one (unless it's an insertion), and also calls
        checkCorruptors on the resulting corruptor candidates. It will make
        100 attempts to generate a new organism, and then it will error out. """

        posToCorrupt = random.randrange(0, len(self.corruptors))
        keepCorruptors = []
        for i, c in enumerate(self.corruptors):
            if i != posToCorrupt:
                keepCorruptors.append(c)
        found = False
        numTries = 0
        while not found:
            numTries += 1
            assert numTries < 100, "WARNING! Took 100 tries on organism " +\
                                   str(self.corruptors)
            newCorBase = random.choice(allowedCorruptors)
            newCor = (newCorBase[0], random.choice(newCorBase[1]))
            candidateCorruptors = sorted(keepCorruptors + [newCor])
            if validCorruptorList(candidateCorruptors):
                found = checkCorruptors(candidateCorruptors)
        return Organism(candidateCorruptors)  # type: ignore

    def mixed(self, other: 'Organism',
              checkCorruptors: Callable[[List[Corruptor]], bool]) -> 'Organism':
        """This is also something you may wish to override in a subclass.
        Currently, it pools the corruptors from self and other, and then
        randomly selects numCorruptors of them. If that passes checkCorruptors,
        then it returns a NEW organism. """
        fullCorruptorPool = sorted(self.corruptors + other.corruptors)
        corruptorPool = [fullCorruptorPool[0]]
        # If there are any duplicated bases, choose one at random.
        for c in fullCorruptorPool[1:]:
            if corruptorPool[-1][0] == c[0]:
                # We have a collision. How do we resolve this?
                newIsSnpDel = c[1] in "ACGTd"
                # We know that the pool is sorted. So insertions will always
                # come after SNPs and deletions.
                # All SNPs and deletions are mutually exclusive.
                # So if the new candidate is either a SNP or a
                # deletion, choose one at random.
                if newIsSnpDel:
                    corruptorPool[-1] = random.choice((c, corruptorPool[-1]))
                else:
                    # The new thing is an insertion.
                    # In this case, the only contention is with deletion.
                    # If we have SNP then insertion, that's no problem
                    # but if we have deletion then insertion, that's
                    # equivalent to a SNP. So just pick one.
                    if corruptorPool[-1][1] == "d":
                        corruptorPool[-1] = random.choice((c, corruptorPool[-1]))
                    else:
                        # Keep the insertion, it's valid.
                        corruptorPool.append(c)
            else:
                corruptorPool.append(c)

        for _ in range(100):
            chosens = sorted(random.sample(corruptorPool, len(self.corruptors)))
            # We know that the full set of corruptorPool forms a valid
            # corruption set, so no need to do any pruning here since any subset
            # of valid corruptors is itself valid.
            if checkCorruptors(chosens):
                assert validCorruptorList(chosens), "Mixing gave invalid organism."
                return Organism(chosens)
        assert False, "Took over 100 attempts to mix organisms" + \
            str(self.corruptors) + str(other.corruptors)


class Population:
    """This is the main class for running the sequence optimization GA."""

    def __init__(self, initialSequence: str, inputLength: int, populationSize: int,
                 numCorruptors: int, allowedCorruptors: List[Corruptor],
                 checkCorruptors: Callable[[List[Corruptor]], bool],
                 fitnessFn: Callable[[Profile, List[Corruptor]], float],
                 numSurvivingParents: int, predictor: utils.BatchPredictor):
        """This is a heck of a constructor, but you need to make a
        lot of choices to use the GA.

        initialSequence is a string of length (input-length + numMutations)
        for your model. The extra length is needed because this GA can have deletions
        and you need sequence to fill in the gaps when that happens. If you limit
        the number of deletions allowed (say, by not having them in allowedCorruptors
        or restricting them through checkCorruptors), then you need to provide
        (input-length + numPossibleDeletions), and if you're only allowing SNPs,
        then numPossibleDeletions = 0.

        inputLength is the length of sequence that is given to your model.

        populationSize is the number of organisms at the end of each generation.

        numCorruptors determines how many corruptors will be applied
        in each organism.

        allowedCorruptors is a list of tuples. Each tuple contains two elements:
        a number, representing the position in the input sequence (starting at
        zero), and the second is a string containing the allowed corruptions at
        the base at the positing given by the number. For example, if your
        sequence is AGGCA, and you want any base to be corruptor to any other
        except that the C cannot be corrupted to a T and the second G cannot
        be corrupted at all, allowedCorruptors would be
        [(0, "CGT"), (1, "ACT"), (3, "AG"), (4, "CGT")].
        In the string of possible corruptors, the capital letters A, C, G, and T
        refer to a SNP of the base at the given position to the new letter,
        a lowercase 'd' means that the base there should be deleted, and a
        letter with a caron, like Ǎ, Č, Ǧ, or Ť, means that the given letter
        should be inserted AFTER the base number.

        checkCorruptors is a function that determines if a list of corruptors
        is valid. It will take a list of tuples, the first element of each
        tuple is a number, representing the base to be corrupted, and the second
        will be a single character, representing the corruption to be applied
        The corruptor locations will always be sorted in increasing order.
        You could use this, for example, to make sure that no two corruptors
        are next to each other. If you don't wish to apply any logic to the
        corruptors, pass in lambda x: True.

        fitnessFn is the fitness function. It takes two arguments:
        The first argument is a list of tuples containing the model outputs.
        These have the same organization as the outputs from batchPredictor.
        The second argument is a list of corruptors, as presented to checkCorruptors.

        numSurvivingParents is the number of parents that will be kept for the next
        generation, usually referred to as elitism in GA terminology.

        predictor is a BatchPredictor that has been set up with the model you
        want to use.
        """
        self.initialSequence = initialSequence
        self.inputLength = inputLength
        self.populationSize = populationSize
        self.numCorruptors = numCorruptors
        self.allowedCorruptors = allowedCorruptors
        self.checkCorruptors = checkCorruptors
        self.fitnessFn = fitnessFn
        self.numSurvivingParents = numSurvivingParents
        self.organisms = []
        self.predictor = predictor
        self._seed()

    def _seed(self) -> None:
        for _ in range(self.populationSize):
            add = False
            numTries = 0
            while not add:
                numTries += 1
                assert numTries < 100, "Too many attempts to create a new organism."
                candidate = self._newOrganism()
                add = True
                # Make sure I haven't already created an identical organism.
                for other in self.organisms:
                    if other == candidate:
                        add = False
                        break
            # We've created a new organism that is ready to go!
            self.organisms.append(candidate)  # type: ignore

    def _newOrganism(self) -> Organism:
        for _ in range(100):
            corLocations = random.sample(self.allowedCorruptors, self.numCorruptors)
            # Since sample is without replacement, I'm guaranteed to not have corruptors in
            # places where I've already put a corruptor.
            # This means that if I'm to have a SNP and insertion at the same place,
            # it has to arise through the evolution, no organism starts that way.
            cors = [(x[0], random.choice(x[1])) for x in corLocations]

            if self.checkCorruptors(sorted(cors)):
                return Organism(cors)
        assert False, "Over 100 attempts to choose corruptors for new organism."

    def runCalculation(self) -> None:
        """Runs the current population through the model, assigns scores
        to each organism, and sorts the organisms by score."""

        for i, organism in enumerate(self.organisms):
            self.predictor.submitString(
                organism.getSequence(self.initialSequence, self.inputLength),
                i)

        for i in range(self.populationSize):
            ret = self.predictor.getOutput()
            self.organisms[ret[1]].profile = ret[0]
            self.organisms[ret[1]].setScore(self.fitnessFn)

        self.organisms.sort(key=lambda x: x.score)

    def _choose1(self) -> Organism:
        """Randomly choose one of the organisms in the population.
        This only makes sense once you've runCalculation(), since it needs
        to know which organisms are good. You may want to override this in
        a subclass to change the selection operator."""
        toChoose = int(random.triangular(0,
                                         self.populationSize,
                                         self.populationSize))
        if toChoose == self.populationSize:
            # In the (extremely unlikely) event that the distribution chooses
            # its maximal value, decrease it by one.
            toChoose = self.populationSize - 1
        return self.organisms[toChoose]

    def _choose2(self) -> Tuple[Organism, Organism]:
        """Randomly choose two of the organisms in the population.
        This guarantees that the two organisms will be different.
        You may want to override this in a subclass to change the
        selection operator."""
        firstChoice = int(random.triangular(0,
                                            self.populationSize,
                                            self.populationSize))
        if firstChoice == self.populationSize:
            firstChoice = firstChoice - 1
        secondChoice = firstChoice
        while secondChoice == firstChoice:
            secondChoice = int(random.triangular(0,
                                                 self.populationSize,
                                                 self.populationSize))
            if secondChoice == self.populationSize:
                secondChoice = secondChoice - 1
        return self.organisms[firstChoice], self.organisms[secondChoice]

    def nextGeneration(self) -> None:
        """Once you've runCalculation(), you call this to create new organisms
        for the next generation. This replaces the .organisms array with the new
        children. If you want to save a list of the best parents, remember that
        the organisms are sorted in *ascending* order of fitness, so the best
        organism is pop.organisms[-1]."""

        # Keep the best parents (elitism)
        ret = set(self.organisms[-self.numSurvivingParents:])
        # Note that ret is a SET, and not a list. sets in Python have the nice
        # property that adding an item that's already in the set does not
        # change its length. So I blindly keep adding new organisms to the set,
        # allowing for duplicates, and this loop keeps doing this until the
        # whole population is the desired size.
        numTries = 0
        while len(ret) < self.populationSize:
            numTries += 1
            assert numTries < self.populationSize * 2, "Too many iterations building generation."
            if random.randrange(2):
                # Do mixing.
                parents = self._choose2()
                child = parents[0].mixed(parents[1], self.checkCorruptors)
                if not child:
                    continue
            else:
                # Do mutation
                parent = self._choose1()
                child = parent.mutated(self.allowedCorruptors, self.checkCorruptors)
                # mutated can return false, in that case no valid child could be generated.
                if not child:
                    continue
            ret.add(child)
        self.organisms = list(ret)


def getCandidateCorruptorList(sequence: str, regions: Optional[List[Tuple[int, int]]] = None,
                              allowDeletion: bool = True,
                              allowInsertion: bool = True) -> List[Corruptor]:
    """Given a sequence (a string), this generates a list of tuples
    that contain all of the possible corruptors for each position in the
    sequence. It will have the format
    [(0, "ACGdǍČǦŤ"), (1, "AGTdǍČǦŤ"), (2, "CGTdǍČǦŤ"), ...]
    where each number is a position in the sequence and the letters are the things
    that can be done at that location. A regular letter means a SNP, a d means deletion
    and the accented letters mean an insertion *after* the position.

    regions, if provided, must be a list of tuples.
    Each tuple contains two numbers, giving a start and stop location
    (left inclusive, right-exclusive, like python array slicing)
    of bases that are corruptor candidates. For example,
    regions=[(100,200), (250,300)] would return
    [(100, "ACG"), (101, "AGT"), ... (199, "CGT"), (250, "ACT"), ... (299,"ACG")]
    The start position in each region tuple must be less than the stop point,
    but this method will check for (and remove) overlapping regions.
    If allowDeletion is True, then the strings will contain 'd', and if
    allowInsertion is True, then the strings will contain 'ǍČǦŤ'.
    """
    if regions is None:
        regions = [(0, len(sequence))]
    candidateCorruptors = []
    indel = ""
    if allowDeletion:
        indel = "d"
    if allowInsertion:
        indel += "ǍČǦŤ"
    removedLetters = {"A": "CGT" + indel,
                      "C": "AGT" + indel,
                      "G": "ACT" + indel,
                      "T": "ACG" + indel}
    for region in regions:
        for basePos in range(region[0], region[1]):
            candidateCorruptors.append((basePos, removedLetters[sequence[basePos]]))
    # Now I've assembled all the candidates. Check for uniqueness.
    sortedCors = sorted(candidateCorruptors)
    ret = [sortedCors[0]]
    for sc in sortedCors[1:]:
        if sc[0] > ret[-1][0]:
            ret.append(sc)
    return ret


def anyCorruptorsCloserThan(corList: List[Corruptor], distance: int) -> bool:
    """A utility function that you can integrate into checkCorruptors
    to see if any corruptors are close to each other. Given a sorted
    list of corruptors (each corruptor a tuple of (position, effect)),
    returns True if there exist two positions that are distance
    apart or less.
    For example, to prevent corruptors on adjacent bases, distance=1.
    To ensure a gap of two bases between each corruptor, distance=2.
    """
    curPos = corList[0][0]
    for c in corList[1:]:
        if c[0] <= curPos + distance:
            return True
        curPos = c[0]
    return False


def removeCorruptors(corruptorList: List[Corruptor],
                     corsToRemove: List[Corruptor]) -> List[Corruptor]:
    """Given a candidate corruptor list, like from getCandidateCorruptorList, and
    a list of tuples giving disallowed corruptors, return a new candidate corruptor
    list where those disallowed corruptors are removed.
    corsToRemove is a list of tuples. The first element of each tuple is the position
    (relative to the start of the input sequence) that needs a corruptor removed, and
    the second is a string containing the forbidden letters.
    For example,
    removeCorruptors([(100, "ACG"), (101, "AT"), (102, "CGT"), (103, "CT"), (104, "AGT")],
                    [(101, "T"), (103, "CT"), (104, "G")])
    -> [(100, "ACG"), (101, "A"), (102, "CGT"), (104, "AT")]
    """
    def removeLetters(original, forbidden):
        return "".join([x for x in original if (x not in forbidden)])
    ret = []
    for c in corruptorList:
        newLetters = c[1]
        for removal in corsToRemove:
            if c[0] == removal[0]:
                newLetters = removeLetters(newLetters, removal[1])
        if len(c[1]):  # There are still letters we can remove here.
            ret.append((c[0], newLetters))
    return ret


def validCorruptorList(corruptorList: List[Corruptor]) -> bool:
    """ A utility to make sure that a list of corruptors
    is fundamentally sound. A valid list satisfies the
    following properties:
    For each pair (c0, c1) in corruptorList:
    c0[0] <= c1[0] (The list is ordered)
    If c0[0] == c1[0],
        c0[1] and c1[1] must be in sorted order.
        Neither c0[1] nor c1[1] are "d"
            (You can't delete a base and do anything else to it.)
        If c0[1] in "ACGT", then c1[1] in "ǍČǦŤ"
            (If you have a SNP, the only other thing
             at that position must be an insertion.)"""
    if len(corruptorList) == 0:
        # The wild-type organism is valid by definition.
        return True
    prev = corruptorList[0]
    for c in corruptorList[1:]:
        if prev[0] > c[0]:
            # Corruptors out of order.
            return False
        if prev[0] == c[0]:
            # Overlapping corruptors.
            if (prev[1] == "d" or c[1] == "d"):
                # A deletion and something else. Not allowed.
                return False
            if ord(prev[1]) > ord(c[1]):
                # The letters are not in order.
                return False
            if (prev[1] in "ACGT" and c[1] not in "ǍČǦŤ"):
                # We have a SNP and something that is not an insertion.
                return False
        prev = c
    return True


def plotTraces(posTraces: List[Tuple[PRED_AR_T, str, str]],
               negTraces: List[Tuple[PRED_AR_T, str, str]],
               xvals: npt.NDArray[np.float32],
               annotations: List[Tuple[Tuple[int, int], str, str]],
               corruptors: List[Corruptor],
               ax: plt.Axes) -> None:
    """Generate a nice little plot including pips for corruptors and boxes for
    annotations.
    posTraces is a list of tuples. Each tuple has three things:
        1. A one-dimensional array of values. The number of values must be the
            same as the number of points in xvals.
        2. A string that will be used as a label for the trace.
        3. A string that will be used for the color of the trace.
    negTraces has the same structure as posTraces, but will be negated before
    being plotted. This is handy to make different conditions visually
    distinct, and of course for chip-nexus data.
    xvals is a one-dimensional array with the x coordinates that will be used
    for the plot. These will usually be genomic coordinates.
    annotations is a list of tuples. Each tuple contains a pair of integers,
    giving the start and stop points of that annotation, a string giving its
    label, and a string giving its color. For example,
    [((431075,431089), "FKH2", "red"), ((431200, 431206), "PHO4", "blue")]
    corruptors has the same format as the corruptors in an organism, but be
    sure you shift the coordinates appropriately so that they line up with
    xvals. (Remember that corruptor coordinates are relative to the start of
    the INPUT, not the output.)
    ax is a matplotlib axes object upon which the plot will be drawn.
    """
    maxesPos = [max(x[0]) for x in posTraces]
    # The negative traces aren't negated yet, so use max.
    maxesNeg = [max(x[0]) for x in negTraces]
    boxHeight = max(maxesPos + maxesNeg) / 20
    for posTrace in posTraces:
        ax.plot(xvals, posTrace[0] + boxHeight, label=posTrace[1], color=posTrace[2])
    for negTrace in negTraces:
        ax.plot(xvals, -negTrace[0] - boxHeight, label=negTrace[1], color=negTrace[2])
    for annot in annotations:
        l, r = annot[0]
        h = boxHeight  # Just for brevity.
        ax.fill([l, l, r, r], [-h, h, h, -h], annot[2], label=annot[1])

    for cor in corruptors:
        left = cor[0] - 5
        right = cor[0] + 5
        h = boxHeight  # Just for brevity.
        if cor[1] in "ACGT":
            # Use a diamond for SNPs.
            corXvals = [left, cor[0], right, cor[0]]
            corYvals = [0, h, 0, -h]
        else:
            # Use a wedge for deletions
            corXvals = [left, left + 4, left, right, right - 4, right]
            corYvals = [-h, 0, h, h, 0, -h]
        ax.fill(corXvals, corYvals, corruptorColors[cor[1]])
    ax.legend()
