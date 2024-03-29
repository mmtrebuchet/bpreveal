"""Useful tools for creating sequences with a desired property."""
from __future__ import annotations
import ast
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
import random
from typing import TypeAlias, Literal
from collections.abc import Callable
import matplotlib.colors
import numpy.typing as npt
import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
from bpreveal import utils
from bpreveal.internal.constants import LOGIT_AR_T, PRED_AR_T, LOGCOUNT_T
import numpy as np
import matplotlib.axes

# Types
CorruptorLetter: TypeAlias = Literal["A"] | Literal["C"] | Literal["G"] | Literal["T"] \
    | Literal["d"] | Literal["Ǎ"] | Literal["Č"] | Literal["Ǧ"] | Literal["Ť"]
"""Any letter that is a valid corruptor."""

Corruptor: TypeAlias = tuple[int, CorruptorLetter]
"""A corruptor gives a coordinate and a change to make.

The change is represented by a single letter. The letters `ACGT` indicate a SNP
at that locus, the letter `d` indicates that that base should be deleted, and
the letters `ǍČǦŤ` indicate that the corresponding base should be inserted
immediately after the locus.

Examples::

    (85, "A"),
    (1525, "Ť"),
    (1603, "d")
"""

Profile: TypeAlias = list[tuple[LOGIT_AR_T, LOGCOUNT_T]]
"""A profile is the result from running a model.

It is a list of (logits, logcounts) tuples.

:type: list[tuple[LOGIT_AR_T, LOGCOUNT_T]
"""

CandidateCorruptor: TypeAlias = tuple[int, str]
"""A candidate corruptor gives all possible mutations at a particular locus.

Examples::

    (85, "ACTd"),
    (1525, "AGTdǍČǦŤ"),
    (1603, "G")

Note that it is possible for a CandidateCorruptor to only allow for one
possible mutation at a locus.
"""

ANNOTATION_T: TypeAlias = tuple[tuple[int, int], str, str] \
    | tuple[tuple[int, int], str, str, float, float]
"""The shape for an annotation object to pass to :func:`plotTraces`.

It contains:

1. a pair of integers, giving the start and stop points of that annotation,
2. a string giving its label,
3. a string giving its color,
4. (Optional) a float giving the bottom of its annotation box, and
5. (Optional) a float giving the top of its annotation box.

"""

# A few variables that you can use to get the accented letters Ǎ, Č, Ǧ, and Ť
# in case they aren't easily typeable on your keyboard:

IN_A: CorruptorLetter = "Ǎ"
"""The letter Ǎ represents inserting an A"""

IN_C: CorruptorLetter = "Č"
"""The letter Č represents inserting a C"""

IN_G: CorruptorLetter = "Ǧ"
"""The letter Ǧ represents inserting a G"""

IN_T: CorruptorLetter = "Ť"
"""The letter Ť represents inserting a T"""

IN_L = "ǍČǦŤ"
"""The four insertion letters"""

IN_D = {"A": "Ǎ", "C": "Č", "G": "Ǧ", "T": "Ť"}
"""A dict mapping a regular letter ACGT to an insertion code ǍČǦŤ"""

CORRUPTOR_TO_IDX: dict[CorruptorLetter, int] =\
    {"A": 0, "C": 1, "G": 2, "T": 3,
     "Ǎ": 4, "Č": 5, "Ǧ": 6, "Ť": 7,
     "d": 8}
"""Use these to map corruptors to integers."""

IDX_TO_CORRUPTOR = "ACGTǍČǦŤd"
"""Given an integer, which corruptor does it represent?
This is the inverse of CORRUPTOR_TO_IDX.
"""

_seqCmap = {"A": (0, 204, 148), "C": (0, 114, 178),
            "G": (240, 228, 66), "T": (213, 94, 0)}
for _k in "ACGT":
    _seqCmap[_k] = tuple((x / 255 for x in _seqCmap[_k]))
corruptorColors = {
    "A": _seqCmap["A"], "C": _seqCmap["C"], "G": _seqCmap["G"], "T": _seqCmap["T"],
    "d": "black",
    "Ǎ": _seqCmap["A"], "Č": _seqCmap["C"], "Ǧ": _seqCmap["G"], "Ť": _seqCmap["T"]}
"""BPreveal's coloring for bases. A is green, C is blue, G is yellow, and T is red.

These colors are drawn from the Wong palette, but with the green lightened a bit.

:type: dict[CorruptorLetter, tuple[float, float, float]]

:meta hide-value:
"""


def corruptorsToArray(corruptorList: list[Corruptor]) -> list[tuple[int, int]]:
    """Convert a list of Corruptors to an array of numbers.

    :param corruptorList: A list of corruptors to serialize.
    :type corruptorList: list[:data:`Corruptor<bpreveal.gaOptimize.Corruptor>`]
    :return: A list of tuples of numbers representing the same information.

    Given a list of corruptor tuples, like ``[(1354, 'C'), (1514, 'Ť'), (1693, 'd')]``,
    convert that to an integer list that can be easily saved. This list will have the
    format ``[ [1354, 1], [1514, 7], [1693, 8] ]``
    where the second number indicates the type of corruptor, as given by CORRUPTOR_TO_IDX.
    Returns a list of lists, not a numpy array!
    If you want an array, use ``np.array(corruptorsToArray(myCorruptors))``
    """
    ret = []
    for corruptor in corruptorList:
        ret.append([corruptor[0], CORRUPTOR_TO_IDX[corruptor[1]]])
    return ret


def arrayToCorruptors(corruptorArray: list[tuple[int, int]]) -> list[Corruptor]:
    """Turn an array of numbers into a Corruptor list.

    :param corruptorArray: A list of tuples of ints.
    :return: Corruptors corresponding to the array.
    :rtype: list[:py:data:`Corruptor<bpreveal.gaOptimize.Corruptor>`]

    Takes an array of numerical corruptors in the form
    ``[ [1354, 1], [1514, 7], [1693, 8] ]``
    and generates the canonical list with letters, as used in the rest of the
    code. This function will work on a list or a numpy array of shape (N x 2).
    Returns a list of tuples, one for each position that was corrupted, like:
    ``[(1354, 'C'), (1514, 'Ť'), (1693, 'd')]``
    """
    ret = []
    for corruptorPair in corruptorArray:
        pos = int(corruptorPair[0])
        idx = int(corruptorPair[1])
        ret.append((pos, IDX_TO_CORRUPTOR[idx]))
    return ret


def stringToCorruptorList(corruptorStr: str) -> list[Corruptor]:
    """Parse a string and turn it into a :py:data:`~Corruptor` list.

    :param corruptorStr: The string to parse.
    :return: A list of Corruptors.
    :rtype: list[:py:data:`~Corruptor`]

    Takes a string representing a list of corruptors and generates
    the actual list as a python object. For example, if the string is
    ``"[(1354, 'C'), (1514, 'Ť'), (1693, 'd')]"``
    this function will parse it to the list of tuples
    ``[(1354, 'C'), (1514, 'Ť'), (1693, 'd')]``

    (The inverse of this function is simply `str` on a corruptor list.)
    """
    ret = []
    tree = ast.parse(corruptorStr)
    elems = tree.body[0].value.elts  # type: ignore
    for e in elems:
        pos = e.elts[0].value  # type: ignore
        cor = e.elts[1].value  # type: ignore
        ret.append((pos, cor))
    return ret


class Organism:
    """Represents the set of corruptors that are to be applied to the input sequence.

    :param corruptors: A list of corruptors that this organism represents.
    :type corruptors: list[:py:data:`~Corruptor`]
    """

    profile: Profile
    score: float
    corruptors: list[Corruptor]

    def __init__(self, corruptors: list[Corruptor]) -> None:
        """Construct an organism with the given corruptors."""
        self.corruptors = sorted(corruptors)
        assert validCorruptorList(self.corruptors), "Invalid corruptors in constructor."

    def getSequence(self, initialSequence: str, inputLength: int) -> str:
        """Apply this organism's corruptors to initialSequence, a string.

        :param initialSequence: A string representing the wild-type sequence that
            this organism will apply its corruptors to.
        :param inputLength: The length of the returned sequence.
        :return: A string of length inputLength.

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

    def setScore(self, scoreFn: Callable[[Profile, list[Corruptor]], float]) -> None:
        """Apply the score function to this organism's profile.

        :param scoreFn: A function that takes this organism's profile and its corruptor
            list and returns a float.
        :type scoreFn: Callable(:py:data:`~Profile`, list[:py:data:`~Corruptor`]) -> float

        This function can only be called after the organism's profile has been set!
        """
        assert self.profile is not None, \
            "Attempting to score organism with no profile."
        self.score = scoreFn(self.profile, self.corruptors)

    def __eq__(self, other: "Organism") -> bool:
        """Return True if this organism has the same corruptors as the other.

        :param other: The organism to check against
        :return: True if they have the same corruptors, False otherwise.

        Does *not* check that the profiles or score are identical!
        """
        return self.cmp(other) == 0

    def __hash__(self) -> int:
        """Return an integer representing a hash of this organism's corruptors.

        :return: An integer unique to this organism's corruptors.

        This means that you can use organisms as a key
        in a dictionary, for example.
        Note that profile and score are not integrated in this hash!
        """
        return str(self.corruptors).__hash__()

    def cmp(self, other: "Organism") -> int:  # pylint: disable=too-many-return-statements
        """A general comparator between two organisms based on their corruptors.

        :param other: The organism to compare against.
        :return: 1 if this organism is after (alphabetically) the other
            one, -1 if this organism comes first, or 0 if the two organisms
            have the same corruptors.

        Note that this does not compare profile or score!
        """
        for i in range(min(len(self.corruptors),
                           len(other.corruptors))):
            mine = self.corruptors[i]
            them = other.corruptors[i]
            if mine[0] < them[0]:
                # My corruptor comes first.
                return -1
            if mine[0] > them[0]:
                # My corruptor comes second.
                return 1
            if mine[1] < them[1]:
                # My letter is earlier.
                return -1
            if mine[1] > them[1]:
                # My letter is later.
                return 1
        if len(self.corruptors) > len(other.corruptors):
            return 1
        if len(self.corruptors) < len(other.corruptors):
            return -1
        # We have identical corruptors, we are the same organism.
        return 0

    def mutated(self, allowedCorruptors: list[CandidateCorruptor],
                checkCorruptors: Callable[[list[Corruptor]], bool]) -> "Organism":
        """Mutate this organism's corruptors.

        :param allowedCorruptors: All corruptors that this organism has access to
        :type allowedCorruptors: list[:py:data:`~CandidateCorruptor`]
        :param checkCorruptors: A function that accepts a list of corruptors and
            returns True if they are a valid combination and False otherwise.
        :type checkCorruptors: Callable[[list[:py:data:`~Corruptor`]], bool]
        :return: A newly-allocated Organism with a new set of corruptors.

        This is something you may want to override in a subclass.
        This returns a NEW organism that has one changed corruptor.
        It chooses one of its current corruptors at random, and then changes
        it to a random selection from allowedCorruptors. (allowedCorruptors
        has the same structure as in a Population).
        It makes sure that the new corruptor doesn't occur on the same base as
        an existing one (unless it's an insertion), and also calls
        checkCorruptors on the resulting corruptor candidates. It will make
        100 attempts to generate a new organism, and then it will error out.
        """
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

    def mixed(self, other: "Organism",
              checkCorruptors: Callable[[list[Corruptor]], bool]) -> "Organism":
        """Make a new organism by combining this one with another.

        :param other: Another organism that this should be mixed with.
        :param checkCorruptors: A function that returns True if a corruptor
            list is valid, and False otherwise.
        :type checkCorruptors: Callable[[list[:py:data:`~Corruptor`]], bool]
        :return: A newly-allocated Organism that is a blend of both parents.

        This is also something you may wish to override in a subclass.
        Currently, it pools the corruptors from self and other, and then
        randomly selects numCorruptors of them. If that passes checkCorruptors,
        then it returns a NEW organism.
        """
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
                elif corruptorPool[-1][1] == "d":
                    # The new thing is an insertion.
                    # In this case, the only contention is with deletion.
                    # If we have SNP then insertion, that's no problem
                    # but if we have deletion then insertion, that's
                    # equivalent to a SNP. So just pick one.
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
    """The main class for running the sequence optimization GA.

    This is a heck of a constructor, but you need to make a lot of
    choices to use the GA.

    :param initialSequence: A string of length (input-length + numMutations)
    :param inputLength: is the length of sequence that is given to your model.
    :param populationSize: is the number of organisms at the end of each generation.
    :param numCorruptors: determines how many corruptors will be applied
        in each organism.
    :param allowedCorruptors: is a list of tuples. (see below)
    :type allowedCorruptors: list[:py:data:`~CandidateCorruptor`]
    :param checkCorruptors: is a function that determines if a list of corruptors
        is valid. (see below)
    :type checkCorruptors: Callable[[list[:py:data:`~Corruptor`]], bool]
    :param fitnessFn: The fitness function. It takes two arguments:
        The first argument is a list of tuples containing the model outputs.
        These have the same organization as the outputs from batchPredictor.
        The second argument is a list of corruptors, as presented to checkCorruptors.
    :type fitnessFn: Callable[[:py:data:`~Profile`, list[:py:data:`~Corruptor`]], float],
    :param numSurvivingParents: The number of parents that will be kept for the next
        generation, usually referred to as elitism in GA terminology.
    :param predictor: A BatchPredictor that has been set up with the model you
        want to use.

    The initial sequence must be longer than needed for your model.
    The extra length is needed because this GA can have deletions
    and you need sequence to fill in the gaps when that happens. If you limit
    the number of deletions allowed (say, by not having them in allowedCorruptors
    or restricting them through checkCorruptors), then you need to provide
    (input-length + numPossibleDeletions), and if you're only allowing SNPs,
    then numPossibleDeletions = 0.

    allowedCorruptors is a list of tuples, with each tuple being a
    :py:data:`~CandidateCorruptor`. Each tuple contains two elements: a number,
    representing the position in the input sequence (starting at zero), and the
    second is a string containing the allowed corruptions at the base at the
    positing given by the number. For example, if your sequence is AGGCA, and
    you want any base to be corruptor to any other except that the C cannot be
    corrupted to a T and the second G cannot be corrupted at all,
    allowedCorruptors would be
    ``[(0, "CGT"), (1, "ACT"), (3, "AG"), (4, "CGT")]``.
    In the string of possible corruptors, the capital letters A, C, G, and T
    refer to a SNP of the base at the given position to the new letter,
    a lowercase 'd' means that the base there should be deleted, and a
    letter with a caron, like Ǎ, Č, Ǧ, or Ť, means that the given letter
    should be inserted AFTER the base number.

    checkCorruptors should take a list of :py:data:`~Corruptor`. These are
    tuples, and the first element of each tuple is a number, representing the
    base to be corrupted, and the second will be a single character,
    representing the corruption to be applied The corruptor locations will
    always be sorted in increasing order. You could use this, for example, to
    make sure that no two corruptors are next to each other. If you don't wish
    to apply any logic to the corruptors, pass in ``lambda x: True``.
    """

    initialSequence: str
    """The un-mutated sequence that all organisms will work with."""
    organisms: list[Organism]
    """The Organisms in this population.

    When the population is created, the organisms will have no profile or score
    information.

    After you execute :py:meth:`runCalculation()<~runCalculation>`, each
    organism will contain profile and score data, and the organisms list will
    be sorted in ascending order of score. So organisms[-1] is the best
    organism in the population.

    After you call :py:meth:`nextGeneration()<~nextGeneration>`, the organisms
    in the population are reset and don't contain score or profile information
    any more.
    """

    def __init__(self, initialSequence: str, inputLength: int, populationSize: int,
                 numCorruptors: int, allowedCorruptors: list[CandidateCorruptor],
                 checkCorruptors: Callable[[list[Corruptor]], bool],
                 fitnessFn: Callable[[Profile, list[Corruptor]], float],
                 numSurvivingParents: int, predictor: utils.BatchPredictor):
        """Construct a population."""
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
        """Create the initial pool of organisms."""
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
        """Construct a random new organism.

        :return: A newly-allocated Organism.
        """
        for _ in range(100):
            corLocations = random.sample(self.allowedCorruptors, self.numCorruptors)
            # Since sample is without replacement, I'm guaranteed to not have
            # corruptors in places where I've already put a corruptor.
            # This means that if I'm to have a SNP and insertion at the same place,
            # it has to arise through the evolution, no organism starts that way.
            cors = [(x[0], random.choice(x[1])) for x in corLocations]

            if self.checkCorruptors(sorted(cors)):  # type: ignore
                return Organism(cors)  # type: ignore
        assert False, "Over 100 attempts to choose corruptors for new organism."

    def runCalculation(self) -> None:
        """Run the GA.

        Runs the current population through the model, assigns scores to each
        organism, and sorts the organisms by score. If you want to save a list
        of the best parents, remember that the organisms are sorted in
        *ascending* order of fitness, so the best organism is
        pop.organisms[-1].
        """
        numInFlight = 0
        for i, organism in enumerate(self.organisms):
            self.predictor.submitString(
                organism.getSequence(self.initialSequence, self.inputLength),
                i)
            numInFlight += 1
            while self.predictor.outputReady():
                ret = self.predictor.getOutput()
                self.organisms[ret[1]].profile = ret[0]
                self.organisms[ret[1]].setScore(self.fitnessFn)
                numInFlight -= 1

        while numInFlight:
            ret = self.predictor.getOutput()
            self.organisms[ret[1]].profile = ret[0]
            self.organisms[ret[1]].setScore(self.fitnessFn)
            numInFlight -= 1

        self.organisms.sort(key=lambda x: x.score)

    def _choose1(self) -> Organism:
        """Single organism selection operator.

        :return: A single Organism from this Population.

        Randomly choose one of the organisms in the population.
        This only makes sense once you've runCalculation(), since it needs
        to know which organisms are good. You may want to override this in
        a subclass to change the selection operator.
        """
        toChoose = int(random.triangular(0,
                                         self.populationSize,
                                         self.populationSize))
        if toChoose == self.populationSize:
            # In the (extremely unlikely) event that the distribution chooses
            # its maximal value, decrease it by one.
            toChoose = self.populationSize - 1
        return self.organisms[toChoose]

    def _choose2(self) -> tuple[Organism, Organism]:
        """Double organism selection operator.

        :return: Two different organisms drawn from this population.

        Randomly choose two of the organisms in the population.
        This guarantees that the two organisms will be different.
        You may want to override this in a subclass to change the
        selection operator.
        """
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
        """Perform mutation and mixing to generate new children.

        Once you've runCalculation(), you call this to create new organisms
        for the next generation. This replaces the .organisms array with the new
        children, and those organisms will not have any profile or score data.
        """
        # Keep the best parents (elitism)
        ret = set(self.organisms[-self.numSurvivingParents:])
        # Note that ret is a SET, and not a list. Sets in Python have the nice
        # property that adding an item that's already in the set does not
        # change its length. So I blindly keep adding new organisms to the set,
        # allowing for duplicates, and this loop keeps doing this until the
        # whole population is the desired size.
        numTries = 0
        while len(ret) < self.populationSize:
            numTries += 1
            assert numTries < self.populationSize * 2, \
                "Too many iterations building generation."
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
                # mutated can return false, in that case no valid child
                # could be generated.
                if not child:
                    continue
            ret.add(child)
        self.organisms = list(ret)


def getCandidateCorruptorList(sequence: str, regions: list[tuple[int, int]] | None = None,
                              allowDeletion: bool = True,
                              allowInsertion: bool = True) -> list[CandidateCorruptor]:
    """Give the corruptors that are possible in a given sequence.

    :param sequence: The DNA sequence that will be used.
    :param regions: An optional list of tuples giving *allowed* regions for mutations.
    :param allowDeletion: Is deletion a valid corruptor type?
    :param allowInsertion: Is insertion a valid corruptor type?
    :return: A list of Corruptors.
    :rtype: list[:py:data:`~Corruptor`]

    Given a sequence (a string), this generates a list of tuples
    that contain all of the possible corruptors for each position in the
    sequence. It will have the format
    ``[(0, "ACGdǍČǦŤ"), (1, "AGTdǍČǦŤ"), (2, "CGTdǍČǦŤ"), ...]``
    where each number is a position in the sequence and the letters are the things
    that can be done at that location. A regular letter means a SNP, a d means deletion
    and the accented letters mean an insertion *after* the position.

    regions, if provided, must be a list of tuples.
    Each tuple contains two numbers, giving a start and stop location
    (left inclusive, right-exclusive, like python array slicing)
    of bases that are corruptor candidates. For example,
    ``regions=[(100,200), (250,300)]`` would return
    ``[(100, "ACG"), (101, "AGT"), ... (199, "CGT"), (250, "ACT"), ... (299,"ACG")]``
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


def anyCorruptorsCloserThan(corList: list[Corruptor], distance: int) -> bool:
    """Are any corruptors close to each other?

    :param corList: The corruptors to consider.
    :type corList: list[:py:data:`~Corruptor`]
    :param distance: The minimum distance that must be between corruptors.
    :return: True if there exist two corruptors from corList that are less than
        distance apart, False otherwise.

    A utility function that you can integrate into checkCorruptors
    to see if any corruptors are close to each other. Given a sorted
    list of corruptors (each corruptor a tuple of (position, effect)),
    For example, to prevent corruptors on adjacent bases, distance=1.
    To ensure a gap of two bases between each corruptor, distance=2.
    """
    curPos = corList[0][0]
    for c in corList[1:]:
        if c[0] <= curPos + distance:
            return True
        curPos = c[0]
    return False


def removeCorruptors(corruptorList: list[CandidateCorruptor],
                     corsToRemove: list[Corruptor]) -> list[CandidateCorruptor]:
    """Take corruptors out of a list.

    :param corruptorList: A list of candidate corruptors, like [(100, "ACG")].
    :type corruptorList: list[:py:data:`~CandidateCorruptor`]
    :param corsToRemove: A list of corruptors.
    :type corsToRemove: list[:py:data:`~Corruptor`]
    :return: A newly-allocated list of candidate corruptors with the given corruptors
        removed.
    :rtype: list[:py:data:`~CandidateCorruptor`]

    Given a candidate corruptor list, like from getCandidateCorruptorList, and
    a list of tuples giving disallowed corruptors, return a new candidate corruptor
    list where those disallowed corruptors are removed.
    corsToRemove is a list of tuples. The first element of each tuple is the position
    (relative to the start of the input sequence) that needs a corruptor removed, and
    the second is a string containing the forbidden letters.

    For example::

        removeCorruptors([(100, "ACG"), (101, "AT"), (102, "CGT"),
                          (103, "CT"), (104, "AGT")],
                         [(101, "T"), (103, "CT"), (104, "G")])
        -> [(100, "ACG"), (101, "A"), (102, "CGT"), (104, "AT")]

    """
    def removeLetters(original, forbidden):
        return "".join([x for x in original if x not in forbidden])
    ret = []
    for c in corruptorList:
        newLetters = c[1]
        for removal in corsToRemove:
            if c[0] == removal[0]:
                newLetters = removeLetters(newLetters, removal[1])
        if len(c[1]):  # There are still letters we can remove here.
            ret.append((c[0], newLetters))
    return ret


def validCorruptorList(corruptorList: list[Corruptor]) -> bool:
    r"""Is a list of corruptors even possible?

    :param corruptorList: The corruptors to consider.
    :type corruptorList: list[:py:data:`~Corruptor`]
    :return: True if the corruptors could be applied by an organism,
        False otherwise.

    A valid list satisfies the following property.
    For each pair :math:`(c_n, c_{n+1})` in corruptorList:

    * :math:`c_n[0] <= c_{n+1}[0]` (The list is ordered by the position of the corruptor.)

    * If :math:`c_n[0] == c_{n+1}[0]`:

        * :math:`(c_n[1], c_{n+1}[1])` must be in sorted order. For stupid reasons,
          sorted order is ``ACGTdČŤǍǦ``.

        * Neither :math:`c_n[1]` nor :math:`c_{n+1}[1]` are ``"d"``.

            * (You can't delete a base and do anything else to it.)

        * :math:`c_n[1] \in` ``"ACGT"`` :math:`\implies c_{n+1}[1] \in` ``"ǍČǦŤ"``.

            * (If you have a SNP, the only other thing
              at that position must be an insertion.)

    """
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
            if prev[1] == "d" or c[1] == "d":
                # A deletion and something else. Not allowed.
                return False
            if ord(prev[1]) > ord(c[1]):
                # The letters are not in order.
                return False
            if prev[1] in "ACGT" and c[1] not in "ǍČǦŤ":
                # We have a SNP and something that is not an insertion.
                return False
        prev = c
    return True


def plotTraces(posTraces: list[tuple[PRED_AR_T, str, str]],
               negTraces: list[tuple[PRED_AR_T, str, str]],
               xvals: npt.NDArray[np.float32],
               annotations: list[ANNOTATION_T],
               corruptors: list[Corruptor],
               ax: matplotlib.axes.Axes) -> None:
    """Generate a nice little plot with pips for corruptors and boxes for annotations.

    :param posTraces: The profiles to plot above the X axis.
    :param negTraces: The profiles to plot below the X axis.
    :param xvals: An array giving the genomic coordinates of your data.
    :param annotations: Annotations that you'd like to put on your plot.
    :param corruptors: A list of Corruptors that you'd like to put on your plot.
    :type corruptors: list[:py:data:`~Corruptor`]
    :param ax: The matplotlib Axes object to draw on.

    posTraces is a list of tuples. Each tuple has three things:

        1. A one-dimensional array of values. The number of values must be the
           same as the number of points in xvals.
        2. A string that will be used as a label for the trace.
        3. A string that will be used for the color of the trace.

    negTraces has the same structure as posTraces, but will be negated before
    being plotted. This is handy to make different conditions visually
    distinct, and of course for chip-nexus data.

    annotations is a list of tuples. Each tuple contains:

        1. a pair of integers, giving the start and stop points of that annotation,
        2. a string giving its label,
        3. a string giving its color,
        4. a float giving the bottom of its annotation box, and
        5. a float giving the top of its annotation box. For example::

            `[((431075,431089), "FKH2", "red", 0.5, 0,7),
            ((431200, 431206), "PHO4", "blue", 0.3, 0.6)]`

    corruptors has the same format as the corruptors in an organism, but be
    sure you shift the coordinates appropriately so that they line up with
    xvals. In other words, the coordinates of the corruptors should be the
    real genomic coordinates of the mutations.
    (Remember that corruptor coordinates are relative to the start of
    the INPUT, not the output.)

    """
    maxesPos = [max(x[0]) for x in posTraces]
    # The negative traces aren't negated yet, so use max.
    maxesNeg = [max(x[0]) for x in negTraces]
    boxHeight = max(maxesPos + maxesNeg) / 20
    for posTrace in posTraces:
        ax.plot(xvals, posTrace[0] + boxHeight, label=posTrace[1], color=posTrace[2])
    for negTrace in negTraces:
        ax.plot(xvals, -negTrace[0] - boxHeight, label=negTrace[1], color=negTrace[2])
    usedLabels = []
    for annot in annotations:
        match annot:
            case ((l, r), label, color):
                h = boxHeight  # Just for brevity.
                ax.fill([l, l, r, r], [-h, h, h, -h], color, label=label)
            case ((l, r), label, color, startFrac, stopFrac):
                bottom = -boxHeight + (2 * boxHeight * startFrac)
                top = -boxHeight + (2 * boxHeight * stopFrac)
                if (color, label) not in usedLabels:
                    ax.fill([l, l, r, r], [bottom, top, top, bottom], color, label=label)
                    usedLabels.append((color, label))
                else:
                    ax.fill([l, l, r, r], [bottom, top, top, bottom], color)

    for cor in corruptors:
        match cor:
            case (pos, corType):
                bottom = -boxHeight
                top = boxHeight
            case (pos, corType, startFrac, stopFrac):
                bottom = -boxHeight + (2 * boxHeight * startFrac)
                top = -boxHeight + (2 * boxHeight * stopFrac)

        left = pos - 5
        right = pos + 5
        # left = cor[0] - 5
        # right = cor[0] + 5
        # h = boxHeight  # Just for brevity.
        midPt = (bottom + top) / 2
        if corType in "ACGT":
            # Use a diamond for SNPs.
            corXvals = [left, cor[0], right, cor[0]]
            corYvals = [midPt, top, midPt, bottom]
        else:
            # Use a wedge for deletions
            corXvals = [left, left + 4, left, right, right - 4, right]
            corYvals = [bottom, midPt, top, top, midPt, bottom]
        ax.fill(corXvals, corYvals, matplotlib.colors.to_hex(corruptorColors[corType]))
    ax.legend()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
