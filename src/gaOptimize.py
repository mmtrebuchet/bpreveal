import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import random

class Organism:
    """This represents the set of mutations that are to be applied to the
    input sequence."""
    def __init__(self, mutations):
        """Construct an organism with the given mutations.
        """

        self.mutations = sorted(mutations)
        self.profile = None
        self.score = None

    def getSequence(self, initialSequence):
        seq = list(initialSequence)
        for m in self.mutations:
            seq[m[0]] = m[1]
        return "".join(seq)

    def setScore(self, scoreFn):
        assert self.profile is not None
        self.score = scoreFn(self.profile)

    def __eq__(self, other):
        return self.cmp(other) == 0

    def cmp(self,other):
        for i in range(len(self.mutations)):
            mine = self.mutations[i]
            them = other.mutations[i]
            if(mine[0] < them[0]):
                # My mutation comes first.
                return -1
            elif (mine[0] > them[0]):
                # My mutation comes second.
                return 1
            else:
                if(mine[1] < them[1]):
                    # My letter is earlier.
                    return -1
                elif(mine[1] > them[1]):
                    # My letter is later.
                    return 1
        #We have identical mutation profiles.
        return 0

    def mutated(self, allowedMutations, checkMutations):
        """This is something you may want to override in a subclass.
        This returns a NEW organism that has a single mutation from this one.
        It chooses one of its current mutations at random, and then changes
        it to a random selection from allowedMutations. (allowedMutations
        has the same structure as in a Population).
        It makes sure that the new mutation doesn't occur on the same base as
        an existing one, and also calls checkMutations on the resulting
        mutation candidates. It will make 100 attempts to generate
        a new organism, and then it will error out. """
        posToMut = random.randrange(0, len(self.mutations))
        keepMutations = [x for x in self.mutations if x[0] != posToMut]
        found = False
        numTries = 0
        while not found:
            numTries += 1
            assert numTries < 100, "Too many attempts to create mutated organism."
            newMutBase = random.choice(allowedMutations)
            newMut = (newMutBase[0], random.choice(newMutBase[1]))
            found = True
            # This is not a very efficient way to generate mutations,
            # but this is not the slow step in this algorithm.
            for i in range(len(self.mutations)):
                if (i != posToMut and self.mutations[i][0] == newMut[0]):
                    #I gave a position that is already taken. Try another
                    found = False
                # Also check to see if the mutations meet the user's muster.
            if found:
                found = checkMutations(sorted(keepMutations + [newMut]))
        return Organism(sorted(keepMutations + [newMut]))

    def crossover(self, other, checkMutations):
        """This is also something you may wish to override in a subclass.
        Currently, it pools the mutations from self and other, and then
        randomly selects numMutations of them. If that passes checkMutations,
        then it returns a new organism. """
        mutationPool = self.mutations + other.mutations
        for _ in range(100):
            chosens = sorted(random.sample(mutationPool, len(self.mutations)))
            add = True
            for i in range(len(chosens)-1):
                if(chosens[i][0] == chosens[i+1][0]):
                    add = False
            if(add and checkMutations(chosens)):
                return Organism(chosens)
        assert False, "Unable to mix parent organisms after 100 attempts."



class Population:
    """This is the main class for running the sequence optimization GA."""
    def __init__(self, initialSequence, populationSize, numMutations,
                 allowedMutations, checkMutations, fitnessFn,
                 numSurvivingParents, predictor):
        """This is a heck of a constructor, but you need to make a
        lot of choices to use the GA.

        initialSequence is a string of length input-length for your model.

        populationSize is the number of organisms at the end of each generation.

        numMutations determines how many mutations will be applied
        in each organism.

        allowedMutations is a list of tuples. Each tuple contains two elements:
        a number, representing the position in the input sequence (starting at
        zero), and the second is a string containing the bases that can replace
        the base at the positing given by the number. For example, if your
        sequence is AGGCA, and you want any base to be mutated to any other
        except that the C cannot be mutated to a T and the second G cannot
        be mutated at all, allowedMutations would be
        [(0, "CGT"), (1, "ACT"), (3, "AG"), (4, "CGT")].

        checkMutations is a function that determines if a list of mutations
        is valid. It will take a list of tuples, the first element of each
        tuple is a number, representing the base to be mutated, and the second
        will be a single character, representing the base that it will be
        mutated to. The mutation locations will always be sorted in
        increasing order. You could use this, for example, to make sure that
        no two mutations are next to each other. If you don't wish to apply
        any logic to the mutations, pass in lambda x: True.

        fitnessFn is the fitness function. It takes two arguments:
        The first argument is a list of tuples containing the model outputs.
        These have the same organization as the outputs from batchPredictor.
        The second argument is a list of mutations, as presented to checkMutations.

        numSurvivingParents is the number of parents that will be kept for the next
        generation, usually referred to as elitism in GA terminology.

        predictor is a BatchPredictor that has been set up with the model you
        want to use.
        """
        self.initialSequence = initialSequence
        self.populationSize = populationSize
        self.numMutations = numMutations
        self.allowedMutations = allowedMutations
        self.checkMutations = checkMutations
        self.fitnessFn = fitnessFn
        self.numSurvivingParents = numSurvivingParents
        self.organisms = []
        self.predictor = predictor

        self._seed()


    def _seed(self):
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
                    if(other == candidate):
                        add = False
                        break
            #We've created a new organism that is ready to go!
            self.organisms.append(candidate)


    def _newOrganism(self):
        for _ in range(100):
            mutLocations = random.sample(self.allowedMutations, self.numMutations)
            # Since sample is without replacement, I'm guaranteed to not have mutations in
            # places where I've already put a mutation.
            muts = [(x[0], random.choice(x[1])) for x in mutLocations]

            if(self.checkMutations(sorted(muts))):
                return Organism(muts)
        assert False, "Over 100 attempts to choose mutations for new organism."

    def runCalculation(self):
        for i, organism in enumerate(self.organisms):
            self.predictor.submitString(organism.getSequence(self.initialSequence, i))
        self.predictor.runBatch()
        for i in range(self.populationSize):
            ret = self.predictor.getOutput()
            self.organisms[ret[1]].profile = ret[0]
            self.organisms[ret[1]].setScore(self.fitnessFn)
        self.organisms.sort(key = lambda x: x.score)

    def _choose1(self):
        toChoose = int(random.triangular(0, self.populationSize, self.populationSize))
        if(toChoose == self.populationSize):
            # In the (extremely unlikely) event that the distribution chooses
            # its maximal value, decrease it by one.
            toChoose = self.populationSize - 1
        return self.organisms[toChoose]

    def _choose2(self):
        firstChoice =  int(random.triangular(0, self.populationSize, self.populationSize))
        if(firstChoice == self.populationSize):
            firstChoice = firstChoice - 1
        secondChoice = firstChoice
        while secondChoice == firstChoice:
            secondChoice =  int(random.triangular(0, self.populationSize, self.populationSize))
            if(secondChoice == self.populationSize):
                secondChoice = secondChoice - 1
        return self.organisms[firstChoice], self.organisms[secondChoice]

    def runGeneration(self):
        # Keep the best parents (elitism)
        ret = self.organisms[-self.numSurvivingParents:]
        numTries = 0
        while(len(ret) < self.populationSize):
            numTries += 1
            assert numTries < self.populationSize * 2, "Too many iterations building next generation."
            if(random.randrange(2)):
                # Do mixing.
                parents = self._choose2()
                child = parents[0].crossover(parents[1], self.checkMutations)
            else:
                # Do mutation
                parent = self._choose1()
                child = parent.mutate(self.allowedMutations, self.checkMutations)
            add = True
            for r in ret:
                #Yes, this is a quadratic check for duplicates. Bite me.
                if(child == r):
                    add = False
                    break
            if(add):
                ret.append(child)
        self.organisms = ret
        self.runCalculation()
