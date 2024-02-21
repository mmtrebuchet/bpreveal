"""A set of callbacks useful for training a model."""
import re
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    ReduceLROnPlateau, Callback
from bpreveal import logUtils
from bpreveal import generators


def getCallbacks(earlyStop: int, outputPrefix: str, plateauPatience: int, heads,
                 trainBatchGen: generators.H5BatchGenerator,
                 valBatchGen: generators.H5BatchGenerator):
    """Return a set of callbacks for your model.

    :param earlyStop: The ``early-stopping-patience`` from the config file.
    :param outputPrefix: The ``output-prefix`` for the model, including directory.
    :param plateauPatience: The ``learning-rate-plateau-patience`` from the config file.
    :param heads: The heads list for your model, to which adaptive loss λ tensors
        have been added.
    :param trainBatchGen: The batch generator for training. Just used to see how many
        batches there will be.
    :param valBatchGen: The batch generator for validation. Just used to see how many
        batches there will be.
    :return: A list of Keras callbacks that you should use to train your model.

    The returned callbacks are:

    EarlyStopping
        Stop training if the validation loss hasn't improved for a while.
    ModelCheckpoint
        Write a checkpoint file every time the validation loss improves.
    ReduceLROnPlateau
        If validation loss hasn't improved for a while, decrease the learning rate.
    ApplyAdaptiveCountsLoss
        Implement the :doc:`adaptive counts loss algorithm<countsLossReweighting>`.

    """
    logUtils.debug("Creating callbacks based on earlyStop "
                   "{0:d}, outputPrefix {1:s}, plateauPatience {2:d}".format(
                       earlyStop, outputPrefix, plateauPatience))
    if logUtils.getLogger().isEnabledFor(logUtils.INFO):
        verbose = 1
    else:
        verbose = 0
    earlyStopCallback = EarlyStopping(monitor="val_loss",
                                      patience=earlyStop,
                                      verbose=verbose,
                                      mode="min",
                                      restore_best_weights=True)

    filepath = "{}.checkpoint.model".format(outputPrefix)
    checkpointCallback = ModelCheckpoint(filepath,
                                         monitor="val_loss",
                                         verbose=verbose,
                                         save_best_only=True,
                                         mode="min")
    plateauCallback = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=plateauPatience, verbose=verbose)
    adaptiveLossCallback = ApplyAdaptiveCountsLoss(heads, 0.3, plateauCallback,
                                                   earlyStopCallback, checkpointCallback)
    displayCallback = DisplayCallback(plateauCallback, earlyStopCallback,
                                      adaptiveLossCallback,
                                      trainBatchGen, valBatchGen)
    return [earlyStopCallback, checkpointCallback, plateauCallback,
            adaptiveLossCallback, displayCallback]


class DisplayCallback(Callback):
    """Replaces the tensorflow progress bar logger with lots of printing to stderr.

    :param trainBatchGen: The training batch generator.
    :param valBatchGen: The validation batch generator.
    :param plateauCallback: The plateau callback, used to access the LR schedule.
    :param earlyStopCallback: The EarlyStopping callback, used to see how long we have left.
    :param adaptiveLossCallback: The adaptive loss callback, used to read λ values.
    """

    epochNumber = 0
    """What is the currently-running epoch number?"""
    batchNumber = 0
    """What is the currently-running training batch number?"""
    printLocationsEpoch = {}
    """For a given data type, what row should it be printed on
    in the epoch pane? For example, "val_loss" might go on row 5."""

    printLocationsBatch = {}
    """What row should each data type go on in the batch pane?"""
    multipliers = {}
    """For a given data type, what constant should it be multiplied by
    for display? This is used to weight profile losses by profile-loss-weight.
    """
    prevEpochLogs = None
    """The logs from last epoch"""
    lastBatchEndTime = 0
    """When did the last batch finish?"""
    lastValBatchEndTime = 0
    """When did the last validation batch finish?"""
    lastEpochEndTime = None
    """When did the last epoch finish?"""
    lastEpochStartTime = None
    """When did the current epoch start?"""
    numEpochs: int
    """What is the maximum number of training epochs?"""
    numBatches: int
    """How many training batches per epoch?"""
    numValBatches: int
    """How many validation batches per epoch?"""
    curEpochWaitTime = None
    """How long between the end of the last epoch and the start of this one?"""
    maxLen = 0
    """Of all the data types, what is the length of the longest name?
    Used to calculate column positions."""

    def __init__(self, plateauCallback, earlyStopCallback,
                 adaptiveLossCallback, trainBatchGen, valBatchGen):
        self.plateauCallback = plateauCallback
        self.earlyStopCallback = earlyStopCallback
        self.adaptiveLossCallback = adaptiveLossCallback
        self.numBatches = len(trainBatchGen)
        self.numValBatches = len(valBatchGen)

    def _calcPositions(self, initialLogs: dict):
        """Assign rows and columns to all log types.

        Given the first set of logs from a training batch,
        work out where all of the logs need to be displayed in their window.
        """
        profileLosses = []
        countsLosses = []
        remainingTerms = []
        for lossName in initialLogs.keys():
            foundInHeads = False
            for head in self.adaptiveLossCallback.heads:
                headName = head["head-name"]
                profileRe = r".*profile_{0:s}_loss".format(headName)
                countsRe = r".*logcounts_{0:s}_loss".format(headName)
                if re.fullmatch(profileRe, lossName):
                    profileLosses.append(lossName)
                    profileLosses.append("val_" + lossName)
                    profileLosses.append("EPOCH_SPACER")
                    self.multipliers[lossName] = head["profile-loss-weight"]
                    self.multipliers["val_" + lossName] = head["profile-loss-weight"]
                    foundInHeads = True
                    break
                if re.fullmatch(countsRe, lossName):
                    countsLosses.append(lossName)
                    countsLosses.append("val_" + lossName)
                    countsLosses.append("EPOCH_SPACER")
                    foundInHeads = True
                    break
            if not foundInHeads:
                if lossName not in ["lr", "epoch", "batch", "loss"]:
                    remainingTerms.append(lossName)
        assert len(remainingTerms) == 0, str(remainingTerms) + " includes unknown loss component. "\
            "not " + str(profileLosses) + "/" + str(countsLosses)

        printOrderEpoch = ["epoch", "EPOCH_SPACER", "loss", "val_loss", "EPOCH_SPACER"]
        printOrderEpoch.extend(profileLosses)
        printOrderEpoch.extend(countsLosses)
        printOrderEpoch.extend(["Best epoch", "Best loss", "Epochs until earlystop",
                           "lr", "Epochs until LR plateau", "EPOCH_SPACER", "Setup time",
                           "Training time", "Validation time",
                           "Seconds / epoch", "Minutes to earlystop"])

        printOrderBatch = ["batch", "loss", "EPOCH_SPACER"]
        printOrderBatch.extend(profileLosses)
        printOrderBatch.extend(countsLosses)
        # Now that we have the orders, just enumerate them and put them in
        # printLocations.
        for i, po in enumerate(printOrderEpoch):
            self.printLocationsEpoch[po] = i + 2
        for i, po in enumerate([x for x in printOrderBatch if x != "EPOCH_SPACER"]):
            self.printLocationsBatch[po] = i + 2
        self.maxLen = max((len(x) for x in printOrderEpoch))

    def on_train_begin(self, logs: dict | None = None):  # pylint: disable=invalid-name
        """Just loads in the total number of epochs."""
        del logs
        params = self.params.get
        autoTotal = params("epochs", params("nb_epochs", None))
        if autoTotal is not None:
            self.numEpochs = autoTotal

    def on_epoch_begin(self, epoch: int, logs: dict | None = None):  # pylint: disable=invalid-name
        """Just sets the timers up, so you can check how long an epoch took at the end."""
        del logs
        self.epochNumber = epoch
        # pylint: disable=attribute-defined-outside-init
        self.batchNumber = 0
        self.curEpochStartTime = time.perf_counter()
        self.firstBatchTime = time.perf_counter() + 1e10
        self.firstValBatchTime = time.perf_counter() + 1e10
        if self.lastEpochEndTime is not None:
            self.curEpochWaitTime = time.perf_counter() - self.lastEpochEndTime
        # pylint: enable=attribute-defined-outside-init

    def formatStr(self, val: str | int | float | tuple[int, int]) -> str:
        """Formats an object to be 10 characters wide.

        If a second object is provided, format as a ratio.

        :param str1: The thing to format
        :param str2: Optional. If provided, format the two inputs as a ratio.
        :return: A 10 character wide formatted string.
        """
        match val:
            case str():
                return "{0:>10s}".format(val)
            case int():
                return "{0:>10d}".format(val)
            case float():
                return "{0:>10.3f}".format(val)
            case int(), int():
                return "{0:>3d} / {1:>4d}".format(*val)
            case _, None:
                return "{0:>10s}".format(str(val))
            case _:
                return "{0:10s}".format("FMT_ERR")

    def on_epoch_end(self, epoch: int, logs: dict | None = None):  # pylint: disable=invalid-name
        """Writes out all the logs for this epoch and the last one at INFO logging level."""
        del epoch
        logs = {k: logs[k] for k in logs.keys()}
        recordEpoch = self.earlyStopCallback.best_epoch
        correctedLoss = self.earlyStopCallback.best
        # Add some extra data to the logs. (Note that I copied the logs dict)
        logs["Best epoch"] = recordEpoch
        logs["Best loss"] = float(correctedLoss)
        logs["Epochs until earlystop"] = (self.earlyStopCallback.wait,
                                          self.earlyStopCallback.patience)
        logs["Epochs until LR plateau"] = (self.plateauCallback.wait,
                                           self.plateauCallback.patience)
        if self.lastEpochEndTime is not None:
            timePerEpoch = time.perf_counter() - self.lastEpochEndTime
            logs["Seconds / epoch"] = timePerEpoch
            logs["Minutes to earlystop"] = timePerEpoch * (
                self.earlyStopCallback.patience - self.earlyStopCallback.wait) / 60
        if self.curEpochWaitTime is not None:
            logs["Setup time"] = self.curEpochWaitTime
        logs["Training time"] = self.lastBatchTime - self.firstBatchTime
        logs["Validation time"] = self.lastValBatchTime - self.firstValBatchTime
        self.lastEpochEndTime = time.perf_counter()
        lines = self._getEpochLines(logs)
        self._writeLogLines(lines, logUtils.info, "E")
        lines = self._getλLines()
        self._writeLogLines(lines, logUtils.debug, "λ")
        self.prevEpochLogs = logs

    def on_train_batch_end(self, batch: int,  # pylint: disable=invalid-name
                           logs: dict | None = None):
        """Write the loss info for the current batch at DEBUG level."""
        # pylint: disable=attribute-defined-outside-init
        self.firstBatchTime = min(time.perf_counter(), self.firstBatchTime)
        self.lastBatchTime = time.perf_counter()
        # pylint: enable=attribute-defined-outside-init
        if len(list(self.printLocationsEpoch.keys())) == 0:
            # We haven't calculated any print locations yet.
            self._calcPositions(logs)
        if (time.perf_counter() - self.lastBatchEndTime > 0.1)\
                or batch in [self.numBatches - 1, 0]:
            self.lastBatchEndTime = time.perf_counter()

            logs = {k: logs[k] for k in logs.keys()}
            self.batchNumber = batch
            self._writeLogLines(self._getBatchLines(logs), logUtils.debug,
                                "BH" if batch == 0 else "B")

    def on_test_batch_end(self, batch: int,  # pylint: disable=invalid-name
                          logs: dict | None = None):
        """Just emit a counter with the batch number at DEBUG level."""
        del logs
        # pylint: disable=attribute-defined-outside-init
        self.firstValBatchTime = min(time.perf_counter(), self.firstValBatchTime)
        self.lastValBatchTime = time.perf_counter()
        # pylint: enable=attribute-defined-outside-init
        if (time.perf_counter() - self.lastValBatchEndTime > 0.1) \
                or batch in [self.numValBatches - 1, 0]:
            self.lastValBatchEndTime = time.perf_counter()
            lines = [((max(self.printLocationsBatch.values()), 1, "Eval batch"),
                      (max(self.printLocationsBatch.values()),
                       20,
                       self.formatStr((batch, self.numValBatches - 1))))]
            self._writeLogLines(lines, logUtils.debug, "V")

    def _writeLogLines(self, lines: list[tuple[int, int, str]], writer, win: str):
        """Actually write the lines to the logger.

        :param writer: The logger to use
        :type writer: logging.logger
        :param lines: A list of tuples containing (row, column, string) data.
        :param win: A character (or two, if you want to include highlighting)
            that determines which window will be used in the display program.
        """
        for line in lines:
            # Condense all of the outputs on one line into one string.
            if len(line) == 0:
                return
            row = line[0][0]
            lineLen = line[-1][1] + len(line[-1][2])
            outChrs = list(" " * lineLen)
            for _, col, text in line:
                for i, c in enumerate(text):
                    outChrs[i + col] = c
            writer("∬{0:d}∬{1:d}∬{2:s}∬{3:s}".format(row, 1, win, ''.join(outChrs)))

    def _getλLines(self) -> list[tuple[int, int, str]]:
        lines = []
        for i, headName in enumerate(self.adaptiveLossCallback.λHistory.keys()):
            λValue = self.adaptiveLossCallback.λHistory[headName][-1]
            lines.append([((i + 2, 1, headName)), (i + 2, 20, "{0:8.3f}".format(λValue))])
        return lines

    def _getEpochLines(self, logs) -> list[tuple[int, int, str]]:
        lines = []
        logs["epoch"] = (self.epochNumber, self.numEpochs)
        logs["lr"] = "{0:10.7f}".format(logs["lr"])
        for lk in logs.keys():
            lines.append([(self.printLocationsEpoch[lk], 1, lk)])

            outStr = self.formatStr(logs[lk])
            lines[-1].append((self.printLocationsEpoch[lk], self.maxLen + 2, outStr))

            if self.prevEpochLogs is not None:
                outStrOld = self.formatStr(self.prevEpochLogs.get(lk, ""))
                lines[-1].append(
                    (self.printLocationsEpoch[lk], self.maxLen + 2 + 14, outStrOld))

        return lines

    def _getBatchLines(self, logs) -> list[tuple[int, int, str]]:

        logs["batch"] = (self.batchNumber, self.numBatches - 1)

        lines = []
        for lk in logs.keys():
            lines.append([(self.printLocationsBatch[lk], 1, lk)])
            outStr = self.formatStr(logs[lk])
            lines[-1].append((self.printLocationsBatch[lk], self.maxLen + 2, outStr))
        return lines


class ApplyAdaptiveCountsLoss(Callback):
    """Implements the adaptive counts loss algorithm.

    This updates the counts loss weights in your model on the fly, so that you can specify a
    target fraction of the loss that is due to counts, and the model will automatically update
    the weight. It currently does not update profile weights, so you can't yet automatically
    balance the losses in a multitask model.
    (You can still manually specify them in the config json, of course!)

    :param heads: Is straight from the json, except with "INTERNAL_counts-loss-weight-variable"
        entries in each head. If present, these variables are the Keras Variables that contain the
        loss weights. These will be updated in this callback.
    :param aggression: A number from 0 to 1.
    :param lrPlateauCallback: The learning rate plateau callback that your model is going to use.
    :param earlyStopCallback: The early stopping callback that your model is going to use.
    :param checkpointCallback: The checkpoint callback that your model is going to use.


    ``aggression`` determines how aggressively the counts loss
    will be re-weighted. Lower values indicate slower changes, and a value of 1.0 means that
    the old loss should be completely discarded at each iteration.
    (If the newly calculated loss weight is ever off by over a factor of two, it's clamped to
    be exactly twice (or half) of the old weight, so that gross instability in the early stages
    doesn't cause the model to explode.)

    The three callbacks are the others used in the model, and the model should run them BEFORE
    this callback is executed.
    This callback messes with their internal state (naughty, naughty!) because changing the
    loss weights could cause the model's loss value to go up even though the model hasn't
    gotten any worse.
    """

    # Straight from the json, with "INTERNAL_counts-loss-weight-variable".
    heads: dict
    # 0 = don't change weights, 1 = ignore history, change weights very fast.
    aggression: float
    # The logs at each epoch.
    logsHistory: list[dict]
    # The history of the counts loss weights for each head. (key is head-name)
    λHistory: dict[str, list]

    def __init__(self, heads: dict, aggression: float,
                 lrPlateauCallback, earlyStopCallback, checkpointCallback):
        """Build the callback."""
        super().__init__()
        self.heads = heads
        logUtils.debug(heads)
        self.aggression = aggression
        self.lrPlateauCallback = lrPlateauCallback
        self.earlyStopCallback = earlyStopCallback
        self.checkpointCallback = checkpointCallback
        self.logsHistory = []
        self.λHistory = {}
        for head in heads:
            self.λHistory[head["head-name"]] = []

    def on_train_begin(self, logs=None):  # pylint: disable=invalid-name,unused-argument
        """Set up the initial guesses for λ.

        :param logs: Ignored.

        At the beginning of training, see which heads are using adaptive weights.
        For those heads, load in an initial guess for λ based on the BPNet heuristic.
        """
        for head in self.heads:
            if "counts-loss-frac-target" in head:
                if "counts-loss-weight" in head:
                    # We've been given a starting point. Don't use the heuristic.
                    λ = head["counts-loss-weight"]
                    logUtils.debug(
                        "An initial counts-loss-weight was provided.")
                    logUtils.debug("Setting λ = {0:f} for head {1:s}"
                                   .format(λ, head["head-name"]))
                    head["INTERNAL_λ-variable"].assign(λ)
                else:
                    # Get the desired λ value. INTERNAL_mean-counts is added by the
                    # generator, which calls addMeanCounts in its constructor.
                    λ = head["INTERNAL_mean-counts"] * \
                        head["counts-loss-frac-target"]
                    # Now for a bit of magic - experimentation has determined this number.
                    λ = λ * 37.0
                    logUtils.info("Estimated initial λ of {0:f} for head {1:s} based on ĉ of {2:f}"
                                  .format(λ, head["head-name"], head["INTERNAL_mean-counts"]))
                    head["INTERNAL_λ-variable"].assign(λ)

    def getLosses(self, epoch: int, headName: str) -> tuple[float, float, float, float]:
        """Get what the losses actually were in a previous epoch using that epoch's λ values.

        :param epoch: The epoch for which you'd like losses.
        :param headName: The head for which you'd like losses.
        :return: A tuple with four losses: (profile, counts, val_profile, val_counts).

        What were the profile, counts, val_profile, and val_counts (in that order)
        losses at the given epoch? The counts weight returned from this function
        uses the counts weight in use at the given epoch.
        This method assumes that the validation and training losses
        match some regexes, and these are based on layer names. If someone
        got really goofy with head names (e.g., one head named "x" and another
        named "profile_x", then this could get messed up.
        """
        epochLosses = self.logsHistory[epoch]
        profileRe = r".*profile_{0:s}_loss".format(headName)
        countsRe = r".*logcounts_{0:s}_loss".format(headName)
        valProfileRe = r"val.*profile_{0:s}_loss".format(headName)
        valCountsRe = r"val.*logcounts_{0:s}_loss".format(headName)

        valProfile = valCounts = profile = counts = None
        for lossName in epochLosses.keys():
            # Match the validation regexes first, because the
            # non-validation losses are just missing the characters
            # 'val', and so they'd also match validation losses.
            if re.fullmatch(valProfileRe, lossName):
                valProfile = lossName
            elif re.fullmatch(valCountsRe, lossName):
                valCounts = lossName
            elif re.fullmatch(profileRe, lossName):
                profile = lossName
            elif re.fullmatch(countsRe, lossName):
                counts = lossName
        if None in [profile, counts, valProfile, valCounts]:
            # We missed one of the losses. Abort!
            # (with some debugging information.
            print("Loss names ", list(epochLosses.keys()))
            print("head name ", headName)
            print("regex matches ", [profile, counts, valProfile, valCounts])
            assert False, "A loss didn't match any regex."
        ret = (epochLosses[profile],
               epochLosses[counts],
               epochLosses[valProfile],
               epochLosses[valCounts])
        return ret

    def whatWouldValLossBe(self, epoch):
        """Determine a previous epoch's validation loss but use the current λ values to do it.

        :param epoch: The epoch number where you'd like to know what the loss would have been.
        :return: A float giving the corrected validation loss at that epoch.

        Had we been using the current λ in a previous epoch, what would
        its validation loss have been?
        """
        # Using the current heads, determine what the loss would have been at the given epoch.
        # (Since we store the loss history, this is pretty easy.)
        logs = self.logsHistory[epoch]
        valTotalLoss = 0
        for head in self.heads:
            _, _, vpl, vcl = self.getLosses(epoch, head["head-name"])
            valTotalLoss += vpl * float(head["profile-loss-weight"])

            λVar = head["INTERNAL_λ-variable"]
            curλ = λVar.read_value()
            oldλ = self.λHistory[head["head-name"]][epoch]
            valTotalLoss += vcl * curλ / oldλ
        logUtils.debug("Calculated new loss of {0:f} on epoch {1:d}, with original loss {2:f}"
                       .format(valTotalLoss, epoch, logs["val_loss"]))
        return valTotalLoss

    def resetCallbacks(self):
        """Manipulate the other callbacks so they don't break when λ changes.

        This is the squirreliest method here. The other callbacks that track model progression
        track the loss of the model at the record-setting epoch. But the definition of loss itself
        is changing during the training, so we need to update their stored idea of what the model's
        loss was during the training.

        For example, consider the following scenario::

            raw_loss  loss_weight  scaled_loss
            10          1           10
            9           1           9
            8           1           8
            7           2           14
            6           2           12
            5           2           10

        A callback that was tracking the loss, looking for the minimum, would claim
        that the best epoch was epoch 3, where the scaled loss was eight.
        We need to go into that callback and say, "no, with our current loss weight,
        the loss on epoch three would *actually* have been 16." so that the
        callback thinks that the loss of 14 on epoch 4 is an improvement.
        """
        # Find which epoch set the record.
        recordEpoch = self.earlyStopCallback.best_epoch
        # Now, set the callbacks to have a corrected loss at the record-setting epoch.
        correctedLoss = self.whatWouldValLossBe(recordEpoch)
        logUtils.debug("Resetting callbacks from old loss {0:f} to new loss {1:f}"
                       .format(self.lrPlateauCallback.best, correctedLoss))
        self.lrPlateauCallback.best = correctedLoss
        self.earlyStopCallback.best = correctedLoss
        self.checkpointCallback.best = correctedLoss

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=invalid-name
        """Update the other callbacks and calculate a new λ.

        :param epoch: The epoch number that just finished.
        :param logs: The history logs from the last epoch.

        """
        assert logs is not None, "Cannot work with empty logs!"
        self.logsHistory.append(logs)

        for head in self.heads:
            if epoch > 0:
                profileLoss, countsLoss, _, _ = self.getLosses(
                    epoch, head["head-name"])
            else:
                # In the first epoch, use the validation losses.
                # Since the actual losses include extremely high
                # values from the very first few batches.
                _, _, profileLoss, countsLoss = self.getLosses(
                    epoch, head["head-name"])

            λVar = head["INTERNAL_λ-variable"]
            curλ = λVar.read_value()
            self.λHistory[head["head-name"]].append(float(curλ))
            countsLossRaw = countsLoss / curλ
            if "counts-loss-frac-target" in head:
                # We want to update the loss.

                # Algebra for this calculation. Let t be the goal fraction,
                # λ₁ is the new loss weight, p is profile loss, c is (raw) counts loss.
                # f is the current counts loss fraction.
                # f ≡ λ₀ c / (λ₀ c + p)
                # We desire that t = f, so
                # t = λ₁ c / (λ₁ c + p)
                # λ₁ c t + p t = λ₁ c
                # λ₁ c t - λ₁ c = - p t
                # λ₁ c (t - 1) = - p t
                # λ₁ c (1 - t) = p t
                # λ₁ = p t / (c * (1 - t))
                target = head["counts-loss-frac-target"]
                correctedλ = target * profileLoss / \
                    (countsLossRaw * (1 - target))
                # Approach the new target weight slowly, using exponential damping.
                if epoch > 1:
                    # Use exponential damping after the second epoch.
                    newλ = curλ * (1 - self.aggression) + \
                        correctedλ * self.aggression
                else:
                    # Jump early in training.
                    newλ = correctedλ
                # If the weight would change by more than a factor of aggression, clamp it down.
                threshold = 2 if epoch > 1 else 10
                if (max(newλ, curλ) / min(newλ, curλ)) > threshold:
                    logUtils.debug("Large λ change detected. Old: {0:f}, new {1:f}"
                                   .format(curλ, newλ))
                    if newλ > curλ:
                        # λ₁ / λ₀ > 2
                        # make λ₁ = λ₀ * 2
                        newλ = curλ * threshold
                    else:
                        # λ₀ / λ₁ > 2
                        # make λ₁ = λ₀ / 2
                        newλ = curλ / threshold
                    logUtils.debug("Clamped new λ to {0:f}".format(newλ))
                    if threshold == 10:
                        # We jumped and had a large threshold - user should choose
                        # a better counts weight.
                        logUtils.warning("A large λ change was detected in the first epoch. "
                                         "Consider changing the starting counts-loss-weight for "
                                         "head {0:s} to a value near {1:f}".format(
                                             head["head-name"], correctedλ))
                # With current loss weight, what is the loss fraction due to counts?
                # (This doesn't enter our calculation, it's for logging.
                scaledCurFrac = countsLoss / (profileLoss + countsLoss)

                logUtils.debug(("countsLoss: head {0:s} λ₀ {1:f}, λ₁ {2:f} (A=1: {3:f})."
                               " frac {4:f}, goal {5:f}. Raw counts loss {6:f} (scaled {7:f})."
                                " Epoch {8:d}")
                               .format(head["head-name"], curλ, newλ,
                                       correctedλ, scaledCurFrac,
                                       head["counts-loss-frac-target"],
                                       countsLossRaw, countsLoss, epoch))

                λVar.assign(newλ)
        # We've updated the loss. But now we have to go mess with the callbacks so that
        # the increased loss value isn't interpreted as the model getting worse.
        self.resetCallbacks()
