import logging
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    ReduceLROnPlateau, Callback
import re


def getCallbacks(earlyStop: int, outputPrefix: str, plateauPatience: int, heads):
    logging.debug("Creating callbacks based on earlyStop "
                  "{0:d}, outputPrefix {1:s}, plateauPatience {2:d}".format(
                      earlyStop, outputPrefix, plateauPatience))
    earlyStopCallback = EarlyStopping(monitor='val_loss',
                                      patience=earlyStop,
                                      verbose=1,
                                      mode='min',
                                      restore_best_weights=True)

    filepath = "{}.checkpoint.model".format(outputPrefix)
    checkpointCallback = ModelCheckpoint(filepath,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='min')
    plateauCallback = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=plateauPatience, verbose=1)
    adaptiveLossCallback = ApplyAdaptiveCountsLoss(heads, 0.3, plateauCallback,
                                                   earlyStopCallback, checkpointCallback)
    return [earlyStopCallback, checkpointCallback, plateauCallback, adaptiveLossCallback]


class ApplyAdaptiveCountsLoss(Callback):
    """This updates the counts loss weights in your model on the fly, so that you can specify a
    target fraction of the loss that is due to counts, and the model will automatically update
    the weight. It currently does not update profile weights, so you can't yet automatically
    balance the losses in a multitask model.
    (You can still manually specify them in the config json, of course!)"""
    # Straight from the json, with "INTERNAL_counts-loss-weight-variable".
    heads: dict
    # 0 = don't change weights, 1 = ignore history, change weights very fast.
    aggression: float
    # The logs at each epoch.
    logsHistory: list[dict]
    # The history of the counts loss weights for each head. (key is head-name)
    weightHistory: dict[str, list]

    def __init__(self, heads: dict, aggression: float,
                 lrPlateauCallback, earlyStopCallback, checkpointCallback):
        """Heads is straight from the json, except with "INTERNAL_counts-loss-weight-variable"
        entries in each head. If present, these variables are the Keras Variables that contain the
        loss weights. These will be updated in this callback.
        aggression is a number from 0 to 1. It determines how aggressively the counts loss
        will be re-weighted. Lower values indicate slower changes, and a value of 1.0 means that
        the old loss should be completely discarded at each iteration.
        (If the newly calculated loss weight is ever off by over a factor of two, it's clamped to
        be exactly twice (or half) of the old weight, so that gross instability in the early stages
        doesn't cause the model to explode.)
        The three callbacks are the others used in the model, and should be run BEFORE
        this callback is executed.
        This callback messes with their internal state (naughty, naughty!) because changing the
        loss weights could cause the model's loss value to go up even though the model hasn't
        gotten any worse.
        """
        super().__init__()
        self.heads = heads
        logging.debug(heads)
        self.aggression = aggression
        self.lrPlateauCallback = lrPlateauCallback
        self.earlyStopCallback = earlyStopCallback
        self.checkpointCallback = checkpointCallback
        self.logsHistory = []
        self.weightHistory = dict()
        for head in heads:
            self.weightHistory[head["head-name"]] = []

    def getLosses(self, epoch: int, headName: str) -> tuple[float, float, float, float]:
        """What were the profile, counts, val_profile, and val_counts (in that order)
        losses at the given epoch? The counts weight returned from this function
        uses the counts weight in use at the given epoch.
        This method assumes that the validation and training losses
        match some regexes, and these are based on layer names. If someone
        got really goofy with head names (e.g., one head named "x" and another
        named "profile_x", then this could get messed up.
        Returns a tuple of floats, representing the four losses
        (profile, counts, val_profile, val_counts) at the given epoch.
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
        # Using the current heads, determine what the loss would have been at the given epoch.
        # (Since we store the loss history, this is pretty easy.)
        logs = self.logsHistory[epoch]
        valTotalLoss = 0
        for head in self.heads:
            _, _, vpl, vcl = self.getLosses(epoch, head["head-name"])
            valTotalLoss += vpl * float(head["profile-loss-weight"])

            weightVar = head["INTERNAL_counts-loss-weight-variable"]
            curWeight = weightVar.read_value()
            oldWeight = self.weightHistory[head["head-name"]][epoch]
            valTotalLoss += vcl * curWeight / oldWeight
        logging.debug("Calculated new loss of {0:f} on epoch {1:d}, with original loss {2:f}"
                      .format(valTotalLoss, epoch, logs["val_loss"]))
        return valTotalLoss

    def resetCallbacks(self):
        """This is the squirreliest method here. The other callbacks that track model progression
        track the loss of the model at the record-setting epoch. But the definition of loss itself
        is changing during the training, so we need to update their stored idea of what the model's
        loss was during the training.
        For example, consider the following scenario:
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
        callback thinks that the loss of 14 on epoch 4 is an improvement."""

        # Find which epoch set the record.

        recordEpoch = self.earlyStopCallback.best_epoch
        # Now, set the callbacks to have a corrected loss at the record-setting epoch.
        correctedLoss = self.whatWouldValLossBe(recordEpoch)
        logging.debug("Resetting callbacks from old loss {0:f} to new loss {1:f}"
                      .format(self.lrPlateauCallback.best, correctedLoss))
        self.lrPlateauCallback.best = correctedLoss
        self.earlyStopCallback.best = correctedLoss
        self.checkpointCallback.best = correctedLoss

    def on_epoch_end(self, epoch, logs=None):
        assert logs is not None, "Cannot work with empty logs!"
        self.logsHistory.append(logs)

        for head in self.heads:
            profileLoss, countsLoss, _, _ = self.getLosses(epoch, head["head-name"])
            weightVar = head["INTERNAL_counts-loss-weight-variable"]
            curWeight = weightVar.read_value()
            self.weightHistory[head["head-name"]].append(float(curWeight))
            countsLossRaw = countsLoss / curWeight
            if "counts-loss-frac-target" in head and epoch > 1:
                # We want to update the loss.

                # Algebra for this calculation. Let t be the goal fraction,
                # α is the new loss weight, p is profile loss, c is (raw) counts loss.
                # f is the current counts loss fraction.
                # f ≡ α c / (α c + p)
                # We desire that t = f, so
                # t = α c / (α c + p)
                # α c t + p t = α c
                # α c t - α c = - p t
                # α c (t - 1) = - p t
                # α c (1 - t) = p t
                # α = p t / (c * (1 - t))
                target = head["counts-loss-frac-target"]
                correctedWeight = target * profileLoss / (countsLossRaw * (1 - target))
                # Approach the new target weight slowly, using exponential damping.
                newWeight = curWeight * (1 - self.aggression) + correctedWeight * self.aggression
                # If the weight would change by more than a factor of aggression, clamp it down.

                if (max(newWeight, curWeight) / min(newWeight, curWeight)) > 2:
                    logging.debug("Large weight change detected. Old: {0:f}, new {1:f}"
                                  .format(curWeight, newWeight))
                    if newWeight > curWeight:
                        # N / C > 2
                        # make N = C * 2
                        newWeight = curWeight * 2
                    else:
                        # C / N > 2
                        # make N = C / 2
                        newWeight = curWeight / 2
                    logging.debug("Clamped new weight to {0:f}".format(newWeight))

                # With current loss weight, what is the loss fraction due to counts?
                # (This doesn't enter our calculation, it's for logging.
                scaledCurFrac = countsLoss / (profileLoss + countsLoss)

                logging.debug(("countsLoss: head {0:s} old cw {1:f}, new cw {2:f} (A=1: {3:f})."
                               " frac {4:f}, goal {5:f}. Raw counts loss {6:f} (scaled {7:f})")
                              .format(head["head-name"], curWeight, newWeight,
                                   correctedWeight, scaledCurFrac,
                                   head["counts-loss-frac-target"],
                                   countsLossRaw, countsLoss))

                weightVar.assign(newWeight)
        # We've updated the loss. But now we have to go mess with the callbacks so that
        # the increased loss value isn't interpreted as the model getting worse.
        self.resetCallbacks()


def tensorboardCallback(logDir: str):
    logging.debug("Creating tensorboard callback in {0:s}".format(logDir))
    from tensorflow.keras.callbacks import TensorBoard
    return TensorBoard(log_dir=logDir,
                       histogram_freq=1,
                       write_graph=True,
                       write_steps_per_second=True,
                       profile_batch=(1, 2000))
