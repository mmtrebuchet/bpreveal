"""A simple set of functions that train with a curses display."""
from bpreveal.callbacks import getCallbacks


def trainModel(model, inputLength, outputLength,  # pylint: disable=unused-argument
               trainBatchGen, valBatchGen, epochs, earlyStop, outputPrefix,
               plateauPatience, heads):
    """Run the training."""
    callbacks = getCallbacks(earlyStop, outputPrefix, plateauPatience, heads,
                             trainBatchGen, valBatchGen)
    history = model.fit(trainBatchGen, epochs=epochs,
                        validation_data=valBatchGen, callbacks=callbacks,
                        verbose=0)
    # Turn the learning rate data into python floats, since they come as
    # numpy floats and those are not serializable.
    history.history["lr"] = [float(x) for x in history.history["lr"]]
    # Add the counts loss weight history to the history json.
    lossCallback = callbacks[3]
    history.history["counts-loss-weight"] = lossCallback.λHistory
    return history
