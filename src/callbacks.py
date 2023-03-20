import numpy as np
import os
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def getCallbacks(earlyStop, outputPrefix, plateauPatience):
    logging.debug("Creating callbacks based on earlyStop {0:d}, outputPrefix {1:s}, plateauPatience {2:d}".format(\
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
    plateauCallback = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=plateauPatience, verbose=1)
    return [earlyStopCallback, checkpointCallback, plateauCallback]

def tensorboardCallback(logDir):
    logging.debug("Creating tensorboard callback in {0:s}".format(logDir))
    from tensorflow.keras.callbacks import TensorBoard
    return TensorBoard(log_dir=logDir,
                       histogram_freq = 1,
                       write_graph=True,
                       write_steps_per_second=True,
                       profile_batch=(1,2000))

