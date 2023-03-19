import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def getCallbacks(early_stop, output_prefix, plateauPatience):
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=early_stop,
                                        verbose=1,
                                        mode='min', 
                                        restore_best_weights=True)

    filepath = "{}.checkpoint.model".format(output_prefix)
    checkpoint_callback = ModelCheckpoint(filepath,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='min')
    plateauCallback = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=plateauPatience, verbose=1)
    return [early_stop_callback, checkpoint_callback, plateauCallback]

def tensorboardCallback(logDir):
    from tensorflow.keras.callbacks import TensorBoard
    return TensorBoard(log_dir=logDir,
                       histogram_freq = 1,
                       write_graph=True,
                       write_steps_per_second=True,
                       profile_batch=(1,2000))

