import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def getCallbacks(early_stop, output_prefix):
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=early_stop,
                                        verbose=1,
                                        mode='min')

    filepath = "{}.checkpoint.model".format(output_prefix)
    checkpoint_callback = ModelCheckpoint(filepath,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='min')

    return [early_stop_callback, checkpoint_callback]

