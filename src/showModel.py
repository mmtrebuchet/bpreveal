#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
import json
import tensorflow as tf
import utils
utils.setMemoryGrowth()
from tensorflow import keras
from keras.models import load_model
import losses



def main(modelFname, pngFile):
    model = load_model(modelFname, custom_objects = {'multinomialNll' : losses.multinomialNll})
    print(model.summary(expand_nested=True, show_trainable=True))
    if(pngFile is not None):
        from tensorflow.keras.utils import plot_model
        plot_model(model, pngFile, show_shapes=True, show_dtype=True, 
                show_layer_names=True, expand_nested=True, show_layer_activations=True)

if (__name__ == "__main__"):
    import sys
    if(len(sys.argv) > 2):
        pngFile = sys.argv[2]
    else:
        pngFile = None
    main(sys.argv[1], pngFile)


