#!/usr/bin/env python3
"""A super simple program that displays a summary of your model and optionally saves an image."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
from bpreveal import utils
import argparse
if __name__ == "__main__":
    utils.setMemoryGrowth()


def main(modelFname: str, pngFile: str | None):
    """Read in the model named by modelFname, show it as text, and optionally save as a png."""
    model = utils.loadModel(modelFname)
    print(model.summary(expand_nested=True, show_trainable=True))
    if pngFile is not None:
        from tensorflow.keras.utils import plot_model
        plot_model(model, pngFile, show_shapes=True, show_dtype=True,
                show_layer_names=True, expand_nested=True, show_layer_activations=True)


def getParser() -> argparse.ArgumentParser:
    """Build the parser."""
    ap = argparse.ArgumentParser(description="Show a text description of your "
                                 "model and optionally save it to a png file.")
    ap.add_argument("--model", help="The name of the Keras model file to show.")
    ap.add_argument("--png", help="(optional) The name of the png-format image to save.")
    return ap


if __name__ == "__main__":
    args = getParser().parse_args()
    main(args.model, args.png)
