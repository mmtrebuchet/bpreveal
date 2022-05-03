#This file contains the custom layers that are needed for the various models.

import tensorflow as tf
from tensorflow import keras

class LinearRegression(keras.layers.Layer):
    """A simple layer that performs a linear regression on its inputs:
    out = input * slope + offset
    where slope and offset are two (scalar) parameters."""

    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__(**kwargs)
        slope_init = tf.ones_initializer()
        self.slope = tf.Variable(
                initial_value = slope_init(shape=(1,), dtype='float32'),
                trainable=True,
                name='slope')
        
        offset_init = tf.zeros_initializer()
        self.offset = tf.Variable(
                initial_value = offset_init(shape=(1,), dtype='float32'),
                trainable=True,
                name='offset')

    def call(self, inputs):
        return inputs * self.slope + self.offset

