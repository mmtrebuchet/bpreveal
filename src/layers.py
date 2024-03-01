"""Custom layers that are needed for the various models."""
import tensorflow as tf
from tensorflow import keras


class LinearRegression(keras.layers.Layer):
    """A simple layer that performs a linear regression on its inputs.

    .. highlight:: none

    Implements the following formula::

        out = input * slope + offset

    where slope and offset are two (scalar) parameters.

    :param kwargs: Passed to the keras Layer initializer.
    """

    def __init__(self, **kwargs):
        """Construct."""
        super().__init__(**kwargs)
        slopeInit = tf.ones_initializer()
        self.slope = tf.Variable(
            initial_value=slopeInit(shape=(1,), dtype='float32'),
            trainable=True,
            name='slope')

        offsetInit = tf.zeros_initializer()
        self.offset = tf.Variable(
            initial_value=offsetInit(shape=(1,), dtype='float32'),
            trainable=True,
            name='offset')

    def call(self, inputs):
        """Actually perform the calculation."""
        return inputs * self.slope + self.offset
