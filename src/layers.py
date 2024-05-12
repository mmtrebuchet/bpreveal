"""Custom layers that are needed for the various models."""
import tensorflow as tf
import tf_keras as keras


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

    def call(self, inputs, *args, **kwargs):
        """Actually perform the calculation.

        Variadic arguments are ignored.
        """
        return inputs * self.slope + self.offset
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
