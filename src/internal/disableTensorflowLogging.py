"""A module you can import that does one thing: It makes Tensorflow shut up.

You can import this before you import tensorflow in your code, and all the warning
messages will go away.

(There are no functions to call here, everything happens globally when you import
this module.)
"""

import os
import warnings
import bpreveal.logUtils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
warnings.simplefilter("ignore")
bpreveal.logUtils.debug("Tensorflow logging successfully disabled.")
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
