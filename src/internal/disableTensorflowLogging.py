"""This is a module you can import that does one thing: It makes Tensorflow shut up.

You can import this before you import tensorflow in your code, and all the warning
messages will go away.

(There are no functions to call here, everything happens globally when you import
this module.)
"""

import os
import bpreveal.logUtils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import warnings
warnings.simplefilter("ignore")
bpreveal.logUtils.debug("Tensorflow logging successfully disabled.")
