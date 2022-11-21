#import tensorflow as tf

import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)   # 参数详解：[]:shape;minval:范围下限；maxval:范围上限；
    return tf.Variable(initial, name=name)    # random_uniform是均匀分布


def zeros(input_dim, output_dim, name=None):
    """All zeros."""
    initial = tf.zeros((input_dim, output_dim), dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(input_dim, output_dim, name=None):
    """All zeros."""
    initial = tf.ones((input_dim, output_dim), dtype=tf.float32)
    return tf.Variable(initial, name=name)