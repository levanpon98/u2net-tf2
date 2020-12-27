import tensorflow as tf


def conv2d(input, filters, dirate):
    conv_s1 = tf.keras.layers.Conv2D(filters=filters)