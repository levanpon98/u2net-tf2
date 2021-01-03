import keras
import tensorflow as tf


def bce_loss(y_true, y_pred):
    # y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = keras.losses.binary_crossentropy(y_true, y_pred[0])
    loss1 = keras.losses.binary_crossentropy(y_true, y_pred[1])
    loss2 = keras.losses.binary_crossentropy(y_true, y_pred[2])
    loss3 = keras.losses.binary_crossentropy(y_true, y_pred[3])
    loss4 = keras.losses.binary_crossentropy(y_true, y_pred[4])
    loss5 = keras.losses.binary_crossentropy(y_true, y_pred[5])
    loss6 = keras.losses.binary_crossentropy(y_true, y_pred[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
