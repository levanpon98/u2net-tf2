
import tensorflow as tf


def bce_loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)

    loss0 = tf.keras.losses.binary_crossentropy(y_true, y_pred[0])
    loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred[1])
    loss2 = tf.keras.losses.binary_crossentropy(y_true, y_pred[2])
    loss3 = tf.keras.losses.binary_crossentropy(y_true, y_pred[3])
    loss4 = tf.keras.losses.binary_crossentropy(y_true, y_pred[4])
    loss5 = tf.keras.losses.binary_crossentropy(y_true, y_pred[5])
    loss6 = tf.keras.losses.binary_crossentropy(y_true, y_pred[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
