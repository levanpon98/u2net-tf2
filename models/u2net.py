"""
Description       : 
Author            : LE VAN PON
Maintainer        :
Date              : 02/01/2021
Version           : 1.0
Usage             : 
Notes             :
"""
import tensorflow as tf


def conv_block(inputs, out_channels, dirate=1, name=''):
    with tf.variable_scope(f'conv2d_{name}', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(inputs, out_channels, 3, strides=1, padding='same', dilation_rate=dirate)
        bn = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(bn)

    return relu


def resize_bilinear(inputs, scale=2):
    b, h, w, c = inputs.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    results = tf.image.resize_bilinear(inputs, [new_h, new_w])

    return results


def rsu7(inputs, mid_channels, out_channels, name='rsu7'):
    with tf.variable_scope('rsu7_' + name, reuse=tf.AUTO_REUSE):
        hx = inputs
        hxin = conv_block(hx, out_channels=out_channels, dirate=1, name='b0')
        hx1 = conv_block(hxin, out_channels=mid_channels, dirate=1, name='b1')

        hx = tf.layers.max_pooling2d(hx1, 2, 2, name='max_pooling_1')

        hx2 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b2')
        hx = tf.layers.max_pooling2d(hx2, 2, 2, name='max_pooling_2')

        hx3 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b3')
        hx = tf.layers.max_pooling2d(hx3, 2, 2, name='max_pooling_3')

        hx4 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b4')
        hx = tf.layers.max_pooling2d(hx4, 2, 2, name='max_pooling_4')

        hx5 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b5')
        hx = tf.layers.max_pooling2d(hx5, 2, 2, name='max_pooling_5')

        hx6 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b6')
        hx7 = conv_block(hx6, out_channels=mid_channels, dirate=2, name='b7')

        hx6d = conv_block(tf.concat([hx7, hx6], axis=3), out_channels=mid_channels, dirate=1, name='hx6d_concat')
        hx6dup = resize_bilinear(hx6d)

        hx5d = conv_block(tf.concat([hx6dup, hx5], axis=3), out_channels=mid_channels, dirate=1, name='hx5d_concat')
        hx5dup = resize_bilinear(hx5d)

        hx4d = conv_block(tf.concat([hx5dup, hx4], axis=3), out_channels=mid_channels, dirate=1, name='hx4d_concat')
        hx4dup = resize_bilinear(hx4d)

        hx3d = conv_block(tf.concat([hx4dup, hx3], axis=3), out_channels=mid_channels, dirate=1, name='hx3d_concat')
        hx3dup = resize_bilinear(hx3d)

        hx2d = conv_block(tf.concat([hx3dup, hx2], axis=3), out_channels=mid_channels, dirate=1, name='hx2d_concat')
        hx2dup = resize_bilinear(hx2d)

        hx1d = conv_block(tf.concat([hx2dup, hx1], axis=3), out_channels=out_channels, dirate=1, name='hx1d_concat')

    return hx1d + hxin


def rsu6(inputs, mid_channels, out_channels, name='rsu6'):
    with tf.variable_scope('rsu6_' + name, reuse=tf.AUTO_REUSE):
        hx = inputs
        hxin = conv_block(hx, out_channels=out_channels, dirate=1, name='b0')
        hx1 = conv_block(hxin, out_channels=mid_channels, dirate=1, name='b1')

        hx = tf.layers.max_pooling2d(hx1, 2, 2, name='max_pooling_1')

        hx2 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b2')
        hx = tf.layers.max_pooling2d(hx2, 2, 2, name='max_pooling_2')

        hx3 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b3')
        hx = tf.layers.max_pooling2d(hx3, 2, 2, name='max_pooling_3')

        hx4 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b4')
        hx = tf.layers.max_pooling2d(hx4, 2, 2, name='max_pooling_4')

        hx5 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b5')
        hx6 = conv_block(hx5, out_channels=mid_channels, dirate=2, name='b6')

        hx5d = conv_block(tf.concat([hx6, hx5], axis=3), out_channels=mid_channels, dirate=1, name='hx5d_concat')
        hx5dup = resize_bilinear(hx5d)

        hx4d = conv_block(tf.concat([hx5dup, hx4], axis=3), out_channels=mid_channels, dirate=1, name='hx4d_concat')
        hx4dup = resize_bilinear(hx4d)

        hx3d = conv_block(tf.concat([hx4dup, hx3], axis=3), out_channels=mid_channels, dirate=1, name='hx3d_concat')
        hx3dup = resize_bilinear(hx3d)

        hx2d = conv_block(tf.concat([hx3dup, hx2], axis=3), out_channels=mid_channels, dirate=1, name='hx2d_concat')
        hx2dup = resize_bilinear(hx2d)

        hx1d = conv_block(tf.concat([hx2dup, hx1], axis=3), out_channels=out_channels, dirate=1, name='hx1d_concat')

    return hx1d + hxin


def rsu5(inputs, mid_channels, out_channels, name='rsu5'):
    with tf.variable_scope('rsu5_' + name, reuse=tf.AUTO_REUSE):
        hx = inputs
        hxin = conv_block(hx, out_channels=out_channels, dirate=1, name='b0')
        hx1 = conv_block(hxin, out_channels=mid_channels, dirate=1, name='b1')

        hx = tf.layers.max_pooling2d(hx1, 2, 2, name='max_pooling_1')

        hx2 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b2')
        hx = tf.layers.max_pooling2d(hx2, 2, 2, name='max_pooling_2')

        hx3 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b3')
        hx = tf.layers.max_pooling2d(hx3, 2, 2, name='max_pooling_3')

        hx4 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b4')
        hx5 = conv_block(hx, out_channels=mid_channels, dirate=2, name='b5')

        hx4d = conv_block(tf.concat([hx5, hx4], axis=3), out_channels=mid_channels, dirate=1, name='hx4d_concat')
        hx4dup = resize_bilinear(hx4d)

        hx3d = conv_block(tf.concat([hx4dup, hx3], axis=3), out_channels=mid_channels, dirate=1, name='hx3d_concat')
        hx3dup = resize_bilinear(hx3d)

        hx2d = conv_block(tf.concat([hx3dup, hx2], axis=3), out_channels=mid_channels, dirate=1, name='hx2d_concat')
        hx2dup = resize_bilinear(hx2d)

        hx1d = conv_block(tf.concat([hx2dup, hx1], axis=3), out_channels=out_channels, dirate=1, name='hx1d_concat')

    return hx1d + hxin


def rsu4(inputs, mid_channels, out_channels, name='rsu4'):
    with tf.variable_scope('rsu4_' + name, reuse=tf.AUTO_REUSE):
        hx = inputs
        hxin = conv_block(hx, out_channels=out_channels, dirate=1, name='b0')
        hx1 = conv_block(hxin, out_channels=mid_channels, dirate=1, name='b1')

        hx = tf.layers.max_pooling2d(hx1, 2, 2, name='max_pooling_1')

        hx2 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b2')
        hx = tf.layers.max_pooling2d(hx2, 2, 2, name='max_pooling_2')

        hx3 = conv_block(hx, out_channels=mid_channels, dirate=1, name='b3')
        hx4 = conv_block(hx, out_channels=mid_channels, dirate=2, name='b4')

        hx3d = conv_block(tf.concat([hx4, hx3], axis=3), out_channels=mid_channels, dirate=1, name='hx3d_concat')
        hx3dup = resize_bilinear(hx3d)

        hx2d = conv_block(tf.concat([hx3dup, hx2], axis=3), out_channels=mid_channels, dirate=1, name='hx2d_concat')
        hx2dup = resize_bilinear(hx2d)

        hx1d = conv_block(tf.concat([hx2dup, hx1], axis=3), out_channels=out_channels, dirate=1, name='hx1d_concat')

    return hx1d + hxin


def rsu4f(inputs, mid_channels, out_channels, name='rsu4f'):
    with tf.variable_scope('rsu4f_' + name, reuse=tf.AUTO_REUSE):
        hx = inputs
        hxin = conv_block(hx, out_channels=out_channels, dirate=1, name='hxin')

        hx1 = conv_block(hxin, out_channels=mid_channels, dirate=1, name='hx1')
        hx2 = conv_block(hx1, out_channels=mid_channels, dirate=2, name='hx2')
        hx3 = conv_block(hx2, out_channels=mid_channels, dirate=4, name='hx3')
        hx4 = conv_block(hx3, out_channels=mid_channels, dirate=8, name='hx4')
        hx3d = conv_block(tf.concat([hx4, hx3], axis=3), out_channels=mid_channels, dirate=4, name='hx3_concat')
        hx2d = conv_block(tf.concat([hx3d, hx2], axis=3), out_channels=mid_channels, dirate=2, name='hx2_concat')
        hx1d = conv_block(tf.concat([hx2d, hx1], axis=3), out_channels=out_channels, dirate=1, name='hx1_concat')

    return hx1d + hxin


def u2net(inputs, out_channels=1):
    with tf.variable_scope('u2net', reuse=tf.AUTO_REUSE):
        hx = inputs
        hx1 = rsu7(hx, 32, 64, name='hx_1')
        hx = tf.layers.max_pooling2d(hx1, 2, 2, name='pool12')

        hx2 = rsu6(hx, 32, 128, name='hx_2')
        hx = tf.layers.max_pooling2d(hx2, 2, 2, name='pool23')

        hx3 = rsu5(hx, 64, 256, name='hx_3')
        hx = tf.layers.max_pooling2d(hx3, 2, 2, name='pool34')

        hx4 = rsu4(hx, 128, 512, name='hx_4')
        hx = tf.layers.max_pooling2d(hx4, 2, 2, name='pool45')

        hx5 = rsu4f(hx, 256, 512, name='hx_5')
        hx = tf.layers.max_pooling2d(hx5, 2, 2, name='pool56')

        hx6 = rsu4f(hx, 256, 512, name='hx_6')
        hx6up = resize_bilinear(hx6)
        hx6_conv = tf.layers.conv2d(hx6, filters=out_channels, kernel_size=(3, 3), padding='same')
        side6 = tf.keras.layers.UpSampling2D(size=(32, 32))(hx6_conv)

        hx5d = rsu4f(tf.concat([hx6up, hx5], axis=3), 256, 512, name='hx5d')
        hx5dup = resize_bilinear(hx5d)
        hx5d_conv = tf.layers.conv2d(hx5d, filters=out_channels, kernel_size=(3, 3), padding='same')
        side5 = tf.keras.layers.UpSampling2D(size=(16, 16))(hx5d_conv)

        hx4d = rsu4(tf.concat([hx5dup, hx4], axis=3), 128, 256, name='hx4d')
        hx4dup = resize_bilinear(hx4d)
        hx4d_conv = tf.layers.conv2d(hx4d, filters=out_channels, kernel_size=(3, 3), padding='same')
        side4 = tf.keras.layers.UpSampling2D(size=(8, 8))(hx4d_conv)

        hx3d = rsu5(tf.concat([hx4dup, hx3], axis=3), 64, 128, name='hx3d')
        hx3dup = resize_bilinear(hx3d)
        hx3d_conv = tf.layers.conv2d(hx3d, filters=out_channels, kernel_size=(3, 3), padding='same')
        side3 = tf.keras.layers.UpSampling2D(size=(4, 4))(hx3d_conv)

        hx2d = rsu6(tf.concat([hx3dup, hx2], axis=3), 32, 64, name='hx2d')
        hx2dup = resize_bilinear(hx2d)
        hx2d_conv = tf.layers.conv2d(hx2d, filters=out_channels, kernel_size=(3, 3), padding='same')
        side2 = resize_bilinear(hx2d_conv)

        hx1d = rsu7(tf.concat([hx2dup, hx1], axis=3), 16, 64, name='hx1d')
        side1 = tf.layers.conv2d(hx1d, filters=out_channels, kernel_size=(3, 3), padding='same')

        fused_output = tf.layers.conv2d(
            tf.concat([side1, side2, side3, side4, side5, side6], axis=3),
            filters=out_channels,
            kernel_size=(1, 1),
            padding='same'
        )

        fused_output_sig = tf.nn.sigmoid(fused_output)
        side1_sig = tf.nn.sigmoid(side1)
        side2_sig = tf.nn.sigmoid(side2)
        side3_sig = tf.nn.sigmoid(side3)
        side4_sig = tf.nn.sigmoid(side4)
        side5_sig = tf.nn.sigmoid(side5)
        side6_sig = tf.nn.sigmoid(side6)

    return tf.stack([fused_output_sig, side1_sig, side2_sig, side3_sig, side4_sig, side5_sig, side6_sig])


# if __name__ == '__main__':
#     input = tf.constant(shape=(1, 320, 320, 3), value=255, dtype=tf.float32)
#     # print(resize_bilinear(input))
#     out = u2net(input, 32)
#     print(out.shape)
