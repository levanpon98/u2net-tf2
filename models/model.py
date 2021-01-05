import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D, Concatenate, Activation, \
    Add, Lambda
from tensorflow.keras import Model, Input


def resize_bilinear(inputs, scale=2):
    b, h, w, c = inputs.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    results = tf.image.resize_bilinear(inputs, [new_h, new_w])

    return results


def ConvBlock(inputs, out_ch=3, dirate=1):
    x = Conv2D(out_ch, (3, 3), strides=1, padding='same', dilation_rate=dirate)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def RSU7(inputs, mid_ch, out_ch):
    hx = inputs
    hxin = ConvBlock(hx, out_ch=out_ch, dirate=1)
    hx1 = ConvBlock(hxin, out_ch=mid_ch, dirate=1)

    hx = MaxPool2D(2, strides=(2, 2))(hx1)

    hx2 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx2)

    hx3 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx3)

    hx4 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx4)

    hx5 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx5)

    hx6 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx7 = ConvBlock(hx6, out_ch=mid_ch, dirate=2)

    hx6d = ConvBlock(Concatenate(axis=3)([hx7, hx6]), out_ch=mid_ch, dirate=1)
    hx6dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx6d)

    hx5d = ConvBlock(Concatenate(axis=3)([hx6dup, hx5]), out_ch=mid_ch, dirate=1)
    hx5dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx5d)

    hx4d = ConvBlock(Concatenate(axis=3)([hx5dup, hx4]), out_ch=mid_ch, dirate=1)
    hx4dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx4d)

    hx3d = ConvBlock(Concatenate(axis=3)([hx4dup, hx3]), out_ch=mid_ch, dirate=1)
    hx3dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx3d)

    hx2d = ConvBlock(Concatenate(axis=3)([hx3dup, hx2]), out_ch=mid_ch, dirate=1)
    hx2dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx2d)

    hx1d = ConvBlock(Concatenate(axis=3)([hx2dup, hx1]), out_ch=out_ch, dirate=1)

    out = Add()([hx1d, hxin])
    return out


def RSU6(inputs, mid_ch=12, out_ch=3):
    hx = inputs
    hxin = ConvBlock(hx, out_ch=out_ch, dirate=1)
    hx1 = ConvBlock(hxin, out_ch=mid_ch, dirate=1)

    hx = MaxPool2D(2, strides=(2, 2))(hx1)

    hx2 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx2)

    hx3 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx3)

    hx4 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx4)

    hx5 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx6 = ConvBlock(hx5, out_ch=mid_ch, dirate=2)

    hx5d = ConvBlock(Concatenate(axis=3)([hx6, hx5]), out_ch=mid_ch, dirate=1)
    hx5dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx5d)

    hx4d = ConvBlock(Concatenate(axis=3)([hx5dup, hx4]), out_ch=mid_ch, dirate=1)
    hx4dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx4d)

    hx3d = ConvBlock(Concatenate(axis=3)([hx4dup, hx3]), out_ch=mid_ch, dirate=1)
    hx3dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx3d)

    hx2d = ConvBlock(Concatenate(axis=3)([hx3dup, hx2]), out_ch=mid_ch, dirate=1)
    hx2dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx2d)

    hx1d = ConvBlock(Concatenate(axis=3)([hx2dup, hx1]), out_ch=out_ch, dirate=1)

    out = Add()([hx1d, hxin])
    return out


def RSU5(inputs, mid_ch=12, out_ch=3):
    hx = inputs
    hxin = ConvBlock(hx, out_ch=out_ch, dirate=1)
    hx1 = ConvBlock(hxin, out_ch=mid_ch, dirate=1)

    hx = MaxPool2D(2, strides=(2, 2))(hx1)

    hx2 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx2)

    hx3 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx3)

    hx4 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx5 = ConvBlock(hx4, out_ch=mid_ch, dirate=2)

    hx4d = ConvBlock(Concatenate(axis=3)([hx5, hx4]), out_ch=mid_ch, dirate=1)
    hx4dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx4d)

    hx3d = ConvBlock(Concatenate(axis=3)([hx4dup, hx3]), out_ch=mid_ch, dirate=1)
    hx3dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx3d)

    hx2d = ConvBlock(Concatenate(axis=3)([hx3dup, hx2]), out_ch=mid_ch, dirate=1)
    hx2dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx2d)

    hx1d = ConvBlock(Concatenate(axis=3)([hx2dup, hx1]), out_ch=out_ch, dirate=1)

    out = Add()([hx1d, hxin])
    return out


def RSU4(inputs, out_ch, mid_ch):
    hx = inputs
    hxin = ConvBlock(hx, out_ch=out_ch, dirate=1)
    hx1 = ConvBlock(hxin, out_ch=mid_ch, dirate=1)

    hx = MaxPool2D(2, strides=(2, 2))(hx1)

    hx2 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx = MaxPool2D(2, strides=(2, 2))(hx2)

    hx3 = ConvBlock(hx, out_ch=mid_ch, dirate=1)
    hx4 = ConvBlock(hx3, out_ch=mid_ch, dirate=1)

    hx3d = ConvBlock(Concatenate(axis=3)([hx4, hx3]), out_ch=mid_ch, dirate=1)
    hx3dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx3d)

    hx2d = ConvBlock(Concatenate(axis=3)([hx3dup, hx2]), out_ch=mid_ch, dirate=1)
    hx2dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx2d)

    hx1d = ConvBlock(Concatenate(axis=3)([hx2dup, hx1]), out_ch=out_ch, dirate=1)

    out = Add()([hx1d, hxin])
    return out


def RSU4F(inputs, mid_ch=12, out_ch=3):
    hx = inputs
    hxin = ConvBlock(hx, out_ch=out_ch, dirate=1)

    hx1 = ConvBlock(hxin, out_ch=mid_ch, dirate=1)
    hx2 = ConvBlock(hx1, out_ch=mid_ch, dirate=2)
    hx3 = ConvBlock(hx2, out_ch=mid_ch, dirate=4)
    hx4 = ConvBlock(hx3, out_ch=mid_ch, dirate=8)
    hx3d = ConvBlock(Concatenate(axis=3)([hx4, hx3]), out_ch=mid_ch, dirate=4)
    hx2d = ConvBlock(Concatenate(axis=3)([hx3d, hx2]), out_ch=mid_ch, dirate=2)
    hx1d = ConvBlock(Concatenate(axis=3)([hx2d, hx1]), out_ch=out_ch, dirate=1)

    out = Add()([hx1d, hxin])
    return out


def stack(inputs):
    return tf.stack(inputs)


def U2NET(inputs, out_ch=1):
    hx = inputs

    hx1 = RSU7(hx, mid_ch=32, out_ch=64)
    hx = MaxPool2D((2, 2), 2)(hx1)

    hx2 = RSU6(hx, mid_ch=32, out_ch=128)
    hx = MaxPool2D((2, 2), 2)(hx2)

    hx3 = RSU5(hx, mid_ch=64, out_ch=256)
    hx = MaxPool2D((2, 2), 2)(hx3)

    hx4 = RSU4(hx, mid_ch=128, out_ch=512)
    hx = MaxPool2D((2, 2), 2)(hx4)

    hx5 = RSU4F(hx, mid_ch=256, out_ch=512)
    hx = MaxPool2D((2, 2), 2)(hx5)

    hx6 = RSU4F(hx, mid_ch=256, out_ch=512)

    hx6up = Lambda(lambda x: resize_bilinear(x, scale=2))(hx6)
    side6 = Lambda(lambda x: resize_bilinear(x, scale=32))(Conv2D(out_ch, (3, 3), padding='same')(hx6))

    hx5d = RSU4F(Concatenate(axis=3)([hx6up, hx5]), mid_ch=256, out_ch=512)
    hx5dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx5d)
    side5 = Lambda(lambda x: resize_bilinear(x, scale=16))(Conv2D(out_ch, (3, 3), padding='same')(hx5d))

    hx4d = RSU4(Concatenate(axis=3)([hx5dup, hx4]), mid_ch=128, out_ch=256)
    hx4dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx4d)
    side4 = Lambda(lambda x: resize_bilinear(x, scale=8))(Conv2D(out_ch, (3, 3), padding='same')(hx4d))

    hx3d = RSU5(Concatenate(axis=3)([hx4dup, hx3]), mid_ch=64, out_ch=128)
    hx3dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx3d)
    side3 = Lambda(lambda x: resize_bilinear(x, scale=4))(Conv2D(out_ch, (3, 3), padding='same')(hx3d))

    hx2d = RSU6(Concatenate(axis=3)([hx3dup, hx2]), mid_ch=32, out_ch=64)
    hx2dup = Lambda(lambda x: resize_bilinear(x, scale=2))(hx2d)
    side2 = Lambda(lambda x: resize_bilinear(x, scale=2))(Conv2D(out_ch, (3, 3), padding='same')(hx2d))

    hx1d = RSU7(Concatenate(axis=3)([hx2dup, hx1]), mid_ch=16, out_ch=64)
    side1 = Conv2D(out_ch, (3, 3), padding='same')(hx1d)

    fused_output = Conv2D(out_ch, (1, 1), padding='same')(
        Concatenate(axis=3)([side1, side2, side3, side4, side5, side6]))

    fused_output = Activation('sigmoid')(fused_output)
    side1 = Activation('sigmoid')(side1)
    side2 = Activation('sigmoid')(side2)
    side3 = Activation('sigmoid')(side3)
    side4 = Activation('sigmoid')(side4)
    side5 = Activation('sigmoid')(side5)
    side6 = Activation('sigmoid')(side6)

    outputs = Lambda(lambda x: stack(x))([fused_output, side1, side2, side3, side4, side5, side6])
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    inputs = Input(shape=(320, 320, 3))
    model = U2NET(inputs)
    model.summary()
