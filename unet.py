import numpy as np

from keras import backend as K
from keras.layers import Activation, Input, PReLU
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.convolutional import Conv3DTranspose as UpConv3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_uresnet_model(input_shape, output_shape, scale=1) :
    input, pred = generate_unet_model(
        input_shape, output_shape[1], scale, True)
    pred = organise_output(pred, output_shape)

    return Model(inputs=[input], outputs=[pred])

def generate_segnet_model(input_shape, fc_layer_filters, scale, use_batchnorm=True) :
    input = Input(shape=input_shape)

    conv1 = get_core_ele(input, np.int32(scale*32), use_batchnorm)
    pool1 = get_pooling(conv1, np.int32(scale*32), use_batchnorm)

    conv2 = get_core_ele(pool1, np.int32(scale*64), use_batchnorm)
    pool2 = get_pooling(conv2, np.int32(scale*64), use_batchnorm)

    conv3 = get_core_ele(pool2, np.int32(scale*128), use_batchnorm)
    pool3 = get_pooling(conv3, np.int32(scale*128), use_batchnorm)

    conv4 = get_core_ele(pool3, np.int32(scale*256), use_batchnorm)
    
    up1 = get_upconv_layer(conv4, np.int32(scale*128))
    conv5 = get_core_ele(up1, np.int32(scale*128), use_batchnorm)

    conv6 = get_core_ele(conv5, np.int32(scale*128), use_batchnorm)
    up2 = get_upconv_layer(conv6, np.int32(scale*64))

    conv7 = get_core_ele(up2, np.int32(scale*64), use_batchnorm)
    up3 = get_upconv_layer(conv7, np.int32(scale*32))

    conv8 = get_core_ele(up3, np.int32(scale*32), use_batchnorm)

    pred = get_conv_fc(conv8, fc_layer_filters)

    return input, pred

def generate_unet_model(input_shape, fc_layer_filters, scale, use_batchnorm=True) :
    input = Input(shape=input_shape)

    conv1 = get_core_ele(input, np.int32(scale*32), use_batchnorm)
    pool1 = get_pooling(conv1, np.int32(scale*32), use_batchnorm)

    conv2 = get_core_ele(pool1, np.int32(scale*64), use_batchnorm)
    pool2 = get_pooling(conv2, np.int32(scale*64), use_batchnorm)

    conv3 = get_core_ele(pool2, np.int32(scale*128), use_batchnorm)
    pool3 = get_pooling(conv3, np.int32(scale*128), use_batchnorm)

    conv4 = get_core_ele(pool3, np.int32(scale*256), use_batchnorm)
    
    up1 = get_upconv_layer(conv4, np.int32(scale*128))
    conv5 = get_core_ele(up1, np.int32(scale*128), use_batchnorm)

    add35 = merge_add(conv3, conv5, use_batchnorm)
    conv6 = get_core_ele(add35, np.int32(scale*128), use_batchnorm)
    up2 = get_upconv_layer(conv6, np.int32(scale*64))

    add22 = merge_add(conv2, up2, use_batchnorm)
    conv7 = get_core_ele(add22, np.int32(scale*64), use_batchnorm)
    up3 = get_upconv_layer(conv7, np.int32(scale*32))

    add13 = merge_add(conv1, up3, use_batchnorm)
    conv8 = get_core_ele(add13, np.int32(scale*32), use_batchnorm)

    pred = get_conv_fc(conv8, fc_layer_filters)

    return input, pred

def merge_add(a, b, use_batchnorm) :
    c = add([a, b])
    c = BatchNormalization(axis=1)(c) if use_batchnorm else c
    return PReLU()(c)

def get_core_ele(input, num_filters, use_batchnorm=False) :
    a = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same')(input)
    b = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same')(input)
    return merge_add(a, b, use_batchnorm)

def get_pooling(input, num_filters, use_batchnorm) :
    a = Conv3D(num_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input)
    a = BatchNormalization(axis=1)(a) if use_batchnorm else a
    return PReLU()(a)

def get_upconv_layer(input, num_filters) :
    return UpConv3D(num_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input)

def get_conv_fc(input, num_filters) :
    fc = Conv3D(num_filters, kernel_size=(1, 1, 1))(input)
    return PReLU()(fc)

def organise_output(input, output_shape) :
    pred = Reshape((output_shape[1], output_shape[0]))(input)
    pred = Permute((2, 1))(pred)
    return Activation('softmax')(pred)