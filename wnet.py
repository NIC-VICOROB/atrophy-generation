import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, PReLU, Lambda
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv3DTranspose as UpConv3D
from keras.layers.merge import add, concatenate, multiply
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam

from unet import generate_unet_model, generate_uresnet_model, get_conv_fc, get_core_ele, generate_segnet_model

K.set_image_dim_ordering('th')

def generate_wnet_model(wparams, segparams):
    input_channels = wparams['input_channels']
    output_channels = wparams['output_channels']
    latent_channels = wparams['latent_channels']
    scale = wparams['scale']
    use_combined_loss = wparams['use_combined_loss']
    patch_shape = wparams['patch_shape']
    loss_weights = wparams['loss_weights']

    f2 = decoder_maker(
        input_channels, patch_shape, output_channels, scale[0], True)
    f3 = decoder_maker(
        input_channels, patch_shape, output_channels, scale[1], False)
    f4 = decoder_maker(
        2 * output_channels, patch_shape, output_channels, scale[2], True)
    
    input_vol = Input(shape=(1, ) + patch_shape)
    input_csf = Input(shape=(1, ) + patch_shape)

    f2_out = f2(concatenate(
        [multiply([input_vol, Lambda(lambda x:1-x)(input_csf)]), input_csf], axis=1))
    f3_out = f3(concatenate([multiply([input_vol, input_csf]), input_csf], axis=1))

    f_out = f4(concatenate([f2_out, f3_out], axis=1))
    
    f = Model(inputs=[input_vol, input_csf], outputs=[f2_out, f3_out, f_out])

    def mae_loss(y_true, y_pred) :
        mask = K.batch_flatten(K.cast(K.not_equal(y_true, 0), 'float32'))

        return mae(y_true, y_pred) * mask
    
    def mse_loss(y_true, y_pred) :
        mask = K.batch_flatten(K.cast(K.not_equal(y_true, 0), 'float32'))
    
        return mse(y_true, y_pred) * mask

    loss = [mae_loss, mae_loss, mae_loss]
    
    f.compile(optimizer='Adam', loss=loss, loss_weights=loss_weights)

    return f

def encoder_maker(input_channels, patch_shape, output_channels, scale) :
    input_shape = (input_channels, ) + patch_shape
    fc_layer_filters = output_channels
    use_batchnorm = False

    inp, pred = generate_unet_model(
        input_shape, fc_layer_filters, scale, use_batchnorm)

    return Model(inputs=[inp], outputs=[pred])

def decoder_maker(input_channels, patch_shape, output_channels, scale, use_skip_connections=True) :
    input_shape = (input_channels, ) + patch_shape
    fc_layer_filters = output_channels
    use_batchnorm = False

    if use_skip_connections :
        inp, pred = generate_unet_model(
            input_shape, fc_layer_filters, scale, use_batchnorm)
    else :
        inp, pred = generate_segnet_model(
            input_shape, fc_layer_filters, scale, use_batchnorm)

    return Model(inputs=[inp], outputs=[pred])

def mae(y_true, y_pred) :
    return K.abs(K.batch_flatten(y_pred - y_true))

def mse(y_true, y_pred) :
    return K.pow(K.batch_flatten(y_pred - y_true), 2)