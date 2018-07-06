import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, PReLU
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.convolutional import Conv3DTranspose as UpConv3D
from keras.layers.merge import add
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam

from unet import generate_unet_model, generate_uresnet_model

K.set_image_dim_ordering('th')

def generate_wnet_model(wparams, segparams):
    input_channels = wparams['input_channels']
    output_channels = wparams['output_channels']
    latent_channels = wparams['latent_channels']
    scale = wparams['scale']
    patch_shape = wparams['patch_shape']

    seg_model_filename = segparams['seg_model_filename']
    seg_model_params_filename = segparams['seg_model_params_filename']
    segmentation_classes = segparams['segmentation_classes']

    S, train_mean, train_std = load_segmentation_model(
        patch_shape, seg_model_filename, seg_model_params_filename, segmentation_classes)

    f1 = encoder_maker(input_channels, patch_shape, latent_channels, scale)
    f2 = encoder_maker(latent_channels, patch_shape, output_channels, scale)
    f = wnet_maker(f1, f2, (input_channels, ) + patch_shape)
    
    def combined_loss(y_true, y_pred) :
        mask = K.cast(K.not_equal(y_true, 0), 'float32')

        l1 = mae(y_true, y_pred * mask)
        h = categorical_crossentropy(
            S((y_true - train_mean) / train_std),
            S((y_pred - train_mean) / train_std))

        return l1 + h

    f.compile(optimizer='Adam', loss=combined_loss)

    return f

def load_segmentation_model(
    patch_shape, seg_model_filename, seg_model_params_filename, segmentation_classes) :
    input_shape = (1, ) + patch_shape
    output_shape = (np.product(patch_shape), segmentation_classes)
    train_params = np.load(seg_model_params_filename).item()
    S = generate_uresnet_model(input_shape, output_shape)
    # S.load_weights(seg_model_filename)
    
    return S, train_params['train_mean'], train_params['train_std']

def wnet_maker(f1, f2, input_shape) :
    input = Input(shape=input_shape)

    f1_out = f1(input)
    f2_out = f2(f1_out)

    return Model(inputs=[input], outputs=[f2_out])

def encoder_maker(input_channels, patch_shape, output_channels, scale) :
    input_shape = (input_channels, ) + patch_shape
    fc_layer_filters = output_channels
    use_batchnorm = False

    inp, pred = generate_unet_model(
        input_shape, fc_layer_filters, scale, use_batchnorm)

    return Model(inputs=[inp], outputs=[pred])

def decoder_maker(input_channels, patch_shape, output_channels, scale) :
    input_shape = (input_channels, ) + patch_shape
    fc_layer_filters = output_channels
    use_batchnorm = False

    inp, pred = generate_unet_model(
        input_shape, fc_layer_filters, scale, use_batchnorm)

    return Model(inputs=[inp], outputs=[pred])

def mae(y_true, y_pred) :
    return K.abs(K.batch_flatten(y_pred - y_true))