import numpy as np

from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Activation, Dropout, GaussianNoise, Input, Lambda, PReLU
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.merge import add, concatenate, maximum
from keras.models import Model
from keras.optimizers import Adam

from keras.engine import Layer

import tensorflow as tf

K.set_image_dim_ordering('th')

class Multimodel(object):
    def __init__(
        self, input_modalities, output_modalities, output_weights, latent_dim, channels, patch_shape, to_process, scale=1):
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.latent_dim = latent_dim
        self.channels = channels
        self.scale = scale
        self.common_merge = 'max'
        self.output_weights = output_weights
        self.ind_outs = True
        self.fuse_outs = True
        self.num_emb = len(input_modalities) + 1
        self.patch_shape = patch_shape
        self.to_process = to_process

    def encoder_maker(self, modality, channels=1, to_process=True) :
        return self.ushape_network_maker(
            modality, self.scale, channels, self.latent_dim, 'enc', self.patch_shape, to_process)

    def decoder_maker(self, modality, channels=1) :
        inp, pred = self.ushape_network_maker(
            modality, self.scale, self.latent_dim, 1, 'dec', self.patch_shape, to_process=True)
        return Model(inputs=[inp], outputs=[pred], name='{}_{}'.format('dec', modality))

    def get_embedding_distance_outputs(self, embeddings):
        if len(self.inputs) == 1:
            print 'Skipping embedding distance outputs for unimodal model'
            return []

        outputs = list()

        ind_emb = embeddings[:-1]
        weighted_rep = embeddings[-1]

        all_emb_flattened = [new_flatten(emb) for emb in ind_emb]
        if len(ind_emb) > 1 :
            concat_emb = concatenate(all_emb_flattened, axis=1, name='em_concat')
        else :
            concat_emb = all_emb_flattened[0]

        outputs.append(concat_emb)
        print 'making output: em_concat', concat_emb, concat_emb.name

        fused_emb = new_flatten(weighted_rep, name='em_fused')
        outputs.append(fused_emb)

        return outputs

    def build(self):
        print 'Latent dimensions: ' + str(self.latent_dim)

        encoders = [self.encoder_maker(m, self.channels[i], self.to_process[i]) for i, m in enumerate(self.input_modalities)]

        ind_emb = [lr for (input, lr) in encoders]
        self.org_ind_emb = [lr for (input, lr) in encoders]
        self.inputs = [input for (input, lr) in encoders]

        assert self.common_merge == 'max'
        print 'Fuse latent representations using ' + str(self.common_merge)
        weighted_rep = maximum(ind_emb, name='combined_em') if len(self.inputs) > 1 else ind_emb[0]
        self.all_emb = ind_emb + [weighted_rep]

        self.decoders = [self.decoder_maker(m) for m in self.output_modalities]
        outputs = get_decoder_outputs(self.output_modalities, self.decoders, self.all_emb)

        outputs += self.get_embedding_distance_outputs(self.all_emb)

        print 'all outputs: ', [o.name for o in outputs]

        out_dict = {'em_%d_dec_%s' % (emi, dec): adhoc_loss for emi in range(self.num_emb) for dec in self.output_modalities}

        get_indiv_weight = lambda mod: self.output_weights[mod] if self.ind_outs else 0.0
        get_fused_weight = lambda mod: self.output_weights[mod] if self.fuse_outs else 0.0
        loss_weights = {}
        for dec in self.output_modalities:
            for emi in range(self.num_emb - 1):
                loss_weights['em_%d_dec_%s' % (emi, dec)] = get_indiv_weight(dec)
            loss_weights['em_%d_dec_%s' % (self.num_emb - 1, dec)] = get_fused_weight(dec)

        if len(self.inputs) > 1:
            out_dict['em_concat'] = embedding_distance
            loss_weights['em_concat'] = self.output_weights['concat']

            out_dict['em_fused'] = embedding_distance
            loss_weights['em_fused'] = 0.0

        print 'output dict: ', out_dict
        print 'loss weights: ', loss_weights

        self.model = Model(inputs=self.inputs, outputs=outputs)
        self.model.compile(optimizer='Adam', loss=out_dict, loss_weights=loss_weights)

    def get_inputs(self, modalities):
        return [self.inputs[self.input_modalities.index(mod)] for mod in modalities]

    def get_embeddings(self, modalities):
        assert set(modalities).issubset(set(self.input_modalities))
        ind_emb = [self.all_emb[self.input_modalities.index(mod)] for mod in modalities]
        org_ind_emb = [self.org_ind_emb[self.input_modalities.index(mod)] for mod in modalities]

        if len(ind_emb) > 1:
            fused_emb = maximum(ind_emb, name='fused_em')
        else:
            fused_emb = ind_emb[0]
        return ind_emb + [fused_emb]

    def get_input(self, modality):
        assert modality in self.input_modalities
        for l in self.model.layers:
            if l.name == 'enc_' + modality + '_input':
                return l.output
        return None

    def predict_z(self, input_modalities, X):
        embeddings = self.get_embeddings(input_modalities)
        inputs = [self.get_input(mod) for mod in input_modalities]
        partial_model = Model(inputs=inputs, outputs=embeddings)
        Z = partial_model.predict(X)
        assert len(Z) == len(embeddings)
        return Z

    def new_decoder_model(self, input_modalities, modality):
        if modality in self.output_modalities:
            print 'Using trained decoder'
            decoder = self.decoders[self.output_modalities.index(modality)]
        else:
            print 'Creating new decoder'
            decoder = self.decoder_maker(modality)
        inputs = [Input(shape=(self.latent_dim, None, None)) for i in range(len(input_modalities) + 1)]

        outputs = [decoder(inpt) for inpt in inputs]
        for outi, out in enumerate(outputs):
            out.name = 'em_%d_dec_%s' % (outi, modality)

        out_dict = {decoder.name: adhoc_loss}
        loss_weights = {decoder.name: 1.0}

        new_model = Model(inputs=inputs, outputs=outputs)
        new_model.compile(optimizer='Adam', loss=out_dict, loss_weights=loss_weights)

        return new_model

    def get_partial_model(self, input_modalities, output_modality):
        assert set(input_modalities).issubset(set(self.input_modalities))
        assert output_modality in self.output_modalities

        inputs = self.get_inputs(input_modalities)
        embeddings = self.get_embeddings(input_modalities)

        decoder = self.decoders[self.output_modalities.index(output_modality)]
        outputs = get_decoder_outputs([output_modality], [decoder], embeddings)
        outputs += self.get_embedding_distance_outputs(embeddings)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def new_encoder_model(self, modality, output_modalities):
        if modality in self.input_modalities:
            print 'Using trained encoder'
            input = self.inputs[self.input_modalities.index(modality)]
            lr = self.all_emb[self.input_modalities.index(modality)]
        else:
            print 'Creating new encoder'
            input, lr = self.encoder_maker(modality)

        decoders = [self.decoders[self.output_modalities.index(mod)] for mod in output_modalities]
        for d in decoders:
            d.trainable = False
        outputs = get_decoder_outputs(output_modalities, decoders, [lr])

        model = Model(input=[input], output=outputs)
        model.compile(optimizer=Adam(), loss={d.name: adhoc_loss for d in decoders},
                      loss_weights={d.name: 1.0 for d in decoders})
        return model

    def get_conv_fc(self, input, num_filters, modality, task='enc', level=1) :
        name_pattern = '{}_{}_{}{}'
        name_conv = name_pattern.format(task, modality, 'conv', level)
        name_act = name_pattern.format(task, modality, 'act', level)
        fc = Conv3D(num_filters, kernel_size=(1, 1, 1), name=name_conv)(input)
        return PReLU(name=name_act)(fc)

    def get_deconv_layer(self, input, num_filters, modality, task='enc', level=1) :
        name = '{}_{}_{}{}'.format(task, modality, 'dconv', level)
        return Conv3DTranspose(num_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2), name=name)(input)

    def get_res_conv_core(self, input, num_filters, modality, task='enc', level=1) :
        name_pattern = '{}_{}_{}{}{}'
        name_a = name_pattern.format(task, modality, 'conv', level, 'a')
        name_b = name_pattern.format(task, modality, 'conv', level, 'b')
        a = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same', name=name_a)(input)
        b = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same', name=name_b)(input)
        c = add([a, b], name=name_pattern.format(task, modality, 'sum', level, ''))
        return PReLU(name=name_pattern.format(task, modality, 'res', level, ''))(c)

    def get_max_pooling_layer(self, input, modality, task='enc', level=1) :
        name = '{}_{}_{}{}'.format(task, modality, 'pool', level)
        return MaxPooling3D(pool_size=(2, 2, 2), name=name)(input)

    def merge_add(self, input, modality, task='enc', level=1) :
        name_pattern = '{}_{}_{}{}'
        merged = add(input, name=name_pattern.format(task, modality, 'add', level))
        return PReLU(name=name_pattern.format(task, modality, 'act', level))(merged)

    def ushape_network_maker(
        self, modality, scale, channels, latent_dim, task, patch_shape, to_process=True):
        input_shape = (channels, ) + patch_shape
        
        inp = Input(shape=input_shape, name='{}_{}_{}{}'.format(task, modality, 'input', ''))

        conv1 = self.get_res_conv_core(inp, np.int16(32*scale), modality, task, 1)
        pool1 = self.get_max_pooling_layer(conv1, modality, task, 1)

        conv2 = self.get_res_conv_core(pool1, np.int16(64*scale), modality, task, 2)
        pool2 = self.get_max_pooling_layer(conv2, modality, task, 2)

        conv3 = self.get_res_conv_core(pool2, np.int16(128*scale), modality, task, 3)
        pool3 = self.get_max_pooling_layer(conv3, modality, task, 3)

        conv4 = self.get_res_conv_core(pool3, np.int16(256*scale), modality, task, 4)

        up1 = self.get_deconv_layer(conv4, np.int16(128*scale), modality, task, 5)
        conv5 = self.get_res_conv_core(up1, np.int16(128*scale), modality, task, 5)

        add35 = self.merge_add([conv3, conv5], modality, task, 6)
        conv6 = self.get_res_conv_core(add35, np.int16(128*scale), modality, task, 6)
        up2 = self.get_deconv_layer(conv6, np.int16(64*scale), modality, task, 6)

        add22 = self.merge_add([conv2, up2], modality, task, 7)
        conv7 = self.get_res_conv_core(add22, np.int16(64*scale), modality, task, 7)
        up3 = self.get_deconv_layer(conv7, np.int16(32*scale), modality, task, 7)

        add13 = self.merge_add([conv1, up3], modality, task, 8) if to_process == True else inp

        pred = self.get_conv_fc(add13, latent_dim, modality, task, 9)

        return inp, pred

def get_decoder_outputs(output_modalities, decoders, embeddings):
    assert len(output_modalities) == len(decoders)

    outputs = list()
    for di, decode in enumerate(decoders):
        for emi, em in enumerate(embeddings):
            out_em = decode(em)
            name = 'em_' + str(emi) + '_dec_' + output_modalities[di]
            l = Lambda(lambda x: x + 0, name=name)(out_em)
            outputs.append(l)
            print 'making output:', em, out_em, name

    return outputs

def embedding_distance(y_true, y_pred):
    return K.var(y_pred, axis=1)

def new_flatten(emb, name=''):
    l = Lambda(lambda x: K.batch_flatten(x))(emb)
    l = Lambda(lambda x: K.expand_dims(x, axis=1), name=name)(l)
    return l

def var(embeddings):
    emb = embeddings[0]
    shape = (emb.shape[1], emb.shape[2], emb.shape[3])
    sz = shape[0] * shape[1] * shape[2]

    flat_embs = [K.reshape(emb, (emb.shape[0], 1, sz)) for emb in embeddings]

    emb_var = K.var(K.concatenate(flat_embs, axis=1), axis=1, keepdims=True)

    return K.reshape(emb_var, embeddings[0].shape)

def mae(y_true, y_pred):
    return K.abs(K.batch_flatten(y_pred - y_true))

def adhoc_loss(y_true, y_pred) :
    mask = K.cast(K.not_equal(y_true, 0), 'float32')
    m = mae(y_true, y_pred*mask)
    cc = categorical_crossentropy(
        S((y_true - train_mean) / train_std),
        S((y_pred - train_mean) / train_std))
    return m + cc

import numpy as np

from keras import backend as K
from keras.layers import Activation, Input, PReLU, Flatten, Dense, Cropping3D, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.convolutional import Conv3DTranspose as Deconv3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_uresnet_model(input_shape, output_shape, num_classes=4, scale=1):
    input = Input(shape=input_shape)

    conv1 = get_res_conv_core(input, np.int32(scale*32))
    pool1 = get_max_pooling_layer(conv1)

    conv2 = get_res_conv_core(pool1, np.int32(scale*64))
    pool2 = get_max_pooling_layer(conv2)

    conv3 = get_res_conv_core(pool2, np.int32(scale*128))
    pool3 = get_max_pooling_layer(conv3)

    conv4 = get_res_conv_core(pool3, np.int32(scale*256))
    
    up1 = get_deconv_layer(conv4, np.int32(scale*128))
    conv5 = get_res_conv_core(up1, np.int32(scale*128))

    add35 = merge_add(conv3, conv5)
    conv6 = get_res_conv_core(add35, np.int32(scale*128))
    up2 = get_deconv_layer(conv6, np.int32(scale*64))

    add22 = merge_add(conv2, up2)
    conv7 = get_res_conv_core(add22, np.int32(scale*64))
    up3 = get_deconv_layer(conv7, np.int32(scale*32))

    add13 = merge_add(conv1, up3)
    conv8 = get_res_conv_core(add13, np.int32(scale*32))

    pred = get_conv_fc(conv8)
    pred = organise_output(pred, output_shape)

    return Model(inputs=[input], outputs=[pred])

def merge_add(a, b) :
    c = add([a, b])
    c = BatchNormalization(axis=1)(c)
    return PReLU()(c)

def get_res_conv_core(input, num_filters) :
    a = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same')(input)
    b = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same')(input)
    return merge_add(a, b)

def get_max_pooling_layer(input) :
    return MaxPooling3D(pool_size=(2, 2, 2))(input)

def get_deconv_layer(input, num_filters) :
    return Deconv3D(num_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input)

def get_conv_fc(input, num_filters=4) :
    fc = Conv3D(num_filters, kernel_size=(1, 1, 1))(input)

    return PReLU()(fc)

def organise_output(input, output_shape) :
    pred = Reshape((4, 32*32*32))(input)
    pred = Permute((2, 1))(pred)
    return Activation('softmax')(pred)

curr_patch_shape = (32, 32, 32)
S = generate_uresnet_model((1, ) + curr_patch_shape, (np.product(curr_patch_shape), 4))
S.load_weights('models/ag_segmenter.h5')
train_mean = 786.1171
train_std = 587.41235
