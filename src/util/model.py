'''
Definition of a model architecture und loss function in Keras.

Author: David Kostka
Date: 15.02.2021
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape, Softmax, concatenate, DepthwiseConv2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
import tensorflow_addons as tfa
import util.utils as util
import numpy as np
import tensorflow as tf


def FeatureModule(x, layers, filters, sep=True):
    sep=False
    if sep:
        for _ in range(layers):
            x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1,1), use_bias=False, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)

            x = Conv2D(filters, (1, 1), strides=(1,1), use_bias=False, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
    else:
        for _ in range(layers):
            x = Conv2D(filters, (3, 3), strides=(1,1), use_bias=False, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
    return x


def ScaleModule(x, filters, sep=True):
    x = Conv2D(filters, (3, 3), strides=(2,2), use_bias=False, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def FeatureExtractor(input_image, layer_list):
    x = input_image
    nr_of_blocks = len(layer_list)

    for i, (layers, filters) in enumerate(layer_list):
        if i == 0:
            x = FeatureModule(x, 1, filters, sep=False)
            x = FeatureModule(x, layers-1, filters)
            x = ScaleModule(x, filters)
        elif i == nr_of_blocks-1:
            x = FeatureModule(x, layers, filters, sep=False)
        else:
            x = FeatureModule(x, layers, filters)
            x = ScaleModule(x, filters)

    return x


def MultiNaoModel(img_size, layer_list, batch_size):
    input_image = Input(shape=(*img_size, 1))
    x = FeatureExtractor(input_image, layer_list)

    output = Conv2D(5*2, (1, 1), strides=(1,1), padding='same')(x)
    
    model = Model(inputs=input_image, outputs=output)

    return model

def multi_nao_loss(anchors, xy_weight=5, wh_weight=10):
    def loss(y_true, y_pred):
        xy_pred, wh_pred, conf_pred = util.activate_feature_map(y_pred, anchors)

        conf_true = y_true[..., 4:5]
        xy_true = y_true[..., 0:2]
        wh_true = y_true[..., 2:4]
        print(conf_pred.shape)
        print(conf_true.shape)
        #obj_mask = tf.expand_dims(conf_true, axis=-1)
        obj_mask = conf_true
        xy_loss = tf.reduce_sum(tf.square(xy_true - xy_pred) * obj_mask) / 2.0
        wh_loss = tf.reduce_sum(tf.square(tf.sqrt(wh_true) - tf.sqrt(wh_pred)) * obj_mask) / 2.0

        fl = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE)

        conf_loss = fl(conf_true, conf_pred)
        conf_loss_sum = tf.reduce_sum(conf_loss)

        return (xy_loss*xy_weight + wh_loss*wh_weight + conf_loss_sum) / (y_true.shape[0] + xy_weight + wh_weight)
    return loss