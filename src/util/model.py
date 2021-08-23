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
import visualkeras

def FeatureModule(x, layers, filters, sep=True):
    sep = False
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

def decode_output(feats):
    #classes = tf.keras.backend.softmax(feats[..., 5:])
    #classes = tf.sigmoid(feats[..., 5:])
    classes = tf.sigmoid(feats[..., 5:])
    conf = tf.sigmoid(feats[..., 4])
    xy = tf.sigmoid(feats[..., 0:2])
    wh = tf.maximum(feats[..., 2:4], 0) #ReLU

    return xy, wh, conf, classes

def MultiNaoModel(img_size, layer_list):
    '''
    Input: Image ((img_size), 1)
    Output: 4x5x5
    '''

    input_image = Input(shape=(*img_size, 1))
    x = FeatureExtractor(input_image, layer_list)

    output = Conv2D(5+2, (1, 1), strides=(1,1), padding='same')(x)

    model = Model(inputs=input_image, outputs=output)

    return model

def multi_nao_loss():
    def loss(y_true, y_pred):
        xy_pred, wh_pred, conf_pred, classes_pred = decode_output(y_pred)

        classes_true = y_true[..., 5:]
        conf_true = y_true[..., 4]
        xy_true = y_true[..., 0:2]
        wh_true = y_true[..., 2:4]

        obj_mask = tf.zeros(tf.shape(y_true)[:4])
        obj_mask = tf.expand_dims(conf_true, axis=-1)

        xy_loss = tf.reduce_sum(tf.square(xy_true - xy_pred) * obj_mask) / 2.0
        wh_loss = tf.reduce_sum(tf.square(tf.sqrt(wh_true) - tf.sqrt(wh_pred)) * obj_mask) / 2.0

        #tf.nn.softmax_cross_entropy_with_logits und sigmoid rausnehmen
        #cross_entr = losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) # From logits?
        #class_loss = tf.nn.softmax_cross_entropy_with_logits(labels=classes_true, logits=classes_pred)
        #class_loss = tf.reduce_sum(class_loss * tf.squeeze(obj_mask))

        #bceFunc = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        #conf_loss = bceFunc(conf_true, conf_pred)
        #conf_loss = tf.reduce_sum(tf.square(conf_true - conf_pred)) 
        fl = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE)

        conf_loss = fl(conf_true, conf_pred)
        conf_loss_sum = tf.reduce_sum(conf_loss)

        class_loss = fl(classes_true, classes_pred)
        class_loss_sum = tf.reduce_sum(class_loss * tf.squeeze(obj_mask))

        #return (xy_loss + wh_loss + conf_loss + class_loss) / y_true.shape[0]
        return (xy_loss*5 + wh_loss*5 + conf_loss_sum + class_loss_sum) / (y_true.shape[0] + 5 + 5)
    return loss

def test_loss():
    # Loss Funktion und Metriken testen
    pred_test = np.array([[1.0, 1.0, 1.0, 1.0, 1], [0.7, 0.8, 0.7, 0.7, 1]])
    true_test = np.array([[1.0, 1.0, 1.0, 1.0, 1], [0.3, 0.3, 0.4, 0.4, 1]])
    print(true_test)
    print(pred_test)
    print('-------- Loss ----------')
    print(md.nao_loss(true_test, pred_test))
