'''
Helper functions to create a TFRecord image Dataset.

Author: David Kostka
Date: 15.02.2021
'''
# TODO: In Klasse convertieren, Generalisieren so dass die Struktur der labels.csv definiert werden kann aber der Rest gleich bleibt (selbe TFRecord Struktur)
# Dafuer Config Datei lesen um an Dataset anpassbar zu sein, spalten, codierung, nr. channels
# und definierbares mapping von CSV Spaltenname zu TFRecord Feature name z.B. {filename:"name", xmin:"minX", ...}

import tensorflow as tf
import pandas as pd
import os
import util.utils as util


def parse_tfrecord(tfrecord):
    IMAGE_FEATURE_MAP = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    img = tf.image.decode_jpeg(x['image/encoded'], channels=1)

    labels = tf.cast(tf.sparse.to_dense(x['image/object/class/label']), tf.float32)
    annotations = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    #paddings = [[0, 10 - tf.shape(annotations)[0]], [0, 0]]
    #annotations = tf.pad(annotations, paddings)

    return img, annotations


def load_tfrecord_dataset(tfrecord_path):
    '''
    Erzeugt ein tf.data.Dataset Objekt aus einer TFRecord File 
    '''
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_optimised_dataset(path, batch_size, grid_shape, shuffle=True):
    dset = load_tfrecord_dataset(path)
    dset_opt = dset.map(lambda x,y: (x, util.preprocess_true_boxes(y, grid_shape)))
    if shuffle: 
        dset_opt = dset_opt.shuffle(buffer_size=512)
    dset_opt = dset_opt.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dset_opt
