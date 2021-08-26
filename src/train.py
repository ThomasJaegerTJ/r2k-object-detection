'''
    This script trains a specified model with the specified train- and testset.
    Model checkpoints get saved locally, incase the system crashes while training.
    It also creates graphs of the model architecture and logs for tensorboard.
    The trained model then gets saved in multiple different formats.

    Author: David Kostka
    Date: 15.02.2021

    python src/train.py --tfrecord_dir=data/datasets/simulator/records --output_dir=data/models/experimodel
'''

import os
from time import time
from absl import app, logging
import json
import util.dataset as ds
import util.model as md
import util.utils as util
import pandas as pd
from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

params = util.get_params()

flags = tf.app.flags
flags.DEFINE_string('tfrecord_dir', '', 'Path to TFRecords trainset folder')
flags.DEFINE_string('tfrecord_test_dir', '', 'Path to TFRecords testset folder')
flags.DEFINE_string('output_dir', '', 'Path to model output folder')
flags.DEFINE_string('chkpts_dir', 'data/checkpoints/', 'Path to TF checkpoints')
flags.DEFINE_string('logs_dir', 'data/logs/', 'Path to logs')
flags.DEFINE_string('history_path', '', 'Path to history CSV')
flags.DEFINE_string('dataset_info_path', '', 'Path to dataset information file')
flags.DEFINE_boolean('export_tflite', False, 'Export model as .tflite')
flags.DEFINE_boolean('augment_data', True, 'Apply Augmentation to the train dataset')
FLAGS = flags.FLAGS


def compile_model(model):
    model.compile(
                    loss=md.multi_nao_loss(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=params.train.lr),
                )

    # Print info and architecture
    logging.info(model.summary())

    return model


def train(model, trainset, valset):
    model_name = os.path.basename(FLAGS.output_dir)
    callbacks = [
                    ModelCheckpoint(FLAGS.chkpts_dir + model_name + '/' + model_name + '_{epoch}.tf', verbose=1, save_weights_only=True),
                    TensorBoard(log_dir=FLAGS.logs_dir + model_name + '/' + '{}'.format(time())),
                    ReduceLROnPlateau(verbose=1)
                ]
    if params.train.early_stopping:
        callbacks.append(EarlyStopping(patience=20, verbose=1, restore_best_weights=True, monitor='val_loss'))

    history = model.fit(
        trainset,
        validation_data=valset,
        epochs=params.train.epochs,
        callbacks=[callbacks]
    )

    return history


def save_model(model, path):
    model_path = os.path.join(path, os.path.basename(path))

    # Save Model as default export
    model.save(model_path)

    # Save as format compatible with CompiledNN
    model.save(model_path + '.hdf5', save_format='h5')

    # Convert and Save to TFLite Model
    if FLAGS.export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert(model)
        with open(model_path + '.tflite', 'wb') as f:
            f.write(tflite_model)


def main(_):
    tf.get_logger().setLevel('ERROR')

    # Multi GPU Support
    # See: https://keras.io/guides/distributed_training/
    #strategy = tf.distribute.MirroredStrategy()
    #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with open(FLAGS.dataset_info_path, 'r') as f:
        ds_info = json.load(f)

    # Open a strategy scope.
    #with strategy.scope():
    model = md.MultiNaoModel((params.dataset.image.size[1], params.dataset.image.size[0]), params.model.layers)
    model = compile_model(model)
    
    tfrecord_paths = [os.path.join(FLAGS.tfrecord_dir, f) for f in os.listdir(FLAGS.tfrecord_dir)]
    tfrecord_test_paths = [os.path.join(FLAGS.tfrecord_test_dir, f) for f in os.listdir(FLAGS.tfrecord_test_dir)]

    #trainset = ds.get_optimised_dataset(tfrecord_paths, params.train.batch_size, params.model.grid.shape)
    #valset = ds.get_optimised_dataset(tfrecord_test_paths, params.train.batch_size, params.model.grid.shape)
    if FLAGS.augment_data:
        fullset = ds.get_augmented_dataset(tfrecord_paths, params.train.batch_size, params.model.grid.shape)
    else:
        fullset = ds.get_optimised_dataset(tfrecord_paths, params.train.batch_size, params.model.grid.shape)
    
    val_count = int(((ds_info['train_split_count']/2) * params.dataset.val_split) / params.train.batch_size)

    valset = fullset.take(val_count)
    trainset = fullset.skip(val_count)
    
    history = train(model, trainset, valset)
    save_model(model, FLAGS.output_dir)
    
    logging.info('Writing history to csv...')
    with open(FLAGS.history_path, 'w') as fd:
        pd.DataFrame(history.history).to_csv(fd)

if __name__ == '__main__':
    app.run(main)
