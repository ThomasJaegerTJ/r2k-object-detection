"""
Referenced: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/generate_tfrecord.py

python src/generate_tfrecord.py --output_dir=data/datasets/simulator/records --images_dir=data/datasets/raw/simulator --images_subdir=images/ --csv_dir=data/datasets/simulator/labels
python src/generate_tfrecord.py --output_dir=data/datasets/ImageTagger/records --images_dir=data/datasets/raw/ImageTagger --csv_dir=data/datasets/ImageTagger/labels
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from collections import namedtuple
import random
import math
import json

import util.utils as util

flags = tf.app.flags
flags.DEFINE_string('csv_dir', '', 'Path to labels CSV folder')
flags.DEFINE_string('images_dir', '', 'Path to folder of datasets containing images')
flags.DEFINE_string('images_subdir', '', 'Images subdirectory per dataset')
flags.DEFINE_string('output_dir', '', 'Path to TFRecord output folder')
flags.DEFINE_string('info_path', '', 'Path to dataset info file')
FLAGS = flags.FLAGS

params = util.get_params()


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def process_image(image):
    if params.dataset.image.channels == 1:
        image = image.convert('L')
    image = image.resize(params.dataset.image.size)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    resized_encoded_png = img_byte_arr.getvalue()

    return resized_encoded_png

def create_tf_example(group):
    with tf.gfile.GFile(os.path.normpath(group.filename), 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    width, height = image.size

    resized_encoded_png = process_image(image)
    filename = group.filename.encode('utf8')

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes.append(row['class'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[resized_encoded_png])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
    }))
    return tf_example


def get_shuffled_labels(csv_dir, images_dir, images_subdir):
    shuffled_labels = []
    csv_files = os.listdir(csv_dir)
    progbar = tf.keras.utils.Progbar(len(csv_files))

    for i, file in enumerate(csv_files):
        ds_name = os.path.splitext(file)[0]
        csv_path = os.path.join(csv_dir, file)
        images_path = os.path.join(images_dir, ds_name, images_subdir)

        examples = pd.read_csv(csv_path)
        examples['filename'] = images_path + examples['filename']
        grouped = split(examples, 'filename')
        shuffled_labels.extend(grouped)

        progbar.update(i+1)

    random.shuffle(shuffled_labels)
    return shuffled_labels


def get_record_writers(count, output_dir, prefix=''):
    record_writers = []

    for i in range(count):
        wr_path = os.path.join(output_dir, prefix + str(i) + '.record')
        record_writers.append(tf.python_io.TFRecordWriter(wr_path))

    return record_writers


def distribute_examples(labelset, record_writers):
    progbar = tf.keras.utils.Progbar(len(labelset))

    for i, group in enumerate(labelset):
        writer = random.choice(record_writers)
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
        progbar.update(i+1)


def main(_):
    test_out_dir = os.path.join(FLAGS.output_dir, 'test')
    train_out_dir = os.path.join(FLAGS.output_dir, 'train')
    util.create_dirs([FLAGS.output_dir, test_out_dir, train_out_dir])
    random.seed(420)

    tf.print('Loading and shuffling labels: ' + FLAGS.csv_dir)
    shuffled_labels = get_shuffled_labels(FLAGS.csv_dir, FLAGS.images_dir, FLAGS.images_subdir)

    test_split_count = math.ceil(len(shuffled_labels) * params.dataset.test_split)
    train_split_count = len(shuffled_labels) - test_split_count
    test_records_count = math.ceil(params.dataset.nr_records * params.dataset.test_split)
    train_records_count = params.dataset.nr_records - test_records_count

    test_record_writers = get_record_writers(test_records_count, test_out_dir, 'test_')
    train_record_writers = get_record_writers(train_records_count, train_out_dir, 'train_')

    tf.print('Distributing {} test examples to {} TFRecords in: {}'.format(test_split_count, test_records_count, FLAGS.output_dir))
    distribute_examples(shuffled_labels[:test_split_count], test_record_writers)
    tf.print('Distributing {} train examples to {} TFRecords in: {}'.format(train_split_count, train_records_count, FLAGS.output_dir))
    distribute_examples(shuffled_labels[test_split_count:], train_record_writers)
    
    # TODO: More metadata like name, imgsize, ...
    tf.print('Writing info file in: ' + FLAGS.info_path)
    with open(FLAGS.info_path, 'w') as fd:
        json.dump( {'test_split': params.dataset.test_split,
                    'test_split_count': test_split_count, 
                    'train_split_count': train_split_count,
                    }, fd)

    tf.print('Done!')


if __name__ == '__main__':
    tf.app.run()
