'''
    This script calculates evaluation metrics on a model and test dataset

    Author: David Kostka
    Date: 15.02.2021
'''

from absl import app, logging
import os
import time
import util.dataset as ds
import util.model as md
import util.utils as util
import src.evaluators.coco_evaluator as coco
import sklearn.metrics as metrics
import numpy as np

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

import src.evaluators.coco_evaluator as coco
from src.bounding_box import BoundingBox

import pickle
import json

params = util.get_params()

flags = tf.app.flags
flags.DEFINE_string('model_path', '', 'Path to model')
flags.DEFINE_string('testset_dir', '', 'Path to tfrecord test dataset')
flags.DEFINE_string('scores_path', '', 'Path to scores json')
flags.DEFINE_string('dataset_info_path', '', 'Path to dataset information file')
FLAGS = flags.FLAGS

def format_predictions(labels, predictions, nr_imgs):
    true_bboxes_list = []
    pred_bboxes_list = []
    #TODO: move yolo_anchors in configs
    yolo_anchors = np.array([(0.1499, 0.3052), (0.0563, 0.0748)], np.float32)

    progbar = tf.keras.utils.Progbar(nr_imgs)
    
    #TODO: This is slow, speed it up somhow (no n^2 for loop)
    for id, (lbl, pred) in enumerate(zip(labels, predictions)):
        true_bboxes = lbl
        pred_bboxes = util.postprocess_model_output(tf.expand_dims(pred, axis=0), params.dataset.image.size, 0.0, yolo_anchors)

        for bbox in true_bboxes:
            # np.max because it throws an error otherwise, don't ask me why
            true_bboxes_list.append(BoundingBox(id, format=2, class_id=np.max(bbox[5]), coordinates=bbox[:4]))
        for bbox in pred_bboxes:
            pred_bboxes_list.append(BoundingBox(id, format=2, class_id=np.max(bbox[5]), coordinates=bbox[:4], bb_type=2, confidence=bbox[4]))
        progbar.update(id+1)

    return true_bboxes_list, pred_bboxes_list

def get_evaluation_summary(true_bboxes_list, pred_bboxes_list, ious, area_ranges):
    coco_results = {}

    #TODO: This is slow, parallelize?
    start = time.perf_counter() 
    for iou in ious:
        for key, area_range in area_ranges.items():
            if key is 'full': key = ''
            coco_metr = coco.get_coco_metrics(true_bboxes_list, pred_bboxes_list, max_dets=80, iou_threshold=iou, area_range=area_range)
            coco_results['Nao.AP@' + str(round(iou*100)) + key] = coco_metr[0]['AP']
            coco_results['Nao.nr_positives'] = str(coco_metr[0]['total positives'])
            coco_results['Ball.AP@' + str(round(iou*100)) + key] = coco_metr[1]['AP']
            coco_results['Ball.nr_positives'] = str(coco_metr[1]['total positives'])
    end = time.perf_counter() 
    logging.info('Calculated metrics in ' + str(end-start) + ' seconds')

    return coco_results

def main(argv):
    tf.get_logger().setLevel('ERROR')

    with open(FLAGS.dataset_info_path, 'r') as f:
        ds_info = json.load(f)

    logging.info('Loading test dataset...')
    tfrecord_paths = [os.path.join(FLAGS.testset_dir, f) for f in os.listdir(FLAGS.testset_dir)]
    
    testset = ds.get_optimised_dataset(tfrecord_paths, 1, params.model.grid.shape, shuffle=False)
    labels = ds.get_optimised_dataset(tfrecord_paths, 1, params.model.grid.shape, shuffle=False).map(lambda x,y: util.postprocess_dataset_labels(y, params.dataset.image.size))

    logging.info('Loading Model...')
    yolo_anchors = np.array([(0.1499, 0.3052), (0.0563, 0.0748)], np.float32)
    model = tf.keras.models.load_model(FLAGS.model_path, custom_objects={'loss': md.multi_nao_loss(yolo_anchors)})

    logging.info('Running model on testset...')
    predictions = model.predict(testset)

    logging.info('Collecting predictions...')
    true_bboxes_list, pred_bboxes_list = format_predictions(labels, predictions, ds_info['test_split_count'])

    logging.info('Calculating metrics...')
    ious = [0.5, 0.75]
    area_ranges = {'small': (0, 24**2), 'medium': (24**2, 64**2), 'large': (64**2, np.inf), 'full': (0, np.inf)}
    eval_results = get_evaluation_summary(true_bboxes_list, pred_bboxes_list, ious, area_ranges)

    print(eval_results)

    logging.info('Writing metrics to json...')
    with open(FLAGS.scores_path, 'w') as fd:
        json.dump(eval_results, fd)


if __name__ == '__main__':
    app.run(main)
