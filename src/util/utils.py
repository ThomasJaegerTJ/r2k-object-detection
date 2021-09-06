'''
Helper functions for data processing and pipeline stages

Author: David Kostka
Date: 15.02.2021
'''

import yaml
from collections import namedtuple
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import albumentations as A          #For Data Augmentatio
from albumentations.augmentations.transforms import CLAHE, ColorJitter, Downscale, ISONoise

class Object(object):
    pass


def get_params(path='params.yaml'):
    with open(path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

        dataset = Object()
        dataset.nr_records = params['dataset']['nr_records']
        dataset.test_split = params['dataset']['test_split']
        dataset.val_split = params['dataset']['val_split']

        dataset.image = Object()
        dataset.image.width = params['dataset']['image']['width']
        dataset.image.height = params['dataset']['image']['height']
        dataset.image.channels = params['dataset']['image']['channels']
        dataset.image.size = (dataset.image.width, dataset.image.height)

        train = Object()
        train.batch_size = params['train']['batch_size']
        train.epochs = params['train']['epochs']
        train.early_stopping = params['train']['early_stopping']
        train.lr = params['train']['lr']

        model = Object()
        model.layers = params['model']['layers']
        model.grid = Object()
        model.grid.h = params['model']['grid']['h']
        model.grid.w = params['model']['grid']['w']
        model.grid.shape = (model.grid.w, model.grid.h)

        wandb = Object()
        wandb.group = params['wandb']['group']

        params = Object()
        params.dataset = dataset
        params.train = train
        params.model = model
        params.wandb = wandb
        return params

def create_dirs(dir_list):
    for path in dir_list:
        if not os.path.exists(path):
            os.makedirs(path)
            print('Created new directory: ' + path)

def unnormalize_corners(boxes, size):
    '''
    Skaliert normalisiere BBOX Koordinaten in Pixel Koordinaten um
    Input: (xmin, ymin, xmin, ymax), (img_height, img_width)
    Werte von 0.0 bis 1.0

    Output: (xmin, ymin, xmin, ymax)
    Skalierte Werte von 0 bis size
    '''
    size = tf.reshape(tf.tile(size, [tf.shape(boxes)[-2]]), [tf.shape(boxes)[-2], tf.shape(size)[0]])
    size = tf.cast(size, tf.float32)
    return tf.concat(
        [tf.multiply(boxes[..., :2], size), tf.multiply(boxes[..., 2:4], size), boxes[..., 4:]],
        axis=-1,
    )

def normalize_corners(boxes, size):
    '''
    Liefert Normalisiere BBOX Koordinaten
    Input: (xmin, ymin, xmax, ymax), (img_height, img_width)
    Werte von 0 bis size

    Output: (xmin, ymin, xmin, ymax)
    Werte von 0.0 bis 1.0
    '''
    size = tf.reshape(tf.tile(size, [tf.shape(boxes)[-2]]), [tf.shape(boxes)[-2], tf.shape(size)[0]])
    size = tf.cast(size, tf.float32)
    return tf.concat(
        [tf.divide(boxes[..., :2], size), tf.divide(boxes[..., 2:4], size), boxes[..., 4:]],
        axis=-1,
    )

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2], boxes[..., 4:]], axis=-1)

# https://keras.io/examples/vision/retinanet/
def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:4]) / 2.0, boxes[..., 2:4] - boxes[..., :2], boxes[..., 4:]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:4] / 2.0, boxes[..., :2] + boxes[..., 2:4] / 2.0, boxes[..., 4:]],
        axis=-1,
    )

def preprocess_true_boxes(true_boxes, grid_shape):
    # https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py

    conv_height = grid_shape[1]
    conv_width = grid_shape[0]
    num_box_params = 5
    num_classes = 2
    matching_true_boxes = tf.zeros(
        (conv_height, conv_width, num_classes, num_box_params),
        dtype=tf.float32)

    true_boxes = convert_to_xywh(true_boxes)
    cell = true_boxes[..., :2] * grid_shape
    cell_rounded = tf.math.floor(cell)
    offset = cell - cell_rounded
    cell_rounded = tf.cast(cell_rounded, tf.int64)
    wh = true_boxes[..., 2:4] #* grid_shape

    classes = tf.one_hot(tf.cast(true_boxes[..., 4], tf.int64), depth=3)
    conf = 1 - classes[..., 0:1]
    indx = tf.cast(tf.maximum(true_boxes[..., 4] - 1, 0), tf.int64)
    local_boxes = tf.concat([tf.cast(offset, tf.float32), tf.cast(wh, tf.float32), tf.cast(conf, tf.float32)], axis=-1)

    return tf.tensor_scatter_nd_update(matching_true_boxes, tf.stack([cell_rounded[:,1], cell_rounded[:,0], indx], axis=-1), local_boxes)

def create_grid(grid_shape):
    grid_y = tf.tile(tf.reshape(tf.range(grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 2, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 2, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, tf.float32)

    return grid


def get_bbox_list_from_feature_map(xy, wh, conf, conf_threshold = 0.5):
    grid_shape = tf.shape(xy)[-4:-2]
    grid = create_grid(grid_shape)

    #xy = (label_feature_map[..., :2] + grid) / tf.cast(grid_shape[::-1], tf.float32)
    #wh = label_feature_map[..., 2:4] / tf.cast(grid_shape[::-1], tf.float32)
    #conf = label_feature_map[..., 4:5]

    xy = (xy + grid) / tf.cast(grid_shape[::-1], tf.float32)
    #wh = wh / tf.cast(grid_shape[::-1], tf.float32)

    found = tf.where(tf.greater(conf, conf_threshold))
    classes = tf.expand_dims(found[..., 3], axis=-1)
    bboxes = tf.concat([xy, wh, conf], axis=-1)
    bbox_list = tf.gather_nd(bboxes, found[..., :-1])

    return tf.concat([bbox_list, tf.cast(classes, tf.float32)], axis=-1)

# TODO: Redundant in model.py
def activate_feature_map(feats, anchors):
    feats = tf.reshape(feats, [-1, 8, 10, 2, 5])
    conf = tf.sigmoid(feats[..., 4:5])
    xy = tf.sigmoid(feats[..., 0:2])
    wh = tf.maximum(feats[..., 2:4], 0.0) * anchors

    return xy, wh, conf

def postprocess_model_output(pred, img_size, conf_threshold, anchors, iou_threshold=0.6):
    grid_shape = tf.shape(pred)[-4:-2]

    pred = tf.reshape(
        pred, [-1, grid_shape[0], grid_shape[1], 2, 5])

    xy, wh, conf = activate_feature_map(pred, anchors)
    bbox_list = get_bbox_list_from_feature_map(xy, wh, conf, conf_threshold)

    bbox_list_corners_norm = convert_to_corners(bbox_list)
    bbox_list_corners = unnormalize_corners(bbox_list_corners_norm, img_size)
    
    supressed_idxs = tf.image.non_max_suppression(bbox_list_corners[..., :4], bbox_list_corners[..., 4], 80, score_threshold=conf_threshold, iou_threshold=iou_threshold)
    bbox_list_suppressed = tf.gather(bbox_list_corners, supressed_idxs)

    return bbox_list_suppressed

def postprocess_dataset_labels(lbls, img_size):
    grid_shape = tf.shape(lbls)[-4:-2]
    lbls = tf.reshape(
        lbls, [-1, grid_shape[0], grid_shape[1], 2, 5])

    bbox_list = get_bbox_list_from_feature_map(lbls[..., 0:2], lbls[..., 2:4], lbls[..., 4:5])

    bbox_list_corners_norm = convert_to_corners(bbox_list)
    bbox_list_corners = unnormalize_corners(bbox_list_corners_norm, img_size)

    return bbox_list_corners
    
bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}

augment = A.Compose([
            A.CLAHE(p = 0.4),
            A.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.4, p = 0.4),
            # A.Downscale(scale_min = 0.25, scale_max = 0.25, p = 0.5),
            A.ISONoise(color_shift = (0.01, 0.07), intensity = (0.1, 0.7), p = 0.4),
            A.MotionBlur(blur_limit = (3,7), p = 0.4), #Blur Image to random levels
            A.RandomBrightnessContrast(brightness_limit = 0.6, contrast_limit = 0.6, p = 0.4), #Random adjustments in brightness and contrast
            A.GaussNoise(var_limit = (10.0, 60.0), mean = 0, p = 0.4),
            A.RGBShift(r_shift_limit = 25, g_shift_limit = 25, b_shift_limit = 25, p = 0.4)
            #A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower = 1, num_shadows_upper = 2, shadow_dimension= = 5, p = 0.5),
            #A.RandomSunFlare(flare_roi = (0, 0, 1, 0.5), angle_lower = 0, angle_upper = 1, num_flare_circles_lower = 6, num_flare_circles_upper = 10, src_radius = 400, src_color = (255, 255, 255), p = 0.5),
            
            #A.HorizontalFlip(p = 0.5)
        ])#, bbox_params=bbox_params)
