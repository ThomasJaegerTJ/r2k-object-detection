'''
Helper functions for data and metric visualization

Author: David Kostka
Date: 15.02.2021
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.ticker as plticker

import util
import numpy as np

def plot_bbox_list(ax, true_bbox_list=None, pred_bbox_list=None):
    if true_bbox_list is not None:
        edge_clr = 'red'
        face_clr = [0,1,0,0.1]
        for bbox in true_bbox_list:
            rect = get_bbox_rectangle(bbox, edge_clr, [0,1,0,0])
            ax.add_patch(rect)

    if pred_bbox_list is not None:
        edge_clr = 'cyan'
        face_clr = [1,0,0,0.1]
        for bbox in pred_bbox_list:
            if bbox[5] == 1:
                edge_clr = 'yellow'
            else:
                edge_clr = 'cyan'
            conf = bbox[4]
            rect = get_bbox_rectangle(bbox, edge_clr, [0,1,0,0])
            ax.add_patch(rect)
            #ax.text(rect.get_x()+2, rect.get_y()+6, "%.3f" % conf, color='white', bbox=dict(facecolor='red', alpha=0.5))
            ax.text(rect.get_x()+2, rect.get_y()+6, "%.3f" % conf, color=edge_clr)

def get_bbox_rectangle(bbox, edge_clr, face_clr):
    '''
    bbox : [xmin, ymin, xmax, ymax]
    '''
    x = bbox[0]
    y = bbox[1]
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    conf = bbox[4]
    rect = Rectangle((bbox[0], bbox[1]), h, w, linewidth=2, edgecolor=edge_clr, facecolor=face_clr)
    return rect

def plot_image_with_grid(img, img_size, grid_shape):
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=img_size[0]/grid_shape[0]))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=img_size[1]/grid_shape[1]))
    ax.grid(which='major', axis='both', linestyle='-')
    ax.imshow(img, cmap='gray')
    return ax

def plot_roc(roc_auc, fpr, tpr, optimal_threshold=None, optimal_idx=None):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    if optimal_threshold and optimal_idx:
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')
        plt.text(fpr[optimal_idx] + 0.02, tpr[optimal_idx] - 0.05, 'trsh: ' + '%.3f' % optimal_threshold)
        plt.text(fpr[optimal_idx] + 0.02, tpr[optimal_idx] - 0.1, 'tpr: ' + '%.3f' % tpr[optimal_idx])
        plt.text(fpr[optimal_idx] + 0.02, tpr[optimal_idx] - 0.15, 'fpr: ' + '%.3f' % fpr[optimal_idx])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def visualize_output_tensor(true_feats, feats, img, img_size, grid_shape, conf_threshold=0.5):
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=img_size[0]/grid_shape[0]))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=img_size[1]/grid_shape[1]))
    ax.grid(which='major', axis='both', linestyle='-')
    ax.imshow(img, cmap='gray')

    for i in range(10):
        for j in range(8):
            pred_conf = feats[0, j, i, 4]
            true_conf = true_feats[0, j, i, 4]
            grid_size = np.array(img_size) / np.array(grid_shape)
            
            if true_conf > conf_threshold:
                x, y = (true_feats[0, j, i, 0:2] + (i,j)) * grid_size
                w, h = true_feats[0, j, i, 2:4] * grid_size
                x = x - (w/2.0)
                y = y - (h/2.0)

                edge_clr = 'green'
                face_clr = [0,1,0,0.1]
                bbox = Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_clr, facecolor=face_clr)
                ax.add_patch(bbox)
            
            if pred_conf > conf_threshold:
                x, y = (feats[0, j, i, 0:2] + (i,j)) * grid_size
                w, h = feats[0, j, i, 2:4] * grid_size
                x = x - (w/2.0)
                y = y - (h/2.0)

                edge_clr = 'red'
                face_clr = [1,0,0,0.1]
                bbox = Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_clr, facecolor=face_clr)
                ax.add_patch(bbox)
                ax.text(x+2, y+6, "%.3f" % pred_conf, color='white', bbox=dict(facecolor='red', alpha=0.5))
    

def to_global_bbox(bbox, cell, size, grid_shape):
    cell_size = (size[0]/grid_shape[0], size[1]/grid_shape[1])

    x = (bbox[0] + cell[0]) * cell_size[0]
    y = (bbox[1] + cell[1]) * cell_size[1]
    w = (cell_size[0] * bbox[2])
    h = (cell_size[1] * bbox[3])
    bbox[0] = x - (w / 2)
    bbox[1] = y - (h / 2)
    bbox[2] = bbox[0] + w
    bbox[3] = bbox[1] + h

    #return util.unnormalize_corners(bbox, size)
    return bbox

def get_bbox_list(feature_map, img_size, grid_shape, conf_threshold=0.5):
    indx = (feature_map[:,:,4] > conf_threshold).nonzero()
    bboxes = feature_map[indx]
    print(bboxes)
    indx = np.transpose(indx)
    indx[:, [0,1]] = indx[:,[1,0]]

    for bbox, ix in zip(bboxes, indx):
        bbox = to_global_bbox(bbox, ix, img_size, grid_shape)
    return bboxes

def visualize_output(bboxes, img, img_size, grid_shape):
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=img_size[0]/grid_shape[0]))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=img_size[1]/grid_shape[1]))
    ax.grid(which='major', axis='both', linestyle='-')

    ax.imshow(img, cmap='gray')
    for bbox in bboxes:
        #bbox = util.unnormalize_corners(bbox, img_size)
        vis.plot_bbox(ax, bbox, text=False)