import math
import os
import numpy as np
from PIL import Image
from imutils import paths
from pycocotools.coco import COCO
from tqdm import tqdm
from tensorflow.python.keras.models import model_from_json
from utils import config
from utils.utils_generator import ipsw_dataset_generator
from utils.nms import non_max_suppression
from utils.utils import get_classes
from utils.calc_map import get_map

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
map_mode == 0: predictions + gts + voc_map
map_mode == 1: predictions
map_mode == 2: gts
map_mode == 3: voc_map
"""

map_mode = 3
classes_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt')
test_dataset_path = open(os.path.join(config.COCO_ORIG_BASE_PATH, "test_set.txt")).read().strip().split()
# test_dataset_path = test_dataset_path[3799:]
# MINOVERLAP --> mAP0.x
MINOVERLAP = 0.25
# visualization of map calculations
map_vis = False
# map result dir
map_out_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_coco/w1/map_result'
model_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_coco/w1/ipsw_vgg16_cifar100_w1_fpl.json'
weights_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_coco/w1/ipsw_vgg16_cifar100_w1_fpl.h5'

if not os.path.exists(map_out_path):
    os.makedirs(map_out_path)

input_shape = [32, 32]
roi_sizes = [[150, 100], [150, 150], [100, 150]]

if not os.path.exists(map_out_path):
    os.makedirs(map_out_path)
if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
    os.makedirs(os.path.join(map_out_path, 'ground-truth'))
if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
    os.makedirs(os.path.join(map_out_path, 'detection-results'))
if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
    os.makedirs(os.path.join(map_out_path, 'images-optional'))

coco = COCO(config.COCO_TEST_ANNOTS)
cats = coco.loadCats(coco.getCatIds())
catIds = [cat['id'] for cat in cats]
coco_classes = [cat['name'] for cat in cats]
class_names, _ = get_classes(classes_path)

if map_mode == 0 or map_mode == 1:

    print("Get predict result.")
    for image_path in tqdm(test_dataset_path):
        image = Image.open(image_path)
        filename = image_path.split(os.path.sep)[-1]
        filename = filename[:filename.rfind(".")]
        image_id = int(filename.lstrip('0'))
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + str(image_id) + ".jpg"))
        # generate rois for prediction
        rois, locs = ipsw_dataset_generator(input_shape=input_shape, scale=1.5, win_step=16,
                                            roi_sizes=roi_sizes, train=False).test_generator(image)
        f = open(os.path.join(map_out_path, "detection-results/" + str(image_id) + ".txt"), "w")
        # load model info
        with open(model_path, 'r') as file:
            model_json = file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        rois = np.array(rois)
        loc_preds, label_preds = model.predict(rois)
        label_idxs = np.argmax(label_preds, axis=1)
        label_scores = np.max(label_preds, axis=1)
        num_pred = len(label_idxs)
        labels = {}
        # loop over the predictions
        for i in range(num_pred):
            # filter out weak detections by ensuring the predicted probability
            # is greater than the minimum probability
            if label_scores[i] >= 0.5:
                # grab the bounding box regression and update the coordinates
                x_start_pred, y_start_pred, x_end_pred, y_end_pred = locs[i]
                w_pred, h_pred = x_end_pred - x_start_pred, y_end_pred - y_start_pred
                tx, ty, tw, th = loc_preds[i]
                x_start_update = w_pred * tx + x_start_pred
                y_start_update = h_pred * ty + y_start_pred
                w_update = math.exp(tw) * w_pred
                h_update = math.exp(th) * h_pred
                box = (x_start_update, x_start_update, x_start_update + w_update, y_start_update + h_update)
                # grab the list of predictions for the label and add the bounding box and probability to the list
                predicted_class = class_names[int(label_idxs[i])]
                L = labels.get(predicted_class, [])
                L.append((box, label_scores[i], label_idxs[i]))
                labels[predicted_class] = L

        # loop over the labels for each of detected objects in the image
        for label in labels.keys():
            boxes = np.array([p[0] for p in labels[label]])
            proba = np.array([p[1] for p in labels[label]])
            # non-maxima suppression
            boxes_ids = non_max_suppression(boxes, overlapThresh=0.1)
            for boxes_id in boxes_ids:
                box = boxes[boxes_id]
                score = proba[boxes_id]
                top, left, bottom, right = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                f.write("%s %s %s %s %s %s\n" % (label, score,
                                                 str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
    print("Get predict result done.")

if map_mode == 0 or map_mode == 2:
    print("Get ground truth result.")
    for image_path in tqdm(test_dataset_path):
        filename = image_path.split(os.path.sep)[-1]
        filename = filename[:filename.rfind(".")]
        image_id = int(filename.lstrip('0'))
        with open(os.path.join(map_out_path, "ground-truth/" + str(image_id) + ".txt"), "w") as new_f:
            annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
            annInfos = coco.loadAnns(annIds)
            for annInfo in annInfos:
                # extract the label and bounding box coordinates
                idx_gt = annInfo['category_id']
                cat = cats[catIds.index(idx_gt)]['name']
                cls_id = coco_classes.index(cat)
                xMin, yMin, bw, bh = annInfo['bbox']
                xMax = xMin + bw
                yMax = yMin + bh
                coordinates = (xMin, yMin, xMax, yMax)
                new_f.write("%s %s %s %s %s\n" % (cat, str(xMin), str(yMin), str(xMax), str(yMax)))
    print("Get ground truth result done.")

if map_mode == 0 or map_mode == 3:
    print("Get map.")
    get_map(MINOVERLAP, True, path=map_out_path)
    print("Get map done.")
