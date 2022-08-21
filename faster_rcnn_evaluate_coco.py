import os
import xml.etree.ElementTree as ET

from PIL import Image
from imutils import paths
from pycocotools.coco import COCO
from tqdm import tqdm

from utils import config
from utils.utils import get_classes
from utils.calc_map import get_coco_map, get_map
from faster_rcnn import faster_rcnn

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
map_mode == 0: predictions + gts + voc_map
map_mode == 1: predictions
map_mode == 2: gts
map_mode == 3: voc_map
map_mode == 4: use coco api to calculate map based with iou 0.50:0.95, need to format predictions and gts
"""

map_mode = 0
classes_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt')
test_dataset_path = list(paths.list_images(config.COCO_TEST_IMAGES))
# MINOVERLAP --> mAP0.x
MINOVERLAP = 0.25
# visualization of map calculations
map_vis = False
# map result dir
map_out_path = './logs_faster_rcnn/vgg11_w10/map_result'

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
    print("Load model.")
    frcnn = faster_rcnn(confidence=0.5, nms_iou=0.3)
    print("Load model done.")

    print("Get predict result.")
    for image_path in tqdm(test_dataset_path):
        image = Image.open(image_path)
        filename = image_path.split(os.path.sep)[-1]
        filename = filename[:filename.rfind(".")]
        image_id = int(filename.lstrip('0'))
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + str(image_id) + ".jpg"))
        frcnn.get_map_txt(image_id, image, class_names, map_out_path)
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

if map_mode == 4:
    print("Get map.")
    get_coco_map(class_names=class_names, path=map_out_path)
    print("Get map done.")
