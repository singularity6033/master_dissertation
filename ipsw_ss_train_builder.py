import time
from sklearn.preprocessing import LabelBinarizer
from utils.utils_generator import ipsw_dataset_generator, ss_dataset_generator
from utils.utils import get_classes
import tensorflow as tf
from utils import config
import matplotlib
import numpy as np
from tqdm import tqdm
import os

matplotlib.use('Agg')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset_name = 'voc'  # voc or coco
method = 'ss'  # ipsw or ss
# VOC2012
if dataset_name == 'voc':
    train_annotation_path = os.path.join(config.VOC_ORIG_BASE_PATH, '2012_train.txt')
    val_annotation_path = os.path.join(config.VOC_ORIG_BASE_PATH, '2012_val.txt')
    classes_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt')
# COCO
elif dataset_name == 'coco':
    train_annotation_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_train_set.txt')
    val_annotation_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_val_set.txt')
    classes_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt')

input_shape = [32, 32]
roi_sizes = [[150, 100], [150, 150], [100, 150]]

class_names, num_classes = get_classes(classes_path)

lb = LabelBinarizer()
lb.fit(list(class_names))

# load train and val set
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()

label_counter_train = np.zeros((num_classes, 1), dtype=int)
label_counter_val = np.zeros((num_classes, 1), dtype=int)

# initialize both the training and validation image generators
if method == 'ipsw':
    gen_train = ipsw_dataset_generator(input_shape, scale=1.5, win_step=16, roi_sizes=roi_sizes,
                                       category=class_names, dataset_name=dataset_name, train=True)
    gen_val = ipsw_dataset_generator(input_shape, scale=1.5, win_step=16, roi_sizes=roi_sizes,
                                     category=class_names, dataset_name=dataset_name, train=False)
elif method == 'ss':
    gen_train = ss_dataset_generator(input_shape, category=class_names, dataset_name=dataset_name, train=True)
    gen_val = ss_dataset_generator(input_shape, category=class_names, dataset_name=dataset_name, train=False)

t0 = time.time()
# build train and validation dataset
print('[INFO] training dataset generating...')
for train_line in tqdm(train_lines):
    label_counter_train = gen_train.train_generator(train_line, label_counter_train)
print('[INFO] validation dataset generating...')
for val_line in tqdm(val_lines):
    label_counter_val = gen_val.train_generator(val_line, label_counter_val)
t1 = time.time()
print('ipsw train builder time spent: ' + str(t1 - t0) + 's')
print('Done')
