import time
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from faster_rcnn import faster_rcnn

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


"""
there are 2 modes representing ways to predict based on faster rcnn
1. mode == 'predict' ---> prediction for single image
2. mode == 'dir_predict' ---> prediction for image list
"""


def frcnn_detect_image(img, model_path, weights_path, chop_idx, stride, out_channel):
    frcnn = faster_rcnn(model_path=model_path, weights_path=weights_path, chop_idx=chop_idx, stride=stride,
                        out_channel=out_channel)
    img = img.resize((300, 300), Image.BICUBIC)
    r_image = frcnn.detect_image(img)
    return r_image
