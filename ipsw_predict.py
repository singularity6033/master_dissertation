import colorsys
import math
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
from utils import config
from utils.utils_generator import ipsw_dataset_generator, ss_dataset_generator
from utils.nms import non_max_suppression
from utils.utils import get_classes
from tensorflow.python.keras.models import model_from_json

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
there are 2 modes representing ways to predict based on faster rcnn
1. mode == 'predict' ---> prediction for single image
2. mode == 'dir_predict' ---> prediction for image list
"""
mode = "dir_predict"
dir_origin_path = "img_test/"
dir_save_path = "img_test_output/"
dataset_name = 'voc'
method = 'ipsw'

if dataset_name == 'voc':
    classes_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt')
elif dataset_name == 'coco':
    classes_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt')
class_names, num_classes = get_classes(classes_path)

hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
input_shape = [32, 32]
roi_sizes = [[150, 150]]


def ipsw_detect_image(input_img, model_path, weights_path, class_list=class_names, color_list=None):
    # set font style and size
    if color_list is None:
        color_list = colors
    font = ImageFont.truetype(font='others/OpenSans-Light.ttf',
                              size=np.floor(3e-2 * np.shape(input_img)[1] + 0.5).astype('int32'))
    thickness = max((np.shape(input_img)[0] + np.shape(input_img)[1]) // 300, 1)
    # generate rois for prediction
    if method == 'ipsw':
        rois, locs = ipsw_dataset_generator(input_shape=input_shape, scale=1.5, win_step=16, roi_sizes=roi_sizes,
                                            train=False).test_generator(input_img)
    elif method == 'ss':
        rois, locs = ss_dataset_generator(input_shape=input_shape, category=class_names, dataset_name='voc',
                                          train=False).test_generator(input_img)
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
        if label_scores[i] >= 0.25:
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

    for label in labels.keys():
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes_ids = non_max_suppression(boxes, overlapThresh=0.6)
        for boxes_id in boxes_ids:
            predicted_class = label
            box = boxes[boxes_id]
            score = proba[boxes_id]
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(input_img.size[1], np.floor(bottom).astype('int32'))
            right = min(input_img.size[0], np.floor(right).astype('int32'))

            label_name = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(input_img)
            label_size = draw.textsize(label_name, font)
            label_name = label_name.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for th in range(thickness):
                draw.rectangle([left + th, top + th, right - th, bottom - th],
                               outline=color_list[class_list.index(predicted_class)])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=color_list[class_list.index(predicted_class)])
            draw.text(text_origin, str(label_name, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
    return input_img


if __name__ == '__main__':
    mp = './logs_sw_ip/vgg11_w7_new/ipsw_vgg11_cifar10_w7_fpl.json'
    wp = './logs_sw_ip/vgg11_w7_new/ipsw_vgg11_cifar10_w7_fpl.h5'
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = ipsw_detect_image(image, mp, wp)
                r_image.show()

    elif mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg',
                                          '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = ipsw_detect_image(image, mp, wp)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    else:
        raise AssertionError("Please specify the correct mode.")
