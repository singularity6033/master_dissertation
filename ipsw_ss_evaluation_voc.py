import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import model_from_json
from tqdm import tqdm
from utils import config
from utils.nms import non_max_suppression, cpu_soft_nms
from utils.utils_generator import ipsw_dataset_generator, ss_dataset_generator
from utils.utils import get_classes
from utils.calc_map import get_map
import tensorflow as tf
import math

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(map_out_path, model_path, weights_path):
    """
    map_mode == 0: predictions + gts + voc_map
    map_mode == 1: predictions
    map_mode == 2: gts
    map_mode == 3: voc_map
    """

    map_mode = 3
    VOCdevkit_path = './dataset/VOCdevkit'
    method = 'ipsw'
    classes_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt')
    class_names, num_classes = get_classes(classes_path)
    test_dataset_path = open(os.path.join(config.VOC_ORIG_BASE_PATH,
                                          "VOC2012/ImageSets/Main/test.txt")).read().strip().split()

    # we choose 25% of testing dataset to evaluate our model
    # test_dataset_path = test_dataset_path[1600:]

    voc_image_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'VOC2012/JPEGImages')
    # MINOVERLAP --> mAP0.x
    MINOVERLAP = 0.25
    # visualization of map calculations
    map_vis = False
    input_shape = [32, 32]
    roi_sizes = [[150, 100], [150, 150], [100, 150]]
    # if os.path.exists(map_out_path):
    #     shutil.rmtree(map_out_path)
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    if map_mode == 0 or map_mode == 1:

        print("Get predict result.")
        for image_path in tqdm(test_dataset_path):
            input_image = Image.open(os.path.join(voc_image_path, image_path + '.jpg'))
            if map_vis:
                input_image.save(os.path.join(map_out_path, "images-optional/" + image_path + ".jpg"))
            if method == 'ipsw':
                rois, locs = ipsw_dataset_generator(input_shape=input_shape, scale=1.5, win_step=16,
                                                    roi_sizes=roi_sizes,
                                                    train=False).test_generator(input_image)
            elif method == 'ss':
                rois, locs = ss_dataset_generator(input_shape=input_shape, category=class_names, dataset_name='voc',
                                                  train=False).test_generator(input_image)
            f = open(os.path.join(map_out_path, "detection-results/" + image_path + ".txt"), "w")
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
                    box = (x_start_update, y_start_update, x_start_update + w_update, y_start_update + h_update)
                    # box = (x_start_pred, y_start_pred, x_end_pred, y_end_pred)
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
                boxes_ids = non_max_suppression(boxes, overlapThresh=0.6)
                # boxes_ids = cpu_soft_nms(boxes, proba, Nt=0.3)
                for boxes_id in boxes_ids:
                    box = boxes[boxes_id]
                    score = proba[boxes_id]
                    left, top, right, bottom = box
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    bottom = min(input_image.size[1], np.floor(bottom).astype('int32'))
                    right = min(input_image.size[0], np.floor(right).astype('int32'))
                    f.write("%s %s %s %s %s %s\n" % (label, score,
                                                     str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
            f.close()
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_path in tqdm(test_dataset_path):
            with open(os.path.join(map_out_path, "ground-truth/" + image_path + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2012/Annotations/" + image_path + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")


if __name__ == '__main__':
    # b_map_out_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # b_model_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # b_weights_path = './sgd_tf2/logs_ipsw_morefc'
    # mn = ['sgd_vgg11_cifar10', 'sgd_vgg11_cifar100', 'sgd_vgg16_cifar10', 'sgd_vgg16_cifar100']
    # for i in range(4):
    #     mop = os.path.sep.join([b_map_out_path, mn[i], 'map_result'])
    #     mp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '.json'])
    #     wp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '.h5'])
    #     main(mop, mp, wp)
    #
    # b_map_out_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_model_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_weights_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc'
    # for i in range(8, 10):
    #     mop = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'map_result'])
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar10_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar10_w' + str(i + 1) + '_fpl.h5'])
    #     main(mop, mp, wp)
    #
    # b_map_out_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    # b_model_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    # b_weights_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    # for i in range(8, 10):
    #     mop = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'map_result'])
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar100_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar100_w' + str(i + 1) + '_fpl.h5'])
    #     main(mop, mp, wp)
    #
    # b_map_out_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_model_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_weights_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc'
    # for i in range(13, 15):
    #     mop = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'map_result'])
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar10_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar10_w' + str(i + 1) + '_fpl.h5'])
    #     main(mop, mp, wp)

    b_map_out_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc_coco'
    b_model_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc_coco'
    b_weights_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc_coco'
    for i in range(1):
        mop = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'map_result'])
        mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.json'])
        wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.h5'])
        main(mop, mp, wp)
