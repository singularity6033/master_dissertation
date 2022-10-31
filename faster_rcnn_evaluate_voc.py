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


def main(map_out_path, model_path, weights_path, chop_idx, stride, out_channel):
    """
    map_mode == 0: predictions + gts + voc_map
    map_mode == 1: predictions
    map_mode == 2: gts
    map_mode == 3: voc_map
    map_mode == 4: use coco api to calculate map based with iou 0.50:0.95, need to format predictions and gts
    """

    map_mode = 0
    VOCdevkit_path = './dataset/VOCdevkit'
    classes_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt')
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Main/test.txt")).read().strip().split()
    # MINOVERLAP --> mAP0.x
    MINOVERLAP = 0.05
    # visualization of map calculations
    map_vis = False

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = faster_rcnn(model_path=model_path, weights_path=weights_path, chop_idx=chop_idx, stride=stride,
                            out_channel=out_channel)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2012/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2012/Annotations/" + image_id + ".xml")).getroot()
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

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")


if __name__ == '__main__':
    # map result dir
    b_map_out_path = './vgg16_cifar10_fpl_tf2/logs_frcnn_voc'
    b_model_path = './vgg16_cifar10_fpl_tf2/model'
    b_weights_path = './vgg16_cifar10_fpl_tf2/logs_frcnn_voc'
    wn = ['ep200-loss2.426-val_loss2.198.h5', 'ep050-loss2.472-val_loss2.173.h5', 'ep050-loss2.182-val_loss1.905.h5',
          'ep050-loss2.389-val_loss2.160.h5', 'ep050-loss2.875-val_loss2.977.h5', 'ep050-loss2.778-val_loss2.843.h5',
          'ep050-loss2.887-val_loss2.960.h5', 'ep050-loss2.781-val_loss2.941.h5', 'ep050-loss2.989-val_loss3.198.h5',
          'ep050-loss2.773-val_loss2.952.h5']

    ci = [-4, -5, -4, -5, -4, -4, -5, -4, -4, -5, -4, -4, -5]
    st = [1, 1, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16]
    oc = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    for i in range(5, 6):
        mop = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'map_result'])
        mp = os.path.sep.join([b_model_path, 's' + str(i + 1) + '_' + str(ci[i]) + '.json'])
        wp = os.path.sep.join([b_weights_path, 'w' + str(i + 1), wn[i - 5]])
        main(mop, mp, wp, ci[i], st[i], oc[i])

    # b_map_out_path = './sgd_tf2/logs_frcnn_voc/vgg16_cifar10'
    # b_model_path = './sgd_tf2/model'
    #
    # mop = os.path.sep.join([b_map_out_path, 'map_result'])
    # mp = os.path.sep.join([b_model_path, 'sgd_vgg16_cifar10_-11.json'])
    # wp = os.path.sep.join([b_map_out_path, 'ep050-loss1.951-val_loss2.023.h5'])
    # main(mop, mp, wp, -11, 16, 512)

    # b_map_out_path = './sgd_tf2/logs_frcnn_voc/vgg16_cifar100'
    # b_model_path = './sgd_tf2/model'
    #
    # mop = os.path.sep.join([b_map_out_path, 'map_result'])
    # mp = os.path.sep.join([b_model_path, 'sgd_vgg16_cifar100_-11.json'])
    # wp = os.path.sep.join([b_map_out_path, 'ep050-loss1.960-val_loss2.066.h5'])
    # main(mop, mp, wp, -11, 16, 512)
