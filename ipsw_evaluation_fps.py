import colorsys
import math
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ipsw_predict import ipsw_detect_image

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(model_path, weight_path, vp, vsp, fpsp, v_fps=25):
    capture = cv2.VideoCapture(vp)
    if not vsp == "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(vsp, fourcc, v_fps, size)

    fps = 0.0
    num_frames = 0
    t0 = time.time()
    while True:
        success, frame = capture.read()
        # pillow image is RGB but opencv image is BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(ipsw_detect_image(frame, model_path, weight_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not vsp == "":
            out.write(frame)
        num_frames += 1
    t1 = time.time()
    fps = num_frames / (t1 - t0)
    f = open(fpsp, "a")
    f.write("%s\n" % (str(fps)))
    f.close()
    capture.release()


if __name__ == '__main__':
    video_path = './video_test/1.mp4'

    b_map_out_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_voc'
    b_model_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_voc'
    b_weights_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_voc'
    fps_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_voc/vgg11_cifar10_w1_to_w10_fps.txt'
    for i in range(10):
        mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar10_w' + str(i + 1) + '_fpl.json'])
        wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar10_w' + str(i + 1) + '_fpl.h5'])
        video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test'])
        main(mp, wp, video_path, video_save_path, fps_path)

    b_map_out_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_voc'
    b_model_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_voc'
    b_weights_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_voc'
    fps_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_voc/vgg11_cifar100_w1_to_w10_fps.txt'
    for i in range(10):
        mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar100_w' + str(i + 1) + '_fpl.json'])
        wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar100_w' + str(i + 1) + '_fpl.h5'])
        video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test'])
        main(mp, wp, video_path, video_save_path, fps_path)

    b_map_out_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_voc'
    b_model_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_voc'
    b_weights_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_voc'
    fps_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_voc/vgg16_cifar10_w1_to_w15_fps.txt'
    for i in range(15):
        mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.json'])
        wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.h5'])
        video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test'])
        main(mp, wp, video_path, video_save_path, fps_path)

    b_map_out_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_voc'
    b_model_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_voc'
    b_weights_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_voc'
    fps_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_voc/vgg16_cifar100_w1_to_w15_fps.txt'
    for i in range(15):
        mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.json'])
        wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.h5'])
        video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test'])
        main(mp, wp, video_path, video_save_path, fps_path)

    b_map_out_path = './sgd_tf2/logs_ipsw'
    b_model_path = './sgd_tf2/logs_ipsw'
    b_weights_path = './sgd_tf2/logs_ipsw'
    mn = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
    b_fps_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_voc'
    for i in range(4):
        mp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '_sgd.json'])
        wp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '_sgd.h5'])
        video_save_path = os.path.sep.join([b_map_out_path, mn[i], 'video_test'])
        fps_path = os.path.sep.join([b_fps_path, mn, 'fps.txt'])
        main(mp, wp, video_path, video_save_path, fps_path)
