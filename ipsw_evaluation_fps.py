import colorsys
import math
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

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
    with tqdm(total=200) as pbar:
        while num_frames < 200:
            # print('processing frame {}'.format(num_frames))
            success, frame = capture.read()
            # pillow image is RGB but opencv image is BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(ipsw_detect_image(frame, model_path, weight_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not vsp == "":
                out.write(frame)
            num_frames += 1
            pbar.update(1)
    t1 = time.time()
    fps = num_frames / (t1 - t0)
    f = open(fpsp, "a")
    f.write("%s\n" % (str(fps)))
    f.close()
    capture.release()


if __name__ == '__main__':
    video_path = './videos/1.mp4'

    # b_map_out_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_model_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_weights_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc'
    # fps_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc/vgg11_cifar10_ipsw_fps.txt'
    # for i in range(9, 10):
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar10_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar10_w' + str(i + 1) + '_fpl.h5'])
    #     video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test.mp4'])
    #     if i == 2:
    #         main(mp, wp, video_path, video_save_path, fps_path)
    #     else:
    #         main(mp, wp, video_path, '', fps_path)

    b_map_out_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    b_model_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    b_weights_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    fps_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc/vgg11_cifar100_ipsw_fps.txt'
    for i in range(9, 10):
        mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar100_w' + str(i + 1) + '_fpl.json'])
        wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg11_cifar100_w' + str(i + 1) + '_fpl.h5'])
        video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test.mp4'])
        if i == 5:
            main(mp, wp, video_path, video_save_path, fps_path)
        else:
            main(mp, wp, video_path, '', fps_path)
    #
    # b_map_out_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_model_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc'
    # b_weights_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc'
    # fps_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc/vgg16_cifar10_ipsw_fps.txt'
    # for i in range(14, 15):
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar10_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar10_w' + str(i + 1) + '_fpl.h5'])
    #     video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test.mp4'])
    #     if i == 2:
    #         main(mp, wp, video_path, video_save_path, fps_path)
    #     else:
    #         main(mp, wp, video_path, '', fps_path)

    # b_map_out_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc'
    # b_model_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc'
    # b_weights_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc'
    # fps_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc/vgg16_cifar100_ipsw_fps.txt'
    # for i in range(14, 15):
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.h5'])
    #     video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test.mp4'])
    #     if i == 6:
    #         main(mp, wp, video_path, video_save_path, fps_path)
    #     else:
    #         main(mp, wp, video_path, '', fps_path)
    #
    # b_map_out_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # b_model_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # b_weights_path = './sgd_tf2/logs_ipsw_morefc'
    # mn = ['sgd_vgg11_cifar10', 'sgd_vgg11_cifar100', 'sgd_vgg16_cifar10', 'sgd_vgg16_cifar100']
    # b_fps_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # for i in range(3, 4):
    #     mp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '.json'])
    #     wp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '.h5'])
    #     video_save_path = os.path.sep.join([b_map_out_path, mn[i], 'video_test'])
    #     fps_path = os.path.sep.join([b_fps_path, mn[i], 'fps.txt'])
    #     main(mp, wp, video_path, '', fps_path)

    # b_map_out_path = './vgg16_cifar100_fpl_tf2/logs_ss_morefc'
    # b_model_path = './vgg16_cifar100_fpl_tf2/logs_ss_morefc'
    # b_weights_path = './vgg16_cifar100_fpl_tf2/logs_ss_morefc'
    # fps_path = './vgg16_cifar100_fpl_tf2/logs_ss_morefc/vgg16_cifar100_ss_fpl_fps1.txt'
    # for i in range(14, 15):
    #     mp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.json'])
    #     wp = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'ipsw_vgg16_cifar100_w' + str(i + 1) + '_fpl.h5'])
    #     video_save_path = os.path.sep.join([b_map_out_path, 'w' + str(i + 1), 'video_test.mp4'])
    #     main(mp, wp, video_path, '', fps_path)

    # b_map_out_path = './sgd_tf2_ndn_best/logs_ss_morefc'
    # b_model_path = './sgd_tf2_ndn_best/logs_ss_morefc'
    # b_weights_path = './sgd_tf2/logs_ss_morefc'
    # mn = 'sgd_vgg16_cifar100'
    # b_fps_path = './sgd_tf2_ndn_best/logs_ss_morefc'
    # mp = os.path.sep.join([b_map_out_path, mn, 'ipsw_' + mn + '.json'])
    # wp = os.path.sep.join([b_map_out_path, mn, 'ipsw_' + mn + '.h5'])
    # # video_save_path = os.path.sep.join([b_map_out_path, mn[i], 'video_test'])
    # fps_path = os.path.sep.join([b_fps_path, mn, 'fps.txt'])
    # main(mp, wp, video_path, '', fps_path)
