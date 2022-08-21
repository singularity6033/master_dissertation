import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from faster_rcnn import faster_rcnn

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

frcnn = faster_rcnn()
"""
there are 2 modes representing ways to predict based on faster rcnn
1. mode == 'predict' ---> prediction for single image
2. mode == 'dir_predict' ---> prediction for image list
"""
mode = "dir_predict"
dir_origin_path = "img_test/"
dir_save_path = "img_test_output/"

if mode == "predict":
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = frcnn.detect_image(image)
            r_image.show()

elif mode == "dir_predict":
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg',
                                      '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            r_image = frcnn.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name))
else:
    raise AssertionError("Please specify the correct mode.")




