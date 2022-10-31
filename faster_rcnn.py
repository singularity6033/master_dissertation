import colorsys
import os
import time

import numpy as np
from PIL import ImageDraw, ImageFont
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.models import load_model

import nets.faster_rcnn_model as frcnn
from utils import config
from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, get_new_img_size, resize_image
from utils.bbox import BBoxUtility


class faster_rcnn(object):
    _defaults = {
        "model_path": './vgg16_cifar100_fpl_tf2/model/s13_-5.json',
        "weights_path": './vgg16_cifar100_fpl_tf2/logs_frcnn_voc/w13/ep100-loss2.237-val_loss2.329.h5',
        "chop_idx": -5,
        "stride": 16,
        "out_channel": 512,
        "classes_path": os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt'),
        "confidence": 0.01,
        "nms_iou": 0.6,
        'anchors_size': [64, 128, 256],
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # initialization
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # get names of class and num of anchors
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.num_classes = self.num_classes + 1
        # bbox utility
        self.bbox_util = BBoxUtility(self.num_classes, classifier_nms=self.nms_iou, max_after_nms=150)

        # one color for one class when drawing bounding box
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.model_rpn, self.model_classifier = \
            frcnn.get_predict_model(self.model_path, self.num_classes, self.chop_idx, self.out_channel)
        self.generate()

    # load model
    def generate(self):
        model_path = os.path.expanduser(self.weights_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # load weights
        self.model_rpn.load_weights(self.weights_path, by_name=True)
        self.model_classifier.load_weights(self.weights_path, by_name=True)
        # print('{} model, anchors, and classes loaded.'.format(self.weights_path))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image = cvtColor(image)
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        # expand one dimension to store batch size
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        rpn_pred = self.model_rpn(image_data)
        rpn_pred = [x.numpy() for x in rpn_pred]

        anchors = get_anchors(input_shape, self.anchors_size, self.stride)
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)

        classifier_pred = self.model_classifier([rpn_pred[2], rpn_results[:, :, [1, 0, 3, 2]]])
        classifier_pred = [x.numpy() for x in classifier_pred]

        results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                          self.confidence)
        # no results return origins
        if len(results[0]) == 0:
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        # set font style and size
        font = ImageFont.truetype(font='others/OpenSans-Light.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // input_shape[0], 1)

        # plot boxes
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for th in range(thickness):
                draw.rectangle([left + th, top + th, right - th, bottom - th], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_fps(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image = cvtColor(image)
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        rpn_pred = self.model_rpn(image_data)
        rpn_pred = [x.numpy() for x in rpn_pred]

        anchors = get_anchors(input_shape, self.anchors_size)
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)

        classifier_pred = self.model_classifier([rpn_pred[2], rpn_results[:, :, [1, 0, 3, 2]]])
        classifier_pred = [x.numpy() for x in classifier_pred]

        results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                          self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            rpn_pred = self.model_rpn(image_data)
            rpn_pred = [x.numpy() for x in rpn_pred]

            anchors = get_anchors(input_shape, self.anchors_size)
            rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
            temp_ROIs = rpn_results[:, :, [1, 0, 3, 2]]

            classifier_pred = self.model_classifier([rpn_pred[2], temp_ROIs])
            classifier_pred = [x.numpy() for x in classifier_pred]

            results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                              self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + str(image_id) + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image = cvtColor(image)
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float64')), 0)

        rpn_pred = self.model_rpn(image_data)
        rpn_pred = [x.numpy() for x in rpn_pred]

        anchors = get_anchors(input_shape, self.anchors_size, self.stride)
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)

        # print('rpn_results size:', rpn_results.shape)
        # print('rpn_results', rpn_results)
        # print('rpn_results_new', rpn_results[:, :, [1, 0, 3, 2]])

        classifier_pred = self.model_classifier([rpn_pred[2], rpn_results[:, :, [1, 0, 3, 2]]])
        # print('classifier_pred size:', classifier_pred)
        classifier_pred = [x.numpy() for x in classifier_pred]

        results = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape,
                                                          self.confidence)
        if len(results[0]) <= 0:
            return
        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return
