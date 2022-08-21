import os
import random

import cv2
import numpy as np
from PIL import Image
import math

from utils import config
from utils.iou import compute_iou
from utils.utils import cvtColor
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

"""
A basic way to convert a image classifier to object detectors (linear SVM + HOG)
there are 2 key ingredients:
1. image pyramid: find objects in images at different scales (sizes)
2. sliding window: a fixed-size rectangle that slides from left-to-right and top-to-bottom within an image
combine with image pyramids, sliding windows allow us to localize objects at different locations and
multiple scales of the input image.
"""


class ipsw_dataset_generator:
    def __init__(self, input_shape=None, scale=None, win_step=None, roi_sizes=None, category=None, dataset_name=None,
                 train=True):
        self.input_shape = input_shape
        self.scale = scale
        self.win_step = win_step  # sliding step
        self.roi_sizes = roi_sizes  # sliding window size
        self.category = category
        self.dataset_name = dataset_name
        self.train = train

    def train_generator(self, input_image_line, label_counter):
        if self.dataset_name == 'voc':
            base_path = config.VOC_IPSW_BASE_PATH
            train_path = config.VOC_IPSW_TRAIN_PATH
            val_path = config.VOC_IPSW_VAL_PATH
        elif self.dataset_name == 'coco':
            base_path = config.COCO_IPSW_BASE_PATH
            train_path = config.COCO_IPSW_TRAIN_PATH
            val_path = config.COCO_IPSW_VAL_PATH
        line = input_image_line.split()
        max_bg_num = 1
        if len(line) <= 1:
            return label_counter
        rois = []
        locs = []
        # load image
        raw_image = Image.open(line[0])
        H, W = raw_image.size
        img_prs = self.image_pyramid(raw_image)
        for img_pr in img_prs:
            # determine the scale factor between the *original* image
            # dimensions and the *current* layer of the pyramid
            scale = W / float(img_pr.size[1])
            # for each layer of the image pyramid, loop over the sliding
            # window locations
            for (x, y, roiOrig) in self.sliding_window(img_pr):
                # scale the (x, y)-coordinates of the ROI with respect to the *original* image dimensions
                x = x * scale
                y = y * scale
                w = roiOrig.size[0] * scale
                h = roiOrig.size[1] * scale
                # update our list of ROIs and associated coordinates
                rois.append(roiOrig)
                locs.append((x, y, x + w, y + h))

        # load gt regions
        gt_boxes = []
        for j in range(1, len(line)):
            xMin, yMin, xMax, yMax, label_id = line[j].split(',')
            gt_boxes.append((float(xMin), float(yMin), float(xMax), float(yMax), label_id))
        imgPath = None
        bg_rois = []
        # loop over the region got from sliding window + image pyramid
        for loc_id, loc in enumerate(locs):
            negative_count = 0
            roi = rois[loc_id]
            # extract locations of predicted bounding boxes
            x_start_pred, y_start_pred, x_end_pred, y_end_pred = loc
            w_pred, h_pred = x_end_pred - x_start_pred, y_end_pred - y_start_pred
            # loop over the ground-truth bounding boxes
            for gtBox in gt_boxes:
                # compute the intersection over union between the two boxes and unpack the ground-truth bounding box
                iou = compute_iou(gtBox[:-1], loc)
                label_id = gtBox[-1]
                # extract locations of predicted bounding boxes
                x_start_gt, y_start_gt, x_end_gt, y_end_gt = gtBox[:-1]
                w_gt, h_gt = x_end_gt - x_start_gt, y_end_gt - y_start_gt
                # determine if the predicted bounding box falls within the ground-truth bounding box
                fullOverlap = x_start_pred >= x_start_gt
                fullOverlap = fullOverlap and y_start_pred >= y_start_gt
                fullOverlap = fullOverlap and x_end_pred <= x_end_gt
                fullOverlap = fullOverlap and y_end_pred <= y_end_gt
                # check to see if the IOU is greater than 75%
                if iou >= 0.8:
                    # extract the output roi
                    # roi = get_random_data(rois[loc_id], self.input_shape, random=self.train)
                    roi = roi.resize((self.input_shape[0], self.input_shape[1]), Image.BICUBIC)
                    label = self.category[int(label_id)]
                    if self.train:
                        imgPath = os.path.sep.join([train_path, label])
                        list_file = open(os.path.sep.join([base_path, 'train_dataset.txt']), 'a', encoding='utf-8')
                    else:
                        imgPath = os.path.sep.join([val_path, label])
                        list_file = open(os.path.sep.join([base_path, 'val_dataset.txt']), 'a', encoding='utf-8')
                    if not os.path.exists(imgPath):
                        os.makedirs(imgPath)
                    filename = "{}.png".format(label_counter[int(label_id), 0])
                    label_counter[int(label_id), 0] += 1
                    filePath = os.path.sep.join([imgPath, filename])
                    roi.save(filePath)
                    # bounding box regression
                    tx = (x_start_gt - x_start_pred) / w_pred
                    ty = (y_start_gt - y_start_pred) / h_pred
                    tw = math.log(w_gt / w_pred)
                    th = math.log(h_gt / h_pred)
                    list_file.write(filePath + ';' + ",".join([str(a) for a in [tx, ty, tw, th]])
                                    + ',' + str(label_id))
                    list_file.write('\n')
                    list_file.close()
        return label_counter

    def test_generator(self, img):
        rois = []
        locs = []
        # load image
        H, W = img.size
        img = img.resize((300, 300), Image.BICUBIC)
        H_s, W_s = H / 300, W / 300
        # H_s, W_s = 1, 1
        img_prs = self.image_pyramid(img)
        # t0 = time.time()
        for img_pr in img_prs:
            # determine the scale factor between the *original* image
            # dimensions and the *current* layer of the pyramid
            scale = W / float(img_pr.size[1])
            # for each layer of the image pyramid, loop over the sliding window locations
            for (x, y, roiOrig) in self.sliding_window(img_pr):
                # scale the (x, y)-coordinates of the ROI with respect to the *original* image dimensions
                x = x * scale * W_s
                y = y * scale * H_s
                w = roiOrig.size[0] * scale * W_s
                h = roiOrig.size[1] * scale * H_s
                # take the ROI and preprocess it so we can later classify
                # random data augmentation in training no such operation in validating
                roi = get_random_data(roiOrig, self.input_shape, random=self.train)
                # roi = roiOrig.resize((self.input_shape[0], self.input_shape[1]), Image.BICUBIC)
                # roi = get_random_data(roiOrig, self.input_shape, random=self.train)
                roi = img_to_array(roi)
                # update our list of ROIs and associated coordinates
                rois.append(roi)
                locs.append((x, y, x + w, y + h))
        # print(len(rois))
        return rois, locs

    def image_pyramid(self, image):
        max_roi_size_h = 0
        max_roi_size_w = 0
        for roi_size in self.roi_sizes:
            if roi_size[0] > max_roi_size_h:
                max_roi_size_h = roi_size[0]
            if roi_size[1] > max_roi_size_w:
                max_roi_size_w = roi_size[1]
        # pyramid image list
        images = [image]
        while True:
            # compute the dimensions of the next image in the pyramid
            h = int(image.size[0] / self.scale)
            w = int(image.size[1] / self.scale)
            image = image.resize((h, w), Image.BICUBIC)
            # if the resized image does not meet the supplied minimum size, then stop constructing the pyramid
            # image pyramid terminate size (top of the pyramid) is roi size
            if image.size[0] < max_roi_size_w or image.size[1] < max_roi_size_h:
                break
            images.append(image)
        return images

    def sliding_window(self, image):
        windows = []
        # slide a window across the image
        for roi_size in self.roi_sizes:
            for y in range(0, image.size[0] - roi_size[1], self.win_step):
                for x in range(0, image.size[1] - roi_size[0], self.win_step):
                    windows.append([x, y, image.crop((x, y, x + roi_size[0], y + roi_size[1]))])
        return windows


"""
Another way to convert a image classifier to object detectors (selective search)
original paper: Uijlings J R R... Selective search for object recognition
there are 2 improvements:
1. Be faster and more efficient than sliding windows and image pyramids
2. Accurately detect the regions of an image that could contain an object
"""


class ss_dataset_generator:
    def __init__(self, input_shape=None, category=None, dataset_name=None, train=True):
        self.input_shape = input_shape
        self.category = category
        self.dataset_name = dataset_name
        self.train = train

    def train_generator(self, input_image_line, label_counter):
        if self.dataset_name == 'voc':
            base_path = config.VOC_SS_BASE_PATH
            train_path = config.VOC_SS_TRAIN_PATH
            val_path = config.VOC_SS_VAL_PATH
        elif self.dataset_name == 'coco':
            base_path = config.COCO_SS_BASE_PATH
            train_path = config.COCO_SS_TRAIN_PATH
            val_path = config.COCO_SS_VAL_PATH
        line = input_image_line.split()
        if len(line) <= 1:
            return label_counter
        rois = []
        locs = []
        # load image
        image = cv2.imread(line[0])
        # run selective search on the image and initialize our list of proposed boxes
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        # loop over the rectangles generated by selective search
        for (x, y, w, h) in rects:
            # convert our bounding boxes from (x, y, w, h) to (startX, startY, endX, endY)
            locs.append((x, y, x + w, y + h))

        # load gt regions
        gt_boxes = []
        for j in range(1, len(line)):
            xMin, yMin, xMax, yMax, label_id = line[j].split(',')
            gt_boxes.append((float(xMin), float(yMin), float(xMax), float(yMax), label_id))
        imgPath = None
        # loop over the region got from sliding window + image pyramid
        for loc in locs:
            negative_count = 0
            # extract locations of predicted bounding boxes
            x_start_pred, y_start_pred, x_end_pred, y_end_pred = loc
            w_pred, h_pred = x_end_pred - x_start_pred, y_end_pred - y_start_pred
            # loop over the ground-truth bounding boxes
            for gtBox in gt_boxes:
                # compute the intersection over union between the two boxes and unpack the ground-truth bounding box
                iou = compute_iou(gtBox[:-1], loc)
                label_id = gtBox[-1]
                # extract locations of predicted bounding boxes
                x_start_gt, y_start_gt, x_end_gt, y_end_gt = gtBox[:-1]
                w_gt, h_gt = x_end_gt - x_start_gt, y_end_gt - y_start_gt
                # determine if the predicted bounding box falls within the ground-truth bounding box
                fullOverlap = x_start_pred >= x_start_gt
                fullOverlap = fullOverlap and y_start_pred >= y_start_gt
                fullOverlap = fullOverlap and x_end_pred <= x_end_gt
                fullOverlap = fullOverlap and y_end_pred <= y_end_gt
                # check to see if the IOU is greater than 75% or less than 0%
                if iou >= 0.8:
                    # extract the output roi
                    roi = image[y_start_pred:y_end_pred, x_start_pred:x_end_pred]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(roi, self.input_shape, interpolation=cv2.INTER_CUBIC)
                    label = self.category[int(label_id)]
                    if self.train:
                        imgPath = os.path.sep.join([train_path, label])
                        list_file = open(os.path.sep.join([base_path, 'train_dataset.txt']), 'a', encoding='utf-8')
                    else:
                        imgPath = os.path.sep.join([val_path, label])
                        list_file = open(os.path.sep.join([base_path, 'val_dataset.txt']), 'a', encoding='utf-8')
                    if not os.path.exists(imgPath):
                        os.makedirs(imgPath)
                    filename = "{}.png".format(label_counter[int(label_id), 0])
                    label_counter[int(label_id), 0] += 1
                    filePath = os.path.sep.join([imgPath, filename])
                    cv2.imwrite(filePath, roi)
                    # bounding box regression
                    tx = (x_start_gt - x_start_pred) / w_pred
                    ty = (y_start_gt - y_start_pred) / h_pred
                    tw = math.log(w_gt / w_pred)
                    th = math.log(h_gt / h_pred)
                    list_file.write(filePath + ';' + ",".join([str(a) for a in [tx, ty, tw, th]])
                                    + ',' + str(label_id))
                    list_file.write('\n')
                    list_file.close()
        return label_counter

    def test_generator(self, img):
        H, W, C = img.shape
        # img = cv2.resize(img, [300, 300], interpolation=cv2.INTER_CUBIC)
        # H_s, W_s = H / 300, W / 300
        H_s, W_s = 1, 1
        rois = []
        locs = []
        # run selective search on the image and initialize our list of proposed boxes
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        random.shuffle(rects)
        # loop over the rectangles generated by selective search
        for (x, y, w, h) in rects[:100]:
            roi = img[y:y + h, x:x + w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, self.input_shape, interpolation=cv2.INTER_CUBIC)
            rois.append(roi)
            # convert our bounding boxes from (x, y, w, h) to (startX, startY, endX, endY)
            # and rescale to original image size
            startX = x * W_s
            startY = y * H_s
            endX = (x + w) * W_s
            endY = (y + h) * H_s
            locs.append((startX, startY, endX, endY))
        return rois, locs


class fit_generator:
    def __init__(self, inputPath=None, input_shape=None, batch_size=None, lb=None, category=None, train=None):
        self.inputPath = inputPath
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lb = lb
        self.category = category
        self.train = train

    def generate(self):
        # open the CSV file for reading
        f = open(self.inputPath, "r")
        while True:
            # initialize our batches of images ,labels and boxes
            images = []
            labels = []
            boxes = []
            while len(images) < self.batch_size:
                # load image
                one_line = f.readline()
                if one_line == "":
                    # reset the file pointer to the beginning of the file and re-read the line
                    f.seek(0)
                    one_line = f.readline()
                one_line = one_line.split(";")
                image = Image.open(one_line[0])
                image = get_random_data(image, self.input_shape, random=self.train)
                # roi = get_random_data(image, self.input_shape, random=self.train)
                images.append(image)
                # load bounding label and boxes info
                for i in range(1, len(one_line)):
                    tx, ty, tw, th, label_id = one_line[i].split(',')
                labels.append(self.category[int(label_id)])
                boxes.append((float(tx), float(ty), float(tw), float(th)))
            labels = self.lb.transform(labels)
            # if self.train is not None:
            #     (images, labels) = next(self.train.flow(np.array(images), labels, batch_size=self.batch_size))
            trainTargets = {
                "class_label": labels,
                "bounding_box": np.array(boxes)
            }
            yield np.array(images), trainTargets


"""
faster rcnn training generator
"""


class faster_rcnn_dataset:
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, n_sample=256,
                 ignore_threshold=0.3, overlap_threshold=0.7):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.n_sample = n_sample
        self.ignore_threshold = ignore_threshold
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def generate(self):
        i = 0
        while True:
            image_data = []
            classifications = []
            regressions = []
            targets = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.annotation_lines)
                # random data augmentation in training no such operation in validating
                image, box = get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
                if len(box) != 0:
                    boxes = np.array(box[:, :4], dtype=np.float32)
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
                    box = np.concatenate([boxes, box[:, -1:]], axis=-1)

                assignment = self.assign_boxes(box)
                classification = assignment[:, 4]
                regression = assignment[:, :]
                # select positive and negative boxes
                # total number training samples is 256
                pos_index = np.where(classification > 0)[0]
                if len(pos_index) > self.n_sample / 2:
                    disable_index = np.random.choice(pos_index, size=(len(pos_index) - self.n_sample // 2),
                                                     replace=False)
                    classification[disable_index] = -1
                    regression[disable_index, -1] = -1

                # balance positive and negative boxes to meet 256 limit
                n_neg = self.n_sample - np.sum(classification > 0)
                neg_index = np.where(classification == 0)[0]
                if len(neg_index) > n_neg:
                    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
                    classification[disable_index] = -1
                    regression[disable_index, -1] = -1

                i = (i + 1) % self.length
                image_data.append(preprocess_input(image))
                classifications.append(np.expand_dims(classification, -1))
                regressions.append(regression)
                targets.append(box)

            yield np.array(image_data), [np.array(classifications, dtype=np.float32),
                                         np.array(regressions, dtype=np.float32)], targets

    def iou(self, box):
        # calculate iou between anchors and gt boxes
        inter_ul = np.maximum(self.anchors[:, :2], box[:2])
        inter_br = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_br - inter_ul
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # area of gt boxes
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # area of anchors
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        union = area_true + area_gt - inter
        iou = inter / union
        return iou

    def encode_ignore_box(self, box, return_iou=True, variances=[0.25, 0.25, 0.25, 0.25]):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_anchors, 1))
        # find anchors which fall in the ignore range
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        # find positive boxes with high ious
        assign_mask = iou > self.overlap_threshold
        # if there is no such a box satisfy the threshold pick the maximum one as positive boxes
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # find corresponding anchors
        assigned_anchors = self.anchors[assign_mask]
        # turn gt boxes into center+wh format
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # turn anchors into center+wh format
        assigned_anchors_center = 0.5 * (assigned_anchors[:, :2] + assigned_anchors[:, 2:4])
        assigned_anchors_wh = assigned_anchors[:, 2:4] - assigned_anchors[:, :2]

        # find result from faster rcnn (inverse prediction)
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes):
        # assignment[:4]: regression results
        # assignment[4]: label (whether has object or not)
        assignment = np.zeros((self.num_anchors, 4 + 1))
        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment

        # calculate ious for each gt box
        apply_along_axis_boxes = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ignored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        # after reshape: shape of ignored_boxes is [num_true_box, num_anchors, 1] where 1 is iou
        ignored_boxes = ignored_boxes.reshape(-1, self.num_anchors, 1)
        ignore_iou = ignored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1
        # after reshape: shape of encoded_boxes is [num_true_box, num_anchors, 4+1] where 4 is encoded results, 1 is iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        # for each anchor find a gt box with maximum iou
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)

        # get encoded gt box with best idx of anchors
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4 represents the prob of background
        # it needs to be assigned to 1 if there is an object
        assignment[:, 4][best_iou_mask] = 1
        # we get predicted results of one input image (not gt)
        return assignment


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    image = cvtColor(image)
    # get image size and target size
    iw, ih = image.size
    h, w = input_shape
    if not random:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        # add gray stripe to remaining images
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        return image_data

    # scale img and warp width and height
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # add gray stripe to remaining images
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flipping
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # warping in chrominance
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1
    return image_data
