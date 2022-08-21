import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class BBoxUtility(object):
    def __init__(self, num_classes, max_pre_nms=12000, rpn_nms=0.7, classifier_nms=0.3, max_after_nms=300):
        # number of classes
        self.num_classes = num_classes
        # maximum number of rps before nms
        self._max_pre_nms = max_pre_nms
        # ious for rpn and classifier
        self.rpn_nms_iou = rpn_nms
        self.classifier_nms_iou = classifier_nms
        # maximum number of rps after nms
        self._max_after_nms = max_after_nms

    @staticmethod
    def _decode_boxes(mbox_loc, anchors, variances):
        # get shapes of anchors
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # get centers of anchors
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])
        # shifts of detected rps wrt. anchors
        # print('mbox_loc: ', mbox_loc[:, 0].shape)
        # print('anchor_width: ', anchor_width.shape)
        detections_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        detections_center_x += anchor_center_x
        detections_center_y = mbox_loc[:, 1] * anchor_height * variances[1]
        detections_center_y += anchor_center_y

        # get shapes of detected rps
        detections_width = np.exp(mbox_loc[:, 2] * variances[2])
        detections_width *= anchor_width
        detections_height = np.exp(mbox_loc[:, 3] * variances[3])
        detections_height *= anchor_height

        # get coordinates of left top and right bottom corners
        detections_tl_x = detections_center_x - 0.5 * detections_width
        detections_tl_y = detections_center_y - 0.5 * detections_height
        detections_br_x = detections_center_x + 0.5 * detections_width
        detections_br_y = detections_center_y + 0.5 * detections_height

        # concatenation
        detections = np.concatenate((detections_tl_x[:, None],
                                     detections_tl_y[:, None],
                                     detections_br_x[:, None],
                                     detections_br_y[:, None]), axis=-1)

        # in case of overflowing (restrict the range of detection rp coordinates to [0, 1])
        detections = np.minimum(np.maximum(detections, 0.0), 1.0)
        return detections

    @staticmethod
    def frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_min = box_yx - (box_hw / 2.)
        box_max = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_min[..., 0:1], box_min[..., 1:2],
                                box_max[..., 0:1], box_max[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def detection_out_rpn(self, predictions, anchors, variances=[0.25, 0.25, 0.25, 0.25]):
        # get confidence of class (-1, 0, 1)
        mbox_conf = predictions[0]
        # get prediction from bounding box regression
        mbox_loc = predictions[1]
        # print('mbox_loc[i]: ', mbox_loc[0].shape)
        # store the final selected region proposals
        results = []
        for i in range(len(mbox_loc)):
            detections = self._decode_boxes(mbox_loc[i], anchors, variances)
            # sort rps by their confidence
            boxes_conf = mbox_conf[i, :, 0]
            boxes_conf_sorted = np.argsort(boxes_conf)[::-1][:self._max_pre_nms]
            # pick a part of all rps as the number is too big
            boxes_conf_to_process = boxes_conf[boxes_conf_sorted]
            boxes_to_process = detections[boxes_conf_sorted, :]
            # use nms to filter out those rps with high ious
            idx = tf.image.non_max_suppression(boxes_to_process, boxes_conf_to_process, self._max_after_nms,
                                               iou_threshold=self.rpn_nms_iou).numpy()
            final_boxes = boxes_to_process[idx]
            results.append(final_boxes)
        return np.array(results)

    def detection_out_classifier(self, predictions, rpn_results, image_shape, input_shape, confidence=0.5,
                                 variances=[0.125, 0.125, 0.25, 0.25]):
        proposal_conf = predictions[0]
        proposal_loc = predictions[1]
        results = []
        for i in range(len(proposal_conf)):
            results.append([])
            # make use of results from classifier and decode rps, determine classes
            detections = []
            # calculate center, width and height of rps
            rpn_results[i, :, 2] = rpn_results[i, :, 2] - rpn_results[i, :, 0]
            rpn_results[i, :, 3] = rpn_results[i, :, 3] - rpn_results[i, :, 1]
            rpn_results[i, :, 0] = rpn_results[i, :, 0] + rpn_results[i, :, 2] / 2
            rpn_results[i, :, 1] = rpn_results[i, :, 1] + rpn_results[i, :, 3] / 2
            for j in range(proposal_conf[i].shape[0]):
                # calculate confidence of rps
                score = np.max(proposal_conf[i][j, :-1])
                label = np.argmax(proposal_conf[i][j, :-1])
                if score < confidence:
                    continue
                # get predicted rps
                x, y, w, h = rpn_results[i, j, :]
                tx, ty, tw, th = proposal_loc[i][j, 4 * label: 4 * (label + 1)]

                x1 = tx * variances[0] * w + x
                y1 = ty * variances[1] * h + y
                w1 = math.exp(tw * variances[2]) * w
                h1 = math.exp(th * variances[3]) * h

                x_min = x1 - w1 / 2.
                y_min = y1 - h1 / 2.
                x_max = x1 + w1 / 2
                y_max = y1 + h1 / 2

                detections.append([x_min, y_min, x_max, y_max, score, label])

            detections = np.array(detections)
            if len(detections) > 0:
                for c in range(self.num_classes):
                    c_confs_m = detections[:, -1] == c
                    if len(detections[c_confs_m]) > 0:
                        boxes_to_process = detections[:, :4][c_confs_m]
                        confs_to_process = detections[:, 4][c_confs_m]
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self._max_after_nms,
                                                           iou_threshold=self.classifier_nms_iou).numpy()
                        results[-1].extend(detections[c_confs_m][idx])

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2]+results[-1][:, 2:4])/2, results[-1][:, 2:4]-results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results
