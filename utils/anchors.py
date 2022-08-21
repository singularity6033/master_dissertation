import math
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

'''
    using 3 scales and 3 aspect ratios to generate 3*3=9 basic anchor boxes
    3 squares (s+m+l) and 6 rectangles 2*(s+m+l)
'''


def generate_anchor_basic(sizes=[64, 128, 256], ratios=[[1, 1], [1, 2], [2, 1]]):
    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))
    # 1*3 to 2*9 --> transpose --> 9*2
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    for i in range(len(ratios)):
        anchors[3 * i: 3 * i + 3, 2] = anchors[3 * i: 3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i: 3 * i + 3, 3] = anchors[3 * i: 3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


'''
    generate a group of anchor boxes based on shape of the feature map
    (each network point has 9 basic anchor boxes)
'''


def generate_anchor_group(shape, anchors, stride=16):
    shift_x = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # compress to 1-d array
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    number_of_shifts = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.reshape(shifts, [number_of_shifts, 1, 4])
    shifted_anchors = np.reshape(shifted_anchors, [number_of_shifts * number_of_anchors, 4])
    return shifted_anchors


def get_vgg_output_length(height, width, pooling_num=4):
    def get_output_length(input_length):
        filter_sizes = [2, 2, 2, 2]
        padding = [0, 0, 0, 0]
        stride = 2
        for i in range(len(filter_sizes[:pooling_num])):
            input_length = math.floor((input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1)
        return input_length
    return get_output_length(height), get_output_length(width)


# stride should be modified to fit the size of feature map
# 16 - 4 * pooling layers
# 8 - 3 * pooling layers
# 4 - 2 * pooling layers
# 2 - 1 * pooling layers
# 1 - 0 * pooling layers
def get_anchors(input_shape, sizes=[64, 128, 256], stride=16, ratios=[[1, 1], [1, 2], [2, 1]]):
    feature_shape = get_vgg_output_length(input_shape[0], input_shape[1], int(math.log2(stride)))
    anchors = generate_anchor_basic(sizes=sizes, ratios=ratios)
    anchors = generate_anchor_group(feature_shape, anchors, stride=stride)
    anchors[:, ::2] /= input_shape[1]
    anchors[:, 1::2] /= input_shape[0]
    anchors = np.clip(anchors, 0, 1)
    return anchors


if __name__ == "__main__":
    a = get_vgg_output_length(32, 32, int(math.log2(1)))
    # y = generate_anchor_group([2, 2], a, 1)
    x = get_anchors([32, 32])
    print(x)
    # x = get_vgg_output_length(32, 32)
