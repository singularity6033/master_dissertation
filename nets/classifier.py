import tensorflow as tf
import tensorflow.keras.backend as K
from nets.vgg import vgg_classifier_layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Flatten, Layer, TimeDistributed


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        self.num_channels = 3
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        input_shape2 = input_shape[1]
        return None, input_shape2[1], self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)
        # shared feature map
        feature_map = x[0]
        # region proposals
        rois = x[1]
        num_rois = tf.shape(rois)[1]
        batch_size = tf.shape(rois)[0]
        # generate index of rps to feed into crop_and_resize
        box_index = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index = tf.tile(box_index, (1, num_rois))
        box_index = tf.reshape(box_index, [-1])
        # find corresponding area in feature map
        rs = tf.image.crop_and_resize(feature_map, tf.reshape(rois, [-1, 4]), box_index,
                                      (self.pool_size, self.pool_size))
        final_output = K.reshape(rs, (batch_size, num_rois, self.pool_size, self.pool_size, self.num_channels))
        return final_output


def get_vgg_classifier(base_layers, input_rois, roi_size, num_classes):
    # batch_size, 37, 37, 512 -> batch_size, num_rois, 7, 7, 512
    out_roi_pool = RoiPoolingConv(roi_size)([base_layers, input_rois])

    # batch_size, num_rois, 7, 7, 512 -> batch_size, num_rois, 4096
    out_layer = vgg_classifier_layers(out_roi_pool)

    # batch_size, num_rois, 4096 -> batch_size, num_rois, num_classes
    out_classification = TimeDistributed(Dense(num_classes, activation='softmax',
                                         kernel_initializer=RandomNormal(stddev=0.02)),
                                         name='dense_class_{}'.format(num_classes))(out_layer)
    # batch_size, num_rois, 4096 -> batch_size, num_rois, 4 * (num_classes-1)
    out_regression = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear',
                                     kernel_initializer=RandomNormal(stddev=0.02)),
                                     name='dense_regress_{}'.format(num_classes))(out_layer)
    return [out_classification, out_regression]
