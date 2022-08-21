from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Reshape


# region proposal network
def get_rpn(base_layers, num_anchors):
    # use 3*3*512 convolutional kernels to concatenate features
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02),
               name='rpn_conv1')(base_layers)
    # use 1*1 convolutional kernel to adjust number of channels
    x_classification = Conv2D(num_anchors, (1, 1), activation='sigmoid',
                              kernel_initializer=RandomNormal(stddev=0.02),
                              name='rpn_out_class')(x)
    x_regression = Conv2D(num_anchors * 4, (1, 1), activation='linear',
                          kernel_initializer=RandomNormal(stddev=0.02),
                          name='rpn_out_regress')(x)

    x_classification = Reshape((-1, 1), name="classification")(x_classification)
    x_regression = Reshape((-1, 4), name="regression")(x_regression)
    return [x_classification, x_regression]
