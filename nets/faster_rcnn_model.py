from utils.change_nn_input_size import change_input_size
from tensorflow.keras.models import model_from_json
from nets.classifier import get_vgg_classifier
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from nets.rpn import get_rpn


def get_model(model_path, weights_path, num_classes, chop_idx=-5, num_anchors=9):
    with open(model_path, 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    # chop off the fc layer and the last pooling layer
    model = Model(inputs=model.input, outputs=model.layers[chop_idx].output)
    # model.summary()
    base_model = change_input_size(model, None, None, 3)
    base_layers = base_model.output
    roi_input = Input(shape=(None, 4))
    # base layer is the shared feature map from previous cnn
    # build region proposal network
    rpn = get_rpn(base_layers, num_anchors)
    # build classifier cnn
    classifier = get_vgg_classifier(base_layers, roi_input, 3, num_classes)
    model_rpn = Model(base_model.input, rpn)
    model_all = Model([base_model.input, roi_input], rpn + classifier)
    return model_rpn, model_all


def get_predict_model(model_path, num_classes, chop_idx=-5, out_channel=512, num_anchors=9):
    with open(model_path, 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    # chop off the fc layer and the last pooling layer
    model = Model(inputs=model.input, outputs=model.layers[chop_idx].output)
    # model.summary()
    base_model = change_input_size(model, None, None, 3)
    base_layers = base_model.output
    roi_input = Input(shape=(None, 4))
    # 512 means output shape of the last conv
    feature_map_input = Input(shape=(None, None, out_channel))
    rpn = get_rpn(base_layers, num_anchors)
    classifier = get_vgg_classifier(feature_map_input, roi_input, 3, num_classes)
    model_rpn = Model(base_model.input, rpn + [base_layers])
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only
