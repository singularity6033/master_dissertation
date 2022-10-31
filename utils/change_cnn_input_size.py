from tensorflow.keras.models import model_from_json


def change_input_size(model, h, w, ch=3):
    model._layers[0]._batch_input_shape = (None, h, w, ch)
    new_model = model_from_json(model.to_json())
    for layer, new_layer in zip(model.layers, new_model.layers):
        new_layer.set_weights(layer.get_weights())
    return new_model
