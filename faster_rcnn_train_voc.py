# import the necessary packages
import time
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.change_cnn_input_size import change_input_size
from utils.utils_generator import fit_generator
from utils.utils import get_classes
import tensorflow as tf
from utils import config
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(save_path, model_path, weights_path, filename):
    # initialize the initial learning rate, number of epochs to train for and batch size
    init_lr = 1e-4
    num_epochs = 200
    batch_size = 32
    input_shape = [32, 32]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_name = 'coco'  # voc or coco
    method = 'ipsw'  # ipsw or ss
    # VOC2012
    if dataset_name == 'voc':
        if method == 'ipsw':
            train_annotation_path = os.path.join(config.VOC_IPSW_BASE_PATH, 'train_dataset.txt')
            val_annotation_path = os.path.join(config.VOC_IPSW_BASE_PATH, 'val_dataset.txt')
        elif method == 'ss':
            train_annotation_path = os.path.join(config.VOC_SS_BASE_PATH, 'train_dataset.txt')
            val_annotation_path = os.path.join(config.VOC_SS_BASE_PATH, 'val_dataset.txt')
        classes_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt')
    # COCO
    elif dataset_name == 'coco':
        if method == 'ipsw':
            train_annotation_path = os.path.join(config.COCO_IPSW_BASE_PATH, 'train_dataset.txt')
            val_annotation_path = os.path.join(config.COCO_IPSW_BASE_PATH, 'val_dataset.txt')
        elif method == 'ss':
            train_annotation_path = os.path.join(config.COCO_SS_BASE_PATH, 'train_dataset.txt')
            val_annotation_path = os.path.join(config.COCO_SS_BASE_PATH, 'val_dataset.txt')
        classes_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt')

    class_names, num_classes = get_classes(classes_path)

    lb = LabelBinarizer()
    lb.fit(list(class_names))

    data_gen = ImageDataGenerator(
        # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # load train and val set
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)
    epoch_step_train = num_train // batch_size
    epoch_step_val = num_val // batch_size

    # load pre-trained VGG CNN (cifar100) and drop off the head FC layer change the input size to (300, 300, 3)
    with open(model_path, 'r') as file:
        model_json = file.read()
    origin_model = model_from_json(model_json)
    origin_model.load_weights(weights_path)
    origin_model.summary()
    base_model = Model(inputs=origin_model.input, outputs=origin_model.layers[-4].output)
    base_model_modified = change_input_size(base_model, input_shape[0], input_shape[1], 3)
    # base_model.summary()
    baseHead = base_model_modified.output
    # pooling_out = base_model_modified.layers[-4].output
    flatten = Flatten(name="flatten_new")(baseHead)
    # construct the classification head of object detection model
    classifierHead = Dense(512, activation="relu", name="fc_1")(flatten)
    classifierHead = Dropout(0.5)(classifierHead)
    classifierHead = Dense(512, activation="relu", name="fc_2")(classifierHead)
    classifierHead = Dropout(0.5)(classifierHead)
    classifierHead = Dense(num_classes, activation="softmax", name="class_label")(classifierHead)
    # construct the regression head of object detection model
    bboxHead = Dense(128, activation="relu", name="bbox_fc1")(flatten)
    bboxHead = Dense(64, activation="relu", name="bbox_fc2")(bboxHead)
    bboxHead = Dense(32, activation="relu", name="bbox_fc3")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
    model = Model(inputs=base_model_modified.input, outputs=(bboxHead, classifierHead))

    # training process
    # initialize both the training and val image generators
    gen_train = fit_generator(train_annotation_path, input_shape, batch_size, lb,
                              category=class_names, train=False).generate()
    gen_val = fit_generator(val_annotation_path, input_shape, batch_size, lb,
                            category=class_names, train=False).generate()

    # freeze all layers before new head
    for layer in base_model_modified.layers:
        layer.trainable = False

    # define a dictionary to set the loss methods
    # categorical cross-entropy for the class label head and mean absolute error for the bounding box head
    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",
    }
    # define a dictionary that specifies the weights per loss
    # both the class label and bounding box outputs will receive equal weight
    lossWeights = {
        "class_label": 1.0,
        "bounding_box": 1.0
    }

    # compile model
    opt = Adam(lr=init_lr)
    model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

    # save the best model
    file_path = os.path.sep.join([save_path, filename + ".h5"])
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fine-tuning the head of the network
    print("[INFO] training head (freeze process)...")
    start_time = time.time()
    H = model.fit(
        gen_train,
        steps_per_epoch=epoch_step_train,
        validation_data=gen_val,
        validation_steps=epoch_step_val,
        epochs=num_epochs,
        callbacks=callbacks_list,
        verbose=1)

    print('training time is: ', time.time() - start_time, 's')

    # Save model to disk
    print("Save freeze model structure to disk")
    # model.save_weights(os.path.sep.join([save_path, filename + ".h5"]))
    model_json = model.to_json()
    with open(os.path.sep.join([save_path, filename + ".json"]), "w") as json_file:
        json_file.write(model_json)

    loss_labels = ["loss", "class_label_loss", "bounding_box_loss",
                   "val_loss", "val_class_label_loss", "val_bounding_box_loss"]
    accuracy_labels = ["class_label_accuracy", "bounding_box_accuracy",
                       "val_class_label_accuracy", "val_bounding_box_accuracy"]

    # print(H.history)
    # save records of loss and accuracy
    for ll in loss_labels:
        with open(os.path.join(save_path, filename + "_" + ll + ".txt"), 'a') as f:
            f.write(str(H.history[ll]))
            f.write("\n")
    for al in accuracy_labels:
        with open(os.path.join(save_path, filename + "_" + al + ".txt"), 'a') as f:
            f.write(str(H.history[al]))
            f.write("\n")

    # plot the training loss and accuracy
    N = num_epochs
    print("[INFO] plotting results...")

    plt.style.use("ggplot")
    plt.figure()
    for ll in ["loss", "val_loss"]:
        plt.plot(np.arange(0, N), H.history[ll])
    plt.legend(['train', 'test'])
    plt.title('loss')
    plt.savefig(os.path.sep.join([save_path, filename + "_loss.png"]), dpi=300, format="png")
    plt.figure()
    for al in ["class_label_accuracy", "val_class_label_accuracy"]:
        plt.plot(np.arange(0, N), H.history[al])
    plt.legend(['train', 'test'])
    plt.title('accuracy')
    plt.savefig(os.path.sep.join([save_path, filename + "_accuracy.png"]), dpi=300, format="png")


if __name__ == '__main__':
    # b_map_out_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # b_model_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
    # b_weights_path = './sgd_tf2/logs_ipsw_morefc'
    # mn = ['sgd_vgg11_cifar10', 'sgd_vgg11_cifar100', 'sgd_vgg16_cifar10', 'sgd_vgg16_cifar100']
    # for i in range(4):
    #     mop = os.path.sep.join([b_map_out_path, mn[i], 'map_result'])
    #     mp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '.json'])
    #     wp = os.path.sep.join([b_map_out_path, mn[i], 'ipsw_' + mn[i] + '.h5'])
    #     main(mop, mp, wp)

    model_names = ['s1_-4', 's2_-5', 's3_-4', 's4_-5', 's5_-4', 's6_-4', 's7_-5', 's8_-4', 's9_-4', 's10_-5',
                   's11_-4', 's12_-4', 's13_-5', 's14_-7', 's15_-9']
    save_root_path = './vgg16_cifar100_fpl_tf2/logs_ipsw_morefc_coco_no_aug'
    model_root_path = './vgg16_cifar100_fpl_tf2/model'
    weights_root_path = './vgg16_cifar100_fpl_tf2/weights'
    for i in range(1, 2):
        sp = os.path.sep.join([save_root_path, 'w' + str(i)])
        mp = os.path.sep.join([model_root_path, model_names[i - 1] + '.json'])
        wp = os.path.sep.join([weights_root_path, 'w' + str(i) + '.h5'])
        fn = 'ipsw_vgg16_cifar100_w' + str(i) + '_fpl'
        main(sp, mp, wp, fn)

    # save_root_path = './vgg16_cifar10_fpl_tf2/logs_ipsw_morefc_0.7'
    # model_root_path = './vgg16_cifar10_fpl_tf2/model'
    # weights_root_path = './vgg16_cifar10_fpl_tf2/weights'
    # for i in range(14, 15):
    #     sp = os.path.sep.join([save_root_path, 'w' + str(i)])
    #     mp = os.path.sep.join([model_root_path, model_names[i - 1] + '.json'])
    #     wp = os.path.sep.join([weights_root_path, 'w' + str(i) + '.h5'])
    #     fn = 'ipsw_vgg16_cifar10_w' + str(i) + '_fpl'
    #     main(sp, mp, wp, fn)
    #
    # model_names = ['s1_-5', 's2_-5', 's3_-4', 's4_-5', 's5_-4', 's6_-5', 's7_-4', 's8_-5', 's9_-7', 's10_-9']
    # save_root_path = './vgg11_cifar10_fpl_tf2/logs_ipsw_morefc_bbox'
    # model_root_path = './vgg11_cifar10_fpl_tf2/model'
    # weights_root_path = './vgg11_cifar10_fpl_tf2/weights'
    # for i in range(9, 10):
    #     sp = os.path.sep.join([save_root_path, 'w' + str(i)])
    #     mp = os.path.sep.join([model_root_path, model_names[i - 1] + '.json'])
    #     wp = os.path.sep.join([weights_root_path, 'w' + str(i) + '.h5'])
    #     fn = 'ipsw_vgg11_cifar10_w' + str(i) + '_fpl'
    #     main(sp, mp, wp, fn)
    #
    # save_root_path = './vgg11_cifar100_fpl_tf2/logs_ipsw_morefc'
    # model_root_path = './vgg11_cifar100_fpl_tf2/model'
    # weights_root_path = './vgg11_cifar100_fpl_tf2/weights'
    # for i in range(9, 10):
    #     sp = os.path.sep.join([save_root_path, 'w' + str(i)])
    #     mp = os.path.sep.join([model_root_path, model_names[i - 1] + '.json'])
    #     wp = os.path.sep.join([weights_root_path, 'w' + str(i) + '.h5'])
    #     fn = 'ipsw_vgg11_cifar100_w' + str(i) + '_fpl'
    #     main(sp, mp, wp, fn)
