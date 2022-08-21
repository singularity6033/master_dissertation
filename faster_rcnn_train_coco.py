import os
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from nets.faster_rcnn_model import get_model
from nets.faster_rcnn_training import (ProposalTargetCreator, classifier_cls_loss, classifier_smooth_l1, rpn_cls_loss,
                                       rpn_smooth_l1)
from utils import config
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.utils_generator import faster_rcnn_dataset
from utils.utils import get_classes
from utils.bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

classes_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt')
model_path = 'vgg11_structure_json/s1_-5.json'
weights_path = 'vgg11_cifar10_weights_tf/w1.h5'
# model_path = './vgg_cifar100/cifar100vgg_scratch.json'
# weights_path = './vgg_cifar100/cifar100vgg_scratch.h5'
input_shape = [300, 300]
anchors_size = [64, 128, 256]

Init_Epoch = 0
Freeze_Epoch = 5
Freeze_batch_size = 8
Freeze_lr = 1e-4

UnFreeze_Epoch = 40
Unfreeze_batch_size = 32
Unfreeze_lr = 1e-5

Freeze_Train = True

train_annotation_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_train_set.txt')
val_annotation_path = os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_val_set.txt')

class_names, num_classes = get_classes(classes_path)
num_classes += 1
anchors = get_anchors(input_shape, anchors_size)

K.clear_session()
model_rpn, model_all = get_model(model_path, weights_path, num_classes)
# model_all.summary()

callback = tf.summary.create_file_writer("logs")
loss_history = LossHistory("logs/")

bbox_util = BBoxUtility(num_classes)
roi_helper = ProposalTargetCreator(num_classes)

# load train and val set
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)
print(len(model_all.layers))

# for standard vgg16 network 65-12
# freeze_layers = 53
freeze_layers = 3
if Freeze_Train:
    for i in range(freeze_layers):
        if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
            model_all.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))

# freeze training training
if True:
    batch_size = Freeze_batch_size
    lr = Freeze_lr
    start_epoch = Init_Epoch
    end_epoch = Freeze_Epoch

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('The data set is too small for training. Please expand the data set.')

    model_rpn.compile(
        loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1()
        }, optimizer=Adam(lr=lr)
    )
    model_all.compile(
        loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes): classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes): classifier_smooth_l1(num_classes - 1)
        }, optimizer=Adam(lr=lr)
    )

    gen = faster_rcnn_dataset(train_lines, input_shape, anchors, batch_size, num_classes, train=True).generate()
    gen_val = faster_rcnn_dataset(val_lines, input_shape, anchors, batch_size, num_classes, train=False).generate()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch, anchors, bbox_util, roi_helper)
        lr = lr * 0.96
        K.set_value(model_rpn.optimizer.lr, lr)
        K.set_value(model_all.optimizer.lr, lr)

if Freeze_Train:
    for i in range(freeze_layers):
        if type(model_all.layers[i]) != tf.keras.layers.BatchNormalization:
            model_all.layers[i].trainable = True

if True:
    batch_size = Unfreeze_batch_size
    lr = Unfreeze_lr
    start_epoch = Freeze_Epoch
    end_epoch = UnFreeze_Epoch

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('the data set is too small for training. please expand the data set.')

    model_rpn.compile(
        loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1()
        }, optimizer=Adam(lr=lr)
    )
    model_all.compile(
        loss={
            'classification': rpn_cls_loss(),
            'regression': rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes): classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes): classifier_smooth_l1(num_classes - 1)
        }, optimizer=Adam(lr=lr)
    )

    gen = faster_rcnn_dataset(train_lines, input_shape, anchors, batch_size, num_classes, train=True).generate()
    gen_val = faster_rcnn_dataset(val_lines, input_shape, anchors, batch_size, num_classes, train=False).generate()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch, anchors, bbox_util, roi_helper)
        lr = lr * 0.96
        K.set_value(model_rpn.optimizer.lr, lr)
        K.set_value(model_all.optimizer.lr, lr)
