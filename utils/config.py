# import the necessary packages
import os

# define the base path to the output of the vgg_cifar100 model
VGG_BASE_PATH = "./vgg_cifar100"
VGG_PLOT_PATH = "./vgg_cifar100/plot"

# define the base path to the *original* coco dataset and then use
# the base path to derive the coco image and annotations directories
COCO_ORIG_BASE_PATH = "./dataset/coco"
COCO_TRAIN_IMAGES = os.path.sep.join([COCO_ORIG_BASE_PATH, "train2017"])
COCO_TRAIN_ANNOTS = os.path.sep.join([COCO_ORIG_BASE_PATH, "annotations/instances_train2017.json"])
COCO_TEST_IMAGES = os.path.sep.join([COCO_ORIG_BASE_PATH, "val2017"])
COCO_TEST_ANNOTS = os.path.sep.join([COCO_ORIG_BASE_PATH, "annotations/instances_val2017.json"])

# define the base path to the original voc dataset and
# then use the base path to derive the voc image and annotations directories
VOC_ORIG_BASE_PATH = "./dataset/VOCdevkit"
VOC_TRAIN_IMAGES = os.path.sep.join([VOC_ORIG_BASE_PATH, "train2017"])
VOC_TRAIN_ANNOTS = os.path.sep.join([VOC_ORIG_BASE_PATH, "annotations/instances_train2017.json"])
VOC_TEST_IMAGES = os.path.sep.join([VOC_ORIG_BASE_PATH, "val2017"])
VOC_TEST_ANNOTS = os.path.sep.join([VOC_ORIG_BASE_PATH, "annotations/instances_val2017.json"])

# define the generated path w.r.t the voc dataset by ipsw
VOC_IPSW_BASE_PATH = "./dataset/voc_ipsw"
VOC_IPSW_TRAIN_PATH = "./dataset/voc_ipsw/train"
VOC_IPSW_VAL_PATH = "./dataset/voc_ipsw/val"

# define the generated path w.r.t the coco dataset by ipsw
COCO_IPSW_BASE_PATH = "./dataset/coco_ipsw"
COCO_IPSW_TRAIN_PATH = "./dataset/coco_ipsw/train"
COCO_IPSW_VAL_PATH = "./dataset/coco_ipsw/val"

# define the generated path w.r.t the voc dataset by ss
VOC_SS_BASE_PATH = "./dataset/voc_ss"
VOC_SS_TRAIN_PATH = "./dataset/voc_ss/train"
VOC_SS_VAL_PATH = "./dataset/voc_ss/val"

# define the generated path w.r.t the coco dataset by ss
COCO_SS_BASE_PATH = "./dataset/coco_ss"
COCO_SS_TRAIN_PATH = "./dataset/coco_ss/train"
COCO_SS_VAL_PATH = "./dataset/coco_ss/val"

# define the path to the base output directory
BASE_OUTPUT = "rcnn_coco"
# define the path to the output model, plot output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "coco_object_detector_rcnn_VGG16_224.h5"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "coco_label_encoder.pickle"])
MODEL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive ROI to be generated from each image
MAX_POSITIVE = 20

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
