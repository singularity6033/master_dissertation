import os
from imutils import paths
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from utils import config

"""
    load train2017 coco dataset.
    coco test2017 set has no annotations and we want to implement evaluation process from scratch.
    therefore, we shuffle train2017 set into train and validation with radio 10:1 (train:val) for training the network
"""

TrainImagePaths = list(paths.list_images(config.COCO_TRAIN_IMAGES))
TestImagePaths = list(paths.list_images(config.COCO_TEST_IMAGES))
img_train_paths, img_val_paths = train_test_split(TrainImagePaths, test_size=0.1)

coco = COCO(config.COCO_TRAIN_ANNOTS)
cats = coco.loadCats(coco.getCatIds())
catIds = [cat['id'] for cat in cats]
coco_classes = [cat['name'] for cat in cats]
class_names = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'coco_classes.txt'), 'w', encoding='utf-8')
# generate a file to store class names of coco dataset
for class_name in coco_classes:
    class_names.write(class_name)
    class_names.write('\n')
class_names.close()


def extract_annotations(imgId, target_file):
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
    annInfos = coco.loadAnns(annIds)
    for annInfo in annInfos:
        # extract the label and bounding box coordinates
        idx_gt = annInfo['category_id']
        cat = cats[catIds.index(idx_gt)]['name']
        cls_id = coco_classes.index(cat)
        xMin, yMin, bw, bh = annInfo['bbox']
        xMax = xMin + bw
        yMax = yMin + bh
        coordinates = (xMin, yMin, xMax, yMax)
        target_file.write(" " + ",".join(str(coordinate) for coordinate in coordinates) + ',' + str(cls_id))


train_file = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_train_set.txt'), 'w')
val_file = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_val_set.txt'), 'w')
test_file = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'test_set.txt'), 'w')

for img_train_id, img_train_path in enumerate(img_train_paths):
    print("[INFO] generate train set {}/{}...".format(img_train_id + 1, len(img_train_paths)))
    filename = img_train_path.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    img_id = int(filename.lstrip('0'))
    train_file.write(img_train_path)
    extract_annotations(img_id, train_file)
    train_file.write('\n')
train_file.close()

for img_val_id, img_val_path in enumerate(img_val_paths):
    print("[INFO] generate validation set {}/{}...".format(img_val_id + 1, len(img_val_paths)))
    filename = img_val_path.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    img_id = int(filename.lstrip('0'))
    val_file.write(img_val_path)
    extract_annotations(img_id, val_file)
    val_file.write('\n')
val_file.close()

for img_test_id, img_test_path in enumerate(TestImagePaths):
    print("[INFO] generate test set {}/{}...".format(img_test_id + 1, len(TestImagePaths)))
    test_file.write(img_test_path)
    test_file.write('\n')
test_file.close()
