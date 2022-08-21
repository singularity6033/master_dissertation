from utils import config
import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes


classes_path = os.path.join(config.VOC_ORIG_BASE_PATH, 'voc_classes.txt')
trainval_percent = 0.9
train_percent = 0.9
VOCdevkit_path = config.VOC_ORIG_BASE_PATH
VOCdevkit_sets = [('2012', 'train'), ('2012', 'val')]
classes, _ = get_classes(classes_path)


def extract_annotation(voc_year, voc_image_id, voc_list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (voc_year, voc_image_id)),
                   encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(float(xml_box.find('xmin').text)), int(float(xml_box.find('ymin').text)),
             int(float(xml_box.find('xmax').text)), int(float(xml_box.find('ymax').text)))
        voc_list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


random.seed(0)
print("Generate txt in ImageSets.")
xml_file_path = os.path.join(VOCdevkit_path, 'VOC2012/Annotations')
saveBasePath = os.path.join(VOCdevkit_path, 'VOC2012/ImageSets/Main')
temp_xml = os.listdir(xml_file_path)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)
        num = len(total_xml)
        list_num = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list_num, tv)
        train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)
trainval = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_trainval.txt'), 'w')
test = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_test.txt'), 'w')
train = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_train.txt'), 'w')
val = open(os.path.join(config.COCO_ORIG_BASE_PATH, 'shuffled_val.txt'), 'w')

for i in list_num:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        trainval.write(name)
        if i in train:
            train.write(name)
        else:
            val.write(name)
    else:
        test.write(name)
trainval.close()
train.close()
val.close()
test.close()
print("Generate txt in ImageSets done.")

print("Generate 2012_train.txt and 2012_val.txt for train.")
for year, image_set in VOCdevkit_sets:
    image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                     encoding='utf-8').read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

        extract_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
print("Generate 2012_train.txt and 2012_val.txt for train done.")
