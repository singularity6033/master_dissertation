import matplotlib.pyplot as plt
import linecache
import numpy as np
from matplotlib import font_manager
import os

font_name = font_manager.FontProperties(fname='./font/Georgia.ttf', size=12, weight=40)
names = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
titles = ['VGG11 (CIFAR10)', 'VGG11 (CIFAR100)', 'VGG16 (CIFAR10)', 'VGG16 (CIFAR100)']
labels_1 = ['subnet-1', 'subnet-2', 'subnet-3', 'subnet-4', 'subnet-5', 'subnet-6', 'subnet-7', 'subnet-8', 'subnet-9',
            'full net']
line_styles = ['dashed', 'solid', 'dashdot']
marker_styles = ['o', 'x', '^', '*', '+']
width = 0.35
x1 = np.arange(10)
x2 = np.arange(13)
file_path = './'
file_path_1 = 'map_result/results_0.05/results.txt'
file_path_2 = 'map_result/results_0.25/results.txt'
saved_path = './plot/fpl_frcnn_vgg16_cifar100.png'
# fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# for i in range(2):
#     plt.figure(i)
#     plt.title(titles[i], fontproperties=font_name)
#     plt.xlabel("mAP (%)", fontproperties=font_name)
#     map_1 = [0.0] * 10
#     map_2 = [0.0] * 10
#     for j in range(10):
#         p_1 = os.path.sep.join([file_path, names[i] + '_fpl_tf2/logs_ipsw_morefc', 'w' + str(j + 1), file_path_1])
#         p_2 = os.path.sep.join([file_path, names[i] + '_fpl_tf2/logs_ipsw_morefc', 'w' + str(j + 1), file_path_2])
#         tmp1 = float(linecache.getline(p_1, 84).split(' = ')[-1].rstrip('%\n'))
#         tmp2 = float(linecache.getline(p_2, 84).split(' = ')[-1].rstrip('%\n'))
#         map_1[j] = tmp1
#         map_2[j] = tmp2
#     rects1 = plt.barh([i - 0.2 for i in x1[::-1]], map_1, height=0.4, color='tab:blue', label='IoU = 0.05')
#     rects2 = plt.barh([i + 0.2 for i in x1[::-1]], map_2, height=0.4, color='tab:orange', label='IoU = 0.25')
#     plt.yticks(x1[::-1], labels_1, fontproperties=font_name)
#     plt.xlim(0, 10)
#     for rect1, rect2 in zip(rects1, rects2):
#         plt.text(rect1.get_width(), rect1.get_y() + rect1.get_height() / 2.,
#                  f'{rect1.get_width():.2f}' + '%', ha='left', va='center', fontsize='medium')
#         plt.text(rect2.get_width(), rect2.get_y() + rect2.get_height() / 2.,
#                  f'{rect2.get_width():.2f}' + '%', ha='left', va='center', fontsize='medium')
#     plt.legend(prop=font_name)
#     plt.savefig(saved_path[i])

labels_2 = ['subnet-1', 'subnet-2', 'subnet-3', 'subnet-4', 'subnet-5', 'subnet-6', 'subnet-7', 'subnet-8', 'subnet-9',
            'subnet-10', 'subnet-11', 'subnet-12', 'subnet-13']

for i in range(3, 4):
    plt.figure(i, figsize=(10, 8))
    plt.title(titles[i], fontproperties=font_name)
    plt.xlabel("mAP (%)", fontproperties=font_name)
    map_1 = [0.0] * 13
    # map_2 = [0.0] * 15
    for j in range(13):
        p_1 = os.path.sep.join([file_path, names[i] + '_fpl_tf2/logs_frcnn_voc', 'w' + str(j + 1), file_path_1])
        # p_2 = os.path.sep.join([file_path, names[i] + '_fpl_tf2/logs_ipsw_morefc', 'w' + str(j + 1), file_path_2])
        tmp1 = float(linecache.getline(p_1, 84).split(' = ')[-1].rstrip('%\n'))
        # tmp2 = float(linecache.getline(p_2, 84).split(' = ')[-1].rstrip('%\n'))
        map_1[j] = tmp1
        # map_2[j] = tmp2
    rects1 = plt.barh([i for i in x2[::-1]], map_1, height=0.4, color='tab:blue', label='IoU = 0.05')
    # rects2 = plt.barh([i + 0.2 for i in x2[::-1]], map_2, height=0.4, color='tab:orange', label='IoU = 0.25')
    plt.yticks(x2[::-1], labels_2, fontproperties=font_name)
    plt.xlim(0, 5)
    for rect1 in rects1:
        plt.text(rect1.get_width(), rect1.get_y() + rect1.get_height() / 2,
                 f'{rect1.get_width():.2f}' + '%', ha='left', va='center', fontsize='medium')
    plt.legend(prop=font_name)
    # plt.show()
    plt.savefig(saved_path, dpi=300)
