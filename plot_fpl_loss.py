import matplotlib.pyplot as plt
import linecache

import numpy as np
from matplotlib import font_manager
import os

font_name = font_manager.FontProperties(fname='./font/Georgia.ttf', size=12, weight=40)
names = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
titles = ['VGG11 (CIFAR10)', 'VGG11 (CIFAR100)', 'VGG16 (CIFAR10)', 'VGG16 (CIFAR100)']
labels = ['subnet1']
colors_1 = ['#512D38', '#B27092', '#F4BFDB', '#FFE9F3', '#87BAAB',
            '#F08700', '#F49F0A', '#EFCA08', '#00A6A6', 'red']
line_styles = ['dashed', 'solid', 'dashdot']
marker_styles = ['o', 'x', '^', '*', '+']

file_path = '.'
t = [i for i in range(1, 201)]
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
for i in range(2):
    axs[0, i].set_title(titles[i], fontproperties=font_name)
    axs[0, i].set_xlabel("Iterations", fontproperties=font_name)
    axs[0, i].set_ylabel("Validation Loss", fontproperties=font_name)
    for j in range(10):
        p = os.path.sep.join([file_path, names[i] + '_fpl_tf2', 'logs_ipsw_morefc', 'w' + str(j + 1),
                              'ipsw_' + names[i] + '_w' + str(j + 1) + '_fpl_val_loss.txt'])
        line = linecache.getline(p, 1)
        loss = list(map(float, line.lstrip('[').rstrip(']\n').split(', ')))
        if j == 9:
            axs[0, i].plot(t, loss, color=colors_1[j], label='full net', linestyle=line_styles[j // 5],
                           marker=marker_styles[j // 2], linewidth=1.5)
        else:
            axs[0, i].plot(t, loss, color=colors_1[j], label='subnet-' + str(j + 1), linestyle=line_styles[j // 5],
                           marker=marker_styles[j // 2], linewidth=1.5)
    axs[0, i].legend(loc="best")
    axs[0, i].grid()

colors_2 = ['#512D38', '#B27092', '#F4BFDB', '#FFE9F3', '#87BAAB',
            '#F08700', '#F49F0A', '#EFCA08', '#00A6A6', '#3A86FF',
            '#FFBE0B', '#FB5607', '#FF006E', '#8338EC', 'red']
for i in range(2, 4):
    axs[1, i - 2].set_title(titles[i], fontproperties=font_name)
    axs[1, i - 2].set_xlabel("Iterations", fontproperties=font_name)
    axs[1, i - 2].set_ylabel("Validation Loss", fontproperties=font_name)
    for j in range(15):
        p = os.path.sep.join([file_path, names[i] + '_fpl_tf2', 'logs_ipsw_morefc', 'w' + str(j + 1),
                              'ipsw_' + names[i] + '_w' + str(j + 1) + '_fpl_val_loss.txt'])
        line = linecache.getline(p, 1)
        loss = list(map(float, line.lstrip('[').rstrip(']\n').split(', ')))
        if j == 14:
            axs[1, i - 2].plot(t, loss, color=colors_2[j], label='full net', linestyle=line_styles[j // 5],
                               marker=marker_styles[j // 3], linewidth=1.5)
        else:
            axs[1, i - 2].plot(t, loss, color=colors_2[j], label='subnet-' + str(j + 1), linestyle=line_styles[j // 5],
                               marker=marker_styles[j // 3], linewidth=1.5)
    axs[1, i - 2].legend(loc="best")
    axs[1, i - 2].grid()
# fig.show()
fig.savefig('./plot/fpl_ipsw_loss.png', dpi=300)
